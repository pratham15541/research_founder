"""
Stage 5 — Prior-Art Saturation Check.

Walks the Literature Evidence Graph up to 2 hops from the candidate
(method_node, domain_node) pair and counts how many papers are reachable.
A high reachability score means the gap is already being addressed — even if
no single paper explicitly combines both axes.

Saturation Verdicts
-------------------
- OPEN                : ≤1 hop-2 paper bridges method+domain
- PARTIALLY_SATURATED : 2–4 bridging papers found within 2 hops
- SATURATED           : ≥5 bridging papers → gap is REJECTED

Output per candidate
--------------------
{
  "saturation_score": float 0–1   (0 = fully open, 1 = fully saturated)
  "saturation_verdict": "OPEN" | "PARTIALLY_SATURATED" | "SATURATED"
  "saturating_papers": [title, ...]
  "hop1_count": int
  "hop2_count": int
  "rejection_reason": None | "SATURATED_GAP"
}
"""

import logging
from typing import List, Dict, Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)

_SATURATION_OPEN_THRESHOLD = 1          # ≤ this → OPEN
_SATURATION_PARTIAL_THRESHOLD = 4       # ≤ this → PARTIALLY_SATURATED
# > partial threshold → SATURATED


class PriorArtChecker:
    """Graph-walk prior-art saturation checker (Stage 5)."""

    @classmethod
    def check_corpus(
        cls,
        candidate_gaps: List[Dict[str, Any]],
        evidence_graph: nx.DiGraph,
        structured_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run Stage 5 over all candidates.
        Attaches `prior_art` dict to each candidate in-place.
        """
        for cand in candidate_gaps:
            result = cls.check_candidate(cand, evidence_graph, structured_records)
            cand["prior_art"] = result
            if result["rejection_reason"]:
                cand.setdefault("pipeline_rejections", []).append(result["rejection_reason"])
        return candidate_gaps

    @classmethod
    def check_candidate(
        cls,
        candidate: Dict[str, Any],
        evidence_graph: nx.DiGraph,
        structured_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check prior-art saturation for a single candidate gap."""
        axis_a: str = candidate.get("axis_a", "")
        axis_b: str = candidate.get("axis_b", "")

        # Locate method_node and domain nodes in evidence graph
        method_node = cls._find_node(evidence_graph, "method", axis_a)
        domain_node = cls._find_node(evidence_graph, "domain", axis_b)

        hop1_papers: set = set()
        hop2_papers: set = set()

        if method_node:
            # Hop 1: papers directly connected to method node
            for neighbor in evidence_graph.predecessors(method_node):
                if evidence_graph.nodes[neighbor].get("node_type") == "paper":
                    hop1_papers.add(neighbor)

            # Hop 2: papers connected to hop1 papers' other method/domain nodes
            for paper_node in hop1_papers:
                for successor in evidence_graph.successors(paper_node):
                    ntype = evidence_graph.nodes[successor].get("node_type", "")
                    if ntype in ("method", "domain", "future_work"):
                        for p2 in evidence_graph.predecessors(successor):
                            if (
                                p2 not in hop1_papers
                                and evidence_graph.nodes[p2].get("node_type") == "paper"
                            ):
                                hop2_papers.add(p2)

        # Also do text-based check using entity_shared edges (new edge type)
        entity_bridging = cls._entity_bridging_count(axis_a, axis_b, structured_records)

        total_reachable = len(hop1_papers) + len(hop2_papers) + entity_bridging

        # Resolve paper titles
        def node_to_title(n: str) -> str:
            return evidence_graph.nodes[n].get("label", n) if evidence_graph.has_node(n) else n

        saturating = [node_to_title(n) for n in list(hop1_papers)[:3] + list(hop2_papers)[:3]]

        # Saturation verdict
        if total_reachable <= _SATURATION_OPEN_THRESHOLD:
            verdict = "OPEN"
            score = max(0.0, total_reachable * 0.15)
            rejection_reason = None
        elif total_reachable <= _SATURATION_PARTIAL_THRESHOLD:
            verdict = "PARTIALLY_SATURATED"
            score = 0.30 + (total_reachable - 1) * 0.10
            rejection_reason = None
        else:
            verdict = "SATURATED"
            score = min(1.0, 0.65 + (total_reachable - 4) * 0.05)
            rejection_reason = "SATURATED_GAP"

        logger.debug(
            "PriorArtChecker: '%s × %s' → %s (reachable=%d)",
            axis_a[:30], axis_b[:30], verdict, total_reachable,
        )

        return {
            "saturation_score": round(score, 3),
            "saturation_verdict": verdict,
            "saturating_papers": saturating[:6],
            "hop1_count": len(hop1_papers),
            "hop2_count": len(hop2_papers),
            "entity_bridging_count": entity_bridging,
            "rejection_reason": rejection_reason,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_node(graph: nx.DiGraph, node_type: str, axis_text: str) -> Optional[str]:
        """Find the best-matching node of given type in the graph."""
        if not axis_text or not graph:
            return None
        tokens = set(axis_text.lower().split())
        best_node = None
        best_score = 0
        for node, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != node_type:
                continue
            label = attrs.get("label", "").lower()
            score = sum(1 for t in tokens if t in label)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node if best_score >= 1 else None

    @staticmethod
    def _entity_bridging_count(
        axis_a: str,
        axis_b: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """
        Count papers that share scientific entities covering BOTH axis_a and axis_b.
        Uses the `scientific_entities` field added in Stage 1.
        """
        import re
        tokens_a = set(re.findall(r"[a-zA-Z]{4,}", axis_a.lower()))
        tokens_b = set(re.findall(r"[a-zA-Z]{4,}", axis_b.lower()))
        if not tokens_a or not tokens_b:
            return 0
        count = 0
        for r in records:
            entities_text = " ".join(str(e).lower() for e in r.get("scientific_entities", []))
            abs_text = r.get("abstract", "").lower()
            combined = entities_text + " " + abs_text
            hit_a = any(t in combined for t in tokens_a)
            hit_b = any(t in combined for t in tokens_b)
            if hit_a and hit_b:
                count += 1
        return count

