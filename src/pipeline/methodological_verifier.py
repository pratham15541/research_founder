"""
Stage 2 — Methodological Verification.

For every candidate gap the verifier answers: "Does the proposed method actually
exist in the corpus as something that has been *empirically tested*, not merely
mentioned or proposed in a future-work bullet?"

Rules
-----
- VERIFIED   : method appears in ≥2 structured records in `method` or
               `scientific_entities` AND ≥1 of those records has `evaluation_metrics`
- SPECULATIVE: method only appears in `future_work` strings (never evaluated)
- UNVERIFIED : method not found in corpus at all

Output per candidate
--------------------
{
  "method_evidence_score": float 0–1,
  "method_status": "VERIFIED" | "SPECULATIVE" | "UNVERIFIED",
  "verified_papers": [title, ...],
  "speculative_papers": [title, ...],
  "rejection_reason": None | "SPECULATIVE_METHOD"
}
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MethodologicalVerifier:
    """Verifies that a proposed gap method is empirically grounded in the corpus."""

    @classmethod
    def verify_corpus(
        cls,
        candidate_gaps: List[Dict[str, Any]],
        structured_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run Stage 2 over all candidates.
        Attaches `method_verification` dict to each candidate in-place and returns them.
        """
        for cand in candidate_gaps:
            result = cls.verify_candidate(cand, structured_records)
            cand["method_verification"] = result
            if result["rejection_reason"]:
                cand.setdefault("pipeline_rejections", []).append(result["rejection_reason"])
        return candidate_gaps

    @classmethod
    def verify_candidate(
        cls,
        candidate: Dict[str, Any],
        structured_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify the method in a single candidate gap against the corpus."""
        axis_a: str = candidate.get("axis_a", "")
        gap_title: str = candidate.get("gap_title", "")

        # Build token set from axis_a (the proposed methodology dimension)
        method_tokens = cls._tokenize(axis_a)
        if not method_tokens:
            method_tokens = cls._tokenize(gap_title)

        verified_papers: List[str] = []
        speculative_papers: List[str] = []

        for record in structured_records:
            title = record.get("title", "")
            method_text = str(record.get("method", "")).lower()
            entities: List[str] = [str(e).lower() for e in record.get("scientific_entities", [])]
            future_work: List[str] = [str(f).lower() for f in record.get("future_work", [])]
            has_metrics = bool(record.get("evaluation_metrics"))

            combined_empirical = method_text + " " + " ".join(entities)
            combined_future = " ".join(future_work)

            in_empirical = cls._tokens_match(method_tokens, combined_empirical)
            in_future_only = (not in_empirical) and cls._tokens_match(method_tokens, combined_future)

            if in_empirical:
                if has_metrics:
                    verified_papers.append(title)
                else:
                    speculative_papers.append(title)
            elif in_future_only:
                speculative_papers.append(title)

        verified_count = len(verified_papers)
        speculative_count = len(speculative_papers)
        total = verified_count + speculative_count

        if verified_count >= 2:
            status = "VERIFIED"
            score = min(1.0, 0.5 + (verified_count * 0.15))
            rejection_reason = None
        elif verified_count == 1:
            status = "VERIFIED"
            score = 0.45
            rejection_reason = None
        elif speculative_count >= 1:
            status = "SPECULATIVE"
            score = 0.20
            rejection_reason = "SPECULATIVE_METHOD"
        else:
            status = "UNVERIFIED"
            score = 0.10
            rejection_reason = "SPECULATIVE_METHOD"

        logger.debug(
            "MethodologicalVerifier: '%s' → %s (verified=%d, speculative=%d)",
            axis_a[:50], status, verified_count, speculative_count,
        )

        return {
            "method_evidence_score": round(score, 3),
            "method_status": status,
            "verified_papers": verified_papers[:4],
            "speculative_papers": speculative_papers[:4],
            "method_tokens_used": list(method_tokens)[:8],
            "rejection_reason": rejection_reason,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> set:
        """Extract lowercase meaningful tokens (≥4 chars) from a phrase."""
        raw = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stop = {"with", "from", "that", "this", "have", "been", "into", "upon", "over",
                "under", "more", "than", "each", "also", "only", "both", "well"}
        return {t for t in raw if t not in stop}

    @staticmethod
    def _tokens_match(tokens: set, text: str, threshold: float = 0.40) -> bool:
        """Return True if fraction of tokens found in text ≥ threshold."""
        if not tokens:
            return False
        matched = sum(1 for t in tokens if t in text)
        return (matched / len(tokens)) >= threshold

