"""
Stage 3 — Domain Verification.

Answers: "Is the target domain (axis_b) an active research area in the corpus
with at least one empirically evaluated paper?"

Rules
-----
- ACTIVE   : ≥1 paper has the domain tag AND has evaluation_metrics
- NASCENT  : domain appears in papers but only in theoretical/survey form
- INACTIVE : domain not meaningfully represented in corpus

Output per candidate
--------------------
{
  "domain_activity_score": float 0–1,
  "domain_maturity": "ACTIVE" | "NASCENT" | "INACTIVE",
  "active_papers": [title, ...],
  "survey_only_papers": [title, ...],
  "rejection_reason": None | "INACTIVE_DOMAIN"
}
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DomainVerifier:
    """Verifies that the proposed gap's target domain is active in the corpus."""

    @classmethod
    def verify_corpus(
        cls,
        candidate_gaps: List[Dict[str, Any]],
        structured_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run Stage 3 over all candidates.
        Attaches `domain_verification` dict to each candidate in-place.
        """
        for cand in candidate_gaps:
            result = cls.verify_candidate(cand, structured_records)
            cand["domain_verification"] = result
            if result["rejection_reason"]:
                cand.setdefault("pipeline_rejections", []).append(result["rejection_reason"])
        return candidate_gaps

    @classmethod
    def verify_candidate(
        cls,
        candidate: Dict[str, Any],
        structured_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify the domain dimension of a single candidate gap."""
        axis_b: str = candidate.get("axis_b", "")
        gap_title: str = candidate.get("gap_title", "")

        domain_tokens = cls._tokenize(axis_b)
        if not domain_tokens:
            domain_tokens = cls._tokenize(gap_title.split(" in ")[-1])

        active_papers: List[str] = []
        survey_only_papers: List[str] = []

        for record in structured_records:
            title = record.get("title", "")
            domain_field = str(record.get("domain", "")).lower()
            domain_tags: List[str] = [str(t).lower() for t in record.get("domain_tags", [])]
            methodology_class = str(record.get("methodology_class", "")).lower()
            abstract = record.get("abstract", "").lower()
            has_metrics = bool(record.get("evaluation_metrics"))

            domain_text = domain_field + " " + " ".join(domain_tags) + " " + abstract

            if cls._tokens_match(domain_tokens, domain_text):
                is_empirical = has_metrics and methodology_class not in ("survey",)
                if is_empirical:
                    active_papers.append(title)
                else:
                    survey_only_papers.append(title)

        active_count = len(active_papers)
        survey_count = len(survey_only_papers)

        if active_count >= 2:
            maturity = "ACTIVE"
            score = min(1.0, 0.55 + active_count * 0.10)
            rejection_reason = None
        elif active_count == 1:
            maturity = "ACTIVE"
            score = 0.50
            rejection_reason = None
        elif survey_count >= 1:
            maturity = "NASCENT"
            score = 0.30
            rejection_reason = None
        else:
            maturity = "INACTIVE"
            score = 0.05
            rejection_reason = "INACTIVE_DOMAIN"

        logger.debug(
            "DomainVerifier: '%s' → %s (active=%d, survey=%d)",
            axis_b[:50], maturity, active_count, survey_count,
        )

        return {
            "domain_activity_score": round(score, 3),
            "domain_maturity": maturity,
            "active_papers": active_papers[:4],
            "survey_only_papers": survey_only_papers[:4],
            "domain_tokens_used": list(domain_tokens)[:8],
            "rejection_reason": rejection_reason,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> set:
        raw = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stop = {"with", "from", "that", "this", "have", "been", "into", "upon", "over",
                "under", "more", "than", "each", "also", "only", "both", "well",
                "benchmark", "evaluation", "corpus", "emerging", "frontier"}
        return {t for t in raw if t not in stop}

    @staticmethod
    def _tokens_match(tokens: set, text: str, threshold: float = 0.35) -> bool:
        if not tokens:
            return False
        matched = sum(1 for t in tokens if t in text)
        return (matched / len(tokens)) >= threshold

