"""
Human Evaluation Scorecard Framework.
Records expert ratings on Gap Relevance, Novelty, Explainability, and Hallucination Rate.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone

class HumanEvaluationManager:
    """Manages human assessment benchmarks for research gaps and generated literature reviews."""

    @staticmethod
    def create_evaluation_record(
        gap_id: str,
        project_title: str,
        relevance_score: int,  # 1 to 5
        novelty_score: int,    # 1 to 5
        explainability_score: int,  # 1 to 5
        hallucination_detected: bool,
        expert_comments: str = ""
    ) -> Dict[str, Any]:
        """Format a single expert assessment record."""
        return {
            "gap_id": gap_id,
            "project_title": project_title,
            "relevance_score": max(1, min(5, relevance_score)),
            "novelty_score": max(1, min(5, novelty_score)),
            "explainability_score": max(1, min(5, explainability_score)),
            "hallucination_detected": hallucination_detected,
            "expert_comments": expert_comments.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @staticmethod
    def aggregate_human_scores(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate human evaluation metrics."""
        if not records:
            return {
                "total_ratings": 0,
                "mean_relevance": 0.0,
                "mean_novelty": 0.0,
                "mean_explainability": 0.0,
                "hallucination_rate_percent": 0.0
            }

        n = len(records)
        mean_rel = sum(r["relevance_score"] for r in records) / n
        mean_nov = sum(r["novelty_score"] for r in records) / n
        mean_exp = sum(r["explainability_score"] for r in records) / n
        hallucinations = sum(1 for r in records if r.get("hallucination_detected"))
        hal_rate = (hallucinations / n) * 100.0

        return {
            "total_ratings": n,
            "mean_relevance": round(mean_rel, 2),
            "mean_novelty": round(mean_nov, 2),
            "mean_explainability": round(mean_exp, 2),
            "hallucination_rate_percent": round(hal_rate, 1)
        }

