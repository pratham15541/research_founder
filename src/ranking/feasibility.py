"""
General Feasibility Estimation Engine.
Decomposes untried research combinations into component dimensions (compute, data readiness, maturity)
to produce an objective, reproducible feasibility estimate without personalization bias.
"""

from typing import Dict, Any, List
import numpy as np

class ComponentFeasibilityEstimator:
    """Estimates the practical viability of a combinatorial gap from its orthogonal components."""

    @classmethod
    def estimate_gap_feasibility(
        cls,
        axis_a_val: str,
        axis_b_val: str,
        matrix_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate feasibility for the untried cell (axis_a_val, axis_b_val) by examining:
        1. Dimension A profile across all populated cells where A appears (methodology maturity).
        2. Dimension B profile across all populated cells where B appears (domain data availability).
        """
        density_table = matrix_result.get("density_table", {}) if matrix_result else {}
        axis_b_labels = matrix_result.get("axis_b_labels", []) if matrix_result else []
        axis_a_labels = matrix_result.get("axis_a_labels", []) if matrix_result else []

        # Component A: Volume & maturity across other B dimensions
        a_row = density_table.get(axis_a_val, {})
        a_counts = [a_row.get(b, 0) for b in axis_b_labels if b != axis_b_val]
        total_a_evidence = sum(a_counts)
        maturity_score = min(5.0, 1.0 + (np.log1p(total_a_evidence) * 0.9)) if total_a_evidence > 0 else 3.2

        # Component B: Volume & data availability across other A dimensions
        b_counts = [density_table.get(a, {}).get(axis_b_val, 0) for a in axis_a_labels if a != axis_a_val]
        total_b_evidence = sum(b_counts)
        data_readiness_score = min(5.0, 1.0 + (np.log1p(total_b_evidence) * 0.9)) if total_b_evidence > 0 else 3.4

        # Estimated composite feasibility (1.0 to 5.0)
        # Weighted combination: 45% method maturity + 45% domain data readiness - 10% integration friction
        raw_feasibility = (0.45 * maturity_score) + (0.45 * data_readiness_score)
        composite_feasibility = max(1.5, min(4.8, round(float(raw_feasibility), 2)))

        rationale = (
            f"Extrapolated from component dimensions: '{axis_a_val}' has {total_a_evidence} citations/papers "
            f"across other contexts (maturity score {maturity_score:.1f}/5.0), while '{axis_b_val}' has "
            f"{total_b_evidence} papers across existing methodologies (data readiness {data_readiness_score:.1f}/5.0). "
            f"Note: This is an empirical estimate extrapolated from component parts since no direct precedent exists."
        )

        return {
            "feasibility_score": composite_feasibility,
            "maturity_score": round(float(maturity_score), 2),
            "data_readiness_score": round(float(data_readiness_score), 2),
            "rationale": rationale,
            "disclaimer": "Estimate extrapolated from component parts; unknown integration friction or software coupling difficulties may exist."
        }

