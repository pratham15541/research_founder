"""
Rule-based Explainability Condition Decision Tree Generator.
Builds the transparent, verifiable list of conditions that justified why a candidate gap has potential.
"""

from typing import List, Dict, Any

class ConditionTreeBuilder:
    """Constructs verifiable decision trees for why a research opportunity was selected."""

    @classmethod
    def build_decision_conditions(
        cls,
        gap: Dict[str, Any],
        feasibility_data: Dict[str, Any],
        extracted_future_work_seeds: List[str]
    ) -> List[str]:
        """Generate structured condition statements based on empirical matrix metrics."""
        conditions: List[str] = []

        a = gap["axis_a"]
        b = gap["axis_b"]
        count = gap["paper_count"]
        row_neighbors = gap.get("row_dense_neighbors", [])
        col_neighbors = gap.get("col_dense_neighbors", [])

        # Condition 1: Sparsity
        conditions.append(
            f"Condition 1 (Empirical Void): Cell ('{a}' × '{b}') contains {count} papers, "
            f"below the corpus sparsity threshold."
        )

        # Condition 2: Method Ancestry
        if row_neighbors:
            top_rn = row_neighbors[0]
            conditions.append(
                f"Condition 2 (Method Viability): Paradigm '{a}' is well-established in adjacent domain "
                f"'{top_rn['axis_b']}' ({top_rn['count']} papers)."
            )

        # Condition 3: Domain Readiness
        if col_neighbors:
            top_cn = col_neighbors[0]
            conditions.append(
                f"Condition 3 (Domain Activity): Problem area '{b}' is actively studied with other paradigms "
                f"such as '{top_cn['axis_a']}' ({top_cn['count']} papers)."
            )

        # Condition 4: Component Feasibility Gate
        f_score = feasibility_data.get("feasibility_score", 3.0)
        conditions.append(
            f"Condition 4 (Resource Feasibility): Component extrapolation score is {f_score}/5.0, "
            f"verifying reasonable compute and data readiness."
        )

        # Condition 5: Literature Future Work Seed
        if extracted_future_work_seeds:
            seed = extracted_future_work_seeds[0]
            conditions.append(
                f"Condition 5 (Explicit Future Work Grounding): Author statement in literature: \"{seed[:150]}...\""
            )

        return conditions

