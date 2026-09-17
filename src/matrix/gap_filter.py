"""
Deterministic Candidate Gap Filter.
Enforces structural adjacency rules: a sparse cell is only a valid research gap if it borders
well-populated neighboring cells in the same row or column. Rejects isolated non-viable voids.
"""

from typing import List, Dict, Any

class CandidateGapFilter:
    """Filters and weights candidate research gaps from a 2D matrix based on structural adjacency."""

    @classmethod
    def filter_candidate_gaps(
        cls,
        matrix_result: Dict[str, Any],
        dense_multiplier: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify candidate gaps based on:
        1. Sparsity: cell_count <= sparsity_threshold
        2. Adjacency: at least 1 neighbor in same row OR same column has count >= dense_multiplier * threshold
        3. Multi-neighbor weighting: bonus score for being bordered along both axes.
        """
        cells = matrix_result["cells"]
        density_table = matrix_result["density_table"]
        threshold = matrix_result["sparsity_threshold"]
        dense_cutoff = max(threshold + 1, dense_multiplier * threshold)

        axis_a_labels = matrix_result["axis_a_labels"]
        axis_b_labels = matrix_result["axis_b_labels"]

        candidate_gaps: List[Dict[str, Any]] = []

        for cell in cells:
            if not cell["is_sparse"]:
                continue

            a = cell["axis_a"]
            b = cell["axis_b"]

            # Exclude unclustered noise / outlier clusters from being proposed as research methodologies
            if any(term in a.lower() for term in ["other", "emerging directions", "outlier", "unclassified", "noise"]):
                continue

            # Check neighbors along same row (same Axis A, different Axis B)
            row_dense_neighbors = []
            for other_b in axis_b_labels:
                if other_b != b:
                    c = density_table[a][other_b]
                    if c >= dense_cutoff:
                        row_dense_neighbors.append({"axis_b": other_b, "count": c})

            # Check neighbors along same column (different Axis A, same Axis B)
            col_dense_neighbors = []
            for other_a in axis_a_labels:
                if other_a != a:
                    c = density_table[other_a][b]
                    if c >= dense_cutoff:
                        col_dense_neighbors.append({"axis_a": other_a, "count": c})

            total_dense_neighbors = len(row_dense_neighbors) + len(col_dense_neighbors)

            # Rule: Must have at least 1 dense neighbor. Isolated cells are dropped.
            if total_dense_neighbors < 1:
                continue

            # Weighting: multi-axis adjacency bonus
            multi_axis_bonus = 1.5 if (row_dense_neighbors and col_dense_neighbors) else 1.0
            structural_gap_weight = round(total_dense_neighbors * multi_axis_bonus, 2)

            gap_record = dict(cell)
            gap_record.update({
                "is_candidate_gap": True,
                "row_dense_neighbors": row_dense_neighbors,
                "col_dense_neighbors": col_dense_neighbors,
                "total_dense_neighbors": total_dense_neighbors,
                "structural_gap_weight": structural_gap_weight,
                "bordered_in_both_dimensions": bool(row_dense_neighbors and col_dense_neighbors)
            })
            candidate_gaps.append(gap_record)

        # Sort candidate gaps by structural weight descending
        candidate_gaps.sort(key=lambda g: g["structural_gap_weight"], reverse=True)
        return candidate_gaps

