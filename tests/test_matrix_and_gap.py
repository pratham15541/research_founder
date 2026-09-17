"""
Unit tests for Combinatorial 2D Matrix Aggregation and Candidate Gap Adjacency Filtering.
"""

import pytest
from src.matrix.aggregator import CombinatorialMatrixAggregator
from src.matrix.gap_filter import CandidateGapFilter

def test_matrix_aggregation_and_sparsity():
    axis_a = ["Diffusion Models", "Transformers", "Graph Neural Networks"]
    axis_b = ["Healthcare", "Robotics", "Finance"]

    # Synthetic corpus with 50 papers
    papers = []
    # Dense cell: Diffusion Models x Healthcare (20 papers)
    for i in range(20):
        papers.append({
            "id": f"p_dh_{i}",
            "title": f"Diffusion in Health {i}",
            "axis_a_tag": "Diffusion Models",
            "axis_b_tag": "Healthcare",
            "tag_confidence": "high",
            "year": 2024
        })

    # Dense cell: Transformers x Robotics (15 papers)
    for i in range(15):
        papers.append({
            "id": f"p_tr_{i}",
            "title": f"Transformers in Robotics {i}",
            "axis_a_tag": "Transformers",
            "axis_b_tag": "Robotics",
            "tag_confidence": "high",
            "year": 2023
        })

    # Moderate cell: Transformers x Healthcare (10 papers)
    for i in range(10):
        papers.append({
            "id": f"p_th_{i}",
            "title": f"Transformers in Health {i}",
            "axis_a_tag": "Transformers",
            "axis_b_tag": "Healthcare",
            "tag_confidence": "high",
            "year": 2024
        })

    # Sparse cell: Diffusion Models x Robotics (0 papers - our target candidate gap!)

    matrix_res = CombinatorialMatrixAggregator.aggregate_matrix(
        papers=papers,
        axis_a_labels=axis_a,
        axis_b_labels=axis_b,
        sparsity_percentage=0.02
    )

    # Assert total papers and cells
    assert matrix_res["total_papers"] == 45
    assert len(matrix_res["cells"]) == 9
    assert matrix_res["density_table"]["Diffusion Models"]["Healthcare"] == 20
    assert matrix_res["density_table"]["Diffusion Models"]["Robotics"] == 0

    # Run Candidate Gap Filtering
    candidate_gaps = CandidateGapFilter.filter_candidate_gaps(matrix_res, dense_multiplier=3)

    # The cell (Diffusion Models, Robotics) should be flagged because:
    # - It has 0 papers (sparse)
    # - It borders (Diffusion Models, Healthcare) with 20 papers in the same row!
    # - It borders (Transformers, Robotics) with 15 papers in the same column!
    gap_ids = [f"{g['axis_a']}__{g['axis_b']}" for g in candidate_gaps]
    assert "Diffusion Models__Robotics" in gap_ids

    diff_robotics_gap = next(g for g in candidate_gaps if g["axis_a"] == "Diffusion Models" and g["axis_b"] == "Robotics")
    assert diff_robotics_gap["bordered_in_both_dimensions"] is True
    assert diff_robotics_gap["structural_gap_weight"] > 1.0

def test_isolated_void_rejection():
    """Verify that empty cells with NO populated neighbors are rejected as non-viable."""
    axis_a = ["Method A", "Method B"]
    axis_b = ["Domain 1", "Domain 2"]

    # Only 1 populated cell: Method A x Domain 1 (10 papers)
    papers = [
        {"axis_a_tag": "Method A", "axis_b_tag": "Domain 1", "tag_confidence": "high", "year": 2024}
        for _ in range(10)
    ]

    matrix_res = CombinatorialMatrixAggregator.aggregate_matrix(
        papers=papers,
        axis_a_labels=axis_a,
        axis_b_labels=axis_b
    )

    candidate_gaps = CandidateGapFilter.filter_candidate_gaps(matrix_res, dense_multiplier=3)
    gap_coords = [(g["axis_a"], g["axis_b"]) for g in candidate_gaps]

    # Method B x Domain 2 has 0 papers and borders NO populated cells along row or column!
    assert ("Method B", "Domain 2") not in gap_coords

