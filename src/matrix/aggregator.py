"""
Deterministic 2D Combinatorial Matrix Aggregator.
Aggregates paper classifications across two orthogonal dimensions without LLM hallucination.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

class CombinatorialMatrixAggregator:
    """Computes empirical density, recency trends, and adaptive sparsity across 2D axes."""

    @classmethod
    def aggregate_matrix(
        cls,
        papers: List[Dict[str, Any]],
        axis_a_labels: List[str],
        axis_b_labels: List[str],
        sparsity_percentage: float = 0.02
    ) -> Dict[str, Any]:
        """
        Build the 2D Cartesian grid across Axis A and Axis B.
        Excludes low-confidence classifications from raw density counts.
        """
        total_valid_papers = 0
        grid: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        low_confidence_grid: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        current_year = datetime.now(timezone.utc).year
        for p in papers:
            a_val = p.get("axis_a_tag")
            b_val = p.get("axis_b_tag")
            conf = p.get("tag_confidence", "high").lower()

            if not a_val or not b_val:
                continue

            total_valid_papers += 1
            if conf == "low":
                low_confidence_grid[(a_val, b_val)].append(p)
            else:
                grid[(a_val, b_val)].append(p)

        # Adaptive sparsity threshold based on corpus size
        sparsity_threshold = max(1, int(np.floor(sparsity_percentage * total_valid_papers)))

        # Build complete 2D matrix structure
        matrix_cells: List[Dict[str, Any]] = []
        density_table: Dict[str, Dict[str, int]] = {a: {b: 0 for b in axis_b_labels} for a in axis_a_labels}

        for a in axis_a_labels:
            for b in axis_b_labels:
                cell_papers = grid.get((a, b), [])
                low_conf_papers = low_confidence_grid.get((a, b), [])
                count = len(cell_papers)
                density_table[a][b] = count

                # Compute growth/recency trend (% published in last 2 years)
                recent_count = sum(1 for p in cell_papers if p.get("year", 0) >= (current_year - 2))
                recency_ratio = (recent_count / count) if count > 0 else 0.0

                # Check if cell is sparse
                is_sparse = count <= sparsity_threshold

                matrix_cells.append({
                    "cell_id": f"{a}__x__{b}",
                    "axis_a": a,
                    "axis_b": b,
                    "paper_count": count,
                    "low_confidence_count": len(low_conf_papers),
                    "papers": cell_papers,
                    "is_sparse": is_sparse,
                    "recency_ratio": round(recency_ratio, 2)
                })

        return {
            "total_papers": total_valid_papers,
            "sparsity_threshold": sparsity_threshold,
            "axis_a_labels": axis_a_labels,
            "axis_b_labels": axis_b_labels,
            "density_table": density_table,
            "cells": matrix_cells
        }
