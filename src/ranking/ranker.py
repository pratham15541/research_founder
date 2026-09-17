"""
Master Gap Ranking & Grounding Engine.
Ranks candidate gaps using grounded neighbor context, feasibility estimates,
adversarial self-critique, and decision tree conditions.
"""

import logging
from typing import List, Dict, Any
from src.ranking.feasibility import ComponentFeasibilityEstimator
from src.ranking.devil_advocate import DevilsAdvocateNode
from src.ranking.condition_tree import ConditionTreeBuilder
from src.ranking.project_synthesizer import ResearchProjectSynthesizer

logger = logging.getLogger(__name__)

class GroundedGapRanker:
    """Ranks and grounds candidate research gaps into actionable dossiers."""

    @classmethod
    def rank_candidate_gaps(
        cls,
        candidate_gaps: List[Dict[str, Any]],
        matrix_result: Dict[str, Any],
        future_work_seeds: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Produce top K ranked gap dossiers with grounded evidence and devil's advocate critique.
        """
        ranked_dossiers: List[Dict[str, Any]] = []

        for gap in candidate_gaps:
            a = gap["axis_a"]
            b = gap["axis_b"]

            # 1. Feasibility estimation
            feasibility_info = ComponentFeasibilityEstimator.estimate_gap_feasibility(
                a, b, matrix_result
            )
            feasibility_score = feasibility_info["feasibility_score"]

            # 2. Gather neighboring supporting papers for grounding
            supporting_evidence = []
            papers_a: List[Dict[str, Any]] = []
            papers_b: List[Dict[str, Any]] = []

            # Gather from row dense neighbors
            for cell in matrix_result["cells"]:
                if cell["axis_a"] == a and cell["axis_b"] != b and len(cell["papers"]) > 0:
                    for p in cell["papers"][:2]:
                        papers_a.append(p)
                        supporting_evidence.append({
                            "paper_id": p.get("id"),
                            "title": p.get("title"),
                            "year": p.get("year"),
                            "source_url": p.get("source_url"),
                            "role": f"Demonstrates maturity of paradigm '{a}' in domain '{cell['axis_b']}'"
                        })
                elif cell["axis_b"] == b and cell["axis_a"] != a and len(cell["papers"]) > 0:
                    for p in cell["papers"][:2]:
                        papers_b.append(p)
                        supporting_evidence.append({
                            "paper_id": p.get("id"),
                            "title": p.get("title"),
                            "year": p.get("year"),
                            "source_url": p.get("source_url"),
                            "role": f"Demonstrates activity in domain '{b}' using methodology '{cell['axis_a']}'"
                        })

            # 3. Adversarial Critique
            counter_argument = DevilsAdvocateNode.generate_critique(
                a, b, papers_a, papers_b
            )

            # 4. Synthesize human-readable project formulation
            proj_info = ResearchProjectSynthesizer.synthesize_project(
                axis_a=a,
                axis_b=b,
                neighbor_papers_a=papers_a,
                neighbor_papers_b=papers_b,
                cell_count=gap.get("paper_count", 0)
            )

            # 5. Novelty & Impact scoring
            novelty_score = min(5.0, 4.5 + (0.2 * gap.get("structural_gap_weight", 1.0)))
            impact_score = min(5.0, 2.5 + (0.5 * len(supporting_evidence)))

            # Composite opportunity score: 35% Novelty + 35% Feasibility + 30% Impact
            composite_score = round(
                (0.35 * novelty_score) + (0.35 * feasibility_score) + (0.30 * impact_score),
                2
            )

            # 6. Build Decision Conditions
            conditions = ConditionTreeBuilder.build_decision_conditions(
                gap, feasibility_info, future_work_seeds
            )

            dossier = {
                "gap_id": f"gap_{a}_{b}".replace(" ", "_"),
                "axis_a": a,
                "axis_b": b,
                "project_title": proj_info.get("project_title", f"{a} in {b}"),
                "core_research_question": proj_info.get("core_research_question", ""),
                "why_it_is_a_gap": proj_info.get("why_it_is_a_gap", ""),
                "suggested_first_experiment": proj_info.get("suggested_first_experiment", ""),
                "practical_impact": proj_info.get("practical_impact", ""),
                "novelty_score": round(novelty_score, 2),
                "feasibility_score": round(feasibility_score, 2),
                "impact_score": round(impact_score, 2),
                "composite_score": composite_score,
                "feasibility_rationale": feasibility_info["rationale"],
                "counter_argument": counter_argument,
                "supporting_evidence": supporting_evidence[:6],
                "grounding_conditions": conditions,
                "total_neighbor_evidence_count": len(supporting_evidence)
            }
            ranked_dossiers.append(dossier)

        # Sort by composite score descending
        ranked_dossiers.sort(key=lambda d: d["composite_score"], reverse=True)
        return ranked_dossiers[:top_k]

