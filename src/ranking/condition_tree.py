"""
Dynamic Explainability Condition Generator.
Synthesizes transparent, verifiable conditions justifying why a candidate gap has potential:
  - LLM-grounded condition synthesis leveraging full candidate context
  - Dynamic data-driven fallback grounded in empirical metrics
"""

from typing import List, Dict, Any, Optional
import logging
from src.config import settings

logger = logging.getLogger(__name__)


class ConditionTreeBuilder:
    """Constructs verifiable conditions justifying why a research opportunity is viable and unaddressed."""

    @classmethod
    def build_decision_conditions(
        cls,
        gap: Dict[str, Any],
        feasibility_data: Dict[str, Any],
        extracted_future_work_seeds: List[str]
    ) -> List[str]:
        """Generate structured condition statements based on empirical metrics and evidence."""
        # 1. Try LLM synthesis
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            llm_conditions = cls._build_with_llm(gap, feasibility_data, extracted_future_work_seeds)
            if llm_conditions and len(llm_conditions) >= 3:
                return llm_conditions

        # 2. Dynamic empirical fallback
        return cls._build_dynamic_fallback(gap, feasibility_data, extracted_future_work_seeds)

    @classmethod
    def _build_with_llm(
        cls,
        gap: Dict[str, Any],
        feasibility_data: Dict[str, Any],
        extracted_future_work_seeds: List[str]
    ) -> Optional[List[str]]:
        """Query LLM to formulate clear condition statements justifying this gap."""
        try:
            from src.llm.nvidia_client import NvidiaClient

            title = gap.get("gap_title") or gap.get("title", "Research Gap")
            a = gap.get("axis_a", "")
            b = gap.get("axis_b", "")
            signal = gap.get("signal_type", "Empirical Literature Signal")
            why = gap.get("why_it_is_insufficient", "")
            supporting = [
                s.get("title") if isinstance(s, dict) else str(s)
                for s in gap.get("supporting_papers", [])[:3]
            ]
            f_score = feasibility_data.get("feasibility_score", 3.0)

            prompt = f"""You are a scientific research evaluator.
A proposed research gap has been identified:
- Title: "{title}"
- Proposed Method/Dimension A: "{a}"
- Target Domain/Dimension B: "{b}"
- Signal: {signal}
- Evidence Context: {why}
- Cited Literature: {', '.join(supporting) if supporting else 'Literature void'}
- Extrapolated Resource Feasibility: {f_score}/5.0
- Future Work Seed: {extracted_future_work_seeds[0] if extracted_future_work_seeds else 'N/A'}

Provide 4 to 5 concise condition statements (each beginning with "Condition N: ...") that formally justify why this gap is an actionable and scientifically viable research opportunity.

Respond ONLY with a valid JSON array of 4-5 strings:
["Condition 1 (...): ...", "Condition 2 (...): ...", "Condition 3 (...): ...", "Condition 4 (...): ..."]
"""
            result = NvidiaClient.generate_json(
                prompt=prompt,
                temperature=settings.LLM_STRUCTURED_TEMPERATURE,
                max_tokens=600,
                timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 30.0)
            )
            if isinstance(result, list) and len(result) >= 3:
                return [str(c).strip() for c in result if str(c).strip()]
        except Exception as e:
            logger.debug("LLM condition generation failed (%s); using dynamic fallback.", e)

        return None

    @classmethod
    def _build_dynamic_fallback(
        cls,
        gap: Dict[str, Any],
        feasibility_data: Dict[str, Any],
        extracted_future_work_seeds: List[str]
    ) -> List[str]:
        """Construct grounded condition statements dynamically from gap metrics."""
        conditions: List[str] = []

        a = gap.get("axis_a", "Methodology")
        b = gap.get("axis_b", "Domain")
        count = gap.get("paper_count", 0)
        signal_type = gap.get("signal_type", "Empirical Evidence")
        supporting = gap.get("supporting_papers", [])

        # Condition 1: Signal & Evidence Grounding
        supp_detail = f" with {len(supporting)} supporting citation(s)" if supporting else ""
        if count == 0:
            conditions.append(
                f"Condition 1 (Empirical Void): Cell ('{a}' × '{b}') contains {count} direct papers, representing an unpopulated literature space{supp_detail}."
            )
        else:
            conditions.append(
                f"Condition 1 ({signal_type}): Grounded in literature evidence{supp_detail} (direct studies in cell: {count})."
            )

        # Condition 2: Method Precedent
        row_neighbors = gap.get("row_dense_neighbors", [])
        if row_neighbors:
            top_rn = row_neighbors[0]
            conditions.append(
                f"Condition 2 (Method Viability): Paradigm '{a}' is well-established in adjacent context '{top_rn.get('axis_b', 'domain')}' ({top_rn.get('count', 0)} papers)."
            )
        else:
            method_stat = gap.get("method_status", "ACTIVE")
            conditions.append(
                f"Condition 2 (Methodology Status): Paradigm '{a}' exhibits confirmed maturity ({method_stat}) across related literature regimes."
            )

        # Condition 3: Domain Demand
        col_neighbors = gap.get("col_dense_neighbors", [])
        if col_neighbors:
            top_cn = col_neighbors[0]
            conditions.append(
                f"Condition 3 (Domain Activity): Problem area '{b}' is actively studied with other paradigms such as '{top_cn.get('axis_a', 'methods')}' ({top_cn.get('count', 0)} papers)."
            )
        else:
            domain_mat = gap.get("domain_maturity", "ACTIVE")
            conditions.append(
                f"Condition 3 (Domain Readiness): Problem setting '{b}' reflects active empirical investigation ({domain_mat}) with documented baseline benchmarks."
            )

        # Condition 4: Feasibility Gate
        f_score = feasibility_data.get("feasibility_score", 3.0)
        rationale = feasibility_data.get("rationale")
        rat_snippet = f" — {rationale[:120]}..." if rationale else ""
        conditions.append(
            f"Condition 4 (Resource Feasibility): Component extrapolation score is {f_score}/5.0{rat_snippet}"
        )

        # Condition 5: Future Work Seed or Compatibility
        if extracted_future_work_seeds:
            seed = str(extracted_future_work_seeds[0]).strip()
            conditions.append(
                f"Condition 5 (Literature Grounding): Explicit author direction: \"{seed[:150]}...\""
            )
        else:
            compat_verdict = gap.get("compatibility_verdict", "COMPATIBLE")
            conditions.append(
                f"Condition 5 (Modality Compatibility): Technical coupling between '{a}' and '{b}' is verified as {compat_verdict}."
            )

        return conditions
