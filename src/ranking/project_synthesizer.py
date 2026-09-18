"""
Human-Readable Research Project Synthesizer.
Transforms abstract matrix coordinates (e.g. 'Method A x Domain B') into concrete,
publishable research proposals with intuitive research questions, gap justifications,
and first-step validation experiments.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class ResearchProjectSynthesizer:
    """Synthesizes human-readable research project dossiers from combinatorial gap coordinates."""

    @classmethod
    def synthesize_project(
        cls,
        axis_a: str,
        axis_b: str,
        neighbor_papers_a: List[Dict[str, Any]],
        neighbor_papers_b: List[Dict[str, Any]],
        cell_count: int = 0
    ) -> Dict[str, str]:
        """
        Generate a human-readable project briefing for a detected research gap.
        Uses NVIDIA AI if configured; falls back to structured domain heuristics.
        """
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            llm_result = cls._synthesize_with_llm(axis_a, axis_b, neighbor_papers_a, neighbor_papers_b, cell_count)
            if llm_result:
                return llm_result
        elif settings.LLM_REQUIRED:
            raise RuntimeError("NVIDIA_API_KEY is required because LLM_REQUIRED=true.")

        return cls._synthesize_with_heuristics(axis_a, axis_b, neighbor_papers_a, neighbor_papers_b, cell_count)

    @classmethod
    def _synthesize_with_llm(
        cls,
        axis_a: str,
        axis_b: str,
        papers_a: List[Dict[str, Any]],
        papers_b: List[Dict[str, Any]],
        cell_count: int
    ) -> Optional[Dict[str, str]]:
        """Call NVIDIA AI API to synthesize a structured, plain-English research project."""
        context_a = "\n".join([f"- {p.get('title')}" for p in papers_a[:3]])
        context_b = "\n".join([f"- {p.get('title')}" for p in papers_b[:3]])

        prompt = f"""You are a principal academic research scientist.
We detected an underexplored combinatorial research gap in the literature between:
Methodology / Paradigm: "{axis_a}"
Application Domain: "{axis_b}"
Current empirical literature count in this exact intersection: {cell_count} papers.

Context of Method in other domains:
{context_a}

Context of Domain with other methods:
{context_b}

TASK:
Synthesize an actionable, inspiring, and clear research project briefing that a graduate student or researcher can understand in 30 seconds.
Respond ONLY with a valid JSON object matching this schema:
{{
  "project_title": "A clear, compelling academic paper title (e.g. 'Physics-Informed Surrogates for Real-Time Flight Dynamics in UAVs')",
  "core_research_question": "A single, crystal-clear hypothesis or question to be tested",
  "why_it_is_a_gap": "A 2-sentence plain-English explanation of why this is unstudied despite both sides having mature literature",
  "suggested_first_experiment": "A 2-step concrete experiment a researcher can run this week using existing open-source benchmarks",
  "practical_impact": "1 sentence describing the real-world scientific or industrial consequence of solving this gap"
}}
"""
        try:
            from src.llm.nvidia_client import NvidiaClient
            data = NvidiaClient.generate_json(prompt=prompt, temperature=settings.LLM_STRUCTURED_TEMPERATURE, max_tokens=1024)
            if data and isinstance(data, dict):
                required_keys = ["project_title", "core_research_question", "why_it_is_a_gap", "suggested_first_experiment"]
                if all(k in data for k in required_keys):
                    return data
        except Exception as e:
            logger.warning(f"NVIDIA LLM project synthesis failed: {e}. Falling back to domain heuristics.")

        return None

    @classmethod
    def _synthesize_with_heuristics(
        cls,
        axis_a: str,
        axis_b: str,
        papers_a: List[Dict[str, Any]],
        papers_b: List[Dict[str, Any]],
        cell_count: int
    ) -> Dict[str, str]:
        """Corpus-derived fallback that formats a readable proposal from neighboring evidence."""
        b_clean = axis_b.replace("&", "and").strip()
        a_clean = axis_a.replace("&", "and").strip()
        evidence_titles = [p.get("title", "") for p in (papers_a + papers_b) if p.get("title")]
        evidence_phrase = "; ".join(evidence_titles[:3]) if evidence_titles else "adjacent retrieved papers"

        project_title = f"{a_clean} for {b_clean}: A Corpus-Grounded Transfer Study"

        # Core research question
        core_research_question = (
            f"Can principles from '{a_clean}' be systematically integrated into '{b_clean}' "
            f"to improve accuracy, robustness, or computational efficiency compared to standard baselines?"
        )

        # Why it is a gap
        n_a = len(papers_a)
        n_b = len(papers_b)
        why_it_is_a_gap = (
            f"While '{a_clean}' is actively demonstrated across adjacent subfields ({n_a}+ active papers) "
            f"and '{b_clean}' possesses rich empirical benchmarks ({n_b}+ papers), only {cell_count} studies currently bridge the two. "
            f"The nearest evidence comes from: {evidence_phrase}."
        )

        # Suggested first experiment
        suggested_first_experiment = (
            f"Step 1: Benchmark a baseline model on standard {b_clean} public datasets. "
            f"Step 2: Inject {a_clean} formulation (e.g. inductive loss constraints or architectural priors) "
            f"and measure relative sample efficiency and out-of-distribution generalization."
        )

        practical_impact = (
            f"If validated, this would clarify whether evidence around {evidence_phrase} transfers into {b_clean}."
        )

        return {
            "project_title": project_title,
            "core_research_question": core_research_question,
            "why_it_is_a_gap": why_it_is_a_gap,
            "suggested_first_experiment": suggested_first_experiment,
            "practical_impact": practical_impact
        }
