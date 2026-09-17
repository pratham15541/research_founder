"""
AI-Powered Research Question & Hypothesis Generator.
Formulates testable scientific research questions, operational variables, and experimental protocols.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class ResearchQuestionGenerator:
    """Generates structured, evidence-backed research questions and hypothesis protocols for detected gaps."""

    @classmethod
    def generate_questions_for_gap(
        cls,
        gap: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate formal hypotheses and experimental variable protocols for a gap."""
        if settings.NVIDIA_API_KEY:
            llm_questions = cls._generate_with_llm(gap)
            if llm_questions:
                return llm_questions

        return cls._generate_fallback(gap)

    @classmethod
    def _generate_with_llm(cls, gap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query NVIDIA AI API for formal hypothesis structure."""
        title = gap.get("project_title", f"{gap.get('axis_a')} in {gap.get('axis_b')}")
        a = gap.get("axis_a")
        b = gap.get("axis_b")
        why = gap.get("why_it_is_a_gap", "")

        prompt = f"""You are a scientific methodologist formulating a grant-winning research plan.
Research Opportunity: "{title}"
Method Paradigm: "{a}"
Application Domain: "{b}"
Context: {why}

TASK:
Formulate a rigorous research protocol and return a JSON object matching this exact structure:
{{
  "primary_research_question": "Exact scientific research question",
  "primary_hypothesis_h1": "Directional hypothesis (H1)",
  "null_hypothesis_h0": "Falsifiable null hypothesis (H0)",
  "variables": {{
    "independent": "Primary variable manipulated",
    "dependent": "Primary metrics measured",
    "control": "Key environmental or algorithmic constants maintained"
  }},
  "experimental_phases": [
    "Phase 1: Baseline dataset preparation & calibration",
    "Phase 2: Algorithmic integration & model training",
    "Phase 3: Stress-testing & ablation evaluation"
  ],
  "expected_contributions": "Key scientific contribution of this study"
}}
"""
        try:
            from src.llm.nvidia_client import NvidiaClient
            data = NvidiaClient.generate_json(prompt=prompt, temperature=0.25, max_tokens=1024)
            if data and isinstance(data, dict):
                if "primary_research_question" in data and "primary_hypothesis_h1" in data:
                    return data
        except Exception as e:
            logger.warning(f"NVIDIA LLM question generation failed: {e}. Using structured fallback.")

        return None

    @classmethod
    def _generate_fallback(cls, gap: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic scientific question formulation."""
        title = gap.get("project_title", f"{gap.get('axis_a')} in {gap.get('axis_b')}")
        a = gap.get("axis_a", "Methodology")
        b = gap.get("axis_b", "Application Domain")

        return {
            "primary_research_question": f"To what extent does incorporating '{a}' into '{b}' enhance prediction accuracy and sample efficiency over conventional baselines?",
            "primary_hypothesis_h1": f"The integration of '{a}' will reduce test error by at least 15% and improve out-of-distribution robustness compared to standard purely data-driven models in {b}.",
            "null_hypothesis_h0": f"There is no statistically significant performance difference between '{a}' and baseline architectures when evaluated on benchmark {b} tasks.",
            "variables": {
                "independent": f"Incorporation of {a} architectural inductive biases and loss constraints",
                "dependent": f"Generalization accuracy, mean squared error, and computational latency",
                "control": f"Training dataset split, optimization hyperparameters, and hardware environment"
            },
            "experimental_phases": [
                f"Phase 1: Establish benchmark performance using standard baseline models on open {b} datasets.",
                f"Phase 2: Implement and train the {a} model incorporating governing domain constraints.",
                f"Phase 3: Conduct ablation studies comparing sample efficiency, convergence rates, and boundary error."
            ],
            "expected_contributions": f"First systematic empirical study demonstrating the cross-domain viability of {a} in {b} with open-source benchmark reproduction code."
        }

