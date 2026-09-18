"""
Adversarial Critique / Devil's Advocate Node.
Forces the system to identify the single strongest technical failure mode or theoretical bottleneck
for every suggested gap to eliminate confirmation bias and LLM sycophancy.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class DevilsAdvocateNode:
    """Generates rigorous counter-arguments and failure modes for proposed research opportunities."""

    @classmethod
    def generate_critique(
        cls,
        axis_a_val: str,
        axis_b_val: str,
        neighbor_papers_a: List[Dict[str, Any]],
        neighbor_papers_b: List[Dict[str, Any]]
    ) -> str:
        """
        Generate the strongest skeptical counter-argument why the combination might be a dead end.
        Uses LLM if available; otherwise derives a skeptical argument from neighboring evidence.
        """
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            llm_critique = cls._query_llm_critique(
                axis_a_val, axis_b_val, neighbor_papers_a, neighbor_papers_b
            )
            if llm_critique:
                return llm_critique
        elif settings.LLM_REQUIRED:
            raise RuntimeError("NVIDIA_API_KEY is required because LLM_REQUIRED=true.")

        evidence_terms = cls._evidence_terms(neighbor_papers_a + neighbor_papers_b)
        evidence_hint = ", ".join(evidence_terms[:5]) if evidence_terms else "the neighboring evidence"
        return (
            f"Evidence-transfer risk: '{axis_a_val}' and '{axis_b_val}' are supported only indirectly through neighboring papers "
            f"centered on {evidence_hint}. The main failure mode is that assumptions validated in those adjacent settings may not "
            f"survive the exact intersection, so the first experiment must test transferability rather than treating the gap as validated."
        )

    @staticmethod
    def _evidence_terms(papers: List[Dict[str, Any]]) -> List[str]:
        text = " ".join(f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers)
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text)
        stop = {"using", "based", "model", "models", "paper", "study", "analysis", "method", "methods", "results"}
        terms: List[str] = []
        for token in tokens:
            cleaned = token.lower()
            if cleaned in stop:
                continue
            titled = cleaned.replace("-", " ").title()
            if titled not in terms:
                terms.append(titled)
        return terms

    @classmethod
    def _query_llm_critique(
        cls,
        axis_a_val: str,
        axis_b_val: str,
        papers_a: List[Dict[str, Any]],
        papers_b: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Call NVIDIA AI API with adversarial reviewer prompt."""
        context_a = "\n".join([f"- {p.get('title')}" for p in papers_a[:3]])
        context_b = "\n".join([f"- {p.get('title')}" for p in papers_b[:3]])

        prompt = f"""You are a skeptical, adversarial senior academic peer reviewer.
A researcher proposes an untried combination between:
Dimension A: "{axis_a_val}"
Dimension B: "{axis_b_val}"

Known papers using Dimension A in other contexts:
{context_a}

Known papers in Dimension B using other methods:
{context_b}

TASK:
Do NOT praise this idea. Provide the single strongest technical, mathematical, or empirical reason why this direction might FAIL, be nonsensical, or be an unviable dead end. Focus on computational bottlenecks, data mismatches, or violation of core theoretical assumptions.

Keep your critique concise (2-3 sentences), highly specific, and authoritative. Return plain text only.
"""
        try:
            from src.llm.nvidia_client import NvidiaClient
            critique = NvidiaClient.generate(prompt=prompt, temperature=settings.LLM_STRUCTURED_TEMPERATURE, max_tokens=1024)
            if critique and len(critique.strip()) > 30:
                return critique.strip()
        except Exception as e:
            logger.warning(f"Adversarial NVIDIA LLM critique call failed: {e}")

        return None
