"""
Adversarial Critique / Devil's Advocate Node.
Forces the system to identify the single strongest technical failure mode or theoretical bottleneck
for every suggested gap to eliminate confirmation bias and LLM sycophancy.
"""

import json
import logging
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
        Uses LLM if available; otherwise applies deterministic skepticism rules.
        """
        from src.llm.llm_router import LLMRouter
        if LLMRouter.is_available():
            llm_critique = cls._query_llm_critique(
                axis_a_val, axis_b_val, neighbor_papers_a, neighbor_papers_b
            )
            if llm_critique:
                return llm_critique

        # Deterministic domain-specific skepticism fallbacks
        b_lower = axis_b_val.lower()
        if "robot" in b_lower:
            return (
                f"Severe Control Loop Latency Barrier: Deploying '{axis_a_val}' in Autonomous Robotics typically "
                f"fails under high-frequency control demands (e.g. 50-500Hz). While '{axis_a_val}' achieves high offline accuracy, "
                f"iterative backward-pass or surrogate forward-pass latency exceeding 10ms will cause catastrophic instability during dynamic maneuvering."
            )
        elif "edge" in b_lower:
            return (
                f"Memory & Quantization Bottleneck: '{axis_a_val}' architectures generally rely on float32/float64 gradient "
                f"precision for numerical stability. Attempting to compress or quantize these models to INT8/INT4 for Low-Power Edge Systems "
                f"frequently destroys mathematical convergence and exceeds strictly budgeted SRAM limits."
            )
        elif "health" in b_lower or "clinic" in b_lower:
            return (
                f"Clinical Distribution Shift & Black-Box Regulatory Risk: The theoretical priors of '{axis_a_val}' assume "
                f"continuous, well-behaved distributions, whereas Healthcare & Clinical data is characterized by sparse patient cohorts, "
                f"extreme class imbalance, and strict FDA explainability mandates that resist black-box surrogate approximations."
            )
        elif "finan" in b_lower or "econ" in b_lower:
            return (
                f"Microstructure Non-Stationarity & Low Signal-to-Noise: Financial market dynamics violate the core stationarity "
                f"and conservation assumptions often embedded in '{axis_a_val}'. High-frequency arbitrage feedback loops and regime shifts "
                f"cause inductive models to overfit rapidly on spurious historical correlations."
            )
        else:
            return (
                f"Stiff Dynamics & Multiscale Error Accumulation: In {axis_b_val}, physical interactions span multiple orders of "
                f"magnitude in space and time. Standard loss formulations in '{axis_a_val}' struggle with stiff differential gradients, "
                f"leading to error accumulation and violation of fundamental conservation laws over extended horizons."
            )

    @classmethod
    def _query_llm_critique(
        cls,
        axis_a_val: str,
        axis_b_val: str,
        papers_a: List[Dict[str, Any]],
        papers_b: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Call LLM (Bedrock or NVIDIA) with adversarial reviewer prompt."""
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
            from src.llm.llm_router import LLMRouter
            critique = LLMRouter.generate(prompt=prompt, temperature=0.3, max_tokens=1024)
            if critique and len(critique.strip()) > 30:
                return critique.strip()
        except Exception as e:
            logger.warning(f"Adversarial LLM critique call failed: {e}")

        return None

