"""
Stage 7 — Compatibility Reasoning.

Answers: "Is applying the proposed method (axis_a) to the proposed domain (axis_b)
scientifically coherent?" This prevents absurd gaps like "apply a vision-only model
to audio data" from passing validation.

Two-tier evaluation
-------------------
1. LLM call (if available): structured JSON verdict
2. Heuristic rule engine:  modality matrix + complexity constraints

Output per candidate
--------------------
{
  "compatibility_score": float 0–1,
  "compatibility_verdict": "COMPATIBLE" | "CONDITIONAL" | "INCOMPATIBLE",
  "compatibility_reasoning": str,
  "known_barriers": [str, ...],
  "rejection_reason": None | "INCOMPATIBLE"
}
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic modality incompatibility rules
# Each entry: (method_keywords, domain_keywords, barrier_message)
# ---------------------------------------------------------------------------
_INCOMPATIBILITY_RULES = [
    (
        {"vision", "image", "pixel", "convolutional", "cnn", "visual"},
        {"audio", "speech", "acoustic", "waveform", "sound"},
        "Vision-only architectures lack temporal 1D inductive bias required for audio signals.",
    ),
    (
        {"audio", "speech", "acoustic"},
        {"image", "pixel", "visual", "computer vision", "segmentation"},
        "Audio-domain models operate on 1D waveforms and cannot directly process 2D image tensors.",
    ),
    (
        {"quadratic", "transformer", "attention"},
        {"real-time", "edge", "embedded", "iot", "fpga", "mobile"},
        "Quadratic-complexity attention mechanisms exceed memory/latency budgets on edge hardware.",
    ),
    (
        {"symbolic", "logic", "prolog", "rule-based"},
        {"continuous", "regression", "physical simulation", "pde", "fluid"},
        "Symbolic rule-based systems cannot naturally represent continuous differential operators.",
    ),
    (
        {"graph neural", "gnn", "graph network"},
        {"tabular", "structured data", "flat feature"},
        "Graph NNs require relational edge structure absent in flat tabular data.",
    ),
]

# Known technical barriers (method → barrier)
_METHOD_BARRIERS = {
    "transformer": ["quadratic memory scaling with sequence length", "requires large pre-training corpora"],
    "diffusion": ["slow iterative inference", "high memory for high-resolution outputs"],
    "physics-informed": ["requires known governing equations", "sensitive to collocation point placement"],
    "reinforcement learning": ["sample-inefficient exploration", "requires simulator or online environment"],
    "federated": ["communication overhead", "statistical heterogeneity across clients"],
}


class CompatibilityReasoner:
    """Evaluates method–domain compatibility (Stage 7)."""

    @classmethod
    def reason_corpus(
        cls,
        candidate_gaps: List[Dict[str, Any]],
        structured_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run Stage 7 over all candidates.
        Attaches `compatibility` dict to each candidate in-place.
        """
        for cand in candidate_gaps:
            result = cls.reason_candidate(cand, structured_records)
            cand["compatibility"] = result
            if result["rejection_reason"]:
                cand.setdefault("pipeline_rejections", []).append(result["rejection_reason"])
        return candidate_gaps

    @classmethod
    def reason_candidate(
        cls,
        candidate: Dict[str, Any],
        structured_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess compatibility of method–domain pair for a single candidate."""
        axis_a: str = candidate.get("axis_a", "")
        axis_b: str = candidate.get("axis_b", "")

        # Try LLM reasoning first
        llm_result = cls._llm_compatibility(axis_a, axis_b, candidate)
        if llm_result:
            return llm_result

        # Fallback: heuristic rule engine
        return cls._heuristic_compatibility(axis_a, axis_b)

    # ------------------------------------------------------------------ #
    # LLM path
    # ------------------------------------------------------------------ #

    @classmethod
    def _llm_compatibility(
        cls,
        axis_a: str,
        axis_b: str,
        candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Query LLM for structured compatibility verdict."""
        try:
            from src.config import settings
            if not settings.DYNAMIC_LLM_ENABLED or not settings.NVIDIA_API_KEY:
                return None
            from src.llm.nvidia_client import NvidiaClient

            prompt = f"""You are an expert research methodology evaluator.

Evaluate whether applying method/approach "{axis_a}" to domain "{axis_b}" is scientifically compatible.

Return a JSON object with this exact schema:
{{
  "compatibility_verdict": "COMPATIBLE or CONDITIONAL or INCOMPATIBLE",
  "compatibility_score": <float 0.0 to 1.0>,
  "compatibility_reasoning": "<2-sentence explanation>",
  "known_barriers": ["<barrier 1>", "<barrier 2>"],
  "input_format_compatible": true or false,
  "assumption_conflicts": ["<conflict if any>"]
}}

Rules:
- COMPATIBLE (score 0.7–1.0): method inputs match domain data modality; no fundamental conflicts
- CONDITIONAL (score 0.4–0.69): technically feasible but requires adapters, reformulation, or architecture changes
- INCOMPATIBLE (score 0.0–0.39): fundamental mismatch in data modality, complexity, or assumptions"""

            data = NvidiaClient.generate_json(
                prompt=prompt,
                temperature=0.2,
                max_tokens=512,
                timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 35.0),
            )
            if data and isinstance(data, dict) and "compatibility_verdict" in data:
                verdict = str(data.get("compatibility_verdict", "CONDITIONAL")).upper()
                score = float(data.get("compatibility_score", 0.6))
                rejection_reason = "INCOMPATIBLE" if verdict == "INCOMPATIBLE" else None
                return {
                    "compatibility_score": round(min(1.0, max(0.0, score)), 3),
                    "compatibility_verdict": verdict,
                    "compatibility_reasoning": str(data.get("compatibility_reasoning", "")),
                    "known_barriers": list(data.get("known_barriers", [])),
                    "rejection_reason": rejection_reason,
                    "source": "llm",
                }
        except Exception as exc:
            logger.debug("CompatibilityReasoner LLM call failed (%s); using heuristics.", exc)
        return None

    # ------------------------------------------------------------------ #
    # Heuristic path
    # ------------------------------------------------------------------ #

    @classmethod
    def _heuristic_compatibility(cls, axis_a: str, axis_b: str) -> Dict[str, Any]:
        """Rule-engine compatibility checker."""
        """Data-driven compatibility evaluator."""
        a_lower = axis_a.lower()
        b_lower = axis_b.lower()

        # Check hard incompatibility rules
        # 1. Modality Conflict Detection
        for method_kw, domain_kw, barrier_msg in _INCOMPATIBILITY_RULES:
            a_hit = any(kw in a_lower for kw in method_kw)
            b_hit = any(kw in domain_kw for kw in domain_kw) if not a_hit else any(kw in b_lower for kw in domain_kw)
            b_hit = any(kw in b_lower for kw in domain_kw)
            if a_hit and b_hit:
                logger.debug("CompatibilityReasoner: INCOMPATIBLE — %s", barrier_msg)
                return {
                    "compatibility_score": 0.15,
                    "compatibility_verdict": "INCOMPATIBLE",
                    "compatibility_reasoning": barrier_msg,
                    "known_barriers": [barrier_msg],
                    "rejection_reason": "INCOMPATIBLE",
                    "source": "heuristic",
                }

        # Collect known barriers for method
        # 2. Known Architecture / Methodology Friction
        barriers: List[str] = []
        for key, barrier_list in _METHOD_BARRIERS.items():
            if key in a_lower:
                barriers.extend(barrier_list)

        # Check for "real-time" or "edge" constraints conflicting with heavy models
        # 3. Resource / Hardware Constraint Check
        heavy_keywords = {"transformer", "diffusion", "large language", "foundation model", "gpt", "bert"}
        edge_keywords = {"real-time", "edge", "embedded", "mobile", "iot", "low-latency"}
        is_heavy = any(kw in a_lower for kw in heavy_keywords)
        is_edge = any(kw in b_lower for kw in edge_keywords)

        if is_heavy and is_edge:
            return {
                "compatibility_score": 0.35,
                "compatibility_verdict": "CONDITIONAL",
                "compatibility_reasoning": (
                    f"Applying '{axis_a}' to '{axis_b}' is conditionally feasible — requires "
                    "quantization, distillation, or model compression to meet edge resource budgets."
                ),
                "known_barriers": barriers + ["Requires model compression for edge deployment"],
                "rejection_reason": None,
                "source": "heuristic",
            }

        # 4. Continuous Compatibility Score
        penalty = min(0.30, len(barriers) * 0.10)
        score = max(0.40, round(0.85 - penalty, 2))
        verdict = "COMPATIBLE" if score >= 0.70 else "CONDITIONAL"
        reasoning = (
            f"'{axis_a}' is broadly applicable to '{axis_b}'; no fundamental modality mismatch detected."
            if not barriers
            else f"'{axis_a}' can be applied to '{axis_b}' with consideration for: {'; '.join(barriers[:2])}."
        )

        return {
            "compatibility_score": score,
            "compatibility_verdict": verdict,
            "compatibility_reasoning": reasoning,
            "known_barriers": barriers[:3],
            "rejection_reason": None,
            "source": "heuristic",
        }

