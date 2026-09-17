"""
Unified LLM Router.
Auto-selects the best available LLM provider:
  1. AWS Bedrock (if USE_AWS=true and credentials present)
  2. NVIDIA NGC / OpenRouter (if NVIDIA_API_KEY is set)
  3. None (deterministic offline fallback — callers handle gracefully)
"""

import logging
from typing import List, Dict, Any, Optional, Generator

from src.config import settings

logger = logging.getLogger(__name__)

# Provider identifiers
_PROVIDER_BEDROCK = "bedrock"
_PROVIDER_NVIDIA = "nvidia"
_PROVIDER_NONE = "none"


def _detect_provider() -> str:
    """Detect which LLM provider to use based on configuration."""
    if settings.is_aws_enabled():
        return _PROVIDER_BEDROCK
    if settings.NVIDIA_API_KEY:
        return _PROVIDER_NVIDIA
    return _PROVIDER_NONE


class LLMRouter:
    """Unified facade that delegates to the best available LLM provider."""

    @classmethod
    def get_active_provider(cls) -> str:
        """Return the name of the active LLM provider."""
        return _detect_provider()

    @classmethod
    def is_available(cls) -> bool:
        """Check whether any LLM provider is configured."""
        return _detect_provider() != _PROVIDER_NONE

    @classmethod
    def generate(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 35.0,
    ) -> Optional[str]:
        """Generate text using the best available provider."""
        provider = _detect_provider()

        if provider == _PROVIDER_BEDROCK:
            from src.llm.bedrock_client import BedrockClient
            return BedrockClient.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        if provider == _PROVIDER_NVIDIA:
            from src.llm.nvidia_client import NvidiaClient
            return NvidiaClient.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        logger.info("No LLM provider configured. Returning None.")
        return None

    @classmethod
    def generate_json(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 35.0,
    ) -> Optional[Any]:
        """Generate structured JSON via the best available provider."""
        provider = _detect_provider()

        if provider == _PROVIDER_BEDROCK:
            from src.llm.bedrock_client import BedrockClient
            return BedrockClient.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        if provider == _PROVIDER_NVIDIA:
            from src.llm.nvidia_client import NvidiaClient
            return NvidiaClient.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        return None

    @classmethod
    def stream_chat(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 45.0,
    ) -> Generator[str, None, None]:
        """Stream chat tokens from the best available provider."""
        provider = _detect_provider()

        if provider == _PROVIDER_BEDROCK:
            from src.llm.bedrock_client import BedrockClient
            yield from BedrockClient.stream_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return

        if provider == _PROVIDER_NVIDIA:
            from src.llm.nvidia_client import NvidiaClient
            yield from NvidiaClient.stream_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return

        yield "No LLM provider is configured. Please set USE_AWS=true or provide NVIDIA_API_KEY."
