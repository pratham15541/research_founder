"""
Unified NVIDIA NGC AI API Client.
Communicates with NVIDIA's hosted foundation models via OpenAI-compatible endpoints.
Supports standard JSON responses and Server-Sent Event (SSE) streaming.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Generator
import requests
from src.config import settings

logger = logging.getLogger(__name__)

class NvidiaClient:
    """Production client for NVIDIA NGC API models with automatic fallback and JSON extraction."""

    FAST_FALLBACK_MODEL = "meta/llama-3.2-11b-vision-instruct"
    _JSON_DECODER = json.JSONDecoder()

    @classmethod
    def generate(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 35.0
    ) -> Optional[str]:
        """Generate text completion from NVIDIA API."""
        api_key = settings.NVIDIA_API_KEY
        if not api_key:
            logger.warning("NVIDIA_API_KEY not set. Skipping LLM generation.")
            return None

        primary_model = model or settings.NVIDIA_MODEL
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Model trial chain: primary model -> fast fallback
        models_to_try = [primary_model]
        if primary_model != cls.FAST_FALLBACK_MODEL:
            models_to_try.append(cls.FAST_FALLBACK_MODEL)

        url = f"{settings.NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        for cand_model in models_to_try:
            payload = {
                "model": cand_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    # Strip reasoning tags <think>...</think> if model generates chain-of-thought
                    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return clean if clean else content.strip()
                elif res.status_code == 404:
                    logger.warning(f"Model '{cand_model}' not found for this account. Trying fallback...")
                    continue
                else:
                    logger.warning(f"NVIDIA API ({cand_model}) error {res.status_code}: {res.text[:200]}")
            except Exception as e:
                logger.warning(f"NVIDIA API call with '{cand_model}' failed or timed out: {e}")
                continue

        return None

    @classmethod
    def generate_json(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 35.0
    ) -> Optional[Any]:
        """Generate structured JSON (object or list) via NVIDIA API with robust regex sanitization."""
        full_prompt = (
            f"{prompt}\n\n"
            "CRITICAL: Return valid JSON ONLY. "
            "Do not include conversational preamble, apologies, or trailing notes. "
            "Ensure the JSON is strictly parsable."
        )

        raw_text = cls.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        if not raw_text:
            return None

        # Clean markdown codeblocks
        clean = raw_text.strip()
        if "```" in clean:
            clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        # Attempt direct parse
        try:
            return json.loads(clean)
        except Exception:
            pass

        # Extract the first balanced JSON object/list. This is more reliable than
        # greedy regex when model output contains prose, examples, or extra braces.
        parsed = cls._parse_embedded_json(clean)
        if parsed is not None:
            return parsed

        logger.warning("Failed to parse JSON from NVIDIA API response.")
        return None

    @classmethod
    def _parse_embedded_json(cls, text: str) -> Optional[Any]:
        """Parse the first valid JSON object/list embedded in model output."""
        for idx, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = cls._JSON_DECODER.raw_decode(text[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        return None

    @classmethod
    def stream_chat(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 45.0
    ) -> Generator[str, None, None]:
        """Stream SSE chunks from NVIDIA API for interactive RAG research assistant."""
        api_key = settings.NVIDIA_API_KEY
        if not api_key:
            yield "NVIDIA_API_KEY is not configured."
            return

        cand_model = model or settings.NVIDIA_MODEL
        url = f"{settings.NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }
        payload = {
            "model": cand_model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data: ") and not decoded.startswith("data: [DONE]"):
                                try:
                                    chunk = json.loads(decoded[6:])
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    pass
                else:
                    yield f"[NVIDIA API returned error {r.status_code}]"
        except Exception as e:
            yield f"[Streaming error: {e}]"
