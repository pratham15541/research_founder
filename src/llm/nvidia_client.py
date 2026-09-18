"""
Unified NVIDIA NGC AI API Client.
Communicates with NVIDIA's hosted foundation models via OpenAI-compatible endpoints.
Supports standard JSON responses and Server-Sent Event (SSE) streaming.
"""

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Generator
import requests
from src.config import settings, BASE_DIR

logger = logging.getLogger(__name__)

class NvidiaClient:
    """OpenAI-compatible LLM client with env-configured provider fallback and JSON extraction."""

    _JSON_DECODER = json.JSONDecoder()
    _DOTENV_CACHE: Optional[Dict[str, str]] = None

    @classmethod
    def _dotenv_values(cls) -> Dict[str, str]:
        """Read .env values without requiring users to export numbered providers."""
        if cls._DOTENV_CACHE is not None:
            return cls._DOTENV_CACHE

        values: Dict[str, str] = {}
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            try:
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
            except Exception as exc:
                logger.warning("Failed to read numbered LLM providers from .env: %s", exc)
        cls._DOTENV_CACHE = values
        return values

    @classmethod
    def _config_value(cls, key: str) -> Optional[str]:
        if cls._DOTENV_CACHE is not None:
            return cls._DOTENV_CACHE.get(key)
        return os.environ.get(key) or cls._dotenv_values().get(key)

    @staticmethod
    def _completion_url(base_url: str) -> str:
        """Accept either provider base URL or full chat completions endpoint."""
        clean = base_url.rstrip("/")
        if clean.endswith("/chat/completions"):
            return clean
        return f"{clean}/chat/completions"

    @classmethod
    def _provider_chain(cls, override_model: Optional[str] = None) -> List[Dict[str, str]]:
        """Build provider fallback chain from NVIDIA_* and NVIDIA_*1..N env vars."""
        providers: List[Dict[str, str]] = []

        def add_provider(api_key: Optional[str], model_name: Optional[str], base_url: Optional[str]) -> None:
            if not api_key or not model_name or not base_url:
                return
            providers.append({
                "api_key": api_key.strip(),
                "model": (override_model or model_name).strip(),
                "base_url": base_url.strip()
            })

        add_provider(settings.NVIDIA_API_KEY, settings.NVIDIA_MODEL, settings.NVIDIA_BASE_URL)

        for idx in range(1, settings.LLM_PROVIDER_MAX + 1):
            add_provider(
                cls._config_value(f"NVIDIA_API_KEY{idx}") or cls._config_value(f"NVIDIA_API_KEY_{idx}"),
                cls._config_value(f"NVIDIA_MODEL{idx}") or cls._config_value(f"NVIDIA_MODEL_{idx}"),
                cls._config_value(f"NVIDIA_BASE_URL{idx}") or cls._config_value(f"NVIDIA_BASE_URL_{idx}")
            )

        # Deduplicate exact provider/model/url triplets while preserving order.
        seen = set()
        unique: List[Dict[str, str]] = []
        for provider in providers:
            key = (provider["api_key"], provider["model"], provider["base_url"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(provider)
        return unique

    @classmethod
    def _call_provider(
        cls,
        provider: Dict[str, str],
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float
    ) -> Optional[str]:
        """Execute a single API request against an OpenAI-compatible provider."""
        url = cls._completion_url(provider["base_url"])
        cand_model = provider["model"]
        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
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
                choices = data.get("choices", [])
                if not choices:
                    return None
                msg = choices[0].get("message", {})
                content = msg.get("content")
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")

                # Handle models where output is placed in reasoning or content is None/empty
                if (content is None or not str(content).strip()) and reasoning:
                    content = reasoning
                elif content is None:
                    content = ""

                # Strip reasoning tags <think>...</think> if model generates chain-of-thought
                clean = re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL).strip()
                return clean if clean else str(content).strip()
            elif res.status_code == 429:
                logger.warning("LLM provider rate-limited (429) for model '%s' at %s. Trying next configured provider.", cand_model, url)
                return None
            elif res.status_code == 404:
                logger.warning("LLM model '%s' not found at %s. Trying next configured provider.", cand_model, url)
                return None
            else:
                logger.warning("LLM provider error for model '%s' at %s: %s %s", cand_model, url, res.status_code, res.text[:200])
                return None
        except Exception as e:
            logger.warning("LLM call with model '%s' failed or timed out: %s", cand_model, e)
            return None

    @classmethod
    def generate(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        """Generate text completion from the configured OpenAI-compatible provider chain."""
        configured_timeout = float(getattr(settings, "LLM_TIMEOUT_SECONDS", 45.0))
        call_timeout = max(float(timeout or 0), configured_timeout)
        providers = cls._provider_chain(override_model=model)
        if not providers:
            if settings.LLM_REQUIRED:
                raise RuntimeError("No LLM providers configured; set NVIDIA_API_KEY/NVIDIA_MODEL/NVIDIA_BASE_URL or numbered variants.")
            logger.warning("No LLM providers configured. Skipping LLM generation.")
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for provider in providers:
            result = cls._call_provider(
                provider=provider,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=call_timeout
            )
            if result:
                return result

        return None

    @classmethod
    def _try_parse_json(cls, text: str) -> Optional[Any]:
        """Attempt multiple extraction and parsing strategies for LLM-generated JSON."""
        if not text:
            return None
        clean = text.strip()

        # 1. Check for markdown code blocks ```json ... ``` or ``` ... ```
        code_block_match = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", clean, re.DOTALL)
        if code_block_match:
            snippet = code_block_match.group(1).strip()
            fixed_snippet = re.sub(r",\s*([\]}])", r"\1", snippet)
            try:
                return json.loads(fixed_snippet)
            except Exception:
                try:
                    return json.loads(snippet)
                except Exception:
                    pass

        # 2. Attempt direct parse
        try:
            return json.loads(clean)
        except Exception:
            pass

        # 3. Clean standard codeblock markers and retry
        if "```" in clean:
            clean_blocks = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
            clean_blocks = re.sub(r"\n?```$", "", clean_blocks).strip()
            try:
                return json.loads(clean_blocks)
            except Exception:
                pass

        # 4. Extract the first balanced JSON object/list embedded in model output
        return cls._parse_embedded_json(clean)

    @classmethod
    def generate_json(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.35,
        max_tokens: int = 4096,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """Generate structured JSON (object or list) with robust parsing and provider-chain fallback."""
        full_prompt = (
            f"{prompt}\n\n"
            "CRITICAL: Return valid JSON ONLY. "
            "Do not include conversational preamble, apologies, or trailing notes. "
            "Ensure the JSON is strictly parsable."
        )

        configured_timeout = float(getattr(settings, "LLM_TIMEOUT_SECONDS", 45.0))
        call_timeout = max(float(timeout or 0), configured_timeout)

        raw_text = cls.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=call_timeout
        )
        if not raw_text:
            return None

        parsed = cls._try_parse_json(raw_text)
        if parsed is not None:
            return parsed

        logger.warning("Failed to parse valid JSON from LLM response.")
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
        timeout: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Stream SSE chunks from NVIDIA API for interactive RAG research assistant."""
        call_timeout = timeout if timeout is not None else getattr(settings, "LLM_TIMEOUT_SECONDS", 90.0)
        providers = cls._provider_chain(override_model=model)
        if not providers:
            yield "No LLM provider is configured."
            return

        provider = providers[0]
        cand_model = provider["model"]
        url = cls._completion_url(provider["base_url"])
        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
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
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=call_timeout) as r:
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data: ") and not decoded.startswith("data: [DONE]"):
                                try:
                                    chunk = json.loads(decoded[6:])
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content_piece = delta.get("content")
                                        if content_piece:
                                            yield content_piece
                                except Exception:
                                    pass
                else:
                    yield f"[NVIDIA API returned error {r.status_code}]"
        except Exception as e:
            yield f"[Streaming error: {e}]"
