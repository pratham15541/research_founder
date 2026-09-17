"""
AWS Bedrock LLM Client.
Provides the same interface as NvidiaClient (generate, generate_json, stream_chat)
but routes all calls through Amazon Bedrock Runtime via boto3.
Supports Claude 3, Titan Text, and Llama 3 models.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Generator

from src.config import settings

logger = logging.getLogger(__name__)


class BedrockClient:
    """Production client for AWS Bedrock models with automatic fallback and JSON extraction."""

    _JSON_DECODER = json.JSONDecoder()

    @classmethod
    def _get_bedrock_client(cls, timeout: Optional[float] = None):
        """Create a boto3 Bedrock Runtime client, pointing to LocalStack when configured."""
        import boto3
        from botocore.config import Config

        cfg_kwargs: Dict[str, Any] = {"retries": {"max_attempts": 2}}
        if timeout is not None:
            cfg_kwargs["connect_timeout"] = 10
            cfg_kwargs["read_timeout"] = int(timeout)

        kwargs: Dict[str, Any] = {
            "service_name": "bedrock-runtime",
            "region_name": settings.AWS_REGION,
            "config": Config(**cfg_kwargs),
        }
        endpoint = settings.get_aws_endpoint_url()
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        if settings.AWS_ACCESS_KEY_ID:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        if settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

        return boto3.client(**kwargs)

    @classmethod
    def _is_claude_model(cls, model_id: str) -> bool:
        return "claude" in model_id.lower() or "anthropic" in model_id.lower()

    @classmethod
    def _is_titan_model(cls, model_id: str) -> bool:
        return "titan" in model_id.lower()

    @classmethod
    def _build_request_body(
        cls,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Build the model-specific request body for Bedrock invoke_model."""
        if cls._is_claude_model(model_id):
            # Anthropic Claude Messages API format
            system_msg = None
            user_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_msgs.append({"role": m["role"], "content": m["content"]})

            body: Dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_msgs,
            }
            if system_msg:
                body["system"] = system_msg
            return body

        elif cls._is_titan_model(model_id):
            # Amazon Titan Text format
            prompt_parts = []
            for m in messages:
                if m["role"] == "system":
                    prompt_parts.append(f"System: {m['content']}")
                elif m["role"] == "user":
                    prompt_parts.append(f"User: {m['content']}")
                else:
                    prompt_parts.append(f"Assistant: {m['content']}")
            prompt_parts.append("Assistant:")

            return {
                "inputText": "\n\n".join(prompt_parts),
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9,
                },
            }

        else:
            # Meta Llama / generic — use Messages API format (Bedrock Converse)
            prompt_parts = []
            for m in messages:
                prompt_parts.append(f"<|{m['role']}|>\n{m['content']}")
            prompt_parts.append("<|assistant|>")

            return {
                "prompt": "\n".join(prompt_parts),
                "max_gen_len": max_tokens,
                "temperature": temperature,
            }

    @classmethod
    def _extract_response_text(cls, model_id: str, response_body: Dict[str, Any]) -> Optional[str]:
        """Extract the generated text from model-specific response format."""
        try:
            if cls._is_claude_model(model_id):
                content_blocks = response_body.get("content", [])
                texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                return "\n".join(texts).strip() if texts else None

            elif cls._is_titan_model(model_id):
                results = response_body.get("results", [])
                if results:
                    return results[0].get("outputText", "").strip()
                return None

            else:
                # Llama / generic
                return response_body.get("generation", "").strip()

        except Exception as e:
            logger.warning(f"Failed to extract text from Bedrock response: {e}")
            return None

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
        """Generate text completion from AWS Bedrock."""
        primary_model = model or settings.AWS_BEDROCK_MODEL_ID
        fallback_model = settings.AWS_BEDROCK_FALLBACK_MODEL_ID

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        models_to_try = [primary_model]
        if primary_model != fallback_model:
            models_to_try.append(fallback_model)

        for cand_model in models_to_try:
            try:
                client = cls._get_bedrock_client(timeout=timeout)
                body = cls._build_request_body(cand_model, messages, temperature, max_tokens)

                response = client.invoke_model(
                    modelId=cand_model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )

                response_body = json.loads(response["body"].read())
                content = cls._extract_response_text(cand_model, response_body)

                if content:
                    # Strip reasoning tags <think>...</think> if model generates chain-of-thought
                    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return clean if clean else content.strip()

            except Exception as e:
                logger.warning(f"Bedrock API call with '{cand_model}' failed: {e}")
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
        timeout: float = 35.0,
    ) -> Optional[Any]:
        """Generate structured JSON via Bedrock with robust extraction."""
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
            timeout=timeout,
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

        # Extract the first balanced JSON object/list
        parsed = cls._parse_embedded_json(clean)
        if parsed is not None:
            return parsed

        logger.warning("Failed to parse JSON from Bedrock API response.")
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
        timeout: float = 45.0,
    ) -> Generator[str, None, None]:
        """Stream text from Bedrock using invoke_model_with_response_stream."""
        cand_model = model or settings.AWS_BEDROCK_MODEL_ID

        try:
            client = cls._get_bedrock_client(timeout=timeout)
            body = cls._build_request_body(cand_model, messages, temperature, max_tokens)

            response = client.invoke_model_with_response_stream(
                modelId=cand_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk["bytes"])
                        # Claude format
                        if "delta" in chunk_data:
                            text = chunk_data["delta"].get("text", "")
                            if text:
                                yield text
                        # Titan format
                        elif "outputText" in chunk_data:
                            yield chunk_data["outputText"]
                        # Generic
                        elif "generation" in chunk_data:
                            yield chunk_data["generation"]

        except Exception as e:
            logger.warning(f"Bedrock streaming failed for '{cand_model}': {e}")
            # Fall back to non-streaming
            try:
                result = cls.generate(
                    prompt=messages[-1]["content"] if messages else "",
                    system_prompt=next((m["content"] for m in messages if m["role"] == "system"), None),
                    model=cand_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if result:
                    yield result
                else:
                    yield f"[Bedrock streaming error: {e}]"
            except Exception as e2:
                yield f"[Bedrock error: {e2}]"
