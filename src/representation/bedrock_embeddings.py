"""
AWS Bedrock Titan Embedding Engine.
Uses Amazon Titan Embed Text v2 for 1024-dimensional dense embeddings.
Same interface as EmbeddingEngine for drop-in replacement.
"""

import logging
import json
from typing import List, Dict, Any, Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class BedrockEmbeddingEngine:
    """Computes dense embeddings via AWS Bedrock Titan Embed Text model."""

    def __init__(self, model_id: Optional[str] = None, dimension: Optional[int] = None):
        self.model_id = model_id or settings.AWS_BEDROCK_EMBEDDING_MODEL_ID
        # Titan v2 natively generates 1024-dimensional vectors unless configured otherwise (supports 256, 512, 1024)
        if dimension is not None:
            self.dimension = dimension
        elif "titan" in self.model_id.lower() and settings.EMBEDDING_DIMENSION == 384:
            # 384 is MiniLM default; if Titan is used without explicit dimension, default to 1024
            self.dimension = 1024
        else:
            self.dimension = settings.EMBEDDING_DIMENSION
        self._client = None

    def _get_client(self):
        """Lazy-init boto3 Bedrock Runtime client."""
        if self._client is None:
            import boto3

            kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": settings.AWS_REGION,
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

            self._client = boto3.client(**kwargs)
            logger.info(f"Bedrock embedding client initialized (model={self.model_id})")

        return self._client

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode list of strings into dense normalized vectors via Bedrock Titan."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings = []
        client = self._get_client()

        for text in texts:
            try:
                request_payload: Dict[str, Any] = {
                    "inputText": text[:8192],  # Titan limit
                }
                if "titan" in self.model_id.lower() and self.dimension in (256, 512, 1024):
                    request_payload["dimensions"] = self.dimension
                    request_payload["normalize"] = True

                body = json.dumps(request_payload)

                response = client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )

                response_body = json.loads(response["body"].read())
                embedding = response_body.get("embedding", [])

                if embedding:
                    vec = np.array(embedding, dtype=np.float32)
                    if vec.shape[0] != self.dimension:
                        self.dimension = vec.shape[0]
                    # Normalize
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    all_embeddings.append(vec)
                else:
                    # Zero vector fallback
                    all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))

            except Exception as e:
                logger.warning(f"Bedrock embedding failed for text (len={len(text)}): {e}")
                all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))

        return np.array(all_embeddings, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single string."""
        return self.embed_texts([text])[0]

    def embed_paper_content(self, title: str, abstract: str) -> List[float]:
        """Encode a single paper title and abstract."""
        combined = f"{title.strip()} [SEP] {abstract.strip()}"
        res = self.embed_texts([combined])
        return res[0].tolist()
