"""
Embedding generation using SentenceTransformers with lightweight fallback.
Produces 384-dimensional dense vectors for titles and abstracts.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Computes dense vector representations for academic papers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _load_model(self):
        """Lazy load SentenceTransformers model to conserve memory."""
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic fallback.")
                self._model = None
            self._initialized = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode list of strings into 384-dimensional normalized vectors."""
        self._load_model()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return np.array(embeddings, dtype=np.float32)

        # Deterministic lightweight fallback (e.g. for testing environments without PyTorch)
        logger.info("Using deterministic hash-based dense embedding fallback.")
        fallback_vecs = []
        for text in texts:
            # Deterministic hash seed based on text content
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            vec = rng.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            fallback_vecs.append(vec)
        return np.array(fallback_vecs, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single string into a 1D normalized vector."""
        return self.embed_texts([text])[0]

    def embed_paper_content(self, title: str, abstract: str) -> List[float]:
        """Encode a single paper title and abstract."""
        combined = f"{title.strip()} [SEP] {abstract.strip()}"
        res = self.embed_texts([combined])
        return res[0].tolist()


def get_embedding_engine(model_name: Optional[str] = None):
    """
    Factory function to return BedrockEmbeddingEngine (if USE_AWS=true)
    or local SentenceTransformer EmbeddingEngine.
    """
    from src.config import settings
    if settings.USE_AWS:
        try:
            from src.representation.bedrock_embeddings import BedrockEmbeddingEngine
            return BedrockEmbeddingEngine()
        except Exception as e:
            logger.warning(f"Could not load BedrockEmbeddingEngine ({e}). Falling back to local.")
    return EmbeddingEngine(model_name or settings.EMBEDDING_MODEL_NAME)

