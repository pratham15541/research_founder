"""
Embedding generation using SentenceTransformers with lightweight fallback.
Produces 384-dimensional dense vectors for titles and abstracts.
"""

import logging
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

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
                logger.warning(f"Could not load SentenceTransformer ({e}). Using content-based fallback.")
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

        # Lightweight content-based fallback for constrained/test environments.
        # This preserves lexical similarity instead of generating random vectors.
        logger.info("Using hashing-vectorizer embedding fallback.")
        vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2)
        )
        return vectorizer.transform(texts).astype(np.float32).toarray()

    def embed_paper_content(self, title: str, abstract: str) -> List[float]:
        """Encode a single paper title and abstract."""
        combined = f"{title.strip()} [SEP] {abstract.strip()}"
        res = self.embed_texts([combined])
        return res[0].tolist()

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single text string for retrieval queries."""
        return self.embed_texts([text])[0]
