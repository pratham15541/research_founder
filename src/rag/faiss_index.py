"""
FAISS Vector Search Engine for Academic Text Chunks.
Provides sub-millisecond semantic similarity search over dense vector embeddings.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

class FAISSVectorIndex:
    """Manages an in-memory or persisted FAISS vector index with metadata mapping."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Use inner product on normalized vectors for exact cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add chunks and their corresponding normalized embeddings to the index."""
        if len(chunks) == 0 or len(embeddings) == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = (embeddings / norms).astype(np.float32)

        self.index.add(normalized)
        self.metadata_store.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top K nearest chunks to the query vector.
        Returns list of chunks with similarity scores.
        """
        if self.index.ntotal == 0:
            return []

        # Ensure correct shape and float32 type
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_store):
                meta = dict(self.metadata_store[idx])
                meta["similarity_score"] = float(round(score, 4))
                results.append(meta)

        return results

    def count(self) -> int:
        """Return total number of indexed vectors."""
        return self.index.ntotal

    def clear(self) -> None:
        """Reset index and metadata store."""
        self.index.reset()
        self.metadata_store.clear()

