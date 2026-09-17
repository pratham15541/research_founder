"""
Vector Store Factory.
Returns the appropriate vector index backend:
  - OpenSearchVectorIndex when OPENSEARCH_ENDPOINT is configured (or LocalStack)
  - FAISSVectorIndex for local in-memory operation (default)
"""

import logging
from typing import Optional
from src.config import settings

logger = logging.getLogger(__name__)


def get_vector_index(dimension: Optional[int] = None):
    """
    Factory: return the best available vector index.

    Priority:
      1. OpenSearch — if USE_AWS=true and (OPENSEARCH_ENDPOINT set or USE_LOCALSTACK=true)
      2. FAISS — local in-memory fallback (always works, zero Docker needed)
    """
    dim = dimension or settings.EMBEDDING_DIMENSION

    if settings.is_aws_enabled() and (settings.OPENSEARCH_ENDPOINT or settings.USE_LOCALSTACK):
        try:
            from src.rag.opensearch_index import OpenSearchVectorIndex
            idx = OpenSearchVectorIndex(dimension=dim)
            logger.info(f"Using OpenSearch vector index (dim={dim})")
            return idx
        except Exception as e:
            logger.warning(f"OpenSearch unavailable ({e}). Falling back to FAISS.")

    from src.rag.faiss_index import FAISSVectorIndex
    logger.info(f"Using FAISS in-memory vector index (dim={dim})")
    return FAISSVectorIndex(dimension=dim)


# Alias for convenience
get_vector_store = get_vector_index

