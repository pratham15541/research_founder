"""
Amazon OpenSearch Vector Search Index.
Drop-in replacement for FAISSVectorIndex with persistent, scalable k-NN search.
Uses the opensearch-py client (Apache 2.0 — AWS open-source).
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class OpenSearchVectorIndex:
    """Manages an OpenSearch k-NN index with the same interface as FAISSVectorIndex."""

    def __init__(self, dimension: int = 384, index_name: Optional[str] = None):
        self.dimension = dimension
        self.index_name = index_name or settings.OPENSEARCH_INDEX_NAME
        self._client = None
        self._ensure_client()
        self._ensure_index()

    def _ensure_client(self) -> None:
        """Create the OpenSearch client, pointing to LocalStack or real AWS."""
        from opensearchpy import OpenSearch

        endpoint = settings.OPENSEARCH_ENDPOINT
        if not endpoint and settings.USE_LOCALSTACK:
            endpoint = f"{settings.LOCALSTACK_ENDPOINT}"

        if not endpoint:
            raise ValueError(
                "OPENSEARCH_ENDPOINT must be set, or USE_LOCALSTACK must be true."
            )

        # Parse host/port from endpoint URL
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        use_ssl = parsed.scheme == "https"
        port = parsed.port or (443 if use_ssl else 9200)

        # Auth configuration: prefer explicit username/password, use SigV4 for AWS, or fallback for LocalStack
        http_auth = None
        connection_class = None
        if settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
            http_auth = (settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD)
        elif not settings.USE_LOCALSTACK and "localhost" not in host and "127.0.0.1" not in host:
            # Real AWS OpenSearch endpoint: use AWS SigV4 signing with IAM credentials
            try:
                import boto3
                from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection

                session = boto3.Session(
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    aws_session_token=settings.AWS_SESSION_TOKEN,
                    region_name=settings.AWS_REGION,
                )
                credentials = session.get_credentials()
                if credentials:
                    service_name = "aoss" if "aoss" in host else "es"
                    http_auth = AWSV4SignerAuth(credentials, settings.AWS_REGION, service_name)
                    connection_class = RequestsHttpConnection
                    logger.info(f"Using AWS SigV4 auth for OpenSearch (service={service_name}, region={settings.AWS_REGION})")
            except Exception as e:
                logger.warning(f"Could not initialize AWSV4SignerAuth ({e}). Falling back.")
        elif settings.USE_LOCALSTACK and settings.AWS_ACCESS_KEY_ID:
            http_auth = (settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY or "")

        client_kwargs: Dict[str, Any] = {
            "hosts": [{"host": host, "port": port}],
            "use_ssl": use_ssl,
            "verify_certs": use_ssl,
            "ssl_show_warn": False,
        }
        if http_auth:
            client_kwargs["http_auth"] = http_auth
        if connection_class:
            client_kwargs["connection_class"] = connection_class

        self._client = OpenSearch(**client_kwargs)
        logger.info(f"OpenSearch client connected to {host}:{port} (ssl={use_ssl})")

    def _ensure_index(self) -> None:
        """Create the k-NN index if it does not exist."""
        try:
            if not self._client.indices.exists(index=self.index_name):
                body = {
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100,
                        }
                    },
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.dimension,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                },
                            },
                            "chunk_id": {"type": "keyword"},
                            "paper_id": {"type": "keyword"},
                            "paper_title": {"type": "text"},
                            "doi": {"type": "keyword"},
                            "year": {"type": "integer"},
                            "section_type": {"type": "keyword"},
                            "text": {"type": "text"},
                        }
                    },
                }
                self._client.indices.create(index=self.index_name, body=body)
                logger.info(f"Created OpenSearch k-NN index: {self.index_name}")
        except Exception as e:
            logger.warning(f"Could not ensure OpenSearch index: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Bulk-index chunks with their embeddings into OpenSearch."""
        if len(chunks) == 0 or len(embeddings) == 0:
            return

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = (embeddings / norms).astype(np.float32)

        bulk_docs = []
        for chunk, emb in zip(chunks, normalized):
            doc_id = chunk.get("chunk_id", str(uuid.uuid4()))
            doc = {
                "_index": self.index_name,
                "_id": doc_id,
                "embedding": emb.tolist(),
                "chunk_id": doc_id,
                "paper_id": chunk.get("paper_id", ""),
                "paper_title": chunk.get("paper_title", ""),
                "doi": chunk.get("doi", ""),
                "year": chunk.get("year", 2024),
                "section_type": chunk.get("section_type", ""),
                "text": chunk.get("text", ""),
            }
            bulk_docs.append(doc)

        if bulk_docs:
            try:
                from opensearchpy.helpers import bulk
                bulk(self._client, bulk_docs, refresh=True)
                logger.info(f"Indexed {len(bulk_docs)} chunks into OpenSearch")
            except Exception as e:
                logger.warning(f"OpenSearch bulk indexing error: {e}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """k-NN search for top-K nearest chunks."""
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": q.tolist(),
                        "k": top_k,
                    }
                }
            },
        }

        try:
            response = self._client.search(index=self.index_name, body=body)
            results: List[Dict[str, Any]] = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit["_source"]
                source["similarity_score"] = float(round(hit.get("_score", 0.0), 4))
                # Remove the embedding vector from results to save memory
                source.pop("embedding", None)
                results.append(source)
            return results
        except Exception as e:
            logger.warning(f"OpenSearch search error: {e}")
            return []

    def count(self) -> int:
        """Return total documents in the index."""
        try:
            resp = self._client.count(index=self.index_name)
            return resp.get("count", 0)
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete and recreate the index."""
        try:
            if self._client.indices.exists(index=self.index_name):
                self._client.indices.delete(index=self.index_name)
            self._ensure_index()
            logger.info(f"OpenSearch index '{self.index_name}' cleared and recreated.")
        except Exception as e:
            logger.warning(f"Could not clear OpenSearch index: {e}")
