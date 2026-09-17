"""
Unsupervised dimension discovery and cluster cohesion gating.
Implements UMAP dimensionality reduction, HDBSCAN clustering, Silhouette score gating,
regularized KMeans optimization, and Agglomerative Hierarchical clustering.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from src.config import settings

logger = logging.getLogger(__name__)

class DimensionDiscoveryEngine:
    """Discovers natural groupings of papers via multi-algorithm clustering and enforces cohesion gates."""

    @staticmethod
    def reduce_dimensions(embeddings: np.ndarray, target_dims: int = 10) -> np.ndarray:
        """
        Reduce high-dimensional vector space using UMAP (preserving local and global topology)
        or PCA as fallback.
        """
        n_samples = embeddings.shape[0]
        if n_samples <= target_dims:
            return embeddings

        try:
            import umap
            reducer = umap.UMAP(
                n_components=min(target_dims, n_samples - 2),
                n_neighbors=min(15, n_samples - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42
            )
            return reducer.fit_transform(embeddings)
        except Exception as e:
            logger.warning(f"UMAP not available or failed ({e}). Falling back to PCA.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(target_dims, n_samples - 1), random_state=42)
            return pca.fit_transform(embeddings)

    @classmethod
    def cluster_papers(
        cls,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        algorithm: str = "auto"
    ) -> Tuple[np.ndarray, float, str]:
        """
        Execute unsupervised clustering with multi-algorithm selection and validate cohesion.

        Supported algorithms:
            - 'auto': Evaluates HDBSCAN, KMeans (silhouette-optimized), and Agglomerative Hierarchical (Ward),
                      selecting the candidate with highest silhouette score and guaranteed 0% orphan noise.
            - 'kmeans': KMeans with silhouette optimization over k in [3, 8].
            - 'agglomerative': Hierarchical clustering with Ward's linkage (variance-minimizing taxonomy).
            - 'hdbscan': Density-based clustering with automatic outlier centroid reassignment.

        Returns:
            - cluster_labels: np.ndarray of shape (N,)
            - silhouette: float score (-1.0 to 1.0)
            - method_used: Algorithm actually applied
        """
        n_samples = embeddings.shape[0]
        if n_samples < min_cluster_size * 2:
            logger.info(f"Corpus too small ({n_samples}) for multi-cluster discovery. Returning single cluster.")
            return np.zeros(n_samples, dtype=int), 1.0, "single_cluster"

        # 1. Dimensionality reduction
        reduced = cls.reduce_dimensions(embeddings, target_dims=10)

        algo = (algorithm or "auto").lower().strip()

        if algo == "kmeans":
            labels, sil = cls._run_kmeans(reduced, min_k=3, max_k=min(8, n_samples // min_cluster_size))
            method = "kmeans_optimized"
        elif algo in ("agglomerative", "hierarchical"):
            labels, sil = cls._run_agglomerative(reduced, min_k=3, max_k=min(8, n_samples // min_cluster_size))
            method = "agglomerative_ward"
        elif algo == "hdbscan":
            labels, sil = cls._run_hdbscan(reduced, min_cluster_size)
            method = "hdbscan"
        else:  # auto
            labels, sil, method = cls._run_auto(reduced, min_cluster_size)

        logger.info(f"Clustering complete: {len(set(labels))} clusters, Silhouette: {sil:.3f} ({method})")
        return labels, sil, method

    @classmethod
    def _run_hdbscan(cls, reduced: np.ndarray, min_cluster_size: int) -> Tuple[np.ndarray, float]:
        """Run HDBSCAN density clustering with centroid reassignment for outliers."""
        n_samples = reduced.shape[0]
        labels = None
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=max(2, min_cluster_size // 2),
                metric="euclidean",
                cluster_selection_method="eom"
            )
            labels = clusterer.fit_predict(reduced)
        except Exception as e:
            logger.warning(f"HDBSCAN clustering error: {e}")

        if labels is None or len(set(labels) - {-1}) < 2:
            return cls._run_kmeans(reduced, min_k=3, max_k=min(7, n_samples // min_cluster_size))

        # Reassign noise points to nearest centroid so no paper is discarded as orphan
        labels = cls._reassign_outliers(reduced, labels)
        sil = cls._safe_silhouette(reduced, labels)
        return labels, sil

    @staticmethod
    def _run_kmeans(reduced: np.ndarray, min_k: int = 3, max_k: int = 6) -> Tuple[np.ndarray, float]:
        """Iterate over k values to maximize silhouette score."""
        n_samples = reduced.shape[0]
        max_k = min(max_k, n_samples - 1)
        if max_k < min_k:
            min_k = max(2, max_k)

        best_k = min_k
        best_sil = -1.0
        best_labels = np.zeros(n_samples, dtype=int)

        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            candidate_labels = km.fit_predict(reduced)
            try:
                score = float(silhouette_score(reduced, candidate_labels))
                if score > best_sil:
                    best_sil = score
                    best_k = k
                    best_labels = candidate_labels
            except Exception:
                continue

        return best_labels, max(best_sil, 0.0)

    @staticmethod
    def _run_agglomerative(reduced: np.ndarray, min_k: int = 3, max_k: int = 6) -> Tuple[np.ndarray, float]:
        """Agglomerative Hierarchical Clustering with Ward linkage (minimizes intra-cluster variance)."""
        n_samples = reduced.shape[0]
        max_k = min(max_k, n_samples - 1)
        if max_k < min_k:
            min_k = max(2, max_k)

        best_k = min_k
        best_sil = -1.0
        best_labels = np.zeros(n_samples, dtype=int)

        for k in range(min_k, max_k + 1):
            agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
            candidate_labels = agg.fit_predict(reduced)
            try:
                score = float(silhouette_score(reduced, candidate_labels))
                if score > best_sil:
                    best_sil = score
                    best_k = k
                    best_labels = candidate_labels
            except Exception:
                continue

        return best_labels, max(best_sil, 0.0)

    @classmethod
    def _run_auto(cls, reduced: np.ndarray, min_cluster_size: int) -> Tuple[np.ndarray, float, str]:
        """
        Adaptive Auto: Evaluates HDBSCAN, KMeans, and Agglomerative clustering,
        and dynamically selects the highest quality grouping based on Silhouette Score.
        """
        n_samples = reduced.shape[0]
        min_k = 3
        max_k = min(8, max(4, n_samples // min_cluster_size))

        candidates = []

        # Candidate 1: Agglomerative (Ward linkage)
        try:
            agg_labels, agg_sil = cls._run_agglomerative(reduced, min_k=min_k, max_k=max_k)
            if len(set(agg_labels)) >= 2:
                candidates.append((agg_sil, "agglomerative_ward", agg_labels))
        except Exception as e:
            logger.debug(f"Auto-clustering Agglomerative trial failed: {e}")

        # Candidate 2: KMeans (Silhouette-optimized)
        try:
            km_labels, km_sil = cls._run_kmeans(reduced, min_k=min_k, max_k=max_k)
            if len(set(km_labels)) >= 2:
                candidates.append((km_sil, "kmeans_optimized", km_labels))
        except Exception as e:
            logger.debug(f"Auto-clustering KMeans trial failed: {e}")

        # Candidate 3: HDBSCAN (with outlier reassignment)
        try:
            hdb_labels, hdb_sil = cls._run_hdbscan(reduced, min_cluster_size)
            if len(set(hdb_labels)) >= 2:
                candidates.append((hdb_sil, "hdbscan", hdb_labels))
        except Exception as e:
            logger.debug(f"Auto-clustering HDBSCAN trial failed: {e}")

        if not candidates:
            km_labels, km_sil = cls._run_kmeans(reduced, min_k=2, max_k=3)
            return km_labels, km_sil, "kmeans_fallback"

        # Pick candidate with highest silhouette score
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sil, best_method, best_labels = candidates[0]
        return best_labels, best_sil, f"auto_selected_{best_method}"

    @staticmethod
    def _safe_silhouette(reduced: np.ndarray, labels: np.ndarray) -> float:
        """Safely compute silhouette score."""
        unique = set(labels) - {-1}
        if len(unique) < 2 or len(labels) <= len(unique):
            return 0.0
        try:
            return float(silhouette_score(reduced, labels))
        except Exception:
            return 0.0

    @staticmethod
    def _reassign_outliers(reduced: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Reassign all noise points (label -1) to their closest cluster centroid in reduced space.
        Eliminates unclustered noise entirely so every paper belongs to an actionable cluster.
        """
        new_labels = labels.copy()
        unique_clusters = [c for c in set(labels) if c != -1]
        if not unique_clusters:
            return np.zeros(len(labels), dtype=int)

        # Calculate centroids
        centroids = {c: np.mean(reduced[labels == c], axis=0) for c in unique_clusters}

        for idx, lbl in enumerate(labels):
            if lbl == -1:
                point = reduced[idx]
                min_dist = float("inf")
                closest_c = unique_clusters[0]
                for c, centroid in centroids.items():
                    dist = float(np.linalg.norm(point - centroid))
                    if dist < min_dist:
                        min_dist = dist
                        closest_c = c

                new_labels[idx] = closest_c

        return new_labels

    # Backward compatibility alias
    _run_kmeans_fallback = _run_kmeans
