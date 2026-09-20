"""
Dynamic Limitation & Future-Work Clustering.

Replaces the CANONICAL_LIMITATION_KEYS and CANONICAL_FUTURE_WORK_KEYS
hardcoded lists with corpus-driven semantic clustering:

  1. Embed all limitation / future-work text snippets.
  2. Silhouette-optimal KMeans to discover categories.
  3. Name each category from its centroid quote — no predefined strings.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def silhouette_optimal_kmeans(
    texts: List[str],
    embedder,
    min_k: int = 2,
    max_k: int = 10,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Embed texts, find the k that maximises silhouette score, and return:
      labels       – cluster id per text
      centroids    – list of representative text (closest to centroid) per cluster
      cluster_ids  – sorted unique cluster ids
    """
    if not texts:
        return np.array([], dtype=int), [], []

    embs: np.ndarray = np.array(embedder.embed_texts(texts), dtype=float)
    n = len(texts)

    if n <= 2:
        return np.zeros(n, dtype=int), [texts[0]], [0]

    actual_max_k = min(max_k, n - 1)
    if actual_max_k < min_k:
        actual_max_k = min_k = max(2, actual_max_k)

    best_k, best_sil, best_labels = min_k, -1.0, np.zeros(n, dtype=int)
    for k in range(min_k, actual_max_k + 1):
        km = KMeans(n_clusters=k, n_init=8, random_state=42)
        candidate_labels = km.fit_predict(embs)
        try:
            sil = float(silhouette_score(embs, candidate_labels))
            if sil > best_sil:
                best_sil, best_k, best_labels = sil, k, candidate_labels
        except Exception:
            continue

    # Pick the text closest to each centroid as the canonical name
    km_final = KMeans(n_clusters=best_k, n_init=8, random_state=42)
    km_final.fit(embs)
    centroids_arr = km_final.cluster_centers_

    representative_texts: List[str] = []
    cluster_ids: List[int] = sorted(set(int(l) for l in best_labels))
    for cid in cluster_ids:
        indices = [i for i, l in enumerate(best_labels) if l == cid]
        centroid = centroids_arr[cid]
        dists = np.linalg.norm(embs[indices] - centroid, axis=1)
        best_local_idx = indices[int(np.argmin(dists))]
        representative_texts.append(texts[best_local_idx])

    logger.debug(
        "silhouette_optimal_kmeans: n=%d → k=%d (sil=%.3f)", n, best_k, best_sil
    )
    return best_labels, representative_texts, cluster_ids

