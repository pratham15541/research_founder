"""
Unit tests for DimensionDiscoveryEngine, Silhouette gating, and KMeans fallback.
"""

import pytest
import numpy as np
from src.representation.clustering import DimensionDiscoveryEngine

def test_clustering_with_cohesion_gating():
    rng = np.random.RandomState(42)

    # 1. Create two well-separated dense clusters (20 points each)
    cluster1 = rng.randn(20, 384) + np.array([5.0] * 384)
    cluster2 = rng.randn(20, 384) - np.array([5.0] * 384)
    embeddings = np.vstack([cluster1, cluster2])

    labels, sil, method = DimensionDiscoveryEngine.cluster_papers(
        embeddings=embeddings,
        min_cluster_size=5
    )

    # Should discover at least 2 clusters with positive silhouette score
    unique_labels = set(labels) - {-1}
    assert len(unique_labels) >= 2
    assert sil > 0.15

def test_kmeans_fallback_on_diffuse_noise():
    rng = np.random.RandomState(42)

    # Create diffuse uniform noise with no distinct peaks (30 points)
    embeddings = rng.uniform(-1.0, 1.0, size=(30, 384))

    labels, sil, method = DimensionDiscoveryEngine.cluster_papers(
        embeddings=embeddings,
        min_cluster_size=5
    )

    # Regardless of whether HDBSCAN or KMeans fallback is triggered,
    # the engine must produce valid labels without crashing
    assert len(labels) == 30
    assert len(set(labels)) >= 1

def test_multi_algorithm_options():
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(15, 384) + np.array([4.0] * 384)
    cluster2 = rng.randn(15, 384) - np.array([4.0] * 384)
    embeddings = np.vstack([cluster1, cluster2])

    for algo in ["auto", "kmeans", "agglomerative", "hdbscan"]:
        labels, sil, method = DimensionDiscoveryEngine.cluster_papers(
            embeddings=embeddings,
            min_cluster_size=5,
            algorithm=algo
        )
        assert len(labels) == 30
        assert len(set(labels)) >= 2
        assert sil >= 0.0
        assert -1 not in labels, f"Algorithm {algo} should not leave unassigned noise points (-1)"


