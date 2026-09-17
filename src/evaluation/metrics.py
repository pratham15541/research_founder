"""
Scientific Evaluation Metrics Engine.
Computes Intrinsic metrics (Topic Coherence, Diversity, Silhouette) and Extrinsic metrics (Precision@K, Recall@K, F1).
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import silhouette_score

class EvaluationMetricsEngine:
    """Calculates quantitative benchmark metrics for research intelligence pipelines."""

    @staticmethod
    def calculate_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate geometric cluster cohesion score (-1.0 to 1.0)."""
        unique_labels = set(labels) - {-1}
        if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
            return 0.0
        try:
            mask = labels != -1
            score = float(silhouette_score(embeddings[mask], labels[mask]))
            return round(score, 3)
        except Exception:
            return 0.0

    @staticmethod
    def calculate_topic_diversity(topic_terms_list: List[List[str]]) -> float:
        """
        Calculate topic diversity (0.0 to 1.0):
        The proportion of unique words across all top-word lists for discovered topics.
        High diversity (e.g. > 0.7) indicates topics are not repetitive.
        """
        if not topic_terms_list:
            return 0.0

        all_words = []
        for term_list in topic_terms_list:
            for w in term_list:
                all_words.append(w.lower().strip())

        if not all_words:
            return 0.0

        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words)
        return round(float(diversity), 3)

    @staticmethod
    def calculate_topic_coherence(topic_terms_list: List[List[str]], embedder: Any) -> float:
        """
        Calculate embedding-based Topic Coherence (approximate C_v):
        Measures semantic similarity between top words within each topic.
        Values typically range from 0.3 (low coherence) to 0.8+ (high coherence).
        """
        if not topic_terms_list:
            return 0.0

        topic_coherences = []
        for term_list in topic_terms_list:
            valid_terms = [t for t in term_list if len(t.strip()) > 2]
            if len(valid_terms) < 2:
                continue

            term_embs = embedder.embed_texts(valid_terms)
            norms = np.linalg.norm(term_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized = term_embs / norms

            # Compute pairwise cosine similarity
            sim_matrix = np.dot(normalized, normalized.T)
            # Take upper triangle without diagonal
            upper_tri_indices = np.triu_indices(len(valid_terms), k=1)
            if len(upper_tri_indices[0]) > 0:
                mean_sim = np.mean(sim_matrix[upper_tri_indices])
                topic_coherences.append(mean_sim)

        if not topic_coherences:
            return 0.5

        return round(float(np.mean(topic_coherences)), 3)

    @staticmethod
    def calculate_retrieval_metrics(
        query: str,
        retrieved_papers: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate Extrinsic Precision@K, Recall@K, and F1 Score for retrieval quality.
        Uses topic keyword overlap heuristic against ground truth relevance.
        """
        top_k_papers = retrieved_papers[:k]
        if not top_k_papers:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_score": 0.0}

        query_terms = [w.lower() for w in query.split() if len(w) > 3]

        relevant_count = 0
        for p in top_k_papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            if any(term in text for term in query_terms):
                relevant_count += 1

        precision_at_k = relevant_count / len(top_k_papers)

        # Total estimated relevant papers in full retrieved set
        total_relevant = sum(
            1 for p in retrieved_papers
            if any(term in f"{p.get('title', '')} {p.get('abstract', '')}".lower() for term in query_terms)
        )
        total_relevant = max(1, total_relevant)
        recall_at_k = min(1.0, relevant_count / total_relevant)

        if (precision_at_k + recall_at_k) > 0:
            f1 = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        else:
            f1 = 0.0

        return {
            "precision_at_k": round(precision_at_k, 3),
            "recall_at_k": round(recall_at_k, 3),
            "f1_score": round(f1, 3),
            "f1_at_k": round(f1, 3)
        }
