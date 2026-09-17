"""
Semantic labeling engine for discovered clusters.
Uses LLM structured output to generate human-readable taxonomy labels,
with TF-IDF n-gram extraction as offline fallback.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import settings

logger = logging.getLogger(__name__)

class ClusterLabelingEngine:
    """Assigns concise, human-readable research dimension labels to clusters."""

    @classmethod
    def label_clusters(
        cls,
        papers: List[Dict[str, Any]],
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """
        Label all clusters present in labels.
        Returns:
            Dict[cluster_id, {"label": str, "short_description": str, "key_terms": List[str]}]
        """
        unique_labels = sorted(list(set(labels)))
        cluster_info: Dict[int, Dict[str, Any]] = {}

        for raw_cid in unique_labels:
            cid = int(raw_cid)
            if cid == -1:
                cluster_info[cid] = {
                    "label": "Other / Emerging Directions",
                    "short_description": "Diverse or outlier papers outside established primary clusters.",
                    "key_terms": ["Outliers", "Emerging", "Cross-Domain"],
                    "paper_count": int(np.sum(labels == raw_cid))
                }
                continue

            indices = np.where(labels == raw_cid)[0]
            cluster_papers = [papers[i] for i in indices]
            cluster_embs = embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)

            # Find 3 papers closest to centroid
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            top_paper_idx = np.argsort(dists)[:3]
            rep_papers = [cluster_papers[i] for i in top_paper_idx]

            # Try LLM labeling
            label_data = cls._label_with_llm(rep_papers)
            if not label_data:
                label_data = cls._label_with_tfidf(cluster_papers)

            label_data["paper_count"] = int(len(cluster_papers))
            cluster_info[cid] = label_data

        # Ensure all cluster labels are unique
        seen_labels: set = set()
        for cid, info in cluster_info.items():
            base_label = info["label"]
            if base_label in seen_labels:
                key_terms = info.get("key_terms", [])
                alt_term = next((t for t in key_terms if t.lower() not in base_label.lower()), None)
                if alt_term:
                    info["label"] = f"{base_label} ({alt_term})"
                else:
                    info["label"] = f"{base_label} (Group {cid + 1})"
            seen_labels.add(info["label"])

        return cluster_info

    @classmethod
    def _label_with_llm(cls, rep_papers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Query LLM (Bedrock or NVIDIA) for a standardized 2-4 word taxonomy label."""
        from src.llm.llm_router import LLMRouter
        if not LLMRouter.is_available():
            return None

        paper_summaries = "\n".join([
            f"- Title: {p.get('title')}\n  Abstract excerpt: {p.get('abstract', '')[:200]}..."
            for p in rep_papers
        ])

        prompt = f"""You are a senior academic taxonomy specialist.
Analyze these representative papers from an unsupervised cluster and return a JSON object with:
1. "label": A concise research theme name (2 to 4 words, e.g. "Diffusion Generative Models" or "Graph Contrastive Learning").
2. "short_description": A 1-sentence technical explanation of this theme.
3. "key_terms": A list of 3-4 key technical keywords.

Papers in cluster:
{paper_summaries}

Respond ONLY with valid JSON in this exact structure:
{{"label": "...", "short_description": "...", "key_terms": ["...", "..."]}}
"""
        try:
            parsed = LLMRouter.generate_json(prompt=prompt, temperature=0.2, max_tokens=1024)
            if parsed and isinstance(parsed, dict) and "label" in parsed and "short_description" in parsed:
                return parsed
        except Exception as e:
            logger.warning(f"LLM labeling failed: {e}. Falling back to TF-IDF.")

        return None

    @classmethod
    def _label_with_tfidf(cls, cluster_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract top informative n-grams using TF-IDF when offline or LLM unavailable."""
        texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in cluster_papers]
        # Common academic filler words to ignore
        academic_stops = {
            "methods", "method", "materials", "material", "study", "analysis",
            "states", "state", "system", "systems", "approach", "approaches",
            "using", "based", "applications", "application", "review", "recent",
            "novel", "new", "qds", "via", "results", "model", "models",
            "paper", "papers", "learning", "data", "proposed", "performance",
            "research", "order", "different", "provides", "provide", "shows"
        }
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            custom_stops = list(ENGLISH_STOP_WORDS.union(academic_stops))

            vec = TfidfVectorizer(max_features=25, stop_words=custom_stops, ngram_range=(1, 2))
            X = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()
            scores = np.asarray(X.sum(axis=0)).flatten()
            top_indices = np.argsort(scores)[::-1][:6]
            top_terms = [feature_names[i].title() for i in top_indices if len(feature_names[i]) > 3]

            # Construct a natural, readable research title
            if len(top_terms) >= 2:
                # Prefer 2-word phrase if available
                phrase_terms = [t for t in top_terms if " " in t]
                if phrase_terms:
                    label = phrase_terms[0]
                else:
                    label = f"{top_terms[0]} & {top_terms[1]}"
            elif top_terms:
                label = f"{top_terms[0]} Modeling"
            else:
                label = "Theoretical Methodology"

            return {
                "label": label,
                "short_description": f"Focuses on methodologies involving {', '.join(top_terms[:3])}.",
                "key_terms": top_terms[:4]
            }
        except Exception:
            return {
                "label": "Scientific Methodology",
                "short_description": "Aggregated cluster of related foundational methods.",
                "key_terms": ["Methods", "Theory", "Applications"]
            }

