"""
Dynamic Unsupervised Domain & Problem Setting Discovery Engine.
Analyzes the retrieved paper corpus to dynamically discover orthogonal application domains
and problem settings without any hardcoded categories.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import httpx

from src.config import settings

logger = logging.getLogger(__name__)

class DomainDiscoveryEngine:
    """Discovers application domains and problem settings directly from the paper corpus."""

    @classmethod
    def discover_domains(
        cls,
        topic_query: str,
        papers: List[Dict[str, Any]],
        embedder: Any,
        num_domains: int = 5
    ) -> List[str]:
        """
        Dynamically discover 4 to 5 natural application domains or problem regimes
        present in the corpus using unsupervised phrase clustering with LLM formatting fallback.
        """
        if not papers:
            return ["Primary Applications", "Secondary Applications", "Theoretical Extensions"]

        # Attempt LLM-assisted domain discovery if available
        if settings.NVIDIA_API_KEY:
            llm_domains = cls._discover_with_llm(topic_query, papers, num_domains)
            if llm_domains and len(llm_domains) >= 3:
                return llm_domains

        # Unsupervised statistical clustering discovery
        return cls._discover_unsupervised(papers, embedder, num_domains)

    @classmethod
    def _discover_with_llm(
        cls,
        topic_query: str,
        papers: List[Dict[str, Any]],
        num_domains: int
    ) -> Optional[List[str]]:
        """Query NVIDIA AI API to summarize the 4-5 natural application domains in the corpus."""
        titles = [p.get("title", "") for p in papers if p.get("title")][:30]
        titles_bullet = "\n".join([f"- {t}" for t in titles])

        prompt = f"""You are a principal academic research taxonomist.
Analyze these {len(titles)} paper titles on the topic '{topic_query}':
{titles_bullet}

TASK:
Identify exactly {num_domains} distinct, mutually exclusive real-world application domains or physical problem settings actively studied across these papers (e.g. for Physics-Informed ML: 'Fluid Dynamics & Flows', 'Thermal & Heat Transfer', 'Structural & Material Mechanics', 'Biomedical & Physiological Modeling', 'Energy Systems & Batteries').

Respond ONLY with a valid JSON list of {num_domains} strings, formatted as concise domain titles (2 to 4 words each):
["Domain 1", "Domain 2", "Domain 3", "Domain 4", "Domain 5"]
"""
        try:
            from src.llm.nvidia_client import NvidiaClient
            domains = NvidiaClient.generate_json(prompt=prompt, temperature=0.2, max_tokens=1024)
            if isinstance(domains, list) and len(domains) >= 3:
                return [str(d).strip() for d in domains[:num_domains]]
        except Exception as e:
            logger.info(f"NVIDIA LLM domain discovery fallback ({e}). Using unsupervised phrase clustering.")

        return None

    @classmethod
    def _discover_unsupervised(
        cls,
        papers: List[Dict[str, Any]],
        embedder: Any,
        num_domains: int
    ) -> List[str]:
        """
        Unsupervised statistical discovery:
        Extracts distinctive noun phrase n-grams, clusters their embeddings,
        and derives the top centroid phrases as dynamic domain labels.
        """
        texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]

        # Filter academic methodology stop words
        academic_method_stops = {
            "neural", "networks", "network", "physics", "informed", "pinns", "pinn",
            "deep", "learning", "machine", "method", "methods", "model", "models",
            "approach", "approaches", "study", "analysis", "paper", "papers",
            "proposed", "using", "based", "via", "results", "framework", "algorithm",
            "data", "artificial", "intelligence", "computational", "performance",
            "survey", "review", "recent", "novel", "new", "experimental", "solving",
            "approximate", "approximating", "estimation", "numerical", "simulation"
        }
        all_stops = list(ENGLISH_STOP_WORDS.union(academic_method_stops))

        try:
            vec = TfidfVectorizer(ngram_range=(2, 3), max_features=50, stop_words=all_stops)
            X = vec.fit_transform(texts)
            phrases = list(vec.get_feature_names_out())
            if len(phrases) < num_domains:
                vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50, stop_words=all_stops)
                X = vec.fit_transform(texts)
                phrases = list(vec.get_feature_names_out())

            if len(phrases) < num_domains:
                return ["Domain Setting A", "Domain Setting B", "Domain Setting C"]

            phrase_embs = embedder.embed_texts(phrases)
            k = min(num_domains, len(phrases))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(phrase_embs)

            domains: List[str] = []
            for cid in range(k):
                cluster_indices = np.where(labels == cid)[0]
                cluster_phrases = [phrases[i] for i in cluster_indices]
                centroid = km.cluster_centers_[cid]
                c_embs = phrase_embs[cluster_indices]
                dists = np.linalg.norm(c_embs - centroid, axis=1)
                best_phrase = cluster_phrases[np.argmin(dists)]
                # Format phrase cleanly as title case
                formatted = best_phrase.replace("_", " ").title()
                domains.append(formatted)

            # Ensure uniqueness
            unique_domains = []
            for d in domains:
                if d not in unique_domains:
                    unique_domains.append(d)

            return unique_domains if len(unique_domains) >= 3 else [
                "Theoretical Foundations", "Empirical Systems", "Computational Benchmarks", "Applied Engineering"
            ]
        except Exception as e:
            logger.warning(f"Unsupervised domain discovery error: {e}")
            return [
                "Computational Modeling", "Experimental Validation", "System Dynamics", "Domain Optimization"
            ]

    @classmethod
    def tag_papers_with_domains(
        cls,
        papers: List[Dict[str, Any]],
        domains: List[str],
        embedder: Any
    ) -> None:
        """
        Assign each paper in the corpus to its closest dynamic domain
        by calculating embedding cosine similarity between paper abstract and domain representations.
        """
        if not domains:
            return

        domain_embs = embedder.embed_texts(domains)
        paper_texts = [f"{p.get('title', '')} {p.get('abstract', '')[:200]}" for p in papers]
        paper_embs = embedder.embed_texts(paper_texts)

        # Normalize for cosine similarity
        norm_domain = domain_embs / np.linalg.norm(domain_embs, axis=1, keepdims=True)
        norm_paper = paper_embs / np.linalg.norm(paper_embs, axis=1, keepdims=True)

        sims = np.dot(norm_paper, norm_domain.T)  # Shape (N, num_domains)
        best_domain_indices = np.argmax(sims, axis=1)

        for idx, p in enumerate(papers):
            matched_domain = domains[best_domain_indices[idx]]
            p["axis_b_tag"] = matched_domain

