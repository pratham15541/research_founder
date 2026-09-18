"""
Dynamic corpus-derived domain and problem-setting discovery.
All Axis B labels are inferred from the retrieved papers; no generic domain buckets are used.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans

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
        present in the corpus using LLM synthesis or corpus phrase clustering.
        """
        if not papers:
            raise ValueError("Cannot discover dynamic domains without a paper corpus.")

        # Attempt LLM-assisted domain discovery if available
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            llm_domains = cls._discover_with_llm(topic_query, papers, num_domains)
            if llm_domains and len(llm_domains) >= 3:
                return llm_domains
        elif settings.LLM_REQUIRED:
            raise RuntimeError("NVIDIA_API_KEY is required for domain discovery because LLM_REQUIRED=true.")

        # Unsupervised statistical clustering discovery
        return cls._discover_unsupervised(papers, embedder, num_domains)

    @classmethod
    def _discover_with_llm(
        cls,
        topic_query: str,
        papers: List[Dict[str, Any]],
        num_domains: int
    ) -> Optional[List[str]]:
        """Query NVIDIA AI API to summarize natural domains in the corpus."""
        paper_summaries = []
        for p in papers[:40]:
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            if title:
                paper_summaries.append(f"- {title}: {abstract[:240]}")
        corpus_bullet = "\n".join(paper_summaries)

        prompt = f"""You are a principal academic research taxonomist.
Analyze this paper corpus on the topic "{topic_query}":
{corpus_bullet}

TASK:
Infer exactly {num_domains} distinct, mutually exclusive application domains, physical problem settings, datasets, or empirical regimes that are actually present in these papers.
Do not use generic buckets. Do not invent fields absent from the corpus. Each label must be grounded in repeated terms from the titles or abstracts.

Respond ONLY with a valid JSON list of {num_domains} strings, formatted as concise domain titles (2 to 4 words each):
["...", "...", "...", "...", "..."]
"""
        try:
            from src.llm.nvidia_client import NvidiaClient
            domains = NvidiaClient.generate_json(
                prompt=prompt,
                temperature=settings.LLM_STRUCTURED_TEMPERATURE,
                max_tokens=1024
            )
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
            vec = TfidfVectorizer(ngram_range=(2, 4), max_features=80, stop_words=all_stops)
            X = vec.fit_transform(texts)
            phrases = list(vec.get_feature_names_out())
            if len(phrases) < num_domains:
                vec = TfidfVectorizer(ngram_range=(1, 3), max_features=80, stop_words=all_stops)
                X = vec.fit_transform(texts)
                phrases = list(vec.get_feature_names_out())

            if len(phrases) < num_domains:
                phrases = cls._extract_title_phrases(papers, all_stops)

            if not phrases:
                raise ValueError("Could not derive dynamic domain labels from the corpus.")

            phrase_embs = embedder.embed_texts(phrases)
            k = min(num_domains, len(phrases))
            if k == 1:
                return [phrases[0].replace("_", " ").title()]

            km = KMeans(n_clusters=k, n_init=10)
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

            if len(unique_domains) >= min(3, k):
                return unique_domains
            return [p.replace("_", " ").title() for p in phrases[:num_domains]]
        except Exception as e:
            logger.warning(f"Unsupervised domain discovery error: {e}")
            phrases = cls._extract_title_phrases(papers, all_stops)
            if phrases:
                return [p.replace("_", " ").title() for p in phrases[:num_domains]]
            raise

    @staticmethod
    def _extract_title_phrases(papers: List[Dict[str, Any]], stop_words: List[str]) -> List[str]:
        """Derive domain candidates directly from title and abstract phrases."""
        texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 3), max_features=40, stop_words=stop_words)
            X = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()
            scores = np.asarray(X.sum(axis=0)).flatten()
            top_indices = np.argsort(scores)[::-1]
            phrases: List[str] = []
            for idx in top_indices:
                phrase = feature_names[idx].strip()
                if len(phrase) < 3:
                    continue
                formatted = phrase.title()
                if formatted not in phrases:
                    phrases.append(formatted)
            return phrases
        except Exception:
            return []

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
        norm_domain = domain_embs / np.maximum(np.linalg.norm(domain_embs, axis=1, keepdims=True), 1e-12)
        norm_paper = paper_embs / np.maximum(np.linalg.norm(paper_embs, axis=1, keepdims=True), 1e-12)

        sims = np.dot(norm_paper, norm_domain.T)  # Shape (N, num_domains)
        best_domain_indices = np.argmax(sims, axis=1)

        for idx, p in enumerate(papers):
            row = sims[idx]
            best_idx = int(best_domain_indices[idx])
            matched_domain = domains[best_idx]
            sorted_scores = np.sort(row)
            best_score = float(row[best_idx])
            margin = float(best_score - sorted_scores[-2]) if len(sorted_scores) > 1 else best_score
            p["axis_b_tag"] = matched_domain
            p["axis_b_confidence"] = round(best_score, 4)
            p["axis_b_confidence_margin"] = round(margin, 4)
            if best_score < 0.12 or margin < 0.015:
                p["tag_confidence"] = "low"
