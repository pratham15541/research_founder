"""
Dynamic Axis B Domain Discovery Engine.
Infers application domains directly from the paper corpus using NVIDIA AI API or
silhouette-optimal statistical KMeans phrase clustering.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
        num_domains: Optional[int] = None,
    ) -> List[str]:
        """
        Dynamically discover natural application domains or problem regimes
        present in the corpus using LLM synthesis or silhouette-optimal unsupervised clustering.
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

        # Unsupervised statistical clustering discovery (silhouette-optimal k)
        return cls._discover_unsupervised(papers, embedder, num_domains)

    @classmethod
    def _discover_with_llm(
        cls,
        topic_query: str,
        papers: List[Dict[str, Any]],
        num_domains: Optional[int] = None,
    ) -> Optional[List[str]]:
        """Query NVIDIA AI API to dynamically infer the natural domains in the corpus."""
        paper_summaries = []
        for p in papers[:40]:
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            if title:
                paper_summaries.append(f"- {title}: {abstract[:240]}")
        corpus_bullet = "\n".join(paper_summaries)

        target_count_str = "between 3 and 6" if num_domains is None else f"exactly {num_domains}"

        prompt = f"""You are a principal academic research taxonomist.
Analyze this paper corpus on the topic "{topic_query}":
{corpus_bullet}

TASK:
Infer {target_count_str} distinct, mutually exclusive application domains, physical problem settings, datasets, or empirical regimes that are actually present in these papers.
Do not use generic buckets. Do not invent fields absent from the corpus. Each label must be grounded in repeated terms from the titles or abstracts.

Respond ONLY with a valid JSON list of strings, formatted as concise domain titles (2 to 4 words each):
["Domain 1", "Domain 2", ...]
"""
        try:
            from src.llm.nvidia_client import NvidiaClient
            domains = NvidiaClient.generate_json(
                prompt=prompt,
                temperature=settings.LLM_STRUCTURED_TEMPERATURE,
                max_tokens=1024
            )
            if isinstance(domains, list) and len(domains) >= 3:
                limit = num_domains or 7
                return [str(d).strip() for d in domains[:limit] if str(d).strip()]
        except Exception as e:
            logger.info(f"NVIDIA LLM domain discovery fallback ({e}). Using unsupervised phrase clustering.")

        return None

    @classmethod
    def _discover_unsupervised(
        cls,
        papers: List[Dict[str, Any]],
        embedder: Any,
        num_domains: Optional[int] = None,
    ) -> List[str]:
        """
        Unsupervised statistical discovery:
        Extracts distinctive noun phrase n-grams, clusters their embeddings with silhouette-optimal k,
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
            if len(phrases) < 4:
                vec = TfidfVectorizer(ngram_range=(1, 3), max_features=80, stop_words=all_stops)
                X = vec.fit_transform(texts)
                phrases = list(vec.get_feature_names_out())

            if len(phrases) < 4:
                phrases = cls._extract_title_phrases(papers, all_stops)

            if not phrases:
                raise ValueError("Could not derive dynamic domain labels from the corpus.")

            phrase_embs = embedder.embed_texts(phrases)
            max_possible_k = min(len(phrases) - 1, 7)

            if num_domains is not None:
                k = max(2, min(num_domains, len(phrases)))
            elif max_possible_k >= 3:
                best_k = 3
                best_sil = -1.0
                for cand_k in range(3, max_possible_k + 1):
                    km_cand = KMeans(n_clusters=cand_k, n_init=10, random_state=42)
                    labels_cand = km_cand.fit_predict(phrase_embs)
                    try:
                        sil = float(silhouette_score(phrase_embs, labels_cand))
                        if sil > best_sil:
                            best_sil = sil
                            best_k = cand_k
                    except Exception:
                        continue
                k = best_k
            else:
                k = max(2, min(len(phrases), 4))

            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(phrase_embs)

            domains: List[str] = []
            for cid in range(k):
                cluster_indices = np.where(labels == cid)[0]
                cluster_phrases = [phrases[i] for i in cluster_indices]
                centroid = km.cluster_centers_[cid]
                c_embs = phrase_embs[cluster_indices]
                dists = np.linalg.norm(c_embs - centroid, axis=1)
                best_phrase = cluster_phrases[np.argmin(dists)]
                formatted = best_phrase.replace("_", " ").title()
                domains.append(formatted)

            # Ensure uniqueness
            unique_domains = []
            for d in domains:
                if d not in unique_domains:
                    unique_domains.append(d)

            if len(unique_domains) >= 2:
                return unique_domains
            return [p.replace("_", " ").title() for p in phrases[:k]]
        except Exception as e:
            logger.warning(f"Unsupervised domain discovery error: {e}")
            phrases = cls._extract_title_phrases(papers, all_stops)
            if phrases:
                return [p.replace("_", " ").title() for p in phrases[:5]]
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
