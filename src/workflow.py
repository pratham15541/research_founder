"""
Master Research Discovery Workflow.
Coordinates the end-to-end pipeline from query expansion and hybrid retrieval to
unsupervised dimension discovery, deterministic 2D matrix aggregation,
and grounded ranking with devil's advocate critique.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.ingestion.hybrid import HybridIngestionEngine
from src.ingestion.pdf_parser import AcademicPDFParser
from src.representation.embeddings import EmbeddingEngine
from src.representation.clustering import DimensionDiscoveryEngine
from src.representation.labeling import ClusterLabelingEngine
from src.representation.domain_discovery import DomainDiscoveryEngine
from src.matrix.aggregator import CombinatorialMatrixAggregator
from src.matrix.gap_filter import CandidateGapFilter
from src.ranking.ranker import GroundedGapRanker
from src.graph.knowledge_graph import KnowledgeGraphBuilder
from src.storage.cache import CompoundingCacheEngine
from src.rag.chunker import AcademicChunker
from src.rag.faiss_index import FAISSVectorIndex
from src.synthesis.literature_review import LiteratureReviewGenerator
from src.synthesis.question_generator import ResearchQuestionGenerator
from src.evaluation.metrics import EvaluationMetricsEngine

logger = logging.getLogger(__name__)

class ResearchWorkflowRunner:
    """End-to-end orchestrator for research landscape mapping and gap discovery."""

    def __init__(self):
        self.ingestion = HybridIngestionEngine()
        self.embedder = EmbeddingEngine(settings.EMBEDDING_MODEL_NAME)
        self.faiss_index = FAISSVectorIndex(dimension=384)

    async def run_pipeline(
        self,
        session: AsyncSession,
        topic_query: str,
        expanded_queries: Optional[List[str]] = None,
        uploaded_pdf_paths: Optional[List[Path]] = None,
        axis_b_predefined: Optional[List[str]] = None,
        target_corpus_size: int = 120,
        clustering_algorithm: str = "auto"
    ) -> Dict[str, Any]:
        """Execute the full 10-step research discovery pipeline."""
        queries = expanded_queries or [topic_query, f"{topic_query} methods", f"{topic_query} applications"]

        # Step 1 & 2: Hybrid Ingestion & Compounding DB Cache
        corpus_papers = await self.ingestion.ingest_topic_corpus(
            session=session,
            queries=queries,
            uploaded_pdf_paths=uploaded_pdf_paths,
            target_corpus_size=target_corpus_size
        )

        if not corpus_papers:
            raise ValueError(f"No papers could be retrieved or parsed for query: {topic_query}")

        paper_dicts = [p.to_dict() for p in corpus_papers]

        # Step 3: Embeddings
        texts = [f"{p['title']} [SEP] {p['abstract']}" for p in paper_dicts]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Unsupervised Clustering & Dimension Discovery (Axis A)
        cluster_labels, sil_score, cluster_method = DimensionDiscoveryEngine.cluster_papers(
            embeddings=embeddings,
            min_cluster_size=settings.MIN_CLUSTER_SIZE,
            algorithm=clustering_algorithm
        )

        # Step 5: Semantic Cluster Labeling
        cluster_info = ClusterLabelingEngine.label_clusters(
            papers=paper_dicts,
            labels=cluster_labels,
            embeddings=embeddings
        )

        # Assign Axis A tags to papers
        axis_a_labels = [cinfo["label"] for cinfo in cluster_info.values()]
        for idx, p in enumerate(paper_dicts):
            cid = cluster_labels[idx]
            p["axis_a_tag"] = cluster_info[cid]["label"]
            p["tag_confidence"] = "high" if cid != -1 else "medium"

        # Define Axis B (Domain / Application or Predefined buckets)
        # Step 5b: Dynamic Unsupervised Discovery of Axis B (Application Domains & Problem Settings)
        if not axis_b_predefined:
            axis_b_labels = [
                "Healthcare & Clinical",
                "Autonomous Robotics",
                "Low-Power Edge Systems",
                "Finance & Economics",
                "Scientific Discovery"
            ]
            axis_b_labels = DomainDiscoveryEngine.discover_domains(
                topic_query=topic_query,
                papers=paper_dicts,
                embedder=self.embedder,
                num_domains=5
            )
        else:
            axis_b_labels = axis_b_predefined

        # Tag Axis B (Rule-based / Keyword classification for demonstration)
        self._tag_axis_b(paper_dicts, axis_b_labels)
        # Tag Axis B dynamically using embedding cosine similarity
        DomainDiscoveryEngine.tag_papers_with_domains(
            papers=paper_dicts,
            domains=axis_b_labels,
            embedder=self.embedder
        )

        # Step 6: Deterministic 2D Matrix Aggregation
        matrix_result = CombinatorialMatrixAggregator.aggregate_matrix(
            papers=paper_dicts,
            axis_a_labels=axis_a_labels,
            axis_b_labels=axis_b_labels,
            sparsity_percentage=settings.SPARSITY_PERCENTAGE
        )

        # Step 7: Candidate Gap Filtering (Adjacency + Sparsity)
        candidate_gaps = CandidateGapFilter.filter_candidate_gaps(
            matrix_result=matrix_result,
            dense_multiplier=3
        )

        # Extract any future work seeds from uploaded PDFs
        future_work_seeds: List[str] = []
        if uploaded_pdf_paths:
            for pdf_path in uploaded_pdf_paths:
                try:
                    parsed = AcademicPDFParser.parse_pdf(pdf_path)
                    seeds = AcademicPDFParser.extract_explicit_future_work_statements(parsed.get("sections", {}))
                    future_work_seeds.extend(seeds)
                except Exception:
                    pass

        # Step 8: Feasibility Estimation, Grounded Ranking & Adversarial Self-Critique
        ranked_gaps = GroundedGapRanker.rank_candidate_gaps(
            candidate_gaps=candidate_gaps,
            matrix_result=matrix_result,
            future_work_seeds=future_work_seeds,
            top_k=5
        )

        # Step 9 & 10: Knowledge Graph Generation
        graph = KnowledgeGraphBuilder.build_graph(
            papers=paper_dicts,
            embeddings=embeddings,
            similarity_threshold=0.65
        )
        graph_html = KnowledgeGraphBuilder.export_pyvis_html(graph)

        # Step 11: Chunk corpus and index in FAISS vector store
        chunks = AcademicChunker.chunk_corpus(paper_dicts)
        self.faiss_index.clear()
        if chunks:
            chunk_texts = [f"{c.get('paper_title')} [{c.get('section_type')}] {c.get('text')}" for c in chunks]
            chunk_embs = self.embedder.embed_texts(chunk_texts)
            self.faiss_index.add_chunks(chunks, chunk_embs)

        # Step 12: Quantitative Evaluation Metrics
        topic_terms_list = [cinfo.get("key_terms", []) for cinfo in cluster_info.values()]
        topic_diversity = EvaluationMetricsEngine.calculate_topic_diversity(topic_terms_list)
        topic_coherence = EvaluationMetricsEngine.calculate_topic_coherence(topic_terms_list, self.embedder)
        retrieval_eval = EvaluationMetricsEngine.calculate_retrieval_metrics(topic_query, paper_dicts)

        # Step 13: Literature Review Synthesis
        lit_review = LiteratureReviewGenerator.generate_review(
            topic_query=topic_query,
            papers=paper_dicts,
            clusters=cluster_info,
            domains=axis_b_labels,
            ranked_gaps=ranked_gaps
        )

        # Step 14: Evidence-Grounded Research Questions
        research_questions = [
            ResearchQuestionGenerator.generate_questions_for_gap(g)
            for g in ranked_gaps[:3]
        ]

        return {
            "topic_query": topic_query,
            "corpus_size": int(len(paper_dicts)),
            "silhouette_score": float(round(sil_score, 3)),
            "cluster_method": cluster_method,
            "discovered_clusters": {str(k): v for k, v in cluster_info.items()},
            "matrix": matrix_result,
            "candidate_gaps_count": int(len(candidate_gaps)),
            "ranked_gaps": ranked_gaps,
            "graph_summary": {
                "nodes": int(graph.number_of_nodes()),
                "edges": int(graph.number_of_edges())
            },
            "graph_html": graph_html,
            "papers": paper_dicts,
            "chunks_count": len(chunks),
            "evaluation_metrics": {
                "silhouette_score": float(round(sil_score, 3)),
                "topic_coherence": topic_coherence,
                "topic_diversity": topic_diversity,
                "retrieval": retrieval_eval
            },
            "literature_review": lit_review,
            "research_questions": research_questions
        }

    @staticmethod
    def _tag_axis_b(papers: List[Dict[str, Any]], axis_b_labels: List[str]) -> None:
        """Categorize papers along Axis B using keyword heuristics or round-robin for balanced spread."""
        domain_keywords = {
            "Healthcare & Clinical": ["medical", "health", "clinical", "patient", "disease", "hospital", "cancer", "biomedical"],
            "Autonomous Robotics": ["robot", "autonomous", "vehicle", "navigation", "motion", "drone", "sensor", "slam"],
            "Low-Power Edge Systems": ["edge", "embedded", "low-power", "mobile", "fpga", "quantization", "latency", "hardware"],
            "Finance & Economics": ["market", "trading", "financial", "stock", "portfolio", "risk", "crypto", "price"],
            "Scientific Discovery": ["physics", "chemistry", "material", "climate", "molecule", "biology", "simulation", "earth"]
        }

        for idx, p in enumerate(papers):
            text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            matched_b = None
            for b_label in axis_b_labels:
                kws = domain_keywords.get(b_label, [])
                if any(kw in text for kw in kws):
                    matched_b = b_label
                    break

            # If no keyword match, assign deterministically based on index hash for spread
            if not matched_b:
                matched_b = axis_b_labels[idx % len(axis_b_labels)]

            p["axis_b_tag"] = matched_b

