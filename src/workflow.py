"""
Master Research Discovery Workflow.
Coordinates the end-to-end evidence-grounded pipeline:
  1. Hybrid retrieval & compounding corpus caching
  2. Unsupervised dimension discovery (Axis A / Axis B)
  3. Structured 15-dimension knowledge extraction per paper
  4. Literature Evidence Graph construction
  5. Multi-signal gap mining (repeated limitations, future directions, contradictions)
  6. 15-category gap taxonomy candidate generation
  7. Adversarial novelty verification & 5-pillar Devil's Advocate reality check
  8. Calibrated confidence scoring and grounded experimental protocol formulation
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
from src.matrix.candidate_generator import MultiSignalGapGenerator
from src.extraction.paper_extractor import AcademicPaperExtractor
from src.graph.evidence_graph import LiteratureEvidenceGraphBuilder
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
    """End-to-end orchestrator for evidence-backed research landscape mapping and gap discovery."""

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
        target_corpus_size: int = 120,
        clustering_algorithm: str = "auto"
    ) -> Dict[str, Any]:
        """Execute the full multi-signal research gap discovery pipeline."""
        queries = expanded_queries or self._expand_queries(topic_query)

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

        # Define Axis B dynamically from retrieved papers
        axis_b_labels = DomainDiscoveryEngine.discover_domains(
            topic_query=topic_query,
            papers=paper_dicts,
            embedder=self.embedder,
            num_domains=5
        )

        # Tag Axis B dynamically
        DomainDiscoveryEngine.tag_papers_with_domains(
            papers=paper_dicts,
            domains=axis_b_labels,
            embedder=self.embedder
        )

        # Step 6: Structured 15-Dimension Knowledge Extraction per Paper
        logger.info("Extracting structured research dimensions from %d papers...", len(paper_dicts))
        structured_records = AcademicPaperExtractor.extract_corpus_records(paper_dicts)

        # Step 7: Literature Evidence Graph & Signal Mining
        logger.info("Mining evidence graph, recurring limitations, and contradictions...")
        evidence_graph = LiteratureEvidenceGraphBuilder.build_evidence_graph(structured_records)
        evidence_graph_html = LiteratureEvidenceGraphBuilder.export_pyvis_evidence_graph_html(evidence_graph)
        limitation_clusters = LiteratureEvidenceGraphBuilder.mine_repeated_limitations(structured_records)
        future_work_clusters = LiteratureEvidenceGraphBuilder.mine_recurring_future_work(structured_records)
        contradictions = LiteratureEvidenceGraphBuilder.mine_contradictions(structured_records)

        # Step 8: 2D Matrix Aggregation (Preserved for Landscape Visualization)
        matrix_result = CombinatorialMatrixAggregator.aggregate_matrix(
            papers=paper_dicts,
            axis_a_labels=axis_a_labels,
            axis_b_labels=axis_b_labels,
            sparsity_percentage=settings.SPARSITY_PERCENTAGE
        )

        # Step 9: Multi-Signal Gap Candidate Generation (8 Signals across 15 Taxonomy Categories)
        candidate_gaps = MultiSignalGapGenerator.generate_candidates(
            paper_records=structured_records,
            limitation_clusters=limitation_clusters,
            future_work_clusters=future_work_clusters,
            contradictions=contradictions,
            matrix_result=matrix_result,
            max_candidates=15
        )

        # Extract future work seeds from uploaded PDFs if any
        future_work_seeds: List[str] = []
        if uploaded_pdf_paths:
            for pdf_path in uploaded_pdf_paths:
                try:
                    parsed = AcademicPDFParser.parse_pdf(pdf_path)
                    seeds = AcademicPDFParser.extract_explicit_future_work_statements(parsed.get("sections", {}))
                    future_work_seeds.extend(seeds)
                except Exception:
                    pass

        # Step 10: Adversarial Novelty Verification, 5-Pillar Challenge & Calibrated Ranking
        ranked_gaps = GroundedGapRanker.rank_candidate_gaps(
            candidate_gaps=candidate_gaps,
            matrix_result=matrix_result,
            future_work_seeds=future_work_seeds,
            paper_records=structured_records,
            embedder=self.embedder,
            top_k=5
        )

        # Step 11: Paper Citation Knowledge Graph Generation
        paper_graph = KnowledgeGraphBuilder.build_graph(
            papers=paper_dicts,
            embeddings=embeddings,
            similarity_threshold=0.65
        )
        paper_graph_html = KnowledgeGraphBuilder.export_pyvis_html(paper_graph)

        # Step 12: Chunk corpus and index in FAISS vector store
        chunks = AcademicChunker.chunk_corpus(paper_dicts)
        self.faiss_index.clear()
        if chunks:
            chunk_texts = [f"{c.get('paper_title')} [{c.get('section_type')}] {c.get('text')}" for c in chunks]
            chunk_embs = self.embedder.embed_texts(chunk_texts)
            self.faiss_index.add_chunks(chunks, chunk_embs)

        # Step 13: Quantitative Evaluation Metrics
        topic_terms_list = [cinfo.get("key_terms", []) for cinfo in cluster_info.values()]
        topic_diversity = EvaluationMetricsEngine.calculate_topic_diversity(topic_terms_list)
        topic_coherence = EvaluationMetricsEngine.calculate_topic_coherence(topic_terms_list, self.embedder)
        retrieval_eval = EvaluationMetricsEngine.calculate_retrieval_metrics(topic_query, paper_dicts)
        full_text_papers = [p for p in paper_dicts if p.get("full_text_available")]
        downloaded_pdf_papers = [p for p in paper_dicts if p.get("pdf_local_path")]

        # Step 14: Literature Review Synthesis
        lit_review = LiteratureReviewGenerator.generate_review(
            topic_query=topic_query,
            papers=paper_dicts,
            clusters=cluster_info,
            domains=axis_b_labels,
            ranked_gaps=ranked_gaps
        )

        # Step 15: Formal Research Questions & Experimental Protocols
        research_questions = [
            ResearchQuestionGenerator.generate_questions_for_gap(g)
            for g in ranked_gaps[:3]
        ]

        return {
            "topic_query": topic_query,
            "expanded_queries": queries,
            "corpus_size": int(len(paper_dicts)),
            "silhouette_score": float(round(sil_score, 3)),
            "cluster_method": cluster_method,
            "discovered_clusters": {str(k): v for k, v in cluster_info.items()},
            "matrix": matrix_result,
            "candidate_gaps_count": int(len(candidate_gaps)),
            "ranked_gaps": ranked_gaps,
            "structured_records": structured_records,
            "limitation_clusters": limitation_clusters,
            "future_work_clusters": future_work_clusters,
            "contradictions": contradictions,
            "evidence_graph_summary": {
                "nodes": int(evidence_graph.number_of_nodes()),
                "edges": int(evidence_graph.number_of_edges())
            },
            "evidence_graph_html": evidence_graph_html,
            "graph_summary": {
                "nodes": int(paper_graph.number_of_nodes()),
                "edges": int(paper_graph.number_of_edges())
            },
            "graph_html": paper_graph_html,
            "papers": paper_dicts,
            "chunks_count": len(chunks),
            "pdf_enrichment": {
                "full_text_available_count": len(full_text_papers),
                "pdf_stored_count": len(downloaded_pdf_papers),
                "download_enabled": settings.ENABLE_FULL_TEXT_DOWNLOAD,
                "download_limit": settings.FULL_TEXT_DOWNLOAD_LIMIT
            },
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
    def _expand_queries(topic_query: str) -> List[str]:
        """Expand user topic into diverse academic retrieval queries."""
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            prompt = f"""You are designing a literature search for a research-gap discovery agent.
Given the topic: "{topic_query}"

Return a JSON list of 5 concise search queries that cover methods, applications, benchmarks, limitations, and emerging directions.
Each query must be 3 to 9 words and must remain tightly relevant to the topic."""
            try:
                from src.llm.nvidia_client import NvidiaClient
                parsed = NvidiaClient.generate_json(
                    prompt=prompt,
                    temperature=settings.LLM_STRUCTURED_TEMPERATURE,
                    max_tokens=768,
                    timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 35.0)
                )
                if isinstance(parsed, list):
                    cleaned = []
                    for item in parsed:
                        q = str(item).strip()
                        if q and q.lower() not in {x.lower() for x in cleaned}:
                            cleaned.append(q)
                    if cleaned:
                        return [topic_query] + cleaned[:5]
            except Exception as exc:
                logger.warning("LLM query expansion failed: %s, using multi-dimensional heuristic expansion.", exc)
        elif settings.LLM_REQUIRED:
            raise RuntimeError("NVIDIA_API_KEY is required for query expansion because LLM_REQUIRED=true.")

        # Robust heuristic expansion covering methods, limitations, benchmarks, and comparisons
        return [
            topic_query,
            f"{topic_query} limitations",
            f"{topic_query} benchmark evaluation",
            f"{topic_query} architectures methods",
            f"{topic_query} comparative study"
        ]
