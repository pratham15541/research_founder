"""
Hybrid Ingestion Orchestrator.
Merges multi-source API retrievals (OpenAlex, Semantic Scholar, arXiv, PubMed) with user-uploaded PDFs,
deduplicates against the PostgreSQL compounding database, and returns a unified corpus.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.ingestion.openalex import OpenAlexRetriever
from src.ingestion.semantic_scholar import SemanticScholarRetriever
from src.ingestion.arxiv_client import ArxivRetriever
from src.ingestion.pubmed_client import PubMedRetriever
from src.ingestion.crossref_client import CrossrefRetriever
from src.ingestion.pdf_parser import AcademicPDFParser
from src.ingestion.full_paper_downloader import FullPaperDownloader
from src.storage.cache import CompoundingCacheEngine
from src.storage.models import Paper

logger = logging.getLogger(__name__)

class HybridIngestionEngine:
    """Coordinates multi-source fetching, PDF parsing, deduplication, and database caching."""

    def __init__(self):
        self.openalex = OpenAlexRetriever()
        self.s2 = SemanticScholarRetriever()
        self.arxiv = ArxivRetriever()
        self.pubmed = PubMedRetriever()
        self.crossref = CrossrefRetriever()

    def _is_biomedical_query(self, query: str) -> bool:
        """Heuristic check to determine if PubMed routing is appropriate."""
        bio_keywords = [
            "cancer", "clinical", "genomic", "disease", "drug", "patient", "medical",
            "biology", "protein", "health", "hospital", "pathology", "cell", "tumor"
        ]
        q_lower = query.lower()
        return any(k in q_lower for k in bio_keywords)

    async def ingest_topic_corpus(
        self,
        session: AsyncSession,
        queries: List[str],
        uploaded_pdf_paths: Optional[List[Path]] = None,
        target_corpus_size: int = 150
    ) -> List[Paper]:
        """
        Execute full hybrid ingestion:
        1. Parse uploaded PDFs
        2. Query external APIs asynchronously across all expanded query variations
        3. Deduplicate across sources and against compounding database
        4. Save new entries to DB and return the merged corpus
        """
        all_raw_papers: List[Dict[str, Any]] = []

        # 1. Parse uploaded PDFs immediately
        if uploaded_pdf_paths:
            for pdf_path in uploaded_pdf_paths:
                try:
                    parsed_doc = AcademicPDFParser.parse_pdf(pdf_path)
                    all_raw_papers.append(parsed_doc)
                    logger.info(f"Ingested uploaded PDF: {pdf_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to parse PDF {pdf_path.name}: {e}")

        # 2. Fetch configured external APIs concurrently
        enabled_sources = {
            s.strip().lower()
            for s in settings.INGESTION_SOURCES.split(",")
            if s.strip()
        }
        tasks = []
        for q in queries:
            if "openalex" in enabled_sources:
                tasks.append(self.openalex.search(q, limit=settings.OPENALEX_LIMIT))
            if "semantic_scholar" in enabled_sources or "s2" in enabled_sources:
                tasks.append(self.s2.search(q, limit=settings.SEMANTIC_SCHOLAR_LIMIT))
            if "arxiv" in enabled_sources:
                tasks.append(self.arxiv.search(q, limit=settings.ARXIV_LIMIT))
            if "pubmed" in enabled_sources:
                tasks.append(self.pubmed.search(q, limit=settings.PUBMED_LIMIT))
            if "crossref" in enabled_sources:
                tasks.append(self.crossref.search(q, limit=settings.CROSSREF_LIMIT))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                all_raw_papers.extend(res)
            elif isinstance(res, Exception):
                logger.warning(f"Retrieval task encountered exception: {res}")

        # 3. Compounding Deduplication against DB
        cached_existing, new_to_insert = await CompoundingCacheEngine.find_existing_papers(
            session, all_raw_papers
        )

        # 4. Insert new papers into PostgreSQL
        inserted_papers = []
        if new_to_insert:
            inserted_papers = await CompoundingCacheEngine.insert_new_papers(session, new_to_insert)

        # Merge cached existing records and newly inserted records
        final_corpus: List[Paper] = cached_existing + inserted_papers

        # Prioritize papers with abstracts and higher citation count
        final_corpus.sort(key=lambda p: (p.is_uploaded, p.citation_count), reverse=True)
        final_corpus = final_corpus[:target_corpus_size]

        if settings.ENABLE_FULL_TEXT_DOWNLOAD:
            enriched = 0
            checked = 0
            for paper in final_corpus:
                if checked >= settings.FULL_TEXT_DOWNLOAD_LIMIT:
                    break
                if paper.full_text and len(paper.full_text.strip()) >= settings.MIN_EXTRACTED_PDF_TEXT_CHARS:
                    continue
                checked += 1
                try:
                    if await FullPaperDownloader.download_and_parse_full_paper(session, paper):
                        enriched += 1
                except Exception as exc:
                    logger.info("Full-text enrichment skipped for %s: %s", paper.id, exc)
            logger.info("Full-text enrichment checked=%s enriched=%s", checked, enriched)

        logger.info(
            f"Hybrid Ingestion complete: Total corpus size = {len(final_corpus)} "
            f"({len(cached_existing)} cached + {len(inserted_papers)} newly fetched)"
        )
        return final_corpus
