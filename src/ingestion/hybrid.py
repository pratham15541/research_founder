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

from src.ingestion.openalex import OpenAlexRetriever
from src.ingestion.semantic_scholar import SemanticScholarRetriever
from src.ingestion.arxiv_client import ArxivRetriever
from src.ingestion.pubmed_client import PubMedRetriever
from src.ingestion.pdf_parser import AcademicPDFParser
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

        # 2. Fetch external APIs concurrently
        tasks = []
        for q in queries:
            tasks.append(self.openalex.search(q, limit=60))
            tasks.append(self.s2.search(q, limit=30))
            tasks.append(self.arxiv.search(q, limit=30))
            if self._is_biomedical_query(q):
                tasks.append(self.pubmed.search(q, limit=30))

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

        logger.info(
            f"Hybrid Ingestion complete: Total corpus size = {len(final_corpus)} "
            f"({len(cached_existing)} cached + {len(inserted_papers)} newly fetched)"
        )
        return final_corpus[:target_corpus_size]

