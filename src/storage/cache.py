"""
Compounding Cache Engine.
Ensures every retrieved or uploaded paper is cached, deduplicated by DOI and normalized title,
and indexed with embeddings so queries become faster and compounding over time.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.models import Paper, PaperEmbedding, PaperSection

logger = logging.getLogger(__name__)

def normalize_title(title: str) -> str:
    """Normalize paper title for fuzzy collision detection and deduplication."""
    if not title:
        return ""
    # Lowercase, remove punctuation and extra spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", title.lower())
    return " ".join(cleaned.split())

class CompoundingCacheEngine:
    """Orchestrates paper persistence, deduplication, and vector caching."""

    @staticmethod
    async def find_existing_papers(
        session: AsyncSession,
        papers_metadata: List[Dict[str, Any]]
    ) -> Tuple[List[Paper], List[Dict[str, Any]]]:
        """
        Partition input papers into already-cached records and new records to insert.
        Deduplication checks exact DOI first, then normalized title.
        """
        dois = [p.get("doi") for p in papers_metadata if p.get("doi")]
        norm_titles = [normalize_title(p.get("title", "")) for p in papers_metadata if p.get("title")]

        cached_papers: List[Paper] = []
        seen_ids = set()

        if dois or norm_titles:
            conditions = []
            if dois:
                conditions.append(Paper.doi.in_(dois))
            if norm_titles:
                conditions.append(Paper.normalized_title.in_(norm_titles))

            stmt = select(Paper).where(or_(*conditions))
            result = await session.execute(stmt)
            for paper in result.scalars().all():
                if paper.id not in seen_ids:
                    cached_papers.append(paper)
                    seen_ids.add(paper.id)

        # Build lookup sets for fast matching
        cached_dois = {p.doi for p in cached_papers if p.doi}
        cached_norm_titles = {p.normalized_title for p in cached_papers}

        new_papers_to_insert: List[Dict[str, Any]] = []
        for p in papers_metadata:
            doi = p.get("doi")
            norm_title = normalize_title(p.get("title", ""))
            
            is_cached = False
            if doi and doi in cached_dois:
                is_cached = True
            elif norm_title and norm_title in cached_norm_titles:
                is_cached = True

            if not is_cached:
                new_papers_to_insert.append(p)
                # Add to local sets to avoid intra-batch duplicates
                if doi:
                    cached_dois.add(doi)
                if norm_title:
                    cached_norm_titles.add(norm_title)

        logger.info(
            f"Compounding Cache: {len(cached_papers)} loaded from DB, "
            f"{len(new_papers_to_insert)} new papers to ingest."
        )
        return cached_papers, new_papers_to_insert

    @staticmethod
    async def insert_new_papers(
        session: AsyncSession,
        papers: List[Dict[str, Any]]
    ) -> List[Paper]:
        """Insert newly retrieved or uploaded papers into the compounding database."""
        created_records: List[Paper] = []
        for item in papers:
            norm_title = normalize_title(item.get("title", ""))
            paper_rec = Paper(
                doi=item.get("doi"),
                normalized_title=norm_title,
                original_title=item.get("title", "Untitled"),
                abstract=item.get("abstract", ""),
                full_text=item.get("full_text"),
                authors=item.get("authors", []),
                publication_year=int(item.get("year", 2024)),
                citation_count=int(item.get("citation_count", 0)),
                source=item.get("source", "unknown"),
                source_url=item.get("source_url", ""),
                is_uploaded=bool(item.get("is_uploaded", False)),
                pdf_local_path=item.get("pdf_local_path")
            )
            session.add(paper_rec)
            created_records.append(paper_rec)

        await session.flush()

        # Add any extracted sections (e.g. from PDF uploads or open access)
        for rec, item in zip(created_records, papers):
            sections = item.get("sections", {})
            for sec_type, sec_content in sections.items():
                if sec_content and len(sec_content.strip()) > 20:
                    sec_rec = PaperSection(
                        paper_id=rec.id,
                        section_type=sec_type,
                        content=sec_content.strip()
                    )
                    session.add(sec_rec)

        await session.commit()
        return created_records

    @staticmethod
    async def save_embedding(
        session: AsyncSession,
        paper_id: Any,
        embedding: List[float],
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        """Store computed embedding vector for a paper."""
        pid = uuid.UUID(paper_id) if isinstance(paper_id, str) else paper_id
        emb_rec = PaperEmbedding(
            paper_id=pid,
            embedding=embedding,
            model_name=model_name
        )
        session.add(emb_rec)
        await session.commit()

    @staticmethod
    async def get_all_papers_by_ids(
        session: AsyncSession,
        paper_ids: List[Any]
    ) -> List[Paper]:
        """Fetch papers by their UUIDs."""
        uuid_ids = [uuid.UUID(pid) if isinstance(pid, str) else pid for pid in paper_ids]
        stmt = select(Paper).where(Paper.id.in_(uuid_ids))
        result = await session.execute(stmt)
        return list(result.scalars().all())

