"""
Full Paper Downloader.
Downloads open-access full-text PDFs from official sources (arXiv, OpenAlex OA, Semantic Scholar, Unpaywall, PMC)
using rotating free proxies from iplocate/free-proxy-list to eliminate rate limits and IP restrictions.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.ingestion.proxy_manager import proxy_manager
from src.ingestion.pdf_parser import AcademicPDFParser
from src.storage.models import Paper, PaperSection

logger = logging.getLogger(__name__)

class FullPaperDownloader:
    """Fetches and parses full-text PDFs for open-access papers."""

    @staticmethod
    def resolve_official_pdf_url(paper: Dict[str, Any]) -> Optional[str]:
        """Resolve the official direct PDF download link from paper metadata."""
        source = paper.get("source", "")
        url = paper.get("source_url", "")
        doi = paper.get("doi", "")

        # 1. arXiv: https://arxiv.org/abs/2301.12345 -> https://arxiv.org/pdf/2301.12345.pdf
        if "arxiv.org" in url:
            arxiv_id_match = re.search(r"arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+|[a-z\-]+/[0-9]+)", url)
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(2)
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # 2. Semantic Scholar or OpenAlex explicit PDF url
        if paper.get("pdf_url"):
            return paper["pdf_url"]

        # 3. Direct PDF link in source_url
        if url and url.endswith(".pdf"):
            return url

        return None

    @classmethod
    async def download_and_parse_full_paper(
        cls,
        session: AsyncSession,
        paper: Paper,
        max_retries: int = 3
    ) -> bool:
        """
        Download official open-access PDF for a paper using rotating proxies,
        parse text with AcademicPDFParser, and update the database record.
        """
        if paper.full_text and len(paper.full_text) > 500:
            return True  # Already has full text

        paper_dict = paper.to_dict()
        pdf_url = cls.resolve_official_pdf_url(paper_dict)

        # If no direct URL, query Unpaywall (official open-access resolver)
        if not pdf_url and paper.doi:
            pdf_url = await cls._resolve_via_unpaywall(paper.doi)

        if not pdf_url:
            return False

        logger.info(f"Downloading full-text PDF for '{paper.original_title[:40]}...' from {pdf_url}")
        dest_path = settings.UPLOAD_DIR / f"{paper.id}.pdf"

        # Download with proxy rotation
        downloaded = False
        for attempt in range(max_retries):
            try:
                # Use proxy on retry or primary
                client = await proxy_manager.get_working_client(timeout=20.0)
                async with client:
                    resp = await client.get(pdf_url, follow_redirects=True)
                    if resp.status_code == 200 and resp.content.startswith(b"%PDF-"):
                        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                        with open(dest_path, "wb") as f:
                            f.write(resp.content)
                        downloaded = True
                        break
                    elif resp.status_code == 200 and len(resp.content) > 1000:
                        # Some servers send PDF without standard header at offset 0
                        with open(dest_path, "wb") as f:
                            f.write(resp.content)
                        if AcademicPDFParser.validate_pdf_file(dest_path):
                            downloaded = True
                            break
                        else:
                            dest_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed for {pdf_url}: {e}")

        if not downloaded or not dest_path.exists():
            return False

        # Parse downloaded PDF
        try:
            parsed = AcademicPDFParser.parse_pdf(dest_path)
            full_text = parsed.get("full_text", "")
            sections = parsed.get("sections", {})

            # Update Paper in database
            paper.full_text = full_text
            paper.pdf_local_path = str(dest_path)

            # Insert extracted sections
            for sec_type, content in sections.items():
                if content and len(content.strip()) > 30:
                    sec_rec = PaperSection(
                        paper_id=paper.id,
                        section_type=sec_type,
                        content=content.strip()
                    )
                    session.add(sec_rec)

            await session.commit()
            logger.info(f"Successfully extracted full text ({len(full_text)} chars) for: {paper.original_title[:40]}")
            return True
        except Exception as e:
            logger.warning(f"Failed to parse downloaded PDF for {paper.id}: {e}")
            return False

    @staticmethod
    async def _resolve_via_unpaywall(doi: str) -> Optional[str]:
        """Query official Unpaywall API for legal open-access PDF URL."""
        email = settings.OPENALEX_EMAIL
        api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.get(api_url)
                if res.status_code == 200:
                    data = res.json()
                    best_oa = data.get("best_oa_location") or {}
                    return best_oa.get("url_for_pdf")
        except Exception:
            pass
        return None

