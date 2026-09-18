"""
Full Paper Downloader.
Downloads open-access full-text PDFs from official sources (arXiv, OpenAlex OA, Semantic Scholar, Unpaywall, PMC)
using rotating free proxies from iplocate/free-proxy-list to eliminate rate limits and IP restrictions.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.ingestion.proxy_manager import proxy_manager
from src.ingestion.pdf_parser import AcademicPDFParser
from src.storage.file_storage import get_file_storage_backend
from src.storage.models import Paper, PaperSection

logger = logging.getLogger(__name__)

class FullPaperDownloader:
    """Fetches and parses full-text PDFs for open-access papers."""

    @staticmethod
    def resolve_official_pdf_url(paper: Dict[str, Any]) -> Optional[str]:
        """Resolve the official direct PDF download link from paper metadata."""
        candidates = FullPaperDownloader.resolve_official_pdf_candidates(paper)
        return candidates[0] if candidates else None

    @staticmethod
    def resolve_official_pdf_candidates(paper: Dict[str, Any]) -> List[str]:
        """Resolve ranked direct PDF candidates from paper metadata."""
        url = paper.get("source_url", "")
        candidates: List[str] = []

        def add_candidate(value: Optional[str]) -> None:
            if not value:
                return
            candidate = str(value).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        # Explicit open-access PDF URLs from APIs should be tried first when present.
        add_candidate(paper.get("pdf_url"))

        # arXiv: https://arxiv.org/abs/2301.12345 -> https://arxiv.org/pdf/2301.12345.pdf
        if "arxiv.org" in url:
            arxiv_id_match = re.search(r"arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+|[a-z\-]+/[0-9]+)", url)
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(2)
                add_candidate(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

        # Direct PDF links, including publisher links with query strings.
        if url and url.lower().split("?", 1)[0].endswith(".pdf"):
            add_candidate(url)

        return candidates

    @classmethod
    async def _resolve_pdf_candidates(cls, paper: Dict[str, Any]) -> List[str]:
        """Resolve all known legal PDF candidates, including DOI-based OA fallbacks."""
        candidates = cls.resolve_official_pdf_candidates(paper)
        doi = paper.get("doi")
        if doi:
            unpaywall_url = await cls._resolve_via_unpaywall(str(doi))
            if unpaywall_url and unpaywall_url not in candidates:
                candidates.append(unpaywall_url)
        return candidates

    @classmethod
    async def download_and_parse_full_paper(
        cls,
        session: AsyncSession,
        paper: Paper,
        max_retries: Optional[int] = None
    ) -> bool:
        """
        Download official open-access PDF for a paper using rotating proxies,
        parse text with AcademicPDFParser, and update the database record.
        """
        if paper.full_text and len(paper.full_text) > 500:
            return True  # Already has full text

        paper_dict = paper.to_dict()
        pdf_urls = await cls._resolve_pdf_candidates(paper_dict)
        if not pdf_urls:
            return False
        # Limit candidate URLs to top 2 to avoid cycling through dead publisher domains
        pdf_urls = pdf_urls[:2]

        logger.info(
            "Downloading full-text PDF for '%s...' using %s candidate URL(s)",
            paper.original_title[:40],
            len(pdf_urls)
        )
        dest_path: Optional[Path] = None
        storage_uri: Optional[str] = None

        max_retries = max_retries or settings.PDF_DOWNLOAD_MAX_RETRIES
        headers = {
            "User-Agent": f"ResearchGraph-Matrix/1.0 (mailto:{settings.OPENALEX_EMAIL})",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.2"
        }

        # Download with direct-first fallback. If direct fails or times out,
        # paywalled/dead publisher domains are fast-skipped.
        downloaded = False
        for pdf_url in pdf_urls:
            skip_candidate = False
            for attempt in range(max_retries):
                if skip_candidate:
                    break

                force_proxy_refresh = attempt > 0 and attempt % max(1, settings.PROXY_REFRESH_ON_FAILURES) == 0

                # Define client modes to attempt
                modes_to_try = []
                if settings.PDF_DOWNLOAD_DIRECT_FIRST:
                    if settings.PROXY_USE_DIRECT_FALLBACK:
                        modes_to_try.append("direct")
                    modes_to_try.append("proxy")
                else:
                    modes_to_try.append("proxy")
                    if settings.PROXY_USE_DIRECT_FALLBACK:
                        modes_to_try.append("direct")

                for mode in modes_to_try:
                    try:
                        if mode == "proxy":
                            client = await proxy_manager.get_working_client(
                                timeout=settings.PDF_DOWNLOAD_TIMEOUT_SECONDS,
                                force_refresh=force_proxy_refresh,
                                headers=headers
                            )
                        else:
                            client = httpx.AsyncClient(
                                timeout=settings.PDF_DOWNLOAD_TIMEOUT_SECONDS,
                                headers=headers
                            )

                        async with client:
                            resp = await client.get(pdf_url, follow_redirects=True)
                            content_type = resp.headers.get("content-type", "").lower()

                            # If server explicitly returns paywall / access denied / server error
                            if resp.status_code in {401, 403, 404, 405, 500, 502, 503}:
                                logger.debug(
                                    "Candidate URL %s returned HTTP %s via %s (paywalled or restricted).",
                                    pdf_url,
                                    resp.status_code,
                                    mode
                                )
                                skip_candidate = True
                                break

                            if resp.status_code != 200 or len(resp.content) < 1000:
                                logger.debug(
                                    "PDF download attempt %s via %s returned status=%s bytes=%s for %s",
                                    attempt + 1,
                                    mode,
                                    resp.status_code,
                                    len(resp.content),
                                    pdf_url
                                )
                                if mode == "direct":
                                    skip_candidate = True
                                    break
                                continue

                            if "html" in content_type and not AcademicPDFParser.validate_pdf_bytes(resp.content):
                                logger.debug(
                                    "PDF download attempt %s via %s returned HTML for %s",
                                    attempt + 1,
                                    mode,
                                    pdf_url
                                )
                                skip_candidate = True
                                break

                            if AcademicPDFParser.validate_pdf_bytes(resp.content):
                                stored = get_file_storage_backend().save_bytes(
                                    content=resp.content,
                                    filename=f"{paper.id}.pdf",
                                    content_type="application/pdf",
                                    file_id=str(paper.id)
                                )
                                validation = AcademicPDFParser.inspect_pdf_file(
                                    stored.local_path,
                                    min_text_chars=settings.MIN_EXTRACTED_PDF_TEXT_CHARS
                                )
                                if validation["is_valid"]:
                                    dest_path = stored.local_path
                                    storage_uri = stored.uri
                                    downloaded = True
                                    break
                                logger.debug("Downloaded bytes were PDF-like but invalid after inspection: %s", validation)
                                stored.local_path.unlink(missing_ok=True)

                    except Exception as client_err:
                        logger.debug(
                            "PDF download attempt %s via %s encountered exception for %s: %s",
                            attempt + 1,
                            mode,
                            pdf_url,
                            client_err
                        )
                        # If direct attempt fails or times out, fast-skip to avoid proxy retries on dead URLs
                        if mode == "direct":
                            skip_candidate = True
                            break
                        if mode == "proxy" and force_proxy_refresh:
                            try:
                                await proxy_manager.get_proxies(force_refresh=True)
                            except Exception:
                                pass

                    if downloaded or skip_candidate:
                        break

                if downloaded or skip_candidate:
                    break

            if not downloaded:
                logger.debug("No usable PDF downloaded from candidate URL: %s", pdf_url)
            if downloaded:
                break

        if not downloaded or not dest_path or not dest_path.exists():
            logger.info(
                "Full-text PDF not accessible for '%s...' (source may be paywalled or require institutional login). Continuing with abstract and metadata.",
                paper.original_title[:45] if paper.original_title else str(paper.id)
            )
            return False

        # Parse downloaded PDF
        try:
            parsed = AcademicPDFParser.parse_pdf(dest_path)
            full_text = parsed.get("full_text", "")
            sections = parsed.get("sections", {})
            if len(full_text.strip()) < settings.MIN_EXTRACTED_PDF_TEXT_CHARS:
                logger.info(
                    "Downloaded PDF for %s but extracted only %s chars; treating as not full-text usable.",
                    paper.id,
                    len(full_text.strip())
                )
                return False

            # Update Paper in database
            paper.full_text = full_text
            paper.pdf_local_path = storage_uri or str(dest_path)

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
        api_url = f"{settings.UNPAYWALL_BASE_URL.rstrip('/')}/{doi}?email={email}"
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
