"""
Semantic Scholar Graph API client.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
import httpx
from src.ingestion.base import BaseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

class SemanticScholarRetriever(BaseRetriever):
    """Retrieves academic papers from Semantic Scholar."""

    _rate_limit_lock = asyncio.Lock()
    _rate_limited_until = 0.0

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.S2_API_KEY
        self.base_url = settings.SEMANTIC_SCHOLAR_BASE_URL
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    @classmethod
    def _is_rate_limited(cls) -> bool:
        return time.monotonic() < cls._rate_limited_until

    @classmethod
    def _set_rate_limit_cooldown(cls, retry_after: Optional[str] = None) -> int:
        cooldown = settings.SEMANTIC_SCHOLAR_RATE_LIMIT_COOLDOWN_SECONDS
        if retry_after:
            try:
                cooldown = max(1, int(float(retry_after)))
            except ValueError:
                pass
        cls._rate_limited_until = time.monotonic() + cooldown
        return cooldown

    async def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search Semantic Scholar with required metadata fields."""
        if self._is_rate_limited():
            logger.info("Semantic Scholar skipped during active rate-limit cooldown.")
            return []

        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,year,citationCount,externalIds,url,authors,openAccessPdf"
        }

        papers: List[Dict[str, Any]] = []
        try:
            async with self._rate_limit_lock:
                if self._is_rate_limited():
                    logger.info("Semantic Scholar skipped during active rate-limit cooldown.")
                    return []

                async with httpx.AsyncClient(timeout=15.0, headers=self.headers) as client:
                    response = await client.get(self.base_url, params=params)
                    if response.status_code == 429:
                        cooldown = self._set_rate_limit_cooldown(response.headers.get("retry-after"))
                        logger.warning(
                            "Semantic Scholar rate limit hit; cooling down this source for %ss. "
                            "Other ingestion sources will continue.",
                            cooldown
                        )
                        return []
                    if response.status_code != 200:
                        logger.warning("Semantic Scholar returned status %s: %s", response.status_code, response.text[:200])
                        return []
                    data = response.json()
                    results = data.get("data", [])

            for item in results:
                title = item.get("title") or ""
                abstract = item.get("abstract") or ""

                # Tier 1 Noise Filter
                if not abstract or len(abstract.strip()) < 100 or not title:
                    continue

                # Extract DOI
                external_ids = item.get("externalIds") or {}
                doi = external_ids.get("DOI")

                authors_data = item.get("authors") or []
                authors = [a.get("name") for a in authors_data if a.get("name")]

                open_access_pdf = item.get("openAccessPdf") or {}
                pdf_url = open_access_pdf.get("url")
                url = pdf_url or item.get("url") or (f"https://doi.org/{doi}" if doi else "")

                papers.append({
                    "doi": doi,
                    "title": title.strip(),
                    "abstract": abstract.strip(),
                    "authors": authors[:5],
                    "year": item.get("year") or 2024,
                    "citation_count": item.get("citationCount") or 0,
                    "source": "semantic_scholar",
                    "source_url": url,
                    "pdf_url": pdf_url,
                    "is_uploaded": False
                })
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {e}")

        logger.info(f"Semantic Scholar retrieved {len(papers)} valid papers for query: '{query}'")
        return papers
