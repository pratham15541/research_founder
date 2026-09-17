"""
Semantic Scholar Graph API client.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
from src.ingestion.base import BaseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

class SemanticScholarRetriever(BaseRetriever):
    """Retrieves academic papers from Semantic Scholar."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.S2_API_KEY
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    async def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search Semantic Scholar with required metadata fields."""
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,year,citationCount,externalIds,url,authors,openAccessPdf"
        }

        papers: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=self.headers) as client:
                response = await client.get(self.BASE_URL, params=params)
                if response.status_code != 200:
                    logger.warning(f"Semantic Scholar returned status {response.status_code}: {response.text[:200]}")
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

                    url = item.get("url") or (f"https://doi.org/{doi}" if doi else "")

                    papers.append({
                        "doi": doi,
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "authors": authors[:5],
                        "year": item.get("year") or 2024,
                        "citation_count": item.get("citationCount") or 0,
                        "source": "semantic_scholar",
                        "source_url": url,
                        "is_uploaded": False
                    })
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {e}")

        logger.info(f"Semantic Scholar retrieved {len(papers)} valid papers for query: '{query}'")
        return papers

