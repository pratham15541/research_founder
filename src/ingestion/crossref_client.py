"""
Crossref REST API client.
Adds a generic metadata source so retrieval does not depend too heavily on
Semantic Scholar or any single rate-limited provider.
"""

import logging
import re
from typing import List, Dict, Any

import httpx

from src.config import settings
from src.ingestion.base import BaseRetriever

logger = logging.getLogger(__name__)


def _clean_abstract(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CrossrefRetriever(BaseRetriever):
    """Retrieves paper metadata from Crossref."""

    def __init__(self):
        self.base_url = settings.CROSSREF_BASE_URL

    async def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "rows": min(limit, 50),
            "sort": "relevance",
            "select": "DOI,title,abstract,author,published-print,published-online,published,URL,is-referenced-by-count,link"
        }
        headers = {
            "User-Agent": f"ResearchGraph-Matrix/1.0 (mailto:{settings.OPENALEX_EMAIL})"
        }

        papers: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                res = await client.get(self.base_url, params=params)
                if res.status_code != 200:
                    logger.warning("Crossref returned status %s: %s", res.status_code, res.text[:200])
                    return []
                items = res.json().get("message", {}).get("items", [])

                for item in items:
                    title_list = item.get("title") or []
                    title = title_list[0].strip() if title_list else ""
                    abstract = _clean_abstract(item.get("abstract", ""))
                    if not title or len(abstract) < 80:
                        continue

                    authors = []
                    for author in item.get("author") or []:
                        given = author.get("given", "")
                        family = author.get("family", "")
                        name = " ".join(x for x in [given, family] if x).strip()
                        if name:
                            authors.append(name)

                    year = 2024
                    for date_key in ("published-print", "published-online", "published"):
                        parts = (item.get(date_key) or {}).get("date-parts") or []
                        if parts and parts[0]:
                            try:
                                year = int(parts[0][0])
                                break
                            except (TypeError, ValueError):
                                pass

                    pdf_url = None
                    for link in item.get("link") or []:
                        content_type = (link.get("content-type") or "").lower()
                        url = link.get("URL")
                        if url and ("pdf" in content_type or url.lower().endswith(".pdf")):
                            pdf_url = url
                            break

                    papers.append({
                        "doi": item.get("DOI"),
                        "title": title,
                        "abstract": abstract,
                        "authors": authors[:5],
                        "year": year,
                        "citation_count": item.get("is-referenced-by-count") or 0,
                        "source": "crossref",
                        "source_url": pdf_url or item.get("URL", ""),
                        "is_uploaded": False
                    })
        except Exception as e:
            logger.error("Error fetching from Crossref: %s", e)

        logger.info("Crossref retrieved %s valid papers for query: '%s'", len(papers), query)
        return papers
