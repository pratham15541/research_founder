"""
OpenAlex REST API client with Polite Pool and abstract inverted-index reconstruction.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
from src.ingestion.base import BaseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct continuous text abstract from OpenAlex inverted index dictionary."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)

class OpenAlexRetriever(BaseRetriever):
    """Retrieves academic papers from OpenAlex with polite pool acceleration."""

    def __init__(self, email: Optional[str] = None):
        self.email = email or settings.OPENALEX_EMAIL
        self.base_url = settings.OPENALEX_BASE_URL
        self.headers = {
            "User-Agent": f"ResearchGraph-Matrix/1.0 (mailto:{self.email})"
        }

    async def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query OpenAlex works with relevance sorting and abstract requirement."""
        params = {
            "search": query,
            "filter": "has_abstract:true",
            "per-page": min(limit, 100),
            "sort": "relevance_score:desc"
        }

        papers: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=self.headers) as client:
                response = await client.get(self.base_url, params=params)
                if response.status_code != 200:
                    logger.warning(f"OpenAlex returned status code {response.status_code}: {response.text[:200]}")
                    return []
                data = response.json()
                results = data.get("results", [])

                for item in results:
                    title = item.get("title") or ""
                    abstract_index = item.get("abstract_inverted_index")
                    abstract = reconstruct_abstract(abstract_index)

                    # Tier 1 Noise Filter: Drop papers with short/missing abstracts
                    if not abstract or len(abstract.strip()) < 100 or not title:
                        continue

                    # Extract authors
                    authorships = item.get("authorships", [])
                    authors = []
                    for auth in authorships:
                        author_name = auth.get("author", {}).get("display_name")
                        if author_name:
                            authors.append(author_name)

                    # Extract DOI and URL
                    doi = item.get("doi")
                    if doi and doi.startswith("https://doi.org/"):
                        doi = doi.replace("https://doi.org/", "")

                    primary_location = item.get("primary_location") or {}
                    landing_url = primary_location.get("landing_page_url") or item.get("id", "")
                    pdf_url = primary_location.get("pdf_url")

                    papers.append({
                        "doi": doi,
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "authors": authors[:5],  # Top 5 authors
                        "year": item.get("publication_year") or 2024,
                        "citation_count": item.get("cited_by_count", 0),
                        "source": "openalex",
                        "source_url": pdf_url or landing_url,
                        "is_uploaded": False
                    })
        except Exception as e:
            logger.error(f"Error fetching from OpenAlex: {e}")

        logger.info(f"OpenAlex retrieved {len(papers)} valid papers for query: '{query}'")
        return papers
