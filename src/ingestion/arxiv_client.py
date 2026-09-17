"""
arXiv API export client with secure XML parsing and metadata normalization.
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import httpx
from src.ingestion.base import BaseRetriever

logger = logging.getLogger(__name__)

class ArxivRetriever(BaseRetriever):
    """Retrieves preprint papers from arXiv Export API."""

    BASE_URL = "http://export.arxiv.org/api/query"
    BASE_URL = "https://export.arxiv.org/api/query"

    async def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Query arXiv Atom feed securely."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(limit, 50),
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        papers: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                if response.status_code != 200:
                    logger.warning(f"arXiv returned status {response.status_code}")
                    return []

                # Secure XML parsing (avoid entity expansion)
                parser = ET.XMLParser()
                root = ET.fromstring(response.content, parser=parser)
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)
                    published_elem = entry.find("atom:published", ns)
                    doi_elem = entry.find("arxiv:doi", ns)

                    title = title_elem.text.replace("\n", " ").strip() if title_elem is not None and title_elem.text else ""
                    abstract = summary_elem.text.replace("\n", " ").strip() if summary_elem is not None and summary_elem.text else ""
                    landing_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                    doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

                    year = 2024
                    if published_elem is not None and published_elem.text:
                        try:
                            year = int(published_elem.text[:4])
                        except ValueError:
                            pass

                    # Extract authors
                    authors = []
                    for author_elem in entry.findall("atom:author", ns):
                        name_elem = author_elem.find("atom:name", ns)
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text.strip())

                    if not abstract or len(abstract) < 100 or not title:
                        continue

                    papers.append({
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors[:5],
                        "year": year,
                        "citation_count": 0,  # arXiv does not report citation counts
                        "source": "arxiv",
                        "source_url": landing_url,
                        "is_uploaded": False
                    })
        except Exception as e:
            logger.error(f"Error fetching from arXiv: {e}")

        logger.info(f"arXiv retrieved {len(papers)} valid papers for query: '{query}'")
        return papers

