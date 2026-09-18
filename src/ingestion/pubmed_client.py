"""
NCBI PubMed E-Utilities retriever for biomedical and life-science papers.
"""

import logging
from typing import List, Dict, Any
import httpx
from src.ingestion.base import BaseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

class PubMedRetriever(BaseRetriever):
    """Retrieves biomedical literature from NCBI Entrez E-Utilities."""

    def __init__(self):
        self.search_url = settings.PUBMED_SEARCH_URL
        self.summary_url = settings.PUBMED_SUMMARY_URL

    async def search(self, query: str, limit: int = 40) -> List[Dict[str, Any]]:
        """Search PubMed database and retrieve document summaries."""
        papers: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # 1. Search for PMIDs
                search_params = {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": min(limit, 50),
                    "sort": "pub_date"
                }
                res = await client.get(self.search_url, params=search_params)
                if res.status_code != 200:
                    return []
                id_list = res.json().get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return []

                # 2. Fetch summaries
                summary_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json"
                }
                sum_res = await client.get(self.summary_url, params=summary_params)
                if sum_res.status_code != 200:
                    return []
                result_data = sum_res.json().get("result", {})

                for pmid in id_list:
                    doc = result_data.get(pmid)
                    if not doc or not isinstance(doc, dict):
                        continue

                    title = doc.get("title", "").strip()
                    # PubMed summary has source/title; abstract can be in description or retrieved
                    # When summary is brief, use title and available descriptors
                    authors_raw = doc.get("authors", [])
                    authors = [a.get("name") for a in authors_raw if a.get("name")]
                    pubdate = doc.get("pubdate", "2024")
                    year = 2024
                    try:
                        year = int(pubdate[:4])
                    except (ValueError, TypeError):
                        pass

                    article_ids = doc.get("articleids", [])
                    doi = None
                    for aid in article_ids:
                        if aid.get("idtype") == "doi":
                            doi = aid.get("value")

                    # Note: Full abstract often requires efetch. When summary doesn't include it,
                    # we use source title and descriptors, or fall back to openalex/arxiv for deep text.
                    abstract = doc.get("source", "") + " - " + title
                    if len(abstract) < 50:
                        continue

                    papers.append({
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors[:5],
                        "year": year,
                        "citation_count": 0,
                        "source": "pubmed",
                        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "is_uploaded": False
                    })
        except Exception as e:
            logger.error(f"Error fetching from PubMed: {e}")

        logger.info(f"PubMed retrieved {len(papers)} valid papers for query: '{query}'")
        return papers
