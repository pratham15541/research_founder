"""
Base abstract class for academic literature retrievers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Abstract interface for scientific literature retrieval clients."""

    @abstractmethod
    async def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Execute literature search.
        Must return normalized dictionary objects with:
        - doi: Optional[str]
        - title: str
        - abstract: str
        - authors: List[str]
        - year: int
        - citation_count: int
        - source: str
        - source_url: str
        """
        pass

