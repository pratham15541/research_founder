"""
Ingestion module for academic papers and PDF parsing.
Ingestion module for academic papers, PDF parsing, proxy rotation, and full paper download.
"""

from src.ingestion.base import BaseRetriever
from src.ingestion.openalex import OpenAlexRetriever
from src.ingestion.semantic_scholar import SemanticScholarRetriever
from src.ingestion.arxiv_client import ArxivRetriever
from src.ingestion.pubmed_client import PubMedRetriever
from src.ingestion.pdf_parser import AcademicPDFParser
from src.ingestion.proxy_manager import ProxyManager, proxy_manager
from src.ingestion.full_paper_downloader import FullPaperDownloader
from src.ingestion.hybrid import HybridIngestionEngine

__all__ = [
    "BaseRetriever",
    "OpenAlexRetriever",
    "SemanticScholarRetriever",
    "ArxivRetriever",
    "PubMedRetriever",
    "AcademicPDFParser",
    "ProxyManager",
    "proxy_manager",
    "FullPaperDownloader",
    "HybridIngestionEngine"
]

