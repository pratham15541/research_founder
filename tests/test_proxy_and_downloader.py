"""
Unit tests for ProxyManager and FullPaperDownloader.
"""

import pytest
from src.ingestion.proxy_manager import ProxyManager
from src.ingestion.full_paper_downloader import FullPaperDownloader

@pytest.mark.asyncio
async def test_proxy_manager_caching_and_rotation(tmp_path):
    # Initialize ProxyManager with temp cache path
    mgr = ProxyManager(cache_ttl_seconds=60)
    mgr.cache_file = tmp_path / "test_proxies.txt"

    # Pre-populate dummy proxies
    with open(mgr.cache_file, "w") as f:
        f.write("http://1.2.3.4:8080\nhttp://5.6.7.8:3128\n")

    proxies = await mgr.get_proxies()
    assert len(proxies) == 2
    assert "http://1.2.3.4:8080" in proxies

    random_proxy = await mgr.get_random_proxy()
    assert random_proxy in ["http://1.2.3.4:8080", "http://5.6.7.8:3128"]

def test_resolve_official_pdf_url():
    # arXiv abstract URL
    paper_arxiv = {
        "source": "arxiv",
        "source_url": "https://arxiv.org/abs/2304.05678",
        "doi": None
    }
    pdf_url = FullPaperDownloader.resolve_official_pdf_url(paper_arxiv)
    assert pdf_url == "https://arxiv.org/pdf/2304.05678.pdf"

    # Explicit openAccessPdf
    paper_s2 = {
        "source": "semantic_scholar",
        "source_url": "https://example.com/paper",
        "pdf_url": "https://openaccess.thecvf.com/content/paper.pdf"
    }
    assert FullPaperDownloader.resolve_official_pdf_url(paper_s2) == "https://openaccess.thecvf.com/content/paper.pdf"

    # Direct PDF in source_url
    paper_direct = {
        "source": "openalex",
        "source_url": "https://research.org/downloads/paper.pdf"
    }
    assert FullPaperDownloader.resolve_official_pdf_url(paper_direct) == "https://research.org/downloads/paper.pdf"

