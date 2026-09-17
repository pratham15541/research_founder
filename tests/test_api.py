"""
Unit tests for FastAPI backend endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "research-landscape-backend"

@pytest.mark.asyncio
async def test_proxies_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/api/proxies")
        assert res.status_code == 200
        data = res.json()
        assert "total_active_proxies" in data
        assert data["source"] == "iplocate/free-proxy-list"

@pytest.mark.asyncio
async def test_pdf_upload_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        dummy_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Title (Physics Informed ML)\n>>\nendobj\n"
        files = {"file": ("test_paper.pdf", dummy_pdf_content, "application/pdf")}
        res = await ac.post("/api/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["filename"] == "test_paper.pdf"
        assert "file_id" in data

