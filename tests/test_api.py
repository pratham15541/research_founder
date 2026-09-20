"""
Unit tests for FastAPI backend endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app
from src.config import settings


def _minimal_pdf_bytes() -> bytes:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Physics Informed ML\n\nAbstract\nThis is a valid uploaded academic PDF for testing.")
    data = doc.tobytes()
    doc.close()
    return data

@pytest.mark.asyncio
async def test_root_endpoint_redirects():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/", follow_redirects=False)
        assert res.status_code == 307
        assert res.headers["location"] == "/docs"

@pytest.mark.asyncio
async def test_favicon_endpoint_returns_204():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/favicon.ico")
        assert res.status_code == 204

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
        assert data["source"] == settings.PROXY_SOURCE_URLS

@pytest.mark.asyncio
async def test_proxy_refresh_endpoint(monkeypatch):
    async def fake_get_proxies(force_refresh=False):
        assert force_refresh is True
        return ["http://1.2.3.4:8080"]

    async def fake_get_random_proxy():
        return "http://1.2.3.4:8080"

    monkeypatch.setattr("src.api.main.proxy_manager.get_proxies", fake_get_proxies)
    monkeypatch.setattr("src.api.main.proxy_manager.get_random_proxy", fake_get_random_proxy)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/proxies/refresh")
        assert res.status_code == 200
        data = res.json()
        assert data["refreshed"] is True
        assert data["total_active_proxies"] == 1
        assert data["sample_proxy"] == "http://1.2.3.4:8080"

@pytest.mark.asyncio
async def test_pdf_upload_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        dummy_pdf_content = _minimal_pdf_bytes()
        files = {"file": ("test_paper.pdf", dummy_pdf_content, "application/pdf")}
        res = await ac.post("/api/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["filename"] == "test_paper.pdf"
        assert "file_id" in data
        assert data["pdf_validation"]["is_valid"] is True
