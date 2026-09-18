"""
Unit tests for Semantic Scholar rate-limit handling.
"""

import pytest

from src.ingestion.semantic_scholar import SemanticScholarRetriever


@pytest.mark.asyncio
async def test_semantic_scholar_429_enters_shared_cooldown(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        status_code = 429
        text = '{"message":"Too Many Requests","code":"429"}'
        headers = {"retry-after": "60"}

        def json(self):
            return {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            calls["count"] += 1
            return FakeResponse()

    monkeypatch.setattr("src.ingestion.semantic_scholar.httpx.AsyncClient", FakeAsyncClient)
    SemanticScholarRetriever._rate_limited_until = 0.0

    retriever = SemanticScholarRetriever()
    assert await retriever.search("physics informed neural networks") == []
    assert await retriever.search("physics informed machine learning") == []

    assert calls["count"] == 1
    assert SemanticScholarRetriever._is_rate_limited() is True

    SemanticScholarRetriever._rate_limited_until = 0.0
