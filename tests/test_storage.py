"""
Unit tests for compounding database caching and deduplication.
Runs with SQLite async in-memory engine.
"""

import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.storage.models import Base, Paper
from src.storage.cache import CompoundingCacheEngine, normalize_title
from src.config import settings
from src.storage.file_storage import LocalFileStorageBackend, get_file_storage_backend

@pytest.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session
        await session.close()
    await engine.dispose()

@pytest.mark.asyncio
async def test_title_normalization():
    t1 = "Deep Learning: A Comprehensive Survey!"
    t2 = "deep learning a comprehensive survey"
    assert normalize_title(t1) == normalize_title(t2)

@pytest.mark.asyncio
async def test_compounding_cache_deduplication(async_session):
    papers_batch_1 = [
        {
            "doi": "10.1234/test.doi.1",
            "title": "Quantum Error Correction with Surface Codes",
            "abstract": "We explore planar code architectures for topological quantum memory.",
            "year": 2024,
            "source": "arxiv",
            "source_url": "https://arxiv.org/abs/1234"
        },
        {
            "doi": "10.1234/test.doi.2",
            "title": "State Space Models for Long Horizon Forecasting",
            "abstract": "Mamba and S4 architectures for sub-quadratic sequence modeling.",
            "year": 2024,
            "source": "openalex",
            "source_url": "https://openalex.org/W1234"
        }
    ]

    # Initial Insert
    cached, to_insert = await CompoundingCacheEngine.find_existing_papers(async_session, papers_batch_1)
    assert len(cached) == 0
    assert len(to_insert) == 2

    inserted = await CompoundingCacheEngine.insert_new_papers(async_session, to_insert)
    assert len(inserted) == 2

    # Second Query with duplicate DOI and duplicate title
    papers_batch_2 = [
        {
            "doi": "10.1234/test.doi.1",  # Same DOI
            "title": "Different Title Representation",
            "abstract": "Duplicate test",
            "year": 2024,
            "source": "s2",
            "source_url": ""
        },
        {
            "doi": None,  # No DOI, but normalized title matches paper 2!
            "title": "State-Space Models for Long-Horizon Forecasting!",
            "abstract": "Duplicate by normalized title match",
            "year": 2024,
            "source": "pubmed",
            "source_url": ""
        },
        {
            "doi": "10.1234/test.doi.3",  # Genuinely new paper
            "title": "Novel Benchmark for Agentic Coding",
            "abstract": "New benchmark dataset.",
            "year": 2025,
            "source": "openalex",
            "source_url": ""
        }
    ]

    cached_2, to_insert_2 = await CompoundingCacheEngine.find_existing_papers(async_session, papers_batch_2)

    # 2 existing papers recognized from compounding cache, only 1 new paper to insert!
    assert len(cached_2) == 2
    assert len(to_insert_2) == 1
    assert to_insert_2[0]["doi"] == "10.1234/test.doi.3"

def test_s3_backend_falls_back_to_local_when_endpoint_unavailable(monkeypatch):
    class BrokenS3Backend:
        def __init__(self):
            raise RuntimeError("Could not connect to the endpoint URL")

    monkeypatch.setattr(settings, "STORAGE_BACKEND", "s3")
    monkeypatch.setattr(settings, "STORAGE_FALLBACK_TO_LOCAL", True)
    monkeypatch.setattr("src.storage.file_storage.S3FileStorageBackend", BrokenS3Backend)

    backend = get_file_storage_backend()

    assert isinstance(backend, LocalFileStorageBackend)
