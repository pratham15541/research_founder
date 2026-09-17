"""
Database connection manager and async session factory.
Provides seamless execution on PostgreSQL 16 + pgvector with adaptive SQLite fallback.
"""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.config import settings
from src.storage.models import Base

logger = logging.getLogger(__name__)

# Determine active database URL
_db_url = settings.DATABASE_URL

# Allow using SQLite for local testing without docker
if "sqlite" in _db_url:
    _engine = create_async_engine(_db_url, echo=False)
else:
    try:
        _engine = create_async_engine(
            _db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL ({e}). Falling back to local SQLite.")
        fallback_path = settings.CACHE_DIR / "compounding_research.db"
        _db_url = f"sqlite+aiosqlite:///{fallback_path}"
        _engine = create_async_engine(_db_url, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def init_db() -> None:
    """Initialize database tables."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency / context generator for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

