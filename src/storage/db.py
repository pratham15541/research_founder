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
    """Initialize database tables, automatically falling back to SQLite if PostgreSQL is unreachable."""
    global _engine, AsyncSessionLocal, _db_url
    try:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Database schema initialized successfully using: {_db_url}")
    except Exception as e:
        if "sqlite" not in _db_url:
            logger.warning(
                f"PostgreSQL connection failed ({e}). Falling back to local SQLite database (no Docker required)..."
            )
            fallback_path = (settings.CACHE_DIR / "compounding_research.db").as_posix()
            _db_url = f"sqlite+aiosqlite:///{fallback_path}"
            _engine = create_async_engine(_db_url, echo=False)
            AsyncSessionLocal = async_sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info(f"Local SQLite database initialized at {_db_url}")
        else:
            raise

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

