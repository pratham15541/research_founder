"""
SQLAlchemy ORM models for research papers, embeddings, sections, and gap analysis.
Supports pgvector vector(384) on PostgreSQL and JSON/Pickle fallback on SQLite.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    JSON,
    Uuid
)
from sqlalchemy.orm import declarative_base, relationship

# Try importing pgvector Vector type, with fallback for SQLite testing
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None

Base = declarative_base()

class Paper(Base):
    """Core academic paper record."""
    __tablename__ = "papers"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    doi = Column(String(255), unique=True, index=True, nullable=True)
    normalized_title = Column(String(512), index=True, nullable=False)
    original_title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    full_text = Column(Text, nullable=True)
    authors = Column(JSON, default=list)
    publication_year = Column(Integer, index=True, nullable=False)
    citation_count = Column(Integer, default=0)
    source = Column(String(64), nullable=False)  # 'openalex', 's2', 'arxiv', 'pubmed', 'uploaded_pdf'
    source_url = Column(Text, nullable=False)
    is_uploaded = Column(Boolean, default=False)
    pdf_local_path = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    embedding_rel = relationship("PaperEmbedding", back_populates="paper", uselist=False, cascade="all, delete-orphan")
    sections = relationship("PaperSection", back_populates="paper", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else "",
            "doi": self.doi,
            "title": self.original_title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.publication_year,
            "citation_count": self.citation_count,
            "source": self.source,
            "source_url": self.source_url,
            "is_uploaded": self.is_uploaded,
            "full_text": self.full_text or "",
            "pdf_local_path": self.pdf_local_path,
            "full_text_available": bool(self.full_text and len(self.full_text.strip()) > 0)
        }

class PaperEmbedding(Base):
    """Vector representation for paper abstracts."""
    __tablename__ = "paper_embeddings"

    paper_id = Column(Uuid, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    
    # Use Vector(384) if pgvector is active, otherwise fallback to JSON array
    if HAS_PGVECTOR and Vector is not None:
        embedding = Column(Vector(384), nullable=False)
    else:
        embedding = Column(JSON, nullable=False)

    model_name = Column(String(128), nullable=False, default="all-MiniLM-L6-v2")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    paper = relationship("Paper", back_populates="embedding_rel")

class PaperSection(Base):
    """Segmented full-text section (methods, limitations, future work)."""
    __tablename__ = "paper_sections"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    paper_id = Column(Uuid, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    section_type = Column(String(64), nullable=False, index=True)  # 'abstract', 'methods', 'limitations', 'future_work'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    paper = relationship("Paper", back_populates="sections")

class AnalysisRun(Base):
    """Historical record of an analysis run on a research topic."""
    __tablename__ = "analysis_runs"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    expanded_queries = Column(JSON, default=list)
    corpus_size = Column(Integer, nullable=False)
    silhouette_score = Column(Numeric(5, 4), nullable=True)
    axis_a_name = Column(String(128), nullable=False)
    axis_b_name = Column(String(128), nullable=False)
    axis_a_labels = Column(JSON, default=list)
    axis_b_labels = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    gap_results = relationship("GapResult", back_populates="run", cascade="all, delete-orphan")

class GapResult(Base):
    """Evaluated candidate or ranked gap for an analysis run."""
    __tablename__ = "gap_results"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id = Column(Uuid, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    axis_a_val = Column(String(128), nullable=False)
    axis_b_val = Column(String(128), nullable=False)
    cell_paper_count = Column(Integer, nullable=False)
    novelty_score = Column(Numeric(3, 2), nullable=False)
    feasibility_score = Column(Numeric(3, 2), nullable=False)
    impact_score = Column(Numeric(3, 2), nullable=False)
    composite_score = Column(Numeric(3, 2), nullable=False)
    counter_argument = Column(Text, nullable=False)
    grounding_evidence = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    run = relationship("AnalysisRun", back_populates="gap_results")
