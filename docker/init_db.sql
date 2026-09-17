-- Database Initialization for Research Landscape & Gap Matrix Tool
-- Enables pgvector and creates relational + vector tables

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Papers Table (Compounding Research Knowledge Base)
CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doi VARCHAR(255) UNIQUE,
    normalized_title VARCHAR(512) NOT NULL,
    original_title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    full_text TEXT,
    authors JSONB DEFAULT '[]'::jsonb,
    publication_year INTEGER NOT NULL,
    citation_count INTEGER DEFAULT 0,
    source VARCHAR(64) NOT NULL, -- 'openalex', 's2', 'arxiv', 'pubmed', 'uploaded_pdf'
    source_url TEXT NOT NULL,
    is_uploaded BOOLEAN DEFAULT FALSE,
    pdf_local_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_norm_title ON papers(normalized_title);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(publication_year);

-- 2. Embeddings Table (Vector Storage using all-MiniLM-L6-v2 384-dim)
CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id UUID PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
ON paper_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 3. Paper Sections Table (For open-access and uploaded PDF full-text)
CREATE TABLE IF NOT EXISTS paper_sections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_type VARCHAR(64) NOT NULL, -- 'abstract', 'methods', 'limitations', 'future_work'
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sections_paper ON paper_sections(paper_id);
CREATE INDEX IF NOT EXISTS idx_sections_type ON paper_sections(section_type);

-- 4. Analysis Runs Table (Historical Topic Runs)
CREATE TABLE IF NOT EXISTS analysis_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    expanded_queries JSONB NOT NULL,
    corpus_size INTEGER NOT NULL,
    silhouette_score NUMERIC(5, 4),
    axis_a_name VARCHAR(128) NOT NULL,
    axis_b_name VARCHAR(128) NOT NULL,
    axis_a_labels JSONB NOT NULL,
    axis_b_labels JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Gap Results Table (Candidate & Ranked Gaps)
CREATE TABLE IF NOT EXISTS gap_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    axis_a_val VARCHAR(128) NOT NULL,
    axis_b_val VARCHAR(128) NOT NULL,
    cell_paper_count INTEGER NOT NULL,
    novelty_score NUMERIC(3, 2) NOT NULL,
    feasibility_score NUMERIC(3, 2) NOT NULL,
    impact_score NUMERIC(3, 2) NOT NULL,
    composite_score NUMERIC(3, 2) NOT NULL,
    counter_argument TEXT NOT NULL,
    grounding_evidence JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

