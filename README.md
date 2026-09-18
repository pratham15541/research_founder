# Research Landscape, Gap Matrix & Knowledge Graph Tool

[![Architecture](https://img.shields.io/badge/Architecture-LangGraph-blue.svg)](docs/02_SYSTEM_ARCHITECTURE.md)
[![Database](https://img.shields.io/badge/Database-PostgreSQL_16_+_pgvector-336791.svg)](docs/03_TECH_STACK_AND_INFRASTRUCTURE.md)
[![Status](https://img.shields.io/badge/Status-Specification_Complete-success.svg)](docs/06_PROGRESS_TASK_PLAN_AND_VERIFICATION.md)

An evidence-backed, combinatorial research intelligence platform that discovers unexplored research opportunities by crossing field dimensions, estimating feasibility from component parts, challenging suggestions with self-critique, and grounding every claim in a compounding paper corpus.

---

## 🌟 Key Differentiators & USPs

1. **Combinatorial 2D Matrix Engine**: Crosses two orthogonal dimensions ($D_A \times D_B$) to calculate cell density, temporal growth, and identify true structural voids.
2. **Component-Extrapolated General Feasibility**: Decomposes untried combinations into component resource profiles (compute, data readiness, theoretical maturity) to produce objective feasibility estimates.
3. **Automated Devil's Advocate / Self-Critique**: Employs an adversarial critique node to force the model to identify the single strongest technical failure mode for every suggested gap.
4. **Compounding PostgreSQL + pgvector Corpus**: Ingests from configurable OpenAlex, Semantic Scholar, arXiv, PubMed, Crossref, and user-uploaded PDF sources, deduplicating and indexing papers once so the system grows faster and smarter with every query.
5. **Double-Column IEEE/ACM PDF Parser + Future Work Seeding**: Extracts structured sections and author-written future work statements from uploaded PDFs to seed matrix gaps.
6. **Condition Decision Trees**: Provides an explainable, rule-grounded trace explaining *why* the AI decided a direction has high potential.
7. **Synchronized Dual-View Visualization**: Real-time bi-directional linking between a 2D Heatmap Matrix and an interactive NetworkX Citation/Similarity Knowledge Graph.

---

## 📚 Complete Project Documentation Suite

The complete architectural, algorithmic, and operational specifications are available in `docs/`:

| Document | Description | Link |
|---|---|---|
| **01: Vision & USPs** | Problem statement, honest competitive positioning, and detailed breakdown of the 6 USPs. | [01_PROJECT_VISION_AND_USPS.md](docs/01_PROJECT_VISION_AND_USPS.md) |
| **02: System Architecture** | High-level architecture, LangGraph state machine, data models, and database schema. | [02_SYSTEM_ARCHITECTURE.md](docs/02_SYSTEM_ARCHITECTURE.md) |
| **03: Tech Stack & Deployment** | Local open-source Docker Compose vs. AWS Serverless production architecture. | [03_TECH_STACK_AND_INFRASTRUCTURE.md](docs/03_TECH_STACK_AND_INFRASTRUCTURE.md) |
| **04: Pipeline & Methodology** | Mathematical formulation for clustering, adjacency filtering, feasibility, and prompt schemas. | [04_PIPELINE_AND_METHODOLOGY_SPEC.md](docs/04_PIPELINE_AND_METHODOLOGY_SPEC.md) |
| **05: Evaluation & Noise Reduction**| Multi-tier noise reduction, Silhouette gating, extrinsic survey benchmarks, and human rubric. | [05_EVALUATION_NOISE_REDUCTION_BENCHMARKS.md](docs/05_EVALUATION_NOISE_REDUCTION_BENCHMARKS.md) |
| **06: Progress & Verification** | Phase roadmap, tech stack task division, and 6 explicit user manual verification protocols. | [06_PROGRESS_TASK_PLAN_AND_VERIFICATION.md](docs/06_PROGRESS_TASK_PLAN_AND_VERIFICATION.md) |

---

## 🚀 Quickstart: Local Open-Source Setup

### 1. Prerequisites
- Docker & Docker Compose ($2.20+$)
- Python 3.11 or 3.12
- Git

### 2. Environment Configuration
Create a `.env` file from the template:
```bash
cp .env.example .env
```
Populate the environment variables:
```ini
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres_secure_pass
POSTGRES_DB=research_db
OPENALEX_EMAIL=your_email@university.edu # For OpenAlex Polite Pool (10 req/sec)
S2_API_KEY=                              # Optional Semantic Scholar API key
INGESTION_SOURCES=openalex,semantic_scholar,arxiv,pubmed,crossref

# OpenAI-compatible LLM providers. The app tries primary, then numbered fallbacks.
NVIDIA_API_KEY=
NVIDIA_MODEL=
NVIDIA_BASE_URL=
NVIDIA_API_KEY1=
NVIDIA_MODEL1=
NVIDIA_BASE_URL1=
```

All paper API URLs are environment-driven in `.env.example`, including OpenAlex,
Semantic Scholar, arXiv, PubMed, Crossref, and Unpaywall. Leave a source out of
`INGESTION_SOURCES` to disable it.

### 3. Launching the Application

You can run either via **Docker Compose** (full stack in containers) or directly on host with **`uv`**:

#### Option A: Run directly with `uv` (Fastest, uses host resources)
Since your `pgvector` container is already running on port 5432:

1. **Start the FastAPI Backend** (Terminal 1):
   ```bash
   uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   Interactive Swagger API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

2. **Start the Streamlit Frontend** (Terminal 2):
   ```bash
   uv run streamlit run src/ui/app.py --server.port=8501 --server.address=127.0.0.1
   ```
   Interactive Web UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

#### Option B: Run everything with Docker Compose
```bash
# Spins up PostgreSQL 16 + pgvector, FastAPI backend, and Streamlit frontend
docker compose -f docker/docker-compose.yml up -d
```

#### Option C: Run with LocalStack S3 storage
LocalStack is included in the Docker Compose stack. To store uploaded and downloaded PDFs in the local S3 mock, set:

```ini
STORAGE_BACKEND=s3
S3_BUCKET=research-uploads
S3_ENDPOINT_URL=http://localstack:4566
S3_REGION=us-east-1
S3_PREFIX=papers
```

Then run:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

For host-based `uv` runs, keep LocalStack running via Compose and use:

```ini
S3_ENDPOINT_URL=http://127.0.0.1:4566
```

The app still keeps a local parser cache in `data/uploads`, while the durable storage URI returned by uploads becomes `s3://research-uploads/papers/...`.
If S3/LocalStack is temporarily unreachable and `STORAGE_FALLBACK_TO_LOCAL=true`, the PDF is kept locally instead of failing the download enrichment step.

### 4. Dynamic PDF download and proxy rotation

Full-text enrichment tries every legal PDF candidate it can derive: explicit API PDF URLs, arXiv PDF URLs, direct publisher PDF URLs, and DOI-based Unpaywall fallbacks. Each candidate is validated as a real parseable PDF before it is accepted, which reduces both false positives and false negatives.

Proxy sources, cache TTL, retry count, and direct fallback are controlled through `.env`:

```ini
PROXY_SOURCE_URLS=https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/http.txt,https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/https.txt
PROXY_CACHE_TTL_SECONDS=1800
PROXY_REFRESH_ON_FAILURES=2
PROXY_USE_DIRECT_FALLBACK=true
PDF_DOWNLOAD_MAX_RETRIES=4
PDF_DOWNLOAD_TIMEOUT_SECONDS=25
PDF_DOWNLOAD_DIRECT_FIRST=true
SEMANTIC_SCHOLAR_RATE_LIMIT_COOLDOWN_SECONDS=900
```

Refresh the proxy cache manually when downloads start getting blocked:

```bash
curl -X POST http://127.0.0.1:8000/api/proxies/refresh
```

---

## 🔬 Testing & Verification Protocols
- **Run the Complete Test Suite (27 tests across API, storage, clustering, matrix, PDF, proxies, RAG, synthesis, and LLM provider fallback)**:
  ```bash
  uv run pytest tests/
  ```
- **15–20 Sample Classification Agreement Audit**:
  ```bash
  uv run python scripts/sample_validation.py
  ```
- **Pre-fetch Topic for Offline Demo (Zero live API calls during judged presentation)**:
  ```bash
  uv run python scripts/prefetch_topic.py --topic "Physics-Informed Neural Networks" --size 60
  ```
