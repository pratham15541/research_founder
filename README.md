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
4. **Compounding PostgreSQL + pgvector Corpus**: Ingests from OpenAlex, Semantic Scholar, arXiv, PubMed, and user-uploaded PDFs, deduplicating and indexing papers once so the system grows faster and smarter with every query.
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
GEMINI_API_KEY=your_gemini_api_key       # For LLM labeling & ranking
S2_API_KEY=                              # Optional Semantic Scholar API key
```

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

---

## 🔬 Testing & Verification Protocols
- **Run the Complete Test Suite (13 tests across API, storage, clustering, matrix, PDF, proxies)**:
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

