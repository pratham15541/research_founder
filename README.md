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

## 🚀 Quickstart: Local Setup (Zero Docker Required!)

The entire system runs natively on Windows/Linux/macOS using standard Python `venv`. No Docker daemon or container installation is needed.

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 2. Environment Configuration
Create a `.env` file from `.env.example`:
```cmd
copy .env.example .env
```
Default `.env` settings will automatically run in local mode (SQLite + in-memory FAISS + local file storage).

### 3. Launching the Application (Standard Python `venv`)

1. **Setup & Install**:
   ```cmd
   .\setup_venv.bat
   ```
   *Or manually:*
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Backend** (Terminal 1):
   ```cmd
   .\run_backend.bat
   ```
   - Interactive Swagger API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

3. **Start the Streamlit Frontend** (Terminal 2):
   ```cmd
   .\run_frontend.bat
   ```
   - Interactive Web UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## ☁️ AWS Open-Source Tech Stack Migration

All components have been modernized to support the **AWS Open-Source Tech Stack** with zero Docker requirement:

| Component | AWS Cloud Service | Local Non-Docker Fallback | Implementation |
|---|---|---|---|
| **LLM Reasoning** | Amazon Bedrock (Claude 3 / Titan / Llama 3) | OpenRouter / NVIDIA NGC / Offline heuristics | `src/llm/bedrock_client.py` & `src/llm/llm_router.py` |
| **Embeddings** | Amazon Titan Embed Text v2 | SentenceTransformers (`all-MiniLM-L6-v2`) | `src/representation/bedrock_embeddings.py` |
| **Vector Store** | Amazon OpenSearch Serverless | In-memory FAISS (`FAISSVectorIndex`) | `src/rag/vector_store.py` & `src/rag/opensearch_index.py` |
| **Document Storage** | Amazon S3 | Local filesystem (`./data/uploads`) | `src/storage/s3_storage.py` |
| **Relational DB** | Amazon Aurora PostgreSQL | Local SQLite (`./data/cache/compounding_research.db`) | SQLAlchemy Async Session |

### Activating AWS (When AWS Credits / Keys are Added)
Once you have your AWS account credentials ready, simply update your `.env`:
```ini
USE_AWS=true
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your-s3-bucket-name
# Optional:
OPENSEARCH_ENDPOINT=https://your-opensearch-domain.us-east-1.es.amazonaws.com
```
Everything automatically routes to AWS over standard HTTPS APIs with **zero Docker required**.

---

## 🔬 Testing & Verification

Run the complete 31-test verification suite directly from your `venv`:
```cmd
.\venv\Scripts\python.exe -m pytest -v
```
All tests verify:
- API endpoints & health checks
- AWS Bedrock client payload formatting & response parsers
- LLM Router auto-detection & graceful offline fallbacks
- S3 Storage local filesystem fallback
- Vector store factory & FAISS indexing
- Clustering, Matrix aggregation, PDF parsing, and Devil's Advocate critique


