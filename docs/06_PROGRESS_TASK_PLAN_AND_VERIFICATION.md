# Document 06: Progress Task Plan & User Manual Verification Guide

## 1. Master Progress Tracker & Phase Breakdown

This tracker documents the development roadmap, technical dependencies, and completion status across all six phases.

### Phase 1: Infrastructure, Docker & Compounding DB
- [x] **Task 1.1**: Master Architecture & Documentation Specifications (`docs/01`–`docs/06`).
- [x] **Task 1.2**: Docker Compose configuration (`postgres` with `pgvector:pg16`, `backend`, `frontend`).
- [x] **Task 1.3**: Database schema migration script (`init_db.sql`) with vector cosine index (`HNSW`).
- [x] **Task 1.4**: SQLAlchemy 2.0 async data models (`Paper`, `PaperEmbedding`, `PaperSection`, `MatrixCell`).
- [x] **Task 1.5**: Compounding Cache Engine (DOI and normalized title deduplication layer).

### Phase 2: Hybrid Multi-Source Ingestion & PDF Parser
- [x] **Task 2.1**: OpenAlex REST API client with Polite Pool email header and rate-limit guard.
- [x] **Task 2.2**: Semantic Scholar Graph API client (with optional API key fallback).
- [x] **Task 2.3**: arXiv API and NCBI PubMed E-Utilities clients (query-weighted routing).
- [x] **Task 2.4**: Double-column academic PDF parser (`PyMuPDF` / `fitz` + layout column ordering).
- [x] **Task 2.5**: Section segmenter for `Abstract`, `Methods`, `Limitations`, and `Future Work`.
- [x] **Task 2.6**: Secure file upload endpoint with `%PDF-` magic-bytes check and UUID sandbox.
- [x] **Task 2.7**: Free Proxy Manager (`iplocate/free-proxy-list`) and Full Paper Downloader.

### Phase 3: Unsupervised Dimension Discovery & Noise Filter
- [x] **Task 3.1**: SentenceTransformer embedding pipeline (`all-MiniLM-L6-v2` / `SciBERT`).
- [x] **Task 3.2**: UMAP dimension reduction ($384 \to 10$ dimensions).
- [x] **Task 3.3**: HDBSCAN clustering engine with outlier absorption logic.
- [x] **Task 3.4**: Silhouette Score cohesion gate ($\bar{S} \ge 0.15$) and automatic KMeans fallback.
- [x] **Task 3.5**: LLM semantic cluster labeling with structured JSON schema.
- [x] **Task 3.6**: Structured paper classification into Axis A $\times$ Axis B with confidence tags.

### Phase 4: Deterministic Matrix Aggregator & Gap Filter
- [x] **Task 4.1**: 2D Grid accumulator ($C(i, j)$ density computation in pure Python/Pandas).
- [x] **Task 4.2**: Dynamic sparsity threshold calculator ($S = \max(1, \lfloor 0.02 \times N \rfloor)$).
- [x] **Task 4.3**: Adjacency filter (must be adjacent to $\ge 1$ populated cell in row or column).
- [x] **Task 4.4**: Isolated empty cell rejector (drops non-viable combinatorial voids).
- [x] **Task 4.5**: Multi-neighbor adjacency weighting score.

### Phase 5: Feasibility Decomposition, Ranking & Self-Critique
- [x] **Task 5.1**: Component feasibility decomposition engine (Resource Profile A $\times$ B).
- [x] **Task 5.2**: Neighbor-grounded ranking prompt (injecting retrieved titles/abstracts).
- [x] **Task 5.3**: Devil's Advocate / Adversarial Critique Node (strongest failure reason).
- [x] **Task 5.4**: Condition decision tree generator (rules triggered for why gap is viable).
- [x] **Task 5.5**: Clickable evidence and citation linker.

### Phase 6: Interactive UI & Dual-View Synchronization
- [x] **Task 6.1**: Streamlit responsive layout with topic search, upload panel, and cache toggles.
- [x] **Task 6.2**: Plotly 2D Heatmap component with gold border highlighting for candidate gaps.
- [x] **Task 6.3**: NetworkX / Pyvis interactive citation and semantic knowledge graph.
- [x] **Task 6.4**: Bi-directional cell click sync: clicking heatmap cell filters graph nodes.
- [x] **Task 6.5**: Gap Dossier slide-out cards with feasibility scores, counter-argument, and papers.
- [x] **Task 6.6**: Pre-fetching cache script for offline, sub-2-second judged presentations.

---

## 2. Task Division by Technology Stack

```
┌───────────────────────────┬──────────────────────────────────────────────────────────────┐
│ Technology / Layer        │ Assigned Responsibilities & Modules                          │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ PostgreSQL 16 + pgvector  │ Compounding paper store, HNSW cosine index, section storage  │
│ FastAPI (Backend)         │ REST endpoints, file upload validation, LangGraph runner     │
│ LangGraph (Orchestration) │ Execution state machine, deterministic & LLM node transitions│
│ PyTorch & Scikit-Learn    │ SentenceTransformers, UMAP, HDBSCAN, Silhouette gating       │
│ PyMuPDF (fitz)            │ Double-column PDF extraction, regex section segmentation     │
│ Streamlit & Plotly        │ 2D Heatmap, gap dossiers, evaluation dashboards              │
│ NetworkX & Pyvis          │ Citation and cosine similarity graph visualization           │
│ Google Gemini / Bedrock   │ Query expansion, cluster labeling, classification, critique  │
└───────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Manual Verification Guide (For the User)

This section details the **exact verification protocols** you should run to validate system accuracy, reliability, and academic rigor before demonstration.

### Protocol 1: Compounding Database & Zero-API Verification
- **Goal**: Confirm that querying an existing topic pulls 100% from PostgreSQL without making redundant API calls.
- **Action**:
  1. Execute a query for *"Diffusion Models in Medical Imaging"* with 50 papers.
  2. Inspect the database: `docker exec -it research_postgres psql -U postgres -d research_db -c "SELECT count(*) FROM papers;"`.
  3. Re-run the identical query.
  4. Inspect logs: assert `Retrieved 50 papers from local compounding cache (0 external API calls)`.
- **Pass Criteria**: Elapsed time $< 500\text{ms}$, 0 external HTTP requests.

### Protocol 2: 15–20 Hand-Labeled Classification Audit
- **Goal**: Ensure the LLM dimensional classification achieves $\ge 80\%$ agreement with human judgment.
- **Action**:
  1. Sample 20 papers from the retrieved corpus using `scripts/sample_validation.py`.
  2. Manually classify each paper into Axis A and Axis B categories.
  3. Compare with the model's classifications in `data/validation/classification_audit.json`.
  4. Calculate agreement: $\text{Agreement} = \frac{\text{Matches}}{20}$.
- **Pass Criteria**: Agreement $\ge 80\%$. Low-confidence classifications must have the `confidence='low'` flag set.

### Protocol 3: Cluster Cohesion & Silhouette Gating Check
- **Goal**: Verify that diffuse or noisy cluster partitions are rejected and handled safely.
- **Action**:
  1. Inject an artificially noisy dataset (e.g., 30 random papers across unrelated fields).
  2. Trigger the clustering step.
  3. Check console logs for Silhouette Score output.
  4. Confirm that if $\bar{S} < 0.15$, the system triggers: `[GATE TRIGGERED] Silhouette score 0.08 < 0.15. Reverting to KMeans fallback with k=4`.
- **Pass Criteria**: Pipeline does not crash, clusters remain well-separated, and no single-paper clusters exist.

### Protocol 4: Candidate Gap Adjacency Validation
- **Goal**: Verify that isolated empty cells are never recommended as research gaps.
- **Action**:
  1. Inspect the generated 2D matrix in the UI.
  2. Locate a corner cell that has 0 papers and whose adjacent row and column cells also have 0 papers (an isolated void).
  3. Verify that this cell is **not** highlighted as a candidate gap.
  4. Locate a cell with 0 papers that is bordered by a cell with $\ge 15$ papers (e.g., *Methodology X* widely used in *Domain Y*, but 0 papers in *Domain Z* where *Domain Z* is otherwise active).
  5. Verify that this cell **is** flagged as a candidate gap.
- **Pass Criteria**: Zero isolated void recommendations; all candidate gaps border active research.

### Protocol 5: Devil's Advocate Critique Reality Check
- **Goal**: Confirm the AI is not sycophantic and provides credible failure modes.
- **Action**:
  1. Open the top-ranked gap dossier card.
  2. Read the `Counter-Argument / Failure Modes` section.
  3. Verify it does not simply say "this requires more research", but instead provides a concrete theoretical, mathematical, or empirical reason (e.g., "high latency makes real-time deployment unviable", "lack of paired training labels").
- **Pass Criteria**: The critique challenges the premise with specific domain constraints.

### Protocol 6: Double-Column PDF Upload & Future Work Extraction
- **Goal**: Confirm user-uploaded PDFs merge seamlessly into the matrix.
- **Action**:
  1. Upload a recent IEEE two-column paper (e.g., a PDF from your own research or advisor).
  2. Confirm the UI shows: `Extracted Abstract, Methods, and 3 Future Work Statements`.
  3. Confirm the uploaded paper appears in the 2D matrix with an `[Uploaded]` badge.
  4. Verify the extracted future work statement seeds a matching candidate gap.
- **Pass Criteria**: Paper is merged into the corpus, classified, and clickable in the knowledge graph.

