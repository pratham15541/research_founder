"""
FastAPI Backend Application for Research Landscape & Gap Matrix Platform.
Provides REST endpoints for research topic analysis, PDF upload, and historical run inspection.
"""

import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.config import settings
from src.storage.db import get_db_session, init_db
from src.storage.models import AnalysisRun, GapResult, Paper
from src.storage.file_storage import get_file_storage_backend
from src.ingestion.pdf_parser import AcademicPDFParser
from src.ingestion.proxy_manager import proxy_manager
from src.workflow import ResearchWorkflowRunner
from src.rag.chat import ResearchChatSession
from src.reports.exporter import ReportExporter
from src.evaluation.human_eval import HumanEvaluationManager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database initialization."""
    logger.info("Initializing research platform database...")
    await init_db()
    app.state.latest_results = None
    app.state.workflow_runner = ResearchWorkflowRunner()
    app.state.chat_session = None
    app.state.human_eval_records = []
    # Pre-warm proxy pool in background
    try:
        await proxy_manager.get_proxies()
    except Exception as e:
        logger.warning(f"Could not pre-warm proxy pool: {e}")
    yield
    logger.info("Shutting down research platform API.")

app = FastAPI(
    title="Research Landscape & Gap Matrix API",
    description="Backend API for evidence-grounded combinatorial research gap discovery.",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware configured for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Streamlit, React, or browser clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Request & Response Models
class AnalyzeRequest(BaseModel):
    topic_query: str = Field(..., json_schema_extra={"example": "Physics-Informed Neural Networks"})
    target_corpus_size: int = Field(default=60, ge=20, le=250)
    expanded_queries: Optional[List[str]] = None
    clustering_algorithm: Optional[str] = Field(default="auto", json_schema_extra={"example": "auto"})

class ChatRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "What are the primary limitations discovered in PINN boundary conditions?"})
    top_k: int = Field(default=4, ge=1, le=10)

class HumanEvalRequest(BaseModel):
    gap_id: str
    project_title: str
    relevance_score: int = Field(..., ge=1, le=5)
    novelty_score: int = Field(..., ge=1, le=5)
    explainability_score: int = Field(..., ge=1, le=5)
    hallucination_detected: bool = False
    expert_comments: Optional[str] = ""

class ExportRequest(BaseModel):
    format: str = Field(default="markdown", json_schema_extra={"example": "markdown"})

class HealthResponse(BaseModel):
    status: str
    service: str
    proxy_pool_size: int

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint confirming API and proxy pool operational status."""
    proxies = await proxy_manager.get_proxies()
    return {
        "status": "healthy",
        "service": "research-landscape-backend",
        "proxy_pool_size": len(proxies)
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Secure academic PDF upload endpoint.
    Validates %PDF- header, saves with unique UUID, and segments sections.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are permitted.")

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.")

    if not AcademicPDFParser.validate_pdf_bytes(content):
        raise HTTPException(status_code=400, detail="Invalid PDF header.")

    try:
        stored = get_file_storage_backend().save_bytes(
            content=content,
            filename=file.filename,
            content_type=file.content_type or "application/pdf",
            file_id=file_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store PDF: {e}")

    validation = AcademicPDFParser.inspect_pdf_file(stored.local_path)
    if not validation["is_valid"]:
        stored.local_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF: {validation['reason']}")

    # Parse sections
    try:
        parsed_doc = AcademicPDFParser.parse_pdf(stored.local_path)
        future_seeds = AcademicPDFParser.extract_explicit_future_work_statements(parsed_doc.get("sections", {}))
        return {
            "file_id": file_id,
            "filename": file.filename,
            "title": parsed_doc.get("title"),
            "sections_detected": list(parsed_doc.get("sections", {}).keys()),
            "future_work_statements": future_seeds,
            "local_path": str(stored.local_path),
            "storage_backend": stored.backend,
            "storage_uri": stored.uri,
            "pdf_validation": validation
        }
    except Exception as e:
        stored.local_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF layout: {e}")

@app.post("/api/analyze")
async def analyze_topic(
    req: AnalyzeRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Execute full 14-step research landscape, gap discovery, RAG indexing, and synthesis pipeline.
    """
    runner = getattr(app.state, "workflow_runner", None) or ResearchWorkflowRunner()
    app.state.workflow_runner = runner
    try:
        results = await runner.run_pipeline(
            session=session,
            topic_query=req.topic_query,
            expanded_queries=req.expanded_queries,
            target_corpus_size=req.target_corpus_size,
            clustering_algorithm=req.clustering_algorithm or "auto"
        )

        # Update app state for downstream chat, review, questions, evaluation, and export
        app.state.latest_results = results
        app.state.chat_session = ResearchChatSession(
            faiss_index=runner.faiss_index,
            embedder=runner.embedder
        )

        # Persist AnalysisRun to historical table
        run_record = AnalysisRun(
            query_text=req.topic_query,
            expanded_queries=req.expanded_queries or [req.topic_query],
            corpus_size=results["corpus_size"],
            silhouette_score=results["silhouette_score"],
            axis_a_name="Discovered Methodology (Unsupervised)",
            axis_b_name="Application Domain",
            axis_a_labels=list(results["matrix"]["axis_a_labels"]),
            axis_b_labels=list(results["matrix"]["axis_b_labels"])
        )
        session.add(run_record)
        await session.flush()

        # Persist ranked gap results
        for g in results["ranked_gaps"]:
            gap_rec = GapResult(
                run_id=run_record.id,
                axis_a_val=g["axis_a"],
                axis_b_val=g["axis_b"],
                cell_paper_count=0,
                novelty_score=g["novelty_score"],
                feasibility_score=g["feasibility_score"],
                impact_score=g["impact_score"],
                composite_score=g["composite_score"],
                counter_argument=g["counter_argument"],
                grounding_evidence=g["supporting_evidence"]
            )
            session.add(gap_rec)

        await session.commit()
        results["run_id"] = run_record.id
        results["run_id"] = str(run_record.id)
        return results
    except Exception as e:
        logger.error(f"Analysis pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/runs")
async def list_runs(session: AsyncSession = Depends(get_db_session)):
    """List historical analysis runs."""
    stmt = select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(20)
    res = await session.execute(stmt)
    runs = res.scalars().all()
    return [
        {
            "run_id": r.id,
            "run_id": str(r.id),
            "query": r.query_text,
            "corpus_size": r.corpus_size,
            "silhouette_score": float(r.silhouette_score) if r.silhouette_score else None,
            "created_at": r.created_at.isoformat()
        }
        for r in runs
    ]

@app.get("/api/proxies")
async def get_proxy_status():
    """Get active proxy pool information."""
    proxies = await proxy_manager.get_proxies()
    sample = await proxy_manager.get_random_proxy()
    return {
        "total_active_proxies": len(proxies),
        "sample_proxy": sample,
        "source": settings.PROXY_SOURCE_URLS
    }

@app.post("/api/proxies/refresh")
async def refresh_proxy_pool():
    """Force-refresh the proxy cache from configured proxy source URLs."""
    proxies = await proxy_manager.get_proxies(force_refresh=True)
    sample = await proxy_manager.get_random_proxy()
    return {
        "total_active_proxies": len(proxies),
        "sample_proxy": sample,
        "source": settings.PROXY_SOURCE_URLS,
        "refreshed": True
    }

@app.post("/api/chat")
async def chat_with_corpus(req: ChatRequest):
    """
    RAG-powered conversational assistant querying indexed papers in FAISS.
    """
    latest_results = getattr(app.state, "latest_results", None)
    runner = getattr(app.state, "workflow_runner", None)
    if not latest_results or not runner or runner.faiss_index.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No indexed academic corpus available. Please execute /api/analyze first."
        )

    if getattr(app.state, "chat_session", None) is None:
        app.state.chat_session = ResearchChatSession(
            faiss_index=runner.faiss_index,
            embedder=runner.embedder
        )

    response = app.state.chat_session.send_message(req.message, top_k=req.top_k)
    return response

@app.get("/api/literature-review")
async def get_literature_review():
    """Get the synthesized automated literature review for the latest analyzed topic."""
    latest_results = getattr(app.state, "latest_results", None)
    if not latest_results:
        raise HTTPException(status_code=400, detail="No active analysis run. Execute /api/analyze first.")
    return latest_results.get("literature_review", {})

@app.get("/api/research-questions")
async def get_research_questions():
    """Get formal research questions and hypothesis protocols for top-ranked gaps."""
    latest_results = getattr(app.state, "latest_results", None)
    if not latest_results:
        raise HTTPException(status_code=400, detail="No active analysis run. Execute /api/analyze first.")
    return latest_results.get("research_questions", [])

@app.get("/api/evaluation")
async def get_evaluation_metrics():
    """Get quantitative intrinsic/extrinsic metrics and aggregate human evaluation scores."""
    latest_results = getattr(app.state, "latest_results", None)
    if not latest_results:
        raise HTTPException(status_code=400, detail="No active analysis run. Execute /api/analyze first.")

    quant_metrics = latest_results.get("evaluation_metrics", {})
    human_records = getattr(app.state, "human_eval_records", [])
    human_agg = HumanEvaluationManager.aggregate_human_scores(human_records)

    return {
        "quantitative_metrics": quant_metrics,
        "human_evaluation": human_agg,
        "total_human_reviews": len(human_records)
    }

@app.post("/api/evaluation/human")
async def submit_human_evaluation(req: HumanEvalRequest):
    """Submit an expert evaluation rating for a research gap proposal."""
    record = HumanEvaluationManager.create_evaluation_record(
        gap_id=req.gap_id,
        project_title=req.project_title,
        relevance_score=req.relevance_score,
        novelty_score=req.novelty_score,
        explainability_score=req.explainability_score,
        hallucination_detected=req.hallucination_detected,
        expert_comments=req.expert_comments or ""
    )
    if not hasattr(app.state, "human_eval_records") or app.state.human_eval_records is None:
        app.state.human_eval_records = []
    app.state.human_eval_records.append(record)

    return {
        "status": "success",
        "saved_record": record,
        "aggregate": HumanEvaluationManager.aggregate_human_scores(app.state.human_eval_records)
    }

@app.post("/api/export")
async def export_analysis_report(req: ExportRequest):
    """Export complete research intelligence report in Markdown, LaTeX, or HTML."""
    latest_results = getattr(app.state, "latest_results", None)
    if not latest_results:
        raise HTTPException(status_code=400, detail="No active analysis run to export. Execute /api/analyze first.")

    topic = latest_results.get("topic_query", "Research_Topic")
    lit_review = latest_results.get("literature_review")
    rqs = latest_results.get("research_questions")
    clean_filename = topic.replace(" ", "_").replace("/", "_")

    fmt = req.format.lower().strip()
    if fmt in ["markdown", "md"]:
        content = ReportExporter.export_markdown(topic, latest_results, lit_review, rqs)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="ResearchGapAI_{clean_filename}.md"'}
        )
    elif fmt in ["latex", "tex"]:
        content = ReportExporter.export_latex(topic, latest_results, lit_review, rqs)
        return Response(
            content=content,
            media_type="application/x-latex",
            headers={"Content-Disposition": f'attachment; filename="ResearchGapAI_{clean_filename}.tex"'}
        )
    elif fmt == "html":
        content = ReportExporter.export_html(topic, latest_results, lit_review)
        return Response(
            content=content,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="ResearchGapAI_{clean_filename}.html"'}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{req.format}'. Choose 'markdown', 'latex', or 'html'.")
