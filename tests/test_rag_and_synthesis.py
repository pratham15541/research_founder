"""
Unit tests for RAG engine, FAISS index, Literature Review, Research Questions, Evaluation Metrics, Exporter, and API.
"""

import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport

from src.rag.chunker import AcademicChunker
from src.rag.faiss_index import FAISSVectorIndex
from src.evaluation.metrics import EvaluationMetricsEngine
from src.evaluation.human_eval import HumanEvaluationManager
from src.reports.exporter import ReportExporter
from src.synthesis.literature_review import LiteratureReviewGenerator
from src.synthesis.question_generator import ResearchQuestionGenerator
from src.api.main import app

def test_academic_chunker():
    papers = [
        {
            "id": "p1",
            "title": "Deep Operator Networks for PDEs",
            "abstract": "We present DeepONet for learning nonlinear operators. " * 10,
            "full_text": "Section 1: Introduction. DeepONet consists of branch and trunk networks. " * 30,
            "sections": {
                "methods": "We formulate the trunk net to evaluate coordinate locations. " * 20,
                "future_work": "Future work includes applying DeepONet to multiscale fluid dynamics."
            }
        }
    ]
    chunks = AcademicChunker.chunk_corpus(papers, max_chunk_words=100, overlap_words=20)
    assert len(chunks) > 0
    assert any(c["section_type"] == "methods" for c in chunks)
    assert any(c["section_type"] == "future_work" for c in chunks)
    assert chunks[0]["paper_id"] == "p1"
    assert chunks[0]["paper_title"] == "Deep Operator Networks for PDEs"

def test_faiss_vector_index():
    index = FAISSVectorIndex(dimension=4)
    chunks = [
        {"chunk_id": "c1", "text": "chunk 1", "paper_title": "Paper 1"},
        {"chunk_id": "c2", "text": "chunk 2", "paper_title": "Paper 2"},
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    index.add_chunks(chunks, embeddings)
    assert index.index.ntotal == 2

    query_emb = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = index.search(query_emb, top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"

    index.clear()
    assert index.index.ntotal == 0

def test_evaluation_metrics():
    # Topic diversity
    terms = [
        ["neural", "network", "physics"],
        ["pde", "solver", "finite"],
        ["loss", "gradient", "optimization"]
    ]
    diversity = EvaluationMetricsEngine.calculate_topic_diversity(terms)
    assert 0.0 < diversity <= 1.0

    # Retrieval metrics
    query = "physics informed neural networks"
    corpus = [
        {"title": "Physics-Informed Neural Networks for Fluid Dynamics", "abstract": "Solving Navier-Stokes equations with PINNs"},
        {"title": "Deep Learning with Neural Networks in Physics", "abstract": "Overview of physical simulations"},
        {"title": "Unrelated Topic on Social Media Sentiments", "abstract": "Twitter sentiment analysis"}
    ]
    metrics = EvaluationMetricsEngine.calculate_retrieval_metrics(query, corpus)
    assert "precision_at_k" in metrics
    assert "recall_at_k" in metrics
    assert "f1_at_k" in metrics
    assert metrics["precision_at_k"] > 0.0

def test_human_evaluation_manager():
    rec = HumanEvaluationManager.create_evaluation_record(
        gap_id="gap_1",
        project_title="PINNs in Solid Mechanics",
        relevance_score=5,
        novelty_score=4,
        explainability_score=4,
        hallucination_detected=False,
        expert_comments="Strong proposal with viable baselines."
    )
    assert rec["gap_id"] == "gap_1"
    assert rec["relevance_score"] == 5

    agg = HumanEvaluationManager.aggregate_human_scores([rec])
    assert agg["total_ratings"] == 1
    assert agg["mean_relevance"] == 5.0
    assert agg["mean_novelty"] == 4.0
    assert agg["hallucination_rate_percent"] == 0.0

def test_report_exporter():
    results = {
        "corpus_size": 25,
        "silhouette_score": 0.35,
        "candidate_gaps_count": 4,
        "ranked_gaps": [
            {
                "project_title": "Adaptive Loss PINNs for Shock Waves",
                "axis_a": "Adaptive Loss Weighting",
                "axis_b": "High-Speed Aerodynamics",
                "composite_score": 4.5,
                "novelty_score": 4.8,
                "feasibility_score": 4.2,
                "impact_score": 4.7,
                "core_research_question": "How can dynamic weighting resolve shock discontinuities?",
                "why_it_is_a_gap": "Little work addresses gradient path balancing at Mach > 3.",
                "suggested_first_experiment": "Benchmark on 1D Sod shock tube problem.",
                "counter_argument": "Extreme gradient stiffness may still cause failure."
            }
        ]
    }
    topic = "Physics-Informed Neural Networks"

    md = ReportExporter.export_markdown(topic, results)
    assert "# ResearchGapAI Report: Physics-Informed Neural Networks" in md
    assert "Adaptive Loss PINNs for Shock Waves" in md

    tex = ReportExporter.export_latex(topic, results)
    assert "\\documentclass[conference]{IEEEtran}" in tex
    assert "Adaptive Loss PINNs for Shock Waves" in tex

    html = ReportExporter.export_html(topic, results)
    assert "<!DOCTYPE html>" in html
    assert "Adaptive Loss PINNs for Shock Waves" in html

def test_synthesis_and_questions():
    gap = {
        "project_title": "Operator Learning for Multiphase Flow",
        "axis_a": "Neural Operator Architecture",
        "axis_b": "Multiphase Porous Media Flow",
        "core_research_question": "Can Fourier neural operators generalize to varying porosity distributions?",
        "why_it_is_a_gap": "Current FNO models assume fixed permeability tensors.",
        "suggested_first_experiment": "Train FNO on 2D Buckley-Leverett simulations.",
        "practical_impact": "Accelerates CO2 sequestration modeling by 1000x."
    }
    rq = ResearchQuestionGenerator.generate_questions_for_gap(gap)
    assert "primary_research_question" in rq
    assert "primary_hypothesis_h1" in rq
    assert "null_hypothesis_h0" in rq
    assert "variables" in rq
    assert "experimental_phases" in rq
    assert isinstance(rq["experimental_phases"], list)
    assert len(rq["experimental_phases"]) == 3

@pytest.mark.asyncio
async def test_extended_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Pre-seed app.state with mock analysis results
        app.state.latest_results = {
            "topic_query": "Physics-Informed Neural Networks",
            "corpus_size": 30,
            "silhouette_score": 0.32,
            "ranked_gaps": [
                {
                    "project_title": "PINNs in Fluid Flow",
                    "axis_a": "Physics-Informed Neural Networks",
                    "axis_b": "Navier-Stokes Fluids",
                    "composite_score": 4.2,
                    "novelty_score": 4.0,
                    "feasibility_score": 4.5,
                    "impact_score": 4.1,
                    "core_research_question": "How to optimize boundary condition weighting?",
                    "why_it_is_a_gap": "Sparse exploration on complex 3D turbulence.",
                    "counter_argument": "High Reynolds numbers induce turbulence spectra that challenge standard MLP capacity.",
                    "suggested_first_experiment": "Run 2D lid-driven cavity test."
                }
            ],
            "evaluation_metrics": {
                "silhouette_score": 0.32,
                "topic_coherence": 0.45,
                "topic_diversity": 0.85,
                "retrieval": {"precision_at_k": 0.8, "recall_at_k": 0.7, "f1_at_k": 0.75}
            },
            "literature_review": {
                "executive_summary": "PINN literature has expanded rapidly since 2019...",
                "thematic_breakdown": [],
                "comparative_synthesis": "Comparison shows strong operator benefits.",
                "identified_voids": "Multiscale stiff systems remain open.",
                "review_markdown": "# PINN Literature Survey\nSummary of findings..."
            },
            "research_questions": [
                {
                    "primary_research_question": "How to optimize boundary condition weighting?",
                    "primary_hypothesis_h1": "Dynamic loss weighting decreases L2 error by >30%.",
                    "null_hypothesis_h0": "Dynamic loss weighting yields no statistically significant decrease.",
                    "variables": {"independent": "Loss weighting scheme", "dependent": "L2 relative error", "controlled": "Collocation density"},
                    "experimental_phases": {"phase_1_baseline_setup": "Standard PINN", "phase_2_hybridization": "Adaptive weights", "phase_3_stress_testing": "Turbulence regime"},
                    "expected_contributions": "Establishes adaptive penalty bounds."
                }
            ]
        }

        # 1. Literature review endpoint
        res_lit = await ac.get("/api/literature-review")
        assert res_lit.status_code == 200
        assert "executive_summary" in res_lit.json()

        # 2. Research questions endpoint
        res_rq = await ac.get("/api/research-questions")
        assert res_rq.status_code == 200
        assert len(res_rq.json()) == 1

        # 3. Evaluation endpoint
        res_eval = await ac.get("/api/evaluation")
        assert res_eval.status_code == 200
        assert "quantitative_metrics" in res_eval.json()

        # 4. Human eval submit endpoint
        eval_payload = {
            "gap_id": "gap_0",
            "project_title": "PINNs in Fluid Flow",
            "relevance_score": 5,
            "novelty_score": 4,
            "explainability_score": 5,
            "hallucination_detected": False,
            "expert_comments": "Looks very promising."
        }
        res_h = await ac.post("/api/evaluation/human", json=eval_payload)
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "success"

        # 5. Export endpoint (Markdown, LaTeX, HTML)
        res_exp_md = await ac.post("/api/export", json={"format": "markdown"})
        assert res_exp_md.status_code == 200
        assert "ResearchGapAI Report" in res_exp_md.text

        res_exp_tex = await ac.post("/api/export", json={"format": "latex"})
        assert res_exp_tex.status_code == 200
        assert "\\documentclass[conference]{IEEEtran}" in res_exp_tex.text

        res_exp_html = await ac.post("/api/export", json={"format": "html"})
        assert res_exp_html.status_code == 200
        assert "<!DOCTYPE html>" in res_exp_html.text
