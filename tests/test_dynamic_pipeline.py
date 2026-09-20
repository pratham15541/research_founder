"""
Tests for dynamic scoring, dynamic taxonomy, silhouette-optimal domain discovery,
and data-driven project synthesis (no static if/else rules).
"""

import pytest
import numpy as np
from src.ranking.gap_validator import GapValidatorAgent
from src.matrix.candidate_generator import MultiSignalGapGenerator, _classify_gap_type_semantic
from src.representation.domain_discovery import DomainDiscoveryEngine
from src.graph.semantic_clustering import silhouette_optimal_kmeans
from src.ranking.condition_tree import ConditionTreeBuilder
from src.ranking.project_synthesizer import ResearchProjectSynthesizer


class MockEmbedder:
    """Deterministic mock embedder for tests."""
    def embed_texts(self, texts):
        np.random.seed(42)
        embs = []
        for t in texts:
            # Deterministic pseudo-embedding based on character ords
            vec = np.zeros(16, dtype=float)
            for idx, ch in enumerate(t[:16]):
                vec[idx] = ord(ch) % 10
            norm = np.linalg.norm(vec)
            embs.append(vec / max(norm, 1e-6))
        return np.array(embs)


def test_continuous_validity_scoring_replaces_static_tiers():
    """Verify that gap validity is determined by continuous validity_score without integer ladder thresholds."""
    candidate = {
        "candidate_id": "c1",
        "gap_title": "Physics-Informed Neural Operators for Hypersonic Flow",
        "axis_a": "Physics-Informed Neural Operators",
        "axis_b": "Hypersonic Flow",
        "supporting_papers": [
            {"title": "Paper 1", "quote": "Limitation in hypersonics"},
            {"title": "Paper 2", "quote": "Shock boundary interaction unaddressed"}
        ],
        "method_verification": {"method_evidence_score": 0.85, "method_status": "VERIFIED"},
        "domain_verification": {"domain_activity_score": 0.80, "domain_maturity": "ACTIVE"},
        "prior_art": {"saturation_score": 0.05, "saturation_verdict": "OPEN"},
        "compatibility": {"compatibility_score": 0.90, "compatibility_verdict": "COMPATIBLE"}
    }

    paper_records = [
        {"title": "Background on Neural Operators", "abstract": "General study on operators", "year": 2023},
        {"title": "Flow aerodynamics", "abstract": "Classical simulation study", "year": 2024}
    ]

    result = GapValidatorAgent.validate_candidate(candidate, paper_records)
    assert "validity_score" in result
    assert isinstance(result["validity_score"], float)
    assert 0.0 <= result["validity_score"] <= 1.0
    assert result["is_valid"] is True
    assert result["status"] in ("True/strong gap", "Potential gap", "Novel intersection")
    assert "confidence_scorecard" in result
    assert result["confidence_scorecard"]["validity_score"] >= 30


def test_dynamic_taxonomy_semantic_classification():
    """Verify that taxonomy classification works through semantic embedding similarity without string matching."""
    candidates = [
        {
            "candidate_id": "cand_1",
            "gap_title": "Absence of High-Temperature Thermodynamic Datasets",
            "axis_a": "High-Temperature Thermodynamics",
            "axis_b": "Materials Science",
            "why_it_is_insufficient": "Empirical datasets under extreme heat conditions are missing from public repositories."
        },
        {
            "candidate_id": "cand_2",
            "gap_title": "Linear Algebra Inversion Bottlenecks on Large Grids",
            "axis_a": "Matrix Inversion",
            "axis_b": "Extreme Grid Resolution",
            "why_it_is_insufficient": "Memory footprint scales quadratically with degrees of freedom."
        }
    ]

    _classify_gap_type_semantic(candidates)
    assert candidates[0]["gap_type"] in ("Dataset Gap", "Knowledge Gap", "Generalization Gap")
    assert candidates[1]["gap_type"] in ("Scalability Gap", "Performance Gap", "Methodological Gap")


def test_silhouette_optimal_domain_discovery():
    """Verify that domain discovery discovers optimal k without fixing num_domains=5."""
    embedder = MockEmbedder()
    papers = [
        {"title": "Aerodynamic drag reduction in supersonic airfoils", "abstract": "Study of supersonic drag."},
        {"title": "Hypersonic shock waves and boundary layer transition", "abstract": "Shock boundary in hypersonics."},
        {"title": "Combustion chamber flame stability in scramjets", "abstract": "Scramjet combustion dynamics."},
        {"title": "Turbulent jet mixing in gas turbine engines", "abstract": "Gas turbine jet mixing."},
        {"title": "Biofluid dynamics in cardiovascular arterial bifurcations", "abstract": "Arterial blood flow."},
        {"title": "Hemodynamic wall shear stress in aortic aneurysms", "abstract": "Aortic hemodynamic wall shear."}
    ]

    domains = DomainDiscoveryEngine.discover_domains(
        topic_query="Fluid Dynamics",
        papers=papers,
        embedder=embedder,
        num_domains=None  # Dynamic silhouette optimization
    )

    assert isinstance(domains, list)
    assert len(domains) >= 2
    # Ensure domain names are clean title case strings
    for d in domains:
        assert isinstance(d, str)
        assert len(d) > 2


def test_dynamic_condition_tree_generation():
    """Verify that ConditionTreeBuilder generates data-driven condition statements."""
    gap = {
        "gap_title": "PINNs in Hypersonics",
        "axis_a": "Physics-Informed Neural Networks",
        "axis_b": "Hypersonics",
        "paper_count": 0,
        "signal_type": "Signal 8: Underexplored Intersection",
        "supporting_papers": [
            {"title": "PINN study", "quote": "PINN verified in laminar regime"}
        ],
        "method_status": "VERIFIED",
        "domain_maturity": "ACTIVE",
        "compatibility_verdict": "COMPATIBLE"
    }

    feasibility = {
        "feasibility_score": 4.2,
        "rationale": "High compute readiness and active open-source tooling."
    }

    # Test LLM or dynamic fallback path
    conditions = ConditionTreeBuilder.build_decision_conditions(
        gap=gap,
        feasibility_data=feasibility,
        extracted_future_work_seeds=["Future work must address high Mach number shocks."]
    )
    assert len(conditions) >= 4
    # Check that conditions reference the method/domain concepts
    all_text = " ".join(conditions).lower()
    assert "pinn" in all_text or "physics" in all_text or "neural" in all_text
    assert "hypersonic" in all_text

    # Also test dynamic fallback explicitly
    fallback_conditions = ConditionTreeBuilder._build_dynamic_fallback(
        gap=gap,
        feasibility_data=feasibility,
        extracted_future_work_seeds=["Future work must address high Mach number shocks."]
    )
    assert len(fallback_conditions) >= 4
    assert any("Physics-Informed Neural Networks" in c for c in fallback_conditions)
    assert any("Hypersonics" in c for c in fallback_conditions)
    assert any("4.2" in c for c in fallback_conditions)


def test_project_synthesizer_uses_extracted_metrics():
    """Verify that ProjectSynthesizer incorporates real extracted metrics instead of hardcoded 15%."""
    papers_a = [
        {"title": "FNO Benchmark", "evaluation_metrics": ["Relative L2 Error", "Peak Memory Latency"], "dataset": "Navier-Stokes PDE Bench"}
    ]
    papers_b = [
        {"title": "Hypersonic Analysis", "evaluation_metrics": ["Shock Wave Error", "Stagnation Pressure Acc"], "dataset": "Hypersonic Wind Tunnel 2024"}
    ]

    # Test dynamic fallback explicitly
    project_fallback = ResearchProjectSynthesizer._synthesize_with_heuristics(
        axis_a="Fourier Neural Operator",
        axis_b="Hypersonic Aerodynamics",
        papers_a=papers_a,
        papers_b=papers_b,
        cell_count=0,
        meta={"gap_type": "Methodological Gap"}
    )
    h1_fb = project_fallback["directional_hypothesis_h1"]
    exp_fb = project_fallback["grounded_experiment"]
    assert "by at least 15%" not in h1_fb
    assert exp_fb["target_dataset"] == "Navier-Stokes PDE Bench"
    assert "Relative L2 Error" in exp_fb["evaluation_metrics"]

    # Test top-level synthesize_project (calls live LLM if available)
    project = ResearchProjectSynthesizer.synthesize_project(
        axis_a="Fourier Neural Operator",
        axis_b="Hypersonic Aerodynamics",
        neighbor_papers_a=papers_a,
        neighbor_papers_b=papers_b,
        cell_count=0,
        candidate_meta={"gap_type": "Methodological Gap"}
    )
    assert "by at least 15%" not in project["directional_hypothesis_h1"]
    assert len(project["grounded_experiment"]["evaluation_metrics"]) >= 1
    assert "project_title" in project


