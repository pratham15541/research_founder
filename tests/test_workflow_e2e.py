import pytest
from unittest.mock import AsyncMock, patch
from src.config import settings
from src.workflow import ResearchWorkflowRunner
from src.storage.models import Paper

@pytest.fixture(autouse=True)
def disable_dynamic_llm():
    old_val = settings.DYNAMIC_LLM_ENABLED
    old_req = settings.LLM_REQUIRED
    settings.DYNAMIC_LLM_ENABLED = False
    settings.LLM_REQUIRED = False
    yield
    settings.DYNAMIC_LLM_ENABLED = old_val
    settings.LLM_REQUIRED = old_req

@pytest.mark.asyncio
async def test_workflow_runner_end_to_end():
    # Construct 10 synthetic papers
    papers = [
        Paper(
            original_title="PINN for High Reynolds Turbulent Flow",
            normalized_title="pinn for high reynolds turbulent flow",
            abstract="We propose physics-informed neural networks for fluid dynamics. Limitation: limited to small synthetic datasets and 2D domains. Future work should evaluate on large-scale 3D turbulent flow datasets. Method improves accuracy significantly.",
            publication_year=2021,
            citation_count=50,
            source="openalex",
            source_url="https://doi.org/10.1000/1",
        ),
        Paper(
            original_title="Fourier Neural Operators for Navier Stokes",
            normalized_title="fourier neural operators for navier stokes",
            abstract="Fourier neural operators map infinite-dimensional spaces. Limitation: high computational memory overhead and evaluation was performed on a small dataset. Future work should address 3D scalability. Method improves accuracy.",
            publication_year=2022,
            citation_count=40,
            source="openalex",
            source_url="https://doi.org/10.1000/2",
        ),
        Paper(
            original_title="DeepONet Benchmarking on Complex Geometries",
            normalized_title="deeponet benchmarking on complex geometries",
            abstract="Deep operator networks for PDEs. Limitation: restricted to small datasets without boundary noise. Future work should evaluate robustness on real-world noisy sensors. Method decreases accuracy under high noise.",
            publication_year=2023,
            citation_count=25,
            source="arxiv",
            source_url="https://doi.org/10.1000/3",
        ),
        Paper(
            original_title="Scalability Limits of Neural Operators in CFD",
            normalized_title="scalability limits of neural operators in cfd",
            abstract="Investigating training time of neural operators. Limitation: our evaluation was limited to small datasets. Future work should evaluate on large-scale datasets across distributed clusters. Findings show method has no significant effect on wall-clock time.",
            publication_year=2024,
            citation_count=15,
            source="arxiv",
            source_url="https://doi.org/10.1000/4",
        ),
        Paper(
            original_title="Generalization Bounds for Physics-Guided AI",
            normalized_title="generalization bounds for physics-guided ai",
            abstract="Theoretical generalization analysis. Limitation: lacks real-world experimental validation. Future work: test across multi-phase flows.",
            publication_year=2025,
            citation_count=5,
            source="s2",
            source_url="https://doi.org/10.1000/5",
        ),
        Paper(
            original_title="Hybrid FNO-PINN for Boundary Layer Turbulence",
            normalized_title="hybrid fno-pinn for boundary layer turbulence",
            abstract="Combining spectral methods with collocation losses. Limitation: requires precise collocation grid sampling. Future work: evaluate on unstructured meshes.",
            publication_year=2024,
            citation_count=12,
            source="openalex",
            source_url="https://doi.org/10.1000/6",
        ),
        Paper(
            original_title="Multiphase CFD Modeling using Neural Surrogates",
            normalized_title="multiphase cfd modeling using neural surrogates",
            abstract="Surrogate models for gas-liquid interface tracking. Limitation: restricted to laminar regimes. Future work: scale to turbulent mixing regimes.",
            publication_year=2023,
            citation_count=18,
            source="openalex",
            source_url="https://doi.org/10.1000/7",
        ),
        Paper(
            original_title="Uncertainty Quantification in Physics-Informed ML",
            normalized_title="uncertainty quantification in physics-informed ml",
            abstract="Bayesian neural networks for PDE parameter estimation. Limitation: sampling is slow. Future work: variational inference acceleration.",
            publication_year=2022,
            citation_count=35,
            source="arxiv",
            source_url="https://doi.org/10.1000/8",
        ),
        Paper(
            original_title="Sparse Sensor Placement for Flow Reconstruction",
            normalized_title="sparse sensor placement for flow reconstruction",
            abstract="Optimal sensor selection for flow state estimation. Limitation: assumes noise-free measurements. Future work: validate with noisy experimental PIV data.",
            publication_year=2021,
            citation_count=45,
            source="s2",
            source_url="https://doi.org/10.1000/9",
        ),
        Paper(
            original_title="Comparative Study of Operator Learning for PDEs",
            normalized_title="comparative study of operator learning for pdes",
            abstract="Comprehensive benchmark across 10 PDE problems. Limitation: limited to small synthetic datasets. Future work: large-scale open benchmark creation. Method improves accuracy on smooth solutions but decreases accuracy on shockwaves.",
            publication_year=2025,
            citation_count=8,
            source="arxiv",
            source_url="https://doi.org/10.1000/10",
        ),
    ]

    runner = ResearchWorkflowRunner()
    
    with patch.object(runner.ingestion, "ingest_topic_corpus", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = papers

        result = await runner.run_pipeline(
            session=None,
            topic_query="physics informed neural networks for turbulence",
            target_corpus_size=10,
            clustering_algorithm="kmeans"
        )

        assert result["corpus_size"] == 10
        assert len(result["structured_records"]) == 10
        assert result["candidate_gaps_count"] > 0
        assert len(result["ranked_gaps"]) > 0

        # Check evidence graph output
        assert "evidence_graph_summary" in result
        assert result["evidence_graph_summary"]["nodes"] > 0
        assert "evidence_graph_html" in result

        # Check top ranked gap attributes
        top_gap = result["ranked_gaps"][0]
        assert "gap_type" in top_gap
        assert "taxonomy_category" in top_gap
        assert "confidence_breakdown" in top_gap
        assert "grounded_experiment" in top_gap
        assert "researcher_verification_checklist" in top_gap
        assert "source_facts" in top_gap
        assert "ai_inferences" in top_gap
        assert "devil_advocate_challenges" in top_gap

        # Check evaluation metrics & literature review
        assert "evaluation_metrics" in result
        assert "literature_review" in result
        assert "research_questions" in result
