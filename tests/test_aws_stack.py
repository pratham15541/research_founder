"""
Unit tests for AWS Open-Source Tech Stack components.
Tests Bedrock client formatting, LLM Router, S3 Storage fallback,
and Vector Store factory — without requiring Docker or real AWS credentials.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.config import settings
from src.llm.llm_router import LLMRouter, _detect_provider
from src.llm.bedrock_client import BedrockClient
from src.storage.s3_storage import S3StorageBackend
from src.rag.vector_store import get_vector_store, get_vector_index
from src.rag.faiss_index import FAISSVectorIndex


def test_llm_router_detects_nvidia_when_aws_disabled():
    """When USE_AWS is false and NVIDIA_API_KEY is present, detect nvidia provider."""
    with patch.object(settings, "USE_AWS", False):
        with patch.object(settings, "NVIDIA_API_KEY", "test-key"):
            assert _detect_provider() == "nvidia"
            assert LLMRouter.is_available() is True


def test_llm_router_detects_bedrock_when_aws_enabled():
    """When USE_AWS is true, detect bedrock provider."""
    with patch.object(settings, "USE_AWS", True):
        assert _detect_provider() == "bedrock"
        assert LLMRouter.is_available() is True


def test_llm_router_detects_none_when_no_keys():
    """When neither AWS nor NVIDIA keys are set, detect none."""
    with patch.object(settings, "USE_AWS", False):
        with patch.object(settings, "NVIDIA_API_KEY", None):
            assert _detect_provider() == "none"
            assert LLMRouter.is_available() is False


def test_bedrock_claude_payload_formatting():
    """Verify BedrockClient formats Claude 3 messages correctly."""
    prompt = "Synthesize research gaps."
    system_prompt = "You are a scientific reviewer."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    body = BedrockClient._build_request_body(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        temperature=0.3,
        max_tokens=2048
    )
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == system_prompt
    assert body["messages"][0]["content"] == prompt
    assert body["max_tokens"] == 2048

    # Test extraction
    extracted = BedrockClient._extract_response_text(
        "anthropic.claude-3-sonnet-20240229-v1:0",
        {"content": [{"type": "text", "text": "Claude analysis result"}]}
    )
    assert extracted == "Claude analysis result"


def test_bedrock_titan_payload_formatting():
    """Verify BedrockClient formats Titan Text payload correctly."""
    prompt = "Test prompt for Titan."
    messages = [{"role": "user", "content": prompt}]
    body = BedrockClient._build_request_body(
        model_id="amazon.titan-text-express-v1",
        messages=messages,
        temperature=0.5,
        max_tokens=1000
    )
    assert "inputText" in body
    assert prompt in body["inputText"]
    assert body["textGenerationConfig"]["temperature"] == 0.5
    assert body["textGenerationConfig"]["maxTokenCount"] == 1000

    # Test extraction
    extracted = BedrockClient._extract_response_text(
        "amazon.titan-text-express-v1",
        {"results": [{"outputText": "Titan response text"}]}
    )
    assert extracted == "Titan response text"


def test_bedrock_llama_payload_formatting():
    """Verify BedrockClient formats Llama 3 prompt correctly."""
    prompt = "Test prompt for Llama."
    system_prompt = "System instructions."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    body = BedrockClient._build_request_body(
        model_id="meta.llama3-70b-instruct-v1:0",
        messages=messages,
        temperature=0.4,
        max_tokens=512
    )
    assert "prompt" in body
    assert "<|system|>" in body["prompt"]
    assert "System instructions." in body["prompt"]
    assert "<|user|>" in body["prompt"]
    assert "Test prompt for Llama." in body["prompt"]
    assert body["max_gen_len"] == 512

    # Test extraction
    extracted = BedrockClient._extract_response_text(
        "meta.llama3-70b-instruct-v1:0",
        {"generation": "Llama generated text"}
    )
    assert extracted == "Llama generated text"


def test_s3_storage_local_filesystem_fallback():
    """Verify S3StorageBackend falls back to local disk when AWS is disabled."""
    with patch.object(settings, "USE_AWS", False):
        backend = S3StorageBackend()
        assert backend.is_s3_active is False

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Research paper test content")
            tmp_path = Path(f.name)

        try:
            # Upload returns local path when S3 is inactive
            result_path = backend.upload_file(tmp_path)
            assert str(tmp_path) == result_path

            # Download returns same local path
            downloaded = backend.download_file("any-key", tmp_path)
            assert downloaded == tmp_path
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def test_vector_store_factory_returns_faiss():
    """Verify get_vector_store returns FAISSVectorIndex for local non-Docker mode."""
    with patch.object(settings, "USE_AWS", False):
        idx = get_vector_store(dimension=384)
        assert isinstance(idx, FAISSVectorIndex)
        assert idx.dimension == 384

        # Verify indexing and searching works
        dummy_chunks = [
            {"chunk_id": "c1", "text": "Quantum machine learning for drug design"},
            {"chunk_id": "c2", "text": "Deep learning architectures for vision"},
        ]
        embeddings = np.random.randn(2, 384).astype(np.float32)
        idx.add_chunks(dummy_chunks, embeddings)
        assert idx.index.ntotal == 2
        assert len(idx.metadata_store) == 2

        # Query
        query_vec = np.random.randn(384).astype(np.float32)
        results = idx.search(query_vec, top_k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] in ["c1", "c2"]


def test_embedding_engine_embed_text():
    """Verify EmbeddingEngine provides embed_text returning 1D vector."""
    from src.representation.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    vec = engine.embed_text("Sample research text")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Should be normalized
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_rag_engine_query_with_embedding_engine():
    """Verify RAGEngine.query runs end-to-end with EmbeddingEngine without crashing."""
    from src.representation.embeddings import EmbeddingEngine
    from src.rag.rag_engine import RAGEngine
    from src.llm.llm_router import LLMRouter

    engine = EmbeddingEngine()
    idx = FAISSVectorIndex(dimension=384)
    chunks = [
        {"chunk_id": "c1", "text": "PINNs enforce boundary conditions by penalizing residual loss.", "paper_title": "PINN Survey", "year": 2023, "paper_id": "p1"},
    ]
    embs = engine.embed_texts([chunks[0]["text"]])
    idx.add_chunks(chunks, embs)

    with patch.object(LLMRouter, "generate", return_value="PINNs enforce boundary conditions via loss weights."):
        result = RAGEngine.query(
            user_query="How do PINNs handle boundary conditions?",
            faiss_index=idx,
            embedder=engine,
            top_k=1
        )
        assert "answer" in result
        assert result["retrieved_chunks_count"] == 1
        assert len(result["citations"]) == 1


def test_aws_session_token_forwarding():
    """Verify Bedrock, S3, and BedrockEmbeddings pass aws_session_token to boto3."""
    mock_boto_mod = MagicMock()
    mock_client = MagicMock()
    mock_boto_mod.client = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto_mod}):
        with patch.object(settings, "AWS_ACCESS_KEY_ID", "ASIATESTKEY"):
            with patch.object(settings, "AWS_SECRET_ACCESS_KEY", "testsecret"):
                with patch.object(settings, "AWS_SESSION_TOKEN", "testsessiontoken"):
                    # Bedrock client
                    BedrockClient._get_bedrock_client()
                    call_kwargs = mock_client.call_args[1]
                    assert call_kwargs.get("aws_session_token") == "testsessiontoken"

                    # S3 backend
                    mock_client.reset_mock()
                    with patch.object(settings, "USE_AWS", True):
                        with patch.object(settings, "S3_BUCKET_NAME", "test-bucket"):
                            S3StorageBackend()
                            call_kwargs = mock_client.call_args[1]
                            assert call_kwargs.get("aws_session_token") == "testsessiontoken"

                    # Bedrock embeddings
                    mock_client.reset_mock()
                    from src.representation.bedrock_embeddings import BedrockEmbeddingEngine
                    emb_engine = BedrockEmbeddingEngine()
                    emb_engine._get_client()
                    call_kwargs = mock_client.call_args[1]
                    assert call_kwargs.get("aws_session_token") == "testsessiontoken"


