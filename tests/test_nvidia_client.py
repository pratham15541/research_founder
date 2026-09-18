from src.llm.nvidia_client import NvidiaClient
from src.config import settings


def test_parse_embedded_json_object_with_surrounding_text():
    text = 'Here is the result:\n{"label": "Neural operators", "score": 0.91}\nDone.'

    parsed = NvidiaClient._parse_embedded_json(text)

    assert parsed == {"label": "Neural operators", "score": 0.91}


def test_parse_embedded_json_list_after_invalid_example():
    text = "Example: [not json]\nActual:\n[\"Fluid dynamics\", \"Materials\", \"Energy\"]"

    parsed = NvidiaClient._parse_embedded_json(text)

    assert parsed == ["Fluid dynamics", "Materials", "Energy"]


def test_completion_url_accepts_base_or_full_endpoint():
    assert NvidiaClient._completion_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1/chat/completions"
    assert NvidiaClient._completion_url("https://openrouter.ai/api/v1/chat/completions") == "https://openrouter.ai/api/v1/chat/completions"


def test_provider_chain_reads_numbered_env(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", "primary-key")
    monkeypatch.setattr(settings, "NVIDIA_MODEL", "primary-model")
    monkeypatch.setattr(settings, "NVIDIA_BASE_URL", "https://primary.example/v1")
    monkeypatch.setattr(settings, "LLM_PROVIDER_MAX", 2)
    monkeypatch.setattr(NvidiaClient, "_DOTENV_CACHE", {
        "NVIDIA_API_KEY1": "fallback-key",
        "NVIDIA_MODEL1": "fallback-model",
        "NVIDIA_BASE_URL1": "https://fallback.example/v1/chat/completions",
    })

    providers = NvidiaClient._provider_chain()

    assert [p["model"] for p in providers] == ["primary-model", "fallback-model"]
    assert providers[1]["api_key"] == "fallback-key"


def test_generate_json_with_markdown_codeblock(monkeypatch):
    sample_text = 'Thinking process here...\n```json\n{"status": "success", "count": 42,}\n```\nExtra notes.'
    monkeypatch.setattr(NvidiaClient, "generate", lambda *args, **kwargs: sample_text)

    parsed = NvidiaClient.generate_json("dummy prompt")
    assert parsed == {"status": "success", "count": 42}


def test_generate_handles_reasoning_content_fallback(monkeypatch):
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "reasoning_content": "Chain of thought conclusion: Target reached."
                    }
                }]
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: MockResponse())
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(settings, "NVIDIA_MODEL", "test-model")
    monkeypatch.setattr(settings, "NVIDIA_BASE_URL", "https://api.example.com/v1")

    result = NvidiaClient.generate("dummy prompt")
    assert result == "Chain of thought conclusion: Target reached."

