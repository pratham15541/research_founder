from src.llm.nvidia_client import NvidiaClient


def test_parse_embedded_json_object_with_surrounding_text():
    text = 'Here is the result:\n{"label": "Neural operators", "score": 0.91}\nDone.'

    parsed = NvidiaClient._parse_embedded_json(text)

    assert parsed == {"label": "Neural operators", "score": 0.91}


def test_parse_embedded_json_list_after_invalid_example():
    text = "Example: [not json]\nActual:\n[\"Fluid dynamics\", \"Materials\", \"Energy\"]"

    parsed = NvidiaClient._parse_embedded_json(text)

    assert parsed == ["Fluid dynamics", "Materials", "Energy"]
