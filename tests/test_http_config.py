import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

import rag._http as http_module


def test_default_ollama_host_is_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    reloaded = importlib.reload(http_module)
    assert reloaded.OLLAMA_HOST == "http://localhost:11434"
    importlib.reload(http_module)  # restore real env for subsequent tests


def test_ollama_host_overridable_via_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    reloaded = importlib.reload(http_module)
    assert reloaded.OLLAMA_HOST == "http://ollama:11434"
    importlib.reload(http_module)  # restore real env for subsequent tests


@pytest.mark.parametrize("raw_host", ["http://ollama:11434/", "http://ollama:11434///"])
def test_trailing_slashes_stripped_from_ollama_host(monkeypatch, raw_host):
    monkeypatch.setenv("OLLAMA_HOST", raw_host)
    reloaded = importlib.reload(http_module)
    assert reloaded.OLLAMA_HOST == "http://ollama:11434"
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    importlib.reload(http_module)  # restore real env for subsequent tests


def test_ollama_post_raises_connection_error_with_url_in_message():
    with pytest.raises(ConnectionError, match="not reachable"):
        http_module.ollama_post("http://localhost:1/api/generate", {"model": "x"}, timeout=2)


def _mock_response(body: bytes):
    """A context-manager mock matching what urllib.request.urlopen() returns."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = False
    return mock_cm


def test_ollama_post_returns_parsed_json_on_success():
    payload = {"response": "the answer", "done": True}
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(payload).encode())):
        result = http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x"})
    assert result == payload


def test_ollama_post_sends_json_encoded_body_and_content_type_header():
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["content_type"] = request.get_header("Content-type")
        captured["method"] = request.get_method()
        return _mock_response(b'{"ok": true}')

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x", "prompt": "hi"})

    assert captured["body"] == {"model": "x", "prompt": "hi"}
    assert captured["content_type"] == "application/json"
    assert captured["method"] == "POST"


def test_ollama_post_raises_runtime_error_on_invalid_json_response():
    with patch("urllib.request.urlopen", return_value=_mock_response(b"not valid json{{{")):
        with pytest.raises(RuntimeError, match="Could not parse"):
            http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x"})


def test_embedder_and_generator_urls_derive_from_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    importlib.reload(http_module)

    import rag.embedder as embedder_module
    import rag.generator as generator_module

    importlib.reload(embedder_module)
    importlib.reload(generator_module)

    assert embedder_module.EMBED_URL == "http://ollama:11434/api/embeddings"
    assert generator_module.GENERATE_URL == "http://ollama:11434/api/generate"

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    importlib.reload(http_module)
    importlib.reload(embedder_module)
    importlib.reload(generator_module)
