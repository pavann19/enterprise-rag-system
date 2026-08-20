import importlib
import json
import urllib.error
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


def test_ollama_post_raises_connection_error_with_url_in_message(monkeypatch):
    monkeypatch.setattr(http_module, "_RETRY_BACKOFF_BASE_SEC", 0)
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


def test_ollama_post_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(http_module, "_RETRY_BACKOFF_BASE_SEC", 0)  # don't actually sleep in tests
    payload = {"response": "recovered"}
    attempts = {"n": 0}

    def _flaky_urlopen(request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise urllib.error.URLError("connection refused")
        return _mock_response(json.dumps(payload).encode())

    with patch("urllib.request.urlopen", side_effect=_flaky_urlopen):
        result = http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x"})

    assert result == payload
    assert attempts["n"] == 2


def test_ollama_post_retries_mid_stream_connection_reset(monkeypatch):
    # Regression test: a connection reset while reading the response body
    # (http.client.RemoteDisconnected, a ConnectionResetError/OSError, NOT
    # a urllib.error.URLError) is a different failure mode than a refused
    # connection, and was not being retried before — found running the
    # real scale benchmark against local Ollama under concurrent load.
    monkeypatch.setattr(http_module, "_RETRY_BACKOFF_BASE_SEC", 0)
    payload = {"response": "recovered"}
    attempts = {"n": 0}

    class _FlakyResponse:
        def __enter__(self):
            attempts["n"] += 1
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            # Raised here, not in __enter__/urlopen(), to accurately mimic
            # the real failure: the connection opens fine, the reset
            # happens while streaming the body.
            if attempts["n"] < 2:
                raise ConnectionResetError("Remote end closed connection without response")
            return json.dumps(payload).encode()

    with patch("urllib.request.urlopen", side_effect=lambda request, timeout=None: _FlakyResponse()):
        result = http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x"})

    assert result == payload
    assert attempts["n"] == 2


def test_ollama_post_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(http_module, "_RETRY_BACKOFF_BASE_SEC", 0)
    monkeypatch.setattr(http_module, "_MAX_RETRIES", 2)
    attempts = {"n": 0}

    def _always_fails(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", side_effect=_always_fails):
        with pytest.raises(ConnectionError, match="after 3 attempt"):
            http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x"})

    assert attempts["n"] == 3  # 1 initial + 2 retries


def test_ollama_post_does_not_retry_invalid_json():
    attempts = {"n": 0}

    def _bad_json(request, timeout=None):
        attempts["n"] += 1
        return _mock_response(b"not valid json{{{")

    with patch("urllib.request.urlopen", side_effect=_bad_json):
        with pytest.raises(RuntimeError, match="Could not parse"):
            http_module.ollama_post("http://localhost:11434/api/generate", {"model": "x"})

    assert attempts["n"] == 1  # malformed JSON is a server-side bug, not transient — no point retrying


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
