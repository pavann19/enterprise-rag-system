import importlib

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
