"""
Integration tests for service/api.py using FastAPI's TestClient.

The corpus ingestion and query pipeline are replaced with fast, deterministic
stubs via monkeypatch — these tests exercise routing, request validation,
lifespan startup/shutdown, and HTTP status-code mapping, not the real
Ollama-backed pipeline (that's covered by the pure-function unit tests
elsewhere and by eval/run_eval.py against a live instance).
"""

import pytest
from fastapi.testclient import TestClient

import service.api as api_module
from validator.json_validator import ValidationError

FAKE_CHUNKS = ["chunk one text", "chunk two text"]
FAKE_METADATA = [{"source": "a.txt"}, {"source": "b.txt"}]


class _FakeVectorStore:
    def __len__(self):
        return len(FAKE_CHUNKS)


def _fake_ingest(**kwargs):
    return FAKE_CHUNKS, FAKE_METADATA, _FakeVectorStore()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "ingest", _fake_ingest)
    with TestClient(api_module.app) as c:
        yield c


# ── /health ──────────────────────────────────────────────────────────────────


def test_health_check_reports_ok_and_corpus_state(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents_loaded"] == 2
    assert "embedding_backend" in body
    assert "embedding_model" in body
    assert "generation_backend" in body
    assert "generation_model" in body


def test_health_check_does_not_invoke_query_pipeline(client, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("query_pipeline must not be called by /health")

    monkeypatch.setattr(api_module, "query_pipeline", _boom)
    resp = client.get("/health")
    assert resp.status_code == 200


# ── /query — request validation ─────────────────────────────────────────────


def test_query_rejects_empty_string(client):
    resp = client.post("/query", json={"query": "   "})
    assert resp.status_code == 422


def test_query_rejects_missing_field(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_rejects_wrong_type(client):
    resp = client.post("/query", json={"query": 12345})
    assert resp.status_code == 422


def test_query_rejects_non_json_body(client):
    resp = client.post("/query", content=b"not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


# ── /query — success and error mapping ──────────────────────────────────────


def test_query_success_returns_pipeline_response(client, monkeypatch):
    fake_response = {
        "query": "what is X?",
        "answer": "X is Y.",
        "sources": [{"text": "chunk one text", "source": "a.txt"}],
        "model": "mistral",
    }
    monkeypatch.setattr(api_module, "query_pipeline", lambda **kwargs: fake_response)

    resp = client.post("/query", json={"query": "what is X?"})
    assert resp.status_code == 200
    assert resp.json() == fake_response


def test_query_passes_request_fields_through_to_pipeline(client, monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return {
            "query": kwargs["query"],
            "answer": "a",
            "sources": [{"text": "t", "source": "s"}],
            "model": "m",
        }

    monkeypatch.setattr(api_module, "query_pipeline", _capture)
    client.post("/query", json={"query": "does it thread through?"})

    assert captured["query"] == "does it thread through?"
    assert captured["chunks"] == FAKE_CHUNKS
    assert captured["metadata"] == FAKE_METADATA


def test_query_returns_503_on_connection_error(client, monkeypatch):
    def _raise(**kwargs):
        raise ConnectionError("backend unreachable at http://ollama:11434")

    monkeypatch.setattr(api_module, "query_pipeline", _raise)
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 503
    assert "backend unreachable" in resp.json()["detail"]


def test_query_returns_500_on_validation_error(client, monkeypatch):
    def _raise(**kwargs):
        raise ValidationError("RAGResponse missing required keys: {'model'}")

    monkeypatch.setattr(api_module, "query_pipeline", _raise)
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 500
    assert "missing required keys" in resp.json()["detail"]


# ── Startup (lifespan) error handling ───────────────────────────────────────


def test_startup_wraps_missing_data_dir_as_runtime_error(monkeypatch):
    def _raise(**kwargs):
        raise FileNotFoundError("no .txt files found")

    monkeypatch.setattr(api_module, "ingest", _raise)
    with pytest.raises(RuntimeError, match="Data directory error"):
        with TestClient(api_module.app):
            pass


def test_startup_wraps_connection_error_as_runtime_error(monkeypatch):
    def _raise(**kwargs):
        raise ConnectionError("Ollama is not reachable")

    monkeypatch.setattr(api_module, "ingest", _raise)
    with pytest.raises(RuntimeError, match="embedding backend unreachable"):
        with TestClient(api_module.app):
            pass
