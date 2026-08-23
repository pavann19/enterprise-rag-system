"""Tests for rag/reranker.py. The cross-encoder model itself is mocked —
these validate re-sorting/score-attachment logic, not model quality."""

from unittest.mock import MagicMock

import pytest

import rag.reranker as reranker_module


@pytest.fixture(autouse=True)
def _clear_model_cache():
    reranker_module._model_cache.clear()
    yield
    reranker_module._model_cache.clear()


def test_rerank_returns_empty_list_unchanged():
    assert reranker_module.rerank("q", []) == []


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9, 0.5]
    monkeypatch.setattr(reranker_module, "_get_model", lambda model_name: fake_model)

    results = [
        {"text": "a", "score": 0.9, "source": "a.txt"},
        {"text": "b", "score": 0.5, "source": "b.txt"},
        {"text": "c", "score": 0.7, "source": "c.txt"},
    ]
    reranked = reranker_module.rerank("query", results)

    assert [r["source"] for r in reranked] == ["b.txt", "c.txt", "a.txt"]
    assert reranked[0]["rerank_score"] == 0.9
    assert reranked[0]["score"] == 0.5  # original cosine score preserved


def test_get_model_caches_by_name(monkeypatch):
    calls = []

    class _FakeCrossEncoder:
        def __init__(self, name):
            calls.append(name)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        MagicMock(CrossEncoder=_FakeCrossEncoder),
    )

    reranker_module._get_model("model-a")
    reranker_module._get_model("model-a")
    reranker_module._get_model("model-b")

    assert calls == ["model-a", "model-b"]


def test_rerank_enabled_flag_reads_env(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    import importlib

    importlib.reload(reranker_module)
    assert reranker_module.RERANK_ENABLED is True
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    importlib.reload(reranker_module)
    assert reranker_module.RERANK_ENABLED is False
