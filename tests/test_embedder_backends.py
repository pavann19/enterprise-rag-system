from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import rag.embedder as embedder_module
from rag.embedder import embed_texts


def test_embed_texts_rejects_empty_list():
    with pytest.raises(ValueError):
        embed_texts([])


def test_embed_texts_rejects_unknown_backend():
    with pytest.raises(ValueError):
        embed_texts(["hello"], backend="nonexistent")


def test_embed_texts_dispatches_to_ollama_backend_by_default(monkeypatch):
    assert embedder_module.EMBED_BACKEND == "ollama"  # default, unless overridden in the environment
    calls = []

    def _fake_ollama(texts, model):
        calls.append((texts, model))
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setitem(
        embedder_module._BACKENDS, "ollama", (_fake_ollama, embedder_module.DEFAULT_OLLAMA_MODEL)
    )

    result = embed_texts(["a", "b"])
    assert result.shape == (2, 4)
    assert calls == [(["a", "b"], embedder_module.DEFAULT_OLLAMA_MODEL)]


def test_embed_texts_explicit_backend_param_overrides_default(monkeypatch):
    calls = []

    def _fake_local(texts, model):
        calls.append((texts, model))
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setitem(
        embedder_module._BACKENDS, "local", (_fake_local, embedder_module.DEFAULT_LOCAL_MODEL)
    )

    embed_texts(["a"], backend="local")
    assert calls == [(["a"], embedder_module.DEFAULT_LOCAL_MODEL)]


def test_embed_texts_local_raises_clear_error_when_sentence_transformers_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="sentence-transformers"):
        embed_texts(["a"], backend="local")


# ── Real backend bodies (ollama_post / SentenceTransformer mocked, not the
# whole backend function) — the tests above bypass _embed_texts_ollama and
# _embed_texts_local entirely by replacing their _BACKENDS entry, so they
# never actually exercise those functions' own logic. These do.


def test_embed_texts_ollama_body_builds_array_from_responses(monkeypatch):
    responses = iter(
        [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    )
    calls = []

    def _fake_ollama_post(url, payload):
        calls.append(payload)
        return next(responses)

    monkeypatch.setattr(embedder_module, "ollama_post", _fake_ollama_post)
    result = embed_texts(["first", "second"], backend="ollama", model="nomic-embed-text")

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result[1], [0.4, 0.5, 0.6])
    assert [c["prompt"] for c in calls] == ["first", "second"]
    assert all(c["model"] == "nomic-embed-text" for c in calls)


def test_embed_texts_ollama_body_raises_runtime_error_on_missing_embedding_key(monkeypatch):
    monkeypatch.setattr(embedder_module, "ollama_post", lambda url, payload: {"unexpected": "shape"})
    with pytest.raises(RuntimeError, match="missing 'embedding' key"):
        embed_texts(["text"], backend="ollama")


def test_embed_texts_local_body_calls_sentence_transformer_encode(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    fake_st_class = MagicMock(return_value=fake_model)

    embedder_module._local_model_cache.clear()
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=fake_st_class)}):
        result = embed_texts(["a", "b"], backend="local", model="test-model")

    fake_st_class.assert_called_once_with("test-model")
    fake_model.encode.assert_called_once_with(["a", "b"], convert_to_numpy=True)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])


def test_embed_texts_local_body_caches_model_across_calls(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0]])
    fake_st_class = MagicMock(return_value=fake_model)

    embedder_module._local_model_cache.clear()
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=fake_st_class)}):
        embed_texts(["a"], backend="local", model="cached-model")
        embed_texts(["b"], backend="local", model="cached-model")

    fake_st_class.assert_called_once()  # second call reused the cached instance
