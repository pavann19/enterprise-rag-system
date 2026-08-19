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
