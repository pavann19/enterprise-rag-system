import numpy as np
import pytest

import rag.ingestion as ingestion_module
from rag.ingestion import ingest


@pytest.fixture
def fake_embed(monkeypatch):
    """Replaces the Ollama-backed embed_texts with a deterministic stub
    keyed on text length, so ingestion tests need no live Ollama instance."""
    calls = {"n": 0}

    def _stub(texts, model="nomic-embed-text"):
        calls["n"] += 1
        return np.array([[float(len(t)), 0.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(ingestion_module, "embed_texts", _stub)
    return calls


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "a.txt").write_text("alpha document about apples and oranges.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta document about bananas and pears.", encoding="utf-8")
    return tmp_path


def test_ingest_raises_on_missing_directory(tmp_path, fake_embed):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "does_not_exist")


def test_ingest_raises_when_no_txt_files(tmp_path, fake_embed):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path)


def test_ingest_returns_chunks_metadata_and_store(corpus_dir, fake_embed):
    chunks, metadata, store = ingest(corpus_dir)
    assert len(chunks) == len(metadata) == len(store)
    sources = {m["source"] for m in metadata}
    assert sources == {"a.txt", "b.txt"}


def test_ingest_without_cache_dir_always_embeds(corpus_dir, fake_embed):
    ingest(corpus_dir, cache_dir=None)
    ingest(corpus_dir, cache_dir=None)
    assert fake_embed["n"] == 2  # no caching -> embedded twice


def test_ingest_with_cache_dir_embeds_once(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    chunks1, metadata1, store1 = ingest(corpus_dir, cache_dir=cache_dir)
    assert fake_embed["n"] == 1

    chunks2, metadata2, store2 = ingest(corpus_dir, cache_dir=cache_dir)
    assert fake_embed["n"] == 1  # second call is a cache hit, no re-embedding

    assert chunks1 == chunks2
    assert metadata1 == metadata2
    assert len(store1) == len(store2)


def test_ingest_cache_invalidated_by_content_change(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    ingest(corpus_dir, cache_dir=cache_dir)
    assert fake_embed["n"] == 1

    (corpus_dir / "a.txt").write_text("a completely different document now.", encoding="utf-8")

    ingest(corpus_dir, cache_dir=cache_dir)
    assert fake_embed["n"] == 2  # content changed -> fingerprint changed -> re-embedded


def test_ingest_cache_invalidated_by_config_change(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    ingest(corpus_dir, cache_dir=cache_dir, chunk_size=300)
    assert fake_embed["n"] == 1

    ingest(corpus_dir, cache_dir=cache_dir, chunk_size=150)
    assert fake_embed["n"] == 2  # different chunk_size -> different fingerprint
