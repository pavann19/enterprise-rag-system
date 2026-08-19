import numpy as np
import pytest

import rag.ingestion as ingestion_module
from rag.ingestion import ingest


@pytest.fixture
def fake_embed(monkeypatch):
    """Replaces the Ollama-backed embed_texts with a deterministic stub
    keyed on text length, so ingestion tests need no live Ollama instance."""
    calls = {"n": 0}

    def _stub(texts, model=None, backend=None):
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


def test_ingest_cache_invalidated_by_embed_model_change(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    ingest(corpus_dir, cache_dir=cache_dir, embed_model="model-a")
    assert fake_embed["n"] == 1

    ingest(corpus_dir, cache_dir=cache_dir, embed_model="model-b")
    assert fake_embed["n"] == 2  # different embed_model -> different fingerprint, even same backend


def test_ingest_cache_invalidated_by_embed_backend_change(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    ingest(corpus_dir, cache_dir=cache_dir, embed_backend="ollama")
    assert fake_embed["n"] == 1

    ingest(corpus_dir, cache_dir=cache_dir, embed_backend="local")
    assert fake_embed["n"] == 2  # different embed_backend -> different fingerprint


def test_ingest_cache_invalidated_by_vector_backend_change(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    ingest(corpus_dir, cache_dir=cache_dir, backend="numpy")
    assert fake_embed["n"] == 1

    ingest(corpus_dir, cache_dir=cache_dir, backend="numpy")  # identical config -> still a cache hit
    assert fake_embed["n"] == 1


def test_ingest_ignores_subdirectories(corpus_dir, fake_embed):
    subdir = corpus_dir / "nested"
    subdir.mkdir()
    (subdir / "c.txt").write_text("a document one level down that should be ignored.", encoding="utf-8")

    chunks, metadata, store = ingest(corpus_dir)
    sources = {m["source"] for m in metadata}
    assert sources == {"a.txt", "b.txt"}  # non-recursive glob — "nested/c.txt" not included


def test_ingest_ignores_non_txt_files(corpus_dir, fake_embed):
    (corpus_dir / "notes.md").write_text("this is markdown, not plaintext.", encoding="utf-8")
    (corpus_dir / "data.json").write_text('{"key": "value"}', encoding="utf-8")

    chunks, metadata, store = ingest(corpus_dir)
    sources = {m["source"] for m in metadata}
    assert sources == {"a.txt", "b.txt"}


def test_ingest_file_order_is_deterministic_across_runs(corpus_dir, fake_embed):
    _, metadata1, _ = ingest(corpus_dir)
    _, metadata2, _ = ingest(corpus_dir)
    assert [m["source"] for m in metadata1] == [m["source"] for m in metadata2]


def test_ingest_chunks_and_metadata_stay_aligned_across_files(corpus_dir, fake_embed):
    # Each chunk's metadata source must match the file it actually came from,
    # not just have the right overall count.
    (corpus_dir / "c.txt").write_text("gamma document about grapes and guavas.", encoding="utf-8")
    chunks, metadata, store = ingest(corpus_dir)

    for chunk, meta in zip(chunks, metadata):
        # each source file's own vocabulary shouldn't leak into another file's chunk
        if meta["source"] == "a.txt":
            assert "apples" in chunk or "alpha" in chunk
        elif meta["source"] == "c.txt":
            assert "grapes" in chunk or "gamma" in chunk


def test_ingest_single_empty_file_among_others_contributes_no_chunks(corpus_dir, fake_embed):
    (corpus_dir / "empty.txt").write_text("   \n  ", encoding="utf-8")
    chunks, metadata, store = ingest(corpus_dir)
    sources = {m["source"] for m in metadata}
    assert "empty.txt" not in sources
    assert sources == {"a.txt", "b.txt"}


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("faiss") is None,
    reason="faiss-cpu not installed",
)
def test_ingest_cache_roundtrip_with_faiss_backend(corpus_dir, fake_embed, tmp_path):
    cache_dir = tmp_path / ".cache"

    chunks1, metadata1, store1 = ingest(corpus_dir, cache_dir=cache_dir, backend="faiss")
    assert fake_embed["n"] == 1

    chunks2, metadata2, store2 = ingest(corpus_dir, cache_dir=cache_dir, backend="faiss")
    assert fake_embed["n"] == 1  # cache hit, no re-embedding
    assert chunks1 == chunks2
    assert len(store1) == len(store2)
