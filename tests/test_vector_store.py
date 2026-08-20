import numpy as np
import pytest

from rag.vector_store import (
    NumpyVectorStore,
    build_vector_store,
    load_vector_store,
)

try:
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import qdrant_client  # noqa: F401

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ── NumpyVectorStore ─────────────────────────────────────────────────────────


def test_numpy_store_identical_vector_scores_one():
    store = NumpyVectorStore(np.array([[1.0, 2.0, 3.0]]))
    hits = store.search(np.array([1.0, 2.0, 3.0]), top_k=1)
    assert hits[0][0] == 0
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_numpy_store_orthogonal_vector_scores_zero():
    store = NumpyVectorStore(np.array([[0.0, 1.0]]))
    hits = store.search(np.array([1.0, 0.0]), top_k=1)
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)


def test_numpy_store_search_sorted_descending():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    store = NumpyVectorStore(corpus)
    hits = store.search(np.array([1.0, 0.0]), top_k=3)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][0] == 0  # index 0 is the identical vector


def test_numpy_store_top_k_clamped_to_size():
    store = NumpyVectorStore(np.array([[1.0, 0.0]]))
    hits = store.search(np.array([1.0, 0.0]), top_k=10)
    assert len(hits) == 1


def test_numpy_store_len():
    store = NumpyVectorStore(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert len(store) == 2


def test_numpy_store_rejects_1d_input():
    with pytest.raises(ValueError):
        NumpyVectorStore(np.array([1.0, 2.0, 3.0]))


def test_numpy_store_rejects_query_dimension_mismatch():
    store = NumpyVectorStore(np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="dimension"):
        store.search(np.array([1.0, 0.0]), top_k=1)


def test_numpy_store_negative_similarity_ranked_last():
    corpus = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    store = NumpyVectorStore(corpus)
    hits = store.search(np.array([1.0, 0.0]), top_k=3)
    assert hits[-1][0] == 1  # the opposite vector scores lowest
    assert hits[-1][1] == pytest.approx(-1.0, abs=1e-6)


def test_numpy_store_handles_tied_scores():
    corpus = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    store = NumpyVectorStore(corpus)
    hits = store.search(np.array([1.0, 0.0]), top_k=3)
    assert len(hits) == 3
    assert all(score == pytest.approx(1.0, abs=1e-6) for _, score in hits)
    assert sorted(idx for idx, _ in hits) == [0, 1, 2]  # every tied vector still returned


def test_numpy_store_single_dimension_vectors():
    # Cosine similarity is magnitude-invariant: any positive 1-D vector is
    # perfectly aligned with a positive query regardless of scale, so 5.0
    # and 0.5 tie at score 1.0 — only sign (direction) affects the score.
    store = NumpyVectorStore(np.array([[5.0], [-3.0], [0.5]]))
    hits = store.search(np.array([1.0]), top_k=3)
    scores_by_index = {idx: score for idx, score in hits}
    assert scores_by_index[0] == pytest.approx(1.0, abs=1e-6)
    assert scores_by_index[2] == pytest.approx(1.0, abs=1e-6)
    assert scores_by_index[1] == pytest.approx(-1.0, abs=1e-6)


def test_numpy_store_save_load_roundtrip(tmp_path):
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    store = NumpyVectorStore(corpus)
    store.save(tmp_path)

    loaded = NumpyVectorStore.load(tmp_path)
    assert len(loaded) == len(store)

    query = np.array([1.0, 0.0])
    original_hits = store.search(query, top_k=3)
    loaded_hits = loaded.search(query, top_k=3)
    assert original_hits == loaded_hits


# ── Factory ────────────────────────────────────────────────────────────────


def test_build_vector_store_numpy_default():
    store = build_vector_store(np.array([[1.0, 0.0]]))
    assert isinstance(store, NumpyVectorStore)


def test_build_vector_store_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_vector_store(np.array([[1.0, 0.0]]), backend="nonexistent")


def test_load_vector_store_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        load_vector_store(tmp_path, backend="nonexistent")


# ── FAISS backend (skipped if faiss is not installed) ───────────────────────


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu not installed")
class TestFaissVectorStore:
    def test_ranking_matches_numpy_backend(self):
        from rag.vector_store import FaissVectorStore

        rng = np.random.default_rng(seed=42)
        corpus = rng.random((20, 8)).astype(np.float32)
        query = rng.random(8).astype(np.float32)

        numpy_hits = NumpyVectorStore(corpus).search(query, top_k=5)
        faiss_hits = FaissVectorStore(corpus).search(query, top_k=5)

        numpy_order = [idx for idx, _ in numpy_hits]
        faiss_order = [idx for idx, _ in faiss_hits]
        assert numpy_order == faiss_order
        for (_, n_score), (_, f_score) in zip(numpy_hits, faiss_hits, strict=True):
            assert n_score == pytest.approx(f_score, abs=1e-4)

    def test_save_load_roundtrip(self, tmp_path):
        from rag.vector_store import FaissVectorStore

        corpus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store = FaissVectorStore(corpus)
        store.save(tmp_path)

        loaded = FaissVectorStore.load(tmp_path)
        assert len(loaded) == len(store)

    def test_rejects_query_dimension_mismatch(self):
        from rag.vector_store import FaissVectorStore

        store = FaissVectorStore(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="dimension"):
            store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)

    def test_top_k_clamped_to_size(self):
        from rag.vector_store import FaissVectorStore

        store = FaissVectorStore(np.array([[1.0, 0.0]], dtype=np.float32))
        hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10)
        assert len(hits) == 1


def test_faiss_backend_raises_clear_error_when_not_installed(monkeypatch):
    if FAISS_AVAILABLE:
        pytest.skip("faiss is installed — this test targets the missing-dependency path")

    with pytest.raises(ImportError, match="faiss-cpu"):
        build_vector_store(np.array([[1.0, 0.0]]), backend="faiss")


# ── Qdrant backend (skipped if qdrant-client is not installed) ──────────────


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantVectorStore:
    def test_ranking_matches_numpy_backend(self):
        from rag.vector_store import QdrantVectorStore

        rng = np.random.default_rng(seed=42)
        corpus = rng.random((20, 8)).astype(np.float32)
        query = rng.random(8).astype(np.float32)

        numpy_hits = NumpyVectorStore(corpus).search(query, top_k=5)
        qdrant_hits = QdrantVectorStore(corpus).search(query, top_k=5)

        numpy_order = [idx for idx, _ in numpy_hits]
        qdrant_order = [idx for idx, _ in qdrant_hits]
        assert numpy_order == qdrant_order
        for (_, n_score), (_, q_score) in zip(numpy_hits, qdrant_hits, strict=True):
            assert n_score == pytest.approx(q_score, abs=1e-4)

    def test_save_load_roundtrip(self, tmp_path):
        from rag.vector_store import QdrantVectorStore

        corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        store = QdrantVectorStore(corpus)
        store.save(tmp_path)

        loaded = QdrantVectorStore.load(tmp_path)
        assert len(loaded) == len(store)

        query = np.array([1.0, 0.0], dtype=np.float32)
        original_hits = store.search(query, top_k=3)
        loaded_hits = loaded.search(query, top_k=3)
        assert [idx for idx, _ in original_hits] == [idx for idx, _ in loaded_hits]

    def test_rejects_query_dimension_mismatch(self):
        from rag.vector_store import QdrantVectorStore

        store = QdrantVectorStore(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="dimension"):
            store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)

    def test_top_k_clamped_to_size(self):
        from rag.vector_store import QdrantVectorStore

        store = QdrantVectorStore(np.array([[1.0, 0.0]], dtype=np.float32))
        hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10)
        assert len(hits) == 1

    def test_top_k_zero_returns_no_results(self):
        from rag.vector_store import QdrantVectorStore

        store = QdrantVectorStore(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=0)
        assert hits == []

    def test_len(self):
        from rag.vector_store import QdrantVectorStore

        store = QdrantVectorStore(np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32))
        assert len(store) == 3


def test_qdrant_backend_raises_clear_error_when_not_installed(monkeypatch):
    if QDRANT_AVAILABLE:
        pytest.skip("qdrant-client is installed — this test targets the missing-dependency path")

    with pytest.raises(ImportError, match="qdrant-client"):
        build_vector_store(np.array([[1.0, 0.0]]), backend="qdrant")
