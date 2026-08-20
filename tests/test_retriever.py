import numpy as np
import pytest

from rag.retriever import retrieve
from rag.vector_store import NumpyVectorStore


def test_retrieve_returns_top_k_sorted_descending():
    query = np.array([1.0, 0.0])
    corpus = np.array(
        [
            [1.0, 0.0],  # identical -> score 1.0
            [0.0, 1.0],  # orthogonal -> score 0.0
            [0.9, 0.1],  # close -> high score
        ]
    )
    chunks = ["exact", "unrelated", "close"]
    metadata = [{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}]
    store = NumpyVectorStore(corpus)

    results = retrieve(query, store, chunks, metadata, top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "exact"
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["source"] == "a.txt"


def test_retrieve_top_k_clamped_to_corpus_size():
    query = np.array([1.0, 0.0])
    store = NumpyVectorStore(np.array([[1.0, 0.0]]))
    results = retrieve(query, store, ["only"], [{"source": "x.txt"}], top_k=5)
    assert len(results) == 1


def test_retrieve_rejects_empty_chunks():
    store = NumpyVectorStore(np.empty((0, 2)))
    with pytest.raises(ValueError):
        retrieve(np.array([1.0, 0.0]), store, [], [])


def test_retrieve_rejects_mismatched_embeddings_length():
    query = np.array([1.0, 0.0])
    store = NumpyVectorStore(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        retrieve(query, store, ["only one chunk"], [{"source": "x.txt"}])


def test_retrieve_rejects_mismatched_metadata_length():
    query = np.array([1.0, 0.0])
    store = NumpyVectorStore(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        retrieve(query, store, ["chunk"], [])


def test_retrieve_propagates_query_dimension_mismatch():
    query = np.array([1.0, 0.0, 0.0])  # 3-D query against a 2-D corpus
    store = NumpyVectorStore(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="dimension"):
        retrieve(query, store, ["chunk"], [{"source": "x.txt"}])


def test_retrieve_defaults_missing_source_to_unknown():
    query = np.array([1.0, 0.0])
    store = NumpyVectorStore(np.array([[1.0, 0.0]]))
    results = retrieve(query, store, ["chunk"], [{}])  # metadata entry has no "source" key
    assert results[0]["source"] == "unknown"


def test_retrieve_preserves_chunk_text_verbatim():
    query = np.array([1.0, 0.0])
    text = "Section 3: Budget Variance Policy — variances exceeding 10% trigger a review."
    store = NumpyVectorStore(np.array([[1.0, 0.0]]))
    results = retrieve(query, store, [text], [{"source": "financial_policy.txt"}])
    assert results[0]["text"] == text


def test_retrieve_top_k_zero_returns_no_results():
    query = np.array([1.0, 0.0])
    store = NumpyVectorStore(np.array([[1.0, 0.0], [0.0, 1.0]]))
    results = retrieve(query, store, ["a", "b"], [{"source": "a.txt"}, {"source": "b.txt"}], top_k=0)
    assert results == []


def test_retrieve_scores_are_rounded():
    query = np.array([1.0, 0.3])
    store = NumpyVectorStore(np.array([[0.7, 0.9]]))
    results = retrieve(query, store, ["chunk"], [{"source": "x.txt"}], top_k=1)
    # round(..., 4) in retrieve() — score should have at most 4 decimal places
    assert results[0]["score"] == round(results[0]["score"], 4)
