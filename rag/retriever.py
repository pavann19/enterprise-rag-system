"""
rag/retriever.py
----------------
Ranks text chunks by semantic similarity to a query embedding.

Retrieval is the only responsibility of this module.
It receives pre-computed embeddings — it does NOT call Ollama.
"""

from typing import List, Tuple

import numpy as np


# ── Internal helper ────────────────────────────────────────────────────────────

def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a query vector and every row in corpus.

    Args:
        query:  1-D array of shape (dim,).
        corpus: 2-D array of shape (n, dim).

    Returns:
        1-D similarity scores of shape (n,), values in [-1, 1].
    """
    # Normalise to unit vectors (add epsilon to avoid division by zero)
    query_norm  = query  / (np.linalg.norm(query)  + 1e-10)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return corpus_norm @ query_norm


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    chunks: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Returns the top-k chunks most semantically similar to the query.

    Args:
        query_embedding:   1-D float array for the query.
        corpus_embeddings: 2-D float array, one row per chunk.
        chunks:            Plaintext chunks aligned with corpus_embeddings.
        top_k:             Maximum number of results to return.

    Returns:
        List of (chunk_text, similarity_score) sorted by score descending.

    Raises:
        ValueError: If chunks and embeddings are misaligned or empty.
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")
    if len(chunks) != len(corpus_embeddings):
        raise ValueError(
            f"Length mismatch: {len(chunks)} chunks vs "
            f"{len(corpus_embeddings)} embeddings."
        )

    scores   = _cosine_similarity(query_embedding, corpus_embeddings)
    top_idx  = np.argsort(scores)[::-1][: min(top_k, len(chunks))]

    return [(chunks[i], float(scores[i])) for i in top_idx]
