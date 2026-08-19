"""
rag/retriever.py
----------------
Retrieves the top-k passages most semantically similar to a query.

Similarity search itself is delegated to a `VectorStore` (rag/vector_store.py)
so this module has no opinion on whether the backend is a raw NumPy array or
a FAISS index — it only translates search results back into chunk text and
source metadata.

Retrieval is the sole responsibility of this module.
It receives a pre-built VectorStore — it does NOT call Ollama or build indices.

Each result carries source-level metadata (filename), enabling cross-document
attribution and auditability in the final structured response.
"""

from typing import Dict, List

import numpy as np

from rag.logging_config import get_logger
from rag.vector_store import VectorStore

log = get_logger(__name__)


def retrieve(
    query_embedding: np.ndarray,
    vector_store: VectorStore,
    chunks: List[str],
    metadata: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """
    Returns the top-k chunks most semantically similar to the query,
    each annotated with its source document filename and similarity score.

    Args:
        query_embedding: 1-D float array for the query.
        vector_store:    A built VectorStore (rag.vector_store) holding the
                          corpus embeddings, one entry per chunk.
        chunks:          Plaintext chunks aligned with vector_store's index order.
        metadata:        Parallel list of dicts, one per chunk.
                         Each dict must contain at least {"source": filename}.
        top_k:           Maximum number of results to return.

    Returns:
        List of dicts: [{"text": str, "score": float, "source": str}, ...]
        Sorted by score descending.

    Raises:
        ValueError: If chunks/metadata/vector_store are misaligned or empty.
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")
    if len(chunks) != len(vector_store):
        raise ValueError(
            f"Length mismatch: {len(chunks)} chunks vs "
            f"{len(vector_store)} vectors in the store."
        )
    if len(chunks) != len(metadata):
        raise ValueError(
            f"Length mismatch: {len(chunks)} chunks vs "
            f"{len(metadata)} metadata entries."
        )

    hits = vector_store.search(query_embedding, top_k)

    results = [
        {
            "text":   chunks[i],
            "score":  round(score, 4),
            "source": metadata[i].get("source", "unknown"),
        }
        for i, score in hits
    ]
    log.debug(
        "retrieve() top-%d scores: %s",
        len(results),
        ", ".join(f"{r['source']}={r['score']}" for r in results),
    )
    return results
