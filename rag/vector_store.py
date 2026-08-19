"""
rag/vector_store.py
--------------------
Pluggable vector-index backend, decoupled from retrieval logic.

Three backends implement the same `VectorStore` interface:
  - NumpyVectorStore:  exact cosine similarity over an in-memory float32
                       array. Zero extra dependencies. Default backend.
  - FaissVectorStore:  exact inner-product search over a FAISS
                       IndexFlatIP, built from L2-normalized vectors
                       (so inner product == cosine similarity). Optional —
                       requires `pip install faiss-cpu`.
  - QdrantVectorStore: cosine search via qdrant-client's embedded/local
                       mode (no Qdrant server to run). Optional — requires
                       `pip install qdrant-client`. See that class's
                       docstring for how its save()/load() differs from
                       the other two backends.

All three support save()/load() so a corpus only needs to be embedded
once; subsequent runs load the persisted index from disk. See
rag/ingestion.py for the cache-hit/cache-miss logic that uses this.

Swapping backends never touches rag/retriever.py or any downstream code —
that is the whole point of hiding the index behind this interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Protocol, Tuple

import numpy as np

from rag.logging_config import get_logger

log = get_logger(__name__)


class VectorStore(Protocol):
    """Common interface every vector backend must implement."""

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Returns [(corpus_index, similarity_score), ...] sorted descending, length <= top_k."""
        ...

    def save(self, path: Path) -> None:
        """Persists the index to disk under `path` (a directory)."""
        ...

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Reconstructs the index previously written by save()."""
        ...

    def __len__(self) -> int:
        ...


# ── NumPy backend ────────────────────────────────────────────────────────────

class NumpyVectorStore:
    """Exact cosine-similarity search over an in-memory float32 array."""

    _FILENAME = "embeddings.npy"

    def __init__(self, embeddings: np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self._embeddings = embeddings.astype(np.float32, copy=False)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if query_embedding.shape[-1] != self._embeddings.shape[-1]:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[-1]} does not match "
                f"corpus dimension {self._embeddings.shape[-1]}. This usually means the "
                f"corpus was embedded with a different model/backend than the query — "
                f"re-run ingestion with a matching embed_model/embed_backend."
            )
        query_norm  = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        corpus_norm = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        )
        scores  = corpus_norm @ query_norm
        top_k   = min(top_k, len(self._embeddings))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / self._FILENAME, self._embeddings)
        log.info("NumpyVectorStore saved — %d vectors → %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorStore":
        embeddings = np.load(path / cls._FILENAME)
        log.info("NumpyVectorStore loaded — %d vectors ← %s", len(embeddings), path)
        return cls(embeddings)

    def __len__(self) -> int:
        return len(self._embeddings)


# ── FAISS backend (optional) ─────────────────────────────────────────────────

class FaissVectorStore:
    """Exact inner-product search over a FAISS IndexFlatIP.

    Vectors are L2-normalized before insertion so inner product equals
    cosine similarity — same ranking semantics as NumpyVectorStore, just
    backed by a purpose-built similarity-search library instead of raw
    NumPy broadcasting.
    """

    _INDEX_FILENAME = "index.faiss"

    def __init__(self, embeddings: np.ndarray):
        try:
            import faiss  # noqa: F401 (imported for the module-level check below)
        except ImportError as exc:
            raise ImportError(
                "faiss is not installed. Run: pip install faiss-cpu\n"
                "Or use backend='numpy' (the default) instead."
            ) from exc

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

        self._faiss = faiss
        self._dim   = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._n     = 0
        self._add(embeddings)

    def _add(self, embeddings: np.ndarray) -> None:
        normalized = self._normalize(embeddings)
        self._index.add(normalized)
        self._n += len(embeddings)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.astype(np.float32, copy=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        return embeddings / norms

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if query_embedding.shape[-1] != self._dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[-1]} does not match "
                f"corpus dimension {self._dim}. This usually means the corpus was "
                f"embedded with a different model/backend than the query — re-run "
                f"ingestion with a matching embed_model/embed_backend."
            )
        query = self._normalize(query_embedding.reshape(1, -1).astype(np.float32))
        top_k = min(top_k, self._n)
        scores, indices = self._index.search(query, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / self._INDEX_FILENAME))
        log.info("FaissVectorStore saved — %d vectors → %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss is not installed. Run: pip install faiss-cpu"
            ) from exc

        instance = object.__new__(cls)
        instance._faiss = faiss
        instance._index = faiss.read_index(str(path / cls._INDEX_FILENAME))
        instance._dim   = instance._index.d
        instance._n     = instance._index.ntotal
        log.info("FaissVectorStore loaded — %d vectors ← %s", instance._n, path)
        return instance

    def __len__(self) -> int:
        return self._n


# ── Qdrant backend (optional) ────────────────────────────────────────────────

class QdrantVectorStore:
    """Cosine search via qdrant-client's embedded/local mode.

    Uses `QdrantClient(":memory:")` — no Qdrant server process, works
    anywhere the other two backends do (Docker, hosted demos). This is a
    deliberate simplification: Qdrant's own on-disk local mode
    (`QdrantClient(path=...)`) holds an exclusive file lock for as long as
    the client is open, which doesn't fit this project's ingest-once/
    query-many-times-in-a-different-process-later cache pattern without
    extra lock-lifecycle handling. Instead, save()/load() persist the raw
    embedding matrix (same as NumpyVectorStore) and load() rebuilds an
    in-memory Qdrant collection from it. You get Qdrant's query engine and
    ranking behavior; you do not get Qdrant's own on-disk index format —
    call this out if you're evaluating Qdrant specifically for its
    storage engine rather than its API.
    """

    _EMBEDDINGS_FILENAME = "qdrant_embeddings.npy"
    _COLLECTION_NAME      = "corpus"

    def __init__(self, embeddings: np.ndarray):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is not installed. Run: pip install qdrant-client\n"
                "Or use backend='numpy' (the default) instead."
            ) from exc

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

        self._embeddings = embeddings.astype(np.float32, copy=False)
        self._dim        = self._embeddings.shape[1]
        self._client = QdrantClient(":memory:")
        self._client.create_collection(
            self._COLLECTION_NAME,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        if len(self._embeddings) > 0:
            self._client.upsert(
                self._COLLECTION_NAME,
                points=[
                    PointStruct(id=i, vector=vec.tolist(), payload={})
                    for i, vec in enumerate(self._embeddings)
                ],
            )

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if query_embedding.shape[-1] != self._dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[-1]} does not match "
                f"corpus dimension {self._dim}. This usually means the corpus was "
                f"embedded with a different model/backend than the query — re-run "
                f"ingestion with a matching embed_model/embed_backend."
            )
        top_k = min(top_k, len(self._embeddings))
        if top_k == 0:
            return []
        hits = self._client.query_points(
            self._COLLECTION_NAME,
            query=query_embedding.astype(np.float32).tolist(),
            limit=top_k,
        ).points
        return [(int(hit.id), float(hit.score)) for hit in hits]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / self._EMBEDDINGS_FILENAME, self._embeddings)
        log.info("QdrantVectorStore saved — %d vectors → %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "QdrantVectorStore":
        embeddings = np.load(path / cls._EMBEDDINGS_FILENAME)
        log.info("QdrantVectorStore loaded — %d vectors ← %s", len(embeddings), path)
        return cls(embeddings)

    def __len__(self) -> int:
        return len(self._embeddings)


# ── Factory ────────────────────────────────────────────────────────────────

_BACKENDS = {
    "numpy":  NumpyVectorStore,
    "faiss":  FaissVectorStore,
    "qdrant": QdrantVectorStore,
}


def build_vector_store(embeddings: np.ndarray, backend: str = "numpy") -> VectorStore:
    """
    Builds a VectorStore from a raw embedding matrix.

    Args:
        embeddings: 2-D float array, shape (n_chunks, dim).
        backend:    "numpy" (default, zero extra deps), "faiss", or "qdrant".

    Raises:
        ValueError:  If backend is not a known name.
        ImportError: If the selected backend's package is not installed.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown vector store backend '{backend}'. "
            f"Available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend](embeddings)


def load_vector_store(path: Path, backend: str = "numpy") -> VectorStore:
    """Loads a previously-saved VectorStore of the given backend from `path`."""
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown vector store backend '{backend}'. "
            f"Available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend].load(path)
