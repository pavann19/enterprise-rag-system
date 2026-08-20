"""
rag/embedder.py
---------------
Vector encoding layer with two interchangeable backends:

  "ollama" (default) — calls Ollama's /api/embeddings endpoint. Fully
                        local, but requires an Ollama server reachable at
                        OLLAMA_HOST — not available on most free hosting
                        platforms (Streamlit Community Cloud, HF Spaces),
                        which don't let you run a persistent background
                        server alongside the app.

  "local"             — runs sentence-transformers in-process. No server,
                        no network call, no API key. This is what makes a
                        hosted public demo possible: the model weights
                        download once (from Hugging Face, at first use) and
                        embeddings run on the platform's own CPU.

Select via the EMBED_BACKEND environment variable, or pass backend=
explicitly to embed_texts(). Both return the same float32 NumPy array
shape, so nothing downstream (rag/vector_store.py, rag/retriever.py) knows
or cares which one produced it. Retrieval and generation are independent
choices — see rag/generator.py for the equivalent split on the generation
side.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rag._http import OLLAMA_HOST, ollama_post

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama")
# Ollama's /api/embeddings endpoint takes one prompt per request — there's
# no batch endpoint to call instead. At corpus scale (hundreds/thousands of
# chunks), one request at a time is the dominant ingestion cost. Concurrent
# requests let Ollama pipeline them instead of round-tripping serially —
# see eval/benchmark_scale.py for the measured before/after. Kept modest by
# default since a single Ollama instance still serializes on one loaded
# model; higher concurrency past a point just queues instead of helping.
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "8"))
# ──────────────────────────────────────────────────────────────────────────────

_local_model_cache: dict = {}


def _embed_texts_ollama(texts: list[str], model: str) -> np.ndarray:
    def _embed_one(text: str) -> list[float]:
        response = ollama_post(EMBED_URL, {"model": model, "prompt": text})
        if "embedding" not in response:
            raise RuntimeError(f"Ollama embedding response missing 'embedding' key.\nGot: {response}")
        return response["embedding"]

    if len(texts) == 1 or EMBED_CONCURRENCY <= 1:
        vectors = [_embed_one(t) for t in texts]
    else:
        # map() preserves input order in its output regardless of which
        # worker finishes first — order must match `texts` so the caller
        # can zip embeddings back to their source chunks.
        with ThreadPoolExecutor(max_workers=min(EMBED_CONCURRENCY, len(texts))) as pool:
            vectors = list(pool.map(_embed_one, texts))

    return np.array(vectors, dtype=np.float32)


def _embed_texts_local(texts: list[str], model: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "backend='local' requires sentence-transformers.\n" "  → pip install sentence-transformers"
        ) from exc

    if model not in _local_model_cache:
        _local_model_cache[model] = SentenceTransformer(model)
    embeddings = _local_model_cache[model].encode(texts, convert_to_numpy=True)
    return embeddings.astype(np.float32, copy=False)


_BACKENDS = {
    "ollama": (_embed_texts_ollama, DEFAULT_OLLAMA_MODEL),
    "local": (_embed_texts_local, DEFAULT_LOCAL_MODEL),
}


def embed_texts(
    texts: list[str],
    model: str = None,
    backend: str = None,
) -> np.ndarray:
    """
    Embeds a list of strings into a 2-D float32 numpy array.

    Args:
        texts:   Non-empty list of strings to embed.
        model:   Model identifier for the chosen backend. Defaults to
                 DEFAULT_OLLAMA_MODEL or DEFAULT_LOCAL_MODEL depending on
                 which backend is active.
        backend: "ollama" or "local". Defaults to the EMBED_BACKEND
                 environment variable (itself defaulting to "ollama").

    Returns:
        np.ndarray of shape (len(texts), embedding_dim). Note: the two
        backends use different embedding models with different
        dimensions — a vector store built with one backend is not
        compatible with the other. Re-ingest after switching backends.

    Raises:
        ValueError:      If `texts` is empty, or backend is unrecognized.
        ImportError:     If backend="local" but sentence-transformers isn't installed.
        ConnectionError: If backend="ollama" and Ollama is unreachable.
        RuntimeError:    If the Ollama response is missing the 'embedding' key.
    """
    if not texts:
        raise ValueError("texts must not be empty.")

    backend = backend or EMBED_BACKEND
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown embed backend '{backend}'. Available: {sorted(_BACKENDS)}")

    embed_fn, default_model = _BACKENDS[backend]
    return embed_fn(texts, model or default_model)
