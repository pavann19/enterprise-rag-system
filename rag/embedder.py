"""
rag/embedder.py
---------------
Lightweight, air-gapped vector encoding layer designed for deterministic
enterprise document indexing workflows.

Converts raw text segments into dense float32 embeddings via Ollama's
/api/embeddings endpoint. All inference is local — zero external API calls,
zero data egress. Embeddings are returned as a NumPy array suitable for
direct ingestion into the in-memory vector computation layer (retriever.py).

Prerequisite:
    ollama pull nomic-embed-text
    ollama serve
"""

from typing import List

import numpy as np

from rag._http import ollama_post

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_URL          = "http://localhost:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
# ──────────────────────────────────────────────────────────────────────────────


def embed_texts(
    texts: List[str],
    model: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """
    Embeds a list of strings into a 2-D float32 numpy array.

    Each string is sent to Ollama individually so that the embedding
    dimension is inferred from the first response.

    Args:
        texts: Non-empty list of strings to embed.
        model: Ollama embedding model name.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).

    Raises:
        ValueError:      If `texts` is empty.
        ConnectionError: If Ollama is unreachable (propagated from _http).
        RuntimeError:    If the Ollama response is missing the 'embedding' key.
    """
    if not texts:
        raise ValueError("texts must not be empty.")

    vectors: List[List[float]] = []

    for text in texts:
        response = ollama_post(EMBED_URL, {"model": model, "prompt": text})

        if "embedding" not in response:
            raise RuntimeError(
                f"Ollama embedding response missing 'embedding' key.\n"
                f"Got: {response}"
            )

        vectors.append(response["embedding"])

    return np.array(vectors, dtype=np.float32)
