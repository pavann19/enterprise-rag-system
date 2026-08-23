"""
app.py
------
Orchestration layer for the Enterprise RAG pipeline.

Coordinates two strictly separated execution phases:

  INGESTION  — delegated to rag/ingestion.py (run once per knowledge base,
               or loaded from cache if the corpus/config is unchanged)
    data/*.txt → ingest() → (chunks, metadata, vector_store)

  QUERY      — retrieval, generation, and schema-validated output (run per request)
    query → embed_texts() → retrieve() → generate_answer() → validate() → RAGResponse

By default, every call is local via Ollama — see rag/embedder.py and
rag/generator.py for the optional cloud backends (EMBED_BACKEND=local,
GEN_BACKEND=anthropic) that exist to make a publicly hosted demo possible,
where a background Ollama server can't run. The validated output conforms
to the RAGResponse TypedDict contract, ensuring reliable integration with
downstream enterprise APIs and audit pipelines.
"""

import json
import os
import sys
from pathlib import Path

from rag.embedder import embed_texts
from rag.generator import generate_answer
from rag.ingestion import ingest
from rag.reranker import RERANK_ENABLED, rerank
from rag.retriever import retrieve
from rag.vector_store import VectorStore
from validator.json_validator import RAGResponse, ValidationError, validate

# ── Configuration ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / ".cache" / "corpus"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND") or "numpy"  # or "faiss"/"qdrant" — see rag/vector_store.py

# Embedding: "ollama" (default, local, needs OLLAMA_HOST reachable) or
# "local" (sentence-transformers, in-process, no server — hosted demos).
EMBED_BACKEND = os.environ.get("EMBED_BACKEND") or "ollama"
_EMBED_MODEL_DEFAULTS = {"ollama": "nomic-embed-text", "local": "all-MiniLM-L6-v2"}
# `or` (not .get(key, default)) so an env var set to "" — e.g. docker-compose's
# ${EMBED_MODEL:-} when the caller didn't override it — is also treated as unset.
EMBED_MODEL = os.environ.get("EMBED_MODEL") or _EMBED_MODEL_DEFAULTS.get(EMBED_BACKEND, "nomic-embed-text")

# Generation: "ollama" (default, local) or "anthropic"/"groq" (cloud,
# opt-in only — each requires its own API key — see rag/generator.py for
# why these exist).
GEN_BACKEND = os.environ.get("GEN_BACKEND") or "ollama"
_GEN_MODEL_DEFAULTS = {
    "ollama": "mistral",
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "openai/gpt-oss-20b",
}
GEN_MODEL = os.environ.get("GEN_MODEL") or _GEN_MODEL_DEFAULTS.get(GEN_BACKEND, "mistral")


# ── Phase 1: Ingestion — see rag/ingestion.py ────────────────────────────────
#
# ingest() is imported directly from rag.ingestion.
# Call signature:
#   ingest(data_dir, chunk_size, chunk_overlap, embed_model, backend, cache_dir)
#      → (chunks, metadata, vector_store)
#
# Passing cache_dir persists the embedded corpus to disk, keyed by a hash of
# the document contents and ingestion config — an unchanged corpus loads
# from disk on the next run instead of re-embedding every chunk via Ollama.


# ── Phase 2: Query pipeline ─────────────────────────────────────────────────────


def query_pipeline(
    query: str,
    chunks: list[str],
    metadata: list[dict[str, str]],
    vector_store: VectorStore,
    gen_model: str = GEN_MODEL,
    gen_backend: str = GEN_BACKEND,
    embed_model: str = EMBED_MODEL,
    embed_backend: str = EMBED_BACKEND,
    top_k: int = TOP_K,
) -> RAGResponse:
    """
    Encodes the query, retrieves top-k passages with source metadata,
    generates a context-grounded answer, and validates the structured response.

    Args:
        query:         User question.
        chunks:        All chunk texts (from ingest()).
        metadata:      Parallel metadata list (from ingest()).
        vector_store:  Built VectorStore over the corpus embeddings (from ingest()).
        gen_model:     Generation model identifier for gen_backend.
        gen_backend:   "ollama" (default) or "anthropic" — see rag/generator.py.
        embed_model:   Embedding model identifier for embed_backend.
        embed_backend: "ollama" (default) or "local" — see rag/embedder.py.
        top_k:         Number of passages to retrieve.

    Returns:
        A validated RAGResponse TypedDict.

    Raises:
        ConnectionError:  If the selected backend's endpoint is unreachable.
        ValidationError:  If the pipeline output fails schema validation.
    """
    # 1. Embed query
    query_embedding = embed_texts([query], model=embed_model, backend=embed_backend)[0]

    # 2. Retrieve top-k passages with source metadata
    results = retrieve(
        query_embedding=query_embedding,
        vector_store=vector_store,
        chunks=chunks,
        metadata=metadata,
        top_k=top_k,
    )
    # results: [{"text": str, "score": float, "source": str}, ...]

    if RERANK_ENABLED:
        results = rerank(query, results)

    passages = [r["text"] for r in results]

    # 3. Generate grounded answer
    answer = generate_answer(query=query, passages=passages, model=gen_model, backend=gen_backend)

    # 4. Build and validate structured response
    raw_response = {
        "query": query,
        "answer": answer,
        "sources": [{"text": r["text"], "source": r["source"]} for r in results],
        "model": gen_model,
    }
    return validate(raw_response)


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in
    # tests/test_app_cli.py, invisible to coverage.py
    # in the parent pytest process
    # Without this, redirecting/capturing stdout on Windows (a log file, a
    # CI runner, subprocess.PIPE) defaults to the console codepage (cp1252)
    # instead of UTF-8, and the box-drawing characters in the prints below
    # crash with UnicodeEncodeError. A real terminal already negotiates
    # UTF-8 correctly, so this only matters for the non-interactive case —
    # which is exactly the case that matters for a CLI meant to be scripted.
    if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    query = sys.argv[1] if len(sys.argv) > 1 else "What is the policy for budget variances exceeding 10%?"

    print("\n── INGESTION ──────────────────────────────")
    try:
        chunks, metadata, vector_store = ingest(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embed_model=EMBED_MODEL,
            embed_backend=EMBED_BACKEND,
            backend=VECTOR_BACKEND,
            cache_dir=CACHE_DIR,
        )
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n── QUERY ──────────────────────────────────")
    print(f"  Query: {query}\n")

    try:
        response = query_pipeline(
            query=query,
            chunks=chunks,
            metadata=metadata,
            vector_store=vector_store,
        )
    except ConnectionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as exc:
        print(f"[VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(response, indent=2))
