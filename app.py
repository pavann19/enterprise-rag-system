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

All inference is local via Ollama. The validated output conforms to
the RAGResponse TypedDict contract, ensuring reliable integration with
downstream enterprise APIs and audit pipelines.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from rag.ingestion import ingest
from rag.embedder  import embed_texts
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.vector_store import VectorStore
from validator.json_validator import validate, ValidationError, RAGResponse

# ── Configuration ───────────────────────────────────────────────────────────────

DATA_DIR       = Path(__file__).parent / "data"
CACHE_DIR      = Path(__file__).parent / ".cache" / "corpus"
EMBED_MODEL    = "nomic-embed-text"
GEN_MODEL      = "mistral"
CHUNK_SIZE     = 300
CHUNK_OVERLAP  = 50
TOP_K          = 3
VECTOR_BACKEND = "numpy"   # or "faiss" — see rag/vector_store.py


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
    chunks: List[str],
    metadata: List[Dict[str, str]],
    vector_store: VectorStore,
    gen_model: str   = GEN_MODEL,
    embed_model: str = EMBED_MODEL,
    top_k: int       = TOP_K,
) -> RAGResponse:
    """
    Encodes the query, retrieves top-k passages with source metadata,
    generates a context-grounded answer, and validates the structured response.

    Args:
        query:        User question.
        chunks:       All chunk texts (from ingest()).
        metadata:     Parallel metadata list (from ingest()).
        vector_store: Built VectorStore over the corpus embeddings (from ingest()).
        gen_model:    Ollama generation model identifier.
        embed_model:  Ollama embedding model identifier.
        top_k:        Number of passages to retrieve.

    Returns:
        A validated RAGResponse TypedDict.

    Raises:
        ConnectionError:  If Ollama is unreachable.
        ValidationError:  If the pipeline output fails schema validation.
    """
    # 1. Embed query
    query_embedding = embed_texts([query], model=embed_model)[0]

    # 2. Retrieve top-k passages with source metadata
    results = retrieve(
        query_embedding = query_embedding,
        vector_store    = vector_store,
        chunks          = chunks,
        metadata        = metadata,
        top_k           = top_k,
    )
    # results: [{"text": str, "score": float, "source": str}, ...]

    passages = [r["text"] for r in results]

    # 3. Generate grounded answer
    answer = generate_answer(query=query, passages=passages, model=gen_model)

    # 4. Build and validate structured response
    raw_response = {
        "query":   query,
        "answer":  answer,
        "sources": [{"text": r["text"], "source": r["source"]} for r in results],
        "model":   gen_model,
    }
    return validate(raw_response)


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the policy for budget variances exceeding 10%?"
    )

    print("\n── INGESTION ──────────────────────────────")
    try:
        chunks, metadata, vector_store = ingest(
            data_dir      = DATA_DIR,
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            embed_model   = EMBED_MODEL,
            backend       = VECTOR_BACKEND,
            cache_dir     = CACHE_DIR,
        )
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n── QUERY ──────────────────────────────────")
    print(f"  Query: {query}\n")

    try:
        response = query_pipeline(
            query        = query,
            chunks       = chunks,
            metadata     = metadata,
            vector_store = vector_store,
        )
    except ConnectionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as exc:
        print(f"[VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(response, indent=2))
