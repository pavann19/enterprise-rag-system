"""
app.py
------
Orchestration layer for the Enterprise RAG pipeline.

Coordinates two strictly separated execution phases:

  INGESTION  — document loading, segmentation, and vector encoding (run once)
    data/*.txt → chunk_text() → embed_texts() → corpus_embeddings (np.ndarray)
                                               → chunk metadata list

  QUERY      — retrieval, generation, and schema-validated output (run per request)
    query → embed_texts() → retrieve() → generate_answer() → validate() → RAGResponse

All inference is local via Ollama. The validated output conforms to
the RAGResponse TypedDict contract, ensuring reliable integration with
downstream enterprise APIs and audit pipelines.

Multi-document support: the ingestion phase walks the entire data/ directory,
loads every .txt file, and builds a single unified embedding corpus. Each chunk
retains its source filename as metadata, enabling cross-document retrieval and
source attribution in the final structured response.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rag.chunker  import chunk_text
from rag.embedder import embed_texts
from rag.retriever import retrieve
from rag.generator import generate_answer
from validator.json_validator import validate, ValidationError, RAGResponse

# ── Configuration ───────────────────────────────────────────────────────────────

DATA_DIR      = Path(__file__).parent / "data"
EMBED_MODEL   = "nomic-embed-text"
GEN_MODEL     = "mistral"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50
TOP_K         = 3


# ── Phase 1: Ingestion ──────────────────────────────────────────────────────────

def ingest(
    data_dir: Path = DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model: str = EMBED_MODEL,
) -> Tuple[List[str], List[Dict[str, str]], np.ndarray]:
    """
    Walks data_dir, loads every .txt file, chunks each document independently,
    and encodes all chunks into a unified embedding matrix.

    Args:
        data_dir:      Directory containing .txt documents.
        chunk_size:    Approximate character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        embed_model:   Ollama embedding model identifier.

    Returns:
        chunks (List[str]):            All chunk texts across all documents.
        metadata (List[Dict[str,str]]): Parallel list; each entry is
                                         {"source": filename}.
        corpus_embeddings (np.ndarray): Shape (n_chunks, embedding_dim).

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no .txt files.
        ConnectionError:   If Ollama is not reachable.
    """
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. "
            "Add documents to the data/ directory before running."
        )

    all_chunks: List[str]            = []
    all_metadata: List[Dict[str, str]] = []

    for filepath in txt_files:
        text   = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        source = filepath.name

        all_chunks.extend(chunks)
        all_metadata.extend({"source": source} for _ in chunks)

        print(f"  [ingest] {source}: {len(chunks)} chunks")

    print(f"  [ingest] Total: {len(all_chunks)} chunks from {len(txt_files)} document(s)")

    corpus_embeddings = embed_texts(all_chunks, model=embed_model)
    return all_chunks, all_metadata, corpus_embeddings


# ── Phase 2: Query pipeline ─────────────────────────────────────────────────────

def query_pipeline(
    query: str,
    chunks: List[str],
    metadata: List[Dict[str, str]],
    corpus_embeddings: np.ndarray,
    gen_model: str   = GEN_MODEL,
    embed_model: str = EMBED_MODEL,
    top_k: int       = TOP_K,
) -> RAGResponse:
    """
    Encodes the query, retrieves top-k passages with source metadata,
    generates a context-grounded answer, and validates the structured response.

    Args:
        query:             User question.
        chunks:            All chunk texts (from ingest()).
        metadata:          Parallel metadata list (from ingest()).
        corpus_embeddings: Precomputed corpus embedding matrix.
        gen_model:         Ollama generation model identifier.
        embed_model:       Ollama embedding model identifier.
        top_k:             Number of passages to retrieve.

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
        query_embedding   = query_embedding,
        corpus_embeddings = corpus_embeddings,
        chunks            = chunks,
        metadata          = metadata,
        top_k             = top_k,
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
        chunks, metadata, corpus_embeddings = ingest()
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n── QUERY ──────────────────────────────────")
    print(f"  Query: {query}\n")

    try:
        response = query_pipeline(
            query             = query,
            chunks            = chunks,
            metadata          = metadata,
            corpus_embeddings = corpus_embeddings,
        )
    except ConnectionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as exc:
        print(f"[VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(response, indent=2))
