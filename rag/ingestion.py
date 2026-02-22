"""
rag/ingestion.py
----------------
Lightweight, air-gapped multi-document ingestion layer designed for
deterministic enterprise workflows.

Walks a directory of plaintext documents, chunks each file independently
using word-boundary-aligned segmentation, and encodes all chunks into a
unified float32 embedding matrix via Ollama.

Each chunk retains source-level metadata ({ "source": filename }), enabling
cross-document retrieval, source attribution, and auditability in the final
structured RAGResponse.

No external calls beyond Ollama (localhost). No database. No file writes.
Drop additional .txt files into the data/ directory and restart — no code
changes required.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rag.chunker         import chunk_text
from rag.embedder        import embed_texts
from rag.logging_config  import get_logger

log = get_logger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest(
    data_dir: Path,
    chunk_size: int    = 300,
    chunk_overlap: int = 50,
    embed_model: str   = "nomic-embed-text",
) -> Tuple[List[str], List[Dict[str, str]], np.ndarray]:
    """
    Walks data_dir, loads every .txt file, chunks each document independently,
    and encodes all chunks into a unified embedding matrix.

    Each chunk is tagged with its source filename, maintaining a strictly
    parallel relationship between chunks, metadata, and corpus_embeddings rows.

    Args:
        data_dir:      Directory containing .txt documents.
        chunk_size:    Approximate character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks to preserve
                       cross-boundary semantic context.
        embed_model:   Ollama embedding model identifier.

    Returns:
        chunks (List[str]):              All chunk texts across all documents.
        metadata (List[Dict[str, str]]): Parallel list; each entry is
                                          {"source": filename}.
        corpus_embeddings (np.ndarray):  Shape (n_chunks, embedding_dim),
                                          dtype float32.

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no .txt files.
        ConnectionError:   If the Ollama embedding endpoint is not reachable.
    """
    log.info("Ingestion started — scanning %s", data_dir)
    txt_files = sorted(Path(data_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. "
            "Add documents to the data/ directory before running."
        )

    log.info("Found %d document(s) to ingest", len(txt_files))
    all_chunks:   List[str]             = []
    all_metadata: List[Dict[str, str]]  = []

    for filepath in txt_files:
        text   = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        source = filepath.name

        all_chunks.extend(chunks)
        all_metadata.extend({"source": source} for _ in chunks)

        log.debug("%s → %d chunks", source, len(chunks))

    log.info(
        "Chunking complete — %d total chunks from %d document(s)",
        len(all_chunks), len(txt_files),
    )

    log.info("Embedding corpus with model '%s' …", embed_model)
    corpus_embeddings = embed_texts(all_chunks, model=embed_model)
    log.info(
        "Ingestion complete — corpus shape %s",
        corpus_embeddings.shape,
    )
    return all_chunks, all_metadata, corpus_embeddings
