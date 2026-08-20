"""
rag/ingestion.py
----------------
Lightweight, air-gapped multi-document ingestion layer designed for
deterministic enterprise workflows.

Walks a directory of documents (.txt, .pdf — see rag/loaders.py for the
full list), chunks each file independently using word-boundary-aligned
segmentation, and encodes all chunks into a VectorStore
(rag/vector_store.py) via Ollama.

Each chunk retains source-level metadata ({ "source": filename }), enabling
cross-document retrieval, source attribution, and auditability in the final
structured RAGResponse.

Embedding is the expensive step (one round-trip per chunk against whichever
embedding backend is configured — see rag/embedder.py). If `cache_dir` is
given, the resulting chunks/metadata/vector-store are persisted to disk
keyed by a hash of (file contents, chunk_size, chunk_overlap, embed_model,
embed_backend, vector backend). A subsequent call with an unchanged corpus
and config loads straight from disk — no re-embedding — turning ingestion
from an every-run cost into a one-time cost that's only repeated when the
corpus or config actually changes.

No database. With the default embed_backend="ollama", no external calls
beyond Ollama (localhost). Drop additional supported files into the data/
directory and restart — no code changes required.
"""

import hashlib
import json
from pathlib import Path

from rag.chunker import chunk_text
from rag.embedder import embed_texts
from rag.loaders import SUPPORTED_EXTENSIONS, load_document_text
from rag.logging_config import get_logger
from rag.vector_store import VectorStore, build_vector_store, load_vector_store

log = get_logger(__name__)

_CHUNKS_FILENAME = "chunks.json"
_METADATA_FILENAME = "metadata.json"
_MANIFEST_FILENAME = "manifest.json"


# ── Cache key ──────────────────────────────────────────────────────────────────


def _corpus_fingerprint(
    doc_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    embed_backend: str,
    backend: str,
) -> str:
    """
    Hashes file contents + ingestion config into a stable cache key.
    Any change to a document's content, or to how it would be chunked/
    embedded/indexed, produces a different fingerprint — so a cache hit
    guarantees the persisted artifacts are equivalent to a fresh run.

    Hashes raw bytes, not extracted text, so this works unchanged for
    binary formats (PDF) as well as plaintext.
    """
    hasher = hashlib.sha256()
    for filepath in doc_files:  # already sorted by caller
        hasher.update(filepath.name.encode("utf-8"))
        hasher.update(filepath.read_bytes())
    hasher.update(
        f"|chunk_size={chunk_size}|overlap={chunk_overlap}"
        f"|embed_model={embed_model}|embed_backend={embed_backend}"
        f"|backend={backend}".encode()
    )
    return hasher.hexdigest()[:16]


def _load_from_cache(cache_path: Path, backend: str) -> tuple[list[str], list[dict[str, str]], VectorStore]:
    chunks = json.loads((cache_path / _CHUNKS_FILENAME).read_text(encoding="utf-8"))
    metadata = json.loads((cache_path / _METADATA_FILENAME).read_text(encoding="utf-8"))
    store = load_vector_store(cache_path, backend=backend)
    return chunks, metadata, store


def _save_to_cache(
    cache_path: Path,
    fingerprint: str,
    chunks: list[str],
    metadata: list[dict[str, str]],
    store: VectorStore,
    backend: str,
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / _CHUNKS_FILENAME).write_text(json.dumps(chunks), encoding="utf-8")
    (cache_path / _METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    (cache_path / _MANIFEST_FILENAME).write_text(
        json.dumps({"fingerprint": fingerprint, "backend": backend, "n_chunks": len(chunks)}),
        encoding="utf-8",
    )
    store.save(cache_path)


# ── Public API ─────────────────────────────────────────────────────────────────


def ingest(
    data_dir: Path,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    embed_model: str = None,
    embed_backend: str = None,
    backend: str = "numpy",
    cache_dir: Path | None = None,
) -> tuple[list[str], list[dict[str, str]], VectorStore]:
    """
    Walks data_dir, loads every supported document (.txt, .pdf — see
    rag/loaders.py), chunks each independently, and builds a VectorStore
    over the embedded chunks.

    Each chunk is tagged with its source filename, maintaining a strictly
    parallel relationship between chunks, metadata, and the vector store's
    index order.

    If `cache_dir` is provided and a cache entry matching the current corpus
    + config already exists, ingestion loads chunks/metadata/index from disk
    and skips chunking, text extraction, and embedding entirely.

    Args:
        data_dir:      Directory containing documents (rag.loaders.SUPPORTED_EXTENSIONS).
        chunk_size:    Approximate character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks to preserve
                       cross-boundary semantic context.
        embed_model:   Model identifier passed to rag.embedder.embed_texts().
                       Defaults to that backend's own default model when None.
        embed_backend: "ollama" or "local" — see rag/embedder.py. Defaults to
                       the EMBED_BACKEND environment variable when None.
        backend:       Vector store backend — "numpy" (default) or "faiss".
        cache_dir:     Optional directory to persist/load a cached corpus index.
                       Pass None (default) to always re-embed from scratch.

    Returns:
        chunks (List[str]):              All chunk texts across all documents.
        metadata (List[Dict[str, str]]): Parallel list; each entry is
                                          {"source": filename}.
        vector_store (VectorStore):      Built (or loaded) index over the
                                          corpus embeddings, ready for
                                          rag.retriever.retrieve().

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no
                            supported documents.
        ImportError:       If a .pdf file is present but pypdf isn't installed.
        ConnectionError:   If embed_backend="ollama" and Ollama is unreachable
                            (only on a cache miss — a cache hit needs no embedding call).
    """
    log.info("Ingestion started — scanning %s", data_dir)
    doc_files = sorted(path for ext in SUPPORTED_EXTENSIONS for path in Path(data_dir).glob(f"*{ext}"))
    if not doc_files:
        raise FileNotFoundError(
            f"No supported documents ({', '.join(SUPPORTED_EXTENSIONS)}) found in "
            f"{data_dir}. Add documents to the data/ directory before running."
        )
    log.info("Found %d document(s) to ingest", len(doc_files))

    fingerprint = _corpus_fingerprint(
        doc_files, chunk_size, chunk_overlap, embed_model, embed_backend, backend
    )
    cache_path = (cache_dir / fingerprint) if cache_dir is not None else None

    if cache_path is not None and (cache_path / _MANIFEST_FILENAME).exists():
        log.info("Cache hit — loading corpus from %s (no re-embedding)", cache_path)
        return _load_from_cache(cache_path, backend)

    if cache_path is not None:
        log.info("Cache miss — embedding corpus (fingerprint=%s)", fingerprint)

    all_chunks: list[str] = []
    all_metadata: list[dict[str, str]] = []

    for filepath in doc_files:
        text = load_document_text(filepath)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        source = filepath.name

        all_chunks.extend(chunks)
        all_metadata.extend({"source": source} for _ in chunks)

        log.debug("%s → %d chunks", source, len(chunks))

    log.info(
        "Chunking complete — %d total chunks from %d document(s)",
        len(all_chunks),
        len(doc_files),
    )

    log.info(
        "Embedding corpus — backend='%s' model='%s' …",
        embed_backend or "<embedder default>",
        embed_model or "<embedder default>",
    )
    corpus_embeddings = embed_texts(all_chunks, model=embed_model, backend=embed_backend)
    vector_store = build_vector_store(corpus_embeddings, backend=backend)
    log.info(
        "Ingestion complete — %d vectors indexed (backend=%s)",
        len(vector_store),
        backend,
    )

    if cache_path is not None:
        _save_to_cache(cache_path, fingerprint, all_chunks, all_metadata, vector_store, backend)
        log.info("Corpus cached → %s", cache_path)

    return all_chunks, all_metadata, vector_store
