"""
service/api.py
--------------
Lightweight FastAPI service layer for the Enterprise RAG pipeline.

Exposes a single REST endpoint:
    POST /query   { "query": str }  →  RAGResponse

The corpus is loaded once at server startup via the lifespan context manager
and held in memory for the lifetime of the process. Inference backend
(local Ollama vs. optional cloud) is whatever app.py's EMBED_BACKEND /
GEN_BACKEND resolve to — see rag/embedder.py and rag/generator.py.

Run with:
    uvicorn service.api:app --host 0.0.0.0 --port 8000

Or from the project root:
    python -m uvicorn service.api:app --reload

The existing CLI entry point (app.py) and Streamlit UI are not affected.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import (
    query_pipeline,
    DATA_DIR,
    CACHE_DIR,
    EMBED_MODEL,
    EMBED_BACKEND,
    GEN_MODEL,
    GEN_BACKEND,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    VECTOR_BACKEND,
)
from rag.ingestion      import ingest
from rag.logging_config import get_logger
from rag.vector_store   import VectorStore
from validator.json_validator import ValidationError

log = get_logger(__name__)


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Input schema for the /query endpoint."""
    query: str


# ── Corpus state (loaded once at startup) ─────────────────────────────────────

class _CorpusState:
    """In-process singleton holding the ingested corpus."""
    chunks:       List[str]
    metadata:     List[Dict[str, str]]
    vector_store: Optional[VectorStore]

    def __init__(self):
        self.chunks       = []
        self.metadata     = []
        self.vector_store = None


_corpus = _CorpusState()


# ── Lifespan (replaces @app.on_event) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the document corpus once at startup; release on shutdown."""
    log.info("Service startup — ingesting document corpus from %s", DATA_DIR)
    try:
        chunks, metadata, vector_store = ingest(
            data_dir      = DATA_DIR,
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            embed_model   = EMBED_MODEL,
            embed_backend = EMBED_BACKEND,
            backend       = VECTOR_BACKEND,
            cache_dir     = CACHE_DIR,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"[startup] Data directory error: {exc}") from exc
    except ConnectionError as exc:
        raise RuntimeError(f"[startup] embedding backend unreachable: {exc}") from exc

    _corpus.chunks       = chunks
    _corpus.metadata     = metadata
    _corpus.vector_store = vector_store
    log.info("Corpus ready — %d chunks from %d document(s)",
             len(chunks), len({m['source'] for m in metadata}))
    yield
    log.info("Service shutdown — corpus released")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Enterprise RAG API",
    description = (
        "Retrieval-Augmented Generation service. Local-only (Ollama) by "
        "default; see GET /health for the inference backends actually "
        "configured on this instance."
    ),
    version = "1.0.0",
    lifespan = lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health_check():
    """
    Liveness probe — confirms the service is running and the corpus is loaded.

    Does NOT trigger embeddings or LLM calls.
    """
    documents_loaded = len({m["source"] for m in _corpus.metadata})
    log.info("Health check — corpus_chunks=%d documents_loaded=%d",
             len(_corpus.chunks), documents_loaded)
    return {
        "status":            "ok",
        "embedding_backend": EMBED_BACKEND,
        "embedding_model":   EMBED_MODEL,
        "generation_backend": GEN_BACKEND,
        "generation_model":  GEN_MODEL,
        "documents_loaded":  documents_loaded,
    }


@app.post("/query", tags=["rag"])
def query(request: QueryRequest):
    """
    Run the full RAG pipeline for a single query.

    - Embeds the query using the configured embedding model
    - Retrieves the top-k semantically similar passages with source metadata
    - Generates a context-grounded answer via the configured generation model
    - Returns a validated RAGResponse object

    Raises:
        422 Unprocessable Entity: if the request body is malformed
        503 Service Unavailable:  if the inference backend is unreachable at query time
        500 Internal Server Error: if the pipeline output fails schema validation
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty.")

    log.info("POST /query — received query='%.80s…'", request.query)
    try:
        response = query_pipeline(
            query         = request.query,
            chunks        = _corpus.chunks,
            metadata      = _corpus.metadata,
            vector_store  = _corpus.vector_store,
            gen_model     = GEN_MODEL,
            gen_backend   = GEN_BACKEND,
            embed_model   = EMBED_MODEL,
            embed_backend = EMBED_BACKEND,
            top_k         = TOP_K,
        )
    except ConnectionError as exc:
        log.error("POST /query failed — inference backend unreachable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValidationError as exc:
        log.error("POST /query failed — validation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log.info("POST /query complete — answer length %d chars", len(response["answer"]))
    return response
