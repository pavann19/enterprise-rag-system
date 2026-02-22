"""
service/api.py
--------------
Lightweight FastAPI service layer for the Enterprise RAG pipeline.

Exposes a single REST endpoint:
    POST /query   { "query": str }  →  RAGResponse

The corpus is loaded once at server startup via the lifespan context manager
and held in memory for the lifetime of the process. All inference remains
local via Ollama — this service adds zero external dependencies beyond the
existing pipeline modules.

Run with:
    uvicorn service.api:app --host 0.0.0.0 --port 8000

Or from the project root:
    python -m uvicorn service.api:app --reload

The existing CLI entry point (app.py) and Streamlit UI are not affected.
"""

from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import (
    query_pipeline,
    DATA_DIR,
    EMBED_MODEL,
    GEN_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from rag.ingestion import ingest
from validator.json_validator import ValidationError


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Input schema for the /query endpoint."""
    query: str


# ── Corpus state (loaded once at startup) ─────────────────────────────────────

class _CorpusState:
    """In-process singleton holding the ingested corpus."""
    chunks:   List[str]
    metadata: List[Dict[str, str]]
    embeddings: np.ndarray

    def __init__(self):
        self.chunks    = []
        self.metadata  = []
        self.embeddings = np.empty((0,), dtype=np.float32)


_corpus = _CorpusState()


# ── Lifespan (replaces @app.on_event) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the document corpus once at startup; release on shutdown."""
    print("[startup] Ingesting document corpus…")
    try:
        chunks, metadata, embeddings = ingest(
            data_dir      = DATA_DIR,
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            embed_model   = EMBED_MODEL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"[startup] Data directory error: {exc}") from exc
    except ConnectionError as exc:
        raise RuntimeError(f"[startup] Ollama unreachable: {exc}") from exc

    _corpus.chunks    = chunks
    _corpus.metadata  = metadata
    _corpus.embeddings = embeddings
    print(f"[startup] Corpus ready — {len(chunks)} chunks loaded.")
    yield
    print("[shutdown] Corpus released.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Enterprise RAG API",
    description = (
        "Air-gapped Retrieval-Augmented Generation service. "
        "All inference is local via Ollama. No external API calls."
    ),
    version = "1.0.0",
    lifespan = lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health_check():
    """Liveness probe — confirms the service is running and corpus is loaded."""
    return {
        "status":      "ok",
        "corpus_size": len(_corpus.chunks),
        "embed_model": EMBED_MODEL,
        "gen_model":   GEN_MODEL,
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
        503 Service Unavailable:  if Ollama becomes unreachable at query time
        500 Internal Server Error: if the pipeline output fails schema validation
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty.")

    try:
        response = query_pipeline(
            query             = request.query,
            chunks            = _corpus.chunks,
            metadata          = _corpus.metadata,
            corpus_embeddings = _corpus.embeddings,
            gen_model         = GEN_MODEL,
            embed_model       = EMBED_MODEL,
            top_k             = TOP_K,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response
