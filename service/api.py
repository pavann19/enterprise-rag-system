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

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_BACKEND,
    EMBED_MODEL,
    GEN_BACKEND,
    GEN_MODEL,
    TOP_K,
    VECTOR_BACKEND,
    query_pipeline,
)
from rag._http import OLLAMA_HOST, ollama_post
from rag.embedder import embed_texts
from rag.generator import generate_answer_stream
from rag.ingestion import ingest
from rag.logging_config import get_logger
from rag.reranker import RERANK_ENABLED, rerank
from rag.retriever import retrieve
from rag.vector_store import VectorStore
from service.rate_limiter import (  # noqa: F401 — RateLimiter re-exported for tests
    RateLimiter,
    get_rate_limiter,
)
from validator.json_validator import ValidationError

log = get_logger(__name__)

# ── Rate limiting ────────────────────────────────────────────────────────────
# Applies to /query only — /health and /ready are cheap, no-inference reads
# meant to be polled freely by orchestrators/load balancers.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
_query_rate_limiter = get_rate_limiter(max_requests=RATE_LIMIT_PER_MINUTE, window_seconds=60)

# ── Auth ─────────────────────────────────────────────────────────────────────
# Optional API-key gate on /query. Unset by default (matches every other env
# var in this service) so a fresh clone still runs with zero config — set
# RAG_API_KEY once this instance is reachable beyond localhost. /health and
# /ready stay open on purpose: orchestrators/load balancers poll them
# without a key.
RAG_API_KEY = os.environ.get("RAG_API_KEY") or None
if RAG_API_KEY is None:
    log.warning(
        "RAG_API_KEY is not set — POST /query is unauthenticated. Set RAG_API_KEY before exposing this instance beyond localhost."
    )


# ── Request / Response models ──────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Input schema for the /query endpoint."""

    query: str


# ── Corpus state (loaded once at startup) ─────────────────────────────────────


class _CorpusState:
    """In-process singleton holding the ingested corpus."""

    chunks: list[str]
    metadata: list[dict[str, str]]
    vector_store: VectorStore | None

    def __init__(self):
        self.chunks = []
        self.metadata = []
        self.vector_store = None


_corpus = _CorpusState()


# ── Lifespan (replaces @app.on_event) ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the document corpus once at startup; release on shutdown."""
    log.info("Service startup — ingesting document corpus from %s", DATA_DIR)
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
    except FileNotFoundError as exc:
        raise RuntimeError(f"[startup] Data directory error: {exc}") from exc
    except ConnectionError as exc:
        raise RuntimeError(f"[startup] embedding backend unreachable: {exc}") from exc

    _corpus.chunks = chunks
    _corpus.metadata = metadata
    _corpus.vector_store = vector_store
    log.info(
        "Corpus ready — %d chunks from %d document(s)", len(chunks), len({m["source"] for m in metadata})
    )
    yield
    log.info("Service shutdown — corpus released")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Retrieval-Augmented Generation service. Local-only (Ollama) by "
        "default; see GET /health for the inference backends actually "
        "configured on this instance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# The Next.js frontend (web/) runs on a different origin (localhost:3000 in
# dev, its own Vercel domain in production) than this API — a browser blocks
# that cross-origin fetch by default without this. CORS_ALLOWED_ORIGINS is a
# comma-separated allowlist, empty by default (no cross-origin access, same
# as before this existed) rather than "*", since "*" combined with
# credentialed requests is a real CSRF-adjacent foot-gun.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )


# ── Request correlation ─────────────────────────────────────────────────────
# Every response carries an X-Request-ID (caller-supplied, if present — lets
# a caller correlate its own logs with ours — otherwise generated here) so a
# single request can be traced through logs across a multi-instance
# deployment without guessing which log line belongs to which call.


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "%s %s request_id=%s status=%d duration_ms=%.1f",
        request.method,
        request.url.path,
        request_id,
        response.status_code,
        duration_ms,
    )
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
def health_check():
    """
    Liveness probe — confirms the process is running and the corpus is loaded.

    Does NOT trigger embeddings or LLM calls, and does NOT check whether the
    inference backend is currently reachable — see GET /ready for that.
    """
    documents_loaded = len({m["source"] for m in _corpus.metadata})
    return {
        "status": "ok",
        "embedding_backend": EMBED_BACKEND,
        "embedding_model": EMBED_MODEL,
        "generation_backend": GEN_BACKEND,
        "generation_model": GEN_MODEL,
        "documents_loaded": documents_loaded,
    }


@app.get("/ready", tags=["ops"])
def readiness_check():
    """
    Readiness probe — checks whether the configured inference backend is
    actually reachable right now, not just whether the process is up.

    Distinct from /health on purpose: an orchestrator should keep routing
    traffic to a live-but-not-yet-ready instance's /health checks (don't
    restart it), while pulling it out of a load balancer's rotation based
    on /ready (don't send it requests it can't serve). Only checks Ollama
    reachability for backend="ollama" — the "local" embedding backend and
    the cloud generation backends (anthropic/groq) don't have a cheap,
    side-effect-free reachability check available, so they report ready
    once the corpus is loaded, same as /health.
    """
    if EMBED_BACKEND != "ollama" and GEN_BACKEND != "ollama":
        return {"status": "ready", "checked_ollama": False}

    try:
        ollama_post(f"{OLLAMA_HOST}/api/embeddings", {"model": EMBED_MODEL, "prompt": "readiness check"})
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {exc}") from exc

    return {"status": "ready", "checked_ollama": True}


@app.post("/query", tags=["rag"])
def query(request: QueryRequest, http_request: Request, x_api_key: str | None = Header(default=None)):
    """
    Run the full RAG pipeline for a single query.

    - Embeds the query using the configured embedding model
    - Retrieves the top-k semantically similar passages with source metadata
    - Generates a context-grounded answer via the configured generation model
    - Returns a validated RAGResponse object

    Requires header `X-API-Key` matching the RAG_API_KEY environment variable,
    if that variable is set (unset = unauthenticated, for local/dev use).

    Rate-limited to RATE_LIMIT_PER_MINUTE requests per client IP (default
    30/min) — this is the one endpoint that actually costs money/compute
    per call (embedding + generation), unlike /health and /ready.

    Raises:
        401 Unauthorized:        if RAG_API_KEY is set and the caller's key is missing/wrong
        422 Unprocessable Entity: if the request body is malformed
        429 Too Many Requests:    if the caller has exceeded the rate limit
        503 Service Unavailable:  if the inference backend is unreachable at query time
        500 Internal Server Error: if the pipeline output fails schema validation
    """
    if RAG_API_KEY is not None and x_api_key != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty.")

    client_key = http_request.client.host if http_request.client else "unknown"
    if not _query_rate_limiter.allow(client_key):
        retry_after = _query_rate_limiter.retry_after_seconds(client_key)
        log.warning("POST /query rate-limited — client=%s retry_after=%.1fs", client_key, retry_after)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min). Retry after {retry_after}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    request_id = getattr(http_request.state, "request_id", "unknown")
    log.info("POST /query — request_id=%s query='%.80s…'", request_id, request.query)
    try:
        response = query_pipeline(
            query=request.query,
            chunks=_corpus.chunks,
            metadata=_corpus.metadata,
            vector_store=_corpus.vector_store,
            gen_model=GEN_MODEL,
            gen_backend=GEN_BACKEND,
            embed_model=EMBED_MODEL,
            embed_backend=EMBED_BACKEND,
            top_k=TOP_K,
        )
    except ConnectionError as exc:
        log.error("POST /query failed — request_id=%s inference backend unreachable: %s", request_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValidationError as exc:
        log.error("POST /query failed — request_id=%s validation error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log.info("POST /query complete — request_id=%s answer_length=%d", request_id, len(response["answer"]))
    return response


def _check_auth_and_rate_limit(request: QueryRequest, http_request: Request, x_api_key: str | None) -> str:
    """Shared guard for /query and /query/stream. Returns the client key used for rate limiting."""
    if RAG_API_KEY is not None and x_api_key != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty.")

    client_key = http_request.client.host if http_request.client else "unknown"
    if not _query_rate_limiter.allow(client_key):
        retry_after = _query_rate_limiter.retry_after_seconds(client_key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min). Retry after {retry_after}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return client_key


@app.post("/query/stream", tags=["rag"])
def query_stream(request: QueryRequest, http_request: Request, x_api_key: str | None = Header(default=None)):
    """
    Same pipeline as POST /query, but streams the generated answer as
    Server-Sent Events instead of waiting for the full answer before
    responding — the same trade-off streamlit_app.py makes with
    st.write_stream, exposed over HTTP for any SSE-capable client.

    Retrieval happens eagerly (fast, and the client needs sources up front);
    only generation is streamed. Emits one `data: <token>` event per
    generated token, then a final `data: [DONE]` sentinel. Errors that occur
    after streaming has started (a mid-stream backend disconnect) are sent
    as a `data: [ERROR] <message>` event, since HTTP status/headers can no
    longer change once the body has started.
    """
    _check_auth_and_rate_limit(request, http_request, x_api_key)

    query_embedding = embed_texts([request.query], model=EMBED_MODEL, backend=EMBED_BACKEND)[0]
    results = retrieve(
        query_embedding=query_embedding,
        vector_store=_corpus.vector_store,
        chunks=_corpus.chunks,
        metadata=_corpus.metadata,
        top_k=TOP_K,
    )
    if RERANK_ENABLED:
        results = rerank(request.query, results)
    passages = [r["text"] for r in results]

    sources_payload = json.dumps([{"text": r["text"], "source": r["source"]} for r in results])

    def event_stream():
        yield f"data: [SOURCES] {sources_payload}\n\n"
        try:
            for token in generate_answer_stream(
                query=request.query, passages=passages, model=GEN_MODEL, backend=GEN_BACKEND
            ):
                yield f"data: {token}\n\n"
        except (ConnectionError, ValueError) as exc:
            yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
