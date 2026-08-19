"""
eval/run_eval.py
-----------------
Retrieval-quality evaluation harness. Runs every query in golden_set.json
through the real ingestion + retrieval pipeline (embeddings via whichever
EMBED_BACKEND app.py resolves to) and scores the result against the
labeled expected source document.

This produces the first *measured* number in this repository — everything
else in the README is either a design decision or an indicative estimate.
Every run's report is timestamped and written to eval/results/ so results
are reproducible and comparable across changes (chunk size, embedding
model, backend) instead of being asserted from memory.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --k 1 3 5 --backend faiss

Requires the configured embedding backend to be reachable (embeddings
only — no generation call is made; this evaluates retrieval, not answer
quality). With the default EMBED_BACKEND="ollama" that means a running
Ollama instance. If it's not reachable, this exits with a clear error
rather than silently reporting zero scores.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_BACKEND,
    EMBED_MODEL,
    VECTOR_BACKEND,
)
from eval.metrics import QueryResult, summarize
from rag.embedder import embed_texts
from rag.ingestion import ingest
from rag.logging_config import get_logger
from rag.retriever import retrieve

log = get_logger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR     = Path(__file__).parent / "results"


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def run(
    top_k_retrieved: int = 5,
    k_values: list = (1, 3, 5),
    backend: str = VECTOR_BACKEND,
) -> dict:
    """
    Runs the golden set end-to-end and returns a full report dict:
    {"config": {...}, "summary": {...}, "per_query": [...]}
    """
    golden_queries = load_golden_set()
    log.info("Loaded %d golden queries", len(golden_queries))

    chunks, metadata, vector_store = ingest(
        data_dir      = DATA_DIR,
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        embed_model   = EMBED_MODEL,
        embed_backend = EMBED_BACKEND,
        backend       = backend,
        cache_dir     = CACHE_DIR,
    )

    per_query = []
    query_results = []

    for item in golden_queries:
        query_embedding = embed_texts([item["query"]], model=EMBED_MODEL, backend=EMBED_BACKEND)[0]
        hits = retrieve(
            query_embedding = query_embedding,
            vector_store    = vector_store,
            chunks          = chunks,
            metadata        = metadata,
            top_k           = top_k_retrieved,
        )
        retrieved_sources = [h["source"] for h in hits]

        result = QueryResult(
            query_id          = item["id"],
            expected_source   = item["expected_source"],
            retrieved_sources = retrieved_sources,
        )
        query_results.append(result)
        per_query.append({
            "id": item["id"],
            "query": item["query"],
            "expected_source": item["expected_source"],
            "retrieved_sources": retrieved_sources,
            "reciprocal_rank": round(
                next(
                    (1.0 / (i + 1) for i, s in enumerate(retrieved_sources)
                     if s == item["expected_source"]),
                    0.0,
                ), 4,
            ),
        })

    summary = summarize(query_results, k_values=k_values)

    return {
        "config": {
            "data_dir":      str(DATA_DIR),
            "embed_backend": EMBED_BACKEND,
            "embed_model":   EMBED_MODEL,
            "chunk_size":   CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "backend":      backend,
            "top_k_retrieved": top_k_retrieved,
        },
        "summary": summary,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation harness")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                         help="top-k cutoffs to report hit-rate/precision at")
    parser.add_argument("--top-k-retrieved", type=int, default=5,
                         help="how many passages retrieve() returns per query")
    parser.add_argument("--backend", default=VECTOR_BACKEND, choices=["numpy", "faiss", "qdrant"])
    args = parser.parse_args()

    try:
        report = run(top_k_retrieved=args.top_k_retrieved, k_values=args.k, backend=args.backend)
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(report["summary"], indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
