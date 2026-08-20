"""
eval/benchmark_scale.py
------------------------
Measures actual behavior at corpus scale — hundreds of documents, not the
4-file demo corpus — instead of asserting a scale claim without evidence.

Generates a synthetic corpus into a temp directory (never touches data/ or
the real golden-set cache), ingests it once serially (EMBED_CONCURRENCY=1)
and once concurrently, and measures retrieval latency percentiles over a
sample of queries against the resulting index. Writes a timestamped JSON
report to eval/results/ (gitignored, same as eval/run_eval.py) and prints
a summary.

Requires a running Ollama instance with nomic-embed-text pulled — same
requirement as eval/run_eval.py.

Usage:
    python -m eval.benchmark_scale
    python -m eval.benchmark_scale --docs 500 --skip-serial
"""

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from rag.embedder import embed_texts
from rag.ingestion import ingest
from rag.logging_config import get_logger
from rag.retriever import retrieve

log = get_logger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# ── Synthetic corpus generation ─────────────────────────────────────────────
# Templated but genuinely varied text (randomized numbers/names/section
# order per document) so chunking and embedding operate on realistic-shaped
# input rather than degenerate repeated strings.

_TOPICS = ["expense_policy", "vendor_management", "capital_planning", "risk_controls", "payroll_processing"]

_SENTENCE_TEMPLATES = [
    "Section {n}: {topic} requests above ${amount:,} require approval from the {role}.",
    "All {topic} submissions must be filed within {days} business days of the triggering event.",
    "The {role} is responsible for reviewing {topic} exceptions on a {freq} basis.",
    "Departments exceeding their {topic} allocation by more than {pct}% must file a variance report.",
    "{topic} documentation is retained for a minimum of {years} years per the records policy.",
    "Any change to {topic} procedures requires sign-off from both Finance and the {role}.",
    "The {topic} committee meets {freq} to review pending requests above the ${amount:,} threshold.",
    "Non-compliance with {topic} controls is escalated to the {role} within {days} business days.",
]

_ROLES = ["CFO", "Controller", "Finance Business Partner", "Audit Committee", "Chief Compliance Officer"]
_FREQS = ["monthly", "quarterly", "weekly", "semi-annually"]


def _generate_document(rng: random.Random, doc_index: int) -> str:
    topic = _TOPICS[doc_index % len(_TOPICS)]
    n_sentences = rng.randint(15, 30)
    lines = [f"{topic.replace('_', ' ').title()} Policy — Document {doc_index}", ""]
    for i in range(n_sentences):
        template = rng.choice(_SENTENCE_TEMPLATES)
        lines.append(
            template.format(
                n=i + 1,
                topic=topic.replace("_", " "),
                amount=rng.choice([5000, 10000, 25000, 50000, 100000]),
                role=rng.choice(_ROLES),
                days=rng.choice([3, 5, 10, 15]),
                freq=rng.choice(_FREQS),
                pct=rng.choice([5, 10, 15, 20]),
                years=rng.choice([3, 5, 7]),
            )
        )
    return "\n\n".join(lines)


def generate_corpus(target_dir: Path, n_docs: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_docs):
        (target_dir / f"doc_{i:04d}.txt").write_text(_generate_document(rng, i), encoding="utf-8")


# ── Benchmark ────────────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


def run(n_docs: int, n_queries: int, skip_serial: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="rag_scale_bench_") as tmp:
        corpus_dir = Path(tmp) / "corpus"
        log.info("Generating %d synthetic documents…", n_docs)
        generate_corpus(corpus_dir, n_docs)

        report: dict = {"n_docs": n_docs}

        # A small subset for the serial baseline — full-corpus serial at
        # hundreds of docs would take a very long time and isn't needed to
        # show the concurrency delta; the ratio holds at any sample size
        # since each embedding call is independent and roughly equal cost.
        if not skip_serial:
            import os

            os.environ["EMBED_CONCURRENCY"] = "1"
            import importlib

            import rag.embedder as embedder_module

            importlib.reload(embedder_module)

            serial_subset = Path(tmp) / "serial_subset"
            generate_corpus(serial_subset, min(30, n_docs))
            t0 = time.perf_counter()
            chunks, _, _ = ingest(
                data_dir=serial_subset, embed_backend="ollama", embed_model="nomic-embed-text", cache_dir=None
            )
            serial_elapsed = time.perf_counter() - t0
            report["serial_baseline"] = {
                "n_docs": min(30, n_docs),
                "n_chunks": len(chunks),
                "elapsed_sec": round(serial_elapsed, 2),
                "chunks_per_sec": round(len(chunks) / serial_elapsed, 2),
            }
            log.info(
                "Serial baseline: %d chunks in %.2fs (%.2f chunks/sec)",
                len(chunks),
                serial_elapsed,
                len(chunks) / serial_elapsed,
            )

            os.environ["EMBED_CONCURRENCY"] = "8"
            importlib.reload(embedder_module)

        log.info("Ingesting full corpus (%d docs) with concurrent embedding…", n_docs)
        t0 = time.perf_counter()
        chunks, metadata, vector_store = ingest(
            data_dir=corpus_dir, embed_backend="ollama", embed_model="nomic-embed-text", cache_dir=None
        )
        full_elapsed = time.perf_counter() - t0
        report["full_ingest"] = {
            "n_docs": n_docs,
            "n_chunks": len(chunks),
            "elapsed_sec": round(full_elapsed, 2),
            "chunks_per_sec": round(len(chunks) / full_elapsed, 2),
            "embedding_matrix_mb": round(
                len(vector_store) * 768 * 4 / 1_000_000, 2
            ),  # nomic-embed-text: 768-dim float32
        }
        log.info(
            "Full ingest: %d chunks in %.2fs (%.2f chunks/sec)",
            len(chunks),
            full_elapsed,
            len(chunks) / full_elapsed,
        )

        if not skip_serial and "serial_baseline" in report:
            speedup = report["serial_baseline"]["chunks_per_sec"] and (
                report["full_ingest"]["chunks_per_sec"] / report["serial_baseline"]["chunks_per_sec"]
            )
            report["concurrency_speedup_factor"] = round(speedup, 2)

        # Retrieval latency at scale — real queries against the real index,
        # not a mock.
        log.info("Measuring retrieval latency over %d queries…", n_queries)
        sample_queries = [
            f"What is the approval threshold for {_TOPICS[i % len(_TOPICS)].replace('_', ' ')}?"
            for i in range(n_queries)
        ]
        # Measured separately on purpose: "query embedding" (a network round
        # trip to Ollama) and "vector search" (pure in-process NumPy
        # cosine similarity) have completely different cost profiles and
        # scale differently — collapsing them into one "retrieval latency"
        # number would hide which one actually dominates, and at small/
        # medium corpus sizes the embedding call dominates by 1-2 orders of
        # magnitude, making pure search look artificially slow by comparison
        # if reported together.
        embed_latencies_ms = []
        search_latencies_ms = []
        for query in sample_queries:
            t0 = time.perf_counter()
            query_embedding = embed_texts([query])[0]
            embed_latencies_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            retrieve(
                query_embedding=query_embedding,
                vector_store=vector_store,
                chunks=chunks,
                metadata=metadata,
                top_k=5,
            )
            search_latencies_ms.append((time.perf_counter() - t0) * 1000)

        def _stats(values: list[float]) -> dict:
            return {
                "mean": round(statistics.mean(values), 3),
                "p50": round(_percentile(values, 0.50), 3),
                "p95": round(_percentile(values, 0.95), 3),
                "p99": round(_percentile(values, 0.99), 3),
                "max": round(max(values), 3),
            }

        report["query_embedding_latency_ms"] = {"n_queries": n_queries, **_stats(embed_latencies_ms)}
        report["vector_search_latency_ms"] = {"n_queries": n_queries, **_stats(search_latencies_ms)}
        log.info(
            "Query embedding latency — p50=%.1fms p95=%.1fms | Vector search latency — p50=%.3fms p95=%.3fms",
            report["query_embedding_latency_ms"]["p50"],
            report["query_embedding_latency_ms"]["p95"],
            report["vector_search_latency_ms"]["p50"],
            report["vector_search_latency_ms"]["p95"],
        )

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-scale benchmark")
    parser.add_argument("--docs", type=int, default=300, help="number of synthetic documents to generate")
    parser.add_argument("--queries", type=int, default=50, help="number of retrieval-latency sample queries")
    parser.add_argument("--skip-serial", action="store_true", help="skip the serial-baseline comparison")
    args = parser.parse_args()

    try:
        report = run(n_docs=args.docs, n_queries=args.queries, skip_serial=args.skip_serial)
    except ConnectionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(report, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"scale_benchmark_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":  # pragma: no cover — manual benchmark script, not part of the test suite
    main()
