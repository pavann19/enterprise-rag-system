"""
eval/load_test.py
-------------------
Concurrent-request load test against a *running* service/api.py instance —
real HTTP requests, real concurrency, not a mock. Measures throughput and
latency percentiles under N concurrent clients hitting POST /query.

This does not start the server itself — point it at one you've already
started:

    uvicorn service.api:app --host 0.0.0.0 --port 8000 &
    python -m eval.load_test --url http://localhost:8000 --concurrency 10 --requests 50

Requires the target service to actually be able to serve /query (corpus
loaded, inference backend reachable) — a fast local model (e.g.
GEN_MODEL=qwen2.5:1.5b) is recommended so the load test measures the
service's own overhead rather than being dominated entirely by generation
time on a large model.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


def _one_request(base_url: str, query: str, timeout: float) -> dict:
    body = json.dumps({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, OSError) as exc:
        return {"ok": False, "error": str(exc), "latency_ms": (time.perf_counter() - t0) * 1000}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"ok": status == 200, "status": status, "latency_ms": elapsed_ms}


def run(base_url: str, concurrency: int, n_requests: int, timeout: float) -> dict:
    queries = [
        "What is the approval threshold for capital expenditures?",
        "What are the standard payment terms for vendors?",
        "What framework does the internal control environment align with?",
    ]
    requests_args = [queries[i % len(queries)] for i in range(n_requests)]

    results = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, base_url, q, timeout) for q in requests_args]
        for future in as_completed(futures):
            results.append(future.result())
    wall_elapsed = time.perf_counter() - wall_start

    successes = [r for r in results if r.get("ok")]
    failures = [r for r in results if not r.get("ok")]
    latencies = [r["latency_ms"] for r in successes]

    report = {
        "base_url": base_url,
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_success": len(successes),
        "n_failure": len(failures),
        "wall_elapsed_sec": round(wall_elapsed, 2),
        "throughput_req_per_sec": round(len(successes) / wall_elapsed, 2) if wall_elapsed > 0 else 0,
    }
    if latencies:
        report["latency_ms"] = {
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(_percentile(latencies, 0.50), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "p99": round(_percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1),
        }
    if failures:
        report["failure_examples"] = [f.get("error", f"HTTP {f.get('status')}") for f in failures[:5]]

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test against a running service/api.py")
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the running service")
    parser.add_argument("--concurrency", type=int, default=10, help="number of concurrent clients")
    parser.add_argument("--requests", type=int, default=50, help="total number of requests to send")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    args = parser.parse_args()

    # Fail fast with a clear message if the target isn't even up, rather
    # than burning the whole run on connection errors.
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=5) as resp:
            resp.read()
    except (urllib.error.URLError, OSError) as exc:
        print(f"[ERROR] {args.url} is not reachable: {exc}", file=sys.stderr)
        sys.exit(1)

    report = run(args.url, args.concurrency, args.requests, args.timeout)
    print(json.dumps(report, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"load_test_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":  # pragma: no cover — manual load-test script, not part of the test suite
    main()
