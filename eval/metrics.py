"""
eval/metrics.py
----------------
Pure, dependency-free retrieval-quality metrics.

Deliberately scoped to *retrieval* quality (did the right document come
back), not answer-quality metrics like faithfulness or answer relevancy.
Those require an LLM-as-judge, which would mean either running a second
local model (extra latency, extra weights to pull) or calling a cloud API —
the latter directly contradicts this project's air-gapped design. Retrieval
metrics need no judge at all: they're plain set/rank arithmetic against a
known-correct label, so they're honest to compute and cheap to run.

Every function here is pure — no I/O, no Ollama, no filesystem — so it can
be unit-tested with synthetic data (see tests/test_eval_metrics.py) without
a live pipeline.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    """One golden-set query's retrieval outcome, ready for scoring."""

    query_id: str
    expected_source: str
    retrieved_sources: list[str]  # ranked, index 0 = top result


def hit_at_k(result: QueryResult, k: int) -> bool:
    """True if expected_source appears anywhere in the top-k retrieved sources."""
    return result.expected_source in result.retrieved_sources[:k]


def precision_at_k(result: QueryResult, k: int) -> float:
    """
    Fraction of the top-k retrieved passages whose source matches expected_source.

    Returns 0.0 if fewer than k results were retrieved (missing slots count
    as misses, not as excluded from the denominator).
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    top_k = result.retrieved_sources[:k]
    matches = sum(1 for source in top_k if source == result.expected_source)
    return matches / k


def reciprocal_rank(result: QueryResult) -> float:
    """1 / (rank of first correct hit, 1-indexed), or 0.0 if never retrieved."""
    for rank, source in enumerate(result.retrieved_sources, start=1):
        if source == result.expected_source:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(results: list[QueryResult]) -> float:
    if not results:
        raise ValueError("results must not be empty.")
    return sum(reciprocal_rank(r) for r in results) / len(results)


def hit_rate_at_k(results: list[QueryResult], k: int) -> float:
    if not results:
        raise ValueError("results must not be empty.")
    return sum(1 for r in results if hit_at_k(r, k)) / len(results)


def mean_precision_at_k(results: list[QueryResult], k: int) -> float:
    if not results:
        raise ValueError("results must not be empty.")
    return sum(precision_at_k(r, k) for r in results) / len(results)


def summarize(results: list[QueryResult], k_values: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """
    Aggregates all metrics across a golden set into a single report dict.

    Args:
        results:  One QueryResult per golden-set query.
        k_values: Which top-k cutoffs to report hit-rate/precision at.

    Returns:
        Flat dict, e.g. {"mrr": 0.83, "hit_rate@1": 0.6, "precision@1": 0.6, ...}
    """
    if not results:
        raise ValueError("results must not be empty.")
    report: dict[str, float] = {
        "n_queries": len(results),
        "mrr": mean_reciprocal_rank(results),
    }
    for k in k_values:
        report[f"hit_rate@{k}"] = hit_rate_at_k(results, k)
        report[f"precision@{k}"] = mean_precision_at_k(results, k)
    return report
