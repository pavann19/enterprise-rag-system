import pytest

from eval.metrics import (
    QueryResult,
    hit_at_k,
    hit_rate_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
    summarize,
)


def _result(expected, retrieved):
    return QueryResult(query_id="q", expected_source=expected, retrieved_sources=retrieved)


# ── hit_at_k ───────────────────────────────────────────────────────────────


def test_hit_at_k_true_when_expected_in_top_k():
    r = _result("a.txt", ["b.txt", "a.txt", "c.txt"])
    assert hit_at_k(r, k=2) is True


def test_hit_at_k_false_when_expected_outside_top_k():
    r = _result("a.txt", ["b.txt", "c.txt", "a.txt"])
    assert hit_at_k(r, k=2) is False


def test_hit_at_k_false_when_never_retrieved():
    r = _result("a.txt", ["b.txt", "c.txt"])
    assert hit_at_k(r, k=5) is False


# ── precision_at_k ───────────────────────────────────────────────────────────


def test_precision_at_k_full_match():
    r = _result("a.txt", ["a.txt", "a.txt", "a.txt"])
    assert precision_at_k(r, k=3) == pytest.approx(1.0)


def test_precision_at_k_partial_match():
    r = _result("a.txt", ["a.txt", "b.txt", "c.txt", "a.txt"])
    assert precision_at_k(r, k=4) == pytest.approx(0.5)


def test_precision_at_k_zero_when_missing_slots_count_as_misses():
    # only 1 result retrieved but k=3 requested -> 2 phantom misses
    r = _result("a.txt", ["a.txt"])
    assert precision_at_k(r, k=3) == pytest.approx(1 / 3)


def test_precision_at_k_rejects_non_positive_k():
    r = _result("a.txt", ["a.txt"])
    with pytest.raises(ValueError):
        precision_at_k(r, k=0)


# ── reciprocal_rank ────────────────────────────────────────────────────────


def test_reciprocal_rank_first_position():
    r = _result("a.txt", ["a.txt", "b.txt"])
    assert reciprocal_rank(r) == pytest.approx(1.0)


def test_reciprocal_rank_third_position():
    r = _result("a.txt", ["b.txt", "c.txt", "a.txt"])
    assert reciprocal_rank(r) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_never_found():
    r = _result("a.txt", ["b.txt", "c.txt"])
    assert reciprocal_rank(r) == 0.0


# ── Aggregates ─────────────────────────────────────────────────────────────


def test_mean_reciprocal_rank_averages_across_queries():
    results = [
        _result("a.txt", ["a.txt"]),  # rr = 1.0
        _result("a.txt", ["b.txt", "a.txt"]),  # rr = 0.5
    ]
    assert mean_reciprocal_rank(results) == pytest.approx(0.75)


def test_mean_reciprocal_rank_rejects_empty_list():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])


def test_hit_rate_at_k_across_queries():
    results = [
        _result("a.txt", ["a.txt", "b.txt"]),  # hit @1
        _result("a.txt", ["b.txt", "a.txt"]),  # miss @1
    ]
    assert hit_rate_at_k(results, k=1) == pytest.approx(0.5)
    assert hit_rate_at_k(results, k=2) == pytest.approx(1.0)


def test_mean_precision_at_k_across_queries():
    results = [
        _result("a.txt", ["a.txt", "a.txt"]),  # precision@2 = 1.0
        _result("a.txt", ["b.txt", "c.txt"]),  # precision@2 = 0.0
    ]
    assert mean_precision_at_k(results, k=2) == pytest.approx(0.5)


def test_hit_rate_at_k_rejects_empty_list():
    with pytest.raises(ValueError):
        hit_rate_at_k([], k=1)


def test_mean_precision_at_k_rejects_empty_list():
    with pytest.raises(ValueError):
        mean_precision_at_k([], k=1)


def test_summarize_produces_expected_keys():
    results = [_result("a.txt", ["a.txt", "b.txt"])]
    report = summarize(results, k_values=[1, 2])
    assert report["n_queries"] == 1
    assert "mrr" in report
    assert report["hit_rate@1"] == pytest.approx(1.0)
    assert report["hit_rate@2"] == pytest.approx(1.0)
    assert report["precision@1"] == pytest.approx(1.0)
    assert report["precision@2"] == pytest.approx(0.5)


def test_summarize_rejects_empty_results():
    with pytest.raises(ValueError):
        summarize([])
