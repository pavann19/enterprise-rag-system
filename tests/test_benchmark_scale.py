"""
Tests for eval/benchmark_scale.py. ingest()/embed_texts()/retrieve() are
mocked, so these need no live Ollama instance and run in milliseconds, not
minutes — the real thing is a manual benchmark you run yourself (see its
docstring), not part of this automated suite.
"""

import json
import sys

import numpy as np
import pytest

import eval.benchmark_scale as benchmark_module


class _FakeVectorStore:
    def __init__(self, n=50):
        self._n = n

    def __len__(self):
        return self._n


@pytest.fixture
def mocked_pipeline(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "ingest",
        lambda **kwargs: (["chunk"] * 50, [{"source": "doc_0000.txt"}] * 50, _FakeVectorStore(50)),
    )
    monkeypatch.setattr(benchmark_module, "embed_texts", lambda texts, **kwargs: np.array([[0.1, 0.2]]))
    monkeypatch.setattr(
        benchmark_module,
        "retrieve",
        lambda **kwargs: [{"text": "chunk", "score": 0.9, "source": "doc_0000.txt"}],
    )


# ── Synthetic corpus generation (real, no mocking — pure/local) ────────────


def test_generate_document_is_non_empty_and_varied():
    import random

    rng = random.Random(1)
    doc_a = benchmark_module._generate_document(rng, 0)
    doc_b = benchmark_module._generate_document(rng, 1)
    assert len(doc_a) > 100
    assert doc_a != doc_b  # different rng draws -> different content


def test_generate_document_is_deterministic_for_a_given_seed():
    import random

    doc_a = benchmark_module._generate_document(random.Random(7), 3)
    doc_b = benchmark_module._generate_document(random.Random(7), 3)
    assert doc_a == doc_b


def test_generate_corpus_writes_n_files(tmp_path):
    target = tmp_path / "corpus"
    benchmark_module.generate_corpus(target, n_docs=5)
    files = sorted(target.glob("*.txt"))
    assert len(files) == 5
    assert all(f.stat().st_size > 0 for f in files)


def test_generate_corpus_is_reproducible_for_the_same_seed(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    benchmark_module.generate_corpus(dir_a, n_docs=3, seed=99)
    benchmark_module.generate_corpus(dir_b, n_docs=3, seed=99)
    for name in ["doc_0000.txt", "doc_0001.txt", "doc_0002.txt"]:
        assert (dir_a / name).read_text() == (dir_b / name).read_text()


# ── _percentile ──────────────────────────────────────────────────────────────


def test_percentile_p50_of_sorted_list():
    assert benchmark_module._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0


def test_percentile_p0_is_minimum():
    assert benchmark_module._percentile([5.0, 1.0, 3.0], 0.0) == 1.0


def test_percentile_clamps_at_the_last_value():
    assert benchmark_module._percentile([1.0, 2.0, 3.0], 0.99) == 3.0


# ── run() ────────────────────────────────────────────────────────────────────


def test_run_skip_serial_report_structure(mocked_pipeline):
    report = benchmark_module.run(n_docs=5, n_queries=3, skip_serial=True)
    assert report["n_docs"] == 5
    assert "serial_baseline" not in report
    assert "concurrency_speedup_factor" not in report
    assert report["full_ingest"]["n_chunks"] == 50
    assert report["query_embedding_latency_ms"]["n_queries"] == 3
    assert report["vector_search_latency_ms"]["n_queries"] == 3


def test_run_with_serial_baseline_includes_speedup_factor(mocked_pipeline):
    report = benchmark_module.run(n_docs=5, n_queries=2, skip_serial=False)
    assert "serial_baseline" in report
    assert "concurrency_speedup_factor" in report


def test_run_propagates_connection_error(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("ollama down")),
    )
    with pytest.raises(ConnectionError):
        benchmark_module.run(n_docs=5, n_queries=1, skip_serial=True)


# ── main() ───────────────────────────────────────────────────────────────────


def test_main_writes_report_and_prints_summary(mocked_pipeline, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(benchmark_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark_scale.py", "--docs", "5", "--queries", "2", "--skip-serial"])

    benchmark_module.main()

    written = list(tmp_path.glob("scale_benchmark_*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["n_docs"] == 5

    captured = capsys.readouterr()
    assert "Full report written to" in captured.out


def test_main_exits_with_error_on_connection_error(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("ollama down")),
    )
    monkeypatch.setattr(sys, "argv", ["benchmark_scale.py", "--docs", "5", "--skip-serial"])

    with pytest.raises(SystemExit) as exc_info:
        benchmark_module.main()

    assert exc_info.value.code == 1
    assert "ollama down" in capsys.readouterr().err
