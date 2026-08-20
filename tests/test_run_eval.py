"""
Tests for eval/run_eval.py — previously completely untested (0% coverage).

ingest()/embed_texts()/retrieve() are all mocked out, so these tests need
no live Ollama instance. What's under test is run_eval's own logic: report
structure, metric aggregation wiring, CLI arg parsing, and error handling.
"""

import json
import sys

import numpy as np
import pytest

import eval.run_eval as run_eval_module

FAKE_GOLDEN_SET = [
    {"id": "q1", "query": "question one", "expected_source": "a.txt"},
    {"id": "q2", "query": "question two", "expected_source": "b.txt"},
]


class _FakeVectorStore:
    def __len__(self):
        return 2


@pytest.fixture
def mocked_pipeline(monkeypatch):
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        run_eval_module,
        "ingest",
        lambda **kwargs: (
            ["chunk a", "chunk b"],
            [{"source": "a.txt"}, {"source": "b.txt"}],
            _FakeVectorStore(),
        ),
    )
    monkeypatch.setattr(run_eval_module, "embed_texts", lambda texts, model, backend: np.array([[0.1, 0.2]]))

    monkeypatch.setattr(
        run_eval_module,
        "retrieve",
        lambda **kwargs: [
            {"text": "chunk a", "score": 0.9, "source": "a.txt"},
            {"text": "chunk b", "score": 0.1, "source": "b.txt"},
        ],
    )
    return {}


def test_run_returns_config_summary_and_per_query(mocked_pipeline):
    report = run_eval_module.run()
    assert set(report.keys()) == {"config", "summary", "per_query"}
    assert len(report["per_query"]) == 2
    assert report["summary"]["n_queries"] == 2


def test_run_per_query_entries_include_reciprocal_rank(mocked_pipeline):
    report = run_eval_module.run()
    for entry in report["per_query"]:
        assert "reciprocal_rank" in entry
        assert 0.0 <= entry["reciprocal_rank"] <= 1.0


def test_run_config_reflects_arguments(mocked_pipeline):
    report = run_eval_module.run(top_k_retrieved=7, k_values=[1, 2], backend="faiss")
    assert report["config"]["top_k_retrieved"] == 7
    assert report["config"]["backend"] == "faiss"


def test_run_summary_matches_metrics_module_output(mocked_pipeline):
    report = run_eval_module.run(k_values=[1])
    # every query retrieved "a.txt" at rank 1 in this fixture; only q1 expects
    # a.txt, so hit_rate@1 should be 0.5 (1 of 2 queries correct)
    assert report["summary"]["hit_rate@1"] == pytest.approx(0.5)


def test_run_raises_on_missing_data_dir(monkeypatch):
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        run_eval_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("no docs")),
    )
    with pytest.raises(FileNotFoundError):
        run_eval_module.run()


def test_run_propagates_connection_error(monkeypatch):
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        run_eval_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("no ollama")),
    )
    with pytest.raises(ConnectionError):
        run_eval_module.run()


# ── main() — CLI entry point ─────────────────────────────────────────────────


def test_main_writes_timestamped_report_and_prints_summary(mocked_pipeline, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_eval_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])

    run_eval_module.main()

    written = list(tmp_path.glob("eval_*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["summary"]["n_queries"] == 2

    captured = capsys.readouterr()
    assert "n_queries" in captured.out
    assert "Full report written to" in captured.out


def test_main_respects_k_and_backend_cli_args(mocked_pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--k", "1", "2", "--backend", "faiss"])

    run_eval_module.main()

    written = list(tmp_path.glob("eval_*.json"))
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["config"]["backend"] == "faiss"
    assert "hit_rate@1" in report["summary"]
    assert "hit_rate@2" in report["summary"]
    assert "hit_rate@5" not in report["summary"]


def test_main_exits_with_error_on_missing_data_dir(monkeypatch, capsys):
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        run_eval_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("no docs found")),
    )
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 1
    assert "no docs found" in capsys.readouterr().err


def test_main_exits_with_error_on_connection_error(monkeypatch, capsys):
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        run_eval_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("ollama unreachable")),
    )
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 1
    assert "ollama unreachable" in capsys.readouterr().err


def test_load_golden_set_reads_real_file():
    queries = run_eval_module.load_golden_set()
    assert len(queries) == 15
    assert all({"id", "query", "expected_source"} <= set(q.keys()) for q in queries)
