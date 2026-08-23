"""
Tests for eval/judge_eval.py. query_pipeline()/generate_answer()/ingest()
are mocked out — the "judge" LLM call is generate_answer() reused (see the
module docstring), so mocking it lets these tests run without any live
Ollama/cloud backend, same pattern as tests/test_run_eval.py.
"""

import json
import sys

import pytest

import eval.judge_eval as judge_eval_module

FAKE_GOLDEN_SET = [
    {"id": "q1", "query": "question one", "expected_source": "a.txt"},
    {"id": "q2", "query": "question two", "expected_source": "b.txt"},
]


class _FakeVectorStore:
    def __len__(self):
        return 2


def _fake_query_pipeline(**kwargs):
    return {
        "query": kwargs["query"],
        "answer": "a grounded answer",
        "sources": [{"text": "the source passage", "source": "a.txt"}],
        "model": kwargs.get("gen_model", "mistral"),
    }


@pytest.fixture
def mocked_pipeline(monkeypatch):
    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module,
        "ingest",
        lambda **kwargs: (
            ["chunk a", "chunk b"],
            [{"source": "a.txt"}, {"source": "b.txt"}],
            _FakeVectorStore(),
        ),
    )
    monkeypatch.setattr(judge_eval_module, "query_pipeline", _fake_query_pipeline)
    monkeypatch.setattr(
        judge_eval_module, "generate_answer", lambda **kwargs: '{"faithfulness": 5, "relevancy": 4}'
    )
    return {}


# ── _parse_judge_response ────────────────────────────────────────────────────


def test_parse_judge_response_extracts_valid_scores():
    scores = judge_eval_module._parse_judge_response('{"faithfulness": 4, "relevancy": 5}')
    assert scores == {"faithfulness": 4, "relevancy": 5}


def test_parse_judge_response_extracts_json_amid_extra_text():
    scores = judge_eval_module._parse_judge_response('Sure! {"faithfulness": 3, "relevancy": 2} Done.')
    assert scores == {"faithfulness": 3, "relevancy": 2}


def test_parse_judge_response_raises_when_no_json_object():
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        judge_eval_module._parse_judge_response("no json here at all")


def test_parse_judge_response_raises_when_scores_out_of_range():
    with pytest.raises(ValueError, match="out of the 1-5 range"):
        judge_eval_module._parse_judge_response('{"faithfulness": 9, "relevancy": 3}')


# ── judge_answer ─────────────────────────────────────────────────────────────


def test_judge_answer_calls_generate_answer_and_parses_result(monkeypatch):
    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return '{"faithfulness": 5, "relevancy": 5}'

    monkeypatch.setattr(judge_eval_module, "generate_answer", _fake_generate)
    scores = judge_eval_module.judge_answer(
        question="what is X?",
        answer="X is Y",
        passage="X is Y per policy.",
        gen_backend="groq",
        gen_model="m",
    )
    assert scores == {"faithfulness": 5, "relevancy": 5}
    assert captured["backend"] == "groq"
    assert captured["model"] == "m"
    assert "what is X?" in captured["passages"][0]


# ── run() ─────────────────────────────────────────────────────────────────────


def test_run_returns_config_summary_and_per_query(mocked_pipeline):
    report = judge_eval_module.run()
    assert set(report.keys()) == {"config", "summary", "per_query"}
    assert len(report["per_query"]) == 2
    assert report["summary"]["n_queries"] == 2
    assert report["summary"]["n_scored"] == 2
    assert report["summary"]["mean_faithfulness"] == 5.0
    assert report["summary"]["mean_relevancy"] == 4.0


def test_run_per_query_entries_include_answer_and_scores(mocked_pipeline):
    report = judge_eval_module.run()
    for entry in report["per_query"]:
        assert entry["answer"] == "a grounded answer"
        assert entry["faithfulness"] == 5
        assert entry["relevancy"] == 4


def test_run_handles_unparseable_judge_response(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(judge_eval_module, "generate_answer", lambda **kwargs: "not valid json at all")
    report = judge_eval_module.run()
    assert report["summary"]["n_scored"] == 0
    assert report["summary"]["n_unparseable"] == 2
    assert report["summary"]["mean_faithfulness"] is None
    assert report["summary"]["mean_relevancy"] is None
    assert all(q["faithfulness"] is None for q in report["per_query"])


def test_run_config_reflects_arguments(mocked_pipeline):
    report = judge_eval_module.run(gen_backend="groq", gen_model="custom-model")
    assert report["config"]["gen_backend"] == "groq"
    assert report["config"]["gen_model"] == "custom-model"


def test_run_raises_on_missing_data_dir(monkeypatch):
    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module, "ingest", lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("no docs"))
    )
    with pytest.raises(FileNotFoundError):
        judge_eval_module.run()


def test_run_propagates_connection_error(monkeypatch):
    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module, "ingest", lambda **kwargs: (_ for _ in ()).throw(ConnectionError("no ollama"))
    )
    with pytest.raises(ConnectionError):
        judge_eval_module.run()


def test_run_propagates_validation_error(monkeypatch):
    from validator.json_validator import ValidationError

    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module,
        "ingest",
        lambda **kwargs: (["chunk a"], [{"source": "a.txt"}], _FakeVectorStore()),
    )
    monkeypatch.setattr(
        judge_eval_module,
        "query_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(ValidationError("bad schema")),
    )
    with pytest.raises(ValidationError):
        judge_eval_module.run()


# ── main() — CLI entry point ─────────────────────────────────────────────────


def test_main_writes_timestamped_report_and_prints_summary(mocked_pipeline, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(judge_eval_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["judge_eval.py"])

    judge_eval_module.main()

    written = list(tmp_path.glob("judge_eval_*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["summary"]["n_queries"] == 2

    captured = capsys.readouterr()
    assert "mean_faithfulness" in captured.out
    assert "Full report written to" in captured.out


def test_main_respects_backend_and_model_cli_args(mocked_pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(judge_eval_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["judge_eval.py", "--gen-backend", "groq", "--gen-model", "llama-3.1-8b-instant"]
    )

    judge_eval_module.main()

    written = list(tmp_path.glob("judge_eval_*.json"))
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["config"]["gen_backend"] == "groq"
    assert report["config"]["gen_model"] == "llama-3.1-8b-instant"


def test_main_exits_with_error_on_missing_data_dir(monkeypatch, capsys):
    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("no docs found")),
    )
    monkeypatch.setattr(sys, "argv", ["judge_eval.py"])

    with pytest.raises(SystemExit) as exc_info:
        judge_eval_module.main()

    assert exc_info.value.code == 1
    assert "no docs found" in capsys.readouterr().err


def test_main_exits_with_error_on_connection_error(monkeypatch, capsys):
    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module,
        "ingest",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("ollama unreachable")),
    )
    monkeypatch.setattr(sys, "argv", ["judge_eval.py"])

    with pytest.raises(SystemExit) as exc_info:
        judge_eval_module.main()

    assert exc_info.value.code == 1
    assert "ollama unreachable" in capsys.readouterr().err


def test_main_exits_with_error_on_validation_error(monkeypatch, capsys):
    from validator.json_validator import ValidationError

    monkeypatch.setattr(judge_eval_module, "_load_golden_set", lambda: FAKE_GOLDEN_SET)
    monkeypatch.setattr(
        judge_eval_module,
        "ingest",
        lambda **kwargs: (["chunk a"], [{"source": "a.txt"}], _FakeVectorStore()),
    )
    monkeypatch.setattr(
        judge_eval_module,
        "query_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(ValidationError("bad schema")),
    )
    monkeypatch.setattr(sys, "argv", ["judge_eval.py"])

    with pytest.raises(SystemExit) as exc_info:
        judge_eval_module.main()

    assert exc_info.value.code == 2
    assert "bad schema" in capsys.readouterr().err


def test_load_golden_set_reads_real_file():
    queries = judge_eval_module._load_golden_set()
    assert len(queries) == 15
    assert all({"id", "query", "expected_source"} <= set(q.keys()) for q in queries)
