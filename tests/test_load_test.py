"""
Tests for eval/load_test.py. The HTTP layer (_one_request) is mocked for
run()/main() tests — this validates the aggregation/reporting logic, not
network behavior, which the manual run against a live service covers.
"""

import json
import sys
import urllib.error

import pytest

import eval.load_test as load_test_module


def test_percentile_p50():
    assert load_test_module._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0


def test_percentile_clamps_at_max():
    assert load_test_module._percentile([1.0, 2.0], 0.99) == 2.0


def test_run_aggregates_successful_requests(monkeypatch):
    monkeypatch.setattr(
        load_test_module,
        "_one_request",
        lambda base_url, query, timeout: {"ok": True, "status": 200, "latency_ms": 42.0},
    )
    report = load_test_module.run("http://x", concurrency=2, n_requests=5, timeout=5)
    assert report["n_success"] == 5
    assert report["n_failure"] == 0
    assert report["latency_ms"]["mean"] == 42.0
    assert "failure_examples" not in report


def test_run_reports_failures_separately(monkeypatch):
    def _fake(base_url, query, timeout):
        return {"ok": False, "error": "connection refused", "latency_ms": 5.0}

    monkeypatch.setattr(load_test_module, "_one_request", _fake)
    report = load_test_module.run("http://x", concurrency=2, n_requests=3, timeout=5)
    assert report["n_success"] == 0
    assert report["n_failure"] == 3
    assert "latency_ms" not in report  # no successful latencies to aggregate
    assert len(report["failure_examples"]) == 3


def test_run_computes_throughput(monkeypatch):
    monkeypatch.setattr(
        load_test_module,
        "_one_request",
        lambda base_url, query, timeout: {"ok": True, "status": 200, "latency_ms": 10.0},
    )
    report = load_test_module.run("http://x", concurrency=5, n_requests=10, timeout=5)
    assert report["throughput_req_per_sec"] > 0
    assert report["n_requests"] == 10


def test_one_request_returns_ok_on_success(monkeypatch):
    from unittest.mock import MagicMock

    fake_response = MagicMock()
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    fake_response.read.return_value = b'{"answer": "ok"}'
    fake_response.status = 200

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: fake_response)
    result = load_test_module._one_request("http://x", "q", timeout=5)
    assert result["ok"] is True
    assert result["status"] == 200
    assert result["latency_ms"] >= 0


def test_one_request_handles_http_error(monkeypatch):
    class _FakeHTTPError(urllib.error.HTTPError):
        pass

    def _raise(request, timeout=None):
        raise urllib.error.HTTPError("http://x/query", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    result = load_test_module._one_request("http://x", "q", timeout=5)
    assert result["ok"] is False
    assert result["status"] == 429


def test_one_request_handles_connection_error(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    result = load_test_module._one_request("http://x", "q", timeout=5)
    assert result["ok"] is False
    assert "error" in result


def test_main_exits_1_when_target_unreachable(monkeypatch, capsys):
    def _raise(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    monkeypatch.setattr(sys, "argv", ["load_test.py", "--url", "http://nope:9999"])

    with pytest.raises(SystemExit) as exc_info:
        load_test_module.main()

    assert exc_info.value.code == 1
    assert "not reachable" in capsys.readouterr().err.lower()


def test_main_writes_report_when_target_reachable(monkeypatch, tmp_path, capsys):
    from unittest.mock import MagicMock

    health_response = MagicMock()
    health_response.__enter__.return_value = health_response
    health_response.__exit__.return_value = False
    health_response.read.return_value = b'{"status": "ok"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: health_response)
    monkeypatch.setattr(
        load_test_module,
        "_one_request",
        lambda base_url, query, timeout: {"ok": True, "status": 200, "latency_ms": 15.0},
    )
    monkeypatch.setattr(load_test_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["load_test.py", "--url", "http://x", "--requests", "3", "--concurrency", "1"]
    )

    load_test_module.main()

    written = list(tmp_path.glob("load_test_*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["n_success"] == 3
