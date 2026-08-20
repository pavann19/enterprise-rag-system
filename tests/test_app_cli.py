"""
Subprocess smoke tests for app.py's `if __name__ == "__main__":` block —
the one piece of this codebase that can't be exercised via direct import
(argv parsing, sys.exit() codes, and stderr output all need a real process
boundary to test honestly).

OLLAMA_HOST is pinned to a guaranteed-unreachable port so these don't
depend on ambient environment state — the same fragility bug fixed in
tests/test_generator.py (a real Ollama container from Docker verification
left something listening on the real default port during that session).

The happy path (successful ingestion + generation + the final
json.dumps(response) print) is intentionally not covered here: it needs a
real, reachable generation backend, which is exactly the kind of live
dependency this project's test suite otherwise avoids everywhere else
(see eval/run_eval.py's own "not yet run — needs live Ollama" note). The
error-handling wiring covered below is what's safely testable without one.
"""

import subprocess
import sys

import pytest

UNREACHABLE_ENV = {"OLLAMA_HOST": "http://localhost:1"}


def _run_app(args, extra_env=None, timeout=30):
    import os

    env = {**os.environ, **UNREACHABLE_ENV, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "app.py", *args],
        env=env,
        capture_output=True,
        text=True,
        # app.py reconfigures its own stdout/stderr to UTF-8 (see its
        # __main__ block) precisely so its em-dash/arrow log output
        # doesn't crash on Windows — but subprocess.run(text=True) decodes
        # the captured bytes using *this* process's preferred encoding by
        # default, which is the Windows console codepage (cp1252), not
        # UTF-8. Without pinning it here, the child now succeeds while the
        # test harness capturing it fails instead.
        encoding="utf-8",
        timeout=timeout,
    )


def test_cli_exits_1_and_prints_error_when_ollama_unreachable():
    result = _run_app([])
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "not reachable" in result.stderr.lower()


def test_cli_accepts_no_query_argument_without_crashing():
    # Ingestion fails (unreachable backend) before the query section ever
    # prints, so the default-query text itself isn't observable from here —
    # what's actually being verified is that argv parsing with zero extra
    # args doesn't raise an IndexError or similar before reaching ingestion.
    result = _run_app([])
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_cli_accepts_a_query_argument_without_crashing():
    result = _run_app(["What is the audit control framework?"])
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_cli_prints_ingestion_header_before_failing():
    result = _run_app([])
    assert "INGESTION" in result.stdout


@pytest.mark.timeout(30)
def test_cli_does_not_hang_indefinitely_on_unreachable_backend():
    # Regression guard: this must fail within a bounded time (connection
    # refused + retries), not hang out an unbounded/very long socket
    # timeout — same class of bug originally fixed in test_generator.py.
    #
    # 30s, not 15s: rag/_http.py retries each transient connection failure
    # (2 retries, ~1.5s of backoff each) — correct and desirable for a
    # single flaky call, but it compounds across the real demo corpus's 86
    # chunks embedded at concurrency=8 (~11 sequential batches), which is
    # exactly what this test exercises during a full Ollama outage. ~16.5s
    # worst case observed; 30s leaves real margin without being "no bound
    # at all."
    result = _run_app([], timeout=30)
    assert result.returncode == 1
