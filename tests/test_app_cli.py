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
def test_cli_does_not_hang_on_unreachable_backend():
    # Regression guard: this must fail fast (connection refused), not wait
    # out a long socket timeout — same class of bug fixed in test_generator.py.
    result = _run_app([], timeout=15)
    assert result.returncode == 1
