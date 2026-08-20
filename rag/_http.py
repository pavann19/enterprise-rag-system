"""
rag/_http.py
------------
Centralized HTTP transport layer for all Ollama API communication.

Provides `ollama_post()` as the single point of control for timeouts,
error handling, and connection governance across the entire pipeline.
All public modules delegate Ollama calls here — no urllib boilerplate
is duplicated in business-logic modules.

Air-gapped by design: every request stays on the private network reachable
at OLLAMA_HOST — no third-party endpoint is ever contacted. That host
defaults to localhost for a bare-metal/venv run, and is overridden to the
"ollama" service name when running under docker-compose (see
docker-compose.yml) — same code, no branching, just a different address on
the same private network.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Retries only cover transient failures (connection refused because the
# server is mid-restart, a request that timed out under load) — not
# malformed requests or a server that's simply not there at all, which
# would just fail the same way N times slower. Kept small: a caller
# blocked in a retry loop for 30+ seconds is often worse than a fast,
# clear failure the caller can act on (retry later, fall back, alert).
_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
_RETRY_BACKOFF_BASE_SEC = 0.5


def ollama_post(
    url: str,
    payload: dict[str, Any],
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Sends a JSON POST request to an Ollama endpoint and returns the response.

    Retries transient connection failures up to OLLAMA_MAX_RETRIES times
    (default 2, i.e. 3 attempts total) with exponential backoff (0.5s, 1s,
    ...) before giving up. A malformed-JSON response is not retried — that
    indicates a real server-side problem a retry won't fix.

    Args:
        url:     Full Ollama endpoint URL.
        payload: Request body as a Python dict (will be JSON-encoded).
        timeout: Socket timeout in seconds, per attempt.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        ConnectionError: If Ollama is still unreachable after all retries.
        RuntimeError:    If the response cannot be decoded as JSON.
    """
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # urllib.error.URLError covers failures while *opening* the connection
    # (refused, DNS, etc.) — urllib wraps socket-level OSErrors into it at
    # that stage. It does NOT cover failures while *reading the response
    # body* after the connection opened successfully: a connection reset
    # mid-stream surfaces as a bare OSError (concretely,
    # http.client.RemoteDisconnected, a ConnectionResetError subclass),
    # raised from inside the `with urlopen(...)` block, past the point
    # urllib does any wrapping. Found for real running eval/benchmark_scale.py
    # at ~1400 chunks with 8 concurrent requests against a local Ollama
    # instance — catching only URLError missed it entirely and the retry
    # logic never engaged.
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_BASE_SEC * (2**attempt))
                continue

        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Could not parse Ollama response as JSON: {exc}") from exc

    raise ConnectionError(
        f"Ollama is not reachable at {url} after {_MAX_RETRIES + 1} attempt(s).\n"
        "  → Make sure Ollama is running:  ollama serve\n"
        f"  Original error: {last_error}"
    ) from last_error
