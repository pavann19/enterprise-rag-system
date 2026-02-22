"""
rag/_http.py
------------
Centralized HTTP transport layer for all Ollama API communication.

Provides `ollama_post()` as the single point of control for timeouts,
error handling, and connection governance across the entire pipeline.
All public modules delegate Ollama calls here — no urllib boilerplate
is duplicated in business-logic modules.

Air-gapped by design: all requests are directed to localhost (127.0.0.1).
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict


def ollama_post(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Sends a JSON POST request to an Ollama endpoint and returns the response.

    Args:
        url:     Full Ollama endpoint URL.
        payload: Request body as a Python dict (will be JSON-encoded).
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        ConnectionError: If Ollama is unreachable (server not running).
        RuntimeError:    If the response cannot be decoded as JSON.
    """
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ollama is not reachable at {url}.\n"
            "  → Make sure Ollama is running:  ollama serve\n"
            f"  Original error: {exc}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse Ollama response as JSON: {exc}") from exc
