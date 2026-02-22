"""
validator/json_validator.py
---------------------------
Lightweight, air-gapped output schema enforcement layer designed for
deterministic enterprise workflows.

Defines the canonical RAGResponse TypedDict and validates pipeline output
against it before returning responses to callers. Raises a typed
ValidationError on any schema violation — no silent failures.

Each source entry carries both the retrieved text and the originating
document filename, enabling cross-document attribution and auditability.
"""

import json
from typing import Any, Dict, List

from typing_extensions import TypedDict

from rag.logging_config import get_logger

log = get_logger(__name__)


# ── Schema definition ──────────────────────────────────────────────────────────

class SourceEntry(TypedDict):
    """A single retrieved passage with source attribution."""
    text:   str   # the chunk text returned by the retriever
    source: str   # originating document filename (e.g. "financial_policy.txt")


class RAGResponse(TypedDict):
    """Canonical output contract for the Enterprise RAG pipeline."""
    query:   str              # original user question
    answer:  str              # LLM-generated, context-grounded answer
    sources: List[SourceEntry]  # top-k retrieved passages with source metadata
    model:   str              # Ollama generation model used


# ── Custom exception ───────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when a RAGResponse fails schema validation."""


# ── Validators ─────────────────────────────────────────────────────────────────

def validate_json_string(raw: str) -> Dict[str, Any]:
    """
    Parses a JSON string and returns the decoded dict.

    Args:
        raw: A JSON-encoded string.

    Returns:
        Decoded Python dict.

    Raises:
        ValidationError: If the string is not valid JSON.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc


def validate(response: Dict[str, Any]) -> RAGResponse:
    """
    Validates a dict against the RAGResponse schema.

    Checks:
      - Required keys are present: query, answer, sources, model
      - query, answer, model are non-empty strings
      - sources is a non-empty list of dicts, each with 'text' and 'source' string fields

    Args:
        response: Dict to validate (typically the raw pipeline output).

    Returns:
        The same dict cast as a typed RAGResponse.

    Raises:
        ValidationError: If any field is missing, wrong type, or blank.
    """
    required_keys = {"query", "answer", "sources", "model"}
    missing = required_keys - response.keys()
    if missing:
        log.error("Validation failed — missing keys: %s", missing)
        raise ValidationError(f"RAGResponse missing required keys: {missing}")

    for key in ("query", "answer", "model"):
        if not isinstance(response[key], str) or not response[key].strip():
            log.error("Validation failed — field '%s' is empty or wrong type", key)
            raise ValidationError(
                f"RAGResponse field '{key}' must be a non-empty string."
            )

    if not isinstance(response["sources"], list) or not response["sources"]:
        log.error("Validation failed — 'sources' is empty or not a list")
        raise ValidationError("RAGResponse 'sources' must be a non-empty list.")

    for i, entry in enumerate(response["sources"]):
        if not isinstance(entry, dict):
            log.error("Validation failed — sources[%d] is not a dict", i)
            raise ValidationError(
                f"RAGResponse sources[{i}] must be a dict, got {type(entry).__name__}."
            )
        for field in ("text", "source"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                log.error(
                    "Validation failed — sources[%d]['%s'] missing or empty", i, field
                )
                raise ValidationError(
                    f"RAGResponse sources[{i}]['{field}'] must be a non-empty string."
                )

    log.info(
        "Validation succeeded — query='%.60s…' sources=%d",
        response.get("query", ""), len(response["sources"]),
    )
    return RAGResponse(**response)  # type: ignore[return-value]
