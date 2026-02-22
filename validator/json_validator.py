"""
validator/json_validator.py
---------------------------
Enforces a structured output schema on every RAG pipeline response.

The schema is defined as a TypedDict so it is both runtime-checkable
and IDE/mypy-visible as a type annotation.
"""

import json
from typing import Any, List, Tuple

# ── Output schema ──────────────────────────────────────────────────────────────

# Canonical structure of a validated RAG response.
# Use this TypedDict as the return-type annotation in callers.
from typing import TypedDict


class RAGResponse(TypedDict):
    """Structured output produced by the RAG pipeline."""
    query:   str         # The original user question
    answer:  str         # The LLM-generated answer
    sources: List[str]   # Top-k retrieved passages used as context
    model:   str         # Ollama generation model that produced the answer


# Runtime field-level schema (field name → expected Python type)
_SCHEMA: dict = {
    "query":   str,
    "answer":  str,
    "sources": list,
    "model":   str,
}

# ── Validation helpers ─────────────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when a RAG response fails schema validation."""


def validate(data: Any) -> RAGResponse:
    """
    Validates a dict against the RAGResponse schema and returns it typed.

    Args:
        data: Object to validate (must be a dict).

    Returns:
        The same dict, typed as RAGResponse.

    Raises:
        ValidationError: With a human-readable message on the first failure.
    """
    if not isinstance(data, dict):
        raise ValidationError(
            f"Response must be a dict, got {type(data).__name__}."
        )

    # Check all required fields exist and have the correct type
    for field, expected_type in _SCHEMA.items():
        if field not in data:
            raise ValidationError(f"Missing required field: '{field}'.")
        if not isinstance(data[field], expected_type):
            actual = type(data[field]).__name__
            raise ValidationError(
                f"Field '{field}' must be {expected_type.__name__}, got {actual}."
            )

    # Check non-empty string fields
    for field in ("query", "answer"):
        if not data[field].strip():
            raise ValidationError(f"Field '{field}' must not be blank.")

    return data  # type: ignore[return-value]


def validate_json_string(json_str: str) -> RAGResponse:
    """
    Parses a raw JSON string and validates it as a RAGResponse.

    Args:
        json_str: Raw JSON string.

    Returns:
        Parsed and validated RAGResponse dict.

    Raises:
        ValidationError: If the string is invalid JSON or fails schema checks.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc

    return validate(data)
