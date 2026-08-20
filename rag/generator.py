"""
rag/generator.py
----------------
Context-grounded answer generation with three interchangeable backends:

  "ollama"    (default) — calls a local Ollama model. Fully local, no API
                          key, no data leaves the machine. Requires an
                          Ollama server reachable at OLLAMA_HOST.

  "anthropic" (optional) — calls the Claude API. Requires the `anthropic`
                          package plus an ANTHROPIC_API_KEY.

  "groq"      (optional) — calls Groq's hosted-inference API (Llama models
                          served on Groq's own chips — free tier, low
                          latency). Requires the `groq` package plus a
                          GROQ_API_KEY.

Both cloud backends exist for one specific reason: a public hosted demo
(Streamlit Community Cloud, HF Spaces, etc.) has no way to run a background
Ollama server, so a fully local deployment can't be reached by someone
clicking a link. Either trades the air-gapped guarantee for a working
public demo — neither is the default, and picking one is a deliberate,
visible choice (an env var), never an implicit fallback.

Injects top-k retrieved passages into a structured RAG prompt and produces
an answer grounded strictly in the provided context, via whichever backend
is selected.

Generation is the sole responsibility of this module.
It expects pre-retrieved context — it does NOT perform retrieval or embedding.
"""

import os
from typing import List

from rag._http          import OLLAMA_HOST, ollama_post
from rag.logging_config import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
GENERATE_URL            = f"{OLLAMA_HOST}/api/generate"
DEFAULT_OLLAMA_MODEL    = "mistral"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GROQ_MODEL      = "openai/gpt-oss-20b"
GEN_BACKEND             = os.environ.get("GEN_BACKEND", "ollama")
# ──────────────────────────────────────────────────────────────────────────────


def _build_prompt(query: str, passages: List[str]) -> str:
    """
    Constructs a RAG prompt with a system instruction, numbered passages,
    and the user's question.

    Args:
        query:    The user's question.
        passages: Retrieved text passages to use as grounding context.

    Returns:
        Formatted prompt string.
    """
    # Format each passage with a numbered label
    formatted_passages = "\n\n".join(
        f"[Passage {i + 1}]\n{passage.strip()}"
        for i, passage in enumerate(passages)
    )

    return (
        # System instruction
        "You are a precise assistant. "
        "Answer the question using ONLY the passages below. "
        "Do not use prior knowledge. "
        "If the answer is not present, say: "
        "'The context does not contain enough information.'\n\n"
        # Retrieved context
        f"Context:\n{formatted_passages}\n\n"
        # User question
        f"Question: {query}\n\n"
        "Answer:"
    )


def _generate_ollama(prompt: str, model: str) -> str:
    try:
        response = ollama_post(
            GENERATE_URL,
            {"model": model, "prompt": prompt, "stream": False},
        )
    except ConnectionError as exc:
        log.error("Generation failed — Ollama unreachable: %s", exc)
        raise
    return response.get("response", "").strip()


def _generate_anthropic(prompt: str, model: str) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "backend='anthropic' requires the anthropic package.\n"
            "  → pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "backend='anthropic' requires the ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIConnectionError as exc:
        raise ConnectionError(f"Anthropic API unreachable: {exc}") from exc

    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


def _generate_groq(prompt: str, model: str) -> str:
    try:
        import groq
    except ImportError as exc:
        raise ImportError(
            "backend='groq' requires the groq package.\n"
            "  → pip install groq"
        ) from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "backend='groq' requires the GROQ_API_KEY environment variable."
        )

    client = groq.Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except groq.APIConnectionError as exc:
        raise ConnectionError(f"Groq API unreachable: {exc}") from exc

    return (response.choices[0].message.content or "").strip()


_BACKENDS = {
    "ollama":    (_generate_ollama, DEFAULT_OLLAMA_MODEL),
    "anthropic": (_generate_anthropic, DEFAULT_ANTHROPIC_MODEL),
    "groq":      (_generate_groq, DEFAULT_GROQ_MODEL),
}


def generate_answer(
    query: str,
    passages: List[str],
    model: str = None,
    backend: str = None,
) -> str:
    """
    Produces a context-grounded answer via the selected backend.

    Args:
        query:    The user's question.
        passages: Top-k retrieved passages (context for the LLM).
        model:    Model identifier for the chosen backend. Defaults to that
                  backend's own DEFAULT_*_MODEL constant when None.
        backend:  "ollama", "anthropic", or "groq". Defaults to the
                  GEN_BACKEND environment variable (itself defaulting to
                  "ollama").

    Returns:
        The generated answer as a plain string.

    Raises:
        ValueError:      If query is blank, passages is empty, or backend
                          is unrecognized.
        ImportError:     If the selected cloud backend's SDK isn't installed.
        RuntimeError:    If the selected cloud backend's API key env var is unset.
        ConnectionError: If the selected backend's endpoint is unreachable.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if not passages:
        raise ValueError("passages must not be empty.")

    backend = backend or GEN_BACKEND
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown generation backend '{backend}'. Available: {sorted(_BACKENDS)}")

    generate_fn, default_model = _BACKENDS[backend]
    model = model or default_model

    log.info("Generating answer — backend='%s' model='%s' query='%.60s…'", backend, model, query)
    prompt = _build_prompt(query, passages)
    answer = generate_fn(prompt, model)

    if not answer:
        log.warning("Generation returned empty response for model='%s'", model)
        return "[The model returned an empty response.]"

    log.info("Generation succeeded — answer length %d chars", len(answer))
    return answer
