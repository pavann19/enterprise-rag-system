"""
rag/generator.py
----------------
Generates a grounded answer from a query and retrieved context passages
by calling a local Ollama generation model.

Generation is the only responsibility of this module.
It expects pre-retrieved context — it does NOT perform retrieval.

Prerequisite:
    ollama pull mistral    (or: llama3, phi3, gemma, etc.)
    ollama serve
"""

from typing import List

from rag._http import ollama_post

# ── Constants ──────────────────────────────────────────────────────────────────
GENERATE_URL      = "http://localhost:11434/api/generate"
DEFAULT_GEN_MODEL = "mistral"
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


def generate_answer(
    query: str,
    passages: List[str],
    model: str = DEFAULT_GEN_MODEL,
) -> str:
    """
    Calls Ollama to produce a context-grounded answer.

    Args:
        query:    The user's question.
        passages: Top-k retrieved passages (context for the LLM).
        model:    Ollama generation model name.

    Returns:
        The generated answer as a plain string.

    Raises:
        ValueError:      If query is blank or passages is empty.
        ConnectionError: If Ollama is unreachable (propagated from _http).
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if not passages:
        raise ValueError("passages must not be empty.")

    prompt   = _build_prompt(query, passages)
    response = ollama_post(
        GENERATE_URL,
        {"model": model, "prompt": prompt, "stream": False},
    )

    answer = response.get("response", "").strip()
    return answer or "[The model returned an empty response.]"
