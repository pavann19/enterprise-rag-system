import pytest

from rag.generator import generate_answer, _build_prompt


def test_build_prompt_includes_query_and_passages():
    prompt = _build_prompt("What is X?", ["Passage one text.", "Passage two text."])
    assert "What is X?" in prompt
    assert "Passage one text." in prompt
    assert "Passage two text." in prompt
    assert "[Passage 1]" in prompt
    assert "[Passage 2]" in prompt


def test_generate_answer_rejects_empty_query():
    with pytest.raises(ValueError):
        generate_answer("", ["some context"])


def test_generate_answer_rejects_empty_passages():
    with pytest.raises(ValueError):
        generate_answer("a question", [])


def test_generate_answer_propagates_connection_error_when_ollama_down():
    # No Ollama server is running in the test environment, and DEFAULT_GEN_MODEL's
    # host (localhost:11434) is not reachable — this exercises the real failure path
    # without requiring a live Ollama instance.
    with pytest.raises(ConnectionError):
        generate_answer("a question", ["some context"])
