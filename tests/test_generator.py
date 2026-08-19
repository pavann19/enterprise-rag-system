import pytest

import rag.generator as generator_module
from rag.generator import generate_answer, _build_prompt


def test_build_prompt_includes_query_and_passages():
    prompt = _build_prompt("What is X?", ["Passage one text.", "Passage two text."])
    assert "What is X?" in prompt
    assert "Passage one text." in prompt
    assert "Passage two text." in prompt
    assert "[Passage 1]" in prompt
    assert "[Passage 2]" in prompt


def test_build_prompt_numbers_passages_in_order():
    prompt = _build_prompt("q", ["first", "second", "third"])
    assert prompt.index("[Passage 1]") < prompt.index("[Passage 2]") < prompt.index("[Passage 3]")


def test_build_prompt_strips_passage_whitespace():
    prompt = _build_prompt("q", ["  padded text  \n"])
    assert "padded text" in prompt
    assert "  padded text  " not in prompt


def test_build_prompt_handles_single_passage():
    prompt = _build_prompt("q", ["only one"])
    assert "[Passage 1]" in prompt
    assert "[Passage 2]" not in prompt


def test_build_prompt_includes_grounding_instruction():
    prompt = _build_prompt("q", ["ctx"])
    assert "ONLY" in prompt
    assert "does not contain enough information" in prompt


def test_build_prompt_handles_special_characters_in_query():
    prompt = _build_prompt('What about "quotes" & <tags>?', ["ctx"])
    assert 'What about "quotes" & <tags>?' in prompt


def test_generate_answer_rejects_empty_query():
    with pytest.raises(ValueError):
        generate_answer("", ["some context"])


def test_generate_answer_rejects_whitespace_only_query():
    with pytest.raises(ValueError):
        generate_answer("   \n\t  ", ["some context"])


def test_generate_answer_rejects_empty_passages():
    with pytest.raises(ValueError):
        generate_answer("a question", [])


def test_generate_answer_propagates_connection_error_when_ollama_down(monkeypatch):
    # Point at a guaranteed-unreachable port (nothing listens on :1) rather
    # than relying on the environment's real OLLAMA_HOST being free — that
    # assumption broke once a real Ollama instance was actually running on
    # localhost:11434 during Docker verification, turning this into a slow
    # 60s timeout instead of an immediate, deterministic connection refusal.
    monkeypatch.setattr(generator_module, "GENERATE_URL", "http://localhost:1/api/generate")
    with pytest.raises(ConnectionError):
        generate_answer("a question", ["some context"])
