import sys

import pytest

import rag.generator as generator_module
from rag.generator import generate_answer


def test_generate_answer_rejects_unknown_backend():
    with pytest.raises(ValueError):
        generate_answer("q", ["ctx"], backend="nonexistent")


def test_generate_answer_dispatches_to_ollama_backend_by_default():
    assert generator_module.GEN_BACKEND == "ollama"  # default, unless overridden in the environment


def test_generate_answer_explicit_backend_param_overrides_default(monkeypatch):
    calls = []

    def _fake_backend(prompt, model):
        calls.append((prompt, model))
        return "a generated answer"

    monkeypatch.setitem(
        generator_module._BACKENDS,
        "anthropic",
        (_fake_backend, generator_module.DEFAULT_ANTHROPIC_MODEL),
    )

    answer = generate_answer("what is X?", ["some context"], backend="anthropic")
    assert answer == "a generated answer"
    assert calls[0][1] == generator_module.DEFAULT_ANTHROPIC_MODEL
    assert "what is X?" in calls[0][0]


def test_generate_answer_returns_placeholder_on_empty_response(monkeypatch):
    monkeypatch.setitem(
        generator_module._BACKENDS,
        "anthropic",
        (lambda prompt, model: "", generator_module.DEFAULT_ANTHROPIC_MODEL),
    )
    answer = generate_answer("q", ["ctx"], backend="anthropic")
    assert "empty response" in answer.lower()


def test_anthropic_backend_raises_clear_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    with pytest.raises(ImportError, match="anthropic"):
        generate_answer("q", ["ctx"], backend="anthropic")


def test_anthropic_backend_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generate_answer("q", ["ctx"], backend="anthropic")
