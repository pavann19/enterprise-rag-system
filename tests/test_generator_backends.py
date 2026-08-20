import sys
from unittest.mock import MagicMock, patch

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
    # A fake module stands in for the real `anthropic` package so this test
    # is deterministic whether or not it's actually installed (it isn't in
    # CI, by design — see the module docstring) — the point here is the
    # api-key check, which runs after the import succeeds either way.
    monkeypatch.setitem(sys.modules, "anthropic", MagicMock())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generate_answer("q", ["ctx"], backend="anthropic")


def test_groq_backend_dispatches_with_default_model(monkeypatch):
    calls = []

    def _fake_backend(prompt, model):
        calls.append((prompt, model))
        return "a groq answer"

    monkeypatch.setitem(
        generator_module._BACKENDS, "groq", (_fake_backend, generator_module.DEFAULT_GROQ_MODEL)
    )

    answer = generate_answer("what is X?", ["some context"], backend="groq")
    assert answer == "a groq answer"
    assert calls[0][1] == generator_module.DEFAULT_GROQ_MODEL


def test_groq_backend_raises_clear_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    with pytest.raises(ImportError, match="groq"):
        generate_answer("q", ["ctx"], backend="groq")


def test_groq_backend_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", MagicMock())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        generate_answer("q", ["ctx"], backend="groq")


# ── Real backend bodies (client class mocked, not the whole backend
# function) — the tests above bypass _generate_anthropic/_generate_groq
# entirely by replacing their _BACKENDS entry, so they never actually
# exercise those functions' own client-construction/response-parsing logic.
# These do, via a fake module injected into sys.modules.


class _FakeAPIConnectionError(Exception):
    pass


def _fake_anthropic_module(create_side_effect=None, response_text="the anthropic answer"):
    fake_module = MagicMock()
    fake_module.APIConnectionError = _FakeAPIConnectionError

    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = response_text
    fake_response = MagicMock()
    fake_response.content = [fake_block]

    fake_client = MagicMock()
    if create_side_effect is not None:
        fake_client.messages.create.side_effect = create_side_effect
    else:
        fake_client.messages.create.return_value = fake_response
    fake_module.Anthropic.return_value = fake_client
    return fake_module, fake_client


def test_anthropic_body_parses_text_blocks_from_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    fake_module, fake_client = _fake_anthropic_module(response_text="grounded answer")

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        answer = generate_answer("q", ["ctx"], backend="anthropic", model="claude-haiku-4-5-20251001")

    assert answer == "grounded answer"
    fake_module.Anthropic.assert_called_once_with(api_key="sk-test-not-real")
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert call_kwargs["messages"] == [{"role": "user", "content": call_kwargs["messages"][0]["content"]}]


def test_anthropic_body_raises_connection_error_on_api_connection_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    fake_module, _ = _fake_anthropic_module(create_side_effect=_FakeAPIConnectionError("network down"))

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        with pytest.raises(ConnectionError, match="Anthropic API unreachable"):
            generate_answer("q", ["ctx"], backend="anthropic")


def _fake_groq_module(create_side_effect=None, response_text="the groq answer"):
    fake_module = MagicMock()
    fake_module.APIConnectionError = _FakeAPIConnectionError

    fake_message = MagicMock()
    fake_message.content = response_text
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = MagicMock()
    if create_side_effect is not None:
        fake_client.chat.completions.create.side_effect = create_side_effect
    else:
        fake_client.chat.completions.create.return_value = fake_response
    fake_module.Groq.return_value = fake_client
    return fake_module, fake_client


def test_groq_body_parses_message_content_from_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    fake_module, fake_client = _fake_groq_module(response_text="grounded groq answer")

    with patch.dict(sys.modules, {"groq": fake_module}):
        answer = generate_answer("q", ["ctx"], backend="groq", model="openai/gpt-oss-20b")

    assert answer == "grounded groq answer"
    fake_module.Groq.assert_called_once_with(api_key="gsk-test-not-real")
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-20b"


def test_groq_body_raises_connection_error_on_api_connection_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    fake_module, _ = _fake_groq_module(create_side_effect=_FakeAPIConnectionError("network down"))

    with patch.dict(sys.modules, {"groq": fake_module}):
        with pytest.raises(ConnectionError, match="Groq API unreachable"):
            generate_answer("q", ["ctx"], backend="groq")


def test_groq_body_handles_none_content_gracefully(monkeypatch):
    # A real Groq/OpenAI-style response can have message.content == None
    # (e.g. a tool-call-only response) — must not raise on .strip().
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    fake_module, _ = _fake_groq_module(response_text=None)

    with patch.dict(sys.modules, {"groq": fake_module}):
        answer = generate_answer("q", ["ctx"], backend="groq")

    assert "empty response" in answer.lower()


def test_ollama_body_returns_stripped_response_field(monkeypatch):
    monkeypatch.setattr(
        generator_module, "ollama_post", lambda url, payload: {"response": "  padded answer  \n"}
    )
    answer = generate_answer("q", ["ctx"], backend="ollama")
    assert answer == "padded answer"
