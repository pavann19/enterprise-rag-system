import sys
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import rag.generator as generator_module
from rag.generator import generate_answer_stream


def test_generate_answer_stream_rejects_blank_query():
    with pytest.raises(ValueError):
        list(generate_answer_stream("   ", ["ctx"]))


def test_generate_answer_stream_rejects_empty_passages():
    with pytest.raises(ValueError):
        list(generate_answer_stream("q", []))


def test_generate_answer_stream_rejects_unknown_backend():
    with pytest.raises(ValueError):
        list(generate_answer_stream("q", ["ctx"], backend="nonexistent"))


def test_generate_answer_stream_dispatches_to_registered_backend(monkeypatch):
    def _fake_stream(prompt, model):
        yield "he"
        yield "llo"

    monkeypatch.setitem(generator_module._STREAMING_BACKENDS, "ollama", _fake_stream)
    tokens = list(generate_answer_stream("q", ["ctx"], backend="ollama"))
    assert tokens == ["he", "llo"]


class _FakeStreamLine(bytes):
    pass


def _fake_ollama_stream_response(lines):
    mock_response = MagicMock()
    mock_response.__enter__.return_value = iter(lines)
    mock_response.__exit__.return_value = False
    return mock_response


def test_stream_ollama_yields_tokens_until_done(monkeypatch):
    lines = [
        b'{"response": "Hel", "done": false}\n',
        b"\n",  # blank line — must be skipped, not passed to json.loads
        b'{"response": "lo", "done": false}\n',
        b'{"response": "", "done": true}\n',
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _fake_ollama_stream_response(lines)
    )
    tokens = list(generator_module._stream_ollama("prompt", "mistral"))
    assert tokens == ["Hel", "lo"]


def test_stream_ollama_exhausts_iterator_when_done_never_true(monkeypatch):
    lines = [b'{"response": "a"}\n', b'{"response": "b"}\n']
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _fake_ollama_stream_response(lines)
    )
    tokens = list(generator_module._stream_ollama("prompt", "mistral"))
    assert tokens == ["a", "b"]


def test_stream_ollama_raises_connection_error_on_url_error(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    with pytest.raises(ConnectionError, match="Could not reach Ollama"):
        list(generator_module._stream_ollama("prompt", "mistral"))


def test_stream_ollama_raises_connection_error_on_os_error(monkeypatch):
    def _raise(request, timeout=None):
        raise ConnectionResetError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    with pytest.raises(ConnectionError, match="Could not reach Ollama"):
        list(generator_module._stream_ollama("prompt", "mistral"))


def test_stream_anthropic_raises_clear_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    with pytest.raises(ImportError, match="anthropic"):
        list(generator_module._stream_anthropic("prompt", "claude-haiku-4-5-20251001"))


def test_stream_anthropic_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", MagicMock())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        list(generator_module._stream_anthropic("prompt", "claude-haiku-4-5-20251001"))


class _FakeAPIConnectionError(Exception):
    pass


def test_stream_anthropic_yields_text_stream(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    fake_module = MagicMock()
    fake_module.APIConnectionError = _FakeAPIConnectionError

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__enter__.return_value.text_stream = iter(["Hel", "lo"])
    fake_stream_ctx.__exit__.return_value = False

    fake_client = MagicMock()
    fake_client.messages.stream.return_value = fake_stream_ctx
    fake_module.Anthropic.return_value = fake_client

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        tokens = list(generator_module._stream_anthropic("prompt", "claude-haiku-4-5-20251001"))

    assert tokens == ["Hel", "lo"]


def test_stream_anthropic_raises_connection_error_on_api_connection_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    fake_module = MagicMock()
    fake_module.APIConnectionError = _FakeAPIConnectionError

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__enter__.side_effect = _FakeAPIConnectionError("network down")
    fake_stream_ctx.__exit__.return_value = False

    fake_client = MagicMock()
    fake_client.messages.stream.return_value = fake_stream_ctx
    fake_module.Anthropic.return_value = fake_client

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        with pytest.raises(ConnectionError, match="Anthropic API unreachable"):
            list(generator_module._stream_anthropic("prompt", "claude-haiku-4-5-20251001"))


def test_stream_groq_raises_clear_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    with pytest.raises(ImportError, match="groq"):
        list(generator_module._stream_groq("prompt", "openai/gpt-oss-20b"))


def test_stream_groq_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", MagicMock())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        list(generator_module._stream_groq("prompt", "openai/gpt-oss-20b"))


def test_stream_groq_yields_delta_content(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    fake_module = MagicMock()
    fake_module.APIConnectionError = _FakeAPIConnectionError

    def _make_chunk(content):
        chunk = MagicMock()
        chunk.choices[0].delta.content = content
        return chunk

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(
        [_make_chunk("Hel"), _make_chunk(None), _make_chunk("lo")]
    )
    fake_module.Groq.return_value = fake_client

    with patch.dict(sys.modules, {"groq": fake_module}):
        tokens = list(generator_module._stream_groq("prompt", "openai/gpt-oss-20b"))

    assert tokens == ["Hel", "lo"]


def test_stream_groq_raises_connection_error_on_api_connection_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    fake_module = MagicMock()
    fake_module.APIConnectionError = _FakeAPIConnectionError
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakeAPIConnectionError("network down")
    fake_module.Groq.return_value = fake_client

    with patch.dict(sys.modules, {"groq": fake_module}):
        with pytest.raises(ConnectionError, match="Groq API unreachable"):
            list(generator_module._stream_groq("prompt", "openai/gpt-oss-20b"))
