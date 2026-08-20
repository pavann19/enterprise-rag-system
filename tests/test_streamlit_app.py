"""
Integration tests for streamlit_app.py using Streamlit's AppTest harness
(streamlit.testing.v1), which runs the script exactly as `streamlit run`
would but without a browser — script exceptions surface as test failures,
and widgets are inspected/driven programmatically.

rag.ingestion.ingest and app.query_pipeline are monkeypatched before each
run so these tests need no live Ollama instance.
"""

from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
import rag.ingestion as ingestion_module


@pytest.fixture(autouse=True)
def _clear_corpus_cache():
    # load_corpus() is @st.cache_resource-decorated in streamlit_app.py, and
    # that cache is process-global — without clearing it, the first test's
    # successful corpus load would be silently reused by every later test
    # regardless of what ingest() is patched to do on that run.
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


FAKE_CHUNKS = ["Section 2: expense authorization text.", "Section 3: budget variance text."]
FAKE_METADATA = [{"source": "financial_policy.txt"}, {"source": "financial_policy.txt"}]


class _FakeVectorStore:
    def __len__(self):
        return len(FAKE_CHUNKS)


def _fake_ingest(*args, **kwargs):
    return FAKE_CHUNKS, FAKE_METADATA, _FakeVectorStore()


@pytest.fixture
def running_app():
    """Runs streamlit_app.py with ingestion mocked out, corpus loaded successfully."""
    with patch.object(ingestion_module, "ingest", side_effect=_fake_ingest):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)
        yield at


def test_app_loads_without_error(running_app):
    assert not running_app.exception


def test_app_shows_corpus_summary(running_app):
    body_text = " ".join(md.value for md in running_app.markdown)
    assert "financial_policy.txt" in body_text


def test_app_shows_backend_captions_in_sidebar(running_app):
    caption_text = " ".join(c.value for c in running_app.sidebar.caption)
    assert "ollama" in caption_text  # default EMBED_BACKEND/GEN_BACKEND


def test_asking_empty_query_shows_warning(running_app):
    running_app.button[0].click().run(timeout=30)
    assert any("enter a question" in w.value.lower() for w in running_app.warning)


def test_asking_a_question_shows_answer_and_sources(running_app):
    fake_response = {
        "query": "What is the Tier 2 approval threshold?",
        "answer": "Tier 2 spending ($5,000-$50,000) requires department head and Finance Business Partner sign-off.",
        "sources": [{"text": "Section 2 text about Tier 2.", "source": "financial_policy.txt"}],
        "model": "mistral",
    }
    with patch("app.query_pipeline", return_value=fake_response) as mock_pipeline:
        running_app.text_input[0].input("What is the Tier 2 approval threshold?").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert mock_pipeline.called
    body_text = " ".join(md.value for md in running_app.markdown)
    assert "Tier 2" in body_text or "Finance Business Partner" in body_text


def test_asking_a_question_shows_validation_error(running_app):
    from validator.json_validator import ValidationError

    with patch("app.query_pipeline", side_effect=ValidationError("bad schema")):
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("validation error" in e.value.lower() for e in running_app.error)


def test_missing_data_directory_shows_error_and_stops():
    with patch.object(ingestion_module, "ingest", side_effect=FileNotFoundError("no docs found")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    assert any("data directory error" in e.value.lower() for e in at.error)


def test_ollama_unreachable_shows_friendly_error():
    with patch.object(
        ingestion_module,
        "ingest",
        side_effect=ConnectionError("Ollama is not reachable at 'http://localhost:11434'"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    assert any("ollama is not reachable" in e.value.lower() for e in at.error)


# ── Cloud-backend sidebar branches ──────────────────────────────────────────
# streamlit_app.py's `from app import GEN_BACKEND` re-reads whatever is
# currently on the already-imported `app` module each time AppTest re-execs
# the script, so patching app.GEN_BACKEND/app.EMBED_BACKEND before .run()
# is what lets these tests exercise the non-default sidebar branches.


def test_sidebar_shows_anthropic_hint_when_gen_backend_is_anthropic():
    with (
        patch.object(app, "GEN_BACKEND", "anthropic"),
        patch.object(ingestion_module, "ingest", side_effect=_fake_ingest),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    caption_text = " ".join(c.value for c in at.sidebar.caption)
    assert "claude api" in caption_text.lower()
    assert "anthropic_api_key" in caption_text.lower()


def test_sidebar_shows_groq_hint_when_gen_backend_is_groq():
    with (
        patch.object(app, "GEN_BACKEND", "groq"),
        patch.object(ingestion_module, "ingest", side_effect=_fake_ingest),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    caption_text = " ".join(c.value for c in at.sidebar.caption)
    assert "groq api" in caption_text.lower()
    assert "groq_api_key" in caption_text.lower()


def test_sidebar_omits_ollama_hint_when_neither_backend_is_ollama():
    with (
        patch.object(app, "GEN_BACKEND", "groq"),
        patch.object(app, "EMBED_BACKEND", "local"),
        patch.object(ingestion_module, "ingest", side_effect=_fake_ingest),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    caption_text = " ".join(c.value for c in at.sidebar.caption)
    assert "ollama serve" not in caption_text.lower()
    assert not any("ollama serve" in code.value for code in at.sidebar.code)


# ── Non-Ollama error branches ────────────────────────────────────────────────


def test_corpus_load_connection_error_shown_generically_for_non_ollama_backend():
    with (
        patch.object(app, "EMBED_BACKEND", "local"),
        patch.object(ingestion_module, "ingest", side_effect=ConnectionError("local model download failed")),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    assert any("embedding backend unreachable" in e.value.lower() for e in at.error)


def test_ollama_generation_connection_error_shown_with_pull_hint(running_app):
    with patch(
        "app.query_pipeline",
        side_effect=ConnectionError("Ollama is not reachable at 'http://localhost:11434'"),
    ):
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("ollama is not reachable" in e.value.lower() for e in running_app.error)
    assert any("ollama pull" in e.value.lower() for e in running_app.error)


def test_corpus_load_import_error_shown_for_missing_dependency():
    with (
        patch.object(app, "EMBED_BACKEND", "local"),
        patch.object(ingestion_module, "ingest", side_effect=ImportError("sentence-transformers missing")),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    assert any("missing dependency" in e.value.lower() for e in at.error)


def test_non_ollama_generation_connection_error_shown_generically(running_app):
    with (
        patch.object(app, "GEN_BACKEND", "groq"),
        patch("app.query_pipeline", side_effect=ConnectionError("Groq API unreachable: timeout")),
    ):
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("generation backend unreachable" in e.value.lower() for e in running_app.error)


def test_generation_import_error_shown_as_misconfiguration(running_app):
    with patch("app.query_pipeline", side_effect=ImportError("backend='groq' requires the groq package")):
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("misconfigured" in e.value.lower() for e in running_app.error)


def test_generation_runtime_error_shown_as_misconfiguration(running_app):
    with patch("app.query_pipeline", side_effect=RuntimeError("GROQ_API_KEY environment variable")):
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("misconfigured" in e.value.lower() for e in running_app.error)
