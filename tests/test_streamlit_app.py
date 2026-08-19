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
        ingestion_module, "ingest",
        side_effect=ConnectionError("Ollama is not reachable at 'http://localhost:11434'"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=30)

    assert not at.exception
    assert any("ollama is not reachable" in e.value.lower() for e in at.error)
