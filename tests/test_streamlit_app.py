"""
Integration tests for streamlit_app.py using Streamlit's AppTest harness
(streamlit.testing.v1), which runs the script exactly as `streamlit run`
would but without a browser — script exceptions surface as test failures,
and widgets are inspected/driven programmatically.

rag.ingestion.ingest and app.query_pipeline are monkeypatched before each
run so these tests need no live Ollama instance.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
import rag.embedder as embedder_module
import rag.generator as generator_module
import rag.ingestion as ingestion_module
import rag.retriever as retriever_module

FAKE_RETRIEVE_RESULTS = [
    {"text": "Section 2 text about Tier 2.", "score": 0.9, "source": "financial_policy.txt"}
]


def _patch_query_stream(answer_tokens=None, retrieve_side_effect=None, stream_side_effect=None):
    """
    streamlit_app.py's query flow calls embed_texts() -> retrieve() ->
    generate_answer_stream() directly (not app.query_pipeline, which is
    only used by the CLI/API — see app.py's query_pipeline docstring).
    Patches the same three functions the script re-imports fresh on every
    AppTest.run(), mirroring the ingestion_module.ingest patch pattern above.
    """
    embed_patch = patch.object(embedder_module, "embed_texts", return_value=[[0.1, 0.2]])
    if retrieve_side_effect is not None:
        retrieve_patch = patch.object(retriever_module, "retrieve", side_effect=retrieve_side_effect)
    else:
        retrieve_patch = patch.object(retriever_module, "retrieve", return_value=FAKE_RETRIEVE_RESULTS)
    if stream_side_effect is not None:
        stream_patch = patch.object(
            generator_module, "generate_answer_stream", side_effect=stream_side_effect
        )
    else:
        tokens = ["an answer"] if answer_tokens is None else answer_tokens
        stream_patch = patch.object(generator_module, "generate_answer_stream", return_value=iter(tokens))
    return embed_patch, retrieve_patch, stream_patch


# AppTest.from_file() resolves a *relative* path against the file that calls
# it (tests/) in some streamlit versions and against the process CWD in
# others — that inconsistency passed locally (streamlit 1.49.1, CWD ==
# repo root by coincidence) and failed on CI's newer streamlit install with
# a FileNotFoundError. An absolute path sidesteps the ambiguity entirely.
STREAMLIT_APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


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
        at = AppTest.from_file(STREAMLIT_APP_PATH)
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
    embed_patch, retrieve_patch, stream_patch = _patch_query_stream(
        answer_tokens=["Tier 2 spending ($5,000-$50,000) requires ", "Finance Business Partner sign-off."]
    )
    with embed_patch, retrieve_patch, stream_patch as mock_stream:
        running_app.text_input[0].input("What is the Tier 2 approval threshold?").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert mock_stream.called
    body_text = " ".join(md.value for md in running_app.markdown)
    assert "Finance Business Partner" in body_text


def test_asking_a_question_shows_empty_response_warning(running_app):
    embed_patch, retrieve_patch, stream_patch = _patch_query_stream(answer_tokens=[])
    with embed_patch, retrieve_patch, stream_patch:
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("empty response" in w.value.lower() for w in running_app.warning)


def test_missing_data_directory_shows_error_and_stops():
    with patch.object(ingestion_module, "ingest", side_effect=FileNotFoundError("no docs found")):
        at = AppTest.from_file(STREAMLIT_APP_PATH)
        at.run(timeout=30)

    assert not at.exception
    assert any("data directory error" in e.value.lower() for e in at.error)


def test_ollama_unreachable_shows_friendly_error():
    with patch.object(
        ingestion_module,
        "ingest",
        side_effect=ConnectionError("Ollama is not reachable at 'http://localhost:11434'"),
    ):
        at = AppTest.from_file(STREAMLIT_APP_PATH)
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
        at = AppTest.from_file(STREAMLIT_APP_PATH)
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
        at = AppTest.from_file(STREAMLIT_APP_PATH)
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
        at = AppTest.from_file(STREAMLIT_APP_PATH)
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
        at = AppTest.from_file(STREAMLIT_APP_PATH)
        at.run(timeout=30)

    assert not at.exception
    assert any("embedding backend unreachable" in e.value.lower() for e in at.error)


def test_ollama_generation_connection_error_shown_with_pull_hint(running_app):
    embed_patch, retrieve_patch, stream_patch = _patch_query_stream(
        stream_side_effect=ConnectionError("Ollama is not reachable at 'http://localhost:11434'")
    )
    with embed_patch, retrieve_patch, stream_patch:
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
        at = AppTest.from_file(STREAMLIT_APP_PATH)
        at.run(timeout=30)

    assert not at.exception
    assert any("missing dependency" in e.value.lower() for e in at.error)


def test_non_ollama_generation_connection_error_shown_generically(running_app):
    embed_patch, retrieve_patch, stream_patch = _patch_query_stream(
        stream_side_effect=ConnectionError("Groq API unreachable: timeout")
    )
    with patch.object(app, "GEN_BACKEND", "groq"), embed_patch, retrieve_patch, stream_patch:
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("generation backend unreachable" in e.value.lower() for e in running_app.error)


def test_generation_import_error_shown_as_misconfiguration(running_app):
    embed_patch, retrieve_patch, stream_patch = _patch_query_stream(
        stream_side_effect=ImportError("backend='groq' requires the groq package")
    )
    with embed_patch, retrieve_patch, stream_patch:
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("misconfigured" in e.value.lower() for e in running_app.error)


def test_generation_runtime_error_shown_as_misconfiguration(running_app):
    embed_patch, retrieve_patch, stream_patch = _patch_query_stream(
        stream_side_effect=RuntimeError("GROQ_API_KEY environment variable")
    )
    with embed_patch, retrieve_patch, stream_patch:
        running_app.text_input[0].input("anything").run(timeout=30)
        running_app.button[0].click().run(timeout=30)

    assert not running_app.exception
    assert any("misconfigured" in e.value.lower() for e in running_app.error)
