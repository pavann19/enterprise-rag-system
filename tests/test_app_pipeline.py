"""
Tests for app.py's query_pipeline() orchestration itself.

Every other test exercises embed_texts/retrieve/generate_answer/validate
in isolation — this file is the one place that checks query_pipeline()
actually wires them together correctly (right arguments passed through,
right shape of the final response), with all four mocked out so no real
Ollama/cloud call happens.
"""

from unittest.mock import patch

import numpy as np
import pytest

import app
from validator.json_validator import ValidationError


class _FakeVectorStore:
    def __len__(self):
        return 3


@pytest.fixture
def fixed_pipeline_mocks():
    with (
        patch("app.embed_texts") as mock_embed,
        patch("app.retrieve") as mock_retrieve,
        patch("app.generate_answer") as mock_generate,
    ):
        mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_retrieve.return_value = [
            {"text": "passage one", "score": 0.9, "source": "a.txt"},
            {"text": "passage two", "score": 0.8, "source": "b.txt"},
        ]
        mock_generate.return_value = "the grounded answer"
        yield {"embed": mock_embed, "retrieve": mock_retrieve, "generate": mock_generate}


def test_query_pipeline_returns_valid_response(fixed_pipeline_mocks):
    response = app.query_pipeline(
        query="what is X?",
        chunks=["passage one", "passage two"],
        metadata=[{"source": "a.txt"}, {"source": "b.txt"}],
        vector_store=_FakeVectorStore(),
    )
    assert response["query"] == "what is X?"
    assert response["answer"] == "the grounded answer"
    assert response["sources"] == [
        {"text": "passage one", "source": "a.txt"},
        {"text": "passage two", "source": "b.txt"},
    ]


def test_query_pipeline_embeds_the_query_not_the_chunks(fixed_pipeline_mocks):
    app.query_pipeline(
        query="what is X?",
        chunks=["passage one"],
        metadata=[{"source": "a.txt"}],
        vector_store=_FakeVectorStore(),
    )
    embed_call = fixed_pipeline_mocks["embed"].call_args
    assert embed_call.args[0] == ["what is X?"]


def test_query_pipeline_passes_retrieved_text_to_generation(fixed_pipeline_mocks):
    app.query_pipeline(
        query="what is X?",
        chunks=["passage one", "passage two"],
        metadata=[{"source": "a.txt"}, {"source": "b.txt"}],
        vector_store=_FakeVectorStore(),
    )
    generate_call = fixed_pipeline_mocks["generate"].call_args
    assert generate_call.kwargs["passages"] == ["passage one", "passage two"]
    assert generate_call.kwargs["query"] == "what is X?"


def test_query_pipeline_uses_explicit_backend_and_model_overrides(fixed_pipeline_mocks):
    app.query_pipeline(
        query="q",
        chunks=["c"],
        metadata=[{"source": "a.txt"}],
        vector_store=_FakeVectorStore(),
        gen_model="custom-gen-model",
        gen_backend="groq",
        embed_model="custom-embed-model",
        embed_backend="local",
    )
    embed_call = fixed_pipeline_mocks["embed"].call_args
    assert embed_call.kwargs["model"] == "custom-embed-model"
    assert embed_call.kwargs["backend"] == "local"

    generate_call = fixed_pipeline_mocks["generate"].call_args
    assert generate_call.kwargs["model"] == "custom-gen-model"
    assert generate_call.kwargs["backend"] == "groq"


def test_query_pipeline_response_model_field_matches_gen_model(fixed_pipeline_mocks):
    response = app.query_pipeline(
        query="q",
        chunks=["c"],
        metadata=[{"source": "a.txt"}],
        vector_store=_FakeVectorStore(),
        gen_model="a-specific-model",
    )
    assert response["model"] == "a-specific-model"


def test_query_pipeline_propagates_connection_error(fixed_pipeline_mocks):
    fixed_pipeline_mocks["generate"].side_effect = ConnectionError("backend down")
    with pytest.raises(ConnectionError):
        app.query_pipeline(
            query="q",
            chunks=["c"],
            metadata=[{"source": "a.txt"}],
            vector_store=_FakeVectorStore(),
        )


def test_query_pipeline_propagates_validation_error_on_malformed_output(fixed_pipeline_mocks):
    fixed_pipeline_mocks["generate"].return_value = ""  # empty answer -> fails validate()
    with patch("app.generate_answer", return_value=""):
        with pytest.raises(ValidationError):
            app.query_pipeline(
                query="q",
                chunks=["c"],
                metadata=[{"source": "a.txt"}],
                vector_store=_FakeVectorStore(),
            )


def test_query_pipeline_applies_rerank_when_enabled(fixed_pipeline_mocks):
    reranked = [{"text": "passage two", "score": 0.8, "rerank_score": 0.99, "source": "b.txt"}]
    with patch.object(app, "RERANK_ENABLED", True), patch("app.rerank", return_value=reranked) as mock_rerank:
        response = app.query_pipeline(
            query="q",
            chunks=["passage one", "passage two"],
            metadata=[{"source": "a.txt"}, {"source": "b.txt"}],
            vector_store=_FakeVectorStore(),
        )
    mock_rerank.assert_called_once()
    assert response["sources"] == [{"text": "passage two", "source": "b.txt"}]


def test_query_pipeline_top_k_forwarded_to_retrieve(fixed_pipeline_mocks):
    app.query_pipeline(
        query="q",
        chunks=["c"],
        metadata=[{"source": "a.txt"}],
        vector_store=_FakeVectorStore(),
        top_k=7,
    )
    retrieve_call = fixed_pipeline_mocks["retrieve"].call_args
    assert retrieve_call.kwargs["top_k"] == 7
