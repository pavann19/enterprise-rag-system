import pytest

from validator.json_validator import validate, validate_json_string, ValidationError


VALID_RESPONSE = {
    "query": "What is the audit control policy?",
    "answer": "The audit control policy follows COSO.",
    "sources": [{"text": "Section 2 — COSO...", "source": "audit_controls.txt"}],
    "model": "mistral",
}


def test_valid_response_passes():
    result = validate(dict(VALID_RESPONSE))
    assert result["query"] == VALID_RESPONSE["query"]
    assert result["sources"][0]["source"] == "audit_controls.txt"


def test_missing_required_key_raises():
    bad = dict(VALID_RESPONSE)
    del bad["model"]
    with pytest.raises(ValidationError):
        validate(bad)


def test_empty_query_raises():
    bad = dict(VALID_RESPONSE)
    bad["query"] = "   "
    with pytest.raises(ValidationError):
        validate(bad)


def test_non_string_answer_raises():
    bad = dict(VALID_RESPONSE)
    bad["answer"] = 12345
    with pytest.raises(ValidationError):
        validate(bad)


def test_empty_sources_list_raises():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = []
    with pytest.raises(ValidationError):
        validate(bad)


def test_source_entry_missing_field_raises():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = [{"text": "some text"}]  # missing "source"
    with pytest.raises(ValidationError):
        validate(bad)


def test_source_entry_not_a_dict_raises():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = ["just a string"]
    with pytest.raises(ValidationError):
        validate(bad)


def test_validate_json_string_valid():
    raw = '{"a": 1}'
    assert validate_json_string(raw) == {"a": 1}


def test_validate_json_string_invalid_raises():
    with pytest.raises(ValidationError):
        validate_json_string("{not valid json")
