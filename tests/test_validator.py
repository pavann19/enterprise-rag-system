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


def test_missing_query_key_raises():
    bad = dict(VALID_RESPONSE)
    del bad["query"]
    with pytest.raises(ValidationError, match="query"):
        validate(bad)


def test_multiple_missing_keys_all_named_in_error():
    bad = {"query": "q"}
    with pytest.raises(ValidationError) as exc_info:
        validate(bad)
    assert "answer" in str(exc_info.value)
    assert "sources" in str(exc_info.value)
    assert "model" in str(exc_info.value)


def test_whitespace_only_answer_raises():
    bad = dict(VALID_RESPONSE)
    bad["answer"] = "   \n\t  "
    with pytest.raises(ValidationError):
        validate(bad)


def test_non_string_model_raises():
    bad = dict(VALID_RESPONSE)
    bad["model"] = None
    with pytest.raises(ValidationError):
        validate(bad)


def test_sources_not_a_list_raises():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = {"text": "t", "source": "s"}  # dict instead of list
    with pytest.raises(ValidationError):
        validate(bad)


def test_source_entry_whitespace_only_field_raises():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = [{"text": "  ", "source": "audit_controls.txt"}]
    with pytest.raises(ValidationError):
        validate(bad)


def test_source_entry_non_string_field_raises():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = [{"text": 42, "source": "audit_controls.txt"}]
    with pytest.raises(ValidationError):
        validate(bad)


def test_second_of_multiple_sources_invalid_still_caught():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = [
        {"text": "valid one", "source": "a.txt"},
        {"text": "", "source": "b.txt"},
    ]
    with pytest.raises(ValidationError, match=r"sources\[1\]"):
        validate(bad)


def test_extra_unexpected_fields_are_tolerated():
    extended = dict(VALID_RESPONSE)
    extended["trace_id"] = "req-123"
    result = validate(extended)
    assert result["trace_id"] == "req-123"


def test_many_sources_all_validated():
    bad = dict(VALID_RESPONSE)
    bad["sources"] = [{"text": f"passage {i}", "source": f"doc{i}.txt"} for i in range(50)]
    bad["sources"].append({"text": "", "source": "bad.txt"})  # last one invalid
    with pytest.raises(ValidationError, match=r"sources\[50\]"):
        validate(bad)


def test_validate_rejects_non_dict_input():
    with pytest.raises(ValidationError):
        validate(["not", "a", "dict"])


def test_validate_json_string_valid():
    raw = '{"a": 1}'
    assert validate_json_string(raw) == {"a": 1}


def test_validate_json_string_invalid_raises():
    with pytest.raises(ValidationError):
        validate_json_string("{not valid json")
