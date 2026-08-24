"""
Property-based tests for validator/json_validator.py using Hypothesis.

tests/test_json_validator.py already covers specific hand-picked cases
(missing key, empty string, wrong type for one field at a time). These
tests instead generate arbitrary combinations of field values/types —
including deeply nested garbage, mixed valid/invalid fields in the same
response, and arbitrary-length source lists — to check the invariant that
matters most for a schema gate: validate() either returns a value that
satisfies every documented constraint, or raises ValidationError. It must
never return something silently malformed.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from validator.json_validator import ValidationError, validate

_non_empty_text = st.text(min_size=1).filter(lambda s: s.strip() != "")

_valid_source = st.fixed_dictionaries({"text": _non_empty_text, "source": _non_empty_text})
_valid_response = st.fixed_dictionaries(
    {
        "query": _non_empty_text,
        "answer": _non_empty_text,
        "model": _non_empty_text,
        "sources": st.lists(_valid_source, min_size=1, max_size=10),
    }
)

# Arbitrary JSON-ish scalars/containers, for throwing genuinely wrong types
# at fields that are supposed to be non-empty strings.
_wrong_type_value = st.one_of(
    st.none(),
    st.integers(),
    st.floats(allow_nan=False),
    st.booleans(),
    st.lists(st.text(), max_size=3),
    st.dictionaries(st.text(max_size=3), st.text(max_size=3), max_size=3),
)


@given(response=_valid_response)
@settings(max_examples=200)
def test_any_well_formed_response_validates_successfully(response):
    result = validate(dict(response))
    assert result["query"] == response["query"]
    assert result["sources"] == response["sources"]


@given(response=_valid_response, key=st.sampled_from(["query", "answer", "model"]))
def test_blanking_any_required_string_field_always_raises(response, key):
    response = dict(response)
    response[key] = ""
    with pytest.raises(ValidationError):
        validate(response)


@given(response=_valid_response, key=st.sampled_from(["query", "answer", "model"]), wrong=_wrong_type_value)
def test_wrong_type_on_any_required_string_field_always_raises(response, key, wrong):
    response = dict(response)
    response[key] = wrong
    with pytest.raises(ValidationError):
        validate(response)


@given(response=_valid_response, missing_key=st.sampled_from(["query", "answer", "model", "sources"]))
def test_removing_any_required_key_always_raises(response, missing_key):
    response = dict(response)
    del response[missing_key]
    with pytest.raises(ValidationError):
        validate(response)


@given(not_a_dict=st.one_of(st.none(), st.integers(), st.text(), st.lists(st.integers())))
def test_non_dict_input_always_raises(not_a_dict):
    with pytest.raises(ValidationError):
        validate(not_a_dict)


@given(
    response=_valid_response, bad_source_field=st.sampled_from(["text", "source"]), wrong=_wrong_type_value
)
def test_wrong_type_in_any_source_entry_field_always_raises(response, bad_source_field, wrong):
    response = dict(response)
    response["sources"] = [dict(response["sources"][0])]
    response["sources"][0][bad_source_field] = wrong
    with pytest.raises(ValidationError):
        validate(response)


@given(response=_valid_response)
def test_empty_sources_list_always_raises(response):
    response = dict(response)
    response["sources"] = []
    with pytest.raises(ValidationError):
        validate(response)
