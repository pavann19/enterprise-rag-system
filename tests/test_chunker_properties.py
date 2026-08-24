"""
Property-based tests for rag/chunker.py using Hypothesis.

The example-based tests in tests/test_chunker.py check specific scenarios
(empty input, one short sentence, a fixed long text). These tests instead
assert invariants that must hold for *any* input Hypothesis can generate —
arbitrary unicode, pathological whitespace, chunk_size/overlap combinations
near their boundary conditions — which is exactly the input space a
hand-picked example set tends to under-sample.
"""

import string
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag.chunker import chunk_text

# Real prose plus enough unicode/whitespace noise to stress word-splitting,
# without spending Hypothesis's budget on inputs no real document contains
# (control characters, unpaired surrogates).
_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"), blacklist_characters="\x00"),
    min_size=0,
    max_size=500,
)


@given(text=_text_strategy)
@settings(max_examples=200)
def test_never_raises_for_default_chunk_params(text):
    chunk_text(text)  # must not raise for any string, given valid chunk_size/overlap


@given(
    text=_text_strategy,
    chunk_size=st.integers(min_value=1, max_value=1000),
)
def test_zero_overlap_never_raises_and_returns_list(text, chunk_size):
    result = chunk_text(text, chunk_size=chunk_size, overlap=0)
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


@given(text=_text_strategy, chunk_size=st.integers(min_value=1, max_value=500))
def test_chunks_are_never_empty_or_whitespace_only(text, chunk_size):
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    for chunk in chunks:
        assert chunk.strip() != ""


@given(text=st.text(alphabet=" \t\n\r", min_size=0, max_size=50))
def test_blank_or_whitespace_only_text_returns_empty_list(text):
    assert chunk_text(text) == []


@given(text=st.text(alphabet=string.printable, min_size=1, max_size=500).filter(lambda t: t.strip()))
def test_every_original_word_appears_in_the_chunked_output(text):
    """
    chunk_text() must not silently drop content — every whitespace-delimited
    word from the input appears at least as many times in the chunked output
    as in the original (overlap can duplicate a word across chunks, but
    never drop one).
    """
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    chunked_words = Counter(" ".join(chunks).split())
    original_words = Counter(text.split())
    for word, count in original_words.items():
        assert chunked_words[word] >= count


@given(chunk_size=st.integers(max_value=0))
def test_non_positive_chunk_size_always_raises(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", chunk_size=chunk_size, overlap=0)


@given(chunk_size=st.integers(min_value=1, max_value=500), extra=st.integers(min_value=0, max_value=1000))
def test_overlap_at_or_above_chunk_size_always_raises(chunk_size, extra):
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=chunk_size, overlap=chunk_size + extra)
