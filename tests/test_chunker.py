import pytest

from rag.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert chunks == [text]


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every chunk must be non-empty and shorter than the raw text
    for chunk in chunks:
        assert chunk.strip()
        assert len(chunk) <= len(text)


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # the overlap words carried into chunk[1] must appear at the start of chunk[1]
    # and at the tail of chunk[0]
    assert first_words[-1] == second_words[0] or any(
        w in second_words[: len(second_words) // 2] for w in first_words[-3:]
    )


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0)


def test_rejects_overlap_greater_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=50, overlap=60)


def test_rejects_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=50, overlap=-1)


def test_zero_overlap_produces_non_repeating_chunks():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=50, overlap=0)
    assert len(chunks) > 1
    # with zero overlap, no word should appear in two consecutive chunks
    for a, b in zip(chunks, chunks[1:], strict=False):  # deliberately offset by one
        assert set(a.split()).isdisjoint(set(b.split()))


def test_single_word_longer_than_chunk_size_still_returned():
    huge_word = "x" * 500
    chunks = chunk_text(huge_word, chunk_size=100, overlap=10)
    assert "".join(chunks).replace(" ", "") == huge_word or huge_word in "".join(chunks)
    assert len(chunks) >= 1


def test_unicode_text_preserved():
    text = "café résumé naïve 日本語 emoji 🎉 " * 10
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert chunks
    reconstructed = " ".join(chunks)
    for token in ["café", "résumé", "naïve", "日本語", "🎉"]:
        assert token in reconstructed


def test_whitespace_normalized_between_words():
    text = "word1\n\nword2\tword3   word4"
    chunks = chunk_text(text, chunk_size=300, overlap=10)
    assert chunks == ["word1 word2 word3 word4"]


def test_chunking_is_deterministic():
    text = " ".join(f"word{i}" for i in range(300))
    first = chunk_text(text, chunk_size=80, overlap=15)
    second = chunk_text(text, chunk_size=80, overlap=15)
    assert first == second


def test_no_chunk_is_only_whitespace():
    text = " ".join(f"word{i}" for i in range(150))
    chunks = chunk_text(text, chunk_size=40, overlap=5)
    assert all(chunk.strip() for chunk in chunks)


def test_text_ending_exactly_on_a_chunk_boundary_has_no_trailing_empty_chunk():
    # When the last word processed pushes current_len over chunk_size AND
    # overlap=0 (so nothing carries into a new current_chunk) AND that was
    # the final word, current_chunk is empty with no more words left to
    # append — the trailing `if current_chunk:` guard must skip it rather
    # than emit an empty final chunk.
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=len(text), overlap=0)
    assert chunks == [text]
    assert all(chunk.strip() for chunk in chunks)
