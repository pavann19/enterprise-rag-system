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
