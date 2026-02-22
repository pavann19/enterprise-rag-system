"""
rag/chunker.py
--------------
Lightweight, air-gapped document segmentation layer designed for
deterministic enterprise workflows.

Segments unstructured input into overlapping, word-boundary-aligned chunks
that preserve cross-boundary semantic context for downstream embedding and
retrieval. Chunk size and overlap are fully configurable to match the
context-window constraints of any Ollama embedding model.

No external calls — operates entirely on local text.
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks by word boundaries.

    Args:
        text:       Raw input text.
        chunk_size: Approximate number of characters per chunk.
        overlap:    Number of characters to overlap between chunks.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If chunk_size or overlap are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")
    if not text or not text.strip():
        return []

    words = text.split()
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1  # +1 for space

        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep last `overlap` characters worth of words for context
            overlap_words: List[str] = []
            overlap_len = 0
            for w in reversed(current_chunk):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current_chunk = overlap_words
            current_len = sum(len(w) + 1 for w in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
