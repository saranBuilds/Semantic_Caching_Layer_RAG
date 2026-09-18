"""Basic sanity tests — expand as each phase is built."""
from src.chunking.chunker import chunk_text


def test_chunk_text_basic():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []