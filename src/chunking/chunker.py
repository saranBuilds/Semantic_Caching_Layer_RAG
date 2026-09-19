"""Splits raw paper text into overlapping chunks for embedding."""
from typing import List


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Simple sliding-window character chunker.
    Good enough for a single paper; swap for a section-aware splitter
    later if you want cleaner citation boundaries.
    """
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return [c for c in chunks if c.strip()]