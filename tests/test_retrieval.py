from __future__ import annotations

from app.retrieval import chunk_text, normalize


def test_normalize_returns_unit_vector() -> None:
    vector = normalize([3.0, 4.0])
    assert vector == [0.6, 0.8]


def test_chunk_text_preserves_content_and_bounds_chunks() -> None:
    text = "First paragraph.\n\n" + ("Second sentence. " * 80)
    chunks = chunk_text(text, max_chars=180, overlap_chars=30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 180 for chunk in chunks)


def test_chunk_text_handles_empty_text() -> None:
    assert chunk_text("") == []

