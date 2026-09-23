"""Pruebas unitarias del detector de contaminación (sin store)."""

from __future__ import annotations

from rag_lab.contamination import foreign_chunks
from rag_lab.store import Chunk


def _chunk(chunk_id: str, agent_id: str, distance: float = 0.1) -> Chunk:
    return Chunk(id=chunk_id, agent_id=agent_id, text=f"texto {chunk_id}", distance=distance)


def test_returns_empty_when_all_chunks_are_own() -> None:
    chunks = [_chunk("faq-01", "faq"), _chunk("faq-02", "faq")]
    assert foreign_chunks(chunks, "faq") == []


def test_returns_only_foreign_chunks_preserving_order() -> None:
    chunks = [
        _chunk("seg-02", "seguimiento", 0.2),
        _chunk("faq-01", "faq", 0.3),
        _chunk("seg-01", "seguimiento", 0.4),
    ]
    result = foreign_chunks(chunks, "faq")
    assert result == [chunks[0], chunks[2]]


def test_empty_input_returns_empty_list() -> None:
    assert foreign_chunks([], "faq") == []


def test_accepts_any_iterable_and_returns_list() -> None:
    chunks = (_chunk("faq-01", "faq"), _chunk("seg-01", "seguimiento"))
    result = foreign_chunks(iter(chunks), "seguimiento")
    assert isinstance(result, list)
    assert result == [chunks[0]]


def test_all_foreign_when_agent_owns_nothing() -> None:
    chunks = [_chunk("faq-01", "faq"), _chunk("seg-01", "seguimiento")]
    assert foreign_chunks(chunks, "otro") == chunks
