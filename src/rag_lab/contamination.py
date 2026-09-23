"""Detector de contaminación de contexto entre agentes."""

from __future__ import annotations

from collections.abc import Iterable

from rag_lab.store import Chunk


def foreign_chunks(chunks: Iterable[Chunk], agent_id: str) -> list[Chunk]:
    """Chunks cuyo `agent_id` difiere de `agent_id`, en el orden recibido.

    Una lista no vacía indica contaminación: el agente recuperó conocimiento de otro.
    """
    return [chunk for chunk in chunks if chunk.agent_id != agent_id]
