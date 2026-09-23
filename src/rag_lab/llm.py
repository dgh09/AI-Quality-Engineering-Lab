"""Interfaz del cliente LLM. El cliente real (compatible con OpenAI) se añade en T9."""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """Fallo al obtener una respuesta del LLM."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...
