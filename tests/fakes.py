"""Dobles de prueba compartidos (importables vía `pythonpath = ["tests"]`)."""

from __future__ import annotations

from collections.abc import Sequence


class FakeLLM:
    """LLM falso y determinista: registra las llamadas y devuelve respuestas guionizadas.

    Con `answers` como texto devuelve siempre ese texto; con una secuencia devuelve
    sus elementos en orden y lanza `AssertionError` si se agotan. `json_modes` guarda
    el `json_mode` de cada llamada, en paralelo a `calls`.
    """

    DEFAULT_ANSWER = "FAKE LLM ANSWER"

    def __init__(self, answers: str | Sequence[str] = DEFAULT_ANSWER) -> None:
        self._fixed = answers if isinstance(answers, str) else None
        self._script = [] if isinstance(answers, str) else list(answers)
        self.calls: list[tuple[str, str]] = []
        self.json_modes: list[bool] = []

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        self.json_modes.append(json_mode)
        if self._fixed is not None:
            return self._fixed
        if len(self.calls) > len(self._script):
            raise AssertionError(
                f"FakeLLM: se agotaron las {len(self._script)} respuestas guionizadas"
            )
        return self._script[len(self.calls) - 1]
