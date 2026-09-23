"""Cliente LLM: interfaz `LLMClient` y cliente real compatible con OpenAI (p. ej. Ollama).

El resto del código depende solo del Protocol `LLMClient`; las pruebas usan `FakeLLM`
(tests/fakes.py) y nunca tocan la red.
"""

from __future__ import annotations

from typing import Protocol

import openai
from openai import OpenAI

from rag_lab.config import Settings

# Un modelo de 3B en CPU puede tardar decenas de segundos por respuesta; 120 s deja
# margen sin colgar indefinidamente la CLI si Ollama no responde.
DEFAULT_TIMEOUT_SECONDS = 120.0
# Sin reintentos del SDK: el peor caso por llamada es timeout × (1 + max_retries), así
# que con 0 queda en 120 s en vez de ~360 s con el default del SDK (2). Con un modelo
# local de 3B en CPU, un timeout o una conexión rechazada casi siempre se repite, así
# que reintentar solo multiplica la espera antes de fallar.
DEFAULT_MAX_RETRIES = 0


class LLMError(Exception):
    """Fallo al obtener una respuesta del LLM."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class OpenAICompatibleClient:
    """`LLMClient` sobre el SDK `openai` apuntando a `Settings.llm_base_url`.

    `model` se recibe aparte para usar el mismo cliente como generador
    (`settings.generator_model`) o como juez (`settings.judge_model`).
    """

    def __init__(
        self,
        settings: Settings,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not model.strip():
            raise ValueError("model no puede estar vacío")
        self.model = model
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Devuelve el texto de la respuesta; cualquier fallo del SDK se convierte en `LLMError`.

        Con `json_mode=True` pide `response_format={"type": "json_object"}`. Una respuesta
        sin contenido (sin choices, `None` o solo espacios) también es `LLMError`.
        """
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise LLMError(
                f"Fallo al llamar al modelo {self.model!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if not response.choices:
            raise LLMError(f"El modelo {self.model!r} devolvió una respuesta sin choices")
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise LLMError(f"El modelo {self.model!r} devolvió una respuesta vacía")
        return content
