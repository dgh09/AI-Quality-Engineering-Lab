"""Configuración del laboratorio leída desde variables de entorno (y opcionalmente un .env)."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_API_KEY = "ollama"  # Ollama ignora la key; no es un secreto real.
DEFAULT_MODEL = "qwen2.5:3b"
# Calibrado con scripts/calibrate_threshold.py (distancia coseno top-1, all-MiniLM-L6-v2,
# modo isolated): los casos 'answer' del golden set llegan como máximo a 0.4601 (gs-01) y
# los 'abstain' empiezan en 0.7734 (gs-09, dominio cruzado). 0.6 queda cerca del punto
# medio (~0.617), con margen de 0.14 sobre el peor 'answer' y de 0.17 bajo el 'abstain'
# más cercano. En modo shared las preguntas cruzadas encuentran el documento del otro
# agente a 0.34 (gs-09) y 0.47 (gs-10), dentro del umbral: el bug llega al LLM.
DEFAULT_RELEVANCE_THRESHOLD = 0.6
DEFAULT_TOP_K = 3
DEFAULT_RUNS_PER_CASE = 3
DEFAULT_PASS_RATE_THRESHOLD = 0.67


def _load_env(env_file: str | Path | None) -> dict[str, str]:
    """Combina el .env (si existe) con os.environ; el entorno real tiene prioridad."""
    merged: dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        for key, value in dotenv_values(env_file).items():
            if value is not None:
                merged[key] = value
    merged.update(os.environ)
    return merged


def _parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} debe ser un número, se recibió {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito, se recibió {raw!r}")
    return value


def _parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} debe ser un entero, se recibió {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str = field(repr=False)  # nunca exponer la key en logs/repr
    generator_model: str
    judge_model: str
    relevance_threshold: float
    top_k: int
    runs_per_case: int
    pass_rate_threshold: float

    def __post_init__(self) -> None:
        if not self.relevance_threshold > 0:
            raise ValueError(
                f"RELEVANCE_THRESHOLD debe ser > 0, se recibió {self.relevance_threshold}"
            )
        if self.top_k < 1:
            raise ValueError(f"TOP_K debe ser >= 1, se recibió {self.top_k}")
        if self.runs_per_case < 1:
            raise ValueError(f"RUNS_PER_CASE debe ser >= 1, se recibió {self.runs_per_case}")
        if not 0 <= self.pass_rate_threshold <= 1:
            raise ValueError(
                f"PASS_RATE_THRESHOLD debe estar en [0, 1], se recibió {self.pass_rate_threshold}"
            )

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        """Construye Settings desde el entorno.

        Si `env_file` apunta a un archivo existente, sus valores se usan como respaldo;
        las variables ya presentes en el entorno nunca se sobrescriben. Con `None` no se
        lee ningún archivo. No modifica `os.environ`.
        """
        env = _load_env(env_file)
        return cls(
            llm_base_url=env.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_api_key=env.get("LLM_API_KEY", DEFAULT_LLM_API_KEY),
            generator_model=env.get("GENERATOR_MODEL", DEFAULT_MODEL),
            judge_model=env.get("JUDGE_MODEL", DEFAULT_MODEL),
            relevance_threshold=_parse_float(
                env, "RELEVANCE_THRESHOLD", DEFAULT_RELEVANCE_THRESHOLD
            ),
            top_k=_parse_int(env, "TOP_K", DEFAULT_TOP_K),
            runs_per_case=_parse_int(env, "RUNS_PER_CASE", DEFAULT_RUNS_PER_CASE),
            pass_rate_threshold=_parse_float(
                env, "PASS_RATE_THRESHOLD", DEFAULT_PASS_RATE_THRESHOLD
            ),
        )
