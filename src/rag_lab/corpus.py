"""Carga y validación del corpus de documentos de los agentes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Directorio de datos del proyecto, resuelto respecto a este archivo (no al CWD).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Archivo de documentos por agente, en el orden de carga.
DOC_FILES: dict[str, str] = {
    "faq": "faq_docs.json",
    "seguimiento": "seguimiento_docs.json",
}

# Agentes válidos, derivados de DOC_FILES para que no puedan divergir.
AGENT_IDS: tuple[str, ...] = tuple(DOC_FILES)

_FIELDS: tuple[str, ...] = ("id", "agent_id", "title", "text")


class CorpusError(Exception):
    """El corpus en disco no existe o no cumple el formato esperado."""


@dataclass(frozen=True)
class Document:
    id: str
    agent_id: str
    title: str
    text: str


def _read_json_list(path: Path) -> list[Any]:
    if not path.is_file():
        raise CorpusError(f"{path.name}: no existe el archivo ({path})")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"{path.name}: el archivo no está codificado en UTF-8 ({exc})") from exc
    except OSError as exc:
        raise CorpusError(f"{path.name}: no se pudo leer el archivo ({exc})") from exc
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path.name}: JSON inválido ({exc})") from exc
    if not isinstance(raw, list):
        raise CorpusError(f"{path.name}: se esperaba una lista JSON, se recibió {type(raw).__name__}")
    return raw


def _parse_document(entry: Any, index: int, file_name: str, expected_agent: str) -> Document:
    if not isinstance(entry, dict):
        raise CorpusError(f"{file_name}: la entrada #{index} no es un objeto JSON")
    label = f"id={entry['id']!r}" if isinstance(entry.get("id"), str) else f"entrada #{index}"
    for field in _FIELDS:
        if field not in entry:
            raise CorpusError(f"{file_name} ({label}): falta el campo {field!r}")
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise CorpusError(
                f"{file_name} ({label}): el campo {field!r} debe ser un texto no vacío, "
                f"se recibió {value!r}"
            )
    agent_id = entry["agent_id"]
    if agent_id not in AGENT_IDS:
        raise CorpusError(
            f"{file_name} ({label}): agent_id {agent_id!r} desconocido; válidos: {AGENT_IDS}"
        )
    if agent_id != expected_agent:
        raise CorpusError(
            f"{file_name} ({label}): agent_id {agent_id!r} no corresponde al archivo "
            f"(se esperaba {expected_agent!r})"
        )
    return Document(id=entry["id"], agent_id=agent_id, title=entry["title"], text=entry["text"])


def load_documents(data_dir: Path = DEFAULT_DATA_DIR) -> list[Document]:
    """Carga los documentos de todos los agentes (faq primero, luego seguimiento).

    Lanza `CorpusError` si falta un archivo, el JSON es inválido o alguna entrada
    no cumple el esquema `{id, agent_id, title, text}`, o si hay ids repetidos.
    """
    documents: list[Document] = []
    seen: dict[str, str] = {}
    for agent_id, file_name in DOC_FILES.items():
        entries = _read_json_list(Path(data_dir) / file_name)
        for index, entry in enumerate(entries):
            doc = _parse_document(entry, index, file_name, agent_id)
            if doc.id in seen:
                raise CorpusError(
                    f"{file_name}: id duplicado {doc.id!r} (ya definido en {seen[doc.id]})"
                )
            seen[doc.id] = file_name
            documents.append(doc)
    return documents
