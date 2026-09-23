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


# --- Golden set --------------------------------------------------------------

GOLDEN_SET_FILE = "golden_set.json"

# Categorías de caso y el comportamiento que cada una exige al agente.
CATEGORIES: tuple[str, ...] = ("in_domain", "cross_domain", "out_of_domain")
EXPECTED_BEHAVIORS: tuple[str, ...] = ("answer", "abstain")
_BEHAVIOR_BY_CATEGORY: dict[str, str] = {
    "in_domain": "answer",
    "cross_domain": "abstain",
    "out_of_domain": "abstain",
}

_GOLDEN_TEXT_FIELDS: tuple[str, ...] = (
    "id",
    "agent_id",
    "question",
    "category",
    "expected_behavior",
)


@dataclass(frozen=True)
class GoldenCase:
    id: str
    agent_id: str
    question: str
    category: str
    expected_behavior: str
    # in_domain: documentos del propio agente que contienen la respuesta.
    # cross_domain: documentos del OTRO agente que serían la fuente contaminante.
    # out_of_domain: vacío.
    expected_source_ids: tuple[str, ...]


def _parse_golden_case(entry: Any, index: int, file_name: str) -> GoldenCase:
    if not isinstance(entry, dict):
        raise CorpusError(f"{file_name}: la entrada #{index} no es un objeto JSON")
    label = f"id={entry['id']!r}" if isinstance(entry.get("id"), str) else f"entrada #{index}"
    for field in (*_GOLDEN_TEXT_FIELDS, "expected_source_ids"):
        if field not in entry:
            raise CorpusError(f"{file_name} ({label}): falta el campo {field!r}")
    for field in _GOLDEN_TEXT_FIELDS:
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise CorpusError(
                f"{file_name} ({label}): el campo {field!r} debe ser un texto no vacío, "
                f"se recibió {value!r}"
            )
    sources = entry["expected_source_ids"]
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise CorpusError(
            f"{file_name} ({label}): el campo 'expected_source_ids' debe ser una lista de "
            f"textos, se recibió {sources!r}"
        )
    agent_id, category, behavior = entry["agent_id"], entry["category"], entry["expected_behavior"]
    if agent_id not in AGENT_IDS:
        raise CorpusError(
            f"{file_name} ({label}): agent_id {agent_id!r} desconocido; válidos: {AGENT_IDS}"
        )
    if category not in CATEGORIES:
        raise CorpusError(
            f"{file_name} ({label}): category {category!r} desconocida; válidas: {CATEGORIES}"
        )
    if behavior not in EXPECTED_BEHAVIORS:
        raise CorpusError(
            f"{file_name} ({label}): expected_behavior {behavior!r} desconocido; "
            f"válidos: {EXPECTED_BEHAVIORS}"
        )
    if _BEHAVIOR_BY_CATEGORY[category] != behavior:
        raise CorpusError(
            f"{file_name} ({label}): la categoría {category!r} es incompatible con "
            f"expected_behavior {behavior!r} (se esperaba {_BEHAVIOR_BY_CATEGORY[category]!r})"
        )
    return GoldenCase(
        id=entry["id"],
        agent_id=agent_id,
        question=entry["question"],
        category=category,
        expected_behavior=behavior,
        expected_source_ids=tuple(sources),
    )


def _check_sources(case: GoldenCase, doc_agents: dict[str, str], file_name: str) -> None:
    label = f"id={case.id!r}"
    for source_id in case.expected_source_ids:
        if source_id not in doc_agents:
            raise CorpusError(
                f"{file_name} ({label}): expected_source_id {source_id!r} no existe en el corpus"
            )
    if case.category == "out_of_domain":
        if case.expected_source_ids:
            raise CorpusError(
                f"{file_name} ({label}): un caso out_of_domain no debe tener "
                f"expected_source_ids, se recibió {list(case.expected_source_ids)!r}"
            )
        return
    if not case.expected_source_ids:
        raise CorpusError(
            f"{file_name} ({label}): un caso {case.category} requiere al menos un "
            f"expected_source_id"
        )
    same_agent = case.category == "in_domain"
    for source_id in case.expected_source_ids:
        source_agent = doc_agents[source_id]
        if (source_agent == case.agent_id) != same_agent:
            rule = "del mismo agente" if same_agent else "del otro agente"
            raise CorpusError(
                f"{file_name} ({label}): {source_id!r} es del agente {source_agent!r}; "
                f"las fuentes de un caso {case.category} deben ser {rule} "
                f"(agent_id={case.agent_id!r})"
            )


def load_golden_set(
    path: Path = DEFAULT_DATA_DIR / GOLDEN_SET_FILE,
    documents: list[Document] | None = None,
) -> list[GoldenCase]:
    """Carga y valida el golden set contra el corpus.

    Si no se pasan `documents`, se cargan con `load_documents()`. Lanza
    `CorpusError` si el archivo no cumple el esquema, hay ids repetidos, la
    categoría no concuerda con `expected_behavior` o las fuentes esperadas no
    existen o no pertenecen al agente que exige la categoría.
    """
    path = Path(path)
    if documents is None:
        documents = load_documents()
    doc_agents = {doc.id: doc.agent_id for doc in documents}
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for index, entry in enumerate(_read_json_list(path)):
        case = _parse_golden_case(entry, index, path.name)
        if case.id in seen:
            raise CorpusError(f"{path.name}: id duplicado {case.id!r}")
        seen.add(case.id)
        _check_sources(case, doc_agents, path.name)
        cases.append(case)
    return cases
