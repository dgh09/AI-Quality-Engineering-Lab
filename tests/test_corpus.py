import json
from pathlib import Path
from typing import Any

import pytest

from rag_lab.corpus import AGENT_IDS, CorpusError, Document, load_documents

FAQ_FILE = "faq_docs.json"
SEG_FILE = "seguimiento_docs.json"


def _doc(doc_id: str, agent_id: str) -> dict[str, Any]:
    return {
        "id": doc_id,
        "agent_id": agent_id,
        "title": f"Title {doc_id}",
        "text": f"Some text for {doc_id}.",
    }


def _write(data_dir: Path, faq: Any, seg: Any) -> None:
    for name, payload in ((FAQ_FILE, faq), (SEG_FILE, seg)):
        content = payload if isinstance(payload, str) else json.dumps(payload)
        (data_dir / name).write_text(content, encoding="utf-8")


@pytest.fixture
def valid_faq() -> list[dict[str, Any]]:
    return [_doc("faq-01", "faq"), _doc("faq-02", "faq")]


@pytest.fixture
def valid_seg() -> list[dict[str, Any]]:
    return [_doc("seg-01", "seguimiento"), _doc("seg-02", "seguimiento")]


# --- Corpus real -----------------------------------------------------------


def test_agent_ids_are_faq_and_seguimiento() -> None:
    assert AGENT_IDS == ("faq", "seguimiento")


def test_real_corpus_has_eight_docs_per_agent() -> None:
    docs = load_documents()

    assert all(isinstance(d, Document) for d in docs)
    assert sum(d.agent_id == "faq" for d in docs) == 8
    assert sum(d.agent_id == "seguimiento" for d in docs) == 8
    assert len(docs) == 16


def test_real_corpus_ids_are_unique_and_follow_naming() -> None:
    docs = load_documents()
    ids = [d.id for d in docs]

    assert len(ids) == len(set(ids))
    assert ids == [f"faq-{i:02d}" for i in range(1, 9)] + [f"seg-{i:02d}" for i in range(1, 9)]


def test_real_corpus_agent_id_matches_source_file() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    for name, agent_id in ((FAQ_FILE, "faq"), (SEG_FILE, "seguimiento")):
        raw = json.loads((data_dir / name).read_text(encoding="utf-8"))
        assert len(raw) == 8
        assert all(entry["agent_id"] == agent_id for entry in raw)


def test_real_corpus_loads_faq_first_then_seguimiento() -> None:
    agent_ids = [d.agent_id for d in load_documents()]

    assert agent_ids == ["faq"] * 8 + ["seguimiento"] * 8


def test_real_corpus_fields_are_non_empty() -> None:
    for doc in load_documents():
        assert doc.title.strip()
        assert doc.text.strip()


def test_default_data_dir_does_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert len(load_documents()) == 16


def test_document_is_frozen() -> None:
    doc = Document(id="x", agent_id="faq", title="t", text="b")

    with pytest.raises(AttributeError):
        doc.text = "changed"  # type: ignore[misc]


# --- Carga desde un directorio arbitrario -----------------------------------


def test_loads_valid_corpus_from_custom_dir(
    tmp_path: Path, valid_faq: list[dict[str, Any]], valid_seg: list[dict[str, Any]]
) -> None:
    _write(tmp_path, valid_faq, valid_seg)

    docs = load_documents(tmp_path)

    assert [d.id for d in docs] == ["faq-01", "faq-02", "seg-01", "seg-02"]
    assert docs[0] == Document(
        id="faq-01", agent_id="faq", title="Title faq-01", text="Some text for faq-01."
    )


# --- Casos de error ----------------------------------------------------------


def test_missing_file_raises_corpus_error(
    tmp_path: Path, valid_faq: list[dict[str, Any]]
) -> None:
    (tmp_path / FAQ_FILE).write_text(json.dumps(valid_faq), encoding="utf-8")

    with pytest.raises(CorpusError, match="no existe") as exc_info:
        load_documents(tmp_path)
    assert SEG_FILE in str(exc_info.value)


def test_invalid_json_raises_corpus_error(
    tmp_path: Path, valid_seg: list[dict[str, Any]]
) -> None:
    _write(tmp_path, "[{not valid json", valid_seg)

    with pytest.raises(CorpusError, match="JSON inválido") as exc_info:
        load_documents(tmp_path)
    assert FAQ_FILE in str(exc_info.value)


def test_json_that_is_not_a_list_raises_corpus_error(
    tmp_path: Path, valid_seg: list[dict[str, Any]]
) -> None:
    _write(tmp_path, {"id": "faq-01"}, valid_seg)

    with pytest.raises(CorpusError, match="se esperaba una lista") as exc_info:
        load_documents(tmp_path)
    assert FAQ_FILE in str(exc_info.value)


def test_entry_that_is_not_an_object_raises_corpus_error(
    tmp_path: Path, valid_faq: list[dict[str, Any]], valid_seg: list[dict[str, Any]]
) -> None:
    _write(tmp_path, [*valid_faq, "just a string"], valid_seg)

    with pytest.raises(CorpusError, match="no es un objeto") as exc_info:
        load_documents(tmp_path)
    assert FAQ_FILE in str(exc_info.value)


@pytest.mark.parametrize("field", ["id", "agent_id", "title", "text"])
def test_missing_field_raises_corpus_error(
    tmp_path: Path,
    valid_faq: list[dict[str, Any]],
    valid_seg: list[dict[str, Any]],
    field: str,
) -> None:
    del valid_seg[1][field]
    _write(tmp_path, valid_faq, valid_seg)

    with pytest.raises(CorpusError, match=f"falta el campo '{field}'") as exc_info:
        load_documents(tmp_path)
    assert SEG_FILE in str(exc_info.value)


@pytest.mark.parametrize("value", ["", "   ", 42, None])
def test_empty_or_non_string_text_raises_corpus_error(
    tmp_path: Path,
    valid_faq: list[dict[str, Any]],
    valid_seg: list[dict[str, Any]],
    value: Any,
) -> None:
    valid_faq[1]["text"] = value
    _write(tmp_path, valid_faq, valid_seg)

    with pytest.raises(
        CorpusError, match="el campo 'text' debe ser un texto no vacío"
    ) as exc_info:
        load_documents(tmp_path)
    assert FAQ_FILE in str(exc_info.value)
    assert "faq-02" in str(exc_info.value)


def test_unknown_agent_id_raises_corpus_error(
    tmp_path: Path, valid_faq: list[dict[str, Any]], valid_seg: list[dict[str, Any]]
) -> None:
    valid_faq[0]["agent_id"] = "billing"
    _write(tmp_path, valid_faq, valid_seg)

    with pytest.raises(CorpusError, match="agent_id 'billing' desconocido") as exc_info:
        load_documents(tmp_path)
    assert "faq-01" in str(exc_info.value)


def test_agent_id_not_matching_file_raises_corpus_error(
    tmp_path: Path, valid_faq: list[dict[str, Any]], valid_seg: list[dict[str, Any]]
) -> None:
    valid_faq[1]["agent_id"] = "seguimiento"
    _write(tmp_path, valid_faq, valid_seg)

    with pytest.raises(CorpusError, match="no corresponde al archivo") as exc_info:
        load_documents(tmp_path)
    assert FAQ_FILE in str(exc_info.value)
    assert "faq-02" in str(exc_info.value)


def test_duplicate_id_within_a_file_raises_corpus_error(
    tmp_path: Path, valid_faq: list[dict[str, Any]], valid_seg: list[dict[str, Any]]
) -> None:
    valid_faq[1]["id"] = "faq-01"
    _write(tmp_path, valid_faq, valid_seg)

    with pytest.raises(CorpusError, match="id duplicado 'faq-01'"):
        load_documents(tmp_path)


def test_duplicate_id_across_files_raises_corpus_error(
    tmp_path: Path, valid_faq: list[dict[str, Any]], valid_seg: list[dict[str, Any]]
) -> None:
    valid_seg[0]["id"] = "faq-02"
    _write(tmp_path, valid_faq, valid_seg)

    with pytest.raises(CorpusError, match="id duplicado 'faq-02'") as exc_info:
        load_documents(tmp_path)
    assert SEG_FILE in str(exc_info.value)


def test_non_utf8_file_raises_corpus_error(
    tmp_path: Path, valid_seg: list[dict[str, Any]]
) -> None:
    _write(tmp_path, "[]", valid_seg)
    (tmp_path / FAQ_FILE).write_bytes(b"\xff\xfe\x00")

    with pytest.raises(CorpusError, match="UTF-8") as exc_info:
        load_documents(tmp_path)
    assert FAQ_FILE in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
