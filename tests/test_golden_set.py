import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rag_lab.corpus import (
    CATEGORIES,
    EXPECTED_BEHAVIORS,
    CorpusError,
    Document,
    GoldenCase,
    load_documents,
    load_golden_set,
)

DOCS: list[Document] = [
    Document(id="faq-01", agent_id="faq", title="t", text="b"),
    Document(id="faq-02", agent_id="faq", title="t", text="b"),
    Document(id="seg-01", agent_id="seguimiento", title="t", text="b"),
    Document(id="seg-02", agent_id="seguimiento", title="t", text="b"),
]


def _case(
    case_id: str,
    agent_id: str = "faq",
    category: str = "in_domain",
    expected_behavior: str = "answer",
    sources: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "agent_id": agent_id,
        "question": f"Question {case_id}?",
        "category": category,
        "expected_behavior": expected_behavior,
        "expected_source_ids": ["faq-01"] if sources is None else sources,
    }


def _valid_cases() -> list[dict[str, Any]]:
    return [
        _case("gs-01", "faq", "in_domain", "answer", ["faq-01"]),
        _case("gs-02", "seguimiento", "in_domain", "answer", ["seg-01", "seg-02"]),
        _case("gs-03", "faq", "cross_domain", "abstain", ["seg-01"]),
        _case("gs-04", "seguimiento", "cross_domain", "abstain", ["faq-02"]),
        _case("gs-05", "faq", "out_of_domain", "abstain", []),
    ]


def _load(tmp_path: Path, payload: Any) -> list[GoldenCase]:
    path = tmp_path / "golden_set.json"
    content = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(content, encoding="utf-8")
    return load_golden_set(path, documents=DOCS)


# --- Golden set real ---------------------------------------------------------


def test_constants() -> None:
    assert CATEGORIES == ("in_domain", "cross_domain", "out_of_domain")
    assert EXPECTED_BEHAVIORS == ("answer", "abstain")


def test_real_golden_set_has_twelve_cases_with_expected_distribution() -> None:
    cases = load_golden_set()

    assert len(cases) == 12
    assert all(isinstance(c, GoldenCase) for c in cases)
    assert Counter((c.agent_id, c.category) for c in cases) == {
        ("faq", "in_domain"): 4,
        ("seguimiento", "in_domain"): 4,
        ("faq", "cross_domain"): 1,
        ("seguimiento", "cross_domain"): 1,
        ("faq", "out_of_domain"): 1,
        ("seguimiento", "out_of_domain"): 1,
    }
    assert [c.id for c in cases] == [f"gs-{i:02d}" for i in range(1, 13)]


def test_real_golden_set_references_are_valid() -> None:
    docs_by_id = {d.id: d for d in load_documents()}

    for case in load_golden_set():
        assert case.question.strip()
        assert all(src in docs_by_id for src in case.expected_source_ids)
        source_agents = {docs_by_id[src].agent_id for src in case.expected_source_ids}
        if case.category == "in_domain":
            assert case.expected_behavior == "answer"
            assert source_agents == {case.agent_id}
        elif case.category == "cross_domain":
            assert case.expected_behavior == "abstain"
            assert source_agents and case.agent_id not in source_agents
        else:
            assert case.expected_behavior == "abstain"
            assert case.expected_source_ids == ()


def test_golden_case_is_frozen_and_sources_are_tuple(tmp_path: Path) -> None:
    case = _load(tmp_path, _valid_cases())[1]

    assert case.expected_source_ids == ("seg-01", "seg-02")
    with pytest.raises(AttributeError):
        case.question = "changed"  # type: ignore[misc]


def test_loads_valid_golden_set_from_custom_path(tmp_path: Path) -> None:
    cases = _load(tmp_path, _valid_cases())

    assert cases[0] == GoldenCase(
        id="gs-01",
        agent_id="faq",
        question="Question gs-01?",
        category="in_domain",
        expected_behavior="answer",
        expected_source_ids=("faq-01",),
    )


# --- Casos de error ----------------------------------------------------------


def test_missing_file_raises_corpus_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="no existe"):
        load_golden_set(tmp_path / "golden_set.json", documents=DOCS)


def test_invalid_json_raises_corpus_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="JSON inválido"):
        _load(tmp_path, "[{oops")


def test_not_a_list_raises_corpus_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="se esperaba una lista"):
        _load(tmp_path, {"id": "gs-01"})


def test_entry_not_object_raises_corpus_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="la entrada #5 no es un objeto"):
        _load(tmp_path, [*_valid_cases(), "text"])


@pytest.mark.parametrize(
    "field",
    ["id", "agent_id", "question", "category", "expected_behavior", "expected_source_ids"],
)
def test_missing_field_raises_corpus_error(tmp_path: Path, field: str) -> None:
    cases = _valid_cases()
    del cases[2][field]

    with pytest.raises(CorpusError, match=f"falta el campo '{field}'"):
        _load(tmp_path, cases)


@pytest.mark.parametrize("value", ["", "  ", 3, None])
def test_empty_question_raises_corpus_error(tmp_path: Path, value: Any) -> None:
    cases = _valid_cases()
    cases[1]["question"] = value

    with pytest.raises(CorpusError, match="gs-02.*'question' debe ser un texto no vacío"):
        _load(tmp_path, cases)


@pytest.mark.parametrize("value", ["faq-01", ["faq-01", 7], None])
def test_sources_not_list_of_strings_raises_corpus_error(tmp_path: Path, value: Any) -> None:
    cases = _valid_cases()
    cases[0]["expected_source_ids"] = value

    with pytest.raises(CorpusError, match="gs-01.*debe ser una lista de textos"):
        _load(tmp_path, cases)


def test_duplicate_id_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[3]["id"] = "gs-01"

    with pytest.raises(CorpusError, match="id duplicado 'gs-01'"):
        _load(tmp_path, cases)


def test_unknown_agent_id_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[0]["agent_id"] = "billing"

    with pytest.raises(CorpusError, match="gs-01.*agent_id 'billing' desconocido"):
        _load(tmp_path, cases)


def test_unknown_category_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[0]["category"] = "trivia"

    with pytest.raises(CorpusError, match="gs-01.*category 'trivia' desconocida"):
        _load(tmp_path, cases)


def test_unknown_expected_behavior_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[0]["expected_behavior"] = "guess"

    with pytest.raises(CorpusError, match="gs-01.*expected_behavior 'guess' desconocido"):
        _load(tmp_path, cases)


@pytest.mark.parametrize(
    ("index", "behavior"),
    [(0, "abstain"), (2, "answer"), (4, "answer")],
)
def test_inconsistent_category_and_behavior_raises_corpus_error(
    tmp_path: Path, index: int, behavior: str
) -> None:
    cases = _valid_cases()
    cases[index]["expected_behavior"] = behavior

    with pytest.raises(
        CorpusError, match=f"gs-0{index + 1}.*es incompatible con expected_behavior '{behavior}'"
    ):
        _load(tmp_path, cases)


def test_unknown_source_id_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[1]["expected_source_ids"] = ["seg-01", "seg-99"]

    with pytest.raises(CorpusError, match="gs-02.*expected_source_id 'seg-99' no existe"):
        _load(tmp_path, cases)


def test_in_domain_without_sources_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[0]["expected_source_ids"] = []

    with pytest.raises(CorpusError, match="gs-01.*in_domain requiere al menos un"):
        _load(tmp_path, cases)


def test_in_domain_with_source_from_other_agent_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[0]["expected_source_ids"] = ["faq-01", "seg-01"]

    with pytest.raises(
        CorpusError, match="gs-01.*'seg-01' es del agente 'seguimiento'.*mismo agente"
    ):
        _load(tmp_path, cases)


def test_cross_domain_without_sources_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[2]["expected_source_ids"] = []

    with pytest.raises(CorpusError, match="gs-03.*cross_domain requiere al menos un"):
        _load(tmp_path, cases)


def test_cross_domain_with_source_from_same_agent_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[2]["expected_source_ids"] = ["faq-01"]

    with pytest.raises(CorpusError, match="gs-03.*'faq-01' es del agente 'faq'.*otro agente"):
        _load(tmp_path, cases)


def test_out_of_domain_with_sources_raises_corpus_error(tmp_path: Path) -> None:
    cases = _valid_cases()
    cases[4]["expected_source_ids"] = ["faq-01"]

    with pytest.raises(CorpusError, match="gs-05.*out_of_domain no debe tener"):
        _load(tmp_path, cases)
