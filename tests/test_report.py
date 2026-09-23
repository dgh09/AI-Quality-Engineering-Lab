"""Reporte Markdown antes (shared) / después (isolated).

Todo se construye a mano (métricas y `CaseResult` fijos): el reporte es una función
pura de sus entradas, sin store ni LLM.
"""

from __future__ import annotations

import pytest

from rag_lab.corpus import GoldenCase
from rag_lab.evaluation import CaseResult, DeterministicMetrics, RunError
from rag_lab.judge import Score, Verdict
from rag_lab.report import (
    JUSTIFICATION_MAX_CHARS,
    ModeReport,
    format_rate,
    render_markdown,
)

_BEFORE_METRICS = DeterministicMetrics(
    total_cases=12,
    contaminated_case_ids=("faq-cross-01", "seg-cross-01"),
    abstention_mismatch_case_ids=("faq-cross-01",),
)
_AFTER_METRICS = DeterministicMetrics(
    total_cases=12, contaminated_case_ids=(), abstention_mismatch_case_ids=()
)

_ANSWER_CASE = GoldenCase(
    id="faq-in-01",
    agent_id="faq",
    question="How long do I have to return an item?",
    category="in_domain",
    expected_behavior="answer",
    expected_source_ids=("faq-02",),
)
_ABSTAIN_CASE = GoldenCase(
    id="faq-cross-01",
    agent_id="faq",
    question="How many vacation days do employees get?",
    category="cross_domain",
    expected_behavior="abstain",
    expected_source_ids=(),
)


def _verdict(
    faithfulness: float = 1.0,
    relevance: float = 1.0,
    abstention: float = 1.0,
    *,
    faithfulness_text: str = "All claims are in the context.",
    relevance_text: str = "It answers the question.",
    abstention_text: str = "Correct choice.",
) -> Verdict:
    return Verdict(
        faithfulness=Score(faithfulness, faithfulness_text),
        relevance=Score(relevance, relevance_text),
        abstention=Score(abstention, abstention_text),
    )


def _result(
    case: GoldenCase, outcomes: tuple[Verdict | RunError, ...], run_passed: tuple[bool, ...]
) -> CaseResult:
    successes = sum(run_passed)
    return CaseResult(
        case=case,
        pass_rate=successes / len(run_passed),
        passed=successes * 3 >= 2 * len(run_passed),
        verdicts=outcomes,
        run_passed=run_passed,
    )


def _deterministic_only() -> tuple[ModeReport, ModeReport]:
    return ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS)


def _llm_section(report: str) -> str:
    assert "## Evaluación con juez LLM" in report
    return report.split("## Evaluación con juez LLM", 1)[1]


def _row_for(section: str, case_id: str) -> str:
    rows = [line for line in section.splitlines() if line.startswith(f"| {case_id} |")]
    assert len(rows) == 1, rows
    return rows[0]


# --- Formato de tasas ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "total", "expected"),
    [(2, 12, "16.7% (2/12)"), (0, 12, "0.0% (0/12)"), (12, 12, "100.0% (12/12)"), (2, 3, "66.7% (2/3)")],
)
def test_format_rate_uses_one_decimal_percent_and_fraction(count, total, expected):
    assert format_rate(count, total) == expected


# --- Sección determinista --------------------------------------------------------------


def test_report_names_both_modes():
    report = render_markdown(*_deterministic_only())
    assert "Antes (shared)" in report
    assert "Después (isolated)" in report


def test_deterministic_table_shows_expected_rates_for_each_mode():
    report = render_markdown(*_deterministic_only())
    assert "| Métrica | Antes (shared) | Después (isolated) |" in report
    assert "| Tasa de contaminación | 16.7% (2/12) | 0.0% (0/12) |" in report
    assert "| Exactitud de abstención | 91.7% (11/12) | 100.0% (12/12) |" in report


def test_deterministic_table_lists_offending_case_ids_and_dash_when_none():
    report = render_markdown(*_deterministic_only())
    assert "| Casos contaminados | faq-cross-01, seg-cross-01 | — |" in report
    assert "| Casos con abstención incorrecta | faq-cross-01 | — |" in report


def test_report_explains_that_contamination_is_measured_on_retrieval_not_by_judge():
    report = render_markdown(*_deterministic_only())
    assert "recuperación" in report
    assert "fidelidad" in report
    assert "contexto contaminado" in report


# --- Sección LLM -----------------------------------------------------------------------


def test_llm_section_is_omitted_without_llm_results():
    report = render_markdown(*_deterministic_only())
    assert "juez LLM" not in report
    assert "Tasa de éxito" not in report


def test_llm_section_is_omitted_when_results_are_empty():
    report = render_markdown(ModeReport(_BEFORE_METRICS, ()), ModeReport(_AFTER_METRICS, ()))
    assert "juez LLM" not in report


def test_llm_section_has_one_subsection_per_mode_with_case_rows():
    passing = _result(_ANSWER_CASE, (_verdict(),) * 3, (True, True, True))
    failing = _result(
        _ABSTAIN_CASE,
        (_verdict(abstention=0.0, abstention_text="It answered from the context."),) * 3,
        (False, False, False),
    )
    report = render_markdown(
        ModeReport(_BEFORE_METRICS, (passing, failing)),
        ModeReport(_AFTER_METRICS, (passing,)),
    )
    section = _llm_section(report)
    assert "### Antes (shared)" in section
    assert "### Después (isolated)" in section
    before, after = section.split("### Después (isolated)", 1)
    assert (
        "| Caso | Agente | Esperado | Tasa de éxito | Resultado | Justificación del juez |"
        in before
    )
    assert _row_for(before, "faq-in-01").startswith(
        "| faq-in-01 | faq | answer | 100.0% (3/3) | ✅ pasa |"
    )
    assert _row_for(before, "faq-cross-01").startswith(
        "| faq-cross-01 | faq | abstain | 0.0% (0/3) | ❌ no pasa |"
    )
    assert "Casos que pasan: 50.0% (1/2)" in before
    assert "Casos que pasan: 100.0% (1/1)" in after


def test_llm_section_marks_mode_without_results():
    passing = _result(_ANSWER_CASE, (_verdict(),) * 3, (True, True, True))
    report = render_markdown(ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (passing,)))
    before = _llm_section(report).split("### Después (isolated)", 1)[0]
    assert "Sin resultados de evaluación LLM" in before


def test_two_of_three_runs_is_reported_as_passing_rate():
    result = _result(
        _ANSWER_CASE,
        (_verdict(), _verdict(relevance=0.2), _verdict()),
        (True, False, True),
    )
    report = render_markdown(ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (result,)))
    row = _row_for(_llm_section(report), "faq-in-01")
    assert "| 66.7% (2/3) | ✅ pasa |" in row


# --- Justificación elegida ---------------------------------------------------------------


def test_failed_answer_case_shows_lowest_decisive_dimension_of_first_failed_run():
    result = _result(
        _ANSWER_CASE,
        (
            _verdict(),
            _verdict(faithfulness=0.9, relevance=0.3, relevance_text="Off-topic reply."),
            _verdict(faithfulness=0.1, faithfulness_text="Invented a fee."),
        ),
        (True, False, False),
    )
    report = render_markdown(ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (result,)))
    row = _row_for(_llm_section(report), "faq-in-01")
    assert row.endswith("| relevance 0.30: Off-topic reply. |")


def test_passing_answer_case_shows_decisive_dimension_of_first_passed_run():
    result = _result(
        _ANSWER_CASE,
        (
            _verdict(relevance=0.1),
            _verdict(faithfulness=0.8, faithfulness_text="Mostly grounded."),
            _verdict(),
        ),
        (False, True, True),
    )
    report = render_markdown(ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (result,)))
    row = _row_for(_llm_section(report), "faq-in-01")
    assert row.endswith("| faithfulness 0.80: Mostly grounded. |")


def test_answer_case_tie_between_dimensions_prefers_faithfulness():
    result = _result(_ANSWER_CASE, (_verdict(),), (True,))
    report = render_markdown(ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (result,)))
    row = _row_for(_llm_section(report), "faq-in-01")
    assert row.endswith("| faithfulness 1.00: All claims are in the context. |")


def test_abstain_case_shows_abstention_justification():
    result = _result(
        _ABSTAIN_CASE,
        (_verdict(faithfulness=0.0, abstention=0.0, abstention_text="It answered anyway."),),
        (False,),
    )
    report = render_markdown(ModeReport(_BEFORE_METRICS, (result,)), ModeReport(_AFTER_METRICS))
    row = _row_for(_llm_section(report), "faq-cross-01")
    assert row.endswith("| abstention 0.00: It answered anyway. |")


def test_run_error_shows_stage_and_message():
    result = _result(
        _ANSWER_CASE,
        (RunError(stage="judge", message="salida inválida"), _verdict(), _verdict()),
        (False, True, True),
    )
    failing = _result(
        _ABSTAIN_CASE,
        (RunError(stage="agent", message="timeout de conexión"),) * 3,
        (False, False, False),
    )
    report = render_markdown(
        ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (result, failing))
    )
    section = _llm_section(report)
    assert _row_for(section, "faq-cross-01").endswith(
        "| error (agent): timeout de conexión |"
    )
    # El caso pasa: se muestra la justificación de la primera ejecución aprobada, y el
    # error queda contado en el resumen del modo (1 + 3 de 6 ejecuciones).
    row = _row_for(section, "faq-in-01")
    assert row.endswith("| faithfulness 1.00: All claims are in the context. |")
    assert "Ejecuciones con error: 4 de 6" in section


# --- Escape y truncado -------------------------------------------------------------------


def test_justification_with_pipe_and_newline_does_not_break_the_table():
    result = _result(
        _ABSTAIN_CASE,
        (_verdict(abstention=0.0, abstention_text="Answered a | b\nfrom context.\r\nBad."),),
        (False,),
    )
    report = render_markdown(ModeReport(_BEFORE_METRICS, (result,)), ModeReport(_AFTER_METRICS))
    row = _row_for(_llm_section(report), "faq-cross-01")
    assert row.endswith("| abstention 0.00: Answered a \\| b from context. Bad. |")
    # 6 columnas → 7 separadores sin escapar.
    assert row.replace("\\|", "").count("|") == 7


def test_long_justification_is_truncated_with_ellipsis():
    long_text = "word " * 100
    result = _result(_ABSTAIN_CASE, (_verdict(abstention=0.0, abstention_text=long_text),), (False,))
    report = render_markdown(ModeReport(_BEFORE_METRICS, (result,)), ModeReport(_AFTER_METRICS))
    row = _row_for(_llm_section(report), "faq-cross-01")
    cell = row.rsplit(" | ", 1)[1].removesuffix(" |")
    assert cell.endswith("…")
    assert JUSTIFICATION_MAX_CHARS - 5 <= len(cell) <= JUSTIFICATION_MAX_CHARS
    assert not cell.endswith(" …")


def test_long_run_error_message_is_truncated():
    result = _result(_ANSWER_CASE, (RunError(stage="judge", message="x" * 500),), (False,))
    report = render_markdown(ModeReport(_BEFORE_METRICS), ModeReport(_AFTER_METRICS, (result,)))
    row = _row_for(_llm_section(report), "faq-in-01")
    cell = row.rsplit(" | ", 1)[1].removesuffix(" |")
    assert cell.startswith("error (judge): xxx")
    assert cell.endswith("…")
    assert len(cell) == JUSTIFICATION_MAX_CHARS


# --- Determinismo ------------------------------------------------------------------------


def test_render_is_deterministic_and_ends_with_single_newline():
    result = _result(_ANSWER_CASE, (_verdict(),) * 3, (True, True, True))
    before = ModeReport(_BEFORE_METRICS, (result,))
    after = ModeReport(_AFTER_METRICS, (result,))
    first = render_markdown(before, after)
    assert first == render_markdown(before, after)
    assert first.endswith("\n") and not first.endswith("\n\n")
