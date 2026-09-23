"""Evaluación de calidad con LLM real (generador + juez) sobre el golden set.

Solo se ejecuta con `pytest -m llm_eval` (el `addopts` de pyproject.toml la excluye
por defecto). Necesita un servidor compatible con OpenAI (Ollama) con los modelos
de `Settings` descargados; si no responde, la prueba se salta con el motivo, sin
fallar ni esperar al timeout de generación: `check_available` solo lista modelos y
tiene su propio timeout corto.

Corre la suite en modo `isolated` con `RUNS_PER_CASE` ejecuciones por caso y afirma
que la proporción de casos que pasan alcanza `PASS_RATE_THRESHOLD` (con la misma
regla de redondeo a dos decimales que `meets_pass_rate` aplica a cada caso). El
reporte Markdown completo se escribe en el `tmp_path` de la prueba para inspección.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_lab.__main__ import LLMUnavailableError, _agents, _connect_llms
from rag_lab.config import Settings
from rag_lab.corpus import load_golden_set
from rag_lab.evaluation import (
    CaseResult,
    RunError,
    RunOutcome,
    deterministic_metrics,
    meets_pass_rate,
    run_suite,
)
from rag_lab.judge import Judge
from rag_lab.llm import LLMClient
from rag_lab.report import ModeReport, format_rate, render_markdown
from rag_lab.store import VectorStore

pytestmark = pytest.mark.llm_eval


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session")
def llm_clients(settings: Settings) -> tuple[LLMClient, LLMClient]:
    """(generador, cliente del juez) ya comprobados; si el LLM no responde, skip.

    Es de ámbito sesión (como `indexed_store`) y va primero en la firma de la prueba
    para que el skip ocurra antes de indexar el corpus.
    """
    try:
        return _connect_llms(settings)
    except LLMUnavailableError as exc:
        pytest.skip(f"LLM no disponible, se omite la evaluación con LLM real: {exc}")


def _describe_run(outcome: RunOutcome, ok: bool) -> str:
    mark = "ok" if ok else "FALLA"
    if isinstance(outcome, RunError):
        return f"{mark} error ({outcome.stage}): {outcome.message}"
    return (
        f"{mark} faithfulness={outcome.faithfulness.score:.2f} "
        f"relevance={outcome.relevance.score:.2f} "
        f"abstention={outcome.abstention.score:.2f}"
    )


def _summary(results: Sequence[CaseResult]) -> str:
    """Una línea por caso con su tasa de éxito y el detalle de cada ejecución."""
    lines = []
    for r in results:
        lines.append(
            f"  {r.case_id} [{r.case.agent_id}, esperado {r.case.expected_behavior}] "
            f"tasa {format_rate(sum(r.run_passed), len(r.run_passed))} "
            f"→ {'pasa' if r.passed else 'NO PASA'}"
        )
        for i, (outcome, ok) in enumerate(zip(r.verdicts, r.run_passed, strict=True), start=1):
            lines.append(f"      ejecución {i}: {_describe_run(outcome, ok)}")
    return "\n".join(lines)


def test_isolated_suite_pass_rate_meets_threshold_with_real_judge(
    llm_clients: tuple[LLMClient, LLMClient],
    indexed_store: VectorStore,
    settings: Settings,
    tmp_path: Path,
) -> None:
    generator, judge_llm = llm_clients
    cases = load_golden_set()

    started = time.perf_counter()
    results = run_suite(
        _agents(indexed_store, settings, generator, isolated=True),
        Judge(judge_llm),
        cases,
        runs=settings.runs_per_case,
        pass_rate_threshold=settings.pass_rate_threshold,
    )
    elapsed = time.perf_counter() - started

    # Reporte completo para inspección (métricas deterministas de ambos modos, sin LLM).
    report_path = tmp_path / "llm_eval_report.md"
    report_path.write_text(
        render_markdown(
            ModeReport(
                metrics=deterministic_metrics(
                    _agents(indexed_store, settings, None, isolated=False), cases
                )
            ),
            ModeReport(
                metrics=deterministic_metrics(
                    _agents(indexed_store, settings, None, isolated=True), cases
                ),
                llm_results=results,
            ),
        ),
        encoding="utf-8",
    )

    passed = sum(r.passed for r in results)
    summary = (
        f"Casos que pasan (isolated, juez {settings.judge_model!r}, generador "
        f"{settings.generator_model!r}, {settings.runs_per_case} ejecuciones/caso): "
        f"{format_rate(passed, len(results))}; umbral {settings.pass_rate_threshold}. "
        f"Duración {elapsed:.0f} s. Reporte: {report_path}\n{_summary(results)}"
    )
    print(summary)
    assert meets_pass_rate(passed, len(results), settings.pass_rate_threshold), summary
