"""Reporte Markdown con los resultados antes (modo shared) y después (modo isolated).

`render_markdown(before, after)` es una función pura: la misma entrada produce
exactamente el mismo texto (sin fechas ni nada dependiente del entorno). La CLI
(T13) la escribe en `reports/report.md` y el README (T15) pega su contenido.

Entrada: un `ModeReport` por modo con las métricas deterministas (siempre) y,
opcionalmente, los resultados de la evaluación con juez LLM (`--with-llm`).

Formato:
- Tasas: `format_rate(k, n)` → porcentaje con un decimal y la fracción exacta,
  p. ej. `16.7% (2/12)`.
- Sección determinista: una tabla con ambos modos en columnas (contaminación,
  exactitud de abstención y los ids de los casos que fallan cada métrica, o `—`),
  seguida de una nota sobre cómo se mide la contaminación.
- Sección LLM ("Evaluación con juez LLM"): solo si algún modo tiene resultados
  (`None` y `()` cuentan como "sin resultados"). Una subsección por modo con una
  fila por caso, en el orden recibido.

Justificación mostrada por caso (una sola, para que la tabla sea legible):
- Ejecución representativa: si el caso pasa, la primera ejecución aprobada; si no,
  la primera ejecución fallida. Así la justificación explica el resultado del caso.
- Si esa ejecución es un `RunError`: `error (<stage>): <mensaje>`.
- Si es un `Verdict`: la dimensión que decide según `Verdict.passed` —`abstention`
  para `abstain`; para `answer`, la de menor score entre `faithfulness` y
  `relevance` (empate → `faithfulness`)—, como `<dimensión> <score 2 decimales>:
  <justificación>`.
- El texto se normaliza a una línea (cualquier espacio en blanco, incluidos los
  saltos de línea, pasa a un espacio), se trunca a `JUSTIFICATION_MAX_CHARS`
  caracteres terminando en `…`, y después se escapa `|` como `\\|` para no romper
  la tabla.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.evaluation import CaseResult, DeterministicMetrics, RunError, RunOutcome
from rag_lab.judge import Score, Verdict

BEFORE_LABEL = "Antes (shared)"
AFTER_LABEL = "Después (isolated)"
# Longitud máxima (antes de escapar `|`) de la celda de justificación, con la elipsis.
JUSTIFICATION_MAX_CHARS = 120
_ELLIPSIS = "…"
_NONE = "—"

_CONTAMINATION_NOTE = (
    "> **Cómo se mide la contaminación.** Es una métrica determinista sobre la "
    "recuperación: un caso está contaminado si el top-k recuperado para su agente "
    "contiene algún chunk de otro agente; no la puntúa el juez. En modo shared "
    "una fidelidad alta del juez no implica ausencia de contaminación: el juez "
    "puntúa la respuesta contra el contexto contaminado que vio el generador, así "
    "que una respuesta copiada de un chunk ajeno puede ser perfectamente \"fiel\"."
)


@dataclass(frozen=True)
class ModeReport:
    """Resultados de un modo (shared o isolated) para el reporte.

    `metrics`: métricas deterministas (sin LLM), siempre presentes.
    `llm_results`: resultados de `run_suite` con juez LLM; `None` si no se ejecutó
    la evaluación con LLM (CLI sin `--with-llm`).
    """

    metrics: DeterministicMetrics
    llm_results: tuple[CaseResult, ...] | None = None


def format_rate(count: int, total: int) -> str:
    """`count/total` como porcentaje con un decimal y la fracción: `16.7% (2/12)`."""
    if total < 1:
        raise ValueError(f"total debe ser >= 1, se recibió {total}")
    return f"{100 * count / total:.1f}% ({count}/{total})"


def _cell(text: str, max_chars: int | None = None) -> str:
    """Texto seguro para una celda de tabla: una línea, truncado y con `|` escapado."""
    flat = " ".join(text.split())
    if max_chars is not None and len(flat) > max_chars:
        flat = flat[: max_chars - len(_ELLIPSIS)].rstrip() + _ELLIPSIS
    return flat.replace("|", "\\|")


def _ids(case_ids: Sequence[str]) -> str:
    return _cell(", ".join(case_ids)) if case_ids else _NONE


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _deterministic_section(before: DeterministicMetrics, after: DeterministicMetrics) -> list[str]:
    def contamination(m: DeterministicMetrics) -> str:
        return format_rate(len(m.contaminated_case_ids), m.total_cases)

    def abstention(m: DeterministicMetrics) -> str:
        return format_rate(m.total_cases - len(m.abstention_mismatch_case_ids), m.total_cases)

    return [
        "## Métricas deterministas (sin LLM)",
        "",
        _row("Métrica", BEFORE_LABEL, AFTER_LABEL),
        _row("---", "---", "---"),
        _row("Tasa de contaminación", contamination(before), contamination(after)),
        _row("Exactitud de abstención", abstention(before), abstention(after)),
        _row("Casos contaminados", _ids(before.contaminated_case_ids), _ids(after.contaminated_case_ids)),
        _row(
            "Casos con abstención incorrecta",
            _ids(before.abstention_mismatch_case_ids),
            _ids(after.abstention_mismatch_case_ids),
        ),
        "",
        _CONTAMINATION_NOTE,
    ]


def _representative(result: CaseResult) -> RunOutcome:
    """Primera ejecución cuyo resultado coincide con el del caso (ver docstring del módulo)."""
    for outcome, ok in zip(result.verdicts, result.run_passed, strict=True):
        if ok == result.passed:
            return outcome
    # Caso aprobado sin ejecuciones aprobadas (umbral 0) o al revés: primera ejecución.
    return result.verdicts[0]


def _justification(result: CaseResult) -> str:
    outcome = _representative(result)
    if isinstance(outcome, RunError):
        text = f"error ({outcome.stage}): {outcome.message}"
    else:
        name, score = _decisive_score(outcome, result.case.expected_behavior)
        text = f"{name} {score.score:.2f}: {score.justification}"
    return _cell(text, JUSTIFICATION_MAX_CHARS)


def _decisive_score(verdict: Verdict, expected_behavior: str) -> tuple[str, Score]:
    if expected_behavior == "abstain":
        return "abstention", verdict.abstention
    if verdict.relevance.score < verdict.faithfulness.score:
        return "relevance", verdict.relevance
    return "faithfulness", verdict.faithfulness


def _llm_mode_section(label: str, results: tuple[CaseResult, ...] | None) -> list[str]:
    lines = [f"### {label}", ""]
    if not results:
        return [*lines, "_Sin resultados de evaluación LLM para este modo._"]
    passed = sum(r.passed for r in results)
    errors = sum(len(r.errors) for r in results)
    runs = sum(len(r.verdicts) for r in results)
    lines += [
        f"Casos que pasan: {format_rate(passed, len(results))} · "
        f"Ejecuciones con error: {errors} de {runs}",
        "",
        _row("Caso", "Agente", "Esperado", "Tasa de éxito", "Resultado", "Justificación del juez"),
        _row("---", "---", "---", "---", "---", "---"),
    ]
    for r in results:
        lines.append(
            _row(
                _cell(r.case_id),
                _cell(r.case.agent_id),
                _cell(r.case.expected_behavior),
                format_rate(sum(r.run_passed), len(r.run_passed)),
                "✅ pasa" if r.passed else "❌ no pasa",
                _justification(r),
            )
        )
    return lines


def _llm_section(before: ModeReport, after: ModeReport) -> list[str]:
    return [
        "## Evaluación con juez LLM",
        "",
        "Tasa de éxito = ejecuciones aprobadas por el juez / ejecuciones del caso. "
        "La justificación corresponde a la primera ejecución con el mismo resultado "
        "que el caso.",
        "",
        *_llm_mode_section(BEFORE_LABEL, before.llm_results),
        "",
        *_llm_mode_section(AFTER_LABEL, after.llm_results),
    ]


def render_markdown(before: ModeReport, after: ModeReport) -> str:
    """Reporte Markdown: métricas deterministas de ambos modos y, si hay resultados
    del juez en algún modo, la tabla de evaluación LLM por caso. Termina en un único
    salto de línea.
    """
    lines = ["# Reporte de evaluación RAG", "", *_deterministic_section(before.metrics, after.metrics)]
    if before.llm_results or after.llm_results:
        lines += ["", *_llm_section(before, after)]
    return "\n".join(lines) + "\n"
