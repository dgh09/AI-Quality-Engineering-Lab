"""Evaluación de los agentes: repeticiones con juez LLM y métricas deterministas.

Dos niveles:

- `run_case` / `run_suite`: cada caso se ejecuta `runs` veces (pregunta al agente →
  veredicto del juez) porque generador y juez no son deterministas. Un caso pasa si
  su tasa de éxito alcanza `pass_rate_threshold` (ver `meets_pass_rate`). Un fallo de
  una ejecución —`LLMError` del generador o `JudgeError` del juez— cuenta como
  ejecución fallida y queda registrado como `RunError` en `CaseResult.verdicts`: no
  se oculta ni aborta la suite. Cualquier otra excepción es un bug y se propaga.

- `deterministic_metrics`: tasa de contaminación y exactitud de abstención, sin juez.
  La contaminación se mide sobre `Retrieval.retrieved`, NUNCA a partir de las
  puntuaciones del juez: en modo shared el juez puntúa la fidelidad contra el
  contexto ya contaminado, así que una respuesta copiada de un chunk ajeno puede
  sacar fidelidad alta. Usa `RagAgent.retrieve` (recuperación + compuerta), así que
  nunca llama al LLM. Funciona igual con agentes `isolated` y shared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol

from rag_lab.agent import AgentResponse, Retrieval
from rag_lab.config import DEFAULT_PASS_RATE_THRESHOLD
from rag_lab.contamination import foreign_chunks
from rag_lab.corpus import GoldenCase
from rag_lab.judge import DEFAULT_MIN_SCORE, Judge, JudgeError, Verdict
from rag_lab.llm import LLMError

# El umbral de tasa de éxito se expresa con dos decimales (0.67 = "2 de 3"), pero
# 2/3 = 0.666… < 0.67. Se compara con una tolerancia de medio centésimo, en
# aritmética exacta: pasa si successes/runs >= threshold - 0.005, es decir, si la
# tasa redondeada a dos decimales (mitad hacia arriba) alcanza el umbral.
PASS_RATE_TOLERANCE = Fraction(1, 200)


class Agent(Protocol):
    """Lo único que la evaluación necesita de un agente (`RagAgent` lo cumple)."""

    def ask(self, question: str) -> AgentResponse: ...


class Retriever(Protocol):
    """Recuperación + compuerta sin LLM (`RagAgent.retrieve` lo cumple)."""

    def retrieve(self, question: str) -> Retrieval: ...


@dataclass(frozen=True)
class RunError:
    """Ejecución fallida por error: `stage` indica si falló el agente o el juez."""

    stage: Literal["agent", "judge"]
    message: str


# Resultado de una ejecución: el veredicto del juez o el error que la hizo fallar.
RunOutcome = Verdict | RunError


@dataclass(frozen=True)
class CaseResult:
    case: GoldenCase
    pass_rate: float
    passed: bool
    # Un elemento por ejecución, en orden.
    verdicts: tuple[RunOutcome, ...]
    # Si cada ejecución pasó (un `RunError` siempre es False), en paralelo a `verdicts`.
    run_passed: tuple[bool, ...]

    @property
    def case_id(self) -> str:
        return self.case.id

    @property
    def errors(self) -> tuple[RunError, ...]:
        """Ejecuciones que fallaron por error (agente o juez), en orden."""
        return tuple(v for v in self.verdicts if isinstance(v, RunError))


def _check_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:  # también rechaza NaN
        raise ValueError(f"pass_rate_threshold debe estar en [0, 1], se recibió {threshold!r}")


def _check_run_params(runs: int, pass_rate_threshold: float, min_score: float) -> None:
    """Valida los parámetros de `run_case`/`run_suite` antes de cualquier llamada."""
    if runs < 1:
        raise ValueError(f"runs debe ser >= 1, se recibió {runs}")
    _check_threshold(pass_rate_threshold)
    if not 0 <= min_score <= 1:  # también rechaza NaN
        raise ValueError(f"min_score debe estar en [0, 1], se recibió {min_score!r}")


def meets_pass_rate(successes: int, runs: int, threshold: float) -> bool:
    """True si `successes / runs` alcanza `threshold` con tolerancia de medio centésimo.

    El umbral se interpreta con dos decimales: 0.67 exige 2 de 3 (0.666… cuenta como
    0.67), 0.66 de 100 no llega a 0.67, y 1.0 exige todas cuando `runs <= 200`. La
    comparación es exacta (`Fraction`, con `threshold` tomado por su valor decimal).
    """
    if runs < 1:
        raise ValueError(f"runs debe ser >= 1, se recibió {runs}")
    if not 0 <= successes <= runs:
        raise ValueError(f"successes debe estar en [0, {runs}], se recibió {successes}")
    _check_threshold(threshold)
    return Fraction(successes, runs) >= Fraction(str(threshold)) - PASS_RATE_TOLERANCE


def _run_once(
    agent: Agent, judge: Judge, case: GoldenCase, min_score: float
) -> tuple[RunOutcome, bool]:
    try:
        response = agent.ask(case.question)
    except LLMError as exc:
        return RunError(stage="agent", message=str(exc)), False
    try:
        verdict = judge.evaluate(
            case.question, response.context, response.answer, case.expected_behavior
        )
    except JudgeError as exc:
        return RunError(stage="judge", message=str(exc)), False
    return verdict, verdict.passed(case.expected_behavior, min_score)


def run_case(
    agent: Agent,
    judge: Judge,
    case: GoldenCase,
    runs: int,
    pass_rate_threshold: float = DEFAULT_PASS_RATE_THRESHOLD,
    min_score: float = DEFAULT_MIN_SCORE,
) -> CaseResult:
    """Ejecuta el caso `runs` veces y lo aprueba según `meets_pass_rate`.

    Cada ejecución: `agent.ask(question)` y luego `judge.evaluate` con el contexto
    que vio el generador (`response.context`), su respuesta y el comportamiento
    esperado; aprueba si `Verdict.passed(expected_behavior, min_score)`. Un
    `LLMError` del agente (el juez no se llama) o un `JudgeError` cuentan como
    ejecución fallida y se registran como `RunError`. `runs < 1`, o
    `pass_rate_threshold` o `min_score` fuera de [0, 1] (o NaN), lanzan
    `ValueError` antes de ejecutar nada.
    """
    _check_run_params(runs, pass_rate_threshold, min_score)
    outcomes = [_run_once(agent, judge, case, min_score) for _ in range(runs)]
    run_passed = tuple(ok for _, ok in outcomes)
    successes = sum(run_passed)
    return CaseResult(
        case=case,
        pass_rate=successes / runs,
        passed=meets_pass_rate(successes, runs, pass_rate_threshold),
        verdicts=tuple(outcome for outcome, _ in outcomes),
        run_passed=run_passed,
    )


def _check_agents(agent_by_id: Mapping[str, object], cases: Sequence[GoldenCase]) -> None:
    missing = [case.id for case in cases if case.agent_id not in agent_by_id]
    if missing:
        raise ValueError(
            f"casos sin agente para su agent_id (disponibles: {sorted(agent_by_id)}): {missing}"
        )


def run_suite(
    agent_by_id: Mapping[str, Agent],
    judge: Judge,
    cases: Sequence[GoldenCase],
    runs: int,
    pass_rate_threshold: float = DEFAULT_PASS_RATE_THRESHOLD,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[CaseResult, ...]:
    """`run_case` para cada caso con el agente de su `agent_id`, en el orden recibido.

    Sin casos devuelve `()`. Parámetros inválidos (como en `run_case`) o un caso sin
    agente en `agent_by_id` lanzan `ValueError` antes de ejecutar nada (no se
    gastan llamadas al LLM).
    """
    _check_run_params(runs, pass_rate_threshold, min_score)
    _check_agents(agent_by_id, cases)
    return tuple(
        run_case(agent_by_id[case.agent_id], judge, case, runs, pass_rate_threshold, min_score)
        for case in cases
    )


# --- Métricas deterministas -----------------------------------------------------------


@dataclass(frozen=True)
class DeterministicMetrics:
    total_cases: int
    # Casos cuyo top-k recuperado contiene al menos un chunk de otro agente.
    contaminated_case_ids: tuple[str, ...]
    # Casos en los que `abstained` no coincide con `expected_behavior`.
    abstention_mismatch_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.total_cases < 1:
            raise ValueError(f"total_cases debe ser >= 1, se recibió {self.total_cases}")

    @property
    def contamination_rate(self) -> float:
        """Fracción de casos con algún chunk ajeno en el top-k recuperado."""
        return len(self.contaminated_case_ids) / self.total_cases

    @property
    def abstention_accuracy(self) -> float:
        """Fracción de casos en que el agente se abstuvo si y solo si se esperaba."""
        return (self.total_cases - len(self.abstention_mismatch_case_ids)) / self.total_cases


def deterministic_metrics(
    retriever_by_id: Mapping[str, Retriever], cases: Sequence[GoldenCase]
) -> DeterministicMetrics:
    """Recupera una vez cada caso con el agente de su `agent_id` y mide contaminación
    y abstención, sin LLM (ni generador ni juez).

    Definiciones (sobre N = número de casos):
    - Tasa de contaminación = casos cuyo `Retrieval.retrieved` (el top-k completo,
      no solo el contexto bajo el umbral) contiene algún chunk con `agent_id` distinto
      del `agent_id` del caso, / N. Mide el fallo de aislamiento de la capa de datos
      aunque el chunk ajeno quede fuera del umbral; en modo isolated debe ser 0.
    - Exactitud de abstención = casos con `abstained == (expected_behavior ==
      "abstain")`, / N, donde `abstained = not retrieval.context`: la misma compuerta
      con la que `RagAgent.ask` se abstiene (vive solo en `RagAgent.retrieve`).

    Sin casos, o con un caso sin agente en `retriever_by_id`, lanza `ValueError`
    (las tasas no están definidas).
    """
    if not cases:
        raise ValueError("se necesitan casos para calcular las métricas deterministas")
    _check_agents(retriever_by_id, cases)
    contaminated: list[str] = []
    mismatched: list[str] = []
    for case in cases:
        retrieval = retriever_by_id[case.agent_id].retrieve(case.question)
        if foreign_chunks(retrieval.retrieved, case.agent_id):
            contaminated.append(case.id)
        abstained = not retrieval.context
        if abstained != (case.expected_behavior == "abstain"):
            mismatched.append(case.id)
    return DeterministicMetrics(
        total_cases=len(cases),
        contaminated_case_ids=tuple(contaminated),
        abstention_mismatch_case_ids=tuple(mismatched),
    )
