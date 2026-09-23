"""Evaluación con repeticiones (juez LLM) y métricas deterministas (sin juez).

Las pruebas de `run_case`/`run_suite` usan un agente falso con respuesta fija y el
`Judge` real sobre un `FakeLLM` con veredictos JSON en secuencia, de modo que cada
ejecución del caso consume un veredicto distinto. `deterministic_metrics` se prueba
sobre un escenario fijo con agentes falsos y, además, sobre el store real en ambos
modos.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from fakes import FakeLLM

from rag_lab.agent import ABSTENTION_MESSAGE, AgentResponse, RagAgent, Retrieval
from rag_lab.config import DEFAULT_PASS_RATE_THRESHOLD, DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K
from rag_lab.corpus import AGENT_IDS, GoldenCase, load_golden_set
from rag_lab.evaluation import (
    CaseResult,
    DeterministicMetrics,
    RunError,
    deterministic_metrics,
    meets_pass_rate,
    run_case,
    run_suite,
)
from rag_lab.judge import Judge, Verdict
from rag_lab.llm import LLMClient, LLMError
from rag_lab.store import Chunk, VectorStore

_FAQ_CHUNK = Chunk("faq-02", "faq", "Returns are accepted within 30 days.", 0.2)
_HR_CHUNK = Chunk("seg-01", "seguimiento", "Employees accrue 1.5 vacation days per month.", 0.3)
_FAR_FAQ_CHUNK = Chunk("faq-07", "faq", "Gift cards never expire.", 0.9)

_ANSWER_CASE = GoldenCase(
    id="c-answer",
    agent_id="faq",
    question="How long do I have to return an item?",
    category="in_domain",
    expected_behavior="answer",
    expected_source_ids=("faq-02",),
)
_ABSTAIN_CASE = GoldenCase(
    id="c-abstain",
    agent_id="faq",
    question="How many vacation days do employees get?",
    category="cross_domain",
    expected_behavior="abstain",
    expected_source_ids=("seg-01",),
)


def _verdict_json(score: float) -> str:
    """Veredicto JSON válido con los tres criterios al mismo `score`."""
    return json.dumps(
        {
            dimension: {"score": score, "justification": f"{dimension} = {score}."}
            for dimension in ("faithfulness", "relevance", "abstention")
        }
    )


_PASS = _verdict_json(1.0)
_FAIL = _verdict_json(0.0)
_INVALID = "not json"


def _answered(*retrieved: Chunk, answer: str = "You have 30 days.") -> AgentResponse:
    return AgentResponse(answer=answer, abstained=False, retrieved=retrieved, context=retrieved)


def _abstained(*retrieved: Chunk) -> AgentResponse:
    return AgentResponse(answer=ABSTENTION_MESSAGE, abstained=True, retrieved=retrieved, context=())


class _FakeAgent:
    """Agente falso: devuelve respuestas por pregunta y registra las preguntas recibidas."""

    def __init__(self, responses: dict[str, AgentResponse]) -> None:
        self._responses = responses
        self.questions: list[str] = []

    def ask(self, question: str) -> AgentResponse:
        self.questions.append(question)
        return self._responses[question]


class _ScriptedAgent:
    """Agente falso que en cada llamada devuelve el siguiente elemento del guion o lo lanza."""

    def __init__(self, script: Sequence[AgentResponse | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    def ask(self, question: str) -> AgentResponse:
        item = self._script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _answer_agent() -> _FakeAgent:
    return _FakeAgent({_ANSWER_CASE.question: _answered(_FAQ_CHUNK)})


# --- meets_pass_rate: resolución de 2/3 frente a 0.67 ---------------------------------


def test_two_of_three_meets_default_threshold_even_though_two_thirds_is_below_067() -> None:
    assert 2 / 3 < 0.67  # por eso no basta con `pass_rate >= threshold`
    assert DEFAULT_PASS_RATE_THRESHOLD == 0.67
    assert meets_pass_rate(2, 3, 0.67) is True


@pytest.mark.parametrize(
    ("successes", "runs", "threshold", "expected"),
    [
        (1, 3, 0.67, False),
        (3, 3, 0.67, True),
        (0, 3, 0.67, False),
        (1, 3, 0.33, True),  # 1/3 = 0.333… también cuenta como 0.33
        (1, 2, 0.5, True),  # inclusivo en el valor exacto
        (1, 2, 0.51, False),  # 0.5 queda por debajo de 0.51 más allá de la tolerancia
        (66, 100, 0.67, False),  # 0.66 no es 0.67
        (0, 3, 0.0, True),
        (2, 3, 1.0, False),
        (3, 3, 1.0, True),
    ],
)
def test_meets_pass_rate_compares_with_two_decimal_tolerance(
    successes: int, runs: int, threshold: float, expected: bool
) -> None:
    assert meets_pass_rate(successes, runs, threshold) is expected


def test_meets_pass_rate_threshold_one_tolerates_one_miss_beyond_200_runs() -> None:
    # Caso límite INTENCIONAL y documentado de la tolerancia de medio centésimo:
    # 200/201 = 0.995… se redondea a 1.00.
    assert meets_pass_rate(200, 201, 1.0) is True


@pytest.mark.parametrize(
    ("successes", "runs", "threshold"),
    [(0, 0, 0.67), (4, 3, 0.67), (-1, 3, 0.67), (2, 3, 1.5), (2, 3, -0.1), (2, 3, float("nan"))],
)
def test_meets_pass_rate_rejects_invalid_arguments(
    successes: int, runs: int, threshold: float
) -> None:
    with pytest.raises(ValueError):
        meets_pass_rate(successes, runs, threshold)


# --- run_case -------------------------------------------------------------------------


def test_run_case_two_of_three_passing_runs_passes_with_threshold_067() -> None:
    agent = _answer_agent()
    judge_llm = FakeLLM([_PASS, _FAIL, _PASS])

    result = run_case(agent, Judge(judge_llm), _ANSWER_CASE, runs=3, pass_rate_threshold=0.67)

    assert result.case_id == "c-answer"
    assert result.pass_rate == pytest.approx(2 / 3)
    assert result.passed is True
    assert len(result.verdicts) == 3
    assert all(isinstance(v, Verdict) for v in result.verdicts)
    assert result.run_passed == (True, False, True)
    assert agent.questions == [_ANSWER_CASE.question] * 3
    assert len(judge_llm.calls) == 3


def test_run_case_one_of_three_passing_runs_fails_with_threshold_067() -> None:
    result = run_case(
        _answer_agent(),
        Judge(FakeLLM([_FAIL, _PASS, _FAIL])),
        _ANSWER_CASE,
        runs=3,
        pass_rate_threshold=0.67,
    )

    assert result.pass_rate == pytest.approx(1 / 3)
    assert result.passed is False
    assert result.run_passed == (False, True, False)


def test_run_case_uses_default_pass_rate_threshold() -> None:
    result = run_case(_answer_agent(), Judge(FakeLLM([_PASS, _PASS, _FAIL])), _ANSWER_CASE, runs=3)

    assert result.passed is True


def test_run_case_judges_with_agent_context_answer_and_expected_behavior() -> None:
    response = AgentResponse(
        answer="You have 30 days.",
        abstained=False,
        retrieved=(_FAQ_CHUNK, _FAR_FAQ_CHUNK),
        context=(_FAQ_CHUNK,),
    )
    agent = _FakeAgent({_ANSWER_CASE.question: response})
    judge_llm = FakeLLM(_PASS)

    run_case(agent, Judge(judge_llm), _ANSWER_CASE, runs=1)

    system, user = judge_llm.calls[0]
    assert _ANSWER_CASE.question in user
    assert "You have 30 days." in user
    assert _FAQ_CHUNK.text in user
    # El juez ve solo el contexto que recibió el generador, no todo el top-k.
    assert _FAR_FAQ_CHUNK.text not in user
    assert "SHOULD ANSWER" in system


def test_run_case_applies_expected_behavior_rule_to_verdict() -> None:
    # abstención alta pero fidelidad/relevancia bajas: pasa solo si se esperaba abstenerse.
    mixed = json.dumps(
        {
            "faithfulness": {"score": 0.0, "justification": "No claims."},
            "relevance": {"score": 0.0, "justification": "Declines."},
            "abstention": {"score": 1.0, "justification": "Declined."},
        }
    )
    agent = _FakeAgent(
        {
            _ANSWER_CASE.question: _abstained(_FAR_FAQ_CHUNK),
            _ABSTAIN_CASE.question: _abstained(_FAR_FAQ_CHUNK),
        }
    )

    as_answer = run_case(agent, Judge(FakeLLM(mixed)), _ANSWER_CASE, runs=1)
    as_abstain = run_case(agent, Judge(FakeLLM(mixed)), _ABSTAIN_CASE, runs=1)

    assert as_answer.passed is False
    assert as_abstain.passed is True


def test_run_case_forwards_min_score_to_verdict_rule() -> None:
    judge_llm = FakeLLM(_verdict_json(0.6))

    lenient = run_case(_answer_agent(), Judge(judge_llm), _ANSWER_CASE, runs=1, min_score=0.5)
    strict = run_case(_answer_agent(), Judge(judge_llm), _ANSWER_CASE, runs=1, min_score=0.7)

    assert lenient.passed is True
    assert strict.passed is False


def test_run_case_judge_error_counts_as_failed_run_and_is_recorded() -> None:
    # Ejecución 2: salida inválida en el intento y en el reintento → JudgeError.
    judge_llm = FakeLLM([_PASS, _INVALID, _INVALID, _PASS])

    result = run_case(
        _answer_agent(), Judge(judge_llm), _ANSWER_CASE, runs=3, pass_rate_threshold=0.67
    )

    assert result.pass_rate == pytest.approx(2 / 3)
    assert result.passed is True
    assert result.run_passed == (True, False, True)
    error = result.verdicts[1]
    assert isinstance(error, RunError)
    assert error.stage == "judge"
    assert "inválida" in error.message
    assert result.errors == (error,)


def test_run_case_judge_errors_can_make_the_case_fail() -> None:
    judge_llm = FakeLLM([_PASS, _INVALID, _INVALID, _INVALID, _INVALID])

    result = run_case(_answer_agent(), Judge(judge_llm), _ANSWER_CASE, runs=3)

    assert result.pass_rate == pytest.approx(1 / 3)
    assert result.passed is False
    assert [type(v) for v in result.verdicts] == [Verdict, RunError, RunError]


def test_run_case_judge_llm_failure_counts_as_failed_run() -> None:
    class _DownLLM:
        def complete(self, system: str, user: str, json_mode: bool = False) -> str:
            raise LLMError("connection refused")

    result = run_case(_answer_agent(), Judge(_DownLLM()), _ANSWER_CASE, runs=2)

    assert result.pass_rate == 0.0
    assert result.passed is False
    assert all(isinstance(v, RunError) and v.stage == "judge" for v in result.verdicts)
    assert "connection refused" in result.verdicts[0].message  # type: ignore[union-attr]


def test_run_case_generator_llm_error_counts_as_failed_run_without_judging() -> None:
    agent = _ScriptedAgent(
        [_answered(_FAQ_CHUNK), LLMError("generator timeout"), _answered(_FAQ_CHUNK)]
    )
    judge_llm = FakeLLM([_PASS, _PASS])

    result = run_case(agent, Judge(judge_llm), _ANSWER_CASE, runs=3, pass_rate_threshold=0.67)

    assert result.run_passed == (True, False, True)
    assert result.passed is True
    error = result.verdicts[1]
    assert isinstance(error, RunError)
    assert error.stage == "agent"
    assert "generator timeout" in error.message
    # La ejecución fallida en el generador no llega al juez.
    assert len(judge_llm.calls) == 2


def test_run_case_does_not_swallow_unexpected_exceptions() -> None:
    agent = _ScriptedAgent([RuntimeError("bug in agent")])

    with pytest.raises(RuntimeError, match="bug in agent"):
        run_case(agent, Judge(FakeLLM(_PASS)), _ANSWER_CASE, runs=1)


@pytest.mark.parametrize("runs", [0, -1])
def test_run_case_rejects_runs_below_one(runs: int) -> None:
    agent = _answer_agent()
    judge_llm = FakeLLM(_PASS)

    with pytest.raises(ValueError, match="runs"):
        run_case(agent, Judge(judge_llm), _ANSWER_CASE, runs=runs)
    assert agent.questions == []
    assert judge_llm.calls == []


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_run_case_rejects_threshold_out_of_range_before_running(threshold: float) -> None:
    agent = _answer_agent()

    with pytest.raises(ValueError, match="pass_rate_threshold"):
        run_case(agent, Judge(FakeLLM(_PASS)), _ANSWER_CASE, runs=1, pass_rate_threshold=threshold)
    assert agent.questions == []


@pytest.mark.parametrize("min_score", [-0.1, 1.1, float("nan")])
def test_run_case_rejects_invalid_min_score_before_running(min_score: float) -> None:
    agent = _answer_agent()
    judge_llm = FakeLLM(_PASS)

    with pytest.raises(ValueError, match="min_score"):
        run_case(agent, Judge(judge_llm), _ANSWER_CASE, runs=1, min_score=min_score)
    assert agent.questions == []
    assert judge_llm.calls == []


@pytest.mark.parametrize("min_score", [-0.1, 1.1, float("nan")])
def test_run_suite_rejects_invalid_min_score_before_running(min_score: float) -> None:
    agent = _answer_agent()
    judge_llm = FakeLLM(_PASS)

    with pytest.raises(ValueError, match="min_score"):
        run_suite({"faq": agent}, Judge(judge_llm), [_ANSWER_CASE], runs=1, min_score=min_score)
    assert agent.questions == []
    assert judge_llm.calls == []


def test_run_suite_rejects_invalid_runs_and_threshold_even_without_cases() -> None:
    judge = Judge(FakeLLM(_PASS))

    with pytest.raises(ValueError, match="runs"):
        run_suite({}, judge, [], runs=0)
    with pytest.raises(ValueError, match="pass_rate_threshold"):
        run_suite({}, judge, [], runs=1, pass_rate_threshold=2.0)


def test_case_result_is_frozen() -> None:
    result = run_case(_answer_agent(), Judge(FakeLLM(_PASS)), _ANSWER_CASE, runs=1)

    assert isinstance(result, CaseResult)
    with pytest.raises(AttributeError):
        result.passed = False  # type: ignore[misc]


# --- run_suite ------------------------------------------------------------------------


def test_run_suite_routes_each_case_to_its_agent_and_keeps_case_order() -> None:
    faq_agent = _FakeAgent({_ANSWER_CASE.question: _answered(_FAQ_CHUNK)})
    seg_case = GoldenCase(
        id="c-seg",
        agent_id="seguimiento",
        question="How many vacation days do I accrue?",
        category="in_domain",
        expected_behavior="answer",
        expected_source_ids=("seg-01",),
    )
    seg_agent = _FakeAgent({seg_case.question: _answered(_HR_CHUNK, answer="1.5 per month.")})
    # 2 ejecuciones por caso: c-answer pasa 2/2, c-seg pasa 0/2.
    judge_llm = FakeLLM([_PASS, _PASS, _FAIL, _FAIL])

    results = run_suite(
        {"faq": faq_agent, "seguimiento": seg_agent},
        Judge(judge_llm),
        [_ANSWER_CASE, seg_case],
        runs=2,
    )

    assert [r.case_id for r in results] == ["c-answer", "c-seg"]
    assert [r.passed for r in results] == [True, False]
    assert faq_agent.questions == [_ANSWER_CASE.question] * 2
    assert seg_agent.questions == [seg_case.question] * 2


def test_run_suite_with_no_cases_returns_empty_tuple() -> None:
    assert run_suite({"faq": _answer_agent()}, Judge(FakeLLM(_PASS)), [], runs=3) == ()


def test_run_suite_rejects_case_without_agent_before_running_anything() -> None:
    agent = _answer_agent()
    orphan = GoldenCase(
        id="c-orphan",
        agent_id="seguimiento",
        question="q",
        category="out_of_domain",
        expected_behavior="abstain",
        expected_source_ids=(),
    )
    judge_llm = FakeLLM(_PASS)

    with pytest.raises(ValueError, match="c-orphan"):
        run_suite({"faq": agent}, Judge(judge_llm), [_ANSWER_CASE, orphan], runs=1)
    assert agent.questions == []
    assert judge_llm.calls == []


# --- deterministic_metrics: escenario fijo --------------------------------------------


def _retrieval(retrieved: Sequence[Chunk], context: Sequence[Chunk] = ()) -> Retrieval:
    return Retrieval(retrieved=tuple(retrieved), context=tuple(context))


class _FakeRetriever:
    """Recuperador falso: devuelve una `Retrieval` fija por pregunta. No tiene `ask`."""

    def __init__(self, retrievals: dict[str, Retrieval]) -> None:
        self._retrievals = retrievals
        self.questions: list[str] = []

    def retrieve(self, question: str) -> Retrieval:
        self.questions.append(question)
        return self._retrievals[question]


def _fixed_scenario() -> tuple[dict[str, _FakeRetriever], list[GoldenCase]]:
    """4 casos del agente faq (abstenerse = contexto vacío tras la compuerta):

    - c1 answer, contexto propio                      → limpio, abstención correcta
    - c2 answer, top-k con chunk ajeno                → contaminado, abstención correcta
    - c3 abstain, el chunk ajeno pasa la compuerta    → contaminado, abstención INCORRECTA
    - c4 abstain, contexto vacío, top-k con ajeno     → contaminado (lejano), correcta
    """
    cases = [
        GoldenCase("c1", "faq", "q1", "in_domain", "answer", ("faq-02",)),
        GoldenCase("c2", "faq", "q2", "in_domain", "answer", ("faq-02",)),
        GoldenCase("c3", "faq", "q3", "cross_domain", "abstain", ("seg-01",)),
        GoldenCase("c4", "faq", "q4", "out_of_domain", "abstain", ()),
    ]
    far_hr = Chunk("seg-05", "seguimiento", "Shift swaps need approval.", 0.95)
    retriever = _FakeRetriever(
        {
            "q1": _retrieval([_FAQ_CHUNK, _FAR_FAQ_CHUNK], [_FAQ_CHUNK]),
            "q2": _retrieval([_FAQ_CHUNK, _HR_CHUNK], [_FAQ_CHUNK, _HR_CHUNK]),
            "q3": _retrieval([_HR_CHUNK, _FAQ_CHUNK], [_HR_CHUNK, _FAQ_CHUNK]),
            "q4": _retrieval([_FAR_FAQ_CHUNK, far_hr]),
        }
    )
    return {"faq": retriever}, cases


def test_deterministic_metrics_on_fixed_scenario() -> None:
    retriever_by_id, cases = _fixed_scenario()

    metrics = deterministic_metrics(retriever_by_id, cases)

    assert metrics == DeterministicMetrics(
        total_cases=4,
        contaminated_case_ids=("c2", "c3", "c4"),
        abstention_mismatch_case_ids=("c3",),
    )
    assert metrics.contamination_rate == pytest.approx(0.75)
    assert metrics.abstention_accuracy == pytest.approx(0.75)
    assert retriever_by_id["faq"].questions == ["q1", "q2", "q3", "q4"]


def test_deterministic_metrics_counts_contamination_from_retrieved_not_context() -> None:
    # Contexto vacío (se abstendría) pero el top-k trae un chunk ajeno.
    case = GoldenCase("c", "faq", "q", "out_of_domain", "abstain", ())
    retriever = _FakeRetriever({"q": _retrieval([_HR_CHUNK])})

    metrics = deterministic_metrics({"faq": retriever}, [case])

    assert metrics.contaminated_case_ids == ("c",)
    assert metrics.contamination_rate == 1.0
    assert metrics.abstention_accuracy == 1.0


def test_deterministic_metrics_uses_case_agent_as_owner() -> None:
    # El mismo chunk es propio para seguimiento y ajeno para faq.
    faq_case = GoldenCase("f", "faq", "q", "in_domain", "answer", ("faq-02",))
    seg_case = GoldenCase("s", "seguimiento", "q", "in_domain", "answer", ("seg-01",))
    retrieval = _retrieval([_HR_CHUNK], [_HR_CHUNK])

    metrics = deterministic_metrics(
        {
            "faq": _FakeRetriever({"q": retrieval}),
            "seguimiento": _FakeRetriever({"q": retrieval}),
        },
        [faq_case, seg_case],
    )

    assert metrics.contaminated_case_ids == ("f",)
    assert metrics.contamination_rate == 0.5


def test_deterministic_metrics_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="casos"):
        deterministic_metrics({"faq": _FakeRetriever({})}, [])


def test_deterministic_metrics_rejects_case_without_agent() -> None:
    case = GoldenCase("c-orphan", "seguimiento", "q", "out_of_domain", "abstain", ())
    retriever = _FakeRetriever({})

    with pytest.raises(ValueError, match="c-orphan"):
        deterministic_metrics({"faq": retriever}, [case])
    assert retriever.questions == []


# --- RagAgent.retrieve y deterministic_metrics sobre el store real ---------------------

GOLDEN_SET: list[GoldenCase] = load_golden_set()
CROSS_DOMAIN_CASES = [c for c in GOLDEN_SET if c.category == "cross_domain"]
CROSS_DOMAIN_IDS = {c.id for c in CROSS_DOMAIN_CASES}


class _ForbiddenLLM:
    """LLM que falla si se le llama: las métricas deterministas no deben usarlo."""

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        raise AssertionError("las métricas deterministas no deben llamar al LLM")


def _real_agents(store: VectorStore, isolated: bool, llm: LLMClient) -> dict[str, RagAgent]:
    return {
        agent_id: RagAgent(
            agent_id,
            store,
            llm,
            threshold=DEFAULT_RELEVANCE_THRESHOLD,
            top_k=DEFAULT_TOP_K,
            isolated=isolated,
        )
        for agent_id in AGENT_IDS
    }


@pytest.mark.parametrize("isolated", [False, True], ids=["shared", "isolated"])
def test_rag_agent_retrieve_matches_ask_without_calling_llm(
    isolated: bool, indexed_store: VectorStore
) -> None:
    # Pregunta cruzada: en shared pasa la compuerta (el LLM se llamaría en ask).
    case = CROSS_DOMAIN_CASES[0]
    llm = FakeLLM()
    agent = _real_agents(indexed_store, isolated, llm)[case.agent_id]

    retrieval = agent.retrieve(case.question)

    assert isinstance(retrieval, Retrieval)
    assert llm.calls == []
    assert retrieval.retrieved == tuple(
        indexed_store.query(
            case.question, k=DEFAULT_TOP_K, agent_id=case.agent_id if isolated else None
        )
    )
    assert retrieval.context == tuple(
        c for c in retrieval.retrieved if c.distance <= DEFAULT_RELEVANCE_THRESHOLD
    )
    response = agent.ask(case.question)
    assert (response.retrieved, response.context) == (retrieval.retrieved, retrieval.context)
    assert response.abstained == (not retrieval.context)


def test_retrieval_is_frozen() -> None:
    retrieval = _retrieval([_FAQ_CHUNK])

    with pytest.raises(AttributeError):
        retrieval.context = (_FAQ_CHUNK,)  # type: ignore[misc]


def test_deterministic_metrics_never_calls_the_llm_even_when_the_gate_passes(
    indexed_store: VectorStore,
) -> None:
    llm = FakeLLM()
    agents = _real_agents(indexed_store, isolated=False, llm=llm)
    # Salvaguarda: en shared las preguntas cruzadas pasan la compuerta, así que `ask`
    # SÍ llamaría al generador; la prueba no es vacía.
    assert any(agents[c.agent_id].retrieve(c.question).context for c in CROSS_DOMAIN_CASES)

    deterministic_metrics(agents, GOLDEN_SET)

    assert llm.calls == []


def test_deterministic_metrics_isolated_has_no_contamination_and_perfect_abstention(
    indexed_store: VectorStore,
) -> None:
    agents = _real_agents(indexed_store, isolated=True, llm=_ForbiddenLLM())

    metrics = deterministic_metrics(agents, GOLDEN_SET)

    assert metrics.total_cases == len(GOLDEN_SET)
    assert metrics.contamination_rate == 0.0
    assert metrics.abstention_accuracy == 1.0


def test_deterministic_metrics_shared_detects_contamination_and_cross_domain_failures(
    indexed_store: VectorStore,
) -> None:
    agents = _real_agents(indexed_store, isolated=False, llm=_ForbiddenLLM())

    metrics = deterministic_metrics(agents, GOLDEN_SET)

    # Las preguntas cruzadas traen el documento del otro agente y pasan la compuerta.
    assert CROSS_DOMAIN_IDS <= set(metrics.contaminated_case_ids)
    assert CROSS_DOMAIN_IDS <= set(metrics.abstention_mismatch_case_ids)
    assert metrics.contamination_rate > 0.0
    assert metrics.abstention_accuracy < 1.0
