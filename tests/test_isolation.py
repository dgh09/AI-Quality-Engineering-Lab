"""Aislamiento entre agentes: aserciones del comportamiento CORRECTO, por modo.

En el modo `shared` (vector store compartido sin filtro por agente) las aserciones
fallan a propósito: están marcadas `xfail(strict=True)` para que el bug quede
documentado y, si alguien lo arregla sin querer, la suite lo señale (XPASS → fallo).

Cuando exista el modo aislado basta con AÑADIR su parámetro a `MODES`; las
aserciones no deben modificarse.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fakes import FakeLLM

from rag_lab.agent import RagAgent
from rag_lab.config import DEFAULT_RELEVANCE_THRESHOLD
from rag_lab.contamination import foreign_chunks
from rag_lab.corpus import AGENT_IDS, GoldenCase, load_golden_set
from rag_lab.store import VectorStore

TOP_K = 3

MODES = [
    pytest.param(
        "shared",
        # raises=AssertionError: solo el fallo de la aserción cuenta como el bug esperado;
        # cualquier otra excepción (fixture, NotImplementedError, API de Chroma) es FAILED.
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason="BUG: vector store compartido sin aislamiento por agent_id",
        ),
    ),
]

GOLDEN_SET: list[GoldenCase] = load_golden_set()
CROSS_DOMAIN_CASES: list[GoldenCase] = [c for c in GOLDEN_SET if c.category == "cross_domain"]

AgentFactory = Callable[[str, str], RagAgent]


@pytest.fixture
def make_agent(indexed_store: VectorStore, fake_llm: FakeLLM) -> AgentFactory:
    """Construye un `RagAgent` sobre el store indexado con el umbral por defecto y top_k=3."""

    def _make(agent_id: str, mode: str) -> RagAgent:
        assert mode in {"shared", "isolated"}, mode
        return RagAgent(
            agent_id,
            indexed_store,
            fake_llm,
            threshold=DEFAULT_RELEVANCE_THRESHOLD,
            top_k=TOP_K,
            isolated=(mode == "isolated"),
        )

    return _make


def test_golden_set_has_cross_domain_cases() -> None:
    # Salvaguarda: sin casos cruzados las pruebas de abajo pasarían en vacío.
    assert len(CROSS_DOMAIN_CASES) >= 2


@pytest.mark.parametrize("mode", MODES)
def test_agent_retrieves_only_its_own_chunks(mode: str, make_agent: AgentFactory) -> None:
    """Ningún agente recupera chunks de otro agente, para ninguna pregunta que se le dirige."""
    offenders: dict[str, list[str]] = {}
    for agent_id in AGENT_IDS:
        agent = make_agent(agent_id, mode)
        for case in (c for c in GOLDEN_SET if c.agent_id == agent_id):
            foreign = foreign_chunks(agent.ask(case.question).retrieved, agent_id)
            if foreign:
                offenders[f"{case.id} ({agent_id})"] = [chunk.id for chunk in foreign]
    assert offenders == {}, (
        f"[{mode}] contaminación: preguntas que recuperaron chunks de otro agente "
        f"(pregunta → ids ajenos): {offenders}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_agent_abstains_on_cross_domain_question(
    mode: str, make_agent: AgentFactory, fake_llm: FakeLLM
) -> None:
    """Ante una pregunta del dominio del otro agente, el agente se abstiene sin llamar al LLM."""
    failures: dict[str, str] = {}
    for case in CROSS_DOMAIN_CASES:
        calls_before = len(fake_llm.calls)
        response = make_agent(case.agent_id, mode).ask(case.question)
        llm_called = len(fake_llm.calls) > calls_before
        if not response.abstained or llm_called:
            failures[f"{case.id} ({case.agent_id})"] = (
                f"abstained={response.abstained}, llm_llamado={llm_called}, "
                f"contexto={[chunk.id for chunk in response.context]}"
            )
    assert failures == {}, (
        f"[{mode}] el agente no se abstuvo ante preguntas de dominio cruzado: {failures}"
    )


def test_shared_store_leaks_other_agent_chunks(make_agent: AgentFactory) -> None:
    """Reproducción del bug: en `shared` la pregunta cruzada trae el documento del otro agente."""
    reached_llm: list[str] = []
    for case in CROSS_DOMAIN_CASES:
        response = make_agent(case.agent_id, "shared").ask(case.question)
        retrieved_ids = [chunk.id for chunk in response.retrieved]
        assert set(case.expected_source_ids) <= set(retrieved_ids), (
            f"{case.id}: se esperaba recuperar {case.expected_source_ids} del otro agente, "
            f"se recuperó {retrieved_ids}"
        )
        if not response.abstained and foreign_chunks(response.context, case.agent_id):
            reached_llm.append(case.id)
    # La fuga en la recuperación se exige para CADA caso. El impacto en el prompt
    # (chunk ajeno dentro del umbral → llega al LLM) depende de la calibración del
    # umbral (T8 puede moverlo; gs-10 está a ~0.47 de 0.5), así que basta con que
    # ocurra en al menos un caso cruzado para demostrar que el bug alcanza al LLM.
    assert reached_llm, (
        "ningún caso de dominio cruzado llevó un chunk ajeno al contexto del LLM "
        f"(casos: {[c.id for c in CROSS_DOMAIN_CASES]})"
    )
