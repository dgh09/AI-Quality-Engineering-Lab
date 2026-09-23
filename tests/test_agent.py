"""Pruebas del RagAgent: compuerta de abstención y construcción del prompt (sin LLM real)."""

from __future__ import annotations

import pytest

from fakes import FakeLLM
from rag_lab.agent import (
    ABSTENTION_MESSAGE,
    AgentResponse,
    RagAgent,
    build_system_prompt,
    build_user_prompt,
)
from rag_lab.corpus import AGENT_IDS, Document
from rag_lab.store import Chunk, VectorStore

SHIPPING_QUESTION = "How much does express shipping cost?"
TOP_K = 3
# Umbral generoso: la distancia coseno está en [0, 2], así que todo chunk pasa.
GENEROUS_THRESHOLD = 2.0


def _doc_text(documents: list[Document], doc_id: str) -> str:
    return next(d.text for d in documents if d.id == doc_id)


# --- Compuerta de abstención ------------------------------------------------------


def test_agent_abstains_without_calling_llm_when_no_chunk_is_within_threshold(
    indexed_store: VectorStore, fake_llm: FakeLLM
) -> None:
    # Umbral derivado de las distancias reales: la mitad de la mínima, así que
    # ningún chunk puede pasar la compuerta.
    min_distance = min(
        c.distance for c in indexed_store.query(SHIPPING_QUESTION, k=TOP_K, agent_id=None)
    )
    assert min_distance > 0, "precondición: se necesita una distancia mínima positiva"
    tiny_threshold = min_distance / 2
    agent = RagAgent("faq", indexed_store, fake_llm, threshold=tiny_threshold, top_k=TOP_K)

    response = agent.ask(SHIPPING_QUESTION)

    assert isinstance(response, AgentResponse)
    assert response.abstained is True
    assert response.answer == ABSTENTION_MESSAGE
    assert response.context == ()
    assert len(response.retrieved) == TOP_K
    assert all(c.distance > tiny_threshold for c in response.retrieved)
    assert fake_llm.calls == []


def test_agent_answers_from_close_chunks_calling_llm_once(
    indexed_store: VectorStore, fake_llm: FakeLLM, documents: list[Document]
) -> None:
    agent = RagAgent("faq", indexed_store, fake_llm, threshold=GENEROUS_THRESHOLD, top_k=TOP_K)

    response = agent.ask(SHIPPING_QUESTION)

    assert response.abstained is False
    assert response.answer == FakeLLM.DEFAULT_ANSWER
    assert len(fake_llm.calls) == 1
    system, user = fake_llm.calls[0]
    assert system == build_system_prompt()
    assert _doc_text(documents, "faq-01") in user
    assert SHIPPING_QUESTION in user
    assert "faq-01" in [c.id for c in response.context]
    assert all(c.distance <= GENEROUS_THRESHOLD for c in response.context)


def test_context_excludes_chunks_above_threshold(
    indexed_store: VectorStore, fake_llm: FakeLLM
) -> None:
    ranked = indexed_store.query(SHIPPING_QUESTION, k=TOP_K, agent_id=None)
    top1, top2 = ranked[0].distance, ranked[1].distance
    assert top1 < top2, "se necesitan distancias distintas para separar top-1 de top-2"
    threshold = (top1 + top2) / 2
    agent = RagAgent("faq", indexed_store, fake_llm, threshold=threshold, top_k=TOP_K)

    response = agent.ask(SHIPPING_QUESTION)

    assert response.abstained is False
    assert [c.id for c in response.context] == [ranked[0].id]
    assert len(response.retrieved) == TOP_K
    assert len(fake_llm.calls) == 1
    _, user = fake_llm.calls[0]
    assert ranked[0].text in user
    for excluded in ranked[1:]:
        assert excluded.text not in user


# --- Prompts (funciones puras) -----------------------------------------------------


def test_system_prompt_does_not_mention_agents_or_filtering() -> None:
    # El aislamiento vive en la capa de datos (T7), nunca en el prompt.
    system = build_system_prompt().lower()

    for agent_id in AGENT_IDS:
        assert agent_id.lower() not in system
    for word in ("agent", "filter", "isolat"):
        assert word not in system


def test_system_prompt_instructs_exact_abstention_message() -> None:
    # Permite detectar la abstención del LLM de forma determinista en la evaluación.
    assert ABSTENTION_MESSAGE in build_system_prompt()


def test_user_prompt_numbers_chunks_and_omits_agent_metadata() -> None:
    chunks = (
        Chunk(id="x-1", agent_id="seguimiento", text="Alpha text.", distance=0.1),
        Chunk(id="x-2", agent_id="seguimiento", text="Beta text.", distance=0.2),
    )

    user = build_user_prompt("What is alpha?", chunks)

    assert "[1] Alpha text." in user
    assert "[2] Beta text." in user
    assert user.index("[1]") < user.index("[2]")
    assert "What is alpha?" in user
    assert "seguimiento" not in user


# --- Validación ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"agent_id": "ventas"}, "agent_id"),
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": -0.5}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"top_k": 0}, "top_k"),
    ],
)
def test_invalid_constructor_arguments_raise_value_error(
    indexed_store: VectorStore, fake_llm: FakeLLM, kwargs: dict[str, object], match: str
) -> None:
    params: dict[str, object] = {"agent_id": "faq", "threshold": 0.5, "top_k": TOP_K}
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        RagAgent(store=indexed_store, llm=fake_llm, **params)  # type: ignore[arg-type]


# --- Modo aislado (T7) -------------------------------------------------------------

VACATION_QUESTION = "How many unused vacation days can employees carry over?"


@pytest.mark.parametrize("agent_id", AGENT_IDS)
def test_isolated_agent_retrieves_only_its_own_chunks(
    indexed_store: VectorStore, fake_llm: FakeLLM, agent_id: str
) -> None:
    agent = RagAgent(
        agent_id, indexed_store, fake_llm, threshold=GENEROUS_THRESHOLD, top_k=TOP_K,
        isolated=True,
    )

    for question in (SHIPPING_QUESTION, VACATION_QUESTION):
        response = agent.ask(question)
        assert len(response.retrieved) == TOP_K
        assert {c.agent_id for c in response.retrieved} == {agent_id}
