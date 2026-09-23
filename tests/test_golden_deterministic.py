"""Golden set completo en modo `isolated`, sin LLM real (FakeLLM).

Cada caso se abstiene o responde según su `expected_behavior` con el umbral por
defecto calibrado (`DEFAULT_RELEVANCE_THRESHOLD`), y los casos in-domain recuperan
sus `expected_source_ids` dentro del top-k. Usa `indexed_store` (solo lectura).
"""

from __future__ import annotations

import pytest
from fakes import FakeLLM

from rag_lab.agent import ABSTENTION_MESSAGE, RagAgent
from rag_lab.config import DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K
from rag_lab.corpus import GoldenCase, load_golden_set
from rag_lab.store import VectorStore

GOLDEN_SET: list[GoldenCase] = load_golden_set()


@pytest.mark.parametrize("case", GOLDEN_SET, ids=[c.id for c in GOLDEN_SET])
def test_isolated_agent_follows_expected_behavior(
    case: GoldenCase, indexed_store: VectorStore, fake_llm: FakeLLM
) -> None:
    agent = RagAgent(
        case.agent_id,
        indexed_store,
        fake_llm,
        threshold=DEFAULT_RELEVANCE_THRESHOLD,
        top_k=DEFAULT_TOP_K,
        isolated=True,
    )

    response = agent.ask(case.question)

    distances = {chunk.id: round(chunk.distance, 4) for chunk in response.retrieved}
    assert all(chunk.agent_id == case.agent_id for chunk in response.retrieved), distances
    if case.expected_behavior == "abstain":
        assert response.abstained is True, f"{case.id}: no se abstuvo; top-k={distances}"
        assert response.answer == ABSTENTION_MESSAGE
        assert response.context == ()
        assert fake_llm.calls == []
    else:
        assert response.abstained is False, f"{case.id}: se abstuvo; top-k={distances}"
        assert response.answer == FakeLLM.DEFAULT_ANSWER
        assert len(fake_llm.calls) == 1
        retrieved_ids = {chunk.id for chunk in response.retrieved}
        context_ids = {chunk.id for chunk in response.context}
        assert set(case.expected_source_ids) <= retrieved_ids, (
            f"{case.id}: fuentes esperadas {case.expected_source_ids} fuera del "
            f"top-{DEFAULT_TOP_K}; top-k={distances}"
        )
        # La fuente esperada no solo se recupera: pasa la compuerta y llega al LLM.
        assert set(case.expected_source_ids) <= context_ids, (
            f"{case.id}: fuentes esperadas {case.expected_source_ids} por encima del "
            f"umbral {DEFAULT_RELEVANCE_THRESHOLD}; top-k={distances}"
        )
