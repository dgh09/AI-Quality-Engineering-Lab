"""Agente RAG: recuperar → compuerta de abstención → generar con un LLM inyectado.

La compuerta de abstención es determinista: si ningún chunk recuperado tiene
`distance <= threshold`, el agente responde `ABSTENTION_MESSAGE` sin llamar al LLM.

Regla de diseño: el aislamiento entre agentes se implementa en la capa de datos
(T7), NUNCA en el prompt. Los prompts no mencionan agentes ni filtros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.corpus import AGENT_IDS
from rag_lab.llm import LLMClient
from rag_lab.store import Chunk, VectorStore

ABSTENTION_MESSAGE = "I don't have information about that in my knowledge base."

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information in the provided context. Do not use outside knowledge. If the "
    "context does not contain the answer, reply with exactly this sentence and "
    f"nothing else: {ABSTENTION_MESSAGE} Keep the answer concise."
)


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    abstained: bool
    # Todos los chunks top-k devueltos por el store (para detectar contaminación).
    retrieved: tuple[Chunk, ...]
    # Subconjunto con distance <= threshold, el que realmente recibe el LLM.
    context: tuple[Chunk, ...]


def build_system_prompt() -> str:
    """Prompt de sistema: responder solo desde el contexto o decir que no se sabe."""
    return _SYSTEM_PROMPT


def build_user_prompt(question: str, context: Sequence[Chunk]) -> str:
    """Prompt de usuario con los chunks numerados ([1], [2], ...) y la pregunta.

    Solo incluye el texto de cada chunk: ni ids ni agent_id.
    """
    numbered = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(context, start=1))
    return f"Context:\n{numbered}\n\nQuestion: {question}"


def select_context(retrieved: Sequence[Chunk], threshold: float) -> tuple[Chunk, ...]:
    """Chunks con `distance <= threshold`, en el orden recibido."""
    return tuple(chunk for chunk in retrieved if chunk.distance <= threshold)


class RagAgent:
    """Agente RAG de un `agent_id` concreto sobre un `VectorStore` y un `LLMClient`.

    Con `isolated=True` la recuperación se filtra por `agent_id` en la capa de
    datos (`VectorStore.query(..., agent_id=...)`); con `isolated=False` usa el
    modo shared (sin filtro, con el bug de contaminación). El prompt es el mismo
    en ambos modos.
    """

    def __init__(
        self,
        agent_id: str,
        store: VectorStore,
        llm: LLMClient,
        threshold: float,
        top_k: int,
        isolated: bool = False,
    ) -> None:
        if agent_id not in AGENT_IDS:
            raise ValueError(f"agent_id {agent_id!r} desconocido; válidos: {AGENT_IDS}")
        if not threshold > 0:  # también rechaza NaN
            raise ValueError(f"threshold debe ser > 0, se recibió {threshold}")
        if top_k < 1:
            raise ValueError(f"top_k debe ser >= 1, se recibió {top_k}")
        self.agent_id = agent_id
        self.threshold = threshold
        self.top_k = top_k
        self.isolated = isolated
        self._store = store
        self._llm = llm

    def ask(self, question: str) -> AgentResponse:
        """Recupera top-k, aplica la compuerta de abstención y, si pasa, llama al LLM."""
        agent_filter = self.agent_id if self.isolated else None
        retrieved = tuple(self._store.query(question, k=self.top_k, agent_id=agent_filter))
        context = select_context(retrieved, self.threshold)
        if not context:
            return AgentResponse(
                answer=ABSTENTION_MESSAGE, abstained=True, retrieved=retrieved, context=()
            )
        answer = self._llm.complete(build_system_prompt(), build_user_prompt(question, context))
        return AgentResponse(answer=answer, abstained=False, retrieved=retrieved, context=context)
