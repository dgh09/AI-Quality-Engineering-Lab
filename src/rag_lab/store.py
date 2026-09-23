"""Vector store sobre Chroma para los documentos de los agentes.

Todos los agentes comparten UNA sola colección. `VectorStore.query` aísla por agente
en la capa de datos cuando recibe `agent_id` (filtro `where` de Chroma). Con
`agent_id=None` conserva el modo shared, con el bug intencional de contaminación de
contexto; se mantiene disponible para el informe antes/después.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chromadb.api import ClientAPI
from chromadb.api.types import Where
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rag_lab.corpus import AGENT_IDS, Document

DEFAULT_COLLECTION_NAME = "knowledge_base"


@dataclass(frozen=True)
class Chunk:
    id: str
    agent_id: str
    text: str
    distance: float  # distancia coseno (0 = idéntico; menor es más relevante)


def _embedding_text(doc: Document) -> str:
    return f"{doc.title}\n{doc.text}"


class VectorStore:
    """Colección única de Chroma con distancia coseno y embeddings locales.

    Usa la función de embeddings por defecto de Chroma (all-MiniLM-L6-v2 vía
    ONNX, local, sin API). Cada documento se guarda y se embebe como
    `title + "\n" + text`, así que `Chunk.text` incluye el título; `agent_id` y
    `title` van en la metadata.

    Los documentos de TODOS los agentes viven en la misma colección. El
    aislamiento se aplica en `query` filtrando por `agent_id`. BUG INTENCIONAL
    (modo shared): con `agent_id=None`, `query` no tiene noción del agente que
    pregunta, por lo que el agente faq puede recuperar documentos de
    seguimiento (contaminación de contexto).
    """

    def __init__(
        self, client: ClientAPI, collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> None:
        self._collection = client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=DefaultEmbeddingFunction(),
        )

    def index(self, documents: Sequence[Document]) -> None:
        """Inserta o actualiza (upsert por id) los documentos; re-indexar no duplica."""
        if not documents:
            return
        self._collection.upsert(
            ids=[doc.id for doc in documents],
            documents=[_embedding_text(doc) for doc in documents],
            metadatas=[{"agent_id": doc.agent_id, "title": doc.title} for doc in documents],
        )

    def count(self) -> int:
        """Número de documentos indexados en la colección."""
        return self._collection.count()

    def query(self, text: str, k: int, *, agent_id: str | None) -> list[Chunk]:
        """Devuelve los `k` chunks más cercanos, ordenados por distancia ascendente.

        `agent_id` es obligatorio y solo por nombre, para que el modo shared sea
        una elección explícita. Con un `agent_id` solo busca entre los documentos
        de ese agente (filtro `where={"agent_id": agent_id}` de Chroma); uno
        desconocido lanza `ValueError`. BUG INTENCIONAL (modo shared): con
        `agent_id=None` busca en los documentos de TODOS los agentes, sin filtro.
        Si no hay documentos candidatos devuelve `[]`; si hay menos de `k`,
        devuelve todos (Chroma 1.5.9 admite `n_results` mayor que los candidatos).
        """
        if k < 1:
            raise ValueError(f"k debe ser >= 1, se recibió {k}")
        if agent_id is not None and agent_id not in AGENT_IDS:
            raise ValueError(f"agent_id {agent_id!r} desconocido; válidos: {AGENT_IDS}")
        where: Where | None = None if agent_id is None else {"agent_id": agent_id}
        result = self._collection.query(
            query_texts=[text],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        texts = result["documents"][0]  # type: ignore[index]
        metadatas = result["metadatas"][0]  # type: ignore[index]
        distances = result["distances"][0]  # type: ignore[index]
        chunks = [
            Chunk(id=i, agent_id=str(m["agent_id"]), text=t, distance=float(d))
            for i, t, m, d in zip(ids, texts, metadatas, distances, strict=True)
        ]
        return sorted(chunks, key=lambda c: c.distance)
