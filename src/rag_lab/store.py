"""Vector store sobre Chroma para los documentos de los agentes.

ATENCIÓN — versión con el bug intencional de contaminación de contexto: todos los
agentes comparten UNA sola colección y `VectorStore.query` no filtra por agente.
El arreglo (filtro por `agent_id`) se implementa en T7.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rag_lab.corpus import Document

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

    BUG INTENCIONAL (se arregla en T7): los documentos de TODOS los agentes
    viven en la misma colección y `query` no tiene noción del agente que
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

    def query(self, text: str, k: int) -> list[Chunk]:
        """Devuelve los `k` chunks más cercanos, ordenados por distancia ascendente.

        BUG INTENCIONAL (se arregla en T7): busca en los documentos de TODOS los
        agentes, sin filtro por `agent_id`. Si el store está vacío devuelve `[]`;
        si hay menos de `k` documentos, devuelve todos.
        """
        if k < 1:
            raise ValueError(f"k debe ser >= 1, se recibió {k}")
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_texts=[text],
            n_results=min(k, total),
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
