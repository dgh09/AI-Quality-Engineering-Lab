import pytest

from rag_lab.corpus import AGENT_IDS, Document
from rag_lab.store import Chunk, VectorStore

SHIPPING_QUESTION = "How much does express shipping cost?"
# Pregunta del dominio de seguimiento (RR. HH.): la haría un cliente al agente faq
# en el caso de dominio cruzado.
VACATION_QUESTION = "How many unused vacation days can employees carry over?"


def test_index_stores_all_documents(store: VectorStore, documents: list[Document]) -> None:
    store.index(documents)
    assert store.count() == len(documents)


def test_reindex_is_idempotent(store: VectorStore, documents: list[Document]) -> None:
    store.index(documents)
    store.index(documents)
    assert store.count() == len(documents)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_query_returns_k_chunks_sorted_by_distance(
    indexed_store: VectorStore, documents: list[Document], k: int
) -> None:
    chunks = indexed_store.query(SHIPPING_QUESTION, k=k, agent_id=None)

    assert len(chunks) == k
    assert all(isinstance(c, Chunk) for c in chunks)
    distances = [c.distance for c in chunks]
    assert distances == sorted(distances)
    doc_agents = {d.id: d.agent_id for d in documents}
    for chunk in chunks:
        assert chunk.id in doc_agents
        assert chunk.agent_id in AGENT_IDS
        assert chunk.agent_id == doc_agents[chunk.id]
        assert chunk.text.strip()


def test_query_k_larger_than_store_returns_all(
    indexed_store: VectorStore, documents: list[Document]
) -> None:
    assert len(indexed_store.query(SHIPPING_QUESTION, k=50, agent_id=None)) == len(documents)


def test_shipping_question_retrieves_shipping_doc_first(indexed_store: VectorStore) -> None:
    top = indexed_store.query(SHIPPING_QUESTION, k=3, agent_id=None)[0]
    assert top.id == "faq-01"
    assert top.agent_id == "faq"


def test_store_has_no_agent_isolation_bug(indexed_store: VectorStore) -> None:
    """Reproduce el bug a nivel de store: no sabe qué agente pregunta.

    Una pregunta de RR. HH. hecha (por ejemplo) al agente faq devuelve un chunk
    de `seguimiento` en top-1. Con el filtro por agent_id de T7, esta prueba
    documenta la llamada sin filtro (`agent_id=None`, modo shared).
    """
    top = indexed_store.query(VACATION_QUESTION, k=3, agent_id=None)[0]
    assert top.agent_id == "seguimiento"


@pytest.mark.parametrize("k", [0, -1])
def test_query_rejects_k_below_one(indexed_store: VectorStore, k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        indexed_store.query(SHIPPING_QUESTION, k=k, agent_id=None)


def test_query_on_empty_store_returns_empty_list(store: VectorStore) -> None:
    assert store.query(SHIPPING_QUESTION, k=3, agent_id=None) == []


def test_query_on_empty_store_still_validates_k(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.query(SHIPPING_QUESTION, k=0, agent_id=None)


# --- Filtro por agent_id (arreglo T7) ------------------------------------------------


def test_query_with_agent_id_returns_only_that_agents_chunks(indexed_store: VectorStore) -> None:
    # Sin filtro, el top-1 de esta pregunta es de `seguimiento` (ver la prueba del bug).
    chunks = indexed_store.query(VACATION_QUESTION, k=3, agent_id="faq")

    assert len(chunks) == 3
    assert {c.agent_id for c in chunks} == {"faq"}
    distances = [c.distance for c in chunks]
    assert distances == sorted(distances)


@pytest.mark.parametrize("agent_id", AGENT_IDS)
def test_query_with_agent_id_and_large_k_returns_exactly_that_agents_docs(
    indexed_store: VectorStore, documents: list[Document], agent_id: str
) -> None:
    own_ids = {d.id for d in documents if d.agent_id == agent_id}
    assert len(own_ids) == 8, "precondición: cada agente tiene 8 documentos"

    chunks = indexed_store.query(SHIPPING_QUESTION, k=50, agent_id=agent_id)

    assert len(chunks) == len(own_ids)
    assert {c.id for c in chunks} == own_ids


def test_query_rejects_unknown_agent_id(indexed_store: VectorStore) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        indexed_store.query(SHIPPING_QUESTION, k=3, agent_id="ventas")
