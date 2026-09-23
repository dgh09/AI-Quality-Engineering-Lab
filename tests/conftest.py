"""Fixtures compartidas: corpus real y vector store en memoria (sin red)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import chromadb
import pytest
from fakes import FakeLLM

from rag_lab.corpus import Document, load_documents
from rag_lab.store import VectorStore


@pytest.fixture(scope="session")
def documents() -> list[Document]:
    return load_documents()


@pytest.fixture(scope="session")
def chroma_client(tmp_path_factory: pytest.TempPathFactory) -> chromadb.ClientAPI:
    """Cliente Chroma para las colecciones temporales de la fixture `store`.

    Es un `PersistentClient` en su propio directorio temporal, distinto del de
    `indexed_store`. En chromadb 1.5.9, crear y borrar (`delete_collection`)
    colecciones repetidamente en un cliente compartido termina corrompiendo el
    segmento HNSW de OTRA colección del mismo cliente (observado tras ~55 ciclos:
    `query` devuelve `[]` o `InternalError: Error creating hnsw segment reader`).
    Por eso los borrados quedan confinados a este cliente. No se usa
    `EphemeralClient`: dentro de un proceso comparten estado y no aíslan.
    """
    return chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("chroma-store")))


@pytest.fixture
def store(chroma_client: chromadb.ClientAPI) -> Iterator[VectorStore]:
    """VectorStore vacío sobre una colección de nombre único, borrada al terminar."""
    name = f"test-{uuid.uuid4().hex}"
    yield VectorStore(chroma_client, collection_name=name)
    chroma_client.delete_collection(name)


@pytest.fixture(scope="session")
def indexed_store(
    tmp_path_factory: pytest.TempPathFactory, documents: list[Document]
) -> VectorStore:
    """VectorStore de SOLO LECTURA con los documentos reales, indexado una vez por sesión.

    Vive en un `PersistentClient` dedicado (directorio temporal propio) del que
    nunca se borra ninguna colección, aislado de los `delete_collection` de la
    fixture `store` (ver `chroma_client`). Las pruebas que indexan o modifican el
    store deben usar `store` (de ámbito función), nunca esta.
    """
    client = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("chroma-indexed")))
    vector_store = VectorStore(client, collection_name="test-indexed")
    vector_store.index(documents)
    return vector_store


@pytest.fixture
def fake_llm() -> FakeLLM:
    """FakeLLM nuevo por prueba, con la respuesta fija `FakeLLM.DEFAULT_ANSWER`."""
    return FakeLLM()
