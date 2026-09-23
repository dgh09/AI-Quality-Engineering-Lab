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
def chroma_client() -> chromadb.ClientAPI:
    # Los EphemeralClient de un mismo proceso pueden compartir estado en memoria;
    # por eso cada prueba usa su propia colección (ver `store`).
    return chromadb.EphemeralClient()


@pytest.fixture
def store(chroma_client: chromadb.ClientAPI) -> Iterator[VectorStore]:
    """VectorStore vacío sobre una colección de nombre único, borrada al terminar."""
    name = f"test-{uuid.uuid4().hex}"
    yield VectorStore(chroma_client, collection_name=name)
    chroma_client.delete_collection(name)


@pytest.fixture(scope="session")
def indexed_store(
    chroma_client: chromadb.ClientAPI, documents: list[Document]
) -> Iterator[VectorStore]:
    """VectorStore de SOLO LECTURA con los documentos reales, indexado una vez por sesión.

    Usa su propia colección de nombre único, que se borra al final de la sesión.
    Las pruebas que indexan o modifican el store deben usar la fixture `store`
    (de ámbito función), nunca esta.
    """
    name = f"test-indexed-{uuid.uuid4().hex}"
    vector_store = VectorStore(chroma_client, collection_name=name)
    vector_store.index(documents)
    yield vector_store
    chroma_client.delete_collection(name)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """FakeLLM nuevo por prueba, con la respuesta fija `FakeLLM.DEFAULT_ANSWER`."""
    return FakeLLM()
