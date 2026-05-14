"""ChromaDB vector store for document chunk retrieval."""

import os
from typing import Any

import chromadb
from chromadb.config import Settings

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma")
_client = None
_collection = None


def get_client() -> chromadb.ClientAPI:
    """Get or create the shared ChromaDB client (runs embedded)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str = "documents") -> chromadb.Collection:
    """Get or create a named collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(name=name)
    return _collection


def add_document_chunks(
    chunks: list[str],
    metadata: list[dict[str, Any]],
    ids: list[str],
) -> None:
    """Add document chunks to the vector store."""
    collection = get_collection()
    collection.add(
        documents=chunks,
        metadatas=metadata,
        ids=ids,
    )


def query(
    query_text: str,
    n_results: int = 5,
    filter_criteria: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Query the vector store for relevant passages.

    Args:
        query_text: The search query.
        n_results: Number of results to return.
        filter_criteria: Optional metadata filters.

    Returns:
        List of results with document, metadata, and distance.
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=filter_criteria,
    )

    output = []
    if results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            output.append(
                {
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
            )
    return output
