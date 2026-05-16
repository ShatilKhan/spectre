"""ChromaDB vector store for document chunk retrieval with text chunking."""

import os
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/home/user/app/data/chroma")
_client = None


def get_client() -> chromadb.ClientAPI:
    """Get or create the shared ChromaDB client (runs embedded, persists to disk)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str = "documents") -> chromadb.Collection:
    """Get or create a named collection."""
    client = get_client()
    return client.get_or_create_collection(name=name)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Args:
        text: Full document text.
        chunk_size: Target characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Index a document's text into ChromaDB by chunking and embedding.

    Args:
        doc_id: Unique document identifier.
        text: Full document text to chunk and store.
        metadata: Optional metadata dict (e.g. file_name, doc_type, page_count).

    Returns:
        Number of chunks stored.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    metadata = metadata or {}
    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        chunk_meta = {**metadata, "chunk_index": i, "doc_id": doc_id}
        ids.append(chunk_id)
        metadatas.append(chunk_meta)

    collection = get_collection()
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def index_pages(
    doc_id: str,
    pages: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> int:
    """Index each OCR page as a separate entry with its page number.

    Args:
        doc_id: Unique document identifier.
        pages: List of dicts with 'page' (int) and 'text' (str).
        metadata: Optional metadata dict.

    Returns:
        Number of pages indexed.
    """
    metadata = metadata or {}
    ids = []
    documents = []
    metadatas = []
    for i, p in enumerate(pages):
        page_num = p.get("page", i + 1)
        text = p.get("text", "").strip()
        if not text:
            continue
        chunk_id = f"{doc_id}_page_{page_num}"
        chunk_meta = {
            **metadata,
            "page": page_num,
            "chunk_index": i,
            "doc_id": doc_id,
        }
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(chunk_meta)

    if not documents:
        return 0
    collection = get_collection()
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


def query(
    query_text: str,
    n_results: int = 5,
    filter_criteria: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Query the vector store for relevant passages.

    Args:
        query_text: The search query (e.g. "What are the payment terms?").
        n_results: Number of results to return.
        filter_criteria: Optional metadata filters (e.g. {"doc_type": "nda"}).

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


def retrieve_for_draft(
    extracted_data: dict[str, Any],
    ocr_text: str,
    pages: list[dict[str, Any]] | None = None,
    doc_id: str | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant passages for draft generation with page-aware indexing.

    First indexes the OCR text by page (so each citation has a page number),
    then queries for passages relevant to the extracted data context.

    Args:
        extracted_data: The structured extraction result.
        ocr_text: Full OCR text from the document.
        pages: List of dicts with 'page' (int) and 'text' (str) per page.
        doc_id: Unique ID (auto-generated if not provided).
        n_results: Max passages to retrieve.

    Returns:
        List of relevant passages with page numbers in metadata.
    """
    if doc_id is None:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    # Index pages (with page numbers) if not already stored
    collection = get_collection()
    existing = collection.get(ids=[f"{doc_id}_page_1"])
    if (not existing or not existing["documents"]) and pages:
        index_pages(doc_id, pages, {
            "doc_id": doc_id,
            "file_name": extracted_data.get("file_name", ""),
            "doc_type": extracted_data.get("doc_type", ""),
        })

    # Build a query from the extracted fields
    query_parts = []
    for key, value in extracted_data.items():
        if isinstance(value, str) and len(value) > 20:
            query_parts.append(value)
        elif isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, str) and len(v2) > 20:
                    query_parts.append(v2)
    query_text = " ".join(query_parts[:3]) if query_parts else ocr_text[:500]

    return query(query_text, n_results=n_results)
