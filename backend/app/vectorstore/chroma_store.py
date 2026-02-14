from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.api import Collection

from app.config import settings, ensure_directories
from app.embeddings.embedder import embed_texts
from app.errors import VectorDbError

logger = logging.getLogger(__name__)

_CLIENT: chromadb.Client | None = None
_COLLECTION: Collection | None = None
_COLLECTION_NAME = "pko_documents"


def _get_client() -> chromadb.Client:
    global _CLIENT
    if _CLIENT is None:
        ensure_directories()
        logger.info("Initializing ChromaDB client at %s", settings.chroma_db_path)
        try:
            _CLIENT = chromadb.PersistentClient(path=str(settings.chroma_db_path))
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB client", exc_info=exc)
            raise VectorDbError("Vector DB error") from exc
    return _CLIENT


def get_vector_store() -> Collection:
    """
    Return the singleton ChromaDB collection used by the application.
    """
    global _COLLECTION
    if _COLLECTION is None:
        client = _get_client()
        try:
            _COLLECTION = client.get_or_create_collection(name=_COLLECTION_NAME)
        except Exception as exc:
            logger.error("Failed to get_or_create_collection", exc_info=exc)
            raise VectorDbError("Vector DB error") from exc
    return _COLLECTION


def add_documents(chunks: List[Dict[str, Any]]) -> int:
    """
    Add chunk documents to the Chroma collection.

    Each chunk is expected to be of the form:
      { "text": str, "metadata": dict }

    Returns the number of chunks successfully added.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = embed_texts(texts)

    collection = get_vector_store()
    try:
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    except Exception as exc:
        logger.error("Failed to add documents to Chroma", exc_info=exc)
        raise VectorDbError("Vector DB error") from exc
    return len(chunks)


def clear_vector_store() -> None:
    """
    Clear all data from the Chroma vector store, including on-disk persistence.
    """
    global _CLIENT, _COLLECTION
    _COLLECTION = None
    _CLIENT = None

    path: Path = settings.chroma_db_path
    try:
        if path.exists():
            shutil.rmtree(path)
        ensure_directories()
    except Exception as exc:
        logger.error("Failed to clear Chroma vector store", exc_info=exc)
        raise VectorDbError("Vector DB error") from exc


def list_indexed_files() -> List[Dict[str, Any]]:
    """
    Return a list of unique source files with basic statistics.

    Each entry:
      {
        "source_file": str,
        "file_type": str | None,
        "num_chunks": int
      }
    """
    collection = get_vector_store()
    try:
        result = collection.get(include=["metadatas"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to read from Chroma collection: %s", exc)
        return []

    metadatas = result.get("metadatas") or []
    stats: Dict[Tuple[str, str | None], int] = {}

    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source_file")
        ftype = meta.get("file_type")
        if not source:
            continue
        key = (str(source), ftype if isinstance(ftype, str) else None)
        stats[key] = stats.get(key, 0) + 1

    files: List[Dict[str, Any]] = []
    for (source, ftype), num_chunks in stats.items():
        files.append(
            {
                "source_file": source,
                "file_type": ftype,
                "num_chunks": num_chunks,
            }
        )
    return files


def count_chunks() -> int:
    """
    Return total number of chunks stored in the collection.
    """
    collection = get_vector_store()
    try:
        return collection.count()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to count Chroma collection: %s", exc)
        return 0


def query_similar(query_text: str, top_k: int) -> Dict[str, Any]:
    """
    Query the vector store for similar chunks.

    Returns the raw Chroma query result so the caller can format context and
    sources as needed.
    """
    collection = get_vector_store()
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results
    except Exception as exc:
        logger.error("Failed to query Chroma", exc_info=exc)
        raise VectorDbError("Vector DB error") from exc

