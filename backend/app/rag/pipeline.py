from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.llm.llm_interface import get_llm_client
from app.vectorstore.chroma_store import query_similar

logger = logging.getLogger(__name__)


def _format_context_and_sources(
    results: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Turn raw Chroma query results into a context string and list of sources.
    """
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    context_lines: List[str] = []
    sources: List[Dict[str, Any]] = []

    for docs_row, metas_row, dists_row in zip(documents, metadatas, distances):
        for doc, meta, dist in zip(docs_row, metas_row, dists_row):
            source_file = (meta or {}).get("source_file")
            chunk_index = (meta or {}).get("chunk_index")
            file_type = (meta or {}).get("file_type")

            context_lines.append(
                f"[source={source_file} chunk={chunk_index}] {doc}"
            )
            sources.append(
                {
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "file_type": file_type,
                    "score": float(dist) if dist is not None else None,
                }
            )

    context = "\n\n".join(context_lines)
    return context, sources


def answer_question(
    question: str, history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    High-level RAG entrypoint.

    - Retrieve top-k chunks from the vector store.
    - Build a strict instruction to answer only from context.
    - Call the configured LLM.

    Returns a dict: { "answer": str, "sources": list[dict] }.
    """
    logger.info("retrieval_start", extra={"top_k": settings.top_k})
    start_retrieval = time.time()
    retrieval = query_similar(question, top_k=settings.top_k)
    elapsed_retrieval = time.time() - start_retrieval
    logger.info(
        "retrieval_end",
        extra={"top_k": settings.top_k, "elapsed": elapsed_retrieval},
    )

    context, sources = _format_context_and_sources(retrieval)

    if not context.strip():
        # Short-circuit if nothing was retrieved
        return {
            "answer": "The answer is not found in the provided documents.",
            "sources": [],
        }

    llm = get_llm_client()
    logger.info("llm_call_start")
    start_llm = time.time()
    answer = llm.generate_answer(context=context, question=question, history=history)
    elapsed_llm = time.time() - start_llm
    logger.info("llm_call_end", extra={"elapsed": elapsed_llm})

    return {
        "answer": answer.strip(),
        "sources": sources,
    }

