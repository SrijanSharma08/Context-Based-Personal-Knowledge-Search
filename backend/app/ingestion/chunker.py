from __future__ import annotations

from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Configure and return a RecursiveCharacterTextSplitter using global settings.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Given a list of normalized document dicts from loader.load_file, return a
    list of chunk dicts with extended metadata, including chunk_index.

    Input document format:
      { "text": str, "metadata": { ... } }

    Output chunk format:
      {
        "id": str (optional, may be filled by caller),
        "text": str,
        "metadata": {
          ...original metadata...,
          "chunk_index": int
        }
      }
    """
    splitter = get_text_splitter()
    chunks: List[Dict] = []
    for doc in documents:
        text: str = doc.get("text", "") or ""
        base_meta: Dict = dict(doc.get("metadata", {}))
        if not text.strip():
            continue
        split_texts = splitter.split_text(text)
        for idx, chunk_text in enumerate(split_texts):
            metadata = dict(base_meta)
            metadata["chunk_index"] = idx
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )
    return chunks

