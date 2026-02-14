from __future__ import annotations

import logging
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.errors import EmbeddingModelError

logger = logging.getLogger(__name__)

_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """
    Lazily load and return the global SentenceTransformer embedder.
    Moves the model to GPU if available.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        logger.info(
            "embedding_model_load_start",
            extra={
                "model_name": settings.embedding_model_name,
                "gpu_available": settings.gpu_available,
            },
        )
        start = time.time()
        try:
            model = SentenceTransformer(settings.embedding_model_name)
            if settings.gpu_available:
                model = model.to("cuda")
            _EMBEDDER = model
        except Exception as exc:
            logger.error("embedding_model_load_failed", exc_info=exc)
            raise EmbeddingModelError("Embedding model missing") from exc
        elapsed = time.time() - start
        logger.info(
            "embedding_model_load_end",
            extra={"model_name": settings.embedding_model_name, "elapsed": elapsed},
        )
    return _EMBEDDER


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts into dense vectors.
    """
    if not texts:
        return []
    model = get_embedder()
    logger.info(
        "embedding_encode_start",
        extra={"batch_size": len(texts)},
    )
    start = time.time()
    try:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.error("embedding_encode_failed", exc_info=exc)
        raise EmbeddingModelError("Embedding model missing") from exc
    elapsed = time.time() - start
    logger.info(
        "embedding_encode_end",
        extra={"batch_size": len(texts), "elapsed": elapsed},
    )
    return embeddings.tolist()

