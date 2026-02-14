from __future__ import annotations


class BackendError(Exception):
    """Base class for backend-specific errors."""


class OllamaUnavailable(BackendError):
    """Ollama service is not reachable."""


class OllamaModelMissing(BackendError):
    """Configured Ollama model is not available locally."""


class OllamaRequestFailed(BackendError):
    """Unexpected error from Ollama during generation."""


class EmbeddingModelError(BackendError):
    """Embedding model is missing or failed to load/use."""


class TesseractError(BackendError):
    """Tesseract OCR is not installed or failed."""


class VectorDbError(BackendError):
    """Vector database is not accessible or failed."""

