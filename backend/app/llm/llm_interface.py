from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.config import Settings, settings


class LLMClient(ABC):
    """
    Common interface for local LLM backends.
    """

    @abstractmethod
    def generate_answer(
        self, context: str, question: str, history: Optional[List[dict]] = None
    ) -> str:
        """
        Generate an answer given RAG context, a question, and optional chat history.
        """


_LLM_CLIENT: Optional[LLMClient] = None


def get_llm_client(cfg: Settings | None = None) -> LLMClient:
    """
    Return the configured LLM implementation (Ollama or HuggingFace).
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    from app.llm.ollama_llm import OllamaClient
    from app.llm.hf_llm import HuggingFaceClient

    cfg = cfg or settings
    if cfg.llm_provider == "ollama":
        _LLM_CLIENT = OllamaClient(cfg)
    elif cfg.llm_provider == "hf":
        _LLM_CLIENT = HuggingFaceClient(cfg)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")
    return _LLM_CLIENT

