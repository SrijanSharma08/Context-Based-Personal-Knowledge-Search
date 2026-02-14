import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field
import torch


class Settings(BaseSettings):
    """
    Central configuration for the backend.

    Values can be overridden via environment variables where useful, but
    sane local defaults are provided for a simple local setup.
    """

    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    data_dir: Path = Field(default_factory=lambda: Path("backend/data"))
    uploads_dir_name: str = "uploads"
    chroma_db_dir_name: str = "chroma_db"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 200

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM configuration
    llm_provider: Literal["ollama", "hf"] = "ollama"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_temperature: float = 0.1
    ollama_max_tokens: int = 512

    # Timeouts (seconds)
    ollama_timeout_seconds: int = 60
    health_timeout_seconds: int = 5
    ocr_timeout_seconds: int = 30

    # HuggingFace Transformers (local models)
    hf_model_name: str = "gpt2"
    hf_temperature: float = 0.7
    hf_max_new_tokens: int = 512

    # Retrieval / RAG
    top_k: int = 6

    class Config:
        env_prefix = "PKO_"
        case_sensitive = False

    @property
    def uploads_path(self) -> Path:
        base = self.project_root / self.data_dir
        return base / self.uploads_dir_name

    @property
    def chroma_db_path(self) -> Path:
        base = self.project_root / self.data_dir
        return base / self.chroma_db_dir_name

    @property
    def gpu_available(self) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


settings = Settings()


def ensure_directories() -> None:
    """
    Ensure that required data directories exist.
    """
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)

