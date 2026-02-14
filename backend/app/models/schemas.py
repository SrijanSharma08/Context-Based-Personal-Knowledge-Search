from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question about the ingested documents.")
    history: Optional[List[ChatHistoryItem]] = Field(
        default=None, description="Optional chat history for conversational context."
    )


class Source(BaseModel):
    source_file: Optional[str] = None
    file_type: Optional[str] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class IngestFileResult(BaseModel):
    filename: str
    file_type: Optional[str] = None
    chunks_added: int
    error: Optional[str] = None


class IngestResponse(BaseModel):
    results: List[IngestFileResult]
    total_chunks: int


class StatusResponse(BaseModel):
    num_files: int
    num_chunks: int
    db_path: str
    gpu_available: bool
    llm_provider: str
    llm_model: str

