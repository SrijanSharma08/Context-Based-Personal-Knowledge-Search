from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.config import ensure_directories, settings
from app.health import collect_health_snapshot
from app.errors import (
    OllamaUnavailable,
    OllamaModelMissing,
    OllamaRequestFailed,
    EmbeddingModelError,
    VectorDbError,
    TesseractError,
)
from app.ingestion.loader import load_file, detect_file_type
from app.ingestion.chunker import chunk_documents
from app.models.schemas import (
    IngestFileResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from app.rag.pipeline import answer_question
from app.vectorstore.chroma_store import (
    add_documents,
    clear_vector_store,
    count_chunks,
    list_indexed_files,
)


LOG_LEVEL = os.environ.get("PKO_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    ensure_directories()

    app = FastAPI(
        title="Personal Knowledge Organizer Backend",
        description="FastAPI backend for document ingestion and RAG-style querying.",
        version="0.1.0",
    )

    # CORS for Tauri (localhost webview)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "tauri://localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Startup dependency checks
    health = collect_health_snapshot()
    for name, data in health.items():
        logger.info(
            "health_check",
            extra={"check": name, "ok": data["ok"], "detail": data["message"]},
        )

    # Ollama and model are hard requirements for queries
    if not health["ollama"]["ok"]:
        raise RuntimeError("Ollama not running. Please start Ollama.")
    if not health["llm_model"]["ok"]:
        raise RuntimeError(str(health["llm_model"]["message"] or "LLM model missing in Ollama."))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        results: List[IngestFileResult] = []
        total_chunks = 0

        for up in files:
            filename = os.path.basename(up.filename or "unknown")
            save_path = settings.uploads_path / filename
            try:
                # Save uploaded file to disk
                content = await up.read()
                save_path.write_bytes(content)

                # Load and chunk
                docs = load_file(save_path)
                chunks = chunk_documents(docs)
                added = add_documents(chunks)
                total_chunks += added

                file_type = None
                if docs:
                    file_type = docs[0].get("metadata", {}).get("file_type")

                results.append(
                    IngestFileResult(
                        filename=filename,
                        file_type=file_type,
                        chunks_added=added,
                        error=None,
                    )
                )
            except TesseractError as exc:
                logger.warning("Tesseract error while ingesting %s: %s", filename, exc)
                ftype = detect_file_type(Path(filename)) or None
                results.append(
                    IngestFileResult(
                        filename=filename,
                        file_type=ftype,
                        chunks_added=0,
                        error=str(exc) or "Tesseract not installed",
                    )
                )
            except EmbeddingModelError as exc:
                logger.warning(
                    "Embedding error while ingesting %s: %s", filename, exc
                )
                ftype = detect_file_type(Path(filename)) or None
                results.append(
                    IngestFileResult(
                        filename=filename,
                        file_type=ftype,
                        chunks_added=0,
                        error="Embedding model missing",
                    )
                )
            except VectorDbError:
                ftype = detect_file_type(Path(filename)) or None
                results.append(
                    IngestFileResult(
                        filename=filename,
                        file_type=ftype,
                        chunks_added=0,
                        error="Vector DB error",
                    )
                )
            except Exception as exc:
                logger.exception("Failed to ingest file %s: %s", filename, exc)
                ftype = detect_file_type(Path(filename)) or None
                results.append(
                    IngestFileResult(
                        filename=filename,
                        file_type=ftype,
                        chunks_added=0,
                        error=str(exc),
                    )
                )

        return IngestResponse(results=results, total_chunks=total_chunks)

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        try:
            rag_result = await run_in_threadpool(
                answer_question,
                req.question,
                [h.dict() for h in (req.history or [])],
            )
        except OllamaUnavailable:
            raise HTTPException(status_code=503, detail="Ollama not running")
        except OllamaModelMissing as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model not found: {settings.ollama_model}",
            ) from exc
        except EmbeddingModelError:
            raise HTTPException(status_code=500, detail="Embedding model missing")
        except VectorDbError:
            raise HTTPException(status_code=500, detail="Vector DB error")
        except OllamaRequestFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("RAG pipeline failed: %s", exc)
            raise HTTPException(status_code=500, detail="RAG pipeline failed.") from exc

        return QueryResponse(
            answer=rag_result["answer"],
            sources=rag_result.get("sources") or [],
        )

    @app.post("/clear")
    async def clear() -> JSONResponse:
        try:
            clear_vector_store()
            # Optionally clear uploads (keep directory)
            for path in settings.uploads_path.glob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
        except VectorDbError:
            raise HTTPException(status_code=500, detail="Vector DB error")
        except Exception as exc:
            logger.exception("Failed to clear vector store: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to clear vector store.")
        return JSONResponse({"status": "ok"})

    @app.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        files = list_indexed_files()
        num_chunks = count_chunks()

        if settings.llm_provider == "ollama":
            llm_model = settings.ollama_model
        else:
            llm_model = settings.hf_model_name

        return StatusResponse(
            num_files=len(files),
            num_chunks=num_chunks,
            db_path=str(settings.chroma_db_path),
            gpu_available=settings.gpu_available,
            llm_provider=settings.llm_provider,
            llm_model=llm_model,
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        snapshot = collect_health_snapshot()
        return JSONResponse(snapshot)

    return app


app = create_app()