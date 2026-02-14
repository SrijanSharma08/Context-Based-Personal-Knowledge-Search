# Personal Knowledge Organizer - Backend (FastAPI)

This is a local-only FastAPI backend for a **Personal Knowledge Organizer** desktop app.  
It ingests documents, builds embeddings, stores them in a persistent Chroma vector store, and serves a RAG-style query API for a Tauri desktop frontend.

The service is designed to:

- Run **locally only** (no external SaaS dependencies).
- Use **GPU acceleration** when available via `torch.cuda.is_available()`.
- Be **UI-agnostic** and consumed over HTTP (e.g. `http://127.0.0.1:8000`).

---

## 1. Environment Setup

### 1.1. Create and activate a virtual environment

From the project root (`SDP Project/`):

```bash
cd "backend"
python -m venv .venv

# On Linux
source .venv/bin/activate
```

### 1.2. Install dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. GPU / CUDA Notes

This backend will automatically attempt to use a GPU if:

- `torch` is installed with CUDA support, and  
- `torch.cuda.is_available()` returns `True`.

You can quickly verify this in a Python shell:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

If this prints `True`, the embedding and (optionally) HuggingFace LLM layers will run on GPU.

> **Note:** Installing a CUDA-enabled PyTorch build depends on your system and drivers. See the official PyTorch installation instructions for the correct pip command for your GPU/driver configuration.

---

## 3. Project Structure

Relevant backend layout:

```text
backend/
├── app/
│   ├── main.py               # FastAPI entrypoint
│   ├── config.py             # All configuration (paths, models, chunk sizes, etc.)
│   ├── ingestion/
│   │   ├── loader.py         # File loading + OCR
│   │   ├── chunker.py        # Text splitting
│   ├── embeddings/
│   │   ├── embedder.py       # GPU-aware embedding loader
│   ├── vectorstore/
│   │   ├── chroma_store.py   # ChromaDB logic
│   ├── llm/
│   │   ├── llm_interface.py  # Abstract base class
│   │   ├── ollama_llm.py     # Ollama implementation
│   │   ├── hf_llm.py         # HuggingFace implementation
│   ├── rag/
│   │   ├── pipeline.py       # Retrieval + prompt + generation
│   └── models/
│       ├── schemas.py        # Pydantic request/response models
├── data/
│   ├── chroma_db/            # Persistent Chroma storage
│   └── uploads/              # Uploaded documents
├── requirements.txt
└── README.md
```

The `data/` directory is where the vector database and uploads live; it is safe to delete in development if you want to reset the index.

---

## 4. Running the Backend Server

From `backend/` with your virtual environment activated:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

This starts the FastAPI app on `http://127.0.0.1:8000`, which is what the Tauri desktop app expects.

For a more production-style run (no auto-reload):

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 5. API Overview

Base URL (default): `http://127.0.0.1:8000`

- `POST /ingest`
  - Multipart form-data upload of one or more files.
  - Supported types: `.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`.
  - Extracts text (with optional Tesseract OCR for images), chunks it, embeds, and stores in ChromaDB.

- `POST /query`
  - JSON body: `{ "question": string, "history": [ ... ] }`.
  - Runs a RAG flow: retrieves top-k chunks, builds a strict prompt, and calls the configured LLM.
  - Returns `{ "answer": string, "sources": [ ... ] }`.

- `POST /clear`
  - Clears the Chroma vector store (and optionally related metadata).

- `GET /status`
  - Returns index statistics (number of files, number of chunks, DB path) and GPU status.

---

## 6. Example `curl` Usage

### 6.1. Ingest documents

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "accept: application/json" \
  -F "files=@/path/to/file1.pdf" \
  -F "files=@/path/to/file2.txt"
```

### 6.2. Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does file1 say about project deadlines?",
    "history": []
  }'
```

### 6.3. Clear the knowledge base

```bash
curl -X POST "http://127.0.0.1:8000/clear"
```

### 6.4. Check status

```bash
curl "http://127.0.0.1:8000/status"
```

---

## 7. Relationship to the Tauri Desktop App

This backend is intended to be consumed by a **Tauri-based desktop application** which:

- Runs as a desktop shell (Rust + WebView),
- Hosts a web UI (Vite + React), and
- Talks to this FastAPI service over HTTP on `http://127.0.0.1:8000`.

The desktop app is started separately and assumes the backend is already running.

