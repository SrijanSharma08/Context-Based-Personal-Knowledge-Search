# Personal Knowledge Organizer - Desktop (Tauri + React)

This is the **desktop UI** for the Personal Knowledge Organizer, built with:

- **Tauri** for the desktop shell
- **Vite + React** for the web frontend
- A separate **Python FastAPI backend** running at `http://127.0.0.1:8000`

The desktop app is intentionally **UI-only** and contains **no ML/RAG logic**. All heavy lifting (ingestion, embeddings, retrieval, LLM) happens in the backend.

---

## 1. Prerequisites

- Node.js (LTS recommended)
- Rust toolchain (via `rustup`)
- Tauri prerequisites for your platform (see Tauri docs)

From the project root:

```bash
cd "desktop"
```

---

## 2. Install Node dependencies

```bash
npm install
```

This installs React, Vite, Tauri CLI, and related tooling.

---

## 3. Backend URL Configuration

The frontend talks to the FastAPI backend via a base URL defined in `src/config.ts`:

```ts
export const BACKEND_BASE_URL = "http://127.0.0.1:8000";
```

If you change the backend host or port, update this constant and rebuild.

> **Note:** The Python backend must be running separately (via `uvicorn app.main:app --host 127.0.0.1 --port 8000`) before you use the desktop app.

---

## 4. Running in Development (Tauri)

From the `desktop/` directory:

```bash
npm run tauri
```

This will:

- Start the Vite dev server for the React app.
- Launch the Tauri shell pointing at the dev server.

You should see:

- A sidebar with file picker, ingest/clear buttons, and a status panel.
- A main chat area where you can ask questions about your ingested documents.

If the backend is not running or unreachable, the UI will show clear error messages (e.g. “Failed to reach backend at http://127.0.0.1:8000”).

---

## 5. Building the Desktop App

To build a distributable Tauri application:

```bash
npm run tauri:build
```

This will:

- Build the React frontend using Vite to `dist/`.
- Package the app via Tauri into a native binary/installer for your platform.

Refer to the Tauri documentation for details on signing and distribution.

---

## 6. Project Structure

```text
desktop/
├── src/
│   ├── main.tsx          # Frontend entry
│   ├── App.tsx           # App root layout
│   ├── api.ts            # API client for FastAPI backend
│   ├── config.ts         # Backend base URL
│   ├── components/
│   │   ├── Sidebar.tsx   # Sidebar: file picker, buttons, status
│   │   ├── Chat.tsx      # Chat interface
│   │   └── Message.tsx   # Single chat message bubble
│   └── styles.css        # Minimal, clean styling
├── src-tauri/
│   └── tauri.conf.json   # Tauri configuration
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## 7. Flow Overview

1. **User selects files** in the sidebar and clicks **“Ingest Documents”**.
   - Frontend calls `POST /ingest` through `ingestFiles()` in `api.ts`.
2. **User asks a question** in the chat.
   - Frontend calls `POST /query` with the question and chat history.
3. **User clears the knowledge base** using **“Clear Knowledge Base”**.
   - Frontend calls `POST /clear`.
4. **Status panel** calls `GET /status` to show:
   - Number of indexed files
   - Number of chunks
   - GPU availability
   - LLM provider/model

This keeps the desktop layer thin and focused on UX, while the Python backend handles all knowledge and model logic.

