import { BACKEND_BASE_URL } from "./config";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
  sources?: Source[];
}

export interface Source {
  source_file?: string | null;
  file_type?: string | null;
  chunk_index?: number | null;
  score?: number | null;
}

export interface IngestFileResult {
  filename: string;
  file_type?: string | null;
  chunks_added: number;
  error?: string | null;
}

export interface IngestResponse {
  results: IngestFileResult[];
  total_chunks: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
}

export interface StatusResponse {
  num_files: number;
  num_chunks: number;
  db_path: string;
  gpu_available: boolean;
  llm_provider: string;
  llm_model: string;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) {
        message += `: ${data.detail}`;
      }
    } catch {
      // ignore JSON errors
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export async function ingestFiles(files: File[]): Promise<IngestResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }

  try {
    const res = await fetch(`${BACKEND_BASE_URL}/ingest`, {
      method: "POST",
      body: form
    });
    return await handleResponse<IngestResponse>(res);
  } catch (err: any) {
    if (err instanceof TypeError) {
      throw new Error(`Failed to reach backend at ${BACKEND_BASE_URL}`);
    }
    throw err;
  }
}

export async function query(
  question: string,
  history: ChatMessage[]
): Promise<QueryResponse> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question,
        history: history.map(({ role, content }) => ({ role, content }))
      })
    });
    return await handleResponse<QueryResponse>(res);
  } catch (err: any) {
    if (err instanceof TypeError) {
      throw new Error(`Failed to reach backend at ${BACKEND_BASE_URL}`);
    }
    throw err;
  }
}

export async function clearDB(): Promise<void> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/clear`, {
      method: "POST"
    });
    await handleResponse<unknown>(res);
  } catch (err: any) {
    if (err instanceof TypeError) {
      throw new Error(`Failed to reach backend at ${BACKEND_BASE_URL}`);
    }
    throw err;
  }
}

export async function getStatus(): Promise<StatusResponse> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/status`);
    return await handleResponse<StatusResponse>(res);
  } catch (err: any) {
    if (err instanceof TypeError) {
      throw new Error(`Failed to reach backend at ${BACKEND_BASE_URL}`);
    }
    throw err;
  }
}

