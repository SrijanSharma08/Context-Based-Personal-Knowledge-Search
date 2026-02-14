import React from "react";
import type { StatusResponse } from "../api";

interface SidebarProps {
  selectedFiles: File[];
  onFilesChange: (files: File[]) => void;
  onIngest: () => void;
  onClear: () => void;
  status: StatusResponse | null;
  loadingIngest: boolean;
  loadingClear: boolean;
  loadingStatus: boolean;
  error: string | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  selectedFiles,
  onFilesChange,
  onIngest,
  onClear,
  status,
  loadingIngest,
  loadingClear,
  loadingStatus,
  error
}) => {
  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    onFilesChange(files);
  };

  return (
    <div className="sidebar">
      <h2>Knowledge Base</h2>

      <div className="section">
        <label className="label">Select documents</label>
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          accept=".txt,.md,.pdf,.png,.jpg,.jpeg"
        />
        <div className="small">
          {selectedFiles.length === 0
            ? "No files selected."
            : `${selectedFiles.length} file(s) selected.`}
        </div>
      </div>

      <div className="section buttons">
        <button
          onClick={onIngest}
          disabled={loadingIngest || selectedFiles.length === 0}
        >
          {loadingIngest ? "Ingesting..." : "Ingest Documents"}
        </button>
        <button onClick={onClear} disabled={loadingClear}>
          {loadingClear ? "Clearing..." : "Clear Knowledge Base"}
        </button>
      </div>

      <div className="section status">
        <div className="label-row">
          <span className="label">Status</span>
          {loadingStatus && <span className="small">Refreshing...</span>}
        </div>
        {status ? (
          <ul className="status-list">
            <li>
              <strong>Connection:</strong> Connected
            </li>
            <li>
              <strong>Files:</strong> {status.num_files}
            </li>
            <li>
              <strong>Chunks:</strong> {status.num_chunks}
            </li>
            <li>
              <strong>GPU:</strong> {status.gpu_available ? "Available" : "CPU only"}
            </li>
            <li>
              <strong>LLM:</strong> {status.llm_provider} ({status.llm_model})
            </li>
            <li className="small">
              <strong>DB path:</strong> {status.db_path}
            </li>
          </ul>
        ) : (
          <div className="small">No status yet. Try refreshing via ingest/query.</div>
        )}
        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
};

