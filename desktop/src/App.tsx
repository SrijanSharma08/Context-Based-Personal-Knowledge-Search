import React, { useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./components/Chat";
import {
  ChatMessage,
  ingestFiles,
  getStatus,
  clearDB,
  query,
  StatusResponse
} from "./api";

export const App: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const [loadingIngest, setLoadingIngest] = useState(false);
  const [loadingClear, setLoadingClear] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [loadingQuery, setLoadingQuery] = useState(false);

  const [sidebarError, setSidebarError] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);

  const refreshStatus = async () => {
    setLoadingStatus(true);
    setSidebarError(null);
    try {
      const s = await getStatus();
      setStatus(s);
    } catch (err: any) {
      setSidebarError(err.message ?? String(err));
    } finally {
      setLoadingStatus(false);
    }
  };

  useEffect(() => {
    // Initial status fetch
    void refreshStatus();
  }, []);

  const handleIngest = async () => {
    if (selectedFiles.length === 0) return;
    setLoadingIngest(true);
    setSidebarError(null);
    try {
      await ingestFiles(selectedFiles);
      await refreshStatus();
    } catch (err: any) {
      setSidebarError(err.message ?? String(err));
    } finally {
      setLoadingIngest(false);
    }
  };

  const handleClear = async () => {
    setLoadingClear(true);
    setSidebarError(null);
    try {
      await clearDB();
      setMessages([]);
      await refreshStatus();
    } catch (err: any) {
      setSidebarError(err.message ?? String(err));
    } finally {
      setLoadingClear(false);
    }
  };

  const handleSend = async (question: string) => {
    setLoadingQuery(true);
    setChatError(null);
    const userMessage: ChatMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    try {
      const resp = await query(question, messages.concat(userMessage));
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: resp.answer,
        sources: resp.sources
      };
      setMessages((prev) => [...prev, assistantMessage]);
      await refreshStatus();
      return resp;
    } catch (err: any) {
      setChatError(err.message ?? String(err));
    } finally {
      setLoadingQuery(false);
    }
  };

  return (
    <div className="app-root">
      <Sidebar
        selectedFiles={selectedFiles}
        onFilesChange={setSelectedFiles}
        onIngest={handleIngest}
        onClear={handleClear}
        status={status}
        loadingIngest={loadingIngest}
        loadingClear={loadingClear}
        loadingStatus={loadingStatus}
        error={sidebarError}
      />
      <Chat
        messages={messages}
        onSend={handleSend}
        loading={loadingQuery}
        error={chatError}
      />
    </div>
  );
};

