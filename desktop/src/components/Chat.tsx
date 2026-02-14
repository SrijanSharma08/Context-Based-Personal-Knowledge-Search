import React, { useState } from "react";
import { Message } from "./Message";
import type { ChatMessage, QueryResponse } from "../api";

interface ChatProps {
  messages: ChatMessage[];
  onSend: (question: string) => Promise<QueryResponse | void>;
  loading: boolean;
  error: string | null;
}

export const Chat: React.FC<ChatProps> = ({ messages, onSend, loading, error }) => {
  const [input, setInput] = useState("");

  const handleSubmit: React.FormEventHandler = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    await onSend(question);
  };

  return (
    <div className="chat">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="placeholder">
            Ask a question about your documents once you have ingested them.
          </div>
        )}
        {messages.map((m, idx) => (
          <Message key={idx} message={m} />
        ))}
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask a question about your documents..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
      {error && <div className="error global-error">{error}</div>}
    </div>
  );
};

