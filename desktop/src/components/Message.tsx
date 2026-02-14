import React from "react";
import type { ChatMessage } from "../api";

interface MessageProps {
  message: ChatMessage;
}

export const Message: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-role">{isUser ? "You" : "Assistant"}</div>
      <div className="message-content">{message.content}</div>
      {!isUser && message.sources && message.sources.length > 0 && (
        <div className="message-sources">
          <div className="sources-title">Sources:</div>
          <ul>
            {message.sources.map((s, idx) => (
              <li key={idx}>
                {s.source_file ?? "Unknown file"} (chunk {s.chunk_index ?? "?"})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

