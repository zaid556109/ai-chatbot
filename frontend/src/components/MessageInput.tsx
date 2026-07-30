// Created by Claude
import React, { useRef, useEffect } from "react";

interface MessageInputProps {
  onSend: (content: string) => void;
  disabled: boolean;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSend, disabled }) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // added by Claude — auto-resize textarea as user types
  const handleInput = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  };

  const handleSend = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    const value = ta.value.trim();
    if (!value || disabled) return;
    onSend(value);
    ta.value = "";
    ta.style.height = "auto";
  };

  // added by Claude — send on Enter, newline on Shift+Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (!disabled) textareaRef.current?.focus();
  }, [disabled]);

  return (
    <div className="input-bar">
      <div className="input-inner">
        <textarea
          ref={textareaRef}
          className="message-textarea"
          placeholder="Message the AI..."
          rows={1}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          className="send-btn"
          onClick={handleSend}
          disabled={disabled}
          title="Send message"
        >
          {/* added by Claude — paper-plane send icon */}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
      <div className="input-hint">
        {disabled ? "Waiting for response…" : "Enter to send · Shift+Enter for new line"}
      </div>
    </div>
  );
};

export default MessageInput;
