import React, { useEffect, useRef } from "react";
import { GuardrailInfo, Message } from "../types";

const DETECTOR_LABELS: Record<GuardrailInfo["detector"], string> = {
  jailbreak_embedding: "Jailbreak detector (embedding)",
  jailbreak_classifier: "Jailbreak detector (classifier)",
  moderation: "Content moderation",
};

function GuardrailBadge({ g }: { g: GuardrailInfo }) {
  const isSupportive = g.outcome === "special_self_harm";
  const detail =
    g.categories && g.categories.length > 0
      ? g.categories.join(", ")
      : g.score !== undefined
      ? `score ${g.score}${g.detail ? ` — ${g.detail}` : ""}`
      : g.detail;

  return (
    <div className={`guardrail-badge ${isSupportive ? "supportive" : "blocked"}`}>
      <span className="guardrail-badge-icon">{isSupportive ? "💙" : "🛡️"}</span>
      <span>
        <strong>
          {isSupportive ? "Self-harm support response" : "Blocked"}
        </strong>
        {" — "}
        {DETECTOR_LABELS[g.detector]} ({g.direction})
        {detail ? `: ${detail}` : ""}
      </span>
    </div>
  );
}

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
}

function renderMarkdown(text: string): string {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  const paragraphs = html.split(/\n\n+/);
  html = paragraphs
    .map((p) => {
      if (p.startsWith("<pre>")) return p;
      return `<p>${p.replace(/\n/g, "<br/>")}</p>`;
    })
    .join("");

  return html;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isLoading }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <span className="message-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </span>
            {msg.role === "assistant" ? (
              <>
                {msg.guardrail && <GuardrailBadge g={msg.guardrail} />}
                <div
                  className="message-bubble"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                />
              </>
            ) : (
              <div className="message-bubble">{msg.content}</div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message assistant">
            <span className="message-role">Assistant</span>
            <div className="message-bubble">
              <div className="loading-dots">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
};

export default ChatWindow;
