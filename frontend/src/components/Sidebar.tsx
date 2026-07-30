// Created by Claude
import React from "react";
import { ChatSummary } from "../types";

interface SidebarProps {
  chats: ChatSummary[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}) => {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNewChat}>
          {/* added by Claude */}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Chat
        </button>
      </div>

      {chats.length > 0 && (
        <div className="sidebar-title">Conversations</div>
      )}

      <div className="chat-list">
        {chats.length === 0 ? (
          <div className="sidebar-empty">No chats yet</div>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${activeChatId === chat.id ? "active" : ""}`}
              onClick={() => onSelectChat(chat.id)}
            >
              <span className="chat-item-title">{chat.title}</span>
              <button
                className="chat-item-delete"
                onClick={(e) => {
                  e.stopPropagation(); // added by Claude
                  onDeleteChat(chat.id);
                }}
                title="Delete chat"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Sidebar;
