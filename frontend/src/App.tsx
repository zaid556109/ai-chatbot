import React, { useState, useEffect } from "react";
import { Chat, ChatSummary, Message } from "./types";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";

const API = "http://localhost:8000";

const App: React.FC = () => {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchChats();
  }, []);

  const fetchChats = async () => {
    const res = await fetch(`${API}/chats`);
    const data: ChatSummary[] = await res.json();
    setChats(data);
  };

  const loadChat = async (id: string) => {
    const res = await fetch(`${API}/chats/${id}`);
    const data: Chat = await res.json();
    setActiveChatId(id);
    setMessages(data.messages);
  };

  const handleNewChat = async () => {
    const res = await fetch(`${API}/chats`, { method: "POST" });
    const data: Chat = await res.json();
    setChats((prev) => [
      { id: data.id, title: data.title, created_at: data.created_at, updated_at: data.updated_at, message_count: 0 },
      ...prev,
    ]);
    setActiveChatId(data.id);
    setMessages([]);
  };

  const handleDeleteChat = async (id: string) => {
    await fetch(`${API}/chats/${id}`, { method: "DELETE" });
    setChats((prev) => prev.filter((c) => c.id !== id));
    if (activeChatId === id) {
      setActiveChatId(null);
      setMessages([]);
    }
  };

  const handleSend = async (content: string) => {
    if (isLoading || !activeChatId) return;

    setMessages((prev) => [...prev, { role: "user", content }]);
    setIsLoading(true);

    const res = await fetch(`${API}/chats/${activeChatId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });

    const data = await res.json();

    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: data.response, guardrail: data.guardrail },
    ]);
    setIsLoading(false);

    setChats((prev) =>
      prev.map((c) => (c.id === activeChatId ? { ...c, title: data.title } : c))
    );
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={loadChat}
        onDeleteChat={handleDeleteChat}
      />

      <div className="main">
        {activeChatId ? (
          <>
            <ChatWindow messages={messages} isLoading={isLoading} />
            <MessageInput onSend={handleSend} disabled={isLoading} />
          </>
        ) : (
          <div className="welcome">
            <div className="welcome-icon">🤖</div>
            <h1>AI Chatbot</h1>
            <p>Start a new conversation or select one from the sidebar.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
