// Created by Claude
export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Chat {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
  message_count?: number;
}

export interface ChatSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}
