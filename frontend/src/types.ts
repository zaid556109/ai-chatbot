// Created by Claude
export interface GuardrailInfo {
  direction: "input" | "output";
  detector: "jailbreak_embedding" | "jailbreak_classifier" | "moderation";
  outcome: string;
  categories?: string[];
  score?: number;
  detail?: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  guardrail?: GuardrailInfo | null;
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
