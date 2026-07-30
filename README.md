# ai-chatbot

A small full-stack chatbot: a Flask + LangChain backend calling OpenAI's `gpt-4o-mini`, with a React/Vite frontend. Built as a base to add content-moderation guardrails around.

## Stack

- **Backend:** Flask, LangChain (`langchain-openai`), file-based chat storage (JSON files under `backend/chats/`)
- **Frontend:** React + TypeScript + Vite

## Setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
python main.py
```

Runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:5173` (the backend's CORS config expects this origin).

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/chats` | List chat summaries |
| `POST` | `/chats` | Create a new chat |
| `GET` | `/chats/:id` | Get a chat with full message history |
| `POST` | `/chats/:id/messages` | Send a message, get the assistant's reply |
| `DELETE` | `/chats/:id` | Delete a chat |

## Status

Guardrails (input/output content moderation, prompt-injection defenses, abuse prevention) are in progress and not yet implemented — the current system prompt is the only constraint on model behavior.
