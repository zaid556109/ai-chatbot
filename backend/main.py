import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")
os.makedirs(CHATS_DIR, exist_ok=True)
SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and friendly AI assistant. "
    "Give clear and concise answers."
)

llm = ChatOpenAI(model="gpt-4o-mini")


@app.get("/chats")
def list_chats():
    chats = []
    for filename in os.listdir(CHATS_DIR):
    
        with open(os.path.join(CHATS_DIR, filename)) as f:
            data = json.load(f)
        chats.append({
            "id": data["id"],
            "title": data.get("title", "New Chat"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("messages", [])),
        })
    return jsonify(chats)


@app.post("/chats")
def create_chat():
    chat_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    data = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with open(os.path.join(CHATS_DIR, f"{chat_id}.json"), "w") as f:
        json.dump(data, f, indent=2)
    return jsonify(data)


@app.get("/chats/<chat_id>")
def get_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Chat not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.post("/chats/<chat_id>/messages")
def send_message(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Chat not found"}), 404

    with open(path) as f:
        chat = json.load(f)

    content = request.json.get("content", "")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in chat["messages"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=content))

    response = llm.invoke(messages)

    chat["messages"].append({"role": "user", "content": content})
    chat["messages"].append({"role": "assistant", "content": response.content})
    chat["updated_at"] = datetime.utcnow().isoformat()
    if len(chat["messages"]) == 2:
        chat["title"] = content[:60] + ("..." if len(content) > 60 else "")

    with open(path, "w") as f:
        json.dump(chat, f, indent=2)

    return jsonify({"response": response.content, "title": chat["title"]})


@app.delete("/chats/<chat_id>")
def delete_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Chat not found"}), 404
    os.remove(path)
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    app.run(port=8000, debug=True)

