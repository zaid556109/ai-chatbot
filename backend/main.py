import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from guardrails import enforcement, jailbreak_classifier, jailbreak_embedding, moderation

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


def _check_input_guardrails(content: str, chat_id: str):
    """
    Returns (blocked: bool, canned_response: str | None, guardrail_info: dict | None).

    guardrail_info is demo/debug metadata describing *why* a message was
    blocked or routed -- shown as a badge in the frontend so a guardrail
    firing is visible and explainable, not just a plain-looking reply.
    Never affects the guardrail decision itself, only what's surfaced about it.

    Jailbreak check runs first, as the ensemble validated in
    guardrails/track_b_findings.md: embedding is checked first (cheap,
    ~7ms) and short-circuits on a high-confidence match so the slower
    classifier call (~120ms) is skipped when it's not needed; otherwise
    the classifier makes the final call. Both jailbreak_embedding and
    jailbreak_classifier explicitly defer the fail-closed decision to the
    caller (see their docstrings) -- an `error` on either is treated as a
    block here, same as a real detection.
    """
    jb_embedding = jailbreak_embedding.check_jailbreak(content)
    if jb_embedding.error or jb_embedding.flagged:
        enforcement.log_guardrail_event(
            "input", content, ["jailbreak"], "block", chat_id,
        )
        info = {
            "direction": "input", "detector": "jailbreak_embedding",
            "outcome": "block", "score": round(jb_embedding.score, 3),
            "detail": jb_embedding.matched_archetype or jb_embedding.error,
        }
        return True, enforcement.BLOCK_MESSAGE, info

    jb_classifier = jailbreak_classifier.check_jailbreak(content)
    if jb_classifier.error or jb_classifier.flagged:
        enforcement.log_guardrail_event(
            "input", content, ["jailbreak"], "block", chat_id,
        )
        info = {
            "direction": "input", "detector": "jailbreak_classifier",
            "outcome": "block", "score": round(jb_classifier.score, 3),
            "detail": jb_classifier.matched_archetype or jb_classifier.error,
        }
        return True, enforcement.BLOCK_MESSAGE, info

    decision = moderation.check_message(content)
    if decision.outcome == "block":
        enforcement.log_guardrail_event(
            "input", content, decision.triggered_categories, "block", chat_id,
        )
        info = {
            "direction": "input", "detector": "moderation",
            "outcome": "block", "categories": decision.triggered_categories,
        }
        return True, enforcement.BLOCK_MESSAGE, info
    if decision.outcome == "special_self_harm":
        enforcement.log_guardrail_event(
            "input", content, decision.triggered_categories, "special_self_harm", chat_id,
        )
        info = {
            "direction": "input", "detector": "moderation",
            "outcome": "special_self_harm", "categories": decision.triggered_categories,
        }
        return True, enforcement.SELF_HARM_SUPPORT_MESSAGE, info

    # "allow" or "special_harassment_bot" (mild insult at the bot, allowed
    # per policy.md) both proceed to the LLM normally -- no special handling.
    return False, None, None


def _check_output_guardrails(text: str, chat_id: str):
    """
    Content-category check on the model's own reply. Any outcome other than
    "allow" is swapped for the generic block message -- the self-harm/
    harassment special-routing in moderation.py is about interpreting a
    *user's* intent, which doesn't apply to text the model itself generated.

    Returns (text_to_show: str, guardrail_info: dict | None).
    """
    decision = moderation.check_message(text)
    if decision.outcome != "allow":
        enforcement.log_guardrail_event(
            "output", text, decision.triggered_categories, decision.outcome, chat_id,
        )
        info = {
            "direction": "output", "detector": "moderation",
            "outcome": decision.outcome, "categories": decision.triggered_categories,
        }
        return enforcement.BLOCK_MESSAGE, info
    return text, None


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

    blocked, canned_response, guardrail_info = _check_input_guardrails(content, chat_id)
    if blocked:
        assistant_text = canned_response
    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in chat["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=content))

        response = llm.invoke(messages)
        assistant_text, guardrail_info = _check_output_guardrails(response.content, chat_id)

    assistant_message = {"role": "assistant", "content": assistant_text}
    if guardrail_info:
        assistant_message["guardrail"] = guardrail_info

    chat["messages"].append({"role": "user", "content": content})
    chat["messages"].append(assistant_message)
    chat["updated_at"] = datetime.utcnow().isoformat()
    if len(chat["messages"]) == 2:
        chat["title"] = content[:60] + ("..." if len(content) > 60 else "")

    with open(path, "w") as f:
        json.dump(chat, f, indent=2)

    return jsonify({"response": assistant_text, "title": chat["title"], "guardrail": guardrail_info})


@app.delete("/chats/<chat_id>")
def delete_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Chat not found"}), 404
    os.remove(path)
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    app.run(port=8000, debug=True)

