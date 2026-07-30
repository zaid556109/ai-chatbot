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

## Guardrails: mode of action

This project's goal isn't just to add a filter — it's to research and evaluate guardrail techniques and document why we picked what we picked. The approach is layered rather than one single check, following the standard defense pattern for LLM apps (input validation → managed moderation → output filtering):

1. **Content-category moderation** (sexual, hate, harassment, violence, self-harm): use OpenAI's Moderation API (`omni-moderation-latest`) directly, on both the incoming user message and the model's reply, rather than a homemade keyword/profanity filter. Reasoning: a bag-of-words profanity classifier can't distinguish "for fuck's sake, that's wrong" (allowed — frustration, not explicit content) from actually explicit content, since it only detects word presence, not intent or category. Category-based classification solves the problem we actually have. This endpoint is free and doesn't need to be built or trained by us.

2. **Prompt-injection / jailbreak detection**: not covered by the Moderation API at all, so this is genuinely researched and compared head-to-head:
   - **Embedding/cosine-similarity** against a maintained set of known jailbreak phrasings (cheap, no training, but recall is bounded by how good the seed set is).
   - **A pretrained classifier** (e.g. LLM Guard's `PromptInjection` scanner) fine-tuned specifically to recognize injection patterns.
   - Both get evaluated against the same labeled test set (precision, recall, false-positive rate, latency, maintenance cost) and the results get written up, not just the winner picked by feel.

**Note on using third-party APIs/models at all:** the API/classifier only produces a score. The actual engineering work — policy design (what blocks vs. what's allowed), curating the labeled test set, the evaluation harness, integration, failure handling, and the automated test suite — is ours regardless of which technique sits underneath. Both guardrail layers below apply that same standard: no layer gets to be "just call one API and stop" — each requires a primary technique *and* a comparison alternative, evaluated side by side with data.

3. **Test methodology**: a labeled test set of message → expected outcome (allow/block) is written *before* any filter code, covering explicit/harmful content, benign profanity, and known jailbreak prompts. Unit tests hit the filter functions directly and deterministically (no LLM call). Adversarial/red-team tests (`promptfoo redteam`) hit the real running endpoint to confirm the guardrails work end-to-end, not just in isolation.

4. **Failure mode**: guardrail checks fail closed (block) rather than fail open, so a Moderation API outage doesn't silently disable moderation.

## Work split (2 people)

Before splitting: confirm with the team lead whether calling third-party moderation/classification APIs is in scope, or whether classification needs to be built from scratch — this changes the size of both tracks equally, so resolve it before dividing work, not after.

**Step 0 — do together, ~1 session:** agree on the category policy (which categories block outright, which are lenient, where profanity-as-frustration falls) and write the shared labeled test set: allow/block examples per content category, plus a jailbreak/prompt-injection corpus. Both tracks depend on this and it sets the ground rules both people are held to, so it isn't split between you.

Both tracks below follow the same shape and are comparable in size: pick a primary technique, pick a comparison alternative, implement both, evaluate both against the shared test set, write up the comparison with numbers, wire the chosen approach into the app, write the automated tests. Neither track is "call an API and stop."

**Track A — Content-category moderation**
- Primary: OpenAI Moderation API. Comparison alternative: a second, non-OpenAI option (e.g. Google's Perspective API, or a self-hosted classifier like Detoxify) — the point is independent signal, not grading OpenAI's model with OpenAI's own classifier.
- Run both against the shared test set; measure precision, recall, false-positive rate, latency, cost, vendor independence
- Decide per-category thresholds/blocking behavior and fail-closed handling for API errors/timeouts
- Wire the chosen approach into `backend/main.py` at input (before `llm.invoke`) and output (on `response.content`), with a canned refusal + logging of blocked attempts
- `pytest` unit tests for this layer against the shared test set

**Track B — Prompt-injection / jailbreak detection**
- Primary: embedding/cosine-similarity against a maintained set of known jailbreak phrasings. Comparison alternative: a pretrained classifier (e.g. LLM Guard's `PromptInjection` scanner)
- Run both against the shared jailbreak corpus; measure precision, recall, false-positive rate, latency, maintenance cost
- Wire the chosen approach (or both, if the comparison shows they catch different things) into `backend/main.py`
- `pytest` unit tests for this layer, plus `promptfoo redteam` against the live endpoint once both layers are wired in

**Merge step — do together:** integrate both layers into `main.py`, run the full test suite + promptfoo against the combined system, update this README's Status section with what shipped and what each comparison concluded.
