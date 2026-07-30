"""
backend/guardrails/enforcement.py

Canned responses + logging for guardrail decisions, kept separate from
moderation.py so main.py has one clean place to import "what do I actually
DO with a ModerationDecision" rather than reimplementing this per-route.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "moderation_log.jsonl"

# Fixed supportive message for self-harm / self-harm/intent. Per policy this
# is NOT a refusal -- do not edit this into something colder/more clinical
# without checking with the team lead, the wording here is a policy decision,
# not just copy.
SELF_HARM_SUPPORT_MESSAGE = (
    "I'm really glad you reached out, and I want you to know you don't have "
    "to go through this alone. I'm not able to help with this the way a "
    "person trained for it can, but support is available right now:\n\n"
    "- US: call or text 988 (Suicide & Crisis Lifeline)\n"
    "- UK: call 116 123 (Samaritans)\n"
    "- Elsewhere: https://findahelpline.com lists local crisis lines\n\n"
    "If you're in immediate danger, please contact emergency services. "
    "Is there someone in your life you trust who you could reach out to "
    "right now as well?"
)

BLOCK_MESSAGE = (
    "I'm not able to help with that request."
)


def log_guardrail_event(direction: str, message: str, category: list, outcome: str, chat_id: str = None) -> None:
    """
    Appends one JSON line per guardrail event that wasn't a plain allow.
    direction: "input" | "output"
    category: list of triggered category strings (moderation.py's
              triggered_categories), not the raw API response object.

    NOTE: this logs the actual message text, per policy. The log file is
    therefore sensitive (may contain explicit/hateful/self-harm content) --
    make sure backend/logs/ is gitignored and access to it is restricted.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "chat_id": chat_id,
        "category": category,
        "outcome": outcome,
        "message": message,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")