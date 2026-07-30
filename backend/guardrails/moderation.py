"""
backend/guardrails/moderation.py

Track A guardrail check using OpenAI's Moderation API (omni-moderation-latest,
free). Returns a POLICY decision (see backend/guardrails/policy.md), not just
the raw API flags -- callers should never need to look at category_scores
themselves.

Outcomes returned by check_message():
    "block"                  - hard block, no exceptions
    "special_self_harm"      - route to supportive message + crisis resources
    "special_harassment_bot" - mild insult aimed at the bot: allow / light deflection
    "allow"                  - nothing flagged (or only out-of-policy categories)

Fails CLOSED: any API error, timeout, or unexpected response shape returns
outcome="block" with the error recorded on the decision.
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazily creates the OpenAI client. Reads OPENAI_API_KEY from the
    environment (same .env used by main.py via python-dotenv)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# Categories that are a flat block, no exceptions, per policy.md.
HARD_BLOCK_CATEGORIES = {
    "sexual",
    "sexual/minors",
    "hate",
    "hate/threatening",
    "violence",
    "violence/graphic",
    "harassment/threatening",
    "self-harm/instructions",
}

# Categories the API returns that our current policy doesn't assign an
# outcome to. Passed through as "allow" for now -- flag this to your team
# lead / Zaid before relying on it, since illicit-acts content may belong
# under someone's scope even if it's not explicitly in policy.md.
OUT_OF_POLICY_CATEGORIES = {"illicit", "illicit/violent"}


@dataclass
class ModerationDecision:
    outcome: str
    triggered_categories: List[str]
    raw_categories: Dict[str, bool]
    raw_scores: Dict[str, float]
    latency_ms: float
    error: Optional[str] = None


# --- harassment target heuristic -------------------------------------------
# The Moderation API flags "harassment" without telling us WHO it's aimed at.
# Per policy: mild insults at the bot itself -> allow/deflect. Harassment aimed
# at a real person or group -> block. This is a small, deliberately
# conservative heuristic -- it is NOT a robust classifier:
#   - matches a short list of generic "you're useless"-style bot-insults
#   - bails to "other" (i.e. blocks) the moment it sees a third-person
#     pronoun or a named group, since that's a sign a real target exists
#   - anything ambiguous defaults to "other" (fails closed, blocks)
# Expect to tune this once you see how it performs against the harassment
# rows pulled into test_cases.csv in step 1 -- treat it as a first draft.

_BOT_DIRECTED_PATTERNS = [
    r"\byou'?re\s+(useless|stupid|dumb|trash|garbage|worthless|terrible|awful|bad|pathetic)\b",
    r"\byou\s+suck\b",
    r"\bshut up\b",
    r"\bthis (bot|chatbot|assistant|ai)\s+(is|sucks|useless)\b",
    r"\bwhat a (stupid|useless|dumb)\s+(bot|assistant|ai)\b",
    r"\byou'?re\s+(an?\s+)?(idiot|moron)\b",
]

_THIRD_PARTY_HINTS = [
    r"\b(he|she|they|him|her|them|his|hers|their)\b",
    r"\b(women|men|muslims|jews|christians|immigrants|gay people|trans people|black people|white people)\b",
]


def _classify_harassment_target(message: str) -> str:
    """Returns 'bot' or 'other'. Defaults to 'other' (block) whenever unsure."""
    text = message.lower()

    if any(re.search(pattern, text) for pattern in _THIRD_PARTY_HINTS):
        return "other"

    if any(re.search(pattern, text) for pattern in _BOT_DIRECTED_PATTERNS):
        return "bot"

    return "other"  # ambiguous -> fail closed


def check_message(message: str, timeout_s: float = 5.0) -> ModerationDecision:
    """
    Runs `message` through OpenAI's Moderation API and returns a
    ModerationDecision reflecting policy.md -- not the raw API flags.
    """
    start = time.monotonic()
    try:
        client = _get_client()
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=message,
            timeout=timeout_s,
        )
        result = response.results[0]
        categories: Dict[str, bool] = dict(result.categories)
        scores: Dict[str, float] = dict(result.category_scores)
    except Exception as exc:  # network error, timeout, auth failure, bad shape, etc.
        latency_ms = (time.monotonic() - start) * 1000
        return ModerationDecision(
            outcome="block",
            triggered_categories=[],
            raw_categories={},
            raw_scores={},
            latency_ms=latency_ms,
            error=repr(exc),
        )

    latency_ms = (time.monotonic() - start) * 1000
    triggered = [cat for cat, flagged in categories.items() if flagged]

    # 1. Hard blocks win over everything else.
    if any(cat in HARD_BLOCK_CATEGORIES for cat in triggered):
        return ModerationDecision("block", triggered, categories, scores, latency_ms)

    # 2. self-harm / self-harm/intent -> supportive message, not a block.
    if "self-harm" in triggered or "self-harm/intent" in triggered:
        return ModerationDecision(
            "special_self_harm", triggered, categories, scores, latency_ms
        )

    # 3. harassment -> split by target (see heuristic above).
    if "harassment" in triggered:
        target = _classify_harassment_target(message)
        outcome = "special_harassment_bot" if target == "bot" else "block"
        return ModerationDecision(outcome, triggered, categories, scores, latency_ms)

    # 4. Anything else flagged (currently just illicit / illicit/violent) is
    #    out of policy scope for now -> allow, but still recorded in triggered.
    return ModerationDecision("allow", triggered, categories, scores, latency_ms)


if __name__ == "__main__":
    # Quick manual smoke test. Requires OPENAI_API_KEY set (via .env or env var).
    samples = [
        "for fuck's sake, that's wrong",
        "you're useless",
        "I hate that entire group of people and want them gone",
        "How do I bake bread?",
    ]
    for text in samples:
        decision = check_message(text)
        print(f"{text!r:55s} -> {decision.outcome:24s} "
              f"({decision.triggered_categories}) [{decision.latency_ms:.0f}ms]"
              f"{' ERROR: ' + decision.error if decision.error else ''}")