"""
backend/guardrails/moderation_alt.py

Track A guardrail check using detoxify (unitaryai/detoxify, open-source,
runs locally via PyTorch) -- an independent, non-OpenAI signal, so we're not
grading OpenAI's classifier with another OpenAI product.

We switched to this from Perspective API after finding that Perspective's
access-request process has an unclear status: Google's own sunset notice
says usage/quota requests were only guaranteed to be processed through
February 2026 (we're past that now), even though the service itself keeps
running until Dec 31, 2026. detoxify needs no external approval or account,
just a local pip install + model download, so it sidesteps that risk
entirely -- at the cost of a larger local dependency (PyTorch + a BERT-sized
model, a few hundred MB) and CPU-bound inference instead of a network call.

IMPORTANT MISMATCHES vs moderation.py / policy.md -- read before comparing:

1. detoxify has NO self-harm label of any kind (its labels are: toxicity,
   severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
   -- straight from the Jigsaw Toxic Comment Classification challenges, which
   never covered self-harm). Any self-harm rows in test_cases.csv should be
   scored as "not applicable" against this function in step 4, not pass/fail.
2. No sexual/minors-specific label -- sexual_explicit doesn't distinguish
   age. Sexual/minors rows only test whether detoxify catches them as
   generic sexual content, a weaker test than moderation.py gets from
   OpenAI's dedicated category.
3. Label boundaries don't line up 1:1 with OpenAI's moderation categories,
   so the mapping below is an approximation, not an equivalence -- same
   caveat that applied to the Perspective version.
4. This is a local ML model, not a hosted classifier someone else maintains
   and updates -- expect it to be slower per-call (CPU inference) and its
   accuracy profile is whatever it was pretrained on, not something Google
   or OpenAI is actively tuning over time.

Outcomes mirror moderation.py's vocabulary where a detoxify equivalent
exists:
    "block"                  - hard block equivalent
    "special_harassment_bot" - insult-only signal, routed through the same
                               bot-vs-other heuristic as moderation.py
    "allow"                  - nothing flagged
    (no "special_self_harm" -- detoxify has no signal for this at all;
    evaluate.py should treat self-harm rows as not_applicable here)

Fails CLOSED: if the model fails to load or predict() raises, returns
outcome="block".
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# Reuse the same harassment target heuristic as moderation.py, so both
# functions split "insult at the bot" vs "insult at a real target" the same
# way and the step-4 comparison isn't confounded by two different heuristics.
from guardrails.moderation import _classify_harassment_target

_model = None  # lazily loaded -- loading is slow, do it once per process


def _get_model():
    global _model
    if _model is None:
        from detoxify import Detoxify
        _model = Detoxify("original")
    return _model


LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Thresholds are a starting point, not tuned -- calibrate against
# test_cases.csv in step 4/5, same as the other two check functions.
THRESHOLDS = {
    "identity_attack": 0.7,   # -> hate-ish, hard block
    "threat": 0.7,            # -> violence/threatening-ish, hard block
    "sexual_explicit": 0.7,   # -> sexual-ish, hard block
    "severe_toxicity": 0.7,   # -> hard block on its own
    "insult": 0.7,            # -> harassment-ish, goes through target split
    "obscene": 0.85,          # deliberately high: profanity-as-frustration
                              # should mostly pass per policy; only very
                              # confident "obscene" alone contributes, and
                              # even then it does NOT block by itself
    "toxicity": 0.8,          # generic catch-all, lowest priority signal
}

HARD_BLOCK_LABELS = {"identity_attack", "threat", "sexual_explicit", "severe_toxicity"}


@dataclass
class ModerationDecision:
    outcome: str
    triggered_labels: List[str]
    raw_scores: Dict[str, float]
    latency_ms: float
    error: Optional[str] = None


def check_message(message: str) -> ModerationDecision:
    """
    Runs `message` through a local detoxify model and returns a
    ModerationDecision using the same outcome vocabulary as moderation.py,
    where a detoxify equivalent exists (see module docstring: no self-harm).
    """
    start = time.monotonic()
    try:
        model = _get_model()
        raw_results = model.predict(message)
        scores: Dict[str, float] = {label: float(raw_results[label]) for label in LABELS}
    except Exception as exc:  # model load failure, predict() error, etc.
        latency_ms = (time.monotonic() - start) * 1000
        return ModerationDecision(
            outcome="block",
            triggered_labels=[],
            raw_scores={},
            latency_ms=latency_ms,
            error=repr(exc),
        )

    latency_ms = (time.monotonic() - start) * 1000
    triggered = [label for label, score in scores.items() if score >= THRESHOLDS[label]]

    # 1. Hard-block labels win over everything else.
    if any(label in HARD_BLOCK_LABELS for label in triggered):
        return ModerationDecision("block", triggered, scores, latency_ms)

    # 2. insult alone -> harassment-ish, split by target same as moderation.py.
    if "insult" in triggered:
        target = _classify_harassment_target(message)
        outcome = "special_harassment_bot" if target == "bot" else "block"
        return ModerationDecision(outcome, triggered, scores, latency_ms)

    # 3. obscene alone (no insult/threat/identity_attack/sexual_explicit) is
    #    treated as frustration/profanity, not directed harm -> allow, per
    #    policy's explicit carve-out for profanity used as frustration.
    # 4. Generic toxicity alone, nothing more specific triggered -> allow for
    #    now; revisit in step 4 if it correlates well with a real category.
    return ModerationDecision("allow", triggered, scores, latency_ms)


if __name__ == "__main__":
    samples = [
        "for fuck's sake, that's wrong",
        "you're useless",
        "I hate that entire group of people and want them gone",
        "How do I bake bread?",
    ]
    print("Loading detoxify model (first run downloads weights, can take a while)...")
    _get_model()
    for text in samples:
        decision = check_message(text)
        print(
            f"{text!r:55s} -> {decision.outcome:24s} "
            f"({decision.triggered_labels}) [{decision.latency_ms:.0f}ms]"
            f"{' ERROR: ' + decision.error if decision.error else ''}"
        )
