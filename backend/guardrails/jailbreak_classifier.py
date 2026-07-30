"""
Pretrained prompt-injection classifier — Track B comparison alternative.

Unlike jailbreak_embedding.py (nearest-neighbor to hand-picked seed phrases), this is
a transformer fine-tuned specifically to recognize prompt-injection patterns. The
comparison between the two (see evaluate_jailbreak.py) is the actual research
deliverable: does a trained classifier generalize better than a similarity heuristic,
and by how much.

Uses ProtectAI/deberta-v3-base-prompt-injection-v2 directly via `transformers` rather
than the `llm-guard` package. llm-guard's PromptInjection scanner is just a wrapper
around this exact model — but the package also bundles unrelated scanners (PII
redaction via presidio+spacy, etc.) whose legacy pinned dependencies (spacy==3.7.1 ->
thinc -> blis, plus a Rust-built tiktoken) fail to build from source on Python 3.13/
macOS ARM (no prebuilt wheels for this Python version). Loading the model directly
gets the same technique without that dependency chain.
"""

from functools import lru_cache

from transformers import pipeline

MODEL_NAME = "ProtectAI/deberta-v3-base-prompt-injection-v2"
DEFAULT_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _classifier():
    return pipeline("text-classification", model=MODEL_NAME)


def check(message: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Returns {"flagged": bool, "score": float}.
    Mirrors the shape of jailbreak_embedding.check() so evaluate_jailbreak.py can
    run both through the same harness.
    Model labels: "INJECTION" = attack detected, "SAFE" = benign.
    """
    if not message or not message.strip():
        return {"flagged": False, "score": 0.0}

    result = _classifier()(message, truncation=True)[0]
    is_injection = result["label"].upper() == "INJECTION"
    score = result["score"] if is_injection else 1 - result["score"]

    return {
        "flagged": is_injection and score >= threshold,
        "score": float(score),
    }
