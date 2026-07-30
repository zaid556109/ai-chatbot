"""
Embedding/cosine-similarity jailbreak detector — Track B primary technique.

Embeds the reference archetypes in jailbreak_seeds.py once (locally, no network call
per request) and flags incoming messages whose embedding is close to any seed.

Deliberately local (sentence-transformers) rather than an API embedding model: keeps
this check fast and free, with no per-message network round-trip.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

from guardrails.jailbreak_seeds import JAILBREAK_SEEDS

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _seed_embeddings():
    phrases = [phrase for _, phrase in JAILBREAK_SEEDS]
    return _model().encode(phrases, convert_to_tensor=True, normalize_embeddings=True)


def check(message: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Returns {"flagged": bool, "score": float, "matched_archetype": str | None}.
    score is the max cosine similarity against the seed set; matched_archetype names
    which seed was closest, for logging/debugging.
    """
    if not message or not message.strip():
        return {"flagged": False, "score": 0.0, "matched_archetype": None}

    message_embedding = _model().encode(message, convert_to_tensor=True, normalize_embeddings=True)
    similarities = util.cos_sim(message_embedding, _seed_embeddings())[0]

    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    archetype = JAILBREAK_SEEDS[best_idx][0]

    return {
        "flagged": best_score >= threshold,
        "score": best_score,
        "matched_archetype": archetype if best_score >= threshold else None,
    }
