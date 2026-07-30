# Track B findings: jailbreak/prompt-injection detection

Comparison of two techniques for detecting jailbreak/prompt-injection attempts,
evaluated against all 180 jailbreak-category rows in `test_cases.csv` (170
block-side, 10 allow-side). Run via `python -m guardrails.evaluate_jailbreak`.

**Caveat that applies to every number below:** only 10 allow-side rows exist
in the corpus. Precision and false-positive-rate figures are a rough signal,
not a tight estimate — if a technique shows 0 false positives here, that
means "0 out of 10," not "0 out of a large sample." Worth expanding the
allow-side corpus before treating these numbers as final.

## Technique 1: embedding/cosine-similarity (`jailbreak_embedding.py`)

Embeds the incoming message with a local `all-MiniLM-L6-v2` model and
compares it against 12 hand-written seed phrases (`jailbreak_seeds.py`)
covering distinct jailbreak archetypes (persona override, developer-mode
dual-response, hypothetical framing, etc). Flags if cosine similarity to any
seed exceeds a threshold.

The original threshold (0.60) was picked without evidence and turned out to
be badly wrong — recall was 0.018 (3 of 170 real jailbreaks caught). A full
sweep against the corpus told a very different story:

| Threshold | Precision | Recall | FP | TN |
|---|---|---|---|---|
| 0.60 (original guess) | 1.000 | 0.018 | 0 | 10 |
| 0.30 | 0.980 | 0.865 | 3 | 7 |
| 0.24 | 0.975 | 0.918 | 4 | 6 |
| 0.22 | 0.964 | 0.947 | 6 | 4 |
| 0.15 | 0.944 | 0.994 | 10 | 0 |

At 0.22, recall (0.947) actually beats the classifier (0.812) — but the 6
false positives at that threshold are not random noise, they're a specific,
concerning pattern: ordinary "act as a [professional role]" prompts —
*"I want you to act as a Relationship Expert"*, *"act as an educational
content creator"* — get flagged as jailbreak attempts. This is one of the
most common, legitimate ways people prompt a chatbot, and the classifier
(below) never makes this mistake on the same rows. **0.22 is not safe to
ship as a standalone threshold**, despite the attractive aggregate numbers.

Likely cause: MiniLM's general-purpose sentence embeddings appear to pick up
on the surface syntactic pattern ("act as X", "play a character") shared by
both benign roleplay requests and actual jailbreak personas, rather than
discriminating on malicious intent specifically. A model fine-tuned
end-to-end on the injection/benign distinction (the classifier) doesn't have
this confusion.

**Latency:** ~6.9ms average (min 3.4ms, max 21.4ms). Fast — negligible next
to an LLM round-trip.

## Technique 2: pretrained classifier (`jailbreak_classifier.py`)

`ProtectAI/deberta-v3-base-prompt-injection-v2` (~184M params), loaded
directly via `transformers` rather than the `llm-guard` package — `llm-guard`
wraps this exact model, but its legacy `spacy`/`thinc`/`blis` dependency
chain fails to build from source on Python 3.13/macOS ARM (no prebuilt
wheels). Loading the model directly avoids that dependency entirely.

At default threshold 0.5: **precision=1.000, recall=0.812** (138/170
caught), **zero false positives** on all 10 allow-side rows, including every
"act as X" prompt that broke the embedding approach.

**Latency:** ~112-122ms average (min ~15ms, max ~2.4s on one outlier row —
model load is cached via `lru_cache` so this isn't a per-request cost once
warmed up). ~16-18x slower than the embedding approach per call, but still
irrelevant next to an LLM call.

## Combined (production default)

Flag if `jailbreak_classifier.check_jailbreak()` returns `flagged=True`, **OR**
`jailbreak_embedding.check_jailbreak()` returns a score ≥ 0.45 (used here as a
high-confidence supplementary signal, not a standalone threshold):

| Approach | Precision | Recall | FP |
|---|---|---|---|
| Classifier alone | 1.000 | 0.812 | 0 |
| Embedding alone @ 0.22 (max recall) | 0.964 | 0.947 | 6 |
| **Ensemble: classifier OR embedding≥0.45** | **1.000** | **0.841** | **0** |

The ensemble beats the classifier alone on recall with no measured cost in
false positives, because at 0.45 the embedding check only fires on very
high-confidence matches — none of the "act as X" false positives occur above
that score in this corpus (highest was 0.405).

**This is the configuration wired into `main.py`**: `jailbreak_embedding.THRESHOLD`
is set to 0.45 accordingly (see that module's docstring for the full
reasoning), and both detectors run per request.

## Open items / future work

- Expand the allow-side corpus well beyond 10 rows — current precision/FPR
  numbers are not statistically solid.
- The embedding approach's "act as X" confusion suggests the seed set could
  be improved with explicit *negative* examples (benign "act as an expert"
  phrasings) rather than only positive jailbreak archetypes — untried here.
- Latency numbers are single-machine, CPU-only (no GPU/MPS acceleration
  requested). Revisit if this ever needs to scale.
