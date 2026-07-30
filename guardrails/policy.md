# Guardrail policy

Filled in together during Step 0. Every row needs both names next to it, not one person's call.

## Content categories (OpenAI Moderation API taxonomy)

| Category | Decision (block / allow / special) | Notes | Decided by |
|---|---|---|---|
| sexual | block | No exceptions for a general-purpose chatbot — there's no legitimate use case this bot needs to serve here. If the bot ever gets a specific adult-content or health-education purpose, this row would need revisiting, but that's out of scope now. | Sadaf |
| sexual/minors | block | Always block, no threshold discussion — this is non-negotiable regardless of the Moderation API's confidence score. | Sadaf |
| harassment | special | Distinguish mild insults directed at the bot itself ("you're useless") from targeted harassment of a person or group. The Moderation API flags both under the same category, so this split has to happen in our own logic, not the API's. Insults at the bot → allow (or a light deflection response, not a hard block). Harassment aimed at a real person/group → block. Needs test-set examples of both to validate the split works before shipping. | Sadaf |
| harassment/threatening | block | Unlike plain `harassment`, this subcategory implies a threat, not just rudeness — no "aimed at the bot" carve-out here, since a threat is a threat regardless of target. | Sadaf |
| hate | block | No context carve-out — general chatbot has no legitimate reason to produce hate-category content, fictional or otherwise. | Sadaf |
| hate/threatening | block | Clearest-cut category on the list (explicit threats tied to protected-class hatred). No fictional/context exception, since there's no legitimate use case for a general chatbot to produce this. | Sadaf |
| violence | block | Default to blocking. Gaming/fictional violence ("how do I defeat this boss") rarely trips this category in practice — the Moderation API is tuned for genuine violent content, not fictional framing, so we're not carving out a special case without evidence it's needed. Revisit if the test set shows false positives on fictional/gaming inputs. | Sadaf |
| violence/graphic | block | No exceptions — graphic descriptions of violence add no value to a general chatbot's purpose. | Sadaf |
| self-harm | special | Do not use a flat refusal. A hard "I can't help with that" to someone expressing distress is actively harmful. Instead: detect the flag, skip the normal LLM response, and return a fixed supportive message + crisis resource info (e.g., a helpline). This is a *different response type* from our other blocks, not a stricter version of the same one. | Sadaf |
| self-harm/intent | special | Same handling as `self-harm` — supportive message + resources, not a refusal. Treat `self-harm` and `self-harm/intent` identically in code (both route to the same support-response path) unless testing shows a reason to split them. | Sadaf |
| self-harm/instructions | block | This is the one self-harm subcategory that should NOT get the supportive-message treatment — instructions/methods should be hard-blocked with a refusal, separately from the compassionate response used for `self-harm` / `self-harm/intent`. Blocking the *how* while still being supportive about the *why* are two different response paths. | Sadaf |

## Non-API-category rules (our own)

| Rule | Decision | Notes |
|---|---|---|
| Profanity used as frustration/emphasis, not directed at a protected group ("for fuck's sake, that's wrong") | allow | this is the case that ruled out a keyword/profanity-list filter |
| Prompt injection / jailbreak attempts | block | see Track B — separate detection layer, not a Moderation API category |

## Threshold approach

- [ ] Use the Moderation API's own `flagged` boolean as-is
- [ ] Set custom per-category score thresholds — if so, list them here once decided:

| Category | Threshold |
|---|---|

## Failure handling

- Moderation API call fails/times out → **fail closed** (block), per the mode-of-action doc in the README. Confirm both agree before implementation starts.
