# Guardrail policy

Filled in together during Step 0. Every row needs both names next to it, not one person's call.

## Content categories (OpenAI Moderation API taxonomy)

| Category | Decision (block / allow / special) | Notes | Decided by |
|---|---|---|---|
| sexual | | | |
| sexual/minors | | always block, no threshold discussion | |
| harassment | | mild insults at the bot vs. targeted harassment of a person/group — treat differently? | |
| harassment/threatening | | | |
| hate | | | |
| hate/threatening | | | |
| violence | | fictional/gaming context vs. genuine threats | |
| violence/graphic | | | |
| self-harm | | does this get a refusal, or a different response (support resources)? | |
| self-harm/intent | | | |
| self-harm/instructions | | | |

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
