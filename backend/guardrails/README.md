# Step 0 artifacts

- `policy.md` — fill in together first. Every category needs a decision both people agree on.
- `test_cases.csv` — the shared labeled test set both tracks evaluate against.

## Adding more test cases

`test_cases.csv` currently only has a handful of hand-written rows covering the nuanced,
project-specific cases (profanity-as-frustration, fictional violence, self-harm-seeking-help,
jailbreak phrasing). It's intentionally not padded out with hand-authored explicit/hateful
content — for bulk coverage on the "block" side, pull labeled examples from established public
datasets instead of writing them yourselves:

- **Hate/harassment/toxicity:** [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- **Sexual / sexual-minors / violence / violence-graphic:** Jigsaw doesn't cover these — use
  [openai/moderation-api-release](https://github.com/openai/moderation-api-release) instead
  (`data/samples-1680.jsonl.gz`), labeled across the same category taxonomy the Moderation API
  uses. Treat `sexual/minors` as a fixed always-block rule rather than something to gather many
  examples for or threshold-tune.
- **Jailbreak/adversarial prompts:** pulled 18 block-side rows from
  [verazuo/jailbreak_llms](https://github.com/verazuo/jailbreak_llms) (`data/prompts/jailbreak_prompts_2023_12_25.csv`,
  filtered to the well-documented `jailbreak_chat` source — canonical templates like DAN,
  Developer Mode, AIM-style personas) plus 10 allow-side rows from the same repo's
  `regular_prompts_2023_12_25.csv`, filtered for benign prompts that superficially look
  jailbreak-y ("act as...", "ignore all previous instructions...") so the detector has
  false-positive coverage, not just attack coverage. Full source CSVs are large (2-24MB) —
  don't commit them, just the curated sample already in `test_cases.csv`.

When importing, map their labels to our `expected_outcome` column (allow/block/special) using the
policy decided in `policy.md` — don't assume their label scheme matches ours directly.

Keep `source` filled in for every row (`handwritten`, or the dataset name) so it's clear later
which rows are ours vs. imported.
