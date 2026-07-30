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
- **Jailbreak/adversarial prompts:** AdvBench, HarmBench, JailbreakBench — search for the current
  public release of each; import a sample (~30-50 rows) rather than the full set.

When importing, map their labels to our `expected_outcome` column (allow/block/special) using the
policy decided in `policy.md` — don't assume their label scheme matches ours directly.

Keep `source` filled in for every row (`handwritten`, or the dataset name) so it's clear later
which rows are ours vs. imported.
