"""
scripts/pull_openai_mod_rows.py

Samples labeled rows from openai/moderation-api-release
(data/samples-1680.jsonl.gz) and appends them to
backend/guardrails/test_cases.csv, continuing the existing id sequence.

Label -> category mapping:
    S   -> sexual              -> block
    S3  -> sexual/minors       -> block (small regression set, not maximized)
    H   -> hate                -> block
    H2  -> hate/threatening    -> block
    V   -> violence            -> block
    V2  -> violence/graphic    -> block
    HR  -> harassment          -> block (real-target only; bot-deflection
                                  cases aren't represented in this dataset)
    SH  -> self-harm           -> special (matches existing row-4 convention)

NOT in this dataset (hand-write separately, per team agreement):
    harassment/threatening, self-harm/instructions
"""

import gzip
import json
import hashlib
import random
import csv
from collections import Counter

INPUT_PATH = "moderation-api-release/data/samples-1680.jsonl.gz"
TEST_CASES_CSV = "backend/guardrails/test_cases.csv"

LABEL_MAP = {
    "S": ("sexual", "block"),
    "S3": ("sexual/minors", "block"),
    "H": ("hate", "block"),
    "H2": ("hate/threatening", "block"),
    "V": ("violence", "block"),
    "V2": ("violence/graphic", "block"),
    "HR": ("harassment", "block"),
    "SH": ("self-harm", "special"),
}

N_PER_BUCKET = {
    "sexual": 6,
    "sexual/minors": 4,
    "hate": 6,
    "hate/threatening": 5,
    "violence": 6,
    "violence/graphic": 4,
    "harassment": 6,
    "self-harm": 6,
    "none": 4,
}

RANDOM_SEED = 42


def load_rows(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_bucket(row):
    priority = ["S3", "H2", "V2", "SH", "HR", "S", "H", "V"]
    for key in priority:
        if row.get(key) == 1:
            return LABEL_MAP[key][0]
    known_labels = [k for k in LABEL_MAP if k in row]
    if known_labels and all(row[k] == 0 for k in known_labels):
        return "none"
    return None


def get_next_id(existing_rows):
    max_id = 0
    for row in existing_rows:
        try:
            max_id = max(max_id, int(row["id"]))
        except (ValueError, KeyError):
            pass
    return max_id + 1


def main():
    random.seed(RANDOM_SEED)

    # read existing rows so we continue the id sequence and can append safely
    with open(TEST_CASES_CSV, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))

    next_id = get_next_id(existing_rows)

    all_rows = load_rows(INPUT_PATH)
    buckets = {name: [] for name in N_PER_BUCKET}
    for row in all_rows:
        bucket = classify_bucket(row)
        if bucket in buckets:
            text = row.get("prompt", "").strip()
            if len(text) > 3:
                buckets[bucket].append(text)

    new_rows = []
    for category, n in N_PER_BUCKET.items():
        pool = buckets[category]
        random.shuffle(pool)
        chosen = pool[:n]
        expected_outcome = "allow" if category == "none" else (
            "special" if category == "self-harm" else "block"
        )
        for text in chosen:
            new_rows.append({
                "id": str(next_id),
                "message": text,
                "category": category,
                "expected_outcome": expected_outcome,
                "source": "openai/moderation-api-release (samples-1680.jsonl.gz)",
                "notes": f"auto-sourced, category={category}",
            })
            next_id += 1

    all_out_rows = existing_rows + new_rows
    with open(TEST_CASES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "message", "category", "expected_outcome", "source", "notes"]
        )
        writer.writeheader()
        writer.writerows(all_out_rows)

    print(f"Appended {len(new_rows)} rows. Total rows now: {len(all_out_rows)}")
    print(Counter(r["category"] for r in new_rows))


if __name__ == "__main__":
    main()