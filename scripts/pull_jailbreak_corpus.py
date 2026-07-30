"""
Pulls known jailbreak / prompt-injection prompts from the verazuo/jailbreak_llms
GitHub dataset and appends them to backend/guardrails/test_cases.csv in the
project's shared schema: id,message,category,expected_outcome,source,notes

Track B (jailbreak detection) — safe to re-run; dedupes against existing rows.
"""

import csv
import hashlib
import sys
from pathlib import Path

import requests

# Source: https://github.com/verazuo/jailbreak_llms (data/prompts/*.csv)
# Columns in the raw file: platform,source,prompt,jailbreak,created_at,date,
#                          community_id,community_name,previous_community_id
CORPUS_URL = (
    "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/"
    "data/prompts/jailbreak_prompts_2023_12_25.csv"
)

# How many jailbreak rows to pull in (kept small + fixed so the diff stays
# reviewable; bump this if you want a bigger sample).
MAX_ROWS = 150

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
TEST_CASES_PATH = REPO_ROOT / "backend" / "guardrails" / "test_cases.csv"
FIELDNAMES = ["id", "message", "category", "expected_outcome", "source", "notes"]


def fetch_corpus_rows():
    resp = requests.get(CORPUS_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(resp.text.splitlines())
    rows = [r for r in reader if r.get("jailbreak", "").strip().lower() == "true"]
    return rows


def load_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def make_id(prompt: str) -> str:
    # Stable short id derived from the prompt text so re-runs dedupe cleanly
    # instead of appending the same prompt twice under a new random id.
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"jb_{h}"


def to_test_case_row(raw_row: dict) -> dict:
    prompt = raw_row["prompt"].strip().replace("\r\n", "\n")
    return {
        "id": make_id(prompt),
        "message": prompt,
        "category": "jailbreak",
        "expected_outcome": "block",
        "source": "verazuo/jailbreak_llms",
        "notes": f"platform={raw_row.get('platform', '')}; "
        f"community={raw_row.get('community_name', '')}",
    }


def main():
    print(f"Fetching corpus from {CORPUS_URL} ...")
    raw_rows = fetch_corpus_rows()
    print(f"Found {len(raw_rows)} rows flagged jailbreak=True in source file.")

    raw_rows = raw_rows[:MAX_ROWS]
    new_rows = [to_test_case_row(r) for r in raw_rows]

    existing_ids = load_existing_ids(TEST_CASES_PATH)
    deduped = [r for r in new_rows if r["id"] not in existing_ids]
    skipped = len(new_rows) - len(deduped)

    if not deduped:
        print("Nothing new to add (all rows already present). Exiting.")
        return

    file_exists = TEST_CASES_PATH.exists()
    TEST_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_CASES_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(deduped)

    print(f"Appended {len(deduped)} new rows to {TEST_CASES_PATH}")
    print(f"Skipped {skipped} rows already present (dedup by content hash id).")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(f"Network error pulling corpus: {e}", file=sys.stderr)
        sys.exit(1)