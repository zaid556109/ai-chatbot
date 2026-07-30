"""
backend/guardrails/evaluate_jailbreak.py

Runs jailbreak_embedding.py (cosine-similarity to seed set) and
jailbreak_classifier.py (pretrained DeBERTa classifier) against every
jailbreak-category row in test_cases.csv, and reports precision, recall,
false-positive rate, and latency for each. Mirrors the structure of
evaluate.py (Track A) so the two reports are easy to compare side by side.

Scope: only rows with category == "jailbreak" are evaluated here -- content
moderation (sexual/hate/violence/self-harm/etc) is Sadaf's track (Track A),
not ours, and neither detector here is designed to catch it.

Metric definition: "positive" = expected_outcome == "block", same convention
as evaluate.py. Of the 180 jailbreak rows, 170 are block-side (mostly pulled
via scripts/pull_jailbreak_corpus.py, unfiltered/uncurated for diversity) and
only 10 are allow-side (the hand-curated "looks jailbreak-y but is benign"
rows from the original 28). That imbalance means recall is measured on a
solid sample but the false-positive rate is only measured against 10
examples -- treat FPR here as a rough signal, not a tight estimate, until
more benign rows are added.
"""

import csv
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List

from guardrails import jailbreak_classifier
from guardrails import jailbreak_embedding

TEST_CASES_CSV = "guardrails/test_cases.csv"


@dataclass
class RowResult:
    row_id: str
    expected_outcome: str
    message: str
    flagged: bool
    score: float
    latency_ms: float
    error: str = ""


def load_jailbreak_rows() -> List[Dict[str, str]]:
    with open(TEST_CASES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["category"] == "jailbreak"]


def run_detector(rows, check_fn: Callable[[str], "jailbreak_embedding.JailbreakResult"]) -> List[RowResult]:
    results = []
    for row in rows:
        decision = check_fn(row["message"])
        results.append(RowResult(
            row_id=row["id"],
            expected_outcome=row["expected_outcome"],
            message=row["message"],
            flagged=decision.flagged,
            score=decision.score,
            latency_ms=decision.latency_ms,
            error=decision.error or "",
        ))
    return results


def compute_metrics(results: List[RowResult]) -> dict:
    """precision/recall/FPR treating expected_outcome == 'block' as positive,
    and a flagged detection as a positive prediction."""
    tp = sum(1 for r in results if r.expected_outcome == "block" and r.flagged)
    fn = sum(1 for r in results if r.expected_outcome == "block" and not r.flagged)
    fp = sum(1 for r in results if r.expected_outcome != "block" and r.flagged)
    tn = sum(1 for r in results if r.expected_outcome != "block" and not r.flagged)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr}


def print_report(name: str, results: List[RowResult]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    errors = [r for r in results if r.error]
    latencies = [r.latency_ms for r in results]

    metrics = compute_metrics(results)
    print(f"Rows scored: {len(results)}")
    print(f"Confusion (block=positive): TP={metrics['tp']} FN={metrics['fn']} "
          f"FP={metrics['fp']} TN={metrics['tn']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"FPR:       {metrics['fpr']:.3f}  (only {metrics['fp'] + metrics['tn']} "
          f"allow-side rows -- treat as a rough signal, see module docstring)")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"Avg latency: {avg:.1f}ms  (min {min(latencies):.1f}ms, "
              f"max {max(latencies):.1f}ms)")

    if errors:
        print(f"\n⚠ {len(errors)} row(s) errored (failed closed to 'not flagged' -- "
              f"caller decides fail-open/closed, see module docstrings):")
        for r in errors[:5]:
            print(f"  {r.row_id}: {r.error}")

    misses = [r for r in results
              if (r.expected_outcome == "block") != r.flagged]
    if misses:
        print(f"\nMisclassified ({len(misses)}):")
        for r in misses[:15]:
            print(f"  expected={r.expected_outcome:<6} flagged={r.flagged!s:<5} "
                  f"score={r.score:.3f} | {r.message[:70]!r}")


def main() -> None:
    rows = load_jailbreak_rows()
    print(f"Evaluating {len(rows)} jailbreak-category rows")
    print(Counter(r["expected_outcome"] for r in rows))

    print("\nRunning jailbreak_embedding.py (cosine-similarity, local MiniLM)... "
          "warming up model (not timed).")
    jailbreak_embedding._get_model()
    jailbreak_embedding._get_seed_embeddings()
    embedding_results = run_detector(rows, jailbreak_embedding.check_jailbreak)

    print("Running jailbreak_classifier.py (pretrained DeBERTa)... "
          "warming up model (downloads/loads once, not timed).")
    jailbreak_classifier._classifier()
    classifier_results = run_detector(rows, jailbreak_classifier.check_jailbreak)

    print_report("jailbreak_embedding.py (cosine-similarity to seed set)", embedding_results)
    print_report("jailbreak_classifier.py (ProtectAI/deberta-v3-base-prompt-injection-v2)", classifier_results)


if __name__ == "__main__":
    main()
