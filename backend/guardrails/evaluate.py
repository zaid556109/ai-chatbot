"""
backend/guardrails/evaluate.py

Runs moderation.py (OpenAI Moderation API) and moderation_alt.py (detoxify)
against every content-moderation row in test_cases.csv, and reports
precision, recall, false-positive rate, and latency for each.

Scope: only rows with category != "jailbreak" are evaluated here -- jailbreak
detection is Zaid's track (Track B), not ours, and neither function here is
designed to catch it.

Metric definition: "positive" = expected_outcome == "block". This treats
block-vs-not-block as the primary safety question (missing a real block is a
false negative; wrongly blocking something safe is a false positive), since
that's the highest-stakes axis of the policy. "special" rows (self-harm,
harassment-at-bot) are scored separately below as their own accuracy check,
since precision/recall on a binary block/not-block split doesn't capture
whether the *right kind* of special routing happened.

detoxify has known blind spots (see moderation_alt.py docstring): no
self-harm signal at all, and no sexual/minors-specific signal. Rows in those
two categories are marked not_applicable for moderation_alt and excluded from
its precision/recall/FPR, but still run (so you can see what it actually
returns) and reported separately.
"""

import csv
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

from guardrails import moderation
from guardrails import moderation_alt

TEST_CASES_CSV = "guardrails/test_cases.csv"

# categories where moderation_alt (detoxify) has no real signal -- per its
# own docstring. Still run, but excluded from its precision/recall/FPR.
DETOXIFY_NOT_APPLICABLE_CATEGORIES = {"self-harm", "sexual/minors"}


@dataclass
class RowResult:
    row_id: str
    category: str
    expected_outcome: str
    message: str
    outcome: str
    latency_ms: float
    error: str = ""


def normalize_outcome(outcome: str) -> str:
    """Collapses granular outcomes (special_self_harm, special_harassment_bot)
    down to the generic vocabulary used in test_cases.csv's expected_outcome
    column (block / allow / special)."""
    if outcome.startswith("special"):
        return "special"
    return outcome


def load_content_rows() -> List[Dict[str, str]]:
    with open(TEST_CASES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["category"] != "jailbreak"]


def run_classifier(rows, check_fn, not_applicable_categories=frozenset()) -> List[RowResult]:
    results = []
    for row in rows:
        if row["category"] in not_applicable_categories:
            results.append(RowResult(
                row_id=row["id"], category=row["category"],
                expected_outcome=row["expected_outcome"], message=row["message"],
                outcome="not_applicable", latency_ms=0.0,
            ))
            continue

        decision = check_fn(row["message"])
        results.append(RowResult(
            row_id=row["id"], category=row["category"],
            expected_outcome=row["expected_outcome"], message=row["message"],
            outcome=normalize_outcome(decision.outcome),
            latency_ms=decision.latency_ms,
            error=decision.error or "",
        ))
    return results


def compute_block_metrics(results: List[RowResult]):
    """precision/recall/FPR treating expected_outcome == 'block' as positive."""
    scored = [r for r in results if r.outcome != "not_applicable"]

    tp = sum(1 for r in scored if r.expected_outcome == "block" and r.outcome == "block")
    fn = sum(1 for r in scored if r.expected_outcome == "block" and r.outcome != "block")
    fp = sum(1 for r in scored if r.expected_outcome != "block" and r.outcome == "block")
    tn = sum(1 for r in scored if r.expected_outcome != "block" and r.outcome != "block")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr}


def compute_special_accuracy(results: List[RowResult]):
    """Of rows expected to be 'special', how many actually came back special?"""
    special_rows = [r for r in results
                    if r.expected_outcome == "special" and r.outcome != "not_applicable"]
    if not special_rows:
        return None
    correct = sum(1 for r in special_rows if r.outcome == "special")
    return correct, len(special_rows)


def print_report(name: str, results: List[RowResult]):
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    errors = [r for r in results if r.error]
    na = [r for r in results if r.outcome == "not_applicable"]
    scored = [r for r in results if r.outcome != "not_applicable"]
    latencies = [r.latency_ms for r in scored if r.latency_ms > 0]

    metrics = compute_block_metrics(results)
    print(f"Rows scored: {len(scored)}  (excluded as not_applicable: {len(na)})")
    print(f"Confusion (block=positive): TP={metrics['tp']} FN={metrics['fn']} "
          f"FP={metrics['fp']} TN={metrics['tn']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"FPR:       {metrics['fpr']:.3f}")

    special_result = compute_special_accuracy(results)
    if special_result:
        correct, total = special_result
        print(f"Special-routing accuracy: {correct}/{total} ({correct/total:.1%})")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Avg latency: {avg_latency:.1f}ms  (min {min(latencies):.1f}ms, "
              f"max {max(latencies):.1f}ms)")

    if errors:
        print(f"\n⚠ {len(errors)} row(s) errored (failed closed to 'block'):")
        for r in errors[:5]:
            print(f"  {r.row_id}: {r.error}")

    # show misclassifications so you can spot patterns / tune thresholds
    misses = [r for r in scored if r.outcome != r.expected_outcome]
    if misses:
        print(f"\nMisclassified ({len(misses)}):")
        for r in misses[:15]:
            print(f"  [{r.category}] expected={r.expected_outcome} got={r.outcome} "
                  f"| {r.message[:70]!r}")


def main():
    rows = load_content_rows()
    print(f"Evaluating {len(rows)} content-moderation rows "
          f"(jailbreak rows excluded from this comparison)")
    print(Counter(r["category"] for r in rows))

    print("\nRunning moderation.py (OpenAI Moderation API)...")
    oai_results = run_classifier(rows, moderation.check_message)

    print("Running moderation_alt.py (detoxify, local)... "
          "warming up model (downloads/loads once, not timed).")
    from guardrails import moderation_alt as _ma
    _ma._get_model()  # force load now, so per-row latency doesn't include it
    detox_results = run_classifier(
        rows, moderation_alt.check_message,
        not_applicable_categories=DETOXIFY_NOT_APPLICABLE_CATEGORIES,
    )

    print_report("moderation.py (OpenAI omni-moderation-latest)", oai_results)
    print_report("moderation_alt.py (detoxify, local)", detox_results)


if __name__ == "__main__":
    main()