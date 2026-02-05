#!/usr/bin/env python3
"""
ClaimDB Evaluation Script

Compares predictions against ground-truth labels from the public test set.
Prints overall accuracy and per-category / per-label breakdowns.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

VALID_LABELS = {"ENTAILED", "CONTRADICTED", "NOT ENOUGH INFO"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ClaimDB predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--ground-truth", required=True, help="Path to ground-truth JSONL (test-public.jsonl)")
    return parser.parse_args()


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["claim_id"]] = d["label"]
    return preds


def load_ground_truth(path):
    gt = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if "label" not in d:
                    continue
                gt[d["claim_id"]] = {
                    "label": d["label"],
                    "category": d.get("category", ""),
                    "db_name": d.get("db_name", ""),
                }
    return gt


def print_table(headers, rows, col_widths=None):
    """Print a simple text table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for row in rows:
                w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    header_line = "".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))


def evaluate(preds, gt):
    # Match predictions to ground truth
    matched = []
    missing = []
    for claim_id, truth in gt.items():
        if claim_id in preds:
            matched.append({
                "claim_id": claim_id,
                "predicted": preds[claim_id],
                "actual": truth["label"],
                "category": truth["category"],
                "db_name": truth["db_name"],
            })
        else:
            missing.append(claim_id)

    extra = [cid for cid in preds if cid not in gt]

    if not matched:
        print("No matching claim_ids found between predictions and ground truth.")
        return

    # Overall accuracy
    correct = sum(1 for m in matched if m["predicted"] == m["actual"])
    total = len(matched)
    print(f"\n{'='*60}")
    print(f"OVERALL: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"{'='*60}")

    if missing:
        print(f"\nMissing predictions for {len(missing)} claims")
    if extra:
        print(f"Extra predictions not in ground truth: {len(extra)}")

    # Parse errors
    parse_errors = [m for m in matched if m["predicted"] not in VALID_LABELS]
    if parse_errors:
        print(f"Parse errors (invalid labels): {len(parse_errors)}")

    # Per-label breakdown (confusion-style)
    print(f"\n--- Per Ground-Truth Label ---")
    rows = []
    for label in sorted(VALID_LABELS):
        subset = [m for m in matched if m["actual"] == label]
        if not subset:
            continue
        label_correct = sum(1 for m in subset if m["predicted"] == m["actual"])
        acc = 100 * label_correct / len(subset) if subset else 0
        rows.append((label, len(subset), label_correct, f"{acc:.1f}%"))
    print_table(["Label", "Count", "Correct", "Accuracy"], rows)

    # Per-category breakdown
    categories = sorted(set(m["category"] for m in matched))
    if any(c for c in categories):  # Only show if there are non-empty categories
        print(f"\n--- Per Category ---")
        rows = []
        for cat in categories:
            cat_name = cat if cat else "(none)"
            subset = [m for m in matched if m["category"] == cat]
            cat_correct = sum(1 for m in subset if m["predicted"] == m["actual"])
            acc = 100 * cat_correct / len(subset) if subset else 0
            rows.append((cat_name, len(subset), cat_correct, f"{acc:.1f}%"))
        print_table(["Category", "Count", "Correct", "Accuracy"], rows)

    # Per-database breakdown
    print(f"\n--- Per Database ---")
    rows = []
    for db in sorted(set(m["db_name"] for m in matched)):
        subset = [m for m in matched if m["db_name"] == db]
        db_correct = sum(1 for m in subset if m["predicted"] == m["actual"])
        acc = 100 * db_correct / len(subset) if subset else 0
        rows.append((db, len(subset), db_correct, f"{acc:.1f}%"))
    print_table(["Database", "Count", "Correct", "Accuracy"], rows)

    # Confusion matrix
    print(f"\n--- Confusion Matrix (rows=actual, cols=predicted) ---")
    labels = sorted(VALID_LABELS)
    confusion = Counter()
    for m in matched:
        confusion[(m["actual"], m["predicted"])] += 1

    header = [""] + labels
    rows = []
    for actual in labels:
        row = [actual]
        for predicted in labels:
            row.append(confusion.get((actual, predicted), 0))
        rows.append(row)
    print_table(header, rows)


def main():
    args = parse_args()

    preds = load_predictions(args.predictions)
    print(f"Loaded {len(preds)} predictions from {args.predictions}")

    gt = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(gt)} ground-truth labels from {args.ground_truth}")

    evaluate(preds, gt)


if __name__ == "__main__":
    main()
