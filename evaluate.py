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


GROUND_TRUTH_PATH = Path(__file__).parent / "test-public.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ClaimDB predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
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


def compute_breakdown(matched, key):
    """Compute accuracy breakdown by a given key."""
    values = sorted(set(m[key] for m in matched))
    breakdown = {}
    for val in values:
        name = val if val else "(none)"
        subset = [m for m in matched if m[key] == val]
        correct = sum(1 for m in subset if m["predicted"] == m["actual"])
        breakdown[name] = {
            "count": len(subset),
            "correct": correct,
            "accuracy": round(100 * correct / len(subset), 1),
        }
    return breakdown


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
        return None

    correct = sum(1 for m in matched if m["predicted"] == m["actual"])
    total = len(matched)
    parse_errors = [m for m in matched if m["predicted"] not in VALID_LABELS]

    # Confusion matrix
    labels = sorted(VALID_LABELS)
    confusion = Counter()
    for m in matched:
        confusion[(m["actual"], m["predicted"])] += 1
    confusion_matrix = {
        actual: {predicted: confusion.get((actual, predicted), 0) for predicted in labels}
        for actual in labels
    }

    # Per-claim results split by correctness
    correct_claims = [
        {"claim_id": m["claim_id"], "predicted": m["predicted"], "actual": m["actual"]}
        for m in matched if m["predicted"] == m["actual"]
    ]
    wrong_claims = [
        {"claim_id": m["claim_id"], "predicted": m["predicted"], "actual": m["actual"]}
        for m in matched if m["predicted"] != m["actual"]
    ]

    results = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": round(100 * correct / total, 1),
            "parse_errors": len(parse_errors),
            "missing_predictions": len(missing),
        },
        "per_label": compute_breakdown(matched, "actual"),
        "per_category": compute_breakdown(matched, "category"),
        "per_database": compute_breakdown(matched, "db_name"),
        "confusion_matrix": confusion_matrix,
        "correct": correct_claims,
        "wrong": wrong_claims,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"OVERALL: {correct}/{total} correct ({results['overall']['accuracy']}%)")
    print(f"{'='*60}")

    if missing:
        print(f"\nMissing predictions for {len(missing)} claims")
    if extra:
        print(f"Extra predictions not in ground truth: {len(extra)}")
    if parse_errors:
        print(f"Parse errors (invalid labels): {len(parse_errors)}")

    print(f"\n--- Per Ground-Truth Label ---")
    rows = [(k, v["count"], v["correct"], f"{v['accuracy']}%") for k, v in results["per_label"].items()]
    print_table(["Label", "Count", "Correct", "Accuracy"], rows)

    categories = results["per_category"]
    if any(k != "(none)" for k in categories):
        print(f"\n--- Per Category ---")
        rows = [(k, v["count"], v["correct"], f"{v['accuracy']}%") for k, v in categories.items()]
        print_table(["Category", "Count", "Correct", "Accuracy"], rows)

    print(f"\n--- Per Database ---")
    rows = [(k, v["count"], v["correct"], f"{v['accuracy']}%") for k, v in results["per_database"].items()]
    print_table(["Database", "Count", "Correct", "Accuracy"], rows)

    print(f"\n--- Confusion Matrix (rows=actual, cols=predicted) ---")
    header = [""] + labels
    rows = [[actual] + [confusion_matrix[actual][p] for p in labels] for actual in labels]
    print_table(header, rows)

    return results


def main():
    args = parse_args()
    predictions_path = Path(args.predictions)

    preds = load_predictions(predictions_path)
    print(f"Loaded {len(preds)} predictions from {predictions_path}")

    gt = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"Loaded {len(gt)} ground-truth labels from {GROUND_TRUTH_PATH}")

    results = evaluate(preds, gt)

    if results:
        output_path = predictions_path.parent / "evaluate.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
