#!/usr/bin/env python3
"""
Analyze wrong predictions from a ClaimDB evaluation run.

Loads the evaluate.json (for wrong claim IDs), ground-truth file (for claim text
and metadata), and per-claim detail files (for model reasoning and SQL messages).
Prints a structured error analysis report.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

GROUND_TRUTH_PATH = Path(__file__).parent / "test-public.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze wrong ClaimDB predictions")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to results directory (e.g. results/test1)",
    )
    parser.add_argument(
        "--show-claims",
        action="store_true",
        help="Print every wrong claim with full reasoning",
    )
    return parser.parse_args()


def load_ground_truth():
    gt = {}
    with open(GROUND_TRUTH_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                gt[d["claim_id"]] = d
    return gt


def load_evaluate(results_dir):
    path = results_dir / "evaluate.json"
    with open(path) as f:
        return json.load(f)


def load_detail(results_dir, claim_id):
    path = results_dir / "details" / f"{claim_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def extract_sql_queries(detail):
    """Extract SQL queries the model ran from the message history."""
    queries = []
    if not detail or "messages" not in detail:
        return queries
    for msg in detail["messages"]:
        inner = msg.get("message", {})
        content = inner.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    cmd = block.get("input", {}).get("command", "")
                    if "sqlite3" in cmd:
                        # Extract just the SQL part
                        queries.append(cmd)
        elif isinstance(content, str) and "sqlite3" in content:
            queries.append(content)
    return queries


def extract_tool_results(detail):
    """Extract tool results (SQL output) from message history."""
    results = []
    if not detail or "messages" not in detail:
        return results
    for msg in detail["messages"]:
        inner = msg.get("message", {})
        content = inner.get("content", "")
        # Tool results come as user messages with tool_result type
        if msg.get("type") == "user" and msg.get("tool_use_result"):
            result_content = inner.get("content", "")
            if isinstance(result_content, str) and result_content.strip():
                results.append(result_content.strip())
            elif isinstance(result_content, list):
                for block in result_content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        if text:
                            results.append(str(text).strip())
    return results


def classify_error(predicted, actual, reasoning, claim_text, sql_queries):
    """Attempt to classify the type of error based on available signals."""
    reasoning_lower = reasoning.lower() if reasoning else ""
    claim_lower = claim_text.lower()

    # ENTAILED predicted as CONTRADICTED — model found different data
    if actual == "ENTAILED" and predicted == "CONTRADICTED":
        if any(word in reasoning_lower for word in ["count", "number", "total"]):
            return "WRONG_COUNT", "Model computed a different count/number than claimed"
        if any(word in reasoning_lower for word in ["not found", "no record", "does not exist", "no match"]):
            return "ENTITY_NOT_FOUND", "Model failed to find the entity referenced in the claim"
        if any(word in reasoning_lower for word in ["shows", "reveals", "actual", "instead", "rather"]):
            return "WRONG_VALUE", "Model found a different value than claimed"
        return "INCORRECT_CONTRADICTION", "Model incorrectly determined claim was contradicted"

    # CONTRADICTED predicted as ENTAILED — model confirmed false data
    if actual == "CONTRADICTED" and predicted == "ENTAILED":
        return "MISSED_CONTRADICTION", "Model failed to catch that the claim was wrong"

    # ENTAILED predicted as NOT ENOUGH INFO
    if actual == "ENTAILED" and predicted == "NOT ENOUGH INFO":
        return "MISSED_ENTAILMENT", "Model thought data was insufficient when it was actually there"

    # CONTRADICTED predicted as NOT ENOUGH INFO
    if actual == "CONTRADICTED" and predicted == "NOT ENOUGH INFO":
        return "MISSED_CONTRADICTION_NEI", "Model thought data was insufficient instead of finding contradiction"

    # NOT ENOUGH INFO predicted as ENTAILED
    if actual == "NOT ENOUGH INFO" and predicted == "ENTAILED":
        return "FALSE_ENTAILMENT", "Model incorrectly confirmed an unverifiable claim"

    # NOT ENOUGH INFO predicted as CONTRADICTED
    if actual == "NOT ENOUGH INFO" and predicted == "CONTRADICTED":
        return "FALSE_CONTRADICTION", "Model incorrectly contradicted an unverifiable claim"

    return "OTHER", "Unclassified error"


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subsection(title):
    print(f"\n--- {title} ---")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    eval_data = load_evaluate(results_dir)
    gt = load_ground_truth()

    wrong = eval_data["wrong"]
    print(f"Analyzing {len(wrong)} wrong predictions from {results_dir}")

    # ── Confusion flow ──────────────────────────────────────────────────
    print_section("MISCLASSIFICATION FLOWS")
    flow_counts = Counter()
    for w in wrong:
        flow_counts[(w["actual"], w["predicted"])] += 1

    rows = sorted(flow_counts.items(), key=lambda x: -x[1])
    print(f"\n{'Actual':<20} {'Predicted':<20} {'Count':>6}  {'% of errors':>10}")
    print("-" * 60)
    for (actual, predicted), count in rows:
        pct = 100 * count / len(wrong)
        print(f"{actual:<20} {predicted:<20} {count:>6}  {pct:>9.1f}%")

    # ── Per-database errors ─────────────────────────────────────────────
    print_section("ERRORS BY DATABASE")
    db_errors = defaultdict(list)
    for w in wrong:
        claim_id = w["claim_id"]
        gt_entry = gt.get(claim_id, {})
        db_name = gt_entry.get("db_name", "unknown")
        db_errors[db_name].append(w)

    db_totals = eval_data.get("per_database", {})
    print(f"\n{'Database':<28} {'Errors':>6} {'Total':>6} {'Error%':>7}  Top misclassification")
    print("-" * 85)
    for db_name in sorted(db_errors, key=lambda d: -len(db_errors[d])):
        errs = db_errors[db_name]
        total = db_totals.get(db_name, {}).get("count", "?")
        pct = f"{100 * len(errs) / total:.1f}%" if isinstance(total, int) else "?"
        # Most common flow for this db
        flows = Counter((w["actual"], w["predicted"]) for w in errs)
        top_flow = flows.most_common(1)[0]
        flow_str = f"{top_flow[0][0]} -> {top_flow[0][1]} ({top_flow[1]}x)"
        print(f"{db_name:<28} {len(errs):>6} {total:>6} {pct:>7}  {flow_str}")

    # ── Per-category errors (for NOT ENOUGH INFO) ──────────────────────
    print_section("ERRORS BY CATEGORY")
    cat_errors = defaultdict(list)
    for w in wrong:
        claim_id = w["claim_id"]
        gt_entry = gt.get(claim_id, {})
        cat = gt_entry.get("category", "") or "(none)"
        cat_errors[cat].append(w)

    cat_totals = eval_data.get("per_category", {})
    print(f"\n{'Category':<20} {'Errors':>6} {'Total':>6} {'Error%':>7}")
    print("-" * 45)
    for cat in sorted(cat_errors, key=lambda c: -len(cat_errors[c])):
        errs = cat_errors[cat]
        total = cat_totals.get(cat, {}).get("count", "?")
        pct = f"{100 * len(errs) / total:.1f}%" if isinstance(total, int) else "?"
        print(f"{cat:<20} {errs.__len__():>6} {total:>6} {pct:>7}")

    # ── Error type classification ───────────────────────────────────────
    print_section("ERROR TYPE CLASSIFICATION")
    error_types = Counter()
    error_descriptions = {}
    claim_errors = []

    for w in wrong:
        claim_id = w["claim_id"]
        gt_entry = gt.get(claim_id, {})
        detail = load_detail(results_dir, claim_id)

        reasoning = detail.get("reasoning", "") if detail else ""
        claim_text = gt_entry.get("claim", "")
        sql_queries = extract_sql_queries(detail)

        etype, edesc = classify_error(
            w["predicted"], w["actual"], reasoning, claim_text, sql_queries
        )
        error_types[etype] += 1
        error_descriptions[etype] = edesc
        claim_errors.append({
            "claim_id": claim_id,
            "db_name": gt_entry.get("db_name", ""),
            "category": gt_entry.get("category", ""),
            "claim": claim_text,
            "extra_info": gt_entry.get("extra_info", ""),
            "actual": w["actual"],
            "predicted": w["predicted"],
            "reasoning": reasoning,
            "error_type": etype,
            "sql_queries": sql_queries,
        })

    print(f"\n{'Error Type':<30} {'Count':>6}  Description")
    print("-" * 90)
    for etype, count in error_types.most_common():
        print(f"{etype:<30} {count:>6}  {error_descriptions[etype]}")

    # ── The dominant error: ENTAILED misclassified as CONTRADICTED ──────
    ent_as_con = [e for e in claim_errors if e["actual"] == "ENTAILED" and e["predicted"] == "CONTRADICTED"]
    if ent_as_con:
        print_section(f"DEEP DIVE: ENTAILED predicted as CONTRADICTED ({len(ent_as_con)} claims)")
        print("\nThese are claims the database SUPPORTS, but the model said were CONTRADICTED.")
        print("The model ran SQL queries and got different results than expected.\n")

        for i, e in enumerate(ent_as_con, 1):
            print(f"  [{i}] claim_id={e['claim_id']}  db={e['db_name']}")
            print(f"      Claim: {e['claim']}")
            if e["extra_info"]:
                print(f"      Hint: {e['extra_info']}")
            print(f"      Reasoning: {e['reasoning'][:300]}")
            print()

    # ── The reverse: CONTRADICTED predicted as ENTAILED ─────────────────
    con_as_ent = [e for e in claim_errors if e["actual"] == "CONTRADICTED" and e["predicted"] == "ENTAILED"]
    if con_as_ent:
        print_section(f"DEEP DIVE: CONTRADICTED predicted as ENTAILED ({len(con_as_ent)} claims)")
        print("\nThese are claims the database CONTRADICTS, but the model confirmed them.")
        print("The model missed an error in the claim.\n")

        for i, e in enumerate(con_as_ent, 1):
            print(f"  [{i}] claim_id={e['claim_id']}  db={e['db_name']}")
            print(f"      Claim: {e['claim']}")
            if e["extra_info"]:
                print(f"      Hint: {e['extra_info']}")
            print(f"      Reasoning: {e['reasoning'][:300]}")
            print()

    # ── Other error flows ──────────────────────────────────────────────
    other_errors = [e for e in claim_errors
                    if not (e["actual"] == "ENTAILED" and e["predicted"] == "CONTRADICTED")
                    and not (e["actual"] == "CONTRADICTED" and e["predicted"] == "ENTAILED")]
    if other_errors:
        print_section(f"OTHER MISCLASSIFICATIONS ({len(other_errors)} claims)")
        for i, e in enumerate(other_errors, 1):
            print(f"  [{i}] claim_id={e['claim_id']}  db={e['db_name']}  category={e['category'] or '(none)'}")
            print(f"      {e['actual']} -> {e['predicted']}")
            print(f"      Claim: {e['claim']}")
            if e["extra_info"]:
                print(f"      Hint: {e['extra_info']}")
            print(f"      Reasoning: {e['reasoning'][:300]}")
            print()

    # ── Full claim listing (if requested) ──────────────────────────────
    if args.show_claims:
        print_section("ALL WRONG CLAIMS — FULL DETAILS")
        for i, e in enumerate(claim_errors, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(claim_errors)}] claim_id={e['claim_id']}")
            print(f"  Database:   {e['db_name']}")
            print(f"  Category:   {e['category'] or '(none)'}")
            print(f"  Actual:     {e['actual']}")
            print(f"  Predicted:  {e['predicted']}")
            print(f"  Error Type: {e['error_type']}")
            print(f"  Claim:      {e['claim']}")
            if e["extra_info"]:
                print(f"  Hint:       {e['extra_info']}")
            print(f"  Reasoning:  {e['reasoning']}")
            if e["sql_queries"]:
                print(f"  SQL Queries ({len(e['sql_queries'])}):")
                for j, q in enumerate(e["sql_queries"], 1):
                    print(f"    [{j}] {q}")

    # ── Summary ─────────────────────────────────────────────────────────
    print_section("SUMMARY")
    total_wrong = len(wrong)
    total = eval_data["overall"]["total"]
    print(f"\n  {total_wrong} wrong out of {total} ({100*total_wrong/total:.1f}% error rate)\n")

    print("  Key findings:")
    if ent_as_con:
        print(f"  - {len(ent_as_con)}/{total_wrong} ({100*len(ent_as_con)/total_wrong:.0f}%) errors are ENTAILED->CONTRADICTED")
        print(f"    The model's SQL queries returned different results than what the claim stated,")
        print(f"    but the claim was actually correct. Likely SQL interpretation issues.")
    if con_as_ent:
        print(f"  - {len(con_as_ent)}/{total_wrong} ({100*len(con_as_ent)/total_wrong:.0f}%) errors are CONTRADICTED->ENTAILED")
        print(f"    The model confirmed claims that were actually false. It missed subtle errors.")

    # Worst databases
    worst_dbs = sorted(db_errors.items(), key=lambda x: -len(x[1]))[:3]
    print(f"\n  Hardest databases:")
    for db_name, errs in worst_dbs:
        total_db = db_totals.get(db_name, {}).get("count", "?")
        print(f"    - {db_name}: {len(errs)} errors ({total_db} total)")

    print()

    # ── Save full analysis to JSON ──────────────────────────────────────
    output_path = results_dir / "error_analysis.json"
    analysis_output = {
        "total_wrong": total_wrong,
        "total": total,
        "error_rate": round(100 * total_wrong / total, 1),
        "misclassification_flows": {
            f"{actual}->{predicted}": count
            for (actual, predicted), count in flow_counts.most_common()
        },
        "errors_by_database": {
            db: len(errs) for db, errs in db_errors.items()
        },
        "errors_by_category": {
            cat: len(errs) for cat, errs in cat_errors.items()
        },
        "error_types": dict(error_types.most_common()),
        "claims": claim_errors,
    }
    with open(output_path, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"  Full analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
