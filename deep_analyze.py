#!/usr/bin/env python3
"""
Deep Error Analysis for ClaimDB Wrong Predictions

Phase 1 (default): Extracts SQL queries from model message history, re-runs them
against the actual databases, and auto-classifies each error using heuristics.

Phase 2 (--reverify): Uses Claude to independently re-verify each wrong claim
with fresh SQL, then classifies who was right (model vs benchmark).

Usage:
    python3 deep_analyze.py --results-dir results/test1
    python3 deep_analyze.py --results-dir results/test1 --reverify --workers 4
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CLAIMS_WITH_SQL_PATH = Path(__file__).parent / "claims_with_sql.jsonl"
GROUND_TRUTH_PATH = Path(__file__).parent / "test-public.jsonl"
DB_DIR_DEFAULT = Path(__file__).parent / "test_dbs"

CLASSIFICATION_LABELS = [
    "BAD_QUESTION",
    "WRONG_SQL",
    "WRONG_INTERPRETATION",
    "UNIQUENESS_PEDANTRY",
    "MISSED_ERROR",
    "CATEGORY_CONFUSION",
    "OFF_BY_SMALL_AMOUNT",
    "NEEDS_REVIEW",
]

REVERIFY_SYSTEM_PROMPT = """
You are auditing a wrong prediction from a database fact-checking benchmark.

DATABASE: {db_path}
Use sqlite3 CLI to query: sqlite3 "{db_path}" "SELECT ..."

A model was asked to verify a claim against this database.
It predicted {predicted}, but the ground truth says {actual}.

Your job:
1. Read the claim and hint carefully.
2. Explore the schema if needed.
3. Write and execute your OWN SQL to independently verify the claim.
4. Compare your findings to what the original model did (shown below).
5. Determine who is right and classify the error.

ORIGINAL MODEL'S WORK:
{model_work}

OUTPUT FORMAT (must be the LAST thing you write):
```json
{{"your_result": "<what your SQL showed>", "who_is_right": "<model|benchmark|ambiguous>", "classification": "<BAD_QUESTION|WRONG_SQL|WRONG_INTERPRETATION|UNIQUENESS_PEDANTRY|MISSED_ERROR|CATEGORY_CONFUSION|OFF_BY_SMALL_AMOUNT>", "explanation": "<brief explanation of what went wrong>"}}
```
""".strip()

REVERIFY_USER_PROMPT = """
CLAIM: {claim}

{extra_info_section}

The model predicted: {predicted}
The ground truth says: {actual}

Model's reasoning: {reasoning}

Please independently verify this claim and classify the error.
""".strip()


# ── Data loading ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Deep analysis of ClaimDB errors")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--db-dir", default=str(DB_DIR_DEFAULT), help="Path to SQLite databases")
    parser.add_argument("--reverify", action="store_true", help="Enable Phase 2: Claude re-verification")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for re-verification")
    parser.add_argument("--model", default="sonnet", help="Model for re-verification (default: sonnet)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per re-verification in seconds")
    return parser.parse_args()


def load_ground_truth():
    """Load ground truth with SQL from claims_with_sql.jsonl, fallback to test-public.jsonl."""
    gt = {}
    # Prefer claims_with_sql.jsonl (has ground truth SQL + original question)
    source = CLAIMS_WITH_SQL_PATH if CLAIMS_WITH_SQL_PATH.exists() else GROUND_TRUTH_PATH
    with open(source) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                gt[d["claim_id"]] = d
    has_sql = sum(1 for v in gt.values() if v.get("sql"))
    print(f"Loaded {len(gt)} claims from {source.name} ({has_sql} with ground truth SQL)")
    return gt


def load_evaluate(results_dir):
    with open(results_dir / "evaluate.json") as f:
        return json.load(f)


def load_detail(results_dir, claim_id):
    path = results_dir / "details" / f"{claim_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── SQL extraction from message history ─────────────────────────────────────

def extract_sql_pairs(messages):
    """Extract paired (command, result) from message history.

    Returns list of dicts with keys:
      command, sql, db_path, result, is_error, is_schema_exploration
    """
    # First pass: collect all tool_use and tool_result blocks
    tool_uses = {}  # id -> command info
    tool_results = {}  # tool_use_id -> result info

    for msg in messages:
        inner = msg.get("message", {})
        content = inner.get("content", "")
        msg_type = msg.get("type", "")

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    tool_id = block.get("id", "")
                    cmd = block.get("input", {}).get("command", "")
                    if tool_id and cmd:
                        tool_uses[tool_id] = cmd
                elif block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    result_text = block.get("content", "")
                    is_error = block.get("is_error", False)
                    if tool_id:
                        tool_results[tool_id] = {
                            "content": result_text,
                            "is_error": is_error,
                        }

    # Pair them up and parse sqlite3 commands
    pairs = []
    for tool_id, cmd in tool_uses.items():
        if "sqlite3" not in cmd:
            continue

        result_info = tool_results.get(tool_id, {"content": "", "is_error": False})

        # Parse the sqlite3 command
        db_path, sql = parse_sqlite3_command(cmd)
        if not sql:
            continue

        is_schema = sql.startswith(".") or sql.upper().startswith("PRAGMA")
        is_select = sql.upper().lstrip().startswith("SELECT") or sql.upper().lstrip().startswith("WITH")

        pairs.append({
            "command": cmd,
            "sql": sql,
            "db_path": db_path,
            "result": result_info["content"],
            "is_error": result_info["is_error"],
            "is_schema_exploration": is_schema,
            "is_verification_query": is_select and not is_schema,
        })

    return pairs


def parse_sqlite3_command(cmd):
    """Parse a sqlite3 command string to extract db_path and SQL query.

    Returns (db_path, sql) or (None, None) if parsing fails.
    """
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None, None

    # Find sqlite3 in parts
    sqlite_idx = None
    for i, p in enumerate(parts):
        if p == "sqlite3" or p.endswith("/sqlite3"):
            sqlite_idx = i
            break

    if sqlite_idx is None or sqlite_idx + 2 >= len(parts) + 1:
        return None, None

    db_path = parts[sqlite_idx + 1] if sqlite_idx + 1 < len(parts) else None
    sql = parts[sqlite_idx + 2] if sqlite_idx + 2 < len(parts) else None

    return db_path, sql


# ── SQL re-execution ────────────────────────────────────────────────────────

def rerun_sql(db_path, sql, timeout=30):
    """Re-run a SQL query against a database and return the result."""
    if not db_path or not sql or not Path(db_path).exists():
        return {"success": False, "result": None, "error": f"DB not found: {db_path}"}

    try:
        result = subprocess.run(
            ["sqlite3", db_path, sql],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "result": result.stdout.strip(),
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "result": None, "error": "Query timed out"}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


# ── Number extraction ───────────────────────────────────────────────────────

def extract_numbers_from_text(text):
    """Extract numeric values from text. Returns list of floats."""
    if not text:
        return []
    # Match integers, decimals, percentages
    pattern = r'(?<!\w)(\d[\d,]*\.?\d*)\s*%?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return numbers


def numbers_close(a, b, tolerance_pct=5, tolerance_abs=1):
    """Check if two numbers are close (within tolerance)."""
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= tolerance_abs
    pct_diff = abs(a - b) / max(abs(a), abs(b)) * 100
    return pct_diff <= tolerance_pct or abs(a - b) <= tolerance_abs


# ── Auto-classification ─────────────────────────────────────────────────────

def normalize_result(result_str):
    """Normalize a SQL result string for comparison.

    Strips whitespace, normalizes number formats (trailing .0), sorts multi-row
    results, and lowercases for case-insensitive comparison.
    """
    if not result_str:
        return ""
    lines = [l.strip() for l in result_str.strip().splitlines() if l.strip()]
    normalized = []
    for line in lines:
        # Normalize trailing .0 (e.g., "397.0" -> "397")
        parts = line.split("|")
        norm_parts = []
        for p in parts:
            p = p.strip()
            # Try to normalize numeric values
            try:
                val = float(p)
                if val == int(val):
                    p = str(int(val))
                else:
                    p = f"{val:.6g}"
            except (ValueError, OverflowError):
                pass
            norm_parts.append(p.lower())
        normalized.append("|".join(norm_parts))
    return "\n".join(sorted(normalized))


def results_match(result_a, result_b):
    """Check if two SQL results are effectively the same."""
    if not result_a or not result_b:
        return False
    return normalize_result(result_a) == normalize_result(result_b)


def get_model_final_result(sql_pairs):
    """Get the model's final verification query result (last successful SELECT)."""
    verification = [p for p in sql_pairs if p["is_verification_query"]]
    # Return the last non-error verification result
    for p in reversed(verification):
        if not p["is_error"] and p["result"] and "Error:" not in p["result"]:
            return p["result"]
    # Fall back to last verification query result
    return verification[-1]["result"] if verification else None


def auto_classify(claim_data, sql_pairs, gt_entry, gt_sql_result=None):
    """Auto-classify an error based on ground truth SQL comparison and heuristics.

    Returns (classification, reason, confidence).
    confidence is 'high', 'medium', or 'low'.
    """
    predicted = claim_data["predicted"]
    actual = claim_data["actual"]
    detail = claim_data.get("detail", {})
    reasoning = detail.get("reasoning", "")
    reasoning_lower = reasoning.lower()
    category = gt_entry.get("category", "")
    claim_text = gt_entry.get("claim", "")
    verification_queries = [p for p in sql_pairs if p["is_verification_query"]]
    model_final_result = get_model_final_result(sql_pairs)

    # ── Category confusion (NEI mishandled) ──
    if category in ("SUBJECTIVE", "COUNTERFACTUAL", "OUT-OF-SCHEMA"):
        if predicted != "NOT ENOUGH INFO":
            return "CATEGORY_CONFUSION", f"Ground truth category is {category} but model predicted {predicted}", "high"

    # ── Uniqueness pedantry (ties) ──
    tie_words = ["tie", "tied", "also has", "not unique", "uniquely", "shares", "same"]
    if any(w in reasoning_lower for w in tie_words):
        if (actual == "ENTAILED" and predicted == "CONTRADICTED") or \
           (actual == "CONTRADICTED" and predicted == "ENTAILED"):
            return "UNIQUENESS_PEDANTRY", f"Model's reasoning mentions ties/uniqueness. Predicted {predicted}, truth is {actual}", "high"

    # ══ Ground truth SQL comparison (new, high-signal) ══
    if gt_sql_result is not None and model_final_result is not None:

        model_matches_gt = results_match(model_final_result, gt_sql_result)

        if model_matches_gt:
            # Model's SQL got the SAME answer as ground truth SQL.
            # But the model still got the label wrong → interpretation/judgment issue.
            if actual == "ENTAILED" and predicted == "CONTRADICTED":
                return "WRONG_INTERPRETATION", (
                    f"Model's SQL result matches ground truth SQL result "
                    f"({gt_sql_result[:80]}), but model said CONTRADICTED instead of ENTAILED. "
                    f"Model had the right data but drew the wrong conclusion."
                ), "high"
            elif actual == "CONTRADICTED" and predicted == "ENTAILED":
                return "MISSED_ERROR", (
                    f"Model's SQL result matches ground truth SQL ({gt_sql_result[:80]}), "
                    f"but model said ENTAILED. The data contradicts the claim and the model missed it."
                ), "high"
        else:
            # Model's SQL got a DIFFERENT answer than ground truth SQL.
            # The model's query approach was wrong.
            if actual == "ENTAILED" and predicted == "CONTRADICTED":
                return "WRONG_SQL", (
                    f"Model's result ({model_final_result[:60]}) differs from "
                    f"ground truth SQL result ({gt_sql_result[:60]}). "
                    f"Model's SQL query was incorrect."
                ), "high"
            elif actual == "CONTRADICTED" and predicted == "ENTAILED":
                return "WRONG_SQL", (
                    f"Model's result ({model_final_result[:60]}) differs from "
                    f"ground truth SQL result ({gt_sql_result[:60]}). "
                    f"Model's SQL was wrong, hiding the contradiction."
                ), "high"

    # ══ Fallback heuristics (when GT SQL not available or no model result) ══

    # ── Off by small amount ──
    if actual == "ENTAILED" and predicted == "CONTRADICTED" and verification_queries:
        reasoning_numbers = extract_numbers_from_text(reasoning)
        claim_numbers = extract_numbers_from_text(claim_text)
        if claim_numbers and reasoning_numbers:
            for cn in claim_numbers:
                if cn < 2:
                    continue
                for rn in reasoning_numbers:
                    if rn < 2:
                        continue
                    if cn != rn and numbers_close(cn, rn, tolerance_pct=3, tolerance_abs=1):
                        return "OFF_BY_SMALL_AMOUNT", f"Claim says {cn}, model found {rn} (off by {abs(cn-rn):.4g})", "medium"

    # ── ENTAILED predicted as CONTRADICTED (fallback) ──
    if actual == "ENTAILED" and predicted == "CONTRADICTED":
        for p in verification_queries:
            sql_upper = p["sql"].upper()
            if "COUNT(DISTINCT" in sql_upper or "COUNT( DISTINCT" in sql_upper:
                return "WRONG_SQL", f"Model used COUNT(DISTINCT ...) which may not match the claim's intent. Got: {p['result']}", "medium"
            if "JOIN" in sql_upper:
                return "WRONG_SQL", f"Model used a JOIN that may produce wrong results. Got: {p['result']}", "low"
        return "NEEDS_REVIEW", f"Model predicted CONTRADICTED but truth is ENTAILED. Reasoning: {reasoning[:200]}", "low"

    # ── CONTRADICTED predicted as ENTAILED (fallback) ──
    if actual == "CONTRADICTED" and predicted == "ENTAILED":
        claim_numbers = extract_numbers_from_text(claim_text)
        if verification_queries and claim_numbers:
            last_result = verification_queries[-1]["result"]
            result_numbers = extract_numbers_from_text(last_result)
            for cn in claim_numbers:
                for rn in result_numbers:
                    if cn != rn and not numbers_close(cn, rn):
                        return "MISSED_ERROR", f"Model's SQL returned {rn} but claim says {cn} — model should have caught this", "medium"

        if any(w in reasoning_lower for w in ["tied", "tie", "also", "maximum"]):
            return "MISSED_ERROR", f"Model confirmed claim despite evidence of ties/ambiguity", "medium"

        return "MISSED_ERROR", f"Model confirmed a false claim. Reasoning: {reasoning[:200]}", "low"

    # ── Other flows ──
    if actual == "NOT ENOUGH INFO" and predicted in ("ENTAILED", "CONTRADICTED"):
        if category:
            return "CATEGORY_CONFUSION", f"Claim is {category} but model predicted {predicted}", "high"
        return "WRONG_INTERPRETATION", f"Model should have recognized this as NOT ENOUGH INFO", "medium"

    if actual in ("ENTAILED", "CONTRADICTED") and predicted == "NOT ENOUGH INFO":
        return "WRONG_INTERPRETATION", f"Model was overcautious, said NEI but data was available ({actual})", "medium"

    return "NEEDS_REVIEW", f"Unclassified: {actual} -> {predicted}", "low"


# ── Phase 2: Claude re-verification ─────────────────────────────────────────

def invoke_claude(prompt, system_prompt, model, timeout, working_dir):
    """Invoke Claude Code CLI and return parsed output."""
    cmd = [
        "claude",
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--model", model,
        "--system-prompt", system_prompt,
        "--dangerously-skip-permissions",
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd, cwd=working_dir,
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timed out", "output": None, "messages": []}
    except Exception as e:
        return {"success": False, "error": str(e), "output": None, "messages": []}

    return parse_stream_json_output(result.stdout, result.stderr, result.returncode)


def parse_stream_json_output(stdout, stderr, returncode):
    """Parse Claude CLI stream-json output."""
    if returncode != 0:
        return {"success": False, "error": stderr[:500], "output": None, "messages": []}

    messages = []
    final_result = ""
    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if event.get("type") in ("assistant", "user"):
                messages.append(event)
            elif event.get("type") == "result":
                final_result = event.get("result", "")
        except json.JSONDecodeError:
            continue

    return {"success": True, "output": final_result, "messages": messages}


def extract_reverify_json(output):
    """Extract the classification JSON from re-verification output."""
    if not output:
        return None
    json_pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(json_pattern, output, re.DOTALL)
    if not matches:
        json_pattern = r'\{"your_result".*?\}'
        matches = re.findall(json_pattern, output, re.DOTALL)
    for match in reversed(matches or []):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def reverify_claim(claim_analysis, args):
    """Re-verify a single wrong claim using Claude."""
    detail = claim_analysis["detail"]
    gt = claim_analysis["gt_entry"]
    db_path = str(Path(args.db_dir).resolve() / f"{detail['db_name']}.sqlite")

    # Build model work summary
    sql_pairs = claim_analysis["sql_pairs"]
    model_work_lines = []
    for p in sql_pairs:
        if p["is_schema_exploration"]:
            continue
        model_work_lines.append(f"SQL: {p['sql']}")
        model_work_lines.append(f"Result: {p['result']}")
        model_work_lines.append("")
    model_work = "\n".join(model_work_lines) if model_work_lines else "(no verification queries found)"

    system_prompt = REVERIFY_SYSTEM_PROMPT.format(
        db_path=db_path,
        predicted=claim_analysis["predicted"],
        actual=claim_analysis["actual"],
        model_work=model_work,
    )

    extra_info = gt.get("extra_info", "").strip()
    extra_info_section = f"HINTS: {extra_info}" if extra_info else ""

    user_prompt = REVERIFY_USER_PROMPT.format(
        claim=gt["claim"],
        extra_info_section=extra_info_section,
        predicted=claim_analysis["predicted"],
        actual=claim_analysis["actual"],
        reasoning=detail.get("reasoning", ""),
    )

    result = invoke_claude(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=args.model,
        timeout=args.timeout,
        working_dir=str(Path(args.db_dir).resolve().parent),
    )

    if result["success"]:
        parsed = extract_reverify_json(result["output"])
        return parsed
    return None


# ── Report generation ───────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def generate_prompt_recommendations(claims_analysis):
    """Generate actionable prompt recommendations based on error patterns."""
    recommendations = []

    classifications = Counter(c["classification"] for c in claims_analysis)

    if classifications.get("UNIQUENESS_PEDANTRY", 0) > 0:
        affected = [c["claim_id"] for c in claims_analysis if c["classification"] == "UNIQUENESS_PEDANTRY"]
        recommendations.append({
            "recommendation": "Add tie-handling rule: 'When a claim says X is the highest/tallest/most, treat it as ENTAILED if X achieves the maximum value, even if tied with others. Only classify as CONTRADICTED if X does NOT achieve the maximum.'",
            "affected_claims": affected,
            "potential_impact": f"Could fix {len(affected)} errors",
            "fixable": True,
        })

    if classifications.get("CATEGORY_CONFUSION", 0) > 0:
        affected = [c["claim_id"] for c in claims_analysis if c["classification"] == "CATEGORY_CONFUSION"]
        # Check which categories were missed
        missed_cats = Counter()
        for c in claims_analysis:
            if c["classification"] == "CATEGORY_CONFUSION":
                missed_cats[c.get("gt_category", "")] += 1
        details = ", ".join(f"{cat}: {n}" for cat, n in missed_cats.most_common())
        recommendations.append({
            "recommendation": f"Strengthen subjective/counterfactual/out-of-schema detection. Missed categories: {details}. Consider adding: 'If the claim contains subjective qualifiers like \"unremarkable\", \"alarming\", \"convenient\", \"eye-catching\", classify as NOT ENOUGH INFO (SUBJECTIVE) even if the factual component can be verified.'",
            "affected_claims": affected,
            "potential_impact": f"Could fix {len(affected)} errors",
            "fixable": True,
        })

    wrong_sql_count = classifications.get("WRONG_SQL", 0)
    if wrong_sql_count > 0:
        affected = [c["claim_id"] for c in claims_analysis if c["classification"] == "WRONG_SQL"]
        recommendations.append({
            "recommendation": "Add SQL verification step: 'Before concluding CONTRADICTED, re-read the claim and verify your SQL matches exactly what the claim asks. Common pitfalls: (1) Using COUNT(DISTINCT x) when the claim counts occurrences, not unique entities. (2) Joining tables unnecessarily. (3) Filtering on the wrong column.'",
            "affected_claims": affected,
            "potential_impact": f"Could fix up to {len(affected)} errors",
            "fixable": True,
        })

    missed_count = classifications.get("MISSED_ERROR", 0)
    if missed_count > 0:
        affected = [c["claim_id"] for c in claims_analysis if c["classification"] == "MISSED_ERROR"]
        recommendations.append({
            "recommendation": "Add double-check step: 'Before concluding ENTAILED, explicitly compare each number/name/ranking in the claim against your SQL results. If any value differs, even slightly, classify as CONTRADICTED.'",
            "affected_claims": affected,
            "potential_impact": f"Could fix up to {len(affected)} errors",
            "fixable": True,
        })

    bad_count = classifications.get("BAD_QUESTION", 0)
    if bad_count > 0:
        affected = [c["claim_id"] for c in claims_analysis if c["classification"] == "BAD_QUESTION"]
        recommendations.append({
            "recommendation": f"These {len(affected)} claims appear to be ambiguous or have incorrect ground truth labels. No prompt change can fix these.",
            "affected_claims": affected,
            "potential_impact": "Not fixable — benchmark quality issue",
            "fixable": False,
        })

    return recommendations


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    db_dir = Path(args.db_dir).resolve()

    eval_data = load_evaluate(results_dir)
    gt = load_ground_truth()

    wrong = eval_data["wrong"]
    print(f"Analyzing {len(wrong)} wrong predictions from {results_dir}")
    print(f"Databases at: {db_dir}")

    # ── Phase 1: Extract, re-run, auto-classify ──

    print_section("PHASE 1: SQL EXTRACTION & AUTO-CLASSIFICATION")

    claims_analysis = []

    for i, w in enumerate(wrong):
        claim_id = w["claim_id"]
        gt_entry = gt.get(claim_id, {})
        detail = load_detail(results_dir, claim_id)

        if not detail:
            print(f"  Warning: no detail file for claim {claim_id}")
            claims_analysis.append({
                "claim_id": claim_id,
                "predicted": w["predicted"],
                "actual": w["actual"],
                "classification": "NEEDS_REVIEW",
                "classification_reason": "No detail file found",
                "confidence": "low",
                "sql_pairs": [],
                "rerun_results": [],
                "detail": {},
                "gt_entry": gt_entry,
                "gt_category": gt_entry.get("category", ""),
            })
            continue

        # Extract SQL pairs
        sql_pairs = extract_sql_pairs(detail.get("messages", []))
        verification_queries = [p for p in sql_pairs if p["is_verification_query"]]

        # Re-run verification queries
        rerun_results = []
        for p in verification_queries:
            # Use the db from the detail file's db_name
            actual_db_path = str(db_dir / f"{detail['db_name']}.sqlite")
            rerun = rerun_sql(actual_db_path, p["sql"])
            original_was_error = (
                p["is_error"]
                or "Exit code" in p["result"]
                or "tool_use_error" in p["result"]
            )
            rerun_results.append({
                "sql": p["sql"],
                "original_result": p["result"],
                "rerun_result": rerun["result"],
                "original_was_error": original_was_error,
                "matches": (
                    original_was_error  # Don't flag errors as mismatches
                    or p["result"].strip() == (rerun["result"] or "").strip()
                ),
                "error": rerun.get("error"),
            })

        # Run ground truth SQL if available
        gt_sql = gt_entry.get("sql", "")
        gt_sql_result = None
        if gt_sql:
            actual_db_path = str(db_dir / f"{detail['db_name']}.sqlite")
            gt_run = rerun_sql(actual_db_path, gt_sql)
            gt_sql_result = gt_run["result"] if gt_run["success"] else None

        # Auto-classify
        classification, reason, confidence = auto_classify(
            {"predicted": w["predicted"], "actual": w["actual"], "detail": detail},
            sql_pairs, gt_entry, gt_sql_result,
        )

        claims_analysis.append({
            "claim_id": claim_id,
            "db_name": detail.get("db_name", ""),
            "claim": gt_entry.get("claim", ""),
            "extra_info": gt_entry.get("extra_info", ""),
            "predicted": w["predicted"],
            "actual": w["actual"],
            "gt_category": gt_entry.get("category", ""),
            "classification": classification,
            "classification_reason": reason,
            "confidence": confidence,
            "reasoning": detail.get("reasoning", ""),
            "original_question": gt_entry.get("question", ""),
            "gt_sql": gt_sql,
            "gt_sql_result": gt_sql_result,
            "sql_pairs": [{
                "sql": p["sql"],
                "result": p["result"],
                "is_schema": p["is_schema_exploration"],
            } for p in sql_pairs],
            "rerun_results": rerun_results,
            "detail": detail,
            "gt_entry": gt_entry,
        })

        # Print progress
        status = f"[{i+1}/{len(wrong)}] claim={claim_id:>6}  {w['actual']:>15} -> {w['predicted']:<15}"
        print(f"  {status}  => {classification} ({confidence})")

    # Check for re-run mismatches (only flag if both original and re-run produced real results)
    mismatches = 0
    for ca in claims_analysis:
        for rr in ca["rerun_results"]:
            if (not rr["matches"]
                    and not rr.get("original_was_error")
                    and rr.get("error") is None
                    and rr.get("rerun_result")  # re-run actually returned something
                    and rr.get("original_result")  # original had a real result
                    and "Error:" not in rr.get("original_result", "")):
                mismatches += 1

    if mismatches:
        print(f"\n  WARNING: {mismatches} SQL re-runs produced different results than original!")
    else:
        print(f"\n  All SQL re-runs match original results (database unchanged)")

    # ── Phase 1 Summary ──

    print_section("PHASE 1 RESULTS: AUTO-CLASSIFICATION")

    classifications = Counter(c["classification"] for c in claims_analysis)
    confidences = Counter(c["confidence"] for c in claims_analysis)

    print(f"\n{'Classification':<25} {'Count':>6}  {'Fixable?':<10}")
    print("-" * 50)
    fixable_map = {
        "BAD_QUESTION": "No",
        "WRONG_SQL": "Maybe",
        "WRONG_INTERPRETATION": "Maybe",
        "UNIQUENESS_PEDANTRY": "Yes",
        "MISSED_ERROR": "Maybe",
        "CATEGORY_CONFUSION": "Yes",
        "OFF_BY_SMALL_AMOUNT": "Unclear",
        "NEEDS_REVIEW": "Unknown",
    }
    for cls, count in classifications.most_common():
        print(f"{cls:<25} {count:>6}  {fixable_map.get(cls, '?'):<10}")

    print(f"\nConfidence: {dict(confidences.most_common())}")

    clearly_fixable = sum(
        1 for c in claims_analysis
        if c["classification"] in ("UNIQUENESS_PEDANTRY", "CATEGORY_CONFUSION")
    )
    needs_review = sum(
        1 for c in claims_analysis
        if c["classification"] == "NEEDS_REVIEW"
    )
    print(f"\nClearly fixable via prompt: {clearly_fixable}/{len(wrong)}")
    print(f"Need deeper review: {needs_review}/{len(wrong)}")

    # ── Per-classification breakdown with examples ──

    for cls in CLASSIFICATION_LABELS:
        items = [c for c in claims_analysis if c["classification"] == cls]
        if not items:
            continue

        print(f"\n--- {cls} ({len(items)} claims) ---")
        for c in items[:5]:  # Show up to 5 examples
            print(f"  claim_id={c['claim_id']}  db={c.get('db_name', '?')}")
            print(f"    {c['actual']} -> {c['predicted']}")
            print(f"    Claim: {c.get('claim', '?')[:120]}")
            if c.get("original_question"):
                print(f"    Original Q: {c['original_question'][:120]}")
            print(f"    Reason: {c['classification_reason'][:150]}")
            if c.get("gt_sql_result") is not None:
                gt_res = c["gt_sql_result"][:80] if c["gt_sql_result"] else "(empty)"
                # Find model's final result
                model_res = None
                for p in c.get("sql_pairs", []):
                    if not p.get("is_schema") and p.get("result"):
                        model_res = p["result"]
                model_res_str = (model_res[:80] if model_res else "(none)")
                print(f"    GT SQL result: {gt_res}")
                print(f"    Model result:  {model_res_str}")
            print()
        if len(items) > 5:
            remaining_ids = [c["claim_id"] for c in items[5:]]
            print(f"  ... and {len(items)-5} more: {remaining_ids}")
            print()

    # ── Phase 2: Claude re-verification (optional) ──

    if args.reverify:
        print_section("PHASE 2: CLAUDE RE-VERIFICATION")

        # Focus on NEEDS_REVIEW and low-confidence items
        to_reverify = [
            c for c in claims_analysis
            if c["classification"] == "NEEDS_REVIEW" or c["confidence"] == "low"
        ]
        print(f"Re-verifying {len(to_reverify)} claims with Claude ({args.model})...")

        write_lock = threading.Lock()
        completed = 0

        def process_reverify(ca):
            nonlocal completed
            result = reverify_claim(ca, args)
            with write_lock:
                completed += 1
                if result:
                    ca["reverification"] = result
                    # Update classification if Claude provided one
                    new_cls = result.get("classification", "")
                    if new_cls in CLASSIFICATION_LABELS:
                        ca["classification"] = new_cls
                        ca["classification_reason"] = result.get("explanation", ca["classification_reason"])
                        ca["confidence"] = "high"
                    who_right = result.get("who_is_right", "")
                    if who_right == "model":
                        ca["classification"] = "BAD_QUESTION"
                        ca["confidence"] = "high"
                    print(f"  [{completed}/{len(to_reverify)}] claim={ca['claim_id']} "
                          f"=> {ca['classification']} (who_right={who_right})")
                else:
                    print(f"  [{completed}/{len(to_reverify)}] claim={ca['claim_id']} => re-verification failed")

        if args.workers <= 1:
            for ca in to_reverify:
                process_reverify(ca)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_reverify, ca): ca for ca in to_reverify}
                for future in as_completed(futures):
                    future.result()  # Propagate exceptions

        # Updated summary
        print_section("PHASE 2 RESULTS: UPDATED CLASSIFICATION")
        classifications = Counter(c["classification"] for c in claims_analysis)
        print(f"\n{'Classification':<25} {'Count':>6}  {'Fixable?':<10}")
        print("-" * 50)
        for cls, count in classifications.most_common():
            print(f"{cls:<25} {count:>6}  {fixable_map.get(cls, '?'):<10}")

    # ── Prompt recommendations ──

    print_section("PROMPT RECOMMENDATIONS")
    recommendations = generate_prompt_recommendations(claims_analysis)

    if not recommendations:
        print("\n  No specific prompt recommendations generated.")
    else:
        for i, rec in enumerate(recommendations, 1):
            fixable_str = "FIXABLE" if rec["fixable"] else "NOT FIXABLE"
            print(f"\n  [{i}] ({fixable_str}) {rec['potential_impact']}")
            print(f"      {rec['recommendation']}")
            if len(rec["affected_claims"]) <= 10:
                print(f"      Claims: {rec['affected_claims']}")
            else:
                print(f"      Claims: {rec['affected_claims'][:10]} + {len(rec['affected_claims'])-10} more")

    # ── Summary ──

    print_section("EXECUTIVE SUMMARY")

    total_wrong = len(wrong)
    bad_questions = sum(1 for c in claims_analysis if c["classification"] == "BAD_QUESTION")
    clearly_fixable = sum(
        1 for c in claims_analysis
        if c["classification"] in ("UNIQUENESS_PEDANTRY", "CATEGORY_CONFUSION")
    )
    maybe_fixable = sum(
        1 for c in claims_analysis
        if c["classification"] in ("WRONG_SQL", "WRONG_INTERPRETATION", "MISSED_ERROR", "OFF_BY_SMALL_AMOUNT")
    )
    still_needs_review = sum(1 for c in claims_analysis if c["classification"] == "NEEDS_REVIEW")

    print(f"\n  Total wrong: {total_wrong}/1000 (7.0% error rate)")
    print(f"")
    print(f"  Bad questions (unfixable):     {bad_questions:>3}")
    print(f"  Clearly fixable via prompt:    {clearly_fixable:>3}")
    print(f"  Maybe fixable via prompt:      {maybe_fixable:>3}")
    print(f"  Needs deeper review:           {still_needs_review:>3}")
    print(f"")

    if still_needs_review > 0 and not args.reverify:
        print(f"  TIP: Run with --reverify to have Claude independently verify the")
        print(f"  {still_needs_review} claims that need deeper review.")
    print()

    # ── Save JSON output ──

    output_path = results_dir / "deep_analysis.json"

    # Clean up detail/gt_entry from output (too large)
    output_claims = []
    for c in claims_analysis:
        output_claims.append({
            "claim_id": c["claim_id"],
            "db_name": c.get("db_name", ""),
            "claim": c.get("claim", ""),
            "extra_info": c.get("extra_info", ""),
            "original_question": c.get("original_question", ""),
            "predicted": c["predicted"],
            "actual": c["actual"],
            "gt_category": c.get("gt_category", ""),
            "classification": c["classification"],
            "classification_reason": c["classification_reason"],
            "confidence": c["confidence"],
            "reasoning": c.get("reasoning", ""),
            "gt_sql": c.get("gt_sql", ""),
            "gt_sql_result": c.get("gt_sql_result"),
            "model_verification_queries": [
                {"sql": p["sql"], "result": p["result"]}
                for p in c.get("sql_pairs", [])
                if not p.get("is_schema")
            ],
            "rerun_results": c.get("rerun_results", []),
            "reverification": c.get("reverification"),
        })

    output_data = {
        "summary": {
            "total_wrong": total_wrong,
            "classifications": dict(classifications.most_common()),
            "bad_questions": bad_questions,
            "clearly_fixable": clearly_fixable,
            "maybe_fixable": maybe_fixable,
            "needs_review": still_needs_review,
        },
        "claims": output_claims,
        "prompt_recommendations": recommendations,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Full analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
