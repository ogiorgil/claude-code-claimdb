#!/usr/bin/env python3
"""
ClaimDB Benchmark Runner

Loops through claims in a JSONL file, invokes Claude Code CLI to classify each
claim as ENTAILED, CONTRADICTED, or NOT ENOUGH INFO against a SQLite database,
and saves predictions + full trajectories.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

VALID_LABELS = {"ENTAILED", "CONTRADICTED", "NOT ENOUGH INFO"}

SYSTEM_PROMPT = """
You are a database fact-checker. Your task is to verify whether a claim is
supported by the data in a SQLite database.

DATABASE: {db_path}
Use sqlite3 CLI to query: sqlite3 "{db_path}" "SELECT ..."

WORKFLOW:
1. Read the claim carefully.
2. Explore the database schema: sqlite3 "{db_path}" ".tables" and ".schema table_name"
3. Write and execute SQL queries to check the facts in the claim.
4. Compare query results against what the claim asserts.

CLASSIFICATION:
- ENTAILED: The data in the database fully supports the claim. Every factual
  assertion in the claim matches the query results.
- CONTRADICTED: The data in the database directly contradicts the claim.
  At least one factual assertion in the claim is wrong according to the data.
- NOT ENOUGH INFO: The database does not contain sufficient information to
  verify or refute the claim.

Be precise with numbers. If a claim says "42" but the data shows "43", that is CONTRADICTED.

OUTPUT FORMAT (must be the LAST thing you write):
```json
{{"label": "<ENTAILED|CONTRADICTED|NOT ENOUGH INFO>", "reasoning": "<brief explanation>"}}
```
""".strip()

USER_PROMPT = """
Verify this claim against the database at {db_path}:

CLAIM: {claim}

{extra_info_section}
Determine if this claim is ENTAILED, CONTRADICTED, or NOT ENOUGH INFO based on the database contents.
Remember to output your final answer in the required JSON format at the end.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ClaimDB benchmark with Claude Code"
    )
    parser.add_argument("--input", required=True, help="Path to claims JSONL file")
    parser.add_argument(
        "--db-dir", required=True, help="Path to directory with SQLite databases"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory (default: results)"
    )
    parser.add_argument(
        "--model", default="opus", help="Claude Code model alias (default: opus)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per claim in seconds (default: 600)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in claims list (default: 0)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="End index in claims list, -1 for all (default: -1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    return parser.parse_args()


def load_claims(input_path, start_index, end_index):
    claims = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    if end_index == -1:
        end_index = len(claims)
    return claims[start_index:end_index]


def setup_output_dir(output_dir):
    """Set up output directory. User is responsible for choosing a unique path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if (output_path / "predictions.jsonl").exists():
        print(f"Resuming in: {output_path}")
    else:
        print(f"Output directory: {output_path}")
    return output_path


def load_completed_ids(output_dir):
    """Load claim_ids that should be skipped.

    If evaluate.json exists, only skip claims that were correct and rewrite
    predictions.jsonl to remove wrong entries so they can be rerun.
    Otherwise, skip all claims already in predictions.jsonl.
    """
    evaluate_file = output_dir / "evaluate.json"
    predictions_file = output_dir / "predictions.jsonl"

    if evaluate_file.exists():
        with open(evaluate_file) as f:
            eval_data = json.load(f)
        correct_ids = {c["claim_id"] for c in eval_data.get("correct", [])}
        wrong_ids = {c["claim_id"] for c in eval_data.get("wrong", [])}
        # Rewrite predictions.jsonl to only keep correct entries
        if predictions_file.exists() and wrong_ids:
            kept = []
            with open(predictions_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data["claim_id"] in correct_ids:
                            kept.append(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
            with open(predictions_file, "w") as f:
                f.write("\n".join(kept) + "\n" if kept else "")
        # Remove stale evaluate.json since we're rerunning wrong claims
        if wrong_ids:
            evaluate_file.unlink()
            print(
                f"Found {len(correct_ids)} correct, {len(wrong_ids)} wrong — rerunning wrong claims"
            )
        else:
            print(f"All {len(correct_ids)} claims correct, nothing to rerun")
        return correct_ids

    # No evaluation yet — skip all previously completed claims
    done_ids = set()
    if predictions_file.exists():
        with open(predictions_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        done_ids.add(data["claim_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    if done_ids:
        print(f"Found {len(done_ids)} already-completed claims, will skip them")
    return done_ids


def save_run_config(output_dir, args):
    """Save run configuration."""
    config_file = output_dir / "run_config.json"
    if config_file.exists():
        return  # Don't overwrite on resume
    config = {
        "input": str(Path(args.input).resolve()),
        "db_dir": str(Path(args.db_dir).resolve()),
        "model": args.model,
        "timeout": args.timeout,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "started_at": datetime.now().isoformat(),
    }
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def append_prediction(output_dir, claim_id, label):
    """Append a single prediction to predictions.jsonl."""
    with open(output_dir / "predictions.jsonl", "a") as f:
        f.write(json.dumps({"claim_id": claim_id, "label": label}) + "\n")


def save_detailed(output_dir, result):
    """Save a detailed result to its own JSON file."""
    details_dir = output_dir / "details"
    details_dir.mkdir(exist_ok=True)
    detail_file = details_dir / f"{result['claim_id']}.json"
    with open(detail_file, "w") as f:
        json.dump(result, f, indent=2)


def invoke_claude(prompt, system_prompt, model, timeout, working_dir):
    """Invoke Claude Code CLI and return parsed output."""
    cmd = [
        "claude",
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--dangerously-skip-permissions",
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Claude CLI timed out after {timeout}s",
            "output": None,
            "messages": [],
            "usage": {},
            "total_cost_usd": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to invoke Claude CLI: {str(e)}",
            "output": None,
            "messages": [],
            "usage": {},
            "total_cost_usd": 0,
        }

    return parse_stream_json_output(result.stdout, result.stderr, result.returncode)


def parse_stream_json_output(stdout, stderr, returncode):
    """Parse Claude CLI stream-json output."""
    if returncode != 0:
        return {
            "success": False,
            "error": f"CLI failed with code {returncode}: {stderr[:500]}",
            "output": None,
            "messages": [],
            "usage": {},
            "total_cost_usd": 0,
        }

    messages = []
    final_result = ""
    total_cost_usd = 0
    usage = {}

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            event_type = event.get("type")

            if event_type in ("assistant", "user"):
                messages.append(event)
            elif event_type == "result":
                final_result = event.get("result", "")
                total_cost_usd = event.get("total_cost_usd", 0)
                usage = event.get("usage", {})
            elif event_type == "error":
                return {
                    "success": False,
                    "error": event.get("error", {}).get("message", "Unknown error"),
                    "output": None,
                    "messages": messages,
                    "usage": {},
                    "total_cost_usd": 0,
                }
        except json.JSONDecodeError:
            continue

    return {
        "success": True,
        "output": final_result,
        "total_cost_usd": total_cost_usd,
        "usage": usage,
        "messages": messages,
    }


def extract_label(output):
    """Extract label and reasoning from Claude's output."""
    if not output:
        return "PARSE_ERROR", "No output from Claude"

    # Look for JSON block in markdown code fence
    json_pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(json_pattern, output, re.DOTALL)

    if not matches:
        # Try raw JSON with label key
        json_pattern = r'\{"label".*?\}'
        matches = re.findall(json_pattern, output, re.DOTALL)

    if matches:
        for match in reversed(matches):
            try:
                data = json.loads(match)
                label = data.get("label", "").strip()
                reasoning = data.get("reasoning", "")
                if label in VALID_LABELS:
                    return label, reasoning
            except json.JSONDecodeError:
                continue

    # Fallback: look for the label text directly in the last few lines
    last_chunk = output[-500:]
    for label in VALID_LABELS:
        if label in last_chunk:
            return label, f"Extracted from text (no JSON found)"

    return "PARSE_ERROR", f"Could not extract label from output: {output[-300:]}"


def process_claim(claim, args):
    """Process a single claim and return the full result dict."""
    db_path = str(Path(args.db_dir).resolve() / f"{claim['db_name']}.sqlite")
    extra_info = claim.get("extra_info", "").strip()

    system_prompt = SYSTEM_PROMPT.format(db_path=db_path)

    extra_info_section = ""
    if extra_info:
        extra_info_section = f"HINTS (column/value meanings): {extra_info}\n\n"

    user_prompt = USER_PROMPT.format(
        db_path=db_path,
        claim=claim["claim"],
        extra_info_section=extra_info_section,
    )

    start_time = time.time()
    result = invoke_claude(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=args.model,
        timeout=args.timeout,
        working_dir=str(Path(args.db_dir).resolve().parent),
    )
    duration = time.time() - start_time

    if result["success"]:
        label, reasoning = extract_label(result["output"])
    else:
        label = "PARSE_ERROR"
        reasoning = f"CLI error: {result['error']}"

    cli_usage = result.get("usage", {})

    return {
        "claim_id": claim["claim_id"],
        "db_name": claim["db_name"],
        "claim": claim["claim"],
        "extra_info": extra_info,
        "predicted_label": label,
        "reasoning": reasoning,
        "messages": result.get("messages", []),
        "usage": {
            "input_tokens": cli_usage.get("input_tokens", 0),
            "output_tokens": cli_usage.get("output_tokens", 0),
            "cache_read_tokens": cli_usage.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": cli_usage.get("cache_creation_input_tokens", 0),
            "cost_usd": result.get("total_cost_usd", 0),
        },
        "duration_seconds": round(duration, 2),
        "success": result["success"],
    }


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        print(f"Error: Database directory not found: {args.db_dir}")
        return

    claims = load_claims(args.input, args.start_index, args.end_index)
    print(f"Loaded {len(claims)} claims from {args.input}")

    output_dir = setup_output_dir(args.output_dir)
    done_ids = load_completed_ids(output_dir)
    save_run_config(output_dir, args)

    total_cost = 0
    completed = len(done_ids)
    errors = 0
    write_lock = threading.Lock()

    pending_claims = [c for c in claims if c["claim_id"] not in done_ids]
    print(f"Processing {len(pending_claims)} claims with {args.workers} worker(s)")

    def handle_result(result):
        nonlocal total_cost, completed, errors
        with write_lock:
            append_prediction(output_dir, result["claim_id"], result["predicted_label"])
            save_detailed(output_dir, result)

            cost = result["usage"].get("cost_usd", 0)
            total_cost += cost
            completed += 1

            if result["predicted_label"] == "PARSE_ERROR":
                errors += 1
                print(
                    f"  [{completed}/{len(claims)}] Claim {result['claim_id']} "
                    f"-> PARSE_ERROR: {result['reasoning'][:100]}"
                )
            else:
                print(
                    f"  [{completed}/{len(claims)}] Claim {result['claim_id']} "
                    f"(db: {result['db_name']}) "
                    f"-> {result['predicted_label']} "
                    f"({result['duration_seconds']:.1f}s, ${cost:.4f})"
                )

    try:
        if args.workers <= 1:
            for claim in pending_claims:
                result = process_claim(claim, args)
                handle_result(result)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_claim, claim, args): claim
                    for claim in pending_claims
                }
                for future in as_completed(futures):
                    result = future.result()
                    handle_result(result)
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Progress saved — re-run to resume.")

    predictions_path = output_dir / "predictions.jsonl"
    print(f"\n{'='*60}")
    print(f"Done! {completed}/{len(claims)} claims processed")
    print(f"Parse errors: {errors}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"\nTo evaluate:")
    print(f"  python3 evaluate.py --predictions {predictions_path}")


if __name__ == "__main__":
    main()
