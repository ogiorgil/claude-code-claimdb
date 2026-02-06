# Claude Code on ClaimDB

Run the ClaimDB benchmark with the Claude Code CLI, then evaluate and analyze results.

**Overview**
- `run_claimdb.py` runs the benchmark and saves predictions plus full per-claim traces.
- `evaluate.py` scores predictions against the public test set and writes `evaluate.json`.
- `analyze_errors.py` and `deep_analyze.py` generate structured error analysis.

**Requirements**
- Python 3
- `sqlite3` CLI available on PATH
- `claude` CLI available on PATH and authenticated

**Repo Layout**
- `test-public.jsonl`: public evaluation set (includes ground-truth labels)
- `test-private.jsonl`: private set (no public labels)
- `test_dbs/`: SQLite databases referenced by claims
- `claims_with_sql.jsonl`: ground-truth SQL used by deep analysis
- `results/`: example run outputs

**Quickstart**
Run the public benchmark and save outputs to a new results directory:

```bash
python3 run_claimdb.py \
  --input test-public.jsonl \
  --db-dir test_dbs \
  --output-dir results/public_run \
  --model opus \
  --workers 1
```

Notes:
- `--workers` controls parallelism.
- The runner resumes if `predictions.jsonl` exists. If `evaluate.json` exists, it will rerun only previously wrong claims.

**Evaluate**
```bash
python3 evaluate.py --predictions results/public_run/predictions.jsonl
```
This prints accuracy breakdowns and writes `results/public_run/evaluate.json`.

**Error Analysis (Fast)**
```bash
python3 analyze_errors.py --results-dir results/public_run
```
Optional:
- `--show-claims` prints every wrong claim with full reasoning and SQL.

**Deep Analysis (SQL Re-run + Optional Re-verify)**
```bash
python3 deep_analyze.py --results-dir results/public_run
```
Optional re-verification with Claude:

```bash
python3 deep_analyze.py --results-dir results/public_run --reverify --workers 4 --model sonnet
```

**Input Format**
Claims are JSONL. Minimal fields:

```json
{"claim_id": 1, "db_name": "formula_1", "claim": "...", "extra_info": "..."}
```

The public set also includes `label` and `category` fields for evaluation.

**Output Files**
Each run writes to `results/<run_name>/`:
- `predictions.jsonl`: `{"claim_id": ..., "label": ...}`
- `details/<claim_id>.json`: full trace, reasoning, SQL tool calls, token usage, cost
- `run_config.json`: run parameters and start time
- `evaluate.json`: scoring summary (created by `evaluate.py`)
- `error_analysis.json`: error breakdown (created by `analyze_errors.py`)
- `deep_analysis.json`: deep analysis report (created by `deep_analyze.py`)

**Saved Results**
Precomputed runs are in `saved_results/`:
- `saved_results/public1`: 94.9% accuracy (949/1000)
- `saved_results/public2`: 94.4% accuracy (944/1000)
- `saved_results/private`: 93.7% accuracy (937/1000)

Final result (public test): **94.4% accuracy (944/1000)** from `saved_results/public2`.
