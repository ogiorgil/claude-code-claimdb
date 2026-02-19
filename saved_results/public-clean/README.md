# Public Dataset — Clean Prompt (No Task-Specific Guidance)

**Accuracy: 87.0% (870/1000)**

This run uses a minimal system prompt that gives Claude no guidance about the specific NOT ENOUGH INFO subcategories in the benchmark (counterfactual, subjective, out-of-schema). The purpose is to isolate how much of the model's performance comes from its own reasoning versus task-specific instructions in the system prompt.

Previous runs (`public1`, `public2`) used a detailed system prompt that explicitly described these subcategories with examples and classification rules. One concern with that approach is that the system prompt effectively encodes knowledge about the benchmark's challenge categories, which could be seen as giving the model an unfair advantage rather than letting it reason from first principles.

This run removes all such guidance. The system prompt only tells Claude to fact-check claims against a SQLite database and classify them as ENTAILED, CONTRADICTED, or NOT ENOUGH INFO — with no elaboration on what NOT ENOUGH INFO means beyond "the database does not contain sufficient information to verify or refute the claim."

## Results Comparison

| Run | Prompt | Overall | ENTAILED | CONTRADICTED | NOT ENOUGH INFO |
|-----|--------|---------|----------|--------------|-----------------|
| public2 | detailed | 94.4% | 87.1% | 96.7% | 99.4% |
| **public-clean** | **clean** | **87.0%** | **84.1%** | **96.1%** | **81.0%** |

The 7.4 percentage point drop is almost entirely driven by NOT ENOUGH INFO claims (99.4% → 81.0%). ENTAILED and CONTRADICTED accuracy barely changes.

### NOT ENOUGH INFO Breakdown by Category

| Category | Clean Prompt | Detailed Prompt (public2) |
|----------|-------------|--------------------------|
| OUT-OF-SCHEMA | 99.1% | ~99%+ |
| SUBJECTIVE | 80.4% | ~99%+ |
| COUNTERFACTUAL | 63.7% | ~99%+ |

Without explicit guidance, Claude struggles most with counterfactual claims ("If X had happened...") — it tries to compute hypothetical answers rather than recognizing these as unverifiable. Subjective claims ("alarmingly high", "impressively low") are the second-hardest category. Out-of-schema claims remain easy since the model naturally recognizes when a database lacks relevant tables/columns.

## Configuration

- **Model**: opus
- **Prompt style**: `clean` (via `--prompt clean`)
- **Dataset**: test-public.jsonl (1000 claims)
- **Date**: 2026-02-18
