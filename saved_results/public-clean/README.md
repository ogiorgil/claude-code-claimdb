# Public Dataset — Clean Prompt (No Task-Specific Guidance)

**Accuracy: 87.0% (870/1000)**

This run uses a minimal system prompt that gives Claude no guidance about the specific NOT ENOUGH INFO subcategories in the benchmark (counterfactual, subjective, out-of-schema). The purpose is to isolate how much of the model's performance comes from its own reasoning versus task-specific instructions in the system prompt.

Previous runs (`public1`, `public2`) used a detailed system prompt that explicitly described these subcategories with examples and classification rules. One concern with that approach is that the system prompt effectively encodes knowledge about the benchmark's challenge categories, which could be seen as giving the model an unfair advantage rather than letting it reason from first principles.

This run removes all such guidance. The system prompt only tells Claude to fact-check claims against a SQLite database and classify them as ENTAILED, CONTRADICTED, or NOT ENOUGH INFO — with no elaboration on what NOT ENOUGH INFO means beyond "the database does not contain sufficient information to verify or refute the claim."

## Results Comparison

| Metric | Detailed Prompt (public1) | Clean Prompt | Delta |
|--------|--------------------------|--------------|-------|
| Overall accuracy | 949/1000 (94.9%) | 870/1000 (87.0%) | -7.9% |
| ENTAILED | 91.6% | 84.1% | -7.5% |
| CONTRADICTED | 96.1% | 96.1% | 0% |
| NOT ENOUGH INFO | 97.0% | 81.0% | -16.0% |

The 7.9pp drop is concentrated in NOT ENOUGH INFO (−16.0%) and ENTAILED (−7.5%). CONTRADICTED accuracy is identical.

## Error Analysis (130 errors)

### 1. Counterfactual claims — 41 errors

Without explicit instructions, the model computes answers to hypothetical questions instead of classifying them as NOT ENOUGH INFO. This is the single biggest failure mode (31.5% of all errors).

In many of these cases, the model applies deterministic arithmetic to existing data with a modified filter or parameter and arrives at a concrete answer:

- **Claim 864** (financial): "If the minimum loan amount had been 300K instead of 250K, approvals would stay the same." Model found 6 loans ≥ 250K but only 2 ≥ 300K — a WHERE clause change.
- **Claim 6195** (thrombosis_prediction): "If patients over 60 were ineligible for inpatient admission, the inpatient percentage among 1930–1940 births would fall below 25%." Model removed the 17 patients over 60, recomputed 39/149 = 26.17% > 25%.
- **Claim 16372** (debit_card_specializing): "If a 10% fuel subsidy were in effect, transactions > 1000 would be lower." Model multiplied each price by 0.9, re-counted: 41 < 56.
- **Claim 742** (financial): "If the bank charged 10/month from 1993–1998, the balance would be lower." Model computed 5835 − (72 × 10) = 5115 < 5835. Tautologically true.

Without guidance, the model defaults to answering computable questions rather than recognizing the benchmark's convention that counterfactual claims should be classified as NOT ENOUGH INFO regardless of computability.

### 2. Subjective claims — 22 errors

Without explicit instructions, the model treats subjective claims as verifiable by grounding adjectives in statistical context. Several of these are defensible:

- **Claim 2805** (codebase_community): "An average post score of 9.0 is unimpressive." Model found 9.0 is 3× the platform average and top 2% of users.
- **Claim 4935** (formula_1): "497 points in a season is underwhelming." That's McLaren in 2011, 2nd in Constructors, 122 points ahead of 3rd.
- **Claim 1385** (toxicology): "Calcium in non-carcinogenic molecules is unremarkable." Calcium appears exactly once — tied for the rarest element in the dataset.
- **Claim 6566** (thrombosis_prediction): "Total bilirubin of 7.9 would be considered critically high." Normal range is 0.1–1.2; 7.9 is 6.5× the upper limit and the highest value in the entire database.

The model treats these as verifiable by marshalling statistical evidence, rather than following the convention that value judgments fall outside what a database can confirm.

### 3. SQL approach differences and tie-breaking — 46 errors

**Reference SQL disagreements (39 claims):** In these cases the model and the reference SQL take different approaches that produce different results. Common patterns include: differing JOIN strategies that affect row counts, use of DISTINCT, string-vs-numeric sort order, and different table/column choices. The same 36 disagreements appeared in both the detailed-prompt and clean-prompt runs (plus 3 new ones), confirming these are independent of prompt style.

The 3 new cases in this run:
- Claim 5098 (formula_1): Different sort behavior for lap times (string vs. numeric).
- Claim 1937 (toxicology): Model found only 50% of relevant molecules are carcinogenic; claim says "all."
- Claim 6018 (european_football_2): Different interpretation of player nationality vs. league country.

**Tie-breaking conventions (7 claims):** The model identifies ties (e.g., 63 heroes tied at strength=100) and says CONTRADICTED because the claim implies uniqueness ("X is the strongest"). The reference SQL uses `LIMIT 1` and accepts the first result. This is a difference in convention around how to handle non-unique superlatives. Claims: 15832, 4204, 1379, 3986, 1591, 36, 5650.

### 4. Genuine model errors — 21 errors

Wrong column choice, missing DISTINCT, over-trusting misleading hints, or saying NOT ENOUGH INFO when it should have said CONTRADICTED (found an entity doesn't exist but didn't check the actual answer). These are clear model mistakes.

## Error Breakdown Summary

| Category | Errors | Notes |
|----------|--------|-------|
| Counterfactual (NEI convention) | 41 | Model computes answers instead of saying NEI |
| Subjective (NEI convention) | 22 | Model grounds adjectives in statistics instead of saying NEI |
| SQL approach differences + tie-breaking | 46 | Different but reasonable SQL interpretations |
| Genuine model errors | 21 | Real mistakes by the model |

The majority of the accuracy drop comes from the model not following the NOT ENOUGH INFO labeling conventions for counterfactual and subjective claims — exactly the guidance that was removed for this run. The 46 SQL approach differences and 21 genuine errors are consistent across prompt styles.

## Configuration

- **Model**: opus
- **Prompt style**: `clean` (via `--prompt clean`)
- **Dataset**: test-public.jsonl (1000 claims)
- **Date**: 2026-02-18
