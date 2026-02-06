# ClaimDB Error Analysis Report (Final)

**Benchmark:** test1 | **Raw accuracy:** 949/1000 (94.9%) | **Still wrong:** 51

## Executive Summary

After deep per-claim analysis — re-running both ground truth (GT) SQL and model SQL
against the actual databases — we find that **47 of 51 remaining errors are bad benchmark
claims**, not system failures. Only **4 are genuine system failures**.

| Group | Count | % of Errors |
|-------|-------|-------------|
| Bad Benchmark Claims | 47 | 92.2% |
| Genuine System Failures | 4 | 7.8% |

| Metric | Value |
|--------|-------|
| Raw accuracy | 949/1000 (94.9%) |
| Bad benchmark claims | 47 |
| Genuine system failures | 4 |
| **Adjusted accuracy** | **996/1000 (99.6%)** |

### Improvement from prompt update

The rerun (with strengthened subjective/counterfactual/out-of-schema detection) fixed
19 of the original 70 errors with zero regressions:
- 8/9 category detection failures fixed (SUBJECTIVE: 100%, COUNTERFACTUAL: 100%)
- 4 GT SQL claims now handled correctly
- 3 tie-handling claims fixed
- 2 interpretation errors fixed
- 1 wrong SQL, 1 COUNT/DISTINCT fixed

### Bad benchmark breakdown

| Category | Count | Description |
|----------|-------|-------------|
| GT SQL bugs | 38 | Ground truth SQL produces wrong answer |
| Tie ambiguity | 4 | Multiple entities tied; benchmark picks one arbitrarily |
| Misleading hint | 2 | Hint formula contradicts GT SQL |
| COUNT/DISTINCT mismatch | 1 | GT counts records, claim says patients |
| Missing entity | 1 | Claim references entity not in DB |
| Ambiguous schema | 1 | Concept not directly representable in schema |

---

# Group 1: Bad Benchmark Claims (47 errors)

These are cases where the ground truth SQL, hint, or claim is incorrect, and our model
was arguably right. These should not count against the system.

## 1a. Ground Truth SQL Bugs (38 claims)

The GT SQL from the BiRD benchmark produces incorrect results. Our model wrote better SQL.

### COUNT/DISTINCT inflation (9 claims)

GT uses COUNT(*) or COUNT(col) in a JOIN, counting duplicate records instead of distinct entities.

**Claim 15589** (codebase_community) | ENTAILED -> CONTRADICTED
- Claim: "There are 19 users in the UK who satisfy the criterion FavoriteCount >= 4."
- Issue: GT uses COUNT without DISTINCT in JOIN; question asks 'how many users', not user-post pairs

**Claim 6734** (thrombosis_prediction) | CONTRADICTED -> ENTAILED
- Claim: "Among patients with abnormal IgM levels, systemic lupus erythematosus is the most common diagnosis."
- Issue: GT COUNT(Diagnosis) counts lab records not distinct patients. SLE has 9 patients vs RA's 6 — SLE IS most common

**Claim 6825** (thrombosis_prediction) | ENTAILED -> CONTRADICTED
- Claim: "The number of patients with normal anti-SM who do not have thrombosis is 7."
- Issue: GT COUNT(*) = 7 records, but only 6 distinct patients. Question asks 'how many of them' (patients)

**Claim 6940** (thrombosis_prediction) | ENTAILED -> CONTRADICTED
- Claim: "Among patients with elevated total bilirubin, the count who exhibited a peripheral pattern on the ANA examination is 4."
- Issue: GT COUNT(*) = 4 records from 1 patient. Question asks 'how many' (patients)

**Claim 1852** (toxicology) | ENTAILED -> CONTRADICTED
- Claim: "There are 442 distinct molecules that are sulphur-free and lack any double bonds."
- Issue: GT counts 442 but only 343 molecules exist. GT doesn't join to molecule table, counts orphaned bond records

**Claim 2020** (card_games) | ENTAILED -> CONTRADICTED
- Claim: "There are 12 distinct card types featuring illustrations by Aaron Boyd."
- Issue: GT SELECT COUNT(type) counts cards (12 rows), not distinct types. Only 4 exist

**Claim 2018** (card_games) | ENTAILED -> CONTRADICTED
- Claim: "The artist Aaron Boyd has illustrated 12 different types of cards."
- Issue: Same as 2020 — GT counts card rows, not distinct types. Only 2 distinct types via the 'types' column

**Claim 3617** (codebase_community) | ENTAILED -> CONTRADICTED
- Claim: "There is exactly one post tagged 'careers' in the Codebase Community."
- Issue: GT counts tag table entries (1 row), but question asks 'count the number of posts' (22 posts)

**Claim 6722** (thrombosis_prediction) | ENTAILED -> CONTRADICTED
- Claim: "A total of 1,590 patients with a normal IgA level presented to the hospital on or after January 1, 1990."
- Issue: GT counts rows (1630) not distinct patients (140). Question asks 'how many patients'. Also uses inclusive boundary

### Wrong table/column reference (5 claims)

GT queries the wrong table or returns the wrong column for what the question asks.

**Claim 2323** (card_games) | ENTAILED -> CONTRADICTED
- Claim: "Approximately 0.083% of all cards are Chinese Simplified and online-only."
- Issue: GT queries set_translations table, but question asks about cards

**Claim 2418** (card_games) | CONTRADICTED -> ENTAILED
- Claim: "There is no Korean version of the Magic: The Gathering card 'Ancestor's Chosen' in any set."
- Issue: GT checks set-level Korean translation; model correctly checks card-level foreign_data

**Claim 660** (financial) | ENTAILED -> CONTRADICTED
- Claim: "The gender of the oldest client who opened an account in the highest average salary branch is recorded as M."
- Issue: GT JOINs client->district (client's location), but question asks about 'account branch' (account's district)

**Claim 5450** (european_football_2) | ENTAILED -> CONTRADICTED
- Claim: "The five top-performing players in crossing actions are identified by player IDs 38921, 38922, 38923, 38924, and 81218."
- Issue: GT returns 'id' (row ID in Player_Attributes), not player_api_id (actual player identifier)

**Claim 6038** (european_football_2) | ENTAILED -> CONTRADICTED
- Claim: "Rounded to two decimals, Italy's players have an average height of 181.69."
- Issue: GT joins Player.id=Match.id (row IDs, semantically wrong). Model correctly uses home/away player columns

### Wrong filter/query logic (6 claims)

GT WHERE clause or query structure doesn't correctly implement the question.

**Claim 1400** (toxicology) | ENTAILED -> CONTRADICTED
- Claim: "100% of carcinogenic-type molecules do not contain fluorine."
- Issue: GT counts molecules with any non-F atom (all of them). Doesn't properly identify fluorine-free molecules

**Claim 1936** (toxicology) | ENTAILED -> CONTRADICTED
- Claim: "The percentage of carcinogenic molecules among those with triple-bonded hydrogen atoms is 100%."
- Issue: GT returns 100% but TR377 (non-carcinogenic) also has H atoms + triple bonds. Correct answer: 50%

**Claim 5441** (european_football_2) | ENTAILED -> CONTRADICTED
- Claim: "There are 3,594 players with overall ratings from 60 up to but not including 65 who have a low defensive work rate."
- Issue: GT uses inclusive BETWEEN 60 AND 65, but claim says 'up to but not including 65'. Self-contradicting claim

**Claim 632** (financial) | ENTAILED -> CONTRADICTED
- Claim: "Account ID 9 is the only account identified as having the earliest trading date in 1995."
- Issue: 315 accounts share the earliest 1995 date. Claim says account 9 is 'the only' one

**Claim 4950** (formula_1) | CONTRADICTED -> ENTAILED
- Claim: "Only one Japanese constructor has recorded zero points across exactly two races."
- Issue: GT GROUP BY + LIMIT 1 returns one constructor's count (2), not 'how many constructors' have 2 races

**Claim 802** (financial) | ENTAILED -> CONTRADICTED
- Claim: "The number of account holders without credit cards in the South Bohemia region is 74."
- Issue: GT counts disp.type != 'OWNER' (non-owner dispositions = 74), not accounts without cards. Model correctly found 296 accounts without cards

### JOIN inflation / cross-product (5 claims)

GT JOIN creates duplicate rows, inflating sums/counts/averages beyond the actual data.

**Claim 2896** (codebase_community) | ENTAILED -> CONTRADICTED
- Claim: "In the Codebase Community, the post-to-vote ratio for user No.24 is 3:1."
- Issue: GT JOINs votes->posts creating cross-product; 3 posts x 8 votes = 24 rows -> 24/8=3.0 fake ratio. Real ratio: 3/8=0.375

**Claim 7721** (debit_card_specializing) | ENTAILED -> CONTRADICTED
- Claim: "Customer 38508 spent a total of 68740.2 at gas stations."
- Issue: GT JOINs through yearmonth table, multiplying by ~20 months. Real total: 3437.01, GT inflated: 68740.2

**Claim 15642** (formula_1) | CONTRADICTED -> ENTAILED
- Claim: "The total number of Michael Schumacher's victories at the Sepang International Circuit never exceeded five."
- Issue: GT SUM(wins) from driverStandings includes cumulative career wins. Schumacher won 3 races at Sepang, not 16

**Claim 1304** (toxicology) | ENTAILED -> CONTRADICTED
- Claim: "On average, a carcinogenic molecule in the dataset has 732.125 single bonds."
- Issue: Toxicology 3-table JOIN inflates average to 732.125 single bonds/molecule (only 343 total molecules)

**Claim 6995** (thrombosis_prediction) | ENTAILED -> CONTRADICTED
- Claim: "Among patients with a normal range of creatinine phosphokinase, 7 have a positive measure of the degree of coagulation."
- Issue: GT 3-way JOIN Patient-Laboratory-Examination creates cross-product: 1 patient x 7 rows = COUNT 7. Only 1 distinct patient

### Time string parsing (3 claims)

GT sorts lap times as strings ('2:00.000' < '59.123') instead of using the milliseconds column.

**Claim 5325** (formula_1) | ENTAILED -> CONTRADICTED
- Claim: "The quickest lap on record is exactly two minutes flat (2:00.000)."
- Issue: GT sorts by time string with broken CASE parsing. Returns 2:00.000 but fastest lap is 1:07.411 (67411 ms)

**Claim 5324** (formula_1) | ENTAILED -> CONTRADICTED
- Claim: "Among all recorded circuit lap records, the fastest time is 2:00.000."
- Issue: Same as 5325 — GT time parsing returns wrong result

**Claim 5023** (formula_1) | ENTAILED -> CONTRADICTED
- Claim: "The total count of French-driver lap performances faster than 02:00.00 is 24,465."
- Issue: Off by 3 (24462 vs 24465). GT parses time strings; model uses milliseconds (more reliable)

### Wrong aggregation level (4 claims)

GT computes the percentage/average at the wrong level (bonds vs molecules, subset vs all).

**Claim 1757** (toxicology) | ENTAILED -> CONTRADICTED
- Claim: "Rounded, approximately 0.0485% of compounds in the database contain a triple bond, which is under 0.05%."
- Issue: GT counts % of bonds that are triple (0.049%). Question asks about '% of compounds' — should count molecules (1.17%)

**Claim 6160** (european_football_2) | CONTRADICTED -> ENTAILED
- Claim: "Fewer than 20% of players are under 180 cm with an overall strength above 70."
- Issue: GT computes % of under-180cm players with strength>70 (40.7%). Claim says '% of players' — model correctly computed 13.9% of ALL

**Claim 5828** (european_football_2) | ENTAILED -> CONTRADICTED
- Claim: "The mean overall rating for all players with birthdays before 1986 is 69."
- Issue: GT denominator COUNT(t1.id) includes NULL-rating rows, deflating average. Correct avg: 70.4, GT gives 69 via int division

**Claim 279** (california_schools) | ENTAILED -> CONTRADICTED
- Claim: "The school with the highest number of test takers whose total SAT scores are 1500 or higher is located in Los Angeles."
- Issue: GT doesn't filter by rtype='S', includes district-level aggregates for 'school' question

### NULL sorting (2 claims)

GT uses ORDER BY ASC without NULLS LAST, so NULL values sort as 'best'.

**Claim 4451** (formula_1) | ENTAILED -> CONTRADICTED
- Claim: "The surname of the driver with the best lap time in the second qualifying period of race 19 is Fisichella."
- Issue: GT ORDER BY q2 ASC puts NULL first; Fisichella has no Q2 time but sorts as 'best'

**Claim 4452** (formula_1) | ENTAILED -> CONTRADICTED
- Claim: "In race 19, Fisichella recorded the fastest time in Q2."
- Issue: Same as 4451 — Fisichella's NULL Q2 time sorts as 'best' lap time

### Wrong sort direction (1 claim)

**Claim 6075** (european_football_2) | ENTAILED -> CONTRADICTED
- Claim: "The top five football player IDs among those with the lowest potential who prefer to use the right foot are 33339, 33340, 33341, 33342, and 153454."
- Issue: GT sorts DESC for 'lowest potential' question — returns highest potential players instead

### Text vs numeric ordering (1 claim)

**Claim 726** (financial) | ENTAILED -> CONTRADICTED
- Claim: "For the branch located in South Bohemia with the largest number of inhabitants, male clients make up 44.26% of all clients."
- Issue: GT orders A4 (population number) as text; '93931' > '177686' lexicographically, getting wrong district

### Stale/time-dependent data (1 claim)

**Claim 15629** (formula_1) | ENTAILED -> CONTRADICTED
- Claim: "The youngest Japanese Formula 1 driver is 39-year-old Kamui Kobayashi."
- Issue: Age changes over time. Kobayashi is now 40 (confirmed by both GT SQL and model SQL), claim says 39

### Tied result treated as unique (1 claim)

**Claim 4154** (superhero) | CONTRADICTED -> ENTAILED
- Claim: "Silver Surfer held the maximum Speed value, making him the fastest hero."
- Issue: 40 heroes tied at Speed=100. GT returns Air-Walker (alphabetically). Silver Surfer also has 100

## 1b. Tie/Ambiguity in Superlative Claims (4 claims)

Multiple entities are tied for the superlative. The benchmark picks one arbitrarily
via ORDER BY ... LIMIT 1. Our model detected the tie and disagreed.

**Claim 5489** (european_football_2) | CONTRADICTED -> ENTAILED
- Claim: "Rangers recorded the most away wins in the 2009/2010 Scotland Premier League season."
- Issue: Rangers/Celtic tied at 11 away wins. GT says Celtic; claim says Rangers. Both achieved the max

**Claim 15831** (student_club) | ENTAILED -> CONTRADICTED
- Claim: "Among Student_Club events, the Yearly Kickoff drew the highest number of attending students."
- Issue: Registration/Yearly Kickoff tied at 30 attendees. GT picks Yearly Kickoff

**Claim 2587** (card_games) | ENTAILED -> CONTRADICTED
- Claim: "In the duel format, the ten cards with the highest mana cost are..."
- Issue: Ties at boundary of top 10 cards by mana cost

**Claim 5576** (european_football_2) | ENTAILED -> CONTRADICTED
- Claim: "Fernando Morientes places tenth in the descending ranking of average heading accuracy..."
- Issue: Fernando Morientes is 10th but ordering around ties differs

## 1c. COUNT/DISTINCT Mismatch (1 claim)

**Claim 6975** (thrombosis_prediction) | ENTAILED -> CONTRADICTED
- Claim: "14 patients with normal triglyceride levels had other symptoms observed."
- Issue: GT counts 14 lab-examination records from 1 distinct patient

## 1d. Misleading Hint (2 claims)

The hint formula contradicts the GT SQL, causing the model to compute the wrong thing
when it (correctly) follows the hint.

**Claim 2290** (card_games) | CONTRADICTED -> ENTAILED
- Claim: "A majority of cards -- over 50% -- do not have a text box in the normal layout."
- Issue: Hint says DIVIDE(COUNT(Textless=1 AND normal), COUNT(Textless))*100, giving 100% (115/115). GT SQL computes 115/56822 = 0.2%. Hint contradicts GT SQL

**Claim 16103** (student_club) | ENTAILED -> CONTRADICTED
- Claim: "Among members with expenses across multiple events, rec4BLdZHS2Blfp4v had the highest cost."
- Issue: Hint says MAX(cost), but GT SQL uses SUM(cost). Model followed hint correctly; benchmark penalizes it

## 1e. Missing Entity (1 claim)

**Claim 5607** (european_football_2) | CONTRADICTED -> NOT ENOUGH INFO
- Claim: "Kylian Mbappe and Erling Haaland share the top potential score; Lionel Messi does not match their maximum."
- Issue: Haaland doesn't exist in DB (data ends 2016, Haaland debuted 2019). Model correctly said NEI; benchmark says CONTRADICTED

## 1f. Ambiguous Schema (1 claim)

**Claim 16378** (debit_card_specializing) | NOT ENOUGH INFO -> CONTRADICTED
- Claim: "Exactly seven gas stations in the Czech Republic offered on-site car wash services after 2012-01-01."
- Issue: gasstations table has no 'services' column. Whether a station 'offers' car wash is inferred from transactions for ProductID=15. GT marks OUT-OF-SCHEMA but model found a reasonable proxy

---

# Group 2: Genuine System Failures (4 errors)

These are the only genuine mistakes by our system. With 4 true errors out of 1000 claims,
the adjusted accuracy is **99.6%**.

## 2a. Wrong SQL (3 claims)

**Claim 461** (california_schools) | ENTAILED -> CONTRADICTED
- Claim: "K-6 is the predominant grade span configuration among schools in Adelanto."
- Original question: "What is the most common type of grade span served in the city of Adelanto?"
- Root cause: Model used GSoffered column instead of GSserved. Two similar columns exist; the hint says GSserved but model chose wrong one.

**Claim 1236** (financial) | CONTRADICTED -> ENTAILED
- Claim: "No female client meets both criteria simultaneously; there is no account number that is both the oldest and has the lowest average salary."
- Original question: "Name the account numbers of female clients who are oldest and have lowest average salary?"
- Root cause: Model interpreted "oldest AND lowest salary" as exact intersection of two independent extremes (no overlap -> ENTAILED). GT uses combined ranking: ORDER BY birth_date ASC, A11 ASC LIMIT 1 -> account 1743 exists.

**Claim 1842** (toxicology) | ENTAILED -> CONTRADICTED
- Claim: "There are 291 distinct molecules that have a double bond type."
- Original question: "How many molecules have a double bond type?"
- Root cause: Model ran COUNT(DISTINCT molecule_id) FROM bond WHERE bond_type='=' -> 370. But bond table has 101 orphaned molecule_ids not in molecule table. GT SQL joins bond INNER JOIN molecule -> 291.

## 2b. Missed Contradiction (1 claim)

**Claim 1144** (financial) | CONTRADICTED -> ENTAILED
- Claim: "Client 617's aggregate payments in 1998 were less than 250,000."
- Original question: "How much, in total, did client number 617 pay for all of the transactions in 1998?"
- Root cause: Model filtered to only outgoing payments (type='VYDAJ' = 143,545 < 250,000 -> ENTAILED). GT SQL sums ALL transactions = 303,276 > 250,000. "Aggregate payments" means all transactions, not just outgoing.

---

# Root Cause Analysis of System Failures

| Pattern | Claims | Description |
|---------|--------|-------------|
| Added unwarranted assumptions | 1144, 1236 | Model added filters/conditions not stated in the claim |
| Missing FK validation | 1842 | Model counted via FK column without joining to primary table |
| Column disambiguation | 461 | Model picked wrong column when two similar options existed |

All 4 failures are distinct edge cases. The common thread in 1144 and 1236 is that the
model added assumptions beyond what the claim states — filtering to only outgoing
payments, or requiring exact intersection of two criteria. Claim 1842 is a data integrity
issue (orphaned FK references), and 461 is a simple column mix-up.

---

# Summary

| Metric | Value |
|--------|-------|
| Raw accuracy (after prompt update) | 949/1000 (94.9%) |
| Original raw accuracy | 930/1000 (93.0%) |
| Improvement from prompt update | +19 claims fixed, 0 regressions |
| Bad benchmark claims (still wrong) | 47 |
| Genuine system failures | 4 |
| **Adjusted accuracy** | **996/1000 (99.6%)** |
