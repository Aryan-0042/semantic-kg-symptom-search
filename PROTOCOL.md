# Experimental Protocol v1.1 (FROZEN, 2026-09-29; amended once, see §12)

The machine-readable twin of this document is `configs/protocol.yaml`; the scripts read their seeds, grids, rules and comparisons from it. If the two ever disagree, the discrepancy is a bug to be fixed by amendment (§12), not a choice.

**Lifecycle** (steps 3–6 are enforced in code by `experiments/03_final_eval.py`):

1. Protocol frozen and committed (this version).
2. The development sweep (`02_dev_sweep.py`) runs on development rows only and writes `configs/selected_params.json`.
3. `selected_params.json` is committed.
4. Lexicon B is delivered, validated and frozen: its SHA-256 goes in `configs/lexicon_B_freeze.json`, which is committed.
5. Clean working tree; `results/rebuild/test/` empty.
6. The final evaluation runs **once**.

---

## 1. Research questions

These are working hypotheses, not conclusions. If the evidence does not support them, the conclusions change; the protocol does not.

- **RQ-A (primary).** How robust is symptom-to-condition retrieval when symptoms are expressed in lay language, and which retrieval signals remain useful under vocabulary mismatch?
- **RQ-B (method contribution).** Does linking lay symptom phrases to symptom-graph nodes by embedding similarity (semantic linking) recover the knowledge-graph signal that exact matching loses?
- **Secondary.** How do candidate-pool size and fusion rule (linear vs RRF) affect KG reranking, under matched conditions?

## 2. Data

| Item | Source | Fixed facts |
|---|---|---|
| Rows | `data/symptom_disease.csv` | 304 deduplicated rows, 41 diseases, 131 symptoms, 3–17 symptoms per row. Row id = 0-based file position (= v3 `original_row_index`). |
| Passages | `data/guideline_corpus.jsonl` | 104 MedlinePlus/CDC passages, one disease tag each. Fixed external text, not derived from rows. |
| Disease ids | `data/protocol/diseases.csv` | id = position in the sorted (Python string order) list of the 41 disease names. Used for the tie-break (§6). |
| Partition | `data/protocol/partition.csv` | seed 2026: 3 rows per disease → **development, 123 rows**; the rest → **test, 181 rows** (2–7 per disease). Written once, then verified on every run. |
| Lexicon A | `data/lexicons/lexicon_A_dev.csv` | 131 symptoms × 2 variants, authored by Claude (disclosed in `LEXICON_A_PROVENANCE.md`). **Development only.** |
| Lexicon B | `data/lexicons/lexicon_B_test.csv` | Authored independently by a co-author per `LEXICON_B_INSTRUCTIONS.md`, without access to lexicon A, code or results. Fallback: fixed-prompt LLM (`LEXICON_B_LLM_FALLBACK.md`). Frozen by hash before the final evaluation. |

## 3. Data flow

Nothing is fine-tuned: MedCPT and SapBERT are used as released. The only fitted components are built from **index rows**: the disease profiles, BM25 statistics, KG incidence and IDF, and the classifier.

| Regime | Query rows | Index rows (profiles, documents, BM25, KG, IDF, classifier, linker vocabulary) | Lexicon | Used for |
|---|---|---|---|---|
| **Development** | the 123 development rows | for each query row, **the other 122 development rows** (leave-one-row-out: 2 rows of its own disease, 3 of every other disease) | A | all design decisions and all tuning |
| **Final test** | the 181 test rows | **one fixed index of the 123 development rows**. No test row ever enters any index. | **B**; A only as a diagnostic (§10) | run once |

Properties, checked in code and by `tests/run_tests.py`:

- A development query never sees a test row, as a query or in its index.
- A final query's index is exactly the 123 development rows.
- No row is ever in its own index.
- The development rows cover all 131 symptoms, so the KG and linker vocabulary never needs test data.
- Every disease has ≥ 2 index rows in every regime. Because each row equals its disease's symptom union minus at most one symptom, and rows are distinct, the profile of every disease is complete in both regimes.

## 4. Query construction

Every condition of a row is generated from **that same row**, so wording and completeness are never confounded with which row was drawn.

| Condition | Content |
|---|---|
| `canonical_full` | all symptoms, clinical names, in the row's column order: *"High fever, cough, fatigue."* |
| `canonical_partial_k2` | 2 symptoms of the row |
| `canonical_partial_k3` | 3 symptoms: the k2 pair plus one more |
| `lay_full` | each symptom as one lay sentence from the lexicon: *"I'm burning up. I keep coughing. …"* |
| `lay_partial_k2`, `lay_partial_k3` | the same subsets as the canonical partials, with the same lay variants as `lay_full` |

**Subsets.**
- One permutation of the row's symptom indices is drawn from `default_rng([7, row_id, 1])`. The k-subset is its first k entries, presented in column order.
- The subsets are therefore nested (k2 ⊂ k3), and every partial query drops ≥ 1 symptom.
- A k-subset exists only when k < n. The 32 three-symptom rows have no k3 query (19 development, 13 test).

**Lay variants.** One variant per (row, symptom) is drawn from `default_rng([7, row_id, 2])`. It is identical across full, k2 and k3.

**Composition.**
- Queries are *composed* one clause per symptom; nothing is substituted into a finished sentence.
- Clauses are split on `[.,;:!?\n]`. Lexicon variants may not contain these characters (validated), so clauses align 1:1 with generating symptoms. This gives a gold standard for linking accuracy.

**Analysis groups.**
- `canonical_full`, `lay_full`.
- `canonical_partial` = k2 ∪ k3 queries, pooled.
- `lay_partial` = k2 ∪ k3 queries, pooled.
- k2 and k3 are also always reported separately.

**Counts.**

| Split | Wording | full | k2 | k3 | partial (k2 + k3) |
|---|---|---|---|---|---|
| Development | per wording | 123 | 123 | 104 | 227 |
| Test | per wording and lexicon | 181 | 181 | 168 | 349 |

Development has 700 queries in total.

## 5. Systems

All systems score **all 41 diseases**. Passage scores become disease scores by **MaxP** (a disease's best passage).

**Condition document** for each passage: `"<disease>. Symptoms: <profile>. <passage>"`. The profile lists every symptom of the disease in the index rows, most frequent first, ties alphabetical.

| System | Definition |
|---|---|
| BM25 | rank_bm25 Okapi, lower-cased `[a-z0-9]+` tokens, MaxP |
| Dense | MedCPT query encoder (max length *L*, §7) · MedCPT article encoder (512), CLS, L2-normalised, cosine, MaxP |
| Hybrid | α·mm(BM25) + (1−α)·mm(Dense); mm = min-max over the 41 diseases |
| RRF | Σ 1/(60 + rank) over BM25 and Dense; average ranks for ties |
| Classifier | TF-IDF (1–2-grams, sublinear tf, `[a-z0-9]+`) + logistic regression (C, lbfgs, max_iter 5000), trained on the `canonical_full` text of the index rows; class probabilities |
| KG(links) | Σ_{s ∈ links ∩ S_d} w(s) / Σ_{s ∈ links} w(s), with w(s) = ln((41 + 1)/(df(s) + 1)) + 1. S_d is the disease's symptom set in the index rows; df counts diseases. Empty links → all zeros. |
| KG-exact | KG(exact links) |
| KG-linked | KG(exact ∪ semantic links). Equal to the proposed system at **β = 1** (same linker and τ): the KG-only baseline, reported separately, outside the hypothesis family |
| **Hybrid+KG-linked** (proposed) | (1−β)·mm(Hybrid) + β·KG-linked, over all 41 diseases, with **0 < β < 1**: β ∈ {0.05, 0.10, …, 0.95} (v1.1) |
| **Hybrid+KG-exact** (H1/H2 control) | (1−β)·mm(Hybrid) + β·KG-exact, with **the same α, L, β, pool (all 41), rule (linear) and tie-break** as the proposed system |
| **Hybrid** (H3/H4 control) | the proposed system with **β = 0**: same α, L, pool and tie-break |
| *secondary:* Hybrid+KG-exact-tuned | as the H1/H2 control, but with its own development-optimal β |
| *secondary:* KG-substring, Hybrid+KG-substring-tuned | the historical substring matcher (§5.1), with its own development-optimal β |

**Matched by construction.**
- H1/H2: the proposed system and its control differ **only in the linking mechanism** (exact vs exact ∪ semantic). Everything else is shared: the KG formula, IDF, α, L, β, pool and rule. On any query where semantic linking adds no node, their scores are identical (verified).
- H3/H4: the two systems differ **only in the KG term** (β vs 0).

### 5.1 Symptom linking

- **Exact (primary baseline).**
  - Normalisation: lower-case, and every run of non-alphanumeric characters becomes one space. It is applied to node names and text alike, so *"Toxic look (typhos)"* becomes *"toxic look typhos"*.
  - Matching runs inside each clause, so it never spans clauses.
  - One left-to-right regex pass, longest name first at each position, bounded by token boundaries: whole words and phrases only.
  - Matches never overlap, so *"internal itching"* does not also yield *"itching"*, and *"coughing"* does not yield *"cough"*.
- **Substring (historical v1–v3 rule).**
  - A node is linked if its name occurs anywhere in the lower-cased, whitespace-collapsed query, including inside longer words or longer node names.
  - It is used **only** in the sensitivity analysis. It is never used to choose between matchers.
- **Semantic (intervention).**
  - linked(q) = exact(q) ∪ {argmax_n cos(e(c), e(n)) : c a clause of q, max_n cos(e(c), e(n)) ≥ τ}.
  - One best node per clause; clauses are lower-cased. It only *adds* to the exact links.
  - Node vocabulary: the KG symptom names of the index rows (all 131).
  - Encoder: the selected linker (§7), max length 64, CLS pooling, L2-normalised.

## 6. Ranking, tie-breaking and metrics

**Deterministic tie-break, identical for every system** (`src/hsr/metrics.py`):
1. Scores are rounded to 12 decimal places, so floating-point noise cannot create or break a tie.
2. Diseases are sorted by descending rounded score.
3. Equal scores are ordered by **ascending disease id** (§2).

The gold rank is p = 1 + #{strictly higher} + #{equal score and smaller id}.

- **Primary metric:** MRR over the full 41-disease ranking (RR = 1/p).
- **Secondary:** Acc@1, Hit@3, Hit@5. All are **binary** per query ([p ≤ k]).
- **Tie diagnostics** (recorded per query, reported per system × group, never used for scoring):
  - whether the gold shares its score with another disease;
  - whether the top score is shared;
  - whether Acc@1 was decided by the tie-break (best-case rank 1, worst-case rank > 1);
  - the best-case and worst-case ranks.
- Macro-F1 is not reported: with 2–7 queries per class it is noise.

## 7. Tuning and model selection (development regime only)

**Objective.** The mean, over the four groups (canonical_full, canonical_partial, lay_full, lay_partial), of the group MRR. A group MRR pools all of its queries. The objective uses development LORO and lexicon A, and the same ranking rule as evaluation (verified: the tuning objective equals the objective recomputed from evaluated per-query results).

**Selection order.** Each step fixes its parameters for the later steps. "Exact tie" means equal to within 1e-12.

| Step | Parameter(s) | Selected on | Grid | Rule for exact ties |
|---|---|---|---|---|
| 0 | MedCPT query max length *L* | Dense | {64, 128, 256, 512} | see below |
| 1 | α (BM25 weight) | Hybrid | {0, 0.1, …, 1} | closest to 0.5, then smaller |
| 2 | C | Classifier | {0.1, 1, 10} | closest to 1, then smaller |
| 3 | **(linker, τ, β) jointly** | Hybrid+KG-linked | linker ∈ {MedCPT, SapBERT}; τ ∈ {0.300, 0.325, …, 0.950}; β ∈ {0.05, 0.10, …, 0.95} (v1.1; 0 and 1 excluded) | β smallest, then τ largest, then MedCPT |
| 4 | β_exact-tuned, β_substring-tuned | the two secondary systems | β grid | smallest |
| 5 | β per (KG, n) for the linear pool rule | pool analysis (§10) | semantic-KG cells: {0.05, …, 0.95}; exact-KG cells: {0, 0.05, …, 1} | smallest |

**Query length (step 0).**
- **64 is the standard, primary setting** (MedCPT's documented query length).
- A longer length is selected only if it beats 64 on the development Dense objective by **≥ 0.01**. Among qualifying lengths, the best objective wins, then the shortest.
- Truncation rates are recorded on development and test queries. On development queries, 35% of `lay_full` queries exceed 64 tokens (1.6% exceed 128); no other condition exceeds 64.
- If a length other than 64 is selected, the paper states it as an **adaptation of MedCPT beyond its standard query setting**. `selected_params.json` records `query_length_is_standard`.

**Linker selection (step 3).**
- MedCPT vs SapBERT is decided on development data only, jointly with τ and β.
- The development record of **both** candidates is kept: each at its own best (τ, β), with per-group MRR and linking accuracy (`results/rebuild/dev/linker_selection.csv`, `linking_accuracy.csv`, full grid in `tuning_linked_joint.csv`).
- The final evaluation loads **only the selected linker**.

**β.** The β selected in step 3 is used by the proposed system *and* its H1/H2 control (§5). The control's own optimum β is reported only as the secondary system Hybrid+KG-exact-tuned.

**β endpoints (v1.1).** The proposed system is a fusion of retrieval and KG, so its β grid excludes both endpoints. β = 0 is the Hybrid control (H3/H4). β = 1 is the KG-linked baseline, reported separately at the selected linker and τ. The secondary systems in step 4 keep the full grid.

## 8. Frozen defaults (not tuned)

- **BM25:** k1 1.5, b 0.75, rank_bm25 defaults otherwise; MaxP aggregation.
- **RRF:** k = 60.
- **KG:** IDF formula as in §5; D = 41.
- **Encoders:**
  - Dense documents: MedCPT article encoder, max length 512. No final-index document exceeds 512 tokens (max 197).
  - Linkers: max length 64; clauses are short.
  - All encoders: CLS pooling, L2 normalisation, each text encoded alone (no padding effects), fp32.
- **Ranking:** round to 12 decimals; tie-break by ascending disease id.
- **Pools:** the primary systems rank all 41 diseases, with no candidate pool.
- **Seeds:** partition 2026; queries 7; bootstrap 11; permutation 13.
- **Recency:** excluded. It is a per-disease, date-dependent constant (a query-independent class prior).

## 9. Primary hypotheses and statistics

The family is four two-sided tests on **test rows, lexicon B**, metric RR, with Holm correction at α = 0.05:

| ID | Proposed (a) | Control (b) | Group | Queries | Diseases |
|---|---|---|---|---|---|
| **H1** | Hybrid+KG-linked | Hybrid+KG-exact (shared β) | lay_full | 181 | 41 |
| **H2** | Hybrid+KG-linked | Hybrid+KG-exact (shared β) | lay_partial (k2 ∪ k3) | 349 | 41 |
| **H3** | Hybrid+KG-linked | Hybrid (β = 0) | lay_full | 181 | 41 |
| **H4** | Hybrid+KG-linked | Hybrid (β = 0) | lay_partial (k2 ∪ k3) | 349 | 41 |

**Test statistic.** For the group's queries q = 1..N, clustered by disease g = 1..41:
- per-query difference d_q = RR_a(q) − RR_b(q);
- cluster sums S_g = Σ_{q∈g} d_q, and cluster sizes n_g;
- **T = (1/N) Σ_q d_q = Σ_g S_g / Σ_g n_g**, which equals ΔMRR.

**Paired randomisation test.**
- Under H0, the system labels are exchangeable within each disease, so each S_g is symmetric about 0.
- For b = 1..100,000 (seed 13): draw independent signs ε_g ∈ {−1, +1}, each with probability ½, and compute T*_b = (1/N) Σ_g ε_g S_g.
- **p = (1 + #{b : |T*_b| ≥ |T| − 10⁻¹²}) / (1 + 100,000).**

**Interval.**
- Disease-clustered percentile bootstrap: resample the 41 diseases with replacement, 10,000 times (seed 11).
- Statistic Σ S_g* / Σ n_g*; 95% interval from the 2.5 and 97.5 percentiles.

**Correction.** Holm step-down over H1–H4. Hk is rejected iff its Holm-adjusted p < 0.05.

**Reporting language.**
- "a improves on b" requires rejection with T > 0.
- Rejection with T < 0 is reported as a degradation.
- Otherwise the result is reported as "no detectable difference", with ΔMRR and its CI.

**Also reported per hypothesis, descriptively (not tests):**
- ΔAcc@1 with its clustered bootstrap CI;
- counts of queries where a or b is better.

## 10. Secondary analyses (pre-specified, descriptive, outside the family)

1. **All systems × groups and × fine conditions.** MRR, Acc@1, Hit@3 and Hit@5, with clustered 95% CIs. The canonical groups are the near-lookup ceiling (RESEARCH_AUDIT §4.1).
2. **Partial by k.** k2 and k3 are reported separately.
3. **Query length.** The development objective per *L*, plus truncation rates per condition (development; test with lexicon B).
4. **Linker selection record.** Both candidates on development data (§7).
5. **Matcher sensitivity.**
   - Hybrid+KG-exact-tuned (the exact control at its own optimum β).
   - KG-substring and Hybrid+KG-substring-tuned (the historical matcher).
   - These describe how much the H1/H2 conclusion depends on the baseline matcher and its β. They do not replace the primary control.
6. **Linking accuracy** per clause, against the generating symptom, for exact, substring and semantic linking (the selected linker at the selected τ):
   - recall = share of clauses whose gold node is linked;
   - precision = correct links / all links.
7. **Lexicon diagnostic.** The lay groups on the same test rows with lexicon A vs B. It measures dependence on the development lexicon and is **not** a primary result.
8. **Tie diagnostics** (§6).
9. **Pool size × fusion rule.** The generator is Hybrid (selected α, L). The pool is the top n ∈ {3, 5, 10, 20, all} by Hybrid; ties at the boundary go to the smaller id.
   - Inside the pool, rule ∈ {**linear**: (1−β)·mm(Hybrid) + β·K, with β tuned per (KG, n) on development data (semantic-KG cells use the proposed system's grid {0.05, …, 0.95}, so that n = all stays identical to the proposed system; exact-KG cells use the full grid); **RRF**: 1/(60 + rank_pool(Hybrid)) + 1/(60 + rank_pool(K)), no free parameter}.
   - KG ∈ {exact, linked (selected linker, τ)}.
   - Diseases outside the pool keep Hybrid's order below the pool.
   - The linear rule at n = all is exactly the corresponding primary system (verified bitwise).
   - This analysis cannot change the primary method.
10. **Latency.** Per stage (query encoding, BM25, dense scoring, linking, KG, fusion); GPU-synchronised, after warm-up; 5 repeats; median and p95; RTX 3050 6 GB. The script will be written before the final evaluation and uses no test labels.

## 11. Stopping and selection rules

1. **Development sweep.** It runs in full once on this frozen protocol. Its output is the only source of every selected value and is committed before the final evaluation.
2. **Bug fixes.** If, after the sweep, the code is found not to implement this protocol, the fix is committed and logged in §12, and the **full** development sweep is rerun. The selection is whatever the rerun produces.
3. **Method changes after the sweep.** Changes to the method, grids or rules made in response to development results are allowed only as a versioned amendment (§12) with a written reason, followed by a full rerun of the sweep. **No such amendment is allowed once lexicon B has been delivered**, since its content could otherwise inform the method.
4. **Lexicon B.**
   - It is used for nothing before the final evaluation except format validation (`01_prepare_protocol.py --check-lexicon`, which prints no performance).
   - After freezing it is never edited.
   - A validation error found before freezing is fixed by its author, without results, and the new hash is recorded.
5. **Final evaluation.**
   - It runs once, with the committed `selected_params.json`. No parameter, grid, system, metric, hypothesis or rule may change after it starts.
   - If it fails for a technical reason, its partial output is moved to `results/rebuild/test_failed_<timestamp>/`. Only a fix that cannot affect scores is allowed. It is rerun, and the failure is disclosed.
6. **Negative or null results.** They are reported as they are. The test set is never used to re-tune, to switch the primary metric, systems or hypotheses, or to choose among analyses. Any analysis not listed here is labelled *exploratory (post hoc)*.
7. **Disclosure.**
   - During implementation, mechanics checks ran on subsets of development rows. Earlier draft pipeline: 12 and 24 rows, whose printed development numbers were seen. This version: 12 rows, with performance output suppressed until after these values were set.
   - Those outputs are git-ignored and were not used to set any value in this protocol. No test row has been scored by any v4 code.

## 12. Amendment log

| Version | Date | Change | Reason |
|---|---|---|---|
| 1.0 | 2026-09-29 | Initial frozen version. | — |
| 1.1 | 2026-09-29 | The proposed system's β grid (step 3, shared by the H1/H2 control) becomes {0.05, 0.10, …, 0.95}. β = 0 is treated as the Hybrid control and β = 1 as the KG-linked-only baseline, reported separately. The semantic-KG cells of the pool analysis use the same interior grid, so that the linear rule at n = all remains identical to the proposed system (§10.9). Nothing else changes: partition, queries, linker candidates, query-length rule, other grids, hypotheses and statistics are as in v1.0. | The primary proposed architecture is intended to contain both retrieval and KG components. At β = 1 it collapses into a pure KG model, and a substantial fraction of lay rank-1 outcomes then depend on the deterministic tie-break: 22% (lay full) and 32% (lay partial) on development data. |
| 1.1 (editorial) | 2026-09-29 | The final evaluation writes to `results/rebuild/test/` instead of `results/final/`, at the user's request; the run-once guard moves with it. No methodological change and no effect on any score. | Output location only. |

**Disclosure for v1.1.** The amendment was made after the v1.0 development sweep (commit `85a730a`, git tag `dev-sweep-v1.0`), in response to its selection of β = 1.0, the top of the v1.0 grid. It was made before lexicon B was delivered, as §11.3 permits, and without any test row having been scored. The full development sweep was then rerun from scratch, with the embedding cache cleared. The v1.0 development outputs are preserved in `legacy/results/rebuild_dev_v1.0/` and `legacy/old_configs/selected_params_v1.0.json`. The amendment lowers the development objective (v1.0: 0.8470 at β = 1) and strengthens the H1/H2 control, which at β = 1 ranked every lay query alphabetically.

## 13. Changes from v3, and why

| v3 | v4 | Reason |
|---|---|---|
| One 123-query split; canonical and paraphrase from different rows | Development/test row partition; 6 conditions from **the same row**; final index = development rows only | Paired conditions; strict separation of test rows |
| Validation canonical-only, saturated at 100% | Development queries include lay and partial conditions (lexicon A) | Tuning must see the target conditions |
| Paraphrase dictionary = test instrument; sequential `str.replace` | Independent lexicons A (development) and B (test); composed queries | Independence; no nested substitution |
| Top-20 rank-discounted passage sum; MRR truncated to the shortlist | MaxP; full 41-disease ranking; one deterministic tie-break | Comparable MRR; no passage-count bias |
| Substring KG matching | Normalised whole-word exact matching (primary baseline) vs semantic linking; substring kept for sensitivity | Isolates the linking effect |
| Recency (date-dependent class prior) | Removed | Invalid signal |
| MedCPT queries at 64 tokens, truncation unreported | 64 standard; 128/256/512 as a development sensitivity with a pre-set rule; truncation reported | Transparency |
| iid bootstrap (2,000), Bonferroni, McNemar | Disease-clustered sign-flip test and bootstrap, Holm | Queries cluster by disease |

## 14. Known limitations

- Queries are generated from structured rows. Clause segmentation is given by construction, so real free text (fillers, negation, run-on sentences) is out of scope.
- The canonical conditions remain near-lookup because of the dataset's structure: every row is its disease's symptom set minus ≤ 1 symptom.
- Lexical diversity is limited: one lexicon per split, with 2 variants per symptom in lexicon A.
- The development index has 2 rows of the held-out disease, while the final index has 3 per disease. With complete profiles in both, this changes only symptom frequencies (profile order, classifier training counts), not symptom sets.
- The deterministic tie-break favours diseases with small ids when scores tie. This matters mainly for KG-only systems and is quantified by the tie diagnostics.
- The guideline passages carry little ranking signal (RESEARCH_AUDIT §5.3); they are kept as the evidence text of each condition document.

## 15. Legacy compatibility

`src/hsr/legacy_v3.py` rebuilds the v3 pipeline from the new package's components. `experiments/00_verify_legacy_v3.py` reproduces the stored v3 per-query correctness with **0 differences in all 8 conditions** (`results/rebuild/legacy_v3_check.csv`). This was re-verified on this version.
