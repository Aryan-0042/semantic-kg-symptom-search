# Research Audit — Symptom-to-Condition Retrieval Project

**Status:** audit only. No method changes, no paper rewrite, and no files moved or deleted.
**Date:** 2026-09-29
**Added by this audit:** this file, plus `audit/` (3 read-only diagnostic scripts and their logs in `audit/outputs/`).

> **Release note (Phase 8).** Several root-level files named below (for example `master_results_verification_v3.py`, `temp.py`, `cureent_paper.tex`, `figs/`, `requirements.txt`) were later moved into `legacy/`; `legacy/README.md` maps each to its new location. The audit text is otherwise unchanged.
**Evidence tags:** **[V]** means verified by running code during this audit. **[R]** means established by reading code or data. **[I]** means inferred (for example from timestamps) and not proven.

---

## 1. Research objective

**What the current paper claims to study** (`cureent_paper.tex`): it compares linear score fusion (CRFS: BM25 + MedCPT dense + KG + recency) with reciprocal rank fusion followed by KG reranking. The headline claim is *"pool size, not fusion rule"* explains the gap.

**What the system actually is** [V]: a **41-class symptom-to-disease classifier** built on a ranking pipeline. Each of the 104 guideline passages is tagged with a disease. Its indexed text is `"Disease: X. Clinical symptoms: <top-8 symptoms of X from training rows>. Guideline evidence: <passage>"`. A query is a list of symptoms, and the prediction is the disease that collects the most rank-discounted score over the top-ranked passages.

Almost all of the ranking signal comes from the injected training symptoms. The guideline passages contribute almost none of it (§5.3).

**Project goal (from the brief):** a reproducible, leakage-free, statistically defensible, roughly 6-page student-conference study in medical symptom-to-condition retrieval. The final research question is **not yet decided**; §13 ranks the candidates the evidence can support.

---

## 2. Headline findings

1. **Every stored result reproduces exactly.** Stored v1, v2 and v3 results show 0 per-query differences for all methods [V] (`audit/outputs/audit_reproduce_history.log`). The historical *numbers* are trustworthy; several *conclusions* drawn from them are not.
2. **Why the "master verification" disagreed:** `master_results_verification_v3.py` builds BM25, dense and KG from `split_dev_train_v3.csv` (148 rows) instead of `split_dev_v3.csv` (181 rows, which is what produced the stored results). It also defines N1 differently: top-3 from linear fusion rather than from the RRF shortlist. Using the 148-row pool reproduces every one of its values [V]. So the verification script was the thing that differed. Neither the code nor the data had drifted.
3. **The "100% canonical CRFS" result is reproducible** with the original protocol (41/41) [V]. It drops to 40/41 only under the verification script's different index pool.
4. **The original "weight collapse" was produced by the tie-break rule, not by the data.** v1 CRFS tuning had **72 of 286** weight configurations tied at 100% dev-val F1 and MRR. The v1 script broke ties by *lowest `w_recency`* (`src/tune_crfs_weights.py:128-138`), which structurally produced `w_rec = 0`. `w_dense = 0` was an arbitrary pick from the tied set [V]. That search used v1 dev-train/dev-val; it never touched the leaky v2 test set. The paper's statement that leakage caused the collapse is therefore wrong.
5. **The v3 "all four signals get meaningful weight" result is also a tie-break artefact.** The weights are the *centroid* of **64** configurations tied at 100% dev-val [V]. Ten of those tied configurations have `w_dense = 0` and 24 have `w_rec = 0`. The validation set is saturated and cannot discriminate between weightings.
6. **The v3 test set is row-disjoint but not semantically held out.** Every one of the 304 rows equals its disease's full symptom set minus at most one symptom [V]. All test symptoms (100%) appear in the dev pool's symptom union for the correct disease [V]. The canonical queries therefore amount to near-lookup: an exact-string KG lookup with no retrieval scores 95.1% on them [V].
7. **The guideline corpus contributes almost nothing.** With guideline text only, BM25 scores 7.3% and dense 18–26%. With the injected symptom profiles they score 65–66% [V].
8. **Recency is conceptually invalid and non-deterministic.** Every passage of a disease shares one page date, so recency is a query-independent *class prior* based on when MedlinePlus last updated a page. It also uses `pd.Timestamp.now()` (`src/pipeline_common.py:239`) [R]. CRFS MRR already drifted from 0.913 to 0.912 between the run date and today [V].
9. **The latency advantage is mostly an implementation artefact.** The constant recency vector is re-parsed from dates for all 104 docs on every query (~21 ms), and KG is recomputed per document instead of per disease (~7 ms vs ~2 ms) [V]. The "3.4× faster" claim reflects this overhead, not the architecture.
10. **Tuning selections rest on one or two validation queries.** Shortlist size 3 beats 8–50 because of a single dev-val query (index 17 of 33) [R] (`results/rrf_shortlist_diagnostic_v3.txt`). The v3 conclusions also depend on the index pool: adding the 33 dev-val rows changes BM25's correctness on 13 of 123 test queries, and changes the CRFS−RRF gap from 9 to 6 queries (p from 0.022 to 0.109) [V].
11. **8 of 82 paraphrased v3 queries contain corrupted text** caused by nested dictionary substitution, for example *"a hacking cough that won't quiting up phlegm"* and *"an my skin is really itchy feeling inside"* [V].
12. **The paper's figures do not match its text.** `make_plots.py` reads **v2** CSVs, so `figs/baseline_comparison.pdf` and `figs/subset_breakdown.pdf` show v2 data under v3 captions. `figs/accuracy_vs_latency.pdf`, which the paper includes, does not exist [R].

---

## 3. Repository map

The folder is not a git repository. No `README` exists.

| Area | Files | Role / status |
|---|---|---|
| Shared pipeline | `src/pipeline_common.py` | **Active core.** It contains the loaders, retrieval-text builder, MedCPT encoder, KG, recency, CRFS, `run_condition`, RRF, `kg_rerank` and `run_condition_rrf_kg_recency`. It is imported by all v2/v3 scripts and by the master verification. |
| | `legacy/pipeline_common.py` (= `legacy/src2/pipeline_common.txt`) | The same linear code without the RRF section. It is **still imported** by `src/tune_crfs_weights.py` and `src/retrieval_eval_final_L2.py` (`from legacy.pipeline_common`). Do not move it. |
| Data preparation | `src/prepare_dataset.py`, `src/generate_urgency_labels.py` | Raw Kaggle `dataset.csv` → `data/symptom_disease.csv` (304 rows, adds `Urgency`). It also writes `split_train/val/test.csv`, which no longer exist and are unused. |
| Corpus | `src/create_disease_url_map.py`, `src/build_guideline_corpus.py`, `src/audit_corpus.py`, `src/audit_source_dates.py` | Web scrape (DuckDuckGo search, then MedlinePlus/CDC) → `data/guideline_corpus.jsonl`. **This cannot be re-run reproducibly** because live pages change. Treat the `.jsonl` as a frozen artefact. |
| Split generation | `make_frozen_test_split.py` (v1), `make_dev_train_val_split.py` (v1), `make_new_frozen_test_split.py` (v2), `make_frozen_test_split_v3.py`, `make_dev_train_val_split_v3.py` | See §4. |
| Paraphrasing | `src/paraphrase_dictionary.py` (v3, +4 terms), `src/paraphrase_dictionary_old.py` (v2) | The test instrument. It has a nested-substitution bug (§11). |
| Tuning | `tune_crfs_weights.py` (v1), `tune_rrf_shortlist.py` (v2), `tune_crfs_weights_v3.py`, `tune_rrf_shortlist_v3.py` | See §6. |
| Evaluation | `retrieval_eval_legacy.py` (v0), `retrieval_eval_final.py` / `_L2.py` (v1), `retrieval_eval_final_v2.py`, `retrieval_eval_final_v3.py` | See §7. |
| Diagnostics (root) | `rrf_shortlist_diagnostic_v3.py`, `rrf_rank_diagnostic_v3.py`, `src/quick_recency_variance_check.py`, `master_results_verification_v3.py` | v3-era diagnostics and the (mis-specified) verification. |
| Utilities | `temp.py` (concatenates CSVs → `combined_results.txt`), `files_structure.py` (prints the tree), `pre_phase3_audit.py` (fully commented out) | Scratch. |
| Out of paper scope | `triage_classifier_train.py`, `igft_dialogue_sim.py`, `build_safety_testset.py`, `finalize_safety_dataset.py`, `latency_benchmark.py` (v2 config), `make_plots.py` | Parts of the wider "assistant" (triage, dialogue, safety) that are excluded from the paper. `data/raw/Suicide_Detection.csv` (167 MB) is used only by the safety-set builder. |
| Stale caches | `models/medcpt_doc_embeddings.npy`, `models/medcpt_retrieval_embeddings.npy` | Both have shape **(108, 768)**, but the corpus now has **104** docs [V]. They were used only by `retrieval_eval_legacy.py`. |
| Paper | `cureent_paper.tex`, `figs/*.pdf` | The figures are v1/v2-era (§2.12). |
| Stray data | `arthritis_uti_subset.csv` (root) | 6 Arthritis/UTI rows, likely used for the v3 paraphrase-coverage fix [I]. |
| References | `reference_papers/` (14 papers) | See §12. |

**Environment:** `requirements.txt` pins `torch==2.2.2`, but the venv has `2.5.1+cu121`. `statsmodels`, which the eval scripts use, is not listed. The GPU is an RTX 3050 6 GB and MedCPT is cached locally [V].

---

## 4. Dataset and split map

### 4.1 Source data [V]

- Kaggle *Disease Symptom Prediction*: 4,920 rows = 41 diseases × 120 copies. Exact deduplication leaves **304 rows** (5–10 per disease, median 7).
- **Generative structure:** every row equals its disease's full symptom union minus **at most one** symptom (304/304). There are 131 distinct symptoms, **80 of which occur in only one disease**. No symptom set maps to two diseases.
- Consequence: any method that has seen other rows of a disease knows the test row's symptom set almost exactly. On canonical wording, the task is close to a lookup.
- Label strings contain typos and double spaces (`Paroymsal  Positional`, `Osteoarthristis`, `Peptic ulcer diseae`, `hepatitis A`). These are cosmetic but visible in tables.

### 4.2 Guideline corpus [V]

- 104 passages: 101 from MedlinePlus, 3 from CDC. Each has 40–100 words (mean 54). There are 1–5 passages per disease (7 diseases have only 1) and no duplicate texts.
- 73/104 passages are dated. All passages of a disease share the same URL and therefore the same date. Five passages are page boilerplate (e.g. "Updated by: … Review provided by VeriMed…").
- The unequal passage count per disease feeds the disease aggregation, which sums over the top 20 passages. This can favour diseases that have more passages [R].

### 4.3 Splits

| Gen. | File | Rows | Construction (seed) | Index pool used in its eval | Test/pool overlap |
|---|---|---|---|---|---|
| v0 | in-script 70/15/15 (`retrieval_eval_legacy.py`) | ~213/46/45 | `train_test_split` seed 42 | train (213) | — (cache mismatch, §7) |
| v1 | `split_test_final.csv` | 82 (2/disease) | `make_frozen_test_split.py` (42) | `split_dev.csv` (222) | 0 exact; **all** symptoms covered |
| v1 | `split_dev_train.csv` / `split_dev_val.csv` | 181 / 41 | `make_dev_train_val_split.py` (43) | — | 0 exact |
| v2 | `split_test_final_v2.csv` | 82 (41 canon. + 41 para.) | `make_new_frozen_test_split.py` (99), drawn from **all 304 rows** | `split_dev.csv` (222) | **58/82 exact** (the paper's disclosure is correct) [V] |
| v3 | `split_test_final_v3.csv` | 123 (41 canon. + 82 para.) | `make_frozen_test_split_v3.py` (157) | `split_dev_v3.csv` (181) = exact complement | 0 exact; **100% symptoms covered**; mean max-Jaccard 0.79 [V] |
| v3 | `split_dev_train_v3.csv` / `split_dev_val_v3.csv` | 148 / 33 | `make_dev_train_val_split_v3.py` (211) | — | 0 exact; 8 diseases absent from dev-val |

**Canonical split (current):** `data/split_test_final_v3.csv` + `data/split_dev_v3.csv` (test and index), with `split_dev_train_v3.csv` / `split_dev_val_v3.csv` for tuning. `original_row_index` refers to rows of `data/symptom_disease.csv`.

**Test reuse:** 37 of the 123 v3 test rows were already v2 test rows, and 31 were v1 test rows [V]. Results on those rows had been seen before the v3 design decisions were made.

**Query form:** `"The patient reports s1, s2, …."` with underscores replaced by spaces. Paraphrased queries apply the fixed dictionary (one lay rewrite per term) to **all** matching terms. In v3 an average of 73% of a query's symptoms are rewritten, but 31% of the canonical symptom strings still appear literally, because several paraphrases keep the clinical word ("cough" → "a hacking cough that won't quit"; "sweating" → "sweating a lot…") [V]. Each canonical query and its paraphrased variants come from **different rows**, so the canonical-vs-paraphrased comparison is confounded by row.

---

## 5. Pipeline map

### 5.1 Execution flow of the v3 result

```
data/raw/dataset.csv
  └─ src/prepare_dataset.py ─→ data/symptom_disease.csv ─┐ src/generate_urgency_labels.py (adds Urgency)
src/make_frozen_test_split_v3.py (seed 157, paraphrase_dictionary.py)
  ├─→ data/split_test_final_v3.csv  (123)   ├─→ data/split_dev_v3.csv (181)   └─→ results/paraphrase_coverage_v3.csv
src/make_dev_train_val_split_v3.py (seed 211) ─→ split_dev_train_v3.csv (148), split_dev_val_v3.csv (33), results/dev_val_coverage_v3.csv
src/tune_crfs_weights_v3.py   (index: dev-train, objective: dev-val) ─→ results/crfs_tuning_report_v3.csv, frozen_config_v3.json
src/tune_rrf_shortlist_v3.py  (index: dev-train, objective: dev-val) ─→ results/rrf_shortlist_tuning_report_v3.csv, frozen_config_rrf_v3.json
src/retrieval_eval_final_v3.py (index: dev_v3 = train+val; queries: test_v3) ─→ 8 result CSVs (§7)
```

The scripts use relative paths and must be run from the project root, with `src` on the import path (e.g. `cd` to the root and run `python src/…`).

### 5.2 Components (`src/pipeline_common.py`)

| Component | Implementation | Notes |
|---|---|---|
| Doc text | `build_retrieval_text` L87-114: `Disease: X. Clinical symptoms: <8 most frequent symptoms of X in the pool>. Guideline evidence: <text>` | The pool's labelled rows are written into every document, which makes this a supervised profile. |
| BM25 | `BM25Okapi` (k1=1.5, b=0.75 defaults) on `text.lower().split()`, whitespace tokens | Punctuation stays attached ("fever," ≠ "fever"). Scores are min-max normalised per query. |
| Dense | MedCPT query encoder (max 64 tokens) and article encoder (max 512), CLS vector, L2-normalised, dot product | Documents are re-encoded on every run (no cache). Min-max normalised for linear fusion. |
| KG | `networkx` graph: disease–document edges from the corpus, disease–symptom edges from pool rows | Built from the same pool as the doc text. |
| KG score | `kg_relevance` L195-232: a query symptom counts as "present" if its KG name is a **substring** of the query. Score = IDF-weighted share of the query's symptoms that the candidate disease has | Disease-level score (identical for all passages of a disease). Exact-string matching, so it is brittle under paraphrase. Two substring collisions exist (`itching`⊂`internal itching`, `joint pain`⊂`hip joint pain`). |
| Recency | `recency_weight` L235-242: `0.5**(age_days/365)` relative to **now**; undated = 0.5 | A per-disease constant, so it acts as a query-independent prior. Non-deterministic over time. |
| CRFS | Weighted sum of the 4 normalised scores (weights on the simplex) | "Hybrid" uses a hard-coded 0.7·BM25 + 0.3·dense (never tuned or motivated). |
| RRF path | RRF(BM25 ranks, dense ranks, k=60) → top-*n* shortlist → sort by (KG, RRF, date) → aggregate using **RRF** scores | Recency acts only as a third-level tie-break. |
| Aggregation | Over the top 20 ranked docs, `disease += score × 1/rank`; the prediction is the argmax | Diseases with more passages can accumulate more score. |

### 5.3 What carries the signal [V] (v3 test, `audit/outputs/audit_diagnostics.log`)

| Document text | BM25 acc. | Dense acc. |
|---|---|---|
| guideline passage only | 7.3% | 17.9% (no disease name) / 26.0% (with name) |
| as published (with injected top-8 training symptoms) | 65.8% | 65.0% |
| KG exact-match lookup alone (no retrieval) | 95.1% canonical / 68.3% paraphrased / 77.2% overall | |

Interpretation: the KG (a symptom-overlap lookup) is the strongest single signal on canonical wording. On paraphrased wording, dense retrieval degrades less than lexical matching and exact-match KG. The guideline evidence is essentially decorative for ranking. It could still serve as the *returned evidence*, but that role is not currently evaluated.

---

## 6. Parameter selection

| Parameter | Script | Selection data | Grid / rule | Outcome | Problem |
|---|---|---|---|---|---|
| CRFS weights v1 | `tune_crfs_weights.py` | index v1 dev-train (181), objective v1 dev-val (41 canonical) | 286-point simplex, step 0.1; macro-F1 → MRR → **lowest w_rec** | (0.4, 0, 0.6, 0) | **72 configs tied at 100%**; tie-break forced w_rec=0 |
| RRF shortlist v2 | `tune_rrf_shortlist.py` | same | {10,15,20,30,50}; F1 → MRR → smallest | 15 | 4 tied at 100% |
| CRFS weights v3 | `tune_crfs_weights_v3.py` | index v3 dev-train (148), objective dev-val (33 canonical) | same grid; F1 → MRR → **centroid of ties** | (0.205, 0.314, 0.378, 0.103) | **64 configs tied at 100%**; the centroid is not an optimum |
| RRF shortlist v3 | `tune_rrf_shortlist_v3.py` | same | {3,5,8,10,15,20,30,50} → smallest | 3 | 3 and 5 at 100%, 8–50 at 97% (**one query**) |
| RRF k | — | — | fixed at 60 | 60 | fine |
| Hybrid α, top_k=20, KG IDF form, recency half-life, 0.5 for undated | — | none | hard-coded | — | undocumented constants |

**Data allowed to influence selection:** by design, dev-train (index) and dev-val (objective). No tuning script loads a test file [R]. In practice, **validation contains only canonical queries**, while 2/3 of the test set is paraphrased. Validation saturates at 100%, so every selection above was decided by a tie-break rule rather than by evidence. Architecture choices were also made after test results had been seen: RRF was introduced after the v1 test, and the attribution design (N1/N2) came after the v2 test.

---

## 7. Historical result map

"Reproduced" means the per-query correctness vectors were regenerated during this audit using `audit/audit_reproduce_history.py`.

| Result file(s) | Date | Produced by | Inputs / config | Reproduced | Status |
|---|---|---|---|---|---|
| `retrieval_comparison_OLD_100pct.csv`, `ablation_OLD_100pct.csv` | 08-04 | `src/retrieval_eval_legacy.py` (renamed by hand afterwards) [I] | 70/15/15 split; **cached 108-doc embeddings** from an earlier corpus version | **Not possible** (that corpus no longer exists) | Obsolete; invalid |
| `source_date_audit.csv` | 08-06 | `src/audit_source_dates.py` | corpus | n/a | Valid descriptive |
| `crfs_tuning_report.csv`, `frozen_config.json` | 08-06 | `src/tune_crfs_weights.py` | v1 dev-train/val | grid saturation confirmed from file | Historical; shows the tie-break artefact |
| `retrieval_comparison.csv`, `ablation.csv`, `final_test_confidence_intervals.csv`, `significance_tests.csv`, `per_query_predictions.csv` | 08-08 12:06 | `src/retrieval_eval_final_L2.py` (or `retrieval_eval_final.py`; same outputs, identical linear code) [I] | v1 test + `split_dev.csv` + `frozen_config.json` | **Yes, 0 diffs** (5 main methods) | Historical, valid numbers |
| `rrf_shortlist_tuning_report.csv`, `frozen_config_rrf.json` | 08-08 12:07 | `src/tune_rrf_shortlist.py` | v1 dev-train/val | — | Historical |
| `retrieval_comparison_v2.csv`, `*_v2.csv` (CI, significance, subset, per-query) | 08-08 12:08 | `src/retrieval_eval_final_v2.py` | v2 test + `split_dev.csv` (58/82 overlap) + v1 weights + shortlist 15 | **Yes, 0 diffs** (6 methods) | Historical, leaky |
| `latency_all_methods_*.csv`, `latency_method_comparison.csv` | 08-08 19:29 | `src/latency_benchmark.py` | v2 config | not re-run | Superseded |
| `latency_raw.csv`, `latency_summary.csv` | 08-08 19:22 | an earlier `latency_benchmark.py` (now commented out) | **simulated** `time.sleep` placeholder | — | Invalid as a measurement |
| `igft_*.csv`, `triage_*.csv` | 08-08 | `igft_dialogue_sim.py`, `triage_classifier_train.py` | full dataset | — | Out of scope |
| `figs/*.pdf` | 08-08 19:10 | `src/make_plots.py` | **v2** CSVs + v1 `ablation.csv` + simulated latency | — | Stale for a v3 paper |
| `corpus_dedup_report*` | 08-09 | `src/audit_corpus.py` | corpus | n/a | Valid (0 duplicates) |
| `paraphrase_coverage_v3.csv`, `dev_val_coverage_v3.csv` | 08-24 | v3 split scripts | — | n/a | Valid descriptive |
| `crfs_tuning_report_v3.csv`, `frozen_config_v3.json` | 08-24 | `src/tune_crfs_weights_v3.py` | v3 dev-train/val | saturation confirmed from file | Valid record; selection uninformative |
| `rrf_shortlist_diagnostic_v3.{csv,txt}` | 08-25 | `rrf_shortlist_diagnostic_v3.py` (root) | v3 dev-train/val | — | Valid diagnostic |
| `rrf_dev_val_rank_diagnostic_v3.csv`, `…distribution_v3.csv` | 08-26 | `rrf_rank_diagnostic_v3.py` (root) | v3 dev-train/val | — | Valid diagnostic |
| `rrf_shortlist_tuning_report_v3.csv`, `frozen_config_rrf_v3.json` | 08-26 | `src/tune_rrf_shortlist_v3.py` | v3 dev-train/val | — | Valid record; decided by one query |
| `retrieval_comparison_v3.csv`, `ablation_attribution_v3.csv`, `final_test_confidence_intervals_v3.csv`, `subset_breakdown_v3.csv`, `significance_tests_v3.csv`, `bootstrap_ci_v3.csv`, `per_query_predictions_v3.csv`, `per_disease_agreement_v3.csv` | 08-28 19:49 | `src/retrieval_eval_final_v3.py` | v3 test + `split_dev_v3.csv` + v3 configs | **Yes, 0 diffs** (all 8 conditions, at both the run date and today) | **Current reference numbers** (metric caveats in §11) |
| `recency_variance_check_v3.csv` | 08-28 19:45 | `src/quick_recency_variance_check.py` | v3 | — | Valid diagnostic |
| `master_verification_v3/*`, `combined_results.txt` | 09-23 | `master_results_verification_v3.py`, then `temp.py` | **dev-train (148) pool**, a different N1 | its values are reproduced exactly with the 148-row pool [V] | **Mis-specified**; its MISMATCH flags are explained |

---

## 8. Reproducibility

**Reproducible** [V]: v1, v2 and v3 test results, per query, on the current machine (torch 2.5.1, CUDA). Retrieval embeddings are recomputed on every run, so MedCPT GPU non-determinism did not flip any prediction.

**Not reproducible:** the v0 "100%" results, whose corpus version is gone and whose embedding cache is misaligned. The guideline corpus build is also not reproducible, because it depends on live web pages.

**Latent risks:** `recency_weight` depends on the current date. Results depend on which rows form the index pool; there is no fixed random seed for torch, but none is needed today. `retrieval_eval_legacy.py` silently loads a stale cache. Paths are relative to the working directory. The requirements file is out of date.

**Why the master verification differed:** the explanation is complete (§2.2). No unexplained discrepancy remains.

---

## 9. Confirmed vs uncertain facts

**Confirmed:**
- All stored v1/v2/v3 numbers regenerate exactly.
- The v2 test shares 58/82 rows with its index pool.
- The v3 test and dev pool are row-disjoint.
- Every test symptom is covered by the dev pool's union for the correct disease.
- v1 and v3 CRFS tuning saturated (72 and 64 ties).
- Shortlist 3 was chosen because of one validation query.
- Recency uses `now()`.
- The guideline text alone gives 7–26% accuracy; a KG-only lookup gives 95% on canonical queries.
- 8 paraphrased queries are corrupted.
- The figures are v2-era, and `accuracy_vs_latency.pdf` is missing.
- The bootstrap uses 2,000 resamples, not the 10,000 stated in the paper.
- The cached embeddings have 108 rows.

**Uncertain:**
- Whether `retrieval_eval_final.py` or `_L2.py` wrote the v1 files (the outputs are identical either way).
- Exactly how the v0 108-doc corpus differed from the current one.
- Whether the "four pre-registered comparisons" were fixed before the v3 run. The same four already appear in the v2 script, which supports pre-specification, but no timestamped plan exists.
- The provenance of the hard-coded Hybrid weights 0.7/0.3.
- Whether the bibliography entries are accurate. Some look doubtful and must be checked (e.g. `bora2024` is probably *Machine Learning and Knowledge Extraction* 6(4), not *Machines*).

---

## 10. Leakage and overlap risks

| # | Risk | Severity | Status |
|---|---|---|---|
| L1 | v2 test rows inside the index pool (58/82) | High | Historical; correctly disclosed; v2 must not be used as evidence |
| L2 | **Near-duplicate rows**: a test row is its disease's symptom set minus ≤1 symptom, and the dev pool covers 100% of test symptoms | High for interpreting canonical accuracy | Inherent to the dataset. It must be stated, and harder query conditions are needed (§14) |
| L3 | Labelled training rows are written into document text and the KG | Not leakage (it is training), but it makes the system a supervised classifier | Must be described honestly; needs a supervised-classifier baseline |
| L4 | The paraphrase dictionary is the test instrument | High if reused | No query-expansion or synonym resource may be derived from `paraphrase_dictionary.py` |
| L5 | Test rows reused across generations (v3∩v2 = 37, v3∩v1 = 31) and design decisions made after seeing test results | Medium | The v3 test is "seen"; new methods must not be selected on it |
| L6 | Validation is canonical-only while the test is 2/3 paraphrased | Medium (selection is blind to the target condition) | Fix in the new protocol |
| L7 | Disease names appear in document text | None (queries never contain disease names) | OK |
| L8 | Refitting the index on dev-train+dev-val after tuning | None (standard practice) | OK, but results are sensitive to it (§2.10) |
| L9 | Triage/IGFT: urgency is a disease-level label; cross-validation over near-duplicate rows | High, but out of scope | Exclude from the paper |

---

## 11. Methodological weaknesses and bugs

1. **Saturated, canonical-only validation** makes every tuned parameter a tie-break artefact (§6).
2. **MRR is truncated and not comparable across methods.** It ranks only the diseases present in the top-20 docs (linear methods) or top-*n* shortlist (RRF, where n=3 gives at most 3 diseases). Everything else scores RR=0 (`pipeline_common.py` L338, L443, L606). The proposed pipeline's MRR (0.813) is therefore structurally capped.
3. **Recency is a date-dependent class prior** (L239); undated=0.5 and the half-life are arbitrary. In the RRF path it changed 0/123 predictions (N2 = Proposed) [V].
4. **Paraphraser nested substitution** (`paraphrase_dictionary.py` L124-127, sequential `str.replace`, no word boundaries). Several "paraphrases" also keep the clinical term.
5. **KG exact substring matching.** Paraphrased queries retain only 31% of their symptom strings literally, and there are 2 substring collisions.
6. **Latency is dominated by implementation overhead** (recomputing a constant recency vector, per-document KG recomputation, pandas `.iloc` inside loops), not by algorithmic cost. MedCPT query encoding (~15–17 ms) dominates every dense method.
7. **Statistics:**
   - Wilson CIs and exact McNemar are implemented correctly [R].
   - The bootstrap uses iid query resampling with n=2000, while the paper says 10,000. It ignores disease clustering (3 queries per disease).
   - Recall-macro equals accuracy by construction, because the test is balanced.
   - Macro-precision/F1 over 3 queries per class are very noisy.
   - With a single split, results depend on the split (§2.10).
   - The canonical subset (n=41) has Wilson half-widths of about ±6–9 pp.
8. **Aggregation bias.** The top-20 rank-discounted sum favours diseases with more passages (1–5 per disease).
9. **"Hybrid" 0.7/0.3 weights** are hard-coded and never tuned or justified.
10. **BM25 tokenisation** keeps punctuation attached to tokens.
11. **Documentation drift.** The paper describes figures, a bootstrap count and a latency rationale that the code does not support.

---

## 12. The current paper against the evidence

**Still valid (numbers reproduce):**
- Dataset and corpus statistics (304 rows, 41 conditions, 104 passages, 73 dated).
- The v2 leakage disclosure (58/82).
- The v3 construction description.
- Table I accuracy/F1/MRR values, the Table II attribution accuracies, and the Table III subset accuracies (as numbers).
- The McNemar counts and p-values.
- The observation that N2 = Proposed (the recency tie-break is inert).
- The exclusion of 8 diseases from dev-val.

**Not supported or wrong:**
- The weight collapse "caused by leakage" and "does not replicate once the leak is removed". Both weightings are tie-break artefacts of a saturated validation set, and the v1 search never touched the leaky data.
- "Every signal receives meaningful weight" (this is a centroid of 64 ties).
- "Leak-free", when used without qualification. The v3 split is row-disjoint but semantically overlapping.
- Latency attributed to candidate-pool architecture, and the "3.4×" figure (implementation overhead).
- "10,000 bootstrap resamples" (the code uses 2,000).
- The figure captions (the figures show v2 data), and the missing `accuracy_vs_latency.pdf`.
- MRR comparisons across methods (truncated rankings).
- "Pool size, not fusion rule" as a headline. It rests on one split, one pool size chosen by one validation query, and a gap that shrinks to non-significance (p=0.11) when only the index pool changes. It is a hypothesis, not a finding.
- The framing as *guideline evidence retrieval* (passages carry almost no signal).
- The claim that recency is a meaningful signal.

**Structural issue:** the paper is framed around correcting its own earlier (unpublished) results. For a first publication, this adds length without adding a scientific contribution.

---

## 13. Reference papers: conventions to follow

The 14 papers under `reference_papers/` are mostly IEEE-conference, 5–6 page medical RAG and retrieval studies. What the stronger ones have in common:

- **Staged or ablation tables.** Components are enabled cumulatively (e.g. S0→S4 in *Intelligent Biomedical Retrieval Pipeline*; RAG/KG/re-rank/CoT columns in the cardiovascular KG-RAG paper).
- **A query-type breakdown table.** Short/general vs long/detailed queries (*MPNet+BM25L*); symptom/treatment/entity categories (*Hybrid Semantic Search*). This maps directly onto our canonical vs lay-paraphrase conditions.
- **A latency breakdown per stage** (*Hybrid Semantic Search*, Table II), rather than a single end-to-end number.
- **Qualitative examples** of synonym handling and failures.
- **A limitations section split into methodological limits, deployment limits and future work** (*Intelligent Biomedical Retrieval Pipeline*).
- **Modest claims.** For example, "suggests lower hallucination risk through better evidence access rather than providing direct proof".

Evaluations are small (50–200 queries), and most papers report no significance tests. Only one uses a paired t-test, one uses Mann-Whitney, and one uses a time-isolated benchmark. Several are weak: *SymptoTrackAI* contains figures copied from a fraud-detection paper, and the autism paper reports 98% on 50 questions. A study with **repeated splits, paired tests and honest benchmark characterisation would stand out in this genre** without needing a novel architecture.

---

## 14. Candidate research questions (ranked)

Each question below is supportable by this repository with modest additional work.

**RQ-A (recommended core): lay-language robustness of symptom-to-condition ranking.**
*How much do lexical, dense biomedical and symptom-graph signals each contribute when patients describe symptoms in lay language, and which way of combining them is robust to that vocabulary mismatch?*
- Evidence already points at a real, measurable effect. Every method loses 15–35 pp from canonical to paraphrased queries. The exact-match KG is the strongest signal but the most brittle. Dense retrieval is the most paraphrase-robust single signal.
- The canonical condition becomes an honestly labelled "near-lookup ceiling".

**RQ-B (a simple method contribution within RQ-A): semantic symptom linking for the KG.**
Replace substring matching with embedding-based linking. Each comma-separated query phrase is linked to its nearest KG symptom node (e.g. with MedCPT) above a threshold chosen on validation. This addresses the largest measured failure (31% literal symptom survival). It is a one-component change, easy to ablate, and not complexity for its own sake.

**RQ-C (secondary analysis): fusion rule vs candidate-pool size.** Keep the N1/N2 attribution idea, but test it as a **curve over pool sizes** across repeated splits instead of one tuned point. It is likely to become a paragraph or a figure, not the headline.

**RQ-D (discussion point, not a headline): evaluation lessons.** Row-disjoint is not the same as held-out for template-generated symptom datasets, and a saturated validation set turns tie-break rules into "findings". This is worth one paragraph in the discussion.

**Drop:** recency as a signal (report its removal in one sentence) and the framing as guideline-passage retrieval (unless the evidence-return role is evaluated separately).

---

## 15. Recommended experiment plan

Do not start until the open decisions in §17 are answered.

**Phase 0: checkpoint.** Run `git init`. Add a `.gitignore` for `venv/`, `__pycache__/` and optionally `data/raw/Suicide_Detection.csv` (167 MB, out of scope). Make an initial commit of the current state and tag it `pre-rebuild-audit`.

**Phase 1: fix the measurement instrument** (new code; legacy code left untouched).
- Paraphraser: single-pass, word-boundary regex substitution (no nesting). Regenerate all paraphrased queries.
- Remove recency, or freeze its reference date if it is kept for a legacy comparison.
- Score **every disease** so that MRR, Hit@3 and Hit@5 are computed on full rankings.
- Efficient implementation: precompute per-disease KG and constant vectors, avoid `.iloc` in hot loops, and time with GPU synchronisation.
- **Regression test:** a "legacy mode" of the new code must reproduce the v3 stored vectors (0 diffs). This anchors the rebuild.

**Phase 2: protocol.**
- **Repeated evaluation instead of one split.** Leave-one-row-out per disease: every row is a test query once, and the index is built from the other rows of all diseases. Alternatively use R≥20 seeded repeats of 1-row-per-disease test splits. Report mean and CI over queries/folds.
- **Controlled query conditions** built from the held-out row: (a) canonical; (b) lay-paraphrased with the **same row** (removes the row confound); (c) *partial reports* with k∈{2,3,4} sampled symptoms (realistic incomplete reports, which gives a difficulty gradient where methods separate).
- **Validation without touching the test instrument.** Tune only on nested inner folds. Build validation paraphrases from a **separate lexicon (A)** and keep the **test lexicon (B)** frozen and independent. Keep tuned parameters to a minimum: parameter-free RRF, one fusion weight at most, a KG-linking threshold. Report sensitivity curves instead of single tuned optima.
- Pre-specify comparisons and the analysis plan in a committed file **before** the final run. A git timestamp gives a verifiable preregistration.

**Phase 3: systems.**
- Baselines: BM25, Dense (MedCPT), linear hybrid, RRF, KG-exact, **KG-semantic-linked**, and a **supervised symptom classifier** (TF-IDF + logistic regression or multinomial NB) as a sanity reference.
- Fusion variants: linear vs RRF, crossed with KG-exact vs KG-linked, over pool sizes {3, 5, 10, 20, all}.

**Phase 4: statistics.** Exact McNemar for key pairs; a **disease-clustered bootstrap** for accuracy differences; Holm correction over the pre-specified family; per-condition reporting.

**Phase 5: latency.** Stage-wise breakdown (encode, BM25, dense, KG, fuse, aggregate) on the RTX 3050 and on CPU; warm-up; ≥5 repeats; median and p95.

**Phase 6: outputs.** Every table and figure regenerated by scripts from final CSVs, with one command documented in the README.

**Do not run** (unnecessary or harmful):
- LLM answer generation or RAG answer evaluation.
- Triage classifier, IGFT dialogue or safety set in the paper.
- Fine-tuning MedCPT (around 150 near-duplicate rows).
- Tuning RRF k.
- Finer CRFS weight grids on saturated validation.
- Recency variants.
- A "v4" single-split rerun.
- Any new configuration evaluated on `split_test_final_v3.csv` for selection.
- A cross-encoder reranker (optional at most; low expected value, since the documents are symptom profiles).

---

## 16. Proposed final repository structure

This is a proposal only; nothing has been moved. Moves must keep the imports noted in §3 working.

```
health_paper/
  README.md                      # how to reproduce, one command per table/figure
  RESEARCH_AUDIT.md
  requirements.txt               # corrected versions (+statsmodels)
  data/
    raw/                         # Kaggle files (Suicide_Detection.csv -> legacy/data or gitignored)
    symptom_disease.csv
    guideline_corpus.jsonl       # frozen artefact
    lexicons/                    # paraphrase lexicon A (dev) and B (test)
  src/hsr/                       # final package: data.py, paraphrase.py, retrievers.py, kg.py, fusion.py, metrics.py, stats.py, latency.py
  experiments/                   # 01_make_folds.py, 02_run_eval.py, 03_stats.py, 04_latency.py, 05_figures.py
  configs/                       # frozen protocol + preregistered comparisons
  results/final/
  paper/                         # main.tex, figs/, refs.bib
  audit/                         # this audit's scripts + logs
  reference_papers/
  legacy/
    scripts/{v0,v1,v2,v3,diagnostics}/   # old eval/tune/split scripts, master verification, temp.py, pre_phase3_audit.py, files_structure.py
    src_common/                  # current src/pipeline_common.py + legacy/pipeline_common.py (imports preserved)
    results/{v0_legacy,v1,v2,v3,master_verification_v3}/
    data_splits/{v1,v2,v3}/
    old_papers/cureent_paper.tex, old_figs/
    old_configs/frozen_config*.json
    models/                      # stale 108-row embedding caches
    out_of_scope/                # triage, IGFT, safety, simulated latency
```

---

## 17. Open decisions for you

1. **Research framing:** go with RQ-A + RQ-B (lay-language robustness with semantic KG linking; recommended), or keep the fusion-rule/pool-size story as the headline?
2. **An independent test paraphrase lexicon (B):** can a co-author write it without seeing results? The alternatives are an LLM with a frozen, disclosed prompt, or reusing the existing dictionary as B, which would then need a new lexicon A for development.
3. **Checkpoint:** may I run `git init` and make the snapshot commit (excluding `venv/` and, if you agree, the 167 MB suicide dataset)?
4. **Scope:** confirm that triage, IGFT and safety material stays out of the 6-page paper.

---

## Appendix: quick answers to the 36 audit questions

| # | Answer | § |
|---|---|---|
| 1 | Ranking 41 diseases from a symptom list, via passage retrieval with injected symptom profiles, KG, fusion and aggregation | 1 |
| 2 | As drafted: linear vs RRF fusion attribution. Supportable: lay-language robustness plus semantic KG linking | 12, 14 |
| 3 | Kaggle disease-symptom dataset (304 unique rows) + 104 MedlinePlus/CDC passages | 4 |
| 4 | Deduplicated Kaggle rows; scraped passages; urgency joined from a manual map | 3, 4 |
| 5–6 | v1/v2/v3 splits; canonical = `split_test_final_v3.csv` + `split_dev_v3.csv` (+ dev-train/val v3) | 4.3 |
| 7 | "The patient reports a, b, c." with lay rewrites from a fixed dictionary for paraphrased queries | 4.3 |
| 8–10 | Passage scoring → disease aggregation; MedCPT dual encoder; `rank_bm25` Okapi on whitespace tokens | 5.2 |
| 11 | Disease–symptom graph from pool rows; IDF-weighted exact-substring overlap; disease-level score | 5.2 |
| 12 | CRFS weighted sum; RRF(k=60) → top-n → KG sort → aggregation; N1/N2 attribution controls | 5.2 |
| 13–14 | Grid search / small grid on dev-val (canonical only, saturated); tie-break rules decide | 6 |
| 15 | 123 queries × 8 conditions: accuracy, macro P/R/F1, truncated MRR, latency, subsets, McNemar, bootstrap | 7 |
| 16–17 | Full mapping table | 7 |
| 18–19 | v2 overlap (historical); near-duplicate structure; instrument reuse; test reuse | 10 |
| 20 | 4920→304 exact deduplication; no duplicate sets; but all rows are near-duplicates within their disease | 4.1 |
| 21 | None, once the index pool is matched; the master verification used a different pool and N1 | 2, 8 |
| 22–23 | Truncated MRR, date-dependent recency, paraphraser bug, KG substring matching; accuracy/Wilson/McNemar correct | 11 |
| 24 | Wilson and McNemar correct; bootstrap 2k (not 10k), iid rather than clustered; single split | 11.7 |
| 25 | Arithmetic correct (reproduced); canonical vs paraphrase is confounded by row; n=41 subset is wide | 4.3, 11 |
| 26 | Computed correctly *within* each method's truncated ranking; not comparable across methods | 11.2 |
| 27 | Query-level macro over diseases; recall-macro equals accuracy (balanced) | 11.7 |
| 28 | No test leakage into selection; selection is uninformative (saturation, one query) | 6 |
| 29 | No: `now()`-dependent, a per-disease prior, arbitrary constants; inert as a tie-break | 11.3 |
| 30 | v3 per-query vectors, tuning reports (as evidence of saturation), v2 disclosure, diagnostics | 7 |
| 31 | v0 OLD_100pct, simulated latency, stale caches, v2-era figures, master verification outputs | 7 |
| 32–33 | See lists | 12 |
| 34 | RQ-A/RQ-B, with RQ-C as a secondary analysis | 14 |
| 35–36 | See plan and the do-not-run list | 15 |
