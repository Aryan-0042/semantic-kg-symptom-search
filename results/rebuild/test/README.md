# Final evaluation outputs (protocol v1.1)

## Pre-registered: written once by `experiments/03_final_eval.py` (commit `33bd63d`)

| File | Contents |
|---|---|
| `primary_tests.csv` | H1–H4: ΔMRR with disease-clustered 95% CI, sign-flip p, Holm-adjusted p, ΔAcc@1 |
| `per_query.csv` | rank, RR, Hit@1/3/5 and tie diagnostics for every query × system (1,590 queries × 12 systems) |
| `summary_{mrr,acc1,hit3,hit5}_by_{group,condition}.csv` | means with disease-clustered 95% CIs |
| `linking_accuracy.csv` | per-clause recall and precision of exact, substring and semantic linking |
| `pool_analysis.csv` | pool size × fusion rule × KG signal (secondary analysis) |
| `tie_diagnostics.csv`, `truncation_rates.csv` | diagnostics |
| `diagnostic_lexicon_A_vs_B.csv` | lay groups with the development lexicon A on the same test rows (diagnostic only) |
| `provenance.json`, `final_eval.log` | commit, parameters, lexicon hashes, counts; run log |

## Added after the final evaluation

| File | Status | Script |
|---|---|---|
| `sensitivity_no_disease_names.csv` | **POST-HOC, not pre-registered** | `experiments/04_posthoc_sensitivity.py` |
| `sensitivity_no_disease_names_tests.csv` | **POST-HOC, not pre-registered** | same |
| `sensitivity_excluded_queries.csv` | the 39 queries excluded by the post-hoc check | same |
| `latency_diagnostics.csv`, `latency_environment.json` | pre-specified in PROTOCOL.md §10.10, but **implemented and run after** the final evaluation | `experiments/05_latency.py` |

The post-hoc check recomputes results from the committed `per_query.csv`, without the 39 lexicon B lay queries whose wording names a disease ("acne", "jaundiced"). The latency diagnostic uses query texts only, never labels. Neither changes any pre-registered output. See `results/rebuild/FINAL_PAPER_DISCLOSURES.md`.
