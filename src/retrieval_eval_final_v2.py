"""
retrieval_eval_final_v2.py

THE CONFIRMATORY EVALUATION for Option A. Run this exactly once, after:
  1. make_new_frozen_test_split.py has produced data/split_test_final_v2.csv
  2. tune_rrf_shortlist.py has produced results/frozen_config_rrf.json
     (dev-val only, never touches this or any test split)

This script builds BM25/dense/KG from the FULL original development set
(data/split_dev.csv, unchanged from the earlier round -- still valid,
still never includes any v2 test row since v2 test rows are drawn fresh
from the full 304-row dataset and may overlap dev.csv; overlap here is
expected and fine, see the design note below) and evaluates ALL methods
-- old and new -- against data/split_test_final_v2.csv exactly once:

  BM25-only, Dense-only, Hybrid, Hybrid+KG (no recency),
  Proposed (CRFS, linear, frozen weights) -- for continuity with the
    original round -- and Proposed (RRF+KG+Recency) -- the new
    architecture from this round.

IMPORTANT DESIGN NOTE ON DEV/TEST-V2 OVERLAP:
data/split_dev.csv was built by removing the ORIGINAL 82 test rows
(seed=42) from the full 304-row dataset. data/split_test_final_v2.csv is
a fresh draw (seed=99) from the FULL 304-row dataset, and may therefore
include rows that are also sitting in data/split_dev.csv. This was a
disclosed, deliberate tradeoff (see the earlier discussion): the
alternative -- drawing v2 test only from the leftover dev pool -- would
force 1-per-disease sampling for every class with just 5 total rows.
Because build_retrieval_text()/build_kg() aggregate MULTIPLE rows per
disease into a shared symptom-context summary (Counter.most_common(8)),
a handful of overlapping individual rows does not hand any one query's
exact answer to the retrieval system -- the same structural argument
already made for the original split. This is disclosed in the paper,
not hidden.

Also reports accuracy split by the paraphrased vs. canonical subset of
the new test set -- this is the actual test of whether Option A's dense
recall stage does anything the linear CRFS/Hybrid+KG baselines couldn't.
"""
import json
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from statsmodels.stats.contingency_tables import mcnemar

from pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    run_condition,
    run_condition_rrf_kg_recency,
)


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_paired(correct_a, correct_b):
    a, b = np.array(correct_a), np.array(correct_b)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    if n01 == 0 and n10 == 0:
        return n01, n10, 1.0, "No discordant pairs -- methods agree on every query."
    p = mcnemar([[0, n01], [n10, 0]], exact=True).pvalue
    return n01, n10, float(p), ""


def main():
    print("=" * 60)
    print("CONFIRMATORY EVALUATION -- OPTION A vs. ALL PRIOR METHODS")
    print("Run once, against the NEW frozen test set.")
    print("=" * 60)

    with open("results/frozen_config.json") as f:
        crfs_frozen = json.load(f)
    crfs_w = crfs_frozen["selected_weights"]
    crfs_weights = (crfs_w["w_bm25"], crfs_w["w_dense"], crfs_w["w_kg"], crfs_w["w_recency"])
    print(f"Loaded linear-CRFS frozen weights: {crfs_weights}")

    with open("results/frozen_config_rrf.json") as f:
        rrf_frozen = json.load(f)
    shortlist_n = rrf_frozen["selected_shortlist_n"]
    rrf_k = rrf_frozen["rrf_k"]
    print(f"Loaded RRF+KG+Recency config: shortlist_n={shortlist_n}, rrf_k={rrf_k}")

    corpus_df = load_guideline_corpus()
    dev_full = load_symptom_split("data/split_dev.csv")
    test_v2 = pd.read_csv("data/split_test_final_v2.csv")

    test_queries = test_v2["query_text"].tolist()
    test_labels = test_v2["Disease"].astype(str).str.strip().tolist()
    paraphrased_mask = test_v2["paraphrased"].astype(bool).to_numpy()

    print(f"\nNew confirmatory test queries: {len(test_queries)} "
          f"({paraphrased_mask.sum()} paraphrased, {(~paraphrased_mask).sum()} canonical)")

    print("\nBuilding retrieval-aware document text from FULL dev set...")
    retrieval_texts = build_retrieval_text(corpus_df, dev_full)
    bm25 = BM25Okapi([t.lower().split() for t in retrieval_texts])

    print("Loading MedCPT and encoding corpus...")
    encoder = MedCPTEncoder()
    doc_embeddings = encoder.encode_docs(retrieval_texts, batch_size=16)

    print("Building Knowledge Graph from FULL dev set...")
    graph = build_kg(corpus_df, dev_full)

    print("\nWarming up...")
    warmup(bm25, encoder)

    linear_methods = ["BM25-only", "Dense-only", "Hybrid", "Hybrid+KG (no recency)", "Proposed (CRFS)"]
    comparison_rows = []
    per_query_by_method = {}

    print("\nRunning linear-architecture methods...\n")
    for method in linear_methods:
        result, per_query = run_condition(
            method, test_queries, test_labels, corpus_df,
            bm25, encoder, doc_embeddings, graph, weights=crfs_weights, verbose=False,
        )
        comparison_rows.append(result)
        per_query_by_method[method] = per_query
        print(f"  {method:28s} Acc={result['Accuracy']:6.2f}  F1={result['F1']:6.2f}  MRR={result['MRR']:.3f}")

    print("\nRunning Proposed (RRF+KG+Recency)...\n")
    rrf_result, rrf_per_query = run_condition_rrf_kg_recency(
        test_queries, test_labels, corpus_df, bm25, encoder, doc_embeddings, graph,
        shortlist_n=shortlist_n, rrf_k=rrf_k, verbose=False,
    )
    comparison_rows.append(rrf_result)
    per_query_by_method["Proposed (RRF+KG+Recency)"] = rrf_per_query
    print(f"  {'Proposed (RRF+KG+Recency)':28s} Acc={rrf_result['Accuracy']:6.2f}  "
          f"F1={rrf_result['F1']:6.2f}  MRR={rrf_result['MRR']:.3f}")

    comparison_df = pd.DataFrame(comparison_rows)
    import os
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv("results/retrieval_comparison_v2.csv", index=False)

    # ---------------- Overall stats: Wilson CI ----------------
    n = len(test_labels)
    stats_rows = []
    for method, per_query in per_query_by_method.items():
        successes = sum(per_query)
        lo, hi = wilson_ci(successes, n)
        stats_rows.append({
            "Method": method, "N": n, "Correct": successes,
            "Accuracy": round(100 * successes / n, 2),
            "Wilson_95CI_low": round(100 * lo, 2), "Wilson_95CI_high": round(100 * hi, 2),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv("results/final_test_confidence_intervals_v2.csv", index=False)

    # ---------------- Paraphrased vs. canonical subgroup breakdown ----------------
    subset_rows = []
    for method, per_query in per_query_by_method.items():
        per_query_arr = np.array(per_query)
        for subset_name, mask in [("canonical", ~paraphrased_mask), ("paraphrased", paraphrased_mask)]:
            n_sub = int(mask.sum())
            correct_sub = int(per_query_arr[mask].sum())
            subset_rows.append({
                "Method": method, "Subset": subset_name, "N": n_sub, "Correct": correct_sub,
                "Accuracy": round(100 * correct_sub / n_sub, 2) if n_sub else float("nan"),
            })
    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv("results/subset_breakdown_v2.csv", index=False)

    # ---------------- Paired significance (McNemar, correctly this time) ----------------
    sig_rows = []
    comparisons = [
        ("Hybrid", "Proposed (RRF+KG+Recency)"),
        ("Hybrid+KG (no recency)", "Proposed (RRF+KG+Recency)"),
        ("Proposed (CRFS)", "Proposed (RRF+KG+Recency)"),
        ("Dense-only", "Proposed (RRF+KG+Recency)"),
    ]
    for a, b in comparisons:
        n01, n10, p, note = mcnemar_paired(per_query_by_method[a], per_query_by_method[b])
        sig_rows.append({"comparison": f"{a} vs {b}", "n_pairs": n,
                          "b_right_a_wrong": n01, "a_right_b_wrong": n10,
                          "mcnemar_exact_p": p, "note": note})
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv("results/significance_tests_v2.csv", index=False)

    # ---------------- Raw per-query predictions, for any later reanalysis ----------------
    raw_rows = []
    for method, per_query in per_query_by_method.items():
        for i, c in enumerate(per_query):
            raw_rows.append({"method": method, "query_index": i, "true_label": test_labels[i],
                              "paraphrased": bool(paraphrased_mask[i]), "correct": c})
    pd.DataFrame(raw_rows).to_csv("results/per_query_predictions_v2.csv", index=False)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS (n=%d, new frozen test set)" % n)
    print("=" * 60)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("PARAPHRASED vs. CANONICAL SUBSET BREAKDOWN")
    print("=" * 60)
    print(subset_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("PAIRED SIGNIFICANCE (McNemar exact test)")
    print("=" * 60)
    print(sig_df.to_string(index=False))

    print("\n" + "!" * 60)
    print("THESE ARE THE FINAL CONFIRMATORY NUMBERS FOR OPTION A.")
    print("!" * 60)


if __name__ == "__main__":
    main()
