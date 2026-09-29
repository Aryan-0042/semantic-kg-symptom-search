"""
tune_crfs_weights.py

Selects the CRFS fusion weights (w_bm25, w_dense, w_kg, w_recency) using
ONLY data/split_dev_train.csv (to build BM25/dense index/KG/retrieval
text) and data/split_dev_val.csv (to build the evaluation queries scored
during the search). The frozen test set (data/split_test_final.csv) is
never loaded by this script -- it is not imported anywhere below.

SEARCH SPACE:
  Weights are constrained to a 4-simplex (w_bm25 + w_dense + w_kg +
  w_recency == 1, all >= 0), searched on a step-0.1 grid over
  (w_bm25, w_dense, w_kg), with w_recency = 1 - sum(others). Any
  combination where w_recency would be negative is skipped. This is
  exactly the set of weak compositions of 10 tenths into 4 non-negative
  parts, C(13,3) = 286 combinations -- enumerable in full given the
  small validation set, so no sampling scheme needs to be justified.
  (If you tighten the step to 0.05 the grid grows to C(23,3) = 1771
  combinations, which is also fine to run but takes proportionally
  longer -- STEP is defined once below if you want to change it.)

CONSTRAINTS:
  - All weights >= 0 (no negative contribution from any signal)
  - Weights sum to 1.0 (CRFS score stays in a comparable [0,1]-ish range)

OBJECTIVE:
  Macro-F1 on the dev-val queries (n = number of diseases represented
  in dev-val, per make_dev_train_val_split.py -- typically close to 41
  but may be slightly fewer for the smallest classes; see that script's
  printed warning). Macro-F1 rather than accuracy because dev-val is by
  construction close to 1-per-class and we want a metric that penalizes
  systematically ignoring any class, not just overall hit rate.

OUTPUT:
  results/crfs_tuning_report.csv  -- every combination tried, with its
      dev-val macro-F1 (and accuracy/MRR for reference)
  results/frozen_config.json      -- the selected winning configuration,
      the objective value that won, and a timestamp -- this is the file
      retrieval_eval_final.py reads. Do not hand-edit this file after
      generating it without re-running the full search.
"""
import json
import time
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from legacy.pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_queries_from_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    precompute_raw_scores,
    fuse_and_evaluate,
)

STEP = 0.1


def generate_weight_grid(step=STEP):
    grid = []
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    for w_bm25 in vals:
        for w_dense in vals:
            for w_kg in vals:
                w_recency = round(1.0 - w_bm25 - w_dense - w_kg, 2)
                if w_recency < 0:
                    continue
                if w_recency > 1.0:
                    continue
                grid.append((float(w_bm25), float(w_dense), float(w_kg), float(w_recency)))
    return grid


def main():
    print("Loading dev-train / dev-val splits and corpus...")
    corpus_df = load_guideline_corpus()
    dev_train = load_symptom_split("data/split_dev_train.csv")
    dev_val = load_symptom_split("data/split_dev_val.csv")

    val_queries, val_labels = build_queries_from_split(dev_val)
    print(f"Validation queries: {len(val_queries)} across {len(set(val_labels))} diseases")

    print("\nBuilding retrieval-aware document text from dev-train ONLY...")
    retrieval_texts = build_retrieval_text(corpus_df, dev_train)

    print("Building BM25 index...")
    bm25 = BM25Okapi([t.lower().split() for t in retrieval_texts])

    print("Loading MedCPT and encoding corpus (dev-train-conditioned text)...")
    encoder = MedCPTEncoder()
    doc_embeddings = encoder.encode_docs(retrieval_texts, batch_size=16)

    print("Building Knowledge Graph from dev-train ONLY...")
    graph = build_kg(corpus_df, dev_train)

    print("\nWarming up encoder/BM25 before timed calls...")
    warmup(bm25, encoder)

    print("\nPrecomputing raw BM25 / MedCPT / KG / recency scores for all "
          f"{len(val_queries)} validation queries (done ONCE, reused across "
          "every weight combination)...")
    cache = precompute_raw_scores(val_queries, val_labels, corpus_df, bm25, encoder, doc_embeddings, graph)

    grid = generate_weight_grid()
    print(f"\nSweeping {len(grid)} weight combinations over cached scores (fusion-only, fast)...\n")

    rows = []
    t0 = time.time()
    for i, weights in enumerate(grid, start=1):
        result, _ = fuse_and_evaluate(cache, corpus_df, weights)
        rows.append({
            "w_bm25": weights[0], "w_dense": weights[1], "w_kg": weights[2], "w_recency": weights[3],
            "val_accuracy": result["Accuracy"], "val_f1": result["F1"], "val_mrr": result["MRR"],
        })
        if i % 50 == 0 or i == len(grid):
            print(f"  [{i}/{len(grid)}] elapsed {time.time()-t0:.2f}s")

    report_df = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)

    import os
    os.makedirs("results", exist_ok=True)
    report_df.to_csv("results/crfs_tuning_report.csv", index=False)

    # in tune_crfs_weights.py, replace the naive .iloc[0] selection with:
    best_f1 = report_df["val_f1"].max()
    tied = report_df[report_df["val_f1"] == best_f1].copy()
    
    if len(tied) > 1:
        print(f"\n{len(tied)} configurations tied at dev-val F1={best_f1}. "
              f"Breaking tie using dev-val MRR (continuous, less prone to "
              f"saturation), then preferring the simplest (lowest w_recency) "
              f"configuration among any remaining ties.")
        tied = tied.sort_values(["val_mrr", "w_recency"], ascending=[False, True])
    
    best = tied.iloc[0]
    frozen_config = {
        "selected_weights": {
            "w_bm25": float(best["w_bm25"]),
            "w_dense": float(best["w_dense"]),
            "w_kg": float(best["w_kg"]),
            "w_recency": float(best["w_recency"]),
        },
        "objective": "macro_F1_on_dev_val",
        "dev_val_f1": float(best["val_f1"]),
        "dev_val_accuracy": float(best["val_accuracy"]),
        "dev_val_mrr": float(best["val_mrr"]),
        "n_combinations_searched": len(grid),
        "search_step": STEP,
        "n_val_queries": len(val_queries),
        "selected_at": pd.Timestamp.now().isoformat(),
        "note": "Selected using data/split_dev_train.csv and data/split_dev_val.csv ONLY. "
                "data/split_test_final.csv was never loaded by this script.",
    }
    with open("results/frozen_config.json", "w") as f:
        json.dump(frozen_config, f, indent=2)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best weights (w_bm25, w_dense, w_kg, w_recency): "
          f"({best['w_bm25']}, {best['w_dense']}, {best['w_kg']}, {best['w_recency']})")
    print(f"dev-val macro-F1  : {best['val_f1']}")
    print(f"dev-val accuracy  : {best['val_accuracy']}")
    print(f"dev-val MRR       : {best['val_mrr']}")
    print("\nFull grid  -> results/crfs_tuning_report.csv")
    print("Frozen config -> results/frozen_config.json")
    print("\nThese weights are now FROZEN. Do not re-run this script and swap")
    print("the config after seeing final test results.")


if __name__ == "__main__":
    main()
