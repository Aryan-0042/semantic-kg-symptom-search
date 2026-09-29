# tune_crfs_weights_v3.py
"""
Selects CRFS fusion weights using ONLY data/split_dev_train_v3.csv and
data/split_dev_val_v3.csv (the corrected, disjoint-by-construction dev
pool). data/split_test_final_v3.csv is never loaded here.

Identical methodology to the v2 tuning script: 286-combination grid,
step 0.1, 4-simplex constraint. Tie-break: max F1 -> max MRR -> if still
tied, average the surviving tied configurations (a valid point on the
simplex, not an arbitrary first-match pick) -- this is the corrected
tiebreak logic already validated in the v2 round after the original
tuning saturated.
"""
import json
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from pipeline_common import (
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
                if w_recency < 0 or w_recency > 1.0:
                    continue
                grid.append((float(w_bm25), float(w_dense), float(w_kg), float(w_recency)))
    return grid


def select_frozen_weights(report_df):
    """Three-stage deterministic tiebreak: F1 -> MRR -> centroid of any
    remaining tied configurations (not an arbitrary first match)."""
    stage1 = report_df[report_df["val_f1"] == report_df["val_f1"].max()]
    stage2 = stage1[stage1["val_mrr"] == stage1["val_mrr"].max()]

    print(f"Tiebreak trace: {len(report_df)} -> F1 tie:{len(stage1)} -> MRR tie:{len(stage2)}")

    if len(stage2) == 1:
        row = stage2.iloc[0]
        method = "unique after 2-stage tiebreak"
    else:
        row = stage2[["w_bm25", "w_dense", "w_kg", "w_recency"]].mean()
        row = row / row.sum()  # defensive renormalization against float drift
        method = f"centroid of {len(stage2)} remaining tied configurations"

    print(f"Final weights selected via: {method}")
    return (float(row["w_bm25"]), float(row["w_dense"]), float(row["w_kg"]), float(row["w_recency"])), method, len(stage2)


def main():
    print("Loading v3 dev-train / dev-val splits and corpus...")
    corpus_df = load_guideline_corpus()
    dev_train = load_symptom_split("data/split_dev_train_v3.csv")
    dev_val = load_symptom_split("data/split_dev_val_v3.csv")

    val_queries, val_labels = build_queries_from_split(dev_val)
    print(f"Validation queries: {len(val_queries)} across {len(set(val_labels))} diseases")
    print("NOTE: expect 33 diseases represented here, not 41 -- 8 diseases were")
    print("excluded from dev-val by the guard logic during the v3 split repair.")
    print("This is expected and already disclosed; see results/dev_val_coverage_v3.csv")

    print("\nBuilding retrieval-aware document text from dev-train-v3 ONLY...")
    retrieval_texts = build_retrieval_text(corpus_df, dev_train)

    print("Building BM25 index...")
    bm25 = BM25Okapi([t.lower().split() for t in retrieval_texts])

    print("Loading MedCPT and encoding corpus...")
    encoder = MedCPTEncoder()
    doc_embeddings = encoder.encode_docs(retrieval_texts, batch_size=16)

    print("Building Knowledge Graph from dev-train-v3 ONLY...")
    graph = build_kg(corpus_df, dev_train)

    print("\nWarming up...")
    warmup(bm25, encoder)

    print("\nPrecomputing raw scores for all validation queries (once, reused across grid)...")
    cache = precompute_raw_scores(val_queries, val_labels, corpus_df, bm25, encoder, doc_embeddings, graph)

    grid = generate_weight_grid()
    print(f"\nSweeping {len(grid)} weight combinations (fusion-only, fast)...\n")

    rows = []
    for weights in grid:
        result, _ = fuse_and_evaluate(cache, corpus_df, weights)
        rows.append({
            "w_bm25": weights[0], "w_dense": weights[1], "w_kg": weights[2], "w_recency": weights[3],
            "val_accuracy": result["Accuracy"], "val_f1": result["F1"], "val_mrr": result["MRR"],
        })

    report_df = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    import os
    os.makedirs("results", exist_ok=True)
    report_df.to_csv("results/crfs_tuning_report_v3.csv", index=False)

    weights, tiebreak_method, n_tied_at_final_stage = select_frozen_weights(report_df)

    frozen_config = {
        "selected_weights": {
            "w_bm25": weights[0], "w_dense": weights[1], "w_kg": weights[2], "w_recency": weights[3],
        },
        "objective": "macro_F1_on_dev_val, tiebreak MRR, tiebreak centroid-of-ties",
        "tiebreak_method": tiebreak_method,
        "n_combinations_searched": len(grid),
        "n_val_queries": len(val_queries),
        "n_diseases_in_val": len(set(val_labels)),
        "diseases_excluded_from_val": 8,
        "selected_at": pd.Timestamp.now().isoformat(),
        "note": "Selected using data/split_dev_train_v3.csv and data/split_dev_val_v3.csv "
                "ONLY. data/split_test_final_v3.csv was never loaded by this script. "
                "Dev pool is disjoint-by-construction from the v3 test set (exact complement).",
    }
    with open("results/frozen_config_v3.json", "w") as f:
        json.dump(frozen_config, f, indent=2)

    print(f"\nFrozen weights: {weights}")
    print("Full grid -> results/crfs_tuning_report_v3.csv")
    print("Frozen config -> results/frozen_config_v3.json")
    print("\nThese weights are now FROZEN.")


if __name__ == "__main__":
    main()