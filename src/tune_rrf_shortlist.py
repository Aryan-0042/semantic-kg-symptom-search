"""
tune_rrf_shortlist.py

Tunes the ONE real hyperparameter of the Option-A architecture --
shortlist_n, the number of RRF-recalled candidates handed to the KG
reranker -- using data/split_dev_train.csv (build index/KG) and
data/split_dev_val.csv (evaluate) ONLY. Reuses the same dev-train/dev-val
split already created for the linear-CRFS tuning round; this is a new
round of dev-only work, not a reopening of anything frozen.

rrf_k is NOT tuned -- fixed at the literature-standard 60 (see
pipeline_common.RRF_K_DEFAULT and its docstring). This is a deliberate
choice to keep the tunable search space as small as possible.

data/split_test_final.csv (the original frozen 82-query set) is never
loaded here. Nor is any new test split -- this script only ever touches
dev-train/dev-val.

OUTPUT: results/rrf_shortlist_tuning_report.csv (all candidates tried)
        results/frozen_config_rrf.json (the selected shortlist_n)
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
    run_condition_rrf_kg_recency,
    RRF_K_DEFAULT,
)

SHORTLIST_CANDIDATES = [10, 15, 20, 30, 50]


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

    print(f"\nTrying shortlist_n in {SHORTLIST_CANDIDATES} (rrf_k fixed at {RRF_K_DEFAULT})...\n")
    rows = []
    for n in SHORTLIST_CANDIDATES:
        result, _ = run_condition_rrf_kg_recency(
            val_queries, val_labels, corpus_df, bm25, encoder, doc_embeddings, graph,
            shortlist_n=n, rrf_k=RRF_K_DEFAULT, verbose=False,
        )
        rows.append({
            "shortlist_n": n, "val_accuracy": result["Accuracy"],
            "val_f1": result["F1"], "val_mrr": result["MRR"],
        })
        print(f"  shortlist_n={n:3d}  Acc={result['Accuracy']:6.2f}  "
              f"F1={result['F1']:6.2f}  MRR={result['MRR']:.3f}")

    report_df = pd.DataFrame(rows)
    import os
    os.makedirs("results", exist_ok=True)
    report_df.to_csv("results/rrf_shortlist_tuning_report.csv", index=False)

    # Same disciplined tiebreak as the CRFS tuning round: F1 -> MRR ->
    # (here) prefer the SMALLEST shortlist that achieves the best score,
    # since a smaller shortlist is cheaper at inference time and a tie
    # on accuracy/F1/MRR gives no evidence a larger one is actually better.
    best_f1 = report_df["val_f1"].max()
    tied_f1 = report_df[report_df["val_f1"] == best_f1]
    best_mrr = tied_f1["val_mrr"].max()
    tied_mrr = tied_f1[tied_f1["val_mrr"] == best_mrr]
    chosen = tied_mrr.sort_values("shortlist_n").iloc[0]

    print(f"\nTiebreak trace: {len(report_df)} candidates -> "
          f"F1 tie:{len(tied_f1)} -> MRR tie:{len(tied_mrr)} -> smallest shortlist_n wins")

    frozen_config = {
        "architecture": "RRF (BM25+Dense recall) -> KG rerank -> recency tiebreak",
        "rrf_k": RRF_K_DEFAULT,
        "rrf_k_tuned": False,
        "rrf_k_source": "literature default (Cormack, Clarke & Buettcher 2009)",
        "selected_shortlist_n": int(chosen["shortlist_n"]),
        "objective": "macro_F1_on_dev_val, tiebreak MRR, tiebreak smallest shortlist_n",
        "dev_val_f1": float(chosen["val_f1"]),
        "dev_val_accuracy": float(chosen["val_accuracy"]),
        "dev_val_mrr": float(chosen["val_mrr"]),
        "n_candidates_searched": len(SHORTLIST_CANDIDATES),
        "n_val_queries": len(val_queries),
        "selected_at": pd.Timestamp.now().isoformat(),
        "note": "Selected using data/split_dev_train.csv and data/split_dev_val.csv ONLY. "
                "No test split (old or new) was loaded by this script.",
    }
    with open("results/frozen_config_rrf.json", "w") as f:
        json.dump(frozen_config, f, indent=2)

    print(f"\nSelected shortlist_n = {chosen['shortlist_n']}")
    print(f"dev-val F1={chosen['val_f1']}  Accuracy={chosen['val_accuracy']}  MRR={chosen['val_mrr']}")
    print("\nFull report -> results/rrf_shortlist_tuning_report.csv")
    print("Frozen config -> results/frozen_config_rrf.json")
    print("\nThis configuration is now FROZEN.")


if __name__ == "__main__":
    main()
