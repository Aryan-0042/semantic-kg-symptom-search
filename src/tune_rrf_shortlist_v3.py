# tune_rrf_shortlist_v3.py
"""
Tunes shortlist_n ONLY, using data/split_dev_train_v3.csv and
data/split_dev_val_v3.csv. rrf_k stays fixed at 60 (literature default,
not tuned -- same reasoning as the v2 round).
"""
import json
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

SHORTLIST_CANDIDATES = [3, 5, 8, 10, 15, 20, 30, 50]


def main():
    print("Loading v3 dev-train / dev-val splits and corpus...")
    corpus_df = load_guideline_corpus()
    dev_train = load_symptom_split("data/split_dev_train_v3.csv")
    dev_val = load_symptom_split("data/split_dev_val_v3.csv")

    val_queries, val_labels = build_queries_from_split(dev_val)
    print(f"Validation queries: {len(val_queries)} across {len(set(val_labels))} diseases")

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
    report_df.to_csv("results/rrf_shortlist_tuning_report_v3.csv", index=False)

    best_f1 = report_df["val_f1"].max()
    tied_f1 = report_df[report_df["val_f1"] == best_f1]
    best_mrr = tied_f1["val_mrr"].max()
    tied_mrr = tied_f1[tied_f1["val_mrr"] == best_mrr]
    chosen = tied_mrr.sort_values("shortlist_n").iloc[0]  # smallest shortlist among final ties

    print(f"\nTiebreak trace: {len(report_df)} candidates -> "
          f"F1 tie:{len(tied_f1)} -> MRR tie:{len(tied_mrr)} -> smallest shortlist_n wins")

    frozen_config = {
        "architecture": "RRF (BM25+Dense recall) -> KG rerank -> recency tiebreak",
        "rrf_k": RRF_K_DEFAULT,
        "rrf_k_tuned": False,
        "selected_shortlist_n": int(chosen["shortlist_n"]),
        "objective": "macro_F1_on_dev_val, tiebreak MRR, tiebreak smallest shortlist_n",
        "dev_val_f1": float(chosen["val_f1"]),
        "dev_val_accuracy": float(chosen["val_accuracy"]),
        "dev_val_mrr": float(chosen["val_mrr"]),
        "n_candidates_searched": len(SHORTLIST_CANDIDATES),
        "n_val_queries": len(val_queries),
        "selected_at": pd.Timestamp.now().isoformat(),
        "note": "Selected using data/split_dev_train_v3.csv and data/split_dev_val_v3.csv "
                "ONLY. No test split (v1, v2, or v3) was loaded by this script.",
    }
    with open("results/frozen_config_rrf_v3.json", "w") as f:
        json.dump(frozen_config, f, indent=2)

    print(f"\nSelected shortlist_n = {chosen['shortlist_n']}")
    print("Frozen config -> results/frozen_config_rrf_v3.json")


if __name__ == "__main__":
    main()