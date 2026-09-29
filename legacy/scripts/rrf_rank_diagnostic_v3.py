import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]  # moved from the project root to legacy/scripts/ at release packaging
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_queries_from_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    reciprocal_rank_fusion,
    RRF_K_DEFAULT,
)


def main():
    print("=" * 80)
    print("RRF DEV-VAL RANK DIAGNOSTIC — V3")
    print("=" * 80)

    # ---------------------------------------------------------
    # IMPORTANT:
    # Only v3 DEV TRAIN + DEV VAL are loaded.
    # Frozen v3 TEST is deliberately never loaded.
    # ---------------------------------------------------------

    corpus_df = load_guideline_corpus(
        "data/guideline_corpus.jsonl"
    )

    dev_train = load_symptom_split(
        "data/split_dev_train_v3.csv"
    )

    dev_val = load_symptom_split(
        "data/split_dev_val_v3.csv"
    )

    queries, labels = build_queries_from_split(
        dev_val
    )

    print("\n[DATA]")
    print("Corpus documents :", len(corpus_df))
    print("Dev-train rows   :", len(dev_train))
    print("Dev-val rows     :", len(dev_val))
    print("Validation queries:", len(queries))
    print("Validation diseases:", len(set(labels)))
    print("RRF k             :", RRF_K_DEFAULT)

    # ---------------------------------------------------------
    # Build same retrieval infrastructure used for v3 tuning
    # ---------------------------------------------------------

    print("\n[BUILD] Retrieval-aware text from DEV-TRAIN only...")
    retrieval_texts = build_retrieval_text(
        corpus_df,
        dev_train
    )

    bm25 = BM25Okapi(
        [t.lower().split() for t in retrieval_texts]
    )

    print("[BUILD] Loading MedCPT...")
    encoder = MedCPTEncoder()

    print("[BUILD] Encoding guideline corpus...")
    doc_embeddings = encoder.encode_docs(
        retrieval_texts,
        batch_size=16
    )

    print("[BUILD] Building KG from DEV-TRAIN only...")
    graph = build_kg(
        corpus_df,
        dev_train
    )

    print("[BUILD] Warming up...")
    warmup(bm25, encoder)
    print("Warm-up complete.")

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    disease_tags = (
        corpus_df["disease_tag"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    rank_records = []

    print("\n[RUN] Computing RRF ranks before KG reranking...\n")

    for q_index, (query, true_label) in enumerate(
        zip(queries, labels)
    ):
        tokens = query.lower().split()

        bm25_scores = bm25.get_scores(tokens)

        q_emb = encoder.encode_queries(
            [query]
        )[0]

        dense_scores = doc_embeddings @ q_emb

        rrf_scores = reciprocal_rank_fusion(
            bm25_scores,
            dense_scores,
            k=RRF_K_DEFAULT
        )

        # Highest RRF score = rank 1.
        ranking = np.argsort(
            rrf_scores
        )[::-1]

        # Find first document belonging to the true disease.
        true_doc_rank = None

        for rank, doc_idx in enumerate(
            ranking,
            start=1
        ):
            if disease_tags[doc_idx] == str(true_label).strip():
                true_doc_rank = rank
                break

        # Disease-level rank: where does the TRUE DISEASE first
        # appear in the RRF document ranking?
        #
        # Also record whether at least one correct document is
        # inside top-3 / top-5 / top-10.
        rank_records.append(
            {
                "query_index": q_index,
                "true_disease": true_label,
                "true_first_document_rank": (
                    true_doc_rank
                    if true_doc_rank is not None
                    else -1
                ),
                "in_top_3": (
                    true_doc_rank is not None
                    and true_doc_rank <= 3
                ),
                "in_top_5": (
                    true_doc_rank is not None
                    and true_doc_rank <= 5
                ),
                "in_top_10": (
                    true_doc_rank is not None
                    and true_doc_rank <= 10
                ),
                "in_top_15": (
                    true_doc_rank is not None
                    and true_doc_rank <= 15
                ),
            }
        )

    df = pd.DataFrame(rank_records)

    # ---------------------------------------------------------
    # Rank distribution
    # ---------------------------------------------------------

    print("=" * 80)
    print("RRF RANK OF FIRST CORRECT DISEASE DOCUMENT")
    print("=" * 80)

    print(
        df[
            [
                "query_index",
                "true_disease",
                "true_first_document_rank",
                "in_top_3",
                "in_top_5",
                "in_top_10",
                "in_top_15",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Compact buckets requested by Claude
    # ---------------------------------------------------------

    def bucket(rank):
        if rank < 0:
            return "not found"
        if rank == 1:
            return "rank 1"
        if rank == 2:
            return "rank 2"
        if rank == 3:
            return "rank 3"
        if rank <= 5:
            return "rank 4-5"
        if rank <= 10:
            return "rank 6-10"
        if rank <= 15:
            return "rank 11-15"
        return "rank >15"

    df["rank_bucket"] = df[
        "true_first_document_rank"
    ].apply(bucket)

    bucket_order = [
        "rank 1",
        "rank 2",
        "rank 3",
        "rank 4-5",
        "rank 6-10",
        "rank 11-15",
        "rank >15",
        "not found",
    ]

    bucket_counts = (
        df["rank_bucket"]
        .value_counts()
        .reindex(bucket_order, fill_value=0)
        .rename_axis("rank_bucket")
        .reset_index(name="count")
    )

    print("\n" + "=" * 80)
    print("RANK DISTRIBUTION")
    print("=" * 80)

    print(
        bucket_counts.to_string(index=False)
    )

    # ---------------------------------------------------------
    # Coverage at different shortlist sizes
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRUE-DISEASE RECALL AT SHORTLIST DEPTH")
    print("=" * 80)

    n = len(df)

    for k in [1, 2, 3, 5, 8, 10, 15, 20, 50]:
        count = int(
            (
                df["true_first_document_rank"] <= k
            ).sum()
        )

        print(
            f"Top-{k:>2}: "
            f"{count}/{n} "
            f"({100.0 * count / n:.2f}%)"
        )

    # ---------------------------------------------------------
    # Key interpretation indicators
    # ---------------------------------------------------------

    top3 = int(df["in_top_3"].sum())
    top5 = int(df["in_top_5"].sum())
    top10 = int(df["in_top_10"].sum())

    print("\n" + "=" * 80)
    print("KEY DIAGNOSTIC")
    print("=" * 80)

    print(
        f"Correct disease has a document in top-3: "
        f"{top3}/{n} ({100.0 * top3 / n:.2f}%)"
    )

    print(
        f"Correct disease has a document in top-5: "
        f"{top5}/{n} ({100.0 * top5 / n:.2f}%)"
    )

    print(
        f"Correct disease has a document in top-10: "
        f"{top10}/{n} ({100.0 * top10 / n:.2f}%)"
    )

    print("\nNo test split was loaded or used.")

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    raw_path = results_dir / "rrf_dev_val_rank_diagnostic_v3.csv"
    bucket_path = results_dir / "rrf_dev_val_rank_distribution_v3.csv"

    df.to_csv(
        raw_path,
        index=False
    )

    bucket_counts.to_csv(
        bucket_path,
        index=False
    )

    print("\nFiles written:")
    print(raw_path)
    print(bucket_path)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()