# """
# latency_benchmark.py
# Benchmarks end-to-end response latency under two deployment modes:
#   - "edge_sim": quantized-model style CPU-only path (torch.no_grad, fp32
#     CPU inference as a stand-in for an on-device quantized model)
#   - "cloud_sim": same model, but with an artificial network round-trip
#     delay sampled from a realistic 4G/3G latency distribution, standing
#     in for a cloud API call

# Run this after you have a trained model (e.g. triage_classifier_train.py)
# or swap `model_forward` for your actual retrieval+generation pipeline.
# """
# import time
# import numpy as np
# import pandas as pd

# def simulate_network_delay(mode="4g"):
#     # rough, literature-typical RTT distributions in seconds
#     params = {"4g": (0.05, 0.02), "3g": (0.15, 0.05), "wifi": (0.01, 0.005)}
#     mu, sigma = params[mode]
#     return max(0.0, np.random.normal(mu, sigma))

# def model_forward(x):
#     # placeholder compute cost -- replace with real model.forward(x)
#     time.sleep(0.01)
#     return x

# def benchmark(n_queries=200):
#     rows = []
#     for _ in range(n_queries):
#         t0 = time.perf_counter()
#         model_forward(None)
#         edge_latency = time.perf_counter() - t0
#         rows.append({"mode": "edge_sim", "latency_ms": edge_latency * 1000})

#         t0 = time.perf_counter()
#         time.sleep(simulate_network_delay("4g"))
#         model_forward(None)
#         cloud_latency = time.perf_counter() - t0
#         rows.append({"mode": "cloud_sim", "latency_ms": cloud_latency * 1000})

#     df = pd.DataFrame(rows)
#     summary = df.groupby("mode")["latency_ms"].agg(["mean", "std", "median"]).reset_index()
#     summary["throughput_qps"] = 1000 / summary["mean"]
#     df.to_csv("results/latency_raw.csv", index=False)
#     summary.to_csv("results/latency_summary.csv", index=False)
#     print(summary)

# if __name__ == "__main__":
#     benchmark()



# """
# latency_benchmark.py

# REAL latency benchmark for the actual retrieval pipeline.

# Measures end-to-end local inference latency for:
#     Query
#       -> BM25
#       -> MedCPT dense retrieval
#       -> RRF fusion
#       -> KG reranking
#       -> disease-level aggregation

# IMPORTANT:
# - This is a REAL benchmark of the local pipeline.
# - It is NOT a measurement of a mobile/ARM device.
# - It is NOT a real cloud API benchmark.
# - Do not describe the result as "edge vs cloud" unless a real
#   edge device and real cloud endpoint are separately benchmarked.

# Outputs:
#     results/latency_raw.csv
#     results/latency_summary.csv
# """

# import os
# import time
# import json
# import numpy as np
# import pandas as pd
# from rank_bm25 import BM25Okapi

# from pipeline_common import (
#     load_guideline_corpus,
#     load_symptom_split,
#     build_retrieval_text,
#     build_kg,
#     MedCPTEncoder,
#     warmup,
#     reciprocal_rank_fusion,
#     kg_rerank,
# )


# SEED = 42
# np.random.seed(SEED)

# TEST_PATH = "data/split_test_final_v2.csv"
# DEV_PATH = "data/split_dev.csv"
# CORPUS_PATH = "data/guideline_corpus.jsonl"
# RRF_CONFIG_PATH = "results/frozen_config_rrf.json"

# N_WARMUP = 5
# N_REPEATS = 3


# def load_configuration():
#     """
#     Load the frozen RRF configuration used by the confirmatory evaluation.
#     """
#     with open(RRF_CONFIG_PATH, "r", encoding="utf-8") as f:
#         config = json.load(f)

#     shortlist_n = int(config["selected_shortlist_n"])
#     rrf_k = int(config["rrf_k"])

#     return shortlist_n, rrf_k


# def run_actual_pipeline(
#     query,
#     corpus_df,
#     bm25,
#     encoder,
#     doc_embeddings,
#     graph,
#     shortlist_n,
#     rrf_k,
#     top_k=20,
# ):
#     """
#     Execute the ACTUAL Proposed RRF + KG + Recency retrieval pipeline.

#     This function intentionally performs the same computational stages
#     used by run_condition_rrf_kg_recency(), but does not calculate paper
#     metrics because this benchmark is only measuring latency.
#     """

#     n_docs = len(corpus_df)

#     # ---------------- BM25 ----------------
#     tokens = query.lower().split()
#     bm25_scores = bm25.get_scores(tokens)

#     # ---------------- MedCPT dense retrieval ----------------
#     q_emb = encoder.encode_queries([query])[0]
#     dense_scores = doc_embeddings @ q_emb

#     # ---------------- RRF ----------------
#     rrf_scores = reciprocal_rank_fusion(
#         bm25_scores,
#         dense_scores,
#         k=rrf_k,
#     )

#     # ---------------- Shortlist ----------------
#     shortlist = np.argsort(rrf_scores)[::-1][:shortlist_n]

#     # ---------------- KG reranking ----------------
#     reranked = kg_rerank(
#         shortlist,
#         rrf_scores,
#         corpus_df,
#         query,
#         graph,
#     )

#     # ---------------- Disease aggregation ----------------
#     disease_scores = {}

#     for rank, idx in enumerate(reranked[:top_k], start=1):
#         disease = str(
#             corpus_df.iloc[idx]["disease_tag"]
#         ).strip()

#         disease_scores[disease] = (
#             disease_scores.get(disease, 0.0)
#             + float(rrf_scores[idx]) * (1.0 / rank)
#         )

#     if not disease_scores:
#         return ""

#     ranking = sorted(
#         disease_scores.items(),
#         key=lambda x: x[1],
#         reverse=True,
#     )

#     return ranking[0][0]


# def synchronize_device():
#     """
#     Synchronize CUDA before/after timing when CUDA is available.

#     Without synchronization, GPU operations may execute asynchronously
#     and the measured latency can be artificially low.
#     """
#     try:
#         import torch

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#     except Exception:
#         pass


# def benchmark(
#     n_repeats=N_REPEATS,
#     n_warmup=N_WARMUP,
# ):
#     print("=" * 70)
#     print("REAL LOCAL RETRIEVAL LATENCY BENCHMARK")
#     print("=" * 70)

#     os.makedirs("results", exist_ok=True)

#     # ------------------------------------------------------------
#     # Load frozen configuration
#     # ------------------------------------------------------------

#     shortlist_n, rrf_k = load_configuration()

#     print(
#         f"Frozen configuration: "
#         f"shortlist_n={shortlist_n}, rrf_k={rrf_k}"
#     )

#     # ------------------------------------------------------------
#     # Load data
#     # ------------------------------------------------------------

#     print("\nLoading guideline corpus...")
#     corpus_df = load_guideline_corpus(CORPUS_PATH)

#     print("Loading development split...")
#     dev_df = load_symptom_split(DEV_PATH)

#     print("Loading frozen v2 test set...")
#     test_df = pd.read_csv(TEST_PATH)

#     if "query_text" not in test_df.columns:
#         raise ValueError(
#             f"{TEST_PATH} does not contain query_text."
#         )

#     queries = (
#         test_df["query_text"]
#         .astype(str)
#         .tolist()
#     )

#     print(f"Test queries available: {len(queries)}")

#     # ------------------------------------------------------------
#     # Build actual retrieval infrastructure
#     # ------------------------------------------------------------

#     print("\nBuilding retrieval-aware document text...")
#     retrieval_texts = build_retrieval_text(
#         corpus_df,
#         dev_df,
#     )

#     print("Building BM25 index...")
#     bm25 = BM25Okapi(
#         [text.lower().split() for text in retrieval_texts]
#     )

#     print("Loading MedCPT...")
#     encoder = MedCPTEncoder()

#     print("Encoding guideline corpus...")
#     doc_embeddings = encoder.encode_docs(
#         retrieval_texts,
#         batch_size=16,
#     )

#     print("Building knowledge graph...")
#     graph = build_kg(
#         corpus_df,
#         dev_df,
#     )

#     # ------------------------------------------------------------
#     # Warm-up
#     # ------------------------------------------------------------

#     print("\nWarming up models before timing...")

#     warmup(
#         bm25,
#         encoder,
#     )

#     # Additional full-pipeline warm-up.
#     for query in queries[:min(n_warmup, len(queries))]:
#         synchronize_device()

#         run_actual_pipeline(
#             query,
#             corpus_df,
#             bm25,
#             encoder,
#             doc_embeddings,
#             graph,
#             shortlist_n,
#             rrf_k,
#         )

#         synchronize_device()

#     print("Warm-up complete.")

#     # ------------------------------------------------------------
#     # Actual benchmark
#     # ------------------------------------------------------------

#     rows = []

#     print(
#         f"\nBenchmarking {len(queries)} queries "
#         f"x {n_repeats} repeat(s)..."
#     )

#     for repeat in range(1, n_repeats + 1):

#         print(f"  Repeat {repeat}/{n_repeats}")

#         for query_index, query in enumerate(queries):

#             synchronize_device()

#             start = time.perf_counter()

#             prediction = run_actual_pipeline(
#                 query,
#                 corpus_df,
#                 bm25,
#                 encoder,
#                 doc_embeddings,
#                 graph,
#                 shortlist_n,
#                 rrf_k,
#             )

#             synchronize_device()

#             elapsed_ms = (
#                 time.perf_counter() - start
#             ) * 1000.0

#             rows.append(
#                 {
#                     "repeat": repeat,
#                     "query_index": query_index,
#                     "latency_ms": elapsed_ms,
#                     "prediction": prediction,
#                 }
#             )

#     raw_df = pd.DataFrame(rows)

#     raw_path = "results/latency_raw.csv"
#     raw_df.to_csv(
#         raw_path,
#         index=False,
#     )

#     # ------------------------------------------------------------
#     # Summary statistics
#     # ------------------------------------------------------------

#     latencies = raw_df["latency_ms"].to_numpy()

#     summary = {
#         "method": "Proposed (RRF+KG+Recency)",
#         "hardware_mode": "local",
#         "n_queries": len(queries),
#         "n_repeats": n_repeats,
#         "total_measurements": len(latencies),
#         "mean_ms": float(np.mean(latencies)),
#         "std_ms": float(np.std(latencies, ddof=1)),
#         "median_ms": float(np.median(latencies)),
#         "p95_ms": float(np.percentile(latencies, 95)),
#         "min_ms": float(np.min(latencies)),
#         "max_ms": float(np.max(latencies)),
#         "throughput_qps": float(
#             1000.0 / np.mean(latencies)
#         ),
#         "shortlist_n": shortlist_n,
#         "rrf_k": rrf_k,
#     }

#     summary_df = pd.DataFrame([summary])

#     summary_path = "results/latency_summary.csv"

#     summary_df.to_csv(
#         summary_path,
#         index=False,
#     )

#     # ------------------------------------------------------------
#     # Print result
#     # ------------------------------------------------------------

#     print("\n" + "=" * 70)
#     print("REAL LOCAL LATENCY RESULTS")
#     print("=" * 70)

#     print(
#         f"Queries:        {len(queries)}"
#     )
#     print(
#         f"Measurements:   {len(latencies)}"
#     )
#     print(
#         f"Mean:           {summary['mean_ms']:.2f} ms"
#     )
#     print(
#         f"Std:            {summary['std_ms']:.2f} ms"
#     )
#     print(
#         f"Median:         {summary['median_ms']:.2f} ms"
#     )
#     print(
#         f"P95:            {summary['p95_ms']:.2f} ms"
#     )
#     print(
#         f"Min:            {summary['min_ms']:.2f} ms"
#     )
#     print(
#         f"Max:            {summary['max_ms']:.2f} ms"
#     )
#     print(
#         f"Throughput:     {summary['throughput_qps']:.3f} queries/s"
#     )

#     print("\nFiles written:")
#     print(f"  {raw_path}")
#     print(f"  {summary_path}")

#     print("\nIMPORTANT:")
#     print(
#         "These are REAL local measurements of the retrieval pipeline."
#     )
#     print(
#         "They are NOT mobile/edge-device measurements and NOT cloud"
#         " round-trip measurements."
#     )

#     return summary_df


# if __name__ == "__main__":
#     benchmark()



"""
latency_benchmark.py

FAIR LOCAL LATENCY BENCHMARK
----------------------------

Measures all retrieval methods under identical conditions:

1. BM25-only
2. Dense-only
3. Hybrid
4. Hybrid+KG (no recency)
5. Proposed (CRFS)
6. Proposed (RRF+KG+Recency)

IMPORTANT:
- Uses the frozen v2 test set.
- Uses the same corpus, BM25 index, MedCPT model, and KG for all methods.
- Uses the frozen CRFS weights.
- Uses the frozen RRF configuration.
- Does NOT modify or rerun the confirmatory evaluation.
- GPU synchronization is used when CUDA is available.
- This measures LOCAL laptop latency only.
- It is NOT mobile/ARM edge latency.
- It is NOT cloud round-trip latency.

Outputs:
    results/latency_all_methods_raw.csv
    results/latency_all_methods_summary.csv
"""

import os
import time
import json
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    reciprocal_rank_fusion,
    kg_rerank,
    crfs_score,
)


SEED = 42
np.random.seed(SEED)

TEST_PATH = "data/split_test_final_v2.csv"
DEV_PATH = "data/split_dev.csv"
CORPUS_PATH = "data/guideline_corpus.jsonl"

CRFS_CONFIG_PATH = "results/frozen_config.json"
RRF_CONFIG_PATH = "results/frozen_config_rrf.json"

N_WARMUP = 5
N_REPEATS = 3


METHODS = [
    "BM25-only",
    "Dense-only",
    "Hybrid",
    "Hybrid+KG (no recency)",
    "Proposed (CRFS)",
    "Proposed (RRF+KG+Recency)",
]


def synchronize_device():
    """
    Synchronize CUDA so asynchronous GPU work is included
    in the measured latency.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def load_configuration():
    """
    Load the frozen configurations used by the evaluation.
    Nothing is tuned here.
    """

    with open(CRFS_CONFIG_PATH, "r", encoding="utf-8") as f:
        crfs_config = json.load(f)

    crfs_w = crfs_config["selected_weights"]

    crfs_weights = (
        float(crfs_w["w_bm25"]),
        float(crfs_w["w_dense"]),
        float(crfs_w["w_kg"]),
        float(crfs_w["w_recency"]),
    )

    with open(RRF_CONFIG_PATH, "r", encoding="utf-8") as f:
        rrf_config = json.load(f)

    shortlist_n = int(rrf_config["selected_shortlist_n"])
    rrf_k = int(rrf_config["rrf_k"])

    return crfs_weights, shortlist_n, rrf_k


def get_raw_scores(
    query,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    need_bm25=True,
    need_dense=True,
    need_kg=True,
    need_recency=True,
):
    """
    Compute only the retrieval components required by a method.
    """

    n_docs = len(corpus_df)

    # ------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------

    if need_bm25:
        tokens = query.lower().split()

        bm25_scores = bm25.get_scores(tokens)

        lo = bm25_scores.min()
        hi = bm25_scores.max()

        if hi > lo:
            bm25_scores = (
                (bm25_scores - lo) /
                (hi - lo)
            )
        else:
            bm25_scores = np.zeros(n_docs)

    else:
        bm25_scores = np.zeros(n_docs)

    # ------------------------------------------------------------
    # Dense / MedCPT
    # ------------------------------------------------------------

    if need_dense:
        q_emb = encoder.encode_queries([query])[0]

        dense_scores = doc_embeddings @ q_emb

        lo = dense_scores.min()
        hi = dense_scores.max()

        if hi > lo:
            dense_scores = (
                (dense_scores - lo) /
                (hi - lo)
            )
        else:
            dense_scores = np.zeros(n_docs)

    else:
        dense_scores = np.zeros(n_docs)

    # ------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------

    if need_kg:
        from pipeline_common import kg_relevance

        kg_scores = np.array([
            kg_relevance(
                query,
                corpus_df.iloc[i]["disease_tag"],
                graph,
            )
            for i in range(n_docs)
        ])

    else:
        kg_scores = np.zeros(n_docs)

    # ------------------------------------------------------------
    # Recency
    # ------------------------------------------------------------

    if need_recency:
        from pipeline_common import recency_weight

        recency_scores = np.array([
            recency_weight(
                corpus_df.iloc[i]["source_date"]
            )
            for i in range(n_docs)
        ])

    else:
        recency_scores = np.zeros(n_docs)

    return (
        bm25_scores,
        dense_scores,
        kg_scores,
        recency_scores,
    )


def aggregate_prediction(
    fused_scores,
    corpus_df,
    top_k=20,
):
    """
    Same disease-level rank-discounted aggregation used
    by the actual retrieval pipeline.
    """

    top_indices = np.argsort(
        fused_scores
    )[::-1][:top_k]

    disease_scores = {}

    for rank, idx in enumerate(
        top_indices,
        start=1,
    ):
        disease = str(
            corpus_df.iloc[idx]["disease_tag"]
        ).strip()

        disease_scores[disease] = (
            disease_scores.get(disease, 0.0)
            + float(fused_scores[idx])
            * (1.0 / rank)
        )

    if not disease_scores:
        return ""

    ranking = sorted(
        disease_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranking[0][0]


def run_method(
    method,
    query,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    crfs_weights,
    shortlist_n,
    rrf_k,
):
    """
    Execute exactly one method.

    The returned value is the prediction.
    Timing is handled externally.
    """

    # ============================================================
    # BM25-only
    # ============================================================

    if method == "BM25-only":

        bm25_scores, _, _, _ = get_raw_scores(
            query,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            need_bm25=True,
            need_dense=False,
            need_kg=False,
            need_recency=False,
        )

        return aggregate_prediction(
            bm25_scores,
            corpus_df,
        )

    # ============================================================
    # Dense-only
    # ============================================================

    if method == "Dense-only":

        _, dense_scores, _, _ = get_raw_scores(
            query,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            need_bm25=False,
            need_dense=True,
            need_kg=False,
            need_recency=False,
        )

        return aggregate_prediction(
            dense_scores,
            corpus_df,
        )

    # ============================================================
    # Hybrid
    # ============================================================

    if method == "Hybrid":

        bm25_scores, dense_scores, _, _ = get_raw_scores(
            query,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            need_bm25=True,
            need_dense=True,
            need_kg=False,
            need_recency=False,
        )

        fused = (
            0.70 * bm25_scores
            + 0.30 * dense_scores
        )

        return aggregate_prediction(
            fused,
            corpus_df,
        )

    # ============================================================
    # Hybrid + KG
    # ============================================================

    if method == "Hybrid+KG (no recency)":

        (
            bm25_scores,
            dense_scores,
            kg_scores,
            _,
        ) = get_raw_scores(
            query,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            need_bm25=True,
            need_dense=True,
            need_kg=True,
            need_recency=False,
        )

        fused = crfs_score(
            bm25_scores,
            dense_scores,
            kg_scores,
            np.zeros(len(corpus_df)),
            crfs_weights,
        )

        return aggregate_prediction(
            fused,
            corpus_df,
        )

    # ============================================================
    # Proposed CRFS
    # ============================================================

    if method == "Proposed (CRFS)":

        (
            bm25_scores,
            dense_scores,
            kg_scores,
            recency_scores,
        ) = get_raw_scores(
            query,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            need_bm25=True,
            need_dense=True,
            need_kg=True,
            need_recency=True,
        )

        fused = crfs_score(
            bm25_scores,
            dense_scores,
            kg_scores,
            recency_scores,
            crfs_weights,
        )

        return aggregate_prediction(
            fused,
            corpus_df,
        )

    # ============================================================
    # Proposed RRF + KG + Recency
    # ============================================================

    if method == "Proposed (RRF+KG+Recency)":

        (
            bm25_scores,
            dense_scores,
            _,
            _,
        ) = get_raw_scores(
            query,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            need_bm25=True,
            need_dense=True,
            need_kg=False,
            need_recency=False,
        )

        # --------------------------------------------------------
        # RRF
        # --------------------------------------------------------

        rrf_scores = reciprocal_rank_fusion(
            bm25_scores,
            dense_scores,
            k=rrf_k,
        )

        shortlist = np.argsort(
            rrf_scores
        )[::-1][:shortlist_n]

        # --------------------------------------------------------
        # KG reranking
        # --------------------------------------------------------

        reranked = kg_rerank(
            shortlist,
            rrf_scores,
            corpus_df,
            query,
            graph,
        )

        # --------------------------------------------------------
        # Disease-level aggregation
        # --------------------------------------------------------

        disease_scores = {}

        for rank, idx in enumerate(
            reranked[:20],
            start=1,
        ):
            disease = str(
                corpus_df.iloc[idx]["disease_tag"]
            ).strip()

            disease_scores[disease] = (
                disease_scores.get(disease, 0.0)
                + float(rrf_scores[idx])
                * (1.0 / rank)
            )

        if not disease_scores:
            return ""

        ranking = sorted(
            disease_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranking[0][0]

    raise ValueError(
        f"Unknown method: {method}"
    )


def benchmark(
    n_repeats=N_REPEATS,
    n_warmup=N_WARMUP,
):

    print("=" * 75)
    print("FAIR LOCAL LATENCY BENCHMARK -- ALL METHODS")
    print("=" * 75)

    os.makedirs(
        "results",
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Frozen configuration
    # ------------------------------------------------------------

    (
        crfs_weights,
        shortlist_n,
        rrf_k,
    ) = load_configuration()

    print(
        "\nFrozen CRFS weights:"
        f" {crfs_weights}"
    )

    print(
        "Frozen RRF configuration:"
        f" shortlist_n={shortlist_n},"
        f" rrf_k={rrf_k}"
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    print("\nLoading guideline corpus...")
    corpus_df = load_guideline_corpus(
        CORPUS_PATH
    )

    print("Loading development split...")
    dev_df = load_symptom_split(
        DEV_PATH
    )

    print("Loading frozen v2 test set...")
    test_df = pd.read_csv(
        TEST_PATH
    )

    if "query_text" not in test_df.columns:
        raise ValueError(
            f"{TEST_PATH} does not contain query_text."
        )

    queries = (
        test_df["query_text"]
        .astype(str)
        .tolist()
    )

    print(
        f"Test queries: {len(queries)}"
    )

    # ------------------------------------------------------------
    # Retrieval infrastructure
    # ------------------------------------------------------------

    print(
        "\nBuilding retrieval-aware document text..."
    )

    retrieval_texts = build_retrieval_text(
        corpus_df,
        dev_df,
    )

    print("Building BM25 index...")

    bm25 = BM25Okapi(
        [
            text.lower().split()
            for text in retrieval_texts
        ]
    )

    print("Loading MedCPT...")

    encoder = MedCPTEncoder()

    print("Encoding guideline corpus...")

    doc_embeddings = encoder.encode_docs(
        retrieval_texts,
        batch_size=16,
    )

    print("Building knowledge graph...")

    graph = build_kg(
        corpus_df,
        dev_df,
    )

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    print(
        "\nWarming up models before timing..."
    )

    warmup(
        bm25,
        encoder,
    )

    # Warm up every method separately.
    warmup_queries = queries[
        :min(
            n_warmup,
            len(queries),
        )
    ]

    for method in METHODS:

        print(
            f"  Warming up: {method}"
        )

        for query in warmup_queries:

            synchronize_device()

            run_method(
                method,
                query,
                corpus_df,
                bm25,
                encoder,
                doc_embeddings,
                graph,
                crfs_weights,
                shortlist_n,
                rrf_k,
            )

            synchronize_device()

    print(
        "Warm-up complete."
    )

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    rows = []

    print(
        f"\nBenchmarking "
        f"{len(queries)} queries x "
        f"{n_repeats} repeats x "
        f"{len(METHODS)} methods..."
    )

    for method in METHODS:

        print(
            f"\n  Method: {method}"
        )

        for repeat in range(
            1,
            n_repeats + 1,
        ):

            print(
                f"    Repeat {repeat}/{n_repeats}"
            )

            for query_index, query in enumerate(
                queries
            ):

                synchronize_device()

                start = time.perf_counter()

                prediction = run_method(
                    method,
                    query,
                    corpus_df,
                    bm25,
                    encoder,
                    doc_embeddings,
                    graph,
                    crfs_weights,
                    shortlist_n,
                    rrf_k,
                )

                synchronize_device()

                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                rows.append(
                    {
                        "method": method,
                        "repeat": repeat,
                        "query_index": query_index,
                        "latency_ms": elapsed_ms,
                        "prediction": prediction,
                    }
                )

    raw_df = pd.DataFrame(
        rows
    )

    raw_path = (
        "results/"
        "latency_all_methods_raw.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary_rows = []

    for method in METHODS:

        method_df = raw_df[
            raw_df["method"] == method
        ]

        latencies = (
            method_df["latency_ms"]
            .to_numpy()
        )

        mean_ms = float(
            np.mean(latencies)
        )

        summary_rows.append(
            {
                "method": method,
                "n_queries": len(queries),
                "n_repeats": n_repeats,
                "measurements": len(latencies),
                "mean_ms": mean_ms,
                "std_ms": float(
                    np.std(
                        latencies,
                        ddof=1,
                    )
                ),
                "median_ms": float(
                    np.median(latencies)
                ),
                "p95_ms": float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),
                "min_ms": float(
                    np.min(latencies)
                ),
                "max_ms": float(
                    np.max(latencies)
                ),
                "throughput_qps": float(
                    1000.0 / mean_ms
                ),
                "shortlist_n": (
                    shortlist_n
                    if method ==
                    "Proposed (RRF+KG+Recency)"
                    else np.nan
                ),
                "rrf_k": (
                    rrf_k
                    if method ==
                    "Proposed (RRF+KG+Recency)"
                    else np.nan
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        "results/"
        "latency_all_methods_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Comparison against proposed
    # ------------------------------------------------------------

    proposed_mean = float(
        summary_df.loc[
            summary_df["method"]
            == "Proposed (RRF+KG+Recency)",
            "mean_ms",
        ].iloc[0]
    )

    comparison_rows = []

    for _, row in summary_df.iterrows():

        method = row["method"]
        mean_ms = float(
            row["mean_ms"]
        )

        if method == "Proposed (RRF+KG+Recency)":
            speedup = 1.0
            percent_difference = 0.0

        else:
            speedup = (
                mean_ms /
                proposed_mean
            )

            percent_difference = (
                (
                    mean_ms -
                    proposed_mean
                )
                / mean_ms
            ) * 100.0

        comparison_rows.append(
            {
                "method": method,
                "mean_latency_ms": mean_ms,
                "proposed_latency_ms": proposed_mean,
                "baseline_to_proposed_ratio": speedup,
                "proposed_latency_reduction_percent": percent_difference,
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        "results/"
        "latency_method_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FAIR LOCAL LATENCY RESULTS")
    print("=" * 75)

    display_cols = [
        "method",
        "mean_ms",
        "std_ms",
        "median_ms",
        "p95_ms",
        "throughput_qps",
    ]

    print(
        summary_df[
            display_cols
        ].to_string(
            index=False,
            formatters={
                "mean_ms": "{:.2f}".format,
                "std_ms": "{:.2f}".format,
                "median_ms": "{:.2f}".format,
                "p95_ms": "{:.2f}".format,
                "throughput_qps": "{:.2f}".format,
            },
        )
    )

    print("\n")
    print("=" * 75)
    print("PROPOSED METHOD VS BASELINES")
    print("=" * 75)

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "mean_latency_ms": "{:.2f}".format,
                "proposed_latency_ms": "{:.2f}".format,
                "baseline_to_proposed_ratio": "{:.2f}x".format,
                "proposed_latency_reduction_percent": "{:.2f}%".format,
            },
        )
    )

    print("\nFiles written:")
    print(
        f"  {raw_path}"
    )
    print(
        f"  {summary_path}"
    )
    print(
        f"  {comparison_path}"
    )

    print("\nIMPORTANT:")
    print(
        "These are REAL local measurements "
        "on the configured machine."
    )
    print(
        "They are NOT mobile/ARM edge-device "
        "measurements."
    )
    print(
        "They are NOT cloud round-trip measurements."
    )

    return (
        summary_df,
        comparison_df,
    )


if __name__ == "__main__":
    benchmark()