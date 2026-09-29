"""
master_results_verification_v3.py

MASTER VERIFICATION FOR THE UPDATED MEDICAL RETRIEVAL PAPER

Verifies:
    - dataset/test-set structure
    - CRFS frozen weights
    - RRF shortlist configuration
    - BM25 / Dense / Hybrid / Hybrid+KG / CRFS metrics
    - Proposed RRF + KG + Recency metrics
    - Precision / Recall / Macro-F1 / Accuracy / MRR
    - Wilson 95% confidence intervals
    - canonical vs paraphrased accuracy
    - N1 / N2 attribution ablations at TOP-3
    - McNemar exact tests
    - bootstrap accuracy-difference confidence intervals
    - per-query predictions and ranks
    - per-disease accuracy
    - latency
    - comparison with manuscript-claimed values
    - comparison with existing stored result files

IMPORTANT:
    This script never overwrites the existing results.
    All verification outputs go to:
        results/master_verification_v3/

The manuscript currently claims:
    RRF shortlist = 3
    RRF k = 60
    CRFS weights approximately:
        (0.205, 0.314, 0.378, 0.103)

The script deliberately tests the manuscript's TOP-3 claim even if the
currently stored frozen RRF config says something else. It also runs the
stored-config shortlist separately when that value differs.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rank_bm25 import BM25Okapi
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from statsmodels.stats.contingency_tables import mcnemar


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]  # moved from the project root to legacy/scripts/ at release packaging

SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

OUT_DIR = RESULTS / "master_verification_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# IMPORT THE AUDITED PIPELINE COMPONENTS
# ============================================================

from pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    kg_relevance,
    crfs_score,
    reciprocal_rank_fusion,
    kg_rerank,
)


# ============================================================
# FILES
# ============================================================

TEST_PATH = DATA / "split_test_final_v3.csv"
DEV_V3_PATH = DATA / "split_dev_v3.csv"
DEV_TRAIN_PATH = DATA / "split_dev_train_v3.csv"
DEV_VAL_PATH = DATA / "split_dev_val_v3.csv"
CORPUS_PATH = DATA / "guideline_corpus.jsonl"

CRFS_CONFIG_PATH = RESULTS / "frozen_config_v3.json"
RRF_CONFIG_PATH = RESULTS / "frozen_config_rrf_v3.json"

STORED_MAIN_PATH = RESULTS / "retrieval_comparison_v3.csv"
STORED_SUBSET_PATH = RESULTS / "subset_breakdown_v3.csv"
STORED_SIG_PATH = RESULTS / "significance_tests_v3.csv"
STORED_BOOTSTRAP_PATH = RESULTS / "bootstrap_ci_v3.csv"
STORED_PER_QUERY_PATH = RESULTS / "per_query_predictions_v3.csv"
STORED_RRF_TUNING_PATH = RESULTS / "rrf_shortlist_tuning_report_v3.csv"


# ============================================================
# MANUSCRIPT CLAIMS
# ============================================================

PAPER_RRF_SHORTLIST = 3
PAPER_RRF_K = 60

PAPER_WEIGHTS_ROUNDED = {
    "w_bm25": 0.205,
    "w_dense": 0.314,
    "w_kg": 0.378,
    "w_recency": 0.103,
}


# Main-table claims from the manuscript.
EXPECTED_MAIN = {
    "BM25-only": {
        "Accuracy": 65.85,
        "Precision": 71.10,
        "Recall": 65.85,
        "F1": 64.35,
        "MRR": 0.752,
    },
    "Dense-only": {
        "Accuracy": 65.04,
        "Precision": 64.83,
        "Recall": 65.04,
        "F1": 60.07,
        "MRR": 0.737,
    },
    "Hybrid": {
        "Accuracy": 71.54,
        "Precision": 73.98,
        "Recall": 71.54,
        "F1": 69.30,
        "MRR": 0.794,
    },
    "Hybrid+KG (no recency)": {
        "Accuracy": 86.99,
        "Precision": 90.85,
        "Recall": 86.99,
        "F1": 85.48,
        "MRR": 0.909,
    },
    "Proposed (CRFS)": {
        "Accuracy": 87.80,
        "Precision": 94.15,
        "Recall": 87.80,
        "F1": 87.14,
        "MRR": 0.913,
    },
    "Proposed (RRF+KG+Recency)": {
        "Accuracy": 80.49,
        "Precision": 82.87,
        "Recall": 80.49,
        "F1": 77.69,
        "MRR": 0.813,
    },
}


EXPECTED_SUBSETS = {
    "BM25-only": {
        "canonical": 87.80,
        "paraphrased": 54.88,
    },
    "Dense-only": {
        "canonical": 73.17,
        "paraphrased": 60.98,
    },
    "Hybrid": {
        "canonical": 87.80,
        "paraphrased": 63.41,
    },
    "Hybrid+KG (no recency)": {
        "canonical": 97.56,
        "paraphrased": 81.71,
    },
    "Proposed (CRFS)": {
        "canonical": 100.00,
        "paraphrased": 81.71,
    },
    "Proposed (RRF+KG+Recency)": {
        "canonical": 95.12,
        "paraphrased": 73.17,
    },
    "N1 (Linear+KG @ Top-3)": {
        "canonical": 95.12,
        "paraphrased": 73.17,
    },
    "N2 (RRF+KG @ Top-3, no recency)": {
        "canonical": 95.12,
        "paraphrased": 73.17,
    },
}


# Manuscript latency claims.
# These are checked for reporting consistency, NOT treated as exact
# pass/fail because latency is hardware/runtime dependent.
PAPER_LATENCY = {
    "BM25-only": (2.19, 1.00),
    "Dense-only": (16.95, 2.48),
    "Hybrid": (23.53, 5.70),
    "Hybrid+KG (no recency)": (35.59, 7.64),
    "Proposed (CRFS)": (72.66, 14.85),
    "Proposed (RRF+KG+Recency)": (21.25, 4.62),
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sync_cuda():
    """Synchronize CUDA so latency measurements do not stop early."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wilson_ci(successes, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")

    p = successes / n

    denom = 1.0 + z * z / n

    center = (
        p + z * z / (2.0 * n)
    ) / denom

    half = (
        z
        * math.sqrt(
            (p * (1.0 - p) / n)
            + (z * z / (4.0 * n * n))
        )
    ) / denom

    return (
        max(0.0, center - half),
        min(1.0, center + half),
    )


def mcnemar_exact(correct_a, correct_b):
    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)

    if len(a) != len(b):
        raise ValueError("McNemar vectors must have equal length.")

    b_better = int(((a == 0) & (b == 1)).sum())
    a_better = int(((a == 1) & (b == 0)).sum())

    if b_better == 0 and a_better == 0:
        return {
            "b_right_a_wrong": 0,
            "a_right_b_wrong": 0,
            "p": 1.0,
        }

    table = [
        [0, b_better],
        [a_better, 0],
    ]

    p = float(
        mcnemar(
            table,
            exact=True,
        ).pvalue
    )

    return {
        "b_right_a_wrong": b_better,
        "a_right_b_wrong": a_better,
        "p": p,
    }


def bootstrap_accuracy_difference(
    correct_a,
    correct_b,
    n_boot=10000,
    seed=157,
):
    """
    Returns:
        mean observed difference A-B in percentage points
        bootstrap 95% CI
    """

    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            "Bootstrap vectors must have equal length."
        )

    observed = (
        (a.mean() - b.mean())
        * 100.0
    )

    rng = np.random.default_rng(seed)

    n = len(a)

    indices = rng.integers(
        0,
        n,
        size=(n_boot, n),
    )

    sample_a = a[indices]
    sample_b = b[indices]

    diffs = (
        sample_a.mean(axis=1)
        - sample_b.mean(axis=1)
    ) * 100.0

    low, high = np.percentile(
        diffs,
        [2.5, 97.5],
    )

    return (
        float(observed),
        float(low),
        float(high),
    )


def aggregate_prediction(
    ranked_indices,
    score_values,
    corpus_df,
    top_k=20,
):
    """
    Same disease-level aggregation logic used by the retrieval pipeline.
    """

    disease_scores = {}

    for rank, idx in enumerate(
        ranked_indices[:top_k],
        start=1,
    ):
        disease = str(
            corpus_df.iloc[idx]["disease_tag"]
        ).strip()

        disease_scores[disease] = (
            disease_scores.get(disease, 0.0)
            + float(score_values[idx])
            * (1.0 / rank)
        )

    ranking = sorted(
        disease_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    prediction = (
        ranking[0][0]
        if ranking
        else ""
    )

    return prediction, ranking


def metric_row(
    method,
    labels,
    predictions,
    reciprocal_ranks,
    latencies,
    shortlist_n=None,
    rrf_k=None,
):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "Method": method,
        "N": len(labels),
        "Correct": int(
            np.sum(
                np.asarray(predictions)
                == np.asarray(labels)
            )
        ),
        "Accuracy": round(
            accuracy * 100,
            2,
        ),
        "Precision": round(
            precision * 100,
            2,
        ),
        "Recall": round(
            recall * 100,
            2,
        ),
        "F1": round(
            f1 * 100,
            2,
        ),
        "MRR": round(
            float(
                np.mean(reciprocal_ranks)
            ),
            3,
        ),
        "Latency_ms_mean": round(
            float(
                np.mean(latencies)
            ) * 1000,
            2,
        ),
        "Latency_ms_std": round(
            float(
                np.std(latencies)
            ) * 1000,
            2,
        ),
        "shortlist_n": shortlist_n,
        "rrf_k": rrf_k,
    }


# ============================================================
# DATA / CONFIG AUDIT
# ============================================================

def audit_data_and_configs():
    print("=" * 90)
    print("DATA / CONFIGURATION AUDIT")
    print("=" * 90)

    test_df = pd.read_csv(
        TEST_PATH
    )

    dev_df = pd.read_csv(
        DEV_V3_PATH
    )

    dev_train_df = pd.read_csv(
        DEV_TRAIN_PATH
    )

    dev_val_df = pd.read_csv(
        DEV_VAL_PATH
    )

    corpus_df = load_guideline_corpus(
        str(CORPUS_PATH)
    )

    print(
        f"Test rows              : {len(test_df)}"
    )
    print(
        f"Test diseases          : {test_df['Disease'].nunique()}"
    )
    print(
        f"Canonical rows         : "
        f"{(~test_df['paraphrased'].astype(bool)).sum()}"
    )
    print(
        f"Paraphrased rows       : "
        f"{test_df['paraphrased'].astype(bool).sum()}"
    )
    print(
        f"Dev v3 rows            : {len(dev_df)}"
    )
    print(
        f"Dev-train v3 rows      : {len(dev_train_df)}"
    )
    print(
        f"Dev-val v3 rows        : {len(dev_val_df)}"
    )
    print(
        f"Guideline documents    : {len(corpus_df)}"
    )

    # --------------------------------------------------------
    # Test structure
    # --------------------------------------------------------

    assert len(test_df) == 123, (
        f"Expected 123 test rows, got {len(test_df)}"
    )

    canonical_mask = (
        ~test_df["paraphrased"].astype(bool)
    )

    paraphrased_mask = (
        test_df["paraphrased"].astype(bool)
    )

    assert canonical_mask.sum() == 41, (
        "Expected 41 canonical queries."
    )

    assert paraphrased_mask.sum() == 82, (
        "Expected 82 paraphrased queries."
    )

    assert (
        test_df.loc[
            canonical_mask,
            "Disease",
        ].nunique()
        == 41
    ), (
        "Canonical subset does not contain "
        "exactly 41 unique diseases."
    )

    canonical_counts = (
        test_df.loc[
            canonical_mask,
            "Disease",
        ]
        .value_counts()
    )

    assert (
        canonical_counts == 1
    ).all(), (
        "Canonical subset does not have "
        "exactly one query per disease."
    )

    # --------------------------------------------------------
    # Test/dev overlap
    # --------------------------------------------------------

    if (
        "original_row_index" in test_df.columns
        and "original_row_index" in dev_df.columns
    ):
        test_ids = set(
            test_df[
                "original_row_index"
            ].astype(int)
        )

        dev_ids = set(
            dev_df[
                "original_row_index"
            ].astype(int)
        )

        overlap = test_ids & dev_ids

        print(
            f"Test/dev row overlap    : {len(overlap)}"
        )

        assert len(overlap) == 0, (
            "v3 test/dev overlap detected."
        )

        union_size = (
            len(test_ids | dev_ids)
        )

        print(
            f"Test + dev union       : {union_size}"
        )

    # --------------------------------------------------------
    # Dev train / dev val overlap
    # --------------------------------------------------------

    if (
        "original_row_index" in dev_train_df.columns
        and "original_row_index" in dev_val_df.columns
    ):
        train_ids = set(
            dev_train_df[
                "original_row_index"
            ].astype(int)
        )

        val_ids = set(
            dev_val_df[
                "original_row_index"
            ].astype(int)
        )

        overlap = train_ids & val_ids

        print(
            f"Dev-train/val overlap  : {len(overlap)}"
        )

        assert len(overlap) == 0, (
            "Dev-train/dev-val overlap detected."
        )

    # --------------------------------------------------------
    # CRFS config
    # --------------------------------------------------------

    with open(
        CRFS_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        crfs_cfg = json.load(f)

    weights = crfs_cfg[
        "selected_weights"
    ]

    print(
        "\nStored CRFS weights:"
    )

    for key in [
        "w_bm25",
        "w_dense",
        "w_kg",
        "w_recency",
    ]:
        print(
            f"  {key:12s}: "
            f"{weights[key]:.9f}"
        )

    weight_sum = sum(
        float(v)
        for v in weights.values()
    )

    print(
        f"  sum         : {weight_sum:.9f}"
    )

    assert abs(weight_sum - 1.0) < 1e-9, (
        "CRFS weights do not sum to 1."
    )

    for key, expected in PAPER_WEIGHTS_ROUNDED.items():
        actual = float(weights[key])

        if abs(actual - expected) > 0.001:
            print(
                f"[WARNING] {key} differs "
                f"from manuscript rounded value."
            )

    # --------------------------------------------------------
    # RRF config
    # --------------------------------------------------------

    with open(
        RRF_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        rrf_cfg = json.load(f)

    stored_shortlist = int(
        rrf_cfg["selected_shortlist_n"]
    )

    stored_rrf_k = int(
        rrf_cfg["rrf_k"]
    )

    print(
        "\nStored RRF config:"
    )

    print(
        f"  shortlist_n : {stored_shortlist}"
    )

    print(
        f"  rrf_k       : {stored_rrf_k}"
    )

    if stored_shortlist != PAPER_RRF_SHORTLIST:
        print(
            "\n[CONFIG MISMATCH]"
        )
        print(
            f"Manuscript claims shortlist={PAPER_RRF_SHORTLIST}, "
            f"but frozen_config_rrf_v3.json says "
            f"shortlist={stored_shortlist}."
        )

    assert stored_rrf_k == PAPER_RRF_K, (
        f"Expected RRF k={PAPER_RRF_K}, "
        f"got {stored_rrf_k}"
    )

    return (
        test_df,
        dev_df,
        dev_train_df,
        dev_val_df,
        corpus_df,
        weights,
        stored_shortlist,
        stored_rrf_k,
    )


# ============================================================
# RUN LINEAR METHODS
# ============================================================

def run_linear_method(
    method_name,
    queries,
    labels,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    crfs_weights,
):
    """
    Reproduces the linear-fusion branch.

    Methods:
        BM25-only
        Dense-only
        Hybrid
        Hybrid+KG (no recency)
        Proposed (CRFS)
    """

    predictions = []
    reciprocal_ranks = []
    correct = []
    latencies = []

    n_docs = len(
        corpus_df
    )

    for query, true_label in zip(
        queries,
        labels,
    ):

        sync_cuda()
        start = time.perf_counter()

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        if method_name != "Dense-only":

            tokens = (
                query.lower().split()
            )

            bm25_scores = (
                bm25.get_scores(
                    tokens
                )
            )

            lo = bm25_scores.min()
            hi = bm25_scores.max()

            if hi > lo:
                bm25_scores = (
                    bm25_scores - lo
                ) / (
                    hi - lo
                )
            else:
                bm25_scores = np.zeros(
                    n_docs
                )

        else:
            bm25_scores = np.zeros(
                n_docs
            )

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        if method_name != "BM25-only":

            q_emb = (
                encoder.encode_queries(
                    [query]
                )[0]
            )

            dense_scores = (
                doc_embeddings @ q_emb
            )

            lo = dense_scores.min()
            hi = dense_scores.max()

            if hi > lo:
                dense_scores = (
                    dense_scores - lo
                ) / (
                    hi - lo
                )
            else:
                dense_scores = np.zeros(
                    n_docs
                )

        else:
            dense_scores = np.zeros(
                n_docs
            )

        # ----------------------------------------------------
        # KG
        # ----------------------------------------------------

        use_kg = method_name in {
            "Hybrid+KG (no recency)",
            "Proposed (CRFS)",
        }

        if use_kg:

            kg_scores = np.asarray([
                kg_relevance(
                    query,
                    corpus_df.iloc[i][
                        "disease_tag"
                    ],
                    graph,
                )
                for i in range(n_docs)
            ])

        else:

            kg_scores = np.zeros(
                n_docs
            )

        # ----------------------------------------------------
        # Recency
        # ----------------------------------------------------

        use_recency = (
            method_name
            == "Proposed (CRFS)"
        )

        if use_recency:

            recency_scores = np.asarray([
                float(
                    0.5
                    if pd.isna(
                        corpus_df.iloc[i][
                            "source_date"
                        ]
                    )
                    else
                    0.5 ** (
                        max(
                            0,
                            (
                                pd.Timestamp.now()
                                - pd.to_datetime(
                                    corpus_df.iloc[i][
                                        "source_date"
                                    ]
                                )
                            ).days
                        )
                        / 365.0
                    )
                )
                for i in range(n_docs)
            ])

        else:

            recency_scores = np.zeros(
                n_docs
            )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        if method_name == "BM25-only":

            fused = bm25_scores

        elif method_name == "Dense-only":

            fused = dense_scores

        elif method_name == "Hybrid":

            # Exact implementation used in the audited pipeline.
            fused = (
                0.70 * bm25_scores
                + 0.30 * dense_scores
            )

        else:

            fused = crfs_score(
                bm25_scores,
                dense_scores,
                kg_scores,
                recency_scores,
                (
                    float(crfs_weights["w_bm25"]),
                    float(crfs_weights["w_dense"]),
                    float(crfs_weights["w_kg"]),
                    float(crfs_weights["w_recency"]),
                ),
            )

        # ----------------------------------------------------
        # Disease-level aggregation
        # ----------------------------------------------------

        ranked_indices = np.argsort(
            fused
        )[::-1]

        prediction, disease_ranking = (
            aggregate_prediction(
                ranked_indices,
                fused,
                corpus_df,
                top_k=20,
            )
        )

        predictions.append(
            prediction
        )

        is_correct = int(
            prediction == true_label
        )

        correct.append(
            is_correct
        )

        true_rank = next(
            (
                rank
                for rank, (
                    disease,
                    _score,
                ) in enumerate(
                    disease_ranking,
                    start=1,
                )
                if disease == true_label
            ),
            None,
        )

        reciprocal_ranks.append(
            1.0 / true_rank
            if true_rank is not None
            else 0.0
        )

        sync_cuda()
        latencies.append(
            time.perf_counter()
            - start
        )

    result = metric_row(
        method_name,
        labels,
        predictions,
        reciprocal_ranks,
        latencies,
    )

    return {
        "result": result,
        "predictions": predictions,
        "correct": correct,
        "rranks": reciprocal_ranks,
        "latencies": latencies,
    }


# ============================================================
# RUN RRF METHOD
# ============================================================

def run_rrf_method(
    method_name,
    queries,
    labels,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    shortlist_n,
    rrf_k,
    use_recency_tiebreak,
):
    """
    Runs the RRF architecture.

    use_recency_tiebreak=True:
        Proposed RRF + KG + Recency

    use_recency_tiebreak=False:
        N2
    """

    predictions = []
    reciprocal_ranks = []
    correct = []
    latencies = []

    n_docs = len(
        corpus_df
    )

    for query, true_label in zip(
        queries,
        labels,
    ):

        sync_cuda()
        start = time.perf_counter()

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        tokens = (
            query.lower().split()
        )

        bm25_scores = (
            bm25.get_scores(
                tokens
            )
        )

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        q_emb = (
            encoder.encode_queries(
                [query]
            )[0]
        )

        dense_scores = (
            doc_embeddings @ q_emb
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        rrf_scores = (
            reciprocal_rank_fusion(
                bm25_scores,
                dense_scores,
                k=rrf_k,
            )
        )

        # ----------------------------------------------------
        # Shortlist
        # ----------------------------------------------------

        shortlist = np.argsort(
            rrf_scores
        )[::-1][
            :shortlist_n
        ]

        # ----------------------------------------------------
        # KG reranking
        # ----------------------------------------------------

        if use_recency_tiebreak:

            reranked = kg_rerank(
                shortlist,
                rrf_scores,
                corpus_df,
                query,
                graph,
            )

        else:

            scored = []

            for idx in shortlist:

                disease = str(
                    corpus_df.iloc[idx][
                        "disease_tag"
                    ]
                ).strip()

                kg_score = (
                    kg_relevance(
                        query,
                        disease,
                        graph,
                    )
                )

                scored.append(
                    (
                        idx,
                        float(
                            kg_score
                        ),
                        float(
                            rrf_scores[idx]
                        ),
                    )
                )

            scored.sort(
                key=lambda x: (
                    x[1],
                    x[2],
                ),
                reverse=True,
            )

            reranked = [
                item[0]
                for item in scored
            ]

        # ----------------------------------------------------
        # Aggregation
        # ----------------------------------------------------

        prediction, disease_ranking = (
            aggregate_prediction(
                reranked,
                rrf_scores,
                corpus_df,
                top_k=20,
            )
        )

        predictions.append(
            prediction
        )

        is_correct = int(
            prediction == true_label
        )

        correct.append(
            is_correct
        )

        true_rank = next(
            (
                rank
                for rank, (
                    disease,
                    _score,
                ) in enumerate(
                    disease_ranking,
                    start=1,
                )
                if disease == true_label
            ),
            None,
        )

        reciprocal_ranks.append(
            1.0 / true_rank
            if true_rank is not None
            else 0.0
        )

        sync_cuda()
        latencies.append(
            time.perf_counter()
            - start
        )

    result = metric_row(
        method_name,
        labels,
        predictions,
        reciprocal_ranks,
        latencies,
        shortlist_n=shortlist_n,
        rrf_k=rrf_k,
    )

    return {
        "result": result,
        "predictions": predictions,
        "correct": correct,
        "rranks": reciprocal_ranks,
        "latencies": latencies,
    }


# ============================================================
# N1: LINEAR @ TOP-3
# ============================================================

def run_n1_top3(
    queries,
    labels,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    crfs_weights,
):
    """
    N1:
        Linear fusion
        -> top-3 candidate pool
        -> KG reranking
        -> aggregation

    Recency is disabled.

    This directly tests candidate-pool size while preserving
    the linear + KG structure.
    """

    predictions = []
    reciprocal_ranks = []
    correct = []
    latencies = []

    weights = (
        float(crfs_weights["w_bm25"]),
        float(crfs_weights["w_dense"]),
        float(crfs_weights["w_kg"]),
        float(crfs_weights["w_recency"]),
    )

    n_docs = len(
        corpus_df
    )

    for query, true_label in zip(
        queries,
        labels,
    ):

        sync_cuda()
        start = time.perf_counter()

        bm25_scores = (
            bm25.get_scores(
                query.lower().split()
            )
        )

        lo = bm25_scores.min()
        hi = bm25_scores.max()

        if hi > lo:
            bm25_scores = (
                bm25_scores - lo
            ) / (
                hi - lo
            )
        else:
            bm25_scores = np.zeros(
                n_docs
            )

        q_emb = (
            encoder.encode_queries(
                [query]
            )[0]
        )

        dense_scores = (
            doc_embeddings @ q_emb
        )

        lo = dense_scores.min()
        hi = dense_scores.max()

        if hi > lo:
            dense_scores = (
                dense_scores - lo
            ) / (
                hi - lo
            )
        else:
            dense_scores = np.zeros(
                n_docs
            )

        kg_scores = np.asarray([
            kg_relevance(
                query,
                corpus_df.iloc[i][
                    "disease_tag"
                ],
                graph,
            )
            for i in range(n_docs)
        ])

        zero_recency = np.zeros(
            n_docs
        )

        fused = crfs_score(
            bm25_scores,
            dense_scores,
            kg_scores,
            zero_recency,
            weights,
        )

        shortlist = np.argsort(
            fused
        )[::-1][:3]

        scored = []

        for idx in shortlist:

            disease = str(
                corpus_df.iloc[idx][
                    "disease_tag"
                ]
            ).strip()

            kg_score = (
                kg_relevance(
                    query,
                    disease,
                    graph,
                )
            )

            scored.append(
                (
                    idx,
                    float(kg_score),
                    float(fused[idx]),
                )
            )

        scored.sort(
            key=lambda x: (
                x[1],
                x[2],
            ),
            reverse=True,
        )

        reranked = [
            x[0]
            for x in scored
        ]

        prediction, disease_ranking = (
            aggregate_prediction(
                reranked,
                fused,
                corpus_df,
                top_k=3,
            )
        )

        predictions.append(
            prediction
        )

        is_correct = int(
            prediction == true_label
        )

        correct.append(
            is_correct
        )

        true_rank = next(
            (
                rank
                for rank, (
                    disease,
                    _score,
                ) in enumerate(
                    disease_ranking,
                    start=1,
                )
                if disease == true_label
            ),
            None,
        )

        reciprocal_ranks.append(
            1.0 / true_rank
            if true_rank is not None
            else 0.0
        )

        sync_cuda()

        latencies.append(
            time.perf_counter()
            - start
        )

    result = metric_row(
        "N1 (Linear+KG @ Top-3)",
        labels,
        predictions,
        reciprocal_ranks,
        latencies,
        shortlist_n=3,
    )

    return {
        "result": result,
        "predictions": predictions,
        "correct": correct,
        "rranks": reciprocal_ranks,
        "latencies": latencies,
    }


# ============================================================
# SUBSET BREAKDOWN
# ============================================================

def build_subset_results(
    outputs,
    test_df,
):
    canonical_mask = (
        ~test_df["paraphrased"]
        .astype(bool)
        .to_numpy()
    )

    paraphrased_mask = (
        test_df["paraphrased"]
        .astype(bool)
        .to_numpy()
    )

    rows = []

    for method, data in outputs.items():

        predictions = np.asarray(
            data["predictions"]
        )

        labels = (
            test_df["Disease"]
            .astype(str)
            .str.strip()
            .to_numpy()
        )

        for subset_name, mask in [
            (
                "canonical",
                canonical_mask,
            ),
            (
                "paraphrased",
                paraphrased_mask,
            ),
        ]:

            n = int(mask.sum())

            correct = int(
                np.sum(
                    predictions[mask]
                    == labels[mask]
                )
            )

            accuracy = (
                100.0
                * correct
                / n
            )

            rows.append({
                "Method": method,
                "Subset": subset_name,
                "N": n,
                "Correct": correct,
                "Accuracy": round(
                    accuracy,
                    2,
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# PER-DISEASE BREAKDOWN
# ============================================================

def build_per_disease_results(
    outputs,
    test_df,
):
    rows = []

    labels = (
        test_df["Disease"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    for method, data in outputs.items():

        predictions = np.asarray(
            data["predictions"]
        )

        diseases = sorted(
            test_df[
                "Disease"
            ].astype(str)
            .str.strip()
            .unique()
        )

        for disease in diseases:

            mask = (
                labels == disease
            )

            n = int(mask.sum())

            correct = int(
                np.sum(
                    predictions[mask]
                    == labels[mask]
                )
            )

            rows.append({
                "Method": method,
                "Disease": disease,
                "N": n,
                "Correct": correct,
                "Accuracy": round(
                    100.0 * correct / n,
                    2,
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# STATISTICAL TESTS
# ============================================================

def build_significance_results(
    outputs,
):
    proposed = (
        outputs[
            "Proposed (RRF+KG+Recency)"
        ]["correct"]
    )

    comparisons = [
        (
            "Hybrid",
            "Proposed (RRF+KG+Recency)",
        ),
        (
            "Hybrid+KG (no recency)",
            "Proposed (RRF+KG+Recency)",
        ),
        (
            "Proposed (CRFS)",
            "Proposed (RRF+KG+Recency)",
        ),
        (
            "Dense-only",
            "Proposed (RRF+KG+Recency)",
        ),
        (
            "N1 (Linear+KG @ Top-3)",
            "Hybrid+KG (no recency)",
        ),
        (
            "N1 (Linear+KG @ Top-3)",
            "Proposed (RRF+KG+Recency)",
        ),
        (
            "N2 (RRF+KG @ Top-3, no recency)",
            "Proposed (RRF+KG+Recency)",
        ),
    ]

    rows = []

    for a, b in comparisons:

        result = mcnemar_exact(
            outputs[a]["correct"],
            outputs[b]["correct"],
        )

        rows.append({
            "comparison":
                f"{a} vs {b}",
            "n_pairs":
                len(
                    outputs[a]["correct"]
                ),
            "b_right_a_wrong":
                result[
                    "b_right_a_wrong"
                ],
            "a_right_b_wrong":
                result[
                    "a_right_b_wrong"
                ],
            "mcnemar_exact_p":
                result["p"],
        })

    return pd.DataFrame(rows)


def build_bootstrap_results(
    outputs,
    n_boot=10000,
):
    comparisons = [
        (
            "Proposed (RRF+KG+Recency)",
            "Hybrid",
        ),
        (
            "Proposed (RRF+KG+Recency)",
            "Hybrid+KG (no recency)",
        ),
        (
            "Proposed (RRF+KG+Recency)",
            "Proposed (CRFS)",
        ),
        (
            "Proposed (RRF+KG+Recency)",
            "Dense-only",
        ),
        (
            "N1 (Linear+KG @ Top-3)",
            "Hybrid+KG (no recency)",
        ),
        (
            "Proposed (RRF+KG+Recency)",
            "N1 (Linear+KG @ Top-3)",
        ),
        (
            "Proposed (RRF+KG+Recency)",
            "N2 (RRF+KG @ Top-3, no recency)",
        ),
    ]

    rows = []

    for a, b in comparisons:

        mean_diff, low, high = (
            bootstrap_accuracy_difference(
                outputs[a]["correct"],
                outputs[b]["correct"],
                n_boot=n_boot,
                seed=157,
            )
        )

        rows.append({
            "comparison":
                f"{a} minus {b}",
            "bootstrap_mean_diff_pp":
                round(mean_diff, 2),
            "bootstrap_95CI_low_pp":
                round(low, 2),
            "bootstrap_95CI_high_pp":
                round(high, 2),
        })

    return pd.DataFrame(rows)


# ============================================================
# VERIFY STORED PER-QUERY RESULTS
# ============================================================

def compare_with_stored_per_query(
    outputs,
):
    if not STORED_PER_QUERY_PATH.exists():
        print(
            "\n[INFO] Stored per-query file "
            "not found. Skipping comparison."
        )
        return

    stored = pd.read_csv(
        STORED_PER_QUERY_PATH
    )

    print("\n" + "=" * 90)
    print("COMPARISON WITH STORED PER-QUERY CORRECTNESS")
    print("=" * 90)

    for method in [
        "BM25-only",
        "Dense-only",
        "Hybrid",
        "Hybrid+KG (no recency)",
        "Proposed (CRFS)",
        "Proposed (RRF+KG+Recency)",
    ]:

        subset = stored[
            stored["method"] == method
        ].sort_values(
            "query_index"
        )

        if subset.empty:
            print(
                f"{method:35s}: NOT FOUND"
            )
            continue

        stored_correct = (
            subset["correct"]
            .astype(int)
            .to_numpy()
        )

        fresh_correct = np.asarray(
            outputs[method]["correct"],
            dtype=int,
        )

        if len(stored_correct) != len(
            fresh_correct
        ):
            print(
                f"{method:35s}: "
                f"LENGTH MISMATCH"
            )
            continue

        matches = int(
            np.sum(
                stored_correct
                == fresh_correct
            )
        )

        print(
            f"{method:35s}: "
            f"{matches}/{len(fresh_correct)} "
            f"correctness entries match"
        )


# ============================================================
# VERIFY MANUSCRIPT CLAIMS
# ============================================================

def compare_main_metrics(
    summary_df,
):
    print("\n" + "=" * 90)
    print("MANUSCRIPT MAIN-TABLE VERIFICATION")
    print("=" * 90)

    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MRR",
    ]

    rows = []

    for method, expected in (
        EXPECTED_MAIN.items()
    ):

        actual_row = summary_df[
            summary_df["Method"] == method
        ]

        if actual_row.empty:
            print(
                f"{method:35s}: NOT FOUND"
            )
            continue

        actual = actual_row.iloc[0]

        all_match = True

        for metric in metric_cols:

            diff = (
                float(actual[metric])
                - expected[metric]
            )

            tolerance = (
                0.01
                if metric != "MRR"
                else 0.001
            )

            match = (
                abs(diff)
                <= tolerance
            )

            if not match:
                all_match = False

        status = (
            "MATCH"
            if all_match
            else "MISMATCH"
        )

        print(
            f"{method:35s}: {status}"
        )

        rows.append({
            "Method": method,
            "Status": status,
        })

    return pd.DataFrame(rows)


def compare_subset_metrics(
    subset_df,
):
    print("\n" + "=" * 90)
    print("CANONICAL / PARAPHRASED VERIFICATION")
    print("=" * 90)

    rows = []

    for method, expected in (
        EXPECTED_SUBSETS.items()
    ):

        for subset_name in [
            "canonical",
            "paraphrased",
        ]:

            actual_row = subset_df[
                (subset_df["Method"] == method)
                &
                (
                    subset_df["Subset"]
                    == subset_name
                )
            ]

            if actual_row.empty:
                rows.append({
                    "Method": method,
                    "Subset": subset_name,
                    "Expected": expected[
                        subset_name
                    ],
                    "Actual": None,
                    "Status": "MISSING",
                })
                continue

            actual = float(
                actual_row.iloc[0][
                    "Accuracy"
                ]
            )

            expected_value = expected[
                subset_name
            ]

            status = (
                "MATCH"
                if abs(
                    actual
                    - expected_value
                ) <= 0.01
                else "MISMATCH"
            )

            print(
                f"{method:35s} "
                f"{subset_name:12s}: "
                f"{actual:6.2f}% "
                f"(expected "
                f"{expected_value:6.2f}%) "
                f"{status}"
            )

            rows.append({
                "Method": method,
                "Subset": subset_name,
                "Expected": expected_value,
                "Actual": actual,
                "Status": status,
            })

    return pd.DataFrame(rows)


# ============================================================
# STORED-ARTIFACT AUDIT
# ============================================================

def audit_existing_results():
    print("\n" + "=" * 90)
    print("EXISTING STORED RESULT AUDIT")
    print("=" * 90)

    # --------------------------------------------------------
    # Existing main table
    # --------------------------------------------------------

    if STORED_MAIN_PATH.exists():

        stored_main = pd.read_csv(
            STORED_MAIN_PATH
        )

        print(
            "\nStored retrieval_comparison_v3.csv:"
        )

        print(
            stored_main[
                [
                    "Method",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "MRR",
                    "Latency_ms_mean",
                    "Latency_ms_std",
                    "shortlist_n",
                    "rrf_k",
                ]
            ].to_string(
                index=False
            )
        )

        proposed_row = stored_main[
            stored_main["Method"]
            == "Proposed (RRF+KG+Recency)"
        ]

        if not proposed_row.empty:

            stored_shortlist = (
                proposed_row.iloc[0][
                    "shortlist_n"
                ]
            )

            if pd.notna(
                stored_shortlist
            ):
                print(
                    "\nStored main-table "
                    f"shortlist = "
                    f"{int(stored_shortlist)}"
                )

    else:
        print(
            "\nStored main table missing."
        )

    # --------------------------------------------------------
    # Existing subset table
    # --------------------------------------------------------

    if STORED_SUBSET_PATH.exists():

        stored_subset = pd.read_csv(
            STORED_SUBSET_PATH
        )

        crfs_canonical = stored_subset[
            (
                stored_subset["Method"]
                == "Proposed (CRFS)"
            )
            &
            (
                stored_subset["Subset"]
                == "canonical"
            )
        ]

        if not crfs_canonical.empty:

            correct = int(
                crfs_canonical.iloc[0][
                    "Correct"
                ]
            )

            n = int(
                crfs_canonical.iloc[0][
                    "N"
                ]
            )

            print(
                "\nStored CRFS canonical:"
            )

            print(
                f"  correct = {correct}"
            )

            print(
                f"  total   = {n}"
            )

            print(
                f"  accuracy= "
                f"{100.0 * correct / n:.2f}%"
            )

    # --------------------------------------------------------
    # Existing bootstrap
    # --------------------------------------------------------

    if STORED_BOOTSTRAP_PATH.exists():

        print(
            "\nStored bootstrap results:"
        )

        print(
            pd.read_csv(
                STORED_BOOTSTRAP_PATH
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Existing RRF tuning report
    # --------------------------------------------------------

    if STORED_RRF_TUNING_PATH.exists():

        tuning = pd.read_csv(
            STORED_RRF_TUNING_PATH
        )

        print(
            "\nStored RRF tuning report:"
        )

        print(
            tuning.to_string(
                index=False
            )
        )

        if (
            "shortlist_n"
            in tuning.columns
        ):

            candidates = sorted(
                tuning[
                    "shortlist_n"
                ]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            print(
                "\nStored shortlist candidates:"
            )

            print(
                candidates
            )

            if (
                PAPER_RRF_SHORTLIST
                not in candidates
            ):
                print(
                    "[WARNING] Stored tuning "
                    "report does not contain "
                    "shortlist=3."
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 90)
    print("MASTER RESULTS VERIFICATION — V3")
    print("=" * 90)

    print(
        f"ROOT: {ROOT}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Existing artifacts first
    # --------------------------------------------------------

    audit_existing_results()

    # --------------------------------------------------------
    # Data/config audit
    # --------------------------------------------------------

    (
        test_df,
        dev_df,
        dev_train_df,
        dev_val_df,
        corpus_df,
        crfs_weights,
        stored_shortlist,
        stored_rrf_k,
    ) = audit_data_and_configs()

    queries = (
        test_df["query_text"]
        .astype(str)
        .tolist()
    )

    labels = (
        test_df["Disease"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # --------------------------------------------------------
    # Build frozen retrieval infrastructure
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "BUILDING FROZEN RETRIEVAL INFRASTRUCTURE"
    )
    print("=" * 90)

    print(
        "\nBuilding retrieval-aware "
        "document text from dev-train-v3..."
    )

    retrieval_texts = (
        build_retrieval_text(
            corpus_df,
            dev_train_df,
        )
    )

    assert len(
        retrieval_texts
    ) == len(
        corpus_df
    )

    print(
        f"Corpus documents: "
        f"{len(corpus_df)}"
    )

    print(
        "\nBuilding BM25..."
    )

    bm25 = BM25Okapi(
        [
            text.lower().split()
            for text in retrieval_texts
        ]
    )

    print(
        "\nLoading MedCPT..."
    )

    encoder = MedCPTEncoder()

    print(
        "\nEncoding corpus..."
    )

    doc_embeddings = (
        encoder.encode_docs(
            retrieval_texts,
            batch_size=16,
        )
    )

    print(
        f"Embedding matrix: "
        f"{doc_embeddings.shape}"
    )

    print(
        "\nBuilding knowledge graph..."
    )

    graph = build_kg(
        corpus_df,
        dev_train_df,
    )

    print(
        f"KG nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"KG edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        "\nWarm-up..."
    )

    warmup(
        bm25,
        encoder,
    )

    print(
        "Warm-up complete."
    )

    # --------------------------------------------------------
    # PRIMARY LINEAR METHODS
    # --------------------------------------------------------

    outputs = {}

    primary_linear_methods = [
        "BM25-only",
        "Dense-only",
        "Hybrid",
        "Hybrid+KG (no recency)",
        "Proposed (CRFS)",
    ]

    print("\n" + "=" * 90)
    print(
        "RUNNING PRIMARY LINEAR METHODS"
    )
    print("=" * 90)

    for method in primary_linear_methods:

        print(
            f"\nRunning {method}..."
        )

        outputs[method] = (
            run_linear_method(
                method,
                queries,
                labels,
                corpus_df,
                bm25,
                encoder,
                doc_embeddings,
                graph,
                crfs_weights,
            )
        )

        print(
            outputs[method]["result"]
        )

    # --------------------------------------------------------
    # PROPOSED RRF TOP-3
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "RUNNING PROPOSED RRF @ TOP-3"
    )
    print("=" * 90)

    outputs[
        "Proposed (RRF+KG+Recency)"
    ] = run_rrf_method(
        "Proposed (RRF+KG+Recency)",
        queries,
        labels,
        corpus_df,
        bm25,
        encoder,
        doc_embeddings,
        graph,
        shortlist_n=3,
        rrf_k=PAPER_RRF_K,
        use_recency_tiebreak=True,
    )

    print(
        outputs[
            "Proposed (RRF+KG+Recency)"
        ]["result"]
    )

    # --------------------------------------------------------
    # ALSO RUN STORED CONFIG IF DIFFERENT
    # --------------------------------------------------------

    stored_rrf_name = (
        f"Proposed RRF @ stored n={stored_shortlist}"
    )

    if stored_shortlist != PAPER_RRF_SHORTLIST:

        print("\n" + "=" * 90)
        print(
            f"RUNNING STORED CONFIG RRF @ "
            f"TOP-{stored_shortlist}"
        )
        print("=" * 90)

        outputs[
            stored_rrf_name
        ] = run_rrf_method(
            stored_rrf_name,
            queries,
            labels,
            corpus_df,
            bm25,
            encoder,
            doc_embeddings,
            graph,
            shortlist_n=stored_shortlist,
            rrf_k=stored_rrf_k,
            use_recency_tiebreak=True,
        )

        print(
            outputs[
                stored_rrf_name
            ]["result"]
        )

    # --------------------------------------------------------
    # N1 TOP-3
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "RUNNING N1 — LINEAR + KG @ TOP-3"
    )
    print("=" * 90)

    outputs[
        "N1 (Linear+KG @ Top-3)"
    ] = run_n1_top3(
        queries,
        labels,
        corpus_df,
        bm25,
        encoder,
        doc_embeddings,
        graph,
        crfs_weights,
    )

    print(
        outputs[
            "N1 (Linear+KG @ Top-3)"
        ]["result"]
    )

    # --------------------------------------------------------
    # N2 TOP-3
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "RUNNING N2 — RRF + KG @ TOP-3, NO RECENCY"
    )
    print("=" * 90)

    outputs[
        "N2 (RRF+KG @ Top-3, no recency)"
    ] = run_rrf_method(
        "N2 (RRF+KG @ Top-3, no recency)",
        queries,
        labels,
        corpus_df,
        bm25,
        encoder,
        doc_embeddings,
        graph,
        shortlist_n=3,
        rrf_k=PAPER_RRF_K,
        use_recency_tiebreak=False,
    )

    print(
        outputs[
            "N2 (RRF+KG @ Top-3, no recency)"
        ]["result"]
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    for method, data in outputs.items():
        summary_rows.append(
            data["result"]
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print("\n" + "=" * 90)
    print(
        "MASTER RESULTS"
    )
    print("=" * 90)

    display_cols = [
        "Method",
        "N",
        "Correct",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MRR",
        "Latency_ms_mean",
        "Latency_ms_std",
        "shortlist_n",
        "rrf_k",
    ]

    print(
        summary_df[
            [
                c
                for c in display_cols
                if c in summary_df.columns
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Wilson intervals
    # --------------------------------------------------------

    ci_rows = []

    for method, data in outputs.items():

        correct_count = int(
            sum(data["correct"])
        )

        low, high = wilson_ci(
            correct_count,
            len(labels),
        )

        ci_rows.append({
            "Method": method,
            "N": len(labels),
            "Correct": correct_count,
            "Accuracy": round(
                100.0
                * correct_count
                / len(labels),
                2,
            ),
            "Wilson_95CI_low": round(
                100.0 * low,
                2,
            ),
            "Wilson_95CI_high": round(
                100.0 * high,
                2,
            ),
        })

    ci_df = pd.DataFrame(
        ci_rows
    )

    print("\n" + "=" * 90)
    print(
        "WILSON 95% CONFIDENCE INTERVALS"
    )
    print("=" * 90)

    print(
        ci_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Subset
    # --------------------------------------------------------

    subset_df = (
        build_subset_results(
            outputs,
            test_df,
        )
    )

    print("\n" + "=" * 90)
    print(
        "CANONICAL / PARAPHRASED RESULTS"
    )
    print("=" * 90)

    print(
        subset_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Per-disease
    # --------------------------------------------------------

    disease_df = (
        build_per_disease_results(
            outputs,
            test_df,
        )
    )

    disease_df.to_csv(
        OUT_DIR
        / "per_disease_results_v3.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Significance
    # --------------------------------------------------------

    sig_df = (
        build_significance_results(
            outputs,
        )
    )

    print("\n" + "=" * 90)
    print(
        "MCNEMAR EXACT TESTS"
    )
    print("=" * 90)

    print(
        sig_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap_df = (
        build_bootstrap_results(
            outputs,
            n_boot=10000,
        )
    )

    print("\n" + "=" * 90)
    print(
        "BOOTSTRAP 95% CIs"
    )
    print("=" * 90)

    print(
        bootstrap_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Attribution checks
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "ATTRIBUTION CHECKS"
    )
    print("=" * 90)

    hybrid_kg = outputs[
        "Hybrid+KG (no recency)"
    ]["correct"]

    n1 = outputs[
        "N1 (Linear+KG @ Top-3)"
    ]["correct"]

    n2 = outputs[
        "N2 (RRF+KG @ Top-3, no recency)"
    ]["correct"]

    proposed = outputs[
        "Proposed (RRF+KG+Recency)"
    ]["correct"]

    # Hybrid+KG -> N1
    pool_drop = (
        np.mean(n1)
        - np.mean(hybrid_kg)
    ) * 100.0

    # N1 -> proposed
    fusion_gap = (
        np.mean(proposed)
        - np.mean(n1)
    ) * 100.0

    # N2 -> proposed
    recency_gap = (
        np.mean(proposed)
        - np.mean(n2)
    ) * 100.0

    n1_prop_discordant = int(
        np.sum(
            np.asarray(n1)
            != np.asarray(proposed)
        )
    )

    n2_prop_discordant = int(
        np.sum(
            np.asarray(n2)
            != np.asarray(proposed)
        )
    )

    print(
        f"Hybrid+KG -> N1 accuracy change: "
        f"{pool_drop:.2f} pp"
    )

    print(
        f"N1 -> Proposed accuracy change: "
        f"{fusion_gap:.2f} pp"
    )

    print(
        f"N2 -> Proposed accuracy change: "
        f"{recency_gap:.2f} pp"
    )

    print(
        f"N1 vs Proposed discordant queries: "
        f"{n1_prop_discordant}"
    )

    print(
        f"N2 vs Proposed discordant queries: "
        f"{n2_prop_discordant}"
    )

    # --------------------------------------------------------
    # Compare with stored per-query file
    # --------------------------------------------------------

    compare_with_stored_per_query(
        outputs
    )

    # --------------------------------------------------------
    # Compare manuscript claims
    # --------------------------------------------------------

    main_claims_df = (
        compare_main_metrics(
            summary_df[
                summary_df["Method"].isin(
                    EXPECTED_MAIN.keys()
                )
            ]
        )
    )

    subset_claims_df = (
        compare_subset_metrics(
            subset_df
        )
    )

    # --------------------------------------------------------
    # Latency comparison
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "LATENCY — MANUSCRIPT VS FRESH MEASUREMENT"
    )
    print("=" * 90)

    for method, paper_value in (
        PAPER_LATENCY.items()
    ):

        actual_row = summary_df[
            summary_df["Method"]
            == method
        ]

        if actual_row.empty:
            continue

        actual_mean = float(
            actual_row.iloc[0][
                "Latency_ms_mean"
            ]
        )

        actual_std = float(
            actual_row.iloc[0][
                "Latency_ms_std"
            ]
        )

        paper_mean, paper_std = (
            paper_value
        )

        print(
            f"{method:35s} "
            f"paper={paper_mean:.2f}±{paper_std:.2f} ms "
            f"fresh={actual_mean:.2f}±{actual_std:.2f} ms"
        )

    # --------------------------------------------------------
    # Save fresh verification outputs
    # --------------------------------------------------------

    summary_df.to_csv(
        OUT_DIR
        / "master_summary_v3.csv",
        index=False,
    )

    ci_df.to_csv(
        OUT_DIR
        / "master_confidence_intervals_v3.csv",
        index=False,
    )

    subset_df.to_csv(
        OUT_DIR
        / "master_subset_breakdown_v3.csv",
        index=False,
    )

    sig_df.to_csv(
        OUT_DIR
        / "master_significance_v3.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        OUT_DIR
        / "master_bootstrap_v3.csv",
        index=False,
    )

    main_claims_df.to_csv(
        OUT_DIR
        / "master_main_claim_check_v3.csv",
        index=False,
    )

    subset_claims_df.to_csv(
        OUT_DIR
        / "master_subset_claim_check_v3.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save rich per-query results
    # --------------------------------------------------------

    per_query_rows = []

    labels_np = (
        test_df["Disease"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    paraphrased_np = (
        test_df["paraphrased"]
        .astype(bool)
        .to_numpy()
    )

    for method, data in outputs.items():

        for i in range(
            len(labels_np)
        ):

            per_query_rows.append({
                "method": method,
                "query_index": i,
                "true_label":
                    labels_np[i],
                "paraphrased":
                    bool(
                        paraphrased_np[i]
                    ),
                "prediction":
                    data["predictions"][i],
                "correct":
                    int(
                        data["correct"][i]
                    ),
                "reciprocal_rank":
                    float(
                        data["rranks"][i]
                    ),
                "latency_ms":
                    float(
                        data[
                            "latencies"
                        ][i]
                        * 1000.0
                    ),
            })

    per_query_df = pd.DataFrame(
        per_query_rows
    )

    per_query_df.to_csv(
        OUT_DIR
        / "master_per_query_v3.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    main_mismatches = int(
        (
            main_claims_df[
                "Status"
            ] == "MISMATCH"
        ).sum()
    )

    subset_mismatches = int(
        (
            subset_claims_df[
                "Status"
            ] == "MISMATCH"
        ).sum()
    )

    print("\n" + "=" * 90)
    print(
        "FINAL VERIFICATION STATUS"
    )
    print("=" * 90)

    print(
        f"Main-table metric mismatches : "
        f"{main_mismatches}"
    )

    print(
        f"Subset metric mismatches     : "
        f"{subset_mismatches}"
    )

    if (
        main_mismatches == 0
        and subset_mismatches == 0
    ):
        print(
            "\n[CORE METRICS MATCH]"
        )
        print(
            "The fresh pipeline reproduces "
            "the manuscript's accuracy, "
            "precision, recall, F1, MRR "
            "and subset values."
        )
    else:
        print(
            "\n[CORE METRICS DO NOT FULLY MATCH]"
        )
        print(
            "Use master_main_claim_check_v3.csv "
            "and master_subset_claim_check_v3.csv "
            "to identify the differences."
        )

    print(
        "\nFiles written to:"
    )

    print(
        OUT_DIR
    )

    print(
        "\nVerification complete."
    )


if __name__ == "__main__":
    main()