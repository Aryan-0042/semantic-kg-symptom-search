"""
retrieval_eval.py

Hybrid medical retrieval evaluation using:
- BM25
- MedCPT dense retrieval
- Knowledge Graph reranking
- CRFS score fusion
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
import torch

from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import json

# ============================================================
# Dataset Loaders
# ============================================================

def load_symptom_disease():

    df = pd.read_csv("data/symptom_disease.csv")

    symptom_cols = [
        c for c in df.columns
        if c.lower().startswith("symptom")
    ]

    df["symptom_set"] = df[symptom_cols].apply(

        lambda row: [
            s.strip()
            for s in row
            if isinstance(s, str) and s.strip()
        ],

        axis=1

    )

    return df


def load_guideline_corpus():

    rows = []

    with open(
        "data/guideline_corpus.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            rows.append(json.loads(line))

    return pd.DataFrame(rows)

def build_retrieval_text(corpus_df, symptom_df):
    """
    Builds a retrieval-oriented representation for each guideline.

    The original guideline text is preserved. We prepend disease-specific
    symptom context so lexical and dense retrieval can connect patient
    symptoms with the corresponding guideline evidence.
    """

    from collections import Counter

    disease_symptoms = {}

    for disease, group in symptom_df.groupby("Disease"):

        counter = Counter()

        for symptom_list in group["symptom_set"]:
            counter.update(symptom_list)

        symptoms = [
            s.replace("_", " ")
            for s, _ in counter.most_common(8)
        ]

        disease_symptoms[disease] = symptoms

    retrieval_texts = []

    for _, row in corpus_df.iterrows():

        disease = row["disease_tag"]
        guideline_text = str(row["text"])

        symptoms = disease_symptoms.get(disease, [])

        symptom_context = ", ".join(symptoms)

        retrieval_text = (
            f"Disease: {disease}. "
            f"Clinical symptoms: {symptom_context}. "
            f"Guideline evidence: {guideline_text}"
        )

        retrieval_texts.append(retrieval_text)

    return retrieval_texts


# ============================================================
# Configuration
# ============================================================

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DOC_MODEL = "ncbi/MedCPT-Article-Encoder"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MedCPT Encoder
# ============================================================

class MedCPTEncoder:

    def __init__(self):

        print(f"\nLoading MedCPT on {DEVICE}...")

        self.device = DEVICE

        self.query_tokenizer = AutoTokenizer.from_pretrained(
            QUERY_MODEL
        )

        self.query_model = AutoModel.from_pretrained(
            QUERY_MODEL
        ).to(self.device).eval()

        self.doc_tokenizer = AutoTokenizer.from_pretrained(
            DOC_MODEL
        )

        self.doc_model = AutoModel.from_pretrained(
            DOC_MODEL
        ).to(self.device).eval()

        print("Models loaded.\n")

    @torch.no_grad()
    def encode_queries(self, texts, batch_size=16):

        embeddings = []

        for start in range(0, len(texts), batch_size):

            batch = texts[start:start + batch_size]

            encoded = self.query_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )

            encoded = {
                k: v.to(self.device)
                for k, v in encoded.items()
            }

            output = self.query_model(**encoded)

            cls = output.last_hidden_state[:, 0]

            cls = torch.nn.functional.normalize(cls, p=2, dim=1)

            embeddings.append(cls.cpu())

        return torch.cat(embeddings).numpy()

    @torch.no_grad()
    def encode_docs(self, texts, batch_size=16):

        embeddings = []

        for start in range(0, len(texts), batch_size):

            batch = texts[start:start + batch_size]

            encoded = self.doc_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            encoded = {
                k: v.to(self.device)
                for k, v in encoded.items()
            }

            output = self.doc_model(**encoded)

            cls = output.last_hidden_state[:, 0]

            cls = torch.nn.functional.normalize(cls, p=2, dim=1)

            embeddings.append(cls.cpu())

        return torch.cat(embeddings).numpy()


# ============================================================
# Knowledge Graph
# ============================================================

def build_kg(corpus_df, symptom_df):

    G = nx.Graph()

    # --------------------------------------------------------
    # Disease ↔ Document relationships
    # --------------------------------------------------------

    for _, row in corpus_df.iterrows():

        disease = str(
            row["disease_tag"]
        ).strip()

        doc = str(
            row["id"]
        ).strip()

        if not disease or not doc:
            continue

        G.add_node(
            disease,
            type="disease"
        )

        G.add_node(
            doc,
            type="document"
        )

        G.add_edge(
            disease,
            doc,
            relation="has_guideline"
        )

    # --------------------------------------------------------
    # Disease ↔ Symptom relationships
    # --------------------------------------------------------

    # IMPORTANT:
    # Explicitly select only the original symptom columns.
    # DO NOT include symptom_set because it contains Python lists.

    symptom_cols = [
        c
        for c in symptom_df.columns
        if c.lower().startswith("symptom_")
        and c.lower() != "symptom_set"
    ]

    print("\nKG symptom columns:")
    print(symptom_cols)

    for _, row in symptom_df.iterrows():

        disease = str(
            row["Disease"]
        ).strip()

        if not disease:
            continue

        G.add_node(
            disease,
            type="disease"
        )

        for col in symptom_cols:

            value = row[col]

            # Safe missing-value check.
            # At this point value should be scalar,
            # not a list/array.
            if pd.isna(value):
                continue

            symptom = str(
                value
            ).strip().lower()

            if not symptom:
                continue

            symptom = symptom.replace(
                "_",
                " "
            )

            symptom_node = (
                f"symptom::{symptom}"
            )

            G.add_node(
                symptom_node,
                type="symptom",
                name=symptom
            )

            G.add_edge(
                disease,
                symptom_node,
                relation="has_symptom"
            )

    return G


def kg_relevance(query, doc_disease, G):
    """
    Computes symptom-specific KG relevance between a patient query
    and a candidate disease.

    Rare/specific symptoms receive higher weights than symptoms
    shared by many diseases.

    The score is independent of BM25/Dense preliminary predictions.
    """

    query = str(query).lower()
    doc_disease = str(doc_disease).strip()

    # --------------------------------------------------------
    # Candidate disease must exist
    # --------------------------------------------------------

    if doc_disease not in G:
        return 0.0

    # --------------------------------------------------------
    # Extract all symptom nodes
    # --------------------------------------------------------

    symptom_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("type") == "symptom"
    ]

    query_symptoms = set()

    for node in symptom_nodes:

        symptom_name = str(
            G.nodes[node].get("name", "")
        ).strip().lower()

        if not symptom_name:
            continue

        if symptom_name in query:
            query_symptoms.add(node)

    if not query_symptoms:
        return 0.0

    # --------------------------------------------------------
    # Candidate disease symptoms
    # --------------------------------------------------------

    candidate_symptoms = {
        node
        for node in G.neighbors(doc_disease)
        if G.nodes[node].get("type") == "symptom"
    }

    if not candidate_symptoms:
        return 0.0

    # --------------------------------------------------------
    # Shared symptoms
    # --------------------------------------------------------

    shared_symptoms = (
        query_symptoms & candidate_symptoms
    )

    if not shared_symptoms:
        return 0.0

    # --------------------------------------------------------
    # Symptom specificity
    #
    # A symptom connected to many diseases is less informative.
    # A symptom connected to fewer diseases is more informative.
    # --------------------------------------------------------

    disease_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("type") == "disease"
    ]

    total_diseases = len(disease_nodes)

    symptom_weights = {}

    for symptom_node in query_symptoms:

        connected_diseases = sum(
            1
            for neighbor in G.neighbors(symptom_node)
            if G.nodes[neighbor].get("type") == "disease"
        )

        # IDF-style weighting
        weight = np.log(
            (total_diseases + 1)
            / (connected_diseases + 1)
        ) + 1.0

        symptom_weights[symptom_node] = weight

    # --------------------------------------------------------
    # Weighted overlap
    # --------------------------------------------------------

    total_query_weight = sum(
        symptom_weights.values()
    )

    shared_weight = sum(
        symptom_weights[symptom]
        for symptom in shared_symptoms
    )

    if total_query_weight == 0:
        return 0.0

    relevance = (
        shared_weight
        / total_query_weight
    )

    return float(relevance)


# ============================================================
# Recency weighting
# ============================================================

def recency_weight(source_date, half_life_days=365):

    from datetime import datetime

    # Undated documents receive a neutral recency score.
    # We do not reward or penalize documents when the source
    # does not provide a reliable date.
    if pd.isna(source_date):
        return 0.5

    try:
        source_dt = pd.to_datetime(source_date)

        age = max(
            0,
            (datetime.now() - source_dt).days
        )

        return float(
            0.5 ** (age / half_life_days)
        )

    except Exception:
        return 0.5


# ============================================================
# CRFS Fusion
# ============================================================

def crfs_score(
    bm25_score,
    dense_score,
    kg_score,
    recency_score,
    weights=(0.50, 0.20, 0.25, 0.05),
):
    """
    Confidence-aware Retrieval Fusion Score (CRFS).

    BM25 provides the primary lexical evidence.
    Dense retrieval provides semantic evidence.
    KG provides symptom-disease evidence.
    Recency provides a small source-freshness adjustment.
    """

    w_bm25, w_dense, w_kg, w_recency = weights

    return (
        w_bm25 * bm25_score
        + w_dense * dense_score
        + w_kg * kg_score
        + w_recency * recency_score
    )


# ============================================================
# Single evaluation run
# ============================================================

def run_condition(
    name,
    queries,
    labels,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    use_kg,
    use_recency,
):

    predictions = []

    reciprocal_ranks = []

    latencies = []

    for query, true_label in zip(queries, labels):

        start = time.perf_counter()

        # ====================================================
        # BM25
        # ====================================================

        tokens = query.lower().split()

        bm25_scores = bm25.get_scores(tokens)

        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()

        if bm25_max > bm25_min:

            bm25_scores = (
                bm25_scores - bm25_min
            ) / (
                bm25_max - bm25_min
            )

        else:

            bm25_scores = np.zeros_like(
                bm25_scores
            )

        # ====================================================
        # Dense Retrieval (MedCPT)
        # ====================================================

        query_embedding = encoder.encode_queries(
            [query]
        )[0]

        dense_scores = (
            doc_embeddings @ query_embedding
        )

        dense_min = dense_scores.min()
        dense_max = dense_scores.max()

        if dense_max > dense_min:

            dense_scores = (
                dense_scores - dense_min
            ) / (
                dense_max - dense_min
            )

        else:

            dense_scores = np.zeros_like(
                dense_scores
            )

        # ====================================================
        # Knowledge Graph
        #
        # IMPORTANT:
        # KG uses the ORIGINAL patient query directly.
        #
        # It does NOT use the Hybrid prediction.
        # This keeps KG independent from BM25/Dense.
        # ====================================================

        if use_kg:

            kg_scores = np.array([

                kg_relevance(
                    query,
                    corpus_df.iloc[i]["disease_tag"],
                    graph
                )

                for i in range(
                    len(corpus_df)
                )

            ])

        else:

            kg_scores = np.zeros(
                len(corpus_df)
            )

        # ====================================================
        # Recency
        # ====================================================

        if use_recency:

            recency_scores = np.array([

                recency_weight(
                    corpus_df.iloc[i]["source_date"],
                    half_life_days=365
                )

                for i in range(
                    len(corpus_df)
                )

            ])

        else:

            recency_scores = np.zeros(
                len(corpus_df)
            )

        # ====================================================
        # Fusion
        # ====================================================

        if name == "BM25-only":

            fused = bm25_scores

        elif name == "Dense-only":

            fused = dense_scores

        elif name == "Hybrid":

            fused = (
                0.70 * bm25_scores
                +
                0.30 * dense_scores
            )

        else:

            fused = crfs_score(
                bm25_scores,
                dense_scores,
                kg_scores,
                recency_scores,
            )

        # ====================================================
        # Aggregate document scores by disease
        # ====================================================

        TOP_K = 20

        top_indices = np.argsort(
            fused
        )[::-1][:TOP_K]

        disease_scores = {}

        for rank, idx in enumerate(
            top_indices,
            start=1
        ):

            disease = str(
                corpus_df.iloc[idx]["disease_tag"]
            ).strip()

            score = float(
                fused[idx]
            )

            # Reciprocal-rank discount
            rank_weight = 1.0 / rank

            disease_scores[disease] = (

                disease_scores.get(
                    disease,
                    0.0
                )

                +

                score * rank_weight
            )

        # ====================================================
        # Final disease ranking
        # ====================================================

        ranking = sorted(
            disease_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not ranking:

            prediction = ""

        else:

            prediction = ranking[0][0]

        predictions.append(
            prediction
        )

        # ====================================================
        # Reciprocal Rank
        # ====================================================

        rank = None

        for i, (
            disease,
            _
        ) in enumerate(
            ranking,
            start=1
        ):

            if disease == true_label:

                rank = i

                break

        if rank is None:

            reciprocal_ranks.append(
                0.0
            )

        else:

            reciprocal_ranks.append(
                1.0 / rank
            )

        # ====================================================
        # Diagnostic output
        # ====================================================

        if (
            prediction != true_label
            and name in [
                "Hybrid+KG (no recency)",
                "Proposed (CRFS)"
            ]
        ):

            print(
                "\n"
                + "!" * 70
            )

            print(
                "MISCLASSIFICATION DIAGNOSTIC"
            )

            print(
                "Query :",
                query
            )

            print(
                "True  :",
                true_label
            )

            print(
                "Pred  :",
                prediction
            )

            print(
                "\nTop candidate components:"
            )

            diagnostic_indices = np.argsort(
                fused
            )[::-1][:5]

            for idx in diagnostic_indices:

                disease = str(
                    corpus_df.iloc[idx][
                        "disease_tag"
                    ]
                ).strip()

                print(
                    f"{disease:35s} "
                    f"BM25={bm25_scores[idx]:.3f} "
                    f"Dense={dense_scores[idx]:.3f} "
                    f"KG={kg_scores[idx]:.3f} "
                    f"Recency={recency_scores[idx]:.3f} "
                    f"CRFS={fused[idx]:.3f} "
                    f"Date={corpus_df.iloc[idx]['source_date']}"
                )

            print(
                "!" * 70
                + "\n"
            )

        # ====================================================
        # Per-query output
        # ====================================================

        print("=" * 60)

        print(
            "Query :",
            query
        )

        print(
            "True  :",
            true_label
        )

        print(
            "\nTop Diseases"
        )

        for disease, score in sorted(
            disease_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:

            print(
                f"{disease:35s} "
                f"{score:.3f}"
            )

        print(
            "\nPred :",
            prediction
        )

        print("=" * 60)

        latencies.append(
            time.perf_counter() - start
        )

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    mrr = np.mean(
        reciprocal_ranks
    )

    return {

        "Method": name,

        "Accuracy": round(
            accuracy * 100,
            2
        ),

        "Precision": round(
            precision * 100,
            2
        ),

        "Recall": round(
            recall * 100,
            2
        ),

        "F1": round(
            f1 * 100,
            2
        ),

        "MRR": round(
            mrr,
            3
        ),

        "Latency_ms_mean": round(
            np.mean(latencies) * 1000,
            2
        ),

        "Latency_ms_std": round(
            np.std(latencies) * 1000,
            2
        ),
    }

# ============================================================
# Main Evaluation Pipeline
# ============================================================

def main():

    print("Loading datasets...")

    corpus_df = load_guideline_corpus()
    symptom_df = load_symptom_disease()

    print(f"Corpus documents : {len(corpus_df)}")
    print(f"Disease samples  : {len(symptom_df)}")
    print(f"Unique diseases  : {symptom_df['Disease'].nunique()}")


    # --------------------------------------------------------
    # Build evaluation queries from held-out TEST split
    # --------------------------------------------------------

    print("\nBuilding held-out test evaluation queries...")

    from sklearn.model_selection import train_test_split

    # symptom_df contains the prepared 304-row dataset.
    # Recreate the same deterministic 70/15/15 split used
    # during dataset preparation.

    
    # --------------------------------------------------------
    # Reproduce the original 70/15/15 stratified split
    # --------------------------------------------------------
    
    from sklearn.model_selection import train_test_split
    
    SEED = 42
    
    # First split:
    # 70% train
    # 30% temporary
    #
    # 304 -> approximately 213 + 91
    
    train_df, temp_df = train_test_split(
        symptom_df,
        test_size=0.30,
        random_state=SEED,
        stratify=symptom_df["Disease"],
    )
    
    # Second split:
    # Instead of stratifying the 30% temporary set, use
    # the same deterministic random seed to divide it into
    # validation and test.
    #
    # This avoids the "least populated class has only 1 member"
    # error caused by stratifying the already-small temporary set.
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=None,
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    print(
        "Unique diseases in test:",
        test_df["Disease"].nunique()
    )
    

    # --------------------------------------------------------
    # Build leakage-controlled held-out evaluation set
    # --------------------------------------------------------

    print("\nBuilding leakage-controlled evaluation queries...")

    SEED = 42

    # --------------------------------------------------------
    # One held-out sample per disease
    # --------------------------------------------------------
    #
    # Each disease contributes exactly ONE test example.
    #
    # The remaining samples are used to construct:
    #   - retrieval-aware document representations
    #   - KG symptom relationships
    #
    # This guarantees that every disease appears in evaluation
    # while preventing the KG from seeing the held-out example.
    # --------------------------------------------------------

    rng = np.random.RandomState(SEED)

    test_indices = []

    for disease, group in symptom_df.groupby(
        "Disease",
        sort=True
    ):

        # Deterministically select one sample from each disease.
        selected_index = rng.choice(
            group.index.to_numpy()
        )

        test_indices.append(
            selected_index
        )

    test_df = symptom_df.loc[
        test_indices
    ].copy()

    train_df = symptom_df.drop(
        index=test_indices
    ).copy()

    print(
        f"Training samples : {len(train_df)}"
    )

    print(
        f"Test samples     : {len(test_df)}"
    )

    print(
        f"Test diseases     : "
        f"{test_df['Disease'].nunique()}"
    )

    # --------------------------------------------------------
    # Verify every disease is represented exactly once
    # --------------------------------------------------------

    test_counts = test_df["Disease"].value_counts()

    print(
        "\nTest samples per disease:"
    )

    print(
        test_counts.to_string()
    )

    assert (
        test_df["Disease"].nunique()
        == symptom_df["Disease"].nunique()
    )

    assert (
        test_counts == 1
    ).all()

    # --------------------------------------------------------
    # Build test queries
    # --------------------------------------------------------

    queries = []
    labels = []

    symptom_cols = [
        c
        for c in test_df.columns
        if c.lower().startswith("symptom_")
        and c.lower() != "symptom_set"
    ]

    for _, row in test_df.iterrows():

        symptoms = []

        for col in symptom_cols:

            value = row[col]

            if pd.isna(value):
                continue

            symptom = str(
                value
            ).strip()

            if not symptom:
                continue

            symptom = symptom.replace(
                "_",
                " "
            )

            symptoms.append(
                symptom
            )

        if not symptoms:
            continue

        query = (
            "The patient reports "
            + ", ".join(symptoms)
            + "."
        )

        queries.append(
            query
        )

        labels.append(
            str(
                row["Disease"]
            ).strip()
        )

    print(
        f"\nEvaluation queries built: "
        f"{len(queries)}"
    )

    print(
        "\nExample Test Queries:\n"
    )

    for query, label in list(
        zip(queries, labels)
    )[:5]:

        print(
            label,
            "->",
            query
        )




    
    

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 index...")

    # --------------------------------------------------------
    # Retrieval-aware document representations
    # --------------------------------------------------------

    print("\nBuilding retrieval-aware document representations...")


    retrieval_texts = build_retrieval_text(
        corpus_df,
        train_df
    )



    print("Example retrieval representation:\n")
    print(retrieval_texts[0][:1000])
    print()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("Building BM25 index...")

    tokenized_docs = [
        text.lower().split()
        for text in retrieval_texts
    ]

    bm25 = BM25Okapi(tokenized_docs)

    # --------------------------------------------------------
    # Dense encoder
    # --------------------------------------------------------

    encoder = MedCPTEncoder()

    import os

    CACHE_FILE = "models/medcpt_retrieval_embeddings.npy"

    if os.path.exists(CACHE_FILE):

        print("\nLoading cached document embeddings...")

        doc_embeddings = np.load(CACHE_FILE)

    else:

        print("\nEncoding guideline corpus (first run)...")

        doc_embeddings = encoder.encode_docs(
            retrieval_texts,
            batch_size=16
        )

        os.makedirs("models", exist_ok=True)

        np.save(CACHE_FILE, doc_embeddings)

        print(f"Saved embeddings -> {CACHE_FILE}")

    # --------------------------------------------------------
    # Knowledge Graph
    # --------------------------------------------------------

    graph = build_kg(
        corpus_df,
        train_df
    )

    # --------------------------------------------------------
    # Baseline comparison (Table II)
    # --------------------------------------------------------

    methods = [

        "BM25-only",

        "Dense-only",

        "Hybrid",

        "Hybrid+KG (no recency)",

        "Proposed (CRFS)",

    ]

    comparison_rows = []

    print("\nRunning baseline comparison...\n")

    for method in methods:

        print(f"\nRunning {method}...")

        use_kg = (
            "KG" in method
            or "Proposed" in method
        )

        use_recency = (
            "Proposed" in method
        )

        result = run_condition(


            method,

            queries,

            labels,

            corpus_df,

            bm25,

            encoder,

            doc_embeddings,

            graph,

            use_kg,

            use_recency,

        )

        comparison_rows.append(result)

    comparison_df = pd.DataFrame(comparison_rows)

    import os
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv(

        "results/retrieval_comparison.csv",

        index=False,

    )

    print("\nSaved:")
    print("results/retrieval_comparison.csv")

    # --------------------------------------------------------
    # Ablation study (Table III)
    # --------------------------------------------------------

    print("\nRunning ablation study...\n")

    ablation_configs = {

        "Full Proposed System":
            dict(
                use_kg=True,
                use_recency=True,
            ),

        "w/o Knowledge Graph":
            dict(
                use_kg=False,
                use_recency=True,
            ),

        "w/o Recency Weight":
            dict(
                use_kg=True,
                use_recency=False,
            ),

        "w/o Both (Hybrid only)":
            dict(
                use_kg=False,
                use_recency=False,
            ),
    }

    ablation_rows = []

    for label, cfg in ablation_configs.items():

        print(f"\nAblation: {label}")

        result = run_condition(

            "Proposed (CRFS)",

            queries,

            labels,

            corpus_df,

            bm25,

            encoder,

            doc_embeddings,

            graph,

            cfg["use_kg"],

            cfg["use_recency"],

        )

        result["Method"] = label

        ablation_rows.append(result)

    ablation_df = pd.DataFrame(

        ablation_rows

    )

    ablation_df.to_csv(

        "results/ablation.csv",

        index=False,

    )

    print("\nSaved:")
    print("results/ablation.csv")

    print("\n========================================")
    print("Retrieval Evaluation Complete")
    print("========================================")
    print(f"Corpus documents : {len(corpus_df)}")
    print(f"Queries          : {len(queries)}")
    print("Comparison table : results/retrieval_comparison.csv")
    print("Ablation table   : results/ablation.csv")
    print("========================================")
    

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()