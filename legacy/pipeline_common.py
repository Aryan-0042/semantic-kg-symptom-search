"""
pipeline_common.py

Single source of truth for the retrieval pipeline components shared by
tune_crfs_weights.py and retrieval_eval_final.py. Architecture is
preserved exactly as audited: BM25 + MedCPT dense + query-based
symptom-IDF Knowledge Graph + recency + CRFS fusion + rank-discounted
disease-level aggregation.

The ONLY behavioral change from the original retrieval_eval.py is in
run_condition(): each method now computes only the components it
actually needs, latency is captured before any print/diagnostic I/O,
and an explicit warm-up hook is provided. Nothing about the KG formula,
the CRFS formula, or the aggregation logic has been altered here --
that preserves the architecture per your instruction. Only tuning
(which weights get plugged in) moves to the validation-only scripts.
"""

import time
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DOC_MODEL = "ncbi/MedCPT-Article-Encoder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data loaders (unchanged from the audited version)
# ============================================================

def load_guideline_corpus(path="data/guideline_corpus.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_symptom_split(path):
    """Loads a symptom_disease-schema CSV (used for split_dev_train.csv,
    split_dev_val.csv, split_dev.csv, split_test_final.csv -- all share
    the same Disease,Symptom_1..17[,Urgency] schema)."""
    df = pd.read_csv(path)
    return df


def get_symptom_cols(df):
    return [c for c in df.columns if c.lower().startswith("symptom_") and c.lower() != "symptom_set"]


def build_queries_from_split(df):
    """Identical query construction logic to the audited version:
    'The patient reports <s1>, <s2>, ...' with underscores -> spaces.
    Used for BOTH validation queries (tuning) and final test queries --
    same construction, different (and never overlapping) source rows."""
    symptom_cols = get_symptom_cols(df)
    queries, labels = [], []
    for _, row in df.iterrows():
        symptoms = []
        for col in symptom_cols:
            value = row[col]
            if pd.isna(value):
                continue
            s = str(value).strip()
            if not s:
                continue
            symptoms.append(s.replace("_", " "))
        if not symptoms:
            continue
        queries.append("The patient reports " + ", ".join(symptoms) + ".")
        labels.append(str(row["Disease"]).strip())
    return queries, labels


# ============================================================
# Retrieval-aware document text (unchanged logic; caller controls
# which dataframe -- dev-train only, or full dev -- gets passed in)
# ============================================================

def build_retrieval_text(corpus_df, symptom_df):
    from collections import Counter

    disease_symptoms = {}
    symptom_cols = get_symptom_cols(symptom_df)

    for disease, group in symptom_df.groupby("Disease"):
        counter = Counter()
        for _, row in group.iterrows():
            for col in symptom_cols:
                v = row[col]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s:
                    counter.update([s])
        symptoms = [s.replace("_", " ") for s, _ in counter.most_common(8)]
        disease_symptoms[disease] = symptoms

    retrieval_texts = []
    for _, row in corpus_df.iterrows():
        disease = row["disease_tag"]
        guideline_text = str(row["text"])
        symptom_context = ", ".join(disease_symptoms.get(disease, []))
        retrieval_texts.append(
            f"Disease: {disease}. Clinical symptoms: {symptom_context}. Guideline evidence: {guideline_text}"
        )
    return retrieval_texts


# ============================================================
# MedCPT encoder (unchanged)
# ============================================================

class MedCPTEncoder:
    def __init__(self):
        print(f"\nLoading MedCPT on {DEVICE}...")
        self.device = DEVICE
        self.query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
        self.query_model = AutoModel.from_pretrained(QUERY_MODEL).to(self.device).eval()
        self.doc_tokenizer = AutoTokenizer.from_pretrained(DOC_MODEL)
        self.doc_model = AutoModel.from_pretrained(DOC_MODEL).to(self.device).eval()
        print("Models loaded.\n")

    @torch.no_grad()
    def encode_queries(self, texts, batch_size=16):
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.query_tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.query_model(**encoded)
            cls = torch.nn.functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
            embeddings.append(cls.cpu())
        return torch.cat(embeddings).numpy()

    @torch.no_grad()
    def encode_docs(self, texts, batch_size=16):
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.doc_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.doc_model(**encoded)
            cls = torch.nn.functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
            embeddings.append(cls.cpu())
        return torch.cat(embeddings).numpy()


# ============================================================
# Knowledge Graph (unchanged architecture, per your instruction
# to preserve it -- Disease<->Document, Disease<->Symptom, query-based
# IDF-weighted relevance, independent of BM25/Dense predictions)
# ============================================================

def build_kg(corpus_df, symptom_df):
    G = nx.Graph()

    for _, row in corpus_df.iterrows():
        disease = str(row["disease_tag"]).strip()
        doc = str(row["id"]).strip()
        if not disease or not doc:
            continue
        G.add_node(disease, type="disease")
        G.add_node(doc, type="document")
        G.add_edge(disease, doc, relation="has_guideline")

    symptom_cols = get_symptom_cols(symptom_df)
    for _, row in symptom_df.iterrows():
        disease = str(row["Disease"]).strip()
        if not disease:
            continue
        G.add_node(disease, type="disease")
        for col in symptom_cols:
            value = row[col]
            if pd.isna(value):
                continue
            symptom = str(value).strip().lower()
            if not symptom:
                continue
            symptom = symptom.replace("_", " ")
            symptom_node = f"symptom::{symptom}"
            G.add_node(symptom_node, type="symptom", name=symptom)
            G.add_edge(disease, symptom_node, relation="has_symptom")

    return G


def kg_relevance(query, doc_disease, G):
    query = str(query).lower()
    doc_disease = str(doc_disease).strip()

    if doc_disease not in G:
        return 0.0

    symptom_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "symptom"]
    query_symptoms = set()
    for node in symptom_nodes:
        name = str(G.nodes[node].get("name", "")).strip().lower()
        if name and name in query:
            query_symptoms.add(node)

    if not query_symptoms:
        return 0.0

    candidate_symptoms = {n for n in G.neighbors(doc_disease) if G.nodes[n].get("type") == "symptom"}
    if not candidate_symptoms:
        return 0.0

    shared = query_symptoms & candidate_symptoms
    if not shared:
        return 0.0

    disease_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "disease"]
    total_diseases = len(disease_nodes)

    symptom_weights = {}
    for s in query_symptoms:
        connected = sum(1 for nb in G.neighbors(s) if G.nodes[nb].get("type") == "disease")
        symptom_weights[s] = np.log((total_diseases + 1) / (connected + 1)) + 1.0

    total_w = sum(symptom_weights.values())
    if total_w == 0:
        return 0.0
    shared_w = sum(symptom_weights[s] for s in shared)
    return float(shared_w / total_w)


def recency_weight(source_date, half_life_days=365):
    if pd.isna(source_date):
        return 0.5
    try:
        age = max(0, (pd.Timestamp.now() - pd.to_datetime(source_date)).days)
        return float(0.5 ** (age / half_life_days))
    except Exception:
        return 0.5


def crfs_score(bm25_score, dense_score, kg_score, recency_score, weights):
    w_bm25, w_dense, w_kg, w_recency = weights
    return w_bm25 * bm25_score + w_dense * dense_score + w_kg * kg_score + w_recency * recency_score


# ============================================================
# run_condition -- CORRECTED: per-method conditional computation,
# latency captured before any I/O, optional warm-up, silent mode
# for grid-search use (verbose printing during a 100+ combo grid
# search would itself reintroduce the print-in-timing-loop bug).
# ============================================================

METHOD_REQUIREMENTS = {
    "BM25-only":              dict(bm25=True,  dense=False, kg=False, recency=False),
    "Dense-only":             dict(bm25=False, dense=True,  kg=False, recency=False),
    "Hybrid":                 dict(bm25=True,  dense=True,  kg=False, recency=False),
    "Hybrid+KG (no recency)": dict(bm25=True,  dense=True,  kg=True,  recency=False),
    "Proposed (CRFS)":        dict(bm25=True,  dense=True,  kg=True,  recency=True),
}


def warmup(bm25, encoder):
    """Absorbs model-load / CUDA-kernel-compile / first-call overhead
    OUTSIDE any timed loop. Call this once before benchmarking."""
    _ = bm25.get_scores(["warmup", "query"])
    _ = encoder.encode_queries(["warmup query"])


def precompute_raw_scores(queries, labels, corpus_df, bm25, encoder, doc_embeddings, graph):
    """
    Computes the four raw, per-document, per-query score arrays ONCE.
    None of BM25, MedCPT dense similarity, or KG relevance depend on the
    CRFS fusion weights -- only the weighted sum does. A weight grid
    search that calls the full retrieval stack once per candidate
    weight vector (286 times, in the default grid) is doing ~286x more
    BM25/MedCPT/KG work than necessary. This function does that work
    exactly once per query; fuse_and_evaluate() below then sweeps
    weights cheaply over the cached arrays.

    Returns: list of dicts, one per query:
      {"bm25": array, "dense": array, "kg": array, "recency": array,
       "true_label": str}
    """
    n_docs = len(corpus_df)
    recency_scores = np.array([recency_weight(corpus_df.iloc[i]["source_date"]) for i in range(n_docs)])

    cache = []
    for query, true_label in zip(queries, labels):
        tokens = query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        lo, hi = bm25_scores.min(), bm25_scores.max()
        bm25_scores = (bm25_scores - lo) / (hi - lo) if hi > lo else np.zeros(n_docs)

        q_emb = encoder.encode_queries([query])[0]
        dense_scores = doc_embeddings @ q_emb
        lo, hi = dense_scores.min(), dense_scores.max()
        dense_scores = (dense_scores - lo) / (hi - lo) if hi > lo else np.zeros(n_docs)

        kg_scores = np.array([kg_relevance(query, corpus_df.iloc[i]["disease_tag"], graph) for i in range(n_docs)])

        cache.append({
            "bm25": bm25_scores, "dense": dense_scores,
            "kg": kg_scores, "recency": recency_scores,
            "true_label": true_label,
        })
    return cache


def fuse_and_evaluate(cache, corpus_df, weights, top_k=20):
    """
    Cheap, weights-only sweep over precomputed score arrays from
    precompute_raw_scores(). Reproduces exactly the same fusion +
    disease-level aggregation + metrics logic as run_condition()'s
    "Proposed (CRFS)" branch, but without re-running BM25/MedCPT/KG.
    Used by tune_crfs_weights.py's grid search.
    """
    predictions, reciprocal_ranks, per_query_correct = [], [], []
    disease_tags = corpus_df["disease_tag"].astype(str).str.strip().to_numpy()

    for entry in cache:
        fused = crfs_score(entry["bm25"], entry["dense"], entry["kg"], entry["recency"], weights)
        top_indices = np.argsort(fused)[::-1][:top_k]

        disease_scores = {}
        for rank, idx in enumerate(top_indices, start=1):
            disease = disease_tags[idx]
            disease_scores[disease] = disease_scores.get(disease, 0.0) + float(fused[idx]) * (1.0 / rank)

        ranking = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        prediction = ranking[0][0] if ranking else ""
        predictions.append(prediction)

        true_label = entry["true_label"]
        rank_of_true = next((i for i, (d, _) in enumerate(ranking, start=1) if d == true_label), None)
        reciprocal_ranks.append(1.0 / rank_of_true if rank_of_true else 0.0)
        per_query_correct.append(1 if prediction == true_label else 0)

    labels = [e["true_label"] for e in cache]
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    return {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "MRR": round(float(np.mean(reciprocal_ranks)), 3),
    }, per_query_correct


def run_condition(
    name,
    queries,
    labels,
    corpus_df,
    bm25,
    encoder,
    doc_embeddings,
    graph,
    weights,
    top_k=20,
    verbose=False,
    requirements_override=None,
):
    """
    weights: (w_bm25, w_dense, w_kg, w_recency) tuple used ONLY for the
    two fused methods ("Hybrid+KG (no recency)" internally forces
    w_recency effectively unused since recency scores are zeroed by the
    'needs' flag below, not by weight -- see note under Fusion.)

    requirements_override: dict with bm25/dense/kg/recency booleans,
    used by the ablation study to force specific components on/off
    while keeping the "Proposed (CRFS)" fusion formula and weights.
    """
    needs = requirements_override if requirements_override is not None else METHOD_REQUIREMENTS[name]

    predictions, reciprocal_ranks, latencies = [], [], []
    per_query_correct = []

    n_docs = len(corpus_df)

    for query, true_label in zip(queries, labels):
        start = time.perf_counter()

        # ---------------- BM25 (only if needed) ----------------
        if needs["bm25"]:
            tokens = query.lower().split()
            bm25_scores = bm25.get_scores(tokens)
            lo, hi = bm25_scores.min(), bm25_scores.max()
            bm25_scores = (bm25_scores - lo) / (hi - lo) if hi > lo else np.zeros(n_docs)
        else:
            bm25_scores = np.zeros(n_docs)

        # ---------------- Dense / MedCPT (only if needed) ----------------
        if needs["dense"]:
            q_emb = encoder.encode_queries([query])[0]
            dense_scores = doc_embeddings @ q_emb
            lo, hi = dense_scores.min(), dense_scores.max()
            dense_scores = (dense_scores - lo) / (hi - lo) if hi > lo else np.zeros(n_docs)
        else:
            dense_scores = np.zeros(n_docs)

        # ---------------- Knowledge Graph (only if needed) ----------------
        if needs["kg"]:
            kg_scores = np.array([
                kg_relevance(query, corpus_df.iloc[i]["disease_tag"], graph) for i in range(n_docs)
            ])
        else:
            kg_scores = np.zeros(n_docs)

        # ---------------- Recency (only if needed) ----------------
        if needs["recency"]:
            recency_scores = np.array([
                recency_weight(corpus_df.iloc[i]["source_date"]) for i in range(n_docs)
            ])
        else:
            recency_scores = np.zeros(n_docs)

        # ---------------- Fusion ----------------
        if name == "BM25-only":
            fused = bm25_scores
        elif name == "Dense-only":
            fused = dense_scores
        elif name == "Hybrid":
            fused = 0.70 * bm25_scores + 0.30 * dense_scores
        else:
            fused = crfs_score(bm25_scores, dense_scores, kg_scores, recency_scores, weights)

        # ---------------- Disease-level aggregation (unchanged) ----------------
        top_indices = np.argsort(fused)[::-1][:top_k]
        disease_scores = {}
        for rank, idx in enumerate(top_indices, start=1):
            disease = str(corpus_df.iloc[idx]["disease_tag"]).strip()
            disease_scores[disease] = disease_scores.get(disease, 0.0) + float(fused[idx]) * (1.0 / rank)

        ranking = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        prediction = ranking[0][0] if ranking else ""
        predictions.append(prediction)

        rank_of_true = next((i for i, (d, _) in enumerate(ranking, start=1) if d == true_label), None)
        reciprocal_ranks.append(1.0 / rank_of_true if rank_of_true else 0.0)
        per_query_correct.append(1 if prediction == true_label else 0)

        # Latency captured HERE -- before any print/diagnostic output.
        latencies.append(time.perf_counter() - start)

        if verbose:
            print(f"[{name}] true={true_label!r} pred={prediction!r} "
                  f"{'OK' if prediction == true_label else 'MISS'}")

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    mrr = float(np.mean(reciprocal_ranks))

    result = {
        "Method": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "MRR": round(mrr, 3),
        "Latency_ms_mean": round(np.mean(latencies) * 1000, 2),
        "Latency_ms_std": round(np.std(latencies) * 1000, 2),
        "N": len(queries),
    }
    return result, per_query_correct
