"""Legacy compatibility mode: the v3 pipeline rebuilt from hsr components.

Reproduces src/retrieval_eval_final_v3.py (8 conditions on the 123-query v3
test set, index built from split_dev_v3.csv) using the new package's data
loading, encoder wrapper and BM25 library, with every legacy behaviour
switched on:

  * profile = 8 most frequent raw symptom strings (Counter insertion order)
  * document "Disease: X. Clinical symptoms: ... Guideline evidence: ..."
  * whitespace tokenisation for BM25, min-max normalised scores
  * MedCPT queries truncated at 64 tokens, documents in padded batches of 16
  * KG query symptoms found by plain substring match, IDF formula as legacy
  * recency 0.5 ** (age_days / 365), undated = 0.5, reference date frozen at
    the v3 run date (legacy used pd.Timestamp.now())
  * top-20 rank-discounted disease aggregation; RRF shortlist + kg_rerank; N1; N2

Its only purpose is regression: if it reproduces the stored v3 per-query
correctness exactly, the shared building blocks behave as before. It is not
used by any v4 experiment.
"""
from collections import Counter
import json

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from . import DATA, RESULTS
from .data import load_corpus, load_rows
from .encoders import MEDCPT_ARTICLE, MEDCPT_QUERY, Encoder

V3_RUN_DATE = "2026-08-28"
TOP_K = 20

METHODS = [
    "BM25-only", "Dense-only", "Hybrid", "Hybrid+KG (no recency)", "Proposed (CRFS)",
    "Proposed (RRF+KG+Recency)", "N1 (Linear+KG @ shortlist)", "N2 (Proposed, no recency tiebreak)",
]


def _profiles(pool_rows):
    groups = {}
    for r in pool_rows:
        groups.setdefault(r.disease, []).append(r)
    out = {}
    for disease in sorted(groups):  # pandas groupby order; irrelevant for a dict lookup
        c = Counter()
        for r in groups[disease]:
            for s in r.symptoms:
                c.update([s])
        out[disease] = [s.replace("_", " ") for s, _ in c.most_common(8)]
    return out


def _documents(corpus, profiles):
    return [
        f"Disease: {rec.disease_tag}. Clinical symptoms: {', '.join(profiles.get(rec.disease_tag, []))}. "
        f"Guideline evidence: {rec.text}"
        for rec in corpus.itertuples(index=False)
    ]


class _LegacyKG:
    """Same graph statistics and arithmetic as pipeline_common.kg_relevance."""

    def __init__(self, corpus, pool_rows):
        self.disease_symptoms = {}
        disease_nodes = {t.strip() for t in corpus.disease_tag}
        for r in pool_rows:
            disease_nodes.add(r.disease)
            names = self.disease_symptoms.setdefault(r.disease, set())
            for s in r.symptoms:
                names.add("symptom::" + s.strip().lower().replace("_", " "))
        self.total_diseases = len(disease_nodes)
        self.connected = Counter(n for ss in self.disease_symptoms.values() for n in ss)
        self.nodes = list(self.connected)

    def relevance(self, query, disease):
        query = str(query).lower()
        disease = str(disease).strip()
        if disease not in self.disease_symptoms:
            return 0.0
        query_symptoms = {n for n in self.nodes if n[len("symptom::"):] in query}
        if not query_symptoms:
            return 0.0
        shared = query_symptoms & self.disease_symptoms[disease]
        if not shared:
            return 0.0
        w = {s: np.log((self.total_diseases + 1) / (self.connected[s] + 1)) + 1.0 for s in query_symptoms}
        total = sum(w.values())
        return float(sum(w[s] for s in shared) / total) if total else 0.0

    def doc_scores(self, query, doc_diseases):
        per = {d: self.relevance(query, d) for d in set(doc_diseases)}
        return np.array([per[d] for d in doc_diseases])


def _recency(corpus, ref=V3_RUN_DATE):
    ref = pd.Timestamp(ref)
    out = []
    for d in corpus.source_date:
        if pd.isna(d):
            out.append(0.5)
        else:
            out.append(float(0.5 ** (max(0, (ref - pd.to_datetime(d)).days) / 365)))
    return np.array(out)


def _mm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros(len(v))


def _aggregate(order, scores, doc_tags, top_k=TOP_K):
    agg = {}
    for rank, idx in enumerate(order[:top_k], start=1):
        d = doc_tags[idx]
        agg[d] = agg.get(d, 0.0) + float(scores[idx]) * (1.0 / rank)
    ranking = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return ranking[0][0] if ranking else ""


def _rrf(bm25_raw, dense_raw, k=60):
    r1 = np.argsort(np.argsort(-bm25_raw)) + 1
    r2 = np.argsort(np.argsort(-dense_raw)) + 1
    return 1.0 / (k + r1) + 1.0 / (k + r2)


def _date_keys(corpus):
    keys = []
    for d in corpus.source_date:
        try:
            keys.append(pd.to_datetime(d).timestamp() if pd.notna(d) else -1.0)
        except Exception:
            keys.append(-1.0)
    return keys


def run():
    """Returns {method: np.array of 0/1 correctness over the 123 v3 queries}."""
    rows = load_rows()
    corpus = load_corpus()
    pool_ids = pd.read_csv(DATA / "split_dev_v3.csv").original_row_index.tolist()
    pool = [rows[i] for i in pool_ids]
    test = pd.read_csv(DATA / "split_test_final_v3.csv")
    queries = test.query_text.tolist()
    labels = test.Disease.astype(str).str.strip().tolist()

    w = json.load(open(RESULTS / "frozen_config_v3.json"))["selected_weights"]
    W = (w["w_bm25"], w["w_dense"], w["w_kg"], w["w_recency"])
    shortlist_n = json.load(open(RESULTS / "frozen_config_rrf_v3.json"))["selected_shortlist_n"]

    docs = _documents(corpus, _profiles(pool))
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    qenc = Encoder(MEDCPT_QUERY, max_length=64, cache=False)
    denc = Encoder(MEDCPT_ARTICLE, max_length=512, cache=False)
    doc_emb = denc.encode_batched(docs, batch_size=16)
    kg = _LegacyKG(corpus, pool)
    rec = _recency(corpus)
    dates = _date_keys(corpus)
    tags = [t.strip() for t in corpus.disease_tag]
    n = len(docs)

    out = {m: [] for m in METHODS}
    for q, label in zip(queries, labels):
        bm_raw = bm25.get_scores(q.lower().split())
        de_raw = doc_emb @ qenc.encode_batched([q], batch_size=1)[0]
        bm, de = _mm(bm_raw), _mm(de_raw)
        kgs = kg.doc_scores(q, tags)
        zeros = np.zeros(n)
        fused = {
            "BM25-only": bm,
            "Dense-only": de,
            "Hybrid": 0.70 * bm + 0.30 * de,
            "Hybrid+KG (no recency)": W[0] * bm + W[1] * de + W[2] * kgs + W[3] * zeros,
            "Proposed (CRFS)": W[0] * bm + W[1] * de + W[2] * kgs + W[3] * rec,
        }
        for m, f in fused.items():
            out[m].append(int(_aggregate(np.argsort(f)[::-1], f, tags) == label))

        rrf = _rrf(bm_raw, de_raw)
        shortlist = np.argsort(rrf)[::-1][:shortlist_n]
        with_date = sorted(((i, kgs[i], float(rrf[i]), dates[i]) for i in shortlist),
                           key=lambda t: (t[1], t[2], t[3]), reverse=True)
        out["Proposed (RRF+KG+Recency)"].append(int(_aggregate([t[0] for t in with_date], rrf, tags) == label))
        no_date = sorted(((i, kgs[i], float(rrf[i])) for i in shortlist), key=lambda t: (t[1], t[2]), reverse=True)
        out["N2 (Proposed, no recency tiebreak)"].append(int(_aggregate([t[0] for t in no_date], rrf, tags) == label))

        # N1: linear fusion (weights renormalised without recency) restricted to the
        # RRF shortlist computed from *normalised* scores, as in retrieval_eval_final_v3.py
        sl = set(np.argsort(_rrf(bm, de))[::-1][:shortlist_n].tolist())
        mask = np.array([1.0 if i in sl else 0.0 for i in range(n)])
        s3 = W[0] + W[1] + W[2]
        f1 = (W[0] / s3) * bm * mask + (W[1] / s3) * de * mask + (W[2] / s3) * kgs * mask
        agg = {}
        for rank, idx in enumerate(np.argsort(f1)[::-1][:TOP_K], start=1):
            if idx in sl:  # non-shortlist positions still consume a rank, as in legacy
                agg[tags[idx]] = agg.get(tags[idx], 0.0) + float(f1[idx]) * (1.0 / rank)
        pred = sorted(agg.items(), key=lambda x: x[1], reverse=True)[0][0] if agg else ""
        out["N1 (Linear+KG @ shortlist)"].append(int(pred == label))
    return {m: np.array(v) for m, v in out.items()}
