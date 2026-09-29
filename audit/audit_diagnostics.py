"""
audit_diagnostics.py  (RESEARCH_AUDIT.md, sections 5 and 11)

Diagnostics on the v3 configuration (index pool = split_dev_v3.csv):
  1. KG-only lookup accuracy (argmax of kg_relevance over the 41 diseases)
  2. BM25-only / Dense-only accuracy when documents contain ONLY guideline
     text, vs the published document text (disease + top-8 training
     symptoms + guideline text)
  3. per-query cost of the KG and recency components as implemented
  4. cached embedding files vs the current corpus size

Run from the project root:  venv\\Scripts\\python audit\\audit_diagnostics.py
Writes nothing.
"""
import sys
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src")
import pipeline_common as pc  # noqa: E402

corpus = pc.load_guideline_corpus()
pool = pc.load_symptom_split("data/split_dev_v3.csv")
G = pc.build_kg(corpus, pool)
test = pd.read_csv("data/split_test_final_v3.csv")
Q, L, P = test.query_text.tolist(), test.Disease.str.strip().tolist(), test.paraphrased.to_numpy()


def kg_only():
    print("== 1. KG-only lookup (strict: a tie at the top counts as wrong)")
    diseases = sorted(corpus.disease_tag.str.strip().unique())
    correct, ties, zero = [], 0, 0
    for q, label in zip(Q, L):
        s = np.array([pc.kg_relevance(q, d, G) for d in diseases])
        top = [d for d, v in zip(diseases, s) if v == s.max()]
        ties += len(top) > 1
        zero += s.max() == 0
        correct.append(int(len(top) == 1 and top[0] == label))
    c = np.array(correct)
    print(f"canonical {c[~P].mean()*100:.2f}%  paraphrased {c[P].mean()*100:.2f}%  overall {c.mean()*100:.2f}%  "
          f"(tied top: {ties}, all-zero: {zero})")


def text_only():
    print("\n== 2. what the document text contributes")
    enc = pc.MedCPTEncoder()
    variants = {
        "guideline text only (+disease name)": [f"Disease: {r.disease_tag}. Guideline evidence: {r.text}" for r in corpus.itertuples()],
        "guideline text only (no disease name)": corpus.text.astype(str).tolist(),
        "as published (disease + top-8 symptoms + text)": pc.build_retrieval_text(corpus, pool),
    }
    for name, texts in variants.items():
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        emb = enc.encode_docs(texts)
        parts = []
        for m in ["BM25-only", "Dense-only"]:
            r, pq = pc.run_condition(m, Q, L, corpus, bm25, enc, emb, G, weights=(0.25,) * 4)
            pq = np.array(pq)
            parts.append(f"{m} {r['Accuracy']:.1f}% (canon {pq[~P].mean()*100:.1f} / para {pq[P].mean()*100:.1f})")
        print(f"{name:48s} | " + " | ".join(parts))


def component_cost():
    print("\n== 3. per-query component cost as implemented (CPU)")
    q = Q[5]
    t = time.perf_counter(); [pc.kg_relevance(q, corpus.iloc[i]["disease_tag"], G) for i in range(len(corpus))]
    kg_docs = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); [pc.recency_weight(corpus.iloc[i]["source_date"]) for i in range(len(corpus))]
    rec = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); [pc.kg_relevance(q, d, G) for d in corpus.disease_tag.unique()]
    kg_dis = (time.perf_counter() - t) * 1000
    print(f"KG over {len(corpus)} docs {kg_docs:.1f} ms | recency recomputed over {len(corpus)} docs {rec:.1f} ms | "
          f"KG once per disease {kg_dis:.1f} ms")


def caches():
    print("\n== 4. cached embeddings vs corpus")
    for f in ["models/medcpt_doc_embeddings.npy", "models/medcpt_retrieval_embeddings.npy"]:
        print(f"{f}: shape {np.load(f).shape}; current corpus has {len(corpus)} documents")


if __name__ == "__main__":
    kg_only()
    component_cost()
    caches()
    text_only()
