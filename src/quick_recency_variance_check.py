# quick_recency_variance_check.py
"""
For each of the 123 test queries, checks whether the RRF-selected top-3
shortlist actually contains more than one distinct recency value. If
recency is identical across all 3 candidates for most/all queries, that
directly confirms why removing it from N1's linear score changed zero
predictions -- it had nothing to discriminate in the first place.
"""
import json
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from pipeline_common import (
    load_guideline_corpus, load_symptom_split, build_retrieval_text,
    build_kg, MedCPTEncoder, warmup, reciprocal_rank_fusion, recency_weight,
)

with open("results/frozen_config_rrf_v3.json") as f:
    rrf_cfg = json.load(f)
shortlist_n, rrf_k = rrf_cfg["selected_shortlist_n"], rrf_cfg["rrf_k"]

corpus_df = load_guideline_corpus()
#dev_full = load_symptom_split("data/split_dev_train_v3.csv")
dev_full = load_symptom_split("data/split_dev_v3.csv")
test_df = pd.read_csv("data/split_test_final_v3.csv")

retrieval_texts = build_retrieval_text(corpus_df, dev_full)
bm25 = BM25Okapi([t.lower().split() for t in retrieval_texts])
encoder = MedCPTEncoder()
doc_embeddings = encoder.encode_docs(retrieval_texts, batch_size=16)
warmup(bm25, encoder)

n_docs = len(corpus_df)
recency_all = np.array([recency_weight(corpus_df.iloc[i]["source_date"]) for i in range(n_docs)])

rows = []
for _, row in test_df.iterrows():
    query = row["query_text"]
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    q_emb = encoder.encode_queries([query])[0]
    dense_scores = doc_embeddings @ q_emb
    rrf_scores = reciprocal_rank_fusion(bm25_scores, dense_scores, k=rrf_k)
    shortlist = np.argsort(rrf_scores)[::-1][:shortlist_n]
    recency_vals = recency_all[shortlist]
    rows.append({
        "Disease": row["Disease"],
        "recency_values_in_shortlist": list(recency_vals),
        "all_identical": len(set(np.round(recency_vals, 6))) == 1,
    })

df = pd.DataFrame(rows)
df.to_csv("results/recency_variance_check_v3.csv", index=False)
print(f"Queries where all {shortlist_n} shortlisted candidates share the same recency value: "
      f"{df['all_identical'].sum()} / {len(df)}")