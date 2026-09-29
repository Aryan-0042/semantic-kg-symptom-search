"""
audit_reproduce_history.py  (RESEARCH_AUDIT.md, section 8)

Re-runs the ORIGINAL pipeline functions (src/pipeline_common.py and
src/retrieval_eval_final_v3.py, unmodified) and compares per-query
correctness with the stored result files:

  v1: split_test_final.csv    + split_dev.csv    + frozen_config.json
  v2: split_test_final_v2.csv + split_dev.csv    + frozen_config.json / frozen_config_rrf.json
  v3: split_test_final_v3.csv + split_dev_v3.csv + frozen_config_v3.json / frozen_config_rrf_v3.json

For v3 it also varies the two suspected causes of the master-verification
mismatch: the index pool (split_dev_v3 = 181 rows vs split_dev_train_v3 =
148 rows) and the recency reference date (recency_weight() uses
pd.Timestamp.now()).

Run from the project root:  venv\\Scripts\\python audit\\audit_reproduce_history.py
Writes nothing into results/.
"""
import importlib
import json
import sys

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src")
import pipeline_common as pc  # noqa: E402
import retrieval_eval_final_v3 as ev  # noqa: E402

LINEAR = ["BM25-only", "Dense-only", "Hybrid", "Hybrid+KG (no recency)", "Proposed (CRFS)"]
RRF = "Proposed (RRF+KG+Recency)"


def stored(path):
    df = pd.read_csv(path)
    return {m: g.sort_values("query_index").correct.to_numpy() for m, g in df.groupby("method")}


def weights(path):
    w = json.load(open(path))["selected_weights"]
    return (w["w_bm25"], w["w_dense"], w["w_kg"], w["w_recency"])


def build(corpus, enc, pool_path):
    pool = pc.load_symptom_split(pool_path)
    texts = pc.build_retrieval_text(corpus, pool)
    return BM25Okapi([t.lower().split() for t in texts]), enc.encode_docs(texts), pc.build_kg(corpus, pool)


def fixed_recency(ref):
    ref = pd.Timestamp(ref)

    def recency_weight(source_date, half_life_days=365):
        if pd.isna(source_date):
            return 0.5
        return float(0.5 ** (max(0, (ref - pd.to_datetime(source_date)).days) / half_life_days))

    return recency_weight


def report(tag, method, pq, ref):
    pq = np.asarray(pq)
    if ref is None:
        print(f"{tag:34s} {method:36s} correct={pq.sum():3d}  (no stored vector)")
    else:
        print(f"{tag:34s} {method:36s} correct={pq.sum():3d} stored={ref.sum():3d} per-query diffs={(pq != ref).sum()}")


def run_linear_and_rrf(tag, Q, L, corpus, enc, infra, W, shortlist, ref, with_rrf=True):
    bm25, emb, G = infra
    for m in LINEAR:
        _, pq = pc.run_condition(m, Q, L, corpus, bm25, enc, emb, G, weights=W)
        report(tag, m, pq, ref.get(m))
    if with_rrf:
        _, pq = pc.run_condition_rrf_kg_recency(Q, L, corpus, bm25, enc, emb, G, shortlist_n=shortlist, rrf_k=60)
        report(tag, RRF, pq, ref.get(RRF))


def main():
    corpus = pc.load_guideline_corpus()
    enc = pc.MedCPTEncoder()

    # ---- v1 and v2 (both used split_dev.csv as the index pool) ----
    infra = build(corpus, enc, "data/split_dev.csv")
    W1 = weights("results/frozen_config.json")
    sl2 = json.load(open("results/frozen_config_rrf.json"))["selected_shortlist_n"]
    Q1, L1 = pc.build_queries_from_split(pc.load_symptom_split("data/split_test_final.csv"))
    run_linear_and_rrf("v1", Q1, L1, corpus, enc, infra, W1, None,
                       stored("results/per_query_predictions.csv"), with_rrf=False)
    t2 = pd.read_csv("data/split_test_final_v2.csv")
    run_linear_and_rrf("v2", t2.query_text.tolist(), t2.Disease.str.strip().tolist(), corpus, enc, infra,
                       W1, sl2, stored("results/per_query_predictions_v2.csv"))

    # ---- v3: vary index pool and recency reference date ----
    W3 = weights("results/frozen_config_v3.json")
    t3 = pd.read_csv("data/split_test_final_v3.csv")
    Q3, L3 = t3.query_text.tolist(), t3.Disease.astype(str).str.strip().tolist()
    ref3 = stored("results/per_query_predictions_v3.csv")
    for pool in ["data/split_dev_v3.csv", "data/split_dev_train_v3.csv"]:
        infra = build(corpus, enc, pool)
        bm25, emb, G = infra
        for date in ["2026-08-28", "now"]:
            importlib.reload(pc)
            if date != "now":
                pc.recency_weight = fixed_recency(date)
            tag = f"v3 pool={pool.split('/')[-1]} ref={date}"
            run_linear_and_rrf(tag, Q3, L3, corpus, enc, infra, W3, 3, ref3)
            n1 = [int(ev.run_N1_linear_at_shortlist(q, corpus, bm25, enc, emb, G, W3, 3) == l) for q, l in zip(Q3, L3)]
            n2 = [int(ev.run_N2_no_recency_tiebreak(q, corpus, bm25, enc, emb, G, 3, 60) == l) for q, l in zip(Q3, L3)]
            report(tag, "N1 (Linear+KG @ shortlist)", n1, ref3.get("N1 (Linear+KG @ shortlist)"))
            report(tag, "N2 (Proposed, no recency tiebreak)", n2, ref3.get("N2 (Proposed, no recency tiebreak)"))


if __name__ == "__main__":
    main()
