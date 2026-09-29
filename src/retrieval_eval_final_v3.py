# retrieval_eval_final_v3.py
"""
THE CONFIRMATORY EVALUATION, v3. Run this exactly once, against the
frozen 123-query test set (split_test_final_v3.csv), using the frozen
configs from Phase 2. Retrieval infrastructure (BM25 index, MedCPT
embeddings, KG) is built from the FULL dev pool (split_dev_v3.csv, 181
rows) -- standard practice: hyperparameters were selected on
dev-train/dev-val, the final model is refit on dev-train+dev-val
combined before touching test.

Runs 8 conditions: the 6 primary methods plus two attribution controls
(N1, N2) that decompose the accuracy gap into candidate-pool-size vs.
fusion-rule vs. recency, per the methodology plan agreed before any v3
result existed.

Whatever this script outputs is the result. Do not re-run it with
different frozen configs after seeing its output.
"""
import json
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from statsmodels.stats.contingency_tables import mcnemar

from pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_queries_from_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    run_condition,
    run_condition_rrf_kg_recency,
    crfs_score,
    reciprocal_rank_fusion,
    kg_rerank,
)


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci_on_difference(correct_a, correct_b, n_boot=2000, seed=42):
    """Percentile bootstrap CI on the accuracy DIFFERENCE (a - b),
    resampling the paired per-query correctness vectors together."""
    rng = np.random.RandomState(seed)
    a = np.array(correct_a)
    b = np.array(correct_b)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs) * 100
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean())


def run_N1_linear_at_shortlist(query, corpus_df, bm25, encoder, doc_embeddings, graph,
                                crfs_weights, shortlist_n, top_k=20):
    """N1: linear fusion restricted to the same shortlist_n RRF would use,
    scored WITHOUT recency (per the attribution design: isolate pool-size
    from fusion-rule, holding recency out of both sides of that comparison).
    """
    from pipeline_common import kg_relevance
    n_docs = len(corpus_df)

    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    lo, hi = bm25_scores.min(), bm25_scores.max()
    bm25_scores = (bm25_scores - lo) / (hi - lo) if hi > lo else np.zeros(n_docs)

    q_emb = encoder.encode_queries([query])[0]
    dense_scores = doc_embeddings @ q_emb
    lo, hi = dense_scores.min(), dense_scores.max()
    dense_scores = (dense_scores - lo) / (hi - lo) if hi > lo else np.zeros(n_docs)

    rrf_for_shortlist = reciprocal_rank_fusion(bm25_scores, dense_scores, k=60)
    shortlist = set(np.argsort(rrf_for_shortlist)[::-1][:shortlist_n].tolist())

    kg_scores = np.array([
        kg_relevance(query, corpus_df.iloc[i]["disease_tag"], graph) if i in shortlist else 0.0
        for i in range(n_docs)
    ])
    bm25_masked = np.array([bm25_scores[i] if i in shortlist else 0.0 for i in range(n_docs)])
    dense_masked = np.array([dense_scores[i] if i in shortlist else 0.0 for i in range(n_docs)])

    # Renormalize the three active weights to sum to 1, since w_recency=0.103
    # is being excluded entirely (not just multiplied by zero) -- using the
    # raw frozen weights unmodified here would silently under-weight the
    # three retained signals relative to how CRFS actually uses them.
    w_bm25, w_dense, w_kg, _ = crfs_weights
    active_sum = w_bm25 + w_dense + w_kg
    w_bm25_n, w_dense_n, w_kg_n = w_bm25 / active_sum, w_dense / active_sum, w_kg / active_sum

    fused = w_bm25_n * bm25_masked + w_dense_n * dense_masked + w_kg_n * kg_scores

    top_indices = np.argsort(fused)[::-1][:top_k]
    disease_scores = {}
    for rank, idx in enumerate(top_indices, start=1):
        if idx not in shortlist:
            continue
        disease = str(corpus_df.iloc[idx]["disease_tag"]).strip()
        disease_scores[disease] = disease_scores.get(disease, 0.0) + float(fused[idx]) * (1.0 / rank)
    ranking = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
    return ranking[0][0] if ranking else ""

def run_N2_no_recency_tiebreak(query, corpus_df, bm25, encoder, doc_embeddings, graph,
                                shortlist_n, rrf_k, top_k=20):
    """N2: identical to Proposed, except kg_rerank's tiebreak uses ONLY
    RRF score (no recency), isolating recency's marginal contribution
    inside the tiebreak role specifically."""
    n_docs = len(corpus_df)
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    q_emb = encoder.encode_queries([query])[0]
    dense_scores = doc_embeddings @ q_emb

    rrf_scores = reciprocal_rank_fusion(bm25_scores, dense_scores, k=rrf_k)
    shortlist = np.argsort(rrf_scores)[::-1][:shortlist_n]

    # Rerank by KG relevance, tiebreak by RRF score ONLY (no recency) --
    # reuse kg_rerank's KG-relevance ordering but drop the recency key.
    from pipeline_common import kg_relevance
    scored = []
    for idx in shortlist:
        disease = corpus_df.iloc[idx]["disease_tag"]
        kg_score = kg_relevance(query, disease, graph)
        scored.append((idx, kg_score, float(rrf_scores[idx])))
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)  # KG desc, then RRF desc -- NO date key
    reranked = [t[0] for t in scored]

    disease_scores = {}
    for rank, idx in enumerate(reranked[:top_k], start=1):
        disease = str(corpus_df.iloc[idx]["disease_tag"]).strip()
        disease_scores[disease] = disease_scores.get(disease, 0.0) + float(rrf_scores[idx]) * (1.0 / rank)
    ranking = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
    return ranking[0][0] if ranking else ""


def main():
    print("=" * 70)
    print("CONFIRMATORY EVALUATION v3 -- run once, 123-query frozen test set")
    print("=" * 70)

    with open("results/frozen_config_v3.json") as f:
        crfs_frozen = json.load(f)
    w = crfs_frozen["selected_weights"]
    crfs_weights = (w["w_bm25"], w["w_dense"], w["w_kg"], w["w_recency"])

    with open("results/frozen_config_rrf_v3.json") as f:
        rrf_frozen = json.load(f)
    shortlist_n = rrf_frozen["selected_shortlist_n"]
    rrf_k = rrf_frozen["rrf_k"]

    print(f"Frozen CRFS weights: {crfs_weights}")
    print(f"Frozen RRF config: shortlist_n={shortlist_n}, rrf_k={rrf_k}")

    corpus_df = load_guideline_corpus()
    dev_full = load_symptom_split("data/split_dev_v3.csv")   # 181 rows, train+val combined
    test_df = pd.read_csv("data/split_test_final_v3.csv")     # 123 rows, NEVER used for tuning

    test_queries = test_df["query_text"].tolist()
    test_labels = test_df["Disease"].astype(str).str.strip().tolist()
    paraphrased_mask = test_df["paraphrased"].astype(bool).to_numpy()

    print(f"\nTest queries: {len(test_queries)} "
          f"({paraphrased_mask.sum()} paraphrased, {(~paraphrased_mask).sum()} canonical)")

    print("\nBuilding retrieval infrastructure from FULL dev pool (v3)...")
    retrieval_texts = build_retrieval_text(corpus_df, dev_full)
    bm25 = BM25Okapi([t.lower().split() for t in retrieval_texts])
    encoder = MedCPTEncoder()
    doc_embeddings = encoder.encode_docs(retrieval_texts, batch_size=16)
    graph = build_kg(corpus_df, dev_full)
    warmup(bm25, encoder)

    linear_methods = ["BM25-only", "Dense-only", "Hybrid", "Hybrid+KG (no recency)", "Proposed (CRFS)"]
    per_query_by_method = {}
    comparison_rows = []

    print("\nRunning primary methods...\n")
    for method in linear_methods:
        result, per_query = run_condition(
            method, test_queries, test_labels, corpus_df,
            bm25, encoder, doc_embeddings, graph, weights=crfs_weights, verbose=False,
        )
        comparison_rows.append(result)
        per_query_by_method[method] = per_query
        print(f"  {method:28s} Acc={result['Accuracy']:6.2f}  F1={result['F1']:6.2f}  MRR={result['MRR']:.3f}")

    print("\nRunning Proposed (RRF+KG+Recency)...\n")
    rrf_result, rrf_per_query = run_condition_rrf_kg_recency(
        test_queries, test_labels, corpus_df, bm25, encoder, doc_embeddings, graph,
        shortlist_n=shortlist_n, rrf_k=rrf_k, verbose=False,
    )
    comparison_rows.append(rrf_result)
    per_query_by_method["Proposed (RRF+KG+Recency)"] = rrf_per_query
    print(f"  {'Proposed (RRF+KG+Recency)':28s} Acc={rrf_result['Accuracy']:6.2f}  "
          f"F1={rrf_result['F1']:6.2f}  MRR={rrf_result['MRR']:.3f}")

    print("\nRunning attribution conditions N1, N2...\n")
    n1_correct, n2_correct = [], []
    for query, true_label in zip(test_queries, test_labels):
        pred_n1 = run_N1_linear_at_shortlist(
            query, corpus_df, bm25, encoder, doc_embeddings, graph, crfs_weights, shortlist_n
        )
        pred_n2 = run_N2_no_recency_tiebreak(
            query, corpus_df, bm25, encoder, doc_embeddings, graph, shortlist_n, rrf_k
        )
        n1_correct.append(1 if pred_n1 == true_label else 0)
        n2_correct.append(1 if pred_n2 == true_label else 0)

    per_query_by_method["N1 (Linear+KG @ shortlist)"] = n1_correct
    per_query_by_method["N2 (Proposed, no recency tiebreak)"] = n2_correct

    for label, pq in [("N1 (Linear+KG @ shortlist)", n1_correct), ("N2 (Proposed, no recency tiebreak)", n2_correct)]:
        acc = 100 * sum(pq) / len(pq)
        print(f"  {label:38s} Acc={acc:6.2f}  Correct={sum(pq)}/{len(pq)}")

    import os
    os.makedirs("results", exist_ok=True)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv("results/retrieval_comparison_v3.csv", index=False)

    attribution_rows = [
        {"Condition": "Hybrid+KG (no recency)", "Accuracy": comparison_df.loc[comparison_df.Method=="Hybrid+KG (no recency)","Accuracy"].iloc[0], "Pool": "full", "Fusion": "linear", "Recency": "no"},
        {"Condition": "N1 (Linear+KG @ shortlist)", "Accuracy": round(100*sum(n1_correct)/len(n1_correct),2), "Pool": f"top-{shortlist_n}", "Fusion": "linear", "Recency": "no"},
        {"Condition": "N2 (Proposed, no recency tiebreak)", "Accuracy": round(100*sum(n2_correct)/len(n2_correct),2), "Pool": f"top-{shortlist_n}", "Fusion": "RRF", "Recency": "no"},
        {"Condition": "Proposed (RRF+KG+Recency)", "Accuracy": rrf_result["Accuracy"], "Pool": f"top-{shortlist_n}", "Fusion": "RRF", "Recency": "yes (tiebreak)"},
    ]
    pd.DataFrame(attribution_rows).to_csv("results/ablation_attribution_v3.csv", index=False)

    n = len(test_labels)
    stats_rows = []
    for method, pq in per_query_by_method.items():
        successes = sum(pq)
        lo, hi = wilson_ci(successes, n)
        stats_rows.append({"Method": method, "N": n, "Correct": successes,
                            "Accuracy": round(100*successes/n, 2),
                            "Wilson_95CI_low": round(100*lo,2), "Wilson_95CI_high": round(100*hi,2)})
    pd.DataFrame(stats_rows).to_csv("results/final_test_confidence_intervals_v3.csv", index=False)

    subset_rows = []
    for method, pq in per_query_by_method.items():
        pq_arr = np.array(pq)
        for subset_name, mask in [("canonical", ~paraphrased_mask), ("paraphrased", paraphrased_mask)]:
            n_sub = int(mask.sum())
            correct_sub = int(pq_arr[mask].sum())
            subset_rows.append({"Method": method, "Subset": subset_name, "N": n_sub, "Correct": correct_sub,
                                 "Accuracy": round(100*correct_sub/n_sub, 2) if n_sub else float("nan")})
    pd.DataFrame(subset_rows).to_csv("results/subset_breakdown_v3.csv", index=False)

    def mcnemar_paired(a_name, b_name):
        a, b = np.array(per_query_by_method[a_name]), np.array(per_query_by_method[b_name])
        n01 = int(((a==0)&(b==1)).sum()); n10 = int(((a==1)&(b==0)).sum())
        p = 1.0 if (n01==0 and n10==0) else mcnemar([[0,n01],[n10,0]], exact=True).pvalue
        return n01, n10, p

    comparisons = [
        ("Hybrid", "Proposed (RRF+KG+Recency)"),
        ("Hybrid+KG (no recency)", "Proposed (RRF+KG+Recency)"),
        ("Proposed (CRFS)", "Proposed (RRF+KG+Recency)"),
        ("Dense-only", "Proposed (RRF+KG+Recency)"),
        ("Hybrid+KG (no recency)", "N1 (Linear+KG @ shortlist)"),
        ("N1 (Linear+KG @ shortlist)", "Proposed (RRF+KG+Recency)"),
        ("N2 (Proposed, no recency tiebreak)", "Proposed (RRF+KG+Recency)"),
    ]
    sig_rows = []
    boot_rows = []
    for a, b in comparisons:
        n01, n10, p = mcnemar_paired(a, b)
        sig_rows.append({"comparison": f"{a} vs {b}", "n_pairs": n, "b_right_a_wrong": n01,
                          "a_right_b_wrong": n10, "mcnemar_exact_p": p})
        lo, hi, mean_diff = bootstrap_ci_on_difference(per_query_by_method[b], per_query_by_method[a])
        boot_rows.append({"comparison": f"{b} minus {a}", "bootstrap_mean_diff_pp": round(mean_diff,2),
                           "bootstrap_95CI_low_pp": round(lo,2), "bootstrap_95CI_high_pp": round(hi,2)})
    pd.DataFrame(sig_rows).to_csv("results/significance_tests_v3.csv", index=False)
    pd.DataFrame(boot_rows).to_csv("results/bootstrap_ci_v3.csv", index=False)

    raw_rows = []
    for method, pq in per_query_by_method.items():
        for i, c in enumerate(pq):
            raw_rows.append({"method": method, "query_index": i, "true_label": test_labels[i],
                              "paraphrased": bool(paraphrased_mask[i]), "correct": c})
    pd.DataFrame(raw_rows).to_csv("results/per_query_predictions_v3.csv", index=False)

    # Per-disease agreement supplementary table
    per_disease_rows = []
    for disease in sorted(set(test_labels)):
        idxs = [i for i, d in enumerate(test_labels) if d == disease]
        row = {"Disease": disease, "N": len(idxs)}
        for method, pq in per_query_by_method.items():
            row[method] = sum(pq[i] for i in idxs)
        per_disease_rows.append(row)
    pd.DataFrame(per_disease_rows).to_csv("results/per_disease_agreement_v3.csv", index=False)

    print("\n" + "!" * 70)
    print("THESE ARE THE FINAL v3 CONFIRMATORY NUMBERS. Do not re-run against")
    print("this test set with different frozen configs after seeing them.")
    print("!" * 70)
    print("\nFiles written: retrieval_comparison_v3.csv, ablation_attribution_v3.csv,")
    print("final_test_confidence_intervals_v3.csv, subset_breakdown_v3.csv,")
    print("significance_tests_v3.csv, bootstrap_ci_v3.csv, per_query_predictions_v3.csv,")
    print("per_disease_agreement_v3.csv")


if __name__ == "__main__":
    main()
