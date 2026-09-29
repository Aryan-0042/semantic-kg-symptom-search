import inspect
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # moved from the project root to legacy/scripts/ at release packaging
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rank_bm25 import BM25Okapi
import pipeline_common as p
from pipeline_common import (
    load_guideline_corpus,
    load_symptom_split,
    build_queries_from_split,
    build_retrieval_text,
    build_kg,
    MedCPTEncoder,
    warmup,
    run_condition_rrf_kg_recency,
    RRF_K_DEFAULT,
)

OUT_TXT = ROOT / "results" / "rrf_shortlist_diagnostic_v3.txt"
OUT_CSV = ROOT / "results" / "rrf_shortlist_diagnostic_v3.csv"

logs = []

def log(msg=""):
    print(msg)
    logs.append(str(msg))

def main():
    log("=" * 80)
    log("RRF SHORTLIST DIAGNOSTIC — V3")
    log("=" * 80)

    corpus_df = load_guideline_corpus("data/guideline_corpus.jsonl")
    dev_train = load_symptom_split("data/split_dev_train_v3.csv")
    dev_val = load_symptom_split("data/split_dev_val_v3.csv")
    val_queries, val_labels = build_queries_from_split(dev_val)

    log("\n[1] DATA / CORPUS")
    log(f"Corpus documents: {len(corpus_df)}")
    log(f"Retrieval rows expected: {len(corpus_df)}")
    log(f"Dev-train rows: {len(dev_train)}")
    log(f"Dev-val rows: {len(dev_val)}")
    log(f"Validation queries: {len(val_queries)}")
    log(f"Validation diseases: {len(set(val_labels))}")
    log(f"RRF k: {RRF_K_DEFAULT}")

    log("\n[2] RRF IMPLEMENTATION CHECK")
    src = inspect.getsource(p.run_condition_rrf_kg_recency)
    relevant = []
    for i, line in enumerate(src.splitlines(), 1):
        low = line.lower()
        if ("shortlist" in low or "reranked" in low or
                "top_k" in low or "argsort(rrf_scores)" in low):
            relevant.append(f"{i:03d}: {line}")
    for line in relevant:
        log(line)
    log(f"\nContains [:shortlist_n]: {'[:shortlist_n]' in src}")
    log(f"Passes shortlist to kg_rerank(): {'kg_rerank(shortlist' in src}")

    log("\n[3] BUILDING REAL V3 DEV-TRAIN RETRIEVAL INFRASTRUCTURE")
    retrieval_texts = build_retrieval_text(corpus_df, dev_train)
    log(f"Retrieval texts: {len(retrieval_texts)}")

    bm25 = BM25Okapi([t.lower().split() for t in retrieval_texts])

    encoder = MedCPTEncoder()
    log("Encoding guideline corpus...")
    doc_embeddings = encoder.encode_docs(retrieval_texts, batch_size=16)

    graph = build_kg(corpus_df, dev_train)
    log("Warming up...")
    warmup(bm25, encoder)
    log("Warm-up complete.")

    log("\n[4] DECISIVE 5 VS 50 CHECK")
    checks = []
    for shortlist_n in (5, 50):
        log(f"\nRunning shortlist_n={shortlist_n}...")
        result, per_query = run_condition_rrf_kg_recency(
            val_queries, val_labels, corpus_df,
            bm25, encoder, doc_embeddings, graph,
            shortlist_n=shortlist_n,
            rrf_k=RRF_K_DEFAULT,
            verbose=False,
        )
        checks.append((shortlist_n, result, per_query))
        log(
            f"shortlist_n={shortlist_n} | "
            f"Accuracy={result['Accuracy']:.2f} | "
            f"F1={result['F1']:.2f} | "
            f"MRR={result['MRR']:.3f} | "
            f"Correct={sum(per_query)}/{len(per_query)}"
        )

    a = checks[0]
    b = checks[1]
    same_metrics = (
        a[1]["Accuracy"] == b[1]["Accuracy"]
        and a[1]["F1"] == b[1]["F1"]
        and a[1]["MRR"] == b[1]["MRR"]
    )
    same_correctness = a[2] == b[2]

    log(f"\n5 vs 50 identical aggregate metrics: {same_metrics}")
    log(f"5 vs 50 identical per-query correctness: {same_correctness}")

    if not same_correctness:
        diffs = [i for i, (x, y) in enumerate(zip(a[2], b[2])) if x != y]
        log(f"Differing validation query indices: {diffs}")

    log("\n[5] FULL 10/15/20/30/50 RECHECK")
    rows = []
    for shortlist_n in (10, 15, 20, 30, 50):
        log(f"Running shortlist_n={shortlist_n}...")
        result, per_query = run_condition_rrf_kg_recency(
            val_queries, val_labels, corpus_df,
            bm25, encoder, doc_embeddings, graph,
            shortlist_n=shortlist_n,
            rrf_k=RRF_K_DEFAULT,
            verbose=False,
        )
        rows.append({
            "shortlist_n": shortlist_n,
            "Accuracy": result["Accuracy"],
            "F1": result["F1"],
            "MRR": result["MRR"],
            "Correct_count": int(sum(per_query)),
            "N": len(per_query),
        })
        log(
            f"  {shortlist_n:>2}: "
            f"Acc={result['Accuracy']:.2f}, "
            f"F1={result['F1']:.2f}, "
            f"MRR={result['MRR']:.3f}, "
            f"Correct={sum(per_query)}/{len(per_query)}"
        )

    grid = pd.DataFrame(rows)
    log("\nGrid results:")
    log(grid.to_string(index=False))

    log("\n[6] INTERPRETATION")
    if len(corpus_df) < 10:
        log("Corpus-size explanation is possible: corpus < 10.")
    else:
        log(
            "Corpus has >=10 documents, so shortlist 10..50 genuinely "
            "truncate the corpus."
        )

    if same_correctness:
        log(
            "5 and 50 have identical per-query correctness. "
            "This requires interpretation from the actual implementation: "
            "either the extra candidates never change the disease decision, "
            "or shortlist_n is not affecting the effective decision path."
        )
    else:
        log("5 and 50 produce different predictions; shortlist_n is affecting the decision path.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(logs) + "\n", encoding="utf-8")
    log(f"\nSaved: {OUT_CSV}")
    log(f"Saved: {OUT_TXT}")

if __name__ == "__main__":
    main()