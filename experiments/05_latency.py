"""Latency diagnostic (PROTOCOL.md section 10.10).

    venv\\Scripts\\python experiments\\05_latency.py

Implemented and run AFTER the one-shot final evaluation (disclosed in
results/rebuild/FINAL_PAPER_DISCLOSURES.md). It uses query texts only, never
labels, and cannot change any score.

Configuration timed: the deployed final system (index = the 123 development
rows; MedCPT query encoder at the selected length; SapBERT linker at the
selected tau; alpha, beta from configs/selected_params.json). Queries: the
1,060 primary final-evaluation queries (clinical + lexicon B lay), one query
at a time (batch size 1), as in interactive use.

Per query, the stages of the proposed Hybrid+KG-linked system are timed
separately with time.perf_counter, synchronising the GPU after every GPU
stage. Models are called directly (Encoder._forward), so the embedding cache
never shortcuts encoding. Index construction (document and node embeddings,
BM25 statistics, KG) is offline and not timed. WARMUP queries are run first
and discarded; then REPEATS full passes are timed. Reported: median, p95
and mean in milliseconds, per stage and in total.

Correctness guard: for every query, the fused scores from the timed code path
must equal the scores of the evaluated pipeline (hsr.scoring.score_components).

Writes latency_diagnostics.csv and latency_environment.json to results/rebuild/test/ (the published run)
or to the folder given with --out, e.g. results/reproduction/latency, when replicating on other hardware.
"""
import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr.fusion import linear_hybrid, linear_kg  # noqa: E402
from hsr.metrics import ROUND_DECIMALS  # noqa: E402
from hsr.protocol import FINAL_DIR, SELECTED_PATH, Setup, build_encoders, git_state, load_checked_lexicon  # noqa: E402
from hsr.queries import GROUPS, build_row_queries, segments  # noqa: E402
from hsr.scoring import FoldIndex, maxp, score_components  # noqa: E402
from hsr.systems import Params  # noqa: E402

REPEATS = 5
WARMUP = 20
STAGES = ["query_encoding", "bm25", "dense_scoring", "linking_exact", "linking_semantic", "kg_scoring", "fusion_ranking"]
TOTALS = {
    "total_proposed": STAGES,
    "total_hybrid_only": ["query_encoding", "bm25", "dense_scoring", "fusion_ranking"],
    "kg_overhead": ["linking_exact", "linking_semantic", "kg_scoring"],
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_query(fi, enc_q, sap, node_emb, names, p, text):
    """One query through the proposed system; returns (stage -> seconds, fused scores)."""
    n = len(fi.diseases)
    t = {}
    sync()
    t0 = time.perf_counter()
    q = enc_q._forward([text])[0]
    sync()
    t1 = time.perf_counter()
    bm = maxp(fi.bm25.scores(text), fi.doc_disease, n)
    t2 = time.perf_counter()
    de = maxp(fi.doc_emb @ q, fi.doc_disease, n)
    t3 = time.perf_counter()
    linked = set(fi.exact_linker.link(text))
    t4 = time.perf_counter()
    segs = segments(text)
    if segs:
        E = np.concatenate([sap._forward([s]) for s in segs])
        sync()
        sims = E @ node_emb.T
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(segs)), best]
        linked |= {names[j] for j, v in zip(best, best_sim) if v >= p.tau}
    t5 = time.perf_counter()
    K = fi.kg.score(linked)
    t6 = time.perf_counter()
    F = linear_kg(linear_hybrid(bm, de, p.alpha), K, p.beta)
    np.lexsort((np.arange(n), -np.round(F, ROUND_DECIMALS)))  # the final ranking
    t7 = time.perf_counter()
    for name, a, b in zip(STAGES, [t0, t1, t2, t3, t4, t5, t6], [t1, t2, t3, t4, t5, t6, t7]):
        t[name] = b - a
    return t, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FINAL_DIR),
                    help="output folder (default: results/rebuild/test, the published run); use another folder to replicate")
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Setup()
    proto = s.protocol
    p = Params(**json.load(open(SELECTED_PATH))["params"])
    lexB, _ = load_checked_lexicon(proto["data"]["lexicon_test"], "B", s.rows)
    queries = [q for r in s.test_rows for q in build_row_queries(r, s.seed, lexB, s.partial_ks)]
    enc = build_encoders(proto, [p.linker], [p.query_max_length])
    enc_q, sap = enc["query"][p.query_max_length], enc["linkers"][p.linker]
    fi = FoldIndex(s.dev_rows, s.corpus, s.diseases, enc["doc"], enc["linkers"], classifier_C=(p.C,))
    linker = fi.linkers[p.linker]
    node_emb, names = linker.node_emb, linker.names

    # Correctness guard: the timed path computes exactly what was evaluated.
    max_diff = 0.0
    for q in queries:
        _, F = timed_query(fi, enc_q, sap, node_emb, names, p, q.text)
        comp = score_components(fi, q.text, {p.query_max_length: enc_q})
        K = fi.kg.score(comp.links[p.linker].linked(p.tau))
        F_eval = linear_kg(linear_hybrid(comp.bm25, comp.dense[p.query_max_length], p.alpha), K, p.beta)
        max_diff = max(max_diff, float(np.abs(F - F_eval).max()))
    assert max_diff < 1e-6, f"timed path differs from the evaluated pipeline (max diff {max_diff})"

    for q in queries[:WARMUP]:
        timed_query(fi, enc_q, sap, node_emb, names, p, q.text)
    recs = []
    for rep in range(REPEATS):
        for q in queries:
            t, _ = timed_query(fi, enc_q, sap, node_emb, names, p, q.text)
            recs.append({"repeat": rep, "query_id": q.query_id, "group": q.group,
                         "n_segments": len(segments(q.text)), **{k: v * 1000 for k, v in t.items()}})
    raw = pd.DataFrame(recs)
    for name, parts in TOTALS.items():
        raw[name] = raw[parts].sum(axis=1)

    rows = []
    for grp, g in [("all", raw)] + [(x, raw[raw.group == x]) for x in GROUPS]:
        for stage in STAGES + list(TOTALS):
            v = g[stage].to_numpy()
            rows.append({"group": grp, "stage": stage, "n_measurements": len(v), "median_ms": np.median(v),
                         "p95_ms": np.percentile(v, 95), "mean_ms": v.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "latency_diagnostics.csv", index=False, lineterminator="\n")

    env = {
        "note": "Implemented and run after the one-shot final evaluation; query texts only, no labels.",
        "date": datetime.datetime.now().isoformat(timespec="seconds"), "git": git_state(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__,
        "queries": len(queries), "repeats": REPEATS, "warmup_queries": WARMUP, "batch_size": 1,
        "params": {"query_max_length": p.query_max_length, "linker": p.linker, "tau": p.tau,
                   "alpha": p.alpha, "beta": p.beta},
        "segments_per_query": {g: float(raw[raw.group == g].n_segments.mean()) for g in GROUPS},
        "correctness_guard_max_abs_diff": max_diff,
    }
    with open(out_dir / "latency_environment.json", "w", newline="\n") as f:
        json.dump(env, f, indent=2)

    pd.set_option("display.width", 200)
    print(f"device {env['device']} | {len(queries)} queries x {REPEATS} repeats | guard max diff {max_diff:.2e}")
    print(out[out.group == "all"].round(3).to_string(index=False))
    print("\ntotal_proposed / kg_overhead median ms by group:")
    print(out[out.stage.isin(["total_proposed", "total_hybrid_only", "kg_overhead", "linking_semantic"])]
          .pivot(index="group", columns="stage", values="median_ms").round(2).to_string())


if __name__ == "__main__":
    main()
