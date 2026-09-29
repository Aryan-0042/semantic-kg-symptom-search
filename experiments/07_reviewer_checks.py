"""POST-HOC robustness checks prompted by a mock peer review (not pre-registered).

    venv\\Scripts\\python experiments\\07_reviewer_checks.py

Nothing is tuned and the frozen system is unchanged. The test queries (lexicon B
and lexicon C, lay conditions) are re-scored with configs/selected_params.json,
and the script first verifies that every system's per-query rank reproduces the
committed results exactly. Then:

  1. untouched records   H1-H4 on the test records never queried in any earlier
                         project version (audit/outputs/test_overlap_by_row.csv)
  2. two-way bootstrap   H1-H4 intervals that resample conditions AND the 131
                         symptom phrasings (query weight = condition weight x mean
                         weight of its symptoms), for crossed phrase dependence
  3. tie bounds          graph-only systems with ties resolved best / worst
  4. pool analysis       hybrid recall@n (the ceiling of any pool) and clustered CIs
                         for linear minus RRF at each pool size
  5. truncation          H3/H4 with MedCPT reading the whole query (512 tokens) in
                         both systems, alpha/beta/tau unchanged

Writes results/rebuild/posthoc_reviewer_checks/.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr import RESULTS, ROOT  # noqa: E402
from hsr.metrics import rank_matrix  # noqa: E402
from hsr.protocol import SELECTED_PATH, Setup, build_encoders, load_checked_lexicon  # noqa: E402
from hsr.queries import build_row_queries  # noqa: E402
from hsr.runner import compute_components  # noqa: E402
from hsr.stats import cluster_bootstrap_ci, cluster_signflip_test  # noqa: E402
from hsr.systems import ALL_SYSTEMS, Params, evaluate, pool_n, pool_system_scores, system_scores  # noqa: E402

OUT = RESULTS / "rebuild" / "posthoc_reviewer_checks"
LABEL = "POST-HOC (not pre-registered)"
COMPARISONS = [("H1", "Hybrid+KG-exact", "lay_full"), ("H2", "Hybrid+KG-exact", "lay_partial"),
               ("H3", "Hybrid", "lay_full"), ("H4", "Hybrid", "lay_partial")]
PROPOSED = "Hybrid+KG-linked"


def ranks(records, p, system):
    return rank_matrix(np.stack([system_scores(r, p, system) for r in records]), [r.true_idx for r in records])["rank"]


def two_way_ci(d, cond, sym_sets, n_sym, n_boot, seed, level=0.95):
    """Percentile CI of sum(w d)/sum(w), resampling conditions and symptoms with replacement."""
    rng = np.random.default_rng(seed)
    _, c_idx = np.unique(cond, return_inverse=True)
    n_c = c_idx.max() + 1
    M = np.zeros((len(d), n_sym))
    for i, ss in enumerate(sym_sets):
        M[i, ss] = 1.0 / len(ss)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        wc = np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)
        ws = np.bincount(rng.integers(0, n_sym, n_sym), minlength=n_sym)
        w = wc[c_idx] * (M @ ws)
        stats[b] = (w * d).sum() / w.sum()
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def main():
    s = Setup()
    proto = s.protocol
    st, seeds = proto["statistics"], proto["seeds"]
    p = Params(**json.load(open(SELECTED_PATH))["params"])
    lexB, _ = load_checked_lexicon(proto["data"]["lexicon_test"], "B", s.rows)
    lexC, _ = load_checked_lexicon("data/lexicons/lexicon_B_human.csv", "H", s.rows)
    dev_ids = {r.row_id for r in s.dev_rows}

    def index_rows_for(r):
        assert r.row_id not in dev_ids
        return s.dev_rows

    def queries_for(r):
        return [q for lex in (lexB, lexC) for q in build_row_queries(r, s.seed, lex, s.partial_ks)
                if q.condition.startswith("lay_")]

    enc = build_encoders(proto, [p.linker], [p.query_max_length, 512])
    records = compute_components(list(s.test_rows), index_rows_for, queries_for, s.corpus, s.diseases, enc,
                                 [p.tau], [p.C], log_every=0)
    df = evaluate(records, p, ALL_SYSTEMS)

    # ---- 0. exact reproduction of the committed per-query ranks ----
    committed = pd.concat([pd.read_csv(RESULTS / "rebuild" / "test" / "per_query.csv"),
                           pd.read_csv(RESULTS / "rebuild" / "test_human_lexicon" / "per_query.csv")])
    committed = committed[committed.lexicon.isin(["B", "H"])].set_index(["query_id", "system"])["rank"]
    mine = df.set_index(["query_id", "system"])["rank"]
    assert len(mine) == len(committed) and (committed.loc[mine.index] == mine).all(), "re-scoring does not reproduce"
    print(f"reproduction: {len(mine)} per-query ranks identical to the committed results")

    OUT.mkdir(parents=True, exist_ok=True)
    sym_index = {k: i for i, k in enumerate(sorted({k for r in s.rows for k in r.symptoms}))}
    by_lex = {lx: [r for r in records if r.query.lexicon == lx] for lx in ("B", "H")}

    # ---- 1 + 2. untouched records and two-way bootstrap ----
    overlap = pd.read_csv(ROOT / "audit" / "outputs" / "test_overlap_by_row.csv")
    untouched = set(overlap.row_id[overlap.n_historical_retrieval_sets == 0])
    rows = []
    for lx, recs in by_lex.items():
        for hid, ctrl, grp in COMPARISONS:
            sub = [r for r in recs if r.query.group == grp]
            d = (1.0 / ranks(sub, p, PROPOSED)) - (1.0 / ranks(sub, p, ctrl))
            cond = np.array([r.query.disease for r in sub])
            syms = [[sym_index[k] for k in r.query.gold_symptoms] for r in sub]
            lo1, hi1 = cluster_bootstrap_ci(d, cond, st["n_boot"], seeds["bootstrap"])[1:]
            lo2, hi2 = two_way_ci(d, cond, syms, len(sym_index), st["n_boot"], seeds["bootstrap"])
            keep = np.array([r.query.row_id in untouched for r in sub])
            du, cu = d[keep], cond[keep]
            obs_u, p_u = cluster_signflip_test(du, cu, st["n_perm"], seeds["permutation"])
            _, lo_u, hi_u = cluster_bootstrap_ci(du, cu, st["n_boot"], seeds["bootstrap"])
            rows.append({"analysis": LABEL, "lexicon": {"B": "B (Gemini)", "H": "C (co-author)"}[lx], "id": hid,
                         "control": ctrl, "group": grp, "n_queries": len(d), "delta_mrr": d.mean(),
                         "ci_condition": f"[{lo1:.3f}, {hi1:.3f}]", "ci_two_way": f"[{lo2:.3f}, {hi2:.3f}]",
                         "two_way_excludes_zero": lo2 > 0,
                         "untouched_n_queries": int(keep.sum()), "untouched_n_conditions": len(np.unique(cu)),
                         "untouched_delta_mrr": obs_u, "untouched_ci": f"[{lo_u:.3f}, {hi_u:.3f}]",
                         "untouched_p_uncorrected": p_u})
    checks = pd.DataFrame(rows)
    checks.to_csv(OUT / "untouched_and_two_way.csv", index=False, lineterminator="\n")

    # ---- 3. tie bounds for graph-only systems ----
    tb = df[df.system.isin(["KG-exact", "KG-linked", PROPOSED])].assign(
        rr_best=lambda x: 1 / x.rank_best, rr_worst=lambda x: 1 / x.rank_worst)
    tb = tb.groupby(["lexicon", "group", "system"])[["rr_worst", "rr", "rr_best"]].mean().reset_index()
    tb.insert(0, "analysis", "tie bounds (descriptive)")
    tb.to_csv(OUT / "tie_bounds.csv", index=False, lineterminator="\n")

    # ---- 4. pool analysis: hybrid recall@n and CIs for linear - RRF ----
    spec = proto["secondary_analyses"]["pool_size"]
    recs = by_lex["B"]
    prow, committed_pool = [], pd.read_csv(RESULTS / "rebuild" / "test" / "pool_analysis.csv")
    committed_pool = committed_pool[committed_pool.lexicon == "B"].set_index(["group", "kg", "rule", "n"]).rr
    hyb_rank = ranks(recs, p, "Hybrid")
    for grp in ("lay_full", "lay_partial"):
        m = np.array([r.query.group == grp for r in recs])
        cond = np.array([r.query.disease for r in recs])[m]
        for n in spec["n"]:
            nn = pool_n(n, len(s.diseases))
            rr = {}
            for kg, rule in product(spec["kg"], spec["rules"]):
                beta = p.pool_beta[f"{kg}:{n}"] if rule == "linear" else None
                S = np.stack([pool_system_scores(r, p, kg, nn, rule, beta) for r, keep in zip(recs, m) if keep])
                rr[(kg, rule)] = 1.0 / rank_matrix(S, [r.true_idx for r, keep in zip(recs, m) if keep])["rank"]
                assert abs(rr[(kg, rule)].mean() - committed_pool[(grp, kg, rule, str(n))]) < 1e-12
            diff = rr[("linked", "linear")] - rr[("linked", "rrf")]
            dm, lo, hi = cluster_bootstrap_ci(diff, cond, st["n_boot"], seeds["bootstrap"])
            prow.append({"analysis": LABEL, "group": grp, "n": str(n), "hybrid_recall_at_n": float((hyb_rank[m] <= nn).mean()),
                         "mrr_linked_linear": rr[("linked", "linear")].mean(), "mrr_linked_rrf": rr[("linked", "rrf")].mean(),
                         "delta_linear_minus_rrf": dm, "ci_low": lo, "ci_high": hi, "ci_excludes_zero": lo > 0})
    pool = pd.DataFrame(prow)
    pool.to_csv(OUT / "pool_recall_and_ci.csv", index=False, lineterminator="\n")

    # ---- 5. truncation: MedCPT reads the whole query in both systems ----
    p512 = Params(**{**p.__dict__, "query_max_length": 512})
    trow = []
    for lx, recs_l in by_lex.items():
        for hid, ctrl, grp in COMPARISONS[2:]:
            sub = [r for r in recs_l if r.query.group == grp]
            cond = np.array([r.query.disease for r in sub])
            for L, pp in ((64, p), (512, p512)):
                a, b = 1.0 / ranks(sub, pp, PROPOSED), 1.0 / ranks(sub, pp, ctrl)
                obs, pv = cluster_signflip_test(a - b, cond, st["n_perm"], seeds["permutation"])
                _, lo, hi = cluster_bootstrap_ci(a - b, cond, st["n_boot"], seeds["bootstrap"])
                trow.append({"analysis": LABEL, "lexicon": {"B": "B (Gemini)", "H": "C (co-author)"}[lx], "id": hid,
                             "group": grp, "query_max_length": L, "mrr_proposed": a.mean(), "mrr_hybrid": b.mean(),
                             "delta_mrr": obs, "ci_low": lo, "ci_high": hi, "p_uncorrected": pv})
    trunc = pd.DataFrame(trow)
    trunc.to_csv(OUT / "truncation_512.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 250)
    print(f"\n{LABEL}\nuntouched test records: {len(untouched)}\n")
    print(checks.drop(columns="analysis").round(4).to_string(index=False))
    print("\ntie bounds (MRR worst / fixed rule / best):\n" + tb.drop(columns="analysis").round(3).to_string(index=False))
    print("\npool:\n" + pool.drop(columns="analysis").round(3).to_string(index=False))
    print("\ntruncation:\n" + trunc.drop(columns="analysis").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
