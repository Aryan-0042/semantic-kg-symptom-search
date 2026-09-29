"""Analyses shared by the development sweep and the final evaluation."""
from itertools import product

import numpy as np
import pandas as pd

from .metrics import rank_matrix
from .stats import cluster_bootstrap_ci, cluster_signflip_test, holm
from .systems import pool_n, pool_system_scores, query_meta


def summary_with_ci(df, metric="rr", by="group", n_boot=10_000, seed=0):
    """Mean of `metric` per (lexicon, <by>, system) with a disease-clustered 95% CI."""
    out = []
    for (lex, cond, sys_), g in df.groupby(["lexicon", by, "system"]):
        mean, lo, hi = cluster_bootstrap_ci(g[metric].to_numpy(dtype=float), g.disease.to_numpy(), n_boot, seed)
        out.append({"lexicon": lex, by: cond, "system": sys_, "n": len(g),
                    "n_diseases": g.disease.nunique(), metric: mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def paired(df, a, b, group, lexicon):
    sel = (df.group == group) & (df.lexicon == lexicon)
    ga = df[sel & (df.system == a)].set_index("query_id").sort_index()
    gb = df[sel & (df.system == b)].set_index("query_id").sort_index()
    assert len(ga) and ga.index.equals(gb.index), "paired comparison needs identical, non-empty query sets"
    return ga, gb


def primary_tests(df, comparisons, lexicon, n_perm=100_000, n_boot=10_000, seed_perm=0, seed_boot=0):
    """H1-H4: T = mean over the group's queries of RR(a) - RR(b); disease-clustered
    sign-flip p-value and bootstrap CI; Holm over the family. The Acc@1
    difference and its clustered CI are reported descriptively."""
    rows = []
    for c in comparisons:
        ga, gb = paired(df, c["a"], c["b"], c["condition"], lexicon)
        clusters = ga.disease.to_numpy()
        diff = ga.rr.to_numpy() - gb.rr.to_numpy()
        obs, p = cluster_signflip_test(diff, clusters, n_perm, seed_perm)
        _, lo, hi = cluster_bootstrap_ci(diff, clusters, n_boot, seed_boot)
        d1 = ga["hit@1"].to_numpy(dtype=float) - gb["hit@1"].to_numpy(dtype=float)
        acc_d, acc_lo, acc_hi = cluster_bootstrap_ci(d1, clusters, n_boot, seed_boot)
        rows.append({"id": c["id"], "a": c["a"], "b": c["b"], "condition": c["condition"], "lexicon": lexicon,
                     "n_queries": len(diff), "n_diseases": len(np.unique(clusters)),
                     "mrr_a": float(ga.rr.mean()), "mrr_b": float(gb.rr.mean()),
                     "delta_mrr": obs, "ci_low": lo, "ci_high": hi, "p": p,
                     "queries_a_better": int((diff > 0).sum()), "queries_b_better": int((diff < 0).sum()),
                     "acc1_a": float(ga["hit@1"].mean()), "acc1_b": float(gb["hit@1"].mean()),
                     "delta_acc1": acc_d, "delta_acc1_ci_low": acc_lo, "delta_acc1_ci_high": acc_hi})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["reject_at_0.05"] = out.p_holm < 0.05
    return out


def tie_diagnostics(df):
    g = df.assign(gold_tied=df.n_tied > 0,
                  acc1_decided_by_tiebreak=(df.rank_best == 1) & (df.rank_worst > 1)) \
          .groupby(["lexicon", "group", "system"])
    return pd.DataFrame({
        "n": g.size(),
        "share_gold_tied": g.gold_tied.mean(),
        "share_top_tied": g.top_tied.mean(),
        "share_acc1_decided_by_tiebreak": g.acc1_decided_by_tiebreak.mean(),
        "mean_n_tied": g.n_tied.mean(),
    }).reset_index()


def truncation_rates(records, lengths):
    meta = query_meta(records)
    rows = []
    for (lex, cond), g in meta.groupby(["lexicon", "condition"]):
        row = {"lexicon": lex, "condition": cond, "n": len(g),
               "median_tokens": float(g.n_query_tokens.median()), "max_tokens": int(g.n_query_tokens.max())}
        for L in lengths:
            row[f"truncated_at_{L}"] = float((g.n_query_tokens > L).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pool_analysis(records, p, spec, n_diseases):
    """Generator Hybrid; pool n x rule (linear with pool_beta[kg:n], RRF) x KG signal."""
    meta = query_meta(records)
    t = np.array([r.true_idx for r in records])
    frames = []
    for kg, n, rule in product(spec["kg"], spec["n"], spec["rules"]):
        beta = p.pool_beta[f"{kg}:{n}"] if rule == "linear" else None
        nn = pool_n(n, n_diseases)
        S = np.stack([pool_system_scores(r, p, kg, nn, rule, beta) for r in records])
        m = rank_matrix(S, t)
        frames.append(meta.assign(kg=kg, n=str(n), rule=rule, beta=beta, rr=m["rr"], **{"hit@1": m["hit@1"], "hit@3": m["hit@3"]}))
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["lexicon", "group", "kg", "n", "rule"], sort=False)[["beta", "rr", "hit@1", "hit@3"]] \
             .mean().reset_index()


def linking_accuracy(records, linkers_taus):
    """Per segment, against the symptom that generated it. linkers_taus: {linker: tau}.
    recall = share of segments whose gold node is linked; precision = correct links / all links."""
    acc = {}
    for rec in records:
        d = rec.link_diag
        if d["gold"] is None:
            continue
        key = (rec.query.lexicon, rec.query.group)
        a = acc.setdefault(key, {"segments": 0})
        a["segments"] += len(d["gold"])
        methods = {"exact": list(d["exact"]), "substring": list(d["substring"])}
        for lk, tau in linkers_taus.items():
            nodes, sims = d["semantic"][lk]
            methods[f"linked_{lk}@{tau:g}"] = [e | ({n} if s >= tau else set())
                                               for e, n, s in zip(d["exact"], nodes, sims)]
        for name, per_seg in methods.items():
            hits = sum(g in links for g, links in zip(d["gold"], per_seg))
            n_links = sum(len(links) for links in per_seg)
            a[f"{name}_hits"] = a.get(f"{name}_hits", 0) + hits
            a[f"{name}_links"] = a.get(f"{name}_links", 0) + n_links
    rows = []
    for (lex, grp), a in sorted(acc.items()):
        row = {"lexicon": lex, "group": grp, "segments": a["segments"]}
        for name in [k[:-5] for k in a if k.endswith("_hits")]:
            row[f"{name}_recall"] = a[f"{name}_hits"] / a["segments"]
            row[f"{name}_precision"] = a[f"{name}_hits"] / a[f"{name}_links"] if a[f"{name}_links"] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
