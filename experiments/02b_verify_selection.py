"""Independent check that configs/selected_params.json follows PROTOCOL.md section 7.

    venv\\Scripts\\python experiments\\02b_verify_selection.py

Re-derives every selection from the saved development tuning tables with rules
written here from the protocol text (not by calling hsr.tuning), checks that
the tables cover exactly the frozen grids, that each tuning objective equals
the objective recomputed from the evaluated per-query results, and that the
development outputs contain no test row and no lexicon B query. Writes
results/rebuild/dev/selection_verification.json; exits non-zero on any failure.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "results" / "rebuild" / "dev"
TOL = 1e-12
GROUPS = ["canonical_full", "canonical_partial", "lay_full", "lay_partial"]

proto = yaml.safe_load(open(ROOT / "configs" / "protocol.yaml", encoding="utf-8"))
sel = json.load(open(ROOT / "configs" / "selected_params.json"))
p = sel["params"]
grids = proto["tuning"]["grids"]
ql = proto["tuning"]["query_length"]
pool_spec = proto["secondary_analyses"]["pool_size"]
checks, notes = [], []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})


def at_max(t):
    return t[t.objective >= t.objective.max() - TOL]


def same_values(a, b):
    return sorted(np.round(np.asarray(a, dtype=float), 9)) == sorted(np.round(np.asarray(b, dtype=float), 9))


tables = {n: pd.read_csv(DEV / f"tuning_{n}.csv") for n in
          ["query_max_length", "alpha", "classifier_C", "linked_joint", "beta_exact_tuned", "beta_substring_tuned", "pool_beta"]}

# ---- grids covered exactly -------------------------------------------------
check("grid: query lengths", same_values(tables["query_max_length"].query_max_length, ql["grid"]))
check("grid: alpha", same_values(tables["alpha"].alpha, grids["alpha"]))
check("grid: C", same_values(tables["classifier_C"].C, grids["classifier_C"]))
j = tables["linked_joint"]
check("grid: joint size = linkers x tau x beta_proposed",
      len(j) == len(grids["linker"]) * len(grids["tau"]) * len(grids["beta_proposed"]), len(j))
check("grid: joint linkers", sorted(j.linker.unique()) == sorted(grids["linker"]))
check("grid: joint tau", same_values(j.tau.unique(), grids["tau"]))
check("grid: joint beta = beta_proposed (v1.1)", same_values(j.beta.unique(), grids["beta_proposed"]))
check("grid: beta_proposed excludes 0 and 1", 0.0 not in grids["beta_proposed"] and 1.0 not in grids["beta_proposed"]
      and same_values(grids["beta_proposed"], [x for x in grids["beta"] if 0 < x < 1]))
check("grid: joint unique cells", not j.duplicated(["linker", "tau", "beta"]).any())
for n in ["beta_exact_tuned", "beta_substring_tuned"]:
    check(f"grid: {n}", same_values(tables[n].beta, grids["beta"]))
pb = tables["pool_beta"]
for kg, g in pb.groupby("kg"):
    want = grids["beta_proposed"] if kg == "linked" else grids["beta"]
    check(f"grid: pool {kg} cells", len(g) == len(pool_spec["n"]) * len(want)
          and all(same_values(c.beta, want) for _, c in g.groupby("n")), len(g))

# ---- step 0: query length ----------------------------------------------------
t = tables["query_max_length"]
base = float(t.loc[t.query_max_length == ql["standard"], "objective"].iloc[0])
qual = t[t.objective >= base + ql["min_gain"] - TOL]
exp_L = ql["standard"] if qual.empty else int(at_max(qual).query_max_length.min())
check("step 0: query_max_length", p["query_max_length"] == exp_L, f"expected {exp_L}, got {p['query_max_length']}")
check("step 0: query_length_is_standard flag", sel["query_length_is_standard"] == (p["query_max_length"] == ql["standard"]))
notes.append({"step": "0 query length", "tied_at_max": len(at_max(t)), "best_gain_over_64": float(t.objective.max() - base)})

# ---- step 1: alpha (closest to 0.5, then smaller) ----------------------------
m = at_max(tables["alpha"]).assign(d=lambda x: (x.alpha - 0.5).abs())
exp_a = float(m[m.d <= m.d.min() + TOL].alpha.min())
check("step 1: alpha", abs(p["alpha"] - exp_a) < 1e-9, f"expected {exp_a}, got {p['alpha']}")
notes.append({"step": "1 alpha", "tied_at_max": len(m)})

# ---- step 2: C (closest to 1, then smaller) -----------------------------------
m = at_max(tables["classifier_C"]).assign(d=lambda x: (x.C - 1.0).abs())
exp_C = float(m[m.d <= m.d.min() + TOL].C.min())
check("step 2: C", abs(p["C"] - exp_C) < 1e-9, f"expected {exp_C}, got {p['C']}")
notes.append({"step": "2 C", "tied_at_max": len(m)})

# ---- step 3: (linker, tau, beta) jointly: beta smallest, tau largest, medcpt --
m = at_max(j)
m = m[m.beta <= m.beta.min() + 1e-9]
m = m[m.tau >= m.tau.max() - 1e-9]
exp_lk = "medcpt" if (m.linker == "medcpt").any() else sorted(m.linker)[0]
m = m[m.linker == exp_lk]
check("step 3: linker", p["linker"] == exp_lk, f"expected {exp_lk}, got {p['linker']}")
check("step 3: tau", abs(p["tau"] - float(m.tau.iloc[0])) < 1e-9, f"expected {float(m.tau.iloc[0])}, got {p['tau']}")
check("step 3: beta", abs(p["beta"] - float(m.beta.iloc[0])) < 1e-9, f"expected {float(m.beta.iloc[0])}, got {p['beta']}")
notes.append({"step": "3 joint", "tied_at_max": len(at_max(j)),
              "tied_linkers": sorted(at_max(j).linker.unique().tolist())})

# ---- step 4: secondary betas (smallest) ----------------------------------------
for key, n in [("beta_exact_tuned", "beta_exact_tuned"), ("beta_substring_tuned", "beta_substring_tuned")]:
    m = at_max(tables[n])
    check(f"step 4: {key}", abs(p[key] - float(m.beta.min())) < 1e-9, f"expected {float(m.beta.min())}, got {p[key]}")
    notes.append({"step": f"4 {key}", "tied_at_max": len(m)})

# ---- step 5: pool betas (smallest) ---------------------------------------------
for (kg, n), g in pb.groupby(["kg", "n"]):
    exp_b = float(at_max(g).beta.min())
    got = p["pool_beta"].get(f"{kg}:{n}")
    check(f"step 5: pool_beta {kg}:{n}", got is not None and abs(got - exp_b) < 1e-9, f"expected {exp_b}, got {got}")
check("step 5: pool linked:all == beta (n = all is the primary system)", p["pool_beta"]["linked:all"] == p["beta"])
check("step 5: pool exact:all == beta_exact_tuned", p["pool_beta"]["exact:all"] == p["beta_exact_tuned"])

# ---- selected values lie on the frozen grids -----------------------------------
def on(v, grid):
    return any(abs(v - x) < 1e-9 for x in grid)


check("values on grids", p["query_max_length"] in ql["grid"] and p["C"] in grids["classifier_C"]
      and p["linker"] in grids["linker"] and on(p["tau"], grids["tau"]) and on(p["alpha"], grids["alpha"])
      and on(p["beta"], grids["beta_proposed"]) and on(p["beta_exact_tuned"], grids["beta"])
      and on(p["beta_substring_tuned"], grids["beta"])
      and all(on(v, grids["beta_proposed"] if k.startswith("linked") else grids["beta"]) for k, v in p["pool_beta"].items()))
check("proposed beta is interior (0 < beta < 1)", 0.0 < p["beta"] < 1.0, p["beta"])

# ---- tuning objective == evaluated objective ------------------------------------
pq = pd.read_csv(DEV / "per_query.csv")


def evaluated(system):
    g = pq[pq.system == system].groupby("group").rr.mean()
    return float(g.reindex(GROUPS).mean())


def tuned(table, **where):
    t = tables[table]
    for k, v in where.items():
        t = t[np.isclose(t[k], v)] if not isinstance(v, str) else t[t[k] == v]
    return float(t.objective.iloc[0])


pairs = {
    "Dense": tuned("query_max_length", query_max_length=p["query_max_length"]),
    "Hybrid": tuned("alpha", alpha=p["alpha"]),
    "Classifier": tuned("classifier_C", C=p["C"]),
    "Hybrid+KG-linked": tuned("linked_joint", linker=p["linker"], tau=p["tau"], beta=p["beta"]),
    "Hybrid+KG-exact-tuned": tuned("beta_exact_tuned", beta=p["beta_exact_tuned"]),
    "Hybrid+KG-substring-tuned": tuned("beta_substring_tuned", beta=p["beta_substring_tuned"]),
}
for s, v in pairs.items():
    check(f"objective consistent: {s}", abs(evaluated(s) - v) < 1e-9, f"tuning {v:.12f} vs evaluated {evaluated(s):.12f}")

# ---- beta endpoints are the reported baselines ------------------------------------
import pickle  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from hsr.metrics import rank_matrix  # noqa: E402
from hsr.systems import Params, system_scores  # noqa: E402
comp_path = ROOT / "cache" / "components_dev.pkl"
if comp_path.exists():
    recs = pickle.load(open(comp_path, "rb"))
    P = Params(**p)
    t_idx = np.array([r.true_idx for r in recs])
    for b_end, base in [(1.0, "KG-linked"), (0.0, "Hybrid")]:
        Pb = Params(**{**p, "beta": b_end})
        r_end = rank_matrix(np.stack([system_scores(r, Pb, "Hybrid+KG-linked") for r in recs]), t_idx)["rank"]
        r_base = rank_matrix(np.stack([system_scores(r, P, base) for r in recs]), t_idx)["rank"]
        check(f"endpoint: proposed at beta={b_end:g} ranks exactly like {base}", np.array_equal(r_end, r_base))
    q_ids = sorted(r.query.query_id for r in recs)
    check("components file matches per-query results", q_ids == sorted(pd.read_csv(DEV / "per_query.csv").query_id.unique()))
else:
    notes.append({"warning": "cache/components_dev.pkl missing; endpoint checks skipped"})

# ---- development data only ------------------------------------------------------
part = pd.read_csv(ROOT / "data" / "protocol" / "partition.csv")
dev_ids = set(part.row_id[part.split == "dev"])
check("per-query rows are development rows only", set(pq.row_id) <= dev_ids, sorted(set(pq.row_id) - dev_ids)[:5])
check("all 123 development rows queried", set(pq.row_id) == dev_ids)
check("no lexicon B query", set(pq.lexicon) <= {"A", "none"}, sorted(set(pq.lexicon)))
check("700 queries per system", (pq.groupby("system").query_id.nunique() == 700).all())
check("identical query sets across systems", pq.groupby("system").query_id.apply(lambda s: tuple(sorted(s))).nunique() == 1)
check("no final results exist (at verification time)", not (ROOT / "results" / "rebuild" / "test").exists())

# ---- provenance -------------------------------------------------------------------
import hashlib  # noqa: E402
sha = hashlib.sha256(open(ROOT / proto["data"]["lexicon_dev"], "rb").read()).hexdigest()
check("lexicon A hash recorded", sel["lexicon_A_sha256"] == sha)
check("protocol version recorded", sel["protocol_version"] == proto["version"] and proto["status"] == "frozen")
head_at_run = sel["git"]["commit"]
tracked_diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=MD", head_at_run, "--",
                               "src", "experiments/02_dev_sweep.py", "configs/protocol.yaml", "data"],
                              cwd=ROOT, capture_output=True, text=True).stdout.split()
check("tracked code/config/data unchanged since the sweep's commit", not tracked_diff, tracked_diff)

out = {"all_ok": all(c["ok"] for c in checks), "selected": p, "checks": checks, "tie_notes": notes}
with open(DEV / "selection_verification.json", "w", newline="\n") as f:
    json.dump(out, f, indent=2)
for c in checks:
    print(("PASS  " if c["ok"] else "FAIL  ") + c["check"] + ("" if c["ok"] else f"   [{c['detail']}]"))
print("\ntie notes:", json.dumps(notes, indent=1))
print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
sys.exit(0 if out["all_ok"] else 1)
