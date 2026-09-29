"""Development-only parameter selection (PROTOCOL.md section 5).

objective(S) = mean over the four condition groups (canonical_full,
canonical_partial, lay_full, lay_partial) of the group's MRR, where a group's
MRR pools all of its queries (partial = k2 and k3 queries together). Ranks use
the shared deterministic tie-break (metrics.py), so tuning scores exactly what
evaluation scores.

Selection order (each step fixes its parameter for the next):
    0  query_max_length     Dense only; keep 64 unless a longer length beats it by >= margin
    1  alpha                Hybrid
    2  C                    Classifier
    3  (linker, tau, beta)  jointly, proposed Hybrid+KG-linked; beta from the interior
                            grid beta_proposed (v1.1: 0 and 1 excluded)
    4  beta_exact_tuned, beta_substring_tuned   secondary systems
    5  pool_beta[kg:n]      linear rule of the secondary pool analysis (linked cells use
                            beta_proposed, so n = all stays identical to the proposed system)
"""
from itertools import product

import numpy as np
import pandas as pd

from .fusion import linear_hybrid, linear_kg
from .metrics import rank_matrix
from .queries import GROUPS
from .systems import Params, pool_n, pool_system_scores


class Objective:
    def __init__(self, records):
        self.true_idx = np.array([r.true_idx for r in records])
        groups = np.array([r.query.group for r in records])
        self.masks = {g: groups == g for g in GROUPS if (groups == g).any()}

    def per_group(self, S):
        rr = rank_matrix(S, self.true_idx)["rr"]
        return {g: float(rr[m].mean()) for g, m in self.masks.items()}

    def __call__(self, S):
        return float(np.mean(list(self.per_group(S).values())))


def select(table, prefer):
    """Keep rows at the maximum objective, then apply tie-break preferences in order."""
    best = table[np.isclose(table.objective, table.objective.max(), rtol=0, atol=1e-12)].copy()
    for col, how in prefer:
        if len(best) == 1:
            break
        if how == "smallest":
            best = best[best[col] == best[col].min()]
        elif how == "largest":
            best = best[best[col] == best[col].max()]
        elif how.startswith("closest:"):
            target = float(how.split(":")[1])
            d = (best[col] - target).abs()
            best = best[d == d.min()]
            best = best[best[col] == best[col].min()]  # symmetric ties -> smaller
        elif how.startswith("equals:"):
            v = how.split(":")[1]
            if (best[col] == v).any():
                best = best[best[col] == v]
        else:
            raise ValueError(how)
    return best.iloc[0]


def select_query_length(table, standard, margin):
    """Keep the standard length unless another length beats its objective by
    at least `margin`; among those, the best objective, then the shortest."""
    base = float(table.loc[table.query_max_length == standard, "objective"].iloc[0])
    better = table[table.objective >= base + margin - 1e-12]
    if better.empty:
        return int(standard)
    return int(select(better, [("query_max_length", "smallest")]).query_max_length)


def _stack(records, fn):
    return np.stack([fn(r) for r in records])


def tune(records, protocol, linkers, n_diseases):
    """Returns (Params, {table name: DataFrame})."""
    spec = protocol["tuning"]
    grids, obj, tables = spec["grids"], Objective(records), {}

    def table(rows, name):
        t = pd.DataFrame(rows)
        tables[name] = t
        return t

    # 0. MedCPT query length (Dense only)
    ql = spec["query_length"]
    t = table([{"query_max_length": L, "objective": obj(_stack(records, lambda r, L=L: r.comp.dense[L])),
                "is_standard": L == ql["standard"]} for L in ql["grid"]], "query_max_length")
    L = select_query_length(t, ql["standard"], ql["min_gain"])

    # 1. alpha
    t = table([{"alpha": a, "objective": obj(_stack(records, lambda r, a=a: linear_hybrid(r.comp.bm25, r.comp.dense[L], a)))}
               for a in grids["alpha"]], "alpha")
    alpha = float(select(t, [("alpha", "closest:0.5")]).alpha)

    # 2. classifier C
    t = table([{"C": C, "objective": obj(_stack(records, lambda r, C=C: r.comp.classifier[C]))}
               for C in grids["classifier_C"]], "classifier_C")
    C = float(select(t, [("C", "closest:1.0")]).C)

    H = [linear_hybrid(r.comp.bm25, r.comp.dense[L], alpha) for r in records]

    def fused(K_of, beta):
        return np.stack([linear_kg(h, K_of(r), beta) for r, h in zip(records, H)])

    # 3. (linker, tau, beta) jointly for the proposed system
    rows = []
    for lk, tau, b in product(linkers, grids["tau"], grids["beta_proposed"]):
        per = obj.per_group(fused(lambda r: r.kg_linked[lk][tau], b))
        rows.append({"linker": lk, "tau": tau, "beta": b, "objective": float(np.mean(list(per.values()))), **per})
    t = table(rows, "linked_joint")
    sel = select(t, [("beta", "smallest"), ("tau", "largest"), ("linker", "equals:medcpt")])
    linker, tau, beta = str(sel.linker), float(sel.tau), float(sel.beta)

    # 4. secondary systems at their own development optimum
    t = table([{"beta": b, "objective": obj(fused(lambda r: r.comp.kg_exact, b))} for b in grids["beta"]], "beta_exact_tuned")
    beta_exact = float(select(t, [("beta", "smallest")]).beta)
    t = table([{"beta": b, "objective": obj(fused(lambda r: r.comp.kg_substring, b))} for b in grids["beta"]],
              "beta_substring_tuned")
    beta_sub = float(select(t, [("beta", "smallest")]).beta)

    p = Params(query_max_length=L, alpha=alpha, C=C, linker=linker, tau=tau, beta=beta,
               beta_exact_tuned=beta_exact, beta_substring_tuned=beta_sub)

    # 5. pool analysis: beta of the linear rule per (kg, n); RRF has no free parameter
    pool_spec, rows, pool_beta = protocol["secondary_analyses"]["pool_size"], [], {}
    for kg, n in product(pool_spec["kg"], pool_spec["n"]):
        nn = pool_n(n, n_diseases)
        beta_grid = grids["beta_proposed"] if kg == "linked" else grids["beta"]
        cell = [{"kg": kg, "n": str(n), "beta": b,
                 "objective": obj(_stack(records, lambda r, b=b: pool_system_scores(r, p, kg, nn, "linear", b)))}
                for b in beta_grid]
        rows += cell
        pool_beta[f"{kg}:{n}"] = float(select(pd.DataFrame(cell), [("beta", "smallest")]).beta)
    table(rows, "pool_beta")
    p = Params(**{**p.__dict__, "pool_beta": pool_beta})
    return p, tables
