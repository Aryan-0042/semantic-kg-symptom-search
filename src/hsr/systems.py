"""Systems = combinations of per-query components under given parameters.

Components are computed once per query (runner.py); every system and every
point of a tuning grid is a cheap function of them, so tuning never re-runs
the retrieval models.

Matched primary comparisons (PROTOCOL.md sections 5 and 9):
    proposed      Hybrid+KG-linked   (1 - beta) * mm(Hybrid) + beta * KG(exact U semantic(linker, tau)),
                                     0 < beta < 1 (v1.1: beta from the interior grid beta_proposed)
    H1/H2 control Hybrid+KG-exact    same alpha, query length, beta, pool (all) and tie-break;
                                     KG(exact) only: the linking mechanism is the only difference
    H3/H4 control Hybrid             the proposed system with beta = 0 (no KG term)
Reported baseline, outside the family:
    KG-linked                        the proposed system with beta = 1 (same linker and tau): KG only
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .fusion import linear_hybrid, linear_kg, pool_rerank, rrf
from .metrics import rank_matrix

PRIMARY_SYSTEMS = (
    "BM25", "Dense", "Hybrid", "RRF", "Classifier",
    "KG-exact", "KG-linked", "Hybrid+KG-exact", "Hybrid+KG-linked",
)
SECONDARY_SYSTEMS = ("KG-substring", "Hybrid+KG-exact-tuned", "Hybrid+KG-substring-tuned")
ALL_SYSTEMS = PRIMARY_SYSTEMS + SECONDARY_SYSTEMS


@dataclass(frozen=True)
class Params:
    query_max_length: int = 64         # MedCPT query truncation for Dense (standard: 64)
    alpha: float = 0.5                 # BM25 weight in Hybrid
    C: float = 1.0                     # classifier inverse regularisation
    linker: str = "medcpt"             # semantic linker model
    tau: float = 0.8                   # semantic link threshold
    beta: float = 0.5                  # KG weight (0 < beta < 1): proposed system AND its matched exact control
    beta_exact_tuned: float = 0.5      # secondary: exact control at its own development optimum
    beta_substring_tuned: float = 0.5  # sensitivity: historical substring matcher
    pool_beta: dict = field(default_factory=dict)  # "<kg>:<n>" -> beta, linear rule, pool analysis


def hybrid(rec, p):
    c = rec.comp
    return linear_hybrid(c.bm25, c.dense[p.query_max_length], p.alpha)


def kg_linked(rec, p):
    return rec.kg_linked[p.linker][p.tau]


def system_scores(rec, p, name):
    c = rec.comp
    if name == "BM25":
        return c.bm25
    if name == "Dense":
        return c.dense[p.query_max_length]
    if name == "Hybrid":
        return hybrid(rec, p)
    if name == "RRF":
        return rrf(c.bm25, c.dense[p.query_max_length])
    if name == "Classifier":
        return c.classifier[p.C]
    if name == "KG-exact":
        return c.kg_exact
    if name == "KG-substring":
        return c.kg_substring
    if name == "KG-linked":
        return kg_linked(rec, p)
    if name == "Hybrid+KG-exact":
        return linear_kg(hybrid(rec, p), c.kg_exact, p.beta)
    if name == "Hybrid+KG-linked":
        return linear_kg(hybrid(rec, p), kg_linked(rec, p), p.beta)
    if name == "Hybrid+KG-exact-tuned":
        return linear_kg(hybrid(rec, p), c.kg_exact, p.beta_exact_tuned)
    if name == "Hybrid+KG-substring-tuned":
        return linear_kg(hybrid(rec, p), c.kg_substring, p.beta_substring_tuned)
    raise KeyError(name)


def pool_n(n, n_diseases):
    return n_diseases if n == "all" else int(n)


def pool_system_scores(rec, p, kg, n, rule, beta=None):
    """Secondary analysis: pool = top-n of Hybrid, then KG reranking by `rule`."""
    K = rec.comp.kg_exact if kg == "exact" else kg_linked(rec, p)
    return pool_rerank(hybrid(rec, p), K, n, rule, beta)


def query_meta(records):
    return pd.DataFrame([{
        "query_id": r.query.query_id, "row_id": r.query.row_id, "disease": r.query.disease,
        "condition": r.query.condition, "group": r.query.group, "lexicon": r.query.lexicon,
        "n_query_tokens": r.n_query_tokens,
    } for r in records])


def evaluate_fn(records, score_fn, system, meta=None):
    """Per-query metrics (and tie diagnostics) of an arbitrary score function."""
    meta = query_meta(records) if meta is None else meta
    S = np.stack([score_fn(r) for r in records])
    m = rank_matrix(S, np.array([r.true_idx for r in records]))
    df = meta.copy()
    df["system"] = system
    for k, v in m.items():
        df[k] = v
    return df


def evaluate(records, p, systems=ALL_SYSTEMS):
    meta = query_meta(records)
    return pd.concat([evaluate_fn(records, lambda r, s=s: system_scores(r, p, s), s, meta) for s in systems],
                     ignore_index=True)


def params_dict(p):
    return asdict(p)
