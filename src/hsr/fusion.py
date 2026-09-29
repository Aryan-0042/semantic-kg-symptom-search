"""Score fusion, candidate pools and reranking over the full disease list.

Every function returns one score per disease (higher is better), so every
system produces a complete ranking of all diseases; the shared deterministic
tie-break (metrics.py) is applied afterwards.

Two-stage systems (secondary pool-size analysis):
  stage 1 (candidate generation)  generator score G over all diseases; the
                                  pool is the top n by G (ties at the pool
                                  boundary go to the smaller disease id)
  stage 2 (reranking)             inside the pool, reorder by
                                  'linear'  (1 - beta) * mm(G) + beta * K
                                            (mm over all diseases, so n = all
                                            equals single-stage linear fusion)
                                  'rrf'     1/(60 + rank_pool(G)) + 1/(60 + rank_pool(K)),
                                            ranks among pool members, average
                                            ranks for ties
  diseases outside the pool keep G's order below every pool member.
"""
import numpy as np
from scipy.stats import rankdata

from .metrics import ROUND_DECIMALS

RRF_K = 60


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def linear_hybrid(bm25, dense, alpha):
    """alpha * mm(BM25) + (1 - alpha) * mm(dense)."""
    return alpha * minmax(bm25) + (1.0 - alpha) * minmax(dense)


def rrf(*score_vectors, k=RRF_K):
    """Reciprocal rank fusion; tied scores share their average rank."""
    return sum(1.0 / (k + rankdata(-np.round(np.asarray(v, dtype=np.float64), ROUND_DECIMALS), method="average"))
               for v in score_vectors)


def linear_kg(G, K, beta):
    return (1.0 - beta) * minmax(G) + beta * np.asarray(K, dtype=np.float64)


def pool_members(G, n):
    """Boolean mask of the top-n diseases by G (ties -> smaller disease id)."""
    G = np.round(np.asarray(G, dtype=np.float64), ROUND_DECIMALS)
    order = np.lexsort((np.arange(len(G)), -G))
    mask = np.zeros(len(G), dtype=bool)
    mask[order[:n]] = True
    return mask


def pool_rerank(G, K, n, rule, beta=None):
    G = np.asarray(G, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    pool = pool_members(G, n)
    if rule == "linear":
        inner = linear_kg(G, K, beta)                     # in [0, 1]
    elif rule == "rrf":
        inner = np.zeros(len(G))
        inner[pool] = rrf(G[pool], K[pool])               # in (0, 2 / (RRF_K + 1)]
    else:
        raise ValueError(rule)
    if pool.all():
        return inner                                      # n = all: exactly single-stage fusion
    # Pool members score in [2, 3], everything else keeps mm(G) in [0, 1].
    return np.where(pool, 2.0 + inner, minmax(G))
