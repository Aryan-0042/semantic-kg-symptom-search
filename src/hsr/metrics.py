"""Ranking metrics over the full disease list, with one deterministic tie-break.

Every system outputs one score per disease in disease-id order (the
alphabetical order of data/protocol/diseases.csv). Ranking rule, shared by
every system:

  1. scores are rounded to ROUND_DECIMALS decimal places, so floating-point
     noise from summation order cannot create or break a tie;
  2. diseases are sorted by descending rounded score;
  3. diseases with equal rounded scores are ordered by ascending disease id.

The gold disease's rank p is therefore 1 + (#diseases scoring strictly higher)
+ (#diseases with an equal score and a smaller id). Per query:

    RR = 1 / p        Hit@k = [p <= k]  (binary; Acc@1 = Hit@1)

Tie diagnostics (recorded, never used for scoring):
    n_tied      other diseases sharing the gold's rounded score
    rank_best   p if every tie were broken in the gold's favour
    rank_worst  p if every tie were broken against it
    top_tied    more than one disease shares the top rounded score
"""
import numpy as np

KS = (1, 3, 5)
ROUND_DECIMALS = 12


def rank_matrix(S, true_idx, ks=KS):
    """S: (queries x diseases) scores; true_idx: (queries,). Returns dict of arrays."""
    S = np.round(np.asarray(S, dtype=np.float64), ROUND_DECIMALS)
    t = np.asarray(true_idx)
    rows = np.arange(len(S))
    s = S[rows, t][:, None]
    greater = (S > s).sum(1)
    equal = (S == s)
    ties_before = (equal & (np.arange(S.shape[1])[None, :] < t[:, None])).sum(1)
    n_tied = equal.sum(1) - 1
    rank = 1 + greater + ties_before
    top = S.max(1, keepdims=True)
    out = {"rank": rank, "rr": 1.0 / rank}
    for k in ks:
        out[f"hit@{k}"] = (rank <= k).astype(np.int8)
    out.update({"n_tied": n_tied, "rank_best": 1 + greater, "rank_worst": 1 + greater + n_tied,
                "top_tied": ((S == top).sum(1) > 1).astype(np.int8)})
    return out


def rank_metrics(scores, true_idx, ks=KS):
    """Single-query convenience wrapper around rank_matrix."""
    m = rank_matrix(np.asarray(scores, dtype=np.float64)[None, :], [true_idx], ks)
    return {k: v[0].item() for k, v in m.items()}
