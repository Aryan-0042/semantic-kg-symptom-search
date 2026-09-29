"""Paired, disease-clustered statistics.

Queries are clustered by disease (the rows of a disease share one profile and
one KG neighbourhood), so resampling and randomisation happen at the disease
level rather than treating queries as independent.

For paired per-query differences d_q (q = 1..N) in clusters g = 1..G, with
cluster sums S_g = sum_{q in g} d_q and sizes n_g:

    T = (1/N) sum_q d_q = sum_g S_g / sum_g n_g          (the difference in MRR)

Sign-flip randomisation test (two-sided): under H0 the two systems' labels
are exchangeable within each disease, so each S_g is symmetric about 0. For
b = 1..B draw independent signs e_g in {-1, +1} with equal probability and set
T*_b = (1/N) sum_g e_g S_g;  p = (1 + #{b : |T*_b| >= |T|}) / (1 + B).

Cluster bootstrap: for b = 1..B draw G diseases with replacement and compute
sum S_g* / sum n_g*; the CI is the percentile interval of these values.
"""
import numpy as np

TOL = 1e-12


def _cluster_sums(values, clusters):
    values = np.asarray(values, dtype=np.float64)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return np.bincount(inv, weights=values), np.bincount(inv)


def cluster_bootstrap_ci(values, clusters, n_boot=10_000, seed=0, level=0.95):
    """(query-weighted mean, lower, upper), resampling diseases with replacement."""
    sums, counts = _cluster_sums(values, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    a = (1 - level) / 2
    return float(sums.sum() / counts.sum()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def cluster_signflip_test(diff, clusters, n_perm=100_000, seed=0):
    """Two-sided paired randomisation test with disease-level sign flips. Returns (T, p)."""
    sums, counts = _cluster_sums(diff, clusters)
    obs = sums.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(sums)))
    null = (signs * sums).sum(1) / counts.sum()
    p = (np.sum(np.abs(null) >= abs(obs) - TOL) + 1) / (n_perm + 1)
    return float(obs), float(p)


def holm(pvalues):
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
