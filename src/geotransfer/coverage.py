"""Directional covariate coverage (the positivity diagnostic of the TGIS paper, pairwise).

cov(t | s) = share of target tiles whose nearest SOURCE tile, in standardised input
feature space, lies within tau_s, where tau_s is the 95th percentile of the source's
own leave-one-out nearest-neighbour distances (the paper's support tolerance).

Unlike any OT divergence, cov(t | s) != cov(s | t): a broad source can cover a narrow
target without the reverse. That makes it a candidate for the antisymmetric part of
transfer, which a symmetric distance cannot reach by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def tau_loo(Xs, q=0.95):
    d, _ = cKDTree(Xs).query(Xs, k=2)
    return float(np.quantile(d[:, 1], q))


def coverage(Xs, Xt, q=0.95):
    tau = tau_loo(Xs, q)
    d, _ = cKDTree(Xs).query(Xt, k=1)
    return float(np.mean(d <= tau))


def coverage_matrix(feats, q=0.95):
    """C[s, t] = cov(t | s). feats: list of (n_r, d) arrays, already standardised
    with ONE global mean/SD over all regions."""
    S = len(feats)
    C = np.full((S, S), np.nan)
    for s in range(S):
        for t in range(S):
            if s != t:
                C[s, t] = coverage(feats[s], feats[t], q)
    return C


def antisym_upper(A):
    """Values A[s,t] - A[t,s] for s < t."""
    S = A.shape[0]
    iu = np.triu_indices(S, 1)
    return (A - A.T)[iu]
