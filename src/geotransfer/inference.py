"""Inference for distance-vs-transfer correlations when cells share regions.

The S(S-1) ordered pairs are not independent: every cell in row s shares the
source model, every cell in column t shares the target. Two tools respect that:

* QAP (quadratic assignment) permutation: relabel the regions of the distance
  matrix with one permutation applied to rows and columns together. Exact for
  S <= 8 (8! = 40,320 relabelings), Monte Carlo above that.
* Region bootstrap: resample regions with replacement and keep only cells whose
  two regions are different originals.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import rankdata, spearmanr


def _offdiag_index(S):
    ii, jj = np.where(~np.eye(S, dtype=bool))
    return ii, jj


def spearman_offdiag(D, O):
    ii, jj = _offdiag_index(D.shape[0])
    d, o = D[ii, jj], O[ii, jj]
    ok = np.isfinite(d) & np.isfinite(o)
    return float(spearmanr(d[ok], o[ok]).correlation)


def qap_spearman(D, O, n_perm=None, rng=None, exact_max=8, batch=5000):
    """Two-sided QAP p-value for Spearman(D, O) over off-diagonal cells.

    Returns (rho_obs, p_value, n_permutations_used, exact_flag)."""
    S = D.shape[0]
    ii, jj = _offdiag_index(S)
    o = O[ii, jj]
    ok = np.isfinite(o) & np.isfinite(D[ii, jj])
    ii, jj, o = ii[ok], jj[ok], o[ok]
    ro = rankdata(o); ro = (ro - ro.mean()) / ro.std()
    rho_obs = float(np.mean(ro * _zrank(D[ii, jj][None, :])[0]))

    exact = S <= exact_max and n_perm is None
    if exact:
        perm_iter = itertools.permutations(range(S))
        total = math.factorial(S)
    else:
        rng = np.random.default_rng(rng)
        n_perm = n_perm or 9999
        perm_iter = (rng.permutation(S) for _ in range(n_perm))
        total = n_perm
    count, done = 0, 0
    while done < total:
        P = np.array(list(itertools.islice(perm_iter, batch)))
        if P.size == 0:
            break
        Dp = D[P[:, ii], P[:, jj]]          # relabel rows and columns together
        r = np.mean(_zrank(Dp) * ro[None, :], axis=1)
        count += int(np.sum(np.abs(r) >= abs(rho_obs) - 1e-12))
        done += len(P)
    p = count / done if exact else (count + 1) / (done + 1)
    return rho_obs, float(p), done, exact


def _zrank(A):
    R = rankdata(A, axis=1)
    R = R - R.mean(axis=1, keepdims=True)
    return R / R.std(axis=1, keepdims=True)


def region_bootstrap(D, O, B=2000, rng=None, stat=spearman_offdiag):
    """Percentile CI for stat(D, O) under resampling of regions."""
    rng = np.random.default_rng(rng)
    S = D.shape[0]
    vals = []
    for _ in range(B):
        idx = rng.integers(0, S, S)
        a, b = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        keep = idx[a] != idx[b]
        d = D[idx[a], idx[b]][keep]
        o = O[idx[a], idx[b]][keep]
        ok = np.isfinite(d) & np.isfinite(o)
        if ok.sum() < 5 or np.unique(d[ok]).size < 3:
            continue
        vals.append(spearmanr(d[ok], o[ok]).correlation)
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)), len(vals)


def paired_region_bootstrap_diff(D1, D2, O, B=2000, rng=None):
    """H3: bootstrap of |rho(D1, O)| - |rho(D2, O)| with the SAME region resample for
    both distances. Returns (observed diff, CI low, CI high, two-sided bootstrap p)."""
    rng = np.random.default_rng(rng)
    S = O.shape[0]
    obs = abs(spearman_offdiag(D1, O)) - abs(spearman_offdiag(D2, O))
    diffs = []
    a, b = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    for _ in range(B):
        idx = rng.integers(0, S, S)
        keep = idx[a] != idx[b]
        o = O[idx[a], idx[b]][keep]
        d1, d2 = D1[idx[a], idx[b]][keep], D2[idx[a], idx[b]][keep]
        ok = np.isfinite(o) & np.isfinite(d1) & np.isfinite(d2)
        if ok.sum() < 5 or np.unique(d1[ok]).size < 3 or np.unique(d2[ok]).size < 3:
            continue
        diffs.append(abs(spearmanr(d1[ok], o[ok]).correlation) - abs(spearmanr(d2[ok], o[ok]).correlation))
    diffs = np.array(diffs)
    p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return float(obs), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), p


def region_jackknife(D, O):
    """Leave-one-region-out values of the off-diagonal Spearman (influence check)."""
    S = D.shape[0]
    out = []
    for r in range(S):
        keep = [i for i in range(S) if i != r]
        out.append(spearman_offdiag(D[np.ix_(keep, keep)], O[np.ix_(keep, keep)]))
    return np.array(out)


def within_target_spearman(D, O):
    """Mean over targets of Spearman(D[:, t], O[:, t]) across foreign sources: the
    question 'for this target, does a closer source transfer better?'."""
    S = D.shape[0]
    rs = []
    for t in range(S):
        src = [s for s in range(S) if s != t and np.isfinite(O[s, t]) and np.isfinite(D[s, t])]
        if len(src) >= 4:
            rs.append(spearmanr(D[src, t], O[src, t]).correlation)
    return float(np.nanmean(rs))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
