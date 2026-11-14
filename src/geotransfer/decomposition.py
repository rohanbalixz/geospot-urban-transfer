"""Variance components, outcome definitions and noise ceilings for transfer matrices.

Convention: Y[k, s, t] is the score of the model trained on source s (seed k),
evaluated on target t. Outcome functions take one S x S matrix and return an
S x S matrix with NaN on the diagonal, so every downstream test uses the same
off-diagonal cells.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------- loading
def load_multiseed(path):
    """Receipts with per_seed[k]['matrix'][source][target] -> (Y[k,s,t], regions, seeds)."""
    d = json.load(open(path))
    seeds = [str(s) for s in d["seeds"]]
    regs = list(d["per_seed"][seeds[0]]["matrix"].keys())
    Y = np.array([[[d["per_seed"][k]["matrix"][s][t] for t in regs] for s in regs] for k in seeds], dtype=float)
    return Y, regs, seeds


def load_single(path, key="matrix"):
    """Receipts with one matrix[source][target] (single seed) -> (Y[s,t], regions)."""
    d = json.load(open(path))
    mat = d[key]
    cols = list(next(iter(mat.values())).keys())
    regs = [r for r in d.get("regions", list(mat.keys())) if r in mat and r in cols]
    return np.array([[mat[s][t] for t in regs] for s in regs], dtype=float), regs


# ---------------------------------------------------------------- outcomes
def _nan_diag(A):
    A = np.array(A, dtype=float)
    np.fill_diagonal(A, np.nan)
    return A


def relative_drop(Y):
    """GeoSpOT's target: (A_t(M_s) - A_t(M_t)) / A_t(M_t)."""
    d = np.diag(Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _nan_diag((Y - d[None, :]) / d[None, :])


def absolute_drop(Y):
    return _nan_diag(Y - np.diag(Y)[None, :])


def within_target(Y):
    """Score minus the target's mean over FOREIGN sources. Removes target difficulty
    and never uses the target's own (in-sample) model."""
    S = Y.shape[0]
    out = np.full((S, S), np.nan)
    for t in range(S):
        foreign = [s for s in range(S) if s != t]
        m = np.mean(Y[foreign, t])
        for s in foreign:
            out[s, t] = Y[s, t] - m
    return out


def pair_specific(Y):
    """Double-centred interaction over the off-diagonal cells (source and target main
    effects removed by alternating row/column centring on the off-diagonal set)."""
    A = _nan_diag(Y)
    for _ in range(50):
        A = A - np.nanmean(A, axis=1, keepdims=True)
        A = A - np.nanmean(A, axis=0, keepdims=True)
    return A


def raw_offdiag(Y):
    return _nan_diag(Y)


OUTCOMES = {
    "relative_drop": relative_drop,
    "absolute_drop": absolute_drop,
    "within_target": within_target,
    "pair_specific": pair_specific,
    "raw_score": raw_offdiag,
}


def offdiag(A):
    S = A.shape[0]
    return np.array([A[s, t] for s in range(S) for t in range(S) if s != t])


# ---------------------------------------------------------------- variance components
def anova3(Y):
    """Random-effects S x T x K ANOVA with one observation per cell.

    Returns (ss_share, var_components, ms, rel_pair_mean) where rel_pair_mean is the
    reliability of the seed-mean pair interaction, 1 - MS_residual / MS_pair.
    """
    K, S, T = Y.shape
    mu = Y.mean()
    yk, ys, yt = Y.mean((1, 2)), Y.mean((0, 2)), Y.mean((0, 1))
    yst, yks, ykt = Y.mean(0), Y.mean(2), Y.mean(1)
    SS = {
        "target": K * S * ((yt - mu) ** 2).sum(),
        "source": K * T * ((ys - mu) ** 2).sum(),
        "seed": S * T * ((yk - mu) ** 2).sum(),
        "pair": K * ((yst - ys[:, None] - yt[None, :] + mu) ** 2).sum(),
        "source_x_seed": T * ((yks - yk[:, None] - ys[None, :] + mu) ** 2).sum(),
        "target_x_seed": S * ((ykt - yk[:, None] - yt[None, :] + mu) ** 2).sum(),
    }
    tot = ((Y - mu) ** 2).sum()
    SS["residual"] = tot - sum(SS.values())
    df = {"target": T - 1, "source": S - 1, "seed": K - 1, "pair": (S - 1) * (T - 1),
          "source_x_seed": (S - 1) * (K - 1), "target_x_seed": (T - 1) * (K - 1),
          "residual": (S - 1) * (T - 1) * (K - 1)}
    MS = {k: SS[k] / df[k] for k in SS}
    e = MS["residual"]
    vc = {"residual": e,
          "pair": max(0.0, (MS["pair"] - e) / K),
          "source_x_seed": max(0.0, (MS["source_x_seed"] - e) / T),
          "target_x_seed": max(0.0, (MS["target_x_seed"] - e) / S)}
    vc["source"] = max(0.0, (MS["source"] - e - K * vc["pair"] - T * vc["source_x_seed"]) / (T * K))
    vc["target"] = max(0.0, (MS["target"] - e - K * vc["pair"] - S * vc["target_x_seed"]) / (S * K))
    vc["seed"] = max(0.0, (MS["seed"] - e - T * vc["source_x_seed"] - S * vc["target_x_seed"]) / (S * T))
    share = {k: SS[k] / tot for k in SS}
    rel = 1 - MS["residual"] / MS["pair"] if MS["pair"] > 0 else float("nan")
    return share, vc, MS, rel


# ---------------------------------------------------------------- reliability and ceilings
def spearman_brown(r, k):
    return k * r / (1 + (k - 1) * r) if r > -1 / (k - 1) else float("nan")


def seed_pair_reliability(Y, outcome):
    """Mean Spearman between single-seed outcome vectors over all seed pairs."""
    K = Y.shape[0]
    v = [offdiag(outcome(Y[k])) for k in range(K)]
    rs = []
    for a, b in itertools.combinations(range(K), 2):
        ok = np.isfinite(v[a]) & np.isfinite(v[b])
        rs.append(spearmanr(v[a][ok], v[b][ok]).correlation)
    return float(np.mean(rs)), float(np.min(rs)), float(np.max(rs))


def split_half_reliability(Y, outcome):
    """Reliability of the K-seed-mean outcome: mean split-half Spearman (floor(K/2) vs rest),
    stepped up to the full K seeds with Spearman-Brown."""
    K = Y.shape[0]
    h = K // 2
    rs = []
    for A in itertools.combinations(range(K), h):
        B = [k for k in range(K) if k not in A]
        a, b = offdiag(outcome(Y[list(A)].mean(0))), offdiag(outcome(Y[B].mean(0)))
        ok = np.isfinite(a) & np.isfinite(b)
        rs.append(spearmanr(a[ok], b[ok]).correlation)
    r = float(np.mean(rs))
    return spearman_brown(r, K / h) if r > 0 else float("nan")


def noise_ceiling(reliability):
    """Largest |rho| any predictor can reach against an outcome with this reliability."""
    return float(np.sqrt(reliability)) if reliability == reliability and reliability > 0 else 0.0


def symmetric_ceiling(O):
    """A symmetric distance d(s,t) = d(t,s) can only predict the symmetric part of a
    transfer outcome.

    Returns (share, rank_ceiling):
      share        fraction of the off-diagonal variance of O that is symmetric,
                   var((O + O^T)/2) / var(O); sqrt(share) bounds Pearson correlation
      rank_ceiling best Spearman any symmetric predictor can reach: correlation of the
                   ranks of O with the pair-averaged ranks (the L2-best symmetric fit
                   in rank space)."""
    from scipy.stats import rankdata
    S = O.shape[0]
    Os = (O + O.T) / 2
    o, s = offdiag(O), offdiag(Os)
    ok = np.isfinite(o) & np.isfinite(s)
    share = float(np.var(s[ok] - s[ok].mean()) / np.var(o[ok] - o[ok].mean()))
    R = np.full((S, S), np.nan)
    ii, jj = np.where(~np.eye(S, dtype=bool))
    R[ii, jj] = rankdata(O[ii, jj])
    Rs = (R + R.T) / 2
    r, rs = offdiag(R), offdiag(Rs)
    ok = np.isfinite(r) & np.isfinite(rs)
    return share, float(np.corrcoef(r[ok], rs[ok])[0, 1]) if np.std(rs[ok]) > 0 else 0.0


def learner_agreement(Y1, Y2, outcome):
    """Agreement of two learners' seed-mean outcomes, raw and corrected for seed noise."""
    a, b = offdiag(outcome(Y1.mean(0))), offdiag(outcome(Y2.mean(0)))
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(spearmanr(a[ok], b[ok]).correlation)
    r1, r2 = split_half_reliability(Y1, outcome), split_half_reliability(Y2, outcome)
    dis = r / np.sqrt(r1 * r2) if (r1 > 0 and r2 > 0) else float("nan")
    return r, float(dis), r1, r2
