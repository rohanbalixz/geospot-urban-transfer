"""Decision value of a distance: choosing which source model to use for a target.

GeoSpOT's data-selection use picks the source closest to the target. Here the
same decision is scored in units of the task metric:

  gain(rule) = score of the source the rule picks - mean score of a random foreign source

and compared with a cross-seed oracle (best source chosen on some seeds, scored
on the held-out seeds), which is the most any rule can expect to gain once
training noise is accounted for. The in-sample oracle is reported only to show
how much of the apparent gain is the winner's curse.
"""
from __future__ import annotations

import itertools

import numpy as np


def _foreign(S, t):
    return [s for s in range(S) if s != t]


def gains_for_split(Y, A, B, D=None):
    """One seed split. Returns per-target arrays for oracle, distance rule, home model."""
    S = Y.shape[1]
    YA, YB = Y[list(A)].mean(0), Y[list(B)].mean(0)
    orc, dist, home = [], [], []
    for t in range(S):
        f = _foreign(S, t)
        base = np.mean(YB[f, t])
        best = max(f, key=lambda s: YA[s, t])
        orc.append(YB[best, t] - base)
        home.append(YB[t, t] - base)
        if D is not None:
            pick = min(f, key=lambda s: D[s, t])
            dist.append(YB[pick, t] - base)
    return np.array(orc), (np.array(dist) if D is not None else None), np.array(home)


def selection_summary(Y, D=None, n_select=None):
    """Average over all seed splits (n_select seeds to choose, the rest to score).

    Returns dict with oracle_gain, distance_gain, captured_fraction, home_gain,
    insample_oracle_gain (winner's curse reference)."""
    K = Y.shape[0]
    n_select = n_select or max(1, K // 2)
    orc_all, dist_all, home_all = [], [], []
    for A in itertools.combinations(range(K), n_select):
        B = [k for k in range(K) if k not in A]
        o, d, h = gains_for_split(Y, A, B, D)
        orc_all.append(o.mean()); home_all.append(h.mean())
        if d is not None:
            dist_all.append(d.mean())
    Ym = Y.mean(0); S = Ym.shape[0]
    ins = np.mean([max(Ym[f, t]) - np.mean(Ym[f, t]) for t in range(S) for f in [_foreign(S, t)]])
    out = {"oracle_gain": float(np.mean(orc_all)), "home_gain": float(np.mean(home_all)),
           "insample_oracle_gain": float(ins), "n_splits": len(orc_all)}
    if D is not None:
        out["distance_gain"] = float(np.mean(dist_all))
        out["captured_fraction"] = (out["distance_gain"] / out["oracle_gain"]
                                    if out["oracle_gain"] > 0 else float("nan"))
    return out


def selection_bootstrap(Y, D, B=2000, rng=None):
    """Target-level bootstrap CI for distance_gain and captured_fraction (all seeds pooled
    as in selection_summary)."""
    rng = np.random.default_rng(rng)
    K, S, _ = Y.shape
    n_select = max(1, K // 2)
    per_t_orc, per_t_dist = np.zeros(S), np.zeros(S)
    splits = list(itertools.combinations(range(K), n_select))
    for A in splits:
        Bs = [k for k in range(K) if k not in A]
        o, d, _ = gains_for_split(Y, A, Bs, D)
        per_t_orc += o / len(splits); per_t_dist += d / len(splits)
    g, f = [], []
    for _ in range(B):
        idx = rng.integers(0, S, S)
        go, gd = per_t_orc[idx].mean(), per_t_dist[idx].mean()
        g.append(gd); f.append(gd / go if go > 0 else np.nan)
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)),
            float(np.nanpercentile(f, 2.5)), float(np.nanpercentile(f, 97.5)))
