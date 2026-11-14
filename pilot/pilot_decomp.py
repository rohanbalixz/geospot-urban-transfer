"""Pilot: how much pair-specific transfer signal exists above seed noise?

Reads the committed receipts only (no data, no training).
Usage: python -I pilot_decomp.py <metrics_dir>
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

M = Path(sys.argv[1])


def load_multiseed(fn):
    d = json.load(open(M / fn))
    seeds = [str(s) for s in d["seeds"]]
    regs = list(d["per_seed"][seeds[0]]["matrix"].keys())
    Y = np.array([[[d["per_seed"][k]["matrix"][s][t] for t in regs] for s in regs] for k in seeds])
    return Y, regs  # Y[k, s, t]


def anova3(Y):
    """Random-effects S x T x K ANOVA, one obs per cell. Returns SS shares and variance comps."""
    K, S, T = Y.shape
    mu = Y.mean()
    yk, ys, yt = Y.mean((1, 2)), Y.mean((0, 2)), Y.mean((0, 1))
    yst, yks, ykt = Y.mean(0), Y.mean(2), Y.mean(1)
    SS = {
        "target": K * S * ((yt - mu) ** 2).sum(),
        "source": K * T * ((ys - mu) ** 2).sum(),
        "seed": S * T * ((yk - mu) ** 2).sum(),
        "pair(SxT)": K * ((yst - ys[:, None] - yt[None, :] + mu) ** 2).sum(),
        "source x seed": T * ((yks - yk[:, None] - ys[None, :] + mu) ** 2).sum(),
        "target x seed": S * ((ykt - yk[:, None] - yt[None, :] + mu) ** 2).sum(),
    }
    tot = ((Y - mu) ** 2).sum()
    SS["residual(SxTxK)"] = tot - sum(SS.values())
    df = {"target": T - 1, "source": S - 1, "seed": K - 1, "pair(SxT)": (S - 1) * (T - 1),
          "source x seed": (S - 1) * (K - 1), "target x seed": (T - 1) * (K - 1),
          "residual(SxTxK)": (S - 1) * (T - 1) * (K - 1)}
    MS = {k: SS[k] / df[k] for k in SS}
    e = MS["residual(SxTxK)"]
    vc = {
        "residual": e,
        "pair": max(0.0, (MS["pair(SxT)"] - e) / K),
        "source x seed": max(0.0, (MS["source x seed"] - e) / T),
        "target x seed": max(0.0, (MS["target x seed"] - e) / S),
    }
    vc["source"] = max(0.0, (MS["source"] - e - K * vc["pair"] - T * vc["source x seed"]) / (T * K))
    vc["target"] = max(0.0, (MS["target"] - e - K * vc["pair"] - S * vc["target x seed"]) / (S * K))
    share = {k: SS[k] / tot for k in SS}
    rel_pair_K = 1 - MS["residual(SxTxK)"] / MS["pair(SxT)"]  # reliability of seed-mean interaction
    return share, vc, rel_pair_K, MS


def offdiag_pairs(S):
    return [(s, t) for s in range(S) for t in range(S) if s != t]


def delta(Yk):
    """GeoSpOT relative drop for one seed: (A_t(M_s) - A_t(M_t)) / A_t(M_t)."""
    S = Yk.shape[0]
    return np.array([(Yk[s, t] - Yk[t, t]) / Yk[t, t] for s, t in offdiag_pairs(S)])


def absdiff(Yk):
    S = Yk.shape[0]
    return np.array([Yk[s, t] - Yk[t, t] for s, t in offdiag_pairs(S)])


def raw(Yk):
    S = Yk.shape[0]
    return np.array([Yk[s, t] for s, t in offdiag_pairs(S)])


def within_target_centered(Yk):
    """Off-diagonal score minus that target's mean over foreign sources (removes target effect)."""
    S = Yk.shape[0]
    out = []
    for s, t in offdiag_pairs(S):
        foreign = [Yk[u, t] for u in range(S) if u != t]
        out.append(Yk[s, t] - np.mean(foreign))
    return np.array(out)


def seed_rank_reliability(Y, f):
    K = Y.shape[0]
    vals = [f(Y[k]) for k in range(K)]
    rs = [spearmanr(vals[a], vals[b]).correlation for a, b in itertools.combinations(range(K), 2)]
    return float(np.mean(rs)), float(np.min(rs)), float(np.max(rs))


def spearman_brown(r1, k):
    return k * r1 / (1 + (k - 1) * r1)


def selection_value(Y, n_select=2):
    """Cross-seed oracle: pick each target's best foreign source on n_select seeds,
    score it on the held-out seeds. Gain is relative to a random foreign source."""
    K, S, _ = Y.shape
    gains, insample, home_vs_best = [], [], []
    for A in itertools.combinations(range(K), n_select):
        B = [k for k in range(K) if k not in A]
        YA, YB = Y[list(A)].mean(0), Y[B].mean(0)
        for t in range(S):
            foreign = [s for s in range(S) if s != t]
            best = max(foreign, key=lambda s: YA[s, t])
            gains.append(YB[best, t] - np.mean([YB[s, t] for s in foreign]))
            insample.append(max(YB[s, t] for s in foreign) - np.mean([YB[s, t] for s in foreign]))
            home_vs_best.append(YB[t, t] - YB[best, t])
    return float(np.mean(gains)), float(np.std(gains) / np.sqrt(len(gains))), float(np.mean(insample)), float(np.mean(home_vs_best))


def report_multiseed(fn, label):
    Y, regs = load_multiseed(fn)
    K, S, _ = Y.shape
    print(f"\n######## {label}  ({fn}: {K} seeds x {S} sources x {S} targets)")
    share, vc, relK, MS = anova3(Y)
    print("Sum-of-squares share:")
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"   {k:18s} {100*v:6.2f}%")
    tot_vc = sum(vc.values())
    print("Variance components (random effects), share of total:")
    for k, v in sorted(vc.items(), key=lambda kv: -kv[1]):
        print(f"   {k:18s} {100*v/tot_vc:6.2f}%   (sd {np.sqrt(v):.4f})")
    r1 = vc["pair"] / (vc["pair"] + vc["residual"]) if (vc["pair"] + vc["residual"]) > 0 else float("nan")
    print(f"Pair-interaction reliability: single seed {r1:.3f}; {K}-seed mean {relK:.3f}"
          f" -> ceiling on |rho| with seed-mean interaction ~ {np.sqrt(max(relK,0)):.2f}")
    for name, f in [("relative drop Delta", delta), ("absolute drop", absdiff),
                    ("raw cross-region score", raw), ("within-target centred score", within_target_centered)]:
        m, lo, hi = seed_rank_reliability(Y, f)
        rK = spearman_brown(m, K) if m > 0 else float("nan")
        ceil = np.sqrt(rK) if rK == rK and rK > 0 else float("nan")
        ceil1 = np.sqrt(m) if m > 0 else float("nan")
        print(f"   seed-to-seed Spearman, {name:28s}: mean {m:+.3f} [{lo:+.2f}, {hi:+.2f}]"
              f"   ceiling: 1 seed {ceil1:.2f}, {K}-seed mean {ceil:.2f}")
    g, se, ins, hvb = selection_value(Y)
    print(f"Source selection: cross-seed oracle gain over random foreign source = {g:+.4f} (se {se:.4f});"
          f" in-sample oracle (winner's curse) {ins:+.4f}; home model minus chosen source {hvb:+.4f}")
    print(f"   for scale: mean score {Y.mean():.3f}; residual seed sd per cell {np.sqrt(vc['residual']):.4f};"
          f" source-x-seed sd {np.sqrt(vc['source x seed']):.4f}")
    # the specific pair the review flagged
    if "east_asia" in regs and "andes" in regs:
        s, t = regs.index("east_asia"), regs.index("andes")
        print("   east_asia->andes Delta by seed:", [round(float((Y[k, s, t] - Y[k, t, t]) / Y[k, t, t]), 3) for k in range(K)],
              "| andes in-region by seed:", [round(float(Y[k, t, t]), 3) for k in range(K)])
    return Y, regs, vc


def report_single(fn, label, key="matrix", noise_var=None):
    d = json.load(open(M / fn))
    mat = d[key]
    regs = [r for r in d.get("regions", list(mat.keys())) if r in mat]
    cols = list(next(iter(mat.values())).keys())
    regs = [r for r in regs if r in cols]
    Y = np.array([[mat[s][t] for t in regs] for s in regs], dtype=float)
    S = len(regs)
    mu = Y.mean(); ys, yt = Y.mean(1), Y.mean(0)
    inter = Y - ys[:, None] - yt[None, :] + mu
    tot = ((Y - mu) ** 2).sum()
    sh = {"target": S * ((yt - mu) ** 2).sum() / tot, "source": S * ((ys - mu) ** 2).sum() / tot,
          "pair + seed noise": (inter ** 2).sum() / tot}
    print(f"\n######## {label}  ({fn}: single seed, {S}x{S})")
    for k, v in sh.items():
        print(f"   {k:18s} {100*v:6.2f}%")
    ms_int = (inter ** 2).sum() / ((S - 1) ** 2)
    if noise_var is not None:
        print(f"   interaction MS {ms_int:.3e} vs residual seed-noise var borrowed from 8-region run {noise_var:.3e}"
              f" -> implied single-seed pair reliability ~ {max(0, 1 - noise_var / ms_int):.2f}")
    od = [(s, t) for s in range(S) for t in range(S) if s != t]
    dl = np.array([(Y[s, t] - Y[t, t]) / Y[t, t] for s, t in od])
    print(f"   Delta: mean {dl.mean():+.3f}, sd {dl.std(ddof=1):.3f}, range [{dl.min():+.3f}, {dl.max():+.3f}];"
          f" diag mean {np.mean(np.diag(Y)):.3f}, off mean {np.mean([Y[s,t] for s,t in od]):.3f}")
    return Y, regs


if __name__ == "__main__":
    Yc, regs, vcc = report_multiseed("multiseed_matrix_cnn.json", "HARMONISED GHSL input, CNN, FoM")
    Yu, _, vcu = report_multiseed("multiseed_matrix_unet.json", "HARMONISED GHSL input, U-Net, FoM")
    report_single("full_matrix_n20_cnn.json", "HARMONISED GHSL input, CNN, FoM, 20 regions", noise_var=vcc["residual"])
    report_single("landsat_temporal_matrix_landsat_history.json", "RAW Landsat input, SegUNet, IoU, 8 regions")
    report_single("landsat_temporal_matrix.json", "RAW Landsat input (other run), SegUNet, IoU, 8 regions")
