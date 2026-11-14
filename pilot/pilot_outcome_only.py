"""Outcome-side checks only (no distance is computed, so a later pre-registration stays blind).

1. Do CNN and U-Net agree on WHICH pairs transfer well (same inputs, same regions)?
   If not, any model-agnostic distance has a ceiling set by the shared part.
2. Raw Landsat: do the two single-seed 8x8 runs agree on pair structure?
3. Raw Landsat 20-region, 5 seeds: how noisy is the in-region (diagonal) score that
   GeoSpOT's relative drop divides by?
Usage: python -I pilot_outcome_only.py <metrics_dir>
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

M = Path(sys.argv[1])


def ms(fn):
    d = json.load(open(M / fn))
    seeds = [str(s) for s in d["seeds"]]
    regs = list(d["per_seed"][seeds[0]]["matrix"].keys())
    return np.array([[[d["per_seed"][k]["matrix"][s][t] for t in regs] for s in regs] for k in seeds]), regs


def single(fn):
    d = json.load(open(M / fn)); mat = d["matrix"]; regs = d["regions"]
    return np.array([[mat[s][t] for t in regs] for s in regs], dtype=float), regs


def dc(Y):  # double-centre: pair-specific part
    return Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()


def wtc(Y):  # within-target centring over foreign sources, off-diagonal cells
    S = Y.shape[0]
    return np.array([Y[s, t] - np.mean([Y[u, t] for u in range(S) if u != t]) for s in range(S) for t in range(S) if s != t])


def off(A):
    S = A.shape[0]
    return np.array([A[s, t] for s in range(S) for t in range(S) if s != t])


def split_half_rel(Y, f):
    """Mean split-half (2 vs 3 seeds) rank reliability of f(seed-mean), Spearman-Brown to 5 seeds."""
    K = Y.shape[0]; rs = []
    for A in itertools.combinations(range(K), 2):
        B = [k for k in range(K) if k not in A]
        rs.append(spearmanr(f(Y[list(A)].mean(0)), f(Y[B].mean(0))).correlation)
    r = float(np.mean(rs)); return r, 2 * r / (1 + r) if r > 0 else float("nan")


Yc, regs = ms("multiseed_matrix_cnn.json")
Yu, regs_u = ms("multiseed_matrix_unet.json")
assert regs == regs_u
print("regions:", regs)

for name, f in [("pair-specific (double-centred)", lambda Y: off(dc(Y))), ("within-target centred", wtc)]:
    rc, rc5 = split_half_rel(Yc, f); ru, ru5 = split_half_rel(Yu, f)
    obs = spearmanr(f(Yc.mean(0)), f(Yu.mean(0))).correlation
    dis = obs / np.sqrt(rc5 * ru5) if rc5 > 0 and ru5 > 0 else float("nan")
    print(f"\n[{name}] split-half reliability (5-seed, Spearman-Brown): CNN {rc5:.2f}, U-Net {ru5:.2f}")
    print(f"   CNN vs U-Net agreement on seed-mean structure: Spearman {obs:+.2f}; disattenuated {dis:+.2f}")

# home model vs foreign: how often is the target's own model the best source? (seed-mean)
for lab, Y in [("CNN", Yc), ("U-Net", Yu)]:
    m = Y.mean(0); S = m.shape[0]
    rank_home = [int((m[:, t] > m[t, t]).sum()) + 1 for t in range(S)]
    print(f"\n{lab}: rank of the target's own model among 8 sources (1 = best), seed-mean: {rank_home}")

# raw Landsat: two single-seed 8x8 runs
A, ra = single("landsat_temporal_matrix_landsat_history.json")
B, rb = single("landsat_temporal_matrix.json")
assert ra == rb
ok = np.isfinite(A).all() and np.isfinite(B).all()
print(f"\nRAW Landsat two 8x8 runs (landsat_history vs landsat8), finite={ok}")
print(f"   raw off-diagonal score agreement: Spearman {spearmanr(off(A), off(B)).correlation:+.2f}")
print(f"   within-target centred agreement:  Spearman {spearmanr(wtc(A), wtc(B)).correlation:+.2f}")
print(f"   pair-specific agreement:          Spearman {spearmanr(off(dc(A)), off(dc(B))).correlation:+.2f}")
print(f"   source means run1: {np.round(A.mean(1),2)}")
print(f"   source means run2: {np.round(B.mean(1),2)}")
print(f"   diag run1: {np.round(np.diag(A),2)}")
print(f"   diag run2: {np.round(np.diag(B),2)}")

# raw Landsat n20 5-seed: noise in the diagonal
d = json.load(open(M / "landsat_temporal_matrix_landsat_history_g64_n20_multiseed.json"))
seeds = [str(s) for s in d["seeds"]]; regs20 = d["regions"]
D = np.array([[d["per_seed"][k]["diag"][r] for r in regs20] for k in seeds])
sd = D.std(0, ddof=1); mn = D.mean(0)
print(f"\nRAW Landsat 20 regions x 5 seeds, in-region IoU (the Delta denominator):")
print(f"   median across-seed SD {np.median(sd):.3f}; median CV {np.median(sd/mn):.2f}; max SD {sd.max():.3f} ({regs20[int(sd.argmax())]})")
print(f"   seeds with a collapsed in-region model (IoU < 0.1): "
      f"{[(regs20[j], seeds[k], D[k,j]) for k in range(len(seeds)) for j in range(len(regs20)) if D[k,j] < 0.1]}")
rs = [spearmanr(D[a], D[b]).correlation for a, b in itertools.combinations(range(len(seeds)), 2)]
print(f"   seed-to-seed Spearman of in-region IoU across regions: mean {np.mean(rs):+.2f} [{min(rs):+.2f}, {max(rs):+.2f}]")
agg = d["aggregate"]
print("   aggregate:", {k: round(v["mean"], 3) for k, v in agg.items()})
