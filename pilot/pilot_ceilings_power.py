"""Pilot 3: ceilings and detectable effects. Outcome-side only.

No real distance is computed here (synthetic geometries only), so the
pre-registration stays blind to how any region distance relates to transfer.

  (a) symmetric ceiling: a symmetric distance d(s,t) = d(t,s) can only predict the
      symmetric part of an outcome. How much of each outcome is symmetric?
  (b) learner agreement: do CNN and U-Net (same inputs, same regions) agree on the
      transfer structure, after correcting for seed noise?
  (c) minimum detectable |rho| under an exact/MC QAP test with 8 and 20 regions.

Usage: python pilot/pilot_ceilings_power.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geotransfer import decomposition as dec  # noqa: E402
from geotransfer import inference as inf  # noqa: E402
from geotransfer import selection as sel  # noqa: E402

RC = ROOT / "data/receipts"
Yc, regs, _ = dec.load_multiseed(RC / "multiseed_matrix_cnn.json")
Yu, _, _ = dec.load_multiseed(RC / "multiseed_matrix_unet.json")
La, _ = dec.load_single(RC / "landsat_temporal_matrix_landsat_history.json")
OUTS = ["relative_drop", "absolute_drop", "within_target", "pair_specific"]

print("(a) reliability, noise ceiling and symmetric ceiling of each outcome (5-seed means)")
print(f"{'regime':28s} {'outcome':15s} {'rel(5)':>7s} {'ceiling':>8s} {'sym share':>10s} {'sym rank ceil':>13s} {'joint ceiling':>14s}")
for lab, Y in [("R1 harmonised, CNN", Yc), ("R2 harmonised, U-Net", Yu)]:
    for o in OUTS:
        f = dec.OUTCOMES[o]
        rel = dec.split_half_reliability(Y, f)
        ceil = dec.noise_ceiling(rel)
        share, rsym = dec.symmetric_ceiling(f(Y.mean(0)))
        print(f"{lab:28s} {o:15s} {rel:7.2f} {ceil:8.2f} {share:10.2f} {rsym:13.2f} {ceil * rsym:14.2f}")
for o in OUTS:
    share, rsym = dec.symmetric_ceiling(dec.OUTCOMES[o](La))
    print(f"{'R3 raw Landsat (1 seed)':28s} {o:15s} {'n/a':>7s} {'n/a':>8s} {share:10.2f} {rsym:13.2f} {'n/a':>14s}")

print("\n(b) do the two learners agree on transfer structure? (same inputs, same 8 regions)")
for o in OUTS:
    r, dis, r1, r2 = dec.learner_agreement(Yc, Yu, dec.OUTCOMES[o])
    print(f"   {o:15s} CNN vs U-Net Spearman {r:+.2f}; corrected for seed noise {dis:+.2f}  (rel CNN {r1:.2f}, U-Net {r2:.2f})")

print("\n(c) selection value available (cross-seed oracle) and home model, in task units")
for lab, Y in [("R1 harmonised, CNN", Yc), ("R2 harmonised, U-Net", Yu)]:
    s = sel.selection_summary(Y)
    print(f"   {lab:22s} oracle {s['oracle_gain']:+.4f} | in-sample oracle {s['insample_oracle_gain']:+.4f}"
          f" | home model vs random foreign {s['home_gain']:+.4f} | mean score {Y.mean():.3f}")

print("\n(d) minimum |rho| a QAP test can call significant (two-sided 5%), synthetic geometries")
rng = np.random.default_rng(20261114)


def critical_rho(O, S, reps, n_perm):
    crit = []
    for _ in range(reps):
        P = rng.normal(size=(S, 2)); D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        ii, jj = np.where(~np.eye(S, dtype=bool))
        null = []
        for _ in range(n_perm):
            p = rng.permutation(S)
            null.append(inf.spearman_offdiag(D[np.ix_(p, p)], O))
        crit.append(np.quantile(np.abs(null), 0.95))
    return float(np.median(crit)), float(np.quantile(crit, 0.1)), float(np.quantile(crit, 0.9))


for lab, O in [("R2 U-Net relative drop, 8 regions", dec.relative_drop(Yu.mean(0))),
               ("R2 U-Net within-target, 8 regions", dec.within_target(Yu.mean(0)))]:
    m, lo, hi = critical_rho(O, 8, reps=40, n_perm=400)
    print(f"   {lab:38s} critical |rho| median {m:.2f} (10-90%: {lo:.2f}-{hi:.2f})")
# 20 regions: synthetic outcome with raw-Landsat-like structure (target 25%, source 36%, rest 39%)
S = 20
a = rng.normal(0, 0.5, S); b = rng.normal(0, 0.6, S); g = rng.normal(0, 0.62, (S, S))
O20 = a[None, :] + b[:, None] + g; np.fill_diagonal(O20, np.nan)
m, lo, hi = critical_rho(dec.within_target(np.nan_to_num(O20)), S, reps=20, n_perm=300)
print(f"   {'synthetic within-target, 20 regions':38s} critical |rho| median {m:.2f} (10-90%: {lo:.2f}-{hi:.2f})")
