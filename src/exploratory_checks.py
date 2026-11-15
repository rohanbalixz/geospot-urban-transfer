"""Exploratory robustness checks (NOT pre-registered; added 2026-11-14 after all
pre-registered results were committed). They reuse the committed distance matrices and
receipts and change nothing in results/R*.json.

E1 power-matched contrast: R3 restricted to the same 8 regions as R1/R2 (exact QAP), so
   the R2-vs-R3 difference is not just 8 vs 20 regions.
E2 near-duplicate regions: R3 without the two '_2' expansion regions (south_asia_2,
   japan_korea_2), which sit next to south_asia and east_asia.
E3 outcome that removes source quality too: R3 pair-specific (double-centred) rho is
   reported in results/R3.json; restated here next to E1/E2.
E4 direction of H5 across regimes (none significant individually).
E5 Sinkhorn convergence: re-solve sampled pairs with 10x the iteration budget.

  python src/exploratory_checks.py   -> results/exploratory_checks.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import ot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_analysis as ra  # noqa: E402
from geotransfer import decomposition as dec  # noqa: E402
from geotransfer import inference as inf  # noqa: E402
from geotransfer import ot_distance as otd  # noqa: E402
from geotransfer import selection as sel  # noqa: E402

out = {"note": "exploratory, not pre-registered; reuses committed distances"}
r3 = json.load(open(ROOT / "results/R3.json"))
Y3, regs3 = ra.load_outcome_array(ra.REGIMES["R3"])
D3 = np.array(r3["primary"]["distance"])


def test(idx, label):
    Y, D = Y3[:, idx][:, :, idx], D3[np.ix_(idx, idx)]
    res = {"regions": [regs3[i] for i in idx]}
    for o in ["within_target", "pair_specific"]:
        O = dec.OUTCOMES[o](Y.mean(0))
        rho, p, n, exact = inf.qap_spearman(D, O, n_perm=None if len(idx) <= 8 else 9999, rng=1)
        lo, hi, _ = inf.region_bootstrap(D, O, B=2000, rng=2)
        res[o] = dict(spearman=rho, qap_p=p, exact=exact, boot_ci95=[lo, hi])
    s = sel.selection_summary(Y, D)
    res["selection"] = dict(distance_gain=s["distance_gain"], oracle_gain=s["oracle_gain"],
                            captured_fraction=s.get("captured_fraction"))
    w = res["within_target"]
    print(f"{label}: within-target rho {w['spearman']:+.3f} (p {w['qap_p']:.4f}, CI {w['boot_ci95'][0]:+.2f} to "
          f"{w['boot_ci95'][1]:+.2f}); pair-specific {res['pair_specific']['spearman']:+.3f} (p {res['pair_specific']['qap_p']:.4f}); "
          f"selection {s['distance_gain']:+.3f} of {s['oracle_gain']:+.3f}")
    return res


eight = [regs3.index(r) for r in json.load(open(ROOT / "results/R2.json"))["regions"]]
out["E1_R3_same_8_regions_as_R2"] = test(eight, "E1 R3 on the 8 R1/R2 regions")
no_dup = [i for i, r in enumerate(regs3) if not r.endswith("_2")]
out["E2_R3_without_near_duplicates"] = test(no_dup, "E2 R3 without *_2 regions")
out["E3_R3_pair_specific_full"] = r3["primary"]["tests"]["pair_specific"]["spearman"]
out["E4_H5_by_regime"] = {k: dict(spearman=json.load(open(ROOT / f"results/{k}.json"))["H5_directional_coverage"]["spearman"],
                                  qap_p=json.load(open(ROOT / f"results/{k}.json"))["H5_directional_coverage"]["qap_p"])
                          for k in ["R1", "R2", "R3"]}
print("E4 H5 by regime:", {k: f"{v['spearman']:+.2f} (p {v['qap_p']:.2f})" for k, v in out["E4_H5_by_regime"].items()})

conv = {}
for key in ["R2", "R3"]:
    cfg = ra.REGIMES[key]; Y, regs = ra.load_outcome_array(cfg)
    S = ra.load_samples(cfg, regs, ra.TILE_SEED)
    scales = otd.global_scales(S, ["feature", "geoclip"])
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(len(S)) for j in range(i + 1, len(S))]
    rel, nwarn = [], 0
    for k in rng.choice(len(pairs), 10, replace=False):
        i, j = pairs[k]
        C = otd.combined_cost(S[i], S[j], feat_comp="feature", loc_comp="geoclip", lam=0.5, p=1, scales=scales,
                              normalization="global", feature_metric="cosine")
        a, b = np.full(C.shape[0], 1 / C.shape[0]), np.full(C.shape[1], 1 / C.shape[1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v1 = ot.sinkhorn2(a, b, C, reg=0.01, method="sinkhorn_log", numItermax=5000, stopThr=1e-9)
            nwarn += len(w)
        v2 = ot.sinkhorn2(a, b, C, reg=0.01, method="sinkhorn_log", numItermax=50000, stopThr=1e-12)
        rel.append(abs(float(v1) - float(v2)) / abs(float(v2)))
    conv[key] = dict(sampled_pairs=10, convergence_warnings=nwarn, max_relative_change_10x_iterations=max(rel))
    print(f"E5 {key}: warnings {nwarn}/10 pairs, max relative change at 10x iterations {max(rel):.1e}")
out["E5_sinkhorn_convergence"] = conv
json.dump(out, open(ROOT / "results/exploratory_checks.json", "w"), indent=1)
