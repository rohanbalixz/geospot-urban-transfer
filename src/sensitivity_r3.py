"""R3 sensitivity checks named in PREREGISTRATION.md and DEVIATIONS.md.

1. Collapsed runs ("with/without"): drop every region whose FINAL source model, in any
   seed, still scored < 0.10 IoU in-region after the pre-registered retries, then rerun
   the primary test (GeoSpOT vs within-target score, QAP) and source selection on the
   remaining regions, reusing the committed primary distance matrix.
2. Scoring: in-region vs out-of-region IoU of the same models on held-out test cities
   and on all tiles (training tiles included, as in the published receipt).

  python src/sensitivity_r3.py   -> results/R3_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geotransfer import decomposition as dec  # noqa: E402
from geotransfer import inference as inf  # noqa: E402
from geotransfer import selection as sel  # noqa: E402

d = json.load(open(ROOT / "data/receipts/landsat_matrix_dump_multiseed.json"))
r3 = json.load(open(ROOT / "results/R3.json"))
regs, seeds = d["regions"], [str(s) for s in d["seeds"]]
assert regs == r3["regions"]
out = {}

collapsed = sorted({r for k in seeds for r in regs
                    if (d["per_seed"][k]["training_log"][r][-1]["in_region_test_iou"] or 0) < 0.10})
keep = [i for i, r in enumerate(regs) if r not in collapsed]
Y = np.array([[[d["per_seed"][k]["matrix_test"][s][t] for t in regs] for s in regs] for k in seeds], float)
D = np.array(r3["primary"]["distance"])
Yk, Dk = Y[:, keep][:, :, keep], D[np.ix_(keep, keep)]
O = dec.within_target(Yk.mean(0))
rho, p, nperm, exact = inf.qap_spearman(Dk, O, n_perm=9999, rng=1)
lo, hi, _ = inf.region_bootstrap(Dk, O, B=2000, rng=2)
s = sel.selection_summary(Yk, Dk)
out["without_collapsed"] = dict(dropped=collapsed, n_regions=len(keep), within_target_spearman=rho, qap_p=p,
                                boot_ci95=[lo, hi], selection=s,
                                with_all=dict(spearman=r3["primary"]["tests"]["within_target"]["spearman"],
                                              qap_p=r3["primary"]["tests"]["within_target"]["qap_p"]))
off = ~np.eye(len(regs), dtype=bool)
for key in ["matrix_test", "matrix_all_tiles"]:
    M = np.array([[[d["per_seed"][k][key][s][t] for t in regs] for s in regs] for k in seeds], float)
    dg = float(np.mean([np.diag(M[k]).mean() for k in range(len(seeds))]))
    of = float(np.mean([M[k][off].mean() for k in range(len(seeds))]))
    out[f"scoring_{key}"] = dict(in_region=dg, out_region=of, home_field_gap=dg - of, retention=of / dg)
out["training"] = dict(models=len(seeds) * len(regs),
                       retrained=sum(len(d["per_seed"][k]["training_log"][r]) > 1 for k in seeds for r in regs),
                       still_collapsed=[(r, k) for k in seeds for r in regs
                                        if (d["per_seed"][k]["training_log"][r][-1]["in_region_test_iou"] or 0) < 0.10])
json.dump(out, open(ROOT / "results/R3_sensitivity.json", "w"), indent=1)
w = out["without_collapsed"]
print(f"without {collapsed}: rho {w['within_target_spearman']:+.3f} (QAP p {w['qap_p']:.4f}, CI {lo:+.2f} to {hi:+.2f}); "
      f"with all: rho {w['with_all']['spearman']:+.3f} (p {w['with_all']['qap_p']:.4f})")
for key in ["matrix_test", "matrix_all_tiles"]:
    x = out[f"scoring_{key}"]
    print(f"{key:17s} gap {x['home_field_gap']:+.3f}, retention {x['retention']:.2f}")
