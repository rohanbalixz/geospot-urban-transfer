"""Re-derive the headline results from committed files only (no data, no GPU).

Checks:
  1. PREREGISTRATION.md was committed before any results file (git history).
  2. Step-0 ceilings (reliability, symmetric and joint ceilings, variance components)
     recomputed from the transfer-matrix receipts match results/R*.json.
  3. The primary GeoSpOT correlation, its exact/MC QAP p-value and the source-selection
     gain, recomputed from the committed distance matrix, match results/R*.json.
  4. Holm-adjusted H2 p-values in results/COMBINED.json follow from the per-regime p-values.

  python verify.py            # exit code 0 if every check passes
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from geotransfer import decomposition as dec  # noqa: E402
from geotransfer import inference as inf  # noqa: E402
from geotransfer import selection as sel  # noqa: E402

RECEIPTS = {"R1": ("multiseed", "data/receipts/multiseed_matrix_cnn.json"),
            "R2": ("multiseed", "data/receipts/multiseed_matrix_unet.json"),
            "R3": ("dump", "data/receipts/landsat_matrix_dump_multiseed.json")}
ok_all, n_checks = True, 0


def check(name, cond, detail=""):
    global ok_all, n_checks
    n_checks += 1
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load_Y(kind, path):
    if kind == "multiseed":
        return dec.load_multiseed(ROOT / path)[0]
    d = json.load(open(ROOT / path)); regs = d["regions"]; seeds = [str(s) for s in d["seeds"]]
    return np.array([[[d["per_seed"][k]["matrix_test"][s][t] for t in regs] for s in regs] for k in seeds], float)


def first_commit_time(path):
    out = subprocess.run(["git", "-C", str(ROOT), "log", "--diff-filter=A", "--format=%ct", "--", path],
                         capture_output=True, text=True).stdout.split()
    return int(out[-1]) if out else None


# 1. commit order
t_pre = first_commit_time("PREREGISTRATION.md")
res_files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "results").glob("R*.json"))
t_res = [first_commit_time(f) for f in res_files]
if t_pre is None or any(t is None for t in t_res):
    print("[SKIP] commit order (needs full git history: clone without --depth)")
else:
    check("pre-registration committed before every results file", all(t > t_pre for t in t_res),
          f"prereg {t_pre}, earliest result {min(t_res)}")

# 2-3. per regime
for key, (kind, rpath) in RECEIPTS.items():
    rf = ROOT / "results" / f"{key}.json"
    if not rf.exists() or not (ROOT / rpath).exists():
        print(f"[SKIP] {key}: results or receipt not present"); continue
    r = json.load(open(rf)); Y = load_Y(kind, rpath)
    c = r["ceilings"]
    for o in ["within_target", "relative_drop", "pair_specific"]:
        f = dec.OUTCOMES[o]
        rel = dec.split_half_reliability(Y, f)
        _, srank = dec.symmetric_ceiling(f(Y.mean(0)))
        check(f"{key} {o}: reliability and joint ceiling",
              np.isclose(rel, c[o]["reliability_5seed"]) and np.isclose(dec.noise_ceiling(rel) * srank, c[o]["joint_ceiling"]),
              f"rel {rel:.3f}, joint {dec.noise_ceiling(rel) * srank:.3f}")
    _, vc, _, _ = dec.anova3(Y); tot = sum(vc.values())
    check(f"{key}: variance components", all(np.isclose(vc[k] / tot, v) for k, v in c["variance_components_share"].items()),
          f"target {vc['target'] / tot:.3f}, pair {vc['pair'] / tot:.3f}")
    D = np.array(r["primary"]["distance"])
    O = dec.within_target(Y.mean(0))
    t = r["primary"]["tests"]["within_target"]
    rho, p, nperm, exact = inf.qap_spearman(D, O, n_perm=None if D.shape[0] <= 8 else 9999, rng=1)
    check(f"{key}: primary GeoSpOT rho and QAP p (within-target)",
          np.isclose(rho, t["spearman"]) and np.isclose(p, t["qap_p"]),
          f"rho {rho:+.3f}, p {p:.4f}, {'exact' if exact else 'MC'} {nperm}")
    s = sel.selection_summary(Y, D); s0 = r["primary"]["tests"]["selection"]
    check(f"{key}: source-selection gains", np.isclose(s["distance_gain"], s0["distance_gain"]) and
          np.isclose(s["oracle_gain"], s0["oracle_gain"]),
          f"nearest {s['distance_gain']:+.4f} vs oracle {s['oracle_gain']:+.4f}")

# 4. Holm
cf = ROOT / "results/COMBINED.json"
if cf.exists():
    comb = json.load(open(cf))
    if comb.get("H2_family_complete"):
        ks = list(comb["H2_holm"]); ps = [comb["H2_holm"][k]["p"] for k in ks]
        adj = inf.holm(ps)
        check("H2 Holm adjustment over R2 and R3", all(np.isclose(a, comb["H2_holm"][k]["p_holm"]) for k, a in zip(ks, adj)),
              ", ".join(f"{k} {a:.4f}" for k, a in zip(ks, adj)))
    else:
        print("[SKIP] Holm: H2 family incomplete")

print(f"\n{'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'} ({n_checks} checks)")
sys.exit(0 if ok_all else 1)
