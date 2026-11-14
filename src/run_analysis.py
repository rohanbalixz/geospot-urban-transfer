"""The pre-registered analysis, end to end. Frozen together with PREREGISTRATION.md.

  python src/run_analysis.py --regimes R1 R2      # refuses unless PREREGISTRATION.md is committed
  python src/run_analysis.py --regimes R3
  python src/run_analysis.py --combine-only       # Holm over R2+R3, learner agreement, SUMMARY.md
Writes results/<regime>/*.json and results/SUMMARY.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geotransfer import coverage as cov  # noqa: E402
from geotransfer import decomposition as dec  # noqa: E402
from geotransfer import inference as inf  # noqa: E402
from geotransfer import ot_distance as otd  # noqa: E402
from geotransfer import selection as sel  # noqa: E402

REGIMES = {
    "R1": dict(label="harmonised GHSL input, CNN, FoM", receipt="data/receipts/multiseed_matrix_cnn.json",
               loader="multiseed", features="data/features/ghsl"),
    "R2": dict(label="harmonised GHSL input, U-Net, FoM", receipt="data/receipts/multiseed_matrix_unet.json",
               loader="multiseed", features="data/features/ghsl"),
    "R3": dict(label="raw Landsat input, SegUNet, IoU (city-disjoint test)",
               receipt="data/receipts/landsat_matrix_dump_multiseed.json",
               loader="dump", features="data/features/landsat"),
}
N_TILES, TILE_SEED, N_RESAMPLE = 200, 0, 5
PRIMARY = dict(feat_comp="feature", loc_comp="geoclip", lam=0.5, p=1, eps=0.01, debiased=True,
               normalization="global", feature_metric="cosine")
SECONDARY = {
    "location_only_geoclip": dict(PRIMARY, lam=0.0),
    "feature_only": dict(PRIMARY, lam=1.0),
    "combined_satclip_l40": dict(PRIMARY, loc_comp="satclip_l40"),
    "location_only_satclip_l40": dict(PRIMARY, loc_comp="satclip_l40", lam=0.0),
    "location_only_satclip_l10": dict(PRIMARY, loc_comp="satclip_l10", lam=0.0),
    "location_only_geodesic": dict(PRIMARY, loc_comp="geodesic", lam=0.0),
    "released_code_style": dict(PRIMARY, debiased=False, normalization="pair_max"),
    "eps_0.005": dict(PRIMARY, eps=0.005),
    "eps_0.05": dict(PRIMARY, eps=0.05),
}
OUTCOMES = ["within_target", "relative_drop", "pair_specific"]   # first is primary


# ---------------------------------------------------------------- guards and loading
def assert_preregistered():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%h %ci", "--", "PREREGISTRATION.md"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        out = ""
    if not out:
        sys.exit("PREREGISTRATION.md is not committed. Commit it before computing any distance.")
    print("pre-registration commits:\n" + out)


def load_outcome_array(cfg):
    if cfg["loader"] == "multiseed":
        Y, regs, _ = dec.load_multiseed(ROOT / cfg["receipt"])
        return Y, regs
    d = json.load(open(ROOT / cfg["receipt"]))
    regs = d["regions"]; seeds = [str(s) for s in d["seeds"]]
    Y = np.array([[[d["per_seed"][k]["matrix_test"][s][t] for t in regs] for s in regs] for k in seeds], float)
    return Y, regs


def load_samples(cfg, regs, rng_seed):
    rng = np.random.default_rng(rng_seed)
    fdir = ROOT / cfg["features"]
    raw = {}
    for r in regs:
        z = np.load(fdir / f"{r}.npz")
        n = len(z["features"])
        idx = np.sort(rng.choice(n, size=min(N_TILES, n), replace=False))
        emb = {}
        for e in ["geoclip", "satclip_l40", "satclip_l10"]:
            p = fdir / f"{r}__{e}.npy"
            if p.exists():
                emb[e] = np.load(p)[idx]
        raw[r] = (z["features"][idx].astype(float), z["latlon"][idx], emb)
    allF = np.concatenate([raw[r][0] for r in regs])
    mu, sd = allF.mean(0), allF.std(0) + 1e-9          # ONE global standardisation
    return [otd.RegionSample(r, features=(raw[r][0] - mu) / sd + 0.0, latlon=raw[r][1], embeddings=raw[r][2])
            for r in regs]


# ---------------------------------------------------------------- one regime
def ceilings(Y):
    out = {}
    share, vc, _, _ = dec.anova3(Y)
    tot = sum(vc.values())
    out["variance_components_share"] = {k: v / tot for k, v in vc.items()}
    for o in OUTCOMES:
        f = dec.OUTCOMES[o]
        rel = dec.split_half_reliability(Y, f)
        sshare, srank = dec.symmetric_ceiling(f(Y.mean(0)))
        out[o] = dict(reliability_5seed=rel, noise_ceiling=dec.noise_ceiling(rel), symmetric_share=sshare,
                      symmetric_rank_ceiling=srank, joint_ceiling=dec.noise_ceiling(rel) * srank,
                      evaluable=bool(rel == rel and rel >= 0.5))
    out["selection_value"] = sel.selection_summary(Y)
    return out


def test_distance(D, Y, ceil, n_boot=2000):
    res = {}
    Ym = Y.mean(0)
    for o in OUTCOMES:
        O = dec.OUTCOMES[o](Ym)
        rho, p, nperm, exact = inf.qap_spearman(D, O, n_perm=None if D.shape[0] <= 8 else 9999, rng=1)
        lo, hi, nb = inf.region_bootstrap(D, O, B=n_boot, rng=2)
        jk = inf.region_jackknife(D, O)
        jc = ceil[o]["joint_ceiling"]
        res[o] = dict(spearman=rho, qap_p=p, qap_perms=nperm, qap_exact=exact, boot_ci95=[lo, hi], boot_n=nb,
                      jackknife_range=[float(np.nanmin(jk)), float(np.nanmax(jk))],
                      within_target_spearman=inf.within_target_spearman(D, O),
                      fraction_of_joint_ceiling=(abs(rho) / jc) if jc > 0 else None,
                      evaluable=ceil[o]["evaluable"])
    s = sel.selection_summary(Y, D)
    s["boot_ci95_gain_and_fraction"] = sel.selection_bootstrap(Y, D, B=n_boot, rng=3)
    res["selection"] = s
    return res


def directional_test(samples, Y):
    """H5: does coverage asymmetry predict the antisymmetric part of the relative drop?"""
    C = cov.coverage_matrix([s.features for s in samples])          # C[s,t] = cov(t|s)
    A_drop = cov.antisym_upper(dec.relative_drop(Y.mean(0)))
    A_cov = cov.antisym_upper(C)
    rho = float(spearmanr(A_cov, A_drop).correlation)
    # QAP on antisymmetric matrices: relabel regions jointly
    S = C.shape[0]; rng = np.random.default_rng(4)
    Ad = dec.relative_drop(Y.mean(0)); Ad = Ad - Ad.T; Ac = C - C.T
    iu = np.triu_indices(S, 1)
    null = []
    perms = (np.array(p) for p in __import__("itertools").permutations(range(S))) if S <= 8 else \
            (rng.permutation(S) for _ in range(9999))
    for p in perms:
        null.append(spearmanr(Ac[np.ix_(p, p)][iu], Ad[iu]).correlation)
    null = np.abs(np.array(null))
    pval = float(np.mean(null >= abs(rho) - 1e-12))
    return dict(spearman=rho, qap_p=pval, n_pairs=int(len(A_drop)), coverage_matrix=C.tolist())


def run_regime(key):
    cfg = REGIMES[key]
    Y, regs = load_outcome_array(cfg)
    out = {"regime": key, "label": cfg["label"], "regions": regs, "n_seeds": int(Y.shape[0])}
    out["ceilings"] = ceilings(Y)
    samples = load_samples(cfg, regs, TILE_SEED)
    D, asym = otd.distance_matrix(samples, **PRIMARY)
    out["primary"] = dict(config=PRIMARY, asymmetry=asym, distance=D.tolist(), tests=test_distance(D, Y, out["ceilings"]))
    print(f"  {key}: primary distance done", flush=True)
    # distance reliability over tile resamples
    Ds = [D] + [otd.distance_matrix(load_samples(cfg, regs, TILE_SEED + i), **PRIMARY)[0] for i in range(1, N_RESAMPLE)]
    rs = [inf.spearman_offdiag(Ds[a], Ds[b]) for a in range(len(Ds)) for b in range(a + 1, len(Ds))]
    out["primary"]["tile_resample_reliability"] = float(np.mean(rs))
    out["secondary"] = {}
    mats = {}
    for name, c in SECONDARY.items():
        needed = c["loc_comp"]
        if c["lam"] < 1 and needed != "geodesic" and needed not in samples[0].embeddings:
            out["secondary"][name] = "skipped: embedding not available"; continue
        Dn, a2 = otd.distance_matrix(samples, **c)
        mats[name] = Dn
        out["secondary"][name] = dict(config=c, asymmetry=a2, distance=Dn.tolist(),
                                      tests=test_distance(Dn, Y, out["ceilings"], n_boot=500))
        print(f"  {key}: {name} done", flush=True)
    # H3: feature-only vs location-only (GeoCLIP), within-target outcome, paired region bootstrap
    if "feature_only" in mats and "location_only_geoclip" in mats:
        O = dec.within_target(Y.mean(0))
        d, lo, hi, p = inf.paired_region_bootstrap_diff(mats["feature_only"], mats["location_only_geoclip"], O, B=2000, rng=5)
        out["H3_feature_vs_location"] = dict(diff_abs_rho=d, ci95=[lo, hi], p=p)
    out["H5_directional_coverage"] = directional_test(samples, Y)
    return out


def combine(outdir):
    """Cross-regime steps: Holm over the H2 family (R2, R3) and R1-R2 learner agreement."""
    res = {k: json.load(open(outdir / f"{k}.json")) for k in REGIMES if (outdir / f"{k}.json").exists()}
    comb = {"regimes_available": list(res)}
    fam = [k for k in ("R2", "R3") if k in res]
    if fam:
        ps = [res[k]["primary"]["tests"]["within_target"]["qap_p"] for k in fam]
        adj = inf.holm(ps) if len(fam) == 2 else ps
        comb["H2_holm"] = {k: dict(p=p, p_holm=float(a), reject_at_0_05=bool(a < 0.05))
                           for k, p, a in zip(fam, ps, adj)}
        comb["H2_family_complete"] = len(fam) == 2
    if "R1" in res and "R2" in res:
        Yc, _ = load_outcome_array(REGIMES["R1"]); Yu, _ = load_outcome_array(REGIMES["R2"])
        comb["H1_learner_agreement_R1_R2"] = {
            o: dict(zip(["spearman", "noise_corrected", "rel_R1", "rel_R2"], dec.learner_agreement(Yc, Yu, dec.OUTCOMES[o])))
            for o in OUTCOMES}
    json.dump(comb, open(outdir / "COMBINED.json", "w"), indent=1)
    return res, comb


def summarise(res, comb):
    L = ["# Results summary (auto-generated by src/run_analysis.py)\n"]
    for k, r in res.items():
        c = r["ceilings"]; t = r["primary"]["tests"]
        L.append(f"## {k}: {r['label']} ({len(r['regions'])} regions, {r['n_seeds']} seeds)\n")
        vc = c["variance_components_share"]
        L.append("Variance: " + ", ".join(f"{a} {100 * v:.1f}%" for a, v in sorted(vc.items(), key=lambda kv: -kv[1])) + "\n")
        L.append("Primary distance: GeoSpOT debiased Sinkhorn, 0.5 feature + 0.5 GeoCLIP, global scale, eps 0.01.\n")
        L.append("| outcome | reliability | joint ceiling | evaluable | GeoSpOT rho | QAP p | 95% CI | share of ceiling |")
        L.append("|---|---|---|---|---|---|---|---|")
        for o in OUTCOMES:
            x, cc = t[o], c[o]
            frac = f"{x['fraction_of_joint_ceiling']:.2f}" if x["fraction_of_joint_ceiling"] is not None else "n/a"
            L.append(f"| {o} | {cc['reliability_5seed']:.2f} | {cc['joint_ceiling']:.2f} | {cc['evaluable']} | "
                     f"{x['spearman']:+.2f} | {x['qap_p']:.4f} | [{x['boot_ci95'][0]:+.2f}, {x['boot_ci95'][1]:+.2f}] | {frac} |")
        s = t["selection"]; ci = s["boot_ci95_gain_and_fraction"]
        L.append(f"\nSource selection: oracle gain {s['oracle_gain']:+.4f}; GeoSpOT-nearest gain {s['distance_gain']:+.4f} "
                 f"(95% CI {ci[0]:+.4f} to {ci[1]:+.4f}); captured {s.get('captured_fraction', float('nan')):.2f} "
                 f"(CI {ci[2]:.2f} to {ci[3]:.2f}); home model {s['home_gain']:+.4f}\n")
        L.append("| secondary distance | within-target rho | QAP p | relative-drop rho | selection gain |")
        L.append("|---|---|---|---|---|")
        for name, v in r["secondary"].items():
            if isinstance(v, str):
                L.append(f"| {name} | {v} | | | |"); continue
            w, d = v["tests"]["within_target"], v["tests"]["relative_drop"]
            L.append(f"| {name} | {w['spearman']:+.2f} | {w['qap_p']:.4f} | {d['spearman']:+.2f} | {v['tests']['selection']['distance_gain']:+.4f} |")
        if "H3_feature_vs_location" in r:
            h = r["H3_feature_vs_location"]
            L.append(f"\nH3 |rho| feature-only minus location-only (within-target): {h['diff_abs_rho']:+.2f} "
                     f"(95% CI {h['ci95'][0]:+.2f} to {h['ci95'][1]:+.2f}, p {h['p']:.3f})")
        h5 = r["H5_directional_coverage"]
        L.append(f"\nH5 coverage asymmetry vs antisymmetric relative drop: rho {h5['spearman']:+.2f}, QAP p {h5['qap_p']:.4f} ({h5['n_pairs']} pairs)\n")
        L.append(f"Distance stability over {N_RESAMPLE} tile resamples: {r['primary']['tile_resample_reliability']:.2f}\n")
    L.append("## Cross-regime\n")
    if "H2_holm" in comb:
        for k, v in comb["H2_holm"].items():
            L.append(f"- H2 {k}: QAP p {v['p']:.4f}, Holm {v['p_holm']:.4f}, reject at 0.05: {v['reject_at_0_05']}")
        if not comb["H2_family_complete"]:
            L.append("- H2 family incomplete (R3 not yet run): Holm not applied.")
    if "H1_learner_agreement_R1_R2" in comb:
        for o, v in comb["H1_learner_agreement_R1_R2"].items():
            L.append(f"- H1 CNN vs U-Net agreement, {o}: {v['spearman']:+.2f}, noise-corrected {v['noise_corrected']:+.2f}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["R1", "R2"])
    ap.add_argument("--combine-only", action="store_true")
    a = ap.parse_args()
    outdir = ROOT / "results"
    assert_preregistered()
    outdir.mkdir(exist_ok=True)
    if not a.combine_only:
        for k in a.regimes:
            r = run_regime(k)
            json.dump(r, open(outdir / f"{k}.json", "w"), indent=1)
            print(f"{k} done", flush=True)
    res, comb = combine(outdir)
    (outdir / "SUMMARY.md").write_text(summarise(res, comb))
    print((outdir / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
