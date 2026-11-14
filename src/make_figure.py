"""Figure for the README: what a distance can predict, and what GeoSpOT reached.

Top: share of cross-region variance by source of variation, per regime.
Bottom: Spearman rho of each distance with the within-target outcome (95% region-
bootstrap CI), against the joint ceiling (shaded: unreachable by any symmetric,
model-agnostic distance at this seed noise) and the QAP 5% critical value.

  python src/make_figure.py      -> results/figure.png, results/figure.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # reference palette slots 1-5, fixed order
COMP = [("target", "Target region"), ("source", "Source region"), ("pair", "Pair-specific"),
        ("source_x_seed", "Training luck (source × seed)"), ("rest", "Other seed noise")]
ROWS = [("primary", "GeoSpOT (feature + GeoCLIP)"), ("feature_only", "Feature only"),
        ("location_only_geoclip", "GeoCLIP only"), ("location_only_satclip_l40", "SatCLIP L40 only"),
        ("location_only_geodesic", "Geodesic only")]
CRIT = {8: 0.29, 20: 0.15}   # pilot QAP critical |rho|, within-target (pre-registration)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load():
    return {k: json.load(open(RES / f"{k}.json")) for k in ("R1", "R2", "R3") if (RES / f"{k}.json").exists()}


def main():
    res = load()
    keys = list(res)
    fig = plt.figure(figsize=(3.3 * len(keys) + 1.2, 6.2))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1, 2.3], hspace=0.55, wspace=0.12)

    # --- top: variance components, one stacked bar per regime (2px surface gaps)
    axv = fig.add_subplot(gs[0, :])
    for yi, k in enumerate(keys):
        vc = res[k]["ceilings"]["variance_components_share"]
        vals = [vc["target"], vc["source"], vc["pair"], vc["source_x_seed"],
                vc["residual"] + vc.get("seed", 0) + vc["target_x_seed"]]
        left = 0.0
        for (cid, lab), v, col in zip(COMP, vals, SERIES):
            axv.barh(yi, v, left=left, height=0.62, color=col, edgecolor=SURFACE, linewidth=2,
                     label=lab if yi == 0 else None)
            if v >= 0.08:
                axv.text(left + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8, color=INK)
            left += v
    axv.set_yticks(range(len(keys)), [f"{k}: {res[k]['label'].split(',')[0]}, {res[k]['label'].split(',')[1].strip()}" for k in keys])
    axv.invert_yaxis(); axv.set_xlim(0, 1); axv.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    axv.set_title("Where cross-region variation comes from (5 seeds per cell)", loc="left", color=INK, fontsize=10)
    for s in ("top", "right"):
        axv.spines[s].set_visible(False)
    axv.legend(ncol=5, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.28), handlelength=1.2)

    # --- bottom: rho per distance, per regime
    for j, k in enumerate(keys):
        r = res[k]; ax = fig.add_subplot(gs[1, j])
        jc = r["ceilings"]["within_target"]["joint_ceiling"]
        S = len(r["regions"]); crit = CRIT.get(S, 0.15)
        ax.axvspan(-1, -jc, color=GRID, lw=0); ax.axvspan(jc, 1, color=GRID, lw=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        for c in (-crit, crit):
            ax.axvline(c, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        for yi, (name, lab) in enumerate(ROWS):
            t = r["primary"]["tests"] if name == "primary" else (r["secondary"].get(name) or {}).get("tests") \
                if isinstance(r["secondary"].get(name), dict) else None
            if t is None:
                continue
            w = t["within_target"]; col = SERIES[0] if name == "primary" else INK2
            ax.plot(w["boot_ci95"], [yi, yi], color=col, lw=2, solid_capstyle="round")
            ax.plot(w["spearman"], yi, "o", ms=8, color=col, mec=SURFACE, mew=2)
            pv = w["qap_p"]; ptxt = "p < 0.001" if pv < 0.001 else (f"p {pv:.3f}" if pv < 0.01 else f"p {pv:.2f}")
            ax.text(0.98, yi - 0.32, f"ρ {w['spearman']:+.2f}, {ptxt}", ha="right", va="center",
                    fontsize=7, color=INK2, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS] if j == 0 else [""] * len(ROWS))
        ax.set_ylim(len(ROWS) - 0.4, -0.7); ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ with within-target score")
        sel = r["primary"]["tests"]["selection"]
        ax.set_title(f"{k} ({S} regions): ceiling ±{jc:.2f}\nnearest-source gain {sel['distance_gain']:+.3f} vs oracle {sel['oracle_gain']:+.3f}",
                     loc="left", fontsize=8.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Shaded: beyond the joint ceiling (seed noise × symmetry). Dashed: QAP 5% critical |ρ|. "
             "Bars: 95% region-bootstrap CI. Negative ρ = closer source transfers better.", fontsize=7, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"figure.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", RES / "figure.png")


if __name__ == "__main__":
    main()
