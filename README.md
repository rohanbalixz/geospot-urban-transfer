# geospot-urban-transfer

[![ci](https://github.com/rohanbalixz/geospot-urban-transfer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rohanbalixz/geospot-urban-transfer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**How much of cross-region transfer can a distance between regions predict, and how much
of that does GeoSpOT capture?** This is a pre-registered test on dense built-up mapping
across world regions, with 5 training seeds per cell:

- **R1 and R2:** nowcasting built-up change from the GHSL history (8 regions).
- **R3:** mapping 2015 built-up extent from a raw Landsat history (20 regions).

GeoSpOT ([Zhang, Betti, Klemmer, Rolf & Alvarez-Melis, 2026](https://arxiv.org/abs/2604.16220))
predicts the accuracy drop between geographic domains from an optimal-transport distance
over features and location embeddings. Its paper evaluates classification and lists other
tasks as open. This repo first measures how much of the transfer structure any distance
could predict, then where GeoSpOT falls inside that limit, and then whether it helps choose
a source model.

The plan and analysis code were committed (`54220f8`, 2026-11-14 16:42 EST) before any
distance between regions was computed. Everything after that is in
[`DEVIATIONS.md`](DEVIATIONS.md), including checks that weaken some readings of the
results. One caveat: the freeze is recorded only by git commit time. The history was later
rewritten to remove private files (dates kept), and the plan was first pushed together with
the results, so there is no independent timestamp.

## Findings

1. **A large part of a transfer matrix is out of reach of any distance.** A distance is
   symmetric, noise-free and the same for every model. From the matrices alone:
   - only 29-37% of the variance in the relative drop is symmetric;
   - training luck (source × seed) is 3-18% of the variance;
   - whether a CNN and a U-Net trained on the same inputs agree about which source is best
     cannot be detected with 8 regions (ρ = −0.31, exact permutation p = 0.19, CI −0.78 to
     +0.44).
2. **In R3 (20 regions), GeoSpOT tracks which pairs of regions transfer well.**
   - ρ = −0.27 with the within-target score (QAP p = 0.0006, Holm 0.0012, CI −0.44 to
     −0.08), 38% of the achievable ceiling.
   - It holds when source quality is removed (pair-specific outcome −0.28; partial
     correlation given source quality −0.31), without each target's nearest source (−0.24,
     p = 0.002), without the near-duplicate regions (−0.30), and without the one collapsed
     model (−0.30).
3. **For choosing a source model, the source's overall quality matters far more, and a
   dataset distance cannot see it.** In R3:
   - picking the source that transfers best to the *other* regions (no target labels, no
     distance) recovers 91% of the oracle's gain;
   - GeoSpOT's nearest source recovers 50%, the same as plain geographic distance (the two
     pick the same source for 15 of 20 targets);
   - GeoSpOT distance is nearly uncorrelated with source quality (ρ = −0.05).

   In R2 the best-overall-source rule captures all of the available gain, while GeoSpOT's
   pick does no better than random.
4. **On 8 regions, the GeoSpOT test is uninformative in every setting.** R1 gives ρ = +0.10
   and R2 ρ = +0.02 (share of ceiling, 95% CI −78% to +75%). R3 restricted to the same 8
   regions gives ρ = −0.09 (p = 0.64). With 8 regions the test detects only |ρ| ≳ 0.29, so
   **this study cannot say whether the input type matters, or only the number of regions.**
5. **Location embeddings rank pairs about as well as input-band statistics.** In R3,
   input-band statistics alone give ρ = −0.24, GeoCLIP −0.26 and SatCLIP-L40 −0.29, and the
   pre-registered comparison (H3) finds no difference. Geodesic distance gives −0.01 over
   all pairs.
6. **Directional coverage did not predict the asymmetric part of the drop (H5).** It was
   positive in all three regimes (+0.59, +0.37, +0.20) and significant in none.

![Variance decomposition and distance tests](results/figure.png)

*Top: where cross-region variation comes from. Bottom: each distance's Spearman ρ with the
within-target score (95% region-bootstrap CI); panel titles compare rules for choosing a
source. R1 and R2 have 8 regions, R3 has 20.*

## Pre-registered hypotheses and outcomes

| | Hypothesis ([`PREREGISTRATION.md`](PREREGISTRATION.md)) | Outcome |
|---|---|---|
| Step 0 | Outcomes with 5-seed reliability < 0.5 are not interpreted | Only R1's relative drop is gated out (0.48) |
| H1 | Report ceilings and CNN-U-Net agreement | Joint ceilings 0.64-0.72 (within-target); agreement not detectable (p = 0.19, exploratory E8) |
| H2 | ρ(GeoSpOT, within-target) < 0 in R2 and R3 (Holm) | **R3 supported** (Holm p = 0.0012); R2 uninformative (p = 0.91) |
| H3 | Input-band statistics beat location alone in R3 | Not supported (difference −0.01, p = 0.87) |
| H4 | Nearest-source gain > 0 in R2 and R3 | **R3 supported** (+0.072, CI excludes 0); R2 not (−0.010, CI −0.026 to +0.006). See E6 for the baseline the plan left out |
| H5 | Coverage asymmetry predicts the antisymmetric drop | Not supported in any regime |

## Primary results

The primary outcome is the **within-target score**: a foreign source's score on a target
minus the mean score of all foreign sources on that target. It removes how hard the target
is and never uses the target's own model.

| Regime | Task, input → model, metric | Regions × seeds | Joint ceiling | GeoSpOT ρ | QAP p | Share of ceiling |
|---|---|---|---|---|---|---|
| R1 (negative control) | change nowcast, GHSL history → CNN, FoM | 8 × 5 | 0.71 | +0.10 | 0.70 (exact) | uninformative |
| R2 | change nowcast, GHSL history → U-Net, FoM | 8 × 5 | 0.64 | +0.02 | 0.91 (exact) | uninformative (CI −78% to +75%) |
| R3 | 2015 extent, Landsat history → SegUNet, IoU | 20 × 5 | 0.72 | **−0.27** | **0.0006** | 38% (CI 11% to 61%) |

**Choosing a source model** (cross-seed protocol: choose on 2 seeds, score on the other 3;
gain over a random foreign source, as a share of the oracle's gain; exploratory check E6):

| Rule | R2 (FoM) | R3 (IoU) | Uses |
|---|---|---|---|
| Oracle (best source on held-out seeds) | +0.020 (100%) | +0.144 (100%) | target labels |
| **Best overall source** (best mean score on the other regions) | **+0.020 (103%)** | **+0.132 (91%)** | no target labels, no distance |
| Nearest by SatCLIP-L40 | −0.010 | +0.100 (69%) | location |
| Rank-sum of source quality and GeoSpOT | +0.002 | +0.086 (60%) | both |
| Nearest by GeoSpOT (pre-registered H4) | −0.010 | +0.072 (50%) | band statistics + location |
| Nearest by geodesic distance | −0.003 | +0.072 (50%) | location |

- **Joint ceiling:** √(5-seed reliability) × the best Spearman any symmetric predictor can
  reach. Negative ρ means a closer source transfers better.
- **R3 scoring:** on held-out test cities.
- **Full tables** (relative drop, pair-specific outcome, 9 secondary distances, H3, H5):
  [`results/SUMMARY.md`](results/SUMMARY.md). Robustness and selection baselines:
  [`results/exploratory_checks.json`](results/exploratory_checks.json). The secondary
  distances are not corrected for multiple comparisons. In R2, one of them (SatCLIP-L40
  alone, ρ = +0.37, p = 0.009) points the wrong way.

**A side result (not pre-registered).** Scoring R3's in-region cell on held-out cities
instead of on all tiles changes the raw-Landsat home-field gap from +0.187 to +0.112 and
retention from 0.59 to 0.71. Part of the raw-input "drop" reported in the cross-region study
was in-sample scoring ([`results/R3_sensitivity.json`](results/R3_sensitivity.json)).

## What this does and does not show

- **It does show** that a GeoSpOT distance computed without target labels tracks which
  region pairs transfer well, on a 20-region dense task. The effect survives every
  robustness check above.
- **It does show** that, for picking a source model, a dataset distance is the wrong tool on
  this task. How good the source model is overall matters more, and a distance between
  datasets cannot see it.
- **It does not show** that harmonised inputs remove the signal. R3 differs from R1/R2 in
  input, task, model, metric, scoring and number of regions, and on matched regions no
  regime shows an effect.
- **It does show** that a transferability evaluation should report how much of the
  transfer matrix is predictable at all. Here, seed noise, asymmetry and the learner bound
  it well below 1.

## Verify in seconds, without data

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy scipy pot pytest
./.venv/bin/python -m pytest -q tests/      # method checks on synthetic inputs
./.venv/bin/python verify.py                # 26 checks re-derived from committed files
```

`verify.py` checks that the pre-registration was committed before every results file. It
also recomputes the reliability and ceilings from the transfer-matrix receipts, the primary
ρ and its QAP p-value from the committed distance matrices, the selection gains and
baselines, the robustness checks, and the Holm adjustment. CI runs both on every push.

## Reproduce everything

| Step | Command | Needs | Time (M3 Pro) |
|---|---|---|---|
| Environment | `make env` | Python 3.11+ (`PYTHON=...`) | 5 min |
| Tile features, R1/R2 | `make features GHSL_DIR=/path/to/ghsl` | global GHSL R2023A rasters | 1 min |
| Location embeddings | `make embeddings` | downloads SatCLIP-ResNet50 checkpoints | 2 min |
| R3 inputs | `make external processed landsat` | about 4 GB from Planetary Computer, no account | 1-2 h |
| R3 matrix | `make r3-matrix` | Apple MPS or CUDA | 10 min |
| Analysis | `make analysis figure verify` | | 30 min |

Derived tile features and embeddings are committed in [`data/features/`](data/README.md), so
`make analysis` reruns every distance and test without downloading imagery. A fresh clone
reruns R1 bit for bit.

## Layout

```
PREREGISTRATION.md   the frozen plan (commit 54220f8)
DEVIATIONS.md        everything after the freeze: operational events, clarifications, added checks
docs/DESIGN.md       why the ceilings, the outcomes and the inference are what they are
src/geotransfer/     decomposition, inference (QAP, bootstrap), selection rules, OT distances,
                     coverage, GHSL features, location embeddings, data rebuild
src/run_analysis.py  the pre-registered analysis; refuses to run before the plan is committed
src/sensitivity_r3.py, src/exploratory_checks.py   R3 collapse check; checks E1-E9
results/             R1-R3 JSON, COMBINED.json, SUMMARY.md, figure, sensitivity, exploratory
pilot/               outcome-only pilots run before the freeze (no distances)
data/                transfer-matrix receipts, cities, derived features and embeddings
rerun/               R3 training script (drop-in for the cross-region repo) and the original
tests/, verify.py    method checks and result re-derivation (run in CI)
```

## Limits

- **Scope.** One family of built-up tasks, three regimes, 8 or 20 regions. No claim goes
  beyond these.
- **Features.** The "features" are per-tile mean and SD of each input band, not the learned
  encoder features (ResNet-50, BERT) GeoSpOT uses. H3 is about these statistics only.
- **R3 models are small.** 474k parameters, 15 epochs, about 6 s each. 13 of 100 collapsed
  on the first try and were retrained under the pre-registered rule; one stayed below 0.10.
- **R1 and R2** reuse published transfer matrices whose in-region cell was scored on all
  tiles. The primary outcome avoids that cell; the relative drop does not.
- **R3 inputs were re-acquired,** because the originals are gone. The rebuilt tile grid
  matches the original on 786 of 786 tiles where it can be checked, but Landsat scenes may
  differ from the first pull. R3's distance uses every tile of each region, including the
  source's test cities, with inputs only and no labels.
- **The plan left out a baseline.** It had no rule for choosing a source that ignores
  distance (E6, added after an external review).
- **GeoSpOT settings.** It is run as in its released code (linear cost, ε = 0.01, cosine,
  SatCLIP-ResNet50), with a debiased divergence and a global cost scale as the primary
  variant. The released-code variant agrees in R3 (ρ = −0.21, p = 0.005), and in R1 and R2
  neither variant is significant.

## Data and acknowledgements

GHSL R2023A (European Commission, JRC; CC BY 4.0). Landsat Collection 2 Level-2 (USGS),
via Microsoft Planetary Computer. GeoCLIP and SatCLIP location encoders (MIT). Training
code from [Cross-Region Source-Invariance in Earth Observation](https://github.com/rohanbalixz/Cross-Region-Source-Invariance-in-Earth-Observation)
(`649e9ad`). GeoSpOT reference implementation:
[haoranzhang7/GeoSpOT](https://github.com/haoranzhang7/GeoSpOT).

## Citation

See [`CITATION.cff`](CITATION.cff). Code is MIT-licensed ([`LICENSE`](LICENSE)); vendored
SatCLIP modules keep their MIT licence in `src/vendor/satclip_le/`.
