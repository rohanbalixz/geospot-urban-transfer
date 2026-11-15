# geospot-urban-transfer

[![ci](https://github.com/rohanbalixz/geospot-urban-transfer/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanbalixz/geospot-urban-transfer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**How much of cross-region transfer can a distance between regions predict, and how much
of that does GeoSpOT capture?** This is a pre-registered test on dense urban-change transfer
matrices (8 and 20 world regions, 5 training seeds each).

GeoSpOT ([Zhang, Betti, Klemmer, Rolf & Alvarez-Melis, 2026](https://arxiv.org/abs/2604.16220))
predicts the accuracy drop between geographic domains from an optimal-transport distance
over features and location embeddings. Its paper evaluates classification and lists other
tasks as open. This repo tests it on a dense prediction task. First it measures how much of
the transfer structure any distance could predict, then where GeoSpOT falls inside that
limit.

The plan and analysis code were committed (`54220f8`, 2026-11-14 16:42 EST) before any
distance between regions was computed. Everything after that, including the checks that
weaken one reading of the results, is in [`DEVIATIONS.md`](DEVIATIONS.md).

## Findings

1. **A large part of a transfer matrix is out of reach of any distance.** A distance is
   symmetric, noise-free and the same for every model. Transfer is none of those. From the
   matrices alone:
   - only 29-37% of the variance in the relative drop is symmetric;
   - training luck (source × seed) is 3-18% of the variance;
   - a CNN and a U-Net trained on the same inputs disagree about which source is best for a
     target (Spearman −0.41 after correcting for seed noise).
2. **With 20 regions on raw Landsat inputs, GeoSpOT predicts which source transfers better.**
   - ρ = −0.27 (QAP p = 0.0006, Holm-adjusted 0.0012, 95% CI −0.44 to −0.08), 38% of the
     achievable ceiling.
   - Choosing the GeoSpOT-nearest source recovers half of the best achievable gain:
     +0.072 IoU (CI +0.034 to +0.122) against +0.144.
   - It holds when source quality is removed too (pair-specific outcome, ρ = −0.28),
     without the two near-duplicate regions (−0.30), and without the one region whose model
     stayed collapsed (−0.30).
3. **On 8 regions, GeoSpOT shows no detectable relationship in any setting.**
   - Harmonised inputs: CNN ρ = +0.10 and U-Net ρ = +0.02. The U-Net's pair structure is
     reproducible across seeds (reliability 0.84), yet GeoSpOT reaches 3% of its ceiling.
   - Raw Landsat restricted to the same 8 regions: ρ = −0.09 (exact p = 0.64).
   - With 8 regions this test can detect only |ρ| ≳ 0.29, so **this study cannot say whether
     the input type matters, or only the number of regions.**
4. **Learned location embeddings predict about as well as image features; raw geography
   does not.** In R3, features alone give ρ = −0.24, GeoCLIP alone −0.26 and SatCLIP-L40
   alone −0.29; the pre-registered comparison (H3) finds no difference. Geodesic distance
   gives −0.01, so "nearby regions are similar" does not explain the effect.
5. **Directional coverage did not predict the asymmetric part of the drop (H5).** It was
   positive in all three regimes (+0.59, +0.37, +0.20) and significant in none.

![Variance decomposition and distance tests](results/figure.png)

*Top: where cross-region variation comes from. Bottom: each distance's Spearman ρ with the
within-target score, with its 95% region-bootstrap CI. Shaded areas are beyond the joint
ceiling. R1 and R2 have 8 regions, R3 has 20; on R3's same 8 regions, GeoSpOT gives
ρ = −0.09 (p = 0.64).*

## Pre-registered hypotheses and outcomes

| | Hypothesis ([`PREREGISTRATION.md`](PREREGISTRATION.md)) | Outcome |
|---|---|---|
| Step 0 | Outcomes with 5-seed reliability < 0.5 are not interpreted | Only R1's relative drop is gated out (0.48) |
| H1 | Report ceilings and CNN-U-Net agreement | Joint ceilings 0.64-0.72 (within-target); learners disagree (−0.41) |
| H2 | ρ(GeoSpOT, within-target) < 0 in R2 and R3 (Holm) | **R3 supported** (Holm p = 0.0012); R2 not (p = 0.91) |
| H3 | Features beat location alone in R3 | Not supported (difference −0.01, p = 0.87) |
| H4 | Nearest-source gain > 0 in R2 and R3 | **R3 supported** (+0.072, CI excludes 0); R2 not (−0.010, CI −0.026 to +0.006) |
| H5 | Coverage asymmetry predicts the antisymmetric drop | Not supported in any regime |

## Primary results

The primary outcome is the **within-target score**: a foreign source's score on a target
minus the mean score of all foreign sources on that target. It removes how hard the target
is and never uses the target's own model.

| Regime | Input → model, metric | Regions × seeds | Joint ceiling | GeoSpOT ρ | QAP p | Share of ceiling | Nearest-source gain vs oracle |
|---|---|---|---|---|---|---|---|
| R1 (negative control) | harmonised GHSL → CNN, FoM | 8 × 5 | 0.71 | +0.10 | 0.70 (exact) | 14% | −0.001 vs +0.003 |
| R2 | harmonised GHSL → U-Net, FoM | 8 × 5 | 0.64 | +0.02 | 0.91 (exact) | 3% | −0.010 vs +0.020 |
| R3 | raw Landsat → SegUNet, IoU | 20 × 5 | 0.72 | **−0.27** | **0.0006** | 38% | **+0.072 vs +0.144** |

- **Joint ceiling:** √(5-seed reliability) × the best Spearman any symmetric predictor can
  reach. Negative ρ means a closer source transfers better.
- **R3 scoring:** on held-out test cities. 13 of 100 models were retrained under the
  pre-registered collapse rule.
- **Full tables** (relative drop, pair-specific outcome, 9 secondary distances, H3, H5):
  [`results/SUMMARY.md`](results/SUMMARY.md). The secondary distances are not corrected for
  multiple comparisons. In R2, one of them (SatCLIP-L40 alone, ρ = +0.37, p = 0.009) points
  the wrong way.

**A side result (not pre-registered).** Scoring R3's in-region cell on held-out cities
instead of on all tiles changes the raw-Landsat home-field gap from +0.187 to +0.112 and
retention from 0.59 to 0.71. The out-of-region score barely moves. Part of the raw-input
"drop" reported in the cross-region study was in-sample scoring
([`results/R3_sensitivity.json`](results/R3_sensitivity.json)).

## What this does and does not show

- **It does show** that, on a 20-region dense transfer task, a GeoSpOT distance computed
  without target labels ranks sources usefully. Picking the nearest source recovers about
  half of what an oracle could, and the result is robust to the checks above.
- **It does not show** that harmonised inputs remove the signal. R3 differs from R1/R2 in
  input, model, task, metric, scoring and number of regions. On matched regions, no regime
  shows an effect.
- **It does show** that any evaluation of a transferability distance should report how much
  of the transfer matrix is predictable at all. On this task, seed noise, asymmetry and the
  learner bound it well below 1.

## Verify in seconds, without data

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy scipy pot pytest
./.venv/bin/python -m pytest -q tests/      # method checks on synthetic inputs
./.venv/bin/python verify.py                # 23 checks re-derived from committed files
```

`verify.py` checks that the pre-registration was committed before every results file. It
also recomputes the reliability and ceilings from the transfer-matrix receipts, the primary
ρ and its QAP p-value from the committed distance matrices, the selection gains, the
robustness checks, and the Holm adjustment. CI runs both on every push.

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
src/geotransfer/     decomposition, inference (QAP, bootstrap), selection, OT distances,
                     coverage, GHSL features, location embeddings, data rebuild
src/run_analysis.py  the pre-registered analysis; refuses to run before the plan is committed
src/sensitivity_r3.py, src/exploratory_checks.py   R3 collapse check; robustness checks
results/             R1-R3 JSON, COMBINED.json, SUMMARY.md, figure, sensitivity, exploratory
pilot/               outcome-only pilots run before the freeze (no distances)
data/                transfer-matrix receipts, cities, derived features and embeddings
rerun/               R3 training script (drop-in for the cross-region repo) and the original
tests/, verify.py    method checks and result re-derivation (run in CI)
```

## Limits

- One task family (built-up mapping and change), three regimes, 8 or 20 regions. No claim
  goes beyond these.
- R1 and R2 reuse published transfer matrices whose in-region cell was scored on all tiles.
  The primary outcome avoids that cell; the relative drop does not.
- R3 inputs were re-acquired, because the originals are gone. The rebuilt tile grid matches
  the original on 786 of 786 tiles where it can be checked, but Landsat scenes may differ
  from the first pull.
- R3's distance uses every tile of each region, including the source's test cities. Its
  features are inputs only, with no labels.
- GeoSpOT is run as in its released code (linear cost, ε = 0.01, cosine,
  SatCLIP-ResNet50), with a debiased divergence and a global cost scale as the primary
  variant. The released-code variant is reported alongside: it agrees in R3 (ρ = −0.21,
  p = 0.005), and in R1 and R2 neither variant is significant.

## Data and acknowledgements

GHSL R2023A (European Commission, JRC; CC BY 4.0). Landsat Collection 2 Level-2 (USGS),
via Microsoft Planetary Computer. GeoCLIP and SatCLIP location encoders (MIT). Training
code from [Cross-Region Source-Invariance in Earth Observation](https://github.com/rohanbalixz/Cross-Region-Source-Invariance-in-Earth-Observation)
(`649e9ad`). GeoSpOT reference implementation:
[haoranzhang7/GeoSpOT](https://github.com/haoranzhang7/GeoSpOT).

## Citation

See [`CITATION.cff`](CITATION.cff). Code is MIT-licensed ([`LICENSE`](LICENSE)); vendored
SatCLIP modules keep their MIT licence in `src/vendor/satclip_le/`.
