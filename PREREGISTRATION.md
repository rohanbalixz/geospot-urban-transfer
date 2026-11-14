# Pre-registration: what can a region distance predict on urban transfer?

Rohan Bali. Frozen 2026-11-14. This file, `src/`, and `tests/` are committed before
any distance between regions is computed. `src/run_analysis.py` refuses to run on real
data until this file is in the git history.

## What was known before this was written

On 2026-11-14 I ran outcome-side pilots (`pilot/`, outputs in `pilot/outputs/`) on the
committed receipts of the cross-region study
(github.com/rohanbalixz/Cross-Region-Source-Invariance-in-Earth-Observation, commit
`649e9ad`): variance components, seed reliability, symmetric shares, learner agreement,
selection value, and QAP critical values on synthetic geometries. **No distance between
real regions (geodesic, location-embedding, feature or OT) had been computed** when this
was frozen. The ceilings for R1 and R2 below are therefore known in advance; the
GeoSpOT results are not.

## Question

GeoSpOT (Zhang, Betti, Klemmer, Rolf, Alvarez-Melis, 2026) predicts the relative accuracy
drop between regions from an OT distance over features and location. Here I ask how much
of cross-region transfer any region distance can predict on a dense urban-change task,
and how much of that GeoSpOT captures.

## Data

Transfer matrices: one model trained per source region, scored on every target region.

| Regime | Input | Model | Metric | Regions x seeds |
|---|---|---|---|---|
| R1 | GHSL history (8 epochs x built-up, volume, population) | CNN | FoM | 8 x 5 |
| R2 | same | U-Net | FoM | 8 x 5 |
| R3 | raw Landsat history (4 epochs x 6 bands = 24 channels) | SegUNet | IoU, city-disjoint test cities | 20 x 5 |

R1 and R2 are the published receipts `multiseed_matrix_{cnn,unet}.json`. R3 is a new run
(`rerun/landsat_matrix_dump.py`): fixed city-disjoint split (30% of cities per region,
chosen by SHA-256 of the city name). Every cell is scored on the target's test cities. A
source model with in-region test IoU < 0.10 is retrained with seed + 1000, at most twice;
the primary analysis uses the replacement, and a with/without check is reported.

The original R3 inputs are no longer available, so they are re-acquired. Per-city GHSL
rasters (`data/processed`) are rebuilt from the global GHSL R2023A rasters with the
repo's conditioning (`src/geotransfer/build_processed.py`), and the Landsat history is
re-streamed from Microsoft Planetary Computer with the repo's own script
(`scripts/acquire/landsat_history.py`, same scene rule). The rebuilt tile grid is checked
against the original tile IDs that survive for 44 cities. Scenes can differ from the
original pull, so R3 is a new run, not a reproduction.

## Distances

Each region is represented by 200 tiles (fixed random draw, seed 0; all tiles if fewer).

- **Features:** mean and SD per input channel per tile (R1/R2: 24 channels; R3: 24 channels;
  48 numbers each),
  standardised with one global mean and SD over all regions in the regime.
- **Location:** GeoCLIP and SatCLIP (L40, L10) embeddings at tile centres; haversine
  distance for geodesic.
- **Primary distance D\*:** debiased Sinkhorn divergence
  S_ε = OT_ε(s,t) − ½OT_ε(s,s) − ½OT_ε(t,t), with ground cost
  0.5·C_feat^p + 0.5·C_loc^p. C_feat is cosine dissimilarity of features, C_loc is cosine
  dissimilarity of GeoCLIP embeddings, and each is divided by one global scale (the max
  over all region pairs). ε = 0.01, p = 1 (the released GeoSpOT code combines costs
  linearly: λ·first + (1−λ)·second, with `reg_e: 0.01`).
- **Secondary:** λ = 0 (location only: GeoCLIP, SatCLIP L40, SatCLIP L10, geodesic),
  λ = 1 (feature only), combined with SatCLIP L40, the released-code style (no self-terms,
  per-pair max normalisation), and ε ∈ {0.005, 0.05}. SatCLIP checkpoints are
  `microsoft/SatCLIP-ResNet50-L40` and `-L10`, the ones GeoSpOT's
  `location_embeddings/utils.py` loads.
- **Directional coverage:** cov(t|s) is the share of target tiles whose nearest source tile
  (standardised features) is within τ_s, where τ_s is the 95th percentile of the source's
  leave-one-out nearest-neighbour distances.

## Outcomes (computed on 5-seed means)

1. **Primary: within-target score.** Score of source s on target t, minus the mean score of
   all foreign sources on t (s ≠ t only).
2. Relative drop Δ = (A_t(M_s) − A_t(M_t)) / A_t(M_t), as in GeoSpOT. In R1 and R2 the
   diagonal A_t(M_t) is scored on all tiles, training tiles included (published receipts).
3. Pair-specific interaction (double-centred off-diagonal matrix).
4. Selection gain: score of the source with the smallest D\* to t, minus the mean over
   foreign sources. Compared with a cross-seed oracle (best source chosen on 2 seeds,
   scored on the other 3, averaged over all splits).

## Step 0: what can be evaluated

Per regime and outcome, I report variance components (target, source, pair, source × seed,
target × seed, residual) and 5-seed split-half reliability with Spearman–Brown. I also
report the noise ceiling √reliability, the symmetric rank ceiling (best Spearman any
symmetric predictor can reach), their product (the joint ceiling), and R1–R2 learner
agreement corrected for seed noise.

**Gate:** an outcome with 5-seed reliability < 0.5 is reported but not interpreted.

## Hypotheses and tests

All correlations are Spearman over off-diagonal cells, tested by QAP (rows and columns of
the distance matrix relabelled together): exact enumeration of all 40,320 relabellings for
8 regions, and 9,999 random relabellings for 20. 95% CIs come from a region bootstrap
(2,000 draws, cells between copies of the same region dropped), with a leave-one-region-out
range.

- **H2 (primary):** ρ(D\*, within-target score) < 0, two-sided QAP, in R2 and R3, with Holm
  correction across these two tests at α = 0.05. R1 is reported as a negative control: its
  oracle selection gain was +0.003 FoM in the pilot (1% of the mean), so little is at stake.
  Every ρ is also reported as a share of the joint ceiling.
- **H3:** in R3, |ρ| for feature-only is greater than for location-only (GeoCLIP), tested
  with a region bootstrap of the difference.
- **H4:** in R2 and R3, the GeoSpOT-nearest source has a positive selection gain, with a
  target-bootstrap 95% CI excluding 0. I report the share of the oracle gain it captures.
- **H5:** the antisymmetric part of Δ, Δ(s→t) − Δ(t→s), correlates with coverage asymmetry
  cov(t|s) − cov(s|t) over unordered pairs (QAP). D\* is symmetric, so it cannot predict
  this part.

Minimum detectable effect (pilot, synthetic geometries): QAP critical |ρ| is about 0.29 for
the within-target outcome with 8 regions, and about 0.15 with 20.

## What would change the conclusions

- If D\* reaches ≥ 0.5 of the joint ceiling in R2, GeoSpOT sees pair structure even where the
  mean home-field gap is zero.
- If D\* is not significant in R3, where the drop is large and reliable, GeoSpOT misses a
  real shift on this task.
- If location-only matches feature-only in R3, location carries the shift here.
- If H5 holds, directional coverage reaches the part of transfer a divergence cannot.

## Deviations

Any change after this commit is listed in `DEVIATIONS.md` with its date and reason. No
claims go beyond these regions, inputs and models.
