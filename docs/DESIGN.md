# Design: what can a region distance predict?

This note explains the reasoning behind `PREREGISTRATION.md`. The plan itself is in that
file. Pilot numbers come from `pilot/outputs/`, computed before the plan was frozen,
using only transfer outcomes and no region distances.

## The problem with testing a distance directly

GeoSpOT (Zhang, Betti, Klemmer, Rolf, Alvarez-Melis, 2026) scores a region distance by
its Spearman correlation with the relative accuracy drop across region pairs. That number
mixes two things: how good the distance is, and how much of the drop any distance could
predict at all. On dense urban-change transfer, the second part is small, for three
reasons. Each one sets a ceiling that can be computed from the transfer matrix alone.

1. **Seed noise.** Retraining every source model with a new seed reshuffles which pairs
   lose. A cell's score is a property of one training run as much as of the region pair.
   The noise ceiling is √reliability, where reliability is the 5-seed split-half rank
   reliability of the outcome (Spearman–Brown corrected).
2. **Symmetry.** Every OT divergence is symmetric, d(s,t) = d(t,s), but transfer is not.
   The relative drop Δ(s→t) carries the difference in quality between the source model and
   the target's own model, which flips sign for Δ(t→s). The symmetric ceiling is the best
   Spearman any symmetric predictor can reach: the correlation between the ranks of the
   outcome and the pair-averaged ranks.
3. **The learner.** A dataset distance is the same for every model. Two architectures
   trained on the same inputs, in the same regions, can disagree about which pairs lose.
   No model-agnostic distance can fit both.

The **joint ceiling** used in the results is noise ceiling × symmetric ceiling.

## Pilot evidence (outcome side only)

| | R1 harmonised, CNN | R2 harmonised, U-Net | R3 raw Landsat (old run, 1 seed) |
|---|---|---|---|
| Target-region share of variance | 95% | 58% | 25% |
| Pair-specific share | 0.4% | 12% | 39% incl. noise |
| Training luck (source × seed) | 2.6% | 18% | n/a |
| Δ 5-seed reliability | 0.48 | 0.84 | n/a |
| Symmetric share of Δ | 0.29 | 0.29 | 0.31 |
| Joint ceiling, Δ | 0.24 | 0.40 | n/a |
| Joint ceiling, within-target score | 0.71 | 0.64 | n/a |
| Cross-seed oracle gain from choosing a source | +0.003 FoM | +0.020 FoM | n/a |

CNN and U-Net trained on the same inputs disagree on Δ (−0.62 after correcting for seed
noise) and on which foreign source is best for a target (−0.41).

## Choices that follow

- **Primary outcome: within-target score.** Each foreign source's score on a target, minus
  the mean over foreign sources. It removes target difficulty, never uses the target's own
  model (whose diagonal score is in-sample in the published matrices), and answers the
  operational question: for this target, does a closer source transfer better? Δ is kept
  for comparability with GeoSpOT.
- **Decision value.** Choosing the GeoSpOT-nearest source is scored in task units against a
  cross-seed oracle: the best source chosen on 2 seeds and scored on the other 3. The
  in-sample oracle overstates the available gain about fourfold in R1 (winner's curse).
- **Inference for dyadic data.** Cells share regions, so pairs are not independent.
  P-values use QAP, relabelling the distance matrix's rows and columns together: exact over
  all 8! = 40,320 relabellings for 8 regions, 9,999 random relabellings for 20. Intervals
  come from a region bootstrap.
- **Regimes.** R1 and R2 share inputs, so any input-based distance gives the same answer
  for both, and their disagreement is measured, not assumed. R3 swaps in raw Landsat
  reflectance, where the published home-field gap is large (+0.21 IoU), as a regime where
  a distance should have the most to find.
- **A directional alternative (H5).** Covariate coverage cov(t|s), the share of target
  tiles inside the source's support, is asymmetric by construction. It is tested against
  the antisymmetric part of Δ, which GeoSpOT cannot reach.

## Distance implementation

The GeoSpOT cost is λ·C_feature + (1−λ)·C_location with λ = 0.5. As in the released code,
it is linear (p = 1), uses cosine costs, ε = 0.01, and SatCLIP-ResNet50 L40/L10 or GeoCLIP
location embeddings. Two choices differ from the released code, and the released-code
variant is reported alongside:

- **Debiased Sinkhorn divergence** S_ε = OT_ε(s,t) − ½OT_ε(s,s) − ½OT_ε(t,t), instead of
  OT_ε(s,t) alone, so identical regions score 0.
- **One global cost scale** shared by all pairs, instead of each pair's own maximum, so
  distances are comparable across pairs.

Features describe the model's own inputs: mean and SD of each input channel per tile, with
one global standardisation. Each region contributes 200 tiles (or all, if fewer), and
distance stability is checked over 5 tile resamples.
