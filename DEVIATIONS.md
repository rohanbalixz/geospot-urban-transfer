# Deviations and run log (after pre-registration commit `075e436`, 2026-11-14 16:42)

Everything that happened after the plan was frozen, including things that are not
deviations from the analysis, so the run can be audited.

## Operational events (no change to the analysis)

1. **R1/R2 run interrupted once (2026-11-14, about 16:58).** While stopping a stalled
   Landsat download I also killed the shell running `run_analysis.py --regimes R1 R2`.
   R1 had finished; R2 was part-way through its secondary distances. Both were rerun from
   scratch with identical code and inputs. The first R1 output is kept as
   `logs/R1_first_run.json`. The rerun reproduces it exactly: all 1,302 values match except
   the 8 NaN diagonal entries of the coverage matrix (NaN never equals NaN).
2. **Landsat download timeouts.** The repo's `landsat_history.py` hung on stalled HTTPS
   reads (GDAL has no default timeout). It was rerun with GDAL network settings only
   (`GDAL_HTTP_TIMEOUT=60`, `GDAL_HTTP_MAX_RETRY=6`, and others; see `logs/`), 4 cities at a
   time. The scene-selection rule and the code are unchanged. Two cities (Bergen,
   Stavanger) failed on dropped connections and one (Bucharest) stalled in a catalogue
   call. All three were re-run individually and succeeded, so all 220 cities were acquired
   (patch fill 72-100%, median 93%; `logs/landsat_fill.txt`).
3. **`data/processed` rebuild check.** For the 44 cities whose original tile records
   survive (TGIS repo), the rebuilt tile grid matches exactly: 786 of 786 tile IDs
   (`data/processed_rebuild_check.json`).

4. **History cleanup before first push (2026-11-14, 17:27).** Two personal planning
   documents that were not part of the study had been committed with the
   pre-registration. They were removed from every commit before anything was pushed.
   Author and commit dates were preserved, and `PREREGISTRATION.md` is byte-identical.
   Commit hashes changed: the pre-registration commit `fe7438e` (the hash printed in
   `logs/run_R1_R2.log`) is now `075e436`, and the R1/R2 results commit `cc1fad6` is now
   `6aa612f`.

5. **R3 timing test (2026-11-14, about 17:40).** Before the full R3 run, one throwaway
   seed (99, not among the pre-registered seeds) was trained on 2 regions (Andes, SSA)
   to time the pipeline (about 6 s per model). Its receipt and weights were deleted. No
   distance was involved.

## Deviations from the pre-registered analysis

1. **How the with/without collapse check is defined (a clarification, not a change).** The
   pre-registration names a with/without check for collapsed R3 runs but not its exact
   form. Because the matrices are scored with final models only, the check drops every
   region whose final model, in any seed, still scored below 0.10 after the two retries.
   That was 1 region (southern_africa, seed 1, 0.096; 13 of 100 models were retrained).
   The primary test is then rerun on the remaining 19 (`src/sensitivity_r3.py`,
   `results/R3_sensitivity.json`).
2. **Added diagnostic (not pre-registered): held-out vs all-tile scoring of the same R3
   models.** Reported because it changes how the published raw-Landsat retention should be
   read: in-region 0.384 vs 0.461, home-field gap +0.112 vs +0.187, retention 0.71 vs 0.59.
   The out-of-region score is unchanged (0.272 vs 0.274).
