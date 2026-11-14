# Deviations and run log (after pre-registration commit `fe7438e`, 2026-11-14 16:42)

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
   time. The scene-selection rule and the code are unchanged.
3. **`data/processed` rebuild check.** For the 44 cities whose original tile records
   survive (TGIS repo), the rebuilt tile grid matches exactly: 786 of 786 tile IDs
   (`data/processed_rebuild_check.json`).

## Deviations from the pre-registered analysis

None so far.
