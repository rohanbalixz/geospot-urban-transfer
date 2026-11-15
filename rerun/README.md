# rerun/

| File | What |
|---|---|
| `landsat_matrix_dump.py` | The R3 training script used for this study. `make external` copies it into the cross-region repo (`scripts/eval/`) pinned at `649e9ad`. |
| `_original_landsat_temporal_matrix.py` | The original script from that repo, unchanged, kept so the differences can be read side by side (`diff -u _original_landsat_temporal_matrix.py landsat_matrix_dump.py`). |

The three changes from the original are listed in the docstring of `landsat_matrix_dump.py`:
full per-seed matrices are saved; every cell is scored on held-out test cities (a fixed,
seed-independent 30% of each region's cities); and collapsed runs (in-region test
IoU < 0.10) are retrained with seed + 1000, at most twice.
