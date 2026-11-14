# Where these receipts come from

Copied unchanged from `results/metrics/` of
github.com/rohanbalixz/Cross-Region-Source-Invariance-in-Earth-Observation
at commit `649e9ad` (main, fetched 2026-11-14).

Only the receipts this experiment reads are copied. Each JSON stores one
experiment's numbers; per-seed full matrices exist only in
`multiseed_matrix_cnn.json` and `multiseed_matrix_unet.json` (8 regions x 5 seeds).
The raw-Landsat multi-seed receipts store summaries and diagonals only, which is
why `rerun/landsat_matrix_dump.py` exists.
