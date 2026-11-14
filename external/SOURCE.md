# external/

`make external` clones the training code this study builds on into `external/cross-region/`
(not tracked here):

- github.com/rohanbalixz/Cross-Region-Source-Invariance-in-Earth-Observation, pinned to
  commit `649e9ad` (Apache-2.0)
- plus `rerun/landsat_matrix_dump.py`, copied to `scripts/eval/` (regime R3)

Its `data/` is rebuilt locally (`make processed`, `make landsat`). See `PREREGISTRATION.md`.
