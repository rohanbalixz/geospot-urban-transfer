# geospot-urban-transfer

What can a distance between regions predict about cross-region transfer, and how much of
that does GeoSpOT capture? This is a pre-registered test on dense urban-change transfer
matrices (8 and 20 world regions, 5 seeds each).

Status: **pre-registration drafted; no region distance computed yet.**
See `PREREGISTRATION.md` for the plan.

## Layout

```
PREREGISTRATION.md        frozen plan (commit this first)
data/receipts/            transfer-matrix receipts copied from the cross-region repo (@649e9ad)
data/cities.json          the 220 cities (20 regions x 11) and their bounding boxes
pilot/                    outcome-side pilots (no distances) and their outputs
src/geotransfer/          decomposition, inference (QAP, bootstrap), selection,
                          OT distances, coverage, GHSL features, location embeddings
src/run_analysis.py       the pre-registered analysis, end to end
tests/                    synthetic checks for every method
rerun/                    raw-Landsat matrix rerun (drop-in for the cross-region repo)
```

## Run

```bash
~/miniforge3/envs/qgis/bin/python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest -q tests/                      # synthetic checks
./.venv/bin/python pilot/pilot_ceilings_power.py            # pilot (outcome side only)

# after PREREGISTRATION.md is committed:
cd src && ../.venv/bin/python -m geotransfer.ghsl_features --regions south_asia ssa east_asia andes mena sea eeca oceania
../.venv/bin/python -m geotransfer.location_embeddings --features ../data/features/ghsl
cd .. && ./.venv/bin/python src/run_analysis.py --regimes R1 R2
```

GHSL rasters are read from `~/Desktop/NeuralTimeCapsule/data` (GHSL R2023A, global).
