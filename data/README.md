# data/

| Path | What | Source and licence |
|---|---|---|
| `receipts/` | Transfer-matrix receipts (R1, R2, and earlier runs used in the pilots), plus `landsat_matrix_dump_multiseed.json` (R3, this study) | Cross-Region Source-Invariance repo @ `649e9ad`; R3 from `rerun/landsat_matrix_dump.py`. MIT / Apache-2.0 |
| `cities.json`, `cities.txt` | The 220 cities (20 regions x 11) and bounding boxes | Parsed from that repo's `scripts/acquire/regions.py` |
| `features/ghsl/{region}.npz` | Per-tile input features for R1/R2: mean and SD of 24 GHSL channels (built-up, volume, population x 8 epochs) and tile-centre lat/lon | Derived from GHSL R2023A (European Commission, JRC), CC BY 4.0 |
| `features/landsat/{region}.npz` | Per-tile input features for R3: mean and SD of 24 Landsat channels (6 bands x 4 epochs), lat/lon, test-city flag | Derived from USGS Landsat Collection 2 Level-2 via Microsoft Planetary Computer; public domain |
| `features/*/{region}__{encoder}.npy` | GeoCLIP, SatCLIP-ResNet50 L40 and L10 embeddings at tile centres, row-aligned with the `.npz` | Computed with `geoclip` (MIT) and `microsoft/SatCLIP-ResNet50-*` (MIT) |
| `processed_rebuild_check.json` | Rebuilt tile grid vs the original for 44 cities (786/786 tile IDs match) | This study |

Raw rasters, Landsat patches and model weights are not redistributed. `make processed`,
`make landsat` and `make r3-matrix` rebuild them.
