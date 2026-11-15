"""Per-tile features of the HARMONISED input (GHSL history), rebuilt from the global
GHSL R2023A rasters already on disk, for the OT distances.

Mirrors the cross-region repo's conditioning so the distance sees what the model sees:
  * city bbox -> city UTM zone, 250 m grid, bilinear reprojection
    (scripts/preprocess/build_city_tiffs.py)
  * built-up fraction = BUILT_S m^2 / source cell area, clipped to [0, 1]
  * volume and population = log1p(value), then divided by the REGION's max
    (scripts/eval/cross_region_eval.py::load_city_rasters)
  * tiles = 128 x 128 px blocks at stride 64 whose 2015 built-up mean > 0.01
    (scripts/common.py::enumerate_tiles_from_grid). 2015 is the label epoch; it is
    used only to decide which tiles exist, exactly as in the transfer matrix, and
    never enters a feature.
  * features per tile: mean and SD of each of the 24 input channels
    (8 epochs 1975-2010 x built-up, volume, population) -> 48 numbers.

Not identical to the repo's per-region GHSL downloads (same product, different
file packaging: four BUILT_S epochs here are the EPSG:4326 3-arcsecond release),
so tile features can differ slightly from the model's exact inputs. State this.

Usage:
  python -m geotransfer.ghsl_features --regions south_asia ssa ... --out data/features/ghsl
  python -m geotransfer.ghsl_features --city lagos --check      # one-city sanity check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

ROOT = Path(__file__).resolve().parents[2]
# Folder holding the global GHSL R2023A GeoTIFFs (BUILT_S, BUILT_V, POP, 1975-2015).
GHSL = Path(os.environ.get("GHSL_DIR", "data/raw/ghsl")).expanduser()
TRAIN_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015
TILE_PX, TILE_STRIDE, RES = 128, 64, 250
TILE_BUILTUP_THRESHOLD = 0.01


def cities():
    return json.load(open(ROOT / "data/cities.json"))["cities"]


def utm_crs_for_bbox(b):
    lon, lat = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    zone = min(60, max(1, int((lon + 180) / 6) + 1))
    return f"EPSG:326{zone:02d}" if lat >= 0 else f"EPSG:327{zone:02d}"


def bbox_in_utm(b, utm):
    t = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    x0, y0 = t.transform(b[0], b[1]); x1, y1 = t.transform(b[2], b[3])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def raster_path(layer, epoch):
    for crs_tag in ("54009_100", "4326_3ss"):
        p = GHSL / f"GHS_{layer}_E{epoch}_GLOBE_R2023A_{crs_tag}_V1_0.tif"
        if p.exists():
            return p, crs_tag
    raise FileNotFoundError(f"{layer} {epoch}")


def read_to_utm(layer, epoch, bbox, utm, dst_transform, shape):
    path, tag = raster_path(layer, epoch)
    with rasterio.open(path) as src:
        sb = transform_bounds("EPSG:4326", src.crs, *[bbox[0] - 0.05, bbox[1] - 0.05, bbox[2] + 0.05, bbox[3] + 0.05])
        win = window_from_bounds(*sb, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = 0.0
        arr[(arr < 0) | (arr > 1e7)] = 0.0
        out = np.zeros(shape, dtype=np.float64)
        reproject(arr, out, src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=CRS.from_string(utm),
                  resampling=Resampling.bilinear, src_nodata=0.0, dst_nodata=0.0)
    out = np.nan_to_num(out)
    if layer == "BUILT_S":
        if tag == "54009_100":
            cell = 100.0 * 100.0
        else:   # 3 arc-second cell area at the city's latitude
            lat = np.radians((bbox[1] + bbox[3]) / 2)
            cell = (3 / 3600 * np.pi / 180 * 6371008.8) ** 2 * np.cos(lat)
        return np.clip(out / cell, 0.0, 1.0)
    return np.log1p(out)


def city_stack(city):
    b = city["bbox"]; utm = utm_crs_for_bbox(b)
    x0, y0, x1, y1 = bbox_in_utm(b, utm)
    w, h = int(np.ceil((x1 - x0) / RES)), int(np.ceil((y1 - y0) / RES))
    T = from_bounds(x0, y0, x1, y1, w, h)
    bu = {e: read_to_utm("BUILT_S", e, b, utm, T, (h, w)) for e in TRAIN_EPOCHS + [TARGET_EPOCH]}
    vol = {e: read_to_utm("BUILT_V", e, b, utm, T, (h, w)) for e in TRAIN_EPOCHS}
    pop = {e: read_to_utm("POP", e, b, utm, T, (h, w)) for e in TRAIN_EPOCHS}
    return bu, vol, pop, T, utm


def tiles(bu2015):
    h, w = bu2015.shape
    for i in range(0, h - TILE_PX + 1, TILE_STRIDE):
        for j in range(0, w - TILE_PX + 1, TILE_STRIDE):
            if bu2015[i:i + TILE_PX, j:j + TILE_PX].mean() > TILE_BUILTUP_THRESHOLD:
                yield i, j


def region_features(region):
    cs = [c for c in cities() if c["region"] == region]
    stacks = {c["name"]: city_stack(c) for c in cs}
    vmax = max(max(v.max() for v in s[1].values()) for s in stacks.values()) or 1.0
    pmax = max(max(p.max() for p in s[2].values()) for s in stacks.values()) or 1.0
    F, LL, ids = [], [], []
    for name, (bu, vol, pop, T, utm) in stacks.items():
        inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
        for i, j in tiles(bu[TARGET_EPOCH]):
            sl = (slice(i, i + TILE_PX), slice(j, j + TILE_PX))
            chans = []
            for e in TRAIN_EPOCHS:
                chans += [bu[e][sl], np.clip(vol[e][sl] / vmax, 0, 1), np.clip(pop[e][sl] / pmax, 0, 1)]
            chans = np.stack(chans)
            F.append(np.concatenate([chans.mean((1, 2)), chans.std((1, 2))]))
            x, y = T * (j + TILE_PX / 2, i + TILE_PX / 2)
            lon, lat = inv.transform(x, y)
            LL.append((lat, lon)); ids.append(f"{name}_{i:05d}_{j:05d}")
    return np.array(F, dtype=np.float32), np.array(LL), np.array(ids)


FEATURE_NAMES = ([f"mean_{c}_{e}" for e in TRAIN_EPOCHS for c in ("bu", "vol", "pop")]
                 + [f"sd_{c}_{e}" for e in TRAIN_EPOCHS for c in ("bu", "vol", "pop")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=[])
    ap.add_argument("--city", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data/features/ghsl")
    a = ap.parse_args()
    if a.city:
        c = next(c for c in cities() if c["name"] == a.city)
        bu, vol, pop, T, utm = city_stack(c)
        n = sum(1 for _ in tiles(bu[TARGET_EPOCH]))
        print(f"{c['name']} ({c['region']}) {utm} grid {bu[2015].shape}  tiles kept: {n}")
        for e in TRAIN_EPOCHS + [TARGET_EPOCH]:
            print(f"  built-up {e}: mean frac {bu[e].mean():.4f}  max {bu[e].max():.3f}"
                  + (f" | log1p vol max {vol[e].max():.2f}  log1p pop max {pop[e].max():.2f}" if e in vol else ""))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    for r in a.regions:
        F, LL, ids = region_features(r)
        np.savez_compressed(a.out / f"{r}.npz", features=F, latlon=LL, tile_ids=ids,
                            feature_names=np.array(FEATURE_NAMES))
        print(f"{r}: {len(F)} tiles, {F.shape[1]} features -> {a.out / (r + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
