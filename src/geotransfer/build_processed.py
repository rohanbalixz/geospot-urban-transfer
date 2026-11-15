"""Rebuild the cross-region repo's data/processed/ from the global GHSL rasters on disk.

The original per-city rasters are gone. This writes the same files the repo's
scripts read (scripts/preprocess/build_city_tiffs.py layout):

  {out}/{region}/{city}/builtup_{1975..2015}.tif   built-up fraction, UTM, 250 m
  {out}/{region}/{city}/builtvol_{1975..2010}.tif  log1p(built volume)
  {out}/{region}/{city}/pop_{1975..2010}.tif       log1p(population)
  {out}/{region}/{city}/eval_metrics.json          tile records (tile_id, i, j,
                                                   bbox_utm, utm_crs) for the
                                                   repo's tile rule; read by
                                                   scripts/acquire/landsat8.py

Conditioning is shared with ghsl_features.city_stack. Only input-side files are
written; no model is run.

Usage (from the project root):
  python -m geotransfer.build_processed --out external/cross-region/data/processed
  python -m geotransfer.build_processed --check-against <folder with original eval_metrics.json files>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from geotransfer.ghsl_features import TARGET_EPOCH, TILE_PX, TRAIN_EPOCHS, cities, city_stack, tiles

ROOT = Path(__file__).resolve().parents[2]


def write_tif(path, arr, T, crs):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=T, nodata=0.0, compress="deflate") as d:
        d.write(arr.astype(np.float32), 1)


def build_city(c, out):
    d = out / c["region"] / c["name"]
    if (d / "eval_metrics.json").exists():
        return len(json.load(open(d / "eval_metrics.json")))
    d.mkdir(parents=True, exist_ok=True)
    bu, vol, pop, T, utm = city_stack(c)
    for e in TRAIN_EPOCHS + [TARGET_EPOCH]:
        write_tif(d / f"builtup_{e}.tif", bu[e], T, utm)
    for e in TRAIN_EPOCHS:
        write_tif(d / f"builtvol_{e}.tif", vol[e], T, utm)
        write_tif(d / f"pop_{e}.tif", pop[e], T, utm)
    recs = []
    for i, j in tiles(bu[TARGET_EPOCH]):
        x0, y1 = T * (j, i); x1, y0 = T * (j + TILE_PX, i + TILE_PX)
        recs.append(dict(tile_id=f"{c['name']}_{i:05d}_{j:05d}", city=c["name"], region=c["region"], i=i, j=j,
                         bbox_utm=[float(x0), float(y0), float(x1), float(y1)], utm_crs=utm))
    json.dump(recs, open(d / "eval_metrics.json", "w"))
    return len(recs)


def check_against(orig_root, out):
    """Compare rebuilt tile IDs with the original eval_metrics.json files that survive."""
    rows = []
    for f in sorted(Path(orig_root).expanduser().glob("*/*/eval_metrics.json")):
        region, city = f.parent.parent.name, f.parent.name
        mine = out / region / city / "eval_metrics.json"
        if region == "conus" or not mine.exists():
            continue
        a = {r["tile_id"] for r in json.load(open(f))}
        b = {r["tile_id"] for r in json.load(open(mine))}
        rows.append((region, city, len(a), len(b), len(a & b)))
    if not rows:
        print("no overlapping cities to check"); return
    tot_a = sum(r[2] for r in rows); tot_b = sum(r[3] for r in rows); both = sum(r[4] for r in rows)
    for r in rows:
        print(f"  {r[0]:12s} {r[1]:20s} original {r[2]:3d}  rebuilt {r[3]:3d}  shared {r[4]:3d}")
    print(f"{len(rows)} cities: original {tot_a} tiles, rebuilt {tot_b}, shared {both} "
          f"({100 * both / max(tot_a, 1):.1f}% of original, {100 * both / max(tot_b, 1):.1f}% of rebuilt)")
    return dict(cities=len(rows), original=tot_a, rebuilt=tot_b, shared=both)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "external/cross-region/data/processed")
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--check-against", type=Path, default=None)
    a = ap.parse_args()
    if a.check_against:
        res = check_against(a.check_against, a.out)
        if res:
            json.dump(res, open(ROOT / "data/processed_rebuild_check.json", "w"), indent=1)
        return
    cs = [c for c in cities() if a.regions is None or c["region"] in a.regions]
    for k, c in enumerate(cs):
        n = build_city(c, a.out)
        print(f"[{k + 1}/{len(cs)}] {c['region']}/{c['name']}: {n} tiles", flush=True)


if __name__ == "__main__":
    main()
