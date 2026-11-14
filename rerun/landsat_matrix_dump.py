"""Raw-Landsat transfer matrix, rerun so the pair-level analysis is possible (regime R3).

Drop-in for the cross-region repo: copy to
  Cross-Region-Source-Invariance-in-Earth-Observation/scripts/eval/landsat_matrix_dump.py
and run from that repo's root:
  python -m scripts.eval.landsat_matrix_dump --dry-run          # split + tile counts only
  python -m scripts.eval.landsat_matrix_dump --seeds 20260525 1 2 3 4

Same data, model (SegUNet), target (GHSL built-up 2015), grid (64) and training
recipe (15 epochs, Adam 1e-3, batch 8) as scripts/eval/landsat_temporal_matrix.py.
Three changes, all decided before any distance is computed:

1. Full per-seed matrices are saved. The existing receipt keeps only summaries and
   the diagonal, which is not enough for pair-level tests.
2. City-disjoint evaluation. The original scored the diagonal on all of a region's
   tiles, 85% of which the model trained on. Here each region's cities are split
   once (seed-independent, by a fixed hash of the city name) into ~70% train and
   ~30% test cities. Models train on train cities only, and EVERY cell (s, t),
   diagonal included, is scored on target t's test cities. The all-tile matrix is
   saved too, for comparison with the published receipt.
3. Collapsed runs. A source model whose in-region test IoU is below 0.10 is
   recorded, then retrained with seed + 1000 (at most twice). Primary analysis uses
   the replacement; the original is kept in the receipt for a with/without check.

Checkpoints are saved under results/landsat_dump/weights_seed{seed}/{region}.pt so
model-aware distances can be computed later without retraining.

UNTESTED: written without access to data/raw/landsat_history. Run --dry-run first.
"""
import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from scripts.acquire.regions import city_by_name
from scripts.common import TILE_PX, enumerate_tiles_from_grid
from scripts.eval.cross_region_eval import TARGET_EPOCH, load_city_rasters
from scripts.eval.multitask_difficulty import SegUNet

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania",
           "weur", "latam", "camcar", "canada", "nordic", "central_europe",
           "china_interior", "south_asia_2", "southern_africa", "mediterranean",
           "brazil_north", "japan_korea_2"]
GRID, CENTER_PX, BU_THRESH = 64, 27, 0.1
TEST_FRAC, COLLAPSE_IOU, MAX_RETRIES = 0.30, 0.10, 2


def l8_to_reflectance(dn):
    return np.clip(dn.astype(np.float32) * 2.75e-5 - 0.2, 0.0, 1.0)


def test_cities(names):
    """Fixed, seed-independent split: the round(30%) of cities with the smallest
    SHA-256 hash of their name (at least 1, never all)."""
    k = min(len(names) - 1, max(1, round(TEST_FRAC * len(names))))
    return sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:k]


def load_region(region, l8dir):
    from pyproj import Transformer
    X, Y, C, LL, IDS = [], [], [], [], []
    for npz_path in sorted(glob.glob(str(l8dir / "*.npz"))):
        city = city_by_name(Path(npz_path).stem)
        if city is None or city.region != region:
            continue
        d = np.load(npz_path, allow_pickle=True)
        patches, tile_ids = d["patches"], list(d["tile_ids"])
        try:
            bu, _, _, transform, crs = load_city_rasters(city, PROC)
        except Exception:
            continue
        gt = bu[TARGET_EPOCH]
        refs = enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                         city_name=city.name, region=region, utm_crs=crs)
        id2ij = {r.tile_id: (r.i, r.j) for r in refs}
        id2ref = {r.tile_id: r for r in refs}
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        c0 = TILE_PX // 2 - CENTER_PX // 2
        for k, tid in enumerate(tile_ids):
            if tid not in id2ij:
                continue
            i, j = id2ij[tid]
            box = gt[i + c0:i + c0 + CENTER_PX, j + c0:j + c0 + CENTER_PX]
            if box.shape != (CENTER_PX, CENTER_PX):
                continue
            tgt = (zoom(box, GRID / CENTER_PX, order=1) > BU_THRESH).astype(np.int64)[:GRID, :GRID]
            if tgt.shape != (GRID, GRID):
                continue
            l8 = l8_to_reflectance(patches[k].reshape(patches.shape[1] * 6, 224, 224))
            l8 = zoom(l8, (1, GRID / 224, GRID / 224), order=1)[:, :GRID, :GRID].astype(np.float32)
            X.append(l8); Y.append(tgt); C.append(city.name); IDS.append(tid)
            b = id2ref[tid].bbox_utm
            lon, lat = inv.transform((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            LL.append((lat, lon))
    if not X:
        return None
    C = np.array(C)
    names = sorted(set(C))
    if len(names) < 2:
        return None
    test_names = test_cities(names)
    test = np.isin(C, test_names)
    return dict(X=torch.from_numpy(np.stack(X)), Y=torch.from_numpy(np.stack(Y)), city=C,
                test=torch.from_numpy(test), test_cities=test_names, n_cities=len(names),
                latlon=np.array(LL), tile_ids=np.array(IDS))


def dump_features(region, d, outdir):
    """Input features for the OT distances (regime R3): mean and SD of each of the 18
    reflectance channels per tile, plus tile-centre lat/lon. Input only, no labels."""
    X = d["X"].numpy()
    F = np.concatenate([X.mean((2, 3)), X.std((2, 3))], axis=1).astype(np.float32)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"{region}.npz", features=F, latlon=d["latlon"],
                        tile_ids=d["tile_ids"], is_test=d["test"].numpy())


def train(data, seed, epochs=15):
    X, Y = data["X"][~data["test"]], data["Y"][~data["test"]]
    if len(X) < 20:
        return None
    torch.manual_seed(seed)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SegUNet(in_ch=X.shape[1], n=2).to(DEV)
    opt, ce = torch.optim.Adam(m.parameters(), 1e-3), nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.arange(len(X), device=DEV)
    for _ in range(epochs):
        m.train(); perm = idx[torch.randperm(len(idx), generator=g, device=DEV)]
        for k in range(0, len(perm), 8):
            b = perm[k:k + 8]; opt.zero_grad(); ce(m(X[b]), Y[b]).backward(); opt.step()
    m.eval(); return m


def iou(model, data, which):
    sel = data["test"] if which == "test" else torch.ones(len(data["X"]), dtype=torch.bool)
    X, Y = data["X"][sel], data["Y"][sel]
    if len(X) == 0:
        return None
    I = U = 0.0
    with torch.no_grad():
        for k in range(0, len(X), 16):
            p = model(X[k:k + 16].to(DEV)).argmax(1).cpu().numpy(); y = Y[k:k + 16].numpy()
            I += np.logical_and(p == 1, y == 1).sum(); U += np.logical_or(p == 1, y == 1).sum()
    return float(I / U) if U > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="landsat_history", choices=["landsat8", "landsat_history"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2, 3, 4])
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    l8dir = REPO / f"data/raw/{a.data}"
    data = {}
    for r in a.regions:
        d = load_region(r, l8dir)
        if d is None:
            print(f"  {r}: no data"); continue
        ntr, nte = int((~d['test']).sum()), int(d['test'].sum())
        print(f"  {r}: {d['n_cities']} cities, test cities {d['test_cities']}, tiles train {ntr} / test {nte}", flush=True)
        if ntr >= 20 and nte >= 5:
            data[r] = d
    ready = list(data)
    split = {r: data[r]["test_cities"] for r in ready}
    if a.dry_run:
        print(f"ready regions: {len(ready)}"); return
    outdir = REPO / "results/landsat_dump"; outdir.mkdir(parents=True, exist_ok=True)
    for r in ready:   # copy results/landsat_dump/features/*.npz to geospot-urban-transfer/data/features/landsat/
        dump_features(r, data[r], outdir / "features")
    receipt = {"input": f"raw Landsat [{a.data}]", "target": "GHSL built-up 2015", "metric": "IoU",
               "grid": GRID, "regions": ready, "seeds": a.seeds, "test_cities": split,
               "collapse_rule": f"in-region test IoU < {COLLAPSE_IOU} -> retrain with seed+1000 (max {MAX_RETRIES})",
               "per_seed": {}}
    for sd in a.seeds:
        t0 = time.time(); models, log = {}, {}
        for r in ready:
            tries, s_used, m = [], sd, None
            for attempt in range(MAX_RETRIES + 1):
                s_used = sd + 1000 * attempt
                m = train(data[r], s_used)
                v = iou(m, data[r], "test") if m is not None else None
                tries.append({"seed": s_used, "in_region_test_iou": v})
                if v is not None and v >= COLLAPSE_IOU:
                    break
            models[r], log[r] = m, tries
            w = outdir / f"weights_seed{sd}"; w.mkdir(exist_ok=True)
            torch.save(m.state_dict(), w / f"{r}.pt")
            print(f"[seed {sd}] {r}: {tries}", flush=True)
        mat_test = {s: {t: iou(models[s], data[t], "test") for t in ready} for s in ready}
        mat_all = {s: {t: iou(models[s], data[t], "all") for t in ready} for s in ready}
        receipt["per_seed"][str(sd)] = {"matrix_test": mat_test, "matrix_all_tiles": mat_all,
                                        "training_log": log, "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(receipt, open(REPO / "results/metrics/landsat_matrix_dump_multiseed.json", "w"), indent=1)
        print(f"[seed {sd}] done in {(time.time() - t0) / 60:.1f} min; receipt updated", flush=True)


if __name__ == "__main__":
    main()
