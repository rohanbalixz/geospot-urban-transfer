"""Location embeddings at tile centres, matching GeoSpOT's released choices:

  geoclip      geoclip.LocationEncoder, input (lat, lon)
  satclip_l40  microsoft/SatCLIP-ResNet50-L40 (Hugging Face), input (lon, lat)
  satclip_l10  microsoft/SatCLIP-ResNet50-L10

GeoSpOT's location_embeddings/utils.py uses exactly these checkpoints. The SatCLIP
location encoder is rebuilt from each checkpoint's own hyper-parameters with the
vendored modules in src/vendor/satclip_le (microsoft/satclip, MIT). Checkpoints are
cached in data/models/satclip/.

Usage (from the project root):
  python -m geotransfer.location_embeddings --features data/features/ghsl
writes data/features/ghsl/{region}__{encoder}.npy, row-aligned with {region}.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HF = {"satclip_l40": ("microsoft/SatCLIP-ResNet50-L40", "satclip-resnet50-l40.ckpt"),
      "satclip_l10": ("microsoft/SatCLIP-ResNet50-L10", "satclip-resnet50-l10.ckpt")}
_cache = {}


def _satclip(name):
    if name in _cache:
        return _cache[name]
    from huggingface_hub import hf_hub_download
    from vendor.satclip_le.location_encoder import LocationEncoder, get_neural_network, get_positional_encoding
    repo, fn = HF[name]
    path = hf_hub_download(repo, fn, local_dir=ROOT / "data/models/satclip")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    pe = get_positional_encoding(hp["le_type"], hp["legendre_polys"], hp["harmonics_calculation"],
                                 hp["min_radius"], hp["max_radius"], hp["frequency_num"])
    nn_ = get_neural_network(hp["pe_type"], pe.embedding_dim, hp["embed_dim"], hp["capacity"], hp["num_hidden_layers"])
    sd = {k[k.index("nnet"):]: v for k, v in ckpt["state_dict"].items() if "nnet" in k}
    m = LocationEncoder(pe, nn_).double(); m.load_state_dict(sd); m.eval()
    _cache[name] = m
    return m


def encode(latlon, name):
    latlon = np.asarray(latlon, dtype=np.float64)
    if name == "geoclip":
        if "geoclip" not in _cache:
            from geoclip import LocationEncoder
            _cache["geoclip"] = LocationEncoder().eval()
        enc = _cache["geoclip"]
        with torch.no_grad():
            return np.concatenate([enc(torch.tensor(latlon[i:i + 4096], dtype=torch.float32)).numpy()
                                   for i in range(0, len(latlon), 4096)]).astype(np.float32)
    if name in HF:
        m = _satclip(name)
        lonlat = torch.tensor(latlon[:, ::-1].copy(), dtype=torch.float64)
        with torch.no_grad():
            return m(lonlat).float().numpy()
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--encoders", nargs="+", default=["geoclip", "satclip_l40", "satclip_l10"])
    a = ap.parse_args()
    for f in sorted(a.features.glob("*.npz")):
        ll = np.load(f)["latlon"]
        for e in a.encoders:
            out = f.with_name(f"{f.stem}__{e}.npy")
            if not out.exists():
                np.save(out, encode(ll, e))
                print(f"{f.stem}: {e} {len(ll)} tiles -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
