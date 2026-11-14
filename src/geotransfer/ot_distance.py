"""GeoSpOT-style OT distances between regions.

Each region is an empirical measure over n tiles, each tile carrying
  * a feature vector (from the model's own inputs, e.g. per-channel statistics), and
  * a location (lat/lon) or a location embedding (GeoCLIP / SatCLIP).

Ground cost between tile i (source) and tile j (target):
    C = lam * Cf**p + (1 - lam) * Cl**p
with Cf the feature cost and Cl the location cost, each divided by ONE global scale
shared by all region pairs (so distances are comparable across pairs). The released
GeoSpOT code divides by each pair's own max instead; that variant is kept as a
sensitivity option ("pair_max").

The primary distance is the debiased Sinkhorn divergence
    S_eps(a, b) = OT_eps(a, b) - 0.5 OT_eps(a, a) - 0.5 OT_eps(b, b),
which is zero for identical measures. The plain OT_eps(a, b) that the released code
returns (ot.sinkhorn2 between source and target only) is kept for comparison.

Two fixes relative to the released code are deliberate and should be stated:
  * lam is attached to named cost components, never to whichever key sorts first;
  * a zero distance is returned, not trapped in an interactive shell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import ot

EARTH_KM = 6371.0088


@dataclass
class RegionSample:
    name: str
    features: Optional[np.ndarray] = None            # (n, d_f)
    latlon: Optional[np.ndarray] = None              # (n, 2) degrees
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n, d_e)


# ---------------------------------------------------------------- pairwise costs
def cosine_cost(X, Z):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)
    return np.clip(1.0 - Xn @ Zn.T, 0.0, 2.0)


def euclid_cost(X, Z):
    return np.sqrt(np.maximum(ot.dist(X, Z, metric="sqeuclidean"), 0.0))


def haversine_cost(A, B):
    la1, lo1 = np.radians(A[:, 0])[:, None], np.radians(A[:, 1])[:, None]
    la2, lo2 = np.radians(B[:, 0])[None, :], np.radians(B[:, 1])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def component_cost(src: RegionSample, tgt: RegionSample, component: str, feature_metric="cosine"):
    if component == "feature":
        f = cosine_cost if feature_metric == "cosine" else euclid_cost
        return f(src.features, tgt.features)
    if component == "geodesic":
        return haversine_cost(src.latlon, tgt.latlon)
    if component in src.embeddings:
        return cosine_cost(src.embeddings[component], tgt.embeddings[component])
    raise KeyError(component)


# ---------------------------------------------------------------- distances
def global_scales(samples, components, feature_metric="cosine", q=1.0):
    """One normalising constant per cost component over ALL ordered region pairs
    (q=1.0 -> global max; q<1 -> that quantile, more robust)."""
    scales = {}
    for c in components:
        vals = []
        for a in samples:
            for b in samples:
                vals.append(np.quantile(component_cost(a, b, c, feature_metric), q))
        scales[c] = float(max(vals)) or 1.0
    return scales


def combined_cost(src, tgt, feat_comp, loc_comp, lam, p, scales, normalization, feature_metric):
    parts = []
    if lam > 0:
        Cf = component_cost(src, tgt, feat_comp, feature_metric)
        Cf = Cf / (Cf.max() if normalization == "pair_max" else scales[feat_comp])
        parts.append(lam * Cf ** p)
    if lam < 1:
        Cl = component_cost(src, tgt, loc_comp, feature_metric)
        Cl = Cl / (Cl.max() if normalization == "pair_max" else scales[loc_comp])
        parts.append((1 - lam) * Cl ** p)
    return sum(parts)


def sinkhorn_value(C, eps):
    n, m = C.shape
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    return float(ot.sinkhorn2(a, b, C, reg=eps, method="sinkhorn_log", numItermax=5000, stopThr=1e-9))


def region_distance(src, tgt, *, feat_comp="feature", loc_comp="geoclip", lam=0.5, p=1,
                    eps=0.01, debiased=True, normalization="global", scales=None,
                    feature_metric="cosine"):
    kw = dict(feat_comp=feat_comp, loc_comp=loc_comp, lam=lam, p=p, scales=scales,
              normalization=normalization, feature_metric=feature_metric)
    st = sinkhorn_value(combined_cost(src, tgt, **kw), eps)
    if not debiased:
        return st
    ss = sinkhorn_value(combined_cost(src, src, **kw), eps)
    tt = sinkhorn_value(combined_cost(tgt, tgt, **kw), eps)
    return max(0.0, st - 0.5 * ss - 0.5 * tt)


def distance_matrix(samples, **kw):
    """S x S matrix of region distances (diagonal 0).

    Every cost used here is symmetric (C(t,s) = C(s,t)^T, uniform weights), so
    OT(s,t) = OT(t,s) and each unordered pair is solved once. Self-terms OT(s,s) are
    solved once per region. `asym` is a solver check: |OT(s,t) - OT(t,s)| for the
    first pair, solved in both directions."""
    comps = [c for c, on in [(kw.get("feat_comp", "feature"), kw.get("lam", 0.5) > 0),
                             (kw.get("loc_comp", "geoclip"), kw.get("lam", 0.5) < 1)] if on]
    if kw.get("normalization", "global") == "global" and kw.get("scales") is None:
        kw["scales"] = global_scales(samples, comps, kw.get("feature_metric", "cosine"))
    ck = dict(feat_comp=kw.get("feat_comp", "feature"), loc_comp=kw.get("loc_comp", "geoclip"),
              lam=kw.get("lam", 0.5), p=kw.get("p", 1), scales=kw.get("scales"),
              normalization=kw.get("normalization", "global"), feature_metric=kw.get("feature_metric", "cosine"))
    eps, debiased = kw.get("eps", 0.01), kw.get("debiased", True)
    S = len(samples)
    selfs = [sinkhorn_value(combined_cost(s, s, **ck), eps) for s in samples] if debiased else None
    D = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            st = sinkhorn_value(combined_cost(samples[i], samples[j], **ck), eps)
            D[i, j] = D[j, i] = max(0.0, st - 0.5 * selfs[i] - 0.5 * selfs[j]) if debiased else st
    rev = sinkhorn_value(combined_cost(samples[1], samples[0], **ck), eps)
    fwd = sinkhorn_value(combined_cost(samples[0], samples[1], **ck), eps)
    return D, float(abs(rev - fwd))
