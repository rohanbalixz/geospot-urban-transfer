"""Synthetic checks: each method must recover a known answer before it touches real data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotransfer import decomposition as dec  # noqa: E402
from geotransfer import inference as inf  # noqa: E402
from geotransfer import ot_distance as otd  # noqa: E402
from geotransfer import selection as sel  # noqa: E402


def simulate(S=8, K=5, sd=None, rng=0):
    sd = sd or dict(target=0.06, source=0.01, pair=0.02, source_x_seed=0.01, residual=0.008)
    r = np.random.default_rng(rng)
    a = r.normal(0, sd["target"], S); b = r.normal(0, sd["source"], S)
    g = r.normal(0, sd["pair"], (S, S))
    c = r.normal(0, sd["source_x_seed"], (K, S))
    e = r.normal(0, sd["residual"], (K, S, S))
    Y = 0.35 + a[None, None, :] + b[None, :, None] + g[None] + c[:, :, None] + e
    return Y, g


def test_anova_recovers_components():
    sd = dict(target=0.06, source=0.01, pair=0.02, source_x_seed=0.01, residual=0.008)
    est = {k: [] for k in sd}
    for rep in range(200):
        Y, _ = simulate(S=12, K=6, sd=sd, rng=rep)
        _, vc, _, _ = dec.anova3(Y)
        for k in sd:
            est[k].append(vc[k])
    for k, v in sd.items():
        m = np.sqrt(np.mean(est[k]))
        assert abs(m - v) / v < 0.25, (k, m, v)


def test_reliability_tracks_truth():
    hi = np.mean([dec.split_half_reliability(simulate(sd=dict(target=.06, source=.01, pair=.03, source_x_seed=.002, residual=.004), rng=i)[0], dec.pair_specific) for i in range(30)])
    lo = np.mean([dec.split_half_reliability(simulate(sd=dict(target=.06, source=.01, pair=.001, source_x_seed=.002, residual=.02), rng=i)[0], dec.pair_specific) for i in range(30)])
    assert hi > 0.8 and (lo != lo or lo < 0.3), (hi, lo)


def test_qap_null_is_calibrated_and_detects_signal():
    rng = np.random.default_rng(1)
    S = 8
    pvals = []
    for rep in range(60):
        P = rng.normal(size=(S, 2)); D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        O = rng.normal(size=(S, S)); np.fill_diagonal(O, np.nan)
        pvals.append(inf.qap_spearman(D, O, n_perm=999, rng=rep)[1])
    assert np.mean(np.array(pvals) < 0.05) <= 0.12   # nominal 5%, 60 reps
    P = rng.normal(size=(S, 2)); D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    O = -D + 0.05 * rng.normal(size=(S, S)); np.fill_diagonal(O, np.nan)
    rho, p, n, exact = inf.qap_spearman(D, O)
    assert exact and n == 40320 and rho < -0.8 and p < 0.01


def test_selection_gain_zero_for_random_distance():
    gains = []
    for rep in range(40):
        Y, _ = simulate(rng=rep)
        D = np.random.default_rng(rep + 1000).random((8, 8)); D = (D + D.T) / 2
        gains.append(sel.selection_summary(Y, D)["distance_gain"])
    assert abs(np.mean(gains)) < 0.004


def test_selection_gain_positive_for_oracle_like_distance():
    Y, g = simulate(sd=dict(target=.06, source=.0, pair=.03, source_x_seed=.002, residual=.004), rng=3)
    D = -(g + g.T) / 2   # closer = better transfer, symmetric part of the true pair effect
    out = sel.selection_summary(Y, D)
    assert out["distance_gain"] > 0 and out["captured_fraction"] > 0.3


def test_sinkhorn_divergence_properties():
    rng = np.random.default_rng(0)
    def region(name, shift):
        n = 60
        return otd.RegionSample(name=name, features=rng.normal(shift, 1, (n, 4)) + 3,
                                latlon=np.column_stack([rng.uniform(-5, 5, n) + 10 * shift, rng.uniform(-5, 5, n)]),
                                embeddings={"emb": rng.normal(shift, 1, (n, 8)) + 3})
    A, A2, B, C = region("A", 0.0), region("A2", 0.0), region("B", 1.0), region("C", 2.0)
    D, asym = otd.distance_matrix([A, A2, B, C], feat_comp="feature", loc_comp="geodesic", lam=0.5, eps=0.01)
    assert asym < 1e-3
    assert D[0, 1] < D[0, 2] < D[0, 3], D
    self_div = otd.region_distance(A, A, loc_comp="geodesic", scales={"feature": 1, "geodesic": 1000})
    assert abs(self_div) < 1e-6
    # lam is bound to named components: lam=1 ignores location entirely
    d_feat_only = otd.region_distance(A, B, feat_comp="feature", loc_comp="geodesic", lam=1.0,
                                      scales={"feature": 1.0, "geodesic": 1.0})
    d_feat_only_far = otd.region_distance(A, B, feat_comp="feature", loc_comp="geodesic", lam=1.0,
                                          scales={"feature": 1.0, "geodesic": 1e-6})
    assert abs(d_feat_only - d_feat_only_far) < 1e-9


def test_symmetric_ceiling():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(8, 8)); sym = (G + G.T) / 2
    np.fill_diagonal(sym, np.nan)
    share, r = dec.symmetric_ceiling(sym)
    assert share > 0.99 and r > 0.99
    anti = (G - G.T) / 2; np.fill_diagonal(anti, np.nan)
    share, r = dec.symmetric_ceiling(anti)
    assert share < 0.01
