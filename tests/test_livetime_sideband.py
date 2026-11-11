import numpy as np

from ambe.plots.livetime_sideband import capture_fraction, livetime, mixture, sql_seconds, T_EDGES_US


def _readouts(tau, lam, n, seed=1):
    rng = np.random.default_rng(seed)
    dt = tau + rng.exponential(1 / lam, n)
    return (1e9 * (1000 + np.cumsum(dt))).astype(np.int64), dt


def test_livetime_recovers_tau_and_rate():
    ts, dt = _readouts(2.0, 9.0, 200_000)
    r = livetime(ts, gap_s=30.0)
    assert abs(r["tau_s"] - 2.0) < 1e-3
    assert abs(r["lambda_hz"] / 9.0 - 1) < 0.01
    assert abs(r["live_s"] / (dt[1:] - 2.0).sum() - 1) < 0.01
    assert abs(r["r_rec_hz"] - 1 / (2.0 + 1 / 9.0)) < 0.005


def test_livetime_unsaturated_and_gaps():
    ts, _ = _readouts(2.0, 0.6, 50_000, seed=2)
    ts = np.concatenate([ts, ts[-1:] + np.int64(500e9)])
    r = livetime(ts, gap_s=30.0)
    assert r["n_gaps"] >= 1
    assert abs(r["lambda_hz"] / 0.6 - 1) < 0.02


def test_mixture_reduces_and_normalises():
    p0 = np.array([0.2, 0.6, 0.15, 0.05])
    fs = np.array([0.9, 0.08, 0.02, 0.0])
    assert np.allclose(mixture(p0, fs, 0.0), p0)
    assert np.isclose(mixture(p0, fs, 0.3).sum(), 1.0)


def test_capture_fraction_pure_shapes():
    e = 29.4 * (np.exp(-T_EDGES_US[:-1] / 29.4) - np.exp(-T_EDGES_US[1:] / 29.4))
    fc, _, _, _ = capture_fraction(np.round(1e5 * e / e.sum()), T_EDGES_US)
    assert fc > 0.99
    fc, _, _, _ = capture_fraction(np.full(len(T_EDGES_US) - 1, 1000.0), T_EDGES_US)
    assert fc < 0.01


def test_sql_seconds():
    assert sql_seconds({"start": "2026-07-17 00:24:12.899537", "stop": "2026-07-17 20:21:34.014407"}) == 71841.11487
