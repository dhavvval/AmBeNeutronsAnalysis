import numpy as np
import pandas as pd
import pytest

from ambe.efficiency_extraction import forward_fold as ff


def _tmpl(K=5):
    # (captures' cluster counts, other clusters, events)
    sig = [((), 0, 170), ((1,), 0, 600), ((1,), 1, 150), ((), 1, 40), ((1,), 2, 30), ((), 3, 5), ((1, 1), 0, 5)]
    return ff.Template([m for m, _, _ in sig], np.array([k for _, k, _ in sig]), np.array([w for *_, w in sig], float), K)


def test_pmf_at_s1_is_the_empirical_multiplicity():
    t = _tmpl()
    M, V = t.pmf(1.0, 0.0)
    n = np.repeat([0, 1, 2, 1, 3, 3, 2], [170, 600, 150, 40, 30, 5, 5])
    emp = np.bincount(np.minimum(n, 5), minlength=6) / len(n)
    assert np.allclose(M, emp)
    assert np.allclose(V, emp * (1 - emp) / len(n))


def test_thinning_mean_and_efficiency():
    t = _tmpl(K=8)
    for s in (1.0, 0.85, 0.5, 1.1):
        M, _ = t.pmf(s, 0.0)
        assert M.sum() == pytest.approx(1.0)
        ek = (t.w * t.k).sum() / t.W
        eko = (t.w * t.sig_ko).sum() / t.W
        assert (np.arange(9) * M).sum() == pytest.approx(eko + s * ek)
    one = t.with_weights(np.where(t.k <= 1, t.w, 0.0))
    assert one.eff(0.85) == pytest.approx(0.85 * one.eff(1.0))


def test_fold_poisson_keeps_norm_and_shifts_mean():
    p = np.array([0.2, 0.7, 0.1] + [0.0] * 17)
    for mu in (0.0, 0.03, -0.02):
        q = ff.fold_poisson(p, mu)
        assert q.sum() == pytest.approx(1.0)
        assert (np.arange(20) * q).sum() == pytest.approx((np.arange(20) * p).sum() + mu, abs=1e-12)


def test_beta_zero_variance_and_stationary():
    nuA, nuF, d = np.array([100.0, 50.0]), np.array([10.0, 2.0]), np.array([120.0, 40.0])
    assert np.allclose(ff.Problem.beta(nuA, nuF, d, np.zeros(2)), 1.0)
    s2 = np.array([0.01, 0.04])
    b = ff.Problem.beta(nuA, nuF, d, s2)
    grad = nuA - d * nuA / (b * nuA + nuF) + (b - 1) / s2
    assert np.allclose(grad, 0.0, atol=1e-9)


def test_asimov_fit_recovers_injected_values():
    K = 4
    t1, t2 = _tmpl(K), _tmpl(K).with_weights(_tmpl(K).w[::-1].copy() * 0 + _tmpl(K).w)
    c = np.array([480.0, 90.0, 15.0, 3.0, 1.0])
    F = np.array([0.2, 0.1, 0.15])
    N = np.array([20000.0, 15000.0, 30000.0])
    truth = [(0.83, 0.03), (0.91, 0.01)]
    gidx = np.array([0, 1, 1])
    d = np.array([(N[r] - F[r] * c.sum()) * t1.pmf(*truth[gidx[r]])[0] + F[r] * c for r in range(3)])
    tm = t1.with_weights(t1.w * 1e6)   # no template uncertainty
    pb = ff.Problem([1, 2, 3], ["a", "b"], gidx, [tm, tm, tm], d, F, c)
    fr = ff.fit(pb)
    assert fr.ok
    assert fr.x[0] == pytest.approx(0.83, abs=2e-4) and fr.x[1] == pytest.approx(0.91, abs=2e-4)
    assert fr.x[2] == pytest.approx(0.03, abs=2e-4) and fr.x[3] == pytest.approx(0.01, abs=2e-4)
    assert np.exp(fr.x[4:]) == pytest.approx(c, rel=1e-3)


def test_factors_md5_is_checked(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame(dict(run=[1], n_ic3000=[10], n_ic3000_bkg_g=[100])).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        ff.factors(dict(factors_csv=str(p), factors_md5="0" * 32))
    f = ff.factors(dict(factors_csv=str(p), factors_md5=ff.md5(p)))
    assert f.F_shared_rel.iloc[0] == pytest.approx(0.1)


def test_foldeff_dispatches_forward(monkeypatch):
    from ambe.efficiency_extraction import recluster_index as ri
    seen = {}
    monkeypatch.setattr(ff, "cli", lambda ctx, argv: seen.setdefault("argv", list(argv)))
    ri.cli_foldeff(None, ["--fold-model", "forward", "--stage", "data"])
    assert seen["argv"] == ["--stage", "data"]


def test_shared_bb_equals_per_run_for_unshared_templates_and_is_stationary():
    K = 4
    ta, tb = _tmpl(K), _tmpl(K).with_weights(_tmpl(K).w * 0.5)
    c = np.array([15.0, 5.0, 2.0, 0.0, 0.0])
    d = np.array([[3000.0, 12000.0, 900.0, 60.0, 9.0], [2500.0, 11000.0, 1000.0, 70.0, 5.0], [800, 3000, 260, 20, 3.0]])
    F = np.array([0.2, 0.1, 0.15])
    x = np.concatenate([[0.9], [0.01], np.log(c + 0.1)])
    sep = ff.Problem([1, 2, 3], ["g"], np.zeros(3, int), [ta, tb, tb.with_weights(tb.w)], d, F, c, bb_share="template")
    run = ff.Problem([1, 2, 3], ["g"], np.zeros(3, int), [ta, tb, tb.with_weights(tb.w)], d, F, c, bb_share="run")
    assert sep.nll(x) == pytest.approx(run.nll(x), rel=1e-10)
    sh = ff.Problem([1, 2, 3], ["g"], np.zeros(3, int), [ta, tb, tb], d, F, c, bb_share="template")
    nuA, nuF, sig2, _ = sh.predict(x)
    beta, _ = sh.beta_shared(nuA, nuF, sig2)
    assert beta[1] == pytest.approx(beta[2])
    nu = beta * nuA + nuF
    g = (nuA * (1 - d / nu))[1:].sum(axis=0) + (beta[1] - 1) / sig2[1]
    assert np.allclose(g, 0.0, atol=1e-6)


def _tmpl_full(K=4):
    sig = [((), 0, 170), ((1,), 0, 600), ((1,), 1, 150), ((), 1, 40), ((1,), 2, 30), ((), 3, 5), ((1, 1), 0, 5), ((), 4, 6)]
    return ff.Template([m for m, _, _ in sig], np.array([k for _, k, _ in sig]), np.array([w for *_, w in sig], float), K)


def test_cov_shift_is_stationary_and_sum_preserving():
    K = 4
    ta, tb = _tmpl_full(K), _tmpl_full(K).with_weights(_tmpl_full(K).w * 0.5)
    c = np.array([15.0, 5.0, 2.0, 0.0, 0.0])
    d = np.array([[3000.0, 12000.0, 900.0, 60.0, 9.0], [2500.0, 11000.0, 1000.0, 70.0, 5.0], [800, 3000, 260, 20, 3.0]])
    F = np.array([0.2, 0.1, 0.15])
    x = np.concatenate([[0.9], [0.01], np.log(c + 0.1)])
    pb = ff.Problem([1, 2, 3], ["g"], np.zeros(3, int), [ta, tb, tb], d, F, c, bb_model="cov")
    nuA, nuF, sig2, NA = pb.predict(x)
    amb, beta, pen = pb.shift_cov(nuA, nuF, NA)
    assert np.allclose(amb.sum(axis=1), NA)                      # the template pmf keeps its sum
    assert np.allclose(amb[1] / NA[1], amb[2] / NA[2])          # one shift per shared template
    # stationarity of the profiled objective in z
    B, first = pb._B, pb._first
    S = np.einsum("nk,tnm,ml->tkl", B, pb._C[first], B) + 1e-14 * np.eye(B.shape[1])[None]
    z = np.linalg.lstsq(B, (amb[first] / NA[first, None] - nuA[first] / NA[first, None]).T, rcond=None)[0].T
    nu = amb + nuF
    for t, rows in enumerate([[0], [1, 2]]):
        g = sum(NA[r] * (1 - d[r] / nu[r]) for r in rows) @ B + np.linalg.solve(S[t], z[t])
        assert np.allclose(g, 0.0, atol=1e-5)
    fr = ff.fit(pb)
    assert fr.ok


def test_expflat_recovers_tau_and_flat_level():
    rng = np.random.default_rng(3)
    edges = np.arange(2.0, 67.0 + 1e-9, 1.0)
    t = 0.5 * (edges[1:] + edges[:-1])
    lam = 20000 * np.exp(-t / 29.0) + 300.0
    h = rng.poisson(lam).astype(float)
    r = ff._expflat(h, np.clip(h, 1, None), t, 1.0)
    assert r["tau_us"] == pytest.approx(29.0, abs=4 * r["tau_err"])
    assert r["C_per_us"] == pytest.approx(300.0, abs=4 * r["C_err"])


def test_neutcapture_recovers_tau_and_flat_level():
    rng = np.random.default_rng(4)
    t = np.arange(2.0, 66.0) + 0.5
    lam = 30000 * (1 - np.exp(-t / 3.0)) * np.exp(-t / 29.0) + 300.0
    h = rng.poisson(lam).astype(float)
    r = ff._neutcapture(h, np.clip(h, 1, None), t)
    assert r["tau_us"] == pytest.approx(29.0, abs=4 * r["tau_err"])
    assert r["C_per_us"] == pytest.approx(300.0, abs=4 * r["C_err"])
