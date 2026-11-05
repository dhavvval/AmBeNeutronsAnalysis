import numpy as np
import pandas as pd

from ambe.plots.datamc import chi2
from ambe.plots.datamc_clustering import CHI_EDGES, NPOS, Sample, convolve_pmf, pos_id, shape_chi2


def test_convolve_pmf_normalised_and_zero_bin():
    a = np.array([0.2, 0.7, 0.08, 0.02, 0.0])
    b = np.array([0.9, 0.08, 0.015, 0.004, 0.001])
    c = convolve_pmf(a, b, 4)
    assert len(c) == 5
    assert np.isclose(c.sum(), 1.0)
    assert np.isclose(c[0], a[0] * b[0])


def test_shape_chi2_matches_datamc_chi2_unweighted():
    rng = np.random.default_rng(1)
    d = rng.poisson(200, 12).astype(float)
    m = rng.poisson(900, 12).astype(float)
    c_ref, ndf_ref = chi2(d, m, d.sum() / m.sum())
    c, ndf, _ = shape_chi2(d, m, m)
    assert np.isclose(c, c_ref) and ndf == ndf_ref


def _frame(n, pid_xyz, **cols):
    x, y, z = pid_xyz
    return pd.DataFrame({"sourceX": x, "sourceY": y, "sourceZ": z, **cols}, index=range(n))


def test_sample_parts_no_bkg_equals_old_chi2_per_position():
    rng = np.random.default_rng(2)
    xyz = (0.0, 100.0, 0.0)
    d_c = _frame(500, xyz, pe=rng.gamma(4, 5, 500))
    m_c = _frame(3000, xyz, pe=rng.gamma(4, 5, 3000), cap=True)
    d_e = _frame(700, xyz, n_accepted_clusters=1)
    m_e = _frame(3500, xyz, n_accepted_clusters=1, n_noise=0)
    for df in (d_c, m_c, d_e, m_e):
        df["pid"] = pos_id(df)
    assert (d_c.pid == 0).all()
    S = Sample("clusterfinder", False, d_c, d_e, m_c, m_e, d_c.iloc[:0], d_e.iloc[:0])
    comp, var = S.parts("pe", CHI_EDGES, 0)
    nd = np.histogram(d_c.pe, CHI_EDGES)[0].astype(float)
    nm = np.histogram(m_c.pe, CHI_EDGES)[0].astype(float)
    c, ndf, _ = shape_chi2(nd, sum(comp.values()), var)
    c_ref, ndf_ref = chi2(nd, nm, nd.sum() / nm.sum())
    assert np.isclose(c, c_ref) and ndf == ndf_ref
    assert len(S.n_d) == NPOS


def test_sample_mult_with_empty_background_is_mc():
    xyz = (0.0, 0.0, 0.0)
    d_e = _frame(4, xyz, n_accepted_clusters=[0, 1, 1, 2])
    m_e = _frame(4, xyz, n_accepted_clusters=[1, 1, 1, 0], n_noise=[0, 0, 1, 0])
    b_e = _frame(2, xyz, n_accepted_clusters=[0, 0])
    empty = pd.DataFrame({"pe": [], "sourceX": [], "sourceY": [], "sourceZ": [], "cap": []})
    for df in (d_e, m_e, empty):
        df["pid"] = pos_id(df)
    S = Sample("peak", True, empty.copy(), d_e, empty.copy(), m_e, empty.copy(), b_e)
    pdat, pm, pmb, pclean, nd = S.mult(4)
    assert nd == 4 and np.allclose(pm, pmb)
    assert np.isclose(pclean[1], 0.5) and np.isclose(pm[1], 0.75)


def test_spe_map_identity_and_monotone():
    from ambe.plots.datamc_clustering import spe_map
    rng = np.random.default_rng(3)
    a = rng.gamma(3, 0.4, 20000)
    f = spe_map(a, a, -0.5, 8.0)
    q = np.linspace(0.2, 3.0, 50)
    assert np.allclose(f(q), q, atol=0.02)
    assert f(np.array([20.0]))[0] == 20.0
    g = spe_map(a, 1.3 * rng.gamma(3, 0.4, 20000), -0.5, 50.0)
    assert np.all(np.diff(g(q)) >= 0)
    assert np.isclose(np.median(g(a)), 1.3 * np.median(a), rtol=0.03)


def test_nfit_recovers_neutcapture_with_fixed_flat():
    from ambe.plots.datamc_clustering import _nfit
    rng = np.random.default_rng(4)
    t = np.linspace(0, 70, 70001)
    pdf = (1 - np.exp(-t / 6.0)) * np.exp(-t / 30.0)
    x = rng.choice(t, 200000, p=pdf / pdf.sum())
    flat = rng.uniform(0, 70, 20000)
    f = _nfit(np.concatenate([x, flat]), 20000 / 70.0)
    assert abs(f["tau"] - 30.0) < 3 * f["tau_err"] + 0.2
    assert abs(f["therm"] - 6.0) < 3 * f["therm_err"] + 0.2


def test_gd_weights_unit_at_nominal_and_closure():
    from ambe.plots.datamc_clustering import GD_NOMINAL, gd_event_weights
    rng = np.random.default_rng(5)
    lg, lh, n = 0.036, 0.004, 400000
    t = rng.exponential(1 / (lg + lh), n)
    is_gd = rng.random(n) < lg / (lg + lh)
    tc = pd.DataFrame(dict(cap_us=t, is_gd=is_gd))
    model = dict(lam_gd=lg, lam_other=lh, n_events=n)
    w, w0, norm = gd_event_weights(tc, model, GD_NOMINAL)
    assert np.allclose(w, 1.0) and np.isclose(norm, 1.0)
    w, _, _ = gd_event_weights(tc, model, 0.8 * GD_NOMINAL)
    lam = 0.8 * lg + lh
    assert np.isclose(np.average(t, weights=w), 1 / lam, rtol=0.01)
    assert np.isclose(w[is_gd].sum() / w.sum(), 0.8 * lg / lam, rtol=0.01)


def test_neutron_comparison_subtracts_background_and_drops_noise():
    from ambe.plots.datamc_clustering import data_minus_bkg, mc_neutron, neutron_chi2, neutron_means
    rng = np.random.default_rng(3)
    xyz = (0.0, 0.0, 0.0)
    sig, bkg = rng.gamma(4, 5, 4000), rng.uniform(0, 60, 400)
    d_c = _frame(4400, xyz, pe=np.concatenate([sig, bkg]))
    d_e = _frame(10000, xyz, n_accepted_clusters=0)
    b_c = _frame(40, xyz, pe=rng.uniform(0, 60, 40))
    b_e = _frame(1000, xyz, n_accepted_clusters=0)
    m_c = _frame(6000, xyz, pe=np.concatenate([rng.gamma(4, 5, 5000), rng.uniform(0, 60, 1000)]),
                 cap=np.r_[np.ones(5000, bool), np.zeros(1000, bool)])
    m_e = _frame(20000, xyz, n_accepted_clusters=0, n_noise=0)
    for df in (d_c, d_e, b_c, b_e, m_c, m_e):
        df["pid"] = pos_id(df)
    S = Sample("peak", False, d_c, d_e, m_c, m_e, b_c, b_e)
    d, vd = data_minus_bkg(S, "pe", CHI_EDGES)
    raw = np.histogram(d_c.pe, CHI_EDGES)[0]
    assert np.allclose(d, raw - np.histogram(b_c.pe, CHI_EDGES)[0] * 10.0)
    assert (vd >= raw).all()
    m, _ = mc_neutron(S, "pe", CHI_EDGES)
    assert np.isclose(m.sum(), 0.5 * ((m_c.pe >= 5) & (m_c.pe < 65) & m_c.cap).sum())
    md, mm = neutron_means(S, "pe")
    assert abs(md - sig.mean()) < 1.5 and abs(mm - 20.0) < 1.0
    c2, ndf = neutron_chi2(S)
    assert ndf == len(CHI_EDGES) - 2 and c2 / ndf < 3


def test_neutron_mult_counts_capture_clusters_and_folds_background():
    from ambe.plots.datamc_clustering import neutron_mult
    xyz = (0.0, 0.0, 0.0)
    m_e = _frame(4, xyz, _source_file="f", eventNumber=np.arange(4), n_accepted_clusters=0, n_noise=0)
    m_c = _frame(5, xyz, _source_file="f", eventNumber=[0, 0, 1, 2, 2], pe=1.0,
                 cap=np.array([True, True, True, False, False]))
    d_e = _frame(8, xyz, n_accepted_clusters=[0, 0, 1, 1, 1, 1, 2, 3])
    b_e = _frame(10, xyz, n_accepted_clusters=[0] * 9 + [1])
    df = [d_e, b_e, m_c, m_e]
    for x in df:
        x["pid"] = pos_id(x)
    S = Sample("peak", False, _frame(1, xyz, pe=1.0).assign(pid=0), d_e, m_c, m_e,
               _frame(1, xyz, pe=1.0).assign(pid=0), b_e)
    pdat, pm, pmb, nd = neutron_mult(S, 3, None)
    assert nd == 8 and np.allclose(pdat, [0.25, 0.5, 0.125, 0.125])
    assert np.allclose(pm, [0.5, 0.25, 0.25, 0.0])
    assert np.allclose(pmb, convolve_pmf(pm, np.array([0.9, 0.1, 0.0, 0.0]), 3))


def test_cf_labels_fast_and_nms_fast_match_per_trigger_production():
    from ambe.clustering.peak_finder import merge_clusters_in_time
    from ambe.mc.optics import assign_clusterfinder_labels
    from ambe.plots.datamc_clustering import cf_labels_fast, nms_fast
    rng = np.random.default_rng(3)
    hs, cs = [], []
    for k in range(40):
        ct = rng.uniform(2000, 67000, rng.integers(0, 6))
        ct = np.concatenate([ct, ct[:1] + 30.0])
        t = np.concatenate([rng.uniform(1500, 67000, 30)] + [c + rng.uniform(-8, 25, 6) for c in ct])
        hs.append(pd.DataFrame({"trigger": k, "t": t}))
        cs.append(pd.DataFrame({"trigger": k, "clusterTime": ct}))
    h, cf = pd.concat(hs, ignore_index=True), pd.concat(cs, ignore_index=True)
    lab = cf_labels_fast(h, cf)
    grp = nms_fast(h.trigger.to_numpy(), lab, h.t.to_numpy(), 50.0)
    for k, idx in h.groupby("trigger").indices.items():
        ref = assign_clusterfinder_labels(h.iloc[idx], cf[cf.trigger == k].reset_index(drop=True))
        assert (lab[idx] == ref).all()
        mref = merge_clusters_in_time(ref, h.t.to_numpy()[idx], 50.0)
        g = grp[idx]
        assert ((g >= 0) == (mref >= 0)).all()
        pairs = set(zip(g[g >= 0], mref[mref >= 0]))
        assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})
