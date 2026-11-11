from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ambe.efficiency_extraction import forward_fold_datamc as dm  # noqa: E402


def _trig(ic, gate, cos, n):
    return pd.DataFrame(dict(IC_adjusted=ic, gate=gate, is_cosmic=cos, n=n))


def test_mask_selections_and_gain():
    t = _trig([2900.0, 3100.0, 900.0, np.nan], [False, False, True, False], [False, True, False, False], [0, 1, 2, 3])
    assert list(dm._mask(t, "ic3000_noveto", 1.0, 3000)) == [False, True, False, False]
    assert list(dm._mask(t, "ic3000_noveto", 1.05, 3000)) == [True, True, False, False]
    assert list(dm._mask(t, "ic3000_veto", 1.0, 3000)) == [False, False, False, False]
    assert list(dm._mask(t, "gate_veto", 1.0, 3000)) == [False, False, True, False]
    assert dm._mask(t, "all_noveto", 1.0, 3000).all()


class _D:
    def __init__(self, frames):
        self.frames = frames

    def run(self, r):
        return self.frames[r], None


def test_clmult_counts_and_factors():
    src = _trig([3500.0, 3500.0, 1000.0, 1000.0], [False, False, True, True], [False, False, False, True], [0, 2, 1, 9])
    bkg = _trig([2950.0, 3200.0, 900.0], [False, False, True], [False, True, False], [1, 0, 0])
    fac = pd.DataFrame(dict(run=[1], port=[5], x=[0.0], y=[0.0], z=[0.0], F_rsb_g=[0.5], F_rsb_g_err=[0.01],
                            alpha=[0.6], g=[1.02]))
    su = SimpleNamespace(blk={"ic_cosmic": 3000.0}, bkg_run=2, fac=fac, labels={1: "p5z0"})
    cm = dm.clmult(_D({1: src, 2: bkg}), su)
    q = cm[cm.selection == "ic3000_noveto"].set_index("k")
    assert q.n_source[0] == 1 and q.n_source[2] == 1
    assert q.n_6264_g.sum() == 2 and q.n_6264_raw.sum() == 1      # 2950 x 1.02 passes only with the gain
    a = cm[cm.selection == "all_noveto"].set_index("k")
    assert a.n_source[dm.KMAX] == 1 and a.n_source.sum() == 4     # overflow bin
    g = cm[cm.selection == "gate_veto"]
    assert g.n_source.sum() == 1 and g.n_6264_g.sum() == 1
    assert (cm.F == 0.5).all() and (cm.alpha == 0.6).all()


def test_fits_lmfit_rows():
    rng = np.random.default_rng(1)
    t = np.concatenate([rng.exponential(29.0, 60000) + rng.exponential(5.0, 60000), rng.uniform(2, 67, 4000)])
    t = t[(t >= 2) & (t <= 67)] * 1000.0
    cl = pd.DataFrame(dict(clusterTime=t))
    mc = pd.DataFrame(dict(clusterTime=t, cap=np.r_[np.ones(len(t) - 3000, bool), np.zeros(3000, bool)]))
    rows = dm._fits("port", "port5", dict(port=5), cl, mc)
    assert [r["sample"] for r in rows] == ["data", "mc_all", "mc_capture"]
    for r in rows:
        assert r["fit_ok"] and r["fitter"].startswith("lmfit least_squares")
        assert 20 < r["tau"] < 40 and r["tau_err"] > 0


def test_out_requires_residual_panel(tmp_path):
    import matplotlib.pyplot as plt
    from ambe.plotting import main_resid_axes
    ctx = SimpleNamespace(plots_dir=tmp_path)
    O = dm.Out(ctx)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.raises(SystemExit):
        O.save(fig, "nores", "simple", "detail")
    fig, M, R = main_resid_axes()
    p = dm._cmp(M[0, 0], R[0, 0], np.arange(3), [4, 9, 16], [2, 3, 4], [4, 9, 8], [0, 0, 0], "a", "b")
    assert np.allclose(p, [0, 0, 2])
    assert dm._chi2(p) == (4.0, 3)
    O.save(fig, "ok", "simple", "detail")
    O.index()
    idx = pd.read_csv(tmp_path / "deck_chain" / "index.csv")
    assert list(idx.name) == ["ok"]
    assert (tmp_path / "deck_chain" / "ok.png").exists() and (tmp_path / "deck_chain" / "ok.pdf").exists()


def test_overlay_reuse_counts_repeats():
    cf = pd.DataFrame(dict(n=[1, 1, 2, 2, 3, 3, 3], cap=[True, False, True, False, False, False, True],
                           clusterTime=[5e3, 9e3, 6e3, 9e3, 9e3, 2e4, 7e3], clusterPE=[10, 20, 11, 20, 20, 30, 12]))
    M = SimpleNamespace(cf=cf)
    r = dm.overlay_reuse(M).set_index("cls")
    assert r.noncapture_clusters["all"] == 4 and r.noncapture_distinct["all"] == 2
    assert r.max_reuse["all"] == 3 and r.capture_share["n3p"] == pytest.approx(1 / 3)
