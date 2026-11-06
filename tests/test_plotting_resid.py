import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from ambe.plotting import draw_diff, draw_pull, main_resid_axes, two_titles  # noqa: E402


def test_main_resid_axes_shape_and_sharex():
    fig, main, resid = main_resid_axes(2, 3)
    assert main.shape == resid.shape == (2, 3)
    assert resid[1, 2].get_shared_x_axes().joined(main[1, 2], resid[1, 2])
    two_titles(fig, "simple", "detail")


def test_draw_pull_values_and_zero_error_bins():
    fig, main, resid = main_resid_axes()
    pull = draw_pull(resid[0, 0], [0, 1, 2], [10, 4, 0], [3, 0, 0], [1, 4, 0], [4, 0, 0])
    assert np.isclose(pull[0], 9 / 5)
    assert np.isnan(pull[1]) and np.isnan(pull[2])
    lo, hi = resid[0, 0].get_ylim()
    assert lo == -hi and hi >= 3.0


def test_draw_diff_values():
    fig, main, resid = main_resid_axes()
    d, e = draw_diff(resid[0, 0], [0, 1], [0.7, 0.6], [0.03, 0.04], [0.8, 0.6], [0.0, 0.03])
    assert np.allclose(d, [-0.1, 0.0])
    assert np.allclose(e, [0.03, 0.05])
    lo, hi = resid[0, 0].get_ylim()
    assert lo == -hi and hi >= 0.13


def test_draw_pull_fixed_limit_and_gid():
    fig, main, resid = main_resid_axes()
    pull = draw_pull(resid[0, 0], [0, 1], [100, 1], [1, 1], [0, 1], [0, 0], lim=10)
    assert pull[0] == 100
    assert resid[0, 0].get_ylim() == (-10.5, 10.5)
    assert resid[0, 0].get_gid() == "residual" and main[0, 0].get_gid() is None


def test_draw_diff_autoscale_option():
    fig, main, resid = main_resid_axes()
    draw_diff(resid[0, 0], [0, 1], [0.65, 0.70], [0.002, 0.002], [0.80, 0.85], [0.0, 0.0], center_zero=False)
    lo, hi = resid[0, 0].get_ylim()
    assert hi < 0
