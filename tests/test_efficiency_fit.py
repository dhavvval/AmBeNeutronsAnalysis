"""Closure tests for the chapter 8 efficiency fit.

These run before the fit is ever pointed at real data. With a 22-event background
denominator and a known numerator/denominator inconsistency in the surrounding code,
the failure mode here is a plausible-looking wrong number, not a crash -- so the
tests target the things that would still print a clean answer while being wrong:
the interval width, the zero bin, and the overflow handling.
"""
import numpy as np
import pytest

from ambe.stats.efficiency_fit import (
    EMPTY_BIN_UNC_COUNTS, analytic_profile_datadriven, analytic_profile_poisson,
    background_scale_factor, build_multiplicity, delta_chi2_interval, fit_datadriven,
    fit_poisson, make_grid, mc_profile_poisson, multiplicity_from_counts)
import pandas as pd


# --------------------------------------------------------------------------- #
# Model correctness
# --------------------------------------------------------------------------- #
def test_analytic_poisson_is_normalised_and_matches_mc():
    rng = np.random.default_rng(1)
    for eff, lam in [(0.0, 0.0), (0.65, 0.07), (1.0, 0.3), (0.3, 0.5)]:
        a = analytic_profile_poisson(eff, lam, 8)
        assert a.sum() == pytest.approx(1.0)
        m = mc_profile_poisson(eff, lam, 8, 2_000_000, rng)
        # 2e6 throws -> per-bin sigma <= ~3.5e-4; allow 5 sigma.
        assert np.max(np.abs(a - m)) < 2e-3, f"analytic vs MC mismatch at {eff},{lam}"


def test_zero_efficiency_zero_background_is_all_zero_bin():
    p = analytic_profile_poisson(0.0, 0.0, 6)
    assert p[0] == pytest.approx(1.0)


def test_unit_efficiency_zero_background_is_all_one_bin():
    p = analytic_profile_poisson(1.0, 0.0, 6)
    assert p[1] == pytest.approx(1.0)


def test_datadriven_profile_with_delta_background_reduces_to_bernoulli():
    """A background that never fires makes the model a pure Bernoulli(eff)."""
    bkg = np.zeros(6); bkg[0] = 1.0
    p = analytic_profile_datadriven(0.42, bkg, 6)
    assert p[0] == pytest.approx(0.58)
    assert p[1] == pytest.approx(0.42)


def test_background_scale_factor_is_one_for_equal_windows():
    assert background_scale_factor(67000, 2000, 2000) == pytest.approx(1.0)
    assert background_scale_factor(67000, 2000, 20000) > 1.0


# --------------------------------------------------------------------------- #
# Interval extraction -- the part nothing else checks
# --------------------------------------------------------------------------- #
def test_delta_chi2_interval_on_exact_parabola():
    """For chi2 = ((x-mu)/sigma)^2, delta-chi2<=1 must give exactly mu +/- sigma."""
    mu, sigma = 0.64, 0.03
    g = np.linspace(0.4, 0.9, 2001)
    c = ((g - mu) / sigma) ** 2
    lo, hi = delta_chi2_interval(g, c, 1.0)
    assert lo == pytest.approx(mu - sigma, abs=1e-3)
    assert hi == pytest.approx(mu + sigma, abs=1e-3)


def test_delta_chi2_interval_two_parameter_width():
    mu, sigma = 0.5, 0.02
    g = np.linspace(0.3, 0.7, 4001)
    c = ((g - mu) / sigma) ** 2
    lo, hi = delta_chi2_interval(g, c, 2.30)
    assert (hi - lo) / 2 == pytest.approx(np.sqrt(2.30) * sigma, rel=1e-2)


# --------------------------------------------------------------------------- #
# Histogram construction -- the zero bin and the cosmic convention
# --------------------------------------------------------------------------- #
def _event_index(run, n_total, n_cosmic, mult_map, cosmic_with_cand=0):
    """Synthesise an EventIndex frame: mult_map is {multiplicity: n_events}."""
    rows = []
    n_cand_events = sum(mult_map.values())
    n_plain_cosmic = n_cosmic - cosmic_with_cand
    assert n_plain_cosmic >= 0
    # non-cosmic candidate events + cosmic events must fit inside the total
    assert n_cand_events + n_cosmic <= n_total, (
        f"fixture over-allocates: {n_cand_events} cand + {n_cosmic} cosmic > {n_total}")
    for k, n in mult_map.items():
        for _ in range(n):
            rows.append(dict(run=run, is_cosmic=False, n_accepted_clusters=k))
    for _ in range(cosmic_with_cand):
        rows.append(dict(run=run, is_cosmic=True, n_accepted_clusters=1))
    for _ in range(n_plain_cosmic):
        rows.append(dict(run=run, is_cosmic=True, n_accepted_clusters=0))
    while len(rows) < n_total:
        rows.append(dict(run=run, is_cosmic=False, n_accepted_clusters=0))
    d = pd.DataFrame(rows)
    d["n_accepted_single"] = 0
    d["n_accepted_multiple"] = 0
    return d


def test_zero_bin_comes_from_the_denominator_not_the_csv():
    ei = _event_index(6265, n_total=1000, n_cosmic=200, mult_map={1: 300, 2: 50})
    h = build_multiplicity(ei, 6265, n_bins=8, denominator="ambe_triggers")
    assert h.ambe_triggers == 800
    assert h.counts[0] == 800 - 350
    assert h.counts.sum() == 800
    assert h.naive_efficiency == pytest.approx(350 / 800)


def test_denominator_choice_changes_the_answer():
    ei = _event_index(6265, n_total=1000, n_cosmic=200, mult_map={1: 300, 2: 50})
    a = build_multiplicity(ei, 6265, denominator="ambe_triggers")
    t = build_multiplicity(ei, 6265, denominator="total_events")
    assert a.naive_efficiency == pytest.approx(350 / 800)
    assert t.naive_efficiency == pytest.approx(350 / 1000)


def test_production_numerator_leaks_cosmic_events():
    """The documented icscan.py:368-406 inconsistency must be reproducible on demand
    and must NOT be the default."""
    ei = _event_index(6062, n_total=1000, n_cosmic=200,
                      mult_map={1: 300}, cosmic_with_cand=40)
    strict = build_multiplicity(ei, 6062, numerator_mode="noncosmic")
    prod = build_multiplicity(ei, 6062, numerator_mode="production")
    assert strict.n_with_candidate == 300
    assert prod.n_with_candidate == 340
    assert prod.naive_efficiency > strict.naive_efficiency


def test_negative_zero_bin_is_refused_not_silently_clipped():
    """production numerator + ambe_triggers denominator can underflow: the numerator
    counts cosmic-vetoed candidate events that the denominator has removed."""
    ei = _event_index(1, n_total=100, n_cosmic=80, mult_map={1: 15},
                      cosmic_with_cand=70)
    # denominator = 100 - 80 = 20; production numerator = 15 + 70 = 85 > 20
    with pytest.raises(SystemExit, match="negative zero bin"):
        build_multiplicity(ei, 1, numerator_mode="production")


def test_overflow_is_counted_not_silently_clipped():
    ei = _event_index(1, n_total=100, n_cosmic=0, mult_map={1: 10, 9: 3})
    h = build_multiplicity(ei, 1, n_bins=8)
    assert h.n_overflow == 3


def test_empty_bin_uncertainty_floor_is_applied():
    h = multiplicity_from_counts(1, {1: 10}, denominator_total=100, n_bins=6)
    u = h.normed_unc
    assert np.all(u > 0)
    assert u[3] == pytest.approx(EMPTY_BIN_UNC_COUNTS / 100)


# --------------------------------------------------------------------------- #
# Closure: zero background must reproduce naive counting
# --------------------------------------------------------------------------- #
def test_zero_background_limit_recovers_naive_counting():
    """With a background that never fires, the fit must return the naive ratio."""
    denom, n_cand = 15628, 10456
    src = multiplicity_from_counts(6265, {1: n_cand}, denominator_total=denom, n_bins=6)
    bkg = multiplicity_from_counts(6264, {}, denominator_total=5000, n_bins=6)
    grid = make_grid({"min": 0.5, "max": 0.85, "step": 0.0005})
    r = fit_datadriven(src, bkg, grid, engine="analytic")
    assert r.best_eff == pytest.approx(n_cand / denom, abs=1e-3)


# --------------------------------------------------------------------------- #
# Injection-recovery: the test that validates the ERROR BARS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eff_true,lam_true", [(0.65, 0.07), (0.45, 0.03)])
def test_injection_recovery_central_value(eff_true, lam_true):
    rng = np.random.default_rng(7)
    n_ev, n_bins = 15628, 8
    p = analytic_profile_poisson(eff_true, lam_true, n_bins)
    counts = rng.multinomial(n_ev, p).astype(float)
    src = multiplicity_from_counts(
        1, {k: int(counts[k]) for k in range(1, n_bins) if counts[k]},
        denominator_total=n_ev, n_bins=n_bins)
    r = fit_poisson(src,
                    make_grid({"min": 0.30, "max": 0.95, "step": 0.002}),
                    make_grid({"min": 0.00, "max": 0.30, "step": 0.002}),
                    engine="analytic")
    assert abs(r.best_eff - eff_true) < 0.02
    assert abs(r.best_lambda - lam_true) < 0.02


def test_injection_recovery_pull_width():
    """~120 pseudo-experiments: the pull (fit-true)/sigma must have width ~1.

    This is the only test of whether the delta-chi2 interval is actually a 1-sigma
    interval. A fit can have a perfect central value and still quote error bars that
    are twice too small, and nothing else here would notice.
    """
    rng = np.random.default_rng(20260910)
    eff_true, lam_true, n_ev, n_bins = 0.65, 0.07, 15628, 8
    p = analytic_profile_poisson(eff_true, lam_true, n_bins)
    eff_grid = make_grid({"min": 0.55, "max": 0.75, "step": 0.001})
    lam_grid = make_grid({"min": 0.00, "max": 0.20, "step": 0.002})

    pulls = []
    for _ in range(120):
        counts = rng.multinomial(n_ev, p).astype(float)
        src = multiplicity_from_counts(
            1, {k: int(counts[k]) for k in range(1, n_bins) if counts[k]},
            denominator_total=n_ev, n_bins=n_bins)
        r = fit_poisson(src, eff_grid, lam_grid, engine="analytic")
        sig = r.eff_err_hi if r.best_eff < eff_true else r.eff_err_lo
        if sig > 0:
            pulls.append((r.best_eff - eff_true) / sig)
    pulls = np.array(pulls)
    assert abs(np.mean(pulls)) < 0.35, f"pull mean {np.mean(pulls):.3f} -- biased fit"
    # delta-chi2=2.30 is a 2-parameter JOINT contour; its 1D projection is wider than
    # 1 sigma, so the pull width is expected below 1. Bracket it loosely but assert it
    # is not wildly off, which is what a broken interval looks like.
    assert 0.3 < np.std(pulls) < 1.6, f"pull width {np.std(pulls):.3f} -- bad errors"


# --------------------------------------------------------------------------- #
# Fidelity to the published method
# --------------------------------------------------------------------------- #
def test_reproduces_teal_position_0():
    """Fed the histogram Teal's own best fit implies, the fitter must return his
    published numbers (thesis tables 8.4, 8.5: eps_n = 0.64, lambda_n = 0.073).

    This is the strongest single check that the port is faithful. It is run against
    his best-fit-implied histogram rather than eyeballed figure heights so the test
    has no reading error in it.
    """
    from ambe.stats.efficiency_fit import analytic_profile_poisson
    N = 9934                       # triggers after cuts, thesis table 8.3
    p = analytic_profile_poisson(0.64, 0.073, 6)
    counts = np.round(p * N).astype(int)
    src = multiplicity_from_counts(
        1594, {k: int(counts[k]) for k in range(1, 6) if counts[k]},
        denominator_total=N, n_bins=6)
    r = fit_poisson(src, make_grid({"min": 0.40, "max": 0.90, "step": 0.001}),
                    make_grid({"min": 0.00, "max": 0.30, "step": 0.001}),
                    engine="analytic")
    assert r.best_eff == pytest.approx(0.640, abs=0.005)
    assert r.best_lambda == pytest.approx(0.073, abs=0.003)


def test_analytic_matches_teal_reference_class():
    """The vendored ProfileLikelihoodBuilder and the analytic engine are the same
    model, so their profiles must agree to within MC noise."""
    from ambe.stats.profile_likelihood import (ProfileLikelihoodBuilder,
                                               ProfileLikelihoodBuilder2D)
    np.random.seed(3)
    bkg = np.array([0.90, 0.07, 0.02, 0.01, 0.0, 0.0]); bkg /= bkg.sum()
    sig = np.array([0.35, 0.5, 0.1, 0.04, 0.01, 0.0])
    B = ProfileLikelihoodBuilder()
    for eff in (0.3, 0.64, 0.9):
        teal = B.BuildMCProfileBkgDist(eff, sig, bkg, 2_000_000)
        assert np.max(np.abs(teal - analytic_profile_datadriven(eff, bkg, 6))) < 2e-3
    B2 = ProfileLikelihoodBuilder2D()
    for eff, lam in ((0.64, 0.073), (0.45, 0.2)):
        teal = B2.BuildMCProfile(eff, lam, sig, 2_000_000)
        assert np.max(np.abs(teal - analytic_profile_poisson(eff, lam, 6))) < 2e-3


def test_calc_chisquare_accepts_none_background_unc():
    """The 1D BuildLikelihoodProfile lets BkgDistUnc default to None; before the fix
    that path raised TypeError on None**2."""
    from ambe.stats.profile_likelihood import ProfileLikelihoodBuilder
    B = ProfileLikelihoodBuilder()
    v = B.CalcChiSquare(np.full(4, 0.25), np.full(4, 0.25), np.full(4, 0.01), None)
    assert v == pytest.approx(0.0)
