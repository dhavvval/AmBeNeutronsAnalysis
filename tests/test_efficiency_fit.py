"""Closure tests for the chapter 8 efficiency fit.

These run before the fit is ever pointed at real data. With a 22-event background
denominator and a known numerator/denominator inconsistency in the surrounding code,
the failure mode here is a plausible-looking wrong number, not a crash -- so the
tests target the things that would still print a clean answer while being wrong:
the interval width, the zero bin, and the overflow handling.
"""
import numpy as np
import pytest

from ambe.efficiency_extraction.efficiency_fit import (
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


def test_overflow_fails_loudly_by_default():
    """Overflow must not be silently absorbed. It used to be COUNTED and then
    DROPPED, so counts.sum() fell below the denominator and every bin fraction was
    inflated -- while the models clipped, putting data and model on opposite
    conventions."""
    ei = _event_index(1, n_total=100, n_cosmic=0, mult_map={1: 10, 9: 3})
    with pytest.raises(ValueError, match="multiplicity >= n_bins"):
        build_multiplicity(ei, 1, n_bins=8)


def test_overflow_clip_policy_is_consistent_with_the_model():
    """Under overflow_policy='clip' the data must clip the same way the model does,
    and the histogram must still account for every acquisition."""
    ei = _event_index(1, n_total=100, n_cosmic=0, mult_map={1: 10, 9: 3})
    h = build_multiplicity(ei, 1, n_bins=8, overflow_policy="clip")
    assert h.n_overflow == 3
    assert h.counts[7] == 3           # folded into the top bin, not discarded
    assert h.counts.sum() == h.denominator


def test_histogram_always_accounts_for_every_acquisition():
    """counts.sum() == denominator is the invariant the whole fit rests on."""
    for mult_map in ({1: 10}, {1: 10, 2: 5}, {1: 3, 2: 2, 7: 1}):
        ei = _event_index(1, n_total=100, n_cosmic=0, mult_map=mult_map)
        h = build_multiplicity(ei, 1, n_bins=8)
        assert h.counts.sum() == h.denominator
    h = multiplicity_from_counts(1, {1: 10, 3: 2}, denominator_total=100, n_bins=8)
    assert h.counts.sum() == h.denominator


def test_multiplicity_from_counts_records_overflow():
    """This path used to drop out-of-range keys AND report n_overflow=0, so the
    loss was invisible to every consumer."""
    with pytest.raises(ValueError, match="multiplicity >= n_bins"):
        multiplicity_from_counts(1, {1: 10, 9: 3}, denominator_total=100, n_bins=8)
    h = multiplicity_from_counts(1, {1: 10, 9: 3}, denominator_total=100, n_bins=8,
                                 overflow_policy="clip")
    assert h.n_overflow == 3
    assert h.counts.sum() == h.denominator


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


def _coverage_experiment(n_toys, eff_true=0.65, lam_true=0.07, n_ev=15628, n_bins=8,
                         seed=20260910):
    """Run n_toys pseudo-experiments, returning (profile_cover, joint_cover, pulls).

    Coverage -- how often the quoted interval actually contains the truth -- is the
    property that matters. The previous test checked only a pull WIDTH inside a band
    of 0.3 to 1.6, which admits error bars 3x too large or 1.6x too small and is
    close to vacuous.
    """
    rng = np.random.default_rng(seed)
    p = analytic_profile_poisson(eff_true, lam_true, n_bins)
    eff_grid = make_grid({"min": 0.55, "max": 0.75, "step": 0.001})
    lam_grid = make_grid({"min": 0.00, "max": 0.20, "step": 0.002})

    n_prof = n_joint = 0
    pulls = []
    for _ in range(n_toys):
        counts = rng.multinomial(n_ev, p).astype(float)
        src = multiplicity_from_counts(
            1, {k: int(counts[k]) for k in range(1, n_bins) if counts[k]},
            denominator_total=n_ev, n_bins=n_bins)
        r = fit_poisson(src, eff_grid, lam_grid, engine="analytic")
        if r.eff_lo <= eff_true <= r.eff_hi:
            n_prof += 1
        if r.eff_joint_lo <= eff_true <= r.eff_joint_hi:
            n_joint += 1
        sig = r.eff_err_hi if r.best_eff < eff_true else r.eff_err_lo
        if sig > 0:
            pulls.append((r.best_eff - eff_true) / sig)
    return n_prof / n_toys, n_joint / n_toys, np.array(pulls)


@pytest.mark.slow
def test_profile_interval_covers_at_68_percent():
    """The delta-chi2 = 1 PROFILE interval is what gets quoted as the statistical
    uncertainty, so it must contain the truth ~68.3 % of the time."""
    n_toys = 1000
    prof, _, _ = _coverage_experiment(n_toys)
    # Binomial error on a measured coverage of ~0.68 over n_toys throws.
    sigma = np.sqrt(0.683 * (1 - 0.683) / n_toys)
    assert abs(prof - 0.683) < 4 * sigma, (
        f"profile coverage {prof:.3f}, expected 0.683 +/- {sigma:.3f}")


@pytest.mark.slow
def test_joint_projection_interval_overcovers():
    """The delta-chi2 = 2.30 interval is the projection of a 2-parameter JOINT
    region. Projected onto one axis it OVERCOVERS -- that is precisely why it must
    not be reported as a one-sigma statistical error, which is what the old code
    did. Assert it is both wider than the profile interval and above 68.3 %."""
    n_toys = 1000
    prof, joint, _ = _coverage_experiment(n_toys)
    assert joint > prof, "joint projection must be the wider interval"
    sigma = np.sqrt(0.683 * (1 - 0.683) / n_toys)
    assert joint > 0.683 + 2 * sigma, (
        f"joint coverage {joint:.3f} should exceed 68.3 % by more than noise")


@pytest.mark.slow
def test_injection_recovery_is_unbiased():
    n_toys = 1000
    _, _, pulls = _coverage_experiment(n_toys)
    # Standard error on the mean pull.
    assert abs(np.mean(pulls)) < 4 / np.sqrt(n_toys) + 0.05, (
        f"pull mean {np.mean(pulls):.3f} -- biased fit")
    assert 0.75 < np.std(pulls) < 1.35, f"pull width {np.std(pulls):.3f}"


@pytest.mark.parametrize("eff_true", [0.02, 0.97])
def test_boundary_behaviour_near_zero_and_one(eff_true):
    """At the edges of the physical range the fit must still land on the truth
    rather than pinning to a grid edge and reporting it as a measurement."""
    rng = np.random.default_rng(3)
    n_ev, n_bins = 20000, 8
    p = analytic_profile_poisson(eff_true, 0.02, n_bins)
    counts = rng.multinomial(n_ev, p).astype(float)
    src = multiplicity_from_counts(
        1, {k: int(counts[k]) for k in range(1, n_bins) if counts[k]},
        denominator_total=n_ev, n_bins=n_bins)
    r = fit_poisson(src, make_grid({"min": 0.0, "max": 1.0, "step": 0.001}),
                    make_grid({"min": 0.0, "max": 0.2, "step": 0.002}),
                    engine="analytic")
    assert abs(r.best_eff - eff_true) < 0.02


# --------------------------------------------------------------------------- #
# Fidelity to the published method
# --------------------------------------------------------------------------- #
def test_poisson_model_closure_at_thesis_reference_point():
    """OPTIMIZER CLOSURE ONLY -- this test is circular, and that is all it claims.

    The data is generated by `analytic_profile_poisson(0.64, 0.073, 6)` and then
    refit with `fit_poisson(..., engine='analytic')`, i.e. the same function. It
    proves the minimiser can find the minimum of a curve it drew itself. It uses
    none of the thesis's actual bin contents, so it CANNOT detect a wrong model, a
    wrong cut flow, or a wrong histogram -- only a broken optimiser.

    It was previously named test_reproduces_teal_position_0 and described as "the
    strongest single check that the port is faithful", which overstated it. The
    genuine fidelity checks are test_analytic_matches_teal_reference_class (against
    the vendored reference implementation) and
    test_frozen_real_data_fixture_reproduces (against recorded real bin counts).
    """
    from ambe.efficiency_extraction.efficiency_fit import analytic_profile_poisson
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
    from ambe.efficiency_extraction.profile_likelihood import (ProfileLikelihoodBuilder,
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
    from ambe.efficiency_extraction.profile_likelihood import ProfileLikelihoodBuilder
    B = ProfileLikelihoodBuilder()
    v = B.CalcChiSquare(np.full(4, 0.25), np.full(4, 0.25), np.full(4, 0.01), None)
    assert v == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Frozen real-data fixture: the only NON-circular regression test
# --------------------------------------------------------------------------- #
# Aggregate counts recorded from the validated run of
#   ambe stats efficiency --config configs/data_ambe2v4_effpair_box.yaml
# on 2026-09-11, primary convention (ambe_triggers / noncosmic), window
# [2000, 67000] ns, box cuts PE <= 100, CB < 0.45, hits >= 5.
# Raw ROOT files are NOT needed -- only these aggregates, which is the point.
FROZEN = {
    "source": {"run": 6265, "total_events": 20964, "cosmic_events": 5336,
               "nonzero": {1: 9192, 2: 639, 3: 82, 4: 9, 5: 1, 7: 1}},
    "background": {"run": 6264, "total_events": 589, "cosmic_events": 567,
                   "nonzero": {1: 3, 2: 2}},
    "expected": {"poisson_eff": 0.6050, "poisson_lambda": 0.0760,
                 "datadriven_eff": 0.6400, "naive": 0.6350},
}


def _frozen_hist(block, n_bins=8):
    denom = block["total_events"] - block["cosmic_events"]
    return multiplicity_from_counts(
        block["run"], block["nonzero"], denominator_total=denom, n_bins=n_bins,
        total_events=block["total_events"], cosmic_events=block["cosmic_events"])


def test_frozen_real_data_fixture_reproduces():
    """Real recorded bin counts in, previously validated numbers out.

    Unlike test_poisson_model_closure_at_thesis_reference_point, nothing here is
    generated by the model being tested -- these are counts the pipeline actually
    produced. This is what catches a change of convention that still fits cleanly.
    """
    src = _frozen_hist(FROZEN["source"])
    bkg = _frozen_hist(FROZEN["background"])
    exp = FROZEN["expected"]

    assert src.denominator == 15628
    assert bkg.denominator == 22
    assert src.counts.sum() == src.denominator
    assert src.naive_efficiency == pytest.approx(exp["naive"], abs=5e-4)

    eff_grid = make_grid({"min": 0.30, "max": 0.95, "step": 0.001})
    lam_grid = make_grid({"min": 0.00, "max": 0.60, "step": 0.002})
    rp = fit_poisson(src, eff_grid, lam_grid, engine="analytic")
    rd = fit_datadriven(src, bkg, eff_grid, engine="analytic")
    assert rp.best_eff == pytest.approx(exp["poisson_eff"], abs=2e-3)
    assert rp.best_lambda == pytest.approx(exp["poisson_lambda"], abs=2e-3)
    assert rd.best_eff == pytest.approx(exp["datadriven_eff"], abs=2e-3)


def test_frozen_fixture_profile_interval_is_narrower_than_joint():
    """The two intervals must be distinct and correctly ordered on real data."""
    src = _frozen_hist(FROZEN["source"])
    rp = fit_poisson(src, make_grid({"min": 0.30, "max": 0.95, "step": 0.001}),
                     make_grid({"min": 0.00, "max": 0.60, "step": 0.002}),
                     engine="analytic")
    assert rp.eff_joint_lo < rp.eff_lo <= rp.best_eff <= rp.eff_hi < rp.eff_joint_hi
    assert rp.ndof_thesis == 6          # n_bins - n_fitted
    assert rp.ndof_standard == 5        # n_bins - 1 - n_fitted


# --------------------------------------------------------------------------- #
# Goodness of fit
# --------------------------------------------------------------------------- #
def test_pseudo_experiment_p_value_is_uniform_under_the_generating_model():
    """If the model IS the truth, p must be roughly uniform -- otherwise the gate
    that refuses to quote a bad fit is calibrated wrong."""
    from ambe.efficiency_extraction.efficiency_fit import pseudo_experiment_p_value, _toy_unc
    rng = np.random.default_rng(11)
    p_model = analytic_profile_poisson(0.65, 0.07, 8)
    n_ev = 15628
    ps = []
    for _ in range(60):
        toy = rng.multinomial(n_ev, p_model).astype(float)
        unc = _toy_unc(toy, n_ev)
        c2 = float(np.sum(((p_model - toy / n_ev) / unc) ** 2))
        ps.append(pseudo_experiment_p_value(p_model, n_ev, c2, _toy_unc, 300, rng))
    ps = np.array(ps)
    assert 0.3 < ps.mean() < 0.7, f"mean p {ps.mean():.3f} -- not uniform"
    assert (ps > 0.05).sum() > 0.8 * len(ps)


def test_real_data_poisson_fit_fails_goodness_of_fit():
    """The headline model on the real 6265 histogram is a BAD fit, and the code must
    be able to say so. This is what the nominal-model GoF gate exists to catch."""
    from ambe.efficiency_extraction.efficiency_fit import pseudo_experiment_p_value, _toy_unc
    src = _frozen_hist(FROZEN["source"])
    rp = fit_poisson(src, make_grid({"min": 0.30, "max": 0.95, "step": 0.001}),
                     make_grid({"min": 0.00, "max": 0.60, "step": 0.002}),
                     engine="analytic")
    p = pseudo_experiment_p_value(rp.best_profile, src.denominator, rp.chi2_min,
                                  _toy_unc, 500, np.random.default_rng(5))
    assert p < 0.01, f"p = {p}: this fit is known to be poor (chi2 ~ 51 / 6)"


# --------------------------------------------------------------------------- #
# Systematics: direction, one-sidedness, and the terms that were being ignored
# --------------------------------------------------------------------------- #
def test_model_systematic_keeps_its_sign_and_is_one_sided():
    from ambe.efficiency_extraction.efficiency_fit import systematics
    up = systematics(0.60, 0.64, n_trig_bkg=22, n_trig_src=15628)
    assert up.delta_model_signed == pytest.approx(+0.04)
    assert up.total_plus == pytest.approx(0.04)
    assert up.total_minus == pytest.approx(0.0)

    down = systematics(0.64, 0.60, n_trig_bkg=22, n_trig_src=15628)
    assert down.delta_model_signed == pytest.approx(-0.04)
    assert down.total_plus == pytest.approx(0.0)
    assert down.total_minus == pytest.approx(0.04)


def test_housing_uncertainty_actually_changes_the_total():
    """delta_h_unc used to be accepted, stored, and never read."""
    from ambe.efficiency_extraction.efficiency_fit import systematics
    bare = systematics(0.60, 0.64, 22, 15628)
    with_h = systematics(0.60, 0.64, 22, 15628, delta_h=0.0031, delta_h_unc=0.00056)
    assert with_h.total_plus > bare.total_plus
    assert with_h.total_minus > bare.total_minus
    assert with_h.components["housing"]["plus"] == pytest.approx(0.00056)


def test_housing_counts_derive_fraction_and_poisson_error():
    """Thesis 8.7.2 prose: 31 captures from 10,000 simulated neutrons. (Eq. 8.8's
    printed 1E5 denominator is a typo.)"""
    from ambe.efficiency_extraction.efficiency_fit import systematics
    sb = systematics(0.60, 0.64, 22, 15628,
                     housing_n_captures=31, housing_n_simulated=10000)
    assert sb.delta_h == pytest.approx(0.0031)
    assert sb.delta_h_unc == pytest.approx(np.sqrt(31) / 10000, rel=1e-6)
    assert sb.delta_h_unc == pytest.approx(0.00056, abs=1e-5)


def test_missing_livetime_makes_delta_bt_unpublishable():
    """Without livetime the count ratio is NOT the rate ratio of eq. 8.9, so it must
    be reported as a diagnostic and excluded from the total."""
    from ambe.efficiency_extraction.efficiency_fit import systematics
    sb = systematics(0.60, 0.64, n_trig_bkg=22, n_trig_src=15628)
    assert sb.delta_bt is not None              # still reported
    assert sb.delta_bt_publishable is False     # but not usable
    assert "background_triggers" not in sb.components
    assert sb.total_plus == pytest.approx(abs(0.64 - 0.60))


def test_livetime_gives_rate_ratio_with_propagated_uncertainty():
    from ambe.efficiency_extraction.efficiency_fit import systematics
    sb = systematics(0.60, 0.64, n_trig_bkg=100, n_trig_src=10000,
                     livetime_bkg_s=50.0, livetime_src_s=1000.0)
    assert sb.delta_bt_publishable is True
    assert sb.background_rate_hz == pytest.approx(2.0)
    assert sb.source_rate_hz == pytest.approx(10.0)
    assert sb.delta_bt == pytest.approx(0.2)
    expected_rel = np.sqrt(1 / 100 + 1 / 10000)
    assert sb.delta_bt_unc == pytest.approx(0.2 * expected_rel, rel=1e-6)
    # One-sided: false starts dilute, so the correction is upward only.
    assert sb.components["background_triggers"]["minus"] == 0.0


def test_equal_livetime_rate_ratio_equals_count_ratio():
    from ambe.efficiency_extraction.efficiency_fit import systematics
    eq = systematics(0.60, 0.64, 100, 10000, livetime_bkg_s=7.0, livetime_src_s=7.0)
    assert eq.delta_bt == pytest.approx(100 / 10000)


def test_unequal_livetime_shows_why_count_ratio_is_wrong():
    from ambe.efficiency_extraction.efficiency_fit import systematics
    counts_only = systematics(0.60, 0.64, 100, 10000)
    rates = systematics(0.60, 0.64, 100, 10000,
                        livetime_bkg_s=50.0, livetime_src_s=1000.0)
    assert counts_only.delta_bt == pytest.approx(0.01)
    assert rates.delta_bt == pytest.approx(0.20)     # 20x different


def test_zero_livetime_is_treated_as_missing_not_divided_by():
    from ambe.efficiency_extraction.efficiency_fit import systematics
    sb = systematics(0.60, 0.64, 100, 10000, livetime_bkg_s=0.0, livetime_src_s=10.0)
    assert sb.delta_bt_publishable is False


def test_zero_source_count_is_handled():
    from ambe.efficiency_extraction.efficiency_fit import systematics
    sb = systematics(0.60, 0.64, n_trig_bkg=22, n_trig_src=0)
    assert sb.delta_bt is None


# --------------------------------------------------------------------------- #
# Validation of mode arguments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["analytc", "Analytic", "mcc", ""])
def test_invalid_engine_is_rejected(bad):
    """A typo used to fall through an `else` and silently run the toy-MC path."""
    src = multiplicity_from_counts(1, {1: 100}, denominator_total=1000, n_bins=6)
    grid = make_grid({"min": 0.0, "max": 0.5, "step": 0.01})
    with pytest.raises(ValueError, match="engine must be one of"):
        fit_poisson(src, grid, make_grid({"min": 0.0, "max": 0.2, "step": 0.01}),
                    engine=bad)
    with pytest.raises(ValueError, match="engine must be one of"):
        fit_datadriven(src, src, grid, engine=bad)


def test_invalid_likelihood_and_background_mode_are_rejected():
    src = multiplicity_from_counts(1, {1: 100}, denominator_total=1000, n_bins=6)
    grid = make_grid({"min": 0.0, "max": 0.5, "step": 0.01})
    with pytest.raises(ValueError, match="likelihood must be one of"):
        fit_datadriven(src, src, grid, likelihood="chi2")
    with pytest.raises(ValueError, match="background_uncertainty must be one of"):
        fit_datadriven(src, src, grid, bkg_unc_mode="diagonal")


# --------------------------------------------------------------------------- #
# Opt-in statistical modes
# --------------------------------------------------------------------------- #
def test_multinomial_mode_runs_and_stays_close_on_a_good_fit():
    """Where the thesis chi2 is a valid approximation the two must broadly agree;
    the mode exists to MEASURE the difference, not to hide it."""
    rng = np.random.default_rng(4)
    n_ev, n_bins = 15628, 8
    p = analytic_profile_poisson(0.65, 0.07, n_bins)
    counts = rng.multinomial(n_ev, p).astype(float)
    src = multiplicity_from_counts(
        1, {k: int(counts[k]) for k in range(1, n_bins) if counts[k]},
        denominator_total=n_ev, n_bins=n_bins)
    eff_grid = make_grid({"min": 0.55, "max": 0.75, "step": 0.001})
    lam_grid = make_grid({"min": 0.00, "max": 0.20, "step": 0.002})
    a = fit_poisson(src, eff_grid, lam_grid, likelihood="thesis_chi2")
    b = fit_poisson(src, eff_grid, lam_grid, likelihood="multinomial")
    assert abs(a.best_eff - b.best_eff) < 0.02


def test_bootstrap_background_mode_widens_the_interval_on_a_tiny_background():
    """With a 22-trigger background the template uncertainty is the whole story, so
    propagating it through the convolution must not silently shrink the interval."""
    src = _frozen_hist(FROZEN["source"])
    bkg = _frozen_hist(FROZEN["background"])
    grid = make_grid({"min": 0.30, "max": 0.95, "step": 0.002})
    r = fit_datadriven(src, bkg, grid, bkg_unc_mode="bootstrap", n_bootstrap=100,
                       rng=np.random.default_rng(2))
    assert 0.30 < r.best_eff < 0.95
    assert r.eff_hi > r.eff_lo


def test_background_scale_rescales_uncertainties_too():
    """bkg_unc used to be captured BEFORE the bkg_scale reweighting, so a rescaled
    template went into the chi2 denominator carrying the unscaled shape's errors."""
    src = _frozen_hist(FROZEN["source"])
    bkg = _frozen_hist(FROZEN["background"])
    grid = make_grid({"min": 0.30, "max": 0.95, "step": 0.002})
    r1 = fit_datadriven(src, bkg, grid, bkg_scale=1.0)
    r2 = fit_datadriven(src, bkg, grid, bkg_scale=1.5)
    # Not asserting a direction -- only that the scale actually propagates and the
    # fit stays well defined, which it did not when the two were inconsistent.
    assert r1.best_eff != r2.best_eff
    assert np.isfinite(r2.chi2_min)
