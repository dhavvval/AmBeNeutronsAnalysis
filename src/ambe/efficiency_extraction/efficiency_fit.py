"""Neutron detection efficiency by multiplicity fit -- Pershing thesis chapter 8.6.

The efficiency quoted elsewhere in this repo is a naive counting ratio,
unique_neutron_triggers / ambe_triggers (data/icscan.py:17-18). That ratio is biased
whenever the delayed window holds background: an acquisition contains at most one
correlated AmBe neutron, so every additional candidate is background, and counting
"has >= 1 candidate" folds it straight into the efficiency. This module fits the
per-acquisition multiplicity distribution instead, M = delta_S + N_B (eq. 8.2), with
delta_S ~ Bernoulli(eps_n) (eq. 8.3) and N_B drawn either from the measured background
run (data-driven, one parameter, eq. 8.5) or from Poisson(lambda_n) (uncorrelated, two
parameters, eq. 8.6). eps_n is the chi2 profile minimum, with the 68.3 % interval at
delta-chi2 <= 1; the gap between the two models is the background-model systematic.

Acquisitions with NO candidate are the most constraining bin and the only bin missing
from the candidate CSVs, which hold just the events that produced a candidate -- it has
to come from the Stage-1 trigger count. Getting that denominator wrong is the easiest
way to produce a confident wrong number, so build_multiplicity takes the denominator
choice explicitly and records its provenance rather than inferring one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import poisson

# Poisson upper 68.3 % confidence limit for zero observed counts. Teal's convention
# for empty bins (phase2.py:131-132): an empty bin has sqrt(N) = 0, which would make
# its chi2 term infinite and let a single empty bin dictate the whole fit. Substituting
# 1.15 counts of uncertainty is the standard Feldman-Cousins-style floor.
EMPTY_BIN_UNC_COUNTS = 1.15

DENOMINATOR_CHOICES = ("ambe_triggers", "total_events")
NUMERATOR_MODES = ("noncosmic", "production")
OVERFLOW_POLICIES = ("fail", "clip")
NOMINAL_MODELS = ("data_driven", "poisson")
LIKELIHOODS = ("thesis_chi2", "multinomial")
BKG_UNC_MODES = ("thesis_diagonal", "bootstrap")
ENGINES = ("analytic", "mc")


@dataclass(frozen=True)
class MultiplicityHist:
    """A per-acquisition candidate-multiplicity histogram, zero bin included.

    `counts[k]` is the number of acquisitions with exactly k accepted delayed
    candidates. `counts[0]` is derived from the trigger count, not observed directly.
    """
    run: int
    counts: np.ndarray
    total_events: int
    cosmic_events: int
    denominator_mode: str
    numerator_mode: str
    provenance: str
    n_overflow: int = 0

    def __post_init__(self):
        """Every acquisition in the denominator must land in exactly one bin."""
        # Overflow acquisitions used to be counted in n_overflow and then dropped from
        # counts, so normed renormalised over a shrunken total and inflated every bin
        # fraction, bin 0 most of all. The model profiles clip overflow into the top
        # bin, so data and model were on opposite conventions whenever overflow existed.
        total = int(round(float(self.counts.sum())))
        if total != self.denominator:
            raise ValueError(
                f"[efficiency_fit] run {self.run}: histogram sums to {total:,} but "
                f"the {self.denominator_mode} denominator is {self.denominator:,} "
                f"(difference {self.denominator - total:+,}). Every acquisition must "
                f"land in exactly one bin. If this is overflow, raise n_bins or set "
                f"overflow_policy='clip'.")

    @property
    def ambe_triggers(self) -> int:
        return self.total_events - self.cosmic_events

    @property
    def denominator(self) -> int:
        return (self.ambe_triggers if self.denominator_mode == "ambe_triggers"
                else self.total_events)

    @property
    def n_with_candidate(self) -> int:
        return int(self.counts[1:].sum())

    @property
    def naive_efficiency(self) -> float:
        """unique_neutron_triggers / denominator -- the number this module corrects."""
        return self.n_with_candidate / self.denominator if self.denominator else np.nan

    @property
    def normed(self) -> np.ndarray:
        return self.counts / self.counts.sum()

    @property
    def normed_unc(self) -> np.ndarray:
        """sqrt(N)/N_tot per bin with the empty-bin floor; source AND background."""
        return _toy_unc(self.counts, self.counts.sum())

    def summary(self) -> str:
        return (f"run {self.run}: total={self.total_events:,} cosmic={self.cosmic_events:,} "
                f"ambe={self.ambe_triggers:,} denom={self.denominator:,} "
                f"({self.denominator_mode}/{self.numerator_mode}) "
                f"hist={list(self.counts.astype(int))} "
                f"naive_eff={self.naive_efficiency:.4f}")


def load_event_index(paths: Iterable[Path] | Path) -> pd.DataFrame:
    """Read one or more EventIndex parquets, one row per Stage-1-admitted event."""
    # n_accepted_clusters is synthesised if absent, matching icscan.py:193-195 so the
    # two modules agree on what an accepted cluster count is.
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"[efficiency_fit] missing EventIndex parquet: {p}")
        frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True)
    if "n_accepted_clusters" not in d.columns:
        d["n_accepted_clusters"] = d["n_accepted_single"] + d["n_accepted_multiple"]
    return d


def build_multiplicity(event_index: pd.DataFrame, run: int, n_bins: int = 8,
                       denominator: str = "ambe_triggers",
                       numerator_mode: str = "noncosmic",
                       provenance: str = "EventIndex",
                       overflow_policy: str = "fail") -> MultiplicityHist:
    """Build the multiplicity histogram for one run, zero bin included."""
    # numerator_mode='noncosmic' lets an event contribute only if it is not
    # cosmic-vetoed, consistent with a denominator that excludes cosmics.
    # 'production' reproduces the production accumulator, which counts a vetoed event
    # in the numerator while subtracting it from the denominator (icscan.py:368-406) --
    # inconsistent, but it is what the published numbers count. On run 6062 the two
    # differ by 2.8 points, more than the whole systematic budget of table 8.6, so both
    # are offered and the caller is expected to report both.
    # overflow_policy='fail' treats multiplicity >= n_bins as an error (merging
    # physically different multiplicities is not a choice to make implicitly, and these
    # runs have none); 'clip' folds it into the top bin, which then means ">= n_bins-1".
    if denominator not in DENOMINATOR_CHOICES:
        raise SystemExit(f"[efficiency_fit] denominator must be one of {DENOMINATOR_CHOICES}")
    if numerator_mode not in NUMERATOR_MODES:
        raise SystemExit(f"[efficiency_fit] numerator_mode must be one of {NUMERATOR_MODES}")
    if overflow_policy not in OVERFLOW_POLICIES:
        raise SystemExit(
            f"[efficiency_fit] overflow_policy must be one of {OVERFLOW_POLICIES}")

    grp = event_index[event_index["run"] == run]
    if grp.empty:
        raise SystemExit(f"[efficiency_fit] no EventIndex rows for run {run}")

    cosmic = grp["is_cosmic"].astype(bool)
    total_events = int(len(grp))
    cosmic_events = int(cosmic.sum())

    # Which events are allowed to contribute candidates to the numerator.
    contributing = grp if numerator_mode == "production" else grp[~cosmic]
    mult = contributing["n_accepted_clusters"].to_numpy(int)
    mult = mult[mult > 0]

    n_overflow = int((mult >= n_bins).sum())
    if n_overflow and overflow_policy == "fail":
        raise ValueError(
            f"[efficiency_fit] run {run}: {n_overflow:,} acquisitions have "
            f"multiplicity >= n_bins={n_bins} (observed max {int(mult.max())}). "
            f"Raise n_bins so every multiplicity has its own bin, or set "
            f"overflow_policy='clip' to fold them into bin {n_bins - 1} and read "
            f"that bin as '>= {n_bins - 1}' everywhere.")
    # Clip rather than discard. The model profiles already clip
    # (analytic_profile_poisson / analytic_profile_datadriven), so discarding here
    # would leave data and model on opposite conventions and shrink counts.sum()
    # below the denominator. bincount raises on a negative multiplicity, which the
    # old min(int(k), n_bins-1) loop silently folded into the TOP bin via counts[-1].
    counts = np.bincount(np.minimum(mult, n_bins - 1),
                         minlength=n_bins).astype(float)

    denom_val = (total_events - cosmic_events if denominator == "ambe_triggers"
                 else total_events)
    zero_bin = denom_val - int(len(mult))
    if zero_bin < 0:
        raise SystemExit(
            f"[efficiency_fit] run {run}: negative zero bin ({zero_bin}). The "
            f"denominator ({denominator}={denom_val:,}) is smaller than the number of "
            f"candidate-bearing events ({len(mult):,}). With numerator_mode="
            f"'{numerator_mode}' this means cosmic-vetoed events are being counted in "
            f"the numerator while excluded from the denominator -- use "
            f"numerator_mode='noncosmic', or denominator='total_events'.")
    counts[0] = zero_bin

    return MultiplicityHist(
        run=int(run), counts=counts, total_events=total_events,
        cosmic_events=cosmic_events, denominator_mode=denominator,
        numerator_mode=numerator_mode, provenance=provenance, n_overflow=n_overflow)


def multiplicity_from_counts(run: int, nonzero: dict[int, int], denominator_total: int,
                             n_bins: int = 8, provenance: str = "explicit",
                             total_events: Optional[int] = None,
                             cosmic_events: int = 0,
                             overflow_policy: str = "fail") -> MultiplicityHist:
    """Build a histogram from an explicit {multiplicity: n_events} map.

    For validation against published numbers (thesis figs 8.18/8.19) and the closure
    tests, where there is no EventIndex to read. overflow_policy behaves as in
    build_multiplicity; this path used to drop out-of-range keys silently.
    """
    if overflow_policy not in OVERFLOW_POLICIES:
        raise SystemExit(
            f"[efficiency_fit] overflow_policy must be one of {OVERFLOW_POLICIES}")
    counts = np.zeros(n_bins, dtype=float)
    n_overflow = 0
    for k, v in nonzero.items():
        if k <= 0:
            raise SystemExit("[efficiency_fit] nonzero map must not contain bin 0")
        if k >= n_bins:
            n_overflow += int(v)
        counts[min(int(k), n_bins - 1)] += int(v)
    if n_overflow and overflow_policy == "fail":
        raise ValueError(
            f"[efficiency_fit] run {run}: {n_overflow:,} acquisitions have "
            f"multiplicity >= n_bins={n_bins}. Raise n_bins, or set "
            f"overflow_policy='clip'.")
    counts[0] = denominator_total - int(sum(nonzero.values()))
    if counts[0] < 0:
        raise SystemExit(f"[efficiency_fit] negative zero bin for run {run}")
    tot = total_events if total_events is not None else denominator_total + cosmic_events
    return MultiplicityHist(
        run=int(run), counts=counts, total_events=int(tot),
        cosmic_events=int(cosmic_events),
        denominator_mode="ambe_triggers" if cosmic_events else "total_events",
        numerator_mode="noncosmic", provenance=provenance, n_overflow=n_overflow)


def background_scale_factor(acq_window_ns: float, sig_start_ns: float,
                            bkg_start_ns: float) -> float:
    """Rescale a flat background rate from one time window to another (phase2.py:169)."""
    # Equals 1 whenever the two windows start together, as they do for the 6264/6265
    # pair, but is kept explicit so it cannot silently go wrong if they diverge.
    num = acq_window_ns - sig_start_ns
    den = acq_window_ns - bkg_start_ns
    if den <= 0:
        raise SystemExit("[efficiency_fit] background window starts at or after the "
                         "end of the acquisition window")
    return num / den


def poisson_pmf(k: np.ndarray, lam) -> np.ndarray:
    """Poisson pmf, broadcasting k against lam; k on the last axis."""
    # scipy handles mu = 0 natively, returning [1, 0, 0, ...], so the lambda grid can
    # start at zero without the np.where guard the hand-rolled gammaln version needed.
    return poisson.pmf(k, mu=lam)


def analytic_profile_poisson(eff, lam, n_bins: int) -> np.ndarray:
    """Exact pmf of M = Bernoulli(eff) + Poisson(lam), binned like the data.

    eff and lam broadcast against each other and the bin axis is last, so passing
    eff_grid[:, None] with lam_grid[None, :] builds the whole (n_eff, n_lam, n_bins)
    scan cube in one shot -- a nested Python loop over ~200k grid points was the
    dominant cost of fit_poisson.
    """
    # P(M=k) = eff * Pois(k-1; lam) + (1-eff) * Pois(k; lam). Not an approximation of
    # the toy MC -- the same model in closed form. It removes the MC sampling noise
    # that otherwise makes delta-chi2 contours jitter between grid points.
    pois = poisson_pmf(np.arange(n_bins), np.asarray(lam)[..., None])
    shifted = np.zeros_like(pois)
    shifted[..., 1:] = pois[..., :-1]
    e = np.asarray(eff, dtype=float)[..., None]
    p = (1.0 - e) * pois + e * shifted
    # Everything above the top bin clips into it, matching the MC path's
    # np.histogram(range=(0, n_bins)) so the two engines compare bin for bin.
    p[..., -1] += np.clip(1.0 - p.sum(axis=-1), 0.0, None)
    return p / p.sum(axis=-1, keepdims=True)


def mc_profile_poisson(eff: float, lam: float, n_bins: int, n_throws: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Toy-MC counterpart of `analytic_profile_poisson`. Teal's path (eq. 8.2-8.4)."""
    n = int(n_throws)
    sig = (rng.random(n) <= eff).astype(np.int64)
    bkg = rng.poisson(lam, n)
    tot = np.clip(sig + bkg, 0, n_bins - 1)
    return np.bincount(tot, minlength=n_bins)[:n_bins] / n


def mc_profile_datadriven(eff: float, bkg_p: np.ndarray, n_bins: int, n_throws: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Toy MC with the background drawn from a measured histogram (eq. 8.2, fig 8.19)."""
    n = int(n_throws)
    p = np.asarray(bkg_p, dtype=float)
    s = p.sum()
    if s <= 0:
        raise SystemExit("[efficiency_fit] background template sums to zero")
    # np.random.choice demands p sum to exactly 1; counts/sum can land at 1 +/- 1e-16.
    p = p / s
    bkg = rng.choice(len(p), n, p=p)
    sig = (rng.random(n) <= eff).astype(np.int64)
    tot = np.clip(sig + bkg, 0, n_bins - 1)
    return np.bincount(tot, minlength=n_bins)[:n_bins] / n


def analytic_profile_datadriven(eff: float, bkg_p: np.ndarray, n_bins: int) -> np.ndarray:
    """Exact convolution of Bernoulli(eff) with the measured background histogram.

    `bkg_p` may carry leading axes, so the bootstrap branch of fit_datadriven convolves
    all its Dirichlet replicas in one call instead of a Python loop per grid point.
    """
    # Written as (1-eff)*b_k + eff*b_{k-1} rather than np.convolve with [1-eff, eff]:
    # the two are the same thing, but this way the Bernoulli stays visible.
    p = np.asarray(bkg_p, dtype=float)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.zeros(p.shape[:-1] + (n_bins,))
    m = min(p.shape[-1], n_bins)
    out[..., :m] += (1.0 - eff) * p[..., :m]
    shifted = np.zeros_like(out)
    shifted[..., 1:min(p.shape[-1] + 1, n_bins)] = p[..., :min(p.shape[-1], n_bins - 1)]
    out += eff * shifted
    out[..., -1] += np.clip(1.0 - out.sum(axis=-1), 0.0, None)
    return out / out.sum(axis=-1, keepdims=True)


def thesis_chi2(model: np.ndarray, data: np.ndarray, data_unc: np.ndarray,
                bkg_unc: Optional[np.ndarray] = None) -> np.ndarray:
    """thesis eq. 8.5 (with bkg_unc) / eq. 8.6 (without); bins on the last axis.

    Anything broadcastable against `data` works, so a single profile returns a scalar
    and a whole (n_eff, n_lam, n_bins) scan cube returns the (n_eff, n_lam) surface.
    That is the only reason this is not three near-copies any more.
    """
    # The data-driven model carries the background histogram's statistical uncertainty
    # in the denominator because that template is itself measured; the Poisson model
    # does not, because its background is a fitted parameter rather than a measurement.
    den = data_unc if bkg_unc is None else np.sqrt(data_unc ** 2 + bkg_unc ** 2)
    if np.any(den <= 0):
        raise SystemExit("[efficiency_fit] zero uncertainty in a chi2 denominator -- "
                         "the empty-bin floor was not applied")
    return np.sum(((model - data) / den) ** 2, axis=-1)


def multinomial_deviance(model: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """-2 ln L against the saturated model, for observed COUNTS; bins on the last axis."""
    # thesis_chi2 treats the normalised bins as independent Gaussians with variance
    # p_i/N. They are neither: multiplicity bins are multinomial, so they are
    # anti-correlated (Cov = -p_i p_j / N, the probabilities must sum to one) and the
    # diagonal variance is p_i(1-p_i)/N. The gap is largest where it matters most --
    # bin 0 holds most of the mass, so the diagonal form overestimates the uncertainty
    # of the single most constraining bin. Offered as `likelihood: multinomial` with
    # the thesis chi2 still the default, so the shift is a measured number.
    c = np.asarray(counts, dtype=float)
    # xlogy gives 0*log(0) = 0 exactly, which is what the old np.where + errstate
    # dance was emulating for empty bins. The 1e-300 clip is load-bearing and stays:
    # without it a model bin of exactly zero against a non-empty count returns -inf,
    # and np.argmin on a surface containing nan returns the nan's index. The whole
    # lambda = 0 column of the scan cube is exactly zero above bin 1.
    # Subtract the saturated model per bin, not per histogram: it keeps the summation
    # order of the scalar form this replaces, so the statistic is bit-identical to the
    # pre-refactor numbers rather than merely equal to 1e-15.
    sat = xlogy(c, c / c.sum())
    m = np.clip(np.asarray(model, dtype=float), 1e-300, None)
    m = m / m.sum(axis=-1, keepdims=True)
    return 2.0 * np.sum(sat - xlogy(c, m), axis=-1)


def bootstrap_background_templates(bkg: "MultiplicityHist", n_draws: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """Dirichlet-resample the background template `n_draws` times."""
    # The thesis adds the background histogram's per-bin uncertainty straight into the
    # chi2 denominator, but the fitted model is a Bernoulli CONVOLUTION of the template:
    # an uncertainty in background bin k propagates into model bins k and k+1, weighted
    # by the efficiency, not into bin k alone. The per-bin approximation matters most
    # when the background run is small -- exactly this regime, 6264 has 22 triggers.
    # Jeffreys prior (alpha = counts + 1/2) rather than a plain multinomial bootstrap,
    # so bins observed as zero still carry mass instead of being frozen at zero.
    alpha = np.asarray(bkg.counts, dtype=float) + 0.5
    return rng.dirichlet(alpha, size=int(n_draws))


def pseudo_experiment_p_value(model_p: np.ndarray, n_events: int, chi2_obs: float,
                              data_unc_fn, n_toys: int,
                              rng: np.random.Generator) -> float:
    """Fraction of toy datasets with a chi2 at least as large as the observed one."""
    # Preferred over the analytic chi2 CDF because that CDF's assumptions do not hold
    # here: the bins are normalised and hence correlated, the empty-bin floor distorts
    # the statistic, and the fitted parameters do not cost exactly one dof each.
    # data_unc_fn(counts, n_tot) supplies a toy's per-bin denominator, so toy and real
    # fit share one uncertainty convention, floor included.
    p = np.clip(np.asarray(model_p, dtype=float), 0.0, None)
    p = p / p.sum()
    n_tot = int(n_events)
    # One batched draw rather than n_toys sequential ones. numpy fills the batch in
    # the same order, so both the toys and the generator state afterwards are
    # identical to the loop -- which matters, because `rng` is shared across the two
    # fits in run() and the published p-values depend on that stream.
    toys = rng.multinomial(n_tot, p, size=int(n_toys)).astype(float)
    unc = data_unc_fn(toys, n_tot)
    c2 = np.sum(((p - toys / n_tot) / unc) ** 2, axis=-1)
    return float(np.count_nonzero(c2 >= chi2_obs)) / float(n_toys)


def _toy_unc(counts: np.ndarray, n_tot: int) -> np.ndarray:
    """sqrt(N)/N_tot per bin with the empty-bin floor -- the one definition of it."""
    unc = np.sqrt(counts) / n_tot
    unc[counts == 0] = EMPTY_BIN_UNC_COUNTS / n_tot
    return unc


def delta_chi2_interval(grid: np.ndarray, chi2_vals: np.ndarray,
                        delta: float) -> tuple[float, float]:
    """Interval where chi2 - chi2_min <= delta; 1.0 for one parameter, 2.30 for two."""
    # Edges are linearly interpolated between grid points rather than snapped to them,
    # so the interval does not quantise to the scan step. A bound running off the end
    # of the grid comes back as the grid edge, and assert_interior refuses to quote it.
    d = np.asarray(chi2_vals) - np.min(chi2_vals)
    inside = np.flatnonzero(d <= delta)
    if inside.size == 0:
        raise SystemExit("[efficiency_fit] no grid point within delta-chi2 -- "
                         "the grid is too coarse or the fit did not converge")
    lo_i, hi_i = inside[0], inside[-1]

    def interp(i_in: int, i_out: int) -> float:
        if i_out < 0 or i_out >= len(grid):
            return float(grid[i_in])
        y0, y1 = d[i_out], d[i_in]
        if y0 == y1:
            return float(grid[i_in])
        f = (delta - y1) / (y0 - y1)
        return float(grid[i_in] + f * (grid[i_out] - grid[i_in]))

    return interp(lo_i, lo_i - 1), interp(hi_i, hi_i + 1)


@dataclass
class FitResult:
    """One model's fit, carrying two intervals deliberately.

    eff_lo/eff_hi are the PROFILE interval at delta-chi2 <= 1 -- the conventional
    68.3 % interval for one parameter after minimising over any nuisance parameter,
    and what should be quoted as the statistical uncertainty. eff_joint_lo/eff_joint_hi
    project the 68.3 % JOINT two-parameter region (delta-chi2 <= 2.30, the thesis
    convention), which OVERCOVERS for a single parameter. The old code profiled and
    then applied 2.30, mixing the two, and called it a one-sigma error.
    """
    model: str
    best_eff: float
    eff_lo: float
    eff_hi: float
    chi2_min: float
    ndof: int
    eff_grid: np.ndarray
    chi2_eff: np.ndarray          # chi2 profiled over any nuisance parameter
    best_profile: np.ndarray
    best_lambda: Optional[float] = None
    lam_lo: Optional[float] = None
    lam_hi: Optional[float] = None
    lam_grid: Optional[np.ndarray] = None
    chi2_lam: Optional[np.ndarray] = None
    chi2_surface: Optional[np.ndarray] = None   # (n_eff, n_lam), 2D fit only
    warnings: list[str] = field(default_factory=list)
    # Joint-region projection (thesis convention). None for the 1-parameter fit,
    # where it is the same statement as the profile interval.
    eff_joint_lo: Optional[float] = None
    eff_joint_hi: Optional[float] = None
    n_params: int = 1
    n_bins_fit: int = 0
    p_value: Optional[float] = None
    p_value_basis: str = "not computed"

    @property
    def ndof_standard(self) -> int:
        """n_bins - 1 - n_fitted; the thesis convention (`ndof`) omits the -1."""
        # Data and model are both normalised probability histograms, so normalisation
        # costs one degree of freedom before any parameter is fitted. Reporting only
        # one of the two conventions silently picks a definition.
        return max(self.n_bins_fit - 1 - self.n_params, 1)

    @property
    def chi2_per_ndof(self) -> float:
        return self.chi2_min / self.ndof if self.ndof > 0 else np.nan

    @property
    def chi2_per_ndof_standard(self) -> float:
        n = self.ndof_standard
        return self.chi2_min / n if n > 0 else np.nan

    @property
    def eff_err_lo(self) -> float:
        return self.best_eff - self.eff_lo

    @property
    def eff_err_hi(self) -> float:
        return self.eff_hi - self.best_eff

    def summary(self) -> str:
        s = (f"{self.model}: eps_n = {self.best_eff:.4f} "
             f"+{self.eff_err_hi:.4f} -{self.eff_err_lo:.4f}  "
             f"chi2/ndof = {self.chi2_min:.2f}/{self.ndof}")
        if self.best_lambda is not None:
            s += f"  lambda_n = {self.best_lambda:.4f}"
        return s


def assert_interior(result: FitResult, name: str = "efficiency") -> None:
    """Refuse to quote a best fit pinned to the edge of its scan grid."""
    # An edge-pinned minimum means the true minimum is outside the grid, so the central
    # value and the interval are both meaningless -- and they still print as perfectly
    # ordinary numbers, which is exactly why this has to be checked.
    g = result.eff_grid
    if result.best_eff <= g[0] or result.best_eff >= g[-1]:
        raise SystemExit(
            f"[efficiency_fit] {name}: best fit {result.best_eff:.4f} is at the edge "
            f"of the scan grid [{g[0]:.4f}, {g[-1]:.4f}]. Widen eff_grid; the "
            f"minimum is outside it and this number is not meaningful.")
    if result.lam_grid is not None and result.best_lambda is not None:
        lg = result.lam_grid
        if result.best_lambda <= lg[0] or result.best_lambda >= lg[-1]:
            raise SystemExit(
                f"[efficiency_fit] {name}: best lambda {result.best_lambda:.4f} is at "
                f"the edge of [{lg[0]:.4f}, {lg[-1]:.4f}]. Widen lambda_grid.")


def make_grid(spec: dict) -> np.ndarray:
    """Build a scan grid from a {min, max, step} config block, inclusive of max."""
    lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n)


def fit_datadriven(source: MultiplicityHist, bkg: MultiplicityHist,
                   eff_grid: np.ndarray, engine: str = "analytic",
                   n_throws: int = 1_000_000, rng: Optional[np.random.Generator] = None,
                   bkg_scale: float = 1.0,
                   likelihood: str = "thesis_chi2",
                   bkg_unc_mode: str = "thesis_diagonal",
                   n_bootstrap: int = 400) -> FitResult:
    """1D scan over eps_n with the background drawn from the measured histogram.

    thesis eq. 8.5; one fitted parameter, so the 68.3 % interval is delta-chi2 <= 1.
    """
    # bkg_scale rescales the template's MEAN to the source's time window (see
    # background_scale_factor); at 1.0 it is used as measured.
    # bkg_unc_mode='thesis_diagonal' adds the template's per-bin sigma to the chi2
    # denominator as in eq. 8.5; 'bootstrap' instead Dirichlet-resamples the template
    # and propagates it THROUGH the Bernoulli convolution, where the uncertainty
    # actually acts. The per-bin addition degrades when the background run is small.
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"likelihood must be one of {LIKELIHOODS}, got {likelihood!r}")
    if bkg_unc_mode not in BKG_UNC_MODES:
        raise ValueError(
            f"background_uncertainty must be one of {BKG_UNC_MODES}, got {bkg_unc_mode!r}")

    rng = rng or np.random.default_rng()
    n_bins = len(source.counts)
    data, data_unc = source.normed, source.normed_unc
    bkg_p, bkg_unc = bkg.normed, bkg.normed_unc

    if bkg_scale != 1.0:
        # Rescaling a discrete multiplicity template is only defined through its
        # mean; reweight bin k by scale**k and renormalise, which is exact for a
        # Poisson template and a reasonable approximation otherwise.
        w = bkg_scale ** np.arange(n_bins)
        bkg_p = bkg_p * w
        s = bkg_p.sum()
        bkg_p = bkg_p / s
        # The uncertainties must be carried through the SAME reweighting. Previously
        # bkg_unc was captured before this block and kept the unscaled shape's
        # errors, so a rescaled template went into the chi2 denominator with
        # uncertainties belonging to a different distribution.
        bkg_unc = bkg_unc * w / s

    warnings: list[str] = []
    if bkg.denominator < 200:
        warnings.append(
            f"background run {bkg.run} has only {bkg.denominator:,} triggers in the "
            f"denominator; its per-bin uncertainty dominates the chi2 and the "
            f"data-driven interval will be correspondingly wide. Treat this model as "
            f"a systematic cross-check, not the primary result.")
    if bkg.n_with_candidate < 20:
        warnings.append(
            f"background run {bkg.run} has only {bkg.n_with_candidate} "
            f"candidate-bearing events; the background template is nearly a delta at "
            f"zero, which drives this fit toward the naive counting ratio.")

    # In bootstrap mode the template uncertainty is propagated through the
    # convolution instead of being added per-bin, so it must NOT also appear in the
    # denominator or it would be counted twice.
    if bkg_unc_mode == "bootstrap":
        templates = bootstrap_background_templates(bkg, n_bootstrap, rng)
        if bkg_scale != 1.0:
            w = bkg_scale ** np.arange(n_bins)
            templates = templates * w
            templates = templates / templates.sum(axis=1, keepdims=True)
        den_bkg_unc = None
    else:
        templates = None
        den_bkg_unc = bkg_unc

    chi2_vals = np.empty(len(eff_grid))
    profiles = []
    for i, eff in enumerate(eff_grid):
        if templates is not None:
            # Mean model over template replicas, plus the spread they induce in each
            # model bin -- that spread IS the propagated background uncertainty.
            reps = analytic_profile_datadriven(eff, templates, n_bins)
            prof = reps.mean(axis=0)
            prop_unc = reps.std(axis=0, ddof=1)
            den = np.sqrt(data_unc ** 2 + prop_unc ** 2)
        elif engine == "analytic":
            prof = analytic_profile_datadriven(eff, bkg_p, n_bins)
            den = None
        else:
            prof = mc_profile_datadriven(eff, bkg_p, n_bins, n_throws, rng)
            den = None
        profiles.append(prof)
        if likelihood == "multinomial":
            chi2_vals[i] = multinomial_deviance(prof, source.counts)
        elif den is not None:
            chi2_vals[i] = thesis_chi2(prof, data, den)
        else:
            chi2_vals[i] = thesis_chi2(prof, data, data_unc, den_bkg_unc)

    best = int(np.argmin(chi2_vals))
    lo, hi = delta_chi2_interval(eff_grid, chi2_vals, 1.0)
    return FitResult(
        model="data-driven", best_eff=float(eff_grid[best]), eff_lo=lo, eff_hi=hi,
        chi2_min=float(chi2_vals[best]), ndof=n_bins - 1,
        eff_grid=eff_grid, chi2_eff=chi2_vals, best_profile=profiles[best],
        warnings=warnings, n_params=1, n_bins_fit=n_bins)


def fit_poisson(source: MultiplicityHist, eff_grid: np.ndarray, lam_grid: np.ndarray,
                engine: str = "analytic", n_throws: int = 1_000_000,
                rng: Optional[np.random.Generator] = None,
                likelihood: str = "thesis_chi2") -> FitResult:
    """2D scan over (eps_n, lambda_n) with an uncorrelated Poisson background (eq. 8.6)."""
    # Two intervals come back. After profiling over lambda, the conventional 68.3 %
    # interval on eps_n alone is delta-chi2 <= 1 -- that is eff_lo/eff_hi. The thesis
    # uses 2.30, the 68.3 % contour for TWO parameters, which is a joint region whose
    # projection onto the eps_n axis overcovers; it is carried separately as
    # eff_joint_lo/eff_joint_hi. See the FitResult docstring.
    # The chi2 surface stays a genuine 2D array sliced by integer index. Teal's driver
    # flattened it and recovered slices by float equality (np.where(x_var == best_eff),
    # phase2.py:235-240), which breaks outright if the minimum is degenerate.
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"likelihood must be one of {LIKELIHOODS}, got {likelihood!r}")
    rng = rng or np.random.default_rng()
    n_bins = len(source.counts)
    data, data_unc = source.normed, source.normed_unc

    if engine == "analytic":
        # The whole (n_eff, n_lam, n_bins) profile cube in one broadcast call.
        cube = analytic_profile_poisson(eff_grid[:, None], lam_grid[None, :], n_bins)
        if likelihood == "multinomial":
            surface = multinomial_deviance(cube, source.counts)
        else:
            surface = thesis_chi2(cube, data, data_unc)
        bi, bj = np.unravel_index(int(np.argmin(surface)), surface.shape)
        best_profile = cube[bi, bj]
    else:
        surface = np.empty((len(eff_grid), len(lam_grid)))
        profiles: dict[tuple[int, int], np.ndarray] = {}
        for i, eff in enumerate(eff_grid):
            for j, lam in enumerate(lam_grid):
                prof = mc_profile_poisson(eff, lam, n_bins, n_throws, rng)
                surface[i, j] = (multinomial_deviance(prof, source.counts)
                                 if likelihood == "multinomial"
                                 else thesis_chi2(prof, data, data_unc, None))
                profiles[(i, j)] = prof
        bi, bj = np.unravel_index(int(np.argmin(surface)), surface.shape)
        best_profile = profiles[(bi, bj)]
    # Profile: for each value of one parameter, minimise over the other.
    chi2_eff = surface.min(axis=1)
    chi2_lam = surface.min(axis=0)
    # delta-chi2 = 1 on the PROFILED curve: the conventional 68.3 % interval for one
    # parameter of interest. This is what gets quoted.
    eff_lo, eff_hi = delta_chi2_interval(eff_grid, chi2_eff, 1.0)
    lam_lo, lam_hi = delta_chi2_interval(lam_grid, chi2_lam, 1.0)
    # delta-chi2 = 2.30: the thesis convention, carried alongside so the two can be
    # compared rather than conflated.
    eff_jlo, eff_jhi = delta_chi2_interval(eff_grid, chi2_eff, 2.30)

    return FitResult(
        model="uncorrelated", best_eff=float(eff_grid[bi]),
        eff_lo=eff_lo, eff_hi=eff_hi,
        eff_joint_lo=eff_jlo, eff_joint_hi=eff_jhi,
        n_params=2, n_bins_fit=n_bins,
        chi2_min=float(surface[bi, bj]), ndof=n_bins - 2,
        eff_grid=eff_grid, chi2_eff=chi2_eff, best_profile=best_profile,
        best_lambda=float(lam_grid[bj]), lam_lo=lam_lo, lam_hi=lam_hi,
        lam_grid=lam_grid, chi2_lam=chi2_lam, chi2_surface=surface)


@dataclass
class SystBudget:
    """An ASYMMETRIC systematic budget (thesis section 8.7).

    The components do not all point the same way: the model difference has a direction
    set by which fit sits higher, while housing captures and background false-start
    triggers both dilute the measured efficiency, so correcting for either pushes it UP
    and is one-sided. The old code took |eps_DD - eps_Pois| and applied the quadrature
    total identically in both directions.
    """
    delta_model_signed: float           # background model (8.7.1), eps_alt - eps_nom
    delta_bt: Optional[float]           # background false starts (8.7.3)
    delta_bt_unc: Optional[float]
    delta_bt_basis: str                 # "rates" | "trigger counts (diagnostic)"
    delta_bt_publishable: bool
    delta_h: Optional[float]            # housing-capture CORRECTION (8.7.2)
    delta_h_unc: Optional[float]        # its uncertainty -- this is the systematic
    background_rate_hz: Optional[float] = None
    source_rate_hz: Optional[float] = None
    housing_n_captures: Optional[int] = None
    housing_n_simulated: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    @property
    def components(self) -> dict:
        """Every term, with its sign convention made explicit."""
        out = {
            "model": {"signed": self.delta_model_signed,
                      "plus": max(self.delta_model_signed, 0.0),
                      "minus": max(-self.delta_model_signed, 0.0)},
        }
        if self.delta_h_unc is not None:
            out["housing"] = {"correction": self.delta_h, "plus": self.delta_h_unc,
                              "minus": self.delta_h_unc,
                              "n_captures": self.housing_n_captures,
                              "n_simulated": self.housing_n_simulated,
                              "convention": "delta_h corrects the efficiency; "
                                            "delta_h_unc is the systematic"}
        if self.delta_bt is not None and self.delta_bt_publishable:
            out["background_triggers"] = {
                "contamination_fraction": self.delta_bt,
                "uncertainty": self.delta_bt_unc,
                "plus": self.delta_bt, "minus": 0.0,
                "background_rate_hz": self.background_rate_hz,
                "source_rate_hz": self.source_rate_hz,
                "basis": self.delta_bt_basis,
                "note": "one-sided: false-start triggers dilute the efficiency, so "
                        "the correction is upward"}
        return out

    @property
    def total_plus(self) -> float:
        parts = [max(self.delta_model_signed, 0.0)]
        if self.delta_h_unc is not None:
            parts.append(self.delta_h_unc)
        if self.delta_bt is not None and self.delta_bt_publishable:
            parts.append(self.delta_bt)
        return float(np.sqrt(np.sum(np.square(parts))))

    @property
    def total_minus(self) -> float:
        parts = [max(-self.delta_model_signed, 0.0)]
        if self.delta_h_unc is not None:
            parts.append(self.delta_h_unc)
        return float(np.sqrt(np.sum(np.square(parts))))

    def summary(self) -> str:
        h = "NOT EVALUATED" if self.delta_h_unc is None else \
            f"{self.delta_h:+.4f}+/-{self.delta_h_unc:.4f}"
        if self.delta_bt is None:
            bt = "NOT EVALUATED"
        elif not self.delta_bt_publishable:
            bt = f"{self.delta_bt:.4f} (DIAGNOSTIC ONLY, {self.delta_bt_basis})"
        else:
            bt = f"{self.delta_bt:.4f}+/-{self.delta_bt_unc:.4f}"
        return (f"delta_model={self.delta_model_signed:+.4f}  delta_bt={bt}  "
                f"delta_h={h}  total=+{self.total_plus:.4f}/-{self.total_minus:.4f}")


def systematics(eff_nominal: float, eff_alternative: float,
                n_trig_bkg: int, n_trig_src: int,
                livetime_bkg_s: Optional[float] = None,
                livetime_src_s: Optional[float] = None,
                livetime_bkg_unc_s: Optional[float] = None,
                livetime_src_unc_s: Optional[float] = None,
                delta_h: Optional[float] = None,
                delta_h_unc: Optional[float] = None,
                housing_n_captures: Optional[int] = None,
                housing_n_simulated: Optional[int] = None) -> SystBudget:
    """Assemble the signed, asymmetric systematic budget of thesis section 8.7."""
    # delta_model is eps_alternative - eps_nominal (8.7.1), SIGNED -- which model is
    # nominal is the caller's explicit choice, so the sign carries meaning.
    # delta_bt is f_B/f_S, the background-trigger contamination fraction (eq. 8.9),
    # defined on post-cut trigger RATES and so needing livetime. Without livetime the
    # count ratio is still reported but flagged NOT PUBLISHABLE and left out of the
    # total; it equals the rate ratio only when both runs had identical livetime.
    # delta_h is the housing-capture fraction (8.7.2). CONVENTION: delta_h is a
    # CORRECTION added to the efficiency and delta_h_unc is the systematic, so a bare
    # fraction with no uncertainty contributes nothing. Pass housing_n_captures and
    # housing_n_simulated to have the fraction and its Poisson error derived here.
    notes: list[str] = []
    delta_model_signed = eff_alternative - eff_nominal

    # --- housing -----------------------------------------------------------
    if housing_n_captures is not None and housing_n_simulated:
        delta_h = housing_n_captures / housing_n_simulated
        delta_h_unc = np.sqrt(max(housing_n_captures, 0)) / housing_n_simulated
    if delta_h is not None and delta_h_unc is None:
        notes.append(
            "delta_h was supplied without delta_h_unc. Under the correction "
            "convention the UNCERTAINTY is the systematic, so a bare fraction "
            "contributes nothing. Supply housing_n_captures/housing_n_simulated, or "
            "delta_h_unc directly.")
    if delta_h is None:
        notes.append("delta_h (housing captures, thesis 8.7.2) is NOT EVALUATED: no "
                     "MC housing-capture fraction has been supplied. The quoted total "
                     "systematic is therefore incomplete and is a LOWER BOUND.")

    # --- background false-start triggers -----------------------------------
    f_b = f_s = None
    delta_bt = delta_bt_unc = None
    publishable = False
    have_lt = livetime_bkg_s is not None and livetime_src_s is not None
    if have_lt and livetime_bkg_s > 0 and livetime_src_s > 0:
        f_b = n_trig_bkg / livetime_bkg_s
        f_s = n_trig_src / livetime_src_s
        if f_s > 0:
            delta_bt = f_b / f_s
            # Relative errors in quadrature: Poisson on both counts, plus livetime
            # uncertainties when supplied.
            rel = [1.0 / np.sqrt(n_trig_bkg) if n_trig_bkg > 0 else 0.0,
                   1.0 / np.sqrt(n_trig_src) if n_trig_src > 0 else 0.0]
            if livetime_bkg_unc_s:
                rel.append(livetime_bkg_unc_s / livetime_bkg_s)
            if livetime_src_unc_s:
                rel.append(livetime_src_unc_s / livetime_src_s)
            delta_bt_unc = delta_bt * float(np.sqrt(np.sum(np.square(rel))))
            publishable = True
            basis = "rates"
        else:
            basis = "rates (source rate is zero -- undefined)"
            notes.append("delta_bt is undefined: the source trigger rate is zero.")
    else:
        if have_lt:
            notes.append(
                "a livetime of zero was supplied; treating it as missing rather than "
                "dividing by it.")
        delta_bt = n_trig_bkg / n_trig_src if n_trig_src else None
        basis = ("trigger counts -- NOT the rate ratio of thesis eq. 8.9, valid only "
                 "if both runs had equal livetime")
        notes.append(
            "delta_bt is DIAGNOSTIC ONLY and is EXCLUDED from the systematic total: "
            "it was computed from trigger counts, not rates. Thesis eq. 8.9 uses "
            "f_B/f_S in Hz. Supply systematics.livetime_s per run in the config to "
            "get a publishable number.")

    return SystBudget(
        delta_model_signed=delta_model_signed,
        delta_bt=delta_bt, delta_bt_unc=delta_bt_unc, delta_bt_basis=basis,
        delta_bt_publishable=publishable,
        delta_h=delta_h, delta_h_unc=delta_h_unc,
        background_rate_hz=f_b, source_rate_hz=f_s,
        housing_n_captures=housing_n_captures,
        housing_n_simulated=housing_n_simulated,
        notes=notes)


def _resolve_event_index(ctx) -> pd.DataFrame:
    """Find the EventIndex parquets this deck produced."""
    import glob
    pats = ctx.inputs.get("event_index") or []
    if isinstance(pats, str):
        pats = [pats]
    paths: list[Path] = []
    for pat in pats:
        paths.extend(Path(p) for p in sorted(glob.glob(str(pat))))
    if not paths:
        raise SystemExit(
            f"[efficiency_fit] no EventIndex parquet matched {pats}.\n"
            f"This deck must be processed with stage1.dump_event_index: true first:\n"
            f"    ambe data process --config <config> --selection box\n"
            f"Without it the per-event cosmic flag does not exist and the fit cannot "
            f"be made self-consistent (see icscan.py:368-406).")
    return load_event_index(paths)


def run(ctx, argv=None):
    """`ambe stats efficiency --config <yaml>` -- thesis chapter 8 extraction."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="ambe stats efficiency")
    ap.add_argument("--engine", choices=["analytic", "mc"], default=None,
                    help="override fit_params.engine")
    ap.add_argument("--both-denominators", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="report ambe_triggers and total_events (default on); "
                         "--no-both-denominators restricts to fit_params.denominator")
    args, _ = ap.parse_known_args(argv or [])

    fp = dict(ctx.fit_params or {})
    if not fp:
        raise SystemExit("[efficiency_fit] config has no fit_params block")
    syst_cfg = dict((ctx.extra or {}).get("systematics") or {})

    src_run = int(fp["source_run"])
    bkg_run = int(fp["background_run"])
    xchk_run = fp.get("crosscheck_run")
    n_bins = int(fp.get("n_bins", 8))
    engine = args.engine or fp.get("engine", "analytic")
    # Validate here as well as in the fit functions. `choices` on the CLI only
    # guards the CLI; a typo in the config used to fall through the `else` branch
    # and silently run the slow, noisier toy-MC path.
    if engine not in ENGINES:
        raise SystemExit(
            f"[efficiency_fit] engine must be one of {ENGINES}, got {engine!r}. "
            f"A misspelling previously ran the toy-MC path silently.")
    n_throws = int(fp.get("n_throws", 1_000_000))
    rng = np.random.default_rng(int(fp.get("seed", 0)))

    overflow_policy = str(fp.get("overflow_policy", "fail"))
    likelihood = str(fp.get("likelihood", "thesis_chi2"))
    bkg_unc_mode = str(fp.get("background_uncertainty", "thesis_diagonal"))
    for val, allowed, name in ((overflow_policy, OVERFLOW_POLICIES, "overflow_policy"),
                               (likelihood, LIKELIHOODS, "likelihood"),
                               (bkg_unc_mode, BKG_UNC_MODES, "background_uncertainty")):
        if val not in allowed:
            raise SystemExit(
                f"[efficiency_fit] {name} must be one of {allowed}, got {val!r}")

    # WHICH MODEL SUPPLIES THE HEADLINE IS A REQUIRED, EXPLICIT CHOICE. There is no
    # default on purpose. Both background models are fit and reported either way;
    # promoting one of them is a physics judgement about whether the background
    # sample is representative, and it used to be made silently in a format string.
    nominal_model = fp.get("nominal_model")
    if nominal_model is None:
        raise SystemExit(
            "[efficiency_fit] fit_params.nominal_model is REQUIRED and has no "
            f"default. Set it to one of {NOMINAL_MODELS}.\n"
            "  poisson      narrow interval, but assumes the background is\n"
            "               uncorrelated in time -- check its p-value first.\n"
            "  data_driven  uses the measured background shape, but inherits the\n"
            "               background run's statistics; with a small background\n"
            "               sample its chi2/ndof looks good for the wrong reason\n"
            "               (see src/ambe/efficiency_extraction/datadriven_chi2.py).\n"
            "Neither is a safe default, which is why you have to say.")
    nominal_model = str(nominal_model)
    if nominal_model not in NOMINAL_MODELS:
        raise SystemExit(
            f"[efficiency_fit] nominal_model must be one of {NOMINAL_MODELS}, "
            f"got {nominal_model!r}")
    gof_min_p = float(fp.get("gof_min_p_value", 0.01))
    gof_override = fp.get("nominal_model_override_justification")
    n_gof_toys = int(fp.get("n_gof_toys", 2000))

    eff_grid = make_grid(fp["eff_grid"])
    lam_grid = make_grid(fp["lambda_grid"])
    acq_window = float(fp.get("acquisition_window_ns", 67000))
    # The fitter normalises the background rate over [signal_start, acq_window]
    # while selection decides what actually enters the histogram. If those two
    # disagree the efficiency is being quoted over a span it was not measured on,
    # so cross-check them rather than trusting a comment.
    cuts_cfg = dict(ctx.cuts or {})
    cut_tmax = cuts_cfg.get("time_max_ns")
    cut_tmin = cuts_cfg.get("time_min_ns")
    sig_start = float(fp.get("signal_window_start_ns", 2000))
    if cut_tmax is not None and float(cut_tmax) != acq_window:
        raise SystemExit(
            f"[efficiency_fit] window mismatch: cuts.time_max_ns={float(cut_tmax):g} "
            f"but fit_params.acquisition_window_ns={acq_window:g}. Selection and the "
            f"fit must use the same delayed window.")
    if cut_tmin is not None and float(cut_tmin) != sig_start:
        raise SystemExit(
            f"[efficiency_fit] window mismatch: cuts.time_min_ns={float(cut_tmin):g} "
            f"but fit_params.signal_window_start_ns={sig_start:g}.")
    scale = background_scale_factor(
        acq_window, sig_start, float(fp.get("bkg_window_start_ns", 2000)))

    ei = _resolve_event_index(ctx)

    print("\n" + "=" * 78)
    print("NEUTRON DETECTION EFFICIENCY -- multiplicity fit (thesis ch. 8.6)")
    print("=" * 78)
    print(f"source run {src_run}   background run {bkg_run}"
          + (f"   cross-check run {xchk_run}" if xchk_run else ""))
    print(f"engine={engine}  n_bins={n_bins}  bkg_scale={scale:.4f}  "
          f"seed={fp.get('seed')}")

    rows: list[dict] = []
    results: dict = {}

    denoms = (["ambe_triggers", "total_events"] if args.both_denominators
              else [fp.get("denominator", "ambe_triggers")])
    num_modes = ["noncosmic", "production"]

    for denom in denoms:
        for num_mode in num_modes:
            src = build_multiplicity(ei, src_run, n_bins, denom, num_mode,
                                     overflow_policy=overflow_policy)
            bkg = build_multiplicity(ei, bkg_run, n_bins, denom, num_mode,
                                     overflow_policy=overflow_policy)
            primary = (denom == fp.get("denominator", "ambe_triggers")
                       and num_mode == fp.get("numerator_mode", "noncosmic"))
            head = f"\n--- denominator={denom}  numerator={num_mode}" \
                   + ("   [PRIMARY]" if primary else "")
            print(head)
            print("   " + src.summary())
            print("   " + bkg.summary())
            for h in (src, bkg):
                if h.n_overflow:
                    print(f"   *** run {h.run}: {h.n_overflow:,} acquisitions exceed "
                          f"n_bins={n_bins} and were folded into bin {n_bins - 1} "
                          f"(overflow_policy='clip'). Read that bin as "
                          f"'>= {n_bins - 1}'. Raise n_bins to avoid merging "
                          f"physically different multiplicities. ***")

            rp = fit_poisson(src, eff_grid, lam_grid, engine=engine,
                             n_throws=n_throws, rng=rng, likelihood=likelihood)
            rd = fit_datadriven(src, bkg, eff_grid, engine=engine,
                                n_throws=n_throws, rng=rng,
                                bkg_scale=scale, likelihood=likelihood,
                                bkg_unc_mode=bkg_unc_mode,
                                n_bootstrap=int(fp.get("n_bootstrap", 400)))
            if primary:
                assert_interior(rp, "poisson")
                assert_interior(rd, "data-driven")
                # Goodness of fit from pseudo-experiments rather than the analytic
                # chi2 CDF -- the bins are normalised (hence correlated), the
                # empty-bin floor distorts the statistic, and fitted parameters do
                # not cost exactly one dof each.
                for r in (rp, rd):
                    r.p_value = pseudo_experiment_p_value(
                        r.best_profile, src.denominator, r.chi2_min,
                        _toy_unc, n_gof_toys, rng)
                    r.p_value_basis = (f"{n_gof_toys:,} multinomial pseudo-experiments "
                                       f"at the best-fit model")
            print("   " + rp.summary())
            print("   " + rd.summary())
            for w in rd.warnings:
                print(f"   [caveat] {w}")

            # The nominal model is the caller's explicit choice; the OTHER model is
            # the alternative whose distance from it is the model systematic.
            r_nom = rp if nominal_model == "poisson" else rd
            r_alt = rd if nominal_model == "poisson" else rp
            sb = systematics(
                r_nom.best_eff, r_alt.best_eff,
                n_trig_bkg=bkg.denominator, n_trig_src=src.denominator,
                livetime_bkg_s=(syst_cfg.get("livetime_s") or {}).get(bkg_run),
                livetime_src_s=(syst_cfg.get("livetime_s") or {}).get(src_run),
                livetime_bkg_unc_s=(syst_cfg.get("livetime_unc_s") or {}).get(bkg_run),
                livetime_src_unc_s=(syst_cfg.get("livetime_unc_s") or {}).get(src_run),
                delta_h=syst_cfg.get("delta_h_housing_fraction"),
                delta_h_unc=syst_cfg.get("delta_h_housing_fraction_unc"),
                housing_n_captures=syst_cfg.get("housing_n_captures"),
                housing_n_simulated=syst_cfg.get("housing_n_simulated"))
            print("   " + sb.summary())

            for r in (rp, rd):
                rows.append(dict(
                    denominator=denom, numerator_mode=num_mode, primary=primary,
                    model=r.model,
                    is_nominal=(r is r_nom),
                    source_run=src_run, background_run=bkg_run,
                    n_trig_source=src.denominator, n_trig_bkg=bkg.denominator,
                    naive_eff_source=src.naive_efficiency,
                    naive_eff_bkg=bkg.naive_efficiency,
                    best_eff=r.best_eff, eff_lo=r.eff_lo, eff_hi=r.eff_hi,
                    eff_err_lo=r.eff_err_lo, eff_err_hi=r.eff_err_hi,
                    eff_joint_lo=r.eff_joint_lo, eff_joint_hi=r.eff_joint_hi,
                    interval_type="profile_delta_chi2_1",
                    best_lambda=r.best_lambda, lam_lo=r.lam_lo, lam_hi=r.lam_hi,
                    chi2=r.chi2_min,
                    ndof=r.ndof, ndof_thesis=r.ndof,
                    ndof_standard=r.ndof_standard,
                    chi2_per_ndof=r.chi2_per_ndof,
                    chi2_per_ndof_standard=r.chi2_per_ndof_standard,
                    p_value=r.p_value,
                    likelihood=likelihood, background_uncertainty=bkg_unc_mode,
                    n_overflow_source=src.n_overflow, n_overflow_bkg=bkg.n_overflow,
                    delta_model_signed=sb.delta_model_signed,
                    delta_bt=sb.delta_bt, delta_bt_unc=sb.delta_bt_unc,
                    delta_bt_basis=sb.delta_bt_basis,
                    delta_bt_publishable=sb.delta_bt_publishable,
                    delta_h=sb.delta_h, delta_h_unc=sb.delta_h_unc,
                    syst_plus=sb.total_plus, syst_minus=sb.total_minus))
            if primary:
                results = dict(poisson=rp, datadriven=rd, syst=sb, src=src, bkg=bkg)

    # Cross-check run, primary convention only.
    if xchk_run:
        xr = int(xchk_run)
        try:
            xsrc = build_multiplicity(ei, xr, n_bins,
                                      fp.get("denominator", "ambe_triggers"),
                                      fp.get("numerator_mode", "noncosmic"),
                                      overflow_policy=overflow_policy)
            xbkg = build_multiplicity(ei, bkg_run, n_bins,
                                      fp.get("denominator", "ambe_triggers"),
                                      fp.get("numerator_mode", "noncosmic"),
                                      overflow_policy=overflow_policy)
            print(f"\n--- cross-check run {xr} (independent position)")
            print("   " + xsrc.summary())
            xp = fit_poisson(xsrc, eff_grid, lam_grid, engine=engine,
                             n_throws=n_throws, rng=rng, likelihood=likelihood)
            xd = fit_datadriven(xsrc, xbkg, eff_grid, engine=engine,
                                n_throws=n_throws, rng=rng, bkg_scale=scale,
                                likelihood=likelihood, bkg_unc_mode=bkg_unc_mode,
                                n_bootstrap=int(fp.get("n_bootstrap", 400)))
            print("   " + xp.summary())
            print("   " + xd.summary())
            x_nom = xp if nominal_model == "poisson" else xd
            x_alt = xd if nominal_model == "poisson" else xp
            for r in (xp, xd):
                rows.append(dict(
                    denominator=fp.get("denominator", "ambe_triggers"),
                    numerator_mode=fp.get("numerator_mode", "noncosmic"),
                    primary=False, model=f"{r.model} (crosscheck {xr})",
                    is_nominal=(r is x_nom),
                    source_run=xr, background_run=bkg_run,
                    n_trig_source=xsrc.denominator, n_trig_bkg=xbkg.denominator,
                    naive_eff_source=xsrc.naive_efficiency,
                    naive_eff_bkg=xbkg.naive_efficiency,
                    best_eff=r.best_eff, eff_lo=r.eff_lo, eff_hi=r.eff_hi,
                    eff_err_lo=r.eff_err_lo, eff_err_hi=r.eff_err_hi,
                    eff_joint_lo=r.eff_joint_lo, eff_joint_hi=r.eff_joint_hi,
                    interval_type="profile_delta_chi2_1",
                    best_lambda=r.best_lambda, lam_lo=r.lam_lo, lam_hi=r.lam_hi,
                    chi2=r.chi2_min,
                    ndof=r.ndof, ndof_thesis=r.ndof,
                    ndof_standard=r.ndof_standard,
                    chi2_per_ndof=r.chi2_per_ndof,
                    chi2_per_ndof_standard=r.chi2_per_ndof_standard,
                    p_value=r.p_value,
                    likelihood=likelihood, background_uncertainty=bkg_unc_mode,
                    n_overflow_source=xsrc.n_overflow, n_overflow_bkg=xbkg.n_overflow,
                    delta_model_signed=(x_alt.best_eff - x_nom.best_eff),
                    delta_bt=None, delta_bt_unc=None,
                    delta_bt_basis="n/a", delta_bt_publishable=False,
                    delta_h=None, delta_h_unc=None,
                    syst_plus=np.nan, syst_minus=np.nan))
        except SystemExit as e:
            print(f"   cross-check skipped: {e}")

    # ---------------- outputs ----------------
    df = pd.DataFrame(rows)
    out_csv = ctx.csv_path("efficiency_fit")
    df.to_csv(out_csv, index=False)
    print(f"\n[efficiency_fit] wrote {out_csv}")

    if results:
        rp, rd, sb = results["poisson"], results["datadriven"], results["syst"]
        src, bkg = results["src"], results["bkg"]
        r_nom = rp if nominal_model == "poisson" else rd
        r_alt = rd if nominal_model == "poisson" else rp

        # GOODNESS-OF-FIT GATE. A model whose p-value says it does not describe the
        # data does not get to supply the headline just because someone named it in
        # a config. The only way past is an explicit written justification, which is
        # then recorded in the output so the reader sees it too.
        gof_ok = r_nom.p_value is None or r_nom.p_value >= gof_min_p
        gof_warnings: list[str] = []
        if not gof_ok:
            msg = (f"nominal model '{nominal_model}' has p = {r_nom.p_value:.4f} "
                   f"< gof_min_p_value = {gof_min_p:g}: it does not describe the "
                   f"multiplicity distribution. chi2 = {r_nom.chi2_min:.1f} over "
                   f"ndof_thesis={r_nom.ndof} / "
                   f"ndof_standard={r_nom.ndof_standard}.")
            gof_warnings.append(msg)
            if not gof_override:
                raise SystemExit(
                    f"[efficiency_fit] REFUSING TO QUOTE A HEADLINE.\n  {msg}\n"
                    f"Both fits and every diagnostic were still written to "
                    f"{out_csv}.\nEither choose the other model, improve the "
                    f"background description, or -- if the poor fit is understood "
                    f"and accepted -- set\n"
                    f"    fit_params.nominal_model_override_justification: "
                    f"\"<why this is quotable anyway>\"\n"
                    f"which will be recorded in the JSON alongside the number.")
            gof_warnings.append(f"OVERRIDDEN BY CONFIG: {gof_override}")
            for w in gof_warnings:
                print(f"[efficiency_fit] *** {w} ***")

        # Housing captures are a CORRECTION (see systematics()); the raw fit value
        # and the corrected one are both reported so neither is hidden.
        eff_raw = r_nom.best_eff
        eff_corrected = eff_raw + (sb.delta_h or 0.0)
        quoted = (f"{eff_corrected:.3f} +{r_nom.eff_err_hi:.3f}(stat) "
                  f"+{sb.total_plus:.3f}(sys) / -{r_nom.eff_err_lo:.3f}(stat) "
                  f"-{sb.total_minus:.3f}(sys)")

        def _model_block(r: FitResult) -> dict:
            return dict(
                eff=r.best_eff, lo=r.eff_lo, hi=r.eff_hi,
                interval_type="profile_delta_chi2_1",
                joint_lo=r.eff_joint_lo, joint_hi=r.eff_joint_hi,
                lam=r.best_lambda, chi2=r.chi2_min,
                ndof_thesis=r.ndof, ndof_standard=r.ndof_standard,
                chi2_per_ndof_thesis=r.chi2_per_ndof,
                chi2_per_ndof_standard=r.chi2_per_ndof_standard,
                p_value=r.p_value, p_value_basis=r.p_value_basis,
                caveats=r.warnings)

        payload = {
            "window_ns": {"minimum": sig_start, "maximum": acq_window},
            "cuts": {
                "charge_balance_max": cuts_cfg.get("charge_balance_max"),
                "cluster_pe_max": cuts_cfg.get("pe_max"),
                "cluster_hits_min": cuts_cfg.get("cluster_hits_min"),
                "time_min_ns": cut_tmin, "time_max_ns": cut_tmax,
                "note": "dataset-specific adaptations of the thesis box, NOT a "
                        "reproduction of its thresholds",
            },
            "nominal_model": nominal_model,
            "nominal_model_override_justification": gof_override,
            "efficiency_raw": eff_raw,
            "efficiency_corrected": eff_corrected,
            "statistical_interval": {
                "type": "profile_delta_chi2_1",
                "delta_chi2": 1.0,
                "low": r_nom.eff_lo, "high": r_nom.eff_hi,
            },
            "joint_projection_interval": {
                "type": "joint_2_parameter_delta_chi2_2.30",
                "delta_chi2": 2.30,
                "low": r_nom.eff_joint_lo, "high": r_nom.eff_joint_hi,
                "note": "thesis convention; projecting a 2-parameter joint region "
                        "onto one axis OVERCOVERS for that parameter",
            },
            "systematics": {
                "plus": sb.total_plus, "minus": sb.total_minus,
                "components": sb.components,
                "notes": sb.notes,
            },
            "goodness_of_fit": {
                "chi2": r_nom.chi2_min,
                "ndof_definition": ("thesis: n_bins - n_fitted (does not subtract the "
                                    "normalisation); standard: n_bins - 1 - n_fitted"),
                "ndof_thesis": r_nom.ndof,
                "ndof_standard": r_nom.ndof_standard,
                "p_value": r_nom.p_value,
                "p_value_basis": r_nom.p_value_basis,
                "p_value_threshold": gof_min_p,
                "passed": gof_ok,
                "warnings": gof_warnings,
            },
            "models": {
                "uncorrelated": _model_block(rp),
                "data_driven": _model_block(rd),
            },
            "background_model_crosschecks": {
                "nominal": nominal_model,
                "alternative_efficiency": r_alt.best_eff,
                "delta_model_signed": sb.delta_model_signed,
                "note": "the model difference is the delta_m systematic; its SIGN is "
                        "kept so the budget can be asymmetric",
            },
            "input_provenance": {
                "source": dict(run=src.run, hist=list(src.counts.astype(int)),
                               total_events=src.total_events,
                               cosmic_events=src.cosmic_events,
                               denominator=src.denominator,
                               denominator_mode=src.denominator_mode,
                               numerator_mode=src.numerator_mode,
                               n_overflow=src.n_overflow,
                               provenance=src.provenance,
                               naive_efficiency=src.naive_efficiency),
                "background": dict(run=bkg.run, hist=list(bkg.counts.astype(int)),
                                   total_events=bkg.total_events,
                                   cosmic_events=bkg.cosmic_events,
                                   denominator=bkg.denominator,
                                   denominator_mode=bkg.denominator_mode,
                                   numerator_mode=bkg.numerator_mode,
                                   n_overflow=bkg.n_overflow,
                                   provenance=bkg.provenance,
                                   naive_efficiency=bkg.naive_efficiency),
                "likelihood": likelihood,
                "background_uncertainty": bkg_unc_mode,
                "overflow_policy": overflow_policy,
                "engine": engine,
                "background_scale_factor": scale,
            },
            "quoted": quoted,
            "config": {k: (v if not isinstance(v, np.ndarray) else list(v))
                       for k, v in fp.items()},
        }
        out_json = ctx.csv_path("efficiency_fit").with_suffix(".json")
        out_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[efficiency_fit] wrote {out_json}")

        print("\n" + "=" * 78)
        print(f"NOMINAL MODEL ({nominal_model}, chosen in the config)")
        print(f"HEADLINE  eps_n = {quoted}")
        if sb.delta_h:
            print(f"          raw {eff_raw:.4f}, corrected {eff_corrected:.4f} "
                  f"for housing captures ({sb.delta_h:+.4f})")
        print(f"          stat interval is the PROFILE delta-chi2=1 interval; the "
              f"thesis delta-chi2=2.30 projection is "
              f"[{r_nom.eff_joint_lo:.4f}, {r_nom.eff_joint_hi:.4f}]")
        print(f"          p = {r_nom.p_value}  ({r_nom.p_value_basis})")
        print(f"          alternative model ({r_alt.model}): {r_alt.best_eff:.4f}")
        print(f"          naive counting ratio for comparison: "
              f"{100 * src.naive_efficiency:.2f} %")
        print("=" * 78)
        for n in sb.notes:
            print(f"[note] {n}")

    return df


def cli(ctx, argv=None):
    run(ctx, argv)
