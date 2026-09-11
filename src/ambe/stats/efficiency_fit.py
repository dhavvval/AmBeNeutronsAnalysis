"""
Neutron detection efficiency by multiplicity fit -- Pershing thesis chapter 8.6.

WHY THIS MODULE EXISTS. The efficiency this repo quotes elsewhere is a naive
counting ratio, `unique_neutron_triggers / ambe_triggers` (data/icscan.py:17-18,
produced in data/processor.py:1294-1308). That ratio is biased whenever the delayed
window contains background candidates: an acquisition with multiplicity > 1 has at
most one correlated AmBe neutron, so every additional candidate is background, and
counting "has >= 1 candidate" folds that background straight into the efficiency.

The thesis instead FITS the per-acquisition multiplicity distribution. Each toy throw
is

    M = delta_S + N_B                                            (thesis eq. 8.2)

with delta_S ~ Bernoulli(eps_n) the one correlated AmBe neutron (eq. 8.3) and N_B the
background contribution, drawn two different ways:

    data-driven   N_B sampled from the measured background-run multiplicity
                  histogram. One free parameter, eps_n. chi2 carries the background
                  histogram's own statistical uncertainty (eq. 8.5).
    uncorrelated  N_B ~ Poisson(lambda_n), lambda_n free. Two free parameters.
                  chi2 uses the source uncertainty only (eq. 8.6).

eps_n is read off the chi2 profile minimum, with the 68.3 % interval taken as
delta-chi2 <= 1 for the 1-parameter fit and <= 2.30 for the 2-parameter fit
(eq. 8.7). The difference between the two models' best fits is the background-model
systematic delta_m (section 8.7.1).

THE ZERO BIN IS THE POINT. Acquisitions with NO candidate are the most constraining
bin in the fit -- they are what distinguishes a low efficiency from a high one -- and
they are the one bin that is NOT in the candidate CSVs, which only contain events that
produced at least one candidate. It has to come from the Stage-1 trigger count. Getting
that denominator wrong is the single easiest way to produce a confident wrong number,
which is why `build_multiplicity` takes an explicit denominator choice and records its
provenance rather than inferring one.

Usage:
    from ambe.stats.efficiency_fit import (
        load_event_index, build_multiplicity, fit_datadriven, fit_poisson)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln

# Poisson upper 68.3 % confidence limit for zero observed counts. Teal's convention
# for empty bins (phase2.py:131-132): an empty bin has sqrt(N) = 0, which would make
# its chi2 term infinite and let a single empty bin dictate the whole fit. Substituting
# 1.15 counts of uncertainty is the standard Feldman-Cousins-style floor.
EMPTY_BIN_UNC_COUNTS = 1.15

DENOMINATOR_CHOICES = ("ambe_triggers", "total_events")
NUMERATOR_MODES = ("noncosmic", "production")


# --------------------------------------------------------------------------- #
# Multiplicity histograms
# --------------------------------------------------------------------------- #
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
        """`unique_neutron_triggers / denominator` -- what this module replaces.
        Kept so every fit can be reported against the number it is correcting."""
        return self.n_with_candidate / self.denominator if self.denominator else np.nan

    @property
    def mean_multiplicity(self) -> float:
        n = self.counts.sum()
        return float(np.average(np.arange(len(self.counts)), weights=self.counts)) if n else np.nan

    @property
    def normed(self) -> np.ndarray:
        return self.counts / self.counts.sum()

    @property
    def normed_unc(self) -> np.ndarray:
        """sqrt(N)/N_tot per bin, with the empty-bin floor applied.

        Applied to BOTH source and background histograms -- see EMPTY_BIN_UNC_COUNTS.
        """
        n_tot = self.counts.sum()
        unc = np.sqrt(self.counts) / n_tot
        unc[self.counts == 0] = EMPTY_BIN_UNC_COUNTS / n_tot
        return unc

    def summary(self) -> str:
        return (f"run {self.run}: total={self.total_events:,} cosmic={self.cosmic_events:,} "
                f"ambe={self.ambe_triggers:,} denom={self.denominator:,} "
                f"({self.denominator_mode}/{self.numerator_mode}) "
                f"hist={list(self.counts.astype(int))} "
                f"naive_eff={self.naive_efficiency:.4f}")


def load_event_index(paths: Iterable[Path] | Path) -> pd.DataFrame:
    """Read one or more EventIndex parquets, one row per Stage-1-admitted event.

    Adds `n_accepted_clusters` if absent, matching icscan.py:193-195 so the two
    modules agree on what an accepted cluster count is.
    """
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
                       provenance: str = "EventIndex") -> MultiplicityHist:
    """Build the multiplicity histogram for one run, zero bin included.

    numerator_mode:
        noncosmic   an event contributes to the numerator only if it is NOT
                    cosmic-vetoed. Self-consistent with a denominator that excludes
                    cosmic events.
        production  reproduces the production accumulator, which counts a
                    cosmic-vetoed event in the numerator while subtracting it from
                    the denominator (icscan.py:368-406). Inconsistent, but it is what
                    the published numbers count, so it is reproducible on demand.

    On run 6062 the two differ by 2.8 percentage points, which is larger than the
    whole systematic budget of thesis table 8.6 -- hence both are offered and the
    caller is expected to report both.
    """
    if denominator not in DENOMINATOR_CHOICES:
        raise SystemExit(f"[efficiency_fit] denominator must be one of {DENOMINATOR_CHOICES}")
    if numerator_mode not in NUMERATOR_MODES:
        raise SystemExit(f"[efficiency_fit] numerator_mode must be one of {NUMERATOR_MODES}")

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

    counts = np.zeros(n_bins, dtype=float)
    # np.histogram(range=(0,n)) CLIPS the overflow into the last bin rather than
    # discarding it, which silently distorts the tail. Count overflow explicitly and
    # let the caller assert it is zero.
    n_overflow = int((mult >= n_bins).sum())
    for k in mult[mult < n_bins]:
        counts[k] += 1

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
                             cosmic_events: int = 0) -> MultiplicityHist:
    """Build a histogram from an explicit {multiplicity: n_events} map.

    For validation against published numbers (thesis figs 8.18/8.19) and for the
    closure tests, where there is no EventIndex to read.
    """
    counts = np.zeros(n_bins, dtype=float)
    for k, v in nonzero.items():
        if k <= 0:
            raise SystemExit("[efficiency_fit] nonzero map must not contain bin 0")
        if k < n_bins:
            counts[k] = v
    counts[0] = denominator_total - int(sum(nonzero.values()))
    if counts[0] < 0:
        raise SystemExit(f"[efficiency_fit] negative zero bin for run {run}")
    tot = total_events if total_events is not None else denominator_total + cosmic_events
    return MultiplicityHist(
        run=int(run), counts=counts, total_events=int(tot),
        cosmic_events=int(cosmic_events),
        denominator_mode="ambe_triggers" if cosmic_events else "total_events",
        numerator_mode="noncosmic", provenance=provenance)


# --------------------------------------------------------------------------- #
# Model profiles
# --------------------------------------------------------------------------- #
def background_scale_factor(acq_window_ns: float, sig_start_ns: float,
                            bkg_start_ns: float) -> float:
    """Rescale a background rate measured in one time window to another.

    thesis phase2.py:169. The background is flat in time, so a rate measured over
    [bkg_start, W] scales to [sig_start, W] by the livetime ratio. Equals 1 whenever
    the two windows start at the same place -- which is the case for the 6264/6265
    pair -- but is kept explicit so it cannot silently become wrong if they diverge.
    """
    num = acq_window_ns - sig_start_ns
    den = acq_window_ns - bkg_start_ns
    if den <= 0:
        raise SystemExit("[efficiency_fit] background window starts at or after the "
                         "end of the acquisition window")
    return num / den


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    """Poisson pmf, in log space so large k cannot overflow.

    Uses gammaln for log(k!) rather than summing logs per element: the scan calls
    this once per grid point, and a per-element Python loop here made a 2D scan
    ~30x slower than the arithmetic warrants.
    """
    k = np.asarray(k)
    if lam == 0:
        return (k == 0).astype(float)
    return np.exp(-lam + k * np.log(lam) - gammaln(k + 1.0))


def poisson_matrix(k: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Poisson pmf for every (lam, k) pair at once -> shape (len(lams), len(k)).

    The 2D scan evaluates the same Poisson pmf for every efficiency at a given
    lambda, so it is computed once per lambda and reused across the efficiency axis.
    """
    k = np.asarray(k, dtype=float)[None, :]
    lams = np.asarray(lams, dtype=float)[:, None]
    out = np.exp(-lams + k * np.log(np.where(lams > 0, lams, 1.0)) - gammaln(k + 1.0))
    return np.where(lams > 0, out, (k == 0).astype(float))


def analytic_profile_poisson(eff: float, lam: float, n_bins: int) -> np.ndarray:
    """Exact pmf of M = Bernoulli(eff) + Poisson(lam), binned like the data.

    P(M=k) = eff * Pois(k-1; lam) + (1-eff) * Pois(k; lam)

    This is not an approximation of the toy MC -- it is the same model evaluated in
    closed form. Using it removes MC sampling noise from the chi2 surface, which is
    what otherwise makes delta-chi2 contour extraction jitter between grid points.

    The last bin absorbs the overflow, matching np.histogram(range=(0, n_bins))'s
    CLIPPING behaviour in the MC path, so the two engines are comparable bin for bin.
    """
    k = np.arange(n_bins)
    p = (1.0 - eff) * _poisson_pmf(k, lam)
    p[1:] += eff * _poisson_pmf(k[:-1], lam)
    # Everything above the top bin clips into it.
    p[-1] += max(0.0, 1.0 - p.sum())
    return p / p.sum()


def mc_profile_poisson(eff: float, lam: float, n_bins: int, n_throws: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Toy-MC counterpart of `analytic_profile_poisson`. Teal's path (eq. 8.2-8.4)."""
    n = int(n_throws)
    sig = (rng.random(n) <= eff).astype(np.int64)
    bkg = rng.poisson(lam, n)
    tot = np.clip(sig + bkg, 0, n_bins - 1)
    return np.bincount(tot, minlength=n_bins)[:n_bins] / n


def mc_profile_datadriven(eff: float, bkg_p: np.ndarray, n_bins: int, n_throws: int,
                          rng: np.random.Generator,
                          bkg_unc: Optional[np.ndarray] = None) -> np.ndarray:
    """Toy MC with the background drawn from a measured histogram (eq. 8.2 + fig 8.19).

    `bkg_unc` fluctuates the TEMPLATE itself per throw-block. Teal does not do this --
    `BuildMCProfileBkgDist` accepts the uncertainty and ignores it, letting it enter
    only the chi2 denominator. Passing it here is opt-in, for the case where the
    background run's own statistics are thin enough that the template uncertainty
    matters as much as the source's.
    """
    n = int(n_throws)
    p = np.asarray(bkg_p, dtype=float)
    if bkg_unc is not None:
        p = np.clip(p + rng.normal(0.0, bkg_unc), 0.0, None)
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
    """Exact convolution of Bernoulli(eff) with the measured background histogram."""
    p = np.asarray(bkg_p, dtype=float)
    p = p / p.sum()
    out = np.zeros(n_bins)
    m = min(len(p), n_bins)
    out[:m] += (1.0 - eff) * p[:m]
    shifted = np.zeros(n_bins)
    shifted[1:min(len(p) + 1, n_bins)] = p[:min(len(p), n_bins - 1)]
    out += eff * shifted
    out[-1] += max(0.0, 1.0 - out.sum())
    return out / out.sum()


# --------------------------------------------------------------------------- #
# Chi-square and interval extraction
# --------------------------------------------------------------------------- #
def chi2(model: np.ndarray, data: np.ndarray, data_unc: np.ndarray,
         bkg_unc: Optional[np.ndarray] = None) -> float:
    """thesis eq. 8.5 (with bkg_unc) / eq. 8.6 (without).

    The data-driven model carries the background histogram's statistical uncertainty
    in the denominator because the background template is itself measured; the
    Poisson model does not, because its background is a fitted parameter rather than
    a measurement.
    """
    den = data_unc if bkg_unc is None else np.sqrt(data_unc ** 2 + bkg_unc ** 2)
    if np.any(den <= 0):
        raise SystemExit("[efficiency_fit] zero uncertainty in a chi2 denominator -- "
                         "the empty-bin floor was not applied")
    return float(np.sum(((model - data) / den) ** 2))


def delta_chi2_interval(grid: np.ndarray, chi2_vals: np.ndarray,
                        delta: float) -> tuple[float, float]:
    """Interval where chi2 - chi2_min <= delta (thesis eq. 8.7).

    delta = 1.0 for one fitted parameter, 2.30 for two, at 68.3 %.

    Edges are linearly interpolated between grid points rather than snapped to them,
    so the interval does not quantise to the scan step. A bound that runs off the end
    of the grid is returned as the grid edge; the caller is expected to notice and
    refuse to quote it (see `assert_interior`).
    """
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

    @property
    def chi2_per_ndof(self) -> float:
        return self.chi2_min / self.ndof if self.ndof > 0 else np.nan

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
    """Refuse to quote a best fit pinned to the edge of its scan grid.

    An edge-pinned minimum means the true minimum is outside the grid, so both the
    central value and the interval are meaningless -- but they still print as
    perfectly ordinary numbers, which is exactly why this has to be checked.
    """
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


# --------------------------------------------------------------------------- #
# The two fits
# --------------------------------------------------------------------------- #
def make_grid(spec: dict) -> np.ndarray:
    """Build a scan grid from a {min, max, step} config block, inclusive of max."""
    lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n)


def fit_datadriven(source: MultiplicityHist, bkg: MultiplicityHist,
                   eff_grid: np.ndarray, engine: str = "analytic",
                   n_throws: int = 1_000_000, rng: Optional[np.random.Generator] = None,
                   template_fluctuate: bool = False,
                   bkg_scale: float = 1.0) -> FitResult:
    """1D scan over eps_n with the background drawn from the measured histogram.

    thesis eq. 8.5. One fitted parameter, so the 68.3 % interval is delta-chi2 <= 1.

    `bkg_scale` rescales the background template's MEAN to the source's time window
    (see `background_scale_factor`). At 1.0 the template is used as measured.
    """
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
        bkg_p = bkg_p / bkg_p.sum()

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

    chi2_vals = np.empty(len(eff_grid))
    profiles = []
    for i, eff in enumerate(eff_grid):
        if engine == "analytic":
            prof = analytic_profile_datadriven(eff, bkg_p, n_bins)
        else:
            prof = mc_profile_datadriven(eff, bkg_p, n_bins, n_throws, rng,
                                         bkg_unc if template_fluctuate else None)
        profiles.append(prof)
        chi2_vals[i] = chi2(prof, data, data_unc, bkg_unc)

    best = int(np.argmin(chi2_vals))
    lo, hi = delta_chi2_interval(eff_grid, chi2_vals, 1.0)
    return FitResult(
        model="data-driven", best_eff=float(eff_grid[best]), eff_lo=lo, eff_hi=hi,
        chi2_min=float(chi2_vals[best]), ndof=n_bins - 1,
        eff_grid=eff_grid, chi2_eff=chi2_vals, best_profile=profiles[best],
        warnings=warnings)


def fit_poisson(source: MultiplicityHist, eff_grid: np.ndarray, lam_grid: np.ndarray,
                engine: str = "analytic", n_throws: int = 1_000_000,
                rng: Optional[np.random.Generator] = None) -> FitResult:
    """2D scan over (eps_n, lambda_n) with an uncorrelated Poisson background.

    thesis eq. 8.6. Two fitted parameters, so the 68.3 % interval on either is
    delta-chi2 <= 2.30 after profiling over the other (eq. 8.7).

    The chi2 surface is kept as a genuine 2D array and sliced by integer index. Teal's
    driver flattened it and recovered the slices with float equality
    (`np.where(x_var == best_eff)`, phase2.py:235-240), which is fragile and breaks
    outright if the minimum is degenerate.
    """
    rng = rng or np.random.default_rng()
    n_bins = len(source.counts)
    data, data_unc = source.normed, source.normed_unc

    if engine == "analytic":
        # Build the whole (eff, lam, bin) profile cube in one shot. The pmf for a
        # given lambda does not depend on the efficiency, so it is evaluated once
        # per lambda and combined across the efficiency axis by broadcasting --
        # a nested Python loop over ~30k grid points was the dominant cost.
        k = np.arange(n_bins)
        pois = poisson_matrix(k, lam_grid)                    # (n_lam, n_bins)
        shifted = np.zeros_like(pois)
        shifted[:, 1:] = pois[:, :-1]                         # Pois(k-1; lam)
        e = np.asarray(eff_grid)[:, None, None]
        cube = (1.0 - e) * pois[None, :, :] + e * shifted[None, :, :]
        # Match the clipping of np.histogram(range=(0, n_bins)): everything above
        # the top bin piles into it rather than vanishing.
        cube[..., -1] += np.clip(1.0 - cube.sum(axis=-1), 0.0, None)
        cube /= cube.sum(axis=-1, keepdims=True)
        den = data_unc
        surface = np.sum(((cube - data) / den) ** 2, axis=-1)
        bi, bj = np.unravel_index(int(np.argmin(surface)), surface.shape)
        best_profile = cube[bi, bj]
    else:
        surface = np.empty((len(eff_grid), len(lam_grid)))
        profiles: dict[tuple[int, int], np.ndarray] = {}
        for i, eff in enumerate(eff_grid):
            for j, lam in enumerate(lam_grid):
                prof = mc_profile_poisson(eff, lam, n_bins, n_throws, rng)
                surface[i, j] = chi2(prof, data, data_unc, None)
                profiles[(i, j)] = prof
        bi, bj = np.unravel_index(int(np.argmin(surface)), surface.shape)
        best_profile = profiles[(bi, bj)]
    # Profile: for each value of one parameter, minimise over the other.
    chi2_eff = surface.min(axis=1)
    chi2_lam = surface.min(axis=0)
    eff_lo, eff_hi = delta_chi2_interval(eff_grid, chi2_eff, 2.30)
    lam_lo, lam_hi = delta_chi2_interval(lam_grid, chi2_lam, 2.30)

    return FitResult(
        model="uncorrelated", best_eff=float(eff_grid[bi]),
        eff_lo=eff_lo, eff_hi=eff_hi,
        chi2_min=float(surface[bi, bj]), ndof=n_bins - 2,
        eff_grid=eff_grid, chi2_eff=chi2_eff, best_profile=best_profile,
        best_lambda=float(lam_grid[bj]), lam_lo=lam_lo, lam_hi=lam_hi,
        lam_grid=lam_grid, chi2_lam=chi2_lam, chi2_surface=surface)


# --------------------------------------------------------------------------- #
# Systematics (thesis section 8.7)
# --------------------------------------------------------------------------- #
@dataclass
class SystBudget:
    delta_m: float                      # background model (8.7.1)
    delta_bt: Optional[float]           # background false starts (8.7.3)
    delta_bt_basis: str                 # "rates" or "trigger counts"
    delta_h: Optional[float]            # housing captures (8.7.2)
    delta_h_unc: Optional[float]
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Quadrature sum of the evaluated components (thesis table 8.6)."""
        parts = [self.delta_m]
        if self.delta_bt is not None:
            parts.append(self.delta_bt)
        if self.delta_h is not None:
            parts.append(self.delta_h)
        return float(np.sqrt(np.sum(np.square(parts))))

    def summary(self) -> str:
        h = "NOT EVALUATED" if self.delta_h is None else f"{self.delta_h:.4f}"
        bt = "NOT EVALUATED" if self.delta_bt is None else f"{self.delta_bt:.4f}"
        return (f"delta_m={self.delta_m:.4f}  delta_bt={bt} ({self.delta_bt_basis})  "
                f"delta_h={h}  total={self.total:.4f}")


def systematics(eff_dd: float, eff_pois: float,
                n_trig_bkg: int, n_trig_src: int,
                livetime_bkg_s: Optional[float] = None,
                livetime_src_s: Optional[float] = None,
                delta_h: Optional[float] = None,
                delta_h_unc: Optional[float] = None) -> SystBudget:
    """Assemble the systematic budget.

    delta_m  |eps_DD - eps_Pois| (8.7.1). Always available once both models run.
    delta_bt f_B/f_S, the background-trigger contamination fraction (eq. 8.9). This
             is defined on post-cut trigger RATES, so it needs livetime. If livetime
             is not supplied it falls back to the trigger-count ratio and SAYS SO --
             passing counts off as rates would be wrong whenever the two runs have
             different livetimes, which is the normal case.
    delta_h  MC-derived housing-capture fraction (8.7.2). None means not evaluated;
             it is never silently treated as zero, which would make an incomplete
             budget look complete.
    """
    notes: list[str] = []
    delta_m = abs(eff_dd - eff_pois)

    if livetime_bkg_s and livetime_src_s:
        f_b = n_trig_bkg / livetime_bkg_s
        f_s = n_trig_src / livetime_src_s
        delta_bt = f_b / f_s if f_s else None
        basis = "rates"
    else:
        delta_bt = n_trig_bkg / n_trig_src if n_trig_src else None
        basis = "trigger counts (NO livetime supplied -- this is NOT the rate ratio "
        basis += "of thesis eq. 8.9 and is only valid if both runs had equal livetime)"
        notes.append(
            "delta_bt was computed from trigger counts, not rates. Thesis eq. 8.9 "
            "uses f_B/f_S in Hz. Supply systematics.livetime_s per run in the config "
            "to get the defined quantity.")

    if delta_h is None:
        notes.append("delta_h (housing captures, thesis 8.7.2) is NOT EVALUATED: no "
                     "MC housing-capture fraction has been supplied. The quoted total "
                     "systematic is therefore incomplete and is a LOWER BOUND.")

    return SystBudget(delta_m=delta_m, delta_bt=delta_bt, delta_bt_basis=basis,
                      delta_h=delta_h, delta_h_unc=delta_h_unc, notes=notes)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _resolve_event_index(ctx, tag: str) -> pd.DataFrame:
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


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def run(ctx, argv=None):
    """`ambe stats efficiency --config <yaml>` -- thesis chapter 8 extraction."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="ambe stats efficiency")
    ap.add_argument("--engine", choices=["analytic", "mc"], default=None,
                    help="override fit_params.engine")
    ap.add_argument("--both-denominators", action="store_true", default=True,
                    help="report ambe_triggers and total_events (default on)")
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
    n_throws = int(fp.get("n_throws", 1_000_000))
    rng = np.random.default_rng(int(fp.get("seed", 0)))

    eff_grid = make_grid(fp["eff_grid"])
    lam_grid = make_grid(fp["lambda_grid"])
    scale = background_scale_factor(
        float(fp.get("acquisition_window_ns", 67000)),
        float(fp.get("signal_window_start_ns", 2000)),
        float(fp.get("bkg_window_start_ns", 2000)))

    ei = _resolve_event_index(ctx, ctx.run_name)

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
            src = build_multiplicity(ei, src_run, n_bins, denom, num_mode)
            bkg = build_multiplicity(ei, bkg_run, n_bins, denom, num_mode)
            primary = (denom == fp.get("denominator", "ambe_triggers")
                       and num_mode == fp.get("numerator_mode", "noncosmic"))
            head = f"\n--- denominator={denom}  numerator={num_mode}" \
                   + ("   [PRIMARY]" if primary else "")
            print(head)
            print("   " + src.summary())
            print("   " + bkg.summary())
            for h in (src, bkg):
                if h.n_overflow:
                    print(f"   *** run {h.run}: {h.n_overflow} acquisitions exceed "
                          f"n_bins={n_bins} and were CLIPPED into the top bin. "
                          f"Raise n_bins. ***")

            rp = fit_poisson(src, eff_grid, lam_grid, engine=engine,
                             n_throws=n_throws, rng=rng)
            rd = fit_datadriven(src, bkg, eff_grid, engine=engine,
                                n_throws=n_throws, rng=rng,
                                template_fluctuate=bool(fp.get("template_fluctuate")),
                                bkg_scale=scale)
            if primary:
                assert_interior(rp, "poisson")
                assert_interior(rd, "data-driven")
            print("   " + rp.summary())
            print("   " + rd.summary())
            for w in rd.warnings:
                print(f"   [caveat] {w}")

            sb = systematics(
                rd.best_eff, rp.best_eff,
                n_trig_bkg=bkg.denominator, n_trig_src=src.denominator,
                livetime_bkg_s=(syst_cfg.get("livetime_s") or {}).get(bkg_run),
                livetime_src_s=(syst_cfg.get("livetime_s") or {}).get(src_run),
                delta_h=syst_cfg.get("delta_h_housing_fraction"),
                delta_h_unc=syst_cfg.get("delta_h_housing_fraction_unc"))
            print("   " + sb.summary())

            for r in (rp, rd):
                rows.append(dict(
                    denominator=denom, numerator_mode=num_mode, primary=primary,
                    model=r.model, source_run=src_run, background_run=bkg_run,
                    n_trig_source=src.denominator, n_trig_bkg=bkg.denominator,
                    naive_eff_source=src.naive_efficiency,
                    naive_eff_bkg=bkg.naive_efficiency,
                    best_eff=r.best_eff, eff_lo=r.eff_lo, eff_hi=r.eff_hi,
                    eff_err_lo=r.eff_err_lo, eff_err_hi=r.eff_err_hi,
                    best_lambda=r.best_lambda, lam_lo=r.lam_lo, lam_hi=r.lam_hi,
                    chi2=r.chi2_min, ndof=r.ndof, chi2_per_ndof=r.chi2_per_ndof,
                    delta_m=sb.delta_m, delta_bt=sb.delta_bt,
                    delta_bt_basis=sb.delta_bt_basis, delta_h=sb.delta_h,
                    syst_total=sb.total))
            if primary:
                results = dict(poisson=rp, datadriven=rd, syst=sb, src=src, bkg=bkg)

    # Cross-check run, primary convention only.
    if xchk_run:
        xr = int(xchk_run)
        try:
            xsrc = build_multiplicity(ei, xr, n_bins,
                                      fp.get("denominator", "ambe_triggers"),
                                      fp.get("numerator_mode", "noncosmic"))
            xbkg = build_multiplicity(ei, bkg_run, n_bins,
                                      fp.get("denominator", "ambe_triggers"),
                                      fp.get("numerator_mode", "noncosmic"))
            print(f"\n--- cross-check run {xr} (independent position)")
            print("   " + xsrc.summary())
            xp = fit_poisson(xsrc, eff_grid, lam_grid, engine=engine,
                             n_throws=n_throws, rng=rng)
            xd = fit_datadriven(xsrc, xbkg, eff_grid, engine=engine,
                                n_throws=n_throws, rng=rng, bkg_scale=scale)
            print("   " + xp.summary())
            print("   " + xd.summary())
            for r in (xp, xd):
                rows.append(dict(
                    denominator=fp.get("denominator", "ambe_triggers"),
                    numerator_mode=fp.get("numerator_mode", "noncosmic"),
                    primary=False, model=f"{r.model} (crosscheck {xr})",
                    source_run=xr, background_run=bkg_run,
                    n_trig_source=xsrc.denominator, n_trig_bkg=xbkg.denominator,
                    naive_eff_source=xsrc.naive_efficiency,
                    naive_eff_bkg=xbkg.naive_efficiency,
                    best_eff=r.best_eff, eff_lo=r.eff_lo, eff_hi=r.eff_hi,
                    eff_err_lo=r.eff_err_lo, eff_err_hi=r.eff_err_hi,
                    best_lambda=r.best_lambda, lam_lo=r.lam_lo, lam_hi=r.lam_hi,
                    chi2=r.chi2_min, ndof=r.ndof, chi2_per_ndof=r.chi2_per_ndof,
                    delta_m=abs(xd.best_eff - xp.best_eff), delta_bt=None,
                    delta_bt_basis="n/a", delta_h=None, syst_total=np.nan))
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
        payload = {
            "headline": {
                "model": "uncorrelated (Poisson) background -- PRIMARY",
                "efficiency": rp.best_eff,
                "stat_plus": rp.eff_err_hi, "stat_minus": rp.eff_err_lo,
                "syst_total": sb.total,
                "quoted": (f"{rp.best_eff:.3f} +{rp.eff_err_hi:.3f}(stat) "
                           f"+{sb.total:.3f}(sys) / -{rp.eff_err_lo:.3f}(stat) "
                           f"-{sb.total:.3f}(sys)"),
            },
            "models": {
                "uncorrelated": dict(eff=rp.best_eff, lo=rp.eff_lo, hi=rp.eff_hi,
                                     lam=rp.best_lambda, chi2=rp.chi2_min,
                                     ndof=rp.ndof),
                "data_driven": dict(eff=rd.best_eff, lo=rd.eff_lo, hi=rd.eff_hi,
                                    chi2=rd.chi2_min, ndof=rd.ndof,
                                    caveats=rd.warnings),
            },
            "systematics": dict(delta_m=sb.delta_m, delta_bt=sb.delta_bt,
                                delta_bt_basis=sb.delta_bt_basis,
                                delta_h=sb.delta_h, total=sb.total,
                                notes=sb.notes),
            "inputs": {
                "source": dict(run=src.run, hist=list(src.counts.astype(int)),
                               total_events=src.total_events,
                               cosmic_events=src.cosmic_events,
                               denominator=src.denominator,
                               naive_efficiency=src.naive_efficiency),
                "background": dict(run=bkg.run, hist=list(bkg.counts.astype(int)),
                                   total_events=bkg.total_events,
                                   cosmic_events=bkg.cosmic_events,
                                   denominator=bkg.denominator,
                                   naive_efficiency=bkg.naive_efficiency),
            },
            "config": {k: (v if not isinstance(v, np.ndarray) else list(v))
                       for k, v in fp.items()},
        }
        out_json = ctx.csv_path("efficiency_fit").with_suffix(".json")
        out_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[efficiency_fit] wrote {out_json}")

        print("\n" + "=" * 78)
        print(f"HEADLINE  eps_n = {payload['headline']['quoted']}")
        print(f"          naive counting ratio for comparison: "
              f"{_fmt_pct(src.naive_efficiency)}")
        print("=" * 78)
        for n in sb.notes:
            print(f"[note] {n}")

    return df


def cli(ctx, argv=None):
    run(ctx, argv)
