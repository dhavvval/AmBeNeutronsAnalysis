#!/usr/bin/env python
"""
Capture-time comparison: IC-only baseline vs the fprompt2d widened window,
and — the decisive test — the *recovered-only* subset (events the widened
window gained and the tight IC window never saw).

Why this script exists instead of `AmBeNeutronAnalyzer`:
the analyzer fits per source position and cannot be pointed at a subset of
events within a tag. The recovered-only sample is defined by an eventID diff
between two tags, so it has to be built here. The histogram/mask/model are
replicated *exactly* from `ambe.plots.basic` so the numbers stay comparable
to the published tau = 29.16 +/- 0.44 us:

  - 70 bins over (0, 70) us, fit mask 2 < bin_centre < 67  (WAS 10 < ... < 67)
  - NeutCapture(t) = A*(1-exp(-t/therm))*exp(-t/tau) + B
  - bounds A (0, inf), therm (0.1, 10), tau (10, 70), B (0, 15)
  - weights 1/sqrt(N)
  - no PE / charge-balance cuts (lines 146-147 of basic.py are commented out)

Two fit variants are run for every sample:

  Bfix  B held at 0, `vary=False` -- what `lmfit_analysis` actually does, so
        this is the apples-to-apples number against 29.16 +/- 0.44 us.
  Bfree B floated in (0, 15) -- the diagnostic. With B pinned at zero a flat
        accidental pedestal has nowhere to go and inflates tau instead;
        floating it measures the pedestal directly.

Discriminator: if the recovered sample is genuine AmBe signal whose neutron
simply failed to reconstruct in the baseline, tau_recovered must sit on
29.16 +/- 0.44 us. If it is accidental coincidence, tau_recovered runs high
(Bfix) and/or B_recovered/A_recovered comes out large (Bfree).

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    MPLBACKEND=Agg python -u fit_capture_time_fprompt_compare.py
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import lmfit

# Fit backends. Both minimise the same chi2 of the same NeutCapture model over the
# same 10-67 us window with the same bounds, so they are meant to agree; they are
# kept as an explicit choice rather than one being swapped in because the published
# tau = 30.53 +- 0.26 us anchor was measured with "scipy", and `ambe.plots.basic`
# uses lmfit. Every caller states which one it wants -- see fit().
BACKENDS = ("scipy", "lmfit")

CAND_DIR = "EventAmBeNeutronCandidatesData"
BASE_TAG = "AmBe2.0v4_gated"
FP_TAG = "AmBe2.0v4_fprompt"
RUNS = [6046, 6056, 6060, 6061, 6062, 6165, 6166, 6186, 6187, 6188, 6189,
        6230, 6231, 6232, 6234, 6235, 6237, 6239, 6241, 6242]

# --- replicated verbatim from ambe/plots/basic.py fitting_config ---
TIME_BINS = 70
TIME_RANGE = (0, 70)
# THE FIT WINDOW WAS 10-67 us AND IS NOW 2-67 us. Changed deliberately, campaign-wide,
# so that one window is used by every capture-time product in the analysis rather than
# the fit window and the selection's own lower bound disagreeing.
#
# 2 us is where the data actually starts: the box cut is t >= 2 us and the cosmic veto
# drops any event containing a cluster below 2 us, so no selection admits anything
# earlier. Fitting from 10 threw away the eight microseconds that contain the
# thermalisation rise -- which is the only part of the range where the
# (1-exp(-t/therm)) term carries information, so therm was being constrained almost
# entirely by its bounds.
#
# EVERY PUBLISHED CAPTURE TIME MOVES BECAUSE OF THIS. The old anchor, tau = 30.53 +-
# 0.26 us over 19 positions and 30.535 +- 0.228 over 26, was measured on 10-67. It is
# superseded, not reproduced. Anything quoting 30.53 predates this change.
FIT_MIN, FIT_MAX = 2.0, 67.0
INIT = dict(A=200.0, therm=5.0, tau=25.0, B=0.0)
BOUNDS_LO = dict(A=0.0, therm=0.1, tau=10.0, B=0.0)
BOUNDS_HI = dict(A=np.inf, therm=10.0, tau=70.0, B=15.0)

USECOLS = ["eventID", "clusterTime", "sourceX", "sourceY", "sourceZ"]


def NeutCapture(t, A, therm, tau, B):
    return A * (1 - np.exp(-t / therm)) * np.exp(-t / tau) + B


# fit_expflat keeps a 10 us start and does NOT follow FIT_MIN. It is a DIFFERENT
# MODEL -- A*exp(-t/tau) + B, with no rise term at all -- and dropping the rise term
# is only legitimate above ~10 us where the rise is >=86% saturated. Fitting it from
# 2 us asks a pure exponential to describe the thermalisation rise it has no
# parameter for: measured on the 28-run sample that gives chi2/ndof 165 against 1.4,
# and a tau of 46 us against 29.5. That is not a weak fit, it is the wrong model on
# that range, so it would report a meaningless flat-pedestal limit. The pedestal
# diagnostic therefore stays on 10-67 and says so wherever it is quoted.
EXPFLAT_MIN = 10.0


def prepare(ct_us, fit_min=None):
    """Exact replica of AmBeNeutronAnalyzer._prepare_fitting_data.

    `fit_min` overrides the module FIT_MIN for callers that need a different lower
    edge -- only fit_expflat does, see EXPFLAT_MIN.
    """
    lo = FIT_MIN if fit_min is None else fit_min
    counts, edges = np.histogram(ct_us, bins=TIME_BINS, range=TIME_RANGE)
    centres = (edges[:-1] + edges[1:]) / 2
    mask = (centres > lo) & (centres < FIT_MAX)
    x, y = centres[mask], counts[mask]
    err = np.sqrt(y).astype(float)
    err[err == 0] = 1e-10
    return x, y, err


def fit(ct_us, float_B, backend):
    """NeutCapture fit on the FIT_MIN-FIT_MAX window (2-67 us). `backend` is REQUIRED.

    No default on `backend` on purpose: "scipy" is the recipe the frozen
    tau = 30.53 +- 0.26 us anchor was measured with, "lmfit" is the recipe
    `ambe.plots.basic.lmfit_analysis` uses, and a caller that does not say which
    one it wants would silently attribute one recipe's number to the other. Both
    return the SAME dict schema, so a caller only has to name the backend.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "lmfit":
        return _fit_lmfit(ct_us, float_B)
    x, y, err = prepare(ct_us)
    if y.sum() < 50:
        return None
    if float_B:
        p0 = [INIT["A"], INIT["therm"], INIT["tau"], 1.0]
        lo = [BOUNDS_LO[k] for k in ("A", "therm", "tau", "B")]
        hi = [BOUNDS_HI[k] for k in ("A", "therm", "tau", "B")]
        f = NeutCapture
    else:
        p0 = [INIT["A"], INIT["therm"], INIT["tau"]]
        lo = [BOUNDS_LO[k] for k in ("A", "therm", "tau")]
        hi = [BOUNDS_HI[k] for k in ("A", "therm", "tau")]
        def f(t, A, therm, tau):
            return NeutCapture(t, A, therm, tau, 0.0)
    try:
        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=err,
                               absolute_sigma=True, bounds=(lo, hi),
                               maxfev=200000)
    except Exception as exc:                                # noqa: BLE001
        print(f"    fit failed: {exc}")
        return None
    perr = np.sqrt(np.diag(pcov))
    resid = (y - f(x, *popt)) / err
    ndof = len(x) - len(popt)
    out = dict(A=popt[0], A_err=perr[0], therm=popt[1], therm_err=perr[1],
               tau=popt[2], tau_err=perr[2],
               chi2=float((resid ** 2).sum()), ndof=ndof,
               redchi=float((resid ** 2).sum() / ndof), N=int(y.sum()))
    out["B"], out["B_err"] = (popt[3], perr[3]) if float_B else (0.0, 0.0)
    return out


def _fit_lmfit(ct_us, float_B):
    """The lmfit backend of fit(). Same model, window, bounds and weights.

    Follows `ambe.plots.basic.AmBeNeutronAnalyzer.lmfit_analysis`:
    lmfit.Model(NeutCapture), the same initial values and min/max on therm and
    tau, and B held with vary=False when float_B is False (which is what that
    method does, and what the 30.53 us anchor was measured with).

    Weighting: lmfit minimises `(data - model) * weights`, so weights = 1/err
    reproduces curve_fit's `sigma=err, absolute_sigma=True` chi2 exactly. With
    `scale_covar=False` the reported stderr is then on the same footing as
    curve_fit's absolute_sigma errors rather than rescaled by sqrt(redchi) --
    without it the two backends' errors would differ by that factor alone and the
    comparison against the anchor error would be meaningless.

    ONE DELIBERATE DEVIATION FROM lmfit_analysis, AND WHY. method="least_squares"
    rather than lmfit's default "leastsq". Default leastsq is MINPACK
    Levenberg-Marquardt, which imposes min/max by transforming the parameters
    internally; on the high-statistics positions that transform walks into a worse
    local minimum and stops there. Measured on the 28-run box-cut sample, position
    (0,-100,-75): leastsq returns therm = 0.33 us (all but pinned at the 0.1 bound)
    and tau = 33.21 with NO error estimate, while the same data under
    "least_squares" -- which is scipy.optimize.least_squares, the bounded TRF
    solver curve_fit itself uses -- returns therm = 4.90 +- 0.58, tau =
    31.65 +- 0.64, chi2/ndof 2.17. 2 of 26 positions failed outright that way. With
    "least_squares" this backend reproduces the scipy backend to every digit it
    prints, position by position, which is the point: the choice of library must not
    be a physics choice.

    Returns None on the same conditions as the scipy path (<50 counts in window,
    or the minimiser not converging), so callers need no extra branch.
    """
    x, y, err = prepare(ct_us)
    if y.sum() < 50:
        return None
    model = lmfit.Model(NeutCapture)
    params = model.make_params(A=INIT["A"], therm=INIT["therm"], tau=INIT["tau"],
                               B=1.0 if float_B else 0.0)
    for k in ("A", "therm", "tau", "B"):
        params[k].min = BOUNDS_LO[k]
        params[k].max = BOUNDS_HI[k]
    if not float_B:
        params["B"].value = 0.0
        params["B"].vary = False
    try:
        res = model.fit(y.astype(float), params, t=x, weights=1.0 / err,
                        scale_covar=False, nan_policy="omit",
                        method="least_squares")
    except Exception as exc:                                # noqa: BLE001
        print(f"    lmfit fit failed: {exc}")
        return None
    if not res.success or res.covar is None:
        print("    lmfit fit did not converge / no covariance")
        return None

    def val(name):
        p = res.params[name]
        # stderr is None for a parameter lmfit could not error-estimate, and 0.0
        # for one sitting on a bound. is_good() already rejects err == 0, so
        # mapping None onto 0.0 routes both into the same existing quality gate
        # instead of putting a NaN into the weighted average.
        return float(p.value), float(p.stderr) if p.stderr is not None else 0.0

    A, A_err = val("A")
    therm, therm_err = val("therm")
    tau, tau_err = val("tau")
    B, B_err = val("B")
    ndof = len(x) - int(res.nvarys)
    chi2 = float(res.chisqr)
    return dict(A=A, A_err=A_err, therm=therm, therm_err=therm_err,
                tau=tau, tau_err=tau_err, chi2=chi2, ndof=ndof,
                redchi=chi2 / ndof, N=int(y.sum()), B=B, B_err=B_err)


def fit_expflat(ct_us):
    """A*exp(-t/tau) + B on EXPFLAT_MIN-67 us (10-67), B free.

    NOTE the window: this one function deliberately does NOT follow FIT_MIN. See the
    comment on EXPFLAT_MIN -- a model with no rise term cannot be fitted across the
    rise. Any number from here must be quoted as a 10-67 us result even though every
    NeutCapture fit in the analysis is now 2-67.

    Why this variant exists: above 10 us the `(1-exp(-t/therm))` rise term of
    NeutCapture is already >=86% saturated for any therm in its (0.1, 10) box,
    so `therm` is barely identifiable and trades off against `tau`. On small
    samples that degeneracy is what makes the fit collapse to a bound (seen at
    2/19 positions in the recovered subset, where therm pinned at 10.0 and tau
    at 10.0 with chi2/ndof ~ 17-20, even though those histograms fall perfectly
    normally). Dropping the unidentifiable rise term and floating the flat
    pedestal instead gives a well-conditioned 3-parameter fit that answers the
    actual question: is there an excess flat (accidental) component?
    """
    x, y, err = prepare(ct_us, fit_min=EXPFLAT_MIN)
    if y.sum() < 50:
        return None
    f = lambda t, A, tau, B: A * np.exp(-t / tau) + B      # noqa: E731
    try:
        popt, pcov = curve_fit(f, x, y, p0=[y[0] * 1.5, 30.0, y[-1] * 0.5],
                               sigma=err, absolute_sigma=True,
                               bounds=([0, 5.0, 0], [np.inf, 200.0, np.inf]),
                               maxfev=200000)
    except Exception as exc:                                # noqa: BLE001
        print(f"    expflat fit failed: {exc}")
        return None
    perr = np.sqrt(np.diag(pcov))
    resid = (y - f(x, *popt)) / err
    ndof = len(x) - 3
    # pedestal fraction: flat counts / total counts inside the fit window
    flat_frac = popt[2] * len(x) / max(y.sum(), 1)
    return dict(A=popt[0], A_err=perr[0], tau=popt[1], tau_err=perr[1],
                B=popt[2], B_err=perr[2], flat_frac=flat_frac,
                chi2=float((resid ** 2).sum()), ndof=ndof,
                redchi=float((resid ** 2).sum() / ndof), N=int(y.sum()))


def is_good(t):
    """Drop fits that did not converge: parameter pinned at a bound (which
    makes curve_fit report stderr exactly 0 and gives it infinite weight),
    or a chi2/ndof that says the model does not describe the data.

    Module-level because both --mode fpcompare and --mode tagset apply the same
    quality gate; the weighted average is only ever taken over fits passing it.
    """
    return ((t.tau_err > 0) & (t.therm_err > 0)
            & (t.tau > BOUNDS_LO["tau"] + 1e-6) & (t.tau < BOUNDS_HI["tau"] - 1e-6)
            & (t.therm > BOUNDS_LO["therm"] + 1e-6)
            & (t.redchi < 3.0))


def load(tag, run):
    p = os.path.join(CAND_DIR, f"EventAmBeNeutronCandidates_{tag}_{run}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=USECOLS)
    d["clusterTime"] = d["clusterTime"] / 1000.0        # ns -> us, as basic.py:143
    d["run"] = run
    return d


def load_tag(tag):
    """Every per-run candidate CSV carrying `tag`, discovered from the filesystem.

    Unlike load(), this does not need a hardcoded run list — which is the point:
    the tagset mode is meant to be pointed at a campaign without editing the
    module. Runs are parsed out of the filename so the returned frame keeps the
    `run` column that the pooled fits ignore but the printout uses.
    """
    pat = os.path.join(CAND_DIR, f"EventAmBeNeutronCandidates_{tag}_*.csv")
    frames = []
    for p in sorted(glob.glob(pat)):
        m = re.search(rf"EventAmBeNeutronCandidates_{re.escape(tag)}_(\d+)\.csv$", p)
        if not m:
            continue
        d = load(tag, int(m.group(1)))
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        raise SystemExit(f"[tagset] no candidate CSVs matched {pat} — refusing to "
                         f"fit an empty sample")
    out = pd.concat(frames, ignore_index=True)
    print(f"  tag {tag:24s}: {out.run.nunique():3d} runs, {len(out):8,d} clusters, "
          f"{out.groupby(['sourceX','sourceY','sourceZ']).ngroups:3d} positions")
    return out


def fit_per_position(d, backend):
    """Per-(sourceX,sourceY,sourceZ) NeutCapture fit, B fixed at 0.

    This is the recipe behind the published weighted-average tau: same 70-bin
    histogram, same 10-67 us window, same bounds, B held at zero. Returned frame
    matches the CaptureTimeFits_*.csv schema. `backend` is passed straight down to
    fit() and is required for the same reason it is required there.
    """
    recs = []
    for pos, grp in d.groupby(["sourceX", "sourceY", "sourceZ"]):
        r = fit(grp.clusterTime.values, float_B=False, backend=backend)
        if r is None:
            print(f"    position {pos}: fit failed / <50 counts in window")
            continue
        r["pos"] = pos
        recs.append(r)
    return pd.DataFrame(recs)


def weighted_tau(g):
    """Inverse-variance weighted mean tau and its error, over converged fits."""
    w = 1.0 / g.tau_err ** 2
    wavg = (g.tau * w).sum() / w.sum()
    werr = float(np.sqrt(1.0 / w.sum()))
    scatter = (w * (g.tau - wavg) ** 2).sum() / (len(g) - 1) if len(g) > 1 else 0.0
    return float(wavg), werr, float(scatter)


def mode_tagset(tags, out_label, backend):
    """Pool the named tags, fit per source position, write one CaptureTimeFits CSV.

    Why this mode exists: the fpcompare mode below is hardwired to the
    baseline-vs-fprompt question (two fixed tags, a fixed 20-run list, and an
    eventID diff between them). Measuring the capture time of an arbitrary
    campaign — e.g. all 28 v4 source runs spread over two Stage-1 tags — needs the
    same fit recipe pointed at a different sample, not a second copy of it.
    """
    print("=" * 78)
    print(f"TAGSET MODE — tags: {', '.join(tags)}   fit backend: {backend}")
    print("=" * 78)
    frames = [load_tag(t) for t in tags]
    d = pd.concat(frames, ignore_index=True)

    key = ["sourceX", "sourceY", "sourceZ"]
    npos = d.groupby(key).ngroups
    print(f"\n  POOLED: {d.run.nunique()} runs, {len(d):,} clusters, {npos} positions")
    dup = d.groupby(key)["run"].nunique()
    shared = dup[dup > 1]
    if len(shared):
        print(f"  note: {len(shared)} position(s) contributed by >1 run "
              f"(expected where two runs sit at the same place):")
        for pos, n in shared.items():
            runs = sorted(d[(d.sourceX == pos[0]) & (d.sourceY == pos[1])
                            & (d.sourceZ == pos[2])].run.unique())
            print(f"    {pos}: {n} runs {runs}")

    print(f"\n  Pooled fit (all positions together), B fixed at 0, {backend}:")
    rp = fit(d.clusterTime.values, float_B=False, backend=backend)
    if rp:
        print(f"    tau = {rp['tau']:.3f} +- {rp['tau_err']:.3f} us   "
              f"therm = {rp['therm']:.2f}   chi2/ndof = {rp['redchi']:.2f}   N = {rp['N']:,}")

    print(f"\n  PER-POSITION FITS, B fixed at 0, {backend}:")
    t = fit_per_position(d, backend)
    if t.empty:
        raise SystemExit("[tagset] no position produced a fit — nothing to write")
    g = t[is_good(t)]
    print(f"    positions fitted : {len(t)}  ({len(g)} converged, {len(t)-len(g)} rejected)")
    for _, r in t[~is_good(t)].iterrows():
        print(f"      REJECTED {str(r['pos']):16s} tau={r['tau']:.2f}+-{r['tau_err']:.2f} "
              f"therm={r['therm']:.2f} chi2/ndof={r['redchi']:.2f} N={r['N']}")
    if not g.empty:
        wavg, werr, scatter = weighted_tau(g)
        print(f"\n    weighted-avg tau  : {wavg:.3f} +- {werr:.3f} us  (converged only)")
        print(f"    scale-factor err  : +- {werr*np.sqrt(max(scatter,1.0)):.3f} us  "
              f"(chi2/ndof of the average = {scatter:.2f})")
        print(f"    unweighted mean   : {g.tau.mean():.3f} +- "
              f"{g.tau.std(ddof=1)/np.sqrt(len(g)):.3f} us")
        print(f"    tau range         : {g.tau.min():.2f} - {g.tau.max():.2f} us")
        print(f"    chi2/ndof range   : {t.redchi.min():.2f} - {t.redchi.max():.2f}")

    p = os.path.join("TriggerSummary", f"CaptureTimeFits_{out_label}.csv")
    if os.path.exists(p):
        print(f"\n  NOTE: overwriting existing {p}")
    t.to_csv(p, index=False)
    print(f"\nwrote {p}  ({len(t)} rows)")
    return 0


def mode_fpcompare(backend):
    base, fp, rec = [], [], []
    print("Building samples (recovered = fprompt eventIDs absent from gated, per run)")
    for r in RUNS:
        dg, df = load(BASE_TAG, r), load(FP_TAG, r)
        if dg is None or df is None:
            print(f"  run {r}: MISSING a tag, skipped")
            continue
        gained = df[~df.eventID.isin(set(dg.eventID))]
        base.append(dg)
        fp.append(df)
        rec.append(gained)
    base = pd.concat(base, ignore_index=True)
    fp = pd.concat(fp, ignore_index=True)
    rec = pd.concat(rec, ignore_index=True)
    print(f"  baseline  : {len(base):7d} clusters, {base.eventID.nunique():7d} uniq eventID/run-pooled")
    print(f"  fprompt2d : {len(fp):7d} clusters")
    print(f"  recovered : {len(rec):7d} clusters  ({len(rec)/len(fp):.1%} of fprompt2d)")

    samples = [("baseline IC 700-1200", base),
               ("fprompt2d 500-2000 x 0.15-0.30", fp),
               ("RECOVERED only (fprompt2d - baseline)", rec)]

    for float_B, label in ((False, "B FIXED at 0  (replicates lmfit_analysis)"),
                           (True, "B FLOATING in (0,15)  (measures pedestal)")):
        print("\n" + "=" * 78)
        print(f"POOLED FIT, all positions together --- {label}")
        print("=" * 78)
        hdr = f"{'sample':38s} {'N':>8s} {'tau (us)':>16s} {'therm (us)':>14s} {'B':>13s} {'chi2/ndof':>10s}"
        print(hdr)
        for name, d in samples:
            r = fit(d.clusterTime.values, float_B, backend=backend)
            if r is None:
                print(f"{name:38s}   fit failed / too few counts")
                continue
            print(f"{name:38s} {r['N']:8d} {r['tau']:8.2f}+-{r['tau_err']:<6.2f} "
                  f"{r['therm']:6.2f}+-{r['therm_err']:<6.2f} "
                  f"{r['B']:6.2f}+-{r['B_err']:<5.2f} {r['redchi']:10.2f}")

    # per-position, B fixed -- reproduces the published weighted-average recipe
    print("\n" + "=" * 78)
    print("PER-POSITION FITS, B fixed at 0 (the recipe behind tau = 29.16 +- 0.44)")
    print("=" * 78)
    key = ["sourceX", "sourceY", "sourceZ"]
    per = {}
    for name, d in samples:
        recs = []
        for pos, grp in d.groupby(key):
            r = fit(grp.clusterTime.values, float_B=False, backend=backend)
            if r is None:
                continue
            r["pos"] = pos
            recs.append(r)
        per[name] = pd.DataFrame(recs)

    for name, t in per.items():
        if t.empty:
            print(f"\n{name}: no successful fits")
            continue
        g = t[is_good(t)]
        print(f"\n{name}")
        print(f"  positions fitted        : {len(t)}  ({len(g)} converged, {len(t)-len(g)} rejected)")
        if len(t) - len(g):
            bad = t[~is_good(t)]
            for _, r in bad.iterrows():
                print(f"    REJECTED {str(r['pos']):16s} tau={r['tau']:.2f}+-{r['tau_err']:.2f} "
                      f"therm={r['therm']:.2f} chi2/ndof={r['redchi']:.2f} N={r['N']}")
        if g.empty:
            continue
        w = 1.0 / g.tau_err ** 2
        wavg = (g.tau * w).sum() / w.sum()
        werr = np.sqrt(1.0 / w.sum())
        # scatter across positions is larger than the fit errors alone -> also
        # quote the spread-based error, which is the honest one for an average
        # over positions that genuinely differ
        chi2_scatter = (w * (g.tau - wavg) ** 2).sum() / (len(g) - 1)
        print(f"  weighted-avg tau        : {wavg:.2f} +- {werr:.2f} us  (converged only)")
        print(f"    scale-factor infl. err: +- {werr*np.sqrt(max(chi2_scatter,1.0)):.2f} us  "
              f"(chi2/ndof of the average = {chi2_scatter:.2f})")
        print(f"  unweighted mean tau     : {g.tau.mean():.2f} +- {g.tau.std(ddof=1)/np.sqrt(len(g)):.2f} us")
        wt = 1.0 / g.therm_err ** 2
        print(f"  weighted-avg therm      : {(g.therm*wt).sum()/wt.sum():.2f} us")
        print(f"  chi2/ndof range         : {t.redchi.min():.2f} - {t.redchi.max():.2f}")
        print(f"  tau range across pos    : {g.tau.min():.2f} - {g.tau.max():.2f} us")

    # ---- paired per-position comparison: recovered vs baseline ----
    print("\n" + "=" * 78)
    print("PAIRED per-position tau difference, recovered - baseline")
    print("=" * 78)
    tb = per["baseline IC 700-1200"]
    tr = per["RECOVERED only (fprompt2d - baseline)"]
    tf = per["fprompt2d 500-2000 x 0.15-0.30"]
    m = (tb.assign(ok=is_good(tb))[["pos", "tau", "tau_err", "ok"]]
         .merge(tr.assign(ok=is_good(tr))[["pos", "tau", "tau_err", "ok"]],
                on="pos", suffixes=("_b", "_r")))
    m = m[m.ok_b & m.ok_r].copy()
    m["d"] = m.tau_r - m.tau_b
    m["d_err"] = np.sqrt(m.tau_err_r ** 2 + m.tau_err_b ** 2)
    m["pull"] = m.d / m.d_err
    print(m[["pos", "tau_b", "tau_err_b", "tau_r", "tau_err_r", "d", "d_err", "pull"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    wd = 1.0 / m.d_err ** 2
    dbar = (m.d * wd).sum() / wd.sum()
    dbar_err = np.sqrt(1.0 / wd.sum())
    print(f"\n  paired positions          : {len(m)}")
    print(f"  weighted mean d(tau)      : {dbar:+.2f} +- {dbar_err:.2f} us  "
          f"-> {abs(dbar/dbar_err):.1f} sigma from zero")
    print(f"  unweighted mean d(tau)    : {m.d.mean():+.2f} us")
    print(f"  positions with d > 0      : {(m.d > 0).sum()} / {len(m)}")

    # ---- model-independent late-time (accidental) probe ----
    print("\n" + "=" * 78)
    print("MODEL-INDEPENDENT accidental probe: late-time / early-time count ratio")
    print("Accidentals are flat in time; capture signal has decayed by 50-67 us.")
    print("A recovered sample dominated by accidentals must show a LARGER ratio.")
    print("=" * 78)
    print(f"{'sample':40s} {'N(10-20us)':>11s} {'N(50-67us)':>11s} {'ratio':>10s}")
    for name, d in samples:
        ct = d.clusterTime.values
        early = int(((ct >= 10) & (ct < 20)).sum())
        late = int(((ct >= 50) & (ct < 67)).sum())
        # normalise by window width so the ratio is per-us
        re_, rl = early / 10.0, late / 17.0
        err = (rl / re_) * np.sqrt(1.0 / max(late, 1) + 1.0 / max(early, 1))
        print(f"{name:40s} {early:11d} {late:11d}   {rl/re_:.4f}+-{err:.4f}")

    # side-by-side per position
    print("\n" + "=" * 78)
    print("PER-POSITION tau, side by side (B fixed)")
    print("=" * 78)
    cols = []
    for name, t in per.items():
        if t.empty:
            continue
        s = t.set_index("pos")[["tau", "tau_err", "redchi", "N"]]
        s.columns = pd.MultiIndex.from_product([[name.split()[0]], s.columns])
        cols.append(s)
    if cols:
        merged = pd.concat(cols, axis=1)
        print(merged.to_string(float_format=lambda v: f"{v:.2f}"))

    # ---- well-conditioned A*exp(-t/tau)+B fit: the decisive test ----
    print("\n" + "=" * 78)
    print("A*exp(-t/tau) + B on 10-67 us, B FREE  (well-conditioned; no rise term)")
    print("=" * 78)
    print(f"{'sample':40s} {'N':>7s} {'tau (us)':>15s} {'B (cts/us-bin)':>17s} "
          f"{'flat frac':>10s} {'chi2/ndof':>10s}")
    for name, d in samples:
        r = fit_expflat(d.clusterTime.values)
        if r is None:
            print(f"{name:40s}  failed")
            continue
        print(f"{name:40s} {r['N']:7d} {r['tau']:7.2f}+-{r['tau_err']:<6.2f} "
              f"{r['B']:8.1f}+-{r['B_err']:<7.1f} {r['flat_frac']:9.3f} {r['redchi']:10.2f}")

    print("\nPer-position, same model (this converges where NeutCapture did not):")
    exp_per = {}
    for name, d in samples:
        recs = []
        for pos, grp in d.groupby(key):
            r = fit_expflat(grp.clusterTime.values)
            if r is None:
                continue
            r["pos"] = pos
            recs.append(r)
        exp_per[name] = pd.DataFrame(recs)
        t = exp_per[name]
        ok = t[(t.tau_err > 0) & (t.redchi < 3.0)]
        if ok.empty:
            print(f"  {name}: no good fits")
            continue
        w = 1.0 / ok.tau_err ** 2
        wavg = (ok.tau * w).sum() / w.sum()
        werr = np.sqrt(1.0 / w.sum())
        print(f"  {name:38s} {len(ok)}/{len(t)} conv   "
              f"tau = {wavg:.2f} +- {werr:.2f} us   "
              f"mean flat frac = {ok.flat_frac.mean():.3f}   "
              f"chi2/ndof {t.redchi.min():.2f}-{t.redchi.max():.2f}")

    eb, er = exp_per["baseline IC 700-1200"], exp_per["RECOVERED only (fprompt2d - baseline)"]
    me = (eb[["pos", "tau", "tau_err", "flat_frac", "redchi"]]
          .merge(er[["pos", "tau", "tau_err", "flat_frac", "redchi"]],
                 on="pos", suffixes=("_b", "_r")))
    me = me[(me.redchi_b < 3) & (me.redchi_r < 3) & (me.tau_err_b > 0) & (me.tau_err_r > 0)].copy()
    me["d"] = me.tau_r - me.tau_b
    me["d_err"] = np.sqrt(me.tau_err_r ** 2 + me.tau_err_b ** 2)
    wd = 1.0 / me.d_err ** 2
    dbar, dbar_err = (me.d * wd).sum() / wd.sum(), np.sqrt(1.0 / wd.sum())
    print(f"\n  paired d(tau) recovered-baseline, expflat model, {len(me)} positions:")
    print(f"    {dbar:+.2f} +- {dbar_err:.2f} us  -> {abs(dbar/dbar_err):.1f} sigma from zero")
    print(f"    mean flat fraction: baseline {me.flat_frac_b.mean():.3f}  "
          f"recovered {me.flat_frac_r.mean():.3f}")

    outdir = "TriggerSummary"
    for name, t in per.items():
        if t.empty:
            continue
        slug = ("baseline" if name.startswith("baseline")
                else "fprompt2d" if name.startswith("fprompt") else "recovered")
        p = os.path.join(outdir, f"CaptureTimeFits_{slug}.csv")
        t.to_csv(p, index=False)
        print(f"\nwrote {p}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        prog="fit_capture_time_fprompt_compare",
        description="Capture-time fits sharing one recipe (70 bins over 0-70 us, "
                    "10-67 us window, NeutCapture with B fixed at 0).")
    # No default on purpose. The two modes fit different samples and answer
    # different questions; silently picking one would let a tagset measurement be
    # read as the fprompt comparison, or vice versa.
    ap.add_argument("--mode", required=True, choices=["fpcompare", "tagset"],
                    help="fpcompare: the original IC-baseline vs fprompt2d vs "
                         "recovered-only study, on the hardcoded 20-run list. "
                         "tagset: pool --tags and fit per source position, "
                         "writing CaptureTimeFits_<out-label>.csv.")
    ap.add_argument("--tags", default=None,
                    help="tagset mode only: comma-separated Stage-1 tags, e.g. "
                         "AmBe2.0v4_gated,AmBe2.0v4_ext")
    ap.add_argument("--out-label", default=None,
                    help="tagset mode only: slug for TriggerSummary/"
                         "CaptureTimeFits_<label>.csv. Must not be 'baseline', "
                         "'fprompt2d' or 'recovered' — those belong to fpcompare.")
    # Required, no default, for the same reason --mode is: the frozen
    # tau = 30.53 +- 0.26 us anchor and every CaptureTimeFits_*.csv already on disk
    # were produced with the scipy minimiser. A silent default would let an lmfit
    # number be compared against, or written over, a scipy one without anyone
    # having said so.
    ap.add_argument("--fit-backend", required=True, choices=list(BACKENDS),
                    help="minimiser for the NeutCapture fit. Same model, window, "
                         "bounds and weights either way; 'scipy' is what the frozen "
                         "anchor was measured with, 'lmfit' is what "
                         "ambe.plots.basic.lmfit_analysis uses.")
    args = ap.parse_args()

    if args.mode == "fpcompare":
        for flag in ("tags", "out_label"):
            if getattr(args, flag) is not None:
                ap.error(f"--{flag.replace('_','-')} is only meaningful with --mode tagset")
        return mode_fpcompare(args.fit_backend)

    if not args.tags or not args.out_label:
        ap.error("--mode tagset requires both --tags and --out-label")
    reserved = {"baseline", "fprompt2d", "recovered"}
    if args.out_label in reserved:
        ap.error(f"--out-label {args.out_label!r} is reserved for --mode fpcompare "
                 f"output; pick another so the frozen anchor file is not overwritten")
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        ap.error("--tags parsed to nothing")
    return mode_tagset(tags, args.out_label, args.fit_backend)


if __name__ == "__main__":
    sys.exit(main())
