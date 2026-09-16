#!/usr/bin/env python
"""Capture-time fitting helpers and per-position CSV fitting command.

Times are in microseconds; defaults use 70 bins over 0–70 us and a 2–67 us fit window.
The scipy and lmfit backends retain the existing model and parameter bounds.
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

BACKENDS = ("scipy", "lmfit")

CAND_DIR = "EventAmBeNeutronCandidatesData"

# --- replicated verbatim from ambe/plots/basic.py fitting_config ---
TIME_BINS = 70
TIME_RANGE = (0, 70)
FIT_MIN, FIT_MAX = 2.0, 67.0
INIT = dict(A=200.0, therm=5.0, tau=25.0, B=0.0)
BOUNDS_LO = dict(A=0.0, therm=0.1, tau=10.0, B=0.0)
BOUNDS_HI = dict(A=np.inf, therm=10.0, tau=70.0, B=15.0)

USECOLS = ["eventID", "clusterTime", "sourceX", "sourceY", "sourceZ"]


def NeutCapture(t, A, therm, tau, B):
    return A * (1 - np.exp(-t / therm)) * np.exp(-t / tau) + B


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

    Shared by the per-position command and presentation adapters; applies the same
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
    histogram, same 2-67 us window, same bounds, B held at zero. Returned frame
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
    """Pool candidate tags, fit by source position and write CaptureTimeFits CSV."""
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


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Fit candidate times by source position")
    parser.add_argument("--tags", required=True, help="Comma-separated candidate tags")
    parser.add_argument("--out-label", required=True)
    parser.add_argument("--fit-backend", required=True, choices=BACKENDS)
    args = parser.parse_args(argv)
    if args.out_label in {"baseline", "fprompt2d", "recovered"}:
        parser.error("Choose a new output label; that label backs historical results")
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        parser.error("--tags parsed to nothing")
    return mode_tagset(tags, args.out_label, args.fit_backend)


if __name__ == "__main__":
    main()
