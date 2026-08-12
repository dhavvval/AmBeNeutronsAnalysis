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

  - 70 bins over (0, 70) us, fit mask 10 < bin_centre < 67
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
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CAND_DIR = "EventAmBeNeutronCandidatesData"
BASE_TAG = "AmBe2.0v4_gated"
FP_TAG = "AmBe2.0v4_fprompt"
RUNS = [6046, 6056, 6060, 6061, 6062, 6165, 6166, 6186, 6187, 6188, 6189,
        6230, 6231, 6232, 6234, 6235, 6237, 6239, 6241, 6242]

# --- replicated verbatim from ambe/plots/basic.py fitting_config ---
TIME_BINS = 70
TIME_RANGE = (0, 70)
FIT_MIN, FIT_MAX = 10.0, 67.0
INIT = dict(A=200.0, therm=5.0, tau=25.0, B=0.0)
BOUNDS_LO = dict(A=0.0, therm=0.1, tau=10.0, B=0.0)
BOUNDS_HI = dict(A=np.inf, therm=10.0, tau=70.0, B=15.0)

USECOLS = ["eventID", "clusterTime", "sourceX", "sourceY", "sourceZ"]


def NeutCapture(t, A, therm, tau, B):
    return A * (1 - np.exp(-t / therm)) * np.exp(-t / tau) + B


def prepare(ct_us):
    """Exact replica of AmBeNeutronAnalyzer._prepare_fitting_data."""
    counts, edges = np.histogram(ct_us, bins=TIME_BINS, range=TIME_RANGE)
    centres = (edges[:-1] + edges[1:]) / 2
    mask = (centres > FIT_MIN) & (centres < FIT_MAX)
    x, y = centres[mask], counts[mask]
    err = np.sqrt(y).astype(float)
    err[err == 0] = 1e-10
    return x, y, err


def fit(ct_us, float_B):
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


def fit_expflat(ct_us):
    """A*exp(-t/tau) + B on 10-67 us, B free.

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
    x, y, err = prepare(ct_us)
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


def load(tag, run):
    p = os.path.join(CAND_DIR, f"EventAmBeNeutronCandidates_{tag}_{run}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=USECOLS)
    d["clusterTime"] = d["clusterTime"] / 1000.0        # ns -> us, as basic.py:143
    d["run"] = run
    return d


def main():
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
            r = fit(d.clusterTime.values, float_B)
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
            r = fit(grp.clusterTime.values, float_B=False)
            if r is None:
                continue
            r["pos"] = pos
            recs.append(r)
        per[name] = pd.DataFrame(recs)

    def is_good(t):
        """Drop fits that did not converge: parameter pinned at a bound (which
        makes curve_fit report stderr exactly 0 and gives it infinite weight),
        or a chi2/ndof that says the model does not describe the data."""
        return ((t.tau_err > 0) & (t.therm_err > 0)
                & (t.tau > BOUNDS_LO["tau"] + 1e-6) & (t.tau < BOUNDS_HI["tau"] - 1e-6)
                & (t.therm > BOUNDS_LO["therm"] + 1e-6)
                & (t.redchi < 3.0))

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


if __name__ == "__main__":
    sys.exit(main())
