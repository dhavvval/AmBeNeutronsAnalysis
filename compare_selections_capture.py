#!/usr/bin/env python3
"""
compare_selections_capture.py

Final like-for-like comparison of the two selections on AmBe DATA, fitting the
SAME observable (earliest-hit cluster time) per source position:

  Selection 1 — OPTICS + frozen MVA   : scored OPTICS parquet, rf_score>=cut,
                                         clusterTime = clusterTime_earliest [ns]
  Selection 2 — ClusterFinder + cut   : raw CF parquet, PE<80/CB<0.45/nHits>9,
                                         clusterTime = native CF clusterTime [ns]

Both fit neut_capture (thermalisation x exp decay + bg), 70 bins over [0,70] us,
window [2,65] us, per source position. Outputs a per-position tau table (both
selections side by side) + capture-time PDF + a multiplicity comparison reusing
plot_multiplicity_compare's stats.

Usage (myboy venv):
  python compare_selections_capture.py \
      --sel1 .../ambe_all__data_features__scored.parquet \
      --sel2 .../cf_raw/v1_raw_all.parquet \
      --score-col rf_score --score-cut 0.643 \
      --out-dir .../compare_selections
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.plotting import set_style                       # noqa: E402
from ambe.plots.combined import neut_capture              # noqa: E402  A*(1-e^-t/th)*e^-t/tau+B
from ambe.data.processor import AmBeNeutronProcessing      # noqa: E402

SOURCE_POSITIONS = AmBeNeutronProcessing().source_positions
PRESEL_PE_MAX, PRESEL_CB_MAX, PRESEL_HITS_MIN = 80.0, 0.45, 9
BINS, TRANGE, FIT_LO, FIT_HI = 70, (0.0, 70.0), 2.0, 65.0
SEL1_C, SEL2_C = "#0077BB", "#EE7733"


def port_of(x, y, z):
    if x == 75:  return "Port 4 (x=75)"
    if z == -75: return "Port 1 (z=-75)"
    if z == 102: return "Port 3 (z=+102)"
    if z == 75:  return "Port 2 (z=+75)"
    if z == 0:   return "Port 5 (z=0)"
    return f"x{x} z{z}"


def fit_tau(t_us):
    c, e = np.histogram(t_us, bins=BINS, range=TRANGE)
    ctr = (e[:-1] + e[1:]) / 2
    m = (ctr > FIT_LO) & (ctr < FIT_HI)
    fx, fy = ctr[m], c[m]
    if fy.sum() < 20:
        return None
    err = np.sqrt(np.maximum(fy, 1))
    try:
        po, pc = curve_fit(neut_capture, fx, fy, p0=[fy.max(), 5, 25, 0], sigma=err,
                           absolute_sigma=True,
                           bounds=([0, .1, .1, 0], [np.inf, 100, 100, np.inf]), maxfev=20000)
    except Exception:
        return None
    pe = np.sqrt(np.diag(pc))
    cv = float(np.sum((fy - neut_capture(fx, *po)) ** 2 / err ** 2))
    nd = max(len(fy) - 4, 1)
    return dict(tau=po[2], tau_err=pe[2], therm=po[1], chi2ndof=cv / nd, n=int(len(t_us)))


def load_sel1(path, score_col, score_cut):
    d = pd.read_parquet(path)
    # Use OPTICS t_mean (mean hit time), NOT clusterTime_earliest. The earliest-hit
    # time piles up at the t>2us prompt-window boundary (giant 2us spike, no
    # thermalisation rise -> therm->0, chi2~53). t_mean reproduces the proper
    # capture-time turn-on and tracks ClusterFinder's clusterTime almost exactly
    # (medians 24.4 vs 22.9 us), so it is the correct OPTICS analogue for the fit.
    if "t_mean" not in d.columns:
        raise SystemExit("Sel1 parquet lacks t_mean — re-extract OPTICS features.")
    d = d[d[score_col] >= score_cut].copy()
    d["t_us"] = d["t_mean"] / 1000.0
    pos = d["run"].map(lambda r: SOURCE_POSITIONS.get(int(r)))
    d = d[pos.notna()].copy()
    d["pos"] = pos[pos.notna()].map(lambda p: (p[0], p[1], p[2]))
    return d


def load_sel2(path):
    d = pd.read_parquet(path)
    m = (d["clusterPE"] < PRESEL_PE_MAX) & \
        ((d["clusterChargeBalance"] < PRESEL_CB_MAX) | d["clusterChargeBalance"].isna()) & \
        (d["clusterHits"] > PRESEL_HITS_MIN)
    d = d[m].copy()
    d["t_us"] = d["clusterTime"] / 1000.0
    d["pos"] = list(zip(d["sourceX"], d["sourceY"], d["sourceZ"]))
    return d


def per_position_table(d1, d2):
    positions = sorted(set(d1["pos"]) | set(d2["pos"]), key=lambda p: (port_of(*p), p[1]))
    rows = []
    for pos in positions:
        lbl = f"{port_of(*pos)}, y={int(pos[1]):+d}"
        r1 = fit_tau(d1[d1["pos"] == pos]["t_us"]) if (d1["pos"] == pos).any() else None
        r2 = fit_tau(d2[d2["pos"] == pos]["t_us"]) if (d2["pos"] == pos).any() else None
        rows.append({
            "position": lbl,
            "sel1_n": r1["n"] if r1 else 0, "sel1_tau": r1["tau"] if r1 else np.nan,
            "sel1_tau_err": r1["tau_err"] if r1 else np.nan,
            "sel1_chi2ndof": r1["chi2ndof"] if r1 else np.nan,
            "sel2_n": r2["n"] if r2 else 0, "sel2_tau": r2["tau"] if r2 else np.nan,
            "sel2_tau_err": r2["tau_err"] if r2 else np.nan,
            "sel2_chi2ndof": r2["chi2ndof"] if r2 else np.nan,
        })
    return pd.DataFrame(rows)


def _legacy_capture_page(pdf, t_us, title):
    """One capture-time page in the EXACT legacy basic.py scipy_curve_fit style:
    figsize (10,6); step hist color='blue' label 'Data'; blue errorbars; red fit
    line lw=2 with the therm/tau/chi2/ndof multi-line label; + a residuals page.
    Returns the fit dict or None."""
    from scipy.stats import chi2 as _chi2
    counts, edges = np.histogram(t_us, bins=BINS, range=TRANGE)
    centers = (edges[:-1] + edges[1:]) / 2
    m = (centers > FIT_LO) & (centers < FIT_HI)
    xdata, ydata = centers[m], counts[m]
    yerr = np.sqrt(np.maximum(ydata, 1))
    if ydata.sum() < 20:
        return None
    try:
        popt, pcov = curve_fit(neut_capture, xdata, ydata,
                               p0=[ydata.max(), 5.0, 25.0, 0.0], sigma=yerr,
                               absolute_sigma=True,
                               bounds=([0, .1, .1, 0], [np.inf, 100, 100, np.inf]),
                               maxfev=20000)
    except Exception as e:
        print(f"  fit failed [{title}]: {e}")
        return None
    perr = np.sqrt(np.diag(pcov))
    yexp = neut_capture(xdata, *popt)
    chi2_value = float(np.sum(((ydata - yexp) ** 2) / (yerr ** 2)))
    ndof = len(ydata) - len(popt)
    chi2_ndof = chi2_value / ndof
    p_value = float(_chi2.sf(chi2_value, ndof))

    # --- capture-time page (legacy look) ---
    plt.figure(figsize=(10, 6))
    plt.hist(t_us, bins=BINS, range=TRANGE, histtype='step', color='blue', label="Data")
    plt.errorbar(xdata, ydata, yerr=yerr, color='blue', linestyle='None', alpha=0.7)
    label = (
        fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
        fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
        fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof}$, " + "\n"
        fr"$\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$"
    )
    plt.plot(xdata, yexp, 'r-', linewidth=2, label=label)
    plt.xlabel(fr"Cluster Time [$\mu s$]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

    # --- residuals page (legacy look) ---
    residuals = (ydata - yexp) / yerr
    plt.figure(figsize=(10, 6))
    plt.plot(xdata, residuals, 'o-')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time [μs]")
    plt.ylabel("Normalized Residual")
    plt.title(f"Fit Residuals — {title}")
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()
    return dict(tau=popt[2], tau_err=perr[2], therm=popt[1], chi2ndof=chi2_ndof,
                p_value=p_value, n=int(len(t_us)))


def capture_pdf(d, label_prefix, out_pdf):
    """All-positions page + one page per source position, legacy style."""
    with PdfPages(out_pdf) as pdf:
        _legacy_capture_page(pdf, d["t_us"], f"{label_prefix} — All Positions")
        positions = sorted(set(d["pos"]), key=lambda p: (port_of(*p), p[1]))
        for pos in positions:
            sub = d[d["pos"] == pos]
            sx, sy, sz = (int(v) for v in pos)
            _legacy_capture_page(pdf, sub["t_us"],
                                 f"{label_prefix} — run positions:({sx}, {sy}, {sz})")


def multiplicity_stats(d1, d2):
    def stats(d):
        # event key = run + per-event tank timestamp (column name differs by source:
        # OPTICS parquet uses 'event_tank_time', CF parquet uses 'eventTankTime').
        tcol = "event_tank_time" if "event_tank_time" in d.columns else "eventTankTime"
        keys = [k for k in ("run", tcol) if k in d.columns]
        if len(keys) < 2:
            return dict(n_events=0, single=np.nan, mean=np.nan, n2=np.nan, ge3=np.nan)
        m = d.groupby(keys).size()
        m1 = m[m >= 1]
        return dict(n_events=int(len(m1)), single=float((m1 == 1).mean()),
                    mean=float(m1.mean()), n2=float((m1 == 2).mean()),
                    ge3=float((m1 >= 3).mean()))
    return stats(d1), stats(d2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sel1", required=True, help="scored OPTICS parquet (Selection 1)")
    p.add_argument("--sel2", required=True, help="raw CF parquet v1_raw_all (Selection 2)")
    p.add_argument("--score-col", default="rf_score")
    p.add_argument("--score-cut", type=float, default=0.643)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    set_style()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d1 = load_sel1(args.sel1, args.score_col, args.score_cut)
    d2 = load_sel2(args.sel2)

    tab = per_position_table(d1, d2)
    tab.to_csv(out / "capture_tau_by_position_compare.csv", index=False)
    # Legacy-style capture-time PDFs, one per selection (all-positions + per-position).
    capture_pdf(d1, f"Selection 1 — OPTICS+MVA (rf>={args.score_cut:.2f})",
                out / "sel1_capture_by_position.pdf")
    capture_pdf(d2, "Selection 2 — ClusterFinder+cut",
                out / "sel2_capture_by_position.pdf")

    s1, s2 = multiplicity_stats(d1, d2)
    print("\n=== CAPTURE TIME by position (Sel1=OPTICS t_mean, Sel2=CF clusterTime) ===")
    print(f"{'position':<24}{'S1 n':>7}{'S1 tau':>8}{'S1 chi2':>8}   {'S2 n':>7}{'S2 tau':>8}{'S2 chi2':>8}")
    for _, r in tab.iterrows():
        print(f"{r.position:<24}{r.sel1_n:>7}{r.sel1_tau:>8.1f}{r.sel1_chi2ndof:>8.2f}   "
              f"{r.sel2_n:>7}{r.sel2_tau:>8.1f}{r.sel2_chi2ndof:>8.2f}")
    print("\n=== MULTIPLICITY ===")
    print(f"  Sel1 OPTICS+MVA : events={s1['n_events']:,} single={s1['single']:.3f} "
          f"mean={s1['mean']:.3f} n2={s1['n2']:.3f} n>=3={s1['ge3']:.3f}")
    print(f"  Sel2 CF+cut     : events={s2['n_events']:,} single={s2['single']:.3f} "
          f"mean={s2['mean']:.3f} n2={s2['n2']:.3f} n>=3={s2['ge3']:.3f}")
    print(f"\n[compare] tau table -> {out/'capture_tau_by_position_compare.csv'}")
    print(f"[compare] pdf       -> {out/'capture_compare_allpos.pdf'}")


if __name__ == "__main__":
    main()
