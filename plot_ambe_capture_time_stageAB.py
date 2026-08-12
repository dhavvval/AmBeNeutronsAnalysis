"""
plot_ambe_capture_time_stageAB.py

Neutron capture-time distribution for AmBe DATA, for clusters surviving the
Stage-A (OPTICS pre-selection) + Stage-B (MVA) selection, fit with the SAME model
and style as src/ambe/plots/combined.py (neut_capture: thermalisation * exp decay
+ bg).  One fit per source PORT (and an all-ports page).

Reuses combined.neut_capture verbatim so the curve/parameters match the existing
capture-time fits; only the input (scored data parquet, MVA-gated) and the
per-port loop are new.

Input : the scored feature parquet (has t_mean [ns], rf_score, port).
Output: <out>  (one multi-page PDF: all-ports + one page per port).

Usage:
  python plot_ambe_capture_time_stageAB.py \
      --scored .../ambe_all__data_features__scored.parquet \
      --score-col rf_score --score-cut 0.643 \
      --out .../ambe_all__capture_time_by_port.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
from matplotlib.backends.backend_pdf import PdfPages

# Reuse the exact capture-time model from the existing toolkit.
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.plots.combined import neut_capture   # A*(1-exp(-t/therm))*exp(-t/tau)+B
from ambe.data.processor import AmBeNeutronProcessing

SOURCE_POSITIONS = AmBeNeutronProcessing().source_positions   # run -> (x,y,z) cm

# Which physical tank port a position belongs to (z/x define the port; y is the
# scan within a port).
def port_of(x, y, z):
    if (x, y, z) == (0, 328, 0): return "Port out (no source)"
    if x == 75:  return "Port 4 (x=75)"
    if z == -75: return "Port 1 (z=-75)"
    if z == 102: return "Port 3 (z=+102)"
    if z == 75:  return "Port 2 (z=+75)"
    if z == 0:   return "Port 5 (z=0)"
    return f"x{x} z{z}"

def position_label(x, y, z):
    return f"{port_of(x,y,z)}, y={int(y):+d} cm"

# Fit window (us), mirrors combined.py defaults.
FIT_LO, FIT_HI = 2.0, 65.0
# 70 bins over [0,70] us = 1 us bins, matching the README fit_params (time_bins: 70).
# NB: finer binning (e.g. 200 bins = 0.35 us) destabilises the 4-parameter fit and
# pulls tau low (verified: 70/140 bins give tau~40 us, 200 bins collapses to ~26 us).
BINS, TRANGE = 70, (0.0, 70.0)


def fit_and_plot(ax, t_us, title):
    """Histogram + neut_capture fit on one axes. Returns (tau, tau_err, n)."""
    counts, edges = np.histogram(t_us, bins=BINS, range=TRANGE)
    centers = (edges[:-1] + edges[1:]) / 2
    m = (centers > FIT_LO) & (centers < FIT_HI)
    fx, fy = centers[m], counts[m]
    fyerr = np.sqrt(np.maximum(fy, 1))
    ax.hist(t_us, bins=BINS, range=TRANGE, histtype="step", color="blue", label="Data")

    tau = tau_err = np.nan
    if fy.sum() > 20:                       # enough stats to attempt a fit
        p0 = [float(fy.max()), 5.0, 25.0, 0.0]
        try:
            popt, pcov = curve_fit(neut_capture, fx, fy, p0=p0, sigma=fyerr,
                                   absolute_sigma=True,
                                   bounds=([0, 0.1, 0.1, 0], [np.inf, 100, 100, np.inf]))
            perr = np.sqrt(np.diag(pcov))
            fexp = neut_capture(fx, *popt)
            chi2v = float(np.sum(((fy - fexp) ** 2) / (fyerr ** 2)))
            ndof = max(len(fy) - len(popt), 1)
            tau, tau_err = popt[2], perr[2]
            label = (fr"$\mathrm{{therm}}={popt[1]:.2f}\pm{perr[1]:.2f}\,\mu s$" + "\n"
                     fr"$\tau={popt[2]:.2f}\pm{perr[2]:.2f}\,\mu s$" + "\n"
                     fr"$\chi^2/\mathrm{{ndof}}={chi2v/ndof:.2f}$")
            ax.plot(fx, fexp, "r-", lw=2, label=label)
        except Exception as e:
            ax.plot([], [], " ", label=f"fit failed: {e}")

    ax.set_xlabel(r"Cluster Time [$\mu s$]")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    return tau, tau_err, int(len(t_us))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--score-col", default="rf_score")
    p.add_argument("--score-cut", type=float, default=0.643,
                   help="MVA cut (default rf @ 80%% MC sig eff = 0.643)")
    p.add_argument("--selection", choices=["mva", "legacy"], default="mva",
                   help="'mva' = Stage-B MVA cut (Selection 1); 'legacy' = "
                        "passes_stage1 PE<80/CB<0.45/nHits>9 (Selection 2) for the "
                        "capture-time comparison.")
    p.add_argument("--by", choices=["position", "port"], default="position",
                   help="'position' = one page per distinct (x,y,z) source position "
                        "(~21 positions); 'port' = one page per tank port (5).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = pd.read_parquet(args.scored)
    # Selection 1 (mva): Stage-B MVA cut on the OPTICS clusters.
    # Selection 2 (legacy): OPTICS clusters passing PE<80/CB<0.45/nHits>9.
    if args.selection == "mva":
        sel = d[d[args.score_col] >= args.score_cut].copy()
        sel_tag = f"Selection 1 — OPTICS + MVA ({args.score_col}≥{args.score_cut:.2f})"
    else:
        if "passes_stage1" not in d.columns:
            raise SystemExit("--selection legacy needs a 'passes_stage1' column")
        sel = d[d["passes_stage1"]].copy()
        sel_tag = "Selection 2 — OPTICS + legacy cut (PE<80/CB<0.45/nHits>9)"
    sel["t_us"] = sel["t_mean"] / 1000.0     # ns -> us, same convention as combined.py

    # Attach the source (x,y,z) per cluster via run -> source_positions, and the
    # chosen grouping key + label.
    pos = sel["run"].map(lambda r: SOURCE_POSITIONS.get(int(r)))
    sel = sel[pos.notna()].copy(); pos = pos[pos.notna()]
    sel["src_x"] = pos.map(lambda p: p[0]); sel["src_y"] = pos.map(lambda p: p[1])
    sel["src_z"] = pos.map(lambda p: p[2])
    if args.by == "position":
        sel["grp"]   = list(zip(sel["src_x"], sel["src_y"], sel["src_z"]))
        sel["label"] = sel.apply(lambda r: position_label(r.src_x, r.src_y, r.src_z), axis=1)
        # sort by port then y for a sensible page order
        sort_key = lambda g: (port_of(*g), g[1])
    else:
        sel["grp"]   = sel.apply(lambda r: port_of(r.src_x, r.src_y, r.src_z), axis=1)
        sel["label"] = sel["grp"]
        sort_key = lambda g: g

    foot = (f"{sel_tag} · fit: thermalisation × exp decay + bg")

    rows = []
    with PdfPages(args.out) as pdf:
        # Page 1: all positions combined
        fig, ax = plt.subplots(figsize=(8, 5))
        tau, te, n = fit_and_plot(ax, sel["t_us"], "AmBe Neutron Capture Time — All Positions")
        fig.text(0.5, 0.005, foot, ha="center", fontsize=8, color="0.4")
        fig.tight_layout(rect=[0, 0.03, 1, 1]); pdf.savefig(fig); plt.close(fig)
        rows.append(("ALL", n, tau, te))

        # One page per group, ordered
        labels = {g: lab for g, lab in zip(sel["grp"], sel["label"])}
        for g in sorted(labels, key=sort_key):
            s = sel[sel["grp"] == g]
            if len(s) == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            tau, te, n = fit_and_plot(ax, s["t_us"],
                                      f"Neutron Capture Time — {labels[g]}")
            fig.text(0.5, 0.005, foot, ha="center", fontsize=8, color="0.4")
            fig.tight_layout(rect=[0, 0.03, 1, 1]); pdf.savefig(fig); plt.close(fig)
            rows.append((labels[g], n, tau, te))

    summary = pd.DataFrame(rows, columns=["group", "n_clusters", "tau_us", "tau_err_us"])
    summary.insert(0, "selection", args.selection)
    csv_out = Path(args.out).with_suffix(".tau_summary.csv")
    summary.to_csv(csv_out, index=False)

    print(f"[capture-time] wrote {args.out}   (by={args.by}, selection={args.selection})")
    print(f"[capture-time] tau summary -> {csv_out}")
    print(f"{'group':<28}{'n_clusters':>11}{'tau_us':>10}{'tau_err':>9}")
    for lab, n, tau, te in rows:
        print(f"{lab:<28}{n:>11}{tau:>10.2f}{te:>9.2f}")


if __name__ == "__main__":
    main()
