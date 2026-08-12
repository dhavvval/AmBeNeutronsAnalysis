#!/usr/bin/env python
"""
The two results from REPORT §7.2 / §7.5 that existed only as numbers:

  fprompt_yield_vs_IC.pdf
      P(>=1 neutron cluster | accepted tag) as a function of IC_adjusted, with
      the old and widened windows marked, over a panel of the tag statistics.
      This is the curve the IC window slides along -- the tool for picking the
      final bounds (§7.4 item 1).

  fprompt_capture_time_recovered_vs_baseline.pdf
      Normalised cluster-time distributions, recovered-only vs baseline, plus
      the ratio -- the model-free evidence that the recovered events are signal.
      Includes what a 5% flat accidental admixture WOULD look like, so the
      exclusion is visible rather than asserted.

Two stacked panels per figure sharing the x axis, never a second y scale on the
same panel.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    MPLBACKEND=Agg python -u plot_fprompt_yield_and_capture.py
"""
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from ambe.plotting import set_style

RUNS = [6046, 6056, 6060, 6061, 6062, 6165, 6166, 6186, 6187, 6188, 6189,
        6230, 6231, 6232, 6234, 6235, 6237, 6239, 6241, 6242]
FEAT = "TriggerSummary/WaveformFeatures_AmBe2.0v4_fprompt_{run}.parquet"
CAND = "EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_AmBe2.0v4_{tag}_{run}.csv"
OUTDIR = "verbose"

# validated categorical slots 1 & 2 (scripts/validate_palette.js, light mode:
# CVD dE 24.7 protan / 32.7 tritan, normal 33.6, all checks PASS)
C_BASE, C_REC = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
OLD_LO, OLD_HI = 700, 1200
NEW_LO, NEW_HI = 500, 2000


def load_features():
    out = []
    for run in RUNS:
        d = pd.read_parquet(FEAT.format(run=run),
                            columns=["timestamp", "IC_adjusted", "accepted"])
        ts = pd.read_csv(CAND.format(tag="fprompt", run=run),
                         usecols=["eventTankTime"]).eventTankTime
        d["has_n"] = d.timestamp.isin(set(ts))
        out.append(d)
    return pd.concat(out, ignore_index=True)


def load_cluster_times():
    base, rec = [], []
    for run in RUNS:
        uc = ["eventID", "clusterTime"]
        g = pd.read_csv(CAND.format(tag="gated", run=run), usecols=uc)
        f = pd.read_csv(CAND.format(tag="fprompt", run=run), usecols=uc)
        base.append(g)
        rec.append(f[~f.eventID.isin(set(g.eventID))])
    b = pd.concat(base).clusterTime.values / 1000.0
    r = pd.concat(rec).clusterTime.values / 1000.0
    win = lambda a: a[(a >= 10) & (a < 67)]
    return win(b), win(r)


def plot_yield_vs_ic(a):
    acc = a[a.accepted]
    edges = np.arange(500, 2050, 50)
    ctr = (edges[:-1] + edges[1:]) / 2
    p, e, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = acc[(acc.IC_adjusted > lo) & (acc.IC_adjusted <= hi)]
        n.append(len(s))
        if len(s) < 50:
            p.append(np.nan); e.append(np.nan); continue
        q = s.has_n.mean()
        p.append(q); e.append(np.sqrt(q * (1 - q) / len(s)))
    p, e, n = np.array(p), np.array(e), np.array(n)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1]})
    for A in (ax, ax2):
        A.axvspan(NEW_LO, OLD_LO, color="#eb6834", alpha=0.10, lw=0)
        A.axvspan(OLD_HI, NEW_HI, color="#eb6834", alpha=0.10, lw=0)
        A.axvspan(OLD_LO, OLD_HI, color="#2a78d6", alpha=0.10, lw=0)

    ax.errorbar(ctr, p, yerr=e, fmt="o", ms=5.5, lw=2.0, color=C_BASE,
                mfc=C_BASE, mec="white", mew=0.8, capsize=2.5, zorder=3)
    pk = int(np.nanargmax(p))
    ax.plot(ctr[pk], p[pk], "*", ms=17, color=C_REC, mec="white", mew=0.9,
            zorder=4)
    # park the label in the empty upper-right, clear of the descending points
    ax.annotate(f"peak  IC $\\approx${ctr[pk]:.0f}\nP = {p[pk]:.3f}",
                (ctr[pk], p[pk]), textcoords="offset points", xytext=(128, -52),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0),
                bbox=dict(fc="white", ec=MUTED, lw=0.6, alpha=0.95, pad=0.35))
    ax.set_ylabel("P( $\\geq$1 neutron cluster | accepted tag )")
    ax.set_title("Neutron yield per accepted tag vs tag-pulse size — AmBe 2.0v4, 20 runs\n"
                 "the old IC window sat on the peak, so widening it can only lower the ratio",
                 fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.5 * (OLD_LO + OLD_HI), ax.get_ylim()[0] + 0.012,
            "old window\n700–1200", ha="center", fontsize=9.5, color=C_BASE)
    for lo, hi in ((NEW_LO, OLD_LO), (OLD_HI, NEW_HI)):
        ax.text(0.5 * (lo + hi), ax.get_ylim()[0] + 0.012, "added by\nwidening",
                ha="center", fontsize=9.5, color=C_REC)

    ax2.bar(ctr, n, width=46, color=MUTED, alpha=0.55, lw=0)
    ax2.set_yscale("log")
    ax2.set_ylabel("accepted tags / bin")
    ax2.set_xlabel("IC_adjusted  (tag-PMT integrated charge)")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTDIR}/fprompt_yield_vs_IC.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUTDIR}/fprompt_yield_vs_IC.pdf/.png")
    plt.close(fig)


def plot_capture(b, r):
    edges = np.linspace(10, 67, 39)
    ctr = (edges[:-1] + edges[1:]) / 2
    hb, _ = np.histogram(b, bins=edges)
    hr, _ = np.histogram(r, bins=edges)
    pb, pr = hb / hb.sum(), hr / hr.sum()
    eb_, er_ = np.sqrt(hb) / hb.sum(), np.sqrt(hr) / hr.sum()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.2]})
    ax.errorbar(ctr, pb, yerr=eb_, fmt="o", ms=4.5, lw=1.8, color=C_BASE,
                mec="white", mew=0.7, label=f"baseline IC 700–1200  (N={len(b):,})")
    ax.errorbar(ctr, pr, yerr=er_, fmt="s", ms=4.5, lw=1.8, color=C_REC,
                mec="white", mew=0.7, label=f"recovered only  (N={len(r):,})")
    ax.set_yscale("log")
    ax.set_ylabel("fraction of clusters / bin")
    ax.set_title("Cluster-time distribution: events the widened window recovered vs the known-signal baseline\n"
                 "normalised to unit area; identical within errors (KS p = 0.49, shape $\\chi^2$ = 6.8/9)",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # coarser bins for the ratio: at 38 bins the per-bin error (~0.06) swamps the
    # 5%/10% reference curves and nothing can be read off it. 12 bins is what the
    # shape chi2 in REPORT §7.2 uses.
    cedges = np.linspace(10, 67, 13)
    cctr = (cedges[:-1] + cedges[1:]) / 2
    cb, _ = np.histogram(b, bins=cedges)
    cr, _ = np.histogram(r, bins=cedges)
    qb, qr = cb / cb.sum(), cr / cr.sum()
    fb, fr = np.sqrt(cb) / cb.sum(), np.sqrt(cr) / cr.sum()
    ratio = qr / qb
    rerr = ratio * np.sqrt((fr / qr) ** 2 + (fb / qb) ** 2)
    ax2.axhline(1.0, color=MUTED, lw=1.4, zorder=1)
    # what a 5% and 10% flat accidental admixture would do -- the thing excluded
    for frac, ls in ((0.05, "--"), (0.10, ":")):
        flat = np.ones(len(qb)) / len(qb)
        pred = ((1 - frac) * qb + frac * flat) / qb
        ax2.plot(cctr, pred, ls, lw=2.2, color=INK, zorder=2,
                 label=f"if {frac:.0%} of it were flat accidentals")
    ax2.errorbar(cctr, ratio, yerr=rerr, fmt="s", ms=7, lw=2.0, color=C_REC,
                 mec="white", mew=0.8, capsize=3, zorder=3, label="measured")
    ax2.set_ylabel("recovered / baseline")
    ax2.set_xlabel("cluster time  ($\\mu$s)")
    ax2.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTDIR}/fprompt_capture_time_recovered_vs_baseline.{ext}",
                    dpi=200, bbox_inches="tight")
    print(f"wrote {OUTDIR}/fprompt_capture_time_recovered_vs_baseline.pdf/.png")
    plt.close(fig)


def main():
    set_style()
    plot_yield_vs_ic(load_features())
    b, r = load_cluster_times()
    plot_capture(b, r)


if __name__ == "__main__":
    sys.exit(main())
