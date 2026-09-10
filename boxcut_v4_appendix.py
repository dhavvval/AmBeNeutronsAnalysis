#!/usr/bin/env python
"""
boxcut_v4_appendix.py — the per-position QA appendices for the AmBe v4 campaign.

Everything in boxcut_v4_campaign.py aggregates all 26 source positions onto a single
figure. That is right for slides and wrong for the question a QA audience actually
asks, which is "show me your worst position". This builds the multi-page appendices.

THREE PRODUCTS (--do is required, no default)

  byposition    One page per source position, 26 pages, WORST EFFICIENCY FIRST. Each
                page: the capture-time fit with tau and chi2/ndof, the four box-cut
                variables, and a stats block. Box cuts only, no MVA.

  features      One page per FEATURE; within a page one panel per port and one curve
                per y position. This is the classic "per port position" comparison.
                Written three times: once for all Stage-2 clusters, then once per
                streamline for the MVA-selected subset.

  positiongrid  One page per source position, each showing MVA neutron vs MVA other
                on twelve leading features. Written once per streamline.

NOTE there is deliberately no single-page "per port" mode here. `ambe plots basic`
(src/ambe/plots/basic.py) already groups by source position and emits the per-port
position book for whichever selection its config names, so the port-position view is
produced by the established pipeline for BOTH decks rather than by a second code path
that only ever knew about the box cuts.

WHY THE FEATURE PLOTS ARE NOT SIMPLY 'PER STREAMLINE'. The 31 features are IDENTICAL
between the two streamlines: both score the same AmBe clusters with the same features
derived the same way, and the streamline changes only which MC labelling trained the
model — hence the score, hence the selection. So the all-Stage-2 feature distributions
are streamline-INDEPENDENT and are written once, labelled as such. The streamline only
becomes meaningful once the MVA cut is applied, which is what the per-streamline files
show. Writing two identical all-cluster files would have implied a difference that is
not there.

Fits use the anchor's own recipe, imported (not reimplemented) from
fit_capture_time_fprompt_compare: 70 bins over 0-70 us, 2-67 us window,
A(1-exp(-t/therm))exp(-t/tau) with B FIXED at 0. The per-page tau therefore matches
CaptureTimeFits_v4_all28.csv exactly, and the weighted mean over the 26 pages
reproduces 29.477 +- 0.193 us — which the byposition mode asserts at the end of its
run. The window was 10-67 before 2026-08-27, where the same quantity was
30.535 +- 0.228; the two are NOT comparable.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    MPLBACKEND=Agg python -u boxcut_v4_appendix.py --do byposition
    MPLBACKEND=Agg python -u boxcut_v4_appendix.py --do features
    MPLBACKEND=Agg python -u boxcut_v4_appendix.py --do positiongrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from make_plots_ccinc_v3_merged import BLUE, GREY, bare
from fit_capture_time_fprompt_compare import fit, prepare, NeutCapture
import boxcut_v4_campaign as C

HERE = Path(__file__).parent
OUT = HERE / "PRESENTATION_PLOTS"

RED = "#b2182b"

# The leading model features in the merged bundle's own order, plus the three
# ClusterFinder summary quantities the box cut is drawn in, plus the score itself.
# Labels follow make_presentable_optics_rf_plots.FEATURE_LABELS — note d_wall and
# vtx_y are METRES (the June-16 poster labels d_wall in cm and is wrong).
FEATURES = [
    ("n_hits",            "Cluster Hits",                    None),
    ("pe_total",          "Total PE [p.e.]",                 (0, 120)),
    ("n_hits_early",      "Early Hits",                      None),
    ("sigma_t_mad",       r"$\sigma_t$ (MAD) [ns]",          (0, 20)),
    ("t_window_80pct",    "80% Time Window [ns]",            (0, 60)),
    ("charge_bal_legacy", "Charge Balance",                   (0, 0.6)),
    ("pe_balance",        "Charge Balance (PE)",              (0, 1)),
    ("spatial_rms",       "Spatial RMS of Hit PMTs [m]",      None),
    ("d_wall",            "Distance to Wall [m]",             (0, 1.6)),
    ("vtx_y",             "Vertex Y [m]",                     (-2, 2)),
    ("beta1",             r"$\beta_1$",                       None),
    ("beta2",             r"$\beta_2$",                       None),
    ("n_fit_hits",        "Hits Used in the Vertex Fit",      None),
    ("fit_rms_ns",        "Fit Timing RMS [ns]",              (0, 20)),
    ("fit_goodness_init", "Fit Goodness (Centroid Vertex)",   None),
    ("cf_clusterPE",      "ClusterFinder PE",                 (0, 110)),
    ("cf_clusterCB",      "ClusterFinder Charge Balance",     (0, 0.5)),
    ("cf_clusterHits",    "ClusterFinder Hits",               (0, 60)),
    ("gbt_score",         "GBT score",                        (0, 1)),
]

# One colour per y position within a port panel. Six entries because port 3 has six.
Y_COLORS = ["#08519c", "#3182bd", "#6baed6", "#e08214", "#b2182b", "#4d4d4d"]


# ════════════════════════════════════════════════════════════════════════════
# loaders
# ════════════════════════════════════════════════════════════════════════════
def load_all():
    """The 28 per-run box-cut candidate CSVs, pooled. clusterTime in MICROSECONDS."""
    frames = []
    for tag in (C.TAG_GATED, C.TAG_EXT):
        for p in sorted(C.CAND.glob(f"EventAmBeNeutronCandidates_{tag}_*.csv")):
            d = pd.read_csv(p, usecols=["clusterTime", "clusterPE",
                                        "clusterChargeBalance", "clusterHits",
                                        "sourceX", "sourceY", "sourceZ",
                                        "eventTankTime"])
            d["run"], d["tag"] = int(p.stem.rsplit("_", 1)[1]), tag
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["clusterTime"] = d.clusterTime / 1000.0
    d["pos"] = list(zip(d.sourceX, d.sourceY, d.sourceZ))
    return d


def _scored(stream, bc=None):
    """Scored AmBe clusters for one streamline, Stage-2 selected, joined to positions.

    The join is on (run, eventTankTime, clusterTime in ns) — the same exact key used by
    boxcut_v4_campaign.do_matched, so this sees the identical 225,810-cluster set.
    """
    f = (C.SCORED_DIR / f"{stream}_cf_gated" /
         f"ambev4neutron_{stream}_cf_gated__scored.parquet")
    if not f.exists():
        raise SystemExit(f"missing {f}")
    keep = {"run", "event_tank_time", "cf_clusterTime", "cf_clusterPE",
            "cf_clusterCB", "cf_clusterHits", "gbt_score"}
    keep |= {c for c, _, _ in FEATURES}
    m = pd.read_parquet(f, columns=sorted(keep))
    m = m[(m.cf_clusterPE > 0) & (m.cf_clusterPE <= 100)
          & (m.cf_clusterCB > 0) & (m.cf_clusterCB < 0.45)
          & (m.cf_clusterTime >= 2000) & (m.cf_clusterHits >= 5)].copy()
    m["key"] = list(zip(m.run, m.event_tank_time.astype("int64"),
                        np.round(m.cf_clusterTime, 3)))
    if bc is None:
        bc = load_all()
    bc = bc.copy()
    bc["key"] = list(zip(bc.run, bc.eventTankTime.astype("int64"),
                         np.round(bc.clusterTime * 1000.0, 3)))
    pos = bc.drop_duplicates("key").set_index("key")[["sourceX", "sourceY", "sourceZ"]]
    m = m.join(pos, on="key")
    n_before = len(m)
    m = m[m.sourceY.notna()].copy()
    if len(m) != n_before:
        print(f"    note: {n_before - len(m):,} scored clusters had no box-cut match "
              f"and carry no source position; dropped")
    m["pos"] = list(zip(m.sourceX, m.sourceY, m.sourceZ))
    m["port"] = m["pos"].map(lambda p: C.PORT_INFO.get(
        tuple(int(v) if float(v).is_integer() else v for v in p), "Unknown"))
    return m


def _hist_range(d, col, rng):
    """Explicit range if given, else the central 99% so tails do not flatten a page."""
    if rng is not None:
        return rng
    v = d[col].replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return (0, 1)
    lo, hi = float(np.nanpercentile(v, 0.5)), float(np.nanpercentile(v, 99.5))
    return (lo, hi) if hi > lo else (lo, lo + 1)


# ════════════════════════════════════════════════════════════════════════════
# --do byposition
# ════════════════════════════════════════════════════════════════════════════
def page(pdf, pos, g, eff, eff_err, port, runs):
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8))
    r = fit(g.clusterTime.values, float_B=False, backend="scipy")

    ax = axes[0, 0]
    x, _y, _e = prepare(g.clusterTime.values)
    ax.hist(g.clusterTime, bins=70, range=(0, 70), histtype="step", color=BLUE, lw=1.3)
    if r:
        ax.plot(x, NeutCapture(x, r["A"], r["therm"], r["tau"], 0.0), "-",
                color=RED, lw=1.6)
        ax.set_title(rf"$\tau = {r['tau']:.2f}\pm{r['tau_err']:.2f}\ \mu s$, "
                     rf"$\chi^2/\mathrm{{ndof}} = {r['redchi']:.2f}$", fontsize=10)
    else:
        ax.set_title("fit failed / too few counts", fontsize=10)
    ax.set_xlabel(r"cluster time [$\mu$s]")
    ax.set_ylabel("clusters")
    bare(ax)

    for ax, (col, lbl, rng, logy) in zip(
            [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]],
            [("clusterPE", "cluster PE", (0, 110), False),
             ("clusterChargeBalance", "charge balance", (0, 0.5), False),
             ("clusterHits", "cluster hits", (0, 60), False),
             ("clusterTime", r"cluster time [$\mu$s]", (0, 70), True)]):
        ax.hist(g[col], bins=55, range=rng, histtype="step", color=BLUE, lw=1.3)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(lbl)
        ax.set_ylabel("clusters")
        bare(ax)

    ax = axes[1, 2]
    ax.axis("off")
    mult = g.groupby(["run", "eventTankTime"]).size()
    txt = (f"position  (x, y, z) = {tuple(int(v) for v in pos)} cm\n"
           f"{port}\n"
           f"runs      {', '.join(str(x) for x in runs)}\n\n"
           f"Stage-2 candidates    {len(g):,}\n"
           f"events with >=1 cand  {mult.shape[0]:,}\n"
           f"mean cand / event     {mult.mean():.3f}\n"
           f"efficiency            {100*eff:.2f} +/- {100*eff_err:.2f} %\n\n"
           f"median PE             {g.clusterPE.median():.1f}\n"
           f"median CB             {g.clusterChargeBalance.median():.3f}\n"
           f"median hits           {g.clusterHits.median():.0f}\n")
    ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=10.5, family="monospace")

    fig.suptitle(f"AmBe v4 box cuts — position {tuple(int(v) for v in pos)}, {port}, "
                 f"efficiency {100*eff:.2f} %", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return r


def run_byposition():
    summ = C.load_summaries()
    C.check_merge(summ)
    d = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "APPENDIX_boxcut_v4_by_position.pdf"

    order = summ.sort_values("efficiency")          # worst first, on purpose
    taus = []
    with PdfPages(out) as pdf:
        for _, row in order.iterrows():
            g = d[d.pos == row.pos]
            if g.empty:
                print(f"  {row.pos}: no candidates, skipped")
                continue
            runs = sorted(g.run.unique())
            r = page(pdf, row.pos, g, row.efficiency, row.eff_err, row.port, runs)
            if r:
                taus.append((row.pos, r["tau"], r["tau_err"]))
                print(f"  page {len(taus):2d}  {str(row.pos):>16s} {row.port:8s} "
                      f"eff {100*row.efficiency:5.2f} %  n={len(g):>6,d}  "
                      f"tau={r['tau']:.2f}+-{r['tau_err']:.2f}")
    t = pd.DataFrame(taus, columns=["pos", "tau", "tau_err"])
    w = 1.0 / t.tau_err ** 2
    mu, er = (t.tau * w).sum() / w.sum(), float(np.sqrt(1 / w.sum()))
    print(f"\nwrote {out}  ({len(t)} pages)")
    print(f"  weighted tau over pages: {mu:.3f} +- {er:.3f} us")
    assert abs(mu - 29.477) < 0.01 and abs(er - 0.193) < 0.01, (
        "appendix fits do not reproduce the headline 29.477 +- 0.193 — the recipe "
        "drifted and the appendix is not the same measurement as G4")
    print("  PASS  reproduces the headline 29.477 +- 0.193 us")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# --do features   (one page per feature, one panel per port, one curve per y)
# ════════════════════════════════════════════════════════════════════════════
def features_pdf(m, out, header):
    ports = [p for p in C.PORT_ORDER if (m.port == p).any()]
    with PdfPages(out) as pdf:
        for col, lbl, rng in FEATURES:
            if col not in m:
                continue
            lo, hi = _hist_range(m, col, rng)
            fig, axes = plt.subplots(1, len(ports), figsize=(3.05 * len(ports), 3.5),
                                     sharey=True)
            axes = np.atleast_1d(axes)
            for ax, port in zip(axes, ports):
                s = m[m.port == port]
                for i, yv in enumerate(sorted(s.sourceY.unique())):
                    v = s.loc[s.sourceY == yv, col]
                    v = v.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(v) < 20:
                        continue
                    ax.hist(v, bins=45, range=(lo, hi), histtype="step", lw=1.3,
                            density=True, color=Y_COLORS[i % len(Y_COLORS)],
                            label=f"y={yv:g} (n={len(v):,})")
                ax.set_xlabel(lbl, fontsize=9)
                ax.set_title(port, fontsize=10)
                ax.legend(frameon=False, fontsize=6.8)
                bare(ax)
            axes[0].set_ylabel("normalised", fontsize=9)
            fig.suptitle(f"{lbl} by source position — {header}", fontsize=11.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"    page: {col}")
    print(f"  wrote {out}")


def run_features():
    OUT.mkdir(parents=True, exist_ok=True)
    bc = load_all()
    m = _scored("truthtag", bc)
    print(f"[features] {len(m):,} Stage-2 clusters, {m.pos.nunique()} positions, "
          f"{m.port.nunique()} ports")
    features_pdf(m, OUT / "APPENDIX_features_by_position_allStage2.pdf",
                 "all Stage-2 clusters (identical for both streamlines)")
    for stream, label in C.STREAMS.items():
        mm = _scored(stream, bc)
        thr, _ = C._thresholds(label)[("GBT", "eff80")]
        sel = mm[mm.gbt_score > thr]
        print(f"[features] {label}: GBT eff80 threshold {thr:.6f}, "
              f"{len(sel):,} of {len(mm):,} kept ({100*len(sel)/len(mm):.2f}%)")
        features_pdf(sel,
                     OUT / f"APPENDIX_features_by_position_{stream}_mvaneutron.pdf",
                     f"MVA neutron @ GBT eff80 — {label}")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# --do positiongrid   (one page per position, MVA neutron vs MVA other)
# ════════════════════════════════════════════════════════════════════════════
def positiongrid_pdf(m, thr, out, header):
    order = m.groupby("pos").size().sort_values(ascending=False).index
    cols = [(c, l, r) for c, l, r in FEATURES if c in m][:12]
    with PdfPages(out) as pdf:
        for pos in order:
            s = m[m.pos == pos]
            kept, rej = s[s.gbt_score > thr], s[s.gbt_score <= thr]
            fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.4))
            for ax, (col, lbl, rng) in zip(axes.ravel(), cols):
                lo, hi = _hist_range(s, col, rng)
                for d_, c_, lab in ((kept, BLUE, f"MVA neutron ({len(kept):,})"),
                                    (rej, RED, f"MVA other ({len(rej):,})")):
                    v = d_[col].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(v) < 20:
                        continue
                    ax.hist(v, bins=40, range=(lo, hi), histtype="step", lw=1.3,
                            density=True, color=c_, label=lab)
                ax.set_xlabel(lbl, fontsize=8.5)
                bare(ax)
            axes[0, 0].legend(frameon=False, fontsize=7.5)
            axes[0, 0].set_ylabel("normalised", fontsize=8.5)
            port = C.PORT_INFO.get(tuple(int(v) for v in pos), "Unknown")
            fig.suptitle(f"position {tuple(int(v) for v in pos)}, {port} — "
                         f"{len(s):,} Stage-2 clusters — {header}", fontsize=11.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"    page: {tuple(int(v) for v in pos)}  n={len(s):,}")
    print(f"  wrote {out}")


def run_positiongrid():
    OUT.mkdir(parents=True, exist_ok=True)
    bc = load_all()
    for stream, label in C.STREAMS.items():
        mm = _scored(stream, bc)
        thr, _ = C._thresholds(label)[("GBT", "eff80")]
        print(f"[positiongrid] {label}: threshold {thr:.6f}, {len(mm):,} clusters")
        positiongrid_pdf(mm, thr, OUT / f"APPENDIX_positiongrid_{stream}.pdf",
                         f"MVA neutron vs other @ GBT eff80 — {label}")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="boxcut_v4_appendix")
    # No default: the modes write different documents and take very different
    # amounts of time; picking one silently is how you get the wrong appendix.
    ap.add_argument("--do", required=True,
                    choices=["byposition", "features", "positiongrid"])
    a = ap.parse_args()
    return {"byposition": run_byposition,
            "features": run_features,
            "positiongrid": run_positiongrid}[a.do]()


if __name__ == "__main__":
    sys.exit(main())
