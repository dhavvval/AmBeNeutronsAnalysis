"""
Truth-driven hit-time windows for CCinc clusters.

Question
--------
The production cluster tag is a composition majority vote: a cluster is signal iff a
single neutron trackID owns >= MIN_MATCH_FRAC of its hits (`cluster_features.
_truth_label_cluster`).  Nothing in that rule looks at *when* the hits arrived.  This
module asks whether the time structure of the hits, measured against a physics t0,
separates neutron-capture light from background light better than composition does --
and, if so, where the boundary actually sits.  The ~50 ns figure that motivated the
study is a hypothesis inherited from `src/ambe/diagnostics/analyze_truth_timing.py` (AmBe MC), not a result.

Two scales, measured separately
-------------------------------
2a  Event level (microseconds).  Per-hit time against the event's neutron-capture
    reference.  This is where the prompt/delayed split at 10 us already lives.
2b  Intra-cluster (nanoseconds).  Per-hit dt against the cluster's own t0, split by
    where the hit came from.  This is the scale the 50 ns hypothesis is about.

Hit categories (per cluster, relative to that cluster's dominant neutron)
------------------------------------------------------------------------
  dominant_neutron  hits carrying the cluster's dominant neutron trackID
  other_neutron     hits from a DIFFERENT neutron in the same cluster
  nonneutron        truth_class == -5, real non-neutron physics
  darknoise         truth_class == 0
  untraced          BackTracker could not place the hit

`other_neutron` is the category that matters and the one the production tag cannot
see: Stage 1 shows most CCinc clusters carry more than one neutron, so the decisive
question is not "neutron vs background" but "can a time cut separate two neutrons
that the clusterer merged".  If the two neutrons' captures are tens of nanoseconds
apart a window helps; if they are coincident, no window can help and the merge has to
be accepted or fixed in the clusterer.

t0 choice
---------
Both the median and the first-hit t0 are computed and reported.  `src/ambe/diagnostics/analyze_truth_timing.py`
uses the median, which is robust but is itself pulled by a second neutron's light --
exactly the effect under study -- so the first-hit reference is carried alongside it
rather than being chosen on faith.

Outputs
-------
  <csv>/<run>__htw_dt_histogram.csv     binned dt vs category, per method/tag
  <csv>/<run>__htw_window_scan.csv      purity/efficiency vs candidate window edge
  <csv>/<run>__htw_neutron_pair_dt.csv  capture-time separation of merged neutrons
  <parquet>/<run>__htw_clusters.parquet per-cluster timing summary
  plots: htw_dt_by_category, htw_window_scan, htw_neutron_pair_dt
"""

from __future__ import annotations

import argparse as _argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext
from ..plotting import save_plot, set_style
from .optics import (run_optics_on_event, _apply_truth_window, _prefilter_hits,
                     assign_clusterfinder_labels, MIN_MATCH_FRAC, NEUTRON_PDG)

CATEGORIES = ("dominant_neutron", "other_neutron", "nonneutron", "darknoise", "untraced")

# dt binning for the intra-cluster study. Fine below 100 ns because that is where the
# hypothesis lives; the long tail is kept so thermalization scatters stay visible
# instead of piling into an overflow bin that reads as zero.
DT_EDGES = np.concatenate([
    np.arange(-200.0, -50.0, 25.0),
    np.arange(-50.0, 200.0, 5.0),
    np.arange(200.0, 1000.0, 50.0),
    np.array([1000.0, 2000.0, 5000.0, 20000.0, 1e9]),
])

# candidate |dt| window edges scanned for the purity/efficiency trade-off
WINDOW_EDGES = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150,
                         200, 300, 500, 1000], dtype=float)


def _categorise(df_ev, mask, dom_tid):
    """Per-hit category array for the member hits of one cluster."""
    tids = df_ev["ancestor_trackID"].to_numpy(int)[mask]
    pdgs = df_ev["ancestor_pdg"].to_numpy(int)[mask]
    cls = df_ev["truth_class"].to_numpy(int)[mask]
    unt = ((df_ev["is_untraced"].to_numpy(int)[mask]) if "is_untraced" in df_ev.columns
           else np.zeros(int(mask.sum()), int))

    cat = np.full(len(cls), "nonneutron", dtype=object)
    is_n = (pdgs == NEUTRON_PDG) & (tids >= 0)
    cat[is_n] = "other_neutron"
    if dom_tid >= 0:
        cat[is_n & (tids == dom_tid)] = "dominant_neutron"
    cat[cls == 0] = "darknoise"
    # untraced last: it is a statement about the BackTracker, not the physics, and must
    # not be overwritten by the truth_class==0 default those hits also carry.
    cat[unt == 1] = "untraced"
    return cat


def extract(ctx: RunContext, method: str, max_groups: Optional[int] = None):
    """Single pass over the pulses table; returns (hist, clusters, pairs)."""
    import pyarrow.parquet as pq
    from . import cc_selection

    feat = ctx.extra.get("features", {}) or {}
    ms = int(feat.get("min_samples", 8))
    xi = float(feat.get("xi", 0.02))
    t_unit = float(feat.get("t_unit_ns", 25.0))
    tw = float(feat.get("truth_window_ns", 75.0))
    prefilter = float(feat.get("hit_prefilter_ns", 0.0))

    pf = pq.ParquetFile(str(ctx.parquet_path(f"{ctx.run_name}__pulses")))
    # x/y/z/pe are only used by run_optics_on_event, but are read for both methods so
    # the two passes see byte-identical input and any difference is the clusterer's.
    cols = ["eventID", "t", "x", "y", "z", "pe", "truth_class", "is_neutron",
            "is_untraced", "ancestor_trackID", "ancestor_pdg", "cc_pass",
            "origin_in_tank"]
    read_cols = [c for c in cols if c in set(pf.schema_arrow.names)]
    cpath = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    clusters = pd.read_parquet(cpath) if cpath.exists() else None

    n_groups = pf.num_row_groups
    todo = range(n_groups if not max_groups else min(max_groups, n_groups))
    if max_groups and max_groups < n_groups:
        print(f"[htw] *** PARTIAL SAMPLE: {max_groups}/{n_groups} groups. Not quotable. ***")

    # (tag, category) -> counts per dt bin, accumulated rather than stored per hit:
    # 2.3e8 hits will not fit in memory and do not need to.
    hist = {}
    cl_rows, pair_rows = [], []

    for i_grp in todo:
        chunk = pf.read_row_group(i_grp, columns=read_cols).to_pandas()
        chunk = cc_selection.apply_residual_filter(chunk, ctx, verbose=False)
        if not len(chunk):
            continue
        for evid in chunk["eventID"].unique():
            df_ev = chunk[chunk["eventID"] == evid].reset_index(drop=True)
            df_cl = (clusters[clusters["eventID"] == evid].reset_index(drop=True)
                     if clusters is not None else None)
            truth_mask = _apply_truth_window(df_ev, tw)
            t_all = df_ev["t"].to_numpy(float)
            tids_all = df_ev["ancestor_trackID"].to_numpy(int)
            pdgs_all = df_ev["ancestor_pdg"].to_numpy(int)

            if method == "optics":
                df_o = _prefilter_hits(df_ev, df_cl, prefilter) if prefilter > 0 else df_ev
                lbls = run_optics_on_event(df_o, min_samples=ms, xi=xi, t_unit_ns=t_unit)
                if len(df_o) < len(df_ev):
                    full = np.full(len(df_ev), -1, int)
                    full[df_o.index] = lbls
                    lbls = full
            else:
                lbls = (assign_clusterfinder_labels(df_ev, df_cl) if df_cl is not None
                        else np.full(len(df_ev), -1, int))

            for cid in [c for c in np.unique(lbls) if c >= 0]:
                mask = (lbls == cid)
                n_hits = int(mask.sum())
                mt = t_all[mask]
                nm = mask & (pdgs_all == NEUTRON_PDG) & (tids_all >= 0)
                if nm.any():
                    u, c = np.unique(tids_all[nm], return_counts=True)
                    o = np.argsort(-c)
                    u, c = u[o], c[o]
                    dom_tid, n_dom = int(u[0]), int(c[0])
                else:
                    u, c = np.array([], int), np.array([], int)
                    dom_tid, n_dom = -1, 0

                truth_tids = set(int(t) for t, p, w in
                                 zip(tids_all, pdgs_all, truth_mask)
                                 if p == NEUTRON_PDG and t >= 0 and w)
                is_sig = int(dom_tid in truth_tids and n_dom / max(n_hits, 1) >= MIN_MATCH_FRAC)
                tag = "signal" if is_sig else "background"

                cat = _categorise(df_ev, mask, dom_tid)
                t0_med = float(np.median(mt))
                t0_first = float(mt.min())
                for ref, t0 in (("median", t0_med), ("first", t0_first)):
                    dt = mt - t0
                    for cname in CATEGORIES:
                        sel = (cat == cname)
                        if not sel.any():
                            continue
                        h, _ = np.histogram(dt[sel], bins=DT_EDGES)
                        key = (tag, ref, cname)
                        hist[key] = hist.get(key, 0) + h

                # capture-time separation between the neutrons sharing this cluster --
                # the quantity that decides whether ANY window can unmerge them
                if len(u) >= 2:
                    med_dom = float(np.median(t_all[nm & (tids_all == dom_tid)]))
                    for k in range(1, len(u)):
                        med_o = float(np.median(t_all[nm & (tids_all == int(u[k]))]))
                        pair_rows.append(dict(
                            eventID=int(evid), method=method, cluster_id=int(cid),
                            tag=tag, tid_a=dom_tid, tid_b=int(u[k]),
                            n_a=n_dom, n_b=int(c[k]),
                            dt_median_ns=abs(med_dom - med_o)))

                cl_rows.append(dict(
                    eventID=int(evid), method=method, cluster_id=int(cid),
                    tag=tag, n_hits=n_hits, dom_tid=dom_tid, n_dom=n_dom,
                    n_contrib_neutrons=len(u),
                    t0_median=t0_med, t0_first=t0_first,
                    dt_spread_ns=float(mt.max() - mt.min()),
                    n_dominant=int((cat == "dominant_neutron").sum()),
                    n_other_neutron=int((cat == "other_neutron").sum()),
                    n_nonneutron=int((cat == "nonneutron").sum()),
                    n_darknoise=int((cat == "darknoise").sum()),
                    n_untraced=int((cat == "untraced").sum())))

    hrows = []
    for (tag, ref, cname), h in hist.items():
        for lo, hi, n in zip(DT_EDGES[:-1], DT_EDGES[1:], h):
            if n:
                hrows.append(dict(tag=tag, t0_ref=ref, category=cname,
                                  dt_lo=lo, dt_hi=hi, n=int(n)))
    return (pd.DataFrame(hrows), pd.DataFrame(cl_rows), pd.DataFrame(pair_rows))


def window_scan(hist: pd.DataFrame, t0_ref: str = "median") -> pd.DataFrame:
    """Purity/efficiency of a |dt| <= edge cut, from the binned dt histogram.

    Signal = dominant_neutron hits.  `other_neutron` is counted separately and NOT
    folded into either side: whether a second neutron's light is signal or background
    depends on the physics question being asked, and collapsing it here would bake in
    an answer the data has not given.
    """
    rows = []
    h = hist[hist.t0_ref == t0_ref]
    for tag, g in h.groupby("tag"):
        tot = {c: g[g.category == c].n.sum() for c in CATEGORIES}
        for e in WINDOW_EDGES:
            inw = g[(g.dt_lo >= -e) & (g.dt_hi <= e)]
            k = {c: inw[inw.category == c].n.sum() for c in CATEGORIES}
            sig, bg = k["dominant_neutron"], k["nonneutron"] + k["darknoise"]
            rows.append(dict(
                tag=tag, t0_ref=t0_ref, window_ns=e,
                n_dominant=int(sig), n_other_neutron=int(k["other_neutron"]),
                n_nonneutron=int(k["nonneutron"]), n_darknoise=int(k["darknoise"]),
                n_untraced=int(k["untraced"]),
                eff_dominant=sig / max(tot["dominant_neutron"], 1),
                eff_other_neutron=k["other_neutron"] / max(tot["other_neutron"], 1),
                purity_vs_bg=sig / max(sig + bg, 1),
                purity_vs_all=sig / max(sum(k.values()), 1)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots -- house style: title only, no gridlines, no reference lines, no text
# --------------------------------------------------------------------------- #

def _p_dt_by_category(hist, ctx, t0_ref="median"):
    h = hist[(hist.t0_ref == t0_ref) & (hist.dt_lo >= -100) & (hist.dt_hi <= 300)]
    for tag, g in h.groupby("tag"):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for cname in CATEGORIES:
            s = g[g.category == cname].sort_values("dt_lo")
            if not len(s):
                continue
            ctr = 0.5 * (s.dt_lo + s.dt_hi)
            w = (s.dt_hi - s.dt_lo).to_numpy()
            ax.step(ctr, s.n / w, where="mid", label=cname)
        ax.set_yscale("log")
        ax.set_xlabel(f"hit time - cluster {t0_ref} t0 [ns]")
        ax.set_ylabel("hits / ns")
        ax.grid(False)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(ctx.title(f"Hit dt by origin, {tag} clusters, {t0_ref} t0"))
        save_plot(fig, ctx, f"htw_dt_by_category_{tag}_{t0_ref}")


def _p_window_scan(scan, ctx):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, g in scan.groupby("tag"):
        g = g.sort_values("window_ns")
        ax.plot(g.window_ns, g.purity_vs_bg, marker="o", ms=3, label=f"{tag}: purity")
        ax.plot(g.window_ns, g.eff_dominant, marker="s", ms=3, ls="--",
                label=f"{tag}: efficiency")
    ax.set_xscale("log")
    ax.set_xlabel("window half-width |dt| [ns]")
    ax.set_ylabel("fraction")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title("Dominant-neutron efficiency and purity vs window"))
    save_plot(fig, ctx, "htw_window_scan")


def _p_pair_dt(pairs, ctx):
    if not len(pairs):
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = np.logspace(0, 5, 60)
    for tag, g in pairs.groupby("tag"):
        ax.hist(np.clip(g.dt_median_ns, 1, None), bins=b, histtype="step", label=tag)
    med = float(pairs.dt_median_ns.median())
    below = float((pairs.dt_median_ns <= 50).mean())
    ax.set_xscale("log")
    ax.set_xlabel("|capture-time separation| of two neutrons in one cluster [ns]")
    ax.set_ylabel("pairs")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title(f"Merged-neutron separation, median {med:.0f} ns, "
                           f"{100*below:.1f}% within 50 ns"))
    save_plot(fig, ctx, "htw_neutron_pair_dt")


def run(ctx: RunContext, args) -> Path:
    set_style()
    hist, cl, pairs = extract(ctx, args.method, args.max_files)
    if hist.empty:
        raise SystemExit("[htw] nothing extracted")
    base = f"{ctx.run_name}__htw_{args.method}"
    hist.to_csv(ctx.csv_path(f"{base}_dt_histogram"), index=False)
    cl.to_parquet(ctx.parquet_path(f"{base}_clusters"), index=False)
    if len(pairs):
        pairs.to_csv(ctx.csv_path(f"{base}_neutron_pair_dt"), index=False)

    scans = pd.concat([window_scan(hist, r) for r in ("median", "first")],
                      ignore_index=True)
    scans.to_csv(ctx.csv_path(f"{base}_window_scan"), index=False)

    if not args.no_plots:
        _p_dt_by_category(hist, ctx, "median")
        _p_window_scan(scans[scans.t0_ref == "median"], ctx)
        _p_pair_dt(pairs, ctx)

    print(scans[scans.t0_ref == "median"].to_string(index=False))
    if len(pairs):
        q = np.percentile(pairs.dt_median_ns, [10, 25, 50, 75, 90])
        print(f"\n[htw] merged-neutron separation ns, p10/25/50/75/90: "
              f"{np.round(q, 1)}")
        print(f"[htw] pairs within 50 ns: {100*(pairs.dt_median_ns<=50).mean():.1f}%  "
              f"within 200 ns: {100*(pairs.dt_median_ns<=200).mean():.1f}%")
    return ctx.csv_path(f"{base}_window_scan")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = _argparse.ArgumentParser(prog="ambe mc timewindows")
    # No default, for the same reason as `ambe mc match --method`: the two clusterings
    # give different answers and a silent default lets the wrong one be quoted.
    p.add_argument("--method", required=True, choices=["clusterfinder", "optics"])
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(list(argv) if argv else [])
    run(ctx, a)
