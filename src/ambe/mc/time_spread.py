"""
Hit-time spread comparison: ClusterFinder clusters vs truth DirectParent neutrons.

Reads directly from ROOT (does not require the processor parquet step) so that
the per-cluster hit-time arrays (Cluster_HitT) are preserved exactly as
ClusterFinder assigned them.

For every event with at least one cluster we compute per-CF-cluster and per-
truth-neutron:
  delta_t  = t_max - t_min  of member hit times
  sigma_t  = std of member hit times

Summary outputs
---------------
  <csv>/<run>__time_spread.csv          per-cluster row table
  <plots>/ts_delta_t_distribution       delta_t CF vs truth (all clusters)
  <plots>/ts_sigma_t_distribution       sigma_t CF vs truth
  <plots>/ts_example_event_NNN          hit-time strip per cluster, one page
                                        per selected multi-cluster event
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs
from ..plotting import save_plot, set_style

NEUTRON_PDG = 2112
BRANCHES = [
    "eventNumber",
    "numberOfClusters",
    "clusterTime", "clusterPE", "clusterHits",
    "Cluster_HitT",
    "DirectParent_HitTime",
    "DirectParent_NeutronAncestorTrackID",
    "DirectParent_NeutronAncestorPDG",
    "DirectParent_NeutronAncestorClass",
]


# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #

def _truth_neutron_clusters(ht, tids, pdgs):
    """
    Group DirectParent hit times by trackID (PDG=2112 only).
    Returns dict: trackID -> np.array of hit times.
    """
    ht   = np.asarray(ht, dtype=float)
    tids = np.asarray(tids, dtype=int)
    pdgs = np.asarray(pdgs, dtype=int)
    mask = (pdgs == NEUTRON_PDG) & (tids != -5)
    clusters = {}
    for tid in np.unique(tids[mask]):
        clusters[int(tid)] = ht[mask & (tids == tid)]
    return clusters


def _spread(times):
    """Return (delta_t, sigma_t) for an array of times. NaN for <2 hits."""
    if len(times) < 2:
        return np.nan, np.nan
    return float(np.ptp(times)), float(np.std(times))


def extract_spreads(root_path: Path) -> pd.DataFrame:
    """
    Read one ROOT file, return a DataFrame with one row per cluster
    (both CF and truth) with columns:
      eventID, source (cf|truth), cluster_id, n_hits,
      t_min, t_max, delta_t, sigma_t, clusterPE (CF only), trackID (truth only)
    """
    rows = []
    with uproot.open(str(root_path)) as f:
        if "Event" not in f:
            raise KeyError(f"No 'Event' tree in {root_path}")
        arr = f["Event"].arrays(BRANCHES, library="ak")

    for i in range(len(arr)):
        evid = int(arr["eventNumber"][i])
        nc   = int(arr["numberOfClusters"][i])

        # --- CF clusters ---
        cf_hit_times = ak.to_list(arr["Cluster_HitT"][i])     # list of lists
        cf_times_raw = ak.to_list(arr["clusterTime"][i])
        cf_pe        = ak.to_list(arr["clusterPE"][i])
        cf_hits      = ak.to_list(arr["clusterHits"][i])

        for cidx in range(nc):
            hits = np.asarray(cf_hit_times[cidx], dtype=float) if cidx < len(cf_hit_times) else np.array([])
            dt, st = _spread(hits)
            rows.append({
                "eventID":    evid,
                "source":     "cf",
                "cluster_id": cidx,
                "n_hits":     len(hits),
                "t_min":      float(hits.min()) if len(hits) else np.nan,
                "t_max":      float(hits.max()) if len(hits) else np.nan,
                "delta_t":    dt,
                "sigma_t":    st,
                "clusterTime": float(cf_times_raw[cidx]) if cidx < len(cf_times_raw) else np.nan,
                "clusterPE":  float(cf_pe[cidx]) if cidx < len(cf_pe) else np.nan,
                "clusterHits": int(cf_hits[cidx]) if cidx < len(cf_hits) else 0,
                "trackID":    np.nan,
            })

        # --- truth neutron clusters ---
        truth = _truth_neutron_clusters(
            arr["DirectParent_HitTime"][i],
            arr["DirectParent_NeutronAncestorTrackID"][i],
            arr["DirectParent_NeutronAncestorPDG"][i],
        )
        for tid, hits in truth.items():
            dt, st = _spread(hits)
            rows.append({
                "eventID":    evid,
                "source":     "truth",
                "cluster_id": tid,
                "n_hits":     len(hits),
                "t_min":      float(hits.min()) if len(hits) else np.nan,
                "t_max":      float(hits.max()) if len(hits) else np.nan,
                "delta_t":    dt,
                "sigma_t":    st,
                "clusterTime": np.nan,
                "clusterPE":  np.nan,
                "clusterHits": np.nan,
                "trackID":    tid,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary plots
# --------------------------------------------------------------------------- #

def _dist_plot(df: pd.DataFrame, col: str, xlabel: str, ctx: RunContext, fname: str):
    set_style()
    cf    = df[df["source"] == "cf"][col].dropna()
    truth = df[df["source"] == "truth"][col].dropna()

    vmax = np.percentile(np.concatenate([cf.values, truth.values]), 99)
    bins = np.linspace(0, vmax, 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cf,    bins=bins, alpha=0.6, density=True, color="steelblue",
            label=f"CF clusters  (n={len(cf)})")
    ax.hist(truth, bins=bins, alpha=0.6, density=True, color="tomato",
            label=f"Truth neutrons (n={len(truth)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(ctx.title(xlabel + " per cluster"))
    ax.legend()
    save_plot(fig, ctx, fname)


def summary_plots(df: pd.DataFrame, ctx: RunContext):
    _dist_plot(df, "delta_t", "Hit-time spread  Δt = t_max − t_min  (ns)", ctx, "ts_delta_t_distribution")
    _dist_plot(df, "sigma_t", "Hit-time spread  σ_t = std(hit times)  (ns)", ctx, "ts_sigma_t_distribution")

    # Scatter: CF delta_t vs matched truth delta_t (events with exactly 1 CF + 1 truth)
    set_style()
    cf_ev    = df[df["source"] == "cf"].groupby("eventID")
    truth_ev = df[df["source"] == "truth"].groupby("eventID")
    pairs = []
    for evid, cf_grp in cf_ev:
        if evid not in truth_ev.groups:
            continue
        tr_grp = truth_ev.get_group(evid)
        if len(cf_grp) == 1 and len(tr_grp) == 1:
            pairs.append((float(cf_grp["delta_t"].iloc[0]),
                          float(tr_grp["delta_t"].iloc[0])))
    if pairs:
        cx, ty = zip(*pairs)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(cx, ty, s=10, alpha=0.4, color="steelblue")
        lim = max(np.nanpercentile(cx, 99), np.nanpercentile(ty, 99)) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="CF = truth")
        ax.set_xlabel("CF cluster Δt (ns)")
        ax.set_ylabel("Truth neutron Δt (ns)")
        ax.set_title(ctx.title("CF vs truth Δt  (1-cluster events)"))
        ax.legend()
        save_plot(fig, ctx, "ts_delta_t_scatter")


# --------------------------------------------------------------------------- #
# Per-event example plots (multi-cluster events)
# --------------------------------------------------------------------------- #

def _example_event_plot(event_cf: list, event_truth: dict, evid: int,
                        ctx: RunContext):
    """
    Strip chart of hit times per cluster for one event.
    Top panel: CF clusters (one row per cluster, colored by cluster index).
    Bottom panel: truth neutrons (one row per trackID).
    """
    set_style()
    n_cf    = len(event_cf)
    n_truth = len(event_truth)
    if n_cf == 0 and n_truth == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    fig.suptitle(ctx.title(f"Event {evid} — hit times per cluster"), fontsize=12)

    cf_colors = cm.tab10(np.linspace(0, 1, max(n_cf, 1)))

    # --- CF panel ---
    ax = axes[0]
    all_cf_t = []
    for cidx, hits in enumerate(event_cf):
        if len(hits) == 0:
            continue
        y = np.full(len(hits), cidx)
        ax.scatter(hits, y, s=12, alpha=0.7, color=cf_colors[cidx % len(cf_colors)],
                   label=f"CF cluster {cidx}  (n={len(hits)})")
        all_cf_t.extend(hits)

    ax.set_ylabel("CF cluster index")
    ax.set_yticks(range(n_cf))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("ClusterFinder clusters", fontsize=10)
    if all_cf_t:
        margin = max(1.0, (max(all_cf_t) - min(all_cf_t)) * 0.05)
        ax.set_xlim(min(all_cf_t) - margin, max(all_cf_t) + margin)

    # --- Truth panel ---
    ax = axes[1]
    truth_colors = cm.tab10(np.linspace(0, 0.5, max(n_truth, 1)))
    all_truth_t = []
    for row, (tid, hits) in enumerate(event_truth.items()):
        if len(hits) == 0:
            continue
        y = np.full(len(hits), row)
        ax.scatter(hits, y, s=12, alpha=0.7, color=truth_colors[row % len(truth_colors)],
                   label=f"trackID {tid}  (n={len(hits)})")
        all_truth_t.extend(hits)

    ax.set_ylabel("Truth neutron index")
    ax.set_yticks(range(n_truth))
    ax.set_xlabel("Hit time (ns)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Truth neutron clusters (DirectParent trackID)", fontsize=10)
    if all_truth_t:
        margin = max(1.0, (max(all_truth_t) - min(all_truth_t)) * 0.05)
        ax.set_xlim(min(all_truth_t) - margin, max(all_truth_t) + margin)

    plt.tight_layout()
    save_plot(fig, ctx, f"ts_example_event_{evid:05d}", subdir="event_examples")


def example_event_plots(root_path: Path, ctx: RunContext,
                        n_examples: int = 12, min_clusters: int = 2):
    """
    Read the ROOT file a second time and plot hit-time strip charts for the
    first `n_examples` events with >= min_clusters CF clusters.
    """
    with uproot.open(str(root_path)) as f:
        arr = f["Event"].arrays(BRANCHES, library="ak")

    plotted = 0
    for i in range(len(arr)):
        nc = int(arr["numberOfClusters"][i])
        if nc < min_clusters:
            continue
        evid = int(arr["eventNumber"][i])
        cf_hit_times = ak.to_list(arr["Cluster_HitT"][i])
        truth = _truth_neutron_clusters(
            arr["DirectParent_HitTime"][i],
            arr["DirectParent_NeutronAncestorTrackID"][i],
            arr["DirectParent_NeutronAncestorPDG"][i],
        )
        _example_event_plot(
            [np.asarray(cf_hit_times[c], dtype=float) for c in range(nc)],
            {tid: hits for tid, hits in truth.items()},
            evid, ctx,
        )
        plotted += 1
        if plotted >= n_examples:
            break
    print(f"[timespread] wrote {plotted} example event plots -> "
          f"{ctx.plots_dir / 'event_examples'}/")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run(ctx: RunContext, n_examples: int = 12, verbose: bool = True) -> Path:
    root_files = resolve_inputs(ctx.inputs["root_files"])
    if len(root_files) != 1:
        raise ValueError("ambe mc timespread expects exactly one root_files entry")
    root_path = root_files[0]

    if verbose:
        print(f"[timespread] reading {root_path.name}")

    df = extract_spreads(root_path)

    out_csv = ctx.csv_path(f"{ctx.run_name}__time_spread")
    df.to_csv(out_csv, index=False)
    if verbose:
        cf    = df[df["source"] == "cf"]
        truth = df[df["source"] == "truth"]
        print(f"[timespread] {len(cf)} CF clusters | {len(truth)} truth neutron clusters")
        print(f"[timespread] CF    Δt  mean={cf['delta_t'].mean():.1f} ns  median={cf['delta_t'].median():.1f} ns")
        print(f"[timespread] Truth Δt  mean={truth['delta_t'].mean():.1f} ns  median={truth['delta_t'].median():.1f} ns")
        print(f"[timespread] wrote {out_csv}")

    summary_plots(df, ctx)
    example_event_plots(root_path, ctx, n_examples=n_examples)

    return out_csv


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc timespread")
    p.add_argument("--examples", type=int, default=12,
                   help="Number of multi-cluster event strip charts to produce")
    args = p.parse_args(list(argv) if argv else [])
    run(ctx, n_examples=args.examples)
