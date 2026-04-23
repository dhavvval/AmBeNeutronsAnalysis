"""
Apple-to-apple comparison: ClusterFinder clusters vs truth neutron clusters.

Truth neutron clusters are defined exactly as in the ROOT macro
diagnose_neutrons_event_by_event.C: each unique DirectParent_NeutronAncestorTrackID
with PDG=2112 defines one truth neutron.  ClusterFinder clusters are matched to
truth neutrons by checking which trackID dominates the member pulses of each cluster.

Per cluster we record one of three outcomes:
  matched   -- CF cluster whose dominant member trackID is a truth neutron (PDG=2112)
  spurious  -- CF cluster whose members contain no truth neutron hits
  split     -- a truth neutron that is the dominant trackID of >1 CF cluster

Per event we record:
  n_truth    -- unique neutron trackIDs with >=1 hit
  n_cf       -- ClusterFinder cluster count
  n_matched  -- CF clusters matched 1-to-1 to a truth neutron
  n_spurious -- CF clusters with no truth neutron
  n_missed   -- truth neutrons not matched by any CF cluster

Outputs
-------
  <csv_dir>/<run>__matching.csv         per-event comparison table
  <csv_dir>/<run>__cluster_outcomes.csv per-cluster outcome table
  plots: multiplicity heatmap, outcome bar chart, matched-time scatter
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext
from ..plotting import save_plot, set_style
from .optics import assign_clusterfinder_labels

NEUTRON_PDG = 2112


# --------------------------------------------------------------------------- #
# Build truth cluster table from pulses
# --------------------------------------------------------------------------- #

def build_truth_clusters(pulses: pd.DataFrame) -> pd.DataFrame:
    """
    One row per unique (eventID, ancestor_trackID) where PDG=2112.
    Mirrors the ROOT macro: each unique neutron trackID is one truth cluster.
    """
    neutron_hits = pulses[
        (pulses["ancestor_pdg"] == NEUTRON_PDG) &
        (pulses["ancestor_trackID"] != -1)
    ].copy()

    if neutron_hits.empty:
        return pd.DataFrame(columns=[
            "eventID", "trackID", "truth_class",
            "n_hits", "t_min", "t_max", "delta_t",
        ])

    grp = neutron_hits.groupby(["eventID", "ancestor_trackID"])
    truth = grp.agg(
        truth_class=("truth_class", "first"),
        n_hits=("t", "count"),
        t_min=("t", "min"),
        t_max=("t", "max"),
    ).reset_index().rename(columns={"ancestor_trackID": "trackID"})
    truth["delta_t"] = truth["t_max"] - truth["t_min"]
    return truth


# --------------------------------------------------------------------------- #
# Match CF clusters to truth neutrons via dominant trackID
# --------------------------------------------------------------------------- #

def match_clusters(
    pulses: pd.DataFrame,
    clusters: pd.DataFrame,
    truth_clusters: pd.DataFrame,
    tmin_margin_ns: float = 5.0,
    tmax_margin_ns: float = 20.0,
    min_match_frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (event_summary, cluster_outcomes).

    cluster_outcomes has one row per CF cluster with columns:
      eventID, cluster_idx, clusterTime, clusterPE, clusterHits,
      dominant_trackID, dominant_frac, is_matched, matched_trackID

    event_summary has one row per event with:
      eventID, n_truth, n_cf, n_matched, n_spurious, n_missed, n_split
    """
    cluster_rows = []
    event_rows = []

    # Index truth clusters for fast lookup
    truth_idx = truth_clusters.set_index(["eventID", "trackID"])

    for evid in clusters["eventID"].unique():
        df_p = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        df_c = clusters[clusters["eventID"] == evid].reset_index(drop=True)
        df_t = truth_clusters[truth_clusters["eventID"] == evid]

        truth_trackIDs = set(df_t["trackID"].tolist())
        n_truth = len(truth_trackIDs)

        labels = assign_clusterfinder_labels(df_p, df_c, tmin_margin_ns, tmax_margin_ns)

        matched_truth_ids = set()
        split_truth_ids = set()
        n_spurious = 0

        for cidx, crow in df_c.iterrows():
            mask = labels == cidx
            members = df_p[mask]

            # Find which neutron trackID (PDG=2112) contributes most hits
            neutron_members = members[members["ancestor_pdg"] == NEUTRON_PDG]
            if not neutron_members.empty:
                tid_counts = neutron_members["ancestor_trackID"].value_counts()
                dom_tid = int(tid_counts.index[0])
                dom_frac = float(tid_counts.iloc[0] / len(members))
            else:
                dom_tid = -1
                dom_frac = 0.0

            # Require the neutron trackID to dominate the cluster by hit fraction,
            # not just be the most common among the neutron-only hits.
            is_matched = (dom_tid in truth_trackIDs) and (dom_frac >= min_match_frac)

            if is_matched:
                if dom_tid in matched_truth_ids:
                    split_truth_ids.add(dom_tid)
                matched_truth_ids.add(dom_tid)
            else:
                n_spurious += 1

            cluster_rows.append({
                "eventID":         evid,
                "cluster_idx":     int(crow["cluster_idx"]),
                "clusterTime":     float(crow["clusterTime"]),
                "clusterPE":       float(crow["clusterPE"]),
                "clusterHits":     int(crow["clusterHits"]),
                "n_member_pulses": int(mask.sum()),
                "dominant_trackID": dom_tid,
                "dominant_frac":   dom_frac,
                "is_matched":      is_matched,
            })

        n_missed = len(truth_trackIDs - matched_truth_ids)
        n_matched = len(matched_truth_ids)
        n_split = len(split_truth_ids)

        event_rows.append({
            "eventID":    evid,
            "n_truth":    n_truth,
            "n_cf":       len(df_c),
            "n_matched":  n_matched,
            "n_spurious": n_spurious,
            "n_missed":   n_missed,
            "n_split":    n_split,
            # True only when CF count equals truth count exactly, no spurious
            # clusters, and no neutron was split across multiple CF clusters.
            "agreement":  n_matched == n_truth and n_spurious == 0 and n_split == 0,
        })

    return pd.DataFrame(event_rows), pd.DataFrame(cluster_rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _multiplicity_heatmap(event_summary: pd.DataFrame, ctx: RunContext):
    """2-D heatmap: truth multiplicity vs CF cluster count (mirrors ROOT macro)."""
    set_style()
    max_val = max(event_summary["n_truth"].max(), event_summary["n_cf"].max()) + 1
    bins = np.arange(-0.5, max_val + 0.5, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    h, xedge, yedge = np.histogram2d(
        event_summary["n_truth"], event_summary["n_cf"], bins=[bins, bins]
    )
    im = ax.imshow(
        h.T, origin="lower", aspect="auto",
        extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
        cmap="Blues",
    )
    # Annotate cells
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if h[i, j] > 0:
                ax.text(i, j, int(h[i, j]), ha="center", va="center", fontsize=9,
                        color="black" if h[i, j] < h.max() * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Events")
    ax.set_xlabel("Truth neutron clusters (unique trackIDs)")
    ax.set_ylabel("ClusterFinder clusters")
    ax.set_title(ctx.title("CF vs truth multiplicity"))
    ax.plot([xedge[0], xedge[-1]], [yedge[0], yedge[-1]], "r--", lw=1, label="perfect agreement")
    ax.legend(fontsize=9)
    save_plot(fig, ctx, "matching_multiplicity_heatmap")


def _outcome_bar(event_summary: pd.DataFrame, ctx: RunContext):
    """Bar chart: fraction of events in each agreement category."""
    set_style()
    n = len(event_summary)
    n_agree  = int(event_summary["agreement"].sum())
    n_extra  = int((event_summary["n_spurious"] > 0).sum())
    n_miss   = int((event_summary["n_missed"] > 0).sum())
    n_split  = int((event_summary["n_split"] > 0).sum())

    labels  = ["Full agreement\n(CF=truth)", "CF has extra\n(spurious)", "CF missed\na neutron", "CF split\na neutron"]
    counts  = [n_agree, n_extra, n_miss, n_split]
    colors  = ["steelblue", "tomato", "orange", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, [100 * c / n for c in counts], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Fraction of events (%)")
    ax.set_ylim(0, 115)
    ax.set_title(ctx.title(f"CF vs truth agreement  (N={n} events)"))
    save_plot(fig, ctx, "matching_outcome_bar")

    print(f"[matching] Agreement summary ({n} events):")
    for lbl, cnt in zip(labels, counts):
        print(f"  {lbl.replace(chr(10),' '):<35s} {cnt:4d}  ({100*cnt/n:.1f}%)")


def _time_comparison(cluster_outcomes: pd.DataFrame, truth_clusters: pd.DataFrame, ctx: RunContext):
    """Scatter: CF clusterTime vs truth t_min for matched clusters."""
    set_style()
    matched = cluster_outcomes[cluster_outcomes["is_matched"]].copy()
    # Bring in truth t_min for matched trackID
    truth_min = truth_clusters.set_index(["eventID", "trackID"])["t_min"]
    matched["truth_t_min"] = matched.apply(
        lambda r: truth_min.get((r["eventID"], r["dominant_trackID"]), np.nan), axis=1
    )
    matched = matched.dropna(subset=["truth_t_min"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(matched["truth_t_min"], matched["clusterTime"],
               s=15, alpha=0.5, color="steelblue")
    lo = min(matched["truth_t_min"].min(), matched["clusterTime"].min()) - 5
    hi = max(matched["truth_t_min"].max(), matched["clusterTime"].max()) + 5
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect match")
    ax.set_xlabel("Truth neutron first-hit time (ns)")
    ax.set_ylabel("CF clusterTime (ns)")
    ax.set_title(ctx.title("Matched: CF time vs truth first-hit time"))
    ax.legend()
    save_plot(fig, ctx, "matching_time_scatter")


def _spurious_vs_matched_features(cluster_outcomes: pd.DataFrame, ctx: RunContext):
    """PE, hit-count, and neutron-fraction distributions for matched vs spurious CF clusters."""
    set_style()
    matched = cluster_outcomes[cluster_outcomes["is_matched"]]
    spurious = cluster_outcomes[~cluster_outcomes["is_matched"]]

    for col, xlabel in [("clusterPE", "Cluster PE (p.e.)"), ("clusterHits", "Cluster hit count")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        vmin = cluster_outcomes[col].quantile(0.01)
        vmax = cluster_outcomes[col].quantile(0.99)
        bins = np.linspace(vmin, vmax, 35)
        ax.hist(matched[col].clip(vmin, vmax),  bins=bins, alpha=0.6, density=True,
                color="steelblue", label=f"Matched  (n={len(matched)})")
        ax.hist(spurious[col].clip(vmin, vmax), bins=bins, alpha=0.6, density=True,
                color="tomato",    label=f"Spurious (n={len(spurious)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(ctx.title(f"Matched vs spurious: {xlabel}"))
        ax.legend()
        save_plot(fig, ctx, f"matching_{col}")

    # Neutron dominant fraction — shows where the 0.5 threshold sits
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(cluster_outcomes["dominant_frac"], bins=bins, edgecolor="black",
            color="slategray", alpha=0.8)
    ax.axvline(0.5, color="red", lw=1.5, ls="--", label="match threshold (0.5)")
    ax.set_xlabel("Dominant neutron trackID hit fraction\n(fraction of cluster hits from the most common neutron ancestor)")
    ax.set_ylabel("CF clusters")
    ax.set_title(ctx.title("Neutron fraction per CF cluster"))
    ax.legend()
    save_plot(fig, ctx, "matching_neutron_frac")


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def run(ctx: RunContext, verbose: bool = True) -> Path:
    pulses_path  = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if not pulses_path.exists():
        raise FileNotFoundError(f"{pulses_path} — run `ambe mc process` first")
    if not clusters_path.exists():
        raise FileNotFoundError(f"{clusters_path} — run `ambe mc process` first")

    pulses   = pd.read_parquet(pulses_path)
    clusters = pd.read_parquet(clusters_path)

    # Verify trackID was loaded (requires re-running processor after the update)
    if "ancestor_trackID" not in pulses.columns:
        raise RuntimeError(
            "pulses parquet is missing 'ancestor_trackID'. "
            "Re-run `ambe mc process` with the updated processor."
        )

    truth_clusters = build_truth_clusters(pulses)
    if verbose:
        print(f"[matching] {pulses['eventID'].nunique()} events  |  "
              f"{len(clusters)} CF clusters  |  "
              f"{len(truth_clusters)} truth neutron clusters")

    event_summary, cluster_outcomes = match_clusters(pulses, clusters, truth_clusters)

    ev_csv  = ctx.csv_path(f"{ctx.run_name}__matching")
    cl_csv  = ctx.csv_path(f"{ctx.run_name}__cluster_outcomes")
    event_summary.to_csv(ev_csv, index=False)
    cluster_outcomes.to_csv(cl_csv, index=False)
    if verbose:
        print(f"[matching] wrote event summary  -> {ev_csv}")
        print(f"[matching] wrote cluster outcomes -> {cl_csv}")

    _multiplicity_heatmap(event_summary, ctx)
    _outcome_bar(event_summary, ctx)
    _time_comparison(cluster_outcomes, truth_clusters, ctx)
    _spurious_vs_matched_features(cluster_outcomes, ctx)

    return ev_csv


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx)
