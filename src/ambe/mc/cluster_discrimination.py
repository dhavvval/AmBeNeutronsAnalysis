"""
Cluster discrimination analysis: truth-label study of ClusterFinder output.

Physics motivation
------------------
AmBe emits one neutron per source event, so ~98-100% of triggered events should
have exactly one neutron cluster.  In practice ClusterFinder returns multiple
clusters in >19% of events, polluting the neutron multiplicity distribution.

This module asks: *given MC truth*, what per-cluster feature best separates the
real neutron cluster from the spurious ones in multi-cluster events?  The answer
guides either a post-selection cut on top of ClusterFinder or a better OPTICS
configuration.

Outputs
-------
  <csv_dir>/<run>__cluster_features.csv   -- enriched per-cluster table
  <plots_dir>/                            -- feature comparison plots + multiplicity
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext
from ..plotting import save_plot, set_style
from .optics import assign_clusterfinder_labels, NEUTRON_CLASSES


# --------------------------------------------------------------------------- #
# Step 1 — enrich clusters with truth info from member pulses
# --------------------------------------------------------------------------- #

def _cluster_spatial_spread(members: pd.DataFrame) -> float:
    if len(members) < 2:
        return 0.0
    return float(np.sqrt(members["x"].var() + members["y"].var() + members["z"].var()))


def enrich_clusters(
    pulses: pd.DataFrame,
    clusters: pd.DataFrame,
    tmin_margin_ns: float = 5.0,
    tmax_margin_ns: float = 20.0,
) -> pd.DataFrame:
    """
    Assign truth labels to each ClusterFinder cluster.

    For every event, uses the same time-proximity membership as the baseline
    (assign_clusterfinder_labels) to attach pulses to clusters, then computes:
      neutron_fraction  -- fraction of member pulses that are true neutrons
      is_true_neutron   -- neutron_fraction >= 0.5
      t_spread          -- std of hit times inside the cluster  (ns)
      spatial_spread    -- sqrt(var_x + var_y + var_z)          (cm)
      n_member_pulses   -- pulses assigned to this cluster
    """
    rows = []
    for evid in clusters["eventID"].unique():
        df_p = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        df_c = clusters[clusters["eventID"] == evid].reset_index(drop=True)

        labels = assign_clusterfinder_labels(df_p, df_c, tmin_margin_ns, tmax_margin_ns)

        for cidx, crow in df_c.iterrows():
            mask = labels == cidx
            members = df_p[mask]
            n = int(mask.sum())
            n_neutron = int(members["is_neutron"].sum()) if n > 0 else 0
            frac = n_neutron / n if n > 0 else 0.0

            rows.append({
                "eventID":          evid,
                "cluster_idx":      int(crow["cluster_idx"]),
                "clusterTime":      float(crow["clusterTime"]),
                "clusterPE":        float(crow["clusterPE"]),
                "clusterHits":      int(crow["clusterHits"]),
                "n_member_pulses":  n,
                "n_neutron_hits":   n_neutron,
                "neutron_fraction": frac,
                "is_true_neutron":  frac >= 0.5,
                "t_spread":         float(members["t"].std()) if n > 1 else 0.0,
                "spatial_spread":   _cluster_spatial_spread(members),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 2 — selection strategies: given multiple clusters, pick one
# --------------------------------------------------------------------------- #

STRATEGIES = {
    "highest_PE":        lambda df: df["clusterPE"].idxmax(),
    "most_hits":         lambda df: df["clusterHits"].idxmax(),
    "earliest":          lambda df: df["clusterTime"].idxmin(),
    "tightest_time":     lambda df: df["t_spread"].idxmin(),
    "tightest_spatial":  lambda df: df["spatial_spread"].idxmin(),
}


def evaluate_strategies(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    For every multi-cluster event, apply each selection strategy and check
    whether it picks the true neutron cluster.

    Returns a DataFrame with one row per (event, strategy).
    """
    multi = enriched.groupby("eventID").filter(lambda g: len(g) > 1)
    rows = []
    for evid, grp in multi.groupby("eventID"):
        has_true = grp["is_true_neutron"].any()
        for name, selector in STRATEGIES.items():
            chosen_idx = selector(grp)
            picked_true = bool(grp.loc[chosen_idx, "is_true_neutron"])
            rows.append({
                "eventID":      evid,
                "strategy":     name,
                "has_true":     has_true,
                "picked_true":  picked_true,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3 — plots
# --------------------------------------------------------------------------- #

def _multiplicity_plot(enriched: pd.DataFrame, ctx: RunContext):
    """Cluster count per event: raw CF vs after each selection strategy."""
    set_style()
    raw_counts = enriched.groupby("eventID").size()

    # After each strategy: always 1 cluster per event (we always pick one).
    # But we want to show how often that selection is "correct" vs just the
    # raw distribution, so plot raw + fraction with >1 cluster.
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(0.5, raw_counts.max() + 1.5, 1)
    ax.hist(raw_counts, bins=bins, edgecolor="black", alpha=0.7, label="ClusterFinder (raw)")
    ax.set_xlabel("Number of clusters per event")
    ax.set_ylabel("Events")
    ax.set_title(ctx.title("Neutron cluster multiplicity (ClusterFinder)"))

    n_total = len(raw_counts)
    n_multi  = int((raw_counts > 1).sum())
    ax.text(0.97, 0.95,
            f"Events with >1 cluster: {n_multi}/{n_total} ({100*n_multi/n_total:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.legend()
    save_plot(fig, ctx, "cf_multiplicity")
    print(f"[discriminate] {n_multi}/{n_total} ({100*n_multi/n_total:.1f}%) events have >1 cluster")


def _feature_comparison_plots(enriched: pd.DataFrame, ctx: RunContext):
    """
    For multi-cluster events, compare feature distributions between
    true-neutron clusters and spurious clusters.
    """
    set_style()
    multi_evids = enriched.groupby("eventID").filter(lambda g: len(g) > 1)["eventID"].unique()
    multi = enriched[enriched["eventID"].isin(multi_evids)]

    true_cl  = multi[multi["is_true_neutron"]]
    spur_cl  = multi[~multi["is_true_neutron"]]

    features = [
        ("clusterPE",       "Cluster PE (p.e.)",          None),
        ("clusterHits",     "Cluster hit count",           None),
        ("clusterTime",     "Cluster time (ns)",           None),
        ("t_spread",        "Hit time spread σ_t (ns)",    None),
        ("spatial_spread",  "Spatial spread (cm)",         None),
        ("neutron_fraction","True-neutron hit fraction",   (0, 1)),
    ]

    for col, xlabel, xlim in features:
        fig, ax = plt.subplots(figsize=(7, 5))
        vmin = multi[col].quantile(0.01)
        vmax = multi[col].quantile(0.99)
        bins = np.linspace(vmin, vmax, 40)

        ax.hist(true_cl[col].clip(vmin, vmax),  bins=bins, alpha=0.6,
                label=f"True neutron  (n={len(true_cl)})",  density=True, color="steelblue")
        ax.hist(spur_cl[col].clip(vmin, vmax),  bins=bins, alpha=0.6,
                label=f"Spurious      (n={len(spur_cl)})",  density=True, color="tomato")

        if xlim:
            ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(ctx.title(f"True vs spurious: {xlabel}"))
        ax.legend()
        save_plot(fig, ctx, f"discrimination_{col}")


def _strategy_summary_plot(strategy_results: pd.DataFrame, ctx: RunContext):
    """Bar chart: fraction of multi-cluster events each strategy picks the right cluster."""
    set_style()
    summary = (
        strategy_results[strategy_results["has_true"]]
        .groupby("strategy")["picked_true"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue" if v == summary.max() else "lightsteelblue" for v in summary.values]
    bars = ax.bar(summary.index, summary.values * 100, color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Correct selection (%)")
    ax.set_title(ctx.title("Selection strategy accuracy\n(multi-cluster events with a true neutron cluster)"))
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, ctx, "strategy_accuracy")

    print("[discriminate] strategy accuracy on multi-cluster events:")
    for strat, acc in summary.items():
        print(f"  {strat:<20s} {100*acc:.1f}%")


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

    if verbose:
        print(f"[discriminate] {len(pulses)} pulses, {len(clusters)} clusters, "
              f"{pulses['eventID'].nunique()} events")

    enriched = enrich_clusters(pulses, clusters)

    out_csv = ctx.csv_path(f"{ctx.run_name}__cluster_features")
    enriched.to_csv(out_csv, index=False)
    if verbose:
        print(f"[discriminate] wrote enriched cluster table -> {out_csv}")

    _multiplicity_plot(enriched, ctx)
    _feature_comparison_plots(enriched, ctx)

    strategy_results = evaluate_strategies(enriched)
    _strategy_summary_plot(strategy_results, ctx)

    strat_csv = ctx.csv_path(f"{ctx.run_name}__strategy_results")
    strategy_results.to_csv(strat_csv, index=False)
    if verbose:
        print(f"[discriminate] wrote strategy results -> {strat_csv}")

    return out_csv


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx)
