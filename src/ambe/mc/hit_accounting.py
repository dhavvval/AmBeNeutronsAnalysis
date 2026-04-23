"""
Hit accounting and CF cluster composition analysis.

Answers two questions:
  1. Are we losing hits? Compare nhits (all detector hits), DirectParent-labeled
     hits, and hits actually inside CF clusters.
  2. What is each CF cluster made of? For every hit inside a cluster, classify it
     as neutron / non-neutron background / dark noise / unclaimed (dark noise not
     traced by BackTracker).

Time-reference note
-------------------
Cluster_HitT and hitT share the same reconstructed time reference (exact match).
DirectParent_HitTime is offset by +14-16 ns (Cherenkov photon travel time from
interaction point to PMT).  Per-event offset is computed from hits whose chankey
appears in both arrays, then used with a ±8 ns tolerance for the truth match.

Outputs
-------
  <csv>/<run>__hit_accounting.csv    per-event counts
  <csv>/<run>__cluster_composition.csv  per-cluster composition
  plots: hit accounting bar, CF composition pie/stacked bar
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs
from ..plotting import save_plot, set_style

NEUTRON_CLASSES = {1, 2, 3, 4}
BRANCHES = [
    "eventNumber", "nhits", "numberOfClusters",
    "hitChankey", "hitT",
    "Cluster_HitT", "Cluster_HitChankey", "Cluster_HitChankeyMC",
    "DirectParent_HitTime", "DirectParent_PMTID",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
]


# --------------------------------------------------------------------------- #
# Per-hit truth classification
# --------------------------------------------------------------------------- #

def _compute_dp_offset(hit_ck, hit_t, dp_pmtid, dp_t):
    """
    Estimate DirectParent_HitTime - hitT offset using chankeys that appear in both.
    Returns median offset (ns); falls back to 15 ns if insufficient overlap.
    """
    ck_set = set(np.unique(hit_ck)) & set(np.unique(dp_pmtid))
    offsets = []
    for ck in ck_set:
        ht_vals = hit_t[hit_ck == ck]
        dp_vals = dp_t[dp_pmtid == ck]
        if len(ht_vals) == 1 and len(dp_vals) == 1:
            offsets.append(float(dp_vals[0] - ht_vals[0]))
    return float(np.median(offsets)) if len(offsets) >= 3 else 15.0


def _classify_hits(hit_ck, hit_t, dp_pmtid, dp_t, dp_class, dp_dn, offset, tol=8.0):
    """
    For each hit in (hit_ck, hit_t), find the best-matching DirectParent entry
    using the measured time offset, then classify as:
      'neutron'     – ancestor class in {1,2,3,4}
      'background'  – ancestor class == -5
      'darknoise'   – ancestor class == 0 or is_darknoise==1
      'unclaimed'   – no DirectParent match (dark noise BackTracker didn't trace)
    Returns array of labels, same length as hit_ck.
    """
    labels = np.full(len(hit_ck), "unclaimed", dtype=object)
    for j, (ck, t) in enumerate(zip(hit_ck, hit_t)):
        idx = np.where(dp_pmtid == ck)[0]
        if not len(idx):
            continue
        dt = np.abs(dp_t[idx] - (t + offset))
        best = idx[np.argmin(dt)]
        if dt.min() > tol:
            continue
        cls = int(dp_class[best])
        dn  = int(dp_dn[best])
        if cls in NEUTRON_CLASSES:
            labels[j] = "neutron"
        elif cls == -5:
            labels[j] = "background"
        else:
            labels[j] = "darknoise"
    return labels


# --------------------------------------------------------------------------- #
# Main extraction
# --------------------------------------------------------------------------- #

def extract(root_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (event_df, cluster_df).

    event_df  – one row per event with hit counts
    cluster_df – one row per CF cluster with composition counts
    """
    ev_rows, cl_rows = [], []

    with uproot.open(str(root_path)) as f:
        if "Event" not in f:
            raise KeyError(f"No 'Event' tree in {root_path}")
        arr = f["Event"].arrays(BRANCHES, library="ak")

    for i in range(len(arr)):
        evid   = int(arr["eventNumber"][i])
        nhits  = int(arr["nhits"][i])
        nc     = int(arr["numberOfClusters"][i])

        hit_ck = np.array(ak.to_list(arr["hitChankey"][i]),  dtype=int)
        hit_t  = np.array(ak.to_list(arr["hitT"][i]),        dtype=float)
        dp_ck  = np.array(ak.to_list(arr["DirectParent_PMTID"][i]),              dtype=int)
        dp_t   = np.array(ak.to_list(arr["DirectParent_HitTime"][i]),            dtype=float)
        dp_cls = np.array(ak.to_list(arr["DirectParent_NeutronAncestorClass"][i]),dtype=int)
        dp_dn  = np.array(ak.to_list(arr["DirectParent_IsDarknoise"][i]),         dtype=int)

        cf_ck_lists  = ak.to_list(arr["Cluster_HitChankeyMC"][i])
        cf_t_lists   = ak.to_list(arr["Cluster_HitT"][i])

        # --- time offset for this event ---
        offset = _compute_dp_offset(hit_ck, hit_t, dp_ck, dp_t)

        # --- classify every detector hit ---
        hit_labels = _classify_hits(hit_ck, hit_t, dp_ck, dp_t, dp_cls, dp_dn, offset)

        # --- which hits are inside any CF cluster? ---
        # Build set of (chankey, rounded_time) in clusters for exact matching
        clustered_keys = set()
        for ck_list, t_list in zip(cf_ck_lists, cf_t_lists):
            for ck, t in zip(ck_list, t_list):
                clustered_keys.add((int(ck), round(float(t), 4)))

        in_cluster = np.array([
            (int(ck), round(float(t), 4)) in clustered_keys
            for ck, t in zip(hit_ck, hit_t)
        ], dtype=bool)

        n_dp         = len(dp_ck)
        n_clustered  = int(in_cluster.sum())
        n_unclustered= nhits - n_clustered

        # composition of all hits
        lbl_counts = {lbl: int((hit_labels == lbl).sum())
                      for lbl in ("neutron", "background", "darknoise", "unclaimed")}

        # composition of clustered hits only
        cl_lbl = hit_labels[in_cluster]
        cl_counts = {lbl: int((cl_lbl == lbl).sum())
                     for lbl in ("neutron", "background", "darknoise", "unclaimed")}

        ev_rows.append({
            "eventID":           evid,
            "nhits":             nhits,
            "n_dp_labeled":      n_dp,
            "n_clustered":       n_clustered,
            "n_unclustered":     n_unclustered,
            "dp_offset_ns":      round(offset, 2),
            **{f"total_{k}": v for k, v in lbl_counts.items()},
            **{f"cf_{k}":    v for k, v in cl_counts.items()},
        })

        # --- per-cluster composition ---
        # rebuild hit label lookup by (chankey, time)
        hit_label_map = {
            (int(ck), round(float(t), 4)): lbl
            for ck, t, lbl in zip(hit_ck, hit_t, hit_labels)
        }
        for cidx, (ck_list, t_list) in enumerate(zip(cf_ck_lists, cf_t_lists)):
            comp = {"neutron": 0, "background": 0, "darknoise": 0, "unclaimed": 0}
            for ck, t in zip(ck_list, t_list):
                lbl = hit_label_map.get((int(ck), round(float(t), 4)), "unclaimed")
                comp[lbl] += 1
            total = sum(comp.values())
            cl_rows.append({
                "eventID":    evid,
                "cluster_idx": cidx,
                "n_hits":     total,
                **comp,
                **{f"frac_{k}": v / total if total else 0.0 for k, v in comp.items()},
            })

    return pd.DataFrame(ev_rows), pd.DataFrame(cl_rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _accounting_plot(ev_df: pd.DataFrame, ctx: RunContext):
    """Stacked bar: average hit breakdown per event."""
    set_style()
    means = {
        "In CF cluster\n(neutron)":    ev_df["cf_neutron"].mean(),
        "In CF cluster\n(background)": ev_df["cf_background"].mean(),
        "In CF cluster\n(dark noise)": ev_df["cf_darknoise"].mean(),
        "In CF cluster\n(unclaimed)":  ev_df["cf_unclaimed"].mean(),
        "Unclustered\n(neutron)":      (ev_df["total_neutron"] - ev_df["cf_neutron"]).mean(),
        "Unclustered\n(background)":   (ev_df["total_background"] - ev_df["cf_background"]).mean(),
        "Unclustered\n(dark noise)":   (ev_df["total_darknoise"] - ev_df["cf_darknoise"]).mean(),
        "Unclustered\n(unclaimed)":    (ev_df["total_unclaimed"] - ev_df["cf_unclaimed"]).mean(),
    }
    colors = [
        "#2196F3", "#FF9800", "#9E9E9E", "#607D8B",   # clustered
        "#90CAF9", "#FFCC80", "#E0E0E0", "#B0BEC5",   # unclustered (lighter)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = 0
    for (label, val), color in zip(means.items(), colors):
        bar = ax.bar("Average event", val, bottom=bottom, color=color,
                     edgecolor="white", linewidth=0.5, label=f"{label} ({val:.1f})")
        bottom += val
    ax.set_ylabel("Mean hit count per event")
    ax.set_title(ctx.title("Hit accounting: where do all nhits go?"))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    save_plot(fig, ctx, "hit_accounting_bar")


def _composition_plot(cl_df: pd.DataFrame, ctx: RunContext):
    """Stacked bar: hit composition across all CF clusters."""
    set_style()
    cols = ["neutron", "background", "darknoise", "unclaimed"]
    colors = ["steelblue", "tomato", "gray", "lightgray"]
    totals = {c: cl_df[c].sum() for c in cols}
    grand  = sum(totals.values())
    fracs  = {c: totals[c] / grand for c in cols}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: aggregate pie
    ax = axes[0]
    wedge_labels = [f"{c}\n{100*fracs[c]:.1f}%\n({totals[c]:,} hits)" for c in cols]
    ax.pie([fracs[c] for c in cols], labels=wedge_labels,
           colors=colors, startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1))
    ax.set_title("All CF cluster hits — composition")

    # Right: per-cluster fraction distributions
    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    for c, color in zip(cols, colors):
        ax.hist(cl_df[f"frac_{c}"], bins=bins, alpha=0.6,
                color=color, label=c, density=True)
    ax.set_xlabel("Fraction of hits per cluster")
    ax.set_ylabel("Density")
    ax.set_title("Per-cluster hit fraction distributions")
    ax.legend()

    plt.tight_layout()
    save_plot(fig, ctx, "hit_composition")


def _neutron_recovery_plot(ev_df: pd.DataFrame, ctx: RunContext):
    """What fraction of neutron hits does CF actually capture?"""
    set_style()
    ev_df = ev_df[ev_df["total_neutron"] > 0].copy()
    recovery = ev_df["cf_neutron"] / ev_df["total_neutron"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(recovery * 100, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(recovery.median() * 100, color="red", ls="--", lw=1.5,
               label=f"median = {recovery.median()*100:.1f}%")
    ax.set_xlabel("Neutron hits captured by CF (% of truth neutron hits in event)")
    ax.set_ylabel("Events")
    ax.set_title(ctx.title("CF neutron hit recovery rate"))
    ax.legend()
    save_plot(fig, ctx, "hit_neutron_recovery")


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def run(ctx: RunContext, verbose: bool = True) -> Path:
    root_files = resolve_inputs(ctx.inputs["root_files"])
    if len(root_files) != 1:
        raise ValueError("ambe mc hitcomp expects exactly one root_files entry")
    root_path = root_files[0]

    if verbose:
        print(f"[hitcomp] reading {root_path.name}")

    ev_df, cl_df = extract(root_path)

    ev_csv = ctx.csv_path(f"{ctx.run_name}__hit_accounting")
    cl_csv = ctx.csv_path(f"{ctx.run_name}__cluster_composition")
    ev_df.to_csv(ev_csv, index=False)
    cl_df.to_csv(cl_csv, index=False)

    if verbose:
        n = len(ev_df)
        print(f"[hitcomp] {n} events processed")
        print(f"\n--- Average hits per event ---")
        print(f"  Total (nhits):           {ev_df['nhits'].mean():.1f}")
        print(f"  With DirectParent label: {ev_df['n_dp_labeled'].mean():.1f}  "
              f"({100*ev_df['n_dp_labeled'].mean()/ev_df['nhits'].mean():.1f}% of nhits)")
        print(f"  Inside CF clusters:      {ev_df['n_clustered'].mean():.1f}  "
              f"({100*ev_df['n_clustered'].mean()/ev_df['nhits'].mean():.1f}% of nhits)")
        print(f"  Unclustered:             {ev_df['n_unclustered'].mean():.1f}  "
              f"({100*ev_df['n_unclustered'].mean()/ev_df['nhits'].mean():.1f}% of nhits)")
        print(f"\n--- Composition of CF cluster hits (all clusters pooled) ---")
        for lbl in ("neutron", "background", "darknoise", "unclaimed"):
            total = cl_df[lbl].sum()
            grand = cl_df[["neutron","background","darknoise","unclaimed"]].sum().sum()
            print(f"  {lbl:<12s}: {total:7,}  ({100*total/grand:.1f}%)")
        print(f"\n--- Neutron hit recovery ---")
        has_n = ev_df[ev_df["total_neutron"] > 0]
        rec = has_n["cf_neutron"] / has_n["total_neutron"]
        print(f"  Median fraction of event's neutron hits captured in CF: {rec.median()*100:.1f}%")
        print(f"  Mean:   {rec.mean()*100:.1f}%")
        print(f"\n[hitcomp] wrote {ev_csv}")
        print(f"[hitcomp] wrote {cl_csv}")

    _accounting_plot(ev_df, ctx)
    _composition_plot(cl_df, ctx)
    _neutron_recovery_plot(ev_df, ctx)

    return ev_csv


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx)
