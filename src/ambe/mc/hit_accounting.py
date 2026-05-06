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
DirectParent_HitTime is offset by +14-16 ns relative to hitT.  This offset is
the PMT SPE waveform peak delay (p1*exp(-p2²) - T0Offset*2ns from the lognormal
model in PMTWaveformSim, ~15.9 ns for typical ANNIE PMTs).  It is NOT the
Cherenkov photon travel time, which is already embedded in hitT.  Per-event
offset is computed from chankeys that appear exactly once in both arrays, then
used with a ±8 ns tolerance for the truth match.

Unclaimed hit diagnostics
-------------------------
A second pass characterises each unclaimed CF-cluster hit across six axes:
  Step 1  dt from cluster median (flat→dark noise, peaked→missed physics)
  Step 2  PE spectrum vs labeled hits
  Step 3  hitPMTType distribution
  Step 4  BackTracker DP entry existence + IsDarknoise flag
  Step 5  Re-match with ±25 ns wide tolerance
  Step 6  PDG of nearest DP entry (non-neutron parent check)

Outputs
-------
  <csv>/<run>__hit_accounting.csv          per-event counts
  <csv>/<run>__cluster_composition.csv     per-cluster composition
  <csv>/<run>__unclaimed_diagnostics.csv   per-unclaimed-CF-hit diagnostics
  plots: hit_accounting_bar, hit_composition, hit_neutron_recovery,
         unclaimed_diag_timing_pe, unclaimed_diag_meta
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
    "hitPE",        # reconstructed PE per hit
    "hitPMTType",   # PMT subsystem flag (tank/MRD/veto)
    "Cluster_HitT", "Cluster_HitChankey",
    "DirectParent_HitTime", "DirectParent_PMTID",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
    "DirectParent_PDGs",      # PDG code of parent particle
    "DirectParent_TrackIDs",  # GEANT4 track ID
]

WIDE_TOL = 25.0  # ns — wide tolerance for Step 5 re-matching


# --------------------------------------------------------------------------- #
# Per-hit truth classification
# --------------------------------------------------------------------------- #

def _compute_dp_offset(hit_ck, hit_t, dp_pmtid, dp_t):
    """
    Estimate DirectParent_HitTime - hitT offset using chankeys that appear in both.

    The offset is the PMT SPE waveform peak delay (~15.9 ns for typical ANNIE
    PMTs), not the Cherenkov photon travel time (which is already in hitT).

    Falls back to 15.0 ns if fewer than 3 unambiguous pairs exist or if the
    computed median is outside the physically plausible range [10, 25] ns.
    """
    shared = set(hit_ck) & set(dp_pmtid)
    offsets = []
    for ck in shared:
        ht_vals = hit_t[hit_ck == ck]
        dp_vals = dp_t[dp_pmtid == ck]
        if len(ht_vals) == 1 and len(dp_vals) == 1:
            offsets.append(float(dp_vals[0] - ht_vals[0]))
    if len(offsets) < 3:
        return 15.0
    med = float(np.median(offsets))
    return med if 10.0 <= med <= 25.0 else 15.0


def _classify_hits(hit_ck, hit_t, dp_pmtid, dp_t, dp_class, dp_dn, offset, tol=8.0):
    """
    For each hit in (hit_ck, hit_t), find the best-matching DirectParent entry
    using the measured time offset, then classify as:
      'neutron'     – ancestor class in {1,2,3,4}
      'background'  – ancestor class == -5
      'darknoise'   – ancestor class == 0 or is_darknoise==1
      'unclaimed'   – no DirectParent match within tol
    Returns array of labels, same length as hit_ck.

    Uses a dict-based O(N+M) lookup instead of an O(N*M) linear scan.
    """
    # Build per-chankey index into the DP arrays once — O(M)
    dp_by_ck: dict[int, list] = {}
    for idx, ck in enumerate(dp_pmtid):
        dp_by_ck.setdefault(int(ck), []).append(idx)

    labels = np.full(len(hit_ck), "unclaimed", dtype=object)
    for j, (ck, t) in enumerate(zip(hit_ck, hit_t)):
        indices = dp_by_ck.get(int(ck))
        if not indices:
            continue
        t_shifted = t + offset
        best = min(indices, key=lambda i: abs(dp_t[i] - t_shifted))
        if abs(dp_t[best] - t_shifted) > tol:
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


def _dp_label_from_class(cls: int, dn: int) -> str:
    """Derive the hit label from DirectParent class and IsDarknoise flag."""
    if cls in NEUTRON_CLASSES:
        return "neutron"
    elif cls == -5:
        return "background"
    else:
        return "darknoise"


def _diagnose_unclaimed(
    hit_ck, hit_t, hit_pe, hit_pmt_type,
    dp_pmtid, dp_t, dp_dn, dp_pdg, dp_class,
    offset: float,
    labels: np.ndarray,
    cf_ck_lists, cf_t_lists,
    evid: int,
) -> list[dict]:
    """
    For each unclaimed hit inside a CF cluster, produce a diagnostic row.

    Covers Steps 1–6 of the unclaimed characterisation analysis:
      - timing relative to cluster median (Step 1)
      - PE and PMT type carried through from caller (Steps 2, 3)
      - DP entry existence + nearest IsDarknoise (Step 4)
      - wide-tolerance re-match at ±WIDE_TOL ns (Step 5)
      - PDG of nearest DP entry (Step 6)

    Returns a list of dicts, one per unclaimed CF hit.
    """
    rows = []

    def rkey(ck, t):
        return (int(ck), round(float(t), 4))

    hit_label_map = {rkey(ck, t): lbl for ck, t, lbl in zip(hit_ck, hit_t, labels)}
    hit_pe_map    = {rkey(ck, t): pe  for ck, t, pe  in zip(hit_ck, hit_t, hit_pe)}
    hit_pmt_map   = {rkey(ck, t): pt  for ck, t, pt  in zip(hit_ck, hit_t, hit_pmt_type)}

    for cidx, (ck_list, t_list) in enumerate(zip(cf_ck_lists, cf_t_lists)):
        if not ck_list:
            continue
        cluster_median_t = float(np.median([float(t) for t in t_list]))

        for ck, t in zip(ck_list, t_list):
            k = rkey(ck, t)
            if hit_label_map.get(k, "unclaimed") != "unclaimed":
                continue

            pe       = hit_pe_map.get(k, np.nan)
            pmt_type = hit_pmt_map.get(k, -1)
            dt_from_cluster_median = float(t) - cluster_median_t

            ck_int = int(ck)
            dp_idx = np.where(dp_pmtid == ck_int)[0]
            has_dp_entry = len(dp_idx) > 0

            if has_dp_entry:
                dts = np.abs(dp_t[dp_idx] - (float(t) + offset))
                best_local = int(np.argmin(dts))
                best = dp_idx[best_local]
                min_dt                  = float(dts[best_local])
                nearest_dp_is_darknoise = int(dp_dn[best])
                nearest_dp_pdg          = int(dp_pdg[best])
                matched_wide            = min_dt <= WIDE_TOL
                wide_label = (
                    _dp_label_from_class(int(dp_class[best]), int(dp_dn[best]))
                    if matched_wide else "unclaimed"
                )
            else:
                min_dt                  = np.nan
                nearest_dp_is_darknoise = -1
                nearest_dp_pdg          = -999
                matched_wide            = False
                wide_label              = "unclaimed"

            rows.append({
                "eventID":                   evid,
                "cluster_idx":               cidx,
                "hit_ck":                    ck_int,
                "hit_t":                     float(t),
                "hit_pe":                    pe,
                "hit_pmt_type":              pmt_type,
                "dt_from_cluster_median_ns": dt_from_cluster_median,
                "has_dp_entry":              has_dp_entry,
                "min_dt_to_dp_ns":           min_dt,
                "nearest_dp_is_darknoise":   nearest_dp_is_darknoise,
                "nearest_dp_pdg":            nearest_dp_pdg,
                "matched_wide":              matched_wide,
                "wide_label":                wide_label,
            })

    return rows


# --------------------------------------------------------------------------- #
# Main extraction
# --------------------------------------------------------------------------- #

def extract(root_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns (event_df, cluster_df, unclaimed_df, labeled_cf_pe).

    event_df      – one row per event with hit counts
    cluster_df    – one row per CF cluster with composition counts
    unclaimed_df  – one row per unclaimed CF cluster hit with diagnostics
    labeled_cf_pe – PE values for non-unclaimed CF hits (for Step 2 comparison)
    """
    ev_rows, cl_rows, unc_rows = [], [], []
    labeled_cf_pe_list: list[float] = []

    with uproot.open(str(root_path)) as f:
        if "Event" not in f:
            raise KeyError(f"No 'Event' tree in {root_path}")
        tree = f["Event"]
        available = set(tree.keys())
        for b in ("hitPE", "hitPMTType", "DirectParent_PDGs", "DirectParent_TrackIDs"):
            if b not in available:
                print(f"[hitcomp] WARNING: branch '{b}' not found — using sentinel values")
        branches_to_load = [b for b in BRANCHES if b in available]
        arr = tree.arrays(branches_to_load, library="ak")

    for i in range(len(arr)):
        evid   = int(arr["eventNumber"][i])
        nhits  = int(arr["nhits"][i])
        nc     = int(arr["numberOfClusters"][i])

        hit_ck = np.array(ak.to_list(arr["hitChankey"][i]),  dtype=int)
        hit_t  = np.array(ak.to_list(arr["hitT"][i]),        dtype=float)
        dp_ck  = np.array(ak.to_list(arr["DirectParent_PMTID"][i]),               dtype=int)
        dp_t   = np.array(ak.to_list(arr["DirectParent_HitTime"][i]),             dtype=float)
        dp_cls = np.array(ak.to_list(arr["DirectParent_NeutronAncestorClass"][i]), dtype=int)
        dp_dn  = np.array(ak.to_list(arr["DirectParent_IsDarknoise"][i]),          dtype=int)

        # Cluster_HitChankey uses the same data-chankey space as hitChankey.
        # Cluster_HitChankeyMC is the detector key from ClusterFinder and is
        # in a different space — using it here would make every in_cluster
        # lookup silently fail.
        cf_ck_lists = ak.to_list(arr["Cluster_HitChankey"][i])
        cf_t_lists  = ak.to_list(arr["Cluster_HitT"][i])

        # New branches with sentinel fallbacks
        hit_pe = (
            np.array(ak.to_list(arr["hitPE"][i]), dtype=float)
            if "hitPE" in available else np.full(len(hit_ck), np.nan)
        )
        hit_pmt = (
            np.array(ak.to_list(arr["hitPMTType"][i]), dtype=int)
            if "hitPMTType" in available else np.full(len(hit_ck), -1, dtype=int)
        )
        # DirectParent_PDGs is a vector-of-vectors (one inner list per hit);
        # take the first element of each inner list as the primary PDG.
        dp_pdg = (
            np.array([inner[0] if (inner and len(inner) > 0) else -999
                      for inner in ak.to_list(arr["DirectParent_PDGs"][i])], dtype=int)
            if "DirectParent_PDGs" in available else np.full(len(dp_ck), -999, dtype=int)
        )

        # --- time offset for this event ---
        offset = _compute_dp_offset(hit_ck, hit_t, dp_ck, dp_t)

        # --- classify every detector hit ---
        hit_labels = _classify_hits(hit_ck, hit_t, dp_ck, dp_t, dp_cls, dp_dn, offset)

        # --- which hits are inside any CF cluster? ---
        clustered_keys = set()
        for ck_list, t_list in zip(cf_ck_lists, cf_t_lists):
            for ck, t in zip(ck_list, t_list):
                clustered_keys.add((int(ck), round(float(t), 4)))

        in_cluster = np.array([
            (int(ck), round(float(t), 4)) in clustered_keys
            for ck, t in zip(hit_ck, hit_t)
        ], dtype=bool)

        n_dp          = len(dp_ck)
        n_clustered   = int(in_cluster.sum())
        n_unclustered = nhits - n_clustered

        # composition of all hits
        lbl_counts = {lbl: int((hit_labels == lbl).sum())
                      for lbl in ("neutron", "background", "darknoise", "unclaimed")}

        # composition of clustered hits only
        cl_lbl    = hit_labels[in_cluster]
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
        hit_label_map = {
            (int(ck), round(float(t), 4)): lbl
            for ck, t, lbl in zip(hit_ck, hit_t, hit_labels)
        }
        hit_pe_map = {
            (int(ck), round(float(t), 4)): pe
            for ck, t, pe in zip(hit_ck, hit_t, hit_pe)
        }
        for cidx, (ck_list, t_list) in enumerate(zip(cf_ck_lists, cf_t_lists)):
            comp = {"neutron": 0, "background": 0, "darknoise": 0, "unclaimed": 0}
            for ck, t in zip(ck_list, t_list):
                lbl = hit_label_map.get((int(ck), round(float(t), 4)), "unclaimed")
                comp[lbl] += 1
            total = sum(comp.values())
            cl_rows.append({
                "eventID":     evid,
                "cluster_idx": cidx,
                "n_hits":      total,
                **comp,
                **{f"frac_{k}": v / total if total else 0.0 for k, v in comp.items()},
            })

        # --- collect PE for labeled (non-unclaimed) CF hits for Step 2 comparison ---
        for ck, t, lbl in zip(hit_ck[in_cluster], hit_t[in_cluster], hit_labels[in_cluster]):
            if lbl != "unclaimed":
                labeled_cf_pe_list.append(
                    hit_pe_map.get((int(ck), round(float(t), 4)), np.nan)
                )

        # --- unclaimed hit diagnostics ---
        unc_rows.extend(_diagnose_unclaimed(
            hit_ck, hit_t, hit_pe, hit_pmt,
            dp_ck, dp_t, dp_dn, dp_pdg, dp_cls,
            offset, hit_labels, cf_ck_lists, cf_t_lists, evid,
        ))

    return (
        pd.DataFrame(ev_rows),
        pd.DataFrame(cl_rows),
        pd.DataFrame(unc_rows),
        np.array(labeled_cf_pe_list, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Plots — existing
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
# Plots — unclaimed hit diagnostics
# --------------------------------------------------------------------------- #

def _unclaimed_timing_pe_plots(
    unc_df: pd.DataFrame, labeled_pe: np.ndarray, ctx: RunContext
):
    """Steps 1–3: timing relative to cluster median, PE spectrum, PMT type."""
    if unc_df.empty:
        return
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1 — Step 1: timing relative to cluster median
    ax = axes[0]
    dt = unc_df["dt_from_cluster_median_ns"].dropna()
    ax.hist(dt, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="red", ls="--", lw=1.2, label="cluster median")
    ax.set_xlabel("Hit time − cluster median (ns)")
    ax.set_ylabel("Unclaimed hits")
    ax.set_title("Step 1: Timing of unclaimed CF hits\n(flat→dark noise  peaked→missed physics)")
    ax.legend(fontsize=8)

    # Panel 2 — Step 2: PE spectrum comparison
    ax = axes[1]
    unc_pe = unc_df["hit_pe"].dropna().values
    lab_pe = labeled_pe[~np.isnan(labeled_pe)]
    if len(unc_pe) > 0 or len(lab_pe) > 0:
        pe_max = max(
            float(np.percentile(unc_pe, 99)) if len(unc_pe) else 5.0,
            float(np.percentile(lab_pe, 99)) if len(lab_pe) else 5.0,
        )
        bins = np.linspace(0, max(pe_max, 5.0), 40)
        if len(unc_pe):
            ax.hist(unc_pe, bins=bins, alpha=0.6, color="tomato",
                    label="unclaimed", density=True)
        if len(lab_pe):
            ax.hist(lab_pe, bins=bins, alpha=0.6, color="steelblue",
                    label="labeled", density=True)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("Hit PE")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Step 2: PE spectrum\n(dark noise peaks at ~1 PE)")

    # Panel 3 — Step 3: PMT type distribution
    ax = axes[2]
    counts = unc_df["hit_pmt_type"].value_counts().sort_index()
    ax.bar([str(k) for k in counts.index], counts.values,
           color="mediumpurple", edgecolor="white")
    ax.set_xlabel("hitPMTType code")
    ax.set_ylabel("Unclaimed hits")
    ax.set_title("Step 3: Unclaimed hits by PMT type\n(BackTracker may not trace all subsystems)")

    plt.tight_layout()
    save_plot(fig, ctx, "unclaimed_diag_timing_pe")


def _unclaimed_meta_plots(unc_df: pd.DataFrame, ctx: RunContext):
    """Steps 4–6: DP entry existence, wide tolerance re-match, PDG distribution."""
    if unc_df.empty:
        return
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    n_total = len(unc_df)

    # Panel 1 — Step 4: DP entry existence + IsDarknoise breakdown
    ax = axes[0, 0]
    no_entry  = int((~unc_df["has_dp_entry"]).sum())
    entry_dn1 = int((unc_df["has_dp_entry"] & (unc_df["nearest_dp_is_darknoise"] == 1)).sum())
    entry_dn0 = int((unc_df["has_dp_entry"] & (unc_df["nearest_dp_is_darknoise"] == 0)).sum())
    entry_dnu = int((unc_df["has_dp_entry"] & (unc_df["nearest_dp_is_darknoise"] == -1)).sum())
    bar_labels  = ["No DP entry", "DP: IsDN=1", "DP: IsDN=0", "DP: flag?"]
    bar_values  = [no_entry, entry_dn1, entry_dn0, entry_dnu]
    bar_colors  = ["#607D8B", "#9E9E9E", "#FF7043", "#FFCC02"]
    bars = ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, bar_values):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{100*val/n_total:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Unclaimed hits")
    ax.set_title("Step 4: BackTracker DP entry status\nfor unclaimed CF hits")
    ax.tick_params(axis="x", labelsize=8)

    # Panel 2 — Step 5: wide tolerance resolution
    ax = axes[0, 1]
    wide_label_order = ["neutron", "background", "darknoise", "unclaimed"]
    wide_colors = {
        "neutron":    "steelblue",
        "background": "tomato",
        "darknoise":  "gray",
        "unclaimed":  "lightgray",
    }
    wide_counts = unc_df["wide_label"].value_counts()
    vals = [int(wide_counts.get(lbl, 0)) for lbl in wide_label_order]
    bars = ax.bar(wide_label_order, vals,
                  color=[wide_colors[l] for l in wide_label_order], edgecolor="white")
    for bar, val in zip(bars, vals):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{100*val/n_total:.1f}%", ha="center", va="bottom", fontsize=8)
    resolved = int(unc_df["matched_wide"].sum())
    ax.set_ylabel("Unclaimed hits")
    ax.set_title(f"Step 5: Re-match at ±{WIDE_TOL:.0f} ns\n"
                 f"{resolved:,}/{n_total:,} ({100*resolved/n_total:.1f}%) resolved")
    ax.tick_params(axis="x", labelsize=8)

    # Panel 3 — Step 6: PDG of nearest DP entry for hits that have one
    ax = axes[1, 0]
    near_miss = unc_df[unc_df["has_dp_entry"] & (unc_df["nearest_dp_pdg"] != -999)]
    if not near_miss.empty:
        pdg_counts = near_miss["nearest_dp_pdg"].value_counts().head(12)
        ax.bar([str(p) for p in pdg_counts.index], pdg_counts.values,
               color="darkorange", edgecolor="white")
        ax.set_xlabel("PDG code")
        ax.set_ylabel("Count")
        ax.set_title("Step 6: PDG of nearest DP entry\n(for unclaimed hits with a near-miss)")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    else:
        ax.text(0.5, 0.5, "No DP entries found\nfor unclaimed hits",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Step 6: PDG distribution (no data)")
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 4 — Summary breakdown (horizontal bars; categories are non-exclusive)
    ax = axes[1, 1]
    c1 = int((
        (unc_df["nearest_dp_is_darknoise"] == 1) &
        ~unc_df["matched_wide"] &
        (unc_df["hit_pe"].fillna(0) <= 1.5)
    ).sum())
    c2 = int((
        unc_df["matched_wide"] |
        ((unc_df["nearest_dp_pdg"] != -999) & (unc_df["nearest_dp_pdg"] != -1))
    ).sum())
    c3 = int((~unc_df["has_dp_entry"] & ~unc_df["matched_wide"]).sum())
    bar_vals   = [c1, c2, c3]
    bar_labels = [
        f"Confirmed dark noise\n(IsDN=1, PE≤1.5, no wide match)",
        f"BackTracker issue\n(wide-matched or near-miss PDG found)",
        f"Genuinely untraceable\n(no DP entry, no wide match)",
    ]
    bar_colors = ["#9E9E9E", "#FF9800", "#607D8B"]
    bars = ax.barh(bar_labels, bar_vals, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, bar_vals):
        if val:
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}  ({100*val/n_total:.1f}%)",
                    va="center", fontsize=8)
    ax.set_xlabel("Unclaimed CF hits")
    ax.set_title(f"Summary of {n_total:,} unclaimed CF hits\n"
                 "(categories non-exclusive — hits may appear in multiple bars)")
    ax.set_xlim(0, max(bar_vals) * 1.35 if bar_vals else 1)

    plt.tight_layout()
    save_plot(fig, ctx, "unclaimed_diag_meta")


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

    ev_df, cl_df, unc_df, labeled_cf_pe = extract(root_path)

    ev_csv  = ctx.csv_path(f"{ctx.run_name}__hit_accounting")
    cl_csv  = ctx.csv_path(f"{ctx.run_name}__cluster_composition")
    unc_csv = ctx.csv_path(f"{ctx.run_name}__unclaimed_diagnostics")
    ev_df.to_csv(ev_csv,   index=False)
    cl_df.to_csv(cl_csv,   index=False)
    unc_df.to_csv(unc_csv, index=False)

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
            grand = cl_df[["neutron", "background", "darknoise", "unclaimed"]].sum().sum()
            print(f"  {lbl:<12s}: {total:7,}  ({100*total/grand:.1f}%)")
        print(f"\n--- Neutron hit recovery ---")
        has_n = ev_df[ev_df["total_neutron"] > 0]
        rec = has_n["cf_neutron"] / has_n["total_neutron"]
        print(f"  Median fraction of event's neutron hits captured in CF: {rec.median()*100:.1f}%")
        print(f"  Mean:   {rec.mean()*100:.1f}%")

        if len(unc_df):
            nu = len(unc_df)
            c1_mask = (
                (unc_df["nearest_dp_is_darknoise"] == 1) &
                ~unc_df["matched_wide"] &
                (unc_df["hit_pe"].fillna(0) <= 1.5)
            )
            c2_mask = (
                unc_df["matched_wide"] |
                ((unc_df["nearest_dp_pdg"] != -999) & (unc_df["nearest_dp_pdg"] != -1))
            )
            c3_mask = ~unc_df["has_dp_entry"] & ~unc_df["matched_wide"]
            print(f"\n--- Unclaimed hit partition ({nu:,} CF cluster hits) ---")
            print(f"  1. Confirmed dark noise      : {c1_mask.sum():6,} ({100*c1_mask.mean():.1f}%)")
            print(f"  2. BackTracker misclassified : {c2_mask.sum():6,} ({100*c2_mask.mean():.1f}%)")
            print(f"  3. Genuinely untraceable     : {c3_mask.sum():6,} ({100*c3_mask.mean():.1f}%)")
            print(f"  (categories may overlap; non-exclusive by design)")

        print(f"\n[hitcomp] wrote {ev_csv}")
        print(f"[hitcomp] wrote {cl_csv}")
        print(f"[hitcomp] wrote {unc_csv}")

    _accounting_plot(ev_df, ctx)
    _composition_plot(cl_df, ctx)
    _neutron_recovery_plot(ev_df, ctx)
    _unclaimed_timing_pe_plots(unc_df, labeled_cf_pe, ctx)
    _unclaimed_meta_plots(unc_df, ctx)

    return ev_csv


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx)
