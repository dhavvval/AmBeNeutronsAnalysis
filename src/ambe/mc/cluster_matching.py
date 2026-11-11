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
from .optics import assign_clusterfinder_labels, MIN_MATCH_FRAC as _MIN_MATCH_FRAC

NEUTRON_PDG = 2112


# --------------------------------------------------------------------------- #
# Build truth cluster table from pulses
# --------------------------------------------------------------------------- #

def build_truth_clusters(pulses: pd.DataFrame) -> pd.DataFrame:
    """
    One row per unique (eventID, ancestor_trackID) where PDG=2112.
    Mirrors the ROOT macro: each unique neutron trackID is one truth cluster.
    """
    # The sentinel is -5, not -1: BackTracker.cpp initialises neutronAncestorId to -5.
    # This has been harmless only because the ancestor_pdg == 2112 term carries the
    # filter on its own; `>= 0` says what is meant.
    neutron_hits = pulses[
        (pulses["ancestor_pdg"] == NEUTRON_PDG) &
        (pulses["ancestor_trackID"] >= 0)
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

def run_legacy(ctx: RunContext, verbose: bool = True) -> Path:
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


# =========================================================================== #
# PROVENANCE: when an event yields >1 neutron-like cluster, is that one neutron
# split across two clusters, or two different neutrons?
# =========================================================================== #
#
# The dataset makes the question sharp: every AmBe MC event has exactly ONE primary
# neutron (PDG 2112) and at most one true capture. So a genuine "different neutrons"
# case REQUIRES a secondary neutron -- NeutronAncestorClass 2 (from a proton), 3 (from
# a neutron) or 4 (other). The class breakdown is therefore part of the answer, not a
# diagnostic.
#
# WHY EXACT MEMBERSHIP (Cluster_Hit*) AND NOT THE +-WINDOW.
# optics.assign_clusterfinder_labels assigns each hit to the cluster whose clusterTime
# is nearest within (-5, +20) ns. A split pair is by hypothesis two clusters tens of ns
# apart -- exactly the regime where that rule reassigns hits away from ClusterFinder's
# own decision. It can MANUFACTURE a split (giving cluster B enough of A's hits to push
# B's dominant fraction over threshold) and it can destroy one (starving B below the hit
# floor). It also drops every hit more than 20 ns after clusterTime, which is precisely
# the straggling light that distinguishes one capture from two. And its member count is
# not clusterHits, so the label would describe a cluster object that neither the box nor
# the MVA ever acts on. Window mode is kept, but as a cross-check (CC5), not the answer.
#
# WHY frac_dom_traced AND NOT THE EXISTING frac_dom_all.
# match_clusters:126 divides the dominant-trackID count by ALL members, untraced ones
# included. With the DirectParent arrays running a few percent short and the box
# requiring only clusterHits >= 5, a 5-hit cluster with 3 untraced hits scores 0.40 and
# is declared spurious. The clusters that get split off are the small ones, so that rule
# is biased AGAINST calling a split a split, in a hit-count-dependent way. Both
# fractions are stored; the label uses the traced one.

import argparse as _argparse
import re as _re
from collections import Counter as _Counter

import awkward as _ak
import uproot as _uproot

from ..io import resolve_inputs as _resolve_inputs, filter_files_with_tree as _filter_tree
from .processor import (_compute_offset, _build_dp_lookup, _match_hit,
                        MATCH_TOL as _MATCH_TOL)

PROV_CATEGORIES = ("SPLIT", "DISTINCT_NEUTRONS", "SPLIT_AND_DISTINCT",
                   "NEUTRON_PLUS_BACKGROUND", "BACKGROUND_ONLY", "AMBIGUOUS")

PROV_DEFAULTS = dict(
    min_frac=0.60,          # frac_dom_traced for a decisive neutron label
    weak_frac=0.40,         # floor for the "weak" neutron label
    min_hits=3,             # minimum hits carrying the dominant trackID
    runnerup_frac=0.30,     # runner-up share that makes a cluster shared
    min_traced_frac=0.50,   # below this, nothing can be said about the cluster
    bg_frac=0.60,
    control_events_per_file=40,
)

# Box cut, byte-identical to data/selection.py:115-119.
PROV_BOX = dict(pe_min=0.0, pe_max=100.0, ccb_min=0.0, ccb_max=0.45,
                ct_min=2000.0, chits_min=5)
COSMIC_CT, COSMIC_PE = 2000.0, 100.0

PROV_BRANCHES = [
    "runNumber", "eventNumber", "numberOfClusters",
    "clusterTime", "clusterPE", "clusterChargeBalance", "clusterHits",
    "Cluster_HitChankey", "Cluster_HitT", "Cluster_HitPE",
    "Cluster_HitX", "Cluster_HitY", "Cluster_HitZ",
    "hitChankey", "hitT",
    "DirectParent_PMTID", "DirectParent_HitTime",
    "DirectParent_NeutronAncestorTrackID", "DirectParent_NeutronAncestorPDG",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
    "DirectParent_ImmediateAncestorClass", "DirectParent_ImmediateAncestorPDG",
    "DirectParent_ImmediateAncestorTrackID",
    "DirectParent_LineageStatus", "DirectParent_TrackIDs",
    "trueNeutCapTime", "trueNeutCapVtxX", "trueNeutCapVtxY", "trueNeutCapVtxZ",
    "trueNeutCapE", "trueNeutCapGammas", "trueNeutCapNucleus",
]

_PROV_LABEL_RE = _re.compile(r"^port(?P<port>\d+)_(?P<z>z(?:minus)?\d+)QE")
_PROV_Z = {"z0": 0.0, "z50": 50.0, "z100": 100.0,
           "zminus50": -50.0, "zminus100": -100.0}


def _label_from_path(path: Path, label_index: dict):
    """(run, label, port, z_cm) for one AmBe MC file, cross-checked against label_map."""
    label = path.parent.name
    m = _PROV_LABEL_RE.match(label)
    if not m or label not in label_index:
        raise SystemExit(f"[provenance] {path}: unparsable or unmapped label {label!r}")
    fm = _re.search(r"_(\d+)_(\d+)\.root$", path.name)
    if not fm:
        raise SystemExit(f"[provenance] cannot parse file index from {path}")
    run = label_index[label] * 1000 + int(fm.group(1)) // 1000
    return run, label, int(m.group("port")), _PROV_Z[m.group("z")]


def _load_label_index(label_map_path: str) -> dict:
    idx = {}
    with open(label_map_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            i, lab = line.split()
            idx[lab] = int(i)
    if not idx:
        raise SystemExit(f"[provenance] no labels parsed from {label_map_path}")
    return idx


def _passes_box(pe, ccb, ct, chits) -> bool:
    b = PROV_BOX
    return (b["pe_min"] < pe <= b["pe_max"] and b["ccb_min"] < ccb < b["ccb_max"]
            and ct >= b["ct_min"] and chits >= b["chits_min"])


def _label_cluster(member_truth, n_hits, cfg) -> dict:
    """Per-cluster provenance label from the DirectParent verdicts of its member hits.

    `member_truth` is a list of (truth_class, is_dn, trackID, pdg, is_untraced,
    fail_reason) tuples, one per member hit, straight out of _match_hit.
    """
    n_traced = n_no_entry = n_time_mismatch = n_bg = n_dn = 0
    tids = []
    for (cls, is_dn, tid, pdg, untraced, reason) in member_truth:
        if reason == "no_entry":
            n_no_entry += 1
        elif reason == "time_mismatch":
            n_time_mismatch += 1
        if untraced:
            continue
        n_traced += 1
        if is_dn or cls == 0:
            n_dn += 1
        elif pdg == NEUTRON_PDG and tid >= 0:
            tids.append((tid, cls))
        elif cls == -5:
            n_bg += 1

    counts = _Counter(t for t, _c in tids)
    ranked = counts.most_common(2)
    dom_tid, n_dom = (ranked[0] if ranked else (-1, 0))
    tid2, n_tid2 = (ranked[1] if len(ranked) > 1 else (-1, 0))
    dom_tie = bool(len(ranked) > 1 and ranked[0][1] == ranked[1][1])
    dom_class = next((c for t, c in tids if t == dom_tid), -5)

    denom = max(n_traced, 1)
    frac_dom_all = n_dom / max(n_hits, 1)
    frac_dom_traced = n_dom / denom

    # First match wins. mixed_neutrons is tested BEFORE the neutron labels: a cluster
    # genuinely carrying two neutrons' light must not be awarded to the larger one --
    # that is exactly the error that turns a DISTINCT event into a SPLIT event.
    if n_traced / max(n_hits, 1) < cfg["min_traced_frac"]:
        lab = "untraced"
    elif n_tid2 >= cfg["min_hits"] and n_tid2 / denom >= cfg["runnerup_frac"]:
        lab = "mixed_neutrons"
    elif n_dom >= cfg["min_hits"] and frac_dom_traced >= cfg["min_frac"]:
        lab = "neutron"
    elif n_dom >= cfg["min_hits"] and frac_dom_traced >= cfg["weak_frac"]:
        lab = "neutron_weak"
    elif n_bg / denom >= cfg["bg_frac"]:
        lab = "background"
    elif n_dn / denom >= cfg["bg_frac"]:
        lab = "darknoise"
    else:
        lab = "ambiguous"

    return dict(n_traced=n_traced, n_neutron=len(tids), n_bg=n_bg, n_dn=n_dn,
                n_no_entry=n_no_entry, n_time_mismatch=n_time_mismatch,
                dom_tid=int(dom_tid), n_dom=int(n_dom), dom_class=int(dom_class),
                dom_tie=dom_tie, tid2=int(tid2), n_tid2=int(n_tid2),
                frac_tid2_traced=n_tid2 / denom,
                frac_dom_all=frac_dom_all, frac_dom_traced=frac_dom_traced,
                prov_label=lab)


_NEUTRON_LABELS = ("neutron", "neutron_weak")
_UNUSABLE_LABELS = ("ambiguous", "untraced", "mixed_neutrons")


def _classify_event(rows) -> dict:
    """Category for one event's ACCEPTED clusters. Mutually exclusive and exhaustive."""
    tids = [r["dom_tid"] for r in rows if r["prov_label"] in _NEUTRON_LABELS]
    n_nonneutron = sum(1 for r in rows if r["prov_label"] in ("background", "darknoise"))
    n_unusable = sum(1 for r in rows if r["prov_label"] in _UNUSABLE_LABELS)
    c = _Counter(tids)
    n_distinct = len(c)
    max_per_tid = max(c.values()) if c else 0

    # AMBIGUOUS first, so no other category is contaminated by a cluster we could not
    # label. A 3-cluster event is handled by this partition, never by a pairwise
    # reduction; SPLIT_AND_DISTINCT is a real configuration there (two clusters from
    # neutron A, one from B) and folding it away would understate SPLIT and DISTINCT.
    if n_unusable:
        cat = "AMBIGUOUS"
    elif max_per_tid >= 2 and n_distinct >= 2:
        cat = "SPLIT_AND_DISTINCT"
    elif max_per_tid >= 2:
        cat = "SPLIT"
    elif n_distinct >= 2:
        cat = "DISTINCT_NEUTRONS"
    elif n_distinct == 1 and n_nonneutron:
        cat = "NEUTRON_PLUS_BACKGROUND"
    elif n_distinct == 0:
        cat = "BACKGROUND_ONLY"
    else:
        cat = "NEUTRON_PLUS_BACKGROUND"
    return dict(n_accepted=len(rows), n_distinct_tids=n_distinct,
                max_per_tid=max_per_tid, n_nonneutron=n_nonneutron,
                n_unusable=n_unusable, category=cat)


def _window_members(hit_ck, hit_t, cluster_times, prompt_ns=2000.0,
                    tmin=-5.0, tmax=20.0):
    """CC5 membership: the +-window rule, on the same residual hits the MVA sees."""
    keep = hit_t > prompt_ns
    ck, t = hit_ck[keep], hit_t[keep]
    out = [[] for _ in cluster_times]
    if len(cluster_times) == 0 or len(t) == 0:
        return out, ck, t
    for j in range(len(t)):
        d = t[j] - cluster_times
        ok = np.where((d > tmin) & (d < tmax))[0]
        if len(ok):
            out[int(ok[np.argmin(np.abs(d[ok]))])].append(j)
    return out, ck, t


def extract_provenance(root_path, run, label, port, z_cm, mode, cfg, seed):
    """Two-tier per-file extraction: everything vectorized, the DP join on few events.

    Tier A (all events, awkward): cluster inventory, box/veto flags, capture truth.
    Tier B (Python loop): only events with >=2 clusters (~1.1 %) plus a seeded sample
    of single-cluster control events for the capture-time cross-check. That is ~50
    events per 1000, which is what makes this minutes rather than hours.
    """
    rng = np.random.default_rng(seed)
    cl_rows, ev_rows, pair_rows = [], [], []
    qa = dict(run=run, label=label, file=Path(root_path).name)

    with _uproot.open(str(root_path)) as fh:
        a = fh["Event"].arrays(PROV_BRANCHES, library="ak")

    n_ev = len(a)
    ncl = _ak.to_numpy(a["numberOfClusters"])
    ct_all = a["clusterTime"]
    pe_all = a["clusterPE"]

    # Event-level prompt veto, identical to Selection.cosmic.
    vetoed = _ak.to_numpy(_ak.any((ct_all < COSMIC_CT) | (pe_all > COSMIC_PE), axis=1))

    multi_idx = np.where(ncl >= 2)[0]
    single_idx = np.where(ncl == 1)[0]
    n_ctrl = min(int(cfg["control_events_per_file"]), len(single_idx))
    ctrl_idx = rng.choice(single_idx, size=n_ctrl, replace=False) if n_ctrl else np.array([], int)
    todo = np.concatenate([multi_idx, ctrl_idx]).astype(int)
    is_ctrl = set(int(i) for i in ctrl_idx)

    qa.update(n_events=n_ev, n_multi=len(multi_idx), n_control=len(ctrl_idx),
              n_vetoed=int(vetoed.sum()))

    n_offset_fallback = 0
    n_hits_mismatch = 0
    n_unmapped = 0
    n_collisions = 0
    n_wide_recovered = 0

    for i in todo:
        i = int(i)
        hit_ck = _ak.to_numpy(a["hitChankey"][i]).astype(int)
        hit_t = _ak.to_numpy(a["hitT"][i]).astype(float)
        dp_ck = _ak.to_numpy(a["DirectParent_PMTID"][i]).astype(int)
        dp_t = _ak.to_numpy(a["DirectParent_HitTime"][i]).astype(float)
        if len(dp_ck) == 0:
            continue
        offset = _compute_offset(hit_ck, hit_t, dp_ck, dp_t)
        if not (10.0 <= offset <= 25.0):
            n_offset_fallback += 1

        dp_tid = _ak.to_numpy(a["DirectParent_NeutronAncestorTrackID"][i]).astype(int)
        dp_pdg = _ak.to_numpy(a["DirectParent_NeutronAncestorPDG"][i]).astype(int)
        dp_cls = _ak.to_numpy(a["DirectParent_NeutronAncestorClass"][i]).astype(int)
        dp_dn = _ak.to_numpy(a["DirectParent_IsDarknoise"][i]).astype(int)
        dp_bgc = _ak.to_numpy(a["DirectParent_ImmediateAncestorClass"][i]).astype(int)
        dp_bgp = _ak.to_numpy(a["DirectParent_ImmediateAncestorPDG"][i]).astype(int)
        dp_bgt = _ak.to_numpy(a["DirectParent_ImmediateAncestorTrackID"][i]).astype(int)
        lookup = _build_dp_lookup(dp_ck, dp_t, dp_cls, dp_dn, dp_tid, dp_pdg,
                                  dp_bgc, dp_bgp, dp_bgt)

        # CC6: how many DP entries were fed by more than one photon. BackTracker
        # attributes such a pulse entirely to directParents[0], so a high rate here
        # would mean the split/distinct call rests on a first-photon coin flip.
        try:
            npar = _ak.to_numpy(_ak.num(a["DirectParent_TrackIDs"][i], axis=1))
            multiparent_ck = set(int(c) for c, nn in zip(dp_ck, npar) if nn > 1)
        except Exception:
            multiparent_ck = set()

        # Flat-hit key set, for the QA degeneracy counters only.
        flat_keys = _Counter((int(c), round(float(t), 4)) for c, t in zip(hit_ck, hit_t))
        n_collisions += sum(1 for v in flat_keys.values() if v > 1)

        cl_ck = a["Cluster_HitChankey"][i]
        cl_t = a["Cluster_HitT"][i]
        cl_pe = a["Cluster_HitPE"][i]
        cl_x, cl_y, cl_z = a["Cluster_HitX"][i], a["Cluster_HitY"][i], a["Cluster_HitZ"][i]
        ctimes = _ak.to_numpy(ct_all[i]).astype(float)
        cpes = _ak.to_numpy(pe_all[i]).astype(float)
        ccbs = _ak.to_numpy(a["clusterChargeBalance"][i]).astype(float)
        chits = _ak.to_numpy(a["clusterHits"][i]).astype(int)

        if mode == "window":
            wm = _window_members(hit_ck, hit_t, ctimes)
            wmembers, wck, wt = wm

        ev_cl_rows = []
        for c in range(len(ctimes)):
            if mode == "window":
                idxs = wmembers[c]
                mck = wck[idxs].astype(int)
                mt = wt[idxs].astype(float)
                mpe = np.zeros(len(idxs))
                mx = my = mz = np.zeros(len(idxs))
            else:
                mck = _ak.to_numpy(cl_ck[c]).astype(int)
                mt = _ak.to_numpy(cl_t[c]).astype(float)
                mpe = _ak.to_numpy(cl_pe[c]).astype(float)
                mx = _ak.to_numpy(cl_x[c]).astype(float)
                my = _ak.to_numpy(cl_y[c]).astype(float)
                mz = _ak.to_numpy(cl_z[c]).astype(float)
                if len(mck) != int(chits[c]):
                    n_hits_mismatch += 1
                n_unmapped += sum(1 for k, t in zip(mck, mt)
                                  if (int(k), round(float(t), 4)) not in flat_keys)

            truth = []
            for k, t in zip(mck, mt):
                v = _match_hit(int(k), float(t), lookup, offset)
                truth.append(v[:6])
                # Wide-tolerance QA: how many time_mismatch hits a 25 ns window would
                # have recovered. Reported only -- never fed into the label, so the
                # answer's sensitivity to the tolerance stays visible instead of baked in.
                if v[5] == "time_mismatch" and \
                        _match_hit(int(k), float(t), lookup, offset, tol=25.0)[5] is None:
                    n_wide_recovered += 1
            lab = _label_cluster(truth, len(mck), cfg)

            npe = float(np.sum(mpe)) if len(mpe) else 0.0
            w = mpe if npe > 0 else np.ones(len(mx))
            cx = float(np.average(mx, weights=w)) if len(mx) else np.nan
            cy = float(np.average(my, weights=w)) if len(my) else np.nan
            cz = float(np.average(mz, weights=w)) if len(mz) else np.nan
            nmp = sum(1 for k in mck if int(k) in multiparent_ck)

            row = dict(runNumber=run, label=label, port=port, z_cm=z_cm,
                       eventNumber=int(_ak.to_numpy(a["eventNumber"][i:i + 1])[0]),
                       cluster_idx=c, numberOfClusters=int(ncl[i]),
                       membership_mode=mode, dp_offset_ns=offset,
                       clusterTime=float(ctimes[c]), clusterPE=float(cpes[c]),
                       clusterHits=int(chits[c]),
                       clusterChargeBalance=float(ccbs[c]),
                       cx_m=cx, cy_m=cy, cz_m=cz,
                       n_member_hits=len(mck),
                       n_multiparent_hits=nmp,
                       frac_multiparent=nmp / max(len(mck), 1),
                       event_vetoed=bool(vetoed[i]),
                       is_control=bool(i in is_ctrl),
                       sel_box=_passes_box(float(cpes[c]), float(ccbs[c]),
                                           float(ctimes[c]), int(chits[c])))
            row.update(lab)
            ev_cl_rows.append(row)
            cl_rows.append(row)

        # ---- event row + truth capture ----
        tc = _ak.to_numpy(a["trueNeutCapTime"][i]).astype(float)
        ev = dict(runNumber=run, label=label, port=port, z_cm=z_cm,
                  eventNumber=int(_ak.to_numpy(a["eventNumber"][i:i + 1])[0]),
                  n_clusters=int(ncl[i]), event_vetoed=bool(vetoed[i]),
                  is_control=bool(i in is_ctrl), n_truecap=len(tc),
                  truecap_t0=float(tc[0]) if len(tc) else np.nan,
                  truecap_t1=float(tc[1]) if len(tc) > 1 else np.nan)
        if len(tc):
            ev["truecap_vx"] = float(_ak.to_numpy(a["trueNeutCapVtxX"][i])[0])
            ev["truecap_vy"] = float(_ak.to_numpy(a["trueNeutCapVtxY"][i])[0])
            ev["truecap_vz"] = float(_ak.to_numpy(a["trueNeutCapVtxZ"][i])[0])
            ev["truecap_nucleus"] = float(_ak.to_numpy(a["trueNeutCapNucleus"][i])[0])
            ev["truecap_gammas"] = float(_ak.to_numpy(a["trueNeutCapGammas"][i])[0])

        # dt_cap per cluster, for CC1
        for r in ev_cl_rows:
            r["dt_cap_ns"] = (r["clusterTime"] - ev["truecap_t0"]
                              if len(tc) else np.nan)

        # Variants. `all` is every cluster; the four selection variants restrict to
        # accepted clusters under the two neutron definitions x the two veto settings.
        # The MVA variants are filled later, once scores are joined; a missing score
        # must stay NaN, never False -- False reads downstream as "the MVA rejected it".
        for vname, veto in (("all", None), ("box_vetoON", True), ("box_vetoOFF", False)):
            if vname == "all":
                sel = ev_cl_rows
            elif veto and vetoed[i]:
                sel = []
            else:
                sel = [r for r in ev_cl_rows if r["sel_box"]]
            ev[f"{vname}_n_accepted"] = len(sel)
            if len(sel) >= 2:
                cls = _classify_event(sel)
                for k, v in cls.items():
                    ev[f"{vname}_{k}"] = v
                for x in range(len(sel)):
                    for y in range(x + 1, len(sel)):
                        ra, rb = sel[x], sel[y]
                        pair_rows.append(dict(
                            runNumber=run, label=label, port=port, z_cm=z_cm,
                            eventNumber=ev["eventNumber"], variant=vname,
                            cluster_idx_a=ra["cluster_idx"], cluster_idx_b=rb["cluster_idx"],
                            tid_a=ra["dom_tid"], tid_b=rb["dom_tid"],
                            same_tid=bool(ra["dom_tid"] >= 0 and ra["dom_tid"] == rb["dom_tid"]),
                            class_a=ra["dom_class"], class_b=rb["dom_class"],
                            label_a=ra["prov_label"], label_b=rb["prov_label"],
                            dt_ns=abs(ra["clusterTime"] - rb["clusterTime"]),
                            dd_m=float(np.sqrt((ra["cx_m"] - rb["cx_m"]) ** 2
                                               + (ra["cy_m"] - rb["cy_m"]) ** 2
                                               + (ra["cz_m"] - rb["cz_m"]) ** 2)),
                            pe_a=ra["clusterPE"], pe_b=rb["clusterPE"],
                            hits_a=ra["clusterHits"], hits_b=rb["clusterHits"],
                            dt_cap_a=ra["dt_cap_ns"], dt_cap_b=rb["dt_cap_ns"],
                            n_truecap=len(tc)))
        ev_rows.append(ev)

    qa.update(n_offset_fallback=n_offset_fallback,
              n_clusterhits_mismatch=n_hits_mismatch,
              n_unmapped_cluster_hits=n_unmapped,
              n_key_collisions=n_collisions,
              n_wide_recovered=n_wide_recovered)
    return (pd.DataFrame(cl_rows), pd.DataFrame(ev_rows),
            pd.DataFrame(pair_rows), qa)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
# PLOT STYLE IS THE HOUSE STYLE AND IS ENFORCED HERE: title only. Explicit
# ax.grid(False) everywhere; no axvline/axhline, no diagonals, no ax.text, no
# annotate, no bar_label, no suptitle, no caption block. Every threshold, median and
# count is folded into the title string. The four legacy plots above break all of
# that; they belong to --mode legacy and are left alone.

# Plot filenames carry the membership mode too, for the same reason the tables do:
# a window cross-check that silently overwrote the exact figures would leave the deck
# showing one mode while the text quoted the other.
_PROV_SFX = ""


def _sp(fig, ctx, name):
    save_plot(fig, ctx, f"{name}{_PROV_SFX}")


def _wilson(k, n, z=1.0):
    """Wilson interval. The DISTINCT fraction is small and sqrt(N)/N fails there."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _cat_counts(ev: pd.DataFrame, variant: str) -> pd.DataFrame:
    col = f"{variant}_category"
    if col not in ev.columns:
        return pd.DataFrame()
    sub = ev[ev[col].notna()]
    n = len(sub)
    rows = []
    for cat in PROV_CATEGORIES:
        k = int((sub[col] == cat).sum())
        lo, hi = _wilson(k, n)
        rows.append(dict(variant=variant, category=cat, n_events=k,
                         frac=k / n if n else np.nan, ci_lo=lo, ci_hi=hi,
                         n_total=n))
    return pd.DataFrame(rows)


def _p_category_bar(counts: pd.DataFrame, ctx, variant="all"):
    d = counts[counts.variant == variant]
    if d.empty:
        return
    n = int(d.n_total.iloc[0])
    frac = {r.category: 100 * r.frac for r in d.itertuples()}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(frac.keys())[::-1], list(frac.values())[::-1], color="steelblue")
    ax.set_xlabel("Fraction of multi-cluster events (%)")
    ax.grid(False)
    ax.set_title(ctx.title(
        f"Origin of the extra cluster: split {frac.get('SPLIT', 0):.1f}%, "
        f"distinct neutrons {frac.get('DISTINCT_NEUTRONS', 0):.1f}%, "
        f"neutron+background {frac.get('NEUTRON_PLUS_BACKGROUND', 0):.1f}%, "
        f"ambiguous {frac.get('AMBIGUOUS', 0):.1f}% (N={n:,})"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_category_bar")


def _p_category_by_selection(counts: pd.DataFrame, ctx):
    variants = [v for v in counts.variant.unique() if v != "all"]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / max(len(variants), 1)
    x = np.arange(len(PROV_CATEGORIES))
    ns = []
    for j, v in enumerate(variants):
        d = counts[counts.variant == v].set_index("category").reindex(PROV_CATEGORIES)
        ns.append(int(d.n_total.iloc[0]) if d.n_total.notna().any() else 0)
        ax.bar(x + j * w, 100 * d.frac.values, width=w, label=v)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(PROV_CATEGORIES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of multi-cluster events (%)")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title("Extra-cluster origin by selection, N = "
                           + "/".join(f"{n:,}" for n in ns)))
    fig.tight_layout()
    _sp(fig, ctx, "prov_category_by_selection")


def _p_pair_dt(pairs: pd.DataFrame, ctx):
    d = pairs[pairs.variant == "all"]
    same = d[d.same_tid].dt_ns
    diff = d[~d.same_tid].dt_ns
    if not len(same) and not len(diff):
        return
    fig, ax = plt.subplots()
    bins = np.logspace(0, 5, 60)
    if len(same):
        ax.hist(same, bins=bins, histtype="step", lw=1.6,
                label=f"same neutron ({len(same):,})")
    if len(diff):
        ax.hist(diff, bins=bins, histtype="step", lw=1.6,
                label=f"different neutrons ({len(diff):,})")
    ax.set_xscale("log")
    ax.set_xlabel("Cluster-pair Δt (ns)")
    ax.set_ylabel("Pairs")
    ax.grid(False)
    ax.legend(frameon=False)
    m1 = np.median(same) if len(same) else np.nan
    m2 = np.median(diff) / 1000 if len(diff) else np.nan
    ax.set_title(ctx.title(f"Cluster-pair time separation: same neutron median "
                           f"{m1:.0f} ns, different neutrons median {m2:.1f} µs"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_pair_dt")


def _p_pair_dd(pairs: pd.DataFrame, ctx):
    d = pairs[(pairs.variant == "all") & pairs.dd_m.notna()]
    same, diff = d[d.same_tid].dd_m, d[~d.same_tid].dd_m
    if not len(same) and not len(diff):
        return
    fig, ax = plt.subplots()
    bins = np.linspace(0, 6, 50)
    if len(same):
        ax.hist(same, bins=bins, histtype="step", lw=1.6,
                label=f"same neutron ({len(same):,})")
    if len(diff):
        ax.hist(diff, bins=bins, histtype="step", lw=1.6,
                label=f"different neutrons ({len(diff):,})")
    ax.set_xlabel("Cluster-pair centroid separation (m)")
    ax.set_ylabel("Pairs")
    ax.grid(False)
    ax.legend(frameon=False)
    m1 = np.median(same) if len(same) else np.nan
    m2 = np.median(diff) if len(diff) else np.nan
    ax.set_title(ctx.title(f"Cluster-pair spatial separation: same neutron median "
                           f"{m1:.2f} m, different neutrons median {m2:.2f} m"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_pair_dd")


def _p_domfrac(cl: pd.DataFrame, ctx, cfg):
    fig, ax = plt.subplots()
    for lab in ("neutron", "neutron_weak", "mixed_neutrons", "ambiguous"):
        d = cl[cl.prov_label == lab].frac_dom_traced
        if len(d):
            ax.hist(d, bins=40, range=(0, 1), histtype="step", lw=1.5,
                    label=f"{lab} ({len(d):,})")
    ax.set_xlabel("Dominant-neutron fraction of traced hits")
    ax.set_ylabel("Clusters")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title(f"Dominant-neutron fraction of traced hits, label "
                           f"threshold {cfg['min_frac']:.2f}, median "
                           f"{cl.frac_dom_traced.median():.2f}"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_domfrac")


def _p_untraced(cl: pd.DataFrame, ctx):
    u = 1 - cl.n_traced / cl.n_member_hits.clip(lower=1)
    s = u[cl.numberOfClusters == 1]
    m = u[cl.numberOfClusters >= 2]
    fig, ax = plt.subplots()
    for d, nm in ((s, "single-cluster events"), (m, "multi-cluster events")):
        if len(d):
            ax.hist(d, bins=40, range=(0, 1), histtype="step", lw=1.6,
                    label=f"{nm} ({len(d):,})")
    ax.set_xlabel("Untraced fraction of cluster hits")
    ax.set_ylabel("Clusters")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title(f"Untraced hit fraction per cluster: median "
                           f"{s.median() if len(s) else float('nan'):.3f} single-cluster, "
                           f"{m.median() if len(m) else float('nan'):.3f} multi-cluster"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_untraced")


def _p_dtcap(cl: pd.DataFrame, ev: pd.DataFrame, ctx, band):
    ctrl = cl[(cl.is_control) & (cl.prov_label == "neutron")].dt_cap_ns.dropna()
    key = ev.set_index(["runNumber", "eventNumber"])
    cats = key["all_category"] if "all_category" in key.columns else None
    fig, ax = plt.subplots()
    bins = np.linspace(-2000, 6000, 80)
    if len(ctrl):
        ax.hist(ctrl, bins=bins, histtype="step", lw=1.6,
                label=f"single-cluster control ({len(ctrl):,})")
    if cats is not None:
        j = cl.join(cats, on=["runNumber", "eventNumber"])
        for cat, nm in (("SPLIT", "split"), ("DISTINCT_NEUTRONS", "distinct neutrons")):
            d = j[j.all_category == cat].dt_cap_ns.dropna()
            if len(d):
                ax.hist(d, bins=bins, histtype="step", lw=1.6,
                        label=f"{nm} ({len(d):,})")
    ax.set_xlabel("clusterTime − trueNeutCapTime (ns)")
    ax.set_ylabel("Clusters")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title(
        f"clusterTime minus trueNeutCapTime: control median {band[0]:.0f} ns, "
        f"IQR {band[1]:.0f} ns, band {band[2]:.0f} to {band[3]:.0f} ns"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_dtcap")


def _p_class_stack(cl: pd.DataFrame, ev: pd.DataFrame, ctx, frac_secondary):
    if "all_category" not in ev.columns:
        return
    j = cl.join(ev.set_index(["runNumber", "eventNumber"])["all_category"],
                on=["runNumber", "eventNumber"])
    cats = [c for c in PROV_CATEGORIES if (j.all_category == c).any()]
    classes = [1, 2, 3, 4, 0, -5]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(cats))
    for k in classes:
        vals = np.array([
            100 * ((j.all_category == c) & (j.dom_class == k)).sum()
            / max((j.all_category == c).sum(), 1) for c in cats])
        ax.bar(cats, vals, bottom=bottom, label=f"class {k}")
        bottom += vals
    ax.set_ylabel("Share of clusters (%)")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_title(ctx.title(f"Neutron ancestor class by category: {frac_secondary:.1f}% "
                           f"of distinct-neutron events contain a secondary neutron"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_class_stack")


def _p_position_heatmap(ev: pd.DataFrame, ctx):
    if "all_category" not in ev.columns:
        return
    sub = ev[ev.all_category.notna()]
    if sub.empty:
        return
    g = sub.groupby(["port", "z_cm"]).all_category.agg(
        lambda s: 100 * (s == "SPLIT").mean())
    piv = g.unstack("port")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"Port {c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:g}" for v in piv.index])
    ax.set_ylabel("Source height (cm)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Split fraction (%)")
    vals = piv.values[np.isfinite(piv.values)]
    ax.set_title(ctx.title(f"Split fraction of multi-cluster events by source position, "
                           f"range {vals.min():.0f}–{vals.max():.0f}%, "
                           f"mean {vals.mean():.0f}%"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_position_heatmap")


def _p_split_pair_size(pairs: pd.DataFrame, ctx):
    d = pairs[(pairs.variant == "all") & pairs.same_tid]
    if d.empty:
        return
    lo_pe = np.minimum(d.pe_a, d.pe_b); hi_pe = np.maximum(d.pe_a, d.pe_b)
    lo_h = np.minimum(d.hits_a, d.hits_b); hi_h = np.maximum(d.hits_a, d.hits_b)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (a1, a2, xl) in zip(axes, [(lo_pe, hi_pe, "Cluster PE"),
                                       (lo_h, hi_h, "Cluster hits")]):
        bins = np.linspace(0, float(np.percentile(a2, 99)), 40)
        ax.hist(a1, bins=bins, histtype="step", lw=1.6, label="smaller")
        ax.hist(a2, bins=bins, histtype="step", lw=1.6, label="larger")
        ax.set_xlabel(xl); ax.set_ylabel("Split pairs"); ax.grid(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("")
    axes[0].set_title(ctx.title(
        f"Split pairs: smaller {np.median(lo_pe):.0f} PE / {np.median(lo_h):.0f} hits, "
        f"larger {np.median(hi_pe):.0f} PE / {np.median(hi_h):.0f} hits "
        f"({len(d):,} pairs)"), fontsize=9)
    fig.tight_layout()
    _sp(fig, ctx, "prov_split_pair_size")


def _p_split_survival(cl: pd.DataFrame, ev: pd.DataFrame, ctx):
    if "all_category" not in ev.columns:
        return
    splits = ev[ev.all_category.isin(["SPLIT", "SPLIT_AND_DISTINCT"])]
    if splits.empty:
        return
    keys = set(zip(splits.runNumber, splits.eventNumber))
    sub = cl[[k in keys for k in zip(cl.runNumber, cl.eventNumber)]]
    g = sub.groupby(["runNumber", "eventNumber"]).sel_box.agg(["sum", "size"])
    both = int((g["sum"] >= 2).sum()); one = int((g["sum"] == 1).sum())
    none = int((g["sum"] == 0).sum()); tot = max(len(g), 1)
    fig, ax = plt.subplots()
    ax.bar(["both kept", "one kept", "neither kept"],
           [100 * both / tot, 100 * one / tot, 100 * none / tot], color="steelblue")
    ax.set_ylabel("Fraction of split events (%)")
    ax.grid(False)
    ax.set_title(ctx.title(f"Survival of split pairs under the box cut: both in "
                           f"{100*both/tot:.1f}%, one in {100*one/tot:.1f}%, "
                           f"neither in {100*none/tot:.1f}% of {tot:,} split events"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_split_survival")


def _p_multiparent(cl: pd.DataFrame, ev: pd.DataFrame, ctx):
    if "all_category" not in ev.columns:
        return
    j = cl.join(ev.set_index(["runNumber", "eventNumber"])["all_category"],
                on=["runNumber", "eventNumber"])
    fig, ax = plt.subplots()
    ms = {}
    for cat in ("SPLIT", "DISTINCT_NEUTRONS"):
        d = j[j.all_category == cat].frac_multiparent.dropna()
        ms[cat] = d.median() if len(d) else np.nan
        if len(d):
            ax.hist(d, bins=40, range=(0, 1), histtype="step", lw=1.6,
                    label=f"{cat} ({len(d):,})")
    ax.set_xlabel("Fraction of cluster hits with more than one DirectParent")
    ax.set_ylabel("Clusters"); ax.grid(False); ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title(
        f"Multi-parent hit fraction: {ms.get('SPLIT', float('nan')):.3f} in split, "
        f"{ms.get('DISTINCT_NEUTRONS', float('nan')):.3f} in distinct-neutron events"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_multiparent")


def _p_ncap(ev: pd.DataFrame, ctx):
    s = ev[ev.n_clusters == 1].n_truecap
    m = ev[ev.n_clusters >= 2].n_truecap
    fig, ax = plt.subplots()
    bins = np.arange(-0.5, 5.5, 1)
    for d, nm in ((s, "single-cluster"), (m, "multi-cluster")):
        if len(d):
            ax.hist(d, bins=bins, histtype="step", lw=1.6,
                    label=f"{nm} events ({len(d):,})")
    ax.set_xlabel("True neutron captures in the event")
    ax.set_ylabel("Events"); ax.grid(False); ax.legend(frameon=False, fontsize=8)
    ax.set_title(ctx.title(f"True neutron captures per event: mean "
                           f"{s.mean() if len(s) else float('nan'):.3f} single-cluster, "
                           f"{m.mean() if len(m) else float('nan'):.3f} multi-cluster"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_ncap")


def _p_mode_confusion(cl_e: pd.DataFrame, cl_w: pd.DataFrame, ctx):
    key = ["runNumber", "eventNumber", "cluster_idx"]
    j = cl_e.set_index(key)[["prov_label"]].join(
        cl_w.set_index(key)[["prov_label"]], lsuffix="_exact", rsuffix="_window",
        how="inner")
    if j.empty:
        return np.nan
    labs = sorted(set(j.prov_label_exact) | set(j.prov_label_window))
    mat = np.array([[((j.prov_label_exact == a) & (j.prov_label_window == b)).sum()
                     for b in labs] for a in labs], float)
    agree = 100 * np.trace(mat) / mat.sum()
    row = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("window membership"); ax.set_ylabel("exact membership")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="row-normalised share")
    ax.set_title(ctx.title(f"Exact vs window hit-to-cluster membership: {agree:.1f}% "
                           f"of {len(j):,} clusters keep the same label"))
    fig.tight_layout()
    _sp(fig, ctx, "prov_mode_confusion")
    return agree


def _run_one_mode(ctx, args, cfg, mode, files, label_index):
    cls, evs, prs, qas = [], [], [], []
    for n, fp in enumerate(files, 1):
        run, label, port, z_cm = _label_from_path(Path(fp), label_index)
        c, e, pr, qa = extract_provenance(fp, run, label, port, z_cm, mode, cfg,
                                          seed=run)
        cls.append(c); evs.append(e); prs.append(pr); qas.append(qa)
        if n % 100 == 0 or n == len(files):
            print(f"  [{mode}] {n}/{len(files)} files", flush=True)
    cat = lambda fs: (pd.concat([f for f in fs if len(f)], ignore_index=True)
                      if any(len(f) for f in fs) else pd.DataFrame())
    return cat(cls), cat(evs), cat(prs), pd.DataFrame(qas)


def attach_mva_variants(ctx: RunContext, scored: dict, threshold: float,
                        score_col: str = "gbt_score") -> Path:
    """Add the two MVA variants to an already-extracted provenance table.

    The per-cluster provenance label does not depend on the selection -- it is a
    statement about which neutron the cluster's light came from. So the MVA variants
    are a re-partition of the SAME labelled clusters, not a second extraction: join the
    scores on, mark which clusters the MVA accepts, and re-run the event classifier.
    Re-extracting would spend hours reproducing labels that are already on disk.

    `scored` maps variant name -> scored parquet path. A cluster the scoring stage never
    saw gets sel_mva = NA, NEVER False: False reads downstream as "the MVA rejected it",
    which is a statement the data does not support. Those clusters are excluded from the
    variant entirely and counted, exactly as MvaSelection.report_coverage does.
    """
    clp = ctx.parquet_path(f"{ctx.run_name}__provenance_clusters")
    evp = ctx.parquet_path(f"{ctx.run_name}__provenance_events")
    cl = pd.read_parquet(clp)
    ev = pd.read_parquet(evp)

    for variant, path in scored.items():
        s = pd.read_parquet(path, columns=["run", "event_tank_time",
                                           "cf_clusterTime", score_col])
        acc = set(zip(s.loc[s[score_col] > threshold, "run"].astype("int64").tolist(),
                      s.loc[s[score_col] > threshold, "event_tank_time"].astype("int64").tolist(),
                      np.round(s.loc[s[score_col] > threshold, "cf_clusterTime"]
                               .to_numpy(float), 3).tolist()))
        seen = set(zip(s.run.astype("int64").tolist(),
                       s.event_tank_time.astype("int64").tolist(),
                       np.round(s.cf_clusterTime.to_numpy(float), 3).tolist()))
        keys = list(zip(cl.runNumber.astype("int64"), cl.eventNumber.astype("int64"),
                        np.round(cl.clusterTime.to_numpy(float), 3)))
        scored_mask = np.array([k in seen for k in keys])
        acc_mask = np.array([k in acc for k in keys])
        cl[f"sel_{variant}"] = pd.array(np.where(scored_mask, acc_mask, None),
                                        dtype="boolean")
        n_unscored = int((~scored_mask).sum())
        print(f"[provenance] {variant}: {int(acc_mask.sum()):,} of "
              f"{int(scored_mask.sum()):,} scored clusters accepted; "
              f"{n_unscored:,} of {len(cl):,} had NO score "
              f"({100*scored_mask.mean():.2f} % coverage)")

        veto_on = variant.endswith("vetoON")
        cats = {}
        for (r, e), g in cl.groupby(["runNumber", "eventNumber"], sort=False):
            if veto_on and bool(g.event_vetoed.iloc[0]):
                continue
            sel = g[g[f"sel_{variant}"].fillna(False).astype(bool)]
            if len(sel) >= 2:
                cats[(int(r), int(e))] = _classify_event(sel.to_dict("records"))
        idx = list(zip(ev.runNumber.astype(int), ev.eventNumber.astype(int)))
        for k in ("n_accepted", "n_distinct_tids", "max_per_tid", "n_nonneutron",
                  "n_unusable", "category"):
            ev[f"{variant}_{k}"] = [cats.get(i, {}).get(k) for i in idx]

    cl.to_parquet(clp, index=False)
    ev.to_parquet(evp, index=False)

    variants = [c[:-len("_category")] for c in ev.columns if c.endswith("_category")]
    counts = pd.concat([_cat_counts(ev, v) for v in variants], ignore_index=True)
    out = ctx.csv_path(f"{ctx.run_name}__prov_category_counts")
    counts.to_csv(out, index=False)
    byp = []
    for (lab, port, z), g in ev.groupby(["label", "port", "z_cm"]):
        for v in variants:
            c = _cat_counts(g, v)
            c["label"], c["port"], c["z_cm"] = lab, port, z
            byp.append(c)
    if byp:
        pd.concat(byp, ignore_index=True).to_csv(
            ctx.csv_path(f"{ctx.run_name}__prov_by_position"), index=False)
    set_style()
    _p_category_by_selection(counts, ctx)
    print(f"[provenance] rewrote {out} with variants {variants}")
    return out


def run_provenance(ctx: RunContext, args) -> Path:
    set_style()
    prov = dict(PROV_DEFAULTS)
    prov.update({k: v for k, v in (ctx.extra.get("provenance") or {}).items()
                 if k in PROV_DEFAULTS})
    if args.control_events is not None:
        prov["control_events_per_file"] = args.control_events

    cfgp = ctx.extra.get("provenance") or {}
    label_map = cfgp.get("label_map")
    if not label_map:
        raise SystemExit("config needs provenance.label_map (the MC label_map.txt)")
    label_index = _load_label_index(label_map)

    files = _resolve_inputs(ctx.inputs["root_files"])
    files, _bad = _filter_tree(files, "Event")
    if args.max_files:
        files = files[:args.max_files]
        print(f"[provenance] *** PARTIAL SAMPLE: {len(files)} files. Not quotable. ***")
    print(f"[provenance] {len(files)} files, mode={args.mode}, "
          f"thresholds {prov}")

    modes = ["exact", "window"] if args.mode == "both" else [args.mode]
    out = {}
    for m in modes:
        out[m] = _run_one_mode(ctx, args, prov, m, files, label_index)

    primary = "exact" if "exact" in out else args.mode
    cl, ev, pairs, qa = out[primary]
    if cl.empty:
        raise SystemExit("[provenance] no clusters extracted")

    # Outputs are suffixed by membership mode. The window pass is a cross-check, not a
    # second opinion to be silently overwritten by the exact pass -- when the two
    # disagree (gate V1) the answer has to be quoted as a RANGE, and that is impossible
    # if only one mode's tables survived.
    sfx = "" if primary == "exact" else f"_{primary}"
    global _PROV_SFX
    _PROV_SFX = sfx
    base = f"{ctx.run_name}{sfx}"
    cl.to_parquet(ctx.parquet_path(f"{base}__provenance_clusters"), index=False)
    ev.to_parquet(ctx.parquet_path(f"{base}__provenance_events"), index=False)
    if len(pairs):
        pairs.to_parquet(ctx.parquet_path(f"{base}__provenance_pairs"), index=False)
    qa.to_csv(ctx.csv_path(f"{base}__prov_qa"), index=False)

    variants = [c[:-len("_category")] for c in ev.columns if c.endswith("_category")]
    counts = pd.concat([_cat_counts(ev, v) for v in variants], ignore_index=True)
    counts.to_csv(ctx.csv_path(f"{base}__prov_category_counts"), index=False)

    # Per position
    byp = []
    for (lab, port, z), g in ev.groupby(["label", "port", "z_cm"]):
        for v in variants:
            c = _cat_counts(g, v)
            c["label"], c["port"], c["z_cm"] = lab, port, z
            byp.append(c)
    if byp:
        pd.concat(byp, ignore_index=True).to_csv(
            ctx.csv_path(f"{base}__prov_by_position"), index=False)

    # CC1 capture band from the single-cluster control sample
    ctrl = cl[(cl.is_control) & (cl.prov_label == "neutron")].dt_cap_ns.dropna()
    if len(ctrl) > 10:
        mu = float(np.median(ctrl))
        iqr = float(np.percentile(ctrl, 75) - np.percentile(ctrl, 25))
    else:
        mu, iqr = np.nan, np.nan
    band = (mu, iqr, mu - 3 * iqr, mu + 3 * iqr)
    cl["in_capture_band"] = cl.dt_cap_ns.between(band[2], band[3])

    # CC4: distinct-neutron events must contain a secondary neutron (class 2/3/4),
    # because there is exactly one primary per event.
    frac_sec, n_contradiction = np.nan, 0
    if "all_category" in ev.columns:
        j = cl.join(ev.set_index(["runNumber", "eventNumber"])["all_category"],
                    on=["runNumber", "eventNumber"])
        dj = j[j.all_category == "DISTINCT_NEUTRONS"]
        if len(dj):
            per = dj.groupby(["runNumber", "eventNumber"]).dom_class.agg(
                lambda s: set(int(x) for x in s))
            frac_sec = 100 * float(np.mean([bool(s & {2, 3, 4}) for s in per]))
            n_contradiction = int(sum(1 for s in per if s == {1}))
        cls_tab = (j.groupby(["all_category", "dom_class"]).size()
                   .rename("n").reset_index())
        cls_tab.to_csv(ctx.csv_path(f"{base}__prov_class_breakdown"), index=False)

    # Threshold scan (V8)
    scan = []
    for mf in (0.40, 0.50, 0.60, 0.70, 0.80):
        for mh in (2, 3, 5):
            lab = np.where((cl.n_dom >= mh) & (cl.frac_dom_traced >= mf),
                           "neutron", "other")
            tmp = cl.assign(_l=lab)
            tmp = tmp[tmp._l == "neutron"]
            g = tmp.groupby(["runNumber", "eventNumber"]).dom_tid.agg(
                lambda s: (len(set(s)), s.value_counts().max() if len(s) else 0))
            multi = ev.set_index(["runNumber", "eventNumber"])
            multi = multi[multi.n_clusters >= 2].index
            gg = [g.get(k, (0, 0)) for k in multi]
            n = max(len(gg), 1)
            scan.append(dict(min_frac=mf, min_hits=mh,
                             split_frac=sum(1 for d, mx in gg if mx >= 2) / n,
                             distinct_frac=sum(1 for d, mx in gg if d >= 2 and mx == 1) / n,
                             n_events=n))
    pd.DataFrame(scan).to_csv(ctx.csv_path(f"{base}__prov_threshold_scan"),
                              index=False)

    # Plots
    if not args.no_plots:
        _p_category_bar(counts, ctx, "all")
        _p_category_by_selection(counts, ctx)
        _p_pair_dt(pairs, ctx)
        _p_pair_dd(pairs, ctx)
        _p_domfrac(cl, ctx, prov)
        _p_untraced(cl, ctx)
        _p_dtcap(cl, ev, ctx, band)
        _p_class_stack(cl, ev, ctx, frac_sec)
        _p_position_heatmap(ev, ctx)
        _p_split_pair_size(pairs, ctx)
        _p_split_survival(cl, ev, ctx)
        _p_multiparent(cl, ev, ctx)
        _p_ncap(ev, ctx)

    agree = np.nan
    if "window" in out and "exact" in out and not args.no_plots:
        agree = _p_mode_confusion(out["exact"][0], out["window"][0], ctx)

    # ---- verification gates ----
    untr_all = float((1 - cl.n_traced / cl.n_member_hits.clip(lower=1)).median())
    same = pairs[(pairs.variant == "all") & pairs.same_tid] if len(pairs) else pairs
    frac_same_far = (float((same.dt_ns > 1000).mean()) if len(same) else 0.0)
    sc = pd.DataFrame(scan)
    swing = abs(float(sc[(sc.min_frac == 0.5) & (sc.min_hits == 3)].split_frac.iloc[0]
                      - sc[(sc.min_frac == 0.7) & (sc.min_hits == 3)].split_frac.iloc[0]))
    gates = [
        ("V1 exact-vs-window label agreement >= 90%",
         (np.isnan(agree) or agree >= 90.0), f"{agree:.1f}%" if not np.isnan(agree) else "not run"),
        ("V2 median untraced fraction <= 0.15", untr_all <= 0.15, f"{untr_all:.3f}"),
        ("V3 no clusterHits mismatch / unmapped / collisions",
         int(qa.n_clusterhits_mismatch.sum()) == 0 and int(qa.n_unmapped_cluster_hits.sum()) == 0,
         f"{int(qa.n_clusterhits_mismatch.sum())} / {int(qa.n_unmapped_cluster_hits.sum())} / "
         f"{int(qa.n_key_collisions.sum())}"),
        ("V4 control dt_cap IQR <= 50 ns and offset fallback <= 1%",
         (iqr <= 50.0 if np.isfinite(iqr) else False)
         and qa.n_offset_fallback.sum() <= 0.01 * qa.n_events.sum(),
         f"IQR {iqr:.0f} ns, fallback {qa.n_offset_fallback.sum()}"),
        ("V5 same-neutron pairs with dt > 1 us <= 5%",
         frac_same_far <= 0.05, f"{100*frac_same_far:.1f}%"),
        ("V6 no DISTINCT event with two class-1 clusters",
         n_contradiction == 0, f"{n_contradiction} events"),
        ("V8 split fraction stable across min_frac 0.5 -> 0.7 (<= 5 pts)",
         swing <= 0.05, f"{100*swing:.1f} pts"),
    ]
    print("\n[provenance] VERIFICATION GATES")
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}  [{detail}]")

    head = counts[counts.variant == "all"].set_index("category").frac.to_dict()
    n_multi = int(counts[counts.variant == "all"].n_total.iloc[0]) if len(counts) else 0
    print(f"\n[provenance] Of {n_multi:,} events with >=2 ClusterFinder clusters: "
          f"{100*head.get('SPLIT',0):.1f}% one neutron split, "
          f"{100*head.get('DISTINCT_NEUTRONS',0):.1f}% distinct neutrons "
          f"({frac_sec:.1f}% of those carry a secondary neutron), "
          f"{100*head.get('NEUTRON_PLUS_BACKGROUND',0):.1f}% neutron+background, "
          f"{100*head.get('AMBIGUOUS',0):.1f}% unlabelled.")

    pd.DataFrame([dict(variant=v, **{c: float(counts[(counts.variant == v)
                 & (counts.category == c)].frac.iloc[0]) for c in PROV_CATEGORIES},
                 n_events=int(counts[counts.variant == v].n_total.iloc[0]))
                  for v in variants]).to_csv(
        ctx.csv_path(f"{base}__prov_headline"), index=False)
    return ctx.csv_path(f"{base}__prov_headline")


# --------------------------------------------------------------------------- #
# Stage 0 -- split census straight from the persisted cluster-feature table
# --------------------------------------------------------------------------- #

def stage0_from_features(ctx: RunContext, features_parquet: Optional[str] = None) -> Path:
    """Split-neutron census from `<run>__cluster_features.parquet` alone.

    `dominant_trackID` is already persisted per (event, method, cluster), so a
    neutron claimed by more than one cluster of the same method is a SPLIT and needs
    no ROOT pass.  This is the cheap gate in front of `--mode exact`.

    Two limitations are structural, not fixable here, and are asserted rather than
    worked around:

      * `_truth_label_cluster` returns `(0, -1)` on every failure path, so a cluster
        that is NOT tagged signal carries no neutron identity at all -- even when it
        is neutron-dominated.  Stage 0 is therefore blind to splitting among
        background-tagged clusters; only `--mode exact` can see it.
      * A cluster holding two neutrons' light is awarded to the larger contributor
        and so reads as one neutron here.  MERGED is invisible to Stage 0.

    Because the truth-neutron set is built from `_apply_truth_window`, which keeps
    each trackID's own median hit by construction, every neutron trackID with a hit
    is always in that set.  The `dom_tid in truth_tids` half of the signal label can
    therefore never fail, and `dom_frac < MIN_MATCH_FRAC` is the *only* reason a
    cluster with neutron light is tagged background.  That equivalence is checked
    below and reported, since it is what makes the background census interpretable.
    """
    set_style()
    if features_parquet is None:
        features_parquet = ctx.parquet_path(f"{ctx.run_name}__cluster_features")
    cols = ["eventID", "method", "cluster_id", "is_truth_neutron", "dominant_trackID",
            "is_prompt_cluster", "dominant_class", "frac_neutron", "frac_nonneutron",
            "frac_untraced", "n_hits", "n_neutron", "t_mean"]
    df = pd.read_parquet(features_parquet, columns=cols)

    # The signal tag and "has a neutron identity" must be the same partition; if they
    # ever diverge the discard-on-failure behaviour above has changed and every
    # background number below would silently mean something else.
    bad = int(((df.dominant_trackID >= 0) != (df.is_truth_neutron == 1)).sum())
    if bad:
        raise SystemExit(f"[stage0] {bad} clusters have dominant_trackID and "
                         "is_truth_neutron disagreeing -- labelling changed, stop.")

    df["is_bkg_tag"] = ((df.is_truth_neutron == 0) & (df.is_prompt_cluster == 0)).astype(int)

    cl_rows, ev_rows = [], []
    for meth, g in df.groupby("method"):
        nd = g[g.dominant_trackID >= 0]
        per_neutron = nd.groupby(["eventID", "dominant_trackID"]).size()
        n_ev = int(g.eventID.nunique())

        ncl = g.groupby("eventID").size()
        multi = set(ncl[ncl >= 2].index)
        gm = nd[nd.eventID.isin(multi)]
        ev = gm.groupby("eventID")["dominant_trackID"].agg(
            n_distinct="nunique", max_per=lambda s: s.value_counts().max())
        n_multi = len(multi)
        n_split = int(((ev.max_per >= 2) & (ev.n_distinct < 2)).sum())
        n_both = int(((ev.max_per >= 2) & (ev.n_distinct >= 2)).sum())
        n_dist = int(((ev.max_per < 2) & (ev.n_distinct >= 2)).sum())

        bk = g[g.is_bkg_tag == 1]
        cl_rows.append(dict(
            method=meth, n_clusters=len(g), n_events=n_ev,
            n_signal=int((g.is_truth_neutron == 1).sum()),
            n_bkg_tag=len(bk), n_prompt_excluded=int(((g.is_truth_neutron == 0) &
                                                      (g.is_prompt_cluster == 1)).sum()),
            n_neutrons=len(per_neutron),
            n_neutrons_split=int((per_neutron >= 2).sum()),
            frac_neutrons_split=float((per_neutron >= 2).mean()),
            n_clusters_on_split=int(per_neutron[per_neutron >= 2].sum()),
            frac_sig_clusters_on_split=float(per_neutron[per_neutron >= 2].sum() / max(len(nd), 1)),
            # the background census Stage 0 *can* do: how much of the MVA's background
            # class is actually neutron light that lost the dom_frac vote
            bkg_with_neutron_hit=float((bk.n_neutron >= 1).mean()) if len(bk) else np.nan,
            bkg_frac_neutron_majority=float((bk.frac_neutron >= 0.5).mean()) if len(bk) else np.nan,
            bkg_dominant_class_is_neutron=float(bk.dominant_class.isin([1, 2, 3, 4]).mean()) if len(bk) else np.nan,
            bkg_t_mean_median_ns=float(bk.t_mean.median()) if len(bk) else np.nan,
            sig_t_mean_median_ns=float(g.loc[g.is_truth_neutron == 1, "t_mean"].median()),
        ))
        ev_rows.append(dict(
            method=meth, n_multicluster_events=n_multi,
            frac_events_multicluster=n_multi / max(n_ev, 1),
            SPLIT=n_split / max(n_multi, 1),
            DISTINCT_NEUTRONS=n_dist / max(n_multi, 1),
            SPLIT_AND_DISTINCT=n_both / max(n_multi, 1),
            NO_SIGNAL_CLUSTER=(n_multi - len(ev)) / max(n_multi, 1),
            n_SPLIT=n_split, n_DISTINCT=n_dist, n_SPLIT_AND_DISTINCT=n_both,
        ))

    cl = pd.DataFrame(cl_rows)
    ev = pd.DataFrame(ev_rows)
    cl.to_csv(ctx.csv_path(f"{ctx.run_name}__stage0_split_census"), index=False)
    ev.to_csv(ctx.csv_path(f"{ctx.run_name}__stage0_event_categories"), index=False)
    print(cl.to_string(index=False))
    print()
    print(ev.to_string(index=False))
    return ctx.csv_path(f"{ctx.run_name}__stage0_split_census")


# --------------------------------------------------------------------------- #
# Stage 1 -- per-neutron breakdown under the SAME membership that made the labels
# --------------------------------------------------------------------------- #

def _neutron_breakdown(mask, df_ev, truth_mask):
    """Full per-trackID decomposition of one cluster, on the production membership.

    `_truth_label_cluster` computes the dominant neutron trackID and then throws it
    away (`return 0, -1`) whenever the cluster fails the label, so the persisted table
    cannot say why a background-tagged cluster is background.  This keeps everything
    that call discards: every contributing neutron, the runner-up, and which of the
    two label conditions actually failed.

    Note `dom_frac`'s denominator is ALL member hits, not traced ones -- that is the
    production definition and the point is to reproduce it, not to improve it.
    """
    tids = df_ev["ancestor_trackID"].to_numpy(int)
    pdgs = df_ev["ancestor_pdg"].to_numpy(int)
    cls = df_ev["truth_class"].to_numpy(int)
    unt = (df_ev["is_untraced"].to_numpy(int) if "is_untraced" in df_ev.columns
           else np.zeros(len(df_ev), int))

    truth_tids = set(int(t) for t, p, w in zip(tids, pdgs, truth_mask)
                     if p == NEUTRON_PDG and t >= 0 and w)

    n_total = int(mask.sum())
    nm = mask & (pdgs == NEUTRON_PDG) & (tids >= 0)
    n_neutron = int(nm.sum())
    n_untraced = int((mask & (unt == 1)).sum())
    n_dn = int((mask & (cls == 0)).sum())
    n_bg = int((mask & (cls == -5)).sum())

    out = dict(n_hits=n_total, n_neutron=n_neutron, n_untraced=n_untraced,
               n_darknoise=n_dn, n_nonneutron=n_bg,
               n_contrib_neutrons=0, dom_tid=-1, n_dom=0, dom_frac=0.0,
               tid2=-1, n_tid2=0, frac_tid2=0.0,
               dom_in_truth=False, fail_reason="no_neutron_hits",
               neutron_share_of_hits=n_neutron / max(n_total, 1))
    if n_neutron == 0:
        return out

    u, c = np.unique(tids[nm], return_counts=True)
    order = np.argsort(-c)
    u, c = u[order], c[order]
    dom_tid, n_dom = int(u[0]), int(c[0])
    dom_frac = n_dom / max(n_total, 1)
    out.update(n_contrib_neutrons=len(u), dom_tid=dom_tid, n_dom=n_dom,
               dom_frac=dom_frac,
               tid2=int(u[1]) if len(u) > 1 else -1,
               n_tid2=int(c[1]) if len(c) > 1 else 0,
               frac_tid2=(int(c[1]) / max(n_total, 1)) if len(c) > 1 else 0.0,
               dom_in_truth=dom_tid in truth_tids)

    # Which half of `dom_tid in truth_tids and dom_frac >= MIN_MATCH_FRAC` failed.
    # _apply_truth_window always keeps each trackID's own median hit, so every neutron
    # trackID with a hit is in truth_tids and `truth_window` should never appear. It is
    # still emitted rather than assumed, so the claim stays falsifiable.
    if dom_tid in truth_tids and dom_frac >= _MIN_MATCH_FRAC:
        reason = "passes"
    elif dom_tid not in truth_tids:
        reason = "truth_window"
    elif n_neutron / max(n_total, 1) < _MIN_MATCH_FRAC:
        # not enough neutron light in the cluster at all -- diluted by background/noise
        reason = "diluted_by_nonneutron"
    else:
        # neutron-majority cluster where no SINGLE neutron owns half the hits:
        # two or more neutrons merged into one cluster
        reason = "merged_neutrons"
    out["fail_reason"] = reason
    return out


def run_stage1(ctx: RunContext, args) -> Path:
    """Per-cluster neutron provenance on CCinc, reproducing the production membership.

    This deliberately does NOT use the ROOT exact-membership path that the AmBe
    provenance study uses.  The question here is why the *persisted* labels are what
    they are, and those were produced by `assign_clusterfinder_labels` / OPTICS running
    on `__pulses.parquet` (`cluster_features.extract_all_features`).  Re-deriving
    membership from `Cluster_HitChankey` would answer a different question and would
    not reproduce the labels under test.
    """
    import pyarrow.parquet as pq
    from . import cc_selection
    from .cluster_features import _truth_label_cluster
    from .optics import (run_optics_on_event, _apply_truth_window, _prefilter_hits,
                         assign_clusterfinder_labels)

    set_style()
    feat = ctx.extra.get("features", {}) or {}
    ms = int(feat.get("min_samples", 8))
    xi = float(feat.get("xi", 0.02))
    t_unit = float(feat.get("t_unit_ns", 25.0))
    tw = float(feat.get("truth_window_ns", 75.0))
    prefilter = float(feat.get("hit_prefilter_ns", 0.0))

    pulses_path = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    pf = pq.ParquetFile(str(pulses_path))
    cols = ["eventID", "pmtID", "t", "x", "y", "z", "pe", "truth_class", "is_neutron",
            "is_untraced", "ancestor_trackID", "ancestor_pdg", "cc_pass",
            "origin_in_tank"]
    avail = set(pf.schema_arrow.names)
    read_cols = [c for c in cols if c in avail]
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    methods = ["clusterfinder", "optics"] if args.method == "both" else [args.method]
    n_groups = pf.num_row_groups
    groups = range(n_groups if not args.max_files else min(args.max_files, n_groups))
    print(f"[stage1] {pf.metadata.num_rows:,} hits in {n_groups} row groups; "
          f"processing {len(list(groups))}. methods={methods} "
          f"OPTICS ms={ms} xi={xi} t_unit={t_unit} truth_window={tw}")
    if args.max_files and args.max_files < n_groups:
        print(f"[stage1] *** PARTIAL SAMPLE: {args.max_files}/{n_groups} groups. "
              "Not quotable. ***")

    rows = []
    for i_grp in groups:
        chunk = pf.read_row_group(i_grp, columns=read_cols).to_pandas()
        if "pe" not in chunk.columns:
            chunk["pe"] = 1.0
        chunk = cc_selection.apply_residual_filter(chunk, ctx, verbose=False)
        if not len(chunk):
            continue
        for evid in chunk["eventID"].unique():
            df_ev = chunk[chunk["eventID"] == evid].reset_index(drop=True)
            df_cl = (clusters[clusters["eventID"] == evid].reset_index(drop=True)
                     if clusters is not None else None)
            truth_mask = _apply_truth_window(df_ev, tw)
            ev_origin = (int(df_ev["origin_in_tank"].iloc[0])
                         if "origin_in_tank" in df_ev.columns else -1)

            lbl = {}
            if "optics" in methods:
                df_opt = _prefilter_hits(df_ev, df_cl, prefilter) if prefilter > 0 else df_ev
                lo = run_optics_on_event(df_opt, min_samples=ms, xi=xi, t_unit_ns=t_unit)
                if len(df_opt) < len(df_ev):
                    full = np.full(len(df_ev), -1, int)
                    full[df_opt.index] = lo
                    lo = full
                lbl["optics"] = lo
            if "clusterfinder" in methods:
                lbl["clusterfinder"] = (assign_clusterfinder_labels(df_ev, df_cl)
                                        if df_cl is not None
                                        else np.full(len(df_ev), -1, int))

            for method, lbls in lbl.items():
                for cid in [c for c in np.unique(lbls) if c >= 0]:
                    mask = (lbls == cid)
                    bd = _neutron_breakdown(mask, df_ev, truth_mask)
                    is_n, dom_tid = _truth_label_cluster(mask, df_ev, truth_mask)
                    rows.append(dict(eventID=int(evid), method=method,
                                     cluster_id=int(cid),
                                     is_truth_neutron=int(is_n),
                                     persisted_dominant_trackID=int(dom_tid),
                                     origin_in_tank=ev_origin,
                                     t_mean=float(df_ev.loc[mask, "t"].mean()),
                                     **bd))
    cl = pd.DataFrame(rows)
    if cl.empty:
        raise SystemExit("[stage1] no clusters extracted")
    cl.to_parquet(ctx.parquet_path(f"{ctx.run_name}__stage1_neutron_provenance"),
                  index=False)

    # Gate: the reproduced label must agree with _truth_label_cluster on this same
    # membership by construction; a mismatch means the breakdown is reading a
    # different cluster than the label did.
    agree = ((cl.dom_in_truth & (cl.dom_frac >= _MIN_MATCH_FRAC)).astype(int)
             == cl.is_truth_neutron).mean()
    print(f"[stage1] label reproduction: {agree:.4%} "
          f"({'PASS' if agree > 0.999 else 'FAIL'})")

    summ = []
    for (meth, sig), g in cl.groupby(["method", "is_truth_neutron"]):
        summ.append(dict(method=meth, tag="signal" if sig else "background",
                         n_clusters=len(g),
                         mean_n_contrib_neutrons=float(g.n_contrib_neutrons.mean()),
                         frac_single_neutron=float((g.n_contrib_neutrons == 1).mean()),
                         frac_multi_neutron=float((g.n_contrib_neutrons >= 2).mean()),
                         median_dom_frac=float(g.dom_frac.median()),
                         median_neutron_share=float(g.neutron_share_of_hits.median()),
                         **{f"fail_{k}": float(v) for k, v in
                            g.fail_reason.value_counts(normalize=True).items()}))
    s = pd.DataFrame(summ).fillna(0.0)
    s.to_csv(ctx.csv_path(f"{ctx.run_name}__stage1_summary"), index=False)
    print(s.to_string(index=False))
    return ctx.csv_path(f"{ctx.run_name}__stage1_summary")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = _argparse.ArgumentParser(prog="ambe mc match")
    # No default. `legacy` is the pre-existing parquet-driven matching and reproduces
    # its output unchanged; `stage0` is the cheap feature-table census; `exact` is the
    # provenance answer; `window` is its membership cross-check; `both` runs exact then
    # window and compares them.
    p.add_argument("--mode", required=True,
                   choices=["legacy", "window", "exact", "both", "attach-mva",
                            "stage0", "stage1"])
    p.add_argument("--features-parquet", default=None,
                   help="stage0 only: override the cluster_features parquet path")
    # No default: which clustering is being characterised has to be stated, because
    # the two give materially different split rates and a silent default would let a
    # number be quoted against the wrong method.
    p.add_argument("--method", choices=["clusterfinder", "optics", "both"],
                   default=None, help="stage1 only: which clustering to characterise")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--control-events", type=int, default=None)
    p.add_argument("--scored-parquet", default=None)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(list(argv) if argv else [])

    # Deliberately no return value: ambe.cli.main() hands whatever the command returns
    # to sys.exit, so returning a Path would exit 1 on success and break shell chaining.
    if a.mode == "legacy":
        run_legacy(ctx)
    elif a.mode == "stage0":
        stage0_from_features(ctx, a.features_parquet)
    elif a.mode == "stage1":
        if a.method is None:
            raise SystemExit("--mode stage1 needs --method {clusterfinder,optics,both}")
        run_stage1(ctx, a)
    elif a.mode == "attach-mva":
        # Post-process an existing exact-mode extraction. The per-cluster provenance
        # label is a statement about the light's origin and does not depend on the
        # selection, so the MVA variants are a re-partition of labels already on disk.
        m = ctx.extra.get("provenance", {}).get("mva")
        if not m:
            raise SystemExit("--mode attach-mva needs a provenance.mva block with "
                             "`scored` (variant -> parquet) and `threshold`")
        attach_mva_variants(ctx, dict(m["scored"]), float(m["threshold"]),
                            m.get("score_col", "gbt_score"))
    else:
        run_provenance(ctx, a)


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    """Kept so `from .cluster_matching import run` still resolves; mode is required."""
    return cli(ctx, argv)
