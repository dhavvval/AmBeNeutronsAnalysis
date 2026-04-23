"""
MC processor: ROOT -> per-hit Parquet + ClusterFinder sidecar.

Every detector hit (hitChankey / hitT) is included, not just those that
BackTracker could trace.  For each hit we attempt a truth match into the
DirectParent arrays using:
  - chankey identity
  - time proximity: DirectParent_HitTime ≈ hitT + offset
    where offset (~14-16 ns) is the per-event Cherenkov photon travel time,
    computed from chankeys that appear in both arrays.

Unmatched hits are assigned truth_class=0 (dark noise) and flagged with
is_untraced=True so they can be distinguished from BackTracker-labeled noise.

This matters for OPTICS training: the ~11% of hits that are genuine PMT dark
noise but BackTracker-invisible were previously absent from the training set.

Class code conventions (from BackTracker.cpp):
   0 dark noise  ·  1 primary neutron  ·  2 secondary n<-p  ·
   3 secondary n<-n  ·  4 secondary n<-other  ·  -5 non-neutron physics bg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs


# Branches needed from the full-hit list
HIT_BRANCHES = [
    "eventNumber",
    "hitChankey", "hitT", "hitX", "hitY", "hitZ",
]

# Branches needed for truth lookup (all optional — graceful fallback if absent)
DP_BRANCHES = [
    "DirectParent_PMTID", "DirectParent_HitTime",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
    "DirectParent_NeutronAncestorTrackID", "DirectParent_NeutronAncestorPDG",
]

CLUSTER_BRANCHES = [
    "eventNumber",
    "clusterTime", "clusterPE", "clusterHits",
    "Cluster_HitT", "Cluster_HitX", "Cluster_HitY", "Cluster_HitZ",
]

NEUTRON_CLASSES = {1, 2, 3, 4}
DEFAULT_DP_OFFSET = 15.0   # ns fallback if offset can't be computed
MATCH_TOL        = 8.0     # ns half-window for truth matching


# --------------------------------------------------------------------------- #
# Truth-matching helpers
# --------------------------------------------------------------------------- #

def _compute_offset(hit_ck, hit_t, dp_ck, dp_t) -> float:
    """
    Estimate DirectParent_HitTime - hitT per event.
    Uses chankeys that appear exactly once in each array for an unambiguous match.
    Falls back to DEFAULT_DP_OFFSET when insufficient overlap.
    """
    shared = set(np.unique(hit_ck)) & set(np.unique(dp_ck))
    offsets = []
    for ck in shared:
        ht = hit_t[hit_ck == ck]
        dt = dp_t[dp_ck == ck]
        if len(ht) == 1 and len(dt) == 1:
            offsets.append(float(dt[0] - ht[0]))
    return float(np.median(offsets)) if len(offsets) >= 3 else DEFAULT_DP_OFFSET


def _build_dp_lookup(dp_ck, dp_t, dp_class, dp_dn, dp_tid, dp_pdg) -> dict:
    """
    Index DirectParent entries by chankey for O(1) per-hit lookup.
    Returns dict: chankey -> list of (dp_time, class, is_darknoise, trackID, pdg)
    """
    lookup: dict[int, list] = {}
    for ck, t, cls, dn, tid, pdg in zip(dp_ck, dp_t, dp_class, dp_dn, dp_tid, dp_pdg):
        ck_i = int(ck)
        lookup.setdefault(ck_i, []).append(
            (float(t), int(cls), int(dn), int(tid), int(pdg))
        )
    return lookup


def _match_hit(ck: int, t: float, dp_lookup: dict, offset: float,
               tol: float = MATCH_TOL):
    """
    Match one hit (chankey, hitT) to its DirectParent truth entry.

    Returns (truth_class, is_darknoise, ancestor_trackID, ancestor_pdg, is_untraced).
    Untraced hits (no DP entry or time mismatch) get truth_class=0, is_untraced=True.
    """
    entries = dp_lookup.get(ck)
    if not entries:
        return 0, 1, -1, -1, True

    t_shifted = t + offset
    best_dt, best = min(
        ((abs(e[0] - t_shifted), e) for e in entries),
        key=lambda x: x[0],
    )
    if best_dt > tol:
        return 0, 1, -1, -1, True

    _, cls, dn, tid, pdg = best
    return cls, dn, tid, pdg, False


# --------------------------------------------------------------------------- #
# Per-file processing
# --------------------------------------------------------------------------- #

def _process_single_file(root_path: Path, tree_name: str, verbose: bool):
    """Read one ROOT file; return (pulse_df, cluster_df)."""
    with uproot.open(str(root_path)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree {tree_name!r} not in {root_path}")
        t = f[tree_name]
        available = set(t.keys())

        wanted_hit     = [b for b in HIT_BRANCHES  if b in available]
        wanted_dp      = [b for b in DP_BRANCHES   if b in available]
        wanted_cluster = [b for b in CLUSTER_BRANCHES if b in available]

        missing_hit = set(HIT_BRANCHES) - available
        if missing_hit and verbose:
            print(f"[mc.processor] WARN {root_path.name}: missing hit branches: {missing_hit}")

        has_dp = len(wanted_dp) > 0
        if not has_dp and verbose:
            print(f"[mc.processor] WARN {root_path.name}: no DirectParent branches — "
                  "all hits will be labeled as untraced dark noise")

        hit_arr     = t.arrays(wanted_hit,     library="ak")
        dp_arr      = t.arrays(wanted_dp,      library="ak") if has_dp else None
        cluster_arr = t.arrays(wanted_cluster, library="ak") if wanted_cluster else None

    n = len(hit_arr)
    pulse_rows, cluster_rows = [], []

    for i in range(n):
        event_id = int(hit_arr["eventNumber"][i])

        hit_ck = np.array(ak.to_list(hit_arr["hitChankey"][i]), dtype=int)
        hit_t  = np.array(ak.to_list(hit_arr["hitT"][i]),       dtype=float)
        hit_x  = np.array(ak.to_list(hit_arr["hitX"][i]),       dtype=float)
        hit_y  = np.array(ak.to_list(hit_arr["hitY"][i]),       dtype=float)
        hit_z  = np.array(ak.to_list(hit_arr["hitZ"][i]),       dtype=float)

        # Build truth lookup for this event
        if has_dp and "DirectParent_PMTID" in wanted_dp:
            dp_ck  = np.array(ak.to_list(dp_arr["DirectParent_PMTID"][i]),               dtype=int)
            dp_t   = np.array(ak.to_list(dp_arr["DirectParent_HitTime"][i]),             dtype=float)
            dp_cls = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorClass"][i]),dtype=int) \
                     if "DirectParent_NeutronAncestorClass" in wanted_dp \
                     else np.zeros(len(dp_ck), dtype=int)
            dp_dn  = np.array(ak.to_list(dp_arr["DirectParent_IsDarknoise"][i]),         dtype=int) \
                     if "DirectParent_IsDarknoise" in wanted_dp \
                     else np.ones(len(dp_ck), dtype=int)
            dp_tid = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorTrackID"][i]), dtype=int) \
                     if "DirectParent_NeutronAncestorTrackID" in wanted_dp \
                     else np.full(len(dp_ck), -1, dtype=int)
            dp_pdg = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorPDG"][i]),  dtype=int) \
                     if "DirectParent_NeutronAncestorPDG" in wanted_dp \
                     else np.full(len(dp_ck), -1, dtype=int)

            offset    = _compute_offset(hit_ck, hit_t, dp_ck, dp_t)
            dp_lookup = _build_dp_lookup(dp_ck, dp_t, dp_cls, dp_dn, dp_tid, dp_pdg)
        else:
            offset    = 0.0
            dp_lookup = {}

        # Emit one row per detector hit
        for j in range(len(hit_ck)):
            ck = int(hit_ck[j])
            t_hit = float(hit_t[j])

            cls, dn, tid, pdg, untraced = _match_hit(ck, t_hit, dp_lookup, offset)

            pulse_rows.append({
                "eventID":          event_id,
                "pmtID":            ck,
                "t":                t_hit,
                "x":                float(hit_x[j]),
                "y":                float(hit_y[j]),
                "z":                float(hit_z[j]),
                "truth_class":      cls,
                "is_darknoise":     dn,
                "is_neutron":       int(cls in NEUTRON_CLASSES),
                "is_untraced":      int(untraced),
                "ancestor_trackID": tid,
                "ancestor_pdg":     pdg,
            })

        # ClusterFinder sidecar
        if cluster_arr is not None:
            ctimes = np.asarray(cluster_arr["clusterTime"][i]) if "clusterTime" in wanted_cluster else []
            cpes   = np.asarray(cluster_arr["clusterPE"][i])   if "clusterPE"   in wanted_cluster else [0.0] * len(ctimes)
            chits  = np.asarray(cluster_arr["clusterHits"][i]) if "clusterHits" in wanted_cluster else [0]   * len(ctimes)
            for cidx, (ct, cpe, ch) in enumerate(zip(ctimes, cpes, chits)):
                cluster_rows.append({
                    "eventID":     event_id,
                    "cluster_idx": cidx,
                    "clusterTime": float(ct),
                    "clusterPE":   float(cpe),
                    "clusterHits": int(ch),
                })

    return pd.DataFrame(pulse_rows), pd.DataFrame(cluster_rows)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run(ctx: RunContext, tree_name: str = "Event", verbose: bool = True) -> tuple[Path, Path]:
    """
    Process all ROOT files in ctx.inputs['root_files'].
    Writes two Parquet files:
        <parquet_dir>/<run_name>__pulses.parquet
        <parquet_dir>/<run_name>__clusterfinder.parquet
    """
    root_files = resolve_inputs(ctx.inputs["root_files"])
    if verbose:
        print(f"[mc.processor] processing {len(root_files)} file(s)")

    pulse_frames, cluster_frames = [], []
    for rp in root_files:
        pf, cf = _process_single_file(rp, tree_name, verbose)
        pf["_source_file"] = rp.name
        cf["_source_file"] = rp.name
        pulse_frames.append(pf)
        cluster_frames.append(cf)

    pulses   = pd.concat(pulse_frames,   ignore_index=True) if pulse_frames   else pd.DataFrame()
    clusters = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()

    pulses_path   = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    pulses.to_parquet(pulses_path,   index=False)
    clusters.to_parquet(clusters_path, index=False)

    if verbose:
        print(f"[mc.processor] wrote {len(pulses):>8d} hits     -> {pulses_path}")
        print(f"[mc.processor] wrote {len(clusters):>8d} clusters -> {clusters_path}")
        if len(pulses):
            counts = pulses["truth_class"].value_counts().sort_index()
            untraced = int(pulses["is_untraced"].sum())
            print("[mc.processor] truth_class histogram (all detector hits):")
            print(counts.to_string())
            print(f"[mc.processor] of which untraced dark noise: "
                  f"{untraced} ({100*untraced/len(pulses):.1f}%)")

    return pulses_path, clusters_path


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc process")
    p.add_argument("--tree", default="Event")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(list(argv) if argv else [])
    run(ctx, tree_name=args.tree, verbose=not args.quiet)
