"""
MC processor: BeamClusterAnalysisMC ROOT -> per-pulse Parquet + ClusterFinder sidecar.

Tree branches used (from ANNIEEventTreeMaker.cpp):
  - per-MCHit:   hitChankey, hitX, hitY, hitZ  (chankey -> position lookup)
  - per-pulse:   DirectParent_PMTID, DirectParent_HitTime,
                 DirectParent_NeutronAncestorClass, DirectParent_IsDarknoise
  - per-cluster: clusterTime, clusterPE, clusterHits (baseline)

Class code conventions (from BackTracker.cpp):
   0 dark noise  ·  1 primary neutron  ·  2 secondary n<-p  ·
   3 secondary n<-n  ·  4 secondary n<-other  ·  -5 non-neutron physics bg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs


PULSE_BRANCHES = [
    "eventNumber",
    "hitChankey", "hitX", "hitY", "hitZ",
    "DirectParent_PMTID", "DirectParent_HitTime",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
]

CLUSTER_BRANCHES = [
    "eventNumber",
    "clusterTime", "clusterPE", "clusterHits",
    "Cluster_HitT", "Cluster_HitX", "Cluster_HitY", "Cluster_HitZ",
]

NEUTRON_CLASSES = {1, 2, 3, 4}


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _per_event_chankey_lookup(chankeys, xs, ys, zs):
    lookup = {}
    for ck, x, y, z in zip(chankeys, xs, ys, zs):
        ck_i = int(ck)
        if ck_i not in lookup:
            lookup[ck_i] = (float(x), float(y), float(z))
    return lookup


def _process_single_file(root_path: Path, tree_name: str, verbose: bool):
    """Read one ROOT file; return (pulse_df, cluster_df)."""
    with uproot.open(str(root_path)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree {tree_name!r} not in {root_path}")
        t = f[tree_name]
        available = set(t.keys())

        wanted_pulse = [b for b in PULSE_BRANCHES if b in available]
        wanted_cluster = [b for b in CLUSTER_BRANCHES if b in available]
        missing = set(PULSE_BRANCHES) - available
        if missing and verbose:
            print(f"[mc.processor] WARN {root_path.name}: missing pulse branches: {missing}")

        pulse_arr = t.arrays(wanted_pulse, library="ak")
        cluster_arr = t.arrays(wanted_cluster, library="ak") if wanted_cluster else None

    n = len(pulse_arr)
    pulse_rows, cluster_rows = [], []
    unmatched = 0

    for i in range(n):
        event_id = int(pulse_arr["eventNumber"][i])
        lookup = _per_event_chankey_lookup(
            pulse_arr["hitChankey"][i],
            pulse_arr["hitX"][i],
            pulse_arr["hitY"][i],
            pulse_arr["hitZ"][i],
        )
        for pmt, tp, cls, dn in zip(
            pulse_arr["DirectParent_PMTID"][i],
            pulse_arr["DirectParent_HitTime"][i],
            pulse_arr["DirectParent_NeutronAncestorClass"][i],
            pulse_arr["DirectParent_IsDarknoise"][i],
        ):
            pmt_i = int(pmt)
            if pmt_i not in lookup:
                unmatched += 1
                continue
            x, y, z = lookup[pmt_i]
            cls_i = int(cls)
            pulse_rows.append({
                "eventID": event_id,
                "pmtID": pmt_i,
                "t": float(tp),
                "x": x, "y": y, "z": z,
                "truth_class": cls_i,
                "is_darknoise": int(dn),
                "is_neutron": int(cls_i in NEUTRON_CLASSES),
            })

        if cluster_arr is not None:
            ctimes = np.asarray(cluster_arr["clusterTime"][i]) if "clusterTime" in wanted_cluster else []
            cpes = np.asarray(cluster_arr["clusterPE"][i]) if "clusterPE" in wanted_cluster else [0.0] * len(ctimes)
            chits = np.asarray(cluster_arr["clusterHits"][i]) if "clusterHits" in wanted_cluster else [0] * len(ctimes)
            for cidx, (ct, cpe, ch) in enumerate(zip(ctimes, cpes, chits)):
                cluster_rows.append({
                    "eventID": event_id,
                    "cluster_idx": cidx,
                    "clusterTime": float(ct),
                    "clusterPE": float(cpe),
                    "clusterHits": int(ch),
                })

    if verbose and unmatched:
        print(f"[mc.processor] {root_path.name}: dropped {unmatched} pulses with no chankey match")

    return pd.DataFrame(pulse_rows), pd.DataFrame(cluster_rows)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run(ctx: RunContext, tree_name: str = "Event", verbose: bool = True) -> tuple[Path, Path]:
    """
    Process all MC ROOT files listed in ctx.inputs['root_files'] and write
    two Parquet files under ctx.parquet_dir:
        <run_name>__pulses.parquet
        <run_name>__clusterfinder.parquet
    Returns (pulses_path, clusters_path).
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

    pulses = pd.concat(pulse_frames, ignore_index=True) if pulse_frames else pd.DataFrame()
    clusters = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()

    pulses_path = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    pulses.to_parquet(pulses_path, index=False)
    clusters.to_parquet(clusters_path, index=False)

    if verbose:
        print(f"[mc.processor] wrote {len(pulses):>8d} pulses  -> {pulses_path}")
        print(f"[mc.processor] wrote {len(clusters):>8d} clusters -> {clusters_path}")
        if len(pulses):
            counts = pulses["truth_class"].value_counts().sort_index()
            print("[mc.processor] truth_class histogram:")
            print(counts.to_string())

    return pulses_path, clusters_path


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc process")
    p.add_argument("--tree", default="Event")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(list(argv) if argv else [])
    run(ctx, tree_name=args.tree, verbose=not args.quiet)
