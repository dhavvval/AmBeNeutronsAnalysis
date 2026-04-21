"""
mc_processor.py
---------------
Convert BeamClusterAnalysisMC ROOT output (ANNIETree_MC.root or BeamCluster_*.root)
into two Parquet files per input file:

  1. <stem>.parquet              -- one row per ADC pulse, with per-pulse truth label
                                    (DirectParent_NeutronAncestorClass) and (x,y,z,t) features
  2. <stem>_clusterfinder.parquet -- one row per ClusterFinder cluster + member hits,
                                    used as the baseline for OPTICS comparison

Class codes in truth_class:
   0  dark noise
   1  primary neutron
   2  secondary neutron from proton
   3  secondary neutron from neutron
   4  secondary neutron from other parent type
  -5  non-neutron physics background
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


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


def _per_event_chankey_lookup(chankeys, xs, ys, zs):
    """Build a {chankey: (x,y,z)} dict for one event from per-MCHit arrays."""
    lookup = {}
    for ck, x, y, z in zip(chankeys, xs, ys, zs):
        ck_i = int(ck)
        if ck_i not in lookup:
            lookup[ck_i] = (float(x), float(y), float(z))
    return lookup


def process_file(root_path, out_parquet, tree_name="Event", verbose=True):
    """Read one BeamCluster ROOT file, write two parquet files."""
    root_path = Path(root_path)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    with uproot.open(str(root_path)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree '{tree_name}' not in {root_path}. Found: {list(f.keys())}")
        t = f[tree_name]
        available = set(t.keys())

        wanted_pulse = [b for b in PULSE_BRANCHES if b in available]
        wanted_cluster = [b for b in CLUSTER_BRANCHES if b in available]
        missing = set(PULSE_BRANCHES) - available
        if missing:
            print(f"[mc_processor] WARNING: missing pulse branches in {root_path.name}: {missing}")

        pulse_arr = t.arrays(wanted_pulse, library="ak")
        cluster_arr = t.arrays(wanted_cluster, library="ak") if wanted_cluster else None

    n_events = len(pulse_arr)
    if verbose:
        print(f"[mc_processor] {root_path.name}: {n_events} events")

    pulse_rows = []
    cluster_rows = []
    unmatched_chankeys = 0

    for i in range(n_events):
        event_id = int(pulse_arr["eventNumber"][i])

        lookup = _per_event_chankey_lookup(
            pulse_arr["hitChankey"][i],
            pulse_arr["hitX"][i],
            pulse_arr["hitY"][i],
            pulse_arr["hitZ"][i],
        )

        pmtids = np.asarray(pulse_arr["DirectParent_PMTID"][i])
        times = np.asarray(pulse_arr["DirectParent_HitTime"][i])
        classes = np.asarray(pulse_arr["DirectParent_NeutronAncestorClass"][i])
        isdn = np.asarray(pulse_arr["DirectParent_IsDarknoise"][i])

        for pmt, t_pulse, cls, dn in zip(pmtids, times, classes, isdn):
            pmt_i = int(pmt)
            if pmt_i not in lookup:
                unmatched_chankeys += 1
                continue
            x, y, z = lookup[pmt_i]
            cls_i = int(cls)
            pulse_rows.append({
                "eventID": event_id,
                "pmtID": pmt_i,
                "t": float(t_pulse),
                "x": x, "y": y, "z": z,
                "truth_class": cls_i,
                "is_darknoise": int(dn),
                "is_neutron": int(cls_i in NEUTRON_CLASSES),
            })

        if cluster_arr is not None:
            ctimes = np.asarray(cluster_arr["clusterTime"][i]) if "clusterTime" in wanted_cluster else []
            cpes = np.asarray(cluster_arr["clusterPE"][i]) if "clusterPE" in wanted_cluster else [0] * len(ctimes)
            chits = np.asarray(cluster_arr["clusterHits"][i]) if "clusterHits" in wanted_cluster else [0] * len(ctimes)
            for cidx, (ct, cpe, ch) in enumerate(zip(ctimes, cpes, chits)):
                cluster_rows.append({
                    "eventID": event_id,
                    "cluster_idx": cidx,
                    "clusterTime": float(ct),
                    "clusterPE": float(cpe),
                    "clusterHits": int(ch),
                })

    if verbose and unmatched_chankeys:
        print(f"[mc_processor] {root_path.name}: {unmatched_chankeys} pulses had no "
              f"matching hitChankey entry and were dropped")

    pulse_df = pd.DataFrame(pulse_rows)
    pulse_df.to_parquet(out_parquet, index=False)

    cluster_out = out_parquet.with_name(out_parquet.stem + "_clusterfinder.parquet")
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_parquet(cluster_out, index=False)

    if verbose:
        print(f"[mc_processor] wrote {len(pulse_df):>8d} pulses  -> {out_parquet}")
        print(f"[mc_processor] wrote {len(cluster_df):>8d} clusters -> {cluster_out}")
        if len(pulse_df):
            counts = pulse_df["truth_class"].value_counts().sort_index()
            print(f"[mc_processor] truth_class histogram:\n{counts.to_string()}")

    return pulse_df, cluster_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="One or more ROOT files to process")
    p.add_argument("--outdir", default="mc_parquet", help="Output directory for parquet files")
    p.add_argument("--tree", default="Event", help="Tree name inside each ROOT file")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    for root_path in args.inputs:
        root_path = Path(root_path)
        out_parquet = outdir / (root_path.stem + ".parquet")
        process_file(root_path, out_parquet, tree_name=args.tree, verbose=not args.quiet)


if __name__ == "__main__":
    main()
