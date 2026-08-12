#!/usr/bin/env python3
"""
selection2_multiplicity.py

Per-position single/multiple neutron statistics for SELECTION 2 (ClusterFinder),
reproducing the user's canonical legacy table EXACTLY.

Definition (matches AmBeNeutronProcessing.ambe_single_cut / ambe_multiple_cut):
  A cluster is a CANDIDATE if  0<PE<=100, 0<CCB<0.45, clusterTime>=2000ns, hits>=5.
  An event is SINGLE   if it has >=1 candidate AND numberOfClusters == 1.
  An event is MULTIPLE if it has >=1 candidate AND numberOfClusters != 1.
  unique_neutron_triggers = single + multiple   (events with >=1 candidate)
  % of multiple neutrons  = multiple / unique_neutron_triggers   (the user's metric)

Input: the reprocessed common-input parquet (must carry 'numberOfClusters').
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Legacy ambe_single_cut thresholds (CutCriteria defaults).
PE_MAX, CCB_MAX, CT_MIN, HITS_MIN = 100.0, 0.45, 2000.0, 5


def port_of(x, y, z):
    if x == 75:  return 4
    if z == -75: return 1
    if z == 102: return 3
    if z == 75:  return 2
    if z == 0:   return 5
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="cf_raw parquet with numberOfClusters")
    p.add_argument("--out", default=None, help="optional output CSV")
    args = p.parse_args()

    d = pd.read_parquet(args.parquet)
    if "numberOfClusters" not in d.columns:
        sys.exit("parquet lacks 'numberOfClusters' — re-run reprocess_ambe_v1.py (updated).")

    # candidate flag per cluster (ambe_single_cut PE/CCB/time/hits part)
    d["is_cand"] = ((d.clusterPE > 0) & (d.clusterPE <= PE_MAX) &
                    (d.clusterChargeBalance > 0) & (d.clusterChargeBalance < CCB_MAX) &
                    (d.clusterTime >= CT_MIN) & (d.clusterHits >= HITS_MIN))
    d["pos"] = list(zip(d.sourceX, d.sourceY, d.sourceZ))

    rows = []
    # per event: does it have >=1 candidate? what is numberOfClusters?
    for pos, sub in d.groupby("pos"):
        ev = sub.groupby("eventTankTime").agg(
            has_cand=("is_cand", "any"),
            ncl=("numberOfClusters", "first"),
        )
        ev = ev[ev["has_cand"]]                       # events with >=1 candidate
        n_single = int((ev["ncl"] == 1).sum())
        n_multi  = int((ev["ncl"] != 1).sum())
        uniq = n_single + n_multi
        rows.append({
            "port": port_of(*pos), "x": pos[0], "y": pos[1], "z": pos[2],
            "single_neutron_candidates": n_single,
            "multiple_neutron_candidates": n_multi,
            "unique_neutron_triggers": uniq,
            "pct_multiple": 100 * n_multi / uniq if uniq else np.nan,
            "single_frac": n_single / uniq if uniq else np.nan,
        })
    tab = pd.DataFrame(rows).sort_values(["port", "z", "y"])

    tot_s = tab.single_neutron_candidates.sum()
    tot_m = tab.multiple_neutron_candidates.sum()
    tot_u = tab.unique_neutron_triggers.sum()
    print(f"{'port':>4}{'x':>5}{'y':>6}{'z':>6}{'single':>9}{'multi':>8}{'uniq':>8}{'%mult':>8}")
    for _, r in tab.iterrows():
        print(f"{int(r.port):>4}{int(r.x):>5}{int(r.y):>6}{int(r.z):>6}"
              f"{int(r.single_neutron_candidates):>9}{int(r.multiple_neutron_candidates):>8}"
              f"{int(r.unique_neutron_triggers):>8}{r.pct_multiple:>8.2f}")
    print("-" * 53)
    print(f"{'TOTAL':>27}{tot_s:>9}{tot_m:>8}{tot_u:>8}{100*tot_m/tot_u:>8.2f}")
    print(f"\nSelection-2 single fraction = {tot_s/tot_u:.4f}   "
          f"multiple fraction = {tot_m/tot_u:.4f}")

    if args.out:
        tab.to_csv(args.out, index=False)
        print(f"[sel2-mult] wrote {args.out}")


if __name__ == "__main__":
    main()
