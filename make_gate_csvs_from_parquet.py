#!/usr/bin/env python3
"""
make_gate_csvs_from_parquet.py

Emit per-run gate CSVs (the eventTankTime set) from the freshly reprocessed
common-input parquet (cf_raw/v1_raw_all.parquet), in the filename format that
analyze_optics_beamcluster_data.build_gate_sets expects:

    EventAmBeNeutronCandidates_<tag>_<run>.csv   with a single column eventTankTime

This guarantees Selection 1 (OPTICS) and Selection 2 (ClusterFinder) run over the
EXACT same IC-gated + cosmic-vetoed event set — NOT the stale mixed-campaign CSVs.
build_gate_sets reads only the eventTankTime column, so that is all we write.

Usage:
  python make_gate_csvs_from_parquet.py \
      --parquet .../cf_raw/v1_raw_all.parquet \
      --out-dir .../cf_raw/gate_csvs --tag v1_raw
"""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="v1_raw")
    args = p.parse_args()

    d = pd.read_parquet(args.parquet, columns=["run", "eventTankTime"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for run, sub in d.groupby("run"):
        ts = sub["eventTankTime"].drop_duplicates().astype("int64")
        fn = out / f"EventAmBeNeutronCandidates_{args.tag}_{int(run)}.csv"
        ts.to_frame("eventTankTime").to_csv(fn, index=False)
        n += 1
    print(f"[gate] wrote {n} per-run gate CSVs -> {out}  "
          f"({d['eventTankTime'].nunique():,} unique events, {d['run'].nunique()} runs)")


if __name__ == "__main__":
    main()
