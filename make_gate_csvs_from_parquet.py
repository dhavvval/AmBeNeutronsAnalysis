#!/usr/bin/env python3
"""
make_gate_csvs_from_parquet.py

Emit per-run gate CSVs (the eventTankTime set) in the filename format that
analyze_optics_beamcluster_data.build_gate_sets expects:

    EventAmBeNeutronCandidates_<tag>_<run>.csv   with a single column eventTankTime

build_gate_sets reads only the eventTankTime column, so that is all we write.

TWO SOURCES (--source is required; there is no sensible default because the two
parquets carry different column names and different meanings)
---------------------------------------------------------------------------
`cf_raw`
    The reprocessed common-input parquet (cf_raw/v1_raw_all.parquet), columns
    `run` + `eventTankTime`. This is the original use: it guarantees Selection 1
    (OPTICS) and Selection 2 (ClusterFinder) run over the EXACT same IC-gated +
    cosmic-vetoed event set, not the stale mixed-campaign CSVs.

`step4`
    The AmBe special-runs diagnostics per-run join
    (AmBe_special_runs_diagnostics/outputs/step4_joined_<run>.parquet), columns
    `run` + `timestamp` + `passes_ic`. Written for the v4 special runs (6243+),
    which have NO gate CSV in EventAmBeNeutronCandidatesData/ — that directory
    stops at 6242. Only rows with passes_ic are emitted.

    `timestamp` here IS eventTimeTank, same clock, verified on run 6266: all 4,182
    diagnostics timestamps match a BeamCluster eventTimeTank exactly (2,171 of them
    passing IC). Re-verify that match on any new run before trusting the gate — a
    different clock silently produces an empty gate, and an empty gate makes
    extract_cf_features_beamcluster_data.py skip the run entirely while reporting
    success.

Usage:
  python make_gate_csvs_from_parquet.py --source cf_raw \
      --parquet .../cf_raw/v1_raw_all.parquet \
      --out-dir .../cf_raw/gate_csvs --tag v1_raw

  python make_gate_csvs_from_parquet.py --source step4 \
      --parquet /exp/annie/data/users/dajana/AmBe_special_runs_diagnostics/outputs/step4_joined_6266.parquet \
      --out-dir .../gate_csvs_6266 --tag AmBe2.0v4_ic
"""
import argparse
from pathlib import Path

import pandas as pd


def load_cf_raw(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["run", "eventTankTime"])


def load_step4(path: str) -> pd.DataFrame:
    """
    The diagnostics join, reduced to the IC-passing triggers.

    Every row is one IC waveform candidate, so the same trigger can appear more
    than once; the caller de-duplicates. `passes_ic` is the waveform cut decision
    and is the whole point of the gate — emitting the un-gated rows would produce
    a "gate" that selects every candidate trigger and quietly means nothing.
    """
    d = pd.read_parquet(path, columns=["run", "timestamp", "passes_ic"])
    n_all = len(d)
    d = d[d["passes_ic"].astype(bool)]
    print(f"[gate] step4: {len(d):,} of {n_all:,} candidates pass the IC cut")
    return d.rename(columns={"timestamp": "eventTankTime"})[["run", "eventTankTime"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["cf_raw", "step4"],
                   help="Which parquet layout to read. No default on purpose — "
                        "the two carry different columns and different meanings.")
    p.add_argument("--parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="v1_raw")
    args = p.parse_args()

    d = load_cf_raw(args.parquet) if args.source == "cf_raw" \
        else load_step4(args.parquet)
    if d.empty:
        raise SystemExit(f"[gate] no rows to write from {args.parquet} — refusing "
                         f"to emit an empty gate, which would silently skip runs")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for run, sub in d.groupby("run"):
        ts = sub["eventTankTime"].drop_duplicates().astype("int64")
        fn = out / f"EventAmBeNeutronCandidates_{args.tag}_{int(run)}.csv"
        ts.to_frame("eventTankTime").to_csv(fn, index=False)
        print(f"[gate]   run {int(run)}: {len(ts):,} gated triggers -> {fn.name}")
        n += 1
    print(f"[gate] wrote {n} per-run gate CSVs -> {out}  "
          f"({d['eventTankTime'].nunique():,} unique events, {d['run'].nunique()} runs)")


if __name__ == "__main__":
    main()
