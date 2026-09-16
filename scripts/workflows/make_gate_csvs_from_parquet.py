#!/usr/bin/env python3
"""Export waveform-gate timestamps from saved diagnostic tables.

Run from the repository root: python -m scripts.workflows.make_gate_csvs_from_parquet
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
