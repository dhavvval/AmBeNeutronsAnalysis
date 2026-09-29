#!/usr/bin/env python3
"""Build a subsampled pulses+clusterfinder pair for an OPTICS sweep.

Why subsample at all: the sweep only needs the config RANKING, and ranking is
stable under subsampling while absolute purities are not. Measured on the
previous 36-config sweep, a 25% sample reproduces the full-sample ranking with
Spearman rho = 0.94; the top configs sit within 0.002 purity of each other, which
is at the noise floor even on the FULL tank sample. Running everything buys
precision on a difference that is not physical.

Sampling is by SOURCE FILE (= parquet row group), not by random rows or random
events. Two reasons:
  1. All hits of one event live in one row group, so file-level sampling keeps
     every sampled event complete. Random-row sampling would shred events.
  2. It streams one row group at a time, so the 5.1 GB tank table never has to
     be held in memory.

Usage:
    python3 make_sweep_sample.py <src_run> <dst_run> [fraction] [--seed N]

    python3 make_sweep_sample.py cc_neutrino_v3world_truthtag sweep_v3_world_truthtag 0.25
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT_ROOT = Path("ambe_output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_run")
    ap.add_argument("dst_run")
    ap.add_argument("fraction", nargs="?", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default=None,
                    help="where to WRITE the sample (e.g. an EAF-isolated AMBE_OUT). "
                         "Defaults to the same root the source is read from.")
    ap.add_argument("--src-root", default=None,
                    help="where to READ the source run from (defaults to ambe_output)")
    a = ap.parse_args()

    src_root = Path(a.src_root) if a.src_root else OUT_ROOT
    dst_root = Path(a.out_root) if a.out_root else src_root
    src_dir = src_root / a.src_run / "parquet"
    dst_dir = dst_root / a.dst_run / "parquet"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_pulses = src_dir / f"{a.src_run}__pulses.parquet"
    if not src_pulses.exists():
        sys.exit(f"missing {src_pulses}")

    pf = pq.ParquetFile(str(src_pulses))
    n_groups = pf.num_row_groups
    n_take = max(1, int(round(n_groups * a.fraction)))

    rng = np.random.default_rng(a.seed)
    take = sorted(rng.choice(n_groups, n_take, replace=False).tolist())
    print(f"[sample] {a.src_run}: {n_groups} row groups "
          f"({pf.metadata.num_rows:,} hits) -> taking {n_take} "
          f"({100*n_take/n_groups:.0f}%), seed={a.seed}")

    dst_pulses = dst_dir / f"{a.dst_run}__pulses.parquet"
    writer = None
    kept_events, kept_rows = set(), 0
    try:
        for i, g in enumerate(take):
            tbl = pf.read_row_group(g)
            if writer is None:
                writer = pq.ParquetWriter(str(dst_pulses), tbl.schema)
            writer.write_table(tbl)
            kept_rows += tbl.num_rows
            kept_events.update(tbl.column("eventID").to_pylist())
            if (i + 1) % 25 == 0 or i + 1 == len(take):
                print(f"[sample]   {i+1}/{len(take)} groups  {kept_rows:,} hits",
                      flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"[sample] wrote {kept_rows:,} hits -> {dst_pulses}")

    # ClusterFinder sidecar: subset to the sampled events. It is small enough to
    # hold in memory (8-133 MB), unlike the pulses table.
    src_cf = src_dir / f"{a.src_run}__clusterfinder.parquet"
    dst_cf = dst_dir / f"{a.dst_run}__clusterfinder.parquet"
    if src_cf.exists():
        cf = pd.read_parquet(src_cf)
        # eventID is only unique WITHIN a source file, so if the sidecar carries
        # _source_file we must match on the pair, not on eventID alone.
        if "_source_file" in cf.columns:
            src_files = set()
            for g in take:
                t = pf.read_row_group(g, columns=["_source_file"])
                src_files.update(t.column("_source_file").to_pylist())
            sub = cf[cf["_source_file"].isin(src_files)]
        else:
            sub = cf[cf["eventID"].isin(kept_events)]
        sub.to_parquet(dst_cf, index=False)
        print(f"[sample] wrote {len(sub):,} CF clusters -> {dst_cf}")
    else:
        print(f"[sample] WARNING no ClusterFinder sidecar at {src_cf}; "
              "the sweep will have no CF baseline row")

    print(f"[sample] sampled events: {len(kept_events):,}")


if __name__ == "__main__":
    main()
