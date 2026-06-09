#!/usr/bin/env python3
"""
merge_multiport_signal.py
=========================
Concatenate all 25 per-port+z feature parquets produced by run_steven_multiport.sh
into a single combined signal parquet for use with mva_analysis.py.

Output location (mimics normal pipeline structure so mva_analysis.py needs no changes):
  ambe_output/mc_steven_multiport/parquet/mc_steven_multiport__cluster_features.parquet

Adds two provenance columns:
  - port      : e.g. "port1", "port4"
  - z_pos     : e.g. "z0", "zminus50"

Usage
-----
    # After run_steven_multiport.sh completes:
    python merge_multiport_signal.py

    # Limit events per config for balanced training:
    python merge_multiport_signal.py --max-events-per-config 10000

    # Dry-run (report what would be merged without writing):
    python merge_multiport_signal.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# All 25 (port, z_label) combinations — must match configs/steven_multiport/
# ---------------------------------------------------------------------------
PORT_Z_COMBOS = [
    (port, zlabel)
    for port in ["port1", "port2", "port3", "port4", "port5"]
    for zlabel in ["z0", "z50", "z100", "zminus50", "zminus100"]
]

BASE_OUTPUT_ROOT = Path("/Users/dajana/Documents/AmBe/AmBeNeutronsAnalysis/ambe_output")
COMBINED_RUN_NAME = "mc_steven_multiport"


def find_parquet(port: str, z_label: str) -> Path | None:
    """Return path to cluster_features parquet for this config, or None if missing."""
    run_name = f"mc_steven_{port}_{z_label}"
    p = BASE_OUTPUT_ROOT / run_name / "parquet" / f"{run_name}__cluster_features.parquet"
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(prog="merge_multiport_signal")
    ap.add_argument(
        "--max-events-per-config", type=int, default=None, metavar="N",
        help="Randomly subsample each config to at most N events before merging "
             "(cluster-level rows; events are the unit). "
             "Use to enforce equal contribution from each port+z combo.",
    )
    ap.add_argument(
        "--method", default="all", choices=["optics", "clusterfinder", "all"],
        help="Clustering method to keep (default: all — preserves both OPTICS and "
             "ClusterFinder rows so a single merged parquet can feed "
             "mva_analysis.py --method {optics,clusterfinder}).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be merged without writing any output.",
    )
    ap.add_argument(
        "--output-root", default=str(BASE_OUTPUT_ROOT), metavar="PATH",
        help=f"Base output directory (default: {BASE_OUTPUT_ROOT}).",
    )
    args = ap.parse_args()

    output_root = Path(args.output_root)
    out_dir = output_root / COMBINED_RUN_NAME / "parquet"
    out_path = out_dir / f"{COMBINED_RUN_NAME}__cluster_features.parquet"

    print(f"[merge] scanning {len(PORT_Z_COMBOS)} port+z combinations ...")

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    total_clusters_before = 0

    for port, zlabel in PORT_Z_COMBOS:
        p = find_parquet(port, zlabel)
        tag = f"{port}_{zlabel}"
        if p is None:
            print(f"  [MISSING] {tag}")
            missing.append(tag)
            continue

        df = pd.read_parquet(p)

        # Filter to the requested clustering method (or keep all)
        if "method" in df.columns and args.method != "all":
            df = df[df["method"] == args.method].reset_index(drop=True)

        n_before = len(df)
        total_clusters_before += n_before

        # Optional per-config event cap (subsample events, not clusters, to
        # preserve intra-event cluster structure in the sample)
        if args.max_events_per_config is not None and "eventID" in df.columns:
            unique_events = df["eventID"].unique()
            if len(unique_events) > args.max_events_per_config:
                rng = pd.np.random.default_rng(seed=42)
                kept = rng.choice(unique_events, size=args.max_events_per_config, replace=False)
                df = df[df["eventID"].isin(kept)].reset_index(drop=True)

        # Add provenance columns
        df.insert(0, "z_pos", zlabel)
        df.insert(0, "port",  port)

        n_after = len(df)
        print(f"  [ok] {tag:25s}  {n_before:>7,} clusters → kept {n_after:>7,}")
        frames.append(df)

    if not frames:
        sys.exit("[merge] ERROR: no parquets found — run run_steven_multiport.sh first.")

    combined = pd.concat(frames, ignore_index=True)
    n_total = len(combined)
    n_signal = int((combined.get("is_truth_neutron", pd.Series(dtype=int)) == 1).sum()) \
               if "is_truth_neutron" in combined.columns else None

    print(f"\n[merge] combined: {n_total:,} clusters from {len(frames)} / {len(PORT_Z_COMBOS)} configs")
    if n_signal is not None:
        print(f"         of which: {n_signal:,} truth-neutron signal clusters")
    if missing:
        print(f"         MISSING ({len(missing)}): {', '.join(missing)}")

    if args.dry_run:
        print("[merge] --dry-run: no file written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[merge] wrote {out_path}")
    print(f"         size: {size_mb:.1f} MB")
    print(f"\nNext step:")
    print(f"  python mva_analysis.py --config configs/mc_steven_multiport_mva.yaml \\")
    print(f"                         --external-background offbeam")
    print(f"  python mva_analysis.py --config configs/mc_steven_multiport_mva.yaml \\")
    print(f"                         --external-background michel")


if __name__ == "__main__":
    main()
