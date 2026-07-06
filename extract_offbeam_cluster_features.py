#!/usr/bin/env python3
"""
Extract cluster-level features from the offbeam per-hit parquet.

Reads beamoff/background_optics_hits.parquet (columns: run, event_number,
cluster_id, is_background, x, y, z, t, pe, pmtID), groups hits by
(run, event_number, cluster_id), calls compute_cluster_features() on each
group, and writes a cluster-feature parquet ready for use as
--external-background in mva_analysis.py.

Output: beamoff/background_optics_cluster_features.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Repo root must be on sys.path (run from repo root, or via `python extract_offbeam...py`)
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ambe.mc.cluster_features import load_geometry, compute_cluster_features

GEO_PATH = (
    "/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/"
    "FullTankPMTGeometry.csv"
)
OFFSETS_PATH = (
    "/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/"
    "TankPMTTimingOffsets.csv"
)
IN_PARQUET  = REPO / "beamoff" / "background_optics_hits.parquet"
OUT_PARQUET = REPO / "beamoff" / "background_optics_cluster_features.parquet"


def extract_features(hits: pd.DataFrame, geo) -> pd.DataFrame:
    rows = []
    group_cols = ["run", "event_number", "cluster_id"]
    for keys, grp in hits.groupby(group_cols, sort=False):
        run, ev, cid = keys
        feats = compute_cluster_features(grp, geo, source_pos_m=None)
        if not feats:
            continue
        row = {"run": run, "event_number": ev, "cluster_id": cid,
               "is_background": 1}
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert offbeam per-hit parquet to cluster features."
    )
    parser.add_argument("--input",  default=str(IN_PARQUET),
                        help="Per-hit parquet (default: beamoff/background_optics_hits.parquet)")
    parser.add_argument("--output", default=str(OUT_PARQUET),
                        help="Output cluster-features parquet")
    parser.add_argument("--geo",     default=GEO_PATH)
    parser.add_argument("--offsets", default=OFFSETS_PATH)
    parser.add_argument("--max-clusters", type=int, default=None,
                        help="Stop after this many clusters (smoke-test mode)")
    args = parser.parse_args()

    print(f"Loading geometry from {args.geo}")
    geo = load_geometry(args.geo, args.offsets)

    print(f"Reading hits from {args.input}")
    hits = pd.read_parquet(args.input)
    print(f"  {len(hits):,} hits, {hits.groupby(['run','event_number','cluster_id']).ngroups:,} clusters")

    if args.max_clusters is not None:
        # Take first N cluster groups for smoke testing
        keys = (hits.groupby(["run", "event_number", "cluster_id"], sort=False)
                    .ngroups)
        group_keys = (hits.groupby(["run", "event_number", "cluster_id"], sort=False)
                          .apply(lambda x: x)
                          .groupby(level=[0,1,2]))
        # Simple approach: keep hits belonging to first max_clusters unique (run,ev,cid)
        unique_keys = hits[["run","event_number","cluster_id"]].drop_duplicates()
        keep = unique_keys.head(args.max_clusters)
        hits = hits.merge(keep, on=["run","event_number","cluster_id"], how="inner")
        print(f"  Smoke-test mode: keeping {args.max_clusters} clusters ({len(hits):,} hits)")

    print("Extracting cluster features...")
    df_feat = extract_features(hits, geo)
    print(f"  {len(df_feat):,} clusters with features extracted")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_path, index=False)
    print(f"Written: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity check
    print("\nColumn count:", len(df_feat.columns))
    print("is_background values:", df_feat["is_background"].value_counts().to_dict())
    if "pe_total" in df_feat.columns:
        print("pe_total: mean={:.1f}  median={:.1f}".format(
            df_feat["pe_total"].mean(), df_feat["pe_total"].median()))


if __name__ == "__main__":
    main()
