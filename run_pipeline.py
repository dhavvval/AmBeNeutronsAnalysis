"""
run_pipeline.py
---------------
Driver: point at one or more BeamClusterAnalysisMC ROOT files, get OPTICS vs
ClusterFinder metrics out.

  python run_pipeline.py /path/to/ANNIETree_MC.root
  python run_pipeline.py /path/to/BeamCluster_*.root --outdir results/

Produces under --outdir:
  <stem>.parquet                    (per-pulse features + truth labels)
  <stem>_clusterfinder.parquet      (ClusterFinder sidecar)
  metrics.parquet                   (per (event, method, hyperparameters))
  metrics.summary.csv               (per-configuration averages)
"""

import argparse
from pathlib import Path

import pandas as pd

from mc_processor import process_file
from mc_optics import train_and_evaluate, summarise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="BeamCluster ROOT files")
    p.add_argument("--outdir", default="mc_results", help="Output directory")
    p.add_argument("--tree", default="Event", help="Tree name")
    p.add_argument("--min-samples", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--xi", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--t-scale", type=float, nargs="*", default=[None])
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for root_path in args.inputs:
        root_path = Path(root_path)
        pulse_parquet = outdir / (root_path.stem + ".parquet")
        cluster_parquet = outdir / (root_path.stem + "_clusterfinder.parquet")

        process_file(root_path, pulse_parquet, tree_name=args.tree)

        t_scale_list = tuple(None if v is None else float(v) for v in args.t_scale)
        metrics = train_and_evaluate(
            pulse_parquet,
            cluster_parquet if cluster_parquet.exists() else None,
            min_samples_list=tuple(args.min_samples),
            xi_list=tuple(args.xi),
            t_scale_list=t_scale_list,
        )
        metrics["source_file"] = root_path.name
        all_metrics.append(metrics)

    if not all_metrics:
        print("[run_pipeline] no metrics produced")
        return

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_parquet(outdir / "metrics.parquet", index=False)

    summary = summarise(metrics_df)
    summary_csv = outdir / "metrics.summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[run_pipeline] summary:")
    print(summary.to_string(index=False))
    print(f"\n[run_pipeline] wrote:\n  {outdir / 'metrics.parquet'}\n  {summary_csv}")


if __name__ == "__main__":
    main()
