"""Run the shared IC waveform gate and neutron-candidate extraction.

Run from the repository root: python -m scripts.workflows.run_waveform_gated_pipeline
"""

import argparse
import re
import sys
from pathlib import Path

from ambe.data.processor import AmBeNeutronProcessing, WaveformConfig, CutCriteria

DEFAULT_DATASET = "/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Dir with BeamCluster_<run>.root AND per-run waveform "
                        "subdirs <run>/AmBeWaveforms_<run>_*.root")
    p.add_argument("--runinfo", default="AmBe2.0v1_gated",
                   help="Tag used in output filenames")
    p.add_argument("--runs", default=None,
                   help="Comma-separated allowlist of run numbers. Without it every "
                        "BeamCluster_<run>.root in --dataset is processed, and the "
                        "job dies partway through on the first run missing from "
                        "AmBeNeutronProcessing.source_positions")
    p.add_argument("--pulse-gamma", type=int, default=None,
                   help="Override the IC lower bound for this run only "
                        "(default: whatever WaveformConfig's active default is)")
    p.add_argument("--pulse-max", type=int, default=None,
                   help="Override the IC upper bound for this run only "
                        "(default: whatever WaveformConfig's active default is)")
    p.add_argument("--no-feature-dump", action="store_true",
                   help="Skip writing TriggerSummary/WaveformFeatures_<tag>_<run>.parquet. "
                        "Keep the dump on unless disk is tight -- it makes every "
                        "later cut retune an offline pandas query instead of "
                        "another full read over dCache")
    args = p.parse_args()

    wf = WaveformConfig()
    if args.pulse_gamma is not None:
        wf.pulse_gamma = args.pulse_gamma
    if args.pulse_max is not None:
        wf.pulse_max = args.pulse_max
    proc = AmBeNeutronProcessing(config=wf, cuts=CutCriteria())

    # BeamCluster files are <dataset>/BeamCluster_<run>.root.
    # waveform_dir is the same dataset dir; process_run_waveforms looks in
    # <waveform_dir>/<run>/ for the AmBeWaveforms files.
    if args.runs:
        wanted = "|".join(r.strip() for r in args.runs.split(",") if r.strip())
        file_pattern = re.compile(rf"BeamCluster_({wanted})\.root")
    else:
        file_pattern = re.compile(r"BeamCluster_(\d+)\.root")

    print(f"Dataset      : {args.dataset}")
    if args.runs:
        print(f"Runs         : {args.runs}")
    print(f"IC window    : {wf.pulse_gamma} < IC_adjusted < {wf.pulse_max}")

    print(f"runinfo tag  : {args.runinfo}")
    print(f"BeamCluster ntuples: ANNIEEventTreeMaker (which_tree=1)\n")

    proc.run_complete_analysis_pipeline(
        data_directory=args.dataset,
        waveform_dir=args.dataset,        # per-run subdirs live inside the dataset
        file_pattern=file_pattern,
        # campaign=1 -> RWM_ folder pattern. Correct for v1 and v4; note that v3
        # (5xxx, new-PMT) carries the AmBe tag pulse in BRF_ instead and needs
        # campaign=2 -- its RWM_ channel is flat noise.
        campaign=1,
        runinfo=args.runinfo,
        which_tree=1,                     # ANNIEEventTreeMaker: numberOfClusters, Cluster_Hit*
        save_waveform_samples=False,
        plot_ic_distributions=True,
        dump_features=not args.no_feature_dump,
    )

    print("\nStage 1+2 done. Candidate CSVs in EventAmBeNeutronCandidatesData/")
    if not args.no_feature_dump:
        print(f"Per-waveform features: TriggerSummary/WaveformFeatures_{args.runinfo}_<run>.parquet")

    print("Next: run OPTICS (Stage 3) on those CSVs — see run_waveform_gated_optics.sh")


if __name__ == "__main__":
    main()
