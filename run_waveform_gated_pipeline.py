"""
run_waveform_gated_pipeline.py

Stage 1+2 of the waveform-gated OPTICS workflow:

  Stage 1  AmBe waveform IC cut (pulse_gamma < IC_adjusted < pulse_max, no 2nd
           pulse) -> set of accepted tank timestamps (good_events)
  Stage 2  keep BeamCluster events whose eventTimeTank is in good_events,
           apply ambe_single_cut (0<PE<=100, CCB<0.45, clusterTime>=2000,
           clusterHits>=5, cosmic veto)
           -> EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_<runinfo>_<run>.csv
           -> TriggerSummary/AmBeWaveformResults_<runinfo>.csv (acceptance counts)

It just calls the existing AmBeNeutronProcessing.run_complete_analysis_pipeline
(src/ambe/data/processor.py) with the WaveformConfig IC window currently active
in that module (flip the comment/uncomment block there for the dataset version
you're processing) and the correct paths for the dataset, where the raw
waveforms live in per-run subdirs <dataset>/<run>/AmBeWaveforms_<run>_pN_pM.root
and the BeamCluster ntuples are <dataset>/BeamCluster_<run>.root.

Run Stage 3 (OPTICS on the resulting candidate CSVs) afterwards with
src/ambe/clustering/optics_data.py + optics_analysis.py.

Stage 1 can also cut on fprompt, a pulse-shape variable (prompt-peak integral /
total-pulse integral) ported from lmlepin9/2x2_neutron_sources. Unlike IC it is
gain-independent, so it needs no per-campaign retuning, and it rejects low-IC
noise (high fprompt) and high-IC pile-up (low fprompt) independently of the IC
window -- which is what lets the IC window be widened to recover efficiency.
See --selection-mode.

Every run also writes TriggerSummary/WaveformFeatures_<tag>_<run>.parquet with
one row per waveform (timestamp included), so cuts can be retuned offline with
tune_waveform_fprompt.py instead of re-reading the ROOT files.

Usage:
  python run_waveform_gated_pipeline.py [--dataset DIR] [--runinfo TAG]

  # historical IC-only cut (default, unchanged)
  python run_waveform_gated_pipeline.py --dataset ... --runinfo v4_prod

  # widened IC window cleaned up by fprompt
  python run_waveform_gated_pipeline.py --dataset ... --runinfo v4_fprompt \
      --selection-mode fprompt2d --pulse-gamma 500 --pulse-max 2000
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
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
    p.add_argument("--selection-mode", default=None,
                   choices=["ic", "ic+fprompt", "fprompt2d"],
                   help="Stage-1 selection. 'ic' (default) is the historical "
                        "IC-window cut. 'ic+fprompt' adds the fprompt pulse-shape "
                        "window on top of the current IC window. 'fprompt2d' is "
                        "the same cut but meant to be paired with a deliberately "
                        "widened IC window (e.g. --pulse-gamma 500 --pulse-max 2000), "
                        "letting fprompt reject the low-IC noise and high-IC pile-up "
                        "that the tight IC window used to remove")
    p.add_argument("--fprompt-min", type=float, default=None,
                   help="Override the fprompt lower bound (default: WaveformConfig's)")
    p.add_argument("--fprompt-max", type=float, default=None,
                   help="Override the fprompt upper bound (default: WaveformConfig's)")
    p.add_argument("--no-feature-dump", action="store_true",
                   help="Skip writing TriggerSummary/WaveformFeatures_<tag>_<run>.parquet. "
                        "Keep the dump on unless disk is tight -- it makes every "
                        "later cut retune an offline pandas query instead of "
                        "another full read over dCache")
    args = p.parse_args()

    # IC window comes from whichever WaveformConfig default is currently active
    # in src/ambe/data/processor.py — do NOT hardcode a version's window here,
    # or this silently applies the wrong cut to whatever dataset --dataset points at.
    # --pulse-gamma/--pulse-max let you override it for a one-off test without
    # touching that shared default (and without affecting other runs).
    wf = WaveformConfig()
    if args.pulse_gamma is not None:
        wf.pulse_gamma = args.pulse_gamma
    if args.pulse_max is not None:
        wf.pulse_max = args.pulse_max
    if args.selection_mode is not None:
        wf.selection_mode = args.selection_mode
    if args.fprompt_min is not None:
        wf.fprompt_min = args.fprompt_min
    if args.fprompt_max is not None:
        wf.fprompt_max = args.fprompt_max

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
    print(f"Selection    : {wf.selection_mode}")
    if wf.selection_mode != "ic":
        print(f"fprompt      : {wf.fprompt_min} < fprompt < {wf.fprompt_max}")
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
        print(f"Tune the cuts offline: python tune_waveform_fprompt.py --runinfo {args.runinfo}")
    print("Next: run OPTICS (Stage 3) on those CSVs — see run_waveform_gated_optics.sh")


if __name__ == "__main__":
    main()
