"""
run_waveform_gated_pipeline.py

Stage 1+2 of the waveform-gated OPTICS workflow:

  Stage 1  AmBe waveform IC cut (v1: 400 < IC_adjusted < 575, no 2nd pulse)
           -> set of accepted tank timestamps (good_events)
  Stage 2  keep BeamCluster events whose eventTimeTank is in good_events,
           apply ambe_single_cut (0<PE<=100, CCB<0.45, clusterTime>=2000,
           clusterHits>=5, cosmic veto)
           -> EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_<runinfo>_<run>.csv
           -> TriggerSummary/AmBeWaveformResults_<runinfo>.csv (acceptance counts)

It just calls the existing AmBeNeutronProcessing.run_complete_analysis_pipeline
(src/ambe/data/processor.py) with the v1 WaveformConfig and the correct paths for
the AmBe2.0v1 dataset, where the raw waveforms live in per-run subdirs
<dataset>/<run>/AmBeWaveforms_<run>_pN_pM.root and the BeamCluster ntuples are
<dataset>/BeamCluster_<run>.root.

Run Stage 3 (OPTICS on the resulting candidate CSVs) afterwards with
src/ambe/clustering/optics_data.py + optics_analysis.py.

Usage:
  python run_waveform_gated_pipeline.py [--dataset DIR] [--runinfo TAG]
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
    args = p.parse_args()

    # v1 IC window (these are already the WaveformConfig defaults; set explicitly
    # so the cut is unambiguous and self-documenting).
    wf = WaveformConfig()
    wf.pulse_gamma = 400   # v1 lower IC bound
    wf.pulse_max   = 575   # v1 upper IC bound

    proc = AmBeNeutronProcessing(config=wf, cuts=CutCriteria())

    # BeamCluster files are <dataset>/BeamCluster_<run>.root.
    # waveform_dir is the same dataset dir; process_run_waveforms looks in
    # <waveform_dir>/<run>/ for the AmBeWaveforms files.
    file_pattern = re.compile(r"BeamCluster_(\d+)\.root")

    print(f"Dataset      : {args.dataset}")
    print(f"IC window    : {wf.pulse_gamma} < IC_adjusted < {wf.pulse_max}  (v1)")
    print(f"runinfo tag  : {args.runinfo}")
    print(f"BeamCluster ntuples: ANNIEEventTreeMaker (which_tree=1)\n")

    proc.run_complete_analysis_pipeline(
        data_directory=args.dataset,
        waveform_dir=args.dataset,        # per-run subdirs live inside the dataset
        file_pattern=file_pattern,
        campaign=1,                       # v1 -> RWM_ folder pattern
        runinfo=args.runinfo,
        which_tree=1,                     # ANNIEEventTreeMaker: numberOfClusters, Cluster_Hit*
        save_waveform_samples=False,
        plot_ic_distributions=True,
    )

    print("\nStage 1+2 done. Candidate CSVs in EventAmBeNeutronCandidatesData/")
    print("Next: run OPTICS (Stage 3) on those CSVs — see run_waveform_gated_optics.sh")


if __name__ == "__main__":
    main()
