#!/usr/bin/env bash
#
# run_waveform_gated_optics.sh
# ----------------------------
# Full waveform-gated OPTICS workflow for the AmBe2.0v1 data, in 3 stages:
#
#   Stage 1+2 : AmBe waveform IC cut (v1: 400<IC<575) -> good_events,
#               then keep BeamCluster events with eventTimeTank in good_events
#               + ambe_single_cut -> EventAmBeNeutronCandidates_<TAG>_<run>.csv
#               (driver: run_waveform_gated_pipeline.py)
#   Stage 3   : per-event OPTICS on those candidate CSVs
#               (src/ambe/clustering/optics_data.py -> writes *_OPTICS.csv)
#
# Heavy step is Stage 1 (reads ALL raw waveforms; ~hundreds of GB total).
# Run it in tmux:
#     tmux new -s wfgate
#     ./run_waveform_gated_optics.sh
#     # detach: Ctrl-b d   reattach: tmux attach -t wfgate
#
# Args:  ./run_waveform_gated_optics.sh [DATASET_DIR] [TAG]
#   DATASET_DIR : dir with BeamCluster_<run>.root and per-run waveform subdirs
#                 (default: /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1)
#   TAG         : output filename tag (default: AmBe2.0v1_gated)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${OPTICS_VENV:-/exp/annie/app/users/dajana/myboy/bin/activate}"
if [[ -f "$VENV" ]]; then source "$VENV"; else
    echo "WARNING: venv not found at $VENV; ensure required packages are importable."; fi

DATASET="${1:-/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1}"
TAG="${2:-AmBe2.0v1_gated}"

echo "=================================================================="
echo " Waveform-gated OPTICS workflow"
echo "   dataset : $DATASET"
echo "   tag     : $TAG"
echo "   IC cut  : 400 < IC_adjusted < 575  (v1)"
echo "=================================================================="

# ---- Stage 1+2: waveform gating -> candidate CSVs ----------------------------
echo ""; echo ">>> STAGE 1+2: waveform IC cut + BeamCluster matching ..."
python -u run_waveform_gated_pipeline.py --dataset "$DATASET" --runinfo "$TAG" \
    2>&1 | tee "waveform_gated_${TAG}_stage12.log"

# ---- Stage 3: OPTICS on the gated candidate CSVs -----------------------------
echo ""; echo ">>> STAGE 3: OPTICS secondary clustering on gated candidates ..."
python -u - "$TAG" <<'PY' 2>&1 | tee "waveform_gated_${TAG}_stage3.log"
import sys, glob
sys.path.insert(0, "src")
from ambe.clustering.optics_data import process_file
tag = sys.argv[1]
pattern = f"EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{tag}_*.csv"
files = sorted(f for f in glob.glob(pattern) if "_OPTICS" not in f)
print(f"OPTICS Stage 3: {len(files)} candidate CSV(s) matching {pattern}")
for f in files:
    process_file(f)          # writes <name>_OPTICS.csv next to each input
print("Stage 3 done. *_OPTICS.csv written with an 'optics_labels' column per event.")
PY

echo ""
echo "DONE."
echo "  Candidates : EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_${TAG}_<run>.csv"
echo "  OPTICS out : EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_${TAG}_<run>_OPTICS.csv"
echo "  Acceptance : TriggerSummary/AmBeWaveformResults_${TAG}.csv"
echo ""
echo "To benchmark OPTICS-vs-ClusterFinder on the gated set, run optics_analysis.py"
echo "(it pairs *_OPTICS.csv with the traditional CSVs and applies PE<60, CB<0.5, hits>10)."
