#!/usr/bin/env bash
#
# run_optics_benchmark.sh
# -----------------------
# Run the OPTICS-vs-ClusterFinder neutron-like-cluster benchmark on the AmBe
# BeamCluster DATA sample.  Designed to run unattended (e.g. in a tmux session
# overnight, or on a local machine where the ROOT files are available).
#
# What it does:
#   * For every BeamCluster_<run>.root, per event:
#       - ClusterFinder clusters: read straight from the file branches.
#       - OPTICS clusters: re-cluster the hits within +-1 us of each CF cluster
#         time (the MC-pipeline "prefilter" convention), via run_optics_on_event.
#   * Applies the Stage-1 pre-selection (clusterPE<60, chargeBalance<0.5,
#     clusterHits>10) to BOTH methods and reports raw + pre-selected
#     ("neutron-like") clusters per event, per source port and combined.
#   * Writes results incrementally and RESUMES: runs already in the stats CSV
#     are skipped, so you can re-run this after an interruption.
#
# Usage:
#   ./run_optics_benchmark.sh [INPUT] [OUTDIR] [JOBS]
#   INPUT  : ROOT file, directory, or glob   (default: the /pnfs AmBe2.0v1 dir)
#   OUTDIR : output directory                (default: optics_beamcluster_benchmark)
#   JOBS   : parallel worker processes       (default: number of CPU cores)
#
# Examples:
#   # Remote server, overnight in tmux, use all cores:
#   tmux new -s optics
#   ./run_optics_benchmark.sh
#   #   (detach with Ctrl-b d ; reattach later with: tmux attach -t optics)
#
#   # Local machine where you copied the files to ./data/, 8 cores:
#   ./run_optics_benchmark.sh ./data/ optics_out 8

set -euo pipefail

# ---- 1. Locate the analysis dir (where this script lives) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 2. Activate the Python environment --------------------------------------
# On dajana's ANNIE remote machine this is the working venv (system python3 has
# a broken numpy).  On a local machine, point this at any venv/conda env that
# has: uproot numpy pandas scikit-learn awkward pyarrow matplotlib.
VENV="${OPTICS_VENV:-/exp/annie/app/users/dajana/myboy/bin/activate}"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
else
    echo "WARNING: venv activate script not found at $VENV"
    echo "         Set OPTICS_VENV=/path/to/venv/bin/activate or ensure the"
    echo "         required packages are importable from your current python."
fi

# ---- 3. Arguments ------------------------------------------------------------
INPUT="${1:-/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/}"
OUTDIR="${2:-optics_beamcluster_benchmark}"
JOBS="${3:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

echo "=========================================================="
echo " OPTICS vs ClusterFinder benchmark — AmBe DATA"
echo "   input  : $INPUT"
echo "   outdir : $OUTDIR"
echo "   jobs   : $JOBS parallel worker(s)"
echo "   python : $(command -v python)"
echo "=========================================================="

# ---- 4. Run ------------------------------------------------------------------
# --jobs N      : process N files in parallel (CPU-bound; set near core count)
# (resume is ON by default: existing runs in the CSV are skipped)
python -u analyze_optics_beamcluster_data.py \
    "$INPUT" \
    --output-dir "$OUTDIR" \
    --jobs "$JOBS" \
    2>&1 | tee -a "${OUTDIR%/}_run.log"

echo ""
echo "DONE.  Outputs in: $OUTDIR/"
echo "  optics_beamcluster_summary.txt   <- headline OPTICS-vs-CF table"
echo "  optics_beamcluster_perport.csv   <- per-port means/medians"
echo "  optics_beamcluster_stats.csv     <- per-event detail"
echo "  optics_beamcluster_rate.pdf      <- comparison plots"
