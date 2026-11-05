#!/bin/bash
# Full-sample capture truth (hit composition of clusters) on EAF. Stages 2-3 (features,
# NN/XGB retraining) exist but are OFF by default (STOP_STAGE=1).
#
# Launch from a JupyterLab terminal on EAF (tmux only exists there):
#     tmux new -d -s capfull ./run_capture_full_eaf.sh
#     tmux attach -t capfull
#
# Prerequisite, on the GPVM (EAF has no /pnfs): the ROOT truth must already be cached
#     ambe mc capture --config configs/cc_neutrino_v3_truthtag.yaml --truth-unit capture --hit-cleaning none --hit-window delayed --cache-only
# This script symlinks those caches into the isolated capfix run directory.
#
# Everything writes under the capfix config's own output_root (see memory note
# eaf-output-isolation: output_root in the config overrides AMBE_OUT).
set -u    # NOT -e: a failing stage must not silently cancel the ones after it

AMBE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$AMBE_REPO" || exit 1
PY=/exp/annie/app/users/dajana/myboy/bin/python
CFG=configs/cc_neutrino_v3_truthtag_capfix.yaml
RUN=cc_neutrino_v3_truthtag_capfix
OUT=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output_capture/$RUN/parquet
SRC=$AMBE_REPO/ambe_output/cc_neutrino_v3_truthtag/parquet
LOG=../EB_BC_TA_work/logs
: "${JOBS:=120}"
: "${START_STAGE:=1}"
# Default: ONLY stage 1 (capture truth / hit composition). Features (2) and MVA (3) are
# off until explicitly asked for with STOP_STAGE=2 or 3.
: "${STOP_STAGE:=1}"
: "${METHODS:=peak clusterfinder optics}"
# Class-balanced weights already handle imbalance; the 3:1 cap threw away ~80% of signal.
: "${MAX_RATIO:=100}"
# Guard: this is sized for EAF (126 cores / 472 GB). On a 4-core / 11 GB gpvm, 120 workers
# each holding a file of hits would exhaust memory on a shared node.
NCPU=$(nproc)
if (( JOBS > NCPU )); then
  echo "REFUSING: JOBS=$JOBS but this host ($(hostname)) has $NCPU cores."
  echo "Run on EAF, or set JOBS<=$NCPU (e.g. JOBS=3 on a gpvm)."
  exit 1
fi
mkdir -p "$LOG" "$OUT"

# One BLAS/OpenMP thread per worker: ~48x faster on EAF (memory: eaf-blas-thread-pinning).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for f in pulses clusterfinder true_captures true_nu_vertices; do
  if [[ ! -e "$SRC/cc_neutrino_v3_truthtag__$f.parquet" ]]; then
    echo "MISSING $SRC/cc_neutrino_v3_truthtag__$f.parquet — run the --cache-only step on the gpvm"
    exit 1
  fi
  ln -sfn "$SRC/cc_neutrino_v3_truthtag__$f.parquet" "$OUT/${RUN}__$f.parquet"
done

run () {                    # run <logfile> <ambe.cli args...>
  local log="$1"; shift
  echo "=== $(date '+%F %T')  START  $*"
  if "$PY" -m ambe.cli "$@" > "$log" 2>&1; then
    echo "=== $(date '+%F %T')  OK     -> $log"
  else
    echo "=== $(date '+%F %T')  FAILED (exit $?) -> $log"; tail -15 "$log"
  fi
}

# 1. Capture-matched truth + clustering benchmark + vertex study, all 498 files.
if (( START_STAGE <= 1 )); then
  run "$LOG/capfull_capture.log" mc capture --truth-unit capture --hit-cleaning none --hit-window delayed --config "$CFG" --jobs "$JOBS" --vertex \
      --timing-offsets off --tag full
fi

# 2. Fixed-geometry features (tank frame, no MC timing offsets) for optics, CF and peak.
if (( START_STAGE <= 2 && STOP_STAGE >= 2 )); then
  run "$LOG/capfull_features.log" mc features --config "$CFG" --jobs "$JOBS"
fi

# 3. NN (primary) + XGBoost, per method, capture label vs the old composition label.
if (( START_STAGE <= 3 && STOP_STAGE >= 3 )); then
  CT="$OUT/${RUN}__capture_truth_clusters__full.parquet"
  for M in $METHODS; do
    run "$LOG/capfull_mva_${M}_capture.log" mc mva --config "$CFG" --method "$M" \
        --keep-prompt-bkg --max-ratio "$MAX_RATIO" --label-scheme capture --capture-table "$CT" \
        --out-tag "$M" --save-model
    run "$LOG/capfull_mva_${M}_composition.log" mc mva --config "$CFG" --method "$M" \
        --keep-prompt-bkg --max-ratio "$MAX_RATIO" --out-tag "$M" --save-model
  done
fi
echo "=== $(date '+%F %T')  ALL STAGES DONE"
