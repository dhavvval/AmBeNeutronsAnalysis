#!/usr/bin/env bash
# EAF variant of run_optics_sweep10_overnight.sh.
#
# Two differences from the overnight script, both deliberate:
#
#   1. PARALLEL. Stage A passes --jobs to `ambe mc optics`, which fans the
#      per-event work across worker processes. Events are independent, so the
#      metrics are identical to a serial run (verified); only row order differs.
#
#   2. ISOLATED OUTPUT. Everything is written under a separate AMBE_OUT root so
#      an EAF run can never overwrite gpvm results. /exp is mounted on BOTH
#      machines and the pipeline's resume guards assume a single writer, so
#      sharing one output tree between them corrupts state.
#
# Usage (in a JupyterLab terminal on EAF, inside tmux):
#   tmux new -s eafsweep
#   bash run_optics_sweep10_eaf.sh            # auto-sizes --jobs
#   JOBS=32 bash run_optics_sweep10_eaf.sh    # or pin it
#
set -uo pipefail

AMBE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV=/exp/annie/app/users/dajana/myboy/bin/activate

# Isolated output root. Override with AMBE_OUT=... to point somewhere else.
export AMBE_OUT="${AMBE_OUT:-/exp/annie/app/users/dajana/eaf_work/ambe_output_eaf}"
OUT_ROOT="$AMBE_OUT"

# Dedicated EAF config: its own run_name AND an absolute output_root.
# NOTE: a config's `output_root:` key OVERRIDES the AMBE_OUT env var, so
# isolation must live in the config -- exporting AMBE_OUT alone is not enough.
SWEEP_CFG=configs/cc_neutrino_optics_sweep10_eaf.yaml
SWEEP_RUN=cc_neutrino_optics_sweep_eaf
SRC_RUN=cc_neutrino_optics_sweep          # where the input sample lives
GPVM_OUT="${AMBE_REPO}/ambe_output"
LOG="${AMBE_OUT}/logs_optics_sweep10_eaf.log"

# Worker count: honour JOBS, else use the cores we are actually allowed.
# Capped at 36 because that is the config-grid width; beyond it the extra
# workers mostly contend rather than help.
if [[ -z "${JOBS:-}" ]]; then
  JOBS=$(python3 -c "import os;print(min(36,len(os.sched_getaffinity(0))))")
fi

mkdir -p "$OUT_ROOT/$SWEEP_RUN/parquet" "$OUT_ROOT/$SWEEP_RUN/csv"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo "  EAF 10% OPTICS sweep — start $(date)"
echo "  host       : $(hostname)"
echo "  cores usable: $(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')"
echo "  --jobs     : $JOBS"
echo "  output root: $OUT_ROOT     (gpvm tree left untouched)"
echo "=================================================================="

# Stage A inputs: symlink the existing 10% sample rather than recopying 137 MB.
for kind in pulses clusterfinder; do
  link="$OUT_ROOT/$SWEEP_RUN/parquet/${SWEEP_RUN}__${kind}.parquet"
  src="$GPVM_OUT/$SRC_RUN/parquet/${SRC_RUN}__${kind}.parquet"
  if [[ ! -e "$link" ]]; then
    [[ -f "$src" ]] || { echo "FATAL: input missing: $src"; exit 1; }
    ln -s "$src" "$link"
    echo "linked $kind sample"
  fi
done

cd "$AMBE_REPO"
source "$VENV"
export PYTHONUNBUFFERED=1

# CRITICAL on a many-core node: numpy/scipy link OpenBLAS, which by default starts
# one thread PER CORE. With N worker processes on a 126-core node that is N*126
# threads competing for 126 cores -- massive oversubscription, and the run ends up
# barely faster than serial. One BLAS thread per worker is what we want, since the
# parallelism is already at the event level.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo; echo "--- Stage A: ambe mc optics (36 configs, --jobs $JOBS) ---"
SECONDS=0
ambe mc optics --config "$SWEEP_CFG" --jobs "$JOBS"
rc=$?
echo
echo "Stage A finished rc=$rc in ${SECONDS}s ($((SECONDS/60))m $((SECONDS%60))s)"
echo "gpvm serial baseline for the whole script was 2h 13m 42s."

SUMMARY="${OUT_ROOT}/${SWEEP_RUN}/csv/${SWEEP_RUN}__optics_summary.csv"
if [[ -f "$SUMMARY" ]]; then
  echo; echo "--- summary head ---"; head -12 "$SUMMARY"
  echo; echo "wrote: $SUMMARY"
else
  echo "WARN: no summary at $SUMMARY"
fi
echo "done $(date)"
