#!/usr/bin/env bash
# ccv3 OPTICS sweeps (tank + world) at 25% on the Fermilab EAF.
#
# WHAT THIS DOES
#   1. Builds a 25% subsample of each ccv3 pulses table (by source file, so events
#      stay whole) under an ISOLATED output root.
#   2. Runs the 24-config OPTICS sweep on each, in parallel across EAF cores.
#   3. Runs compare_sweeps_pick_uniform.py to pick ONE config good for both.
#
# WHY 25% AND NOT EVERYTHING
#   The sweep needs the config RANKING, not absolute purities. Measured on the
#   previous sweep: a 25% sample reproduces the full-sample ranking at Spearman
#   rho = 0.94, while the top configs sit within 0.002 purity of each other --
#   at the noise floor even on the FULL 2.49M-event tank sample. Full tank would
#   cost ~15.6 h at --jobs 126 to resolve a difference that is not physical.
#
# WHY THE OUTPUT ROOT IS SEPARATE
#   /exp is mounted on BOTH the gpvm and EAF, and the pipeline's resume guards
#   assume a single writer. Writing into the shared ambe_output/ from EAF while
#   anything runs on the gpvm corrupts state. Everything here goes to AMBE_OUT.
#
# USAGE (EAF, inside tmux)
#   tmux new -s v3sweeps
#   bash run_sweeps_v3_eaf.sh                  # both, 25%, auto --jobs
#   FRACTION=0.10 bash run_sweeps_v3_eaf.sh    # cheaper pilot
#   ONLY=world JOBS=64 bash run_sweeps_v3_eaf.sh
#
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV=/exp/annie/app/users/dajana/myboy/bin/activate
SRC_ROOT="${REPO}/ambe_output"                     # read the ccv3 products from here
export AMBE_OUT="${AMBE_OUT:-/exp/annie/app/users/dajana/eaf_work/ambe_output_eaf}"

FRACTION="${FRACTION:-0.25}"
SEED="${SEED:-42}"
ONLY="${ONLY:-both}"                               # both | tank | world

# Workers. Default to the cores we may actually use, capped at 64: each worker
# holds its own event slice plus the CF frame, and going wider mostly contends.
if [[ -z "${JOBS:-}" ]]; then
  JOBS=$(python3 -c "import os;print(min(64,len(os.sched_getaffinity(0))))")
fi

mkdir -p "$AMBE_OUT"
LOG="${AMBE_OUT}/logs_sweeps_v3_eaf.log"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo "  ccv3 OPTICS sweeps — start $(date)"
echo "  host        : $(hostname)"
echo "  usable cores: $(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')"
echo "  --jobs      : $JOBS"
echo "  fraction    : $FRACTION   (seed $SEED)"
echo "  source root : $SRC_ROOT"
echo "  output root : $AMBE_OUT   <- gpvm tree untouched"
echo "  running     : $ONLY"
echo "=================================================================="

cd "$REPO"
source "$VENV"
export PYTHONUNBUFFERED=1

run_one () {
  local tag="$1" src_run="$2" cfg="$3"
  local dst_run="sweep_v3_${tag}_truthtag"

  echo; echo "################ $tag ################"
  if [[ -f "${AMBE_OUT}/${dst_run}/parquet/${dst_run}__pulses.parquet" ]]; then
    echo "[$tag] sample already present — reusing it"
  else
    echo "[$tag] building ${FRACTION} sample from ${src_run}"
    python3 make_sweep_sample.py "$src_run" "$dst_run" "$FRACTION" \
        --seed "$SEED" --src-root "$SRC_ROOT" --out-root "$AMBE_OUT" || {
      echo "[$tag] FATAL: sampling failed"; return 1; }
  fi

  echo; echo "[$tag] sweeping (24 configs, --jobs $JOBS)"
  SECONDS=0
  ambe mc optics --config "$cfg" --jobs "$JOBS"
  local rc=$?
  echo "[$tag] sweep rc=$rc in ${SECONDS}s ($((SECONDS/60))m)"
  return $rc
}

[[ "$ONLY" == "both" || "$ONLY" == "world" ]] && \
  run_one world cc_neutrino_v3world_truthtag configs/sweep_v3_world_truthtag.yaml
[[ "$ONLY" == "both" || "$ONLY" == "tank" ]] && \
  run_one tank  cc_neutrino_v3_truthtag      configs/sweep_v3_tank_truthtag.yaml

# ---- pick the uniform config ----
T="${AMBE_OUT}/sweep_v3_tank_truthtag/csv/sweep_v3_tank_truthtag__optics_summary.csv"
W="${AMBE_OUT}/sweep_v3_world_truthtag/csv/sweep_v3_world_truthtag__optics_summary.csv"
if [[ -f "$T" && -f "$W" ]]; then
  echo; echo "########## uniform-config selection ##########"
  python3 compare_sweeps_pick_uniform.py "$T" "$W"
else
  echo; echo "skipping comparison — need BOTH summaries:"
  [[ -f "$T" ]] || echo "  missing $T"
  [[ -f "$W" ]] || echo "  missing $W"
fi
echo; echo "done $(date)"
