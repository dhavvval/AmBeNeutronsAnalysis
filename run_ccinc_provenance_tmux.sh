#!/bin/bash
# Write new products in this checkout; archived inputs remain configurable.
AMBE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AMBE_OUT="${AMBE_OUT:-$AMBE_REPO/ambe_output}"
# Remaining CCinc v3 cluster-provenance / hit-time-window runs, queued for tmux.
#
# Run under tmux so the work survives losing the terminal:
#     tmux new -d -s ccincprov ./run_ccinc_provenance_tmux.sh
#     tmux attach -t ccincprov
#
# Sequential on purpose: each stage is already single-process CPU-bound over the same
# pulses table, so running them in parallel only contends for page cache and makes the
# per-stage timings unreadable.
set -u   # NOT -e: one failing stage must not silently cancel the ones after it

cd "$AMBE_REPO" || exit 1
PY=/exp/annie/app/users/dajana/myboy/bin/python

run () {            # run <logfile> <args...>
  local log="$1"; shift
  echo "=== $(date '+%H:%M:%S')  START  $*"
  if "$PY" -m ambe.cli "$@" > "$log" 2>&1; then
    echo "=== $(date '+%H:%M:%S')  OK     $* -> $log"
  else
    echo "=== $(date '+%H:%M:%S')  FAILED $* (exit $?) -> $log"
    tail -15 "$log"
  fi
}

# World sample: the tank numbers are already in hand, world is the other half of the
# merged training set and has to be characterised the same way before either is quoted.
run logs_stage1_ccinc_v3_world.log \
    mc match --config configs/cc_neutrino_v3world_truthtag.yaml \
    --mode stage1 --method both

run logs_htw_world_cf.log \
    mc timewindows --config configs/cc_neutrino_v3world_truthtag.yaml \
    --method clusterfinder

run logs_htw_world_optics.log \
    mc timewindows --config configs/cc_neutrino_v3world_truthtag.yaml \
    --method optics

echo "=== $(date '+%H:%M:%S')  ALL STAGES DONE"
