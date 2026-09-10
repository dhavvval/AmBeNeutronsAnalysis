#!/usr/bin/env bash
# Stage 1 + Stage 2 for BOTH neutron definitions, campaign and special runs.
#
# Four tags, none of them protected, so nothing that backs a published number is at
# risk (processor.py refuses the _gated / _ext / _fprompt tags outright):
#
#   AmBe2.0v4_all28_box     28 source runs, box cuts        -> deck 2
#   AmBe2.0v4_all28_mva     28 source runs, MVA neutron     -> deck 3
#   AmBe2.0v4_special_box    6 special runs, box cuts       -> deck 2
#   AmBe2.0v4_special_mva    6 special runs, MVA neutron    -> deck 3
#
# Run SEQUENTIALLY on purpose. Every stage is I/O bound on /pnfs waveform reads, so
# running them in parallel would contend for the same dCache mount without finishing
# any sooner, and would interleave four logs.
#
# Usage:
#   nohup bash run_stage1_both_selections.sh > logs_stage1_chain.log 2>&1 &
#
# Optional: WAIT_PID=<pid> to start only after an already-running job finishes.
set -uo pipefail

cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate
export MPLBACKEND=Agg PYTHONPATH=src

if [[ -n "${WAIT_PID:-}" ]]; then
    echo "[chain] waiting for pid $WAIT_PID to finish first..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[chain] pid $WAIT_PID done"
fi

rc_total=0
run_one() {
    local cfg=$1 sel=$2 log=$3
    if [[ -f "SKIP_$log" ]]; then echo "[chain] skipping $cfg"; return 0; fi
    echo ""
    echo "=================================================================="
    echo "[chain] $(date '+%F %T')  config=$cfg  selection=$sel"
    echo "=================================================================="
    python -u -m ambe.cli data process --config "configs/$cfg" --selection "$sel" \
        > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[chain] FAILED rc=$rc -- see $log"
        tail -20 "$log"
        rc_total=1
    else
        echo "[chain] OK -- $(grep -c 'Processing Run:' "$log") runs, log $log"
    fi
    return 0
}

run_one data_ambe2v4_all28_mva.yaml   mva logs_stage1_all28_mva.log
run_one data_ambe2v4_special_box.yaml box logs_stage1_special_box.log
run_one data_ambe2v4_special_mva.yaml mva logs_stage1_special_mva.log

echo ""
echo "[chain] $(date '+%F %T') all done, rc_total=$rc_total"
echo "[chain] trigger summaries produced:"
ls -la TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_{all28,special}_{box,mva}.csv 2>/dev/null
echo "[chain] published files, mtimes must be UNCHANGED:"
ls -la TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_gated.csv \
       TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_ext.csv \
       TriggerSummary/CaptureTimeFits_baseline.csv
exit $rc_total
