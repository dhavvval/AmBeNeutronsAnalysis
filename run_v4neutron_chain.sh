#!/usr/bin/env bash
#
# run_v4neutron_chain.sh — wait for the AmBe2.0v4_ext Stage-1 pass to finish, then
# run the full 28-run neutron campaign end to end.
#
# Stage 1 for the 8 runs that were never processed (6243, 6244, 6246, 6247, 6249,
# 6250, 6251, 6252) is already running under tag AmBe2.0v4_ext. Everything after it
# is mechanical, so it is chained here rather than babysat.
#
# The completion check polls the Stage-1 LOG, not `pgrep`: a pgrep pattern specific
# enough to match the pipeline also matches this script's own command line, and the
# watcher then waits on itself forever.
#
set -uo pipefail
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis

S1LOG=logs_stage1_v4ext.log
CHAINLOG=logs_v4neutron_chain.log
: > $CHAINLOG

echo "=== chain start $(date) ===" | tee -a $CHAINLOG

# ── 1. wait for Stage 1 ─────────────────────────────────────────────────────
while true; do
  if grep -qE "ANALYSIS PIPELINE COMPLETE|PIPELINE COMPLETE" $S1LOG 2>/dev/null; then
    echo "[chain] Stage 1 complete $(date)" | tee -a $CHAINLOG; break
  fi
  if grep -qE "Traceback" $S1LOG 2>/dev/null; then
    echo "[chain] Stage 1 FAILED — aborting" | tee -a $CHAINLOG
    grep -A 15 Traceback $S1LOG | tail -25 | tee -a $CHAINLOG
    exit 1
  fi
  # If the python process is gone but neither marker appeared, it died silently.
  if ! ps -C python -o args= 2>/dev/null | grep -q run_waveform_gated_pipeline; then
    sleep 30
    if ! grep -qE "PIPELINE COMPLETE" $S1LOG 2>/dev/null; then
      echo "[chain] Stage 1 process gone with no completion marker — aborting" \
        | tee -a $CHAINLOG
      tail -20 $S1LOG | tee -a $CHAINLOG
      exit 1
    fi
  fi
  sleep 120
done

n=$(ls EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_AmBe2.0v4_ext_*.csv \
      2>/dev/null | wc -l)
echo "[chain] AmBe2.0v4_ext candidate CSVs: $n (expect 8)" | tee -a $CHAINLOG
if [ "$n" -ne 8 ]; then
  echo "[chain] wrong count — aborting rather than running a partial campaign" \
    | tee -a $CHAINLOG
  exit 1
fi

# ── 2-4. gate, extract, score the 28 source runs ────────────────────────────
# The extractor resumes per-run shards, so the 20 already extracted are skipped and
# only the 8 new ones run.
echo "[chain] neutron campaign $(date)" | tee -a $CHAINLOG
bash run_ccinc_v3_ambe6266.sh --campaign v4neutron all 2>&1 | tail -40 | tee -a $CHAINLOG
rc=${PIPESTATUS[0]}
echo "[chain] driver exit $rc" | tee -a $CHAINLOG
[ "$rc" -eq 0 ] || exit 1

# ── 5. classify ─────────────────────────────────────────────────────────────
source /exp/annie/app/users/dajana/myboy/bin/activate
export PYTHONUNBUFFERED=1
python ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4neutron \
  > logs_classify_v4neutron.log 2>&1
echo "[chain] classify exit $? $(date)" | tee -a $CHAINLOG
grep -viE "tensorflow|cuda|cuInit|TF-TRT" logs_classify_v4neutron.log | tail -30 \
  | tee -a $CHAINLOG

echo "=== chain done $(date) ===" | tee -a $CHAINLOG
