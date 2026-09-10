#!/usr/bin/env bash
#
# run_ccinc_v3_ambe6266.sh — score AmBe RUN 6266 (AmBe2.0v4) with the merged
# tank+world models, ungated and IC-gated, as the first v4 validation of the
# CCinc v3 selection on data.
#
# WHY 6266. It is a genuine source-in run (AmBe_special_runs_diagnostics REPORT,
# step 7). Of the neighbouring special runs, 6264 is ~2/3 cosmic-induced captures
# and 6270 is still corrupt (FIX_6270.md), so neither is usable and neither is
# scored here. This is a ONE-RUN pilot: it tests that the v4 extraction sits on the
# same footing as the June v1/v3 campaign, not that the selection is validated.
#
# WHY BOTH GATES. No EventAmBeNeutronCandidates CSV exists for 6266 — that
# directory stops at run 6242 — so the IC gate is derived here from the special-runs
# diagnostics join (see stage 1 below). Running ungated as well is what makes the
# gate's effect measurable rather than assumed.
#
# CLUSTERFINDER ONLY. OPTICS does not transfer to AmBe data (KS-to-MC-signal
# 0.42-0.64 vs 0.11-0.16, and the OPTICS data median sits BELOW the MC background
# median — REPORT_ccinc_v3_classifier_selection.md §4), and reco-tag/OPTICS fails
# outright with tau pinned at the fit bound. Both CF configurations are scored;
# truth-tag/CF is the deliverable, reco-tag/CF is an MC-best cross-check and is
# NOT data-applicable (its FV and muon kinematics read truth branches).
#
# THE TWO CLOBBERING HAZARDS from run_ccinc_v3_ambe_stage3.sh apply unchanged and
# are handled the same way — per-configuration staging directories with symlinks:
#   1. the scored output name derives from the INPUT name, and the configurations
#      share input files;
#   2. every frozen .pkl still stores the PRE-rename NN companion filename, so an
#      unstaged run silently produces three scores instead of four.
# Do not "simplify" the staging away.
#
# Usage (no silent defaults):
#   bash run_ccinc_v3_ambe6266.sh all
#   bash run_ccinc_v3_ambe6266.sh gate            # stage 1 only
#   bash run_ccinc_v3_ambe6266.sh extract         # stage 2 only
#   bash run_ccinc_v3_ambe6266.sh score           # stage 3 only
#
set -uo pipefail

cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate
export PYTHONUNBUFFERED=1

BASE=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output
GEO=/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry
AMBEDIR=/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4
CANDDIR=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/EventAmBeNeutronCandidatesData
S4DIR=/exp/annie/data/users/dajana/AmBe_special_runs_diagnostics/outputs
STEP4=$S4DIR/step4_joined_6266.parquet

# ── campaign selector ───────────────────────────────────────────────────────
# run6266  a single special run, gate derived here, ungated AND IC-gated.
# v4gated  the AmBe2.0v4_gated Stage-1+2 campaign: 20 runs whose gate CSVs are
#          the pipeline's OWN Stage-2 output, so nothing is derived and only the
#          gated mode exists. This is the set the AmBe efficiency/capture-time
#          numbers are quoted on, and it carries port positions for 19 positions.
CAMPAIGN=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --campaign) CAMPAIGN="${2:-}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$CAMPAIGN" in
  run6266|v4gated|v4neutron|v4special) ;;
  *) echo "Usage: $0 --campaign {run6266|v4gated|v4neutron|v4special} all|gate|extract|score [...]" >&2
     echo "  (--campaign is required; the two write to different directories and" >&2
     echo "   v4gated has no gate-derivation stage)" >&2
     exit 1 ;;
esac
if [ ${#ARGS[@]} -eq 0 ]; then
  echo "Usage: $0 --campaign {run6266|v4gated} all|gate|extract|score [...]" >&2
  exit 1
fi

if [ "$CAMPAIGN" = "run6266" ]; then
  OUT=$BASE/ambe_data/ccinc_v3_ambe6266
  MODES="ungated gated"
  RUNS="6266"
  PREFIX=ambe6266
  LOG=logs_ccinc_v3_ambe6266.log
  ALL_STAGES="gate extract score"
  JOBS=1
elif [ "$CAMPAIGN" = "v4neutron" ]; then
  # ALL 28 AmBe2.0v4 source runs. The 20 with a Stage-1 candidate CSV under the
  # AmBe2.0v4_gated tag, plus the 8 processed later under AmBe2.0v4_ext (same
  # 700-1200 IC window, written to a separate tag so the 20-run
  # TriggerSummary/AmBeWaveformResults_AmBe2.0v4_gated.csv -- plain to_csv, no
  # append -- is not overwritten).
  #
  # Excluded: 6254/6256 (pulser, ZERO IC-passing triggers -- 0 of 7,932 and 0 of
  # 4,254, so the gate has nothing to select) and 6264/6265/6266/6270 (labelled
  # no-source; --campaign v4special).
  OUT=$BASE/ambe_data/ccinc_v3_ambe_v4neutron
  MODES="gated"
  RUNS="6046 6056 6060 6061 6062 6165 6166 6186 6187 6188 6189 6230 6231 6232 6234 6235 6237 6239 6241 6242 6243 6244 6246 6247 6249 6250 6251 6252"
  PREFIX=ambev4neutron
  LOG=logs_ccinc_v3_ambe_v4neutron.log
  ALL_STAGES="gate extract score"
  JOBS=2
elif [ "$CAMPAIGN" = "v4special" ]; then
  # The four runs labelled no-source in the diagnostics' step7_source_presence.csv.
  # Kept SEPARATE from the neutron campaign on purpose and never merged into a
  # capture-time or efficiency number: 6264 is cosmic-contaminated (10x the prompt
  # rate, tau 48 us), 6265/6266/6270 are labelled no-source but behave like source-in
  # runs, and BeamCluster_6270.root is still the corrupted merge (11,195 duplicated
  # events of 46,211 -- FIX_6270.md).
  #
  # They have no Stage-1 candidate CSV, so the gate is derived from the diagnostics
  # join, which applies the SAME 700-1200 IC window plus second-pulse veto.
  OUT=$BASE/ambe_data/ccinc_v3_ambe_v4special
  MODES="gated"
  RUNS="6264 6265 6266 6270"
  PREFIX=ambev4special
  LOG=logs_ccinc_v3_ambe_v4special.log
  ALL_STAGES="gate extract score"
  JOBS=2
else
  OUT=$BASE/ambe_data/ccinc_v3_ambe_v4gated
  MODES="gated"
  # The 20 runs of AmBe2.0v4_gated. 6264/6265/6266/6270 are deliberately absent:
  # they have NO entry in src/ambe/data/processor.py source_positions, so they
  # carry no port position and an unfiltered pipeline run dies on them.
  RUNS="6046 6056 6060 6061 6062 6165 6166 6186 6187 6188 6189 6230 6231 6232 6234 6235 6237 6239 6241 6242"
  PREFIX=ambev4gated
  LOG=logs_ccinc_v3_ambe_v4gated.log
  ALL_STAGES="gate extract score"
  JOBS=2
fi
STAGE=$OUT/scored

WHAT="${ARGS[*]}"
[ "$WHAT" = "all" ] && WHAT="$ALL_STAGES"

mkdir -p "$OUT"
: > $LOG
echo "=== CCinc v3 — AmBe [$CAMPAIGN]  $(date) ===" | tee -a $LOG
echo "stages: $WHAT   runs: $RUNS   modes: $MODES" | tee -a $LOG
rc_all=0

# ────────────────────────────────────────────────────────────────────────────
# 1. Derive the IC gate
# ────────────────────────────────────────────────────────────────────────────
# `timestamp` in the diagnostics join IS eventTimeTank: verified on 6266, all
# 4,182 candidate timestamps match a BeamCluster eventTimeTank exactly, 2,171 of
# them passing IC. An empty or mismatched gate makes the extractor skip the run
# while reporting success, so the row count below is the thing to check.
#
# For v4gated there is nothing to derive: the gate IS the pipeline's own Stage-2
# candidate list. But it CANNOT be pointed at CANDDIR directly — build_gate_sets
# globs every EventAmBeNeutronCandidates_*.csv and unions the timestamps per run
# number, and AmBe2.0v4_gated and AmBe2.0v4_fprompt cover the SAME 20 runs. On run
# 6062 that union is 9,195 events against gated's 7,755 (+18.6%), and neither tag
# is a subset of the other — so the unfiltered directory gates on an event set that
# belongs to no campaign and silently breaks the closure against 181,636
# candidates. Stage a tag-filtered symlink directory instead.
if [[ " $WHAT " == *" gate "* ]]; then
  echo "" | tee -a $LOG
  echo "######## 1. gate  $(date) ########" | tee -a $LOG
  if [ "$CAMPAIGN" = "run6266" ]; then
    python make_gate_csvs_from_parquet.py --source step4 \
        --parquet "$STEP4" --out-dir "$OUT/gate_csvs" \
        --tag AmBe2.0v4_ic 2>&1 | tee -a $LOG
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc_all=1
  elif [ "$CAMPAIGN" = "v4special" ]; then
    rm -rf "$OUT/gate_csvs"; mkdir -p "$OUT/gate_csvs"
    for r in $RUNS; do
      python make_gate_csvs_from_parquet.py --source step4 \
          --parquet "$S4DIR/step4_joined_${r}.parquet" \
          --out-dir "$OUT/gate_csvs" --tag AmBe2.0v4_special 2>&1 | tee -a $LOG
      [ "${PIPESTATUS[0]}" -eq 0 ] || rc_all=1
    done
  elif [ "$CAMPAIGN" = "v4neutron" ]; then
    rm -rf "$OUT/gate_csvs"; mkdir -p "$OUT/gate_csvs"
    n=0
    for r in $RUNS; do
      src=""
      for tag in AmBe2.0v4_gated AmBe2.0v4_ext; do
        c=$CANDDIR/EventAmBeNeutronCandidates_${tag}_${r}.csv
        [ -e "$c" ] && src=$c
      done
      if [ -z "$src" ]; then
        echo "ERROR: no Stage-1 candidate CSV for run $r under either tag" | tee -a $LOG
        rc_all=1; continue
      fi
      ln -sfn "$src" "$OUT/gate_csvs/$(basename "$src")"
      ev=$(python -c "import pandas as pd,sys;print(pd.read_csv(sys.argv[1],usecols=['eventTankTime']).eventTankTime.nunique())" "$src")
      cl=$(( $(wc -l < "$src") - 1 ))
      echo "  run $r [$(basename "$src" | sed 's/EventAmBeNeutronCandidates_//;s/_[0-9]*\.csv//')]: $ev gated events, $cl Stage-2 candidates" | tee -a $LOG
      n=$((n+1))
    done
    echo "[gate] staged $n gate CSVs -> $OUT/gate_csvs" | tee -a $LOG
  else
    rm -rf "$OUT/gate_csvs"; mkdir -p "$OUT/gate_csvs"
    n=0
    for r in $RUNS; do
      src=$CANDDIR/EventAmBeNeutronCandidates_AmBe2.0v4_gated_${r}.csv
      if [ ! -e "$src" ]; then
        echo "ERROR: no candidate CSV for run $r" | tee -a $LOG; rc_all=1; continue
      fi
      ln -sfn "$src" "$OUT/gate_csvs/$(basename "$src")"
      ev=$(python -c "import pandas as pd,sys;print(pd.read_csv(sys.argv[1],usecols=['eventTankTime']).eventTankTime.nunique())" "$src")
      cl=$(( $(wc -l < "$src") - 1 ))
      echo "  run $r: $ev gated events, $cl Stage-2 candidates" | tee -a $LOG
      n=$((n+1))
    done
    echo "[gate] staged $n tag-filtered gate CSVs -> $OUT/gate_csvs" | tee -a $LOG
  fi
fi

# ────────────────────────────────────────────────────────────────────────────
# 2. ClusterFinder features, ungated and gated
# ────────────────────────────────────────────────────────────────────────────
# JOBS is 1 for run6266 (ONE input file, so extra workers buy nothing) and 2 for
# v4gated (20 files). One worker per FILE, and each holds its own event batch, so
# do not raise it further on an 11 GB node. load_file() reads in 5,000-event batches
# (analyze_optics_beamcluster_data.LOAD_BATCH_EVENTS) because reading this file
# whole is an out-of-memory kill on an 11 GB node.
if [[ " $WHAT " == *" extract "* ]]; then
  for r in $RUNS; do
    [ -e "$AMBEDIR/BeamCluster_${r}.root" ] || {
      echo "ERROR: missing BeamCluster_${r}.root" | tee -a $LOG; rc_all=1; }
  done
  # The extractor takes ONE positional input (file, glob or directory). For
  # v4gated we hand it the whole AmBe2.0v4 directory and let the gate select the
  # runs: with --gate-csv-dir it drops every run that has no gate CSV, printing
  # which ones. Since gate_csvs holds exactly the 20 tag-filtered runs, that is
  # precisely this campaign — and 6264/6265/6266/6270 are excluded by
  # construction rather than by a second list that could drift out of sync.
  [ "$CAMPAIGN" = "run6266" ] && INPUT=$AMBEDIR/BeamCluster_6266.root \
                              || INPUT=$AMBEDIR
  for MODE in $MODES; do
    echo "" | tee -a $LOG
    echo "######## 2. extract CF features [$MODE]  $(date) ########" | tee -a $LOG
    GATE_ARG=()
    [ "$MODE" = "gated" ] && GATE_ARG=(--gate-csv-dir "$OUT/gate_csvs")
    python extract_cf_features_beamcluster_data.py "$INPUT" \
        --run-name "${PREFIX}_${MODE}" --output-dir "$OUT" \
        "${GATE_ARG[@]}" \
        --geometry $GEO/FullTankPMTGeometry.csv \
        --offsets  $GEO/TankPMTTimingOffsets.csv \
        --jobs $JOBS 2>&1 | tee -a $LOG
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc_all=1
  done
fi

# ────────────────────────────────────────────────────────────────────────────
# 3. Score with the merged CF models
# ────────────────────────────────────────────────────────────────────────────
if [[ " $WHAT " == *" score "* ]]; then
  for MODE in $MODES; do
    DAT=$OUT/${PREFIX}_${MODE}__data_features_cf.parquet
    if [ ! -e "$DAT" ]; then
      echo "ERROR: missing $DAT — run the extract stage first" | tee -a $LOG
      rc_all=1; continue
    fi
    for STREAM in truthtag recotag; do
      RN=cc_neutrino_v3_${STREAM}
      SRC=$BASE/$RN/parquet
      PKL=$SRC/${RN}__mva_frozen__keepprompt__merged__cf.pkl
      KERAS=$SRC/${RN}__mva_frozen__keepprompt__merged__nn__cf.keras
      KERAS_EXPECTED=${RN}__mva_frozen__keepprompt__merged__nn.keras
      if [ ! -e "$PKL" ]; then
        echo "ERROR [$STREAM/$MODE]: missing $PKL" | tee -a $LOG
        rc_all=1; continue
      fi
      [ -e "$KERAS" ] || echo "WARN [$STREAM/$MODE]: no NN companion — only three \
scores will be produced" | tee -a $LOG

      D=$STAGE/${STREAM}_cf_${MODE}
      mkdir -p "$D"
      ln -sfn "$PKL" "$D/$(basename "$PKL")"
      [ -e "$KERAS" ] && ln -sfn "$KERAS" "$D/$KERAS_EXPECTED"
      ln -sfn "$DAT" "$D/${PREFIX}_${STREAM}_cf_${MODE}.parquet"

      echo "" | tee -a $LOG
      echo "######## 3. score ${STREAM}/cf [$MODE]  $(date) ########" | tee -a $LOG
      python mva_analysis.py \
          --score-data "$D/${PREFIX}_${STREAM}_cf_${MODE}.parquet" \
          --model "$D/$(basename "$PKL")" 2>&1 | tee -a $LOG
      rc=${PIPESTATUS[0]}
      echo "---- exit $rc for ${STREAM}/cf [$MODE] ----" | tee -a $LOG
      [ "$rc" -eq 0 ] || rc_all=1
    done
  done
fi

echo "" | tee -a $LOG
echo "=== done  $(date)  overall rc=$rc_all ===" | tee -a $LOG
# The two failure modes that report success if unchecked: features silently
# median-filled (name drift between MC and data), and a missing NN.
echo "=== sanity: absent features and NN presence ===" | tee -a $LOG
grep -aE "feature\(s\) absent|NN companion|nn_score" $LOG | tee -a /dev/null
echo "" | tee -a $LOG
echo "scored outputs:" | tee -a $LOG
ls -la $STAGE/*/*__scored.parquet 2>&1 | tee -a $LOG

exit $rc_all
