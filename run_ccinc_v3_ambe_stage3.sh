#!/usr/bin/env bash
#
# run_ccinc_v3_ambe_stage3.sh — Stage 3 of the CCinc v3 campaign: apply the four
# merged tank+world models to real AmBe data.
#
# Deliberately never run before now (world report §6, open item 6). Nothing is
# trained here: mva_analysis.py --score-data loads a frozen artifact, rebuilds the
# feature matrix in the stored order with the stored MC training medians, and applies
# RF/GBT/XGB plus the companion NN. Verified beforehand: all 31 (truth-tag) / 21
# (reco-tag) features the models expect are present in the AmBe features parquets, so
# nothing is median-filled wholesale.
#
# Each model is applied ONLY to its own clustering method's clusters — an OPTICS-trained
# model on OPTICS clusters — following score_real_data.sh.
#
# TWO CLOBBERING HAZARDS, both handled by staging each configuration in its own
# directory. Do not "simplify" this away.
#
#   1. The output name is derived from the INPUT name (<stem>__scored.parquet). All four
#      configurations share two input files, so scoring them in place would have the
#      four runs overwrite each other's results — and
#      ambe_all__data_features_optics_forscore__scored.parquet already exists from the
#      June campaign, so the first run would also destroy that.
#
#   2. Every frozen .pkl stores its NN companion filename as it was BEFORE
#      run_world_merged.sh renamed the artifacts to __merged__{optics,cf}. The stored
#      name (<run>__mva_frozen__keepprompt__merged__nn.keras) no longer exists on disk,
#      so score_data() prints "WARN: NN companion file missing" and silently returns
#      three scores instead of four. Both methods' pkl expect the SAME keras basename in
#      the same directory, so they cannot be fixed in place at once. Staging gives each
#      configuration a private directory where the correct keras file carries the name
#      that configuration's pkl is looking for.
#
# Symlinks throughout — the frozen models are ~250 MB each and the data ~55 MB; nothing
# is copied and nothing under cc_neutrino_v3_*/ or ambe_data/ is modified.
#
# Usage (no silent defaults — name what you want):
#   bash run_ccinc_v3_ambe_stage3.sh all
#   bash run_ccinc_v3_ambe_stage3.sh truthtag/optics recotag/cf
#
set -uo pipefail

cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate
export PYTHONUNBUFFERED=1

BASE=/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output
AMBE=$BASE/ambe_data
STAGE=$BASE/ccinc_v3_ambe_stage3
LOG=logs_ccinc_v3_ambe_stage3.log

# The _forscore variants are the ones to use: they are the same clusters plus
# clusterTime_earliest, which the capture-time fit needs. Note
# ambe_all__data_features_clusterfinder.parquet (without _forscore) is EMPTY (0 rows) —
# picking it would score nothing and report success.
declare -A DATA=(
  [optics]=$AMBE/ambe_all__data_features_optics_forscore.parquet
  [cf]=$AMBE/ambe_all__data_features_clusterfinder_forscore.parquet
)

if [ $# -eq 0 ]; then
  echo "Usage: $0 all | <streamline>/<optics|cf> [...]" >&2
  exit 1
fi

if [ "${1:-}" = "all" ]; then
  CONFIGS="truthtag/optics truthtag/cf recotag/optics recotag/cf"
else
  CONFIGS="$*"
fi

: > $LOG
echo "=== CCinc v3 Stage 3 — AmBe scoring  $(date) ===" | tee -a $LOG
echo "configurations: $CONFIGS" | tee -a $LOG

rc_all=0
for CFG in $CONFIGS; do
  STREAM="${CFG%%/*}"
  MSFX="${CFG##*/}"
  case "$STREAM" in truthtag|recotag) ;; *)
    echo "ERROR: unknown streamline '$STREAM' in '$CFG'" | tee -a $LOG; exit 1;; esac
  case "$MSFX" in optics|cf) ;; *)
    echo "ERROR: method must be optics or cf, got '$MSFX' in '$CFG'" | tee -a $LOG
    exit 1;; esac

  RN=cc_neutrino_v3_${STREAM}
  SRC=$BASE/$RN/parquet
  PKL=$SRC/${RN}__mva_frozen__keepprompt__merged__${MSFX}.pkl
  KERAS=$SRC/${RN}__mva_frozen__keepprompt__merged__nn__${MSFX}.keras
  # The name the pkl's stored metadata will look for, in the model's own directory.
  KERAS_EXPECTED=${RN}__mva_frozen__keepprompt__merged__nn.keras
  DAT=${DATA[$MSFX]}

  for f in "$PKL" "$DAT"; do
    if [ ! -e "$f" ]; then
      echo "ERROR [$CFG]: missing $f" | tee -a $LOG
      rc_all=1; continue 2
    fi
  done
  [ -e "$KERAS" ] || echo "WARN [$CFG]: no NN companion at $(basename "$KERAS") — \
only three scores will be produced" | tee -a $LOG

  D=$STAGE/${STREAM}_${MSFX}
  mkdir -p "$D"
  ln -sfn "$PKL" "$D/$(basename "$PKL")"
  [ -e "$KERAS" ] && ln -sfn "$KERAS" "$D/$KERAS_EXPECTED"
  ln -sfn "$DAT" "$D/ambe_${STREAM}_${MSFX}.parquet"

  echo "" | tee -a $LOG
  echo "################ SCORE ${CFG}  $(date) ################" | tee -a $LOG
  echo "  model: $(basename "$PKL")" | tee -a $LOG
  echo "  data : $(basename "$DAT")" | tee -a $LOG
  echo "  stage: $D" | tee -a $LOG
  python mva_analysis.py \
      --score-data "$D/ambe_${STREAM}_${MSFX}.parquet" \
      --model "$D/$(basename "$PKL")" 2>&1 | tee -a $LOG
  rc=${PIPESTATUS[0]}
  echo "---- exit $rc for ${CFG} $(date) ----" | tee -a $LOG
  [ "$rc" -eq 0 ] || rc_all=1
done

echo "" | tee -a $LOG
echo "=== Stage 3 done  $(date)  overall rc=$rc_all ===" | tee -a $LOG

# The two failure modes that report success if unchecked: features silently
# median-filled (feature-name drift between MC and data), and a missing NN.
echo "" | tee -a $LOG
echo "=== sanity: features absent from the data, and NN presence ===" | tee -a $LOG
grep -aE "feature\(s\) absent|NN companion|nn_score|frac>0.5" $LOG | tee -a /dev/null
echo "" | tee -a $LOG
echo "scored outputs:" | tee -a $LOG
ls -la $STAGE/*/*__scored.parquet 2>&1 | tee -a $LOG

exit $rc_all
