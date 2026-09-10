#!/usr/bin/env bash
# Run the full CC-inclusive productionv2 neutron pipeline end-to-end.
# Launch inside tmux and walk away:
#   tmux new-session -s ccinc
#   bash run_ccinc_productionv2.sh
#   # check progress: tail -f logs_ccinc_productionv2.log
#
# RESUME: set START_STAGE to skip already-completed stages (their outputs are
# reused from disk).
#   START_STAGE=1  skip Stage 0 (ROOT process); reuse __pulses.parquet
#   START_STAGE=2  skip Stage 0+1; reuse __cluster_features.parquet
#   START_STAGE=3  skip Stage 0+1+2; reuse trained MVA models, run Stage 3+4 only
# Stages: 0=process  1=features  2=MVA training (MC bkg only)  3=score AmBe
#         data + downstream  4=box-cut benchmarks.  (default START_STAGE=0)
# Stage-2/3/4 path variables are defined unconditionally so a START_STAGE=3
# resume can reference the MVA model/score paths written by an earlier run.
set -euo pipefail

START_STAGE="${START_STAGE:-0}"

# RUN selects which run_name (config + output dir + model files) to use.
# Default = the original productionv2. Override to run the prompt-inclusive
# iteration's downstream on its better model, e.g.:
#   RUN=cc_neutrino_productionv2_inclprompt MODEL_TAG=__keepprompt \
#     START_STAGE=3 bash run_ccinc_productionv2.sh
RUN="${RUN:-cc_neutrino_productionv2}"
MODEL_TAG="${MODEL_TAG:-}"   # e.g. __keepprompt for the inclprompt model files

REPO=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis
VENV=/exp/annie/app/users/dajana/myboy/bin/activate
CONFIG="configs/${RUN}.yaml"
LOG="${REPO}/logs_ccinc_productionv2.log"
OUT_ROOT=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output
GEO=/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/FullTankPMTGeometry.csv
OFFSETS=/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/TankPMTTimingOffsets.csv

cd "$REPO"
source "$VENV"
exec > >(tee -a "$LOG") 2>&1
echo "========================================================"
echo "  ccinc productionv2 pipeline  —  start $(date)  (START_STAGE=${START_STAGE})"
echo "========================================================"

# ── Stage 0: MC process (ROOT → per-hit parquet) ─────────────────────────
if [[ "${START_STAGE}" -le 0 ]]; then
  echo
  echo "--- Stage 0: ambe mc process ---"
  ambe mc process --config "$CONFIG"
else
  echo "--- Stage 0: SKIPPED (START_STAGE=${START_STAGE}) — reusing __pulses.parquet ---"
fi

# ── Stage 1: OPTICS features (frozen params ms=8 xi=0.1 t=25ns) ──────────
if [[ "${START_STAGE}" -le 1 ]]; then
  echo
  echo "--- Stage 1: ambe mc features ---"
  ambe mc features --config "$CONFIG"
else
  echo "--- Stage 1: SKIPPED (START_STAGE=${START_STAGE}) — reusing __cluster_features.parquet ---"
fi

# Path variables used by Stage 2 (write) AND Stage 3/4 (read) — define them
# UNCONDITIONALLY so a START_STAGE=3 resume can still reference them.
PKL_MCA="${OUT_ROOT}/${RUN}/parquet/${RUN}__mva_frozen${MODEL_TAG}.pkl"
MC_SCORES="${OUT_ROOT}/${RUN}/parquet/${RUN}__mva_scores${MODEL_TAG}.parquet"
MC_FEATURES="${OUT_ROOT}/${RUN}/parquet/${RUN}__cluster_features.parquet"
# Offbeam background (MC-B) intentionally NOT used — single MVA trained on MC
# internal background only (Stage 2a).

# ── Stage 2: Train MVA — signal vs MC internal background only, 1:1 ──────
# Offbeam background (former 2b/2c) intentionally dropped — single MC-bkg model.
# --max-ratio 1.0: 1:1 signal:background balance.
if [[ "${START_STAGE}" -le 2 ]]; then
  echo
  echo "--- Stage 2: MVA train (pure MC background, 1:1) ---"
  python mva_analysis.py \
      --config "$CONFIG" \
      --cc-only \
      --bkg-mode all \
      --max-ratio 1.0 \
      --save-model
else
  echo "--- Stage 2: SKIPPED (START_STAGE=${START_STAGE}) — reusing trained MVA model ---"
fi

# ── Pick the best classifier + derive its operating cut (adaptive) ───────
# Replaces the old hardcoded `rf_score` + stale `0.643`. pick_best_model_cut.py
# reads the MVA scores parquet, picks the highest-test-AUC classifier, and
# derives its score threshold at 80% MC signal efficiency. If a different model
# (gbt/xgb/nn) wins on a future dataset, the whole pipeline follows automatically.
echo
echo "--- Picking best MVA model + operating cut ---"
eval "$(python pick_best_model_cut.py --scores "$MC_SCORES" --sig-eff 0.80)"
: "${BEST_SCORE_COL:=rf_score}"   # fallback if the picker failed
: "${BEST_SCORE_CUT:=0.5}"
echo "[pipeline] using BEST_SCORE_COL=${BEST_SCORE_COL}  BEST_SCORE_CUT=${BEST_SCORE_CUT}"

# ── Stage 3: Score AmBe data + downstream per MVA streamline ─────────────
OPTICS_DATA="${OUT_ROOT}/ambe_data/ambe_all__data_features_optics.parquet"
CF_DATA="${OUT_ROOT}/ambe_data/ambe_all__data_features_cf.parquet"

# Declare streamlines: "label:data_parquet:pkl:mc_scores_parquet:mc_method"
# NOTE: --score-data always writes <input_stem>__scored.parquet next to the input.
# Both optics streamlines (and both CF streamlines) share one input parquet, so we
# rename the scored output to a per-streamline path immediately after scoring to
# avoid the second streamline overwriting the first.
STREAMLINES=(
  "ccinc_optics_mcbkg:${OPTICS_DATA}:${PKL_MCA}:${MC_SCORES}:optics"
  "ccinc_cf_mcbkg:${CF_DATA}:${PKL_MCA}:${MC_SCORES}:clusterfinder"
)

for ENTRY in "${STREAMLINES[@]}"; do
  IFS=: read -r TAG DATA_PAR PKL MC_PAR MC_METHOD <<< "$ENTRY"
  echo
  echo "=== Streamline: ${TAG} ==="

  OUT_S="${OUT_ROOT}/ambe_data/${TAG}"
  mkdir -p "$OUT_S"

  # Stage 3a — score data, then rename to a per-streamline scored parquet
  echo "--- ${TAG}: score AmBe data ---"
  python mva_analysis.py \
      --score-data "$DATA_PAR" \
      --model      "$PKL"
  DATA_STEM="$(basename "${DATA_PAR%.parquet}")"
  DEFAULT_SCORED="$(dirname "$DATA_PAR")/${DATA_STEM}__scored.parquet"
  SCORED="${OUT_S}/${TAG}__scored.parquet"
  mv -f "$DEFAULT_SCORED" "$SCORED"

  # Stage 3b — per-run neutron rate summary (best model; sig-eff derives the cut)
  echo "--- ${TAG}: neutron rate summary (${BEST_SCORE_COL}) ---"
  python summarize_ambe_neutrons.py \
      --scored    "$SCORED" \
      --mc-scores "$MC_PAR" \
      --score-col "$BEST_SCORE_COL" \
      --sig-eff   0.80 \
      --out       "${OUT_S}/rate_best"

  # Stage 3c — multiplicity + score PDF (best model + derived cut)
  echo "--- ${TAG}: multiplicity PDF (${BEST_SCORE_COL} @ ${BEST_SCORE_CUT}) ---"
  python plot_ambe_neutron_multiplicity.py \
      --scored      "$SCORED" \
      --mc-features "$MC_FEATURES" \
      --score-col   "$BEST_SCORE_COL" \
      --score-cut   "$BEST_SCORE_CUT" \
      --out         "${OUT_S}/${TAG}__multiplicity.pdf"

  # Stage 3d — capture-time fit (lmfit only, skip slow pymc; best model + derived cut)
  echo "--- ${TAG}: capture-time fit (${BEST_SCORE_COL} @ ${BEST_SCORE_CUT}) ---"
  python capture_time_stageAB_from_parquet.py \
      --scored    "$SCORED" \
      --score-col "$BEST_SCORE_COL" \
      --score-cut "$BEST_SCORE_CUT" \
      --out-dir   "${OUT_S}/capture_stageAB" \
      --methods   lmfit

  # Stage 3e — data/MC KS comparison  (SKIPPED per request — pure leaf diagnostic,
  # nothing downstream depends on it. Uncomment to restore.)
  # echo "--- ${TAG}: data/MC comparison ---"
  # python analyze_data_mc_comparison.py \
  #     --mc-parquet  "$(dirname "$MC_FEATURES")" \
  #     --run-name    cc_neutrino_productionv2 \
  #     --mc-method   "$MC_METHOD" \
  #     --output-dir  "${OUT_S}/compare_selections" \
  #     --geo         "$GEO" \
  #     --offsets     "$OFFSETS"

done

# ── Stage 4: Box-cut streamlines (S3 OPTICS+cuts, S6 CF+cuts) ────────────
# compare_selections_capture.py contrasts the MVA selection (--sel1, scored)
# against the box-cut selection (--sel2, raw CF). Reuse the mcbkg scored
# parquets written per-streamline in Stage 3 as the MVA side.
OPTICS_SCORED_MCA="${OUT_ROOT}/ambe_data/ccinc_optics_mcbkg/ccinc_optics_mcbkg__scored.parquet"
CF_SCORED_MCA="${OUT_ROOT}/ambe_data/ccinc_cf_mcbkg/ccinc_cf_mcbkg__scored.parquet"

echo
echo "--- Stage 4 S3 (ccinc_optics_boxcuts): OPTICS + box cuts ---"
OUT_S3="${OUT_ROOT}/ambe_data/ccinc_optics_boxcuts"
python benchmark_selection_2x2.py \
    --features  "$MC_FEATURES" \
    --model     "$PKL_MCA" \
    --score-col "$BEST_SCORE_COL" \
    --score-cut "$BEST_SCORE_CUT" \
    --out-dir   "${OUT_S3}/benchmark"
# compare_selections_capture SKIPPED — stale script: its load_sel2 expects raw-CF
# columns (clusterPE/clusterTime/sourceX) but $CF_DATA is the feature parquet
# (pe_total/t_mean/...). Leaf step; nothing downstream needs it. Uncomment +
# remap load_sel2 to restore.
# python compare_selections_capture.py \
#     --sel1      "$OPTICS_SCORED_MCA" \
#     --sel2      "$CF_DATA" \
#     --score-col rf_score \
#     --out-dir   "${OUT_S3}/capture_compare"

echo
echo "--- Stage 4 S6 (ccinc_cf_boxcuts): CF + box cuts ---"
OUT_S6="${OUT_ROOT}/ambe_data/ccinc_cf_boxcuts"
python benchmark_selection_2x2.py \
    --features  "$MC_FEATURES" \
    --model     "$PKL_MCA" \
    --score-col "$BEST_SCORE_COL" \
    --score-cut "$BEST_SCORE_CUT" \
    --out-dir   "${OUT_S6}/benchmark"
# compare_selections_capture SKIPPED — see note in S3 block above (stale schema).
# python compare_selections_capture.py \
#     --sel1      "$CF_SCORED_MCA" \
#     --sel2      "$CF_DATA" \
#     --score-col rf_score \
#     --out-dir   "${OUT_S6}/capture_compare"

echo
echo "========================================================"
echo "  all done  —  $(date)"
echo "  log: $LOG"
echo "========================================================"
