#!/usr/bin/env bash
# ============================================================================
# ISOLATED SANDBOX: two-streamline MVA (train OPTICS *and* ClusterFinder models)
# ============================================================================
# Variant of run_ccinc_xi02_compcut.sh that implements the originally-intended
# design: a SEPARATE frozen MVA per clustering method, instead of scoring CF
# clusters with the OPTICS-trained model.
#
# SAFETY MODEL (why this cannot harm the fully-fledged analysis):
#   * NEW run_name  = cc_neutrino_xi02_compcut_cftrain  -> ALL Stage-2+ outputs
#     land under  ${OUT_ROOT}/cc_neutrino_xi02_compcut_cftrain/  (a fresh folder).
#   * Stage 0/1 are NEVER run here (START_STAGE forced >= 2). The expensive
#     __pulses.parquet / __cluster_features.parquet are reused from the existing
#     cc_neutrino_xi02_compcut run by SYMLINK (read-only; originals untouched).
#   * Stage-3 AmBe data streamline dirs are suffixed *_cftrain so they never
#     collide with the existing ccinc_xi02cc_{optics,cf}_mcbkg dirs.
#   * Stage-4 benchmark dirs are suffixed *_cftrain too.
#   The original cc_neutrino_xi02_compcut/ tree is a pure read-only input here.
#
# Run in a detached tmux session and walk away:
#   cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
#   tmux new-session -s cftrain
#   START_STAGE=2 bash run_ccinc_xi02_compcut_cftrain.sh
#   # detach: Ctrl-b d     reattach: tmux attach -t cftrain
#   # watch progress from anywhere:
#   #   tail -f /exp/annie/app/users/dajana/AmBeNeutronsAnalysis/logs_cc_neutrino_xi02_compcut_cftrain.log
#
# What it does (Stage 0/1 are NOT run — reused via symlink):
#   Stage 2a : REUSES the already-trained sandbox OPTICS model+scores (skips if present)
#   Stage 2b : trains the ClusterFinder model  (-> ...__cf.pkl, ...__cf.parquet scores)
#   Stage 3  : scores AmBe data — OPTICS data by OPTICS model, CF data by CF model
#   Stage 4  : box-cut(80/0.45/9)-vs-MVA benchmark, each method with its OWN model
# Resumable: re-running skips any model+scores already on disk (idempotent).
# ============================================================================
set -euo pipefail

# Stage 0/1 must NOT run here — they would recompute into the new run_name and
# waste hours. Force at least Stage 2; default to 2.
START_STAGE="${START_STAGE:-2}"
if [[ "${START_STAGE}" -lt 2 ]]; then
  echo "ERROR: this sandbox script reuses Stage-0/1 from cc_neutrino_xi02_compcut."
  echo "       Run with START_STAGE>=2 (Stage 0/1 are symlinked, not recomputed)."
  exit 1
fi

# ---- identities ----
SRC_RUN="cc_neutrino_xi02_compcut"                 # where Stage-0/1 already live
RUN="${RUN:-cc_neutrino_xi02_compcut_cftrain}"     # where NEW outputs go
MODEL_TAG="${MODEL_TAG:-__keepprompt}"

REPO=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis
VENV=/exp/annie/app/users/dajana/myboy/bin/activate
CONFIG="configs/${RUN}.yaml"
LOG="${REPO}/logs_${RUN}.log"
OUT_ROOT=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output

# Composition cut thresholds (Stages 3 + 4)
COMP_N_NEUTRON_MAX=10
COMP_FRAC_NONNEUTRON_MIN=0.40

cd "$REPO"
source "$VENV"
exec > >(tee -a "$LOG") 2>&1
echo "========================================================"
echo "  ccinc xi02_compcut CFTRAIN sandbox  —  start $(date)  (START_STAGE=${START_STAGE})"
echo "  NEW run_name : ${RUN}"
echo "  reuses Stage-0/1 from : ${SRC_RUN} (symlink)"
echo "  trains TWO models: OPTICS + ClusterFinder (separate)"
echo "========================================================"

NEW_PARQ="${OUT_ROOT}/${RUN}/parquet"
SRC_PARQ="${OUT_ROOT}/${SRC_RUN}/parquet"
mkdir -p "$NEW_PARQ"

# ── Pre-flight: symlink reusable Stage-0/1 outputs into the new run folder ──
echo
echo "--- Pre-flight: symlinking reusable Stage-0/1 parquet from ${SRC_RUN} ---"
for f in __pulses.parquet __cluster_features.parquet; do
  SRC="${SRC_PARQ}/${SRC_RUN}${f}"
  DST="${NEW_PARQ}/${RUN}${f}"
  if [[ ! -e "$SRC" ]]; then echo "ERROR: missing reusable input $SRC"; exit 1; fi
  ln -sfn "$SRC" "$DST"
  echo "  linked ${DST##*/}  ->  ${SRC}"
done

# ---- path variables (Stage 2 writes / Stage 3+4 read) ----
PKL_OPTICS="${NEW_PARQ}/${RUN}__mva_frozen${MODEL_TAG}.pkl"          # OPTICS model (default name)
PKL_CF="${NEW_PARQ}/${RUN}__mva_frozen${MODEL_TAG}__cf.pkl"          # CF model (explicit name)
# NOTE: training ALWAYS writes scores to ${RUN}__mva_scores${MODEL_TAG}.parquet
# (the DEFAULT_SCORES name below). Both per-method targets MUST differ from that
# default, or the post-train `mv` becomes a no-op (mv file onto itself) and the
# next training silently overwrites it. So both get explicit method suffixes.
MC_SCORES_OPTICS="${NEW_PARQ}/${RUN}__mva_scores${MODEL_TAG}__optics.parquet"
MC_SCORES_CF="${NEW_PARQ}/${RUN}__mva_scores${MODEL_TAG}__cf.parquet"
MC_FEATURES="${NEW_PARQ}/${RUN}__cluster_features.parquet"

# ── Stage 2: Train BOTH MVA streamlines cleanly in the sandbox ────────────
# CRITICAL: mva_analysis.py ALWAYS writes the per-cluster MC scores to the
# config-derived name  ${RUN}__mva_scores${MODEL_TAG}.parquet  — it ignores the
# --save-model path for the SCORES file. Since both trainings share ${RUN}, the
# 2nd training would overwrite the 1st's scores. So after EACH training we
# immediately MOVE the default-named scores parquet to a method-specific name.
# (This is exactly the bug that, via a symlink, clobbered the original OPTICS
# scores on the first attempt — here there are no symlinks to write through.)
# mva_analysis.py writes these THREE artifacts to method-AGNOSTIC default names —
# so OPTICS and CF (same run_name) would clobber each other. After each training
# we rename all three to a per-method name so BOTH streamlines' models, scores,
# AND plots/CSVs (RF signal-vs-bkg discriminator + NN loss/accuracy history) are
# preserved side by side.
DEFAULT_SCORES="${NEW_PARQ}/${RUN}__mva_scores${MODEL_TAG}.parquet"
DEFAULT_PLOT="${OUT_ROOT}/${RUN}/plots/${RUN}__mva${MODEL_TAG}.pdf"
DEFAULT_CSV="${OUT_ROOT}/${RUN}/csv/${RUN}__mva_summary${MODEL_TAG}.csv"

# Defensive move: error if source missing; skip if src==dst (never silently no-op).
preserve_artifacts () {  # $1=method-suffix (optics|cf)  $2=scores-dst
  local suf="$1" sdst="$2"
  if [[ ! -e "$DEFAULT_SCORES" ]]; then echo "ERROR: training wrote no scores at $DEFAULT_SCORES"; exit 1; fi
  if [[ "$DEFAULT_SCORES" -ef "$sdst" ]]; then echo "ERROR: scores dst equals default name ($sdst) — would no-op"; exit 1; fi
  mv -f "$DEFAULT_SCORES" "$sdst";  echo "[pipeline] scores -> ${sdst##*/}"
  # plot PDF (contains the score-distribution discriminator + NN history pages)
  if [[ -e "$DEFAULT_PLOT" ]]; then
    mv -f "$DEFAULT_PLOT" "${DEFAULT_PLOT%.pdf}__${suf}.pdf"
    echo "[pipeline] plot   -> ${RUN}__mva${MODEL_TAG}__${suf}.pdf  (ROC, sig-vs-bkg score dist, NN loss/auc history)"
  fi
  # feature-importance / AUC summary CSV
  if [[ -e "$DEFAULT_CSV" ]]; then
    mv -f "$DEFAULT_CSV" "${DEFAULT_CSV%.csv}__${suf}.csv"
    echo "[pipeline] csv    -> ${RUN}__mva_summary${MODEL_TAG}__${suf}.csv"
  fi
}

echo
echo "--- Stage 2a: MVA train (method=optics) ---"
if [[ -e "$PKL_OPTICS" && -e "$MC_SCORES_OPTICS" ]]; then
  echo "  reusing existing sandbox OPTICS model + scores (already trained) — skip"
else
  python mva_analysis.py \
      --config "$CONFIG" --method optics \
      --cc-only --bkg-mode all --max-ratio 1.0 --keep-prompt-bkg \
      --save-model "$PKL_OPTICS"
  preserve_artifacts optics "$MC_SCORES_OPTICS"   # rename BEFORE CF training overwrites defaults
fi

echo
echo "--- Stage 2b: MVA train (method=clusterfinder) ---"
if [[ -e "$PKL_CF" && -e "$MC_SCORES_CF" ]]; then
  echo "  reusing existing sandbox CF model + scores — skip"
else
  python mva_analysis.py \
      --config "$CONFIG" --method clusterfinder \
      --cc-only --bkg-mode all --max-ratio 1.0 --keep-prompt-bkg \
      --save-model "$PKL_CF"
  preserve_artifacts cf "$MC_SCORES_CF"
fi

# ── Pick best classifier + operating cut, PER method (anchored on each MC) ──
echo
echo "--- Picking best model + cut (OPTICS) ---"
eval "$(python pick_best_model_cut.py --scores "$MC_SCORES_OPTICS" --sig-eff 0.80)"
: "${BEST_SCORE_COL:=rf_score}"; : "${BEST_SCORE_CUT:=0.5}"
OPT_SCORE_COL="$BEST_SCORE_COL"; OPT_SCORE_CUT="$BEST_SCORE_CUT"
echo "[pipeline] OPTICS: ${OPT_SCORE_COL} @ ${OPT_SCORE_CUT}"

echo "--- Picking best model + cut (CF) ---"
eval "$(python pick_best_model_cut.py --scores "$MC_SCORES_CF" --sig-eff 0.80)"
: "${BEST_SCORE_COL:=rf_score}"; : "${BEST_SCORE_CUT:=0.5}"
CF_SCORE_COL="$BEST_SCORE_COL"; CF_SCORE_CUT="$BEST_SCORE_CUT"
echo "[pipeline] CF: ${CF_SCORE_COL} @ ${CF_SCORE_CUT}"

# ── Stage 3: Score AmBe data with EACH method's OWN model ─────────────────
OPTICS_DATA="${OUT_ROOT}/ambe_data/ambe_all__data_features_optics.parquet"
CF_DATA="${OUT_ROOT}/ambe_data/ambe_all__data_features_cf.parquet"

MC_FEAT_CC="${NEW_PARQ}/${RUN}__cluster_features_compcut.parquet"
echo
echo "--- Applying composition cut to MC features (Stage 4 benchmark input) ---"
python - <<PYEOF
import pandas as pd
df = pd.read_parquet("${MC_FEATURES}")
keep = ~((df["n_neutron"] <= ${COMP_N_NEUTRON_MAX}) & (df["frac_nonneutron"] >= ${COMP_FRAC_NONNEUTRON_MIN}))
df[keep].to_parquet("${MC_FEAT_CC}", index=False)
print(f"MC composition cut: {(~keep).sum()} removed, {keep.sum()} kept")
PYEOF

# Streamlines: "label:data_parquet:pkl:mc_scores:mc_method:score_col:score_cut"
STREAMLINES=(
  "ccinc_xi02cc_optics_cftrain:${OPTICS_DATA}:${PKL_OPTICS}:${MC_SCORES_OPTICS}:optics:${OPT_SCORE_COL}:${OPT_SCORE_CUT}"
  "ccinc_xi02cc_cf_cftrain:${CF_DATA}:${PKL_CF}:${MC_SCORES_CF}:clusterfinder:${CF_SCORE_COL}:${CF_SCORE_CUT}"
)

for ENTRY in "${STREAMLINES[@]}"; do
  IFS=: read -r TAG DATA_PAR PKL MC_PAR MC_METHOD SCOL SCUT <<< "$ENTRY"
  echo
  echo "=== Streamline: ${TAG}  (model=${PKL##*/}, ${SCOL} @ ${SCUT}) ==="
  OUT_S="${OUT_ROOT}/ambe_data/${TAG}"
  mkdir -p "$OUT_S"

  COMP_CUT_DATA="${OUT_S}/${TAG}__data_compcut.parquet"
  echo "--- ${TAG}: composition cut on data (skipped if truth cols absent) ---"
  python - <<PYEOF
import pandas as pd, shutil
df = pd.read_parquet("${DATA_PAR}")
if "n_neutron" in df.columns and "frac_nonneutron" in df.columns:
    keep = ~((df["n_neutron"] <= ${COMP_N_NEUTRON_MAX}) & (df["frac_nonneutron"] >= ${COMP_FRAC_NONNEUTRON_MIN}))
    df[keep].to_parquet("${COMP_CUT_DATA}", index=False)
    print(f"Data compcut [${TAG}]: {(~keep).sum()} removed, {keep.sum()} kept")
else:
    shutil.copy("${DATA_PAR}", "${COMP_CUT_DATA}")
    print(f"Data compcut [${TAG}]: SKIPPED — no truth cols ({len(df)} passed through)")
PYEOF

  echo "--- ${TAG}: score AmBe data with its OWN model ---"
  python mva_analysis.py --score-data "$COMP_CUT_DATA" --model "$PKL"
  DATA_STEM="$(basename "${COMP_CUT_DATA%.parquet}")"
  mv -f "$(dirname "$COMP_CUT_DATA")/${DATA_STEM}__scored.parquet" "${OUT_S}/${TAG}__scored.parquet"
  SCORED="${OUT_S}/${TAG}__scored.parquet"

  echo "--- ${TAG}: neutron rate summary (${SCOL}) ---"
  python summarize_ambe_neutrons.py --scored "$SCORED" --mc-scores "$MC_PAR" \
      --score-col "$SCOL" --sig-eff 0.80 --out "${OUT_S}/rate_best"

  echo "--- ${TAG}: multiplicity PDF (${SCOL} @ ${SCUT}) ---"
  python plot_ambe_neutron_multiplicity.py --scored "$SCORED" --mc-features "$MC_FEATURES" \
      --score-col "$SCOL" --score-cut "$SCUT" --out "${OUT_S}/${TAG}__multiplicity.pdf"

  echo "--- ${TAG}: capture-time fit (${SCOL} @ ${SCUT}) ---"
  python capture_time_stageAB_from_parquet.py --scored "$SCORED" \
      --score-col "$SCOL" --score-cut "$SCUT" --out-dir "${OUT_S}/capture_stageAB" --methods lmfit
done

# ── Stage 4: benchmark each method with ITS OWN model ─────────────────────
echo
echo "--- Stage 4 (ccinc_xi02cc_optics_boxcuts_cftrain): OPTICS + own model ---"
python benchmark_selection_2x2.py --features "$MC_FEAT_CC" --model "$PKL_OPTICS" \
    --score-col "$OPT_SCORE_COL" --score-cut "$OPT_SCORE_CUT" \
    --out-dir "${OUT_ROOT}/ambe_data/ccinc_xi02cc_optics_boxcuts_cftrain/benchmark"

echo
echo "--- Stage 4 (ccinc_xi02cc_cf_boxcuts_cftrain): CF + own CF model ---"
python benchmark_selection_2x2.py --features "$MC_FEAT_CC" --model "$PKL_CF" \
    --score-col "$CF_SCORE_COL" --score-cut "$CF_SCORE_CUT" \
    --out-dir "${OUT_ROOT}/ambe_data/ccinc_xi02cc_cf_boxcuts_cftrain/benchmark"

echo
echo "========================================================"
echo "  CFTRAIN sandbox done  —  $(date)"
echo "  outputs under: ${OUT_ROOT}/${RUN}/  and  ambe_data/ccinc_xi02cc_*_cftrain/"
echo "  log: $LOG"
echo "========================================================"
