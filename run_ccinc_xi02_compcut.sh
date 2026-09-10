#!/usr/bin/env bash
# Run the CC-inclusive xi=0.02 + composition-cut neutron pipeline end-to-end.
# Variant of run_ccinc_productionv2.sh with:
#   - OPTICS ms=8, xi=0.02, t_unit=25 ns  (tighter than productionv2 xi=0.10)
#   - Composition cut applied before MVA scoring:
#       reject cluster if n_neutron <= 10 AND frac_nonneutron >= 0.40
#   - --keep-prompt-bkg in Stage 2 (same as inclprompt; gives AUC ~0.718)
#
# Launch inside tmux and walk away:
#   tmux new-session -s xi02cc
#   bash run_ccinc_xi02_compcut.sh
#   # check progress: tail -f logs_<RUN>.log   (default logs_cc_neutrino_xi02_compcut.log)
#
# RESUME: set START_STAGE to skip already-completed stages.
#   START_STAGE=1  skip Stage 0 (ROOT process); reuse __pulses.parquet
#   START_STAGE=2  skip Stage 0+1; reuse __cluster_features.parquet
#   START_STAGE=3  skip Stage 0+1+2; reuse trained MVA models, run Stage 3+4 only
# Stages: 0=process  1=features  1b=background composition report
#         2=MVA training  3=score AmBe data + downstream
#         4=box-cut benchmarks.  (default START_STAGE=0)
#
# ENV KNOBS (every default reproduces the original xi02_compcut behaviour):
#   RUN=<name>          run_name AND config basename (configs/<RUN>.yaml)
#   STOP_STAGE=N        last stage to run, default 4. Use STOP_STAGE=2 for an
#                       MC-only campaign: process, features, background report,
#                       MVA training — and skip Stage 3/4, which are AmBe-data
#                       products and have nothing to do with the MC sample.
#   METHODS="a b"       clustering methods to TRAIN at Stage 2, space-separated.
#                       Default "optics" = one model, historical behaviour.
#                       METHODS="optics clusterfinder" trains one model per
#                       method and preserves both sets of artifacts (see
#                       preserve_artifacts below — mva_analysis.py's mode_tag
#                       does NOT encode --method, so without this the second
#                       training silently overwrites the first).
#   TAG_PREFIX=<pfx>    prefix for the Stage-3/4 AmBe streamline dir names,
#                       default ccinc_xi02cc. Change it whenever RUN changes or
#                       two campaigns will write into the same ambe_data dirs.
#   LOG=<path>          log file, default logs_<RUN>.log
#   REUSE_HITS=<path>   Stage 0 copies hits from this existing __pulses.parquet and
#                       only recomputes cc_pass for this config's cc_stream, instead
#                       of re-reading every ROOT file. Use for a SECOND selection
#                       streamline over input files already processed once: the
#                       per-hit content is stream-independent (all hits are written
#                       regardless of cc_pass), so only cc_pass differs. Saves hours.
#                       Unset = full ROOT pass, as before.
#
# Example — productionv3 tank/fmvmrd, CCinc+MRD+FMV, both methods, MC only:
#   RUN=cc_neutrino_v3_mrdfmv STOP_STAGE=2 METHODS="optics clusterfinder" \
#     bash run_ccinc_xi02_compcut.sh
#
# NOTE: Stage 0 is identical to productionv2. If disk / time is tight, symlink
# the pulses parquet and set START_STAGE=1:
#   ln -s ../cc_neutrino_productionv2/parquet/cc_neutrino_productionv2__pulses.parquet \
#         ${OUT_ROOT}/cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__pulses.parquet
#   START_STAGE=1 bash run_ccinc_xi02_compcut.sh
set -euo pipefail

START_STAGE="${START_STAGE:-0}"
STOP_STAGE="${STOP_STAGE:-4}"

RUN="${RUN:-cc_neutrino_xi02_compcut}"
MODEL_TAG="${MODEL_TAG:-__keepprompt}"
METHODS="${METHODS:-optics}"
TAG_PREFIX="${TAG_PREFIX:-ccinc_xi02cc}"

REPO=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis
VENV=/exp/annie/app/users/dajana/myboy/bin/activate
CONFIG="configs/${RUN}.yaml"
LOG="${LOG:-${REPO}/logs_${RUN}.log}"
OUT_ROOT=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output
GEO=/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/FullTankPMTGeometry.csv
OFFSETS=/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/TankPMTTimingOffsets.csv

# Composition cut thresholds applied before MVA scoring (Stages 3 + 4)
COMP_N_NEUTRON_MAX=10
COMP_FRAC_NONNEUTRON_MIN=0.40

cd "$REPO"
source "$VENV"
# stdout below is a pipe into tee, so Python block-buffers it and per-file progress
# lines sit invisible in an 8 KB buffer for minutes at a time. On a multi-hour Stage 0
# that is indistinguishable from a hang.
export PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG") 2>&1
echo "========================================================"
echo "  ccinc xi02_compcut pipeline  —  start $(date)"
echo "  RUN=${RUN}  CONFIG=${CONFIG}"
echo "  START_STAGE=${START_STAGE}  STOP_STAGE=${STOP_STAGE}"
echo "  METHODS='${METHODS}'  TAG_PREFIX=${TAG_PREFIX}"
echo "  OPTICS: ms=8, xi=0.02, t_unit=25ns"
echo "  Composition cut: n_neutron <= ${COMP_N_NEUTRON_MAX}  AND  frac_nonneutron >= ${COMP_FRAC_NONNEUTRON_MIN}"
echo "========================================================"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config $CONFIG not found (RUN=${RUN} must match configs/<RUN>.yaml)"
  exit 1
fi

# ── Stage 0: MC process (ROOT → per-hit parquet) ─────────────────────────
if [[ "${START_STAGE}" -le 0 ]]; then
  echo
  if [[ -n "${REUSE_HITS:-}" ]]; then
    echo "--- Stage 0: ambe mc process --reuse-hits (cc_pass only) ---"
    echo "    hit source: ${REUSE_HITS}"
    if [[ ! -e "${REUSE_HITS}" ]]; then
      echo "ERROR: REUSE_HITS=${REUSE_HITS} does not exist"; exit 1
    fi
    ambe mc process --config "$CONFIG" --reuse-hits "${REUSE_HITS}"
  else
    echo "--- Stage 0: ambe mc process ---"
    ambe mc process --config "$CONFIG"
  fi
else
  echo "--- Stage 0: SKIPPED (START_STAGE=${START_STAGE}) — reusing __pulses.parquet ---"
fi

# ── Stage 1: OPTICS features (ms=8 xi=0.02 t=25ns) ───────────────────────
if [[ "${START_STAGE}" -le 1 ]]; then
  echo
  echo "--- Stage 1: ambe mc features (ms=8 xi=0.02 t=25ns) ---"
  ambe mc features --config "$CONFIG"
else
  echo "--- Stage 1: SKIPPED (START_STAGE=${START_STAGE}) — reusing __cluster_features.parquet ---"
fi

# ── Stage 1b: background composition report ──────────────────────────────
# Quantifies what the delayed background actually IS, along three ancestry axes
# (immediate ancestor / first non-EM ancestor / generator primary) plus the
# literal lineage chains and the per-cluster dominant-species yields. Diagnostic
# only — the MVA below stays binary and still lumps every non-neutron cluster
# into one background class.
if [[ "${START_STAGE}" -le 1 ]]; then
  echo
  echo "--- Stage 1b: background composition report ---"
  python report_background_composition.py \
      --config "$CONFIG" \
      --methods ${METHODS} \
      --comp-n-neutron-max "$COMP_N_NEUTRON_MAX" \
      --comp-frac-nonneutron-min "$COMP_FRAC_NONNEUTRON_MIN"
else
  echo "--- Stage 1b: SKIPPED (START_STAGE=${START_STAGE}) ---"
fi

# Path variables used by Stage 2 (write) AND Stage 3/4 (read) — define UNCONDITIONALLY.
PKL_MCA="${OUT_ROOT}/${RUN}/parquet/${RUN}__mva_frozen${MODEL_TAG}.pkl"
MC_SCORES="${OUT_ROOT}/${RUN}/parquet/${RUN}__mva_scores${MODEL_TAG}.parquet"
MC_FEATURES="${OUT_ROOT}/${RUN}/parquet/${RUN}__cluster_features.parquet"

# ── Stage 2: Train MVA — signal vs MC internal background, 1:1, incl. prompt ──
# --keep-prompt-bkg triples the background pool (prompt clusters included),
# which is the decisive fix that lifted AUC from 0.587 → 0.718 in productionv2.
# The MVA is trained on xi=0.02 cluster features (different shape from xi=0.10).
N_METHODS="$(wc -w <<< "${METHODS}")"

# mva_analysis.py writes the scores parquet, plot PDF and summary CSV to
# method-AGNOSTIC names derived from run_name + mode_tag. Training two methods
# under one run_name therefore clobbers the first method's artifacts. When more
# than one method is trained, each one's artifacts are moved to a per-method name
# immediately after its training. Same approach as run_..._cftrain.sh.
preserve_artifacts () {   # $1 = method suffix (optics|cf|...)
  local suf="$1"
  local d_scores="${OUT_ROOT}/${RUN}/parquet/${RUN}__mva_scores${MODEL_TAG}.parquet"
  local d_plot="${OUT_ROOT}/${RUN}/plots/${RUN}__mva${MODEL_TAG}.pdf"
  local d_csv="${OUT_ROOT}/${RUN}/csv/${RUN}__mva_summary${MODEL_TAG}.csv"
  if [[ ! -e "$d_scores" ]]; then
    echo "ERROR: training wrote no scores at $d_scores"; exit 1
  fi
  mv -f "$d_scores" "${d_scores%.parquet}__${suf}.parquet"
  echo "[pipeline] scores -> ${RUN}__mva_scores${MODEL_TAG}__${suf}.parquet"
  [[ -e "$d_plot" ]] && mv -f "$d_plot" "${d_plot%.pdf}__${suf}.pdf" && \
    echo "[pipeline] plot   -> ${RUN}__mva${MODEL_TAG}__${suf}.pdf"
  [[ -e "$d_csv" ]] && mv -f "$d_csv" "${d_csv%.csv}__${suf}.csv" && \
    echo "[pipeline] csv    -> ${RUN}__mva_summary${MODEL_TAG}__${suf}.csv"
  return 0
}

if [[ "${STOP_STAGE}" -le 1 ]]; then
  echo
  echo "========================================================"
  echo "  stopped after Stage 1b (STOP_STAGE=${STOP_STAGE})  —  $(date)"
  echo "  log: $LOG"
  echo "========================================================"
  exit 0
fi

if [[ "${START_STAGE}" -le 2 ]]; then
  echo
  echo "--- Stage 2: MVA train (pure MC background, 1:1, incl. prompt) ---"
  for M in ${METHODS}; do
    SUF="$([[ "$M" == "clusterfinder" ]] && echo cf || echo "$M")"
    echo "--- Stage 2 [${M}] ---"
    if [[ "${N_METHODS}" -eq 1 ]]; then
      # Single method: keep the historical default artifact names untouched.
      python mva_analysis.py \
          --config "$CONFIG" \
          --method "$M" \
          --cc-only \
          --bkg-mode all \
          --max-ratio 1.0 \
          --keep-prompt-bkg \
          --save-model
    else
      python mva_analysis.py \
          --config "$CONFIG" \
          --method "$M" \
          --cc-only \
          --bkg-mode all \
          --max-ratio 1.0 \
          --keep-prompt-bkg \
          --save-model "${OUT_ROOT}/${RUN}/parquet/${RUN}__mva_frozen${MODEL_TAG}__${SUF}.pkl"
      preserve_artifacts "$SUF"
    fi
  done
else
  echo "--- Stage 2: SKIPPED (START_STAGE=${START_STAGE}) — reusing trained MVA model ---"
fi

if [[ "${STOP_STAGE}" -le 2 ]]; then
  echo
  echo "========================================================"
  echo "  stopped after Stage 2 (STOP_STAGE=${STOP_STAGE})  —  $(date)"
  echo "  Stage 3 (AmBe data scoring) and Stage 4 (box-cut benchmarks) not run."
  echo "  log: $LOG"
  echo "========================================================"
  exit 0
fi

# Stage 3+ assumes ONE model per run (the default-named PKL_MCA / MC_SCORES).
# Per-method Stage 3/4 already exists as run_ccinc_xi02_compcut_cftrain.sh —
# don't half-implement a second copy of it here.
if [[ "${N_METHODS}" -gt 1 ]]; then
  echo "ERROR: METHODS='${METHODS}' trains one model per method, which Stage 3/4"
  echo "       here cannot consume (they read the single default-named model)."
  echo "       Use STOP_STAGE=2, or run run_ccinc_xi02_compcut_cftrain.sh for the"
  echo "       per-method Stage 3/4 chain."
  exit 1
fi

# ── Pick the best classifier + derive its operating cut (adaptive) ───────
echo
echo "--- Picking best MVA model + operating cut ---"
eval "$(python pick_best_model_cut.py --scores "$MC_SCORES" --sig-eff 0.80)"
: "${BEST_SCORE_COL:=rf_score}"
: "${BEST_SCORE_CUT:=0.5}"
echo "[pipeline] using BEST_SCORE_COL=${BEST_SCORE_COL}  BEST_SCORE_CUT=${BEST_SCORE_CUT}"

# ── Stage 3: Score AmBe data + downstream per MVA streamline ─────────────
OPTICS_DATA="${OUT_ROOT}/ambe_data/ambe_all__data_features_optics.parquet"
CF_DATA="${OUT_ROOT}/ambe_data/ambe_all__data_features_cf.parquet"

# Composition-cut filtered MC features parquet (used in Stage 4 benchmark)
MC_FEAT_CC="${OUT_ROOT}/${RUN}/parquet/${RUN}__cluster_features_compcut.parquet"
echo
echo "--- Applying composition cut to MC features for Stage 4 benchmark ---"
python - <<PYEOF
import pandas as pd
df = pd.read_parquet("${MC_FEATURES}")
keep = ~((df["n_neutron"] <= ${COMP_N_NEUTRON_MAX}) & (df["frac_nonneutron"] >= ${COMP_FRAC_NONNEUTRON_MIN}))
df[keep].to_parquet("${MC_FEAT_CC}", index=False)
n_removed = (~keep).sum()
n_kept    = keep.sum()
print(f"MC composition cut: {n_removed} clusters removed, {n_kept} kept  "
      f"({100*n_removed/(n_removed+n_kept):.1f}% rejected)")
PYEOF

# Streamlines: "label:data_parquet:pkl:mc_scores_parquet:mc_method"
STREAMLINES=(
  "${TAG_PREFIX}_optics_mcbkg:${OPTICS_DATA}:${PKL_MCA}:${MC_SCORES}:optics"
  "${TAG_PREFIX}_cf_mcbkg:${CF_DATA}:${PKL_MCA}:${MC_SCORES}:clusterfinder"
)

for ENTRY in "${STREAMLINES[@]}"; do
  IFS=: read -r TAG DATA_PAR PKL MC_PAR MC_METHOD <<< "$ENTRY"
  echo
  echo "=== Streamline: ${TAG} ==="

  OUT_S="${OUT_ROOT}/ambe_data/${TAG}"
  mkdir -p "$OUT_S"

  # Apply composition cut to the data feature parquet before scoring.
  # The cut uses truth columns (n_neutron, frac_nonneutron) which only exist in MC.
  # Real AmBe data parquets have no truth labels, so we skip the cut there and
  # score the full data parquet directly.
  COMP_CUT_DATA="${OUT_S}/${TAG}__data_compcut.parquet"
  echo "--- ${TAG}: applying composition cut to data features (skipped if truth cols absent) ---"
  python - <<PYEOF
import pandas as pd, shutil
df = pd.read_parquet("${DATA_PAR}")
if "n_neutron" in df.columns and "frac_nonneutron" in df.columns:
    keep = ~((df["n_neutron"] <= ${COMP_N_NEUTRON_MAX}) & (df["frac_nonneutron"] >= ${COMP_FRAC_NONNEUTRON_MIN}))
    df[keep].to_parquet("${COMP_CUT_DATA}", index=False)
    n_removed = (~keep).sum()
    n_kept    = keep.sum()
    print(f"Data composition cut [${TAG}]: {n_removed} clusters removed, {n_kept} kept  "
          f"({100*n_removed/(n_removed+n_kept):.1f}% rejected)")
else:
    shutil.copy("${DATA_PAR}", "${COMP_CUT_DATA}")
    print(f"Data composition cut [${TAG}]: SKIPPED — no truth columns in data parquet ({len(df)} clusters passed through)")
PYEOF

  # Stage 3a — score comp-cut data, then rename to per-streamline scored parquet
  echo "--- ${TAG}: score AmBe data (post comp-cut) ---"
  python mva_analysis.py \
      --score-data "$COMP_CUT_DATA" \
      --model      "$PKL"
  DATA_STEM="$(basename "${COMP_CUT_DATA%.parquet}")"
  DEFAULT_SCORED="$(dirname "$COMP_CUT_DATA")/${DATA_STEM}__scored.parquet"
  SCORED="${OUT_S}/${TAG}__scored.parquet"
  mv -f "$DEFAULT_SCORED" "$SCORED"

  # Stage 3b — per-run neutron rate summary
  echo "--- ${TAG}: neutron rate summary (${BEST_SCORE_COL}) ---"
  python summarize_ambe_neutrons.py \
      --scored    "$SCORED" \
      --mc-scores "$MC_PAR" \
      --score-col "$BEST_SCORE_COL" \
      --sig-eff   0.80 \
      --out       "${OUT_S}/rate_best"

  # Stage 3c — multiplicity + score PDF
  echo "--- ${TAG}: multiplicity PDF (${BEST_SCORE_COL} @ ${BEST_SCORE_CUT}) ---"
  python plot_ambe_neutron_multiplicity.py \
      --scored      "$SCORED" \
      --mc-features "$MC_FEATURES" \
      --score-col   "$BEST_SCORE_COL" \
      --score-cut   "$BEST_SCORE_CUT" \
      --out         "${OUT_S}/${TAG}__multiplicity.pdf"

  # Stage 3d — capture-time fit (lmfit)
  echo "--- ${TAG}: capture-time fit (${BEST_SCORE_COL} @ ${BEST_SCORE_CUT}) ---"
  python capture_time_stageAB_from_parquet.py \
      --scored    "$SCORED" \
      --score-col "$BEST_SCORE_COL" \
      --score-cut "$BEST_SCORE_CUT" \
      --out-dir   "${OUT_S}/capture_stageAB" \
      --methods   lmfit

done

if [[ "${STOP_STAGE}" -le 3 ]]; then
  echo
  echo "========================================================"
  echo "  stopped after Stage 3 (STOP_STAGE=${STOP_STAGE})  —  $(date)"
  echo "  log: $LOG"
  echo "========================================================"
  exit 0
fi

# ── Stage 4: Box-cut benchmarks (using comp-cut-filtered MC features) ─────
OPTICS_SCORED_MCA="${OUT_ROOT}/ambe_data/${TAG_PREFIX}_optics_mcbkg/${TAG_PREFIX}_optics_mcbkg__scored.parquet"
CF_SCORED_MCA="${OUT_ROOT}/ambe_data/${TAG_PREFIX}_cf_mcbkg/${TAG_PREFIX}_cf_mcbkg__scored.parquet"

echo
echo "--- Stage 4 (${TAG_PREFIX}_optics_boxcuts): OPTICS + box cuts ---"
OUT_S3="${OUT_ROOT}/ambe_data/${TAG_PREFIX}_optics_boxcuts"
python benchmark_selection_2x2.py \
    --features  "$MC_FEAT_CC" \
    --model     "$PKL_MCA" \
    --score-col "$BEST_SCORE_COL" \
    --score-cut "$BEST_SCORE_CUT" \
    --out-dir   "${OUT_S3}/benchmark"

echo
echo "--- Stage 4 (${TAG_PREFIX}_cf_boxcuts): CF + box cuts ---"
OUT_S6="${OUT_ROOT}/ambe_data/${TAG_PREFIX}_cf_boxcuts"
python benchmark_selection_2x2.py \
    --features  "$MC_FEAT_CC" \
    --model     "$PKL_MCA" \
    --score-col "$BEST_SCORE_COL" \
    --score-cut "$BEST_SCORE_CUT" \
    --out-dir   "${OUT_S6}/benchmark"

echo
echo "========================================================"
echo "  all done  —  $(date)"
echo "  log: $LOG"
echo "========================================================"
