#!/usr/bin/env bash
# run_ambe_lucho_mva.sh
# ─────────────────────────────────────────────────────────────────────────────
# Apply the lucho100k frozen OPTICS+CF MVA models to ALL AmBe data runs, end to
# end, in ONE tmux session. Mirrors run_ambe_steven_mva.sh but:
#   • driven by the lucho100k frozen models (weak: test AUC ~0.70 — see notes);
#   • runs BOTH methods (optics + clusterfinder) — produces a full output set each;
#   • produces rf/gbt/xgb summaries (per-score) so the three trees can be compared;
#   • FIRST builds the missing pieces the lucho freeze didn't produce:
#       Part A: ClusterFinder-method DATA features (the data parquet was OPTICS-only)
#               via extract_cf_features_beamcluster_data.py over the BeamCluster ntuples;
#       Part B: the optics MC-scores anchor (the freeze saved only CF MC scores).
#
# Usage:
#   tmux new -s lucho_mva
#   bash run_ambe_lucho_mva.sh            # all 33 runs, jobs=2 (4-core box)
#   JOBS=4 bash run_ambe_lucho_mva.sh     # override parallelism
#
# Re-running is safe: CF feature shards already done are skipped (resume); the
# optics anchor + scored parquets are overwritten idempotently.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

JOBS="${JOBS:-2}"                       # parallel files for CF extraction (cores-2 on a 4-core box)
SIG_EFF="${SIG_EFF:-0.80}"              # MC signal-efficiency working point for the cut

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="/exp/annie/app/users/dajana/AmBeNeutronsAnalysis"
VENV="/exp/annie/app/users/dajana/myboy/bin/activate"
OUT="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/ambe_data"
DATA_DIR="/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1"
GATE_DIR="${REPO}/EventAmBeNeutronCandidatesData"
GEOM="/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/FullTankPMTGeometry.csv"
OFFSETS="/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/TankPMTTimingOffsets.csv"
CONFIG="${REPO}/configs/mc_lucho_100k.yaml"
RUN_NAME="ambe_all"

# Lucho frozen models + MC parquets
LUCHO_DIR="${REPO}/ambe_output/mc_lucho_100k/parquet"
MODEL_OPT="${LUCHO_DIR}/mc_lucho_100k_optics_frozen.pkl"
MODEL_CF="${LUCHO_DIR}/mc_lucho_100k_cf_frozen.pkl"
MC_FEATS_ALL="${LUCHO_DIR}/mc_lucho_100k__cluster_features.parquet"   # both methods
MC_SCORES_CF="${LUCHO_DIR}/mc_lucho_100k__mva_scores.parquet"         # CF-only (from freeze)

# Per-method MC anchors (optics regenerated in Part B; CF is the freeze output)
MC_FEATS_OPT="${LUCHO_DIR}/mc_lucho_100k__cluster_features_optics.parquet"
MC_FEATS_CFM="${LUCHO_DIR}/mc_lucho_100k__cluster_features_clusterfinder.parquet"
MC_SCORES_OPT="${LUCHO_DIR}/mc_lucho_100k__mva_scores_optics.parquet"

# Data feature parquets
FEATS_OPT_SRC="${OUT}/${RUN_NAME}__data_features.parquet"             # existing OPTICS-only
FEATS_CF="${OUT}/${RUN_NAME}__data_features_cf.parquet"               # built in Part A
FEATS_BOTH="${OUT}/${RUN_NAME}__data_features_bothmethods.parquet"    # concat (Part A)

STATS_CSV="${REPO}/optics_beamcluster_benchmark_gated/optics_beamcluster_stats.csv"
LOG="${REPO}/logs_ambe_lucho_mva.log"

cd "${REPO}"
source "${VENV}"
exec > >(tee -a "${LOG}") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "AmBe lucho100k MVA pipeline (BOTH methods)   jobs=${JOBS}  sig_eff=${SIG_EFF}"
echo "start: $(date)"
echo "  NOTE: lucho frozen MVA is weak (test AUC ~0.70). Outputs are a lucho"
echo "        comparison set; the steven model remains the stronger selector."
echo "════════════════════════════════════════════════════════════════"

# ── Part A: ClusterFinder-method DATA features over all BeamCluster runs ──────
echo
echo "[A] extracting ClusterFinder-method DATA features (resumable) ..."
python -u extract_cf_features_beamcluster_data.py \
    "${DATA_DIR}/" \
    --run-name "${RUN_NAME}" --output-dir "${OUT}" \
    --gate-csv-dir "${GATE_DIR}" \
    --geometry "${GEOM}" --offsets "${OFFSETS}" \
    --jobs "${JOBS}" || { echo "FATAL: CF feature extraction failed"; exit 1; }
[[ -f "${FEATS_CF}" ]] || { echo "FATAL: CF features parquet not written: ${FEATS_CF}"; exit 1; }

echo
echo "[A] patching method='optics' on a COPY of the OPTICS data features + concat with CF ..."
python - "${FEATS_OPT_SRC}" "${FEATS_CF}" "${FEATS_BOTH}" <<'PYEOF'
import sys, pandas as pd
opt_src, cf_path, both_out = sys.argv[1], sys.argv[2], sys.argv[3]
opt = pd.read_parquet(opt_src)
if "method" not in opt.columns:
    opt["method"] = "optics"
elif (opt["method"] != "optics").any():
    opt["method"] = "optics"
cf = pd.read_parquet(cf_path)
print(f"  optics rows: {len(opt):,}  | CF rows: {len(cf):,}")
both = pd.concat([opt, cf], ignore_index=True, sort=False)   # aligns columns; optics gets NaN for clusterTime_earliest
print(f"  combined: {len(both):,} rows  | method counts: {both['method'].value_counts().to_dict()}")
both.to_parquet(both_out, index=False)
print(f"  wrote -> {both_out}")
PYEOF
[[ -f "${FEATS_BOTH}" ]] || { echo "FATAL: combined parquet not written"; exit 1; }

# ── Part B: build per-method MC feature subsets + regenerate optics MC anchor ─
echo
echo "[B] splitting MC cluster_features by method + scoring optics MC anchor ..."
python - "${MC_FEATS_ALL}" "${MC_FEATS_OPT}" "${MC_FEATS_CFM}" <<'PYEOF'
import sys, pandas as pd
allp, opt_out, cf_out = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(allp)
opt = df[df["method"] == "optics"].reset_index(drop=True)
cf  = df[df["method"] == "clusterfinder"].reset_index(drop=True)
opt.to_parquet(opt_out, index=False); cf.to_parquet(cf_out, index=False)
print(f"  MC optics features: {len(opt):,} -> {opt_out}")
print(f"  MC CF     features: {len(cf):,} -> {cf_out}")
PYEOF

# Score the optics MC features through the optics frozen model -> optics MC anchor.
# mva_analysis.py --score-data writes <stem>__scored.parquet next to the input.
echo "[B] scoring optics MC features with mc_lucho_100k_optics_frozen.pkl ..."
python mva_analysis.py --config "${CONFIG}" \
    --score-data "${MC_FEATS_OPT}" --model "${MODEL_OPT}" \
    || { echo "FATAL: optics MC anchor scoring failed"; exit 1; }
MC_OPT_SCORED="${MC_FEATS_OPT%.parquet}__scored.parquet"
cp -f "${MC_OPT_SCORED}" "${MC_SCORES_OPT}"
echo "[B] optics MC anchor -> ${MC_SCORES_OPT}"
echo "[B] CF     MC anchor  -> ${MC_SCORES_CF}  (from the lucho freeze)"

# ── Per-method pipeline driver ────────────────────────────────────────────────
# $1 method (optics|clusterfinder)  $2 frozen model pkl  $3 MC scores anchor  $4 MC features
run_method () {
  local METHOD="$1" MODEL="$2" MC_SCORES="$3" MC_FEATS="$4"
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo ">>> METHOD = ${METHOD}"
  echo "════════════════════════════════════════════════════════════════"

  # Stage 1: split the combined data parquet to this method, then score it.
  local SPLIT="${OUT}/${RUN_NAME}__data_features_${METHOD}_forscore.parquet"
  python - "${FEATS_BOTH}" "${METHOD}" "${SPLIT}" <<'PYEOF'
import sys, pandas as pd
both, method, out = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(both)
sub = df[df["method"] == method].reset_index(drop=True)
print(f"  [{method}] {len(sub):,} data clusters to score")
if len(sub) == 0:
    sys.exit(f"ERROR: no {method} data rows — cannot score")
sub.to_parquet(out, index=False)
PYEOF
  if [[ ! -s "${SPLIT}" ]]; then echo "  [${METHOD}] skipped (no data rows)"; return 0; fi

  python mva_analysis.py --config "${CONFIG}" --score-data "${SPLIT}" --model "${MODEL}" \
      || { echo "FATAL: scoring ${METHOD} failed"; exit 1; }
  local SCORED="${SPLIT%.parquet}__scored.parquet"
  local CANON="${OUT}/${RUN_NAME}__data_features_${METHOD}__scored_lucho.parquet"
  cp -f "${SCORED}" "${CANON}"
  echo "  [${METHOD}] scored -> ${CANON}"

  # Stage 2: per-run neutron-rate summary for rf / gbt / xgb (each into its own dir).
  local RF_THR=""
  for SCORE in rf_score gbt_score xgb_score; do
    local RDIR="${OUT}/lucho_${METHOD}_rate_${SCORE%%_*}"
    mkdir -p "${RDIR}"
    echo "  [${METHOD}] summary (${SCORE}) ..."
    python summarize_ambe_neutrons.py \
        --scored "${CANON}" --mc-scores "${MC_SCORES}" \
        --score-col "${SCORE}" --sig-eff "${SIG_EFF}" --out "${RDIR}" \
        || echo "  WARN: summary ${METHOD}/${SCORE} failed (continuing)"
  done

  # Derive the rf threshold @ sig-eff from THIS method's MC anchor (for Stage 4).
  RF_THR=$(python - "${MC_SCORES}" "${SIG_EFF}" <<'PYEOF'
import sys, numpy as np, pandas as pd
mc = pd.read_parquet(sys.argv[1]); se = float(sys.argv[2])
y = mc["dominant_class"].isin([1,2,3,4]).to_numpy().astype(int)
test = mc["in_test"].to_numpy().astype(bool) if "in_test" in mc else np.ones(len(mc), bool)
sig = mc["rf_score"].to_numpy()[test & (y == 1)]
print(f"{float(np.quantile(sig, 1-se)):.4f}" if len(sig) else "nan")
PYEOF
)
  echo "  [${METHOD}] rf_score @ ${SIG_EFF} sig-eff threshold = ${RF_THR}"

  # Stage 3: multiplicity + score PDF (gbt @ 0.5 default), MC feature overlays.
  local STATS_ARG=""
  [[ -f "${STATS_CSV}" ]] && STATS_ARG="--stats-csv ${STATS_CSV}"
  echo "  [${METHOD}] multiplicity + score PDF ..."
  python plot_ambe_neutron_multiplicity.py \
      --scored "${CANON}" ${STATS_ARG} \
      --mc-features "${MC_FEATS}" \
      --score-col gbt_score --score-cut 0.5 \
      --out "${OUT}/${RUN_NAME}__neutron_multiplicity_lucho_${METHOD}.pdf" \
      || echo "  WARN: multiplicity plot ${METHOD} failed (continuing)"

  # Stage 4: capture-time fit (lmfit) at the method-derived rf threshold.
  local CAPDIR="${OUT}/capture_stageAB_lucho_${METHOD}"
  mkdir -p "${CAPDIR}"
  if [[ "${RF_THR}" != "nan" && -n "${RF_THR}" ]]; then
    echo "  [${METHOD}] capture-time fit (lmfit, rf>${RF_THR}) ..."
    python capture_time_stageAB_from_parquet.py \
        --scored "${CANON}" --score-col rf_score --score-cut "${RF_THR}" \
        --out-dir "${CAPDIR}" --methods lmfit \
        || echo "  WARN: capture-time fit ${METHOD} failed (continuing)"
  else
    echo "  [${METHOD}] capture-time fit SKIPPED (no finite rf threshold)"
  fi

  # Stage 5: data / MC feature comparison (KS) for this method.
  local CMPDIR="${OUT}/compare_selections_lucho_${METHOD}"
  mkdir -p "${CMPDIR}"
  echo "  [${METHOD}] data/MC comparison ..."
  python analyze_data_mc_comparison.py \
      --data-dir "${GATE_DIR}" \
      --mc-parquet "${LUCHO_DIR}" \
      --run-name mc_lucho_100k \
      --mc-method "${METHOD}" \
      --output-dir "${CMPDIR}" \
      || echo "  WARN: data/MC comparison ${METHOD} failed (continuing)"
  echo "  [${METHOD}] done."
}

# ── Run both methods ──────────────────────────────────────────────────────────
run_method optics        "${MODEL_OPT}" "${MC_SCORES_OPT}" "${MC_FEATS_OPT}"
run_method clusterfinder "${MODEL_CF}"  "${MC_SCORES_CF}"  "${MC_FEATS_CFM}"

echo
echo "════════════════════════════════════════════════════════════════"
echo "DONE: $(date)"
echo "Outputs under ${OUT}/ :"
echo "  combined features        : ${RUN_NAME}__data_features_bothmethods.parquet"
echo "  scored (per method)      : ${RUN_NAME}__data_features_{optics,clusterfinder}__scored_lucho.parquet"
echo "  rate summaries           : lucho_{optics,clusterfinder}_rate_{rf,gbt,xgb}/"
echo "  multiplicity PDFs        : ${RUN_NAME}__neutron_multiplicity_lucho_{optics,clusterfinder}.pdf"
echo "  capture-time fits        : capture_stageAB_lucho_{optics,clusterfinder}/"
echo "  data/MC comparisons      : compare_selections_lucho_{optics,clusterfinder}/"
echo "════════════════════════════════════════════════════════════════"
