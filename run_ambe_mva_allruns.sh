#!/usr/bin/env bash
# run_ambe_mva_allruns.sh
# ─────────────────────────────────────────────────────────────────────────────
# Full AmBe-data neutron MVA pipeline over ALL AmBe2.0v1 runs, end to end.
# Paste this whole thing into a tmux session and walk away.
#
#   Stage 1  extract MVA features for every OPTICS cluster on every gated run
#            (MC-matched Stage A: prompt-window t>2000ns, no CF-prefilter,
#             source_pos OFF; cosmic veto per-event). Parallel + resumable.
#   Stage 2  score all clusters with the frozen MC-trained model (RF/GBT/XGB).
#   Stage 3  per-run neutron-rate + multiplicity summary (MC-anchored threshold).
#   Stage 4  per-run multiplicity PDFs (Stage A / cuts comparison / Stage B / diagnostics).
#
# Re-running is safe: feature shards already done are skipped (resume). To force a
# clean re-run, delete  <OUT>/_feature_shards/  first.
#
# Usage:
#   tmux new -s ambe
#   bash run_ambe_mva_allruns.sh            # all 33 runs, jobs=8
#   bash run_ambe_mva_allruns.sh 4          # override jobs=4
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

JOBS="${1:-8}"                       # parallel files; tune to the box (cores-2)
SCORE_COL="${SCORE_COL:-rf_score}"   # primary model (RF, per analysis decision)
SIG_EFF="${SIG_EFF:-0.80}"           # MC signal-efficiency working point for the cut
PLOT_CUT="${PLOT_CUT:-0.7}"          # raw score cut used only in the per-run Stage-B page

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="/exp/annie/app/users/dajana/AmBeNeutronsAnalysis"
VENV="/exp/annie/app/users/dajana/myboy/bin/activate"
DATA_DIR="/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1"
GATE_DIR="${REPO}/EventAmBeNeutronCandidatesData"
OUT="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/ambe_data"
MC="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/cc_neutrino/parquet"
MODEL="${MC}/cc_neutrino__mva_frozen.pkl"
MC_SCORES="${MC}/cc_neutrino__mva_scores.parquet"
MC_FEATS="${MC}/cc_neutrino__cluster_features.parquet"
STATS_CSV="${REPO}/optics_beamcluster_benchmark_gated/optics_beamcluster_stats.csv"
CONFIG="${REPO}/configs/cc_neutrino_optics.yaml"
RUN_NAME="ambe_all"
LOG="${REPO}/logs_ambe_mva_allruns.log"

cd "${REPO}"
source "${VENV}"
exec > >(tee -a "${LOG}") 2>&1
echo "════════════════════════════════════════════════════════════════"
echo "AmBe MVA all-runs pipeline   jobs=${JOBS}  score=${SCORE_COL}  sig_eff=${SIG_EFF}"
echo "start: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Stage 0: ensure the frozen model exists (re-freeze if missing) ────────────
if [[ ! -f "${MODEL}" ]]; then
  echo "[0] frozen model missing — training + freezing from MC ..."
  python mva_analysis.py --config "${CONFIG}" --no-nn --no-cv --save-model || exit 1
else
  echo "[0] frozen model present: ${MODEL}"
fi

# ── Stage 1: feature extraction over ALL runs (resumable) ─────────────────────
echo "[1] extracting features over all runs in ${DATA_DIR} ..."
python -u analyze_optics_beamcluster_data.py \
    "${DATA_DIR}/" \
    --features-out --run-name "${RUN_NAME}" --output-dir "${OUT}" \
    --gate-csv-dir "${GATE_DIR}" --jobs "${JOBS}" || exit 1
FEATS="${OUT}/${RUN_NAME}__data_features.parquet"
echo "[1] features -> ${FEATS}"

# ── Stage 2: score all clusters with the frozen model ─────────────────────────
echo "[2] scoring with frozen model ..."
python mva_analysis.py --config "${CONFIG}" --score-data "${FEATS}" || exit 1
SCORED="${OUT}/${RUN_NAME}__data_features__scored.parquet"
echo "[2] scored -> ${SCORED}"

# ── Stage 3: per-run neutron-rate + position-dependence summary ───────────────
echo "[3] per-run summary (MC-anchored ${SCORE_COL} @ ${SIG_EFF} signal eff) ..."
python summarize_ambe_neutrons.py \
    --scored "${SCORED}" --mc-scores "${MC_SCORES}" \
    --score-col "${SCORE_COL}" --sig-eff "${SIG_EFF}" --out "${OUT}" || exit 1

# ── Stage 4: combined multiplicity / comparison PDF ───────────────────────────
echo "[4] multiplicity + comparison plots ..."
python plot_ambe_neutron_multiplicity.py \
    --scored "${SCORED}" --mc-features "${MC_FEATS}" --stats-csv "${STATS_CSV}" \
    --score-col "${SCORE_COL}" --score-cut "${PLOT_CUT}" \
    --out "${OUT}/${RUN_NAME}__neutron_multiplicity.pdf" || exit 1

echo "════════════════════════════════════════════════════════════════"
echo "DONE: $(date)"
echo "Outputs in ${OUT}:"
echo "  ${RUN_NAME}__data_features.parquet          (all-run features)"
echo "  ${RUN_NAME}__data_features__scored.parquet  (+ rf/gbt/xgb scores)"
echo "  ${RUN_NAME}__neutron_rate_by_run.csv        (per-run table)"
echo "  ${RUN_NAME}__neutron_summary.pdf            (rate + position dependence)"
echo "  ${RUN_NAME}__neutron_multiplicity.pdf       (Stage A/cuts/Stage B/diagnostics)"
echo "════════════════════════════════════════════════════════════════"
