#!/usr/bin/env bash
# run_ambe_steven_mva.sh
# ─────────────────────────────────────────────────────────────────────────────
# Apply the steven-multiport frozen OPTICS MVA to all AmBe data runs.
# Runs end-to-end in a tmux session without babysitting.
#
# What this does:
#   Stage 0  Patch the data features parquet to add method='optics' (one-shot,
#            idempotent — safe to re-run).
#   Stage 1  Score with the frozen steven-multiport OPTICS model
#            (rf_score, gbt_score, xgb_score, nn_score).
#   Stage 2  Neutron-rate + multiplicity summary (MC-anchored threshold via
#            mc_steven_multiport OPTICS scores — true AmBe MC anchor).
#   Stage 3  Multiplicity + score-distribution PDF (Stage A vs Stage B),
#            with MC feature overlays from mc_steven_multiport cluster_features.
#            Legacy box-cut reference: PE<80 / CB<0.45 / hits>=10.
#   Stage 4  Capture-time fit (lmfit only — fast; add pymc manually if wanted).
#   Stage 5  Data/MC feature comparison (KS tests + overlay histograms)
#            using mc_steven_multiport cluster_features vs real data.
#
# Usage:
#   tmux new -s ambe_steven
#   bash run_ambe_steven_mva.sh
#
# Outputs land in: $OUT  (same dir as the existing ambe_data outputs)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="/exp/annie/app/users/dajana/AmBeNeutronsAnalysis"
VENV="/exp/annie/app/users/dajana/myboy/bin/activate"
OUT="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/ambe_data"
CC_DIR="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/cc_neutrino/parquet"

# Frozen OPTICS model (steven multiport)
MODEL_DIR="${REPO}/mc_steven"           # frozen .pkl + .keras files live here
MODEL="${MODEL_DIR}/mc_steven_multiport_optics_frozen.pkl"
CONFIG="${REPO}/configs/mc_steven_multiport_mva.yaml"

# MC AmBe steven multiport parquets (copied from laptop)
MC_STEVEN_DIR="${REPO}/ambe_output/mc_steven_multiport/parquet"
MC_AMBE_FEATS="${MC_STEVEN_DIR}/mc_steven_multiport__cluster_features.parquet"
MC_AMBE_SCORES="${MC_STEVEN_DIR}/mc_steven_multiport__mva_scores_optics.parquet"

# Data parquets
FEATS="${OUT}/ambe_all__data_features.parquet"
SCORED="${OUT}/ambe_all__data_features__scored_steven.parquet"

# MC reference for MC-anchored threshold — use steven OPTICS scores (true AmBe MC anchor)
MC_SCORES="${MC_AMBE_SCORES}"
MC_FEATS="${MC_AMBE_FEATS}"

# Stats CSV from gated OPTICS benchmark (for Stage-A CF comparison in multiplicity plot)
STATS_CSV="${REPO}/optics_beamcluster_benchmark_gated/optics_beamcluster_stats.csv"

LOG="${REPO}/logs_ambe_steven_mva.log"

cd "${REPO}"
source "${VENV}"
exec > >(tee -a "${LOG}") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "AmBe steven-multiport MVA scoring pipeline"
echo "start: $(date)"
echo "  features:  ${FEATS}"
echo "  model:     ${MODEL}"
echo "  scored ->  ${SCORED}"
echo "════════════════════════════════════════════════════════════════"

# ── Stage 0: inject method='optics' column (idempotent) ──────────────────────
echo
echo "[0] patching method column ..."
python - "${FEATS}" <<'PYEOF'
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_parquet(p)
if "method" in df.columns and df["method"].iloc[0] == "optics":
    print(f"  method column already present — skip")
    sys.exit(0)
df["method"] = "optics"
df.to_parquet(p, index=False)
print(f"  patched {len(df):,} rows  ->  {p}")
PYEOF

# ── Stage 1: score with frozen steven-multiport OPTICS model ─────────────────
echo
echo "[1] scoring with steven-multiport OPTICS frozen model ..."

# score_real_data.sh splits by method, scores, and writes <stem>_optics__scored.parquet.
# We then copy/rename to the canonical SCORED path so downstream stages use it.
bash "${REPO}/score_real_data.sh" \
    "${FEATS}" \
    "${MODEL_DIR}" \
    "${CONFIG}"

# score_real_data.sh writes alongside FEATS:
#   ambe_all__data_features_optics__scored.parquet
RAW_SCORED="${OUT}/ambe_all__data_features_optics__scored.parquet"
if [[ ! -f "${RAW_SCORED}" ]]; then
    echo "ERROR: expected scored output not found: ${RAW_SCORED}" >&2
    exit 1
fi
cp "${RAW_SCORED}" "${SCORED}"
echo "[1] scored -> ${SCORED}"

# ── Stage 2: per-run neutron-rate summary ────────────────────────────────────
echo
echo "[2] per-run neutron-rate summary (MC-anchored rf_score @ 0.80 sig-eff) ..."
python summarize_ambe_neutrons.py \
    --scored      "${SCORED}" \
    --mc-scores   "${MC_SCORES}" \
    --score-col   rf_score \
    --sig-eff     0.80 \
    --out         "${OUT}"
echo "[2] summary -> ${OUT}/ambe_all__neutron_rate_by_run.csv"

# ── Stage 3: multiplicity + score-distribution PDF ───────────────────────────
echo
echo "[3] multiplicity + score PDF (Stage A → B) ..."
STATS_ARG=""
[[ -f "${STATS_CSV}" ]] && STATS_ARG="--stats-csv ${STATS_CSV}"
python plot_ambe_neutron_multiplicity.py \
    --scored      "${SCORED}" \
    ${STATS_ARG} \
    --mc-features "${MC_FEATS}" \
    --score-col   gbt_score \
    --score-cut   0.5 \
    --out         "${OUT}/ambe_all__neutron_multiplicity_steven.pdf"
echo "[3] -> ${OUT}/ambe_all__neutron_multiplicity_steven.pdf"

# ── Stage 4: capture-time fit (lmfit only) ───────────────────────────────────
echo
echo "[4] capture-time fit (Stage A+B, lmfit) ..."
CAPTURE_DIR="${OUT}/capture_stageAB_steven"
mkdir -p "${CAPTURE_DIR}"
# Threshold derived from steven MC test-set: rf_score @ 80% signal efficiency = 0.8423.
# (The old 0.643 was from the cc_neutrino model — do not use it here.)
python capture_time_stageAB_from_parquet.py \
    --scored    "${SCORED}" \
    --score-col rf_score \
    --score-cut 0.8423 \
    --out-dir   "${CAPTURE_DIR}" \
    --methods   lmfit
echo "[4] -> ${CAPTURE_DIR}/"

# ── Stage 5: data / MC score-distribution + feature comparison ───────────────
echo
echo "[5] data vs MC feature comparison ..."
COMPARE_DIR="${OUT}/compare_selections_steven"
mkdir -p "${COMPARE_DIR}"

python analyze_data_mc_comparison.py \
    --data-dir   "${REPO}/EventAmBeNeutronCandidatesData" \
    --mc-parquet "${MC_STEVEN_DIR}" \
    --run-name   mc_steven_multiport \
    --mc-method  optics \
    --output-dir "${COMPARE_DIR}"
echo "[5] -> ${COMPARE_DIR}/"

echo
echo "════════════════════════════════════════════════════════════════"
echo "DONE: $(date)"
echo "Outputs:"
echo "  scored parquet :    ${SCORED}"
echo "  rate CSV        :   ${OUT}/ambe_all__neutron_rate_by_run.csv"
echo "  neutron summary :   ${OUT}/ambe_all__neutron_summary.pdf"
echo "  multiplicity PDF:   ${OUT}/ambe_all__neutron_multiplicity_steven.pdf"
echo "  capture time    :   ${CAPTURE_DIR}/"
echo "  data/MC features:   ${COMPARE_DIR}/"
echo "════════════════════════════════════════════════════════════════"
