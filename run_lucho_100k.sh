#!/usr/bin/env bash
# ============================================================
# 100k-event Lucho pipeline + frozen MVA models
# Input:  /Users/dajana/Documents/ANNIEEvent/Lucho_ANNIEEvent/*.root  (10 files x 10k events)
# Output: ambe_output/mc_lucho_100k/
#
# Steps:
#   1) ambe mc process   (annie venv)
#   2) ambe mc features  (annie venv)  -- no `ambe mc optics` grid sweep
#   3) mva_analysis.py --method optics        --save-model   (annie311 venv, needs TF)
#   4) mva_analysis.py --method clusterfinder --save-model   (annie311 venv, needs TF)
#
# Usage:  bash run_lucho_100k.sh
# ============================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

CFG="configs/mc_lucho_100k.yaml"
VENV_PIPE="$HOME/venvs/annie/bin/activate"
VENV_MVA="$HOME/venvs/annie311/bin/activate"
T0=$SECONDS

echo "=================================================="
echo "  ANNIE AmBe — Lucho 100k pipeline + freeze"
echo "  Config: $CFG"
echo "  Started: $(date)"
echo "=================================================="

# ---------- 1) processor + 2) features (annie) ----------
(
  # shellcheck disable=SC1090
  source "$VENV_PIPE"

  echo ""
  echo "[1/4] ROOT → per-hit parquet  (10 files, ~100k events)"
  time ambe mc process --config "$CFG"

  echo ""
  echo "[2/4] Feature extraction  (OPTICS + ClusterFinder methods, no prefilter)"
  time ambe mc features --config "$CFG"
)

# ---------- 3+4) MVA train + freeze, both methods (annie311) ----------
(
  # shellcheck disable=SC1090
  source "$VENV_MVA"

  for METHOD in optics clusterfinder; do
    SHORT=${METHOD/clusterfinder/cf}
    OUT="ambe_output/mc_lucho_100k/parquet/mc_lucho_100k_${SHORT}_frozen.pkl"
    echo ""
    echo "[3-4] MVA train + freeze  (method=${METHOD}, bkg=all non-neutron)"
    echo "      → ${OUT}"
    time python mva_analysis.py \
        --config "$CFG" \
        --method "$METHOD" \
        --bkg-mode all \
        --save-model "$OUT"
  done
)

echo ""
echo "=================================================="
echo "  Done in $(( SECONDS - T0 ))s  ($(date))"
echo "  Parquets:       ambe_output/mc_lucho_100k/parquet/"
echo "  Frozen models:  ambe_output/mc_lucho_100k/parquet/mc_lucho_100k_{optics,cf}_frozen.pkl"
echo "                  (+ companion __nn.keras for the NN)"
echo "=================================================="
