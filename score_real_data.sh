#!/usr/bin/env bash
#
# score_real_data.sh — apply frozen OPTICS+MVA and ClusterFinder+MVA models
# to a real-data cluster_features parquet, in one shot.
#
# Usage:
#   ./score_real_data.sh <DATA_FEATURES_PARQUET> [MODEL_DIR] [CONFIG]
#
# Arguments:
#   DATA_FEATURES_PARQUET  Required. Path to ambe mc features output for real data.
#                          Must contain a 'method' column with 'optics' / 'clusterfinder' values
#                          and the 33 feature columns the frozen models expect.
#   MODEL_DIR              Optional. Directory holding the four frozen-model files.
#                          Defaults to the directory of this script.
#                          Required files in MODEL_DIR:
#                            mc_steven_multiport_optics_frozen.pkl
#                            mc_steven_multiport_optics_frozen__nn.keras
#                            mc_steven_multiport_cf_frozen.pkl
#                            mc_steven_multiport_cf_frozen__nn.keras
#   CONFIG                 Optional. YAML config used only for path resolution.
#                          Default: configs/mc_steven_multiport_mva.yaml (relative to CWD).
#
# Outputs (alongside the input parquet):
#   <stem>_optics.parquet                  split: method==optics rows
#   <stem>_clusterfinder.parquet           split: method==clusterfinder rows
#   <stem>_optics__scored.parquet          with rf_score, gbt_score, xgb_score, nn_score
#   <stem>_clusterfinder__scored.parquet   same
#
# Activate the venv with tensorflow + xgboost + scikit-learn + joblib + pandas
# BEFORE running this script.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <DATA_FEATURES_PARQUET> [MODEL_DIR] [CONFIG]" >&2
  exit 1
fi

DATA_PARQUET="$1"
MODEL_DIR="${2:-$(cd "$(dirname "$0")" && pwd)}"
CONFIG="${3:-configs/mc_steven_multiport_mva.yaml}"

OPTICS_MODEL="${MODEL_DIR}/mc_steven_multiport_optics_frozen.pkl"
CF_MODEL="${MODEL_DIR}/mc_steven_multiport_cf_frozen.pkl"

# ── Sanity checks ─────────────────────────────────────────────────────────
for f in "$DATA_PARQUET" "$OPTICS_MODEL" "$CF_MODEL" "$CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: file not found: $f" >&2
    exit 1
  fi
done
for m in "$OPTICS_MODEL" "$CF_MODEL"; do
  nn="${m%.pkl}__nn.keras"
  if [[ ! -f "$nn" ]]; then
    echo "WARN: NN companion missing for $(basename "$m"): $(basename "$nn") — only tree scores will be produced." >&2
  fi
done

DATA_DIR="$(cd "$(dirname "$DATA_PARQUET")" && pwd)"
STEM="$(basename "$DATA_PARQUET" .parquet)"
OPT_PARQUET="${DATA_DIR}/${STEM}_optics.parquet"
CF_PARQUET="${DATA_DIR}/${STEM}_clusterfinder.parquet"

echo "=== inputs ==="
echo "  data:   $DATA_PARQUET"
echo "  models: $MODEL_DIR"
echo "  config: $CONFIG"
echo

# ── Step 1: split data parquet by clustering method ──────────────────────
echo "=== Step 1: split data by method column ==="
python - "$DATA_PARQUET" "$OPT_PARQUET" "$CF_PARQUET" <<'PYEOF'
import sys, pandas as pd
src, opt_out, cf_out = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(src)
print(f"  loaded {len(df):,} clusters from {src}")
if "method" not in df.columns:
    sys.exit("ERROR: data parquet has no 'method' column — cannot split.")
counts = df["method"].value_counts().to_dict()
print(f"  method counts: {counts}")
opt = df[df["method"] == "optics"].reset_index(drop=True)
cf  = df[df["method"] == "clusterfinder"].reset_index(drop=True)
if len(opt) == 0:
    print(f"  WARN: no OPTICS rows in data parquet")
if len(cf) == 0:
    print(f"  WARN: no ClusterFinder rows in data parquet")
opt.to_parquet(opt_out, index=False)
cf.to_parquet(cf_out,  index=False)
print(f"  wrote {len(opt):,} OPTICS rows -> {opt_out}")
print(f"  wrote {len(cf):,} CF rows     -> {cf_out}")
PYEOF
echo

# ── Step 2: score OPTICS rows ────────────────────────────────────────────
if [[ -s "$OPT_PARQUET" ]]; then
  echo "=== Step 2: score OPTICS rows with OPTICS-trained model ==="
  python mva_analysis.py \
    --config "$CONFIG" \
    --score-data "$OPT_PARQUET" \
    --model "$OPTICS_MODEL"
  echo
fi

# ── Step 3: score CF rows ────────────────────────────────────────────────
if [[ -s "$CF_PARQUET" ]]; then
  echo "=== Step 3: score ClusterFinder rows with CF-trained model ==="
  python mva_analysis.py \
    --config "$CONFIG" \
    --score-data "$CF_PARQUET" \
    --model "$CF_MODEL"
  echo
fi

OPT_SCORED="${OPT_PARQUET%.parquet}__scored.parquet"
CF_SCORED="${CF_PARQUET%.parquet}__scored.parquet"

# ── Step 4: comparison summary ───────────────────────────────────────────
echo "=== Step 4: pass-rate + per-event multiplicity comparison ==="
python - "$OPT_SCORED" "$CF_SCORED" <<'PYEOF'
import os, sys, pandas as pd

frames = {}
for label, path in [("OPTICS", sys.argv[1]), ("CF", sys.argv[2])]:
    if os.path.exists(path):
        frames[label] = pd.read_parquet(path)
    else:
        print(f"  (no scored parquet for {label}: {path})")

if not frames:
    sys.exit("No scored parquets to summarise.")

# Pass-rate table per (selection, score, threshold)
score_cols = ["rf_score", "gbt_score", "xgb_score", "nn_score"]
thresholds = [0.5, 0.7, 0.9]

print(f"\n{'selection':<8}  {'score':<10} {'thr':<5}  {'n_pass':>8}  {'n_total':>8}  {'frac':>7}")
print("-" * 60)
for name, df in frames.items():
    avail = [c for c in score_cols if c in df.columns]
    for sc in avail:
        for thr in thresholds:
            n = int((df[sc] > thr).sum())
            frac = (n / len(df) * 100) if len(df) > 0 else 0.0
            print(f"{name:<8}  {sc:<10} {thr:<5}  {n:>8d}  {len(df):>8d}  {frac:6.1f}%")
    print()

# Per-event multiplicity at gbt_score > 0.5
print("--- per-event passing-cluster counts (gbt_score > 0.5) ---")
for name, df in frames.items():
    if "eventID" not in df.columns or "gbt_score" not in df.columns:
        print(f"  {name}: skipped (eventID or gbt_score missing)")
        continue
    n_events_total = df["eventID"].nunique()
    passing = df[df["gbt_score"] > 0.5]
    n_per_event = passing.groupby("eventID").size()
    n_events_with_pass = int((n_per_event > 0).sum())
    pct = (n_events_with_pass / n_events_total * 100) if n_events_total else 0.0
    print(f"  {name:<8}: {n_events_with_pass:,} / {n_events_total:,} events have ≥1 passing cluster ({pct:.1f}%)")
    if len(n_per_event) > 0:
        dist = n_per_event.value_counts().sort_index().to_dict()
        print(f"            multiplicity distribution: {dist}")
PYEOF

echo
echo "=== Done ==="
echo "  scored parquets:"
echo "    $OPT_SCORED"
echo "    $CF_SCORED"
