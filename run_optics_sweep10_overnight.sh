#!/usr/bin/env bash
# Unattended overnight runner: 10% OPTICS sweep (36-config grid) + composition-cut
# efficiency on the top configs. Safe to launch in a detached tmux and walk away.
#
#   tmux new-session -d -s optics10 'bash run_optics_sweep10_overnight.sh'
#   tail -f logs_optics_sweep10.log
#
# Stages:
#   A) ambe mc optics  on the 10% sample (run_name cc_neutrino_optics_sweep) — 36 configs
#   B) pick top-5 OPTICS configs from the summary CSV (best purity at recall>=baseline-0.05,
#      plus the lowest-spurious one), write per-config feature YAMLs (symlink 10% sample in)
#   C) ambe mc features for each top config
#   D) full composition-cut grid (n_neutron x frac_nonneutron) per config -> table in the log
set -uo pipefail   # NOTE: not -e — we want the script to finish all configs even if one fails

REPO=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis
VENV=/exp/annie/app/users/dajana/myboy/bin/activate
OUT_ROOT=/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output
SWEEP_CFG=configs/cc_neutrino_optics_sweep10.yaml
SWEEP_RUN=cc_neutrino_optics_sweep
LOG="${REPO}/logs_optics_sweep10.log"
SUMMARY="${OUT_ROOT}/${SWEEP_RUN}/csv/${SWEEP_RUN}__optics_summary.csv"

cd "$REPO"
source "$VENV"
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "  10% OPTICS sweep + composition-cut runner — start $(date)"
echo "=================================================================="

# ---- Stage A: 36-config sweep on the 10% sample ----
echo; echo "--- Stage A: ambe mc optics (36 configs, 10% sample) ---"
ambe mc optics --config "$SWEEP_CFG"
if [[ ! -f "$SUMMARY" ]]; then
  echo "FATAL: sweep summary not written ($SUMMARY) — aborting."; exit 1
fi

# ---- Stage B: pick top configs + write feature YAMLs (Python does the selection) ----
echo; echo "--- Stage B: pick top-5 configs + write feature YAMLs ---"
SEL_FILE="${OUT_ROOT}/${SWEEP_RUN}/csv/top_configs_selected.txt"
python3 - "$SUMMARY" "$OUT_ROOT" "$SWEEP_RUN" "$REPO" "$SEL_FILE" <<'PY'
import sys, os
import pandas as pd
summary, out_root, sweep_run, repo, sel_file = sys.argv[1:6]
df = pd.read_csv(summary)
op = df[df['method'] == 'optics'].copy()
base = op[(op['min_samples']==8)&(op['xi']==0.10)&(op['t_unit_ns']==25.0)]
b_rec = float(base['mean_recall'].iloc[0]) if len(base) else op['mean_recall'].max()
# eligible: recall not collapsed (>= baseline - 0.05); top-4 by purity + lowest-spurious + baseline
elig = op[op['mean_recall'] >= b_rec - 0.05]
top_pur = elig.sort_values('mean_purity', ascending=False).head(4)
low_spur = op.sort_values('mean_real_spurious').head(1)
sel = pd.concat([base, top_pur, low_spur]).drop_duplicates(subset=['min_samples','xi','t_unit_ns']).head(6)

tmpl = """run_name: {run}
display_label: "OPTICS feat {label} (10% sample) ms={ms} xi={xi} t={t}"
campaign: "Composition-cut validation of OPTICS config {label} (10% sample)"
output_root: {out}
inputs:
  root_files:
    - /pnfs/annie/persistent/users/dajana/output/genie_wcsim_tank/gridtest/ANNIEEvent_cc_neutrino_*.root
cuts:
  min_pulses_per_event: 0
  cc_stream: ccinc_truth
  cc_cuts: {{require_cc: true, require_fsl_muon: true, mu_p_min_mev: 600.0, mu_p_max_mev: 1200.0, cos_theta_min: 0.8, require_fv: true, fv_radius_cm: 100.0, fv_y_max_cm: 100.0, fv_require_z_negative: false, nhit_min: 4}}
fit_params: {{}}
residual:
  cc_only: true
  prompt_window_ns: 2000
optics:
  min_samples: [{ms}]
  xi: [{xi}]
  t_unit_ns: [{t}]
  truth_window_ns: 75.0
  hit_prefilter_ns: 2000.0
features:
  min_samples: {ms}
  xi: {xi}
  t_unit_ns: {t}
  hit_prefilter_ns: 2000.0
  truth_window_ns: 75.0
geometry:
  pmt_geometry:   /exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/FullTankPMTGeometry.csv
  timing_offsets: /exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/TankPMTTimingOffsets.csv
"""
src = f'{out_root}/{sweep_run}/parquet'
lines = []
for _, r in sel.iterrows():
    ms = int(r['min_samples']); xi = float(r['xi']); t = float(r['t_unit_ns'])
    label = f"ms{ms}_xi{int(round(xi*100)):02d}_t{int(t)}"
    run = f"sweep10_feat_{label}"
    pdir = f'{out_root}/{run}/parquet'
    os.makedirs(pdir, exist_ok=True)
    for kind in ['pulses', 'clusterfinder']:
        link = f'{pdir}/{run}__{kind}.parquet'
        target = f'{src}/{sweep_run}__{kind}.parquet'
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
    with open(f'{repo}/configs/{run}.yaml', 'w') as f:
        f.write(tmpl.format(run=run, label=label, ms=ms, xi=xi, t=t, out=out_root))
    lines.append(f"{run}\t{ms}\t{xi}\t{t}")
    print(f"  selected {run}: ms={ms} xi={xi} t={t}  (purity={r['mean_purity']:.4f} recall={r['mean_recall']:.4f})")
with open(sel_file, 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f"  wrote selection -> {sel_file}")
PY

if [[ ! -f "$SEL_FILE" ]]; then echo "FATAL: no config selection written — aborting."; exit 1; fi

# ---- Stage C: features for each selected config ----
echo; echo "--- Stage C: ambe mc features per selected config ---"
while IFS=$'\t' read -r RUN MS XI T; do
  [[ -z "$RUN" ]] && continue
  echo; echo "=== features: $RUN (ms=$MS xi=$XI t=$T) ==="
  ambe mc features --config "configs/${RUN}.yaml" || echo "WARN: features failed for $RUN (continuing)"
done < "$SEL_FILE"

# ---- Stage D: full composition-cut grid per config ----
echo; echo "--- Stage D: composition-cut grid (n_neutron x frac_nonneutron) per config ---"
while IFS=$'\t' read -r RUN MS XI T; do
  [[ -z "$RUN" ]] && continue
  FEAT="${OUT_ROOT}/${RUN}/parquet/${RUN}__cluster_features.parquet"
  [[ ! -f "$FEAT" ]] && { echo "skip $RUN: no cluster_features"; continue; }
  echo; echo "############## CUT GRID: $RUN (ms=$MS xi=$XI t=$T) ##############"
  python3 - "$FEAT" "$RUN" <<'PY'
import sys
import pandas as pd
feat_path, run = sys.argv[1], sys.argv[2]
feat = pd.read_parquet(feat_path)
sub = feat[feat['method'] == 'optics'].copy()
mo = int((sub['is_truth_neutron']==1).sum()); so = int((sub['is_truth_neutron']==0).sum())
tot_h = int(sub['n_hits'].sum()); tot_n = int(sub['n_neutron'].sum())
NN=[1,2,3,4,5,8,10,9999]; FN=[0.30,0.40,0.50,0.60,0.70,0.80,0.90]
print(f"{run}: OPTICS clusters={len(sub)}  matched={mo}  spurious={so}")
print(f"NO-CUT hit-purity = {100*tot_n/tot_h:.1f}%  (sig.eff=100% spur.rej=0%)")
print(f"{'nn<=':>6} {'fn>=':>6} {'Sig.eff':>9} {'Purity':>9} {'Spur.rej':>9} {'kept':>8}")
print('-'*60)
rows=[]
for nn in NN:
    for fn in FN:
        rej=(sub['n_neutron']<=nn)&(sub['frac_nonneutron']>=fn)
        kept=sub[~rej]
        mk=int((kept['is_truth_neutron']==1).sum()); sk=int((kept['is_truth_neutron']==0).sum())
        kh=int(kept['n_hits'].sum()); kn=int(kept['n_neutron'].sum())
        sig=mk/mo if mo else float('nan'); pur=kn/kh if kh else float('nan')
        rr=1-sk/so if so else float('nan')
        rows.append((nn,fn,sig,pur,rr,len(kept)))
# sort by purity among sig.eff>=0.99 (the efficiency-preserving regime), then print all
for nn,fn,sig,pur,rr,nk in sorted(rows,key=lambda r:(r[3] if r[2]>=0.99 else -1),reverse=True):
    lbl='inf' if nn==9999 else str(nn)
    star=' *' if sig>=0.99 else ''
    print(f"{lbl:>6} {fn:>6.2f} {100*sig:8.1f}% {100*pur:8.1f}% {100*rr:8.1f}% {nk:>8}{star}")
print("(* = sig.eff>=99%; rows sorted by purity within that regime)")
PY
done < "$SEL_FILE"

echo; echo "=================================================================="
echo "  ALL DONE — $(date)"
echo "  summary CSV : $SUMMARY"
echo "  selection   : $SEL_FILE"
echo "  cut grids   : above in this log ($LOG)"
echo "=================================================================="
