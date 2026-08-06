# Run the steven MVA pipeline on AmBe data using the lucho100k frozen models — BOTH methods

## Context

The `lucho100k` MC run finished cleanly (rc=0) and froze working models
(`mc_lucho_100k_{optics,cf}_frozen.pkl` + NN companions). The goal is to replicate
`run_ambe_steven_mva.sh` — the 5-stage data pipeline already run for the steven-multiport model —
but driven by the lucho frozen models, producing the full output set for **both methods (optics +
clusterfinder)** so the trees (rf/gbt/xgb) can be compared.

**Why this grew beyond simple script-wiring:**
1. The frozen lucho MVA is **weak** — test AUC ≈ 0.70 (RF 0.707 best, NN 0.693 worst); the signal
   cap left only ~4.5k signal vs ~1.5k background. We run it as-is (user decision); just flag the
   caveat in results.
2. The lucho freeze saved **only clusterfinder-method MC scores** (`mc_lucho_100k__mva_scores.parquet`
   = 4,716 CF rows, 0 optics rows) → the optics MC anchor is missing (NaN threshold). Must be
   regenerated.
3. The AmBe **DATA** features parquet is **100% optics** (232,836 optics rows, 0 CF rows) — data was
   extracted with OPTICS only. So there is no CF data to score yet. **User pointed to the BeamCluster
   ntuples** at `/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/BeamCluster_<run>.root` (33 runs)
   which already contain ClusterFinder clusters per event → we extract CF-method data features from
   those and add them to the workflow.

## Key technical findings (verified)

- **Feature extractor is method-agnostic:** `compute_cluster_features(df_cluster, geo, source_pos_m)`
  in `src/ambe/mc/cluster_features.py:502` takes a per-hit dataframe `[x,y,z,t,pe,pmtID]` and emits
  the same 47 features (33 feed the MVA) regardless of OPTICS vs CF origin.
- **OPTICS data worker** = `analyze_optics_beamcluster_data.py:extract_features_one_file()` (605-712).
  It reads BeamCluster files, applies a **cosmic veto** (any CF cluster with clusterTime<2000ns OR
  clusterPE>100 → drop whole event), a **prompt-window cut** (keep t>2000ns), runs OPTICS, then calls
  `compute_cluster_features(..., source_pos_m=None)`. The CF extractor must mirror every one of these
  conventions exactly (geometry CSVs, `source_pos_m=None`, per-run port/source via
  `SOURCE_POSITIONS`, `charge_balance_legacy`, `passes_preselection`, bookkeeping columns) so the 33
  columns line up with what `mc_lucho_100k_cf_frozen.pkl` expects.
- **The BeamCluster file already stores hit→cluster membership** as doubly-jagged branches:
  `Cluster_HitX/Y/Z/T/Q/PE/Chankey` have awkward type `event * var(cluster) * var(hit) * float`.
  So for event i, cluster j: hits are `Cluster_HitX[i][j]`, `Cluster_HitT[i][j]`, etc. The existing
  `load_file()` (263-306) only reads FLAT per-event hit arrays + cluster summaries — it does NOT read
  these nested per-cluster hit arrays. That's the one new read path needed.
- **Score column derivation** (`summarize_ambe_neutrons.py:mc_threshold`, line 41) keys on
  `dominant_class ∈ {1,2,3,4}` and `in_test`. The lucho MC scores have these. CF-method thresholds
  (already computable from the existing file): rf=0.6164, gbt=0.5086, xgb=0.5513 @ 80% sig-eff.

## Approach

### Part A — Produce CF-method DATA features from BeamCluster ntuples (NEW code)

New script `extract_cf_features_beamcluster_data.py` in the repo, closely modeled on
`analyze_optics_beamcluster_data.py`'s feature path. Per file/run:
- Read the `Event` tree's nested cluster-hit branches with uproot/awkward
  (`numberOfClusters`, `clusterTime/PE/ChargeBalance/Hits`, `Cluster_HitX/Y/Z/T/Q/Chankey`).
- Apply the **same** event gating + cosmic veto + (NO prompt cut on CF hits — the CF clusters are
  already the reconstructed objects; the prompt-window cut is an OPTICS-input step. Mirror what the
  MC CF path does in `cluster_features.py:extract_all_features` — CF clusters are taken as-is, not
  re-filtered by t>2000. This must be matched to MC, not to the OPTICS data worker. **Verify against
  the MC CF path before finalizing** — see Open item below.)
- For each CF cluster build `df_clust = [x,y,z,t,pe,pmtID]` from `Cluster_Hit*[i][j]`
  (rename Chankey→pmtID), call `compute_cluster_features(df_clust, geo, source_pos_m=None)`.
- Emit rows with the SAME bookkeeping columns as the optics worker (`run, port, event_number,
  event_tank_time, cluster_id, clusterTime_earliest, passes_stage1`) + the 47 features +
  `method="clusterfinder"`.
- Reuse, not reimplement: `load_geometry`, `SOURCE_POSITIONS`, `port_name`,
  `charge_balance_legacy`, `passes_preselection`, `run_number_from_path` (import from the optics
  script or its shared modules).
- Output `ambe_all__data_features_cf.parquet`, then concatenate with the existing optics parquet
  into `ambe_all__data_features_bothmethods.parquet` (optics rows tagged method='optics', CF rows
  method='clusterfinder').

### Part B — Regenerate the missing optics MC scores anchor

Score the optics MC cluster_features through `mc_lucho_100k_optics_frozen.pkl` to produce
`mc_lucho_100k__mva_scores_optics.parquet` (carrying dominant_class + in_test + rf/gbt/xgb/nn_score),
giving a valid optics threshold. Use `mva_analysis.py --score-data --model` (the optics MC features
already exist in `mc_lucho_100k__cluster_features.parquet`, method='optics' subset).
The CF MC anchor already exists (`mc_lucho_100k__mva_scores.parquet`).

### Part C — The two-method pipeline runner (`run_ambe_lucho_mva.sh`)

Mirror `run_ambe_steven_mva.sh`. Differences handled: call `mva_analysis.py --score-data --model`
directly (the shared `score_real_data.sh` hard-codes steven filenames at lines 43-44 — leave it
untouched). For EACH method (optics, clusterfinder):
- **Stage 1** score the method's data rows with the matching lucho frozen pkl →
  `ambe_all__data_features_<method>__scored_lucho.parquet`.
- **Stage 2** per-run neutron-rate summary ×3 scores (rf/gbt/xgb, sig-eff 0.80), anchored on that
  method's lucho MC scores → `lucho_<method>_rate_{rf,gbt,xgb}/`.
- **Stage 3** multiplicity + score PDF (`--mc-features` = lucho MC features filtered to that method).
- **Stage 4** capture-time lmfit with a **lucho+method-derived** rf threshold (from Stage 2; NOT
  steven's 0.8423) → `capture_stageAB_lucho_<method>/`.
- **Stage 5** data/MC KS comparison (`--mc-method <method>`) → `compare_selections_lucho_<method>/`.

Outputs land under `/exp/.../AmBeNeutronAnalysis/ambe_output/ambe_data/` (the mount with free space).
Run in tmux, `set -euo pipefail`, tee to `logs_ambe_lucho_mva.log`.

## Files

- **New:** `extract_cf_features_beamcluster_data.py` (Part A), `run_ambe_lucho_mva.sh` (Part C).
- **Reused unchanged:** `compute_cluster_features` + helpers in `src/ambe/mc/cluster_features.py`,
  `analyze_optics_beamcluster_data.py` helpers (imported), `mva_analysis.py`,
  `summarize_ambe_neutrons.py`, `plot_ambe_neutron_multiplicity.py`,
  `capture_time_stageAB_from_parquet.py`, `analyze_data_mc_comparison.py`.
- **No edits** to `score_real_data.sh` or any steven config.

## Open item to resolve during implementation (do NOT guess)

The MC CF path (`cluster_features.py:extract_all_features`, ~848-998) defines how CF clusters are
built in MC — in particular whether CF clusters are subject to the t>2000ns residual filter or taken
whole. Part A's CF data extractor MUST replicate the MC CF treatment (not the OPTICS data treatment)
so data and the CF frozen model are on the same footing. Read that function first and match it
exactly; this is the single correctness-critical detail.

## Verification

- `bash -n run_ambe_lucho_mva.sh`; `python -c "import ast; ast.parse(open('extract_cf_features_beamcluster_data.py').read())"`.
- Smoke-test Part A on ONE BeamCluster file (`--max-events` small) → confirm the 33 feature columns
  exist, dtypes match the optics parquet, `method=='clusterfinder'`, and row count is sane vs the
  file's total `clusterHits`.
- Confirm regenerated `mc_lucho_100k__mva_scores_optics.parquet` has nonzero optics rows and the
  derived optics threshold is finite.
- Launch full run in tmux; tail the log. Per method, confirm non-empty:
  scored parquet (rf/gbt/xgb/nn_score), `lucho_<method>_rate_*` CSVs+PDFs, multiplicity PDF,
  `capture_*` lmfit summary CSV, `compare_*` KS CSV+PDF.

## Caveats to surface in results

- Frozen lucho MVA AUC ≈ 0.70 → lower-purity selection than steven; valid as a lucho comparison set.
- CF-method data features are NEWLY produced here (didn't exist before) — call out that the CF data
  pipeline depends on the new extractor and was validated by the smoke-test above.
