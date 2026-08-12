# OPTICS Hyperparameter Sweep — Handoff

Goal: find OPTICS hyperparameters that produce **purer clusters** than the current
production config `min_samples=8, xi=0.10, t_unit_ns=25`. Pick this up in a fresh chat.

---

## Why we're doing this (the finding that motivated it)

Purity study on the full productionv2 cluster_features parquet
(`cc_neutrino_productionv2__cluster_features.parquet`, 26,585 OPTICS clusters):

- **Clusters are NOT pure** — mean neutron-hit fraction ~0.65; median ~0.73.
  Only ~2.6% of OPTICS clusters are 100% pure neutron; ~6% are pure background.
  The bulk (47.6%) sit in the 50–80%-neutron bin.
- **OPTICS over-merges**: **41.5% of ALL background hits get absorbed into
  neutron-DOMINATED clusters** (vs only 7.7% of neutron hits lost into bkg
  clusters). The asymmetry = background being swept INTO signal clusters, not
  signal being fragmented out.
- Signal clusters (dominant=neutron) average frac_neutron 0.762, frac_nonneutron
  0.139, frac_darknoise 0.098 — i.e. ~24% contamination.
- The dominant contaminant is class **-5 (non-neutron physics)**, which is 29% of
  all delayed-residual hits and is temporally/spatially close to neutron captures.

So the clusters are mixed objects given a majority-vote label. We want to see if
tighter OPTICS (smaller xi, larger min_samples, different t_unit_ns) splits the
mixed clusters into purer ones WITHOUT shredding real neutron captures.

ClusterFinder for comparison: slightly purer (51.3% of signal clusters >=80%
neutron vs OPTICS 41.7%), 37.7% bkg-into-signal. Also over-merges.

---

## The tool — `ambe mc optics` (NO custom script needed)

`ambe mc optics --config <cfg>` already sweeps the grid:
- Reads `min_samples`, `xi`, `t_unit_ns` LISTS from the config's `optics:` block
  (`src/ambe/mc/optics.py::_grid_from_ctx`).
- Re-clusters every delayed-residual event under each (ms, xi, t_unit) combo.
- Evaluates against truth trackIDs (proper capture-level matching, MIN_MATCH_FRAC
  in `event_metrics`) AND runs the ClusterFinder baseline.
- Writes `<run>__metrics.parquet` (per-event-per-config) and
  `<run>__optics_summary.csv` (ranked summary — THIS is what to bring back to chat).

---

## THE ONE GOTCHA — OOM on the full file

`train_and_evaluate` does `pd.read_parquet(pulses_path)` on the FULL pulses parquet
(`src/ambe/mc/optics.py` ~line 486). On productionv2 that's 1.4 GB / 78M rows →
~12-15 GB RAM → OOM on the 11 GB gpvm (same issue that hit Stage 1 features, which
was fixed there with a chunked read but optics.py was NOT touched).

A grid sweep re-clusters every event under N configs, so you do NOT want all 376
files anyway. **Use a 10% sample** under a separate run_name.

### Build the 10% sample (one-time)
`/tmp/make_sweep_sample.py` was drafted to do exactly this — samples ~40/408 row
groups (whole files, so every event keeps all its hits) from the productionv2
pulses + clusterfinder parquets into a new run_name `cc_neutrino_optics_sweep`:
```
out: /exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/
       cc_neutrino_optics_sweep/parquet/
         cc_neutrino_optics_sweep__pulses.parquet
         cc_neutrino_optics_sweep__clusterfinder.parquet
```
(Re-create it if /tmp was cleared — it chunk-reads the pulses by row group so it
won't OOM, and filters the small CF sidecar to the sampled events.)

---

## The sweep config to write

Create `configs/cc_neutrino_optics_sweep.yaml` — same as
`cc_neutrino_productionv2.yaml` but with run_name `cc_neutrino_optics_sweep` and a
BROAD optics grid. Current production = ms=8 xi=0.10 t_unit=25 (include it as the
baseline reference point).

```yaml
run_name: cc_neutrino_optics_sweep
display_label: "CC Neutrino — OPTICS hyperparameter sweep (10% sample)"
campaign: "OPTICS sweep on cc_neutrino_productionv2 10% sample"
output_root: /exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output

inputs:
  root_files:
    - /pnfs/annie/persistent/users/dajana/output/genie_wcsim_tank/gridtest/ANNIEEvent_cc_neutrino_*.root
  # NOTE: optics reads the PARQUET (pulses), not ROOT — inputs here are unused by
  # `ambe mc optics`. The sample parquet must already exist under this run_name.

cuts:
  min_pulses_per_event: 0
  cc_stream: ccinc_truth
  cc_cuts:
    require_cc: true
    require_fsl_muon: true
    mu_p_min_mev: 600.0
    mu_p_max_mev: 1200.0
    cos_theta_min: 0.8
    require_fv: true
    fv_radius_cm: 100.0
    fv_y_max_cm: 100.0
    fv_require_z_negative: false
    nhit_min: 4

residual:
  cc_only: true
  prompt_window_ns: 2000

optics:
  # BROAD grid — 5 x 5 x 4 = 100 configs. Trim if too slow on 10%.
  min_samples: [5, 8, 12, 15, 20]
  xi:          [0.02, 0.05, 0.10, 0.15, 0.20]
  t_unit_ns:   [10, 25, 50, 75]
  truth_window_ns: 75.0
  hit_prefilter_ns: 0.0     # 0 = cluster ALL residual hits (clean purity study)

geometry:
  pmt_geometry:   /exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/FullTankPMTGeometry.csv
  timing_offsets: /exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/TankPMTTimingOffsets.csv
```

Tuning intuition for fixing the over-merge:
- **larger min_samples** (12-20) → fewer, denser clusters; rejects loose bkg halos
- **smaller xi** (0.02-0.05) → steeper reachability cut → splits merged blobs
- **smaller t_unit_ns** (10) → tighter in time → separates the prompt-ish bkg tail
  from the true capture; **larger t_unit_ns** (75) → merges more in time (worse for
  purity, but check recall)

---

## Run it (in tmux, takes a while — 100 configs x ~thousands of events)

```bash
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate

# 1. build the 10% sample (one-time)
python3 /tmp/make_sweep_sample.py     # or re-create from the snippet in this doc

# 2. run the sweep
ambe mc optics --config configs/cc_neutrino_optics_sweep.yaml

# outputs:
#   <out>/cc_neutrino_optics_sweep/parquet/cc_neutrino_optics_sweep__metrics.parquet
#   <out>/cc_neutrino_optics_sweep/csv/cc_neutrino_optics_sweep__optics_summary.csv
```

Bring `cc_neutrino_optics_sweep__optics_summary.csv` back to chat to pick the winner.

---

## What to look at in the summary (per config)

`event_metrics` / `summarise` report per-config metrics. Key ones for THIS goal:
- **purity** of clusters (neutron-hit fraction) — want HIGHER than baseline 0.65
- **recall / efficiency** of neutron captures — must NOT drop much (don't trade all
  the signal away for purity)
- **n_clusters / fragmentation** — watch for over-splitting (too many tiny clusters)
- compare every config against the **ms=8 xi=0.10 t_unit=25 baseline row**

The winner = highest purity / lowest bkg-contamination that keeps capture recall
comparable to baseline. If NO config beats baseline meaningfully, that's also a
valid finding: the over-merge is intrinsic (bkg physically overlaps the capture),
and the answer is to lean on the MVA's aggregate-feature separation + the
dominant_class labeling rather than cluster purity.

---

## Reference paths
| What | Path |
|---|---|
| Source | `/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/src/ambe/mc/optics.py` |
| Full features (purity study source) | `.../ambe_output/cc_neutrino_productionv2/parquet/cc_neutrino_productionv2__cluster_features.parquet` |
| Sample sampler | `/tmp/make_sweep_sample.py` |
| venv | `source /exp/annie/app/users/dajana/myboy/bin/activate` |
