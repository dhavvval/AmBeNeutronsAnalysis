# AmBeNeutronsAnalysis

ANNIE experiment AmBe neutron analysis toolkit. Covers:

- **Data pipeline** — real-data waveform processing, efficiency heatmaps, capture-time fits
- **MC pipeline** — ROOT → per-pulse Parquet, OPTICS clustering vs ClusterFinder baseline
- **Statistics** — Bayesian and profile-likelihood fits

---

## Installation

```bash
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
pip install -e .
```

Optional Bayesian fitting dependencies (PyMC/ArviZ):

```bash
pip install -e ".[bayes]"
```

After installation the `ambe` command is available in your shell.

---

## Output location

All outputs (plots, Parquet, CSV) are written **outside the repo** so they are never accidentally committed.

Priority order:

1. `output_root:` key in your config YAML
2. `$AMBE_OUT` environment variable
3. `~/ambe_analysis_output/`

Set it once and forget:

```bash
export AMBE_OUT=/exp/annie/persistent/users/dajana/ambe_output
```

Each run gets its own subdirectory: `$AMBE_OUT/<run_name>/plots/`, `.../parquet/`, `.../csv/`.

---

## Quick start

### Real-data analysis

1. Copy the example config and edit it:

```bash
cp configs/example_data_amb2v4.yaml configs/my_run.yaml
# edit run_name, campaign, input paths, cuts
```

2. Run the full data pipeline:

```bash
ambe pipeline data --config configs/my_run.yaml
```

Or run stages individually:

```bash
ambe data process   --config configs/my_run.yaml   # waveform processing -> CSVs
ambe plots combined --config configs/my_run.yaml   # capture-time fit, multiplicity, ...
ambe plots heatmap  --config configs/my_run.yaml   # efficiency heatmaps + residuals
ambe stats core     --config configs/my_run.yaml   # statistical summary
```

### MC OPTICS training

1. Make sure `ANNIEEventTreeMakerConfig` has these flags set to `1`:
   - `TankHitInfo_fill` — provides chankey → (x,y,z) lookup
   - `DirectParent_MCHit_fill` — per-pulse truth labels (NeutronAncestorClass)
   - `TankCluster_fill` — ClusterFinder output for baseline comparison

2. Run the MC pipeline on `BeamCluster_*.root` files:

```bash
cp configs/example_mc_optics.yaml configs/mc_trial01.yaml
# edit root_files glob, optics hyperparameter grid

ambe pipeline mc --config configs/mc_trial01.yaml
# or step-by-step:
ambe mc process --config configs/mc_trial01.yaml   # ROOT -> Parquet
ambe mc optics  --config configs/mc_trial01.yaml   # OPTICS sweep + ClusterFinder baseline
```

---

## Config YAML reference

```yaml
run_name: my_run           # short identifier, used in every output filename
campaign: "AmBe 2.0v4"    # appears in every plot title
description: "..."         # free text, printed at startup

# Optional: override output root for this run only
# output_root: /some/path

inputs:
  candidate_csvs:
    - EventAmBeNeutronCandidatesData/*.csv
  trigger_summary:
    - TriggerSummary/AmBeTriggerSummary_v4.csv
  root_files:                      # MC only
    - /path/to/BeamCluster_*.root

cuts:
  pe_max: 100.0
  charge_balance_max: 0.45
  min_pulses_per_event: 3          # MC only

fit_params:
  time_bins: 70
  time_range: [0, 70]
  fit_min_time: 2.0
  fit_max_time: 67.0
  initial_amplitude: 200.0
  initial_thermal_time: 5.0
  initial_capture_time: 25.0
  initial_background: 0.0

reference:                         # for residual heatmap
  comparison_campaign: "AmBe 1.0"

optics:                            # MC only; swept as a grid
  min_samples: [5, 10, 20]
  xi: [0.01, 0.05, 0.10]
  t_scale: [null, 1.0, 5.0]       # null = StandardScaler only
```

---

## Package layout

```
src/ambe/
  cli.py             # `ambe` entry point
  context.py         # RunContext dataclass (config -> paths + title helpers)
  paths.py           # output directory resolution ($AMBE_OUT hierarchy)
  plotting.py        # set_style(), save_plot()
  io.py              # glob expansion, CSV loading helpers
  pipeline.py        # one-shot pipeline drivers

  mc/
    processor.py     # ROOT -> per-pulse Parquet + ClusterFinder sidecar
    optics.py        # OPTICS hyperparameter sweep + baseline metrics
    basic_plots.py   # summary plots on MC output
    wcsim.py         # WCSim-specific utilities

  data/
    processor.py     # real-data waveform -> neutron candidate CSVs
    eff.py           # efficiency calculation helpers
    analysis_run.py  # run-level bookkeeping

  plots/
    basic.py         # 1D/2D histograms, capture-time fits (AmBeNeutronAnalyzer)
    combined.py      # multiplicity, PE vs CB, capture-time fit (ctx-native)
    heatmap.py       # efficiency + statistics + residual heatmaps (ctx-native)
    phase2.py        # Phase II campaign plots

  clustering/
    optics_data.py   # per-event OPTICS on data CSVs
    optics_analysis.py
    dbscan.py
    knn.py
    isolation_forest.py

  stats/
    core.py          # statistical summary plots
    profile_likelihood.py
    waveform.py
```

Truth-label class codes (from `BackTracker.cpp`):

| Code | Meaning |
|------|---------|
| 0    | Dark noise |
| 1    | Primary neutron |
| 2    | Secondary neutron ← proton |
| 3    | Secondary neutron ← neutron |
| 4    | Secondary neutron ← other |
| −5   | Non-neutron physics background |

---

## Development

```bash
pip install -e ".[dev]"
ruff check src/
pytest tests/
```
