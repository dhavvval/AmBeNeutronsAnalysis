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

**There is one output tree for all server-side work:**

```
/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output
```

Every plot, Parquet and CSV lands under it, and each run gets its own subdirectory:
`<output_root>/<run_name>/plots/`, `.../parquet/`, `.../csv/`. Data-campaign runs sit
one level deeper, under `ambe_output/ambe_data/<run_name>/`.

Outputs live inside this directory but are **never repo content** — `.gitignore` has an
explicit rule for each output tree, so 20+ GB can sit here without ever being staged.

Priority order for resolving the root:

1. `output_root:` key in your config YAML  ← what all 49 server-side configs use
2. `$AMBE_OUT` environment variable
3. `~/ambe_analysis_output/`

### If you have older scripts or notes

Until 2026-09-09 the outputs were split across **three** locations, so no path could be
guessed:

| old location | held |
|---|---|
| `AmBeNeutronAnalysis/` (no "s") | `ambe_output/`, `logs/`, `beamoff/`, `mc_steven/` |
| `AmBeNeutronsAnalysis/` (this one) | `TriggerSummary/`, `verbose/`, `EventAmBeNeutronCandidatesData/` |
| `/exp/annie/app/users/dajana/ambe_output` | a few MC workspace runs |

All three are merged here. **Both old paths are now symlinks to this tree**, so older
scripts, notes and job definitions keep resolving rather than failing:

```
/exp/annie/app/users/dajana/AmBeNeutronAnalysis -> AmBeNeutronsAnalysis
/exp/annie/app/users/dajana/ambe_output         -> AmBeNeutronsAnalysis/ambe_output
```

Configs whose `output_root` is under `/Users/...` or `/sessions/...` are laptop and
sandbox runs; those are deliberately left pointing off-server.

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
ambe data icscan    --config configs/my_run.yaml --mode differential  # IC-window scan

ambe plots combined --config configs/my_run.yaml   # capture-time fit, multiplicity, ...
ambe plots heatmap  --config configs/my_run.yaml   # efficiency heatmaps + residuals
ambe stats core     --config configs/my_run.yaml   # statistical summary
```

### IC-window scan

`ambe data icscan` re-tallies the efficiency for many Stage-1 IC windows offline, with
no further waveform read. It needs a config whose `stage1` block sets
`dump_event_index: true` and whose IC gate is at least as wide as the scan range --
see `configs/data_ambe2v4_port5center_icscan.yaml`. A waveform rejected at Stage 1 is
never Stage-2 processed, so the scan can only ever narrow the pass window, never widen
it; `icscan` asserts that rather than silently returning zeros.

`--mode` is required and picks the question:

```bash
ambe data icscan --config <cfg> --mode closure       # re-tally the production window,
                                                     # check cell-for-cell vs the frozen summary
ambe data icscan --config <cfg> --mode differential  # disjoint slices -- INDEPENDENT points,
                                                     # the only mode a flatness test is valid on
ambe data icscan --config <cfg> --mode left          # walk the lower edge, upper fixed
ambe data icscan --config <cfg> --mode right         # walk the upper edge, lower fixed
ambe data icscan --config <cfg> --mode widthsweep    # differential at widths 50..500
ambe data icscan --config <cfg> --mode grid          # every arbitrary (lower, upper) pair
```

`differential` takes `--width` to set the slice width (default `icscan.step`). Slice
width is the scan RESOLUTION: structure narrower than a slice is averaged away, while
wider slices have smaller error bars. Those pull chi2/dof in opposite directions, so
**chi2/dof is not comparable between widths** — `widthsweep` prints one row per width
for exactly that reason.

`left`/`right` windows are nested and so share most of their events: their error bars
are per-point only, never point-to-point. Run `closure` first -- if it does not pass,
no other mode's output means anything.

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
