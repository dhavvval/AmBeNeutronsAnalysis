# Configuration guide

All `ambe` CLI configs require `run_name`. Paths are resolved from the working
directory, so run from the repository root. Copy a config, update paths and use a
new tag/run name before processing. The same YAML reader accepts several workflow
blocks; a command uses only the blocks it understands.

| Workflow | Blocks to set | Starting config |
|---|---|---|
| Real-data extraction | `stage1`, `cuts`; `mva` for scored selection | `data_ambe2v4_all28_box.yaml`, `data_ambe2v4_all28_mva.yaml` |
| Special-run extraction | As above plus `stage1.extra_source_positions` | `data_ambe2v4_special_box.yaml` / `_mva.yaml` |
| Latest source campaign | `stage1`, `cuts` | `data_ambe2v5sandi_box.yaml` |
| Source/background efficiency fit | `inputs`, `efficiency`, plotting blocks | `data_ambe2v4_effpair_box.yaml` |
| Plot existing products | `inputs`, `cuts`, `fit_params` | `data_ambe2v4_all28.yaml` |
| IC-window diagnostic | `icscan`, saved feature/index paths | `data_ambe2v4_port5center_icscan.yaml` |
| MC ROOT extraction and CC selection | `inputs.root_files`, `cuts.cc_stream`, `cuts.cc_cuts`, `residual` | `cc_neutrino_v3_truthtag.yaml`, `cc_neutrino_v3_recotag.yaml` |
| OPTICS grid search | Matching extraction `run_name`, `optics` lists | `cc_neutrino_optics_sweep10.yaml` / `sweep3.yaml` |
| Feature extraction | `features` scalars, `geometry` | `sweep10_feat_*.yaml`, `sweep3_feat_*.yaml` |
| AmBe MC candidate selection | `stage1_mc`, `cuts`; `mva` if used | `mc_ambe_qe150_*_veto*.yaml` |
| Michel features | `michel`, `geometry`; `features` for tagged-hit OPTICS | `mc_michel_mva.yaml` |

## A real-data config

```yaml
run_name: my_data_box
output_root: ./ambe_output
stage1:
  data_directory: /path/to/beamcluster
  waveform_dir: /path/to/waveforms
  file_pattern: 'BeamCluster_(\d+)\.root'
  campaign: 1                     # 1: RWM_; 2: BRF_
  which_tree: 1                   # ANNIEEventTreeMaker
  tag: my_data_box
  ic_min: 700
  ic_max: 1200
  dump_event_index: true
  dump_good_events: true
  plot_ic: true
  legacy_ordering: false
cuts:
  pe_max: 100
  charge_balance_max: 0.45
  time_min_ns: 2000
  time_max_ns: 67000
  cluster_hits_min: 5
inputs:
  candidate_csvs:
    - EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_my_data_box_*.csv
  trigger_summary:
    - TriggerSummary/AmBeTriggerSummary_my_data_box.csv
fit_params: {}
```

The IC bounds are strict. Cluster time bounds are inclusive, in ns. Source positions
in `stage1.extra_source_positions` use cm; `features.source_position_m` uses metres.
IC thresholds must suit the campaign; 700–1200 is the current v4 example. The event
veto uses the existing shared `CutCriteria` defaults (<2000 ns or >100 PE), independently
of box PE bounds; changing `cuts.pe_max` does not change the event-veto threshold.
`selection_mode: ic` is accepted for compatibility; other waveform modes are rejected.

For MVA add `mva.scored` (a list of parquet paths), `mva.threshold`, and optionally
`mva.score_col`/`point`. Scores must have `run`, `event_tank_time`, `cf_clusterTime`
and the selected score column. Do not train on the AmBe evaluation sample merely
to fill missing scores; use the frozen artifact on the required feature cohort.

## OPTICS settings

`optics.min_samples`, `optics.xi`, and `optics.t_unit_ns` are lists for a sweep.
`features.min_samples`, `features.xi`, and `features.t_unit_ns` are scalars for the
chosen setting. `geometry.pmt_geometry` and `geometry.timing_offsets` supply the
geometry/timing tables. Keep the residual/prefilter/truth-window definitions with
the setting when comparing results. A feature config is not a new extraction unless
you explicitly run `mc process` with it; reuse or populate its matching input products.

`cc_neutrino_v3_truthtag` and `_recotag` select different event samples; neither
should be applied as an additional cut on the other. Their pilot configs are kept
because `scripts/pilots.sh` constructs those names dynamically. The `v3world`
variants and `steven_multiport/` are retained as inputs to merged training workflows.

## Retention decisions

The current deck configs, optics grids and selected-setting configs are retained,
as are configs named by executable launchers and their dynamic pilot/multiport
families. Local paths alone are not evidence that a config is unused. This is a
source-reference audit, not a record of how often you ran each file.

Fourteen obsolete/test configs were removed from this branch. Their exact contents
remain on `feature/blessedPlotcuts`; see [cleanup record](CLEANUP_PLAN.md)
for the list. Historical reports can still mention those names. They are evidence
of earlier analyses, not the current command guide. Read [workflow guide](WORKFLOWS.md)
for maintained commands and [script index](SCRIPTS.md) for specialised studies.
