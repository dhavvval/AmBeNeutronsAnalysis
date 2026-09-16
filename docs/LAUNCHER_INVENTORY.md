# Launcher and script inventory

Static audit of the checked-in Bash launchers after the September 2026 cleanup. “Referenced” means a command or an import is present, not that a job ran successfully. Commented commands are excluded. Conditional stages count as references. ROOT inputs and server execution logs were not supplied.

## Counts

| Group | Modules | Directly invoked by Bash | Additional imported dependencies | No Bash path found |
|---|---:|---:|---:|---:|
| Diagnostics (`src/ambe/diagnostics`) | 20 | 5 | 0 | 15 |
| Workflow helpers (`scripts/workflows`) | 12 | 7 | 0 | 5 |

The former `scripts/diagnostics` contained 20 modules. Eighteen now share the
packaged diagnostics directory with `runtable` and `compare_veto`; two redundant
capture-time-only modules were deleted. `datadriven_chi2` moved to
`efficiency_extraction` and is counted there, not here. Package `__init__.py` files
are excluded.

## Launcher dependencies

Names in the next table omit their `ambe.diagnostics.` or `scripts.workflows.` prefixes. It includes Python imports and calls through other checked-in shell scripts. “None” can mean a launcher calls the main `ambe` CLI or MVA module directly.

| Launcher | Diagnostics reached | Workflow helpers reached |
|---|---|---|
| `run_ambe_lucho_mva.sh` | `analyze_data_mc_comparison`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity` | `analyze_optics_beamcluster_data`, `extract_cf_features_beamcluster_data`, `summarize_ambe_neutrons` |
| `run_ambe_mva_allruns.sh` | `plot_ambe_neutron_multiplicity` | `analyze_optics_beamcluster_data`, `summarize_ambe_neutrons` |
| `run_ambe_steven_mva.sh` | `analyze_data_mc_comparison`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity` | `summarize_ambe_neutrons` |
| `run_ccinc_productionv2.sh` | `benchmark_selection_2x2`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity` | `pick_best_model_cut`, `summarize_ambe_neutrons` |
| `run_ccinc_provenance_tmux.sh` | None | None |
| `run_ccinc_v3_ambe6266.sh` | None | `analyze_optics_beamcluster_data`, `extract_cf_features_beamcluster_data`, `make_gate_csvs_from_parquet` |
| `run_ccinc_v3_ambe_stage3.sh` | None | None |
| `run_ccinc_xi02_compcut.sh` | `benchmark_selection_2x2`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity`, `report_background_composition` | `pick_best_model_cut`, `summarize_ambe_neutrons` |
| `run_ccinc_xi02_compcut_cftrain.sh` | `benchmark_selection_2x2`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity` | `pick_best_model_cut`, `summarize_ambe_neutrons` |
| `run_lucho_100k.sh` | None | None |
| `run_mc_qe150_mva_chain.sh` | None | `analyze_optics_beamcluster_data`, `derive_vetoon_scored`, `extract_cf_features_beamcluster_data` |
| `run_mc_qe150_plots.sh` | None | None |
| `run_optics_benchmark.sh` | None | `analyze_optics_beamcluster_data` |
| `run_optics_sweep10_overnight.sh` | None | None |
| `run_stage1_both_selections.sh` | None | None |
| `run_v4neutron_chain.sh` | None | `analyze_optics_beamcluster_data`, `extract_cf_features_beamcluster_data`, `make_gate_csvs_from_parquet` |
| `run_waveform_gated_optics.sh` | None | `run_waveform_gated_pipeline` |
| `score_real_data.sh` | None | None |
| `scripts/full_streamlines.sh` | `benchmark_selection_2x2`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity`, `report_background_composition` | `pick_best_model_cut`, `summarize_ambe_neutrons` |
| `scripts/pilots.sh` | `benchmark_selection_2x2`, `capture_time_stageAB_from_parquet`, `plot_ambe_neutron_multiplicity`, `report_background_composition` | `pick_best_model_cut`, `summarize_ambe_neutrons` |
| `tmux_run_lucho_100k.sh` | None | None |

`ambe.cli` dispatch uses command-name strings. Its registry is checked separately: `runtable` remains available through `ambe diagnostics runtable` (and the `data runtable` alias), but none of these Bash launchers selects that command. `compare_veto` is a standalone audit command and is exercised by the synthetic tests. These are not production dependencies merely because the CLI imports a registry.

## Scripts with no Bash path found

| Module | Other Python importers | Suggested treatment |
|---|---|---|
| `ambe.diagnostics.analyze_composition_cut` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.analyze_containment` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.analyze_debug_files` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.analyze_gamma_tagging` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.analyze_truth_timing` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.analyze_twoneutron_multiplicity` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.audit_merged_labels` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.compare_clustersearcher` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.compare_veto` | None found | Keep tested veto audit |
| `ambe.diagnostics.plot_illustrative_cc_hit_timing` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.plot_multiplicity_compare` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.plot_multiplicity_per_streamline` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.plot_real_cc_hit_timing` | None found | Review as manual study; archive if its dataset is retired |
| `ambe.diagnostics.runtable` | None found | Keep CLI utility |
| `ambe.diagnostics.selection2_multiplicity` | None found | Review as manual study; archive if its dataset is retired |
| `scripts.workflows.analyze_optics_data_rate` | None found | Review as manual study; archive if its dataset is retired |
| `scripts.workflows.extract_offbeam_cluster_features` | None found | Review as manual study; archive if its dataset is retired |
| `scripts.workflows.merge_multiport_signal` | None found | Review as manual study; archive if its dataset is retired |
| `scripts.workflows.reprocess_ambe_v1` | None found | Review as manual study; archive if its dataset is retired |
| `scripts.workflows.tune_mva` | None found | Review as manual study; archive if its dataset is retired |

## Organisation and next decisions

- Keep `src/ambe/diagnostics` as the single home for analysis checks. Keep `scripts/workflows` for campaign orchestration and conversion helpers, and `scripts/presentation` for report/deck assembly.
- Within diagnostics, the use cases are selection/capture comparisons, multiplicity, MC truth and labels, and timing/reconstruction. The module names preserve those distinctions; avoid another move until deciding which manual studies remain useful.
- `capture_time_stageAB_from_parquet` is used by production launchers and delegates to the `basic.py` analyzer. It is an adapter for scored parquet inputs, not another independent per-position fitting engine.
- The two old capture-only diagnostics (`plot_ambe_capture_time_stageAB` and
  `compare_selections_capture`) duplicated the maintained fitter and had no live
  Bash call, so they were removed. Git history and `feature/blessedPlotcuts`
  preserve them.
- Review the no-Bash-path workflow helpers before deleting them: off-beam feature extraction, multiport merging and MVA tuning can be prerequisites run manually before the launchers.
- No no-reference candidate was deleted solely on this static evidence. The retired plot/statistics modules explicitly requested for removal were deleted separately.

Historical launcher/config descriptions are retained in `docs/reports`. Server-specific raw input paths and environments still need to exist before a launcher can execute. This inventory establishes code dependencies, not dataset availability.
