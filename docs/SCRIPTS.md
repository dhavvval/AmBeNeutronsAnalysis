# Script index

Run these commands from the repository root after `python -m pip install -e .`.
The maintained analysis package is `src/ambe`; these scripts adapt its outputs
or perform specialised studies. They are not imported by the production processor.

Retired waveform-shape experiments: `tune_waveform_fprompt.py`, `plot_fprompt_yield_and_capture.py`, `plot_efficiency_heatmap_fprompt_vs_baseline.py`, `compare_ic_vs_reference_selection.py`, `analyze_efficiency_vs_ic.py`.
Their code remains on `feature/blessedPlotcuts`. Shared capture-time helpers now
live in `ambe.plots.basic`; MVA training/scoring lives in `ambe.mc.mva_analysis`.

## Presentation

| Module (`python -m …`) | Purpose |
|---|---|
| `scripts.presentation.boxcut_v4_appendix` | Build per-position and feature appendix plots from saved campaign products. |
| `scripts.presentation.boxcut_v4_campaign` | Build campaign, selection-comparison and capture-time maps from saved products. |
| `scripts.presentation.boxcut_v4_special` | Build special-run waveform and cluster plots using shared processor selections. |
| `scripts.presentation.ccinc_v3_ambe_closure` | Apply frozen MC scores to AmBe closure samples and plot capture-time comparisons. |
| `scripts.presentation.ccinc_v3_bg_differential` | Plot species composition and differential feature distributions in CC-inclusive MC. |
| `scripts.presentation.ccinc_v3_stats` | Calculate CC-inclusive selection and model statistics for the presentation builders. |
| `scripts.presentation.collect_presentation_plots` | Collect figures listed in `docs/FIGURES.md` into the five presentation decks. |
| `scripts.presentation.make_collab_slides_xi02_compcut` | Assemble the earlier composition-cut collaboration slide PDF. |
| `scripts.presentation.make_plots_ccinc_v3_merged` | Build CC-inclusive merged-sample model and selection plots. |
| `scripts.presentation.make_presentable_optics_rf_plots` | Style saved OPTICS and classifier results for presentations. |

## Workflows

| Module (`python -m …`) | Purpose |
|---|---|
| `scripts.workflows.analyze_optics_beamcluster_data` | Extract and evaluate OPTICS clusters from BeamCluster real-data files. |
| `scripts.workflows.analyze_optics_data_rate` | Select off-beam OPTICS hits and estimate background cluster rates. |
| `scripts.workflows.derive_vetoon_scored` | Restrict an existing scored MC table to events passing the cosmic veto. |
| `scripts.workflows.extract_cf_features_beamcluster_data` | Extract ClusterFinder features from waveform-gated real-data ROOT files. |
| `scripts.workflows.extract_offbeam_cluster_features` | Compute cluster features from selected off-beam hits. |
| `scripts.workflows.make_gate_csvs_from_parquet` | Export waveform-gate timestamps from saved diagnostic tables. |
| `scripts.workflows.merge_multiport_signal` | Merge per-position MC signal features for classifier training. |
| `scripts.workflows.pick_best_model_cut` | Choose score thresholds from a saved MC score table. |
| `scripts.workflows.reprocess_ambe_v1` | Run the shared processor with AmBe v1 waveform settings. |
| `scripts.workflows.run_waveform_gated_pipeline` | Run the shared IC waveform gate and neutron-candidate extraction. |
| `scripts.workflows.summarize_ambe_neutrons` | Summarize saved AmBe candidates and trigger counts. |
| `scripts.workflows.tune_mva` | Tune classifiers using the maintained MC feature preparation functions. |

## Diagnostics

| Module (`python -m …`) | Purpose |
|---|---|
| `ambe.diagnostics.analyze_composition_cut` | Inspect truth composition and proposed MC composition cuts. |
| `ambe.diagnostics.analyze_containment` | Study MC containment and boundary effects. |
| `ambe.diagnostics.analyze_data_mc_comparison` | Compare saved real-data and MC cluster features. |
| `ambe.diagnostics.analyze_debug_files` | Inspect debug ROOT branches and truth/cluster associations. |
| `ambe.diagnostics.analyze_gamma_tagging` | Study gamma tags and neutron associations in MC. |
| `ambe.diagnostics.analyze_truth_timing` | Inspect truth timing and reconstructed timing in MC. |
| `ambe.diagnostics.analyze_twoneutron_multiplicity` | Study reconstructed multiplicity in the two-neutron MC sample. |
| `ambe.diagnostics.audit_merged_labels` | Audit signal/background labels and composite keys in merged training samples. |
| `ambe.diagnostics.benchmark_selection_2x2` | Compare box and frozen MVA selection on matched MC/data samples. |
| `ambe.diagnostics.capture_time_stageAB_from_parquet` | Fit scored OPTICS cluster times with the shared basic plotting analyzer. |
| `ambe.diagnostics.compare_clustersearcher` | Compare ClusterSearcher and OPTICS reconstruction. |
| `ambe.diagnostics.plot_ambe_neutron_multiplicity` | Plot accepted cluster multiplicity with scored-data comparisons. |
| `ambe.diagnostics.plot_illustrative_cc_hit_timing` | Draw illustrative CC-inclusive hit timing distributions. |
| `ambe.diagnostics.plot_multiplicity_compare` | Compare saved multiplicity distributions across selections. |
| `ambe.diagnostics.plot_multiplicity_per_streamline` | Plot multiplicity separately for the two CC-inclusive event selections. |
| `ambe.diagnostics.plot_real_cc_hit_timing` | Plot reconstructed CC-inclusive hit timing from saved products. |
| `ambe.diagnostics.report_background_composition` | Report background species and populations in model training samples. |
| `ambe.diagnostics.selection2_multiplicity` | Inspect multiplicity for the second stored event selection. |

## Other packaged tools

- `python -m ambe.efficiency_extraction.datadriven_chi2 --help`: background-template uncertainty study.
- `python -m ambe.diagnostics.compare_veto --help`: compare legacy/current event indices.
- `ambe diagnostics runtable --config CONFIG`: compare runs from saved summaries.
- `ambe plots icscan --config CONFIG`: scan saved IC-window selections.

The scripts in `presentation/` are presentation/report adapters. `ccinc_v3_stats`
also exports tables used by other adapters, and `make_plots_ccinc_v3_merged` still
provides their shared plotting style. Keep those dependencies together. The
diagnostic timing/multiplicity scripts can also feed reports, so they are not
classified as presentation-only merely because their plots appear in a deck.

Several exploratory scripts still contain sample-specific input paths. Check
their module constants or `--help` before running. No recorded invocation history
is available; this index describes dependencies, not how often a script was used.

All diagnostic modules now live in `src/ambe/diagnostics`. The former
`scripts/diagnostics` directory was merged into it. See the
[launcher inventory](LAUNCHER_INVENTORY.md) for evidence of use and review candidates.
