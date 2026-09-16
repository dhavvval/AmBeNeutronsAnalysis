# AmBeNeutronsAnalysis

ANNIE AmBe data and Monte Carlo analysis: waveform selection, neutron candidates,
OPTICS and ClusterFinder studies, classifier training, efficiency fits and plots.

## Start here

- [Workflow guide](WORKFLOWS.md): commands, inputs, outputs and analysis stages.
- [Config guide](CONFIGS.md): which YAML to copy and which fields to edit.
- [Script index](SCRIPTS.md): presentation builders, diagnostics and workflow helpers.
- [Michel samples](MICHEL.md): selection differences and feature extraction.
- [Follow-up review](FOLLOWUP_REVIEW.md): answers to the plotting/statistics review.
- [Launcher inventory](LAUNCHER_INVENTORY.md): active code references and review candidates.
- [Output locations](OUTPUTS.md): current defaults and legacy inputs.
- [Cleanup record](CLEANUP_PLAN.md): changes, retained studies and validation limits.

```bash
python -m pip install -e '.[dev]'
ambe --help
ambe data process --config configs/data_ambe2v4_all28_box.yaml --selection box
```

Edit server-specific paths and choose a new output tag before processing. ROOT files,
geometry/timing tables, frozen models and generated analysis products are not included.

## Code layout

| Directory | Responsibility |
|---|---|
| `src/ambe/data` | Waveform processing and shared box/MVA event selection |
| `src/ambe/mc` | MC extraction, CC selection, OPTICS, features and MVA |
| `src/ambe/efficiency_extraction` | Multiplicity efficiency fits, figures and model diagnostics |
| `src/ambe/plots` | Plots from saved analysis products, including IC scans |
| `src/ambe/diagnostics` | Run tables, veto comparisons and exploratory studies |
| `src/ambe/leak` | LAPPD light-leak studies |
| `scripts/presentation` | Deck-specific adapters and figure collection |
| `scripts/workflows` | Specialised extraction/training launch helpers |
| `PRESENTATION_PLOTS` | Generated deck exports; inventories live in docs/presentations |

The data processor writes `TriggerSummary/` and `EventAmBeNeutronCandidatesData/`
under the repository root, independently of the working directory. MC and standard plot commands use `output_root/run_name/`
(or the `AMBE_OUT` fallback). Run repository utilities as modules from this directory,
for example `python -m scripts.presentation.collect_presentation_plots --do check`.

`ambe data eff` remains an alias for `ambe data process`. Older `data icscan` and
`data runtable` commands remain aliases for `plots icscan` and `diagnostics runtable`.
The historical implementation and deleted experiments are preserved on
`feature/blessedPlotcuts`; this cleanup is on `cleanup/blessedPlotcuts-codebase`.

## Tests

```bash
pytest -q -m 'not slow'
```

Historical reports in `docs/reports` document their original samples and results. They
can refer to retired configs or earlier cuts; use the guides above for current commands.
