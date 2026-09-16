# Output locations and older analyses

New managed outputs default to the current `AmBeNeutronsAnalysis` checkout. The
shared resolver uses an explicit `output_root`, then `AMBE_OUT`, then
`<checkout>/ambe_output`. Relative output roots are resolved against the checkout,
not the caller's working directory. Checked-in configs now use `ambe_output` or
`ambe_output/ambe_data` instead of one machine's absolute checkout path. MVA
training uses this same resolver.

An output root containing the misspelled directory component
`AmBeNeutronAnalysis` is rejected by the shared resolver. Explicit absolute output
overrides remain supported, including temporary directories for tests. Standalone
utilities with an explicit `--out`/`--output-dir` still write where requested:
use a destination under this checkout. Run repository scripts from its root.

| Product | Default location below the checkout |
|---|---|
| Waveform features, event indices, trigger summaries | `TriggerSummary/` |
| Selected candidates | `EventAmBeNeutronCandidatesData/` |
| Sampled waveform plots | `verbose/` |
| Standard MC/plot/efficiency products | `ambe_output/<run_name>/` (or config subdirectory) |
| Basic capture-time numeric fits | `<output_root>/<run_name>/csv/capture_time_fits.csv` |
| Presentation figures | `PRESENTATION_PLOTS/` and the builders' configured plot directories |
| Curated Markdown | `docs/`, including `reports/`, `efficiency/`, `presentations/` |
| Generated CC cut-flow Markdown | `docs/generated/<run_name>/` (ignored by Git) |

Stage-1 processor paths are anchored to this checkout even when the caller runs
elsewhere. `output_root` does not move its compatibility CSV/parquet directories.
The Bash launchers that previously hardcoded the server checkout now derive their
repository path from their own location. Raw ROOT data, geometry and virtualenv
paths remain machine-specific inputs.

## Reading an older analysis

Do not rename, move or overwrite an old dataset to accommodate this cleanup.
Point `inputs.candidate_csvs`, `inputs.event_index`, `inputs.waveform_features` or
an explicit input argument at the archived files. This includes outputs stored
in a former `AmBeNeutronAnalysis` directory, the old home output tree, or a prior
`AmBeNeutronsAnalysis` checkout. The spelling check applies to the shared output
resolver, not input paths. No archived files or filesystem symlinks were changed.

Some historical presentation builders still read frozen server products explicitly.
In particular, their `CaptureTimeFits_<tag>.csv` inputs are old fit tables; new
basic plotting runs write `capture_time_fits.csv` beside their other run CSVs.
Update a presentation's input deliberately when choosing a new analysis sample.
The collector can search historical figure sources while writing new deck exports
in this checkout. Markdown from `EFFICIENCY_EXTRACT/` moved to `docs/efficiency/`;
its numerical CSV/JSON artifacts were not moved.

MVA `--score-data` and OPTICS' CSV adapter write derived files alongside their
input feature/candidate table. When reusing an external archived table, first
copy it into a new analysis directory under this checkout, then score that copy.
The launcher chains already stage their working feature tables in this checkout.

Historical notes in `docs/reports` describe their original paths and physics
settings. They are records of earlier results, not evidence that those results
were rerun or revalidated during cleanup.
