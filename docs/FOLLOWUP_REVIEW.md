# Plotting, statistics and script cleanup review

Work is on `cleanup/blessedPlotcuts-codebase`; `feature/blessedPlotcuts` remains
preserved. This follows the first cleanup recorded in [CLEANUP_PLAN](CLEANUP_PLAN.md).
The plan was to trace callers, remove retired paths, consolidate live helpers,
update launchers/documentation, then verify with synthetic data. No real ROOT data
or recorded job-execution history was available.

## Questions about `src/ambe/plots`

1. **`basic.py`: keep `run_analysis`.** The CLI calls `run(ctx)`, which resolves
   the config and calls `AmBeNeutronAnalyzer.run_analysis` to group source positions
   and build the plot book. Removing that method would break the CLI. Removed the
   obsolete hardcoded example `main`, its stale default candidate glob, unused
   imports and lengthy repeated comments. Its SciPy/lmfit paths now own the shared
   fit implementation used by presentation adapters. Basic plotting also saves
   its existing lmfit results to a CSV; it does not refit to produce that table.
2. **`combined.py`: retained.** It remains a CLI command and is called by the MC
   plot launcher and data pipeline.
3. **`heatmap.py`: retained.** It remains the standard summary/heatmap command.
   Its counting/trigger fractions should not be confused with a model-based,
   background-corrected multiplicity efficiency.
4. **`phase2.py`: removed.** At the starting revision it was already a retired
   command containing the older reference driver as a string. No live analysis
   called that driver. Removed the module and CLI registration. Its history is
   preserved in Git; the file alone does not establish authorship of the original
   reference implementation.
5. **`waveform_cuts.py`: keep as a plotting adapter.** It already reads processor
   `WaveformFeatures_<tag>_<run>` products; it does not reprocess ROOT waveforms.
   It now resolves those products under this checkout, accepts the processor's
   CSV fallback, and verifies saved acceptance flags with the processor's
   configured IC gate and second-pulse veto. Missing, empty or inconsistent
   products fail with a useful error. Run `ambe data process ... --selection box`
   first, then `ambe plots waveformcuts` with the matching config.
6. **`efficiency_thesis.py`: not the sole calculator.** It builds multiplicity
   figures by calling `efficiency_fit.py`. Both now live in `efficiency_extraction`.
   Removed its duplicate capture-time plot/fit path. Plotting now forwards the
   config's likelihood, overflow and background-uncertainty settings to the same
   numerical functions. The full validated numerical report remains the
   `ambe stats efficiency` command; the figures alone are not that report.

## Questions about `src/ambe/stats` and diagnostics

1. **`capture_time.py`: removed.** It duplicated the per-position analysis but
   also supplied helpers used by three presentation modules. Those live callers
   now import the fitting, loading and quality-mask helpers from `plots/basic.py`.
   Removed the separate tagset CLI/driver and the redundant MC launcher step.
   Kept the distinct exponential-plus-flat late-time diagnostic used by the AmBe
   closure comparison, also in `basic.py`. No capture-time calculation remains in
   a `stats` directory. The active scored-parquet adapter and styled capture
   pages now use the same solver: removed their obsolete `leastsq` override,
   including a process-wide monkeypatch, and a conflicting capture-time bound.
   These adapters now follow the basic analyzer’s bounded `least_squares` and
   absolute-covariance convention; old rendered values may therefore differ.
2. **`core.py`: removed.** The usable-looking CLI entry only raised
   `NotImplementedError`; its other helper had no external callers. Removed the
   registry entry and failing final step in the data pipeline. The pipeline also
   now forwards its selection arguments to the processor.
3. **Efficiency modules: grouped together.** `efficiency_fit.py`,
   `profile_likelihood.py`, `efficiency_thesis.py` and `datadriven_chi2.py` are in
   `src/ambe/efficiency_extraction/`. Imports and commands were updated.
   `ambe stats efficiency` and `ambe plots efficiency` remain valid command names.
   `profile_likelihood` is a legacy stochastic reference exercised by tests,
   rather than an unused production entry point. See the explanation below.
4. **`waveform.py`: removed.** It was a hardcoded, import-time historical waveform
   summary plotter with no live callers. Processor waveform plots and the retained
   source/no-source adapter cover the maintained workflows; no claim is made that
   every old figure had identical styling.
5. **Markdown and output paths: consolidated.** All tracked Markdown is now in
   `docs/`; package metadata, links and figure-collector paths were updated.
   Generated CC Markdown also goes under `docs/generated`. New managed output
   defaults resolve to this checkout, including the processor and MVA paths.
   [OUTPUTS](OUTPUTS.md) records explicit overrides, older inputs and the two
   adapters that write derived tables beside their inputs.
6. **Two diagnostics directories: merged.** Previously the package directory held
   CLI/reusable diagnostics and `scripts/diagnostics` held the loose studies moved
   during the first cleanup. That division was not useful enough to retain.
   Eighteen script modules now live alongside `runtable` and `compare_veto` in
   `src/ambe/diagnostics`; the two redundant capture-only modules were removed.
   Launchers use `python -m ambe.diagnostics.<name>`.
7. **`datadriven_chi2.py`: moved into `efficiency_extraction`.** It studies how
   uncertainty in the measured background histogram changes the data-driven fit.
8. **Usage inventory: completed.** Of the former 20 diagnostic scripts, five have
   live Bash calls, two capture-only duplicates were removed, and 13 retained
   modules have no Bash path in this checkout. With the two retained package
   utilities there are 20 diagnostics total. Seven of 12 workflow helpers have
   Bash calls; five do not. No additional modules are reached only through
   imports. [LAUNCHER_INVENTORY](LAUNCHER_INVENTORY.md) lists every launcher and
   review candidate, including special-run and presentation dependencies.
   `run_ccinc_productionv2.sh` reaches three diagnostics and two workflow helpers;
   `run_ccinc_v3_ambe_stage3.sh` calls `ambe.mc.mva_analysis` directly and reaches
   neither group. No-reference scripts were retained for your next review.

## How to read the efficiency code

The model treats the number of candidates in one acquisition as the sum of one
possible detected AmBe neutron (a Bernoulli trial with efficiency epsilon) and a
background count. The **Poisson model** fits a background mean lambda. The
**data-driven model** takes the background shape from a no-source run. Both fit
the multiplicity distribution, including acquisitions with zero candidates.

| Module | Responsibility |
|---|---|
| `efficiency_fit.py` | Build event multiplicities, fit both models, calculate intervals and fit-quality diagnostics, write CSV/JSON summaries |
| `efficiency_thesis.py` | Draw distributions, model overlays and parameter scans using those fit functions |
| `profile_likelihood.py` | Original stochastic reference for regression comparisons |
| `datadriven_chi2.py` | Diagnose sensitivity to the background template's statistical uncertainty |

Git history shows the main extraction added in `420fb17` (2026-09-11), then its
statistical checks strengthened in `1f9c361` (2026-09-14). `7e2a0eb` added the
waveform companion figures. The profile reference predates those additions
(the package reorganisation is visible in `6bc8a4b`). The reorganisation here
preserves the numerical fitter and the physics settings.

For the next scientific review, I recommend checking:

- Whether the source and background selections, time windows and trigger
  denominators describe comparable acquisitions.
- Whether the no-source sample has enough events to constrain its multiplicity
  shape. A small reduced chi-square can result from a very uncertain template.
- Whether the best fit and interval lie inside the scan grid, and whether the
  selected background model actually describes the data. The numerical command
  checks grid boundaries and requires an explicit nominal-model choice; it also
  records any configured exception to its fit-quality requirement.
- Whether one possible correlated neutron plus background is appropriate for the
  sample being interpreted. The fitted epsilon is conditional on this model and
  the chosen acquisition/selection definition.
- Whether the reported uncertainty convention is the intended one. The code
  supports the thesis chi-square convention, a multinomial alternative and
  background bootstrap options. Plot labels still follow the thesis chi-square
  figure convention; use the numerical CSV/JSON metadata to identify alternatives.

The plotting command recomputes the selected fits. It does not run the numerical
command's complete cross-check, systematic and goodness-of-fit reporting pipeline.
A useful future improvement is to render figures directly from the saved fit
results and metadata; that requires a defined result-loading interface.

## Validation

Validation completed: **116 fast tests passed**, with three slow tests deselected;
90 Python files and 21 Bash launchers passed syntax checks. All 27 CLI targets
resolved, the three capture-time presentation callers imported, and all 134
figure assignments parsed from the relocated index. All 40 repository Markdown
files are under `docs/`, and the current guide links resolve.

Synthetic capture samples reproduce the five pre-consolidation numerical results
(SciPy/lmfit, fixed/floating background, and the late-time exponential model).
Regression tests cover shared fitting, waveform-product consistency, output-path
resolution, CLI registration and existing efficiency/selection behaviour. Python
syntax/import and Bash syntax checks do not require ROOT files. They do not verify
real campaign yields, ROOT branch compatibility or visual agreement of regenerated
physics figures. A representative waveform ROOT file, its matched BeamCluster file
and the matching config/products will be needed for that later check.
