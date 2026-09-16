# Cleanup plan and commit sequence

Work branch: `cleanup/blessedPlotcuts-codebase`, based on `e1313c5`.
The `feature/blessedPlotcuts` branch remains at that commit. Pre-existing local
edits were saved to `/private/tmp/ambe-before-cleanup.patch`; the retirement of
`analysis_run.py` and `eff.py` incorporates those edits, including resolving the
conflict markers already present in `eff.py`.

1. Consolidate maintained entry points: retire legacy efficiency and capture-time
   drivers, remove unused clustering experiments, relocate diagnostics and MVA.
2. Remove fprompt from waveform processing and its active callers; retain the IC
   baseline, integration and second-pulse veto. Shorten narrative comments.
3. Prune requested presentation outputs, adjust PE/CB axes, and share special-run
   selection logic with the processor.
4. Repair Michel OPTICS extraction and describe the distinct Michel samples.
5. Reduce active configuration clutter with a reference inventory, and document
   commands, configuration families, analysis ownership and remaining validation.

Validation: compile all Python, check config and command references, run synthetic
selection/waveform/order tests and existing statistics tests. Real ROOT data,
geometry, model artifacts and generated analysis tables are not supplied; no new
physics efficiency, cut optimization or regenerated campaign plots will be claimed.

Deferred physics decisions: do not change cosmic-veto thresholds, the default
ordering policy, the shared delayed window, or either Michel selection's thresholds.
Quantify their effects on matched samples before changing the measurement.

## Completed implementation

- Deleted the legacy efficiency implementation and retired data driver; `data eff`
  dispatches directly to the maintained processor. The existing conflict markers
  disappeared with the obsolete `eff.py` file.
- Removed DBSCAN/kNN/isolation-forest experiments after finding no live imports or
  CLI registrations. Retained the two older OPTICS CSV experiments for review.
- Moved `icscan` to plots and `runtable`/the chi-square study to diagnostics.
- Moved MVA into `ambe.mc`, and reusable capture-time helpers into `ambe.stats`.
- Organized all 42 remaining root Python utilities under `scripts/`; updated imports,
  file-relative repository paths, shell launch commands and documentation references.
- Removed five fprompt-only studies, waveform fprompt calculations/config fields and
  interactive processor setup. Kept historical output-tag protections.
- Shared event selection with special-run summaries and shared box masks with their
  cluster cutflows. Preserved main-processor default and legacy-ordering behavior.
- Removed presentation slots A3 (streamline overlap), B2 (CV AUC boxplot), C3
  (truth-tag CF top-six shape), G3 (consistency table), G4/G5 (capture comparisons).
  Retained their reusable calculations where other outputs depend on them. Removed
  M20 because its fprompt feature no longer exists. Updated PE/CB display ranges
  for both selections. Deck 3 did not contain the named G3/G4/G5 counterparts, so
  its pooled capture-time fit and residual plots are retained.
- The request for deck-2 "backup plot 15" has no uniquely identifiable match in
  the tracked inventory. Left the unidentified backup unchanged pending a filename.
- Fixed the Michel OPTICS wrapper's missing target and documented the actual,
  different Michel selections, including the partial nature of `--exact-cuts`.
- Added a matched-event veto comparison diagnostic; no physics thresholds retuned.

Special-run counts can change on regeneration: the summary now uses the full
processor waveform cohort and its current ordering policy; cluster diagnostics
now include the processor's 67 us upper bound. These are alignment fixes, not
validated new campaign numbers. Previously saved plots/results are not overwritten
by this cleanup. `PRESENTATION_PLOTS` contains only tracked inventories here.

## Removed configs

These were not referenced by executable launchers for the retained workflows;
pilot configs constructed dynamically by `scripts/pilots.sh` were retained.
The retired fprompt config no longer describes a supported waveform mode.

- `configs/cc_neutrino_ccinc_test.yaml`
- `configs/cc_neutrino_ccinc_truth.yaml`
- `configs/cc_neutrino_v3_mrdfmv_pilot.yaml`
- `configs/cc_neutrino_v3pilot.yaml`
- `configs/cc_neutrino_v3world_recotag_pilot.yaml`
- `configs/cc_neutrino_v3world_truthtag_pilot.yaml`
- `configs/data_ambe2v1.yaml`
- `configs/data_ambe2v3.yaml`
- `configs/data_ambe2v4_fprompt.yaml`
- `configs/mc_ambe_server_workspace.yaml`
- `configs/mc_local_trial_sandbox.yaml`
- `configs/mc_neutrinoint.yaml`
- `configs/mc_server_run.yaml`
- `configs/mc_twoneutron_workspace.yaml`

## Commit groups

1. Prune obsolete configurations and shorten active data configuration descriptions.
2. Consolidate analysis modules, organize scripts, share selection logic and add regression checks.
3. Document maintained workflows, code ownership, presentation inventory and validation; configure pytest to discover repository script modules.

Commits use the repository's configured author; no AI co-author trailers are added.

## Validation limits

Synthetic tests and source/config/CLI checks do not validate real ROOT branch layouts,
physics efficiencies, model transfer or full plot regeneration. For that next step,
supply matched waveform and BeamCluster files, geometry/timing tables and frozen
score/model products. A small matched source/no-source pair plus a representative
MC/dirt-muon sample is sufficient to start, with the original config and run identities.

## Checks completed

- 103 tests passed; three slow statistical coverage tests deselected.
- All Python sources parsed; no stale imports of relocated root modules found.
- All 29 CLI targets resolve to defined functions.
- All remaining YAML configs parse and declare `run_name`.
- Fourteen maintained command help invocations succeeded.
- Twenty-one shell launchers passed `bash -n`.
- All 134 retained presentation slots belong to exactly one deck.
- `git diff --check` passed.

Tests ran in a temporary Python 3.11 environment using the existing analysis
packages and a separately installed pytest. No package was installed into the
user's existing analysis environment.

Recorded implementation commits: `79e2a2d` (configs), `b65171f` (code and presentation inventory).
