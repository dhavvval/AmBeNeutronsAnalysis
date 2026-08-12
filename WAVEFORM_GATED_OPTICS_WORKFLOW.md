# Waveform-Gated OPTICS Benchmark — Workflow

## Goal
Run the OPTICS secondary-clustering benchmark **only on BeamCluster events that pass
the AmBe waveform pre-selection**, instead of on all ~1.09M events. Expected ~9x
speedup (waveform-passed events are ~11% of all BeamCluster events) AND a physically
cleaner sample (only AmBe-gamma-tagged events).

## The chain already exists — confirmed
All three stages have working code in the toolkit:

1. **Waveform IC cut** — `AmBeNeutronProcessing.analyze_waveform`
   ([src/ambe/data/processor.py:299](src/ambe/data/processor.py#L299)).
   Integrates each PMT pulse in [pulse_start=300, pulse_end=1200] ns, computes
   `IC_adjusted`, accepts if `pulse_gamma < IC_adjusted < pulse_max` AND no second
   pulse. Accepted timestamps → `good_events` set.
   - **IC window = v1: `400 < IC < 575`** (WaveformConfig defaults, lines 24/28). CONFIRMED.

2. **Match to BeamCluster events** — `process_events_efficient`
   ([src/ambe/data/processor.py:656](src/ambe/data/processor.py#L656)).
   Keeps BeamCluster events where **`eventTimeTank ∈ good_events`** (join key = tank
   timestamp), applies `ambe_single_cut` (0<PE≤100, CCB<0.45, clusterTime≥2000,
   clusterHits≥5, cosmic veto), writes survivors to
   `EventAmBeNeutronCandidates_<runinfo>_<run>.csv`.

3. **OPTICS secondary clustering** — `src/ambe/clustering/optics_data.py` consumes
   those CSVs, runs per-event OPTICS; `optics_analysis.py` applies OPTICS-stage cuts
   (PE<60, CB<0.5, hits>10) and compares to ClusterFinder.

```
raw waveforms ──[IC cut v1: 400<IC<575]──▶ good_events (timestamps)
                                              │ match on eventTimeTank
BeamCluster.root ─────────────────────────────┴─[ambe_single_cut]─▶ EventAmBeNeutronCandidates_*.csv
                                                                          │
                                                                  [OPTICS] ─▶ [PE<60,CB<0.5,hits>10 vs CF]
```

## Inputs — verified present
- **v1 raw waveforms**: `/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/<run>/AmBeWaveforms_<run>_pN_pM.root`
  (per-run subdirs, each file holds thousands of `RWM_<timestamp>` histograms).
  Large: run 4499 ≈ 2.4 GB / 4 files; run 4589 ≈ 16 files. The `RWM_` (campaign-1)
  folder pattern is correct for v1.
- **BeamCluster**: `/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/BeamCluster_<run>.root`
  (TTree `Event`, branch `eventTimeTank` is the join key).

## Decisions (user-confirmed)
- Regenerate the gating from scratch (do NOT trust the existing CH5 candidate CSVs).
- IC window = **v1, 400 < IC < 575**.

## Execution plan
Stage 1+2 are driven by the existing `run_full_analysis`-style entry
(`AmBeNeutronProcessing` + `process_run_waveforms` + `process_events_efficient`),
which already writes `EventAmBeNeutronCandidates_*.csv` and
`TriggerSummary/AmBeWaveformResults_*.csv` (acceptance counts).

1. Confirm `WaveformConfig` = v1 (pulse_gamma=400, pulse_max=575) — it already is.
2. Run Stage 1+2 per run: `process_run_waveforms(run, waveform_dir=<AmBe2.0v1>, campaign=1)`
   then `process_events_efficient(...)`, writing fresh candidate CSVs.
   - This is the heavy step (reading ~all waveforms). Parallelize per-run like the
     OPTICS benchmark (process pool), and log acceptance rate per run.
3. Run Stage 3 OPTICS on the fresh candidate CSVs (`optics_data.py`), then the
   OPTICS-vs-CF comparison (`optics_analysis.py`), reusing the same OPTICS params
   (ms=8, xi=0.10, t_unit=25 ns) and Stage-1 cuts (PE<60, CB<0.5, hits>10) as the
   full-sample benchmark for an apples-to-apples comparison.

## Verification
- Acceptance rate per run should be ~40-50% of waveform acquisitions and the
  resulting candidate events ~10-12% of BeamCluster events (matches the existing
  AmBe2.0v1CH5 CSVs and AmBeWaveformResults trigger summary — cross-check counts).
- Spot-check a few `eventTimeTank` values land inside `good_events`.
- Compare the waveform-gated OPTICS-vs-CF cluster rate against the full-sample
  benchmark currently running: gating should raise the neutron-like fraction.

## Open question for the user
The existing candidate CSVs are tagged `CH5` (a specific PMT channel selection?).
Need to confirm whether the regeneration should reproduce that same channel
configuration or use the full PMT set.
