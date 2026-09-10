# fprompt pulse-shape cut for AmBe Stage-1 — report (2026-07-27)

> ## ⚠ CAPTURE-TIME FIT WINDOW CHANGED — 2026-08-27
>
> **Every capture-time number in this document was measured on a 10–67 µs fit window and
> has been superseded.** The analysis now fits **2–67 µs** everywhere — 2 µs is where the
> data starts (box cut `t ≥ 2 µs`, cosmic veto removes anything earlier), so the old
> window discarded the whole thermalisation rise.
>
> | quantity | this document (10–67) | **current (2–67)** |
> |---|---:|---:|
> | anchor, 19 positions | 30.53 ± 0.26 | **29.417 ± 0.221** |
> | campaign τ, 26 positions | 30.535 ± 0.228 | **29.477 ± 0.193** |
> | campaign therm | 5.50 ± 0.22 | **6.52 ± 0.11** |
> | MVA τ, 26 positions | 30.97 ± 0.26 | **29.33 ± 0.23** |
> | box τ, matched set | 31.509 ± 0.250 | **30.444 ± 0.210** |
> | MVA τ, matched set | 30.521 ± 0.263 | **28.898 ± 0.229** |
> | matched shift box → MVA | −0.989 ± 0.363, 2.73σ | **−1.546 ± 0.311, 4.98σ** |
>
> Efficiencies, cut flows, multiplicities and background composition are **unaffected** —
> only quantities derived from a capture-time fit moved. `REPORT_capturetime_maps_and_deck5.md`
> §7 has the full list and the reasoning. Read the τ values below as historical.



Adds a second, orthogonal axis to the Stage-1 AmBe waveform selection: `fprompt`,
a pulse-*shape* variable, alongside the existing `IC_adjusted` pulse-*size* cut.

Ported from [`PMT_analysis_utils.PMT_wvfm_selection`](https://github.com/lmlepin9/2x2_neutron_sources/blob/1af4c3ab5eb28f2de6ef0da241758b39ae361c74/AmBe/light/PMT_analysis_utils.py#L159)
(lmlepin9 / 2x2_neutron_sources).

Pick this up in a fresh chat by pointing Claude Code at this file. Companion docs:
`AMBE_DATA_PROCESSING_HANDOFF.md` (the v4 campaign this builds on),
`WAVEFORM_GATED_OPTICS_WORKFLOW.md`.

---

## 1. Why

Stage 1 (`src/ambe/data/processor.py::analyze_waveform`) accepted a tag-PMT
waveform on **one** variable: `pulse_gamma < IC_adjusted < pulse_max`, plus an
ad-hoc "second pulse after bin 1200" veto.

With only one axis, the window placement *is* the efficiency/purity trade, and it
has to be re-tuned every campaign because it tracks PMT gain (400 v1 / 590 v3 /
700 v4). `AMBE_DATA_PROCESSING_HANDOFF.md` leaves exactly this open — v4
efficiency came out at 56.3% ("not satisfactory") and the 700-1200 vs 600-1400
question was never resolved.

`fprompt = (prompt-peak integral) / (total-pulse integral)` is independent of
pulse size, so it can reject non-BGO-shaped waveforms without the IC window
having to do that job — which is what allows the IC window to be widened.

## 2. Definition as implemented

```
fprompt = sum(ADC[350:400] - baseline_pre) / sum(ADC[350:1200] - baseline_pre)
```

Bin index = ADC sample (2 ns/sample; the histogram x-axis has width-1 bins).
Windows are applied on the axis values, matching how the IC window is applied,
so they survive a binning change.

**Two baselines, deliberately — they are not interchangeable:**

| variable | baseline | why |
|---|---|---|
| `IC_adjusted` | `norm.fit(hist_values)` over the whole record | **unchanged**, so every previously published IC number stays byte-identical |
| `fprompt` | median of bins `[0, 250)` (pre-pulse) | IC integrates ~900 samples, so a baseline error δ merely shifts it by 900δ. fprompt is a ratio of a 50-sample numerator to an 850-sample denominator, where δ *biases* the ratio instead of cancelling. `norm.fit` runs over the pulse as well as the noise, so using it for fprompt produced an unphysical `fprompt > 1` tail (p99 = 1.32 on 6062, up to 3.7 on v3 5740). |

Denominator guard: total integral ≤ `fprompt_min_integral` (50 ADC·samples) →
`fprompt = NaN`, which **fails** any fprompt-based mode.

## 3. What the data says (run 6062, all 38,451 waveforms)

```
IC_adjusted     fp<0.15   in band   fp>0.30    undef     total
[0, 250)              0        29      1500     1293      2822
[250, 500)            0       304       635      104      1043
[500, 700)            0      1059        81        0      1140
[700, 1200)           7     26878        80        0     26965   <- current v4 window
[1200, 1400)          6      3280         0        1      3287
[1400, 2000)         11      1763         0        0      1774
[2000, inf)         887       350         0        0      1237
```

Read this honestly:

- **fprompt does not clean up what we already keep.** Inside 700-1200 it removes
  0.35% (15215 → 15161). No purity win there, and the change does not claim one.
- **Its value is enabling a wider IC window.** 500-2000 × fprompt 0.15-0.30
  accepts **18842 vs 15215, +23.8% Stage-1 acceptance**, because fprompt
  independently kills the two populations that made the tight window necessary:
  low-IC noise (99% of IC<250 is `fp>0.30` or undefined) and high-IC pile-up
  (72% of IC>2000 is `fp<0.15`).
- **fprompt is gain-independent**, so it needs no per-campaign retuning:
  v1 run 4499 median IC 369 / fprompt 0.228; v4 run 6062 median IC 965 /
  fprompt 0.224 — 2.6× gain, same shape.
- **It doubles as a per-run DQ monitor.** v3 run 5740 (median IC 39, fprompt
  0.618) is the lowest-acceptance v3 run at 20.9% — a gain/threshold problem the
  IC cut silently turned into low acceptance.

### fprompt does not replace the second-pulse veto

Checked, not assumed. Inside the IC window the veto rejects ~45% of waveforms,
and the veto-PASS / veto-FAIL sets have the **same** fprompt distribution
(medians 0.222 vs 0.221, Δ = 0.001, full statistics). fprompt's denominator ends
*at* bin 1200; the veto looks *after* it. They catch different failures, so the
change is purely additive and the veto is untouched.

### Mean waveform per region (page 6 of the tuning PDF)

Confirms the interpretation rather than assuming it:

| region | shape |
|---|---|
| low-IC / high-fprompt | narrow ~80 ADC spike, no tail (+ slight undershoot) — not scintillation |
| in-box | ~145 ADC peak with the characteristic BGO exponential decay |
| high-IC / low-fprompt | ~660 ADC peak with a much longer/larger tail — pile-up |

## 4. Code changes

| File | Change |
|---|---|
| `src/ambe/data/processor.py` | `WaveformConfig`: `selection_mode`, `fprompt_*`, `width_threshold`. New `compute_waveform_features()` (single pass, all features) and `passes_selection()`. `analyze_waveform()` now returns `(IC_adjusted, baseline, is_accepted, features)`. `process_run_waveforms()` gained `feature_dump_path`; `run_complete_analysis_pipeline()` gained `dump_features` (default `True`) and logs the active selection. |
| `run_waveform_gated_pipeline.py` | `--selection-mode`, `--fprompt-min`, `--fprompt-max`, `--no-feature-dump`, `--runs` |
| `tune_waveform_fprompt.py` | **new** — offline cut tuning from the parquet dumps |
| `configs/data_ambe2v4_fprompt.yaml` | **new** — heatmap config for the widened-window tag |

`src/ambe/data/eff.py` and `analysis_run.py` are the legacy non-import-path
copies and were deliberately left alone.

### Selection modes

- `"ic"` — the historical cut. **Default**, so nothing changes unless asked.
- `"ic+fprompt"` — adds the fprompt window on top of the current IC window.
- `"fprompt2d"` — same cut, meant to be paired with a widened IC window.

The second-pulse veto applies in all three.

### Per-waveform feature dump — the piece with the most leverage

Every run now writes `TriggerSummary/WaveformFeatures_<tag>_<run>.parquet`:

```
timestamp (int64), accepted (bool), IC_adjusted, baseline, baseline_sigma,
baseline_pre, fprompt, prompt_integral, total_integral, peak_amp, peak_bin,
width_over_thresh, second_pulse (bool)
```

`timestamp` is in the schema, and `good_events` is nothing but a set of
timestamps — so **the entire Stage-1 decision is re-derivable offline**.
Retuning the IC or fprompt window becomes a pandas query instead of another
multi-hour read over dCache (which has hung on this dataset four times).
~2.3 MB per 38k waveforms (~60 B/waveform), so ~50 MB for a 20-run campaign.

## 5. Verification

| # | Check | Result |
|---|---|---|
| 1 | **Regression gate** — 6062, `selection_mode="ic"`, 700-1200 | **PASS**, exactly **15215 / 38451**, matching `AmBeWaveformResults_AmBe2.0v4_gated.csv` |
| 2 | Two implementations agree (`passes_selection` vs the tuning script's `accept_mask`) | 15215 / 15161 / 18842 in all three modes |
| 3 | Feature dump vs the 2000-waveform design sample | reproduces at full statistics (see §3) |
| 4 | Veto/fprompt independence at full statistics | Δ median = 0.001 |
| 5 | Efficiency vs the baseline | **DONE** — ratio falls 1.83 pp, yield rises 23.0%; cause isolated to the IC widening, not fprompt. See §7.1, §7.5 |
| 6 | Capture-time τ invariance | **DONE, PASS** — recovered sample is indistinguishable from baseline signal. See §7 |
| 7 | Cross-campaign (v1 4499, v3 5789/BRF) | **NOT RUN** |

Artifacts:
- `TriggerSummary/WaveformFeatures_regress6062_6062.parquet`
- `verbose/fprompt_tuning_regress6062.pdf` (6 pages: fprompt 1D; fprompt-vs-IC
  2D with cut boxes; veto independence; acceptance scans; per-run DQ; region
  mean-waveform overlays, absolute + peak-normalised)

## 6. Caveats — read before quoting any number

1. **+23.8% is Stage-1 *acceptance*, not efficiency.** The 56.3% in the handoff
   is `unique_neutron_triggers / ambe_triggers`, computed after Stage 2 and
   `ambe_single_cut`. They move independently: if some recovered tags sit on
   events with no neutron cluster, acceptance rises while efficiency does not.
   Everything measured so far is upstream of the quantity the change is
   justified by.
2. **The band tilts** — centre ~0.28 at low IC, ~0.18 above IC 2000 (hence 887
   of 1237 waveforms above 2000 fall below `fprompt_min`). A horizontal box is
   an approximation, defensible and matching the reference, but it means the IC
   upper bound is partly enforced by the fprompt floor. **500-2000 × 0.15-0.30
   is a starting point, not a tuned answer** — the acceptance-scan page is the
   tool for settling the final bounds.
3. **`fprompt > 1` is not eliminated**, only made rare: 545/38451 = 1.4% on
   6062, all at IC < 500, median peak amplitude 183 ADC. These are a fast spike
   followed by AC-coupling undershoot, so the tail integral goes small/negative.
   All are rejected as `fp > max`, so the cut is unaffected — but don't write
   "eliminated".
4. **All numbers except the run-6186 partial are single-run (6062).**
5. `dump_features` defaults to `True`, so `processor.main()`'s interactive path
   now writes parquet files it previously didn't. Benign, but it is a behaviour
   change.

## 7. Results of the 20-run pass (completed 2026-07-27 18:01)

The pass below finished all 20 runs. Both open checks are now closed.
**Verdict: the widened window recovers real signal. Adopt it.**

### 7.1 Efficiency — the ratio falls, the yield rises

Same definition for both tags (`unique_neutron_triggers / ambe_triggers`).
Headline figures **exclude run 6242**, whose input grew between the two passes
(§7.6), so both sides see identical data — 18 positions:

| | ambe_triggers | unique_neutron | efficiency |
|---|---|---|---|
| baseline IC 700-1200 | 266,817 | 151,785 | 0.5689 |
| fprompt2d 500-2000 × 0.15-0.30 | 339,130 | 186,716 | 0.5506 |
| change | **+27.1%** | **+23.0%** | **−1.83 pp** |

Including 6242 gives +27.2% / +23.0% / −1.85 pp — the dataset difference is
proportional and changes nothing, but the table above is the exact number.

Stage-1 acceptance, with both selections re-derived offline from the *same*
feature dumps so 6242 cannot bias either side (all 20 runs, 874,100 waveforms):
**364,569 → 465,636, i.e. 41.71% → 53.27%, +27.7%**.

The efficiency drop is **uniform**: all 19 positions fall, by 0.98 to 2.41 pp.
That uniformity is itself evidence against a position-dependent artifact.

The **marginal** efficiency of the newly admitted triggers is **48.3%** vs
56.9% for the ones already kept. So the widened window admits triggers that are
somewhat *less* likely to carry a reconstructed neutron — exactly the dilution
§6.1 warned about. It is a ratio effect, not evidence of background. That
question is settled by §7.2.

#### Efficiency heatmaps by port × Y position

Standard three-plot set, one per tag, via the usual entry point:

```bash
MPLBACKEND=Agg PYTHONPATH=src python -m ambe.cli plots heatmap --config configs/data_ambe2v4_fprompt.yaml
MPLBACKEND=Agg PYTHONPATH=src python -m ambe.cli plots heatmap --config configs/data_ambe2v4.yaml
```

→ `AmBeNeutronsAnalysis/ambe_output/ambe_data/AmBe2.0v4_{fprompt,gated}/plots/`
(`efficiency_`, `statistics_`, `residual_efficiency_heatmap__*.{pdf,png}`).
Note `output_root` is the sibling **AmBeNeutronsAnalysis** dir, no "s".

Plus a like-for-like three-panel comparison the CLI cannot produce — its
residual panel only compares against the hardcoded `REFERENCE_DATA`
campaigns (AmBe 1.0 / 2.0v2), not against the v4 IC baseline:

```bash
MPLBACKEND=Agg python -u plot_efficiency_heatmap_fprompt_vs_baseline.py
```

→ `verbose/efficiency_heatmap_fprompt_vs_baseline.{pdf,png}` — baseline and
fprompt2d on a **shared** colour scale, then the per-cell Δ on a 0-centred
diverging scale. Every cell of the Δ panel is negative; the 6242 cell is
outlined dotted because its input grew between passes (§7.6).

| Y (cm) | Port 1 | Port 5 | Port 2 | Port 3 | Port 4 |
|---|---|---|---|---|---|
| 100 | 54.11 → 52.62 (−1.49) | 63.04 → 61.23 (−1.80) | 54.12 → 51.91 (−2.21) | 40.87 → 39.58 (−1.29) | 54.52 → 53.06 (−1.46) |
| 50 | 62.58 → 60.49 (−2.09) | — | 61.93 → 59.87 (−2.07) | — | — |
| 0 | 64.14 → 62.22 (−1.92) | 71.82 → 69.51 (−2.31) | 63.17 → 61.09 (−2.09) | 50.82 → 49.43 (−1.39) | 65.29 → 62.88 (−2.41) |
| −50 | 61.80 → 59.45 (−2.35) | — | 60.77 → 58.93 (−1.84) | — | — |
| −60 | — | — | — | 46.99 → 46.00 (−0.98) | — |
| −100 | 50.50 → 48.66 (−1.84) | 62.97 → 60.70 (−2.28) | 52.41 → 50.74 (−1.67) | — | 49.70 → 48.32 (−1.38) |

The geometry is unchanged by the cut: Port 5 y=0 stays the best cell (71.8 →
69.5) and Port 3 y=100 the worst (40.9 → 39.6). The Δ shows no port or
y-dependence — it is a flat pedestal shift, consistent with §7.2's finding that
the recovered events are the same physics, not a new population.

**Do not compare −1.85 pp against the handoff's 56.3%.** That figure is an
unweighted mean over 14 positions; the 0.5730 above is trigger-weighted over
19. Both are computed here the same way, so the *difference* is valid; the
absolute numbers are not interchangeable.

### 7.2 Capture time — the recovered sample is signal, not accidentals

Built the **recovered-only** sample: fprompt2d candidate events whose
`eventID` never appears in the baseline tag, per run. The subset relation is
clean — only **362 of 127,559** baseline events (0.28%) are lost by fprompt2d,
matching the 0.35% the cut removes inside the old window — so
"recovered = fprompt2d − baseline" is a valid framing.
**21,772 recovered events, 20,607 clusters in the 10-67 µs window.**

Model-free tests, recovered vs baseline over 10-67 µs — these are the decisive
ones and they all agree:

| test | result |
|---|---|
| mean cluster time | 30.650 vs 30.731 µs → **−0.080 ± 0.113 µs (−0.7σ)** |
| KS two-sample | **D = 0.0062, p = 0.49** |
| normalised shape χ², 10 bins | **6.8 / 9 dof, p = 0.66** |
| late/early rate ratio, N(50-67)/N(10-20) per µs | 0.2689 ± 0.0059 vs 0.2717 ± 0.0022 |
| quartiles p25/p50/p75/p90 | agree to ≤ 0.15 µs at every quantile |

The normalised ratio is flat at 1.0 across all ten bins with no time trend. For
scale, a 5% flat admixture would tilt it from 0.977 to 1.080 — the measured
last bin is 0.989 ± 0.038, so **a ≥5% accidental component is excluded at
~2.4σ and 10% decisively.** Accidentals are flat in time; nothing flat is
being added.

Fitted τ, `NeutCapture` with `B` fixed at 0 (the `lmfit_analysis` recipe),
per position then weighted-averaged, identical machinery for every sample:

| sample | positions | τ (µs) |
|---|---|---|
| baseline IC 700-1200 | 19/19 | 30.53 ± 0.26 |
| fprompt2d | 19/19 | 30.91 ± 0.23 |
| recovered only | 14/19 conv. | 28.74 ± 0.75 |

Paired per-position difference recovered − baseline: **−1.50 ± 0.82 µs, 1.8σ**,
and *negative*. With `B` pinned at zero an accidental pedestal has nowhere to
go and would push τ **up**; it went slightly down. Combined with the shape
tests above, the extra statistics are recovered signal.

Script: `fit_capture_time_fprompt_compare.py`. Per-position fits in
`TriggerSummary/CaptureTimeFits_{baseline,fprompt2d,recovered}.csv`.

Plotted in `verbose/fprompt_capture_time_recovered_vs_baseline.{pdf,png}` via
`plot_fprompt_yield_and_capture.py` — normalised overlay above, ratio below,
with the curves a 5% and 10% flat accidental admixture *would* trace so the
exclusion is visible rather than asserted. The measured ratio stays flat on 1.0
while both contamination curves climb away from the data past ~40 µs.

### What §7.2 does and does not establish

It establishes that the recovered **neutron candidates are genuine captures** —
their time structure is that of the known signal, to high precision.

It does **not** establish that every newly admitted *tag* is a real AmBe tag.
These tests only see events that produced a cluster, so they are blind to the
51.7% of newly admitted triggers that produced none. The 48.3% marginal
efficiency is equally consistent with "real AmBe tags whose neutron simply
failed to reconstruct" and with "a slice of non-AmBe tags contributing zero
neutrons." **Still open**, and it matters downstream: if some newly admitted
triggers are not AmBe, the `ambe_triggers` denominator of this tag is
contaminated and any future *absolute* efficiency claim built on it inherits
that. For the purpose this change was made for — more real capture statistics —
the result is a pass regardless: +23.0% genuine captures.

### 7.3 Three caveats on the fits themselves — read before quoting τ

1. **τ = 29.16 ± 0.44 µs is not reproducible from the files on disk, and is
   not the right anchor any more.** Re-running the published recipe on the
   *same 15 runs / 14 positions* gives **30.65 ± 0.30 µs**. The reason is
   bookkeeping, not physics: every `EventAmBeNeutronCandidates_AmBe2.0v4_gated_*.csv`
   was regenerated **2026-07-10**, four days after `AMBE_DATA_PROCESSING_HANDOFF.md`
   (07-06) recorded 29.16. The CSVs that number was fitted on no longer exist.
   Part of the error difference is also that lmfit scales stderr by
   √(χ²/ndof) while `curve_fit(absolute_sigma=True)` does not.
   **New anchor: baseline τ = 30.53 ± 0.26 µs (20 runs, 19 positions).**
2. **Pooled all-position fits are unreliable — don't use them.** χ²/ndof comes
   out 9-14 because positions have genuinely different τ and amplitude, so one
   exponential cannot describe the sum. Per-position fitting then averaging is
   the only defensible route. All τ values quoted above are per-position.
3. **The `A·exp(-t/τ) + B` variant is degenerate here; ignore its numbers.**
   It reports τ_recovered = 24.76 ± 1.49 and a 5.6% flat fraction, which looks
   alarming and is an artifact: `B` sits on its lower bound with ±55 count
   uncertainty, i.e. unconstrained, and 2/19 positions give χ²/ndof ≈ 20. The
   model-free shape tests in §7.2 directly contradict it and they carry no
   model assumption, so they win. This is why §7.2 leads with KS/χ² rather
   than a fitted τ.

Separately, 5 of 19 recovered-sample `NeutCapture` fits were rejected. Three
had `therm` collapse to its bound while returning perfectly sensible τ
(30.94, 30.93, 31.33) — including them would move the recovered average *up*,
toward better agreement, so the rejection is conservative. Two,
(0,100,75) and (75,-100,0), failed hard with τ pinned at 10 µs and χ²/ndof
17-20. Their histograms were inspected directly and **fall normally**
(mean 29.6 and 30.7 µs, against 38.5 µs for flat and 26.8 µs for pure
τ = 30): the failure is the `therm`/`tau` degeneracy — above 10 µs the
`(1-exp(-t/therm))` rise term is ≥86% saturated for any allowed `therm`, so it
is barely identifiable and trades off against τ on small samples. Not a data
problem.

### 7.4 Remaining work

1. **Refine the window bounds** from the acceptance-scan page — 500-2000 ×
   0.15-0.30 is still the untuned starting point of §6.2, now known to be safe.
2. **Cross-campaign check** — v1 (4499) and v3 (5789, `--campaign 2` → `BRF_`)
   through feature extraction, confirming one fprompt window works untuned.
3. Optionally re-derive the efficiency heatmap via
   `configs/data_ambe2v4_fprompt.yaml` for the per-position plots; the
   efficiency numbers themselves are already in §7.1 from the trigger summaries.

### 7.5 Why the efficiency fell — it is the IC widening, not fprompt

Measured, not argued. `timestamp` in the feature dumps equals `eventTankTime` in
the candidate CSVs (exact 1:1), so every accepted tag can be labelled "did this
event yield ≥1 neutron cluster?" and P(neutron | tag) becomes a function of
`IC_adjusted`. Script: `analyze_efficiency_vs_ic.py`.

**Decomposition** (P keeps cosmics in the denominator, so it sits ~22% low in
absolute terms; the steps are ratios, where that cancels):

| step | N tags | P | Δ |
|---|---|---|---|
| baseline IC 700-1200, no fprompt | 364,569 | 0.4572 | — |
| + fprompt 0.15-0.30, same IC window | 362,985 | 0.4573 | **+0.016 pp** |
| + widen IC to 500-2000 (= fprompt2d) | 465,636 | 0.4407 | **−1.663 pp** |

Rescaled by 1/(1−0.22) the widening step is −2.13 pp, against the −1.83 pp
observed in §7.1. **fprompt's own contribution is +0.016 pp — it is not the
cause, and if anything it is very slightly beneficial.**

**fprompt is mildly purifying.** Scored on the *gated* CSVs (which did Stage-2
process fprompt's rejects — the fprompt CSVs cannot answer this, see below), it
discards 0.43% of baseline-accepted tags while losing only 0.40% of the
neutron-bearing ones: P = 0.4211 ± 0.0124 for what it rejects vs
0.4573 ± 0.0008 for what it keeps, ~3σ worse than average.

**The mechanism is the shape of P(neutron | tag) vs IC:**

```
   500- 600  N=  2933  P=0.3174
   600- 700  N= 12729  P=0.3353
   700- 800  N= 42211  P=0.3961   <- old window
   800- 900  N= 78717  P=0.4434   <- old window
   900-1000  N= 90957  P=0.4674   <- old window
  1000-1100  N= 81840  P=0.4736   <- old window
  1100-1200  N= 69260  P=0.4814   <- old window, and the PEAK
  1200-1400  N= 59614  P=0.4397
  1400-1600  N= 15492  P=0.3043
  1600-2000  N= 11883  P=0.2387
```

Plotted in `verbose/fprompt_yield_vs_IC.{pdf,png}` (50-unit bins, the widened
and old windows shaded, tag statistics on a panel below) via
`plot_fprompt_yield_and_capture.py`.

The curve peaks at 1100-1200 and falls away on both sides. **The old 700-1200
window was sitting on that peak**, so any widening necessarily admits
lower-yield events and the ratio can only fall. 700-1200 was close to optimal
for the *ratio* — it was never optimal for the *yield*. The two cannot be
maximised together.

Crucially, **every tag in those flanking bands has already passed fprompt**, so
the residual inefficiency there is *not* identifiable from pulse shape and no
tightening of fprompt can recover it. Physically it is consistent with §3's
region interpretation: low IC is weak / partial-energy tags (a Compton-scattered
or otherwise incomplete gamma, which need not be accompanied by a detectable
neutron), high IC is pile-up, where "one tag" is not one AmBe decay.

That the drop is a **flat pedestal across every port and Y position** (§7.1
table, no geometric structure) is what you expect from a global IC-window
effect rather than anything position- or detector-dependent.

> **Analysis trap worth remembering.** A waveform rejected at Stage 1 is never
> Stage-2 processed, so it cannot appear in that tag's candidate CSV. Asking
> "do fprompt's rejects contain neutrons?" against the *fprompt* CSVs returns
> P = 0.0000 for all 1,584 of them — by construction, not by physics. It has to
> be scored against the *gated* CSVs, which is where the real 0.4211 comes from.

### 7.6 Dataset caveat: run 6242 grew between the two passes

6242 reports 71,412 total waveforms in the baseline CSV but 74,960 in the
fprompt2d pass. The baseline was **not** truncated — every
`AmBeWaveforms_6242_*.root` carries mtime **2026-07-15 12:20**, after the
07-10 baseline pass, so the run legitimately gained ~3,548 acquisitions. It is
the only one of the 20 that differs.

Handled exactly rather than argued away, in both places it could matter:
§7.1's headline efficiency **drops 6242 entirely** (18 positions), and §7.1's
Stage-1 acceptance **re-derives both selections from the same feature dumps**,
so neither side is inflated. The re-derivation is exact — applying
`700 < IC_adjusted < 1200 & ~second_pulse` to
`WaveformFeatures_AmBe2.0v4_fprompt_6062.parquet` reproduces the published
15215 / 38451 exactly. This is the §4 feature-dump payoff in practice: a
denominator mismatch fixed with a pandas query instead of another dCache read.

Note the baseline CSV's own 6242 row is stale as a result — it reports
30,481 / 71,412 for a run that now has 74,960 waveforms.

### Command that produced this (finished, ~1 h 10 min)

```bash
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate
MPLBACKEND=Agg python -u run_waveform_gated_pipeline.py \
    --dataset /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4 \
    --runinfo AmBe2.0v4_fprompt \
    --runs 6046,6056,6060,6061,6062,6165,6166,6186,6187,6188,6189,6230,6231,6232,6234,6235,6237,6239,6241,6242 \
    --selection-mode fprompt2d --pulse-gamma 500 --pulse-max 2000
```

Log: `logs/waveform_fprompt2d_v4_stage12.log`. New tag, so nothing overwrites
`AmBe2.0v4_gated`.

### Reproducing §7.2

```bash
source /exp/annie/app/users/dajana/myboy/bin/activate
MPLBACKEND=Agg python -u fit_capture_time_fprompt_compare.py
```

Self-contained — replicates `ambe.plots.basic`'s histogram, 10-67 µs mask and
model exactly, but builds the recovered-only subset (an eventID diff between
two tags), which `AmBeNeutronAnalyzer` cannot express.

### Offline retuning (no ROOT reads)

```bash
python tune_waveform_fprompt.py --runinfo AmBe2.0v4_fprompt \
    --proposed 500 2000 0.15 0.30
# add --waveform-dir <dataset> --campaign 1 for the mean-waveform overlays
```

## 8. Incidental findings (not fixed here)

- **The AmBe tag channel is campaign-dependent, and the `AmBeWaveform` README is
  only half right.** It says the tag lives in the RWM auxiliary channel; that
  holds for v1 (4499) and v4 (6062), but **v3 (5740, 5789) carries the pulse in
  `BRF_` and its `RWM_` is flat noise**. `campaign=1` → `RWM_`, `campaign=2` →
  `BRF_` already handles this; the misleading comment in
  `run_waveform_gated_pipeline.py` was corrected.
- **Runs 6264, 6265, 6266, 6270** sit in the v4 dataset dir but have no
  `source_positions` entry, so an unfiltered pipeline run raises partway through.
  That is what `--runs` is for. I don't know where those sources were, so no
  positions were invented.
- **`AmBeWaveformResults_AmBe2.0v4_gated.csv` has a row with `run=12377`** —
  that's 6188 + 6189 summed by the position groupby, not a real run.
- **The second-pulse veto rejects 42-45% of all waveforms** using threshold
  `7 + sigma + baseline`, where `sigma` comes from `norm.fit` over the full
  34992-sample record and is therefore inflated by the pulse itself — so the
  threshold varies event-by-event for reasons unrelated to noise. It is applied
  over ~68 µs, where any baseline offset accumulates. Worth a separate look.
- **~5%** of records show activity before bin 265 (pre-pulses) while the fprompt
  window opens at 350.
- `plot_waveform_sample` labels its x-axis "Time (ns)" but plots bin index
  (2 ns/sample). Cosmetic.

## 9. Reference implementation — how it differs, verified against the source

Fetched verbatim from the pinned commit, not paraphrased:

```python
def PMT_wvfm_selection(offbeam_wvfm_v1, fprompt_min=0.001, fprompt_max=0.2, width=15):
    pmt_wvfm_v1 = np.array(offbeam_wvfm_v1[:,0,16,:]*(-1), dtype=np.int64) / 4
    pmt_wvfm_v2 = pmt_wvfm_v1 - np.mean(pmt_wvfm_v1[0:50])
    noise_sum_all  = np.sum(pmt_wvfm_v2[:,5:100], axis=-1)
    noise_sum_fast = np.sum(pmt_wvfm_v2[:,5:18],  axis=-1)
    fast_light = np.sum(pmt_wvfm_v2[:,105:118], axis=-1) - noise_sum_fast
    all_light  = np.sum(pmt_wvfm_v2[:,105:200], axis=-1) - noise_sum_all
    fprompt_mask = ((fast_light/all_light) > fprompt_min)*((fast_light/all_light) < fprompt_max)
    thd_mask     = (pmt_wvfm_v2[:,100:300] > 70)
    pick_out_low = (np.sum(thd_mask[:,5:120], axis=-1) <= width)
    valid_trig   = (fprompt_mask==1) * (pick_out_low==0)
    return valid_trig
```

**The structural difference: there is no integrated-charge window anywhere in
the reference.** Its size gate is `pick_out_low` — the *number of samples* above
70 ADC (absolute bins 105-220) must exceed `width`. So:

| | size / light gate | shape gate | pile-up gate |
|---|---|---|---|
| reference | over-threshold **width** > 15 | fprompt < 0.2 | none explicit |
| here | **`IC_adjusted`** window | fprompt < 0.30 | fprompt > 0.15 |

Three real differences beyond the numeric windows:

1. **width vs IC as the size gate** — see below; measured to be near-equivalent.
2. **Their fprompt band is effectively one-sided.** `fprompt_min = 0.001` is
   zero in practice, so they only reject *high* fprompt (narrow spikes). Our
   `0.15` lower bound is what rejects **pile-up** (long tails dilute the prompt
   fraction; §3: 72% of IC > 2000 sits at fp < 0.15). The reference has no
   equivalent, because `pick_out_low` also targets narrow pulses, not wide ones.
3. **Double baseline correction.** They subtract `mean(first 50 samples)` from
   every sample *and then* subtract a **matched-width** noise-window sum from
   both numerator (13 samples, bins 5-18) and denominator (95 samples, bins
   5-100). We subtract a pre-pulse *median* over bins [0, 250) only. Same
   intent; theirs additionally cancels residual baseline drift.

The bands are not numerically comparable, because the window *ratios* differ:
their numerator is 13/95 = 13.7% of the denominator window, ours is
50/850 = 5.9%. A flat waveform gives fprompt ≈ 0.137 on their definition and
≈ 0.059 on ours, and each band sits above its own flat value.

### width vs IC — measured, not asserted (2026-07-28)

The earlier claim here was that `IC_adjusted` "measures total light better."
That was an assertion; on the 20-run campaign the two are **near-substitutes**:

- Spearman correlation `width_over_thresh` vs `IC_adjusted`: **0.915** over all
  874,100 waveforms, **0.869** within the fprompt2d-accepted set.
- P(neutron | tag) vs width has the *same peaked shape* as vs IC — 0.377 at the
  low end, peaking at **0.472** around width 468-499, falling to 0.343 at the
  high end. Same story, same optimum-in-the-middle.

So swapping IC for a width cut would not have changed the efficiency/yield
trade of §7.5. The decision to keep IC and not add width was harmless — but for
the reason that the two are equivalent here, **not** because IC is better.

**A caveat on how not to read that.** A rank statistic says these variables do
not discriminate at all: AUC = 0.5003 (IC), 0.5005 (width), 0.4955 (peak_amp),
and the medians of neutron-bearing vs not are nearly identical (IC 1008 vs
1004). That is **not** evidence they carry no information — it is a consequence
of the relationship being **non-monotonic**. Both tails are depleted and the
middle is enriched, which a monotonic rank measure like AUC is blind to. The
peaked curves are the real signal; the AUC only rules out using either variable
as a one-sided threshold.

### 9.1 Head-to-head: our IC window vs the reference's structure (2026-07-28)

Script: `compare_ic_vs_reference_selection.py`. Both selections applied to the
same 874,100 waveforms.

```
A = 700 < IC_adjusted < 1200                AND not second_pulse
B = 0.15 < fprompt < 0.30 AND width > W     AND not second_pulse   (no charge window)
```

Thresholds are not transferable from the reference, so W is scanned and the
headline is taken at **matched acceptance**: for the same number of tags, which
selection captures more neutrons?

**Result 1 — the reference's width gate is essentially inert here.** From W = 0
to W = 300 the accepted count moves 478,059 → 474,809 and the neutron count
moves 205,200 → 205,198. The reference's own `width > 15` out of a 115-bin
window (~13%) maps to roughly our `> 110`, which sits squarely in that dead
region. **Transplanted at its nominal setting, their width cut does nothing on
our data** — selection B reduces to the fprompt band alone.

**Result 2 — pushed hard enough to matter, width is the *worse* gate.** At
matched acceptance (W = 450):

| | N tags | coverage | P(neutron) | neutrons |
|---|---|---|---|---|
| **A**: IC window (ours) | 364,569 | 100% | **0.4578 ± 0.0008** | 166,903 |
| **B**: fprompt + width | 330,984 | 97.9% | 0.4495 ± 0.0009 | 145,591 |

A wins on purity *and* keeps more tags. The disagreement is the sharpest part:

| | N | P(neutron) |
|---|---|---|
| kept by both | 236,975 | **0.4720** |
| **A only** — IC keeps, reference drops | 127,594 | 0.4315 |
| **B only** — reference keeps, IC drops | 94,009 | **0.3882** |

What the IC window uniquely keeps is markedly more neutron-rich than what the
width gate uniquely keeps (0.4315 vs 0.3882). On this data IC is the better
size gate — which is what §9's original claim asserted, now actually measured.

**Result 3 — with no charge ceiling, B admits extreme pile-up we cannot even
evaluate.** 7,073 of the B-only waveforms (7.5%) have `IC_adjusted` **above
2000, ranging to 18,489** (median 2,627). No Stage-2 pass ever processed those,
so their neutron content is **unknown, not zero**, and they are excluded from
B's P above. They exist because the reference structure has nothing that can
reject a very large pulse: `pick_out_low` removes *narrow* pulses, and our
`fprompt > 0.15` catches most pile-up but not the ~28% of IC > 2000 that sneaks
through the band (§3).

**Result 4 — adding width on top of IC buys purity at a bad price:**

| selection | N | P(neutron) |
|---|---|---|
| A + fprompt band (our shipped `ic+fprompt`) | 362,985 | 0.4580 |
| + width > 400 | 352,877 | 0.4602 |
| + width > 450 | 236,975 | 0.4720 |
| + width > 500 | 48,413 | 0.4808 |

`width > 450` buys +1.4 pp purity for a 35% statistics loss; `> 500` buys
+2.3 pp for 87%. Not worth it while the analysis is statistics-limited, which
retroactively justifies not cutting on width — again, now for a measured reason.

**Limitation, stated plainly.** Our `width_over_thresh` counts samples above
**10 ADC over 850 bins**; the reference counts above **70 ADC over 115 bins**.
The inertness in Result 1 is partly because a 10 ADC threshold is very low, so
almost everything clears it. A faithful transplant needs the width recomputed at
a 70-ADC-equivalent threshold — that is one config value
(`WaveformConfig.width_threshold`, `src/ambe/data/processor.py:55`) plus a
re-read of the waveforms (~1 h 10 min), and it has **not** been done. Results
2-4 do not depend on it, since they compare gates at matched acceptance rather
than at the reference's absolute numbers.

`width_over_thresh` here = samples above 10 ADC in bins [350, 1200) after
pre-pulse-median subtraction; it is recorded and plotted but still not cut on.
The numeric windows differ throughout because their record is ~1000 samples with
the prompt peak at bin ~115 (inverted, /4, channel 16); ours is 34992 samples
with the peak at bin ~366.
