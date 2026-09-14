# Stage-1 waveform study — does the AmBe small PMT fire with no source, and can that flat background be subtracted?

Separate study, same three runs (6264 no-source / 6265 source / 6266 cross-check),
same deck. New code, run 2026-09-11:

```bash
ambe plots waveformcuts --config configs/data_ambe2v4_effpair_box.yaml
```

- module `src/ambe/plots/waveform_cuts.py`, registered as `("plots","waveformcuts")` in `src/ambe/cli.py`
- new config block `waveform_cuts.sideband_ic` in `configs/data_ambe2v4_effpair_box.yaml`
- figures in `plots_waveform/`, tables `waveform_cutflow.csv` and `waveform_sideband.csv`

Input is `TriggerSummary/WaveformFeatures_AmBe2.0v4_effpair_box_<run>.parquet` — one row
per **recorded waveform**, carrying `IC_adjusted`, `second_pulse` and the `accepted`
verdict. Nothing is re-derived from raw waveforms, so these figures show exactly the
selection the efficiency denominators were built from. Closure: `accepted.sum()` returns
589 / 20,964 / 39,391, the three expected Stage-1 totals.

---

## Answer in one line

**Yes, the small PMT fires in the no-source run — 808 of 18,496 waveforms land inside the
700–1200 IC window — but it produces a smooth continuum, not a 4.4 MeV gamma peak, and it
cannot be subtracted at the waveform stage. It is removed downstream, by the cosmic veto
and then by the multiplicity fit.**

---

## 1. The overlay — `wfcut_ic_overlay`, `wfcut_ic_zoom`

`IC_adjusted` for every recorded waveform, normalised per waveform so the runs are
comparable despite very different lengths.

| | 6264 (no-source) | 6265 (source) | 6266 (cross-check) |
|---|---:|---:|---:|
| recorded waveforms | 18,496 | 41,248 | 75,278 |
| in IC window 700–1200 | 808 (4.37 %) | 29,442 (71.38 %) | 53,933 (71.65 %) |
| Stage-1 accepted | 589 (3.18 %) | 20,964 (50.82 %) | 39,391 (52.33 %) |
| median IC | **2,549** | **926** | **916** |
| second-pulse veto rate, all | 33.1 % | 28.5 % | 26.9 % |
| second-pulse veto rate, in window | 27.1 % | 28.8 % | 27.0 % |

What to look at:

- **6265 and 6266 have a clear peak at IC ≈ 850–950.** That is the AmBe 4.4 MeV gamma in
  the source PMT. It is what the 700–1200 window was cut to select, and the window sits
  right on it.
- **6264 has no peak at all.** Its distribution is a low-IC pedestal spike (27 % of its
  waveforms below IC = 100) plus a broad cosmic population at IC ≈ 5,000, and between them
  a genuinely **flat continuum** — ~0.6–0.9 % of waveforms per 100 IC, essentially
  constant from 600 all the way to 2,500. The window sits on that flat part.
- **Inside the window the no-source level is ~20× below the source peak** (≈2×10⁻³ vs
  ≈4×10⁻² per bin, per recorded waveform). The overlap region is where the whole question
  lives.
- **The second-pulse veto is source-blind**: 27–29 % in every run, in and out of the window.
  It is removing pile-up, not a source-correlated population.
- In `wfcut_ic_zoom` the Stage-1-accepted subset is drawn **filled** under each outline, so
  the window edges and the veto are both visible without ruling reference lines on the plot.

This already answers "how would you remove it with a cut": **you can't.** A narrower IC
window removes the flat background and the signal in the same proportion, because under the
window the background is flat and the signal is a broad peak sitting on top of it. That is
the same conclusion `ambe data icscan` reached from the other direction — efficiency
**plateaus** over 800–1250 rather than peaking, so there is no window choice that buys
purity for free.

## 2. The cut flow — `wfcut_cutflow`

Percent of recorded waveforms surviving each cut, Stage 1 and Stage 2 on one axis:

| | recorded | IC window | + 2nd-pulse veto (Stage-1 accepted) | − cosmic veto (ambe_triggers) |
|---|---:|---:|---:|---:|
| 6264 (no-source) | 18,496 | 808 | 589 | **22** |
| 6265 (source) | 41,248 | 29,442 | 20,964 | **15,628** |
| 6266 (cross-check) | 75,278 | 53,933 | 39,391 | **30,681** |

**The no-source run does not collapse at the IC cut — it collapses at the cosmic veto.**
589 → 22 is a 96.3 % kill. Almost everything the small PMT tagged in the no-source run was
an event with a prompt or bright tank cluster, i.e. a cosmic. That is the single most
useful thing in this study: the waveform stage can flag these triggers but cannot reject
them; the *tank* does the rejecting, one stage later.

The source runs lose only ~25 % at the same step.

## 3. The subtraction attempt — `wfcut_sideband_6265`, `wfcut_sideband_6266`

The only waveform-level subtraction available without livetime: normalise the no-source IC
shape to the source run's **high-IC sideband**, where the source contributes least, then
read off the area of the scaled template under the window.

    s = N_source(sideband) / N_no-source(sideband),   sideband = 1600–2500

| | 6265 vs 6264 | 6266 vs 6264 |
|---|---:|---:|
| scale s | 0.742 | 1.298 |
| flat component of in-window waveforms | 600 / 29,442 = **2.04 %** | 1,049 / 53,933 = **1.95 %** |
| flat component of Stage-1 accepted | 437 / 20,964 = **2.08 %** | 765 / 39,391 = **1.94 %** |
| flat component of `ambe_triggers`, after the cosmic veto | 16 / 15,628 = **0.10 %** | 28 / 30,681 = **0.09 %** |

Two things to take from this:

1. **Two independent runs give the same answer at every level** (2.08 vs 1.94 %, 0.10 vs
   0.09 %), and they agree at every sideband choice I scanned. The method is internally
   consistent.
2. **After the cosmic veto the source-independent trigger background is ~0.1 %** — three
   orders below anything in the efficiency error budget. At the *trigger* level the flat
   background is simply not a problem.

In the figure itself, the black "source − scaled no-source" curve lies on top of the blue
source curve through the entire peak. Visually: **the subtraction removes nothing.** That
is the result, not a failure of the plot.

### Every number here is an upper bound, and the sideband choice matters

The source run's sideband is flat background **plus the high-IC tail of its own gamma peak**,
so `s` is overestimated at every choice. There is no per-run livetime anywhere in this repo
to normalise by instead — the same reason δ_bt is unquotable in the efficiency fit (the
waveform `timestamp` column is absolute epoch with a zero minimum, so its span is not a
livetime).

Sideband dependence, both runs:

| sideband | 6265 | 6266 |
|---|---:|---:|
| 1400–2500 | 3.48 % | 3.30 % |
| **1600–2500 (config)** | **2.08 %** | **1.94 %** |
| 1600–3000 | 1.49 % | 1.43 % |
| 2000–4000 | 0.50 % | 0.55 % |

A factor of ~7 across plausible choices, and it has not converged by 2000–4000 — the source
tail extends that far. So the honest statement is a **bound, O(0.5–3.5 %) of Stage-1
accepted**, not a measurement. This is why `waveform_cuts.sideband_ic` has **no default**
and the figure is skipped outright if the config does not set it: the choice is an analyst
decision that has to be recorded, not a constant hidden in plotting code.

## 4. Where the flat background actually gets removed

The word "flat background" is doing double duty, and separating the two is the point of
this study:

**(a) Flat background of TRIGGERS** — the small PMT fires without a source gamma. Measured
above: ≤2.1 % of Stage-1 accepted, **~0.1 % of `ambe_triggers`** once the cosmic veto has
run. It is never subtracted. It sits in the *denominator*, where at 0.1 % it dilutes ε_n
far below the quoted errors. Nothing needs to be done about it.

**(b) Flat background of delayed CLUSTERS inside otherwise good triggers** — accidental
clusters in the 2–67 µs window that look like neutron captures. **This is the one that
matters, and it is not a waveform quantity at all.** No IC cut can see it, because by
construction it lives inside triggers that passed every waveform cut.

It is removed in the multiplicity fit, two ways, and that is the direct link back to
`README.md`:

- **Poisson model** — it is the fitted parameter **λ_n = 0.076 candidates per trigger**,
  extracted from the shape of the multiplicity distribution (the M ≥ 2 bins, which one AmBe
  neutron cannot populate alone). λ_n agrees between 6265 (0.076) and 6266 (0.084), so it
  is a detector property, as it should be.
- **Data-driven model** — it is 6264's own measured multiplicity histogram
  **[17, 3, 2, 0, …]** over its 22 surviving triggers, convolved with the Bernoulli.

So run 6264 earns its keep twice over, and at two different stages: at the waveform stage
it shows the IC window is sitting on a flat, irreducible continuum; at the cluster stage its
22 surviving triggers *are* the background template. And its weakness is the same in both
places — 22 triggers is not much, which is exactly why the data-driven fit's ±0.108 is
twenty times the Poisson error and why its flattering χ²/ndof = 0.44 is an artefact.

## 5. Caveats

1. The sideband numbers are **upper bounds** and are **sideband-dependent by a factor ~7**
   (§3). Quote them as bounds.
2. **No livetime**, so nothing here is a rate. All normalisations are per recorded waveform,
   and the runs' recording rates are not directly comparable — the DAQ dead time τ ≈ 2 s
   caps the recorded rate at 0.5 Hz, so recorded-waveform counts are not proportional to
   exposure.
3. 6264 is the no-source run that measures the IC false-start rate. It is **not** a
   cosmic-free run — 96.3 % of its Stage-1 survivors are cosmic-vetoed, and it has real
   cosmic-induced captures. Do not read its 22 triggers as "pure accidentals".
4. These figures characterise the **IC window position**, not the window *width* choice.
   `ambe data icscan` does the latter, and found the efficiency plateaus over 800–1250 with
   a cliff above 1250. The two should be read together.
5. The 2–67 µs delayed window and the box cuts are untouched here — this study stops at the
   trigger level.
