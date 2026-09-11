# Neutron detection efficiency by multiplicity fit — runs 6264 / 6265

Thesis chapter 8.6 method (Pershing), implemented for ANNIE 2.0 data.
Generated 2026-09-10. Reproduce with:

```bash
ambe data process     --config configs/data_ambe2v4_effpair_box.yaml --selection box
ambe stats efficiency --config configs/data_ambe2v4_effpair_box.yaml
ambe plots efficiency --config configs/data_ambe2v4_effpair_box.yaml
```

## Headline

**ε_n = 0.605 +0.007(stat) −0.006(stat) ±0.035(sys)**, run 6265 at position (0,0,0),
uncorrelated (Poisson) background model, `ambe_triggers` denominator, self-consistent
non-cosmic numerator. Delayed window [2, 67] µs, box cuts.

The systematic is a **lower bound**: δ_h (housing captures) is not evaluated, and
δ_bt is a trigger-count ratio rather than the rate ratio of eq. 8.9 because per-run
livetime is not recorded anywhere in this repo.

For comparison, the naive counting ratio on the same events is 0.635. The fit is
lower because it attributes part of the observed candidate rate to background —
which is the entire point of the method.

## Why the fit differs from naive counting

Naive counting asks "what fraction of triggers had ≥1 candidate". That folds
background candidates into the efficiency. The fit instead models each acquisition as

    M = δ_S + N_B,   δ_S ~ Bernoulli(ε_n)

and uses the *shape* of the multiplicity distribution — in particular the
multiplicity ≥ 2 bins, which a single AmBe neutron cannot populate on its own — to
separate ε_n from the background rate λ_n.

## All variants

| denominator | numerator | model | ε_n | ±stat | λ_n | χ²/ndof | naive |
|---|---|---|---:|---:|---:|---:|---:|
| ambe_triggers | noncosmic | **uncorrelated (PRIMARY)** | **0.605** | +0.007/−0.006 | 0.076 | 51.4/6 | 0.635 |
| ambe_triggers | noncosmic | data-driven | 0.640 | ±0.108 | — | 3.1/7 | 0.635 |
| ambe_triggers | production | uncorrelated | 0.642 | ±0.006 | 0.076 | 51.6/6 | 0.669 |
| ambe_triggers | production | data-driven | 0.687 | ±0.108 | — | 3.3/7 | 0.669 |
| total_events | noncosmic | uncorrelated | 0.429 | ±0.006 | 0.078 | 50.7/6 | 0.473 |
| total_events | noncosmic | data-driven | 0.440 | ±0.005 | — | 98.1/7 | 0.473 |
| total_events | production | uncorrelated | 0.457 | ±0.006 | 0.078 | 51.0/6 | 0.499 |
| total_events | production | data-driven | 0.464 | ±0.006 | — | 111.0/7 | 0.499 |

Every convention is reported because the choice moves the answer by more than any
uncertainty in it. Two choices matter:

**Denominator.** `ambe_triggers = total − cosmic` is the thesis definition (§8.6) and
the repo's own convention (`icscan.py`, `heatmap.py`). Using `total_events` instead
drops ε_n by ~0.18 — far larger than the quoted errors.

**Cosmic numerator.** The production accumulator counts a cosmic-vetoed event in the
numerator while subtracting it from the denominator (documented at
`icscan.py:368-406`). On 6265 that is **532 of 15,628 AmBe triggers**, worth
**3.4 percentage points** — larger than the whole systematic budget of thesis table
8.6. `noncosmic` is self-consistent and is the default; `production` reproduces the
legacy number (0.6691) exactly and exists only for comparison.

## Inputs

| | 6264 (background) | 6265 (source) |
|---|---:|---:|
| Stage-1 admitted triggers | 589 | 20,964 |
| cosmic-vetoed | 567 (96.3 %) | 5,336 (25.5 %) |
| **ambe_triggers** | **22** | **15,628** |
| multiplicity (noncosmic) | [17, 3, 2, 0, …] | [5704, 9192, 639, 82, 9, 1, 0, 1] |
| naive efficiency | 0.227 | 0.635 |
| capture-time τ, [15,67] µs | — | **31.2 ± 3.9 µs**, χ²/ndof 1.24 |

Run 6266 (cross-check, position (0,100,0)): 39,391 triggers, 8,710 cosmic-vetoed,
30,681 ambe_triggers, naive efficiency 0.578.

**6265 is source-like on the data**: its delayed clusters fit a clean Gd-capture
exponential with τ = 31.2 ± 3.9 µs, consistent with Teal's 37 ± 7 µs and this repo's
30.5 µs. 6264's 7 clusters are sparse and flat in time. This settles the labelling —
`data_ambe2v4_special_box.yaml` calls both runs "no-source", which is wrong for 6265.

## Cross-check: run 6266 at an independent position

| run | position | ambe_triggers | naive | ε_n (Poisson) | λ_n |
|---|---|---:|---:|---:|---:|
| 6265 | (0, 0, 0) | 15,628 | 0.635 | **0.605 ± 0.007** | 0.076 |
| 6266 | (0, 100, 0) | 30,681 | 0.578 | **0.540 ± 0.005** | 0.084 |

Two things support the extraction:

1. **The positional trend is right.** Efficiency falls as the source moves away from
   the tank centre — 0.605 at centre, 0.540 at y = +100 cm. Thesis fig. 8.29 shows
   the same behaviour (0.64 → 0.35 over a comparable range).
2. **λ_n agrees across independent runs** (0.076 vs 0.084). The background rate is a
   property of the detector, not of the source position, so consistency here is a
   genuine check — it is the cross-check Teal performs in table 8.5, where his four
   positions give 0.067–0.073.

## Caveats that affect how the number should be quoted

1. **The Poisson fit's χ²/ndof is 51/6.** Not a code defect: 46 of the 51 comes from
   multiplicity bins 3–4, where the model *underestimates the tail*. This is exactly
   the failure Teal reports in §8.6.2 — "the uncorrelated background model fit
   underestimates the high multiplicity tail" — where he gets χ²/ndof = 11/5 at
   position 0. Our sample is 1.6× larger, so the same mismatch has more leverage. It
   means the true background is **not** purely uncorrelated in time, and the honest
   reading is that δ_m (0.035) covers real model uncertainty rather than noise.

2. **6264 has only 22 AmBe triggers.** Its background template is nearly a delta at
   zero, so the data-driven fit is statistics-starved: ±0.108, sixteen times the
   Poisson error. It is reported as a systematic cross-check, not a competing
   measurement. The code emits this caveat automatically.

3. **δ_h is NOT EVALUATED.** No MC housing-capture fraction exists for this campaign.
   The code refuses to substitute zero. Supply
   `systematics.delta_h_housing_fraction` when an MC number exists.

4. **δ_bt is a count ratio, not a rate ratio.** Thesis eq. 8.9 needs livetime, which
   is not in any file here (the waveform `timestamp` column is absolute epoch with a
   zero minimum, so its span is not a livetime). Supply
   `systematics.livetime_s` per run to get the defined quantity.

5. **Do not compare against the published 57.23 %.** That is the 28-run campaign
   number; 6265 is a special run deliberately excluded from the campaign map.

## Validation

20 tests in `tests/test_efficiency_fit.py`, all passing. The ones that matter:

- **Teal reproduction** — fed the histogram his published best fit implies, the
  fitter returns ε_n = 0.640 and λ_n = 0.073 with χ² = 0.11, matching thesis tables
  8.4 and 8.5 exactly.
- **Analytic ≡ Teal's toy MC** — the closed-form pmf and the vendored
  `ProfileLikelihoodBuilder` agree to < 2×10⁻³ (MC noise) on both models, so the
  speedup is not a change of model.
- **Injection–recovery** — central value recovered to < 0.02 over 120
  pseudo-experiments; pull mean and width checked, which is the only test of whether
  the Δχ² interval is really a 1σ interval.
- **Zero-background limit** — reduces to naive counting exactly.
- **Stage-1 closure** — the reprocess reproduces 589/567 and 20,964/5,336 exactly,
  matching three independent prior derivations.
- **Legacy reproduction** — `production` mode returns 0.6691, the number the existing
  pipeline publishes, confirming the port is faithful to what it replaces.

## Implementation

| file | role |
|---|---|
| `src/ambe/stats/efficiency_fit.py` | histograms, both models, χ² intervals, systematics, driver |
| `src/ambe/plots/efficiency_thesis.py` | thesis figs 8.18–8.24 analogues + capture time |
| `src/ambe/stats/profile_likelihood.py` | Teal's classes, kept as reference; four defects fixed |
| `configs/data_ambe2v4_effpair_box.yaml` | the deck, with `dump_event_index: true` |
| `tests/test_efficiency_fit.py` | the closure suite |
| `src/ambe/plots/phase2.py` | retired; original preserved as a string, now imports cleanly |

The 2D scan is vectorised (whole (ε, λ, bin) cube in one broadcast) — 190× faster
than the nested loop, which made the 120-experiment pull test practical.
