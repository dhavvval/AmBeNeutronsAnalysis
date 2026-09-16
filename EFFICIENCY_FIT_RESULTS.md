# Neutron detection efficiency by multiplicity fit — runs 6264 / 6265

Thesis chapter 8.6 method (Pershing), applied to ANNIE 2.0 data. This is an
application of the method to a **different dataset**, not a reproduction of the
thesis's four numerical efficiencies — see "Dataset-specific adaptations" below.

Generated 2026-09-10, revised 2026-09-11 after an external code review. Reproduce:

```bash
ambe data process     --config configs/data_ambe2v4_effpair_box.yaml --selection box
ambe stats efficiency --config configs/data_ambe2v4_effpair_box.yaml
ambe plots efficiency --config configs/data_ambe2v4_effpair_box.yaml
```

## Headline

**ε_n = 0.605 +0.005(stat) −0.004(stat) +0.035/−0.000(sys)**, run 6265 at position
(0,0,0), uncorrelated (Poisson) background model, `ambe_triggers` denominator,
self-consistent non-cosmic numerator. Delayed window [2, 67] µs, box cuts.

**This number is preliminary and the code says so at runtime.** The nominal model
fails its goodness-of-fit gate (p < 10⁻³ from 2,000 pseudo-experiments) and is quoted
only under an explicit written override recorded in the config and in the output
JSON. See "Why neither model is safe" below.

Three things changed in the 2026-09-11 revision and moved the quoted numbers:

- **The statistical interval is now the profile Δχ²=1 interval** (±0.005/0.004), not
  the projection of the Δχ²=2.30 joint two-parameter region (±0.007/0.006). The old
  code profiled over λ and *then* applied 2.30, which is neither statement. The
  2.30 projection is still reported, as `joint_projection_interval`.
- **The systematic is asymmetric.** The data-driven fit sits *above* the nominal one,
  so the model term is one-sided: +0.035/−0.000, not ±0.035.
- **δ_bt no longer enters the total at all.** Without livetime it is a trigger-count
  ratio, not the rate ratio of eq. 8.9; it is now reported as a diagnostic and
  excluded. It was previously folded into the quadrature sum.

The systematic remains a **lower bound**: δ_h (housing captures) is still not
evaluated, and δ_bt is still unquotable for want of per-run livetime.

For comparison, the naive counting ratio on the same events is 0.635. The fit is
lower because it attributes part of the observed candidate rate to background —
which is the entire point of the method.

## Why neither background model is safe here

The review asked for the nominal model to be configurable rather than hard-coded.
It is now a **required** config key with no default, because on this dataset neither
choice is defensible by itself:

| | ε_n | χ²/ndof (thesis) | p-value | verdict |
|---|---:|---:|---:|---|
| uncorrelated (Poisson) | 0.605 | 51.4/6 | <10⁻³ | bad fit, **known** failure mode |
| data-driven | 0.640 | 3.07/7 | 0.39 | good-looking fit, **wrong reason** |

The Poisson model underestimates the high-multiplicity tail — 46 of the 51.4 comes
from bins 3–4. This is the same failure the thesis reports in §8.6.2 (χ²/ndof = 11/5
at position 0); our sample is 1.6× larger, so the same physical mismatch carries more
leverage. It means the real background is not Poisson in time, which is a statement
about the detector rather than a defect in the extraction.

The data-driven fit is not the better alternative. Its p ≈ 0.39 is an artefact of
background run 6264 having **22 AmBe triggers**: the template's per-bin uncertainty
dominates the χ² denominator from bin 2 upward, so large residuals go unpenalised.
Removing σ_bkg moves the same fit from 0.640 to 0.351 with χ²/ndof ≈ 854. Run
`src/ambe/diagnostics/datadriven_chi2.py --source 6265 --bkg 6264` to reproduce the scan;
it also shows the crossover — the background run would need roughly N ≳ 50 triggers
before its χ²/ndof stops being dominated by its own statistics.

Poisson is therefore nominal, and δ_m (the **signed** distance to the data-driven fit)
is carrying real model uncertainty rather than noise.

## Effect of the corrected statistical treatments

Both are opt-in (`likelihood`, `background_uncertainty` in `fit_params`); the thesis
behaviour stays the default so published comparisons hold. Measured shifts:

| run | Poisson χ²→multinomial | data-driven diagonal→bootstrap |
|---|---:|---:|
| 6265 | 0.605 → 0.601 (−0.004) | 0.640 ±0.108 → 0.589 ±0.028 |
| 6046 | 0.653 → 0.651 (−0.002) | 0.698 ±0.108 → 0.635 ±0.026 |

The multinomial likelihood barely moves the Poisson fit — the diagonal Gaussian
approximation is adequate at this sample size, which is worth knowing rather than
assuming. The background treatment is a different story: propagating the template
uncertainty *through the Bernoulli convolution* instead of adding it per-bin narrows
the data-driven interval by a factor of ~4 and moves the central value down by
0.05–0.06. The thesis per-bin formula is badly conservative when the background run
is this small, which is exactly the regime the χ²/ndof artefact lives in.

## Why the fit differs from naive counting

Naive counting asks "what fraction of triggers had ≥1 candidate". That folds
background candidates into the efficiency. The fit instead models each acquisition as

    M = δ_S + N_B,   δ_S ~ Bernoulli(ε_n)

and uses the *shape* of the multiplicity distribution — in particular the
multiplicity ≥ 2 bins, which a single AmBe neutron cannot populate on its own — to
separate ε_n from the background rate λ_n.

## All variants

Statistical errors are the profile Δχ²=1 interval. χ² is reported against both the
thesis dof convention (n_bins − n_fitted) and the standard one (n_bins − 1 −
n_fitted), which subtracts the normalisation constraint; the code no longer picks one
silently.

| denominator | numerator | model | ε_n | stat | λ_n | χ²/ndof<sub>thesis</sub> | χ²/ndof<sub>std</sub> | syst | naive |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ambe_triggers | noncosmic | uncorrelated **(NOMINAL)** | **0.605** | +0.005/−0.004 | 0.076 | 51.4/6 | 51.4/5 | +0.035/−0.000 | 0.635 |
| ambe_triggers | noncosmic | data-driven | 0.640 | +0.108/−0.108 | — | 3.1/7 | 3.1/6 | +0.035/−0.000 | 0.635 |
| ambe_triggers | production | uncorrelated **(NOMINAL)** | 0.642 | +0.004/−0.004 | 0.076 | 51.6/6 | 51.6/5 | +0.045/−0.000 | 0.669 |
| ambe_triggers | production | data-driven | 0.687 | +0.108/−0.109 | — | 3.3/7 | 3.3/6 | +0.045/−0.000 | 0.669 |
| total_events | noncosmic | uncorrelated **(NOMINAL)** | 0.429 | +0.005/−0.003 | 0.078 | 50.7/6 | 50.7/5 | +0.011/−0.000 | 0.473 |
| total_events | noncosmic | data-driven | 0.440 | +0.006/−0.005 | — | 98.1/7 | 98.1/6 | +0.011/−0.000 | 0.473 |
| total_events | production | uncorrelated **(NOMINAL)** | 0.457 | +0.004/−0.004 | 0.078 | 51.0/6 | 51.0/5 | +0.007/−0.000 | 0.499 |
| total_events | production | data-driven | 0.464 | +0.005/−0.006 | — | 111.0/7 | 111.0/6 | +0.007/−0.000 | 0.499 |

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

1. **The Poisson fit fails its goodness-of-fit gate** (χ²/ndof = 51.4/6, p < 10⁻³).
   Not a code defect — see "Why neither background model is safe" above. The driver
   now **refuses to print a headline** in this situation unless the config supplies
   `nominal_model_override_justification`, which is then recorded next to the number
   in the JSON. That justification is written out in
   `configs/data_ambe2v4_effpair_box.yaml`.

2. **6264 has only 22 AmBe triggers.** Its background template is nearly a delta at
   zero, so the data-driven fit is statistics-starved: ±0.108, more than twenty times
   the Poisson error. It is a systematic cross-check, not a competing measurement,
   and its flattering χ²/ndof is an artefact of exactly this. The code emits both
   caveats automatically.

3. **δ_h is NOT EVALUATED.** No MC housing-capture fraction exists for this campaign.
   The code refuses to substitute zero. Convention, now enforced: `delta_h` is a
   **correction** added to the efficiency and `delta_h_unc` is the **systematic**, so
   a bare fraction with no uncertainty contributes nothing and the code says so.
   Prefer supplying `housing_n_captures` / `housing_n_simulated` and letting the
   fraction and its Poisson error be derived. (For Teal's numbers that is 31/10,000 =
   0.0031 ± 0.00056. Thesis eq. 8.8 prints a 1E5 denominator; the prose says 31 of
   10,000, so the equation has a typo.)

4. **δ_bt is a count ratio, not a rate ratio, and is now EXCLUDED from the total.**
   Thesis eq. 8.9 needs livetime, which is not in any file here (the waveform
   `timestamp` column is absolute epoch with a zero minimum, so its span is not a
   livetime). It is reported as a diagnostic only. Supply `systematics.livetime_s`
   (and optionally `livetime_unc_s`) per run to get a publishable number with
   propagated uncertainty.

5. **Do not compare against the published 57.23 %.** That is the 28-run campaign
   number; 6265 is a special run deliberately excluded from the campaign map. Note
   also that the campaign number was produced by the order-dependent cluster loop,
   which is now off by default — reproducing it requires `stage1.legacy_ordering:
   true`.

6. **Run 6046 has no epoch-matched background run.** 6264 belongs to 6265's
   deployment window; 6046 is from the March 2026 sub-campaign (6046, 6056, 6060,
   6061, 6062), for which no no-source run has been processed anywhere in this repo.
   Its deck reuses 6264 anyway and says so. Separately verified: the misleading
   χ²/ndof is driven by 6264's *statistics*, not by the epoch mismatch — pairing
   6264 with its own run 6265 gives the same artefact (p = 0.39 vs 0.41).

## Dataset-specific adaptations

The thresholds below are **not** the thesis's, and this analysis does not claim to
reproduce its numbers. They are adaptations for ANNIE 2.0 data:

| cut | here | thesis | note |
|---|---|---|---|
| charge balance | < 0.45 | < 0.4 | looser; retuned for this detector configuration |
| cluster PE | ≤ 100 | — | also serves as the cosmic-veto threshold |
| cluster hits | ≥ 5 | — | |
| delayed window | [2, 67] µs | — | now enforced at candidate construction, not assumed |
| Stage-1 gate | IC window + second-pulse veto | — | no thesis equivalent |

**These were not optimised on the same data used for the final measurement in any
documented way**, and no per-cut systematic has been assigned for threshold
variation. Both are gaps. `ambe data icscan` characterises the IC-window
sensitivity; nothing equivalent exists for the box thresholds. Until it does, the
efficiency should be read as conditional on this cut set rather than as a
detector property.

## Validation

95 tests across `tests/test_efficiency_fit.py`, `tests/test_selection.py` and
`tests/test_processor_ordering.py`, all passing. The ones that matter:

- **Frozen real-data fixture** — recorded aggregate bin counts from the validated run
  go in, the previously validated ε_n and λ_n come out. Nothing in it is generated by
  the model under test, so unlike the closure test below it can actually catch a
  changed convention. This is the genuine regression test.
- **Analytic ≡ Teal's toy MC** — the closed-form pmf and the vendored
  `ProfileLikelihoodBuilder` agree to < 2×10⁻³ (MC noise) on both models, so the
  speedup is not a change of model. With the fixture above, this is the real fidelity
  evidence.
- **Poisson closure at the thesis reference point** — fed a histogram generated from
  `analytic_profile_poisson(0.64, 0.073)`, the fitter returns those values. This is
  **circular** and now says so in its name and docstring: it tests the optimiser, not
  the physics. It was previously named `test_reproduces_teal_position_0` and billed
  as "the strongest single check that the port is faithful", which it is not.
- **Interval coverage** — 1,000 pseudo-experiments each, asserting that the Δχ²=1
  profile interval covers at 68.3 % within 4 binomial σ, and separately that the
  Δχ²=2.30 projection **over**covers. Coverage is the property that matters; the old
  test checked only a pull width inside a 0.3–1.6 band over 120 throws, which admits
  error bars 3× too large and is close to vacuous.
- **Cut boundaries** — every edge of the box pinned explicitly (1999/2000/66999/
  67000/67001 ns, PE at 100/100.01, CB at 0.45, hits at 5), for both `BoxSelection`
  and `MvaSelection`. There were no selection tests at all before.
- **Cluster-order invariance** — every permutation of a multi-cluster event must give
  the same cosmic verdict and the same candidate multiplicity. This is the test the
  two-pass processor fix exists to satisfy.
- **Histogram integrity** — `counts.sum() == denominator` for every histogram, and
  overflow either fails loudly or clips the same way the models do.
- **Goodness-of-fit calibration** — the pseudo-experiment p-value is ~uniform when the
  model is the truth, and correctly flags the real 6265 Poisson fit as bad.
- **Systematics directionality** — signed model term, one-sided housing and
  background-trigger terms, `delta_h_unc` actually changing the total, and livetime
  absence blocking a publishable δ_bt.
- **Zero-background limit** — reduces to naive counting exactly.
- **Stage-1 closure** — the reprocess reproduces 589/567 and 20,964/5,336 exactly,
  matching three independent prior derivations.
- **Legacy reproduction** — `production` mode returns 0.6691, the number the existing
  pipeline publishes, confirming the port is faithful to what it replaces.

## Implementation

| file | role |
|---|---|
| `src/ambe/stats/efficiency_fit.py` | histograms, both models, intervals, GoF, systematics, driver |
| `src/ambe/plots/efficiency_thesis.py` | thesis figs 8.18–8.24 analogues + capture time |
| `src/ambe/stats/profile_likelihood.py` | Teal's classes, kept as reference; four defects fixed |
| `src/ambe/data/selection.py` | the neutron definitions + the shared delayed window |
| `src/ambe/data/processor.py` | two-pass event loop; `legacy_ordering` for the old one |
| `src/ambe/diagnostics/datadriven_chi2.py` | the σ_bkg-dominance scan behind §"Why neither model" |
| `configs/data_ambe2v4_effpair_box.yaml` | the deck, with `dump_event_index: true` |
| `tests/test_efficiency_fit.py` | fitter closure, coverage, GoF, systematics |
| `tests/test_selection.py` | cut boundaries for both selections |
| `tests/test_processor_ordering.py` | cluster-order invariance and the window, end to end |
| `src/ambe/plots/phase2.py` | retired; original preserved as a string, now imports cleanly |

### Config keys added in the 2026-09-11 revision

`cuts.time_min_ns`, `cuts.time_max_ns`, `cuts.cluster_hits_min` — authoritative, and
cross-checked against `fit_params.acquisition_window_ns` at startup (previously
`ct_min`/`chits_min` were unreachable dataclass defaults and there was no upper bound
at all). `fit_params.nominal_model` (**required**), `gof_min_p_value`, `n_gof_toys`,
`nominal_model_override_justification`, `likelihood`, `background_uncertainty`,
`n_bootstrap`, `overflow_policy`. `systematics.livetime_unc_s`,
`housing_n_captures`, `housing_n_simulated`. `stage1.legacy_ordering`.

The 2D scan is vectorised (whole (ε, λ, bin) cube in one broadcast) — 190× faster
than the nested loop, which is what makes the 1,000-experiment coverage tests
practical.
