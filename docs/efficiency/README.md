> Historical analysis note: results and commands describe the original run. See [the current workflow guide](../WORKFLOWS.md) and [output locations](../OUTPUTS.md) before rerunning. No numerical results were regenerated during cleanup.

# Efficiency extract — Poisson vs data-driven, the event streamline, and the cosmic veto

Pulled out of `EFFICIENCY_FIT_RESULTS.md`, `configs/data_ambe2v4_effpair_box.yaml`,
`src/ambe/efficiency_extraction/efficiency_fit.py`, `src/ambe/data/{selection,processor,icscan}.py` and the
`AmBe2.0v4_effpair_box` outputs. Nothing here is recomputed — it is the 2026-09-11 run,
copied so the numbers, the plots and the reasoning sit in one place.

Source of every number: `efficiency_fit.csv` / `efficiency_fit.json` in the repository’s `EFFICIENCY_EXTRACT/` directory
(copies of `ambe_output/ambe_data/AmBe2.0v4_effpair_box/csv/`).

---

## 1. The two efficiency results

Both models fit the **same** multiplicity histogram from run 6265. They differ only in what
they assume about the background term.

Each acquisition is modelled as

    M = δ_S + N_B        δ_S ~ Bernoulli(ε_n)

M is the number of delayed neutron candidates in the trigger. δ_S is "did the one AmBe
neutron get detected" — 0 or 1, never more. N_B is however many background candidates
happened to land in the same 65 µs. The fit uses the **shape** of the distribution:
multiplicity ≥ 2 bins cannot be populated by the AmBe neutron alone, so they measure the
background, and the rest measures ε_n.

### 1a. Poisson / uncorrelated background (NOMINAL)

Background candidates arrive independently at a constant rate λ_n per trigger:

    P(M=k) = ε_n · Pois(k−1; λ_n) + (1 − ε_n) · Pois(k; λ_n)

λ_n is a **fitted** parameter, so this is a 2-parameter fit and its χ² carries no
background-template uncertainty.

| | value |
|---|---|
| **ε_n** | **0.605  +0.005 / −0.004 (stat)  +0.035 / −0.000 (sys)** |
| λ_n | 0.076 +0.003/−0.004 candidates per trigger |
| stat interval | profile Δχ²=1: [0.6007, 0.6096] |
| joint projection | Δχ²=2.30: [0.5986, 0.6118] (thesis convention, overcovers) |
| χ² | 51.4 / 6 (thesis ndof) = 51.4 / 5 (standard) |
| p-value | < 10⁻³, from 2,000 multinomial pseudo-experiments |
| naive counting ratio | 0.635 |

### 1b. Data-driven background (cross-check)

The background term is not assumed — it is the **measured** multiplicity histogram of
no-source run 6264, convolved with the same Bernoulli:

    P(M=k) = ε_n · B(k−1) + (1 − ε_n) · B(k)

with B = 6264's normalised histogram. One free parameter (ε_n). The template's per-bin
statistical error is added to the χ² denominator (thesis eq. 8.5).

| | value |
|---|---|
| **ε_n** | **0.640 ± 0.108 (stat)** |
| χ² | 3.07 / 7 = 0.44 |
| p-value | 0.38 |

### 1c. Why the good-looking fit is the one you should not quote

| model | ε_n | χ²/ndof | p | verdict |
|---|---:|---:|---:|---|
| uncorrelated (Poisson) | 0.605 | 51.4/6 | <10⁻³ | bad fit, **known** failure mode |
| data-driven | 0.640 | 3.07/7 | 0.38 | good fit, **wrong reason** |

- Poisson fails because it underestimates the high-multiplicity tail — 46 of the 51.4 χ²
  comes from bins 3–4. This is the same failure the thesis reports in §8.6.2 (χ²/ndof = 11/5
  at position 0). It says the real background is **not Poisson in time**: a statement about
  the detector, not a bug in the extraction.
- Data-driven looks fine only because **6264 has 22 AmBe triggers**. Its template
  uncertainty dominates the χ² denominator from bin 2 upward, so big residuals go
  unpenalised. Drop σ_bkg and the same fit moves 0.640 → 0.351 with χ²/ndof ≈ 854.
  Reproduce: `src/ambe/efficiency_extraction/datadriven_chi2.py --source 6265 --bkg 6264`. The background
  run needs N ≳ 50 triggers before its own statistics stop dominating.

So Poisson is nominal, and the **signed** gap to the data-driven fit (+0.035) is the δ_m
model systematic — a real model uncertainty, not noise. The headline is printed only under
the written override recorded in the config and in the JSON.

### 1d. All eight convention variants

| denominator | numerator | model | ε_n | stat | λ_n | χ²/ndof | syst | naive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ambe_triggers | noncosmic | Poisson **(NOMINAL)** | **0.605** | +0.005/−0.004 | 0.076 | 51.4/6 | +0.035/−0.000 | 0.635 |
| ambe_triggers | noncosmic | data-driven | 0.640 | ±0.108 | — | 3.1/7 | +0.035/−0.000 | 0.635 |
| ambe_triggers | production | Poisson | 0.642 | ±0.004 | 0.076 | 51.6/6 | +0.045/−0.000 | 0.669 |
| ambe_triggers | production | data-driven | 0.687 | ±0.108 | — | 3.3/7 | +0.045/−0.000 | 0.669 |
| total_events | noncosmic | Poisson | 0.429 | +0.005/−0.003 | 0.078 | 50.7/6 | +0.011/−0.000 | 0.473 |
| total_events | noncosmic | data-driven | 0.440 | ±0.006 | — | 98.1/7 | +0.011/−0.000 | 0.473 |
| total_events | production | Poisson | 0.457 | ±0.004 | 0.078 | 51.0/6 | +0.007/−0.000 | 0.499 |
| total_events | production | data-driven | 0.464 | ±0.006 | — | 111.0/7 | +0.007/−0.000 | 0.499 |

Cross-check at an independent position:

| run | position | ambe_triggers | naive | ε_n (Poisson) | ε_n (data-driven) | λ_n |
|---|---|---:|---:|---:|---:|---:|
| 6265 | (0, 0, 0) | 15,628 | 0.635 | **0.605 ± 0.005** | 0.640 ± 0.108 | 0.076 |
| 6266 | (0, 100, 0) | 30,681 | 0.578 | **0.540 ± 0.003** | 0.559 ± 0.108 | 0.084 |

λ_n agrees across two independent runs (0.076 vs 0.084) — the background rate is a property
of the detector, not of source position, so this is a genuine check. The positional fall-off
(0.605 at centre → 0.540 at y = +100 cm) matches thesis fig. 8.29.

### 1e. Inputs

| | 6264 (background) | 6265 (source) | 6266 (cross-check) |
|---|---:|---:|---:|
| Stage-1 admitted triggers | 589 | 20,964 | 39,391 |
| cosmic-vetoed | 567 (96.3 %) | 5,336 (25.5 %) | 8,710 (22.1 %) |
| **ambe_triggers** | **22** | **15,628** | **30,681** |
| multiplicity (noncosmic) | [17, 3, 2, 0, …] | [5704, 9192, 639, 82, 9, 1, 0, 1] | — |
| naive efficiency | 0.227 | 0.635 | 0.578 |
| capture-time τ, [2,67] µs (lmfit) | — | 29.7 ± 0.9 µs, χ²/ndof 1.53 | — |

### 1f. Why ε_n is ~60 % — where the number actually comes from

**The strongest justification is arithmetic, not the quality of the fit.** For the Poisson
model the zero bin has a closed form:

    P(M=0) = (1 − ε_n) · e^(−λ_n)      ⟹      ε_n = 1 − (N₀/N) · e^(λ_n)

    N₀/N = 5704 / 15628 = 0.3650
    ε_n  = 1 − 0.3650 × e^0.076 = 1 − 0.3938 = 0.6062        (the fit returns 0.605)

ε_n ≈ 60 % because **36.5 % of AmBe triggers contain no delayed candidate at all**. That is
an observed count, not a modelling choice. The fit only corrects it by e^λ = 1.079 — the
chance a background candidate filled an event that would otherwise have been empty. Nothing
done to the background model can move the answer far, because the zero bin pins it. The
naive ratio (0.635) and the fit (0.605) differ only by that 8 % correction.

So "why 60 %?" is really "why are 36.5 % of triggers empty?"

**Window truncation accounts for ~14 % of it.** Integrating the fitted capture shape
(τ = 29.69, therm = 6.31 µs):

| | fraction of all captures |
|---|---:|
| before 2 µs (removed by the box cut / cosmic veto) | 1.1 % |
| **inside the [2, 67] µs window** | **86.2 %** |
| after 67 µs (past the end of the acquisition) | 12.7 % |

12.7 % of captures simply happen after readout ends, so **ε_n cannot exceed ~0.86 by
construction**. This is a property of the acquisition window, not of the detector.

**The rest, ~30 %, is detector.** Dividing out the window: 0.605 / 0.862 = **0.702**, i.e.
the detector finds ~70 % of the captures that occur inside its window. The remainder is
capture on H rather than Gd (2.2 MeV instead of the ~8 MeV Gd cascade, largely below the
clustering threshold), neutrons escaping the tank, and clusters failing the box
(PE ≤ 100, CB < 0.45, hits ≥ 5).

**External corroboration**

| check | result |
|---|---|
| thesis, same method, position 0 | 0.64 — we get 0.605 |
| positional trend 6265 → 6266 | 0.605 (centre) → 0.540 (y = +100 cm); thesis fig 8.29 shows 0.64 → 0.35 over a comparable range |
| λ_n across independent runs | 0.076 vs 0.084 (thesis table 8.5: 0.067–0.073 over four positions) |
| λ_n gated vs ungated (§6) | 0.076 vs 0.078 |
| data-driven vs Poisson, ungated (§6) | 0.558 vs 0.563 — they agree once 6264 has statistics |
| naive counting bound | 0.635 — the fit must sit below it, and does |

**The honest caveat.** None of this rescues the χ²: the Poisson fit fails its
goodness-of-fit gate (p < 10⁻³), and the ungated fits of §6 fail too. The justification for
0.605 is the zero-bin arithmetic plus these six independent consistency checks, **not** the
quality of the fit. A failing χ² says the background is not Poisson in time; it does not say
ε_n is wrong, because ε_n is anchored by a bin the background model barely touches.

### 1g. This is the multiplicity model, not counting

Two different quantities, and the difference is the point of the method:

- **Naive counting** — "what fraction of `ambe_triggers` had ≥ 1 candidate?" = 0.635. Every
  background candidate landing in an otherwise-empty trigger is counted as a detected
  neutron, so this is biased **high**.
- **Multiplicity fit** — model each acquisition as `M = δ_S + N_B`, `δ_S ~ Bernoulli(ε_n)`,
  and fit the *shape* of the whole histogram `[5704, 9192, 639, 82, 9, 1, 0, 1]`. The M ≥ 2
  bins are the lever: **one AmBe neutron cannot produce two candidates**, so everything above
  M = 1 measures λ_n. Feeding that back removes the bias: 0.635 → 0.605.

The two coincide exactly in the zero-background limit; `tests/test_efficiency_fit.py` asserts it.

---

## 2. Plots (`plots/`, both PDF and PNG)

| file | thesis fig | what it shows |
|---|---|---|
| `eff_multiplicity__*` | 8.18 / 8.19 | the two multiplicity distributions the fit runs on — 6265 source (left), 6264 background (right), log y |
| `eff_bestfit_overlay__*` | 8.20 | **the money plot**: 6265 data points with both best-fit models overlaid — Poisson (solid, ε=0.605, λ=0.076) and data-driven (dashed, ε=0.640) |
| `eff_chi2_1d_datadriven__*` | 8.21 | 1D Δχ² profile of the data-driven model with the Δχ²=1 band shaded — this is where the ±0.108 comes from |
| `eff_chi2_2d__*` | 8.22 | 2D Δχ² map over (ε_n, λ_n) for the Poisson model, cyan = the Δχ²=2.30 68.3 % contour, star = best fit |
| `eff_chi2_projections__*` | 8.23 / 8.24 | the two 1D cuts through that 2D minimum, one per parameter |
| `eff_capture_time_6265__*` | — | delayed-cluster capture time for 6265, τ = 29.7 ± 0.9 µs. This is what proves 6265 is source-like |
| `eff_capture_time_6266__*` | — | same for 6266 |

Regenerate any of them:

```bash
ambe data process     --config configs/data_ambe2v4_effpair_box.yaml --selection box
ambe stats efficiency --config configs/data_ambe2v4_effpair_box.yaml
ambe plots efficiency --config configs/data_ambe2v4_effpair_box.yaml
```

---

## 3. Event selection streamline

Two stages. Stage 1 works on **waveforms** and decides which triggers exist at all; Stage 2
works on **tank clusters** inside those triggers and decides which are neutron candidates.

```
  raw AmBe trigger (one waveform + its tank clusters)
        |
  ======  STAGE 1 : waveform gate -- "was this a real AmBe trigger?"  ======
        |          src/ambe/data/processor.py:400-518
        |
        |  baseline, sigma = norm.fit(whole record)
        |  IC = sum(samples in [pulse_start, pulse_end] - baseline)
        |  IC_adjusted = (ns_per_sample / impedance) * IC
        |
        +--[ CUT 1 ]  700 < IC_adjusted < 1200 .................. else DROP
        |             the AmBe 4.4 MeV gamma tag in the source PMT
        |
        +--[ CUT 2 ]  no sample after pulse_end exceeds
        |             baseline + sigma + 7 ADC  ................. else DROP
        |             "second-pulse veto": kills pile-up / double triggers
        |
        v   admitted -> its eventTimeTank joins good_events
      total_events        6264: 589   6265: 20,964   6266: 39,391
        |
  ======  STAGE 2 : cluster loop, two passes  ======
        |          src/ambe/data/processor.py:1040-1080
        |
        |  PASS 1 -- COSMIC VERDICT, over EVERY cluster in the event:
        |    cosmic  <=>  (cluster_time < 2000 ns)  OR  (cluster_PE > 100)
        |
        +--[ any cluster cosmic ]--> event is COSMIC
        |        cosmic_events++ ; contributes NO candidates
        |        6264: 567 (96.3%)   6265: 5,336 (25.5%)   6266: 8,710 (22.1%)
        |
        v   survivors
      ambe_triggers = total_events - cosmic_events   <-- THE DENOMINATOR
        |   6264: 22        6265: 15,628       6266: 30,681
        |
        |  PASS 2 -- candidates, only inside surviving events:
        +--[ box cut ]  0 < cluster_PE <= 100
        |               0 < charge_balance < 0.45
        |               2,000 <= cluster_time <= 67,000 ns
        |               cluster_hits >= 5
        v
      multiplicity M = number of accepted clusters in that event  <-- THE NUMERATOR
      6265: [5704, 9192, 639, 82, 9, 1, 0, 1] over M = 0..7
        |
  ======  FIT  ======
      M = Bernoulli(eps_n) + background,  background = Poisson(lambda_n) or 6264's histogram
```

Three things worth knowing about this chain:

- The cosmic veto and the delayed window `[2, 67] µs` live on `Selection` (the **base**
  class) and are deliberately not overridable by `BoxSelection` / `MvaSelection`
  (`selection.py:100-116`). Box and MVA must share a denominator and a livetime span or the
  two analyses are not comparable, which is the entire reason both are run.
- The window upper edge is enforced at candidate construction now. The fitter normalises the
  background rate over exactly `[2, 67] µs`, so a candidate accepted outside it would enter
  the numerator with no livetime backing it. Config `cuts.time_max_ns` and
  `fit_params.acquisition_window_ns` are cross-checked at startup and the code exits if they
  disagree.
- An offline IC-window scan can only ever **narrow** the Stage-1 window, never widen it —
  the event index only contains what Stage 1 already admitted.

---

## 4. The cosmic veto, in detail

### What it is

One line, `src/ambe/data/selection.py:100-103`:

```python
def cosmic(self, ct, cpe):
    return (ct < self.cuts.cosmic_ct_threshold or   # 2000 ns
            cpe > self.cuts.cosmic_pe_threshold)    # 100 PE
```

A **cluster** is called cosmic if it is either **early** (< 2 µs after the trigger) or
**bright** (> 100 photoelectrons). If **any** cluster in an event is cosmic, the **whole
event** is thrown away.

### Why those two conditions

A cosmic muon crossing the tank deposits a large, prompt burst of Cherenkov light. Both
thresholds are different views of the same thing:

- **Bright (> 100 PE).** A thermal-neutron capture on Gd releases an ~8 MeV gamma cascade
  that gives a small cluster — tens of PE. A through-going muon gives hundreds to thousands.
  100 PE separates them cleanly.
- **Early (< 2 µs).** A neutron from the AmBe source has to thermalise before it captures.
  That takes ~30 µs (the measured τ is 29.4 µs; 29.7 ± 0.9 µs on 6265 specifically). Nothing
  from the source can produce a capture cluster in the first 2 µs. Light there is prompt —
  muon, or the AmBe gamma itself — not a neutron.

Note that 100 PE and 2 µs are also exactly the box cut's `pe_max` and `ct_min`. That is not a
coincidence: the veto is the *complement* of the neutron box on those two axes. A cluster is
either a plausible delayed neutron, or it is prompt/bright enough to condemn the whole
acquisition.

### Why it vetoes the whole EVENT, not just the cluster

This is the part that usually trips people up. The veto is not "ignore this cluster". It is
"this acquisition was a muon event, throw it out".

The reason: if a muon went through the tank, that muon produces its **own** spallation
neutrons and afterglow. Those show up in the delayed window looking exactly like AmBe
neutron captures. You cannot distinguish them cluster by cluster. So the only safe move is to
discard the entire trigger — its prompt cluster *and* every delayed cluster in it.

This is also why the veto is a property of the **event**: whether a muon went through does
not depend on which cluster your loop happened to look at first.

### Where it enters the efficiency

    ambe_triggers = total_events − cosmic_events        <- denominator (thesis §8.6)

Vetoed events come **out of the denominator**. They are not failures of neutron detection —
they are acquisitions that were never a clean AmBe trigger in the first place, so asking
"did we detect the neutron" is not meaningful for them.

The denominator choice matters more than any uncertainty in the result: using `total_events`
instead drops ε_n from 0.605 to 0.429, ~0.18 — thirty times the statistical error.

The veto rate is also a useful diagnostic in its own right. On 6264 it is **96.3 %**: that
run has almost no AmBe triggers left (22) precisely because it is the no-source run, so
nearly everything that fired was cosmic. On 6265 it is 25.5 %.

### The bug it used to have — the "cosmic leak"

This is the other thing worth understanding, because it is worth 3.4 percentage points.

The **original single-pass loop** did this:

```
for each cluster k in the event:
    if cosmic(k):   count the event as cosmic; break
    if box(k):      bank k as a neutron candidate
```

The `break` only fires when iteration **reaches** the cosmic cluster. So if a box-passing
cluster happened to be stored *before* the cosmic one in the array, it was already banked.
That event then:

- incremented `cosmic_events`  → **subtracted from the denominator**, and
- contributed a candidate       → **counted in the numerator**.

The same event on both sides of the ratio, in opposite directions. And the verdict depended
on **array order**, which is not a physical quantity.

Size of it: on run 6265 that is **532 of 15,628 AmBe triggers**, worth **3.4 pp** — bigger
than the entire systematic budget of thesis table 8.6. On run 6062 it was 410 of 14,638,
worth 2.8 pp. Only multi-cluster events can do it (documented at `icscan.py:368-406`).

**The fix** (`processor.py::select_event_clusters`) is the two-pass loop drawn in §3: decide the cosmic
verdict over *all* clusters first, then collect candidates only from events that survived.
A vetoed event now contributes nothing to the numerator, ever. `tests/test_processor_ordering.py`
asserts that every permutation of a multi-cluster event gives the same verdict and the same
multiplicity.

**The legacy conventions were removed on 2026-09-22.** `numerator_mode` and
`stage1.legacy_ordering` no longer exist, and `build_multiplicity` refuses an EventIndex in
which a cosmic-vetoed event carries candidates. The leak-inclusive numbers quoted above
(0.642, 0.6691 naive) and the published 57.23 % can no longer be regenerated. v4 was
reprocessed with the single path as `AmBe2.0v4_all28_box_veto`.

### One knob you will see and should not touch on data

`AmBeNeutronProcessor(cosmic_veto=False)` disables the veto for a whole dataset. It exists
**only for MC**: in MC there are no cosmics and no AmBe 4.4 MeV gamma (every event has
exactly one primary, PDG 2112), so the sub-2 µs clusters the veto removes there are
neutron-induced prompt light, not muons. It is a dataset-level switch, never a per-selection
one, so box and MVA always keep the same denominator (`processor.py:92-104`).

---

## 5. Cross-study: efficiency with NO waveform gate (tank information only)

**Not in the code base, deliberately.** `run()` never passes `good_events=None` — the
infrastructure blocks a data deck from being *silently* ungated, because it would report an
efficiency against the wrong denominator (`processor.py:925-932`). These numbers were
produced by tallying Stage 2 straight off `ambe_output/stage_v4/effpair/BeamCluster_<run>.root`
using the package's own `BoxSelection(CutCriteria(pe_max=100, ccb_max=0.45, ct_min=2000,
ct_max=67000, chits_min=5))`, so the cuts are identical to the gated run and the only thing
removed is the `eventTimeTank ∈ good_events` filter. `n_bins = 10` is needed: ungated 6266
reaches multiplicity 8.

Recording it here rather than wiring it in, pending a decision on which definition the
efficiency should use.

### Sample

| run | | gated | ungated |
|---|---|---:|---:|
| 6265 | events | 20,964 | **41,586** |
| | cosmic | 25.5 % | 29.7 % |
| | `ambe_triggers` | 15,628 | **29,219** |
| 6264 | events | 589 | **18,804** |
| | cosmic | **96.3 %** | 66.4 % |
| | `ambe_triggers` | **22** | **6,312** |
| 6266 | events | 39,391 | **75,625** |
| | cosmic | 22.1 % | 27.6 % |
| | `ambe_triggers` | 30,681 | **54,731** |

Ungated multiplicity histograms, `ambe_triggers` denominator, noncosmic numerator:

    6264   [6084, 190, 34, 3, 0, 0, 1, 0, 0, 0]
    6265   [11922, 15989, 1157, 136, 13, 1, 0, 1, 0, 0]
    6266   [25026, 27284, 2156, 229, 31, 4, 0, 0, 1, 0]

### Results, `ambe_triggers` denominator

| run | naive | Poisson ε_n | λ_n | χ²/ndof | data-driven ε_n | χ²/ndof |
|---|---:|---:|---:|---:|---:|---:|
| 6265 **gated** | 0.635 | **0.605 ± 0.005** | 0.076 | 51.4/6 | 0.640 **± 0.108** | 3.1/7 |
| 6265 **ungated** | 0.592 | **0.558 ± 0.004** | 0.078 | 73.9/8 | 0.563 **± 0.005** | 200.2/9 |
| 6266 gated | 0.578 | 0.540 ± 0.003 | 0.084 | 75.5/6 | 0.559 ± 0.108 | 2.6/7 |
| 6266 ungated | 0.543 | 0.501 ± 0.003 | 0.086 | 113.0/8 | 0.509 ± 0.004 | 309.5/9 |

With `total_events` instead: 6265 ungated gives 0.365 (Poisson) / 0.389 (data-driven),
6266 gives 0.335 / 0.363.

### What this shows — three findings, two of them more useful than the efficiency

**(a) The waveform gate is worth −0.047 on ε_n** (0.605 → 0.558, ~4.7 pp). Ungated is lower,
as it should be: triggers with no AmBe gamma have no neutron to find and land in the zero bin.

**(b) The data-driven model becomes a real measurement.** This is the significant one.
Gated, 6264 has 22 triggers, the data-driven fit gives ±0.108, and its χ²/ndof = 0.44 is the
artefact documented in §1c. Ungated it has **6,312** triggers: the error collapses **20×** to
±0.005, and the two models then **agree to 0.005** (0.558 vs 0.563) instead of differing by
0.035. The δ_m model systematic — which is the *entire* systematic budget of the gated
result — essentially vanishes.

Its χ²/ndof also goes honest, 200/9 instead of 3.07/7. That is the artefact disappearing,
exactly as `src/ambe/efficiency_extraction/datadriven_chi2.py` predicted would happen past N ≳ 50.

**(c) λ_n is invariant under gating: 0.076 → 0.078 (6265), 0.084 → 0.086 (6266).** The
background candidate rate per trigger is the same whether or not the waveform gate is
applied. This is a strong internal check: it confirms λ_n is a property of the detector, and
that the flat background under the delayed window is genuinely independent of the small PMT
— the same conclusion the sideband study in `WAVEFORM_STUDY.md` reaches from the waveform
side, by a completely different route.

**Both fits remain bad fits** (Poisson χ² 73.9/8 and 113.0/8), so the non-Poisson
high-multiplicity tail is not a gating artefact either.

### Open question this raises

The gated result's dominant systematic exists only because the background run is
statistics-starved, and that starvation is caused by the gate itself (96.3 % of 6264's
Stage-1 survivors are cosmic, leaving 22). Ungated, the same pair gives a δ_m of 0.005
instead of 0.035 — but at a lower central value and against a denominator that includes
triggers with no AmBe gamma. Which definition is the right one for the published efficiency
is **not settled here**.

---

## 6. Caveats carried over

1. The Poisson fit **fails its goodness-of-fit gate** and is quoted under an explicit written
   override recorded in the config and the JSON. Read 0.605 as preliminary.
2. 6264's 22 AmBe triggers make the data-driven fit a systematic cross-check, not a
   competing measurement.
3. **δ_h (housing captures) is NOT EVALUATED** — no MC housing-capture fraction exists for
   this campaign. The quoted systematic is a **lower bound**.
4. **δ_bt is EXCLUDED** — it is a trigger-count ratio, and thesis eq. 8.9 needs rates. No
   per-run livetime exists in this repo (the waveform `timestamp` column is absolute epoch
   with a zero minimum, so its span is not a livetime).
5. **Do not compare to the published 57.23 %** — that is the 28-run campaign number produced
   by the order-dependent loop; 6265 is a special run deliberately excluded from it.
6. Cut thresholds here are ANNIE-2.0 adaptations (CB < 0.45 vs thesis 0.4, plus PE/hits/
   window cuts with no thesis equivalent), not optimised on independent data in any
   documented way, and no per-cut systematic has been assigned. Read ε_n as conditional on
   this cut set rather than as a detector property.
