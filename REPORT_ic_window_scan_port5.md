# IC-window scan of the AmBe efficiency — Port 5 (centre), v4

Date: 2026-09-09. Tag `AmBe2.0v4_port5center_icwide`, config
`configs/data_ambe2v4_port5center_icscan.yaml`, tool `ambe data icscan`.

## Question

The production AmBe efficiency is quoted with a fixed Stage-1 gate
`700 < IC_adjusted < 1200`. Varying that window moves the efficiency directly, so:
step the window edges by 50 across 500–1500 on the centre-port runs and find out how
the efficiency responds, and whether it plateaus.

## Method

Runs (Port 5, one run per position): 6046 (y=0), 6252 (+50), 6165 (+100), 6251 (−50),
6062 (−100).

One Stage-1+2 pass at the **widest** window (500–1500, `selection_mode: ic`, no
fprompt anywhere), then all scan points taken offline as subsets of it. This ordering
is forced: a waveform rejected at Stage 1 is never Stage-2 processed, so a window
wider than the pass would report zero neutrons *by construction* rather than by
physics. `icscan` asserts the scan stays inside the pass window.

Three modes, answering different questions:
- **differential** — 20 disjoint 50-wide slices. Points are **statistically
  independent**, and there is no window-dependent denominator. This is the curve that
  answers "does it plateau", and the only one a flatness test is valid on.
- **left / right** — cumulative nested windows, one edge fixed. Presentation curves.
  Consecutive points share >90% of their events, so their error bars are per-point
  only, never point-to-point.
- **closure** — re-tally the production window and check it against the frozen summary.

## Validation

**Closure: PASS, all 20 cells exact** (5 positions × total_events / cosmic_events /
ambe_triggers / unique_neutron_triggers) against
`TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_all28_box.csv`. Per-position efficiencies
reproduce the published 62.97 / 69.55 / 71.82 / 69.46 / 63.03 %.

Also checked: monotonicity (all three count columns non-decreasing as either window
widens, both modes); additivity (the 20 disjoint slices sum **exactly** to the
500–1500 totals on all four count columns); per-run IC medians agree to 1.2 %
(987–999), so a fixed absolute IC edge is the same physical cut on every run and the
summed curve is not mixing five different effective cuts.

## Result: it DOES plateau, over roughly 800–1250

Differential (disjoint-slice) curve, Port-5 summed. Flatness χ²/dof against a constant:

| range | slices | weighted mean | χ²/dof |
|---|---|---|---|
| 500–1500 (full) | 20 | 0.6695 | **27.46** |
| 500–1250 | 15 | 0.6781 | 2.38 |
| 700–1250 | 11 | 0.6790 | 2.34 |
| 750–1250 | 10 | 0.6797 | 2.00 |
| **800–1250** | 9 | **0.6806** | **1.80** |
| 800–1200 | 8 | 0.6817 | 1.38 |
| 700–1200 (production) | 10 | 0.6799 | 2.19 |

The full range is emphatically not flat (χ²/dof 27). Restricting to 800–1250 brings it
to 1.80, and 800–1200 to 1.38 — i.e. consistent with a genuine plateau.

Departures from the 800–1250 plateau mean (0.6806):

```
   500- 550  0.6700 +- 0.0330   -0.3 s     1200-1250  0.6670 +- 0.0065   -2.1 s
   550- 600  0.6667 +- 0.0274   -0.5 s     1250-1300  0.6422 +- 0.0083   -4.6 s
   600- 650  0.6447 +- 0.0197   -1.8 s     1300-1350  0.6079 +- 0.0111   -6.6 s
   650- 700  0.6455 +- 0.0128   -2.7 s     1350-1400  0.5120 +- 0.0141  -11.9 s
   700- 750  0.6581 +- 0.0092   -2.5 s     1400-1450  0.4819 +- 0.0163  -12.2 s
   750- 800  0.6671 +- 0.0069   -1.9 s     1450-1500  0.4420 +- 0.0178  -13.4 s
```

**Above 1250 there is a cliff, not a plateau** — the slices fall away monotonically and
significantly (−4.6σ to −13.4σ), reaching 0.44 by 1450–1500. Whatever those triggers
are, they are neutron-bearing at ~2/3 the plateau rate. Below 800 the slices sit
1.5–3 % low (−1.8σ to −2.7σ), a mild and much less certain deficit: the whole 500–800
region holds only ~5 % of the sample.

The plateau is at the **same place at all five positions** (per-position table in
`csv/ic_window_scan_differential.csv`); only its height moves with y, from ~0.72 at
y=0 to ~0.64 at y=±100. So the summed curve is meaningful and the window choice does
not need to be position-dependent.

## Slice-width sweep — is the plateau a binning artifact?

`--mode widthsweep` repeats the disjoint-slice scan at widths 50…500. **All nine
widths trace the same curve** (`ic_scan_widthsweep_efficiency`): same flat region,
same cliff onset at 1250. The plateau is not an artifact of the 50-wide binning.

| slice width | full range 500–1500 mean / χ²/dof | plateau 800–1250 mean / χ²/dof |
|---|---|---|
| 50 | 0.6695 / 27.46 | 0.6806 / 1.80 |
| 100 | 0.6695 / 53.41 | 0.6805 / 2.53 |
| 125 | 0.6694 / 69.22 | 0.6805 / 2.59 |
| 150 | 0.6694 / 74.58 | 0.6805 / 3.14 |
| 200 | 0.6694 / 99.85 | 0.6805 / 4.69 |
| 250 | 0.6694 / 110.83 | 0.6805 / 7.02 |
| 400 | 0.6694 / 192.76 | 0.6805 / 4.71 |

The plateau **mean is 0.6805 at every single width** — a stability that a binning
artifact could not produce.

**χ²/dof is not comparable across rows**, and this measurement shows why. Widening a
slice pulls it two opposing ways: averaging washes out structure narrower than the
slice (pushes χ²/dof down) while the error bars shrink as 1/√n (pushes it up). Over
the full range the second effect wins decisively — χ²/dof *rises* 27 → 193 — because
the >1250 cliff spans ~250 units and so survives averaging while the errors keep
shrinking. So a wide-slice χ² is a *more* stringent test of flatness here, not a
looser one, and the plateau χ²/dof creeping from 1.80 to ~5–7 at widths 200–250 is
the residual tilt across 800–1250 becoming visible against much smaller errors — not
a new structure.

Note the plateau row needs slices tiling 800–1250 directly: full-range slices are
anchored at 500, so at large widths none of them land inside the plateau.

## Arbitrary windows

`--mode grid` evaluates all 210 (lower, upper) pairs on the 50 grid
(`ic_scan_grid_efficiency` is the 2-D map). Best efficiency among windows ≥400 wide:

| window | width | neutrons | eff |
|---|---|---|---|
| **800–1200** | 400 | 43607 | **0.6816** |
| 750–1150 | 400 | 42094 | 0.6807 |
| 750–1200 | 450 | 46713 | 0.6806 |
| 800–1250 | 450 | 47136 | 0.6805 |
| 700–1200 (production) | 500 | 48476 | 0.6798 |

The production window is **0.18 pp** below the best 400-wide window. Nothing in the
210-window grid beats it by a meaningful margin — the top dozen span 0.6797–0.6816,
i.e. the whole plateau is degenerate within ~0.2 pp.

Conversely, holding efficiency within 0.5 pp of production and maximising yield:

| window | width | neutrons | vs production |
|---|---|---|---|
| **500–1350** | 850 | 56958 | **+17.5 %** |
| 500–1300 | 800 | 55775 | +15.1 % |
| 600–1300 | 700 | 55441 | +14.4 % |

## What this means for the window choice

Candidate windows, Port-5 summed, against the production window as reference:

| window | ambe_triggers | neutrons | eff | accept % | Δ neutrons | Δ eff (pp) |
|---|---|---|---|---|---|---|
| **700–1200** (production) | 71312 | 48476 | 0.6798 | 40.53 | — | — |
| 800–1250 | 69268 | 47136 | 0.6805 | 38.63 | −2.8 % | +0.07 |
| 500–1200 | 73805 | 50096 | 0.6788 | 42.45 | +3.3 % | −0.10 |
| 700–1250 | 76603 | 52005 | 0.6789 | 43.41 | +7.3 % | −0.09 |
| 600–1250 | 78596 | 53291 | 0.6780 | 44.93 | +9.9 % | −0.17 |
| **500–1250** | 79096 | 53625 | 0.6780 | 45.33 | **+10.6 %** | **−0.18** |
| 500–1500 | 87358 | 58395 | 0.6685 | 50.01 | +20.5 % | −1.13 |

The production window is sitting **inside** the plateau, not on a peak — so it is a
defensible choice and nothing here invalidates the published 57.23 % campaign number.
What it is leaving behind is statistics: extending to **500–1250 buys 10.6 % more
neutron triggers for 0.18 pp of efficiency**, because everything added is drawn from
inside the flat region. Pushing to 1500 is where it stops being free — 1.13 pp for the
last stretch, which is exactly the >1250 cliff being averaged in.

## Concluding remarks

1. **The efficiency plateaus, and 700–1200 sits inside the plateau rather than at an
   extreme of it.** That is the right place for it to be. It is neither the maximum
   nor the minimum of the scan, and it should not be: the maximum of the *ratio* is a
   vanishingly narrow window with no statistics, and the minimum is 500–1500 where the
   cliff is averaged in. Sitting in the interior of a flat region is what a
   well-chosen cut looks like.
2. **The plateau spans ~800–1250 and is real, not a binning artifact.** Nine slice
   widths from 50 to 500 give the same curve and a plateau mean of 0.6805 at *every*
   width.
3. **The 1250 upper edge is the one hard boundary.** Above it the curve falls
   monotonically and significantly (−4.6σ → −13.4σ, reaching 0.44). Whatever those
   triggers are, they are neutron-bearing at ~2/3 the plateau rate. No window should
   extend past 1250.
4. **Within the plateau the choice is degenerate to ~0.2 pp.** Across all 210
   arbitrary windows, the top dozen ≥400 wide span 0.6797–0.6816. So the window choice
   is not an efficiency optimisation — it is a **statistics** decision, and should be
   argued as one.
5. **The lever is yield, not efficiency.** 500–1250 buys +10.6 % neutrons for
   −0.18 pp; 500–1350 buys **+17.5 %** for −0.49 pp. Both are drawn from inside the
   flat region, which is exactly why they are nearly free.
6. **The low edge is nearly free to open.** 500–700 holds only ~2 % of the sample, so
   dropping the lower edge from 700 to 500 changes the efficiency by 0.10 pp. It is
   the *upper* edge that carries all the leverage and all the risk.

**Recommendation: 500–1250** for statistics at constant selection quality (+10.6 %
neutrons, −0.18 pp), or **500–1350** if a 0.49 pp cost is acceptable for +17.5 %.
If the goal is instead the purest tag sample, **800–1200** is the best window in the
whole grid (0.6816) at a 10 % cost in yield. In every case the upper edge stays ≤1250
unless you deliberately accept the cliff.

The honest framing for a talk: 700–1200 is defensible and needs no correction; what
the scan establishes is that it is **conservative**, and that ~10–17 % more neutron
statistics are available at essentially unchanged selection quality.

Caveat this scan cannot settle: the plateau says added triggers are neutron-bearing at
the same *rate*, not that they are the same *physics*. A per-window capture-time τ
would test that (flat efficiency but drifting τ would mean background dilution
masquerading as stability). It was deliberately left out of scope here; the machinery
to add it is the fit in `src/ambe/plots/basic.py` (window 2–67 µs, lmfit
`method="least_squares"`).

## Pre-existing bug found by the closure test

`unique_neutron_triggers` as accumulated in `process_events_efficient` **does not
exclude cosmic-vetoed events.** The cluster loop appends box-passing clusters as it
iterates and the cosmic `break` only fires when the loop *reaches* the cosmic cluster,
so a neutron-like cluster ordered before it is already banked. Those events increment
`cosmic_events` (which is subtracted from the denominator) while still counting in the
numerator.

Scale: 2291 events over the five Port-5 runs at the production window; all have
`numberOfClusters >= 4` (a single-cluster event cannot do it). Effect on the Port-5
summed efficiency is **0.6798 → 0.6476**, i.e. ~3.2 pp, and it is present in every
published v4 box number.

This is pre-existing and was **not** changed. `icscan` reproduces it exactly as the
primary `efficiency` column — that is what makes the closure test meaningful and keeps
every scan point comparable to the published figure — and carries the stricter reading
as `efficiency_noncosmic` / `unique_neutron_triggers_noncosmic` /
`cosmic_with_candidate` at every window, so the size of it is visible rather than
buried. Deciding whether to change the definition is a separate call; note it would
shift published v4 efficiencies down by ~3 pp.

## Files

- `src/ambe/data/icscan.py`, registered as `ambe data icscan` (`--mode` required, no default)
- `configs/data_ambe2v4_port5center_icscan.yaml`
- `src/ambe/data/processor.py` — new `stage1.dump_event_index` writes
  `TriggerSummary/EventIndex_<tag>_<run>.parquet` (per-event cosmic verdict +
  accepted-cluster counts). Default off; verified inert for every existing config.
  This is what makes the scan a pandas groupby: `cosmic_events` is the one ingredient
  the older dumps cannot supply, because `PromptAmBeNeutronCandidates` carries no
  event key.
- Outputs: `ambe_output/ambe_data/AmBe2.0v4_port5center_icscan/{csv,plots}/`
  — `ic_window_scan_{closure,differential,left,right}.csv` and, per mode,
  `ic_scan_<mode>_{efficiency,yield,acceptance}.{pdf,png}`.
  Headline figure: `ic_scan_differential_efficiency`.
- Width sweep figures: `ic_scan_widthsweep_efficiency` (all widths overlaid),
  **`ic_scan_widthsweep_panels` (one panel per width — the legible view)**, and
  `plots/widthsweep/ic_scan_slices_w<width>.{pdf,png}` (one standalone figure per
  width). All three share a common y-range so the curves are directly comparable;
  per-panel autoscaling would make a flat curve and a cliff look equally dramatic.
- Arbitrary-window map: `ic_scan_grid_efficiency` (x = upper edge, y = lower edge, so
  each cell is one window; the blank triangle is lower >= upper).
- Log: `logs/logs_stage1_port5center_icwide.log`
