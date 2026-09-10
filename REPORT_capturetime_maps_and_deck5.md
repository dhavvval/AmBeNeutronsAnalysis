# Capture-time / thermalisation-time maps, and deck 5 for the special runs

> **⚠ THE FIT WINDOW CHANGED AFTER THIS REPORT WAS FIRST WRITTEN.** Every capture-time
> number below is now on **2–67 µs**; it was 10–67 µs. See §7 for what moved and why.
> The frozen anchor is **29.417 ± 0.221 µs** (19 positions) / **29.477 ± 0.193 µs**
> (26 positions). Anything reading 30.53 / 30.535 / 31.509 is a 10–67 number and has
> been superseded.

Four questions were asked. Answers first, then what was built.

---

## 1. Why was there no port-position thermal / capture-time fit?

Partly there was, and partly the capability existed and was orphaned. Three separate
things:

**(a) Per-position τ did exist, but only as a scatter.** Deck 2 slot **G4**
(`06_capture_time_by_position.pdf`) fits τ at each of the 26 positions and quotes the
weighted **29.477 ± 0.193 µs**; deck 4 **L3** does the same for both selections. Neither
was ever drawn on the **port × y grid** — they plot against position *index*, so a τ map
could not be laid beside the efficiency map for the same cells. That is now J9 / K9.

**(b) `therm` was never plotted anywhere, for any deck.** Not a missing fit — a
discarded output. The fit is
`NeutCapture(t) = A·(1 − exp(−t/therm))·exp(−t/tau) + B`, so **the same fit that gives
τ gives therm**, and `therm` / `therm_err` have been sitting in
`TriggerSummary/CaptureTimeFits_v4_all28.csv` and in
`boxcut_v4_capturetime_by_position.csv` since the campaign was fitted. `do_capture()` in
`boxcut_v4_campaign.py` reads that CSV and consumes the `tau` columns only. Nothing else
read the `therm` columns. That is now J10 / K10.

**(c) The code to draw exactly these two maps already existed and was unreachable.**
`src/ambe/plots/basic.py::generate_summary_plots()` builds port × y heatmaps of
thermalisation time *and* capture time from `self.lmfit_summary`, and its docstring says
so. It requires `lmfit_analysis()` to have run first. But the CLI entry point,
`basic.run()`, calls

```python
analyzer.run_analysis(file_pattern=patterns,
                      tasks=["2d_histograms", "1d_histograms"])
```

— no `lmfit_fit`, no `summary`. So neither `lmfit_analysis()` (the per-position fit
pages) nor `generate_summary_plots()` (the two maps) was reachable from `ambe plots
basic`, and the deck build goes through `boxcut_v4_campaign.py` and `ambe plots heatmap`,
neither of which touches them. The capability was written, then stranded behind a task
list that does not include it.

**Now fixed, and split deliberately.** `lmfit_fit` is added to that task list, so the
per-position fit pages — each one showing **therm AND tau** with their errors, χ² and
χ²/ndof — land in `APPENDIX_perposition_boxcuts.pdf` (312 pages, up from 286) and its MVA
twin, which is where they were asked for. `summary` is **not** added: its maps are now
deck figures J9/J10/K9/K10 with a convergence gate and the weighted mean in the title,
and it writes to `OutputPlots/…_<campaign_tag>_LMFIT.png` where `campaign_tag` is
`AmBe2.0v4` for *both* selections — so running the two books in sequence would silently
overwrite one map with the other under a filename naming neither.

**(d) Deck 3 had no per-position τ in its running order at all** — only F3
(`classified_capture_time`, kept vs rejected) in *backup*, and no therm. Now K9/K10.

## 2. lmfit as the fit backend

`fit_capture_time_fprompt_compare.fit()` now takes a **required** `backend` argument,
`"scipy"` or `"lmfit"`, and the CLI takes a **required** `--fit-backend`. No default on
either: a silent default would let one recipe's number be attributed to the other. All
existing call sites (`boxcut_v4_campaign.py` ×5, `boxcut_v4_appendix.py`,
`ccinc_v3_ambe_closure.py`) are pinned to `"scipy"`, so the backend change on its own
moved nothing. (The *window* change of §7 did move everything, separately and
deliberately.)

**One deviation from `lmfit_analysis`, and it matters.** The lmfit backend uses
`method="least_squares"`, not lmfit's default `"leastsq"`. Default leastsq is MINPACK
Levenberg–Marquardt, which imposes `min`/`max` by transforming the parameters
internally, and on the high-statistics positions that transform walks into a worse local
minimum and stops. Measured at position (0,−100,−75):

| minimiser | therm [µs] | tau [µs] | χ²/ndof | errors? |
|---|---|---|---|---|
| lmfit `leastsq` (default) | **0.33** (pinned at the 0.1 bound) | 33.21 | — | **none returned** |
| lmfit `least_squares` | 4.90 ± 0.58 | 31.65 ± 0.64 | 2.17 | yes |
| scipy `curve_fit` (bounded TRF) | 4.90 ± 0.58 | 31.65 ± 0.64 | 2.17 | yes |

2 of 26 positions failed outright under default leastsq. With `least_squares` — which
*is* `scipy.optimize.least_squares`, the solver `curve_fit` uses — the two backends agree
to every printed digit, position by position. The choice of library must not be a physics
choice.

## 3. The two new maps

`python -u boxcut_v4_campaign.py --do captureheat` → four figures + one CSV.

| | box cuts | MVA neutron |
|---|---:|---:|
| positions converged | 26 / 26 | 26 / 26 |
| weighted **τ** [µs] | **29.476 ± 0.193** | **29.331 ± 0.228** |
| τ range | 27.62 – 33.42 | 25.93 – 35.74 |
| weighted **therm** [µs] | **6.516 ± 0.107** | 6.868 ± 0.133 |
| therm range | 5.20 – 8.31 | 4.61 – 9.93 |

The τ column reproduces the campaign anchors independently, on a different minimiser:
G4's 29.477 ± 0.193 and L3's 29.329 ± 0.228. Same geometry, port order, orientation and
`value±err` cell format as the efficiency heatmap, so the maps can be laid side by side.

**On the 2–67 µs window therm is a measurement, not just a flatness check.** That was the
main reason for the window change. On 10–67 the `(1 − exp(−t/therm))` rise is already
≥86 % saturated for any therm inside its (0.1, 10) box, so therm was weakly identified,
traded off against τ, and pinned near a bound at 2 of 26 positions. On 2–67 the fit sees
the rise: all 26 positions converge, the weighted error drops from ±0.22 to ±0.11 µs, and
the spread tightens from 3.61–7.97 to 5.20–8.31. The physics claim — that thermalisation
is a property of the water and should be flat across the grid — is now testable rather
than assumed.

## 4. What 100 % means on the normalised statistics heatmap

It was **the busiest cell of that sample**, and nothing said so on the figure.

`_load_and_prepare()` in `src/ambe/plots/heatmap.py` does
`unique_neutron_triggers / max(unique_neutron_triggers) × 100`. Both decks' maps top out
at 100 at **Port 1, y = −100 cm** — but at a different absolute number:

| deck | tag | 100 % = |
|---|---|---:|
| **deck 2** | `AmBe2.0v4_all28_box` | **27,948** neutron triggers |
| **deck 3** | `AmBe2.0v4_all28_mva` | **21,616** neutron triggers |
| deck 4 (covered pair) | `..._box_covered` / `..._mva_covered` | 26,637 / 21,604 |

So a deck-2 cell reading 30 % and a deck-3 cell reading 30 % are **not the same
exposure**, and the two statistics maps are not comparable cell for cell even though both
are normalised to 100. The efficiency maps are comparable (they share a denominator); the
statistics maps are not.

Fixed rather than only documented: the absolute reference is now printed into the title
(`— 100 % = 27,948 neutron triggers at Port 1, y = −100 cm`) and the colourbar is labelled
*Percentage of the busiest cell (%)*. Both maps regenerated.

---

## 5. Deck 5 — the special runs as their own analysis

`PRESENTATION_PLOTS/deck5_special_runs/`, 20 figures in the running order plus 6 in
backup and 2 appendix books, built by `python -u boxcut_v4_special.py --do diag` then
`--do all`.

**Legends name what a curve IS, not its run number** — "dummy trigger (LAPPD on)", "dummy
trigger (LAPPD off)", "no source", "regular run (Port 5, y = 100 cm)". **Three different
kinds of run**: 6254/6256 are dummy (pulser) triggers (`kind=pulser` in the diagnostics),
so the readout fires on a schedule and each event is a random slice of the tank —
they measure the **accidental cluster rate**. 6264 is a genuine **no-source run** with the
real trigger logic live, so its in-gate waveforms are **false starts of the IC trigger**. The run number stays on
every panel title and in every table. **Every 1D figure is emitted twice**, area-normalised
and in raw counts, on separate pages: a normalised plot cannot show that the LAPPD-on run
contributes 4,404 clusters against the regular run's 146,470, and a counts plot buries the
two dummy triggers. **Every after-the-box panel is re-ranged** to the range the cut admits
(PE 0–100, CB 0–0.45, hits 0–60, multiplicity and per-trigger charge from the data), with
the kept fraction moved into the panel title.

The problem with the old presentation: for the runs that matter most the campaign
selection returns a single number — 6254 and 6256 give **zero** IC-passing triggers, 6264
gives **zero** Stage-2 candidates. Three bars at zero is a true summary and a useless one.
So deck 5 runs the **same flow** (IC waveform stage → tank cluster stage → Stage-2 box)
and reports the **distributions at each stage**, **ungated as well as gated**.

Ungated is not a relaxation. For 6254/6256 it is the only option that exists: **0 of
7,933** and **0 of 25,525** waveforms land in the 700–1200 window, so every gated figure
of those runs is blank. 6266 is on every axis as the source-in contrast — without it there
is nothing to call the others anomalous *with respect to*. No efficiency, no τ.

### Stage 1 — the IC waveform stage (M1–M8, including one basic IC page per run)

| run | kind | waveforms | in 700–1200 | + no 2nd pulse | accepted |
|---|---|---:|---:|---:|---:|
| 6254 dummy trigger, LAPPD ON | pulser | 7,933 | **0** | 0 | **0** |
| 6256 dummy trigger, LAPPD OFF | pulser | 25,525 | **0** | 0 | **0** |
| 6264 **no source** | real trigger logic | 18,497 | **808** | 589 | **589** |
| 6266 regular run, source in | source in | 75,279 | 53,933 | 39,391 | 39,391 |

M2 is the one to show: the dummy triggers never leave a narrow peak at IC_adjusted ≈ 0 — **no BGO
gamma at all**; 6266 shows the AmBe BGO peak at 800–1000; 6264 is a flat cosmic tail out
to 4000 that leaks 808 waveforms into the gate with no source in the tank.

### Stage 2 — the tank, ungated and after the box (M9–M18)

| run | kind | triggers | clusters | clusters/trigger | survive the box |
|---|---|---:|---:|---:|---:|
| 6254 | dummy trigger, LAPPD on | 7,932 | 4,404 | 0.56 | 1,260 (**28.6 %**) |
| 6256 | dummy trigger, LAPPD off | 25,524 | 13,837 | 0.54 | 4,068 (**29.4 %**) |
| 6264 | **no source** | 18,496 | **176,634** | **9.55** | 10,323 (**5.8 %**) |
| 6266 | regular run | 75,278 | 146,470 | 1.95 | 50,577 (**34.5 %**) |

M13 is the requested figure: charge balance vs cluster PE, per run, before the box on top
and after it below, the after row re-ranged so the survivors fill the frame. M14–M16 do
the same for time-vs-PE, hits-vs-PE and time-vs-CB.

Two things fall out of the cut flow (M17) and the leave-one-out (M18):

* **`hits ≥ 5` is a strict no-op** — the survivor count is identical with and without it,
  at every run. ClusterFinder already requires 5 hits. Independently reproduces the
  earlier finding.
* 6264 loses **94.2 %** of its clusters, and it is charge balance and time that do it,
  not PE.
* The two dummy triggers sit at 0.55 clusters/trigger with **84–85 % of triggers empty**,
  and their cluster-time distribution is **flat across 0–70 µs** — no capture shape at
  all, which is exactly what a random slice of the tank should look like.

### M19 — the nothing-to-detect charge question, and it is two numbers, not one

6254 and 6256 are a matched LAPPD ON/OFF pair of **dummy triggers** — no source, and a
readout that fires on a schedule — so any difference between them is instrumental.

| estimator, per trigger | 6254 (ON) | 6256 (OFF) | difference |
|---|---:|---:|---:|
| summed **hit** PE (all tank hits) | 642.40 | 421.75 | **+220.65 ± 13.89 (15.9σ)** |
| tank **hits** | 162.85 | 44.24 | **+118.61 ± 1.24 (95.9σ)** |
| summed **cluster** PE | 279.21 | 285.32 | **−6.11 ± 17.70 (0.35σ)** |

**The LAPPD leak raises the hit stream without producing clusters.** Both estimators
reproduce the diagnostics' own Step G to the digit (`tot_mean` 642.3995 / 421.7504,
`clus_mean` 279.2065 / 285.3153), computed here from the BeamCluster ntuples by a wholly
independent route. This is why mean tank PE cannot stand in for cluster PE anywhere in the
analysis — the two disagree in size *and* in significance. Say which estimator you mean,
every time.

### Deck reorganisation

Deck 2's **H1–H4** and deck 3's backup **F4/F5** moved **into deck 5**. The special runs
are now described in one place instead of appearing as four bars at the end of two other
decks, which was the point of the request. H2/H3/H4 and the IC-vs-fprompt 2D sit in
deck 5's `backup/`, not the running order. `check_assignment()` still enforces that every
slot lands in exactly one deck.

Both inherited appendices were dropped — `APPENDIX_IC_waveforms_all28runs.pdf` is the 28
**source** runs and `APPENDIX_perposition_specialruns_boxcuts.pdf` is per source
**position**, which is close to meaningless for a run with no source. Replaced by
`APPENDIX_specialruns_by_run.pdf` (one page per run: IC spectrum, three tank variables,
PE-vs-CB before and after, multiplicity, and a stats block) and
`APPENDIX_IC_waveforms_specialruns.pdf` (the per-run IC page as a book).

### One denominator mismatch to expect — it is not a contradiction

M1 says 6256 has 25,525 waveforms; H1 says 4,254. The M slots read the **pipeline's own**
`WaveformFeatures_AmBe2.0v4_special_box_*.parquet` (every part file `ambe data process`
saw); H1 reads the special-runs diagnostics join, which covers a narrower part range.
Both report zero IC-passing triggers, which is the claim. Quote M1's denominator for
anything about what the run *contains*. Carried in deck 5's README.

---

## 6. Conclusive remarks — how these runs bear on the regular campaign

Four statements, in decreasing order of how much they should change what anyone does.

### 6.1 There is a large accidental cluster floor, and the IC gate is what removes it

Measured on the **dummy triggers**, which is exactly what they are for: the readout fires
on a schedule, so each event is a random slice of the tank, and whatever the box accepts
there is accidental by construction.

With nothing to detect the box still accepts **0.159 clusters per trigger** — and the two
dummy triggers agree to 0.3 % (0.1589 LAPPD on, 0.1594 LAPPD off), so it is a property of
the detector, not of either run. Against the regular run's 0.672 accepted clusters per
trigger that floor is **24 % of the yield**; extrapolated naively onto the campaign's
380,072 Stage-1 AmBe triggers, **≈ 56,000 accidental clusters of 236,792 candidates,
23.6 %**.

It is not there. Fitting the **gated** 28-run campaign with `A·exp(−t/τ) + B` and letting
the pedestal float returns **B = 0.000 ± 62.9 counts/bin** against a mean bin content of
3,428 — a flat component of **0.0 %, with a 1σ upper bound of 1.8 %**. So the gate
suppresses the accidental floor by **at least a factor ~13**, measured two independent
ways on two independent samples.

That is the strongest single defence of the Stage-1 selection in the analysis, and it is
worth a slide: *the IC gate is not a convenience cut, it is what makes the capture-time
fit meaningful.*

**The caveat, stated because the 23.6 % number invites over-reading:** a dummy trigger and
an IC-gated AmBe trigger sample different time windows at different rates, so the
extrapolation is an order-of-magnitude statement, not a subtraction. The 1.8 % limit on
the gated sample is the measurement.

### 6.2 The accidental floor is flat in time, so it is a τ bias and not a yield error

The dummy triggers' accepted clusters are consistent with uniform over 10–67 µs
(χ²/ndof 1.97 and 2.58, 19 bins). A flat contamination does not subtract cleanly from a
yield; it pulls a single-exponential fit **upward** in τ. Two consequences:

* The per-position τ maps (J9/K9) hold at **30.54 ± 0.23** and **30.97 ± 0.26 µs** with B
  fixed at 0 — legitimate *only* because 6.1 shows there is no pedestal to absorb. If the
  gate is ever loosened, B must be floated and the anchor re-derived.
* The no-source run is the counter-example: it is **not** flat (χ²/ndof 12.2), because it
  carries a cosmic-follower peak near 8 µs on top of the floor. A run can fail this test
  in two different ways and 6264 fails it in both.

### 6.3 Run 6264 measures the IC trigger's FALSE-START rate — a real systematic of the AmBe trigger

This is the finding the deck was worth building for, and it is *only* available from a
no-source run: 6264 has the real trigger logic live and no source, so every waveform that
lands in the IC gate is a trigger the DAQ took as an AmBe start when no neutron was
emitted. A dummy-trigger run cannot measure this at all — it never exercises the trigger
logic, which is why 6254/6256's "0 of 7,933 in the gate" is the expected result and **not**
a limit on false starts.

| | rate of accepted IC triggers | live time |
|---|---:|---:|
| 6264, no source | **0.00833 Hz** (589 in 70,728 s) | 19.6 h |
| 6266, source in | 0.24919 Hz (39,391 in 158,076 s) | 43.9 h |

**False starts are 3.34 % of the AmBe triggers** at 6264's cosmic loading. 6264 is
cosmic-loaded though — 66.6 % cosmic-vetoed against 13.6 % for the source-in specials, a
factor 4.9 — so scaled to a normal run's loading the false-start rate is **≈ 0.7 % of AmBe
triggers**.

**What it does to the measurement, and why it is already bounded.** A false start produces
a trigger with no neutron in it, so its 2–67 µs window contains only accidental clusters —
at the dummy triggers' 0.159/trigger against a true trigger's 0.672. The predicted flat
contamination of the accepted-cluster sample is therefore

* **0.79 %** at 6264's cosmic loading,
* **≈ 0.16 %** at a normal run's loading,

against the campaign's measured **0.0 %, 1σ < 1.8 %**. The prediction sits a factor 2.3
inside the limit at the pessimistic loading and a factor 11 inside it at the realistic one.
So the false-start systematic is **real, identified, and demonstrably sub-percent** — it
belongs in the systematics table with a number rather than as a hand-wave, and it does not
threaten the τ anchor.

**Two things it does mean.** First, 6264 must never be used as a no-neutron background
sample: its tank carries 9.55 clusters per trigger, five times the regular run, with a tail
reaching **112 box-accepted clusters in a single trigger**, and its IC waveforms are
visibly a second population (prompt fraction spread to 1.0, width-over-threshold piled up
near 850 bins, baseline σ to 50 ADC). It is a cosmic sample wearing a no-source label.
Second, because the false-start rate scales with cosmic loading rather than with the
source, it is a *run-condition* systematic: a run taken under unusually high cosmic
loading carries proportionally more of it, and the cosmic-veto fraction already recorded
per position (G6) is the handle for flagging that.


### 6.4 The LAPPD leak does not touch the neutron selection

The leak is real and large in the hit stream — **+220.7 ± 13.9 PE** and **+118.6 ± 1.2
hits** per trigger — and completely absent from reconstructed clusters (**−6.1 ± 17.7 PE**,
0.35σ). The two dummy triggers also accept the same number of box clusters per trigger to
0.3 %. So for the regular campaign: **the LAPPD state does not bias the neutron
efficiency, the capture time, or the yield**, and runs with the LAPPD on and off may be
merged without correction. What it *does* forbid is using mean tank PE as a stand-in for
cluster PE anywhere — those two estimators disagree here in size *and* in significance,
and this is the cleanest available demonstration of it.

### 6.5 What is unchanged

`hits ≥ 5` is confirmed a **strict no-op** on all four runs, independently of the campaign
result: ClusterFinder already requires 5 hits, so the box is really three cuts. And once
the cosmic veto has run, PE > 100 and t < 2 µs cannot occur either — which leaves charge
balance as effectively the whole box. Nothing in these runs changes that, but they are the
cleanest place to see it, because the leave-one-out table (M18) shows it on samples with
wildly different composition and gets the same answer every time.

---

## Files changed

| file | change |
|---|---|
| `fit_capture_time_fprompt_compare.py` | `fit(..., backend)` required; `_fit_lmfit()` added; `--fit-backend` required |
| `boxcut_v4_campaign.py` | `--do captureheat`; `_fit_positions()`, `_heat()`, `do_captureheat()`; existing call sites pinned to `"scipy"` |
| `boxcut_v4_special.py` | deck-5 section: `--do diag` + 11 per-product choices, `load_wf()`, `load_clusters_ungated()`, descriptive run labels, normalised+counts twins, re-ranged after-cut panels, per-run IC pages, per-run appendix book — 20 figures, 2 books, 2 CSVs |
| `boxcut_v4_appendix.py`, `ccinc_v3_ambe_closure.py` | call sites pinned to `"scipy"` |
| `src/ambe/plots/heatmap.py` | statistics heatmap now names its own 100 % reference |
| `src/ambe/plots/basic.py` | `lmfit_fit` added to the CLI task list, so the per-position lmfit fits (therm AND tau) land in `APPENDIX_perposition_{boxcuts,mvaneutron}.pdf`; minimiser `basinhopping` → `least_squares` |
| `FIGURES.md` | J9/J10, K9/K10, new group **M** (M1–M12) |
| `collect_presentation_plots.py` | group regexes widened to `[A-HJ-NZ]`; deck5 added; H1–H4 and F4/F5 reassigned |

New CSVs: `boxcut_v4_captureheat_by_position.csv`,
`boxcut_v4_specialdiag_cutflow.csv`, `boxcut_v4_specialdiag_summary.csv`.

## Reproduce

```bash
source /exp/annie/app/users/dajana/myboy/bin/activate
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
MPLBACKEND=Agg python -u boxcut_v4_campaign.py --do captureheat     # J9/J10, K9/K10
MPLBACKEND=Agg python -u boxcut_v4_special.py  --do diag            # M1-M19 + both books
MPLBACKEND=Agg PYTHONPATH=src python -u -m ambe.cli plots heatmap \
    --config configs/data_ambe2v4_all28_box.yaml                    # J2 with its 100 % named
MPLBACKEND=Agg PYTHONPATH=src python -u -m ambe.cli plots heatmap \
    --config configs/data_ambe2v4_all28_mva.yaml                    # K2
MPLBACKEND=Agg PYTHONPATH=src python -u -m ambe.cli plots basic \
    --config configs/data_ambe2v4_all28_box.yaml                    # per-position lmfit book
MPLBACKEND=Agg PYTHONPATH=src python -u -m ambe.cli plots basic \
    --config configs/data_ambe2v4_all28_mva.yaml                    # its MVA twin
python -u collect_presentation_plots.py --do build                  # all five decks
```

---

## 7. The fit window moved to 2–67 µs, campaign-wide

Asked for explicitly and applied uniformly: **every NeutCapture fit in the analysis now
runs on 2–67 µs.** It was 10–67. The two are not comparable and nothing tries to
reconcile them.

**Why 2 µs is the right lower edge.** It is where the data starts. The box cut is
`t ≥ 2 µs`, and the cosmic veto drops any event containing a cluster below 2 µs, so no
selection in this analysis admits anything earlier. The old window discarded the eight
microseconds that contain the thermalisation rise — the only part of the range where the
`(1 − exp(−t/therm))` term carries information.

### What moved

| quantity | 10–67 µs | **2–67 µs** |
|---|---:|---:|
| anchor, 19 positions (`CaptureTimeFits_baseline`) | 30.53 ± 0.26 | **29.417 ± 0.221** |
| campaign τ, 26 positions (G4 / J9) | 30.535 ± 0.228 | **29.477 ± 0.193** |
| campaign therm (J10) | 5.50 ± 0.22 | **6.52 ± 0.11** |
| therm spread, box | 3.61 – 7.97, 2 positions pinned | **5.20 – 8.31, all 26 converged** |
| MVA τ, 26 positions (K9 / L3) | 30.97 ± 0.26 | **29.33 ± 0.23** |
| box τ on the matched set (G13 / L3) | 31.509 ± 0.250 | **30.444 ± 0.210** |
| matched MVA τ (G13) | 30.521 ± 0.263 | **28.898 ± 0.229** |
| matched shift, box → MVA (G13) | −0.989 ± 0.363, 2.73σ | **−1.546 ± 0.311, 4.98σ** |
| MVA-discarded τ (G13) | 33.890 ± 0.661 | **34.746 ± 0.533** |
| like-for-like agreement, B floating (G8) | +0.334 ± 1.835, 0.18σ | **−0.188 ± 1.754, 0.11σ** |
| per-position residual (L3) | −0.543 ± 0.363, 1.50σ | **−1.115 ± 0.310, 3.59σ** |
| median χ²/ndof, box | 1.37 | 1.51 |

### Three things this changes in the argument, not just in the numbers

**1. `therm` became a measurement.** The weighted error halves (±0.22 → ±0.11 µs), the two
positions that used to pin at a bound now converge, and the spread tightens. The old
caveat — "read the flatness, not the cell" — was a consequence of the window, not of the
physics. It no longer applies with the same force.

**2. τ moved *into* the band this analysis quotes for capture on hydrogen in water.** The
deck's own justification for the 80 % working point rejects τ values of 34.6–35.9 µs as
"outside the 25–30 µs band". The old campaign anchor, 30.53, sat just outside that band.
The new one, **29.48**, sits inside it. That is a consistency improvement, not a
degradation, and it is the strongest single argument that 2–67 is the better window.

**3. The MVA result got sharper, and its direction is unchanged.** On the matched set the
MVA moves τ from 30.444 to 28.898 — **4.98σ** against the old 2.73σ — and what it discards
sits at 34.746 ± 0.533, further from the anchor than before. The per-position residual
(L3) goes from 1.50σ to **3.59σ**. The MVA-selected τ now lands essentially on the 29.42
anchor while the box sits 3.3σ above it, which is a cleaner statement of the same physics
than the old numbers made.

The median χ²/ndof rises 1.37 → 1.51 because the fit now has to describe the rise as well
as the tail, on eight more bins. That is the honest cost, and it is small.

### The one deliberate exception

`fit_expflat` — the flat-pedestal diagnostic — **keeps the 10–67 µs window**, via its own
`EXPFLAT_MIN` constant. It is a different model, `A·exp(−t/τ) + B` with no rise term, and
dropping the rise term is only legitimate above ~10 µs where the rise is saturated.
Fitting it from 2 µs asks a pure exponential to describe a rise it has no parameter for:
measured on the 28-run sample that gives χ²/ndof **165** against 1.4 and τ = 46 µs against
29.5. That is not a weak fit, it is the wrong model on that range, and it would report a
meaningless pedestal limit. **The §6.1 flat-pedestal limit (0.0 %, 1σ < 1.8 %) is
therefore a 10–67 µs number** and is quoted as such.

### Guards

The asserts that used to pin the old anchor now pin the new one, so the recipe still
cannot drift silently:

* `boxcut_v4_campaign.TAU_ANCHOR` = 29.417 ± 0.221, asserted in `do_capture()`.
* `boxcut_v4_appendix.run_byposition()` asserts the 26 appendix pages reproduce
  29.477 ± 0.193 — verified, they do.
* `ccinc_v3_ambe_closure.TAU_ANCHOR` = 29.417 ± 0.221.

### Reproduce

```bash
python -u fit_capture_time_fprompt_compare.py --mode tagset \
    --tags AmBe2.0v4_gated,AmBe2.0v4_ext --out-label v4_all28 --fit-backend scipy
python -u fit_capture_time_fprompt_compare.py --mode fpcompare --fit-backend scipy
python -u boxcut_v4_campaign.py --do capture       # G4/G5
python -u boxcut_v4_campaign.py --do captureheat   # J9/J10/K9/K10
python -u ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4neutron
python -u ccinc_v3_ambe_closure.py --do closure  --dataset ambe6266
python -u boxcut_v4_campaign.py --do validation    # G8/G9
python -u boxcut_v4_campaign.py --do matched       # G13
python -u boxcut_v4_campaign.py --do agreement     # G14/G15/G16
python -u boxcut_v4_campaign.py --do residuals     # L1-L6
python -u boxcut_v4_appendix.py  --do byposition
PYTHONPATH=src python -u -m ambe.cli plots basic --config configs/data_ambe2v4_all28_box.yaml
PYTHONPATH=src python -u -m ambe.cli plots basic --config configs/data_ambe2v4_all28_mva.yaml
python -u collect_presentation_plots.py --do build
```
