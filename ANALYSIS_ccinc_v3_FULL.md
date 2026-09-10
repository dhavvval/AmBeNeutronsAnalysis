# CCinc v3 — Full Analysis: Neutron Selection from CC-Inclusive MC, Background Anatomy, and AmBe Validation

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



**Produced 2026-08-24.** One pass end to end, for presenting. Every number here is
either recomputed read-only from the stored artifacts or cited to the report that owns
it. The four backing reports are **frozen** and are not edited by this one:

| report | what it owns |
|---|---|
| `REPORT_ccinc_v3_streamlines.md` | tank-only two-streamline study, Stages 0→2 |
| `REPORT_ccinc_v3_world_merged.md` | world campaign, labelling rule (§0), lineage method (§0.0), merged AUCs (§5), dirt-neutron decomposition (§4.4/§4.5) |
| `REPORT_ccinc_v3_classifier_selection.md` | classifier verdict, differential lineage, streamline discrepancy, June AmBe closure |
| `WORLD_v3_REPORT.md` | first look at the world sample — **read its SCOPE CORRECTION block** |

New in this document: **§5 background anatomy** (is a background cluster one particle?
which particle is discriminable?), **§6 AmBe run 6266** (the first AmBe2.0v4 validation),
and **§6.6 the new neutron definition on the AmBe pipeline's own Stage-1+2 neutrons**
across the 20-run `AmBe2.0v4_gated` campaign.

---

## 0. The result in one page

**The selection.** Train a cluster-level discriminator to separate neutron-capture
clusters produced by an in-tank CC-inclusive neutrino interaction from everything else,
on merged tank + world-volume `productionv3` MC. **Use GBT on ClusterFinder clusters.**
For real data the deliverable is **truth-tag / ClusterFinder GBT**: AUC 0.672, log-loss
0.646, purity 0.590 at 80% signal efficiency on the 1:1 test set.

**What limits it, in one sentence.** 59.5% of the background light is genuine neutron
capture from an out-of-tank interaction, and no property of the light distinguishes it
from an in-tank capture — so the ceiling is physics, not features.

**What the background actually is** (§5, new):

- Background clusters are **not** one particle each. Only **40%** of the background
  clusters that carry any traced non-EM light are single-particle; the median leading
  particle holds **0.75** of that light.
- The mixing is entirely on the **tank** side: tank background is median purity **0.63**,
  **17.6%** single-particle, **2.19** particles per cluster, and its dominant mixture is
  μ⁻ + π⁺ in near-equal shares (72.4% of mixed clusters, mean shares 0.43 / 0.45).
- The **world** background is single-source: 96.7% of dirt-neutron clusters carry **no**
  non-EM background light at all, and 86.2% of the world non-neutron clusters are
  single-particle.
- Per-particle GBT AUC against signal: **out-of-tank captures 0.705** (easiest),
  μ⁺ 0.645, μ⁻ 0.615, π⁻ 0.599, p 0.579, π⁺ 0.561, **π⁰ 0.552** (hardest). The
  aggregate 0.672 is an average over populations that differ by 0.15 in AUC.

**On the AmBe pipeline's own neutrons** (§6.6, new): run the new definition over **all
28 AmBe2.0v4 source runs — 26 positions, 225,945 Stage-2 neutron candidates** — and at
the MC 80% working point it calls **79.0% neutron and 21.0% other**. The two definitions
largely agree: it keeps 79.0% of what Stage 2 accepts but only **19.6%** of what Stage 2
rejects, a 4.0× contrast (11× at eff50). What it discards is measurably background —
mean capture time **+1.10 ± 0.09 µs later (+12.5σ)** and late/early ratio **×1.15** —
i.e. enriched in a flat, non-exponential component. And the kept fraction runs
**64.2–86.0% across positions** for a threshold nominally set to 80%, driven by light
yield (`n_hits` r = +0.91) and not at all by geometry (`d_wall` r = +0.01).

**The special runs** (§6.7, new) are kept separate and never merged. 6254/6256 (pulser)
have **zero** IC-passing triggers and cannot enter the streamline. **6264 contributes
nothing**: 93.3% of its gated events are cosmic-vetoed, leaving zero clusters. 6265, 6266
and 6270 give 79–86%, at the high end of the neutron campaign's per-position range —
consistent with the diagnostics' finding that they behave source-in despite a no-source
label.

**AmBe run 6266** (§6, new): the v4 extraction reproduces the capture-time physics —
τ = **29.69 ± 3.73 µs** at the GBT 80% working point against the **30.53 ± 0.26 µs**
anchor (pull −0.23), and the MVA pulls the fit from 33.80 ± 3.74 (all clusters,
χ²/ndof 2.49) onto the anchor. But the **score calibration does not transfer**: 6266's
score distribution sits closer to the MC *background* (KS 0.10–0.14) than to the MC
*signal* (0.18–0.23), the reverse of the June v1/v3 sample. §6.4 shows this is a
**source-position** effect, not an extraction bug.

**Two numbers that are easy to misquote:**

1. The merged **+0.082 AUC gain over tank-only is a change of question**, not a better
   answer to the old one. Per-population AUC: dirt neutrons 0.689–0.711 (easiest), tank
   muon/pion background 0.579–0.604 — against the tank-only model's own 0.605
   (world report §4.4). Always quote the merged AUC with this attached.
2. The ξ = 0.02 composition-cut campaign's **0.75 AUC is not comparable** to v3's 0.69:
   different sample, different signal definition, the composition cut already applied
   (it removes the hardest 13.4% of clusters), and a 5.3× smaller test set.

---

## 1. Samples and the labelling rule

| sample | path | files | events |
|---|---|---:|---:|
| tank | `.../genie_wcsim/productionv3/tank/fmvmrd/` | **498** (not 499) | 2,490,000 |
| world | `.../genie_wcsim_world/productionv3/world/fmvmrd/` | 500 | 343,027 |

`ANNIEEvent_cc_neutrino_v3_24.root` is an **empty ROOT file** — debris from an
interrupted production. `io.filter_files_with_tree` skips it loudly. The world sample is
78.4% out-of-tank / 21.6% in-tank (74,022 events).

**The discriminator is where the neutrino interacted, not where light was deposited**
(world report §0):

| population | cuts applied | label |
|---|---|---|
| world, interaction **outside** the tank | **none** | **background**, even a genuine neutron capture |
| world, in tank **and** in FV | full CC selection | neutron-dominated → signal, else background |
| world, in tank but **outside** FV | — | **dropped** from both classes (2,325 clusters) |
| tank (99.9% in-tank) | full CC selection | neutron-dominated → signal, else background |

Not gating the out-of-tank harvest on the CC cuts is deliberate: only 31 of 502
out-of-tank events in file 0 pass the truth-tag cuts, so gating would discard ~94% of the
contamination being modelled. The FV cut is a **muon phase-space cut** and plays no part
in the label.

**The trap that makes this non-obvious.** In world samples `trueVtx{X,Y,Z}` is the WCSim
*primary start point* — a sentinel `(0, 14.4602, −168.100)` for ~67% of events, and the
tank-entry point on the r = 152.4 cm wall for the rest. The interaction vertex is
`trueNuIntxVtx_{X,Y,Z}` in GENIE coordinates; tank coords = GENIE + `(0, +14.466,
−168.100)` cm. Validated on all 343,027 events: median deviation 0.001 cm, p99 0.27 cm,
only 110 events (0.218%) beyond 1 cm and those sit **on** the tank wall, where the two
definitions legitimately differ. That residual ~0.2% in/out ambiguity is irreducible from
the stored branches.

Figures: `V3MERGED__cutflow_streamlines_table`, `V3MERGED__cutflow_nminus1_table`,
`V3MERGED__merged_training_sizes`.

---

## 2. Streamlines, and how much they disagree

Two selections are carried throughout: **truth-tag** (`ccinc_truthtag`) and **reco-tag**
(`ccinc_recotag`). MRD + FMV + promptPE roughly halves the selection relative to the old
truth-only stream (800 → 415 per 25k events).

**The two streamlines never disagree about a label** — 100.00% agreement on every shared
cluster, both methods. They disagree only about *which events to include*, and the
reco-tag-only population is **98.8% signal**, made of μ⁺ 51% / π⁻ 35% wrong-lepton events
entering its signal class (classifier-selection report §3; independently reproduces
streamlines §1.2).

**Reco-tag is not data-applicable** — its FV and muon kinematics still read truth
branches. It is carried as an MC upper bound, never as a deliverable.

Figure: `V3MERGED__streamline_overlap_table`.

---

## 3. Merged training and the classifier verdict

Merging the world sample multiplied the background from 8,392 to 36,008 clusters per
class (4.3×; 9.6× for reco-tag, whose tank-only background was only 3,178).

**All 16 merged test AUCs** (world report §5), reproduced from the stored score tables
before anything downstream runs — `python ccinc_v3_stats.py --do guard` asserts them to
1e-3 and exits non-zero otherwise:

| configuration | RF | GBT | XGB | NN | gain vs tank-only |
|---|---:|---:|---:|---:|---:|
| Truth-tag / OPTICS | 0.656 | 0.661 | **0.662** | 0.649 | +0.060 |
| Truth-tag / ClusterFinder | 0.665 | **0.672** | 0.669 | 0.667 | +0.080 |
| Reco-tag / OPTICS | 0.684 | **0.689** | 0.688 | 0.664 | +0.111 |
| Reco-tag / ClusterFinder | 0.688 | **0.694** | 0.690 | 0.687 | +0.079 |

CV spread is 0.003–0.005, so the merged gain is 12–25× the scatter. **GBT or XGBoost wins
every merged configuration** — RF led three of four tank-only, and that ordering flipped.

**AUC alone cannot pick the model**: the GBT/XGB gap is inside the CV scatter in 3 of 4
configurations. Log-loss breaks the tie and agrees with AUC in all four — GBT wins 3 of 4.
**The NN ranks acceptably and calibrates badly** (worst log-loss everywhere, 0.655–0.679
vs GBT's 0.633–0.651) and transfers worst to data. Never pick it on AUC alone.

**Both reco-tag merged trainings used 21 of 31 features**, not 31 — the world rows'
fitted-vertex block is too sparse. The two highest-AUC configurations have the fewest
inputs.

**The merged gain is not geometry.** Separation is led by multiplicity (`n_hits_early`
0.379σ OPTICS, `n_hits` 0.470σ CF, against the tank-only best of 0.147σ), while `d_wall`
ranks 20th and `vtx_y` 29th of 31, and the four geometry features carry 11.8% (RF) /
9.6% (GBT) of the trained importance. This was tested and closed — do not restate it as
an open caveat.

Figures: `V3STATS__roc_by_config`, `V3STATS__effpurity_scan`, `V3STATS__loss_comparison`,
`V3STATS__model_verdict_table`, `V3MERGED__auc_tank_vs_merged`, `V3MERGED__auc_cv_boxplot`,
`V3MERGED__auc_by_campaign`, `V3MERGED__feature_importances`,
`V3MERGED__feature_separation`, `V3MERGED__nn_loss_curves`, `V3MERGED__roc_merged`.

---

## 4. What the background is made of

The species axis is `origin_pdg` — the **first ancestor that is neither e± nor γ**. This
is the middle ground between `bg_class` (stops at the gamma, so it reads "photon" for
~40% of hits) and `root_pdg` (over-shoots to the generator primary). `origin_pdg == −5`
means a **pure EM chain**, not a tracing failure.

**Normalised background light budget, one denominator, tank + world** (truth-tag/OPTICS,
world report §4.4):

| component | tank | world | total |
|---|---:|---:|---:|
| neutron capture | 6.1% | 53.5% | **59.5%** |
| μ⁻ | 6.3% | 5.0% | 11.3% |
| π⁺ | | | 5.4% |
| no complete non-EM chain | | | 6.1% |
| π⁰ / p / π⁻ / μ⁺ | | | 1.5 / 1.2 / 1.0 / 0.7% |
| dark noise / other | | | 13.1% |

By clusters: tank background 23.3%, dirt neutrons 63.7%, world non-neutron 13.0%.

**Dirt-neutron clusters are purer than signal**, not contaminated — median
`frac_neutron` 0.846 vs 0.773, median `frac_nonneutron` **0.000** vs 0.125. What separates
them is light yield: ~15% fewer hits, ~20% less charge, at identical `d_wall` (0.048σ).
**The mechanism is not established — do not present one.**

Two things not to say: never quote `sigma_t_mad` **means** as a timing difference (9.9 vs
30.7 ns; the medians are 4.5 vs 5.4 and the separation is 0.051σ — a pure tail effect),
and never histogram `ImmediateAncestorClass` alone (it never says "neutron").

**A total-charge box cut will not work.** The species mix is flat in `pe_total` to within
30% across a decade. Only μ⁺ has structure, and the discriminator already removes it. What
survives into the signal-like tail is π⁺/π⁻/p; μ⁻ sits flat at 50–68% of the background.

Figures: `V3MERGED__bkg_composition_tank_vs_world`, `V3MERGED__bkg_composition_streams`,
`V3MERGED__training_background_by_particle`, `V3BG__*__top6_stacked`,
`V3BG__*__top6_shape`, `V3BG__dominance_summary`.

---

## 5. Background anatomy — one particle, or several? *(new)*

`python ccinc_v3_stats.py --do spurious`. Read-only over the four merged score tables,
behind the 16-AUC guard, and it asserts the integrated species shares still reproduce
world report §4.2 before computing anything.

### 5.1 The question

§4 gives the composition **integrated over all clusters**. That is a statement about the
light, not about the clusters, and the two coincide only if a typical background cluster
is made by a single particle. Whether they do decides how the per-species feature plots
in §5.3 may be read.

### 5.2 The answer: the tank background is a mixture, the world background is not

Per cluster, over the traced non-EM background hits (`n_origin_traced` as denominator):

| population | clusters | no non-EM light | median leading share | single-particle | particles/cluster | median frac_n |
|---|---:|---:|---:|---:|---:|---:|
| SIGNAL | 39,133 | 31.6% | 1.00 | 77.1% | 1.25 | 0.80 |
| BKG tank | 8,252 | 2.9% | **0.63** | **17.6%** | **2.19** | 0.25 |
| BKG world (dirt n) | 26,139 | **96.7%** | 1.00 | 96.2% | 1.04 | 0.88 |
| BKG world non-neutron | 4,742 | 39.6% | 1.00 | 86.2% | 1.18 | 0.00 |
| **BKG all** | 39,133 | **70.0%** | **0.75** | **40.1%** | 1.86 | 0.80 |

(Truth-tag / ClusterFinder. The percentages in the last four columns are over clusters
that have traced non-EM light; the "no non-EM light" column is the rest. All four
configurations agree — the tank single-particle fraction is 14.8–19.0% and the dirt-n
"no non-EM light" fraction 94.0–96.8%.)

Three things follow:

1. **70% of background clusters carry no traced non-EM light at all.** They are not
   tracing failures — they are the out-of-tank captures, median 88% neutron light. The
   "background" of this analysis is mostly neutron capture in the wrong place.
2. **The tank background is genuinely mixed** and is more mixed than the signal
   (2.19 particles vs 1.25; 17.6% single-particle vs 77.1%). Its dominant mixture is
   **μ⁻ + π⁺ in near-equal shares** — present in 72.4% of the 7,030 mixed clusters,
   mean shares 0.433 / 0.447. Next are μ⁻+p (19.4%), μ⁻+π⁰ (17.1%), μ⁻+π⁻ (12.8%).
3. **The world non-neutron background is single-particle** (86.2%), so it *is* legitimate
   to speak of "a π⁺ cluster" there.

**Consequence for reading §5.3:** a per-species curve is the distribution of clusters
that particle **dominates**, not of a pure single-particle population — and that caveat
bites hardest exactly where the interesting background lives, on the tank side.

Figures: `V3SPUR__purity`, `V3SPUR__n_species`, `V3SPUR__cooccurrence`,
`V3SPUR__species_census`. CSVs: `ccinc_v3_spurious_{species_shares,purity,byspecies,cooccurrence}.csv`.

### 5.3 Per-particle feature plots and per-particle discrimination

`make_presentable_optics_rf_plots.py --bkg-split origin` replaces the single grey
"Background" histogram with one curve per dominant background particle, on the same
ranked features, for all four configurations. `--bkg-split none` reproduces the
established pages byte-for-byte (verified).

**Per-particle one-vs-signal AUC, truth-tag / ClusterFinder:**

| particle | clusters | % of bkg | median leading share | GBT AUC | best feature (σ) |
|---|---:|---:|---:|---:|---|
| no non-EM origin (out-of-tank capture) | 27,393 | 70.0 | — | **0.705** | `n_hits` (0.636) |
| μ⁺ | 273 | 0.7 | 0.59 | 0.645 | `pe_total` (0.702) |
| μ⁻ | 6,463 | 16.5 | 1.00 | 0.615 | `charge_bal_legacy` (0.311) |
| π⁻ | 512 | 1.3 | 0.71 | 0.599 | `n_hits` (0.206) |
| p | 728 | 1.9 | 1.00 | 0.579 | `charge_bal_legacy` (0.328) |
| π⁺ | 2,727 | 7.0 | 0.67 | 0.561 | `charge_bal_legacy` (0.174) |
| π⁰ | 1,010 | 2.6 | 0.67 | 0.552 | `charge_bal_legacy` (0.160) |

K⁺ (7), K⁻ (13) and "other" (7) are below the 200-cluster floor and are logged and not
drawn — they are 0.24% of the background, not a gap in coverage.

Reading it:

- The classifier's headline 0.672 is an average over populations spanning **0.552 to
  0.705**. The part it does best on is the out-of-tank capture — the part that is 70% of
  the background by clusters and that no feature of the light can distinguish on physics
  grounds; what it is actually separating there is light **yield** (~15% fewer hits).
- **π⁺ and π⁰ are essentially inseparable** (AUC 0.55–0.56; best single-feature
  separation 0.16–0.17σ). Together they are 9.6% of the background and there is no handle
  on them in the current feature set.
- μ⁺ separates best of the charged species (0.645, `pe_total` 0.702σ) but is 0.7% of the
  background — removing it changes nothing. This is the same conclusion the total-charge
  box-cut study reached from the other direction (§4).
- `charge_bal_legacy` is the best feature for four of the six charged species, while
  `n_hits`/`pe_total` lead only for the out-of-tank captures and μ⁺. **The features that
  drive the merged AUC are not the features that separate the charged background.**

Figures (per configuration prefix `TT_CF_GBT__`, `TT_OPTICS_XGB__`, `RT_OPTICS_GBT__`,
`RT_CF_GBT__`): `*_features_byspecies`, `*_byspecies_top2_<feature>`.
CSV: `ccinc_v3_species_separation.csv` (per config × particle × feature: importance,
medians, σ separation, and the four one-vs-signal AUCs).

**How this differs from `V3BG__*`.** `ccinc_v3_bg_differential.py` shows the
**composition of a bin** along a feature axis (hit-weighted, deciles): "what is this bin
made of". §5.3 shows the **distribution of each particle** overlaid on the signal: "where
does this particle live". Both are kept; they answer different questions and should not
be presented as alternatives.

---

## 6. AmBe validation — run 6266 (AmBe2.0v4) *(new)*

`bash run_ccinc_v3_ambe6266.sh all` then
`python ccinc_v3_ambe_closure.py --do all --dataset ambe6266`.

### 6.1 What this can and cannot test

AmBe data has no truth labels: there is no efficiency and no purity here. Two things are
measurable — **data/MC score agreement** and **capture-time closure** against the
τ = **30.53 ± 0.26 µs** anchor (`REPORT_fprompt_waveform_cut.md` §7; the older 29.16
anchor is dead, the CSVs it was fitted on no longer exist). The fitter is **imported**
from `fit_capture_time_fprompt_compare.py` so the binning, 10–67 µs window and functional
form match the anchor. Capture time comes from `t_mean`.

**This is a one-run pilot.** Run 6266 is a genuine source-in run (special-runs
diagnostics, step 7); 6264 is ~2/3 cosmic-induced captures and 6270 is still corrupt, so
neither is scored. ClusterFinder only — OPTICS does not transfer to AmBe data.
**v4 recorded rates are not comparable across campaigns**: DAQ dead time τ ≈ 2 s caps the
recorded rate at 0.5 Hz. Yields and shapes are fine; rates are not.

### 6.2 The IC gate, derived here

No `EventAmBeNeutronCandidates_*_6266.csv` exists anywhere — that directory stops at run
6242. The gate was derived from the diagnostics join
`step4_joined_6266.parquet` via `make_gate_csvs_from_parquet.py --source step4`:
**2,171 of 4,182** IC candidates pass. `timestamp` there **is** `eventTimeTank`, verified:
all 4,182 match a BeamCluster `eventTimeTank` exactly. Both gates were then run, so the
gate's effect is measured rather than assumed.

| | events | CF clusters | clusters/event | Stage-1 pass |
|---|---:|---:|---:|---:|
| 6266 ungated | 37,260 | 43,284 | 1.162 | 55.4% |
| 6266 IC-gated | 1,248 | 1,403 | 1.124 | 56.9% |
| June v1 pooled (31 runs, gated) | 79,633 | 93,809 | 1.178 | 64.0% |

**The gate does not change the cluster multiplicity** — 1.16 → 1.12 per trigger, and the
June sample sits at 1.18. What it changes is which triggers are in the sample at all
(75,625 → 1,248) and their purity. An earlier version of this table read 2.88 → 1.14 and
claimed the gate collapsed multiplicity; that was wrong, and came from keying events on
`event_number`, which restarts per part file (see §6.6.1). **The gated sample is the
like-for-like comparison to June; the ungated one is where the statistics are.**

### 6.3 Capture-time closure — it works

Truth-tag / ClusterFinder, ungated:

| selection | clusters | data pass frac | MC eff | τ [µs] | χ²/ndof | pull vs anchor |
|---|---:|---:|---:|---|---:|---:|
| all clusters | 43,284 | 1.000 | — | 33.80 ± 3.74 | 2.49 | — |
| passes_stage1 | 23,962 | 0.554 | — | 28.70 ± 4.09 | 1.98 | — |
| GBT @ eff50 | 12,502 | 0.289 | 0.500 | 27.08 ± 5.56 | 1.45 | −0.62 |
| **GBT @ eff80** | 28,140 | 0.650 | 0.800 | **29.69 ± 3.73** | 1.79 | **−0.23** |
| GBT @ eff90 | 35,100 | 0.811 | 0.900 | 31.07 ± 3.61 | 1.84 | +0.15 |

All twelve model × working-point fits on the ungated sample are usable and **all are
consistent with the anchor within 2σ**. The realised data pass rate tracks the MC signal
efficiency closely (0.650 vs 0.800 at eff80, 0.811 vs 0.900 at eff90), so the MC-calibrated
threshold is landing roughly where it was meant to.

**What the MVA is responsible for is the change from baseline to selected**, because the
anchor was fitted on a different pipeline. That change is the result: 33.80 ± 3.74 with
χ²/ndof 2.49 for all clusters → 29.69 ± 3.73 with χ²/ndof 1.79 at eff80. The selection
removes the flat background that was inflating τ and lands on the anchor.

**The gated sample cannot fit a τ.** All 12 gated fits fail the quality gate — every one
has τ_err between 10 and 33 µs (one NN point pins at the 10 µs bound). The central values
are not alarming (RF eff80 30.27 ± 10.60, XGB eff80 24.68 ± 16.28), they are simply
uninformative: **1,403 clusters from one run is not enough to measure a 30 µs lifetime.**
This is a statistics statement about a one-run pilot, not a statement about the gate.

Figures: `V3AMBE_AMBE6266__capture_time_selected`, `V3AMBE_AMBE6266__tau_summary`.

### 6.4 Score calibration does **not** transfer — and why

| sample | model | data median | KS to MC signal | KS to MC bkg | closer to |
|---|---|---:|---:|---:|---|
| June v1 (gated) — *see §6.10.1: this is v1 only, not v1/v3* | GBT | 0.546 | **0.115** | 0.183 | signal |
| 6266 gated | GBT | 0.510 | 0.204 | **0.141** | background |
| 6266 ungated | GBT | 0.504 | 0.212 | **0.104** | background |

On the June sample the CF models sat closer to the MC **signal**, which is the statement
"the model transferred". On 6266 they sit closer to the MC **background**, for every model
and both gates. The score median drops by ~0.04 and the Stage-1 pass rate by 7 points.

**This is a source-position effect, not an extraction bug.** Comparing feature medians:

| feature | MC signal | June data | 6266 gated | 6266 / June |
|---|---:|---:|---:|---:|
| `vtx_y` [m] | 0.147 | 0.020 | **0.520** | — |
| `d_wall` [m] | 0.858 | 0.843 | **0.675** | 0.80 |
| `n_hits` | 15.0 | 12.0 | 11.0 | 0.92 |
| `n_hits_early` | 14.0 | 10.0 | 9.0 | 0.90 |
| `n_fit_hits` | 9.0 | 7.0 | 6.0 | 0.86 |
| `sigma_t_mad` [ns] | 3.75 | 4.22 | 4.53 | 1.07 |
| `charge_bal_legacy` | 0.282 | 0.349 | 0.378 | 1.08 |

Run 6266's clusters sit **half a metre higher and 0.17 m closer to the wall** than the
June pooled set, and consequently collect ~10% fewer hits with ~7% worse timing — which
pushes them down exactly the axes the classifier ranks first. Run 6266 is **not in
`SOURCE_POSITIONS`** (the map stops at 6256) and its port reads `unknown`; the June sample
is 31 runs in the 4xxx range across five ports. So this comparison is cross-campaign
**and** cross-position, and the score shift is the expected consequence.

**Note `d_wall` is in metres.** The presentable script used to label it `[cm]`; that wrong
label is on the June-16 poster figures too.

### 6.5 Verdict on 6266

The v4 pipeline reproduces the physics — capture time closes on the anchor and the MVA
improves it. The **absolute score scale does not transfer between AmBe campaigns and
source positions**, so an MC-calibrated threshold quoted as "80% efficient" delivers 65%
on this run. Before pooling 6266 with the June runs, its port must be added to
`SOURCE_POSITIONS`; before quoting a τ from the gated stream, more than one run is needed.

---

## 6.6 The new neutron definition on the AmBe pipeline's own neutrons *(new)*

`bash run_ccinc_v3_ambe6266.sh --campaign v4gated all` then
`python ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4gated`.

### 6.6.1 What is being compared

The AmBe analysis pipeline already defines a neutron. Stage 1 gates on the IC waveform
cut; Stage 2 keeps clusters with `0 < clusterPE ≤ 100`, `0 < CB < 0.45`,
`clusterTime ≥ 2000 ns`, `clusterHits ≥ 5` after a cosmic veto
(`clusterTime < 2000 ns` or `clusterPE > 100 PE`); and it computes efficiency and
capture time per source position. This section asks what the CCinc v3 discriminator
calls *those* clusters.

Sample: **all 28 AmBe2.0v4 source runs, 26 source positions across all five ports.**
Twenty had a Stage-1 candidate CSV already (`AmBe2.0v4_gated`); the other eight
(6243, 6244, 6246, 6247, 6249, 6250, 6251, 6252) had never been Stage-1 processed and
were run here under a separate tag `AmBe2.0v4_ext` with the same 700–1200 IC window —
separate because `AmBeWaveformResults_<tag>.csv` is written with a plain `to_csv`, so
reusing the tag would have overwritten the 20-run acceptance table the published 57.30%
efficiency rests on.

**Not in this sample, and why:** 6254 and 6256 are pulser runs with **zero** IC-passing
triggers (0 of 7,932 and 0 of 4,254) — the gate has nothing to select, so they cannot
enter the streamline at all. 6264/6265/6266/6270 are labelled no-source and are reported
separately in §6.7.

**Two methodological points, both measured rather than assumed.**

1. **The candidate CSVs cannot be scored directly.** They carry everything needed —
   cluster summary, full hit lists, source position — but their hit lists are
   ClusterFinder's *full stored membership*, whereas every MC cluster the model trained
   on was built from delayed-residual hits within (−5, +20) ns of `clusterTime`. On run
   6062 that window keeps **81.4%** of the stored hits (median 0.816; exact agreement in
   5.2% of clusters), and `n_hits`/`n_hits_early`/`pe_total` are the model's three
   leading features. Scoring the CSVs would have fed the model ~23% more light than it
   ever saw and biased every score upward. Features are therefore re-derived through the
   MC-matched path, and ClusterFinder's own summary is carried alongside as
   `cf_clusterTime/PE/CB/Hits/Number` so the Stage-2 cut is applied *exactly*, to the
   same table, with no join.
2. **The gate directory must be tag-filtered.** `build_gate_sets` globs every
   `EventAmBeNeutronCandidates_*.csv` and unions by run number, and `AmBe2.0v4_gated`
   and `AmBe2.0v4_fprompt` cover the **same 20 runs**. On run 6062 the union is 9,195
   events against gated's 7,755 — **+18.6%, and neither tag is a subset of the other**.
   (The June v1 extraction was not affected: `AmBe2.0v1` and `AmBe2.0v1CH5` share runs
   but their gate sets are identical.)

**Closure.** 188,915 CF clusters over 123,010 events; applying the Stage-2 definition
gives **173,354 candidates**, against the pipeline's own **181,636** — a 4.6%
under-count, in the expected direction: the extractor's cosmic veto drops the whole
event while the pipeline's `break`s out of its cluster loop and keeps candidates found
earlier. Membership regression: mean `n_hits / cf_clusterHits` = **0.870**, i.e. the
MC-matched window is running (1.0 would mean raw membership leaked in). Stage-1 pass
rate 63.4%, matching June's 64.0% — this campaign sits on the same footing as v1/v3,
unlike run 6266's 56.9% (§6.4).

### 6.6.2 The split

Truth-tag / ClusterFinder, the data deliverable. **Percentages are of the pipeline's own
Stage-2 AmBe neutrons.**

**225,945 Stage-2 AmBe neutron candidates.**

| working point | GBT threshold | MC eff | classified **neutron** | classified **other** | of Stage-2-*rejected* clusters, % called neutron |
|---|---:|---:|---:|---:|---:|
| eff50 | 0.59 | 0.50 | **42.7%** | 57.3% | 3.8% |
| **eff80** | **0.42** | **0.80** | **79.0%** | **21.0%** | **19.6%** |
| eff90 | 0.33 | 0.90 | **91.6%** | 8.5% | 40.0% |

All four models agree to within a few points at each point (eff80: RF 78.7%, GBT 79.0%,
XGB 78.3%, NN 75.6%). Reco-tag/CF is the same to <1 point — it is carried as a
cross-check and is not data-applicable. **Adding the 8 new runs moved nothing**: the
20-run answer was 79.0/21.0 with 19.7% of rejected, against 79.0/21.0 and 19.6% here.
That stability across a 30% larger sample is itself the result.

**The two definitions largely agree, and the discriminator is doing real work.** At
eff80 the MVA keeps 79.0% of what Stage 2 accepts but only 19.7% of what Stage 2
rejects — a **4.0× contrast**. At eff50 the contrast is 11× (42.3% vs 3.8%). If the MVA
carried no information about the AmBe cuts these two numbers would be equal.

**Single vs multiple.** Single-neutron candidates (`clusterNumber == 1`) are kept more
often than multiples at every point — 81.1% vs 71.9% at eff80. Multi-cluster events are
where the pile-up and accidentals live, so this is the expected direction.

### 6.6.3 Is what it rejects actually background?

**The per-position capture-time fits cannot answer this, and it would be wrong to claim
they do.** All 38 "MVA other" fits fail the quality gate — but every one fails on
`tau_err` (10–33 µs) with χ²/ndof near 1.3. The fits are fine; the rejected sample is
simply too small per position to pin a 30 µs lifetime. "No usable fit" is a statistics
statement, not evidence of background. (The same trap is on record from the fprompt
study: a flat pedestal is degenerate with a long τ, and `lmfit_analysis` fixes `B` at 0
so a pedestal *inflates* τ rather than being absorbed.)

Settled model-free instead, on the anchor's own 10–67 µs window, pooling positions —
legitimate here because these are shape statistics, not fits:

| | kept (147,275) | rejected (39,483) | difference |
|---|---:|---:|---|
| mean capture time [µs] | 30.762 ± 0.040 | 31.859 ± 0.078 | **+1.097 ± 0.088 (+12.5σ)** |
| late/early ratio (30–67 / 10–30 µs) | 0.8150 ± 0.0043 | 0.9329 ± 0.0094 | **×1.145** |
| KS distance | — | — | **D = 0.035, p = 3×10⁻³⁴** |

**The rejected fraction is measurably flatter.** A flat accidental component raises the
late/early ratio and pushes the mean later, and both move together here at >10σ. Reco-tag
reproduces it independently (+11.9σ, ×1.133). So the 21% the MVA discards is enriched in
a flat, non-exponential component — which is what "background" means on this axis — and
that is also *why* its per-position fits will not converge.

The kept sample fits well: weighted **τ = 29.98 µs** over 45 positions (median χ²/ndof
1.26) against the **30.53 ± 0.26 µs** anchor, versus 30.41 µs (χ²/ndof 1.36) for all
candidates. The improvement is small — the pipeline's own cuts already remove most of
the background — but it is in the right direction on both τ and χ².

### 6.6.4 The score scale depends on source position — now measured

§6.4 inferred from a single run that the score scale moves with where the source sits.
With **26 positions across all five ports** that is now a measurement. GBT at eff80,
kept fraction per position:

| port | positions | mean | range |
|---|---:|---:|---|
| port5_z0 | 5 | 83.7% | 80.0 – 86.0 |
| port1_z-75 | 5 | 80.7% | 75.6 – 84.6 |
| port4_x75 | 5 | 79.4% | 72.7 – 82.7 |
| port2_z75 | 5 | 76.9% | 69.3 – 81.5 |
| port3_z102 | 6 | 70.3% | 64.2 – 75.0 |

**Total spread 64.2% – 86.0%, i.e. 21.8 percentage points** across positions, for a
threshold nominally calibrated to 80%. Port 3 is systematically worst, which is also the
port with the lowest pipeline efficiency (46.1% in v4).

What drives it is **light yield, not geometry**, and the full sample sharpens this
considerably. Across the 26 positions the kept fraction correlates with `median_n_hits`
at **r = +0.91** and `median_pe_total` at **+0.86**, but with `median_d_wall` at
**+0.01** — i.e. no dependence on distance to the wall at all. That is the same conclusion the MC
reached from the other direction — the merged separation is multiplicity-led while
`d_wall` ranks 20th of 31 features — and it means the position dependence is a light-
collection effect, not the classifier learning where the source is.

Practically: **a threshold quoted as "80% efficient" delivers 64–86% depending on
position.** Any efficiency the new definition is used to compute has to be either
calibrated per position or quoted with this spread attached.

Figures: `V3AMBEN__classified_fraction`, `V3AMBEN__classified_by_position`,
`V3AMBEN__classified_capture_time`. CSVs:
`ccinc_v3_ambe_classification{,_byposition,_capturetime,_shapetest}_ambepipe_v4neutron.csv`.
(The `V3AMBEPIPE__*` / `_ambepipe_v4gated` set is the same study on the first 20 runs and
is superseded by this one.)

---

## 6.7 The special runs — reported separately *(new)*

`bash run_ccinc_v3_ambe6266.sh --campaign v4special all` then
`python ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4special`.

Six runs are not ordinary source runs and are **never merged** into a capture time or
efficiency number. The split follows the diagnostics' own
`step7_source_presence.csv`, not a judgement call.

**Two cannot be processed at all.** 6254 and 6256 are `kind=pulser`, no source, and have
**zero** IC-passing triggers — 0 of 7,932 and 0 of 4,254 candidates. That is a result,
not a limitation: with no AmBe source there is nothing for the IC gate to select.

**Four were run**, gated from the diagnostics join (verified to be the same 700–1200 IC
window plus second-pulse veto the pipeline applies), since they have no Stage-1 CSV:

| run | port / Z | gated triggers | cosmic-vetoed | Stage-2 candidates | **% neutron** (GBT @ eff80) |
|---|---|---:|---:|---:|---:|
| 6265 | 5 / 0 | 2,150 | 25.1% | 1,127 | **86.2%** |
| 6270 | 4 / 0 | 2,312 | 26.6% | 1,485 | **83.4%** |
| 6266 | 5 / 100 | 2,171 | 20.2% | 1,049 | **79.0%** |
| **6264** | 5 / 0 | 149 | **93.3%** | **0** | — |

**Run 6264 contributes nothing to the neutron streamline.** 139 of its 149 gated events
are cosmic-vetoed, leaving zero clusters — a clean quantitative confirmation of the
diagnostics' finding that it is cosmic-dominated (10× the prompt rate, τ = 48 µs). The
other three veto at 20–27%, so 6264 is roughly 4× higher.

The three that survive land at **79–86%**, inside the neutron campaign's per-position
range and at its high end — consistent with the diagnostics' finding that 6265/6266/6270
behave like source-in runs despite carrying a no-source label. Their of-rejected figure
is 36.9% against the neutron campaign's 19.6%, i.e. the MVA is much less decisive about
what Stage 2 discarded here; on ~1,000 clusters per run that is expected. **No τ is
quoted for any special run** — every per-run fit fails the quality gate on statistics,
and the per-run shape test has too few rejected clusters to run.

**6270 carries an additional caveat**: `BeamCluster_6270.root` is still the corrupted
merge — 11,195 duplicated events of 46,211 (24%), stale chunk still in scratch
(`FIX_6270.md`). Its numbers are reported but should not be quoted without that.

Positions for the four come from the run/port/Z table and live in `SPECIAL_POSITIONS` in
`ccinc_v3_ambe_closure.py`, **deliberately not** added to `src/ambe/data/processor.py`'s
`source_positions`: that map drives the pipeline's own efficiency heatmaps, and putting
labelled-no-source runs into it would silently fold them into the campaign efficiency.

Figures: `V3AMBESPEC__classified_fraction`, `V3AMBESPEC__classified_by_position`.

---

## 6.8 The traditional box cuts, over the whole v4 campaign *(new)*

`python fit_capture_time_fprompt_compare.py --mode tagset --tags AmBe2.0v4_gated,AmBe2.0v4_ext --out-label v4_all28`
then `python boxcut_v4_campaign.py --do all`.

**No MVA and no MC appear anywhere in this section.** The neutron definition is the AmBe
pipeline's own, and the point is to establish that it behaves as expected across all 26
source positions before anything is asked of the MVA. Everything is computed from
artifacts already on disk; no ROOT file is re-read and Stage 1 is not re-run.

### 6.8.1 The sample, and why it needed merging at config level

All **28 AmBe2.0v4 source runs, 26 positions, five ports**. Twenty sit under Stage-1 tag
`AmBe2.0v4_gated` (19 positions), the eight later ones under `AmBe2.0v4_ext` (7
positions). They are separate tags on disk because `AmBeWaveformResults_<tag>.csv` is
written with a plain `to_csv` — reprocessing the eight under `_gated` would have destroyed
the 20-run acceptance table the published 57.30% rests on.

The merge is therefore done in `configs/data_ambe2v4_all28.yaml`, by listing both files;
`resolve_inputs`/`load_csvs` already support lists, so no plotting code changed. **The
merge is only valid because the two tags share no source position** — 19 + 7 = 26 unique.
That is asserted at every run, not assumed. Two traps sit next to it:

- Globbing `AmBeTriggerSummary_AmBe2.0v4_*.csv` instead would also match `_fprompt`,
  which covers the **same 20 runs** under a different Stage-1 cut, double-counting 19 of
  the 26 positions.
- `src/ambe/data/eff.py` looks like the box-cut implementation and is not: it is the
  unported legacy module (`run()` raises), it calls `input()` at import, its `AmBe()` has
  `clusterHits >= 5` **commented out** (lines 24–25), and it hardcodes the IC window as
  `500 < IC < 1400` (line 219) rather than 700–1200. The maintained path is
  `processor.py`'s `CutCriteria` / `ambe_single_cut`.

**Run 6248 is absent because it does not exist on disk.** No `BeamCluster_6248.root`, no
waveform directory, anywhere under `/pnfs/.../dajana` or `/exp/annie/data/users/dajana`.
It is in `processor.py:151` `source_positions` at (75, 50, 0) as a position entry only, so
that position is covered solely by the LAPPD debug pair 6249/6250. Including it needs a
grid transfer first.

### 6.8.2 The cut flow

| stage | events | of Stage-1 |
|---|---:|---:|
| IC-gated triggers (Stage 1) | 474,192 | 100% |
| − cosmic veto | −94,120 | −19.85% |
| = AmBe triggers | 380,072 | 80.15% |
| single-neutron events | 174,583 | 36.82% |
| multi-neutron events | 42,949 | 9.06% |
| **= neutron triggers (Stage 2)** | **217,532** | **45.87%** |

The 217,532 and the 236,792 candidate rows behind it reproduce `PIPELINE_REFERENCE` in
`ccinc_v3_ambe_closure.py` **exactly**, so the box-cut route and the MVA route are
demonstrably reading the same sample.

### 6.8.3 Efficiency — and the consistency statement that matters

Campaign efficiency, `unique_neutron_triggers / ambe_triggers`: **57.23 ± 0.08 %**, range
**40.31 % → 71.82 %** across the 26 positions.

| sample | positions | AmBe triggers | neutron triggers | efficiency |
|---|---:|---:|---:|---:|
| 20 runs (published) | 19 | 290,843 | 166,665 | **57.30 ± 0.09 %** |
| 8 runs (added) | 7 | 89,229 | 50,867 | 57.01 ± 0.17 % |
| **28 runs (all)** | **26** | **380,072** | **217,532** | **57.23 ± 0.08 %** |

**Adding eight runs moved the campaign efficiency by −0.07 pp.** That is the QA result:
a 31% larger trigger sample and 7 new positions changed nothing outside the error.

The position spread is real and is not a detector fault. Port 3 holds the four lowest
positions (40.3–50.8%) and port 5 y=0 the highest (71.8%); the same ordering appears in
the AmBe 1.0 and 2.0v2 reference maps, and — independently, from the other direction — in
the MVA's own kept-fraction ranking (§6.6.4, `n_hits` r = +0.91). Two positions agree to
better than 0.1 pp with each other across ports, which is the internal cross-check.

Two shape checks come out flat, as they should:

- **Cosmic-veto fraction 19.85 %** campaign-wide, 15.57–22.78 % across positions. A
  source-position-dependent cosmic rate would have meant a livetime or gating problem.
- **Multi-neutron share 19.74 %** of neutron triggers, 18.01–23.67 %.

### 6.8.4 Capture time on 26 positions — the anchor is stable

Fits use the anchor's own recipe, reused rather than reimplemented: 70 bins over 0–70 µs,
10–67 µs window, `A(1−e^{−t/therm})e^{−t/τ} + B` with **B fixed at 0**, and the `is_good`
quality gate. `fit_capture_time_fprompt_compare.py` gained a `--mode` argument (no
default: `fpcompare` keeps its original behaviour, `tagset` fits an arbitrary campaign),
so there is one fit recipe in the repo and not two.

| sample | positions | weighted τ [µs] | median χ²/ndof | clusters in window |
|---|---:|---|---:|---:|
| frozen anchor (`CaptureTimeFits_baseline.csv`) | 19 | **30.529 ± 0.260** | 1.32 | 149,507 |
| all 28 runs (`CaptureTimeFits_v4_all28.csv`) | 26 | **30.535 ± 0.228** | 1.37 | 194,729 |

**Extending 19 → 26 positions moves τ by +0.005 µs and shrinks the error by 12 %. All 26
fits converge.** The frozen file is not touched: re-running the new mode on the 20-run tag
alone reproduces `CaptureTimeFits_baseline.csv` **byte-identically** and returns
30.529 ± 0.260, which is the regression that licenses everything above.

**Do not quote a pooled fit.** Pooling all 26 positions into one histogram gives
τ = 32.90 ± 0.16 µs with χ²/ndof 14.7 and `therm` pinned near its lower bound. That is not
a measurement — with ~195k clusters drawn from positions whose light collection genuinely
differs, a single thermalisation+exponential cannot describe the sum, and χ² says so. The
per-position-then-weighted recipe exists for exactly this reason.

Figures: `V4BOX__cutflow_table`, `V4BOX__efficiency_by_position`,
`V4BOX__consistency_table`, `V4BOX__capture_time_by_position`,
`V4BOX__capture_time_anchor_comparison`, `V4BOX__cosmic_fraction`,
`V4BOX__multiplicity_share`, `V4BOX__cut_variables`. CSVs: `boxcut_v4_{cutflow,
efficiency_by_position,consistency,capturetime_by_position}.csv`.

---

## 6.9 Box cuts on the special runs, separately *(new)*

`python boxcut_v4_special.py --do all`.

Same cuts, same code path (`processor.py`'s `cosmic_cut` / `ambe_single_cut` /
`ambe_multiple_cut`), **no efficiency and no τ**. Stage 1 comes from the diagnostics join
`step4_joined_<run>.parquet` because none of these runs has a Stage-1 CSV; positions come
from `SPECIAL_POSITIONS` in the closure script and are deliberately **not** added to
`processor.source_positions`, which drives the campaign heatmaps.

| run | position | IC candidates | gated | cosmic-vetoed | AmBe triggers | Stage-2 candidates |
|---|---|---:|---:|---:|---:|---:|
| 6254 | pulser, no source | 7,932 | **0** | — | — | **0** |
| 6256 | pulser, no source | 4,254 | **0** | — | — | **0** |
| **6264** | port 5, y=0 | 4,624 | 149 | **98.0 %** | 3 | **0** |
| 6265 | port 5, y=0 | 4,124 | 2,150 | 25.1 % | 1,611 | 1,184 |
| 6266 | port 5, y=100 | 4,182 | 2,171 | 20.3 % | 1,731 | 1,100 |
| 6270 | port 4, x=75 | 4,366 | 2,312 | 20.9 % | 1,828 | 1,239 |

**The gated-trigger counts reproduce §6.7's MVA route exactly for all four runs** — 149,
2,150, 2,171, 2,312. Two independent gates over two independent code paths agreeing to
the event is the check worth stating.

**Getting 6270 to agree required de-duplicating on `eventTimeTank`, and that is a result
about the file, not a fitting choice.** `BeamCluster_6270.root` is still the corrupted
merge: **11,279 duplicated events of 46,211 (24.4 %)**, confirming `FIX_6270.md`
independently. 629 of the duplicates fall inside the gate, so an undeduplicated loop
reports 2,941 gated triggers (+27 %) and every downstream yield is inflated by the same
factor. 6265 carries 337 duplicates (0.8 %) from the same class of fault, none of them
gated. The event key is `eventTimeTank`, never `eventNumber`.

**6264 contributes nothing**, for the reason the diagnostics predicted: 146 of its 149
gated events are cosmic-vetoed, leaving three AmBe triggers and zero Stage-2 candidates.
Note the box-cut veto fraction is **98.0 %** against §6.7's **93.3 %**. Both give zero
candidates, and the difference is expected rather than a discrepancy — the MVA route
re-derives cluster features in a (−5, +20) ns delayed-residual window, so its `clusterPE`
differs and the >100 PE veto flips on a handful of events. Quote 98.0 % for the box-cut
analysis and 93.3 % for the MVA one; do not mix them.

The three that survive give **0.64–0.74 Stage-2 candidates per AmBe trigger** and cluster
shapes indistinguishable from source-in runs, consistent with the diagnostics' finding
that 6265/6266/6270 behave source-in despite a no-source label. **No τ and no efficiency
is quoted for any of them**, and 6270's numbers should not be quoted at all without the
corruption caveat attached.

Figures: `V4BOXSPEC__cutflow_table`, `V4BOXSPEC__yield_per_trigger`,
`V4BOXSPEC__cosmic_fraction`, `V4BOXSPEC__cluster_shapes`. CSVs:
`boxcut_v4_special_{summary,candidates}.csv`.

---

## 6.10 The MVA definition validated against the box cuts — and June retired *(new)*

`python ccinc_v3_ambe_closure.py --do all --dataset june` (reproduction) and
`--do closure --dataset ambepipe_v4neutron` (the v4-native replacement), then
`python boxcut_v4_campaign.py --do validation`.

### 6.10.1 What "June" was, and why it had to be replaced

The τ = **31.30 ± 2.28 µs** and the "the model transferred" score-agreement numbers quoted
throughout §6.4 come from a sample built in June from 31 gated runs on **old hardware**.

**Correction: that sample is AmBe2.0v1 only, not "v1/v3 pooled".** The label
"June v1/v3" in §6.4's table, and the `--dataset june` docstring in
`ccinc_v3_ambe_closure.py:112` ("the pooled AmBe2.0v1/v3 features"), are both wrong.
Counted directly from the staged score table, the 31 runs are **4499–4687 — every one in
the 4xxx range**, i.e. the v1 campaign minus 4604/4605. The `AmBe2.0v3` tag is entirely
5xxx (5740–5828) and **contributes zero runs**. So the June reference is one campaign
older than it has been described as, which strengthens rather than weakens the case for
replacing it. Read "June v1/v3" as "June v1" wherever it appears.

It was the only *multi-run, multi-position* AmBe
sample the MVA had ever been validated on: §6 contributed one v4 run (6266, whose gated
fits all fail on statistics) and §6.6 measures the kept/rejected split rather than an
MVA-selected τ. So a v1/v3-era number was carrying the validation, against a v4-derived
anchor.

**June reproduces exactly on current code** (the scored staging is intact, 176 MB):
τ = **31.303 ± 2.283 µs** at truth-tag/ClusterFinder GBT eff80, and GBT data median
0.5464 with KS 0.1147 to MC signal against 0.1827 to MC background — closer to signal, as
documented. Nothing has drifted. (The `d_wall [cm]` mislabel is already fixed in
`make_presentable_optics_rf_plots.py:170`; it survives only on the old June-16 poster
PDFs, which this code does not regenerate.)

### 6.10.2 The v4-native validation, and why it must be per position

Running the closure on all 28 v4 runs gives τ ≈ 31.5 µs at eff80 — with **χ²/ndof 8–12**,
i.e. unusable. **This is the opposite of the 6266 problem.** 6266 failed for want of
clusters; here there are ~181k of them, pooled across 26 positions with genuinely
different light collection, and a single thermalisation+exponential cannot describe the
sum. The same thing happens to the box cuts when pooled (§6.8.4, χ²/ndof 14.7). Per
position, both fit.

So the validation is done per position, on the **same 26 positions**, under **one** fit
recipe. The one methodological difference has to be stated: the anchor recipe holds the
flat pedestal **B at 0**, the closure **floats B** in (0, 15). B and τ are correlated, so
floating B inflates τ's error roughly 6×. The like-for-like row recomputes the box-cut
number with B floating too.

| selection (26 AmBe positions, truth-tag/CF, GBT eff80) | positions | weighted τ [µs] | median χ²/ndof |
|---|---:|---|---:|
| box cuts, B fixed at 0 (anchor recipe) | 26 | 30.535 ± 0.228 | 1.37 |
| box cuts, B floating | 26 | 29.501 ± 1.187 | 1.40 |
| MVA input (all Stage-2), B floating | 24 | 30.414 ± 1.371 | 1.36 |
| **MVA neutron @ eff80**, B floating | 23 | **29.835 ± 1.400** | **1.27** |
| MVA other @ eff80, B floating | 1 | no usable fit | 1.31 |

**Like-for-like the MVA and the box cuts agree to 0.18σ** (+0.334 ± 1.835 µs), and the MVA
*improves* the median χ²/ndof from 1.36 to 1.27 — it is removing something that did not fit
an exponential. That, plus the "MVA other" column having only one converged fit out of 26
(statistics, per §6.6.3 — the background claim rests on the model-free shape test), is the
fresh multi-position validation June was standing in for.

**June is now a historical cross-check, not the reference.** Never compare the 0.228 and
1.400 errors directly; that is a recipe difference, not a measurement difference.

### 6.10.3 The strict comparison — identical clusters, and it is the best result here

The table above still contains a 4.6% sample difference: the box cuts are measured on the
pipeline's 236,792 Stage-2 clusters, the MVA on the extractor's 225,945. The gap is the
documented §6.6.1 effect — the extractor's cosmic veto drops the whole event, the pipeline
`break`s out of its cluster loop and keeps candidates found earlier in the same event.

Joining **cluster by cluster** on `(run, event_tank_time, clusterTime)` removes it:
**225,810 clusters match exactly — 99.94 % of the MVA sample and 95.36 % of the box-cut
sample** (10,982 box-cut-only, 135 MVA-only). Both selections are then fitted on that one
set, per position, same recipe.

The working-point threshold is **read back from the classification CSV** rather than
hardcoded: truth-tag/CF GBT eff80 = **0.4236212** at MC efficiency 0.800025. That matters —
an earlier version of this table used a rounded 0.420, which admitted ~1,100 extra
clusters and made the count disagree with §6.6.2 for no physical reason.

| selection (identical 225,810 clusters, 26 positions) | clusters | τ [µs], B = 0 | τ [µs], B free |
|---|---:|---|---|
| box cuts | 225,810 | 31.509 ± 0.250 | 30.481 ± 1.353 |
| MVA input, all matched clusters | 225,810 | 31.509 ± 0.250 | 30.481 ± 1.353 |
| **MVA neutron @ eff80** (GBT > 0.4236) | 178,408 | **30.521 ± 0.263** | 29.745 ± 1.376 |
| **MVA other @ eff80** (GBT ≤ 0.4236) | 47,402 | **33.890 ± 0.661** | 28.416 ± 3.367 |

Two internal checks worth stating. Row 2 reproducing row 1 **exactly** confirms the join
is a join and not a re-selection. And the MVA-neutron count reconciles with §6.6.2 to the
cluster: **178,520** on the full MVA sample (79.010%, exactly what the classification
reports) and **178,408** on the matched set — the **112**-cluster difference being precisely
the MVA-only clusters the join drops.

**This is the cleanest statement in the whole AmBe chapter.** On *identical* clusters, the
MVA moves τ from **31.509 ± 0.250 → 30.521 ± 0.263**, i.e. onto the frozen anchor
**30.53 ± 0.26** (agreement to 0.03 µs) — a −0.989 ± 0.363 µs shift, **2.73σ**, in the
right direction. And what it discards sits at **33.890 ± 0.661 µs**, +2.4 µs later than the
sample it came from. That is the background claim made with a fit rather than only with the
model-free shape test of §6.6.3, and it is possible only because the cluster sets are
identical.

With B floating the same shift is −0.736 ± 1.930 µs (0.38σ) — consistent, just far less
sensitive, because floating B on 26 separate positions costs most of the resolving power.
**Quote the B = 0 row**: it is the anchor's own recipe, and it is the comparison that
distinguishes anything.

One caveat to state: the matched box-cut τ (31.509) is higher than the full-sample 30.535
of §6.8.4, because the 10,982 excluded clusters are precisely those in events the
extractor called cosmic. Removing them raises τ. So the matched numbers are internally
comparable to each other and **not** to §6.8.4's — do not mix the two tables.

Figures: `V4BOX__matched_boxcut_vs_mva_table`. CSV: `boxcut_v4_matched_mva.csv`.

### 6.10.4 The full cross-method agreement — both streamlines, all models, all points

`python boxcut_v4_campaign.py --do agreement`. §6.10.3 answers the question at one working
point on one streamline; this is the same comparison everywhere, on three axes.

**Capture time** (GBT, per position, B = 0, identical clusters):

| streamline | point | τ box cuts | τ MVA neutron | τ MVA other | Δ [µs] | signif. |
|---|---|---|---|---|---|---:|
| Truth-tag / CF | eff50 | 31.509 ± 0.250 | 29.746 ± 0.366 | 31.856 ± 0.350 | −1.763 ± 0.443 | 3.98σ |
| **Truth-tag / CF** | **eff80** | 31.509 ± 0.250 | **30.521 ± 0.263** | **33.890 ± 0.661** | −0.989 ± 0.363 | 2.73σ |
| Truth-tag / CF | eff90 | 31.509 ± 0.250 | 30.948 ± 0.251 | 34.521 ± 1.315 | −0.561 ± 0.354 | 1.58σ |
| Reco-tag / CF | eff50 | 31.509 ± 0.250 | 29.878 ± 0.366 | 31.984 ± 0.347 | −1.631 ± 0.444 | 3.68σ |
| Reco-tag / CF | eff80 | 31.509 ± 0.250 | 30.654 ± 0.262 | 33.120 ± 0.663 | −0.855 ± 0.362 | 2.36σ |
| Reco-tag / CF | eff90 | 31.509 ± 0.250 | 30.912 ± 0.253 | 33.331 ± 1.155 | −0.597 ± 0.355 | 1.68σ |

**The trend is the physics.** As the cut tightens, the kept sample's τ falls monotonically
(30.95 → 30.52 → 29.75) and the *discarded* sample's τ rises monotonically (33.33/34.52 →
33.89 → 31.86). A tighter cut throws away more, and what it throws away is progressively
more background-like. A selection that was cutting at random would move neither. Note eff50
overshoots — 29.75 is *below* the 30.53 anchor — so eff80 is not merely conventional here,
it is where the kept sample lands on the anchor. The two streamlines agree throughout, at
most 0.13 µs apart.

**Efficiency per source position.** Neither number is a truth efficiency (AmBe data has no
truth). Both are the *same observable* — the fraction of AmBe triggers yielding at least
one accepted delayed cluster — under two definitions of "accepted", with the same
denominator and counted over the same 225,810 clusters.

| | campaign efficiency, matched basis |
|---|---:|
| box cuts | **54.57 %** |
| MVA @ GBT eff90 | 50.92 % |
| MVA @ GBT eff80 | 44.72 % |
| MVA @ GBT eff50 | 24.76 % |

The MVA is strictly tighter than the box cuts at every point, as it must be — it is a cut
*applied on top of* them. (54.57 % rather than §6.8.3's 57.23 % because the matched basis
excludes the 10,982 extractor-cosmic clusters; use 54.57 % only inside this table.)

**Per-position, the two definitions agree almost perfectly: r = 0.993** at truth-tag GBT
eff80 over 26 positions, and **0.946–0.999 across all 24 model × point × streamline
combinations**. The MVA changes the *normalisation* but preserves the position ordering
entirely — which is the strongest available statement that it is not learning the source
position, and it agrees from the data side with §6.6.4's finding that the kept fraction
tracks `n_hits` (r = +0.91) and not `d_wall` (r = +0.01).

**All four models agree** on the kept fraction at every point: eff80 gives GBT 79.01 %,
RF 78.70 %, XGB 78.28 %, NN 75.64 % (truth-tag). NN is the outlier at eff50 (39.51 % vs
~42.7 % for the trees), consistent with its known poor calibration (§3).

Figures: `V4BOX__agreement_capture_time_table`, `V4BOX__agreement_efficiency_by_position`,
`V4BOX__agreement_kept_fraction`. CSVs: `boxcut_v4_agreement_{counts,efficiency,
capturetime}.csv`.

Figures: `V3AMBEN__capture_time_selected`, `V3AMBEN__tau_summary` (v4-native closure),
`V4BOX__validation_mva_vs_boxcut_table`, `V4BOX__validation_tau_scatter`, and
`V3AMBE__{tau_summary,score_data_mc_overlay,capture_time_selected}` for June. CSVs:
`boxcut_v4_validation_mva.csv`, `ccinc_v3_ambe_closure{,_ambepipe_v4neutron}.csv`.

---

## 7. Limits and further scope

**Physics limits, already established — these are answers, not open questions:**

- **The AUC ceiling is physics.** 52–68% of the background light is genuine out-of-tank
  neutron capture. No feature of the light separates it. Further gain has to come from a
  handle *outside* the cluster — MRD/FMV coincidence, event-level topology, beam timing —
  not from more cluster features.
- **A total-charge box cut will not work** (§4), and **π⁺/π⁰ have no handle at all** in
  the current feature set (§5.3).
- **The merged gain is not geometry** (§3), and **dirt-neutron clusters are pure, not
  mixed** (§4).

**Open, ranked by what they would change:**

1. **Calibrate the working point per source position.** This is now the largest known
   systematic on any efficiency the new definition produces: §6.6.4 measures a
   **20.4-point spread** (65.5–85.9%) across 19 positions for one nominal 80% threshold,
   correlating with light yield (r = +0.87 on `n_hits`) not geometry. Either fit the
   threshold per position, or quote every efficiency with this spread attached. Doing it
   per position is tractable — the 19-position table already exists.
2. **Extend the classification study to v1, v3 and v4_fprompt.** §6.6 covers
   `AmBe2.0v4_gated` only. v1 (90,423 candidates) and v3 (228,012) would test whether the
   79% split is campaign-stable; `AmBe2.0v4_fprompt` on the same 20 runs would answer a
   sharper question — does the new definition call the fprompt-recovered +23% yield
   neutrons, or is it recovering what the MVA would throw away? Each needs one extraction
   pass (~40 min per campaign) and nothing else.
3. **Scale the single-run 6266 validation.** The gated stream needs ~10× the statistics
   of 6266 to fit a τ. The clean v4 set excludes 6264 (2/3 cosmic at 20.5σ — also not a
   valid no-neutron reference) and 6270 (corrupt, `FIX_6270.md`). Only 6249/6250 and
   6254/6256 are clean LAPPD on/off pairs. **Add the 62xx runs to `SOURCE_POSITIONS`
   first** — those runs have no port entry at all, and §6.6.4 shows position is not a
   detail that can be averaged over.
4. **Diagnose the Stage-1 pass-rate asymmetry** (31.9% OPTICS vs 64.0% ClusterFinder; 56.9%
   for 6266 CF). Measured, never explained. It is the leading suspect for why OPTICS does
   not transfer to data.
5. **Test a `frac_neutron` signal definition.** §4.5 found 1,176 tank background clusters
   (14.0%, 27.9% of the tank background's neutron light) are majority-neutron and labelled
   background only because `dominant_class` is a plurality across four neutron class
   codes. Worth testing — but it shrinks the already-limiting background class.
6. **Attack the tank background as a mixture, not as species.** §5.2 shows tank background
   clusters average 2.19 particles with μ⁻+π⁺ in near-equal shares. Per-species cuts are
   the wrong tool there by construction; a mixture-aware target (or a cut on the μ⁻+π⁺
   topology) is the thing to try.
7. **Fix the frozen-`.pkl` NN filename at the source.** Every merged bundle still stores
   the pre-rename `__merged__nn.keras`. Both drivers work around it with staged symlinks;
   any other consumer loses the NN silently and reports success.
8. **Quantify the ~0.2% wall ambiguity** (§1) as a label systematic instead of a footnote.
9. **Reco-tag: build reco proxies for the FV and muon kinematics, or stop quoting it.**
   It is currently the best MC number and cannot be applied to data.

---

## 8. How to reproduce

```bash
source /exp/annie/app/users/dajana/myboy/bin/activate
export PYTHONUNBUFFERED=1
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis

# 0. the guard — run this first, everything else asserts it internally
python ccinc_v3_stats.py --do guard          # PASS: all 16 AUCs reproduce to 0.001

# 1. classifier selection, dirt-neutron decomposition, background anatomy
python ccinc_v3_stats.py --do models
python ccinc_v3_stats.py --do dirtn
python ccinc_v3_stats.py --do spurious       # §5.1-5.2
python ccinc_v3_bg_differential.py --config all

# 2. per-species feature plots (§5.3), one call per configuration.
#    --bkg-split none reproduces the established pages byte-for-byte.
B=/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output
python make_presentable_optics_rf_plots.py \
  --mc-scores $B/cc_neutrino_v3_truthtag/parquet/cc_neutrino_v3_truthtag__mva_scores__keepprompt__merged__cf.parquet \
  --frozen    $B/cc_neutrino_v3_truthtag/parquet/cc_neutrino_v3_truthtag__mva_frozen__keepprompt__merged__cf.pkl \
  --style poster --rank-model gbt --mc-only --score-cut auto \
  --bkg-split origin --config-tag truthtag/cf \
  --title "ClusterFinder + GBT — Top 6 Discriminating Features" \
  --name-prefix TT_CF_GBT__presentable --figures-dir slide_plots_ccinc_v3_merged \
  --out slide_plots_ccinc_v3_merged/TT_CF_GBT__presentable_ALL.pdf
# repeat with (truthtag, optics, xgb, TT_OPTICS_XGB), (recotag, optics, gbt,
# RT_OPTICS_GBT), (recotag, cf, gbt, RT_CF_GBT). NOTE the file suffix is `cf`,
# not `clusterfinder`.

# 3. AmBe run 6266 (§6) — ~15 min, dominated by the ungated extraction
bash run_ccinc_v3_ambe6266.sh --campaign run6266 all
python ccinc_v3_ambe_closure.py --do all --dataset ambe6266
python ccinc_v3_ambe_closure.py --do all --dataset june      # the June reference

# 4. The AmBe pipeline's own neutrons (§6.6) — all 28 source runs, ~45 min.
#    Stage 1 first for the 8 runs that were never processed, under a SEPARATE tag so
#    the 20-run TriggerSummary (plain to_csv, no append) is not overwritten:
MPLBACKEND=Agg python -u run_waveform_gated_pipeline.py \
  --dataset /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4 \
  --runinfo AmBe2.0v4_ext --runs 6243,6244,6246,6247,6249,6250,6251,6252
#    Then gate/extract/score/classify. The gate stage stages a TAG-FILTERED symlink
#    dir; do not point the extractor at EventAmBeNeutronCandidatesData directly
#    (AmBe2.0v4_gated and _fprompt share runs and build_gate_sets would union them,
#    +18.6% on run 6062).
bash run_ccinc_v3_ambe6266.sh --campaign v4neutron all
python ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4neutron

# 5. The special runs (§6.7) — separate results, never merged. ~3 min.
bash run_ccinc_v3_ambe6266.sh --campaign v4special all
python ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4special

# 6. The traditional box cuts, whole v4 campaign (§6.8) — ~4 min, no ROOT reads.
#    The fitter's --mode has NO default: fpcompare is the original baseline-vs-fprompt
#    study, tagset fits an arbitrary campaign. Reserved out-labels (baseline,
#    fprompt2d, recovered) are refused so the frozen anchor cannot be overwritten.
#    Run the regression FIRST — it must reproduce CaptureTimeFits_baseline.csv
#    byte-identically and return 30.529 +- 0.260.
python -u fit_capture_time_fprompt_compare.py --mode tagset \
  --tags AmBe2.0v4_gated --out-label regress20
python -u fit_capture_time_fprompt_compare.py --mode tagset \
  --tags AmBe2.0v4_gated,AmBe2.0v4_ext --out-label v4_all28
MPLBACKEND=Agg python -u boxcut_v4_campaign.py --do all

# 7. Box cuts on the special runs (§6.9) — ~2 min. Reads BeamCluster directly, so
#    this one does hit dCache. No efficiency and no tau is produced, by design.
MPLBACKEND=Agg python -u boxcut_v4_special.py --do all

# 8. The MVA-vs-box-cut validation and the June reproduction (§6.10).
python -u ccinc_v3_ambe_closure.py --do all     --dataset june               # ~20 min
python -u ccinc_v3_ambe_closure.py --do closure --dataset ambepipe_v4neutron # ~25 min
MPLBACKEND=Agg python -u boxcut_v4_campaign.py --do validation
MPLBACKEND=Agg python -u boxcut_v4_campaign.py --do matched     # §6.10.3, identical clusters
MPLBACKEND=Agg python -u boxcut_v4_campaign.py --do agreement   # §6.10.4, all streamlines/models/points

# 9. Collect every talk figure into ONE folder (PDF + PNG + MANIFEST.csv with the
#    absolute source path of each). Reads figure names from FIGURES.md, so that file
#    stays the single source of truth. --do check reports without writing.
python -u collect_presentation_plots.py --do build
```

**Two output trees, one letter apart.** `slide_plots_ccinc_v3_merged/` lives under
`AmBeNeutron**s**Analysis` (the repo); `ambe plots heatmap`/`combined` write under
`AmBeNeutronsAnalysis/ambe_output/ambe_data/<run_name>/plots/` (**no** `s`). Both exist.
Step 9 above is the only thing that should be copied off the node.

**Environment traps.** `matplotlib.use("Agg")` is forced in `mva_analysis.py` — with
`DISPLAY` set it picked TkAgg and a training died in Tk teardown with exit 134 after
writing its scores but before its importance CSV. `PYTHONUNBUFFERED=1` matters because
stdout into a `tee` pipe block-buffers and a silent multi-hour stage is
indistinguishable from a hang.

**Memory.** `analyze_optics_beamcluster_data.load_file` now reads in 5,000-event batches
(`LOAD_BATCH_EVENTS`). Reading a v4 BeamCluster file whole — 75,625 events for 6266 — is
an out-of-memory kill on an 11 GB node. Same events, same order, same contents.

**Two checks the classify mode prints and you should read.** `mean n_hits /
cf_clusterHits` must sit near 0.87 — 1.0 means ClusterFinder's raw membership leaked in
and every score is biased high. And the Stage-2 candidate and event counts must land
within a few percent of the pipeline's own (`PIPELINE_REFERENCE`); they are a slight
under-count by design, −4.6% on both (§6.6.1).

**Artifact traps that are still live.** Every merged `.pkl` stores the pre-rename NN
filename, so an unstaged `score_data()` silently returns three scores instead of four.
Scored output names derive from the *input* name, so scoring configurations in place
clobbers across them. `ambe_all__data_features_clusterfinder.parquet` (without
`_forscore`) is **empty** — using it scores nothing and reports success.
