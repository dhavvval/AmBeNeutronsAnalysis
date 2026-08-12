# CCinc v3 — World-Volume Background & Merged Tank+World Training
## productionv3 world/fmvmrd · Stages 0 → 2 · merged with tank/fmvmrd

**Runs:** `cc_neutrino_v3world_truthtag`, `cc_neutrino_v3world_recotag` (+ merged trainings)
**Date produced:** 2026-08-04 / 05
**MC input:** `/pnfs/annie/persistent/users/dajana/output/genie_wcsim_world/productionv3/world/fmvmrd/ANNIEEvent_cc_neutrino_world_*.root`
**Companion report:** `REPORT_ccinc_v3_streamlines.md` (the tank-only two-streamline study)

Motivation: the tank-only study left the MVA background-starved — 8,392 background
clusters against 48,934 signal, so 1:1 balancing threw away 83% of the signal and
capped training at 8,392 per class. The world sample simulates the volume *around*
the detector, so it supplies the one background class a tank-only simulation cannot
contain: neutrino interactions in the dirt / concrete / MRD steel whose final-state
particles propagate into the tank and deposit light there.

---

## 0.0 How the particle lineage is tracked

Every composition number in this report and in the companion tank report rests on a
per-hit ancestry walk, so it is worth stating exactly what is being traced before any
percentage is quoted.

**What is stored.** For each PMT hit, WCSim's `DirectParentID` chain is walked back and
the result is written per hit into `<run>__pulses.parquet`:

| column | meaning |
|---|---|
| `ancestor_pdg` | the direct parent of the Cherenkov photon — nearly always `e±` or `γ` |
| `lineage_chain` | the literal chain as a string, nearest ancestor first (`e- <- gamma <- pi0`) |
| `lineage_depth` | how many generations were walked |
| **`origin_pdg`** | **first ancestor that is neither `e±` nor `γ`** — the physics axis |
| `root_pdg` | the GENIE primary at the top of the chain |
| `lineage_status` | 1 = complete chain; anything else = untraced |
| `is_gamma_mediated`, `is_carrier_mediated` | whether the light reached the PMT via a gamma or a charged carrier |

**The walk.** `e±` and `γ` are treated as transparent: the walk skips whole EM
generations and stops at the first hadron or muon. That ancestor is `origin_pdg`. This
is what makes the composition tables physically meaningful — without it, ~40% of all
hits would be labelled "photon" because a neutron capture emits gammas, which says
nothing about what produced the neutron.

**Three rules that are easy to get wrong** (all three have bitten this analysis before):

1. **Join, never zip.** The lineage arrays are *not* row-aligned with the `hit*` arrays
   — they run about 5% short in both the tank and the world sample. The per-hit merge
   joins on `DirectParent_PMTID` + `HitTime`. Row-order zipping silently mislabels hits.
2. **`origin_pdg == -5` means pure EM, not untraced.** It marks a chain with no non-EM
   ancestor at all, i.e. a `γ` is the GENIE primary. Untraced is `lineage_status != 1`,
   which runs at ~11% of delayed hits. Conflating the two would move 14% of the tank
   background (43% of the world background) into the wrong bin.
3. **Never histogram `ImmediateAncestorClass` on its own.** It never says "neutron", and
   it reads "photon" for ~40% of hits because of capture gammas.

**How the axes are used.** Three axes are reported and they answer different questions:
`truth_class` (is this hit neutron light, non-neutron physics, or dark noise),
`origin` (which particle ultimately made it — the physics axis, and the one quoted
throughout), and `root` (which generator primary it came from, used as a cross-check).
At cluster level the same walk produces `n_origin_<species>` counts, whose denominator
is *background hits with a complete non-EM chain* — that is the denominator behind
every per-particle percentage in §4.2.

The runtime prerequisite is easy to forget and fails silently:
`/exp/annie/app/users/dajana/WCSim/WCSim` must be prepended to `LD_LIBRARY_PATH`
before running `Analyse`, or the stock `libWCSimRoot.so` loads without
`DirectParentID` and ~92% of hits come back untraced on *any* input.

---

## 0. Design — how signal and background are defined

The discriminator is **where the neutrino interacted**, not where the light landed.

| population | cuts applied | label |
|---|---|---|
| world, interaction **outside** the tank | **none** — no FV, no CC cuts, harvested unconditionally | **background** |
| world, interaction **inside** tank **and inside FV** | full CC-inclusive selection | neutron-dominated → **signal**; other → **background** |
| world, interaction inside tank but **outside FV** | — | **dropped** (neither class) |
| tank (99.9% in-tank, FV-selected) | full CC-inclusive selection | neutron-dominated → **signal**; other → **background** |

Three points that matter for interpretation:

1. **Out-of-tank light is background even when it is a genuine neutron capture.** A
   neutron born in the dirt that drifts in and captures on hydrogen produces exactly
   the signal-like light — but no in-tank neutrino made it, so it is background. This
   is the population the tank-only training had zero examples of.
2. **No CC-inclusive cuts are applied to the out-of-tank harvest.** Those cuts exist
   to select a muon-neutrino *signal* sample; gating background on them would discard
   most of the contamination and make the background a function of the signal
   selection. Measured: 24,629 of the 27,346 harvested out-of-tank clusters fail
   `cc_pass` — gating would have lost 90% of the new background.
3. **In-tank-but-outside-FV events are dropped, not forced into a class.** Calling an
   in-tank capture "background" would teach the model the opposite of the truth;
   calling a non-FV event "signal" would break the phase-space match with the
   FV-selected tank signal. 2,325 clusters, 7% of the gated world sample.
   (`--no-fv-signal` keeps them as signal.)

**The FV cut is a muon phase-space cut and plays no part in the label.** It is applied
to events we treat as selected signal, never to the background harvest.

### The two streamlines stay separate

The truth-tag and reco-tag selections are **never mixed**. Each is merged only with its
own world counterpart, and each produces its own model per clustering method:

```
tank truthtag + world truthtag  ->  model (optics), model (clusterfinder)
tank recotag  + world recotag   ->  model (optics), model (clusterfinder)
```

Four merged trainings, no cross-streamline combination. The world *background* happens
to be numerically identical in both merges (27,346 clusters) because the out-of-tank
harvest ignores `cc_pass`, and `residual.cc_only: false` makes the two world runs'
cluster tables identical — they differ only in `cc_pass`. Only the world *signal*
contribution differs between streamlines.

---

## 1. Sample and processing

| | value |
|---|---|
| files | **500**, `ANNIEEvent_cc_neutrino_world_<N>.root`, N = 0…499, no gaps |
| bad / empty files | **none** (contrast: the tank sample has one empty file) |
| events | **343,027** (min 614, max 761 per file, mean 686) |
| total size | 1.07 GB (tank: 17.2 GB for 2.49 M events) |
| mean `nhits` | 40 (tank: 90) |
| **interactions inside tank** | **74,022 (21.6%)** |
| **interactions outside tank** | **269,005 (78.4%)** |
| clusters extracted | **144,902** (optics 69,440, ClusterFinder 75,462) |

Runtime: Stage 0 ≈ 35 min, Stage 1 ≈ 100 min per stream. The second stream reused the
first's hits (`REUSE_HITS`), since per-hit content is stream-independent.

### 1.1 The vertex trap — essential for any future world work

**`trueVtx{X,Y,Z}` is NOT the interaction vertex in world files.** It is the WCSim
*primary-particle start point*:

- a **sentinel** `(0, 14.4602, −168.100)` for ~67% of world events (co-occurring with
  `trueMuonEnergy == −9999`, i.e. no primary muon)
- the **tank-entry point** on the r = 152.4 cm wall for most of the rest

The real interaction vertex is `trueNuIntxVtx_{X,Y,Z}`, in GENIE coordinates:

```
x_tank = trueNuIntxVtx_X
y_tank = trueNuIntxVtx_Y + 14.466      # cm
z_tank = trueNuIntxVtx_Z - 168.100     # cm
```

ANNIE's symmetry axis is **y**; r = hypot(x, z); units cm. Validated on the full
sample: on in-tank events with a valid primary, the corrected GENIE vertex reproduces
`trueVtx` with **median deviation 0.001 cm, p99 = 0.27 cm**. Only 110 events (0.218%)
exceed 1 cm and their median r is 152.1 cm — they sit *on* the tank wall, where the two
definitions legitimately differ. A coordinate-convention change would shift ~100% of
events, so this tail is not one.

Implemented as `_augment_origin_columns()` in `src/ambe/mc/cc_selection.py`, which emits
`_nuvtx_{x,y,z}_tank`, `_nuvtx_r`, `origin_in_tank`, `origin_in_fv`, and validates the
offset at runtime with a loud warning above a 2% failure rate.

### 1.2 Which world production is sound

`WORLD_v3_REPORT.md` records that world GENIE truth is unusable. **That applies only to
the 7/30 hand-built `EB_BC_TA/ANNIEEvent_bt_v3world.root`, not to this 7/31 grid
production.** Both have 656 events in file 0, which makes them easy to confuse.

| ntuple | in-tank frac | \|trueVtx − corrected GENIE vtx\| | corr(muE, fslE) |
|---|---|---|---|
| `bt_v3world.root` (7/30) | 0.8% (5/656) | n/a (1 usable event) | n/a |
| `world/fmvmrd/world_0.root` (7/31) | 23.5% (154/656) | 0.14 / 0.055 / 0.19 cm | **1.000000** |

The 0.8% in-tank fraction on the old file *is* the symptom: with GENIE matched to the
wrong events the vertices are effectively random, so almost nothing lands in the tank.

---

## 2. World cut-flows (343,027 events)

| cut | truthtag | % prev | recotag | % prev |
|---|---:|---:|---:|---:|
| Total | 343,027 | — | 343,027 | — |
| p_mu ∈ [600,1200) MeV/c | 92,086 | 26.8% | 92,086 | 26.8% |
| cos θ > 0.8 | 56,794 | 61.7% | 56,794 | 61.7% |
| trueCC == 1 | 43,358 | 76.3% | — | — |
| FSL is muon | 41,716 | 96.2% | — | — |
| NoVeto == 1 | — | — | 56,089 | 98.8% |
| MRD-tagged | — | — | 4,833 | **8.6%** |
| promptPE ∈ [500,3000) | — | — | 2,084 | 43.1% |
| nhits ≥ 4 | 22,904 | **54.9%** | 2,084 | 100.0% |
| **final `cc_pass`** | **22,904 (6.7%)** | | **2,084 (0.6%)** | |

No fiducial-volume cut — see §0. Two behaviours differ sharply from the tank sample and
are worth a slide:

- **MRD tagging keeps only 8.6%** (tank: 61.4%). Most world events interact outside the
  tank, so few produce an MRD-coincident track.
- **`nhits ≥ 4` is NOT a no-op (54.9%)**, whereas it is exactly 100.0% in every tank
  configuration. World events deposit far less light (mean `nhits` 40 vs 90).

---

## 3. What the new background is made of

### 3.1 Out-of-tank delayed background — the population the merge adds

Delayed residual (t > 10 µs), non-neutron hits, **restricted to out-of-tank
interactions**. 71,894 hits.

| origin (first non-EM ancestor) | world out-of-tank | tank (truthtag) |
|---|---:|---:|
| μ⁻ | **44.31%** | 50.65% |
| **pure EM (γ is the primary)** | **43.50%** | 13.82% |
| π⁺ | 4.54% | 17.27% |
| p | 3.65% | 5.39% |
| π⁻ | 2.57% | 2.10% |
| μ⁺ | 0.75% | 3.24% |
| **π⁰** | **0.43%** | 7.46% |
| K⁻ | 0.19% | ~0 |

**Physics: charged and neutral pions are absorbed in the dirt and concrete before
reaching the tank.** What survives the trip is penetrating muons and gammas — so
pure-EM triples (13.8% → 43.5%) while π⁰ nearly vanishes (7.5% → 0.43%). This
independently reproduces the particle-entering-tank census in `WORLD_v3_REPORT.md`
(γ 39.7%, n 30.6%, p 13.9%, μ⁻ 10.3%).

### 3.2 Composition within the *selected* world events (Stage 1b)

Different population from §3.1 — Stage 1b's scope is `cc_pass` events, which for the
world are predominantly in-tank. Quoted for completeness, **not** the background the
merge uses.

| origin | truthtag (22,904 evts) | recotag (2,084 evts) |
|---|---:|---:|
| μ⁻ | 66.91% (0.916/evt) | 50.10% (1.031/evt) |
| pure EM | 14.78% (0.202) | 14.53% (0.299) |
| π⁺ | 8.86% (0.121) | 16.86% (0.347) |
| π⁰ | 3.59% (0.049) | 5.34% (0.110) |
| p | 3.00% (0.041) | 5.32% (0.109) |
| μ⁺ | 1.73% (0.024) | 4.48% (0.092) |
| π⁻ | 1.03% (0.014) | 3.27% (0.067) |

Cluster classes, OPTICS, same scope:

| dominant class | truthtag | recotag |
|---|---:|---:|
| secondary n ← n | 36.50% | 40.91% |
| primary neutron | 32.15% | 38.26% |
| non-neutron physics | 23.07% | 12.72% |
| secondary n ← p | 3.88% | 5.72% |
| secondary n ← other | 3.54% | 1.88% |
| dark noise | 0.86% | 0.51% |

---

## 4. The merged training set

Truth-tag streamline, OPTICS (ClusterFinder is equivalent within a few %):

| source | population | clusters |
|---|---|---:|
| tank | signal (neutron-dominated, in FV) | 48,934 |
| tank | background | 8,392 |
| world | out-of-tank background | **27,346** |
| world | — of which **neutron-dominated** | **22,949** |
| world | in-tank + in-FV signal | 1,441 |
| world | in-tank + in-FV background | 270 |
| world | in-tank, outside FV → **dropped** | 2,325 |

| | tank only | **tank + world** |
|---|---:|---:|
| signal | 48,934 | **50,375** |
| background | **8,392 ← limiting** | **36,008** |
| after 1:1 cap, per class | **8,392** | **36,008** |
| features | 31 | 31 |

**Background grows 4.3×, and it is still the limiting side** — so the balanced training
set grows 4.3× as well. (Signal does *not* become limiting: 50,375 > 36,008.)

**The character of the background changes.** 22,949 of the 27,346 added clusters are
neutron-dominated, so ~64% of the merged background is genuine neutron-capture light.
Before the merge, background was almost entirely non-neutron physics (muon and pion
light). The model is therefore now largely being asked to separate **in-tank captures
from out-of-tank captures** — a position/topology question rather than a light-shape
question, which shifts the discriminating power toward `d_wall` and `vtx_y`.

### 4.1 Audit — all checks pass on the real merged frame

| check | result |
|---|---|
| out-of-tank harvest unconditional | **PASS** — 24,629 of 27,346 kept clusters have `cc_pass == 0` |
| in-tank rows all satisfy `cc_pass` | **PASS** — 0 violations |
| world contributes reclassified captures | **PASS** — 22,949 |
| composite split key `(_source_run, eventID)` | **PASS** — 2 events would have been wrongly merged by a bare `eventID` |
| tank baseline unchanged by the new code | **PASS** — 0.602 / 0.605 / 0.584 / 0.599 reproduced exactly |

Reproduce with `python audit_merged_labels.py --config configs/cc_neutrino_v3_truthtag.yaml
--merge-run cc_neutrino_v3world_truthtag --method optics`.

---

### 4.2 What the training background is made of, per particle

§3 describes the delayed background at *hit* level over whole samples. This section is
narrower and is the one to quote next to the signal-vs-background feature plots: it is
the composition of the **background class actually used for training**, i.e. the
background-labelled clusters of the merged frame.

Two numbers have to be quoted together, because "% of background per particle" alone is
misleading here.

**First: how much of the background is neutron light at all.** Averaged over the hits in
background-labelled clusters (truth-tag / OPTICS):

| | clusters | neutron-dominated | neutron light | non-neutron light | dark noise / other |
|---|---:|---:|---:|---:|---:|
| tank component | 8,392 | 0 | 24.7% | 65.2% | 10.2% |
| world out-of-tank | 27,616 | 22,949 | **70.9%** | 15.1% | 14.0% |
| **merged background** | **36,008** | **22,949 (64%)** | **59.5%** | **27.4%** | 13.1% |

So ~60% of the light in the merged background class is *genuine neutron-capture light*.
It is background only because the neutrino interacted outside the tank. The tank
component behaves the other way round — a quarter neutron light, two thirds muon and
pion light.

**Second: what the non-neutron part of it is.** Percentages of traced non-neutron
background hits, on the `origin` axis (truth-tag / OPTICS):

| origin | merged | tank component | world out-of-tank |
|---|---:|---:|---:|
| μ⁻ | **53.3%** | 43.2% | **75.1%** |
| π⁺ | **25.5%** | 32.6% | 10.2% |
| π⁰ | 7.2% | 9.8% | 1.4% |
| p | 5.8% | 5.4% | 6.7% |
| π⁻ | 4.6% | 4.5% | 4.8% |
| μ⁺ | 3.4% | 4.3% | 1.4% |
| K± | 0.2% | 0.1% | 0.4% |

Reading of the table: **muons make just over half of the non-neutron background and
pions just over a third**, and the split moves strongly with where the interaction
happened. In the world component muons rise to 75% and pions collapse to 16% combined —
the same dirt-absorption effect as §3.1, now measured on the training clusters rather
than on all hits. The reco-tag streamline shifts the merged column to μ⁻ 62.7% /
π⁺ 20.4% / π⁰ 2.3%, because its tank component is 2.6× smaller and the world component
therefore dominates the average more.

ClusterFinder agrees within ~1 point on every row. Full four-configuration table:
`slide_plots_ccinc_v3_merged/table_merged_background_composition.csv`.

**Denominator discipline.** These are `n_origin_<species>` counts: background
(non-neutron) hits with a complete non-EM chain, pure-EM chains excluded from both
numerator and denominator. `frac_bg_*_of_bg` is *not* used anywhere — it can exceed 1
(up to 25.0) because `n_bg_photon` counts capture gammas.

### 4.3 Which features separate the two classes

Computed on the merged frame with the §0 label, as pooled-σ separation of the class
means. Best six per configuration:

| rank | truth-tag / OPTICS | truth-tag / CF | reco-tag / OPTICS | reco-tag / CF |
|---|---|---|---|---|
| 1 | n_hits_early 0.379 | n_hits 0.470 | n_hits_early 0.415 | n_hits 0.536 |
| 2 | n_fit_hits 0.363 | charge_bal_legacy 0.466 | n_fit_hits 0.390 | n_hits_early 0.517 |
| 3 | n_hits 0.317 | beta1 0.462 | n_hits 0.354 | charge_bal_legacy 0.514 |
| 4 | charge_bal_legacy 0.249 | n_hits_early 0.458 | charge_bal_legacy 0.279 | beta1 0.506 |
| 5 | beta1 0.206 | n_fit_hits 0.434 | beta1 0.227 | n_fit_hits 0.477 |
| 6 | fit_rms_ns 0.196 | pe_total 0.351 | fit_converged 0.208 | pe_total 0.425 |

**The tank-only study's central complaint is answered.** There, the best single feature
reached 0.147σ (`fit_goodness_init`) and the conclusion was that the feature set does not
discriminate at single-cluster level. On the merged frame the best feature reaches
0.379σ under OPTICS and 0.470σ under ClusterFinder — a factor 2.5 to 3.2 — and the
leaders are hit multiplicity, charge balance and isotropy.

**And it is not geometry.** This was the explicit worry recorded as open item 4 of the
handoff: the merged label separates in-tank from out-of-tank interactions, so the model
could be learning that out-of-tank light sits near the wall rather than learning anything
about neutrons. Measured ranks out of 31 features:

| feature | tt / OPTICS | tt / CF | rt / OPTICS | rt / CF |
|---|---|---|---|---|
| `d_wall` | 20th (0.038σ) | 15th (0.080σ) | 11th (0.134σ) | 13th (0.138σ) |
| `d_wall_fit` | 12th (0.080σ) | 19th (0.056σ) | 10th (0.151σ) | 16th (0.095σ) |
| `vtx_y` | 29th (0.010σ) | 29th (0.008σ) | 21st (0.037σ) | 27th (0.013σ) |
| `vtx_fit_y` | 16th (0.064σ) | 20th (0.051σ) | 14th (0.107σ) | 17th (0.095σ) |

No geometry feature is in the top nine of any configuration. The trained importances say
the same thing: on truth-tag / OPTICS the four geometry features carry **11.8% of the
random-forest and 9.6% of the GBT importance combined**, with `d_wall` and `vtx_y` ranked
7th and 8th; the forest's top feature is `sigma_t_mad` at only 6.1%, and GBT concentrates
on `n_hits_early` (26.7%), `sigma_t_mad` (14.2%) and `pe_total` (10.4%) — multiplicity,
timing and charge shape.

Physically this is what should be expected: an out-of-tank interaction deposits light
along a longer, less localised path through the tank, and the tank component of the
background is muon light. Both make background clusters systematically busier than a
localised in-tank capture. **The merged gain is a light-shape and multiplicity effect,
not a wall-proximity artifact.**

Separation tables for all four configurations:
`slide_plots_ccinc_v3_merged/table_merged_feature_separation.csv`.

## 5. Merged MVA results

Tank-only baseline → merged, test-set AUC. **All four merged trainings finished on
2026-08-05** (truth-tag/OPTICS 11:05, truth-tag/CF 11:24, reco-tag/OPTICS 11:42,
reco-tag/CF 11:58). Regenerate the table with `python make_plots_ccinc_v3_merged.py`,
which parses the AUCs straight out of `logs_merged_tankworld.log` — they are not written
to any CSV.

| | truthtag OPTICS | truthtag CF | recotag OPTICS | recotag CF |
|---|---:|---:|---:|---:|
| AUC RF | 0.602 → **0.656** | 0.591 → **0.665** | 0.578 → **0.684** | 0.624 → **0.688** |
| AUC GBT | 0.605 → **0.661** | 0.585 → **0.672** | 0.574 → **0.689** | 0.612 → **0.694** |
| AUC XGBoost | 0.584 → **0.662** | 0.583 → **0.669** | 0.559 → **0.688** | 0.599 → **0.690** |
| AUC NN | 0.599 → **0.649** | 0.593 → **0.667** | 0.571 → **0.664** | 0.608 → **0.687** |
| background per class | 8,392 → **36,008** | 8,252 → **39,133** | 3,178 → **30,633** | 3,077 → **33,798** |

Gains, all four configurations:

| | RF | GBT | XGBoost | NN | mean |
|---|---:|---:|---:|---:|---:|
| truth-tag / OPTICS | +0.054 | +0.056 | +0.078 | +0.050 | **+0.060** |
| truth-tag / ClusterFinder | +0.074 | +0.087 | +0.086 | +0.074 | **+0.080** |
| reco-tag / OPTICS | +0.106 | +0.115 | +0.129 | +0.093 | **+0.111** |
| reco-tag / ClusterFinder | +0.064 | +0.082 | +0.091 | +0.079 | **+0.079** |

Mean gain over all 16 matched model/configuration pairs: **+0.082**. The best merged
model is **reco-tag / ClusterFinder GBT at 0.694** (CV 0.6908 ± 0.0031).

**Two orderings worth naming.** *GBT and XGBoost beat the random forest in every merged
configuration*, which was not true tank-only (RF led three of four there) — so the
best-model choice for a plot or a cut is now GBT or XGBoost, not RF. And the *reco-tag
streamline gains most*, which follows from where it started: its tank-only training had
only 3,178 background clusters per class, so the world sample multiplies its background
9.6× against the truth-tag stream's 4.3×. Reco-tag/OPTICS was the *worst* of the eight
tank-only trainings (0.578) and reco-tag/CF is now the best merged one. Read that as the
background statistics finally being adequate, not as the reco tagging becoming better
physics — and the reco streamline is still not data-applicable (§6).

The cross-validation spread tightened at the same time. Truth-tag / OPTICS:
CV 0.6523 ± 0.0034 (RF) / 0.6564 ± 0.0037 (GBT) / 0.6536 ± 0.0032 (XGB) against a
tank-only 0.5763 ± 0.0049 / 0.5810 ± 0.0069 / 0.5736 ± 0.0081. Truth-tag / ClusterFinder:
0.6680 ± 0.0044 / 0.6718 ± 0.0046 / 0.6692 ± 0.0054. Reco-tag / OPTICS:
0.6832 ± 0.0048 / 0.6850 ± 0.0042 / 0.6850 ± 0.0045. Reco-tag / ClusterFinder:
0.6851 ± 0.0038 / 0.6908 ± 0.0031 / 0.6883 ± 0.0038. **The gains are 12–25× the
fold-to-fold scatter**, unlike the truth-vs-reco differences in the tank study, which were
pure noise. ClusterFinder gains more than OPTICS in the truth-tag stream, consistent with
its larger feature separations (§4.3) and its larger background.

> **The truth-tag / ClusterFinder training exited 134 (SIGABRT) on its first pass, and has
> since been recovered.** It aborted in matplotlib/Tk teardown *after* computing the AUCs
> and writing the score table, model and plots, so the numbers above were never in doubt —
> but its `__mva_summary__keepprompt__merged__cf.csv` (feature importances) was not
> written. Fixed at the source: `mva_analysis.py` now calls `matplotlib.use("Agg")` before
> importing pyplot, since with `DISPLAY` set it was picking TkAgg.
> **RESOLVED 2026-08-05:** that single training was re-run to completion (exit 0, log
> `logs_merged_tt_cf_rerun.log`), reproducing AUC 0.665 / 0.672 / 0.669 / 0.667 exactly and
> writing the missing CSV. **All four merged configurations now have a complete artifact
> set** — importance CSV, ROC/discriminator PDF, score table, frozen model + NN.
>
> A separate naming hazard was fixed in the same pass. `mva_analysis.py`'s `mode_tag` does
> **not** encode `--method`, so every method writes to the same
> `__keepprompt__merged.*` filenames and the second training silently overwrites the
> first — the trap `run_ccinc_xi02_compcut.sh` documents and guards against. The first
> merged run lost its OPTICS artifacts this way (AUCs survived only because they are in the
> log). `run_world_merged.sh` now renames each training's outputs to `__merged__{optics,cf}`
> immediately after it finishes. **Any merged artifact without a method suffix is from the
> clobbered first run — do not use it.**

The NN training curves are worth one look before quoting the NN number (figure
`V3MERGED__nn_loss_curves.pdf`): the merged network's **validation loss turns upward after
about five epochs while its validation AUC keeps climbing to 0.66**. It is becoming
over-confident rather than worse at ranking, which is exactly why early stopping monitors
`val_auc` and not `val_loss`. Best epoch 28 of 43.

Two caveats on the ClusterFinder columns, once they land: the merged **feature
separations are markedly larger under ClusterFinder** (best 0.470σ vs 0.379σ, see §4.3),
so a larger CF gain would be expected, and CF also has the larger background (39,133 vs
36,008 per class).

### 5.1 The xi=0.02 + composition-cut campaign, as a reference

Added 2026-08-05 at request, read-only: the AUCs are recomputed from that campaign's
stored score tables, which reproduce `REPORT_ccinc_xi02_compcut_FULL.md` §2.2 / §4b.1
exactly for the three tree models.

| | RF | GBT | XGBoost | NN | test set |
|---|---:|---:|---:|---:|---:|
| xi02 + compcut, OPTICS | **0.7545** | 0.7535 | 0.7488 | 0.7495 | 2,728 (1,353 signal) |
| xi02 + compcut, CF (own model) | 0.7093 | **0.7130** | 0.7013 | 0.7116 | 2,494 (1,259 signal) |
| v3 merged, best (reco-tag / CF) | 0.688 | **0.694** | 0.690 | 0.687 | 14,480 |

**Do not read this as "the older campaign discriminates better".** The two are not on one
scale:

- **Different sample and signal definition.** xi02+compcut is the AmBe-oriented
  CC-neutrino sample; v3 merged asks a harder question — separate in-tank captures from
  *out-of-tank* captures, where 64% of the background is genuine capture light.
- **The composition cut is applied** in xi02+compcut, which removes 13.4% of clusters,
  concentrated in the background-dominated species (46% of π⁺-origin, 61% of π⁻-origin).
  Removing the hardest background before training raises AUC by construction.
- **Its test set is 5.3× smaller** (2,728 vs 14,480 clusters), so its AUC carries a
  correspondingly wider error and its CV spread is ±0.009–0.013 against v3's ±0.003–0.005.

The one thing that *is* directly comparable is the ordering of models within a campaign,
and it differs: xi02 auto-picked RF (all four within 0.006, a tie), while every merged v3
configuration prefers GBT or XGBoost by a resolvable margin.

The NN entries above come from the stored score tables and read 0.7495 / 0.7116; that
report quotes the **seeded** retrain, 0.7526 / 0.7156. The score tables predate the
seeding. A 0.003 difference, immaterial here — but do not "correct" one against the other,
they are different runs.

### Not the deliverable — world-only trainings

`run_ccinc_xi02_compcut.sh` with `STOP_STAGE=2` also trains a **world-only** model
(world signal vs world background) as a byproduct. Those artifacts exist in the world
run directories but must **not** be quoted: they train on the world sample alone, which
is not the combined-statistics goal, and they were produced before a signal-definition
fix (see §6). Recorded here only so nobody mistakes them for the merged result.

---

## 6. Caveats and known issues

- **The FV cut acts as a near-total NC veto in the tank sample** — see
  `REPORT_ccinc_v3_streamlines.md` §1.2. NC events have no primary muon, so `trueVtx` is
  a sentinel at r = 168.1 cm and fails the 100 cm FV for a bookkeeping reason. Only
  1,159 of 758,396 NC events pass. This means the reco streamline's apparent NC purity
  is inherited from a truth cut, not earned by detector tagging, and will not carry over
  to data.
- **A bug was found and fixed on 2026-08-05** in the FV-drop logic: it was applied to the
  background side only, leaving in-tank/outside-FV neutron clusters in the *signal*
  class (world signal read 3,411 instead of 1,441). Any merged artifact produced before
  09:53 on 2026-08-05, and all world-only Stage-2 artifacts, use the buggy definition.
  The corrected merged run is the one reported in §5.
- **World fitted-vertex features are sparse.** The world-only trainings fell back to 21
  of 31 features because the Gauss-Newton fitted-vertex block dropped below 50% non-NaN
  coverage. The *merged* training keeps all 31 (the tank rows carry the coverage), but the
  world rows are contributing NaN-imputed values for that block — a caveat on how much
  the fitted-vertex features can be trusted on the world population.
- **~0.2% in/out ambiguity is irreducible.** Interactions on the water/steel boundary
  (r ≈ 152.4 cm) cannot be cleanly assigned from the stored branches.
- **`nhits ≥ 4` is a real cut here (54.9%)**, unlike in the tank sample. Any tank/world
  comparison of absolute efficiency must account for it.
- The reco streamline's FV and muon-kinematic cuts still use truth branches, so neither
  streamline is yet data-applicable — unchanged from the tank study.
- **The geometry-artifact worry (handoff open item 4) is resolved, not open** — see §4.3.
  Geometry features rank 10th at best and carry ~10% of the importance. What remains open
  is the *other* half of that item: because 64% of the merged background is genuine
  out-of-tank capture light, a model trained this way learns "did an in-tank neutrino
  make this" and not "is this a neutron capture". That is the intended target for a
  CC-inclusive selection, but it is not a neutron tagger in the AmBe sense.
- **`mva_analysis.py` now writes `mva_label` into the score tables** (added 2026-08-05).
  Score tables produced before that — including the truth-tag / OPTICS merged one — lack
  it, and re-deriving the label from `dominant_class` alone gives the **wrong** answer on
  a merged frame (out-of-tank captures are neutron-dominated but labelled background); it
  reads AUC ≈ 0.54 instead of 0.66. Re-derive as
  `where(origin_in_tank == 0, 0, dominant_class in {1,2,3,4})`.

---

## 7. Reproducing everything

```bash
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis

# world processing, both streams (Stages 0/1/1b/2)
bash /exp/annie/data/users/dajana/ccinc_v3_streamlines/run_world_full.sh

# audit + the four merged trainings (the deliverable)
bash /exp/annie/data/users/dajana/ccinc_v3_streamlines/run_world_merged.sh
```

Configs: `configs/cc_neutrino_v3world_{truthtag,recotag}{,_pilot}.yaml`
Logs: `logs_full_v3world_{truthtag,recotag,BOTH}.log`, `logs_merged_tankworld.log`
Merged artifacts are tagged `__merged` so they never overwrite the tank-only baselines.

### 7.1 The figures

```bash
source /exp/annie/app/users/dajana/myboy/bin/activate
python make_plots_ccinc_v3_merged.py
```

Writes `slide_plots_ccinc_v3_merged/` — one standalone PDF **and** PNG (200 dpi) per
figure, plain figures in the same format as `slide_plots_ccinc_xi02/`, to be dropped
straight into a talk. Greyscale only (series separated by fill level, hatching and line
style, never by hue) and Helvetica metrics throughout via Nimbus Sans, mathtext included —
`pdffonts` on any output shows only `NimbusSans-*`. All files are prefixed `V3MERGED__`:

| figure | content |
|---|---|
| `cutflow_streamlines_table` | both cut-flows, tank and world, truth-tag vs reco-tag, as a **table** (§2) |
| `cutflow_nminus1_table` | N−1 cut-flow: what each cut removes on its own (tank only — the CSV exists only there) |
| `compcut_impact_table` | composition cut per origin species: clusters removed and signal lost (§2.1 of the tank report) |
| `streamline_overlap_table` | truth-tag / reco-tag / both / union event counts (§1.1 of the tank report) |
| `bkg_composition_streams` | delayed background by origin particle, tt vs rt (§3) |
| `bkg_composition_tank_vs_world` | tank in-tank vs world out-of-tank background (§3.1) |
| `merged_training_sizes` | 8,392 → 36,008 background growth (§4) |
| `training_background_by_particle` | §4.2 — neutron vs non-neutron light, and the non-neutron part per particle |
| `feature_separation` | §4.3 — separations, merged vs tank-only |
| `feature_importances` | the geometry check, tank vs merged, geometry highlighted |
| `auc_tank_vs_merged` | §5 — AUC per model per configuration, bars |
| `auc_cv_boxplot` | §5 — box plot of the 5 cross-validation folds, tank vs merged, test AUC marked |
| `auc_by_campaign` | §5.1 — the xi=0.02 + composition-cut campaign next to v3, with the non-comparability stated on the figure |
| `nn_loss_curves` | NN loss and AUC per epoch, parsed out of the logs |
| `roc_merged` | ROC per configuration from the `__merged` score tables |

Everything is read live from the run directories and the logs, so **re-running it after
each merged training lands fills in the panels marked "pending"** — no edits needed. Two
cached intermediates (the streamline overlap and the world out-of-tank hit composition,
both expensive) are written as `cache_*.csv` there; delete them to force a recompute. The
two tables this report quotes are dumped alongside:
`table_merged_background_composition.csv` and `table_merged_feature_separation.csv`.

### 7.2 The per-feature signal-vs-background plots

Those come from the **existing** presentable-plots script, not from the one above, so the
format matches `slide_plots_ccinc_xi02/OPTICS_RF__presentable_features.pdf` exactly:
top-6 features by the trained model's own importances, "Neutron" vs "Background", panel
titles `Discriminating Feature: <name>`. It only *reads* the frozen `.pkl` and the score
table — nothing is refitted.

Three flags were added to it for this campaign, none with a silent default:

- `--style {poster,bw}` — `poster` is the original June-16 blue/DejaVu look, `bw` is black
  and white with Helvetica metrics (Nimbus Sans, mathtext included; `pdffonts` confirms no
  DejaVu in the output).
- `--rank-model {rf,gbt,xgb}` — whose importances order the features and whose score the
  cut page uses. **Use the best model for the run: GBT or XGBoost on every merged
  configuration** (§5), not RF.
- `--mc-only` — MC feature pages only. Required when `--scored` is absent, so a missing
  AmBe input can never silently drop the multiplicity and capture-time pages. Stage 3 was
  never run for v3, so all four merged runs use it.

`--score-cut auto` (the new default) reads the 80%-signal-efficiency threshold off the
existing score table — the 20th percentile of test-set signal scores, the same definition
the June deck used. `--figures-dir` writes each page as its own PDF + PNG as well as into
the multi-page `*_ALL.pdf`.

```bash
B=/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output
python make_presentable_optics_rf_plots.py \
  --mc-scores $B/cc_neutrino_v3_recotag/parquet/cc_neutrino_v3_recotag__mva_scores__keepprompt__merged__cf.parquet \
  --frozen    $B/cc_neutrino_v3_recotag/parquet/cc_neutrino_v3_recotag__mva_frozen__keepprompt__merged__cf.pkl \
  --style bw --rank-model gbt --mc-only --score-cut auto \
  --title "ClusterFinder + GBT — Top 6 Discriminating Features" \
  --name-prefix RT_CF_GBT__presentable \
  --figures-dir slide_plots_ccinc_v3_merged \
  --out slide_plots_ccinc_v3_merged/RT_CF_GBT__presentable_ALL.pdf
```

Produced for all four merged configurations, each with its best model:

| prefix | configuration | model | 80%-eff cut |
|---|---|---|---:|
| `TT_OPTICS_XGB__presentable_*` | truth-tag / OPTICS | XGBoost (0.662) | 0.435 |
| `TT_CF_GBT__presentable_*` | truth-tag / ClusterFinder | GBT (0.672) | 0.424 |
| `RT_OPTICS_GBT__presentable_*` | reco-tag / OPTICS | GBT (0.689) | 0.430 |
| `RT_CF_GBT__presentable_*` | reco-tag / ClusterFinder | GBT (0.694) | 0.417 |

Each prefix gives four figures: `_features` (top-6 panel), `_features_top2_<feat>` (one
wide page per top-2 feature), `_features_cut` (top-6 after the 80%-efficiency cut), and
`_ALL.pdf` (all pages in one file).

**The signal mask matters here.** `signal_mask()` uses `mva_label` when the score table
has it and otherwise rebuilds it from `origin_in_tank`. On a merged frame,
`dominant_class` alone would call every out-of-tank capture signal and the plots would
show the wrong split (it reads AUC ≈ 0.54 instead of 0.66 — see §6).
