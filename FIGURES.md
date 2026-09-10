# CCinc v3 — figure ordering for a talk

Every figure is a standalone PDF + PNG in `slide_plots_ccinc_v3_merged/`. **Nothing is
renamed** — the reports reference these files by name. This is an ordering sheet, not a
new figure set. Drop `.pdf` / `.png` as needed.

Section numbers refer to `ANALYSIS_ccinc_v3_FULL.md`.

---

## ⚠ THE CAPTURE-TIME FIT WINDOW CHANGED — 2026-08-27

**Every capture-time number in this file, and in every figure it names, is now fitted on
2–67 µs. It used to be 10–67 µs. Numbers from the two windows are NOT comparable.**

Why 2: that is where the data starts. The box cut is `t ≥ 2 µs` and the cosmic veto drops
any event containing an earlier cluster, so no selection admits anything before it.
Fitting from 10 discarded the eight microseconds that contain the thermalisation rise —
the only part of the range where the `(1 − exp(−t/therm))` term carries information.

What moved:

| quantity | 10–67 µs (old) | **2–67 µs (current)** |
|---|---:|---:|
| anchor, 19 positions | 30.53 ± 0.26 | **29.417 ± 0.221** |
| campaign τ, 26 positions (G4/J9) | 30.535 ± 0.228 | **29.477 ± 0.193** |
| campaign therm (J10) | 5.50 ± 0.22, 2 positions failing | **6.52 ± 0.11, all 26 converged** |
| MVA τ, 26 positions (K9/L3) | 30.97 ± 0.26 | **29.33 ± 0.23** |
| matched box → MVA (G13) | 31.509 → 30.521, 2.73σ | **30.444 → 28.898, 4.98σ** |

Two things worth saying out loud. `therm` is now genuinely constrained rather than pinned
near its bounds — the fit finally sees the rise it is meant to describe. And the campaign τ
moves **into** the 25–30 µs band this analysis itself quotes for capture on hydrogen in
water (see "Why the working point is 80 % signal efficiency" below); the old 30.53 sat just
outside it.

**One function deliberately keeps the old window:** `fit_expflat`, the flat-pedestal
diagnostic. It is `A·exp(−t/τ) + B` with no rise term, so fitting it across the rise is
simply the wrong model (χ²/ndof 165 against 1.4). It stays on 10–67 and any number from it
must be quoted as a 10–67 result.

Anything elsewhere in the repo still quoting 30.53 / 30.535 / 31.509 predates this change.

---

---

## A. Sample and selection (§1–2) — 4 figures

| # | figure | says |
|---|---|---|
| A1 | `V3MERGED__cutflow_streamlines_table` | the two selections, cut by cut |
| A2 | `V3MERGED__cutflow_nminus1_table` | which cut is actually doing the work |
| A3 | `V3MERGED__streamline_overlap_table` | the streamlines never disagree about a *label*, only about which events they include |
| A4 | `V3MERGED__merged_training_sizes` | what merging the world sample bought: background 8,392 → 36,008 per class |

## B. Training and classifier choice (§3) — 8 figures

| # | figure | says |
|---|---|---|
| B1 | `V3MERGED__auc_tank_vs_merged` | the +0.082 merged gain. **Say §4.4 out loud here** — it is a change of question, not a better answer |
| B2 | `V3MERGED__auc_cv_boxplot` | the gain is 12–25× the CV scatter; the GBT/XGB gap is not |
| B3 | `V3STATS__roc_by_config` | all four models, all four configurations, one scale |
| B4 | `V3STATS__loss_comparison` | log-loss breaks the GBT/XGB tie, and shows the NN calibrates badly |
| B5 | `V3STATS__model_verdict_table` | **the verdict slide**: GBT on ClusterFinder |
| B6 | `V3STATS__effpurity_scan` | purity 0.590 at 80% efficiency — and that this is a 1:1 test set, not a physical purity |
| B7 | `selection_2x2_benchmark__ambe__gbt__eff` | **the validation B6 cannot give**: efficiency of the box cuts and of the MVA neutron on the *natural* MC cluster population, OPTICS vs ClusterFinder. CF: box **0.930**, MVA @0.424 **0.806** |
| B8 | `selection_2x2_benchmark__ambe__gbt__purity` | the same for purity — CF: box **0.869**, MVA **0.883**, with **5,348** fakes against the box's **7,044** (−24 %). This is the physical purity; B6's is not |

*Optional:* `V3MERGED__auc_by_campaign` if the ξ=0.02 0.75 is going to come up — but only
with the "not comparable" caveat attached (§0).

*What to say at B7/B8, and the distinction that matters:* B6's purity is computed on a
**1:1 balanced** test set, so it is a ranking diagnostic, not a physical purity. B7/B8 are
the natural-population numbers — `benchmark_selection_2x2.py --box ambe --score-col
gbt_score --score-cut 0.423621`, i.e. the *same* box the AmBe data analysis applies and the
*same* frozen threshold deck 3 uses. Two things worth stating:

- The frozen cut 0.423621 returns **efficiency 0.806** on ClusterFinder here, independently
  reproducing the 80 % working point it was derived at. That is a closure test, not a
  coincidence.
- The MVA buys **purity 0.869 → 0.883** and **7,044 → 5,348 fakes (−24 %)** for
  **0.930 → 0.806** efficiency. It is a trade, not a free win — say so.

*The stale table to stop quoting:* `PRESENTATION_DEEP_DIVE_MC_TO_OUTPUTS.md` §Stage 4 lists
purities of 0.977 / 0.996. Those were measured with the **legacy** box (PE<80, CB<0.45,
nHits>9) and an **rf** cut at 0.547 — neither the box the data analysis applies nor the
current deliverable. B7/B8 supersede them. `--box legacy` still reproduces them if anyone
asks where they came from.

*The denominator to state once:* 99,062 of 115,706 MC clusters are neutron-dominated, so
this population is signal-rich by construction and these purities are not transferable to
AmBe data, which has a different background mix. Purity here ranks the two selections
against each other; it is not a prediction for the data sample.

### Why the working point is 80 % signal efficiency

Asked at every talk, so have the answer ready. It is a **physics-closure** argument, not
an optimisation on data. The ablation (`PRESENTATION_DEEP_DIVE_MC_TO_OUTPUTS.md`, Stage 3):

| threshold rule | fitted τ | good fits of 21 positions | single-neutron fraction |
|---|---:|---:|---:|
| **fixed 80 % MC signal efficiency** | **26.7 ± 2.1 µs** | **20 / 21** | 0.969 |
| Youden's J | 34.6 ± 2.5 µs | 10 / 21 | 0.825 |
| max-F1 | — | — | 0.349 |

Youden's-J and max-F1 reach 94–97 % MC efficiency, but the extra accepted clusters are
enough fake background to drag the fitted capture time to 34.6–35.9 µs — outside the
25–30 µs band for neutron capture on hydrogen in water — and to collapse fit quality
(10/21 and 17/21 converging). The fixed 80 % point is the only rule where **both** RF and
NN land in the physically correct band with stable fits at **every** source position. That
is the justification: it is chosen on external physics, so it cannot be accused of being
tuned to make the AmBe answer come out right.

**State the provenance honestly:** that table was measured in the OPTICS/RF era. The
current deliverable is the merged GBT on ClusterFinder, where the corresponding threshold
is 0.423621. B7 shows that cut independently returns **efficiency 0.806** on the natural MC
population, reproducing the 80 % point it was derived at — so the working point transfers,
but the τ-vs-threshold ablation above has **not** been redone for the current model. Say
"this was established for RF on OPTICS and the working point carries over" rather than
implying the numbers are current.

## C. What the background is (§4) — 5 figures

| # | figure | says |
|---|---|---|
| C4 | `V3MERGED__bkg_containment_table__truthtag_cf` | the same budget as numbers, for the backup slide and for anyone who wants to read the percentages off |
| C0 | `V3MERGED__bkg_containment_normalized__truthtag_cf` | the containment view: every component as a share of ALL delayed background light, tank vs outside the tank, out-of-tank neutron capture kept in the normalisation (61.1% of the total) |
| C1 | `V3MERGED__bkg_composition_tank_vs_world` | 59.5% of the background light is real neutron capture, mostly from the world |
| C2 | `V3MERGED__training_background_by_particle` | of the non-neutron remainder: μ⁻ 53%, π⁺ 26%, π⁰ 7%, p 6%, π⁻ 5%, μ⁺ 3% |
| C3 | `V3BG__truthtag_cf__top6_shape` | where each species lives along the leading features — *composition of a bin* |

## D. Background anatomy — the new block (§5) — 11 figures

D6–D8 ask D1/D2's question over ALL the light in the cluster — capture, charged
parent, pure EM, dark noise — so no cluster drops out. D1/D2 ask it on the
charged-parent axis only, where 70% of background clusters have nothing to show.
Run D8 → D6 → D7 first, then D1/D2 as the charged-particle zoom.

| # | figure | says |
|---|---|---|
| D9 | `V3SPUR__light_sources_table` | the all-light census as numbers: per source, share of the light, clusters it dominates, and how pure those are |
| D10 | `V3SPUR__light_split_table` | the same composition on the train half and the test half separately — they agree to 0.3 pp, so the block may be quoted on either |
| D0 | `V3SPUR__light_purity_table` | signal against background on every mixture metric at once — the table that says purity is not the discriminant |
| D8 | `V3SPUR__light_sources_census` | what the background light IS: **neutron capture (background) is 61.1% of the background hits and dominates 76.8% of clusters**, then μ⁻ 11.3%, dark noise 10.9%, pure EM 6.6%, π⁺ 5.5%. Says "hits", not "light": the quantity is a hit count and "61.1% of the light" gets requoted as a PE fraction |
| D6 | `V3SPUR__n_sources_alllight` | a cluster is never one thing: mean 2.33 sources in background, 2.83 in signal |
| D7 | `V3SPUR__purity_alllight` | the leading source holds a median 0.83 of a background cluster and 0.80 of a signal one — composition purity is NOT what separates them |
| D1 | `V3SPUR__n_species` | the charged-particle zoom: where a cluster has charged-parent light, it is usually a mixture |
| D2 | `V3SPUR__purity` | the leading particle holds a median 0.75 of a background cluster's traced light |
| D3 | `V3SPUR__cooccurrence` | what the mixtures are, over **all** the background light and therefore over all 39,133 background clusters rather than the 30% with a traced charged parent: **neutron capture + dark noise in 78.4% of the 25,848 mixed clusters**, then capture + μ⁻ 27.4%, μ⁻ + dark noise 27.1%, μ⁻ + π⁺ 19.8% |
| D11 | `V3SPUR__cooccurrence_matrix` | backup for D3: every source pair at once as a matrix, for the pair that is not in D3's top ten |
| D4 | `V3SPUR__species_census` | **the table to leave up**: per particle — share of background, purity, and GBT AUC against signal (0.552 π⁰ → 0.705 out-of-tank capture) |
| D5 | `TT_CF_GBT__presentable_features_byspecies_logy` | the six leading features with one curve per **source** instead of one grey background. The 27,393-cluster `no non-EM origin` class is split by its dominant light source into **neutron capture 25,724 / γ,e± 1,534 / dark noise 135** — the same partition D8 uses — so the largest background class is no longer one grey curve. All curves solid, separated by colour; the signal band is drawn at α=0.30 so the curves are readable over it. **LOG y**: dark noise is 135 clusters piled into the first bin or two, so on a linear axis its density there sets the y limit and squashes the other nine curves into the bottom fifth of every panel |
| D12 | `TT_CF_GBT__presentable_features_byspecies` | backup: the same page on a **linear** y axis, solid lines — the version to show if the log axis is more than the audience wants |
| D13 | `TT_CF_GBT__presentable_features_byspecies_dashed` | backup: linear y, **dashed** — a dash pattern per class as well as a colour, for print and for the panels where two curves overlap |
| D14 | `TT_CF_GBT__presentable_features_byspecies_logy_dashed` | backup: log y **and** dashed, the two fixes together |
| D15 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise` | backup: linear y, solid, **dark noise left off entirely** — the other way to stop 135 clusters setting the y limit, and the clearest of the eight on a linear axis. Every remaining curve is identical to D12's: each class is area-normalised on its own, so dropping one frees the axis without renormalising the rest |
| D16 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise_logy` | backup: log y, solid, no dark noise |
| D17 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise_dashed` | backup: linear y, dashed, no dark noise |
| D18 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise_logy_dashed` | backup: log y, dashed, no dark noise |

*Zoom slides if there is time:* `TT_CF_GBT__presentable_byspecies_top2_pe_total`,
`TT_CF_GBT__presentable_byspecies_top2_n_hits` — full-width, one feature each.

*The distinction to state once, at D5:* `V3BG__*` (C3) shows what a **bin** is made of;
`*_byspecies` (D5) shows where a **particle** lives. They look similar and answer
different questions.

*The caveat to state once, at D1:* only 17.6% of tank background clusters are
single-particle, so a per-species curve is "clusters this particle dominates", not "pure
π⁺ clusters".

## E. AmBe validation, run 6266 (§6) — 3 figures

| # | figure | says |
|---|---|---|
| E1 | `V3AMBE_AMBE6266__capture_time_selected` | **the closure**: 33.41 ± 3.11 all clusters → 30.40 ± 3.37 at GBT eff80, against the 29.42 ± 0.22 anchor (pull +0.29) |
| E2 | `V3AMBE_AMBE6266__tau_summary` | every model × working point against the anchor band; and that all 12 *gated* fits fail the quality gate on statistics |
| E3 | `V3AMBE_AMBE6266__score_data_mc_overlay` | the honest one: 6266's scores sit closer to MC **background**, not signal — a source-position effect (§6.4) |

*The June reference, if the two campaigns are being compared:* `V3AMBE__score_data_mc_overlay`
and `V3AMBE__tau_summary` — same plots on the pooled v1/v3 sample, where the CF models
sat closer to MC signal.

## F. The new definition on the AmBe pipeline's own neutrons (§6.6–6.7) — 4 figures

| # | figure | says |
|---|---|---|
| F1 | `V3AMBEN__classified_fraction` | **the answer to "what %"**: of 225,945 Stage-2 AmBe neutrons over all 28 source runs, the MVA calls 79.0% neutron and 21.0% other at the MC 80% point (42.7% at eff50, 91.6% at eff90); all four models agree |
| F2 | `V3AMBEN__classified_by_position` | the kept fraction runs **64.2–86.0% across 26 positions** for one nominal 80% threshold — driven by light yield (`n_hits` r = +0.91), not geometry (`d_wall` r = +0.01) |
| F3 | `V3AMBEN__classified_capture_time` | kept vs rejected τ per position against the anchor band. **State the caveat with it:** the rejected fits fail on σ (10–33 µs at χ²/ndof ≈ 1.3), so the background claim rests on the model-free shape test, not on these |
| F4 | `V3AMBESPEC__classified_fraction` | the special runs, **separate and never merged**: 6265/6266/6270 at 79–86%; 6264 gives zero clusters (93.3% cosmic-vetoed); 6254/6256 have zero IC-passing triggers |
| F5 | `V3AMBESPEC__classified_by_position` | the same four runs placed against the campaign's per-position range — backup for F4 |
| F6 | `V3AMBEN__capture_time_selected` | **the v4-native MVA validation**: the frozen definition applied to all 28 AmBe runs, capture time at each working point. Replaces June as the multi-position reference |
| F7 | `V3AMBEN__tau_summary` | every model × working point on the 28-run sample against the anchor band. **State the caveat with it:** the *pooled* fits fail on χ²/ndof (8–12) — see the note below |

*The number to say at F1, because it is what makes the split meaningful:* the MVA keeps
79.0% of what Stage 2 accepts but only **19.6%** of what Stage 2 rejects — a 4.0×
contrast, 11× at eff50. Equal numbers would mean the MVA knew nothing about the AmBe cuts.
Worth adding: the 20-run answer was 79.0/19.7, so a 30% larger sample moved nothing.

*The number to say at F3:* what the MVA discards is **+1.10 ± 0.09 µs later (+12.5σ)**
with a late/early ratio **×1.15** — flatter, i.e. enriched in a non-exponential
component. That is the background claim, and it is model-free.

*The caveat that must travel with F6/F7 — say it, do not let someone find it:* the
**pooled** 28-run fits are unusable, χ²/ndof 8–12 at eff80/eff90. This is **not** the
6266 problem (too few clusters); it is the opposite. With ~181k clusters pooled over 26
positions whose light collection genuinely differs, one thermalisation+exponential no
longer describes the sum, and χ² notices. **Per position it fits fine at both ends** —
that is the real validation, and it lives on G8/G9:

| selection (26 AmBe positions, truth-tag/CF GBT eff80) | positions | weighted τ [µs] | median χ²/ndof |
|---|---:|---|---:|
| box cuts, B fixed at 0 (anchor recipe) | 26 | 29.477 ± 0.193 | 1.51 |
| box cuts, B floating | 26 | 29.501 ± 1.187 | 1.40 |
| MVA input (all Stage-2), B floating | 24 | 30.414 ± 1.371 | 1.36 |
| **MVA neutron @ eff80**, B floating | 23 | **29.835 ± 1.400** | **1.27** |
| MVA other @ eff80, B floating | 1 | no usable fit | 1.31 |

Like-for-like (both B floating): **MVA − box cuts = −0.188 ± 1.754 µs, 0.11σ.** Never
compare the 0.193 and 1.365
errors directly — the anchor recipe fixes B at 0, the MVA closure floats it, and B/τ are
correlated, so floating B inflates the error ~6×. That is a recipe difference, not a
measurement difference.

## G. Traditional box cuts, the full v4 campaign (§6.8) — 7 figures

The neutron definition is the AmBe pipeline's own — IC gate 700–1200 + second-pulse
veto, then `0 < PE ≤ 100`, `0 < CB < 0.45`, `t ≥ 2 µs`, `hits ≥ 5` after the cosmic
veto. This is the QA baseline group F is measured against, so it stands on its own.

**This group splits across two decks**, which is why `collect_presentation_plots.py`
assigns decks per slot and not per group. G1–G7 and G10–G12 are the box cuts alone,
with no MVA and no MC anywhere in them, and go to `deck2_ambe_boxcuts`. G8, G9 and
G13–G16 put the MVA next to the box cuts and go to `deck4_boxcut_vs_mva`.

| # | figure | says |
|---|---|---|
| G1 | `V4BOX__cutflow_table` | the box cuts stage by stage: 474,192 IC-gated triggers → 19.85% cosmic-vetoed → 380,072 AmBe triggers → **217,532 neutron triggers** (45.9% of Stage 1) |
| G2 | `V4BOX__efficiency_by_position` | **the headline**: campaign efficiency **57.23 ± 0.08 %** over 28 runs / 26 positions, range **40.3–71.8 %**; port 3 lowest, port 5 y=0 highest |
| G3 | `V4BOX__consistency_table` | **the QA statement**: the 8 new runs move the campaign efficiency by **−0.07 pp** (57.30 → 57.23). Adding 30% more data changed nothing |
| G4 | `V4BOX__capture_time_by_position` | τ per position, 26 positions, **all 26 fits converge**; weighted **29.477 ± 0.193 µs** |
| G5 | `V4BOX__capture_time_anchor_comparison` | extending 19 → 26 positions moves τ by **+0.005 µs** and shrinks the error 12%. The anchor is stable |
| G6 | `V4BOX__cosmic_fraction` | cosmic veto **19.85 %** campaign-wide, flat to 7 pp across 26 positions — a detector-stability check |
| G7 | `V4BOX__multiplicity_share` | multi-neutron share **19.74 %**, range 18.0–23.7 % |
| G8 | `V4BOX__validation_mva_vs_boxcut_table` | **the validation slide**: the frozen MVA definition and the box cuts on the *same* 26 AmBe positions under *one* fit recipe — agreement **0.11σ** (−0.188 ± 1.754 µs) |
| G9 | `V4BOX__validation_tau_scatter` | the same per position rather than pooled — MVA-selected τ against box-cut τ |
| G10 | `efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28` | **the canonical heatmap**: port × y efficiency with per-cell binomial SE, all 26 positions |
| G11 | `residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28` | the same against the AmBe 1.0 reference map — shows the 40–72 % pattern is not new |
| G12 | `statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28` | exposure per cell, so nobody reads a low-statistics cell as a physics effect |
| G13 | `V4BOX__matched_boxcut_vs_mva_table` | **the strict comparison, and the best result in the AmBe chapter**: on the *same* 225,810 clusters the MVA moves τ **30.444 ± 0.210 → 28.898 ± 0.229** (−1.546 ± 0.311, **4.98σ**) and what it discards sits at **34.746 ± 0.533** |
| G14 | `V4BOX__agreement_capture_time_table` | the same comparison at **every** working point and **both** streamlines: the shift is monotonic (eff50 −1.76 µs / 4.0σ → eff90 −0.56 µs / 1.6σ) and the discarded sample gets later as the cut tightens |
| G15 | `V4BOX__agreement_efficiency_by_position` | **the agreement slide**: MVA-neutron vs box-cut efficiency, per position — **r = 0.993** over 26 positions. The two definitions rank the positions identically |
| G16 | `V4BOX__agreement_kept_fraction` | all four models × three working points × both streamlines on one page — the models agree to a few points everywhere |

*Backup:* `V4BOX__cut_variables` — the four box-cut variables with the cut positions, for
anyone who asks what the box actually is.

*The number to say at G2:* the spread is **not** a detector problem — the same 40–72%
pattern reproduces in AmBe 1.0 and 2.0v2, and the MVA sees the same ordering
independently (F2, `n_hits` r = +0.91). Port 3 is furthest from the PMT-dense region.

*The comparison to make once, between G4 and F3:* box-cut τ **29.477 ± 0.193** vs
MVA-selected τ **29.835 ± 1.400** on the *same* 26 positions. Strictly like-for-like (both
with B floating) that is **29.038 ± 1.101** vs **28.849 ± 1.365** — agreement to 0.11σ.
The box-cut anchor recipe fixes B at 0, which is why its quoted error is ~6× smaller;
say that rather than comparing the two error bars directly.

## H. Traditional box cuts, the special runs (§6.9) — 4 figures

Reported separately and **never merged**. No efficiency and no τ is quoted for any of them.

| # | figure | says |
|---|---|---|
| H1 | `V4BOXSPEC__cutflow_table` | all six runs on one page: 6254/6256 have **zero** IC-passing triggers (0 of 7,932 and 0 of 4,254); 6264/6265/6266/6270 gated at 149 / 2,150 / 2,171 / 2,312 |
| H2 | `V4BOXSPEC__yield_per_trigger` | **6264 gives zero Stage-2 candidates**; the other three sit at 0.64–0.74 candidates per AmBe trigger |
| H3 | `V4BOXSPEC__cosmic_fraction` | why: 6264 cosmic-vetoes at **98.0 %** against 20–25 % for the other three — ~4× higher |
| H4 | `V4BOXSPEC__cluster_shapes` | the four box-cut variables per run, area-normalised. 6265/6266/6270 are indistinguishable from source-in runs |

*State with H1:* the gated-trigger counts reproduce the MVA route (§6.7) **exactly** for
all four runs, which is the cross-check that the two independent gates agree.

*State with H2/H3:* 6264's cosmic-veto fraction is **98.0 %** here against 93.3 % on the
MVA route. Both give zero candidates. The difference is real and expected — the MVA route
re-derives cluster features in a (−5, +20) ns delayed-residual window, so its `clusterPE`
differs and the >100 PE veto flips on a handful of events. Quote 98.0 % for the box-cut
analysis and 93.3 % for the MVA one; do not mix them.

*The 6270 caveat, mandatory:* `BeamCluster_6270.root` is still the corrupted merge —
**11,279 duplicated events of 46,211 (24.4 %)**, confirmed here, matching `FIX_6270.md`.
Deduplicating on `eventTimeTank` is what makes its 2,312 gated triggers agree with §6.7;
an undeduplicated count reads 2,941 (+27%). Never quote 6270 without this.

## J. Deck 2 core — box cuts, from the established pipeline *(new)*

The parallel set. Every figure here has a **K** twin produced by the *same* command on
the *same* runs with only the neutron definition swapped, so decks 2 and 3 can be laid
side by side figure for figure. Produced by
`ambe plots {heatmap,combined,basic} --config configs/data_ambe2v4_all28_box.yaml`
against tag `AmBe2.0v4_all28_box`, which reproduces the published selection: campaign
efficiency **57.25 ± 0.08 %** vs 57.23 ± 0.08, cosmic veto 19.85 % exactly, and 27 of
28 runs identical to the published candidate counts.

| # | figure | says |
|---|---|---|
| J1 | `efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_box` | **the deck-2 headline**: port × y efficiency with per-cell binomial SE, 26 positions, one tag |
| J2 | `statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_box` | exposure per cell, so a low-statistics cell is not read as a physics effect |
| J3 | `residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_box` | against the AmBe 1.0 reference map — the 40–72 % pattern is not new |
| J4 | `neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28_box` | the pooled capture-time fit. **Quote G4's per-position weighted τ, never this** — pooled χ²/ndof is 3.22 |
| J5 | `capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28_box` | residuals of J4, which is where the pooled χ² comes from |
| J6 | `neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28_box` | neutron multiplicity per trigger |
| J7 | `cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28_box` | PE vs charge balance — the 2D the box is drawn in |
| J8 | `delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28_box` | Δt between first and subsequent clusters in multi-neutron events |
| J9 | `V4CAPHEAT__capture_time_heatmap_box` | **the map that was missing**: τ per port × y on the same grid as J1, 26 positions, all converge, weighted **29.48 ± 0.19 µs**, spread 27.62–33.42 — reproduces G4's per-position anchor with lmfit |
| J10 | `V4CAPHEAT__thermal_time_heatmap_box` | the *other* constant the same fit measures: thermalisation time per port × y, weighted **6.52 ± 0.11 µs**, spread 5.20–8.31, all 26 converged. On the 2–67 µs window therm is genuinely constrained — the fit now sees the rise it describes |

*Why J9/J10 exist, since it is the first question asked about them:* the per-position τ
was only ever shown as a 26-point scatter against position index (G4), and `therm` was
never shown at all even though the **same** NeutCapture fit returns it and it has been
sitting in `CaptureTimeFits_v4_all28.csv` all along. J9/J10 put both on the port × y
grid the efficiency uses, so a τ map and an efficiency map can be laid side by side for
the same cells. Produced by `boxcut_v4_campaign.py --do captureheat`, fit backend
**lmfit** (`--fit-backend lmfit`), B fixed at 0, on the **2–67 µs** window.

## K. Deck 3 core — MVA neutron, the same pipeline *(new)*

Twin of J. Same runs, same Stage-1 IC gate (700 < IC_adjusted < 1200 + second-pulse
veto), same cosmic veto, same three commands — the neutron is `gbt_score > 0.423621`
with **no PE / CB / time / hits box**.

| # | figure | says |
|---|---|---|
| K1 | `efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_mva` | **the deck-3 headline**, to be put beside J1. On covered triggers: **45.88 %** against the box's 55.99 % |
| K2 | `statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J2's twin |
| K3 | `residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J3's twin, same AmBe 1.0 reference |
| K4 | `neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J4's twin. Pooled — backup, not the result |
| K5 | `capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J5's twin |
| K6 | `neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J6's twin |
| K7 | `cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J7's twin — **the honest one**: the MVA sample extends past CB = 0.45, which the box forbids |
| K8 | `delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J8's twin |
| K9 | `V4CAPHEAT__capture_time_heatmap_mva` | J9's twin: τ per port × y under the MVA neutron, 26 positions, weighted **29.33 ± 0.23 µs**, spread 25.93–35.74 — reproduces L3's MVA number |
| K10 | `V4CAPHEAT__thermal_time_heatmap_mva` | J10's twin: thermalisation time per port × y, weighted **6.87 ± 0.13 µs**, spread 4.61–9.93 |

*The caveat that must travel with K1, and it is not optional:* the MVA scoring stage saw
**225,810 of the 237,590** clusters Stage 1 finds — a 4.96 % dropout, the documented
Stage-2 closure gap. A cluster with no score is not a rejected cluster. K1 is therefore
computed on **covered triggers** only (every box cluster in the event scored):
370,391 triggers, identical on both sides, giving box **55.99 %** vs MVA **45.88 %**,
**−10.11 pp**. Deck 2's J1 keeps the full sample at 57.25 % because it needs no MVA.
Never compare 57.25 % with 45.88 % — compare 55.99 % with 45.88 %.

## L. Deck 4 — the two definitions differenced *(new)*

Box minus MVA, on the same covered triggers and the same scored clusters. Produced by
`boxcut_v4_campaign.py --do scoredsubset` then `--do residuals`. Nothing here is a
purity: AmBe data has no per-cluster truth label, so an overlap is an agreement. The
purity is B7/B8, from MC.

| # | figure | says |
|---|---|---|
| L1 | `V4RES__residual_efficiency_by_position` | **the deck-4 headline**: both efficiencies and their difference per position. On the same 370,389 covered triggers: box **55.99 %**, MVA **46.15 %**, **−9.84 pp** |
| L2 | `V4RES__residual_efficiency_heatmap_boxvsmva` | the same as a port × y map |
| L3 | `V4RES__residual_capture_time_by_position` | per-position τ both ways: box **30.444 ± 0.210**, MVA **29.329 ± 0.228**, Δ **−1.115 ± 0.310 (3.59σ)** — the MVA lands on the 29.42 anchor, the box sits 3.3σ above it. Per position, never pooled |
| L4 | `V4RES__residual_cluster_overlap_table` | the disagreement, cluster by cluster: **178,404** both call a neutron, **47,406** box-only (the MVA rejects them), **3,922** MVA-only (outside the box, all on CB ≥ 0.45). Agreement, **not** purity |
| L5 | `V4RES__residual_clusters_by_position` | clusters selected per position, both, plus the ratio |
| L6 | `V4RES__residual_multiplicity_by_position` | share of neutron triggers with **more than one accepted neutron**, per position, both plus the residual: box **7.88 %**, MVA **5.92 %**. **NOT** G7's 19.74 % — see the note |

*The closure to state at L3:* the box side reproduces the published G13 numbers exactly
— 225,810 scored clusters and τ = 30.444 ± 0.210 µs. That is what licenses the MVA side
of the same figure.

*The multiplicity definition trap, state it at L6.* Two different quantities are both
called "multiplicity" in this analysis and they differ by 2.5×. **Both count events** —
an earlier version of this note said G7 counts clusters, and that is wrong. What differs
is *which* events they count:

- **G7 / the pipeline's `mult_share` = 19.74 % (box), 17.23 % (MVA)** counts events that
  had more than one cluster **in total before any selection** *and* yielded at least one
  accepted neutron. The source is `Selection.multiple()` in `src/ambe/data/selection.py`,
  `return cn != 1 and self.accepts(...)`, incremented once per event: `cn` is
  `numberOfClusters` off the ntuple, so the *condition* is selection-independent and the
  number barely moves. It is the number in the published tables.
- **L6 = 7.88 % (box) / 5.92 % (MVA)** counts events in which the selection **accepted**
  more than one neutron. This is the one that actually responds to changing the neutron
  definition, which is why deck 4 uses it.

On the full sample rather than the covered pair the second definition gives **7.85 %
(box)** and **5.92 % (MVA)** of neutron triggers — 17,140 and 10,128 events — so the
covered/full distinction moves it by 0.03 pp and nothing rests on it.

Both are correct; they answer different questions. Never put them in the same sentence
without saying which is which.

*Why the MVA takes clusters the box refuses.* Of the four box cuts, three are already
implied upstream and only one bites: `PE > 100` and `t < 2 µs` are impossible after the
cosmic veto (it drops any event containing such a cluster), and `hits < 5` cannot occur
because ClusterFinder itself requires 5. So every one of the 3,922 MVA-only clusters is a
**charge-balance** case, CB ≥ 0.45. Once the cosmic veto is applied, "the box" is
effectively a charge-balance cut — worth saying out loud, because it reframes what the MVA
is actually replacing.

*What is NOT claimed here.* No purity. AmBe data carries no per-cluster truth label, so the
overlap above measures agreement between two selections, and two selections can agree while
both admitting the same background. Efficiency and purity come from MC on B7/B8.

## M. Deck 5 — the special runs as their own analysis *(new)*

The four runs the campaign can only report as zeros, run through the *same* flow the
campaign runs — Stage-1 IC waveform gate, then the tank cluster stage, then the Stage-2
box — but reporting the **distributions at each stage** instead of the survivor count,
and reporting them **ungated** as well as gated. Produced by
`boxcut_v4_special.py --do diag`.

Ungated is not a relaxation of the analysis. For the two dummy triggers it is the only
option that exists: a scheduled readout carries no BGO gamma, so **0 of 7,933** and
**0 of 25,525** waveforms land in the 700–1200 IC window and every gated plot of those two
is blank. What the runs contain is still a measurement.

**These are three different kinds of run and the labels must not blur them.** 6254/6256 are
**dummy (pulser) triggers** — `kind=pulser` in the diagnostics' `step8_cosmic.csv`, "dummy
trigger, LAPPD ON/OFF" in `stepG_charge_summary.csv`. The readout fires on a schedule, not
on anything in the detector, so each event is a *random slice of the tank*: they measure
the **accidental cluster rate**. 6264 is a genuine **no-source run** — the real trigger
logic is live, the source is simply absent — so its 808 in-gate waveforms are **false
starts of the IC trigger**, which a dummy-trigger run can never measure because it never
exercises the trigger logic at all.

**Legends name what a curve IS, not which run it came from.** A legend entry reading
"6254" makes these figures unreadable without the run table in hand, and what the runs
are is the whole point of the deck. The run number stays on every panel title and in
every table.

| legend text | run | kind | the question it answers |
|---|---|---|---|
| dummy trigger (LAPPD on) | 6254 | pulser | what charge does the tank register with nothing to detect, and what is the accidental cluster rate? |
| dummy trigger (LAPPD off) | 6256 | pulser | 6254's matched partner, so any difference between them is instrumental |
| no source | 6264 | real trigger logic, source absent | what does a **false start of the IC trigger** look like, in IC and in the tank? |
| source in (Port 5, y = 0 cm) | 6265 | source in | **the contrast, everywhere in the deck**, and the only one that is apple to apple: 6265 sits at (0,0,0), the same position as 6264 to the centimetre, so a difference between them is the source and not the geometry. Labelled no-source in the run database and behaves source-in (τ = 35.5 µs, 2,150 of 4,124 IC candidates passing). **6266 is no longer drawn anywhere in this deck** — it is source-in at Port 5 but y = 100 cm, so using it put a ~10 pp geometry difference inside every comparison. It keeps its row in H1 for provenance |

**Every 1D figure is emitted twice, normalised and in raw counts, on separate pages.**
Neither answers the other's question: a normalised plot cannot show that the LAPPD-on
dummy trigger contributes 4,404 clusters against the regular run's 146,470, and a counts
plot buries the two dummy triggers entirely. **Every before/after pair is re-ranged after the
box** — on the before-cut axes the survivors occupy a ninth of the frame and the rest is
empty, which reads as "almost nothing survived" instead of showing the shape of what
did. The kept fraction is on each after-panel title, which is what the shared axes were
there to convey.

No efficiency and no τ is quoted for any run in this deck.

**The stage figures use two run sets, and they are never mixed in one frame.** The
earlier version of M13–M16 put all four runs side by side and contrasted 6264 (Port 5,
y = 0) with 6266 (Port 5, **y = 100**). Both halves of that were wrong for the question.
A dummy trigger has **zero** IC-passing waveforms, so any frame that shows the AmBe
waveform cut is blank for it and the cut cannot be a shared row; and efficiency varies
by roughly 10 pp between y = 0 and y = 100 on the campaign's own map, so part of any
6264-vs-6266 difference is geometry rather than the source. So:

* **M13–M16 + M25** — 6264 against **6265**, both at Port 5, (0, 0, 0), through all
  three stages: before the AmBe waveform cut, after it, after the box.
* **M21–M24 + M26** — 6254 against 6256, the two dummy triggers, ungated, **box only**.

**6265 also replaces 6266 in every 1D overlay, per-run page and cut flow (M1–M12,
M17–M18)**, for the same reason. `DIAG_RUNS` is now `[6254, 6256, 6264, 6265]`.

**Titles say what the figure is and nothing else.** The commentary that used to ride
in them — which cut is doing the work, why a row is ungated, what range the after-box
panels are drawn on — belongs in the talk and in this file, not on the slide.


| # | figure | says |
|---|---|---|
| M1 | `V4SPECDIAG__ic_cutflow_table` | **start here**: Stage-1 waveform cut flow per run. dummy trigger (LAPPD on) **0 of 7,933**, (LAPPD off) **0 of 25,525**, no source **808 in window → 589 accepted**, regular run 53,933 → 39,391 |
| M2 | `V4SPECDIAG__ic_adjusted_window` | the same as a spectrum, counts on log y. Both dummy triggers never leave a narrow peak at 0 — **no BGO gamma at all**, as a scheduled readout should not have; the regular run shows the AmBe BGO peak at 800–1000; the no-source run is a flat cosmic tail to 4000 that leaks **808 false starts** into the gate |
| M3 | `V4SPECDIAG__ic_features_1d` | the six IC waveform features ungated, area-normalised — where the false-trigger population differs in shape from a real one |
| M4 | `V4SPECDIAG__ic_features_1d_counts` | M3 in raw counts on a log y, so the exposure of each run is visible rather than divided out |
| M5 | `V4SPECDIAG__ic_page_6254` | the basic IC page for the **dummy trigger (LAPPD on)** — one run, its own axes, its own counts. The special-run twin of the campaign's per-run IC book |
| M6 | `V4SPECDIAG__ic_page_6256` | the same for the **dummy trigger (LAPPD off)** |
| M7 | `V4SPECDIAG__ic_page_6264` | the same for the **no-source run** — this is the false-start IC distribution, and it is visibly a second population: prompt fraction spread to 1.0, width-over-threshold piled up near 850 bins, baseline σ to 50 ADC |
| M8 | `V4SPECDIAG__ic_page_6265` | the same for the **source-in run at the same position as 6264**, for comparison. Was 6266 (Port 5, y = 100); 6265 sits at (0,0,0) like 6264, so the whole deck now compares like with like |
| M9 | `V4SPECDIAG__tank_features_1d` | the tank side ungated, area-normalised: the four cut variables plus multiplicity and per-trigger charge, on ranges that run **past** every cut bound so what the box discards is visible. **9.55 clusters/trigger for the no-source run** against 1.95 for the regular run and 0.55 for the two dummy triggers |
| M10 | `V4SPECDIAG__tank_features_1d_counts` | M9 in raw counts |
| M11 | `V4SPECDIAG__tank_features_1d_afterbox` | the same six panels **after** the box, on the ranges the box admits, area-normalised. Multiplicity is recomputed as *accepted* clusters per trigger, which is the number that matters |
| M12 | `V4SPECDIAG__tank_features_1d_afterbox_counts` | M11 in raw counts |
| M13 | `V4SPECDIAG__pair_stages_clusterchargebalance_vs_clusterpe` | **the headline ask**: charge balance vs cluster PE for **6264 (no source) against 6265 (source in), both at Port 5, (0,0,0)** — rows are the streamline, ungated → AmBe waveform cut → box cut. Kept fractions **6.3 % → 7.0 % (787 clusters)** for no source against **41.3 % → 48.8 % (15,165)** for source in |
| M14 | `V4SPECDIAG__pair_stages_clustertime_vs_clusterpe` | the same three stages for cluster time vs PE. **This is the one to leave up**: after the box, 6265 shows the capture-time tail out to 67 µs and 6264 shows only a prompt clump near 8 µs |
| M15 | `V4SPECDIAG__pair_stages_clusterhits_vs_clusterpe` | the same for cluster hits vs PE |
| M16 | `V4SPECDIAG__pair_stages_clustertime_vs_clusterchargebalance` | the same for cluster time vs charge balance |
| M25 | `V4SPECDIAG__pair_stage_cutflow_table` | the same streamline as numbers, so the panel counts are checkable: 176,634 → 11,210 → 787 for no source, 75,259 → 31,081 → 15,165 for source in |
| M21 | `V4SPECDIAG__dummy_stages_clusterchargebalance_vs_clusterpe` | the **dummy triggers on their own**, 6254 (LAPPD on) vs 6256 (LAPPD off), before and after the box. **No AmBe waveform cut row exists**: both have 0 IC-passing waveforms, so that stage is empty by construction rather than by choice. Kept 28.6 % / 29.4 % |
| M22 | `V4SPECDIAG__dummy_stages_clustertime_vs_clusterpe` | the same for cluster time vs PE |
| M23 | `V4SPECDIAG__dummy_stages_clusterhits_vs_clusterpe` | the same for cluster hits vs PE |
| M24 | `V4SPECDIAG__dummy_stages_clustertime_vs_clusterchargebalance` | the same for cluster time vs charge balance |
| M26 | `V4SPECDIAG__dummy_stage_cutflow_table` | the dummy streamline as numbers: 4,404 → 1,260 and 13,837 → 4,068 |
| M17 | `V4SPECDIAG__tank_cutflow_table` | Stage-2 cumulative cut flow on ungated clusters. The no-source run loses **94.2 %** of its 176,634 clusters, and it is charge balance and time that do it |
| M18 | `V4SPECDIAG__tank_nminus1_table` | the same leave-one-out, which is what identifies *which* cut is working. **`hits ≥ 5` is a strict no-op** at every run — ClusterFinder already requires 5 |
| M19 | `V4SPECDIAG__nosource_charge_6254_vs_6256` | **the nothing-to-detect charge answer, on the two DUMMY TRIGGERS, and it is two numbers not one**: hit-level charge differs by **+220.7 ± 13.9 PE (15.9σ)** and hit count by **+118.6 ± 1.2 (95.9σ)**, but CLUSTER-level charge **agrees** (−6.1 ± 17.7 PE, 0.35σ). The LAPPD leak raises the hit stream without producing clusters |
| M20 | `V4SPECDIAG__ic_adjusted_vs_fprompt_2d` | backup: IC_adjusted vs prompt fraction, one panel per run. Dropped from the running order — M2 and M3 already carry both of its projections. The figure to produce if anyone asks whether the gate could be moved |

*The number that must travel with M19, because it is the reason mean tank PE is not
usable as a stand-in for cluster PE anywhere in this analysis:* the two estimators
reproduce the diagnostics' own Step G to the digit — `tot_mean` 642.40 / 421.75 and
`clus_mean` 279.21 / 285.32 — and they disagree in both size **and** significance. Quote
which estimator you mean, every time.

*The no-source statement, in one line, and it is a systematic:* 6264 has the real trigger
logic live and no source, yet 808 waveforms land in the IC gate and 589 survive the
second-pulse veto. With nothing to emit a neutron those are **false starts of the IC
trigger** — 0.00833 Hz against 0.24919 Hz for the source-in run, i.e. **3.34 % of AmBe
triggers at 6264's cosmic loading**, or **≈0.7 %** scaled to a normal run's loading
(66.6 % vs 13.6 % cosmic-vetoed, 4.9×). Its tank carries 9.55 clusters per trigger with a
tail reaching **112 box-accepted clusters in a single trigger**, which is why the
campaign's cosmic veto removes 98.0 % of it and why 6264 must never be used as a
no-neutron background sample.

*Why the dummy triggers cannot measure that, and what they measure instead:* a scheduled
readout never exercises the trigger logic, so it has no false-start rate to report — 0 of
7,933 and 0 of 25,525 in the gate is the expected result, not a limit on false starts.
What a dummy trigger does give, and nothing else does, is the **accidental cluster rate**:
the box accepts **0.159 clusters per trigger** from a random slice of the tank, and the two
runs agree on it to 0.3 %.


## Z. Backup and zoom figures — collected, not slotted

Everything the prose above calls "backup", "zoom" or "optional", plus the headline page
of each alternate configuration. These are gathered so the folder is complete; none is
in the running order.

| # | figure | says |
|---|---|---|
| Z1 | `V3MERGED__roc_merged` | the merged ROC on its own |
| Z2 | `V3MERGED__feature_importances` | which features the GBT actually uses |
| Z3 | `V3MERGED__feature_separation` | per-feature separation power |
| Z4 | `V3MERGED__nn_loss_curves` | why the NN calibrates badly |
| Z5 | `V3MERGED__auc_by_campaign` | the ξ=0.02 comparison — **only** with the "not comparable" caveat (§0) |
| Z6 | `V3MERGED__bkg_composition_streams` | background composition per streamline |
| Z7 | `V3MERGED__compcut_impact_table` | what the composition cut removes |
| Z8 | `TT_CF_GBT__presentable_byspecies_top2_pe_total` | D5 zoom, full width, one feature |
| Z9 | `TT_CF_GBT__presentable_byspecies_top2_n_hits` | D5 zoom, full width, one feature |
| Z10 | `RT_CF_GBT__presentable_features_byspecies` | reco-tag / ClusterFinder, same page as D5 |
| Z11 | `TT_OPTICS_XGB__presentable_features_byspecies` | truth-tag / OPTICS, same page as D5 |
| Z12 | `RT_OPTICS_GBT__presentable_features_byspecies` | reco-tag / OPTICS, same page as D5 |
| Z13 | `V3BG__truthtag_optics__top6_shape` | C3 for the OPTICS clustering |
| Z14 | `V3AMBE__score_data_mc_overlay` | the June score overlay — the "model transferred" baseline that 6266 reverses |
| Z15 | `V3AMBE__tau_summary` | June τ per model × working point; **31.30 ± 2.28 µs** at truth-tag/CF GBT eff80 |
| Z16 | `V3AMBE__capture_time_selected` | June capture time, the fit itself |
| Z17 | `V4BOX__cut_variables` | what the box actually is: the four cut variables with the cut positions |
| Z18 | `V3AMBEPIPE__classified_fraction` | the superseded 20-run version of F1 — keep only to answer "did adding runs change it?" |
| Z19 | `neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28` | the **pooled** 28-run box-cut capture-time fit, χ²/ndof 3.22. Backup only — quote the per-position weighted τ (G4), never this |
| Z20 | `capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28` | residuals of Z19, which is where the pooled fit's χ² comes from |
| Z21 | `neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28` | neutron multiplicity per trigger, campaign-pooled |
| Z22 | `cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28` | PE vs charge balance, the 2D the box cut is drawn in |
| Z23 | `delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28` | Δt between first and subsequent clusters in multi-neutron events |

## Appendices — multi-page PDFs, not slots

Not in the tables above because they are multi-page documents rather than single
figures, so the collector does not index them. They live in `PRESENTATION_PLOTS/`
alongside everything else.

| file | pages | what |
|---|---:|---|
| `APPENDIX_boxcut_v4_by_position.pdf` | 26 | **one page per source position, worst efficiency first.** Each page: the capture-time histogram with the fitted curve drawn, τ and χ²/ndof in the panel title, the four box-cut variables (one log-y), and a stats block (candidates, events, mean candidates/event, efficiency, median PE/CB/hits). Box cuts only, no MVA |
| `APPENDIX_features_by_position_allStage2.pdf` | 19 | **one page per feature, one panel per port, one curve per y position.** All 225,811 Stage-2 clusters. This is the classic per-port-position feature comparison |
| `APPENDIX_features_by_position_truthtag_mvaneutron.pdf` | 19 | the same, restricted to MVA neutron @ GBT eff80, truth-tag (threshold 0.423621, 178,409 kept = 79.01 %) |
| `APPENDIX_features_by_position_recotag_mvaneutron.pdf` | 19 | the same for reco-tag (threshold 0.416366, 177,907 kept = 78.79 %) |
| `APPENDIX_positiongrid_truthtag.pdf` | 26 | one page per position, **MVA neutron vs MVA other** overlaid on twelve leading features. Truth-tag |
| `APPENDIX_positiongrid_recotag.pdf` | 26 | the same for reco-tag |
| `verbose/AllRuns_IC_adjusted_AmBe2.0v4_gated.pdf` | 20 | Stage-1 IC waveform distribution, one page per run. Not copied into `PRESENTATION_PLOTS/` — diagnostic, not presentational |
| `verbose/AllRuns_IC_adjusted_AmBe2.0v4_ext.pdf` | 8 | the same for the 8 later runs |

*Why there is only ONE all-Stage-2 feature file and not one per streamline:* the 31
features are **identical** between truth-tag and reco-tag. Both score the same AmBe
clusters with the same features derived the same way; the streamline changes only which
MC labelling trained the model, hence the score, hence the selection. Writing two
identical all-cluster files would imply a difference that is not there. The streamline
only becomes meaningful once the MVA cut is applied — which is what the two
`_mvaneutron` files and the two `positiongrid` files show.

*Why worst-first ordering in `by_position`:* page 1 is (0, −100, 102) at 40.31 %, the
position a reviewer will ask about. Its τ = 28.84 ± 1.63 µs is perfectly ordinary — low
efficiency at port 3 is a light-collection effect, not a capture-physics one, and this
appendix is how you show that in one page.

### Where the capture-time *histograms* live, and what G4 is not

**G4 is a summary of 26 fitted τ values — it is not a capture-time histogram, and it is
not a substitute for one.** The histograms with drawn fits exist in two places:

- **Per position, with the fit curve**: `APPENDIX_boxcut_v4_by_position.pdf`, 26 pages.
- **Pooled, from `src/ambe/plots/combined.py`**: Z19 `neutron_capture_time_fit__…` plus
  Z20 residuals — produced by `ambe plots combined --config configs/data_ambe2v4_all28.yaml`.

**Why the headline τ does not come from `combined.py`.** Its fit is a *different recipe*
from the anchor's, and the two are not interchangeable:

| | `combined.py` | anchor recipe (`fit_capture_time_fprompt_compare`) |
|---|---|---|
| bins over 0–70 µs | 200 | **70** |
| fit window | 2–65 µs | **10–67 µs** |
| pedestal `B` | **free**, (0, ∞) | **fixed at 0** |
| `therm` bound | (0.1, 100) | (0.1, 10) |
| scope | pooled, one config | **per position**, then inverse-variance weighted |

τ = 29.417 ± 0.221 µs was measured with the right-hand recipe, so only that recipe
produces a number comparable to it — which is why G4/G5 and the appendix use it, and why
`combined.py`'s pooled fit (χ²/ndof **3.22**) is filed as backup rather than as the
result. `combined.py` also has no per-position loop, so it cannot produce G4 at all.
Both are correct; they answer different questions. Quote the appendix or G4 for τ, and
show Z19 when someone wants to see one histogram and one curve.

---

## Short version — 14 slides

A1 · A4 · B1 · B5 · B6 · C1 · C2 · **D1 · D4 · D5** · **E1 · E3** · **F1 · F2**  (+ F4 if the special runs come up)

That keeps all three new results (the background is a mixture on the tank side and a
single source on the world side; the v4 capture-time closes but the score scale does not
transfer; the new definition keeps 79% of the pipeline's own AmBe neutrons and what it
drops is measurably flatter) and drops the method detail to backup.

## Numbers to have ready

- merged AUCs 0.656–0.694; deliverable **truth-tag / ClusterFinder GBT 0.672**
- per-population AUC **0.579–0.604** (tank μ/π) vs **0.689–0.711** (dirt n) — quote with any merged AUC
- background: 59.5% neutron-capture light; **70%** of background clusters carry no traced non-EM light
- tank background **2.19 particles/cluster**, **17.6%** single-particle; signal **1.25 / 77.1%**
- per-particle GBT AUC **0.552 (π⁰) → 0.705 (out-of-tank capture)**
- 6266: τ = **30.40 ± 3.37 µs** at eff80, anchor **29.42 ± 0.22**, pull +0.29
- 6266 gate: 2,171 of 4,182 IC candidates; 75,625 → 1,248 triggers (multiplicity is flat at ~1.12–1.18 everywhere)
- AmBe pipeline neutrons (28 runs), MC eff80: **79.0% neutron / 21.0% other**; of Stage-2-rejected clusters only **19.6%** called neutron
- rejected sample is **+1.10 ± 0.09 µs later (+12.5σ)**, late/early **×1.15**, KS D = 0.035
- kept fraction **64.2–86.0%** over 26 positions; correlates with `n_hits` (+0.91), not `d_wall` (+0.01)
- Stage-2 closure 225,945 / 207,530 vs the pipeline's 236,792 / 217,532 (−4.6% on both)
- specials: 6265/6266/6270 at 79–86%; **6264 → 0 clusters** (93.3% cosmic-vetoed); 6254/6256 → 0 IC-passing triggers
- the event key is **`event_tank_time`**; `event_number` restarts per part file (ratio 0.40 on 6266 ungated)
- `d_wall` is in **metres** (the June-16 poster labels it cm — wrong)
