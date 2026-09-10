# CCinc v3 — Classifier Selection, Differential Background Lineage, and AmBe Closure

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



**Date produced:** 2026-08-05
**Companion reports** (read for anything about the samples or the training itself):

- `REPORT_ccinc_v3_streamlines.md` — the tank-only two-streamline study, Stages 0→2.
  **Frozen. Nothing here recomputes or supersedes it.**
- `REPORT_ccinc_v3_world_merged.md` — the world-volume campaign and the four merged
  tank+world trainings. §0 has the labelling rule, §0.0 the lineage method, §5 the AUCs.
  **§4.4 (added 2026-08-06) is required reading before quoting any AUC in §2 below**: it
  decomposes the merged AUC by background population and shows the dirt-neutron class is
  the *easiest* part of the background (0.689–0.711) while the tank muon/pion background is
  the hardest (0.579–0.604) — i.e. the merged gain is a change of question, not a better
  answer to the old one. §4.5 explains why the tank side of the background carries neutron
  light at all.

This report answers the question the campaign was built for and the two above stop short
of: **which classifier, in which streamline, with which clustering method** — and then
what limits it, how much the two streamlines actually disagree, and whether any of it
survives contact with AmBe data.

Everything is read-only over existing artifacts. No retraining, no reprocessing.

---

## 0. The answer, up front

**Use GBT on ClusterFinder clusters.**

| criterion | verdict |
|---|---|
| MC discrimination (AUC) | GBT best in 3 of 4 configurations, XGBoost in the 4th — but the GBT/XGB gap is **inside the CV scatter in 3 of 4**, so they are a tie |
| MC loss (log-loss, Brier) | **GBT best in 3 of 4**, XGBoost in the 4th. Agrees with AUC in all four |
| NN | **worst or joint-worst on loss in all four** (log-loss 0.655–0.679 vs GBT's 0.633–0.651) despite competitive AUC — it ranks acceptably and calibrates badly |
| Random Forest | never best merged. It led three of four *tank-only*; that ordering flipped |
| transfer to AmBe data | **ClusterFinder transfers, OPTICS does not** — KS-to-MC-signal 0.11–0.16 vs 0.42–0.64 |
| AmBe capture-time closure | every usable CF fit is consistent with τ = 30.53 ± 0.26 µs; **reco-tag/OPTICS GBT fails outright** |

The MC-best configuration (reco-tag / ClusterFinder GBT, AUC 0.694) and the
best-transferring one are the same, which is the one piece of luck in this analysis. But
**reco-tag is not data-applicable** (its FV and muon kinematics still read truth
branches), so the deliverable for real data is **truth-tag / ClusterFinder GBT**:
AUC 0.672, log-loss 0.646, purity 0.590 at 80% signal efficiency, and
τ = 31.30 ± 2.28 µs on AmBe against the 30.53 ± 0.26 anchor.

Two negative results that matter as much as the positive one:

- **A total-charge box cut will not help.** The background species mix is nearly flat in
  `pe_total` (§2.2). What the classifier removes is μ⁺; what it cannot remove is
  π⁺, π⁻ and protons, and those sit *preferentially in the signal-like tail*.
- **The two streamlines do not disagree about labels at all** — 100% label agreement on
  every shared cluster (§3). They disagree only about *which events to include*, and the
  reco-tag-only population is 99% μ⁺/π⁻-origin wrong-lepton events entering its **signal**
  class.

---

## 1. Method and the guard that makes it trustworthy

Everything below is computed from the four merged score tables:

```
<BASE>/cc_neutrino_v3_{truthtag,recotag}/parquet/
    cc_neutrino_v3_*__mva_scores__keepprompt__merged__{optics,cf}.parquet
```

Three of the four carry `mva_label`. **The truth-tag/OPTICS table does not** — it was
written at 11:05 on 2026-08-05, before the fix that added the column at 11:26. Its label
is re-derived with the documented vertex rule (out-of-tank ⇒ background regardless of
composition; in-tank ⇒ neutron-dominated). Re-deriving it from `dominant_class` alone is
the trap: an out-of-tank capture is neutron-dominated and still background, and that
mistake reads AUC 0.54 instead of 0.66.

So `ccinc_v3_stats.py` **asserts all 16 merged AUCs reproduce world report §5 to 1e-3
before computing anything**, and exits non-zero otherwise. It passes:

```
[guard] PASS — all 16 AUCs reproduce the report to 0.001
```

Test-set sizes: 14,480 (tt/OPTICS) · 15,779 (tt/CF) · 10,193 (rt/OPTICS) · 10,295 (rt/CF),
matching `ccinc_v3_effpurity_by_config.csv` exactly. Every number in §2–§4 sits behind
that assertion.

**One thing to know before reading any purity below.** The merged trainings are 1:1
capped, so the test set has nS ≈ nB by construction. Purity here is the purity of *that*
set and is a relative figure of merit between models on a common sample — it is not a
physical purity, because the true signal:background ratio in data is not 1:1.

---

## 2. Which classifier

### 2.1 All four models on one scale

Test-set AUC with 5-fold CV, and the two losses. Loss is the new information: the
reports quote no loss for the tree models at all, and only a training curve for the NN.

| configuration | model | features | AUC | CV ± σ | log-loss | Brier |
|---|---|---:|---:|---:|---:|---:|
| Truth-tag / OPTICS | RF | 31 | 0.6563 | 0.6523 ± 0.0034 | 0.6537 | 0.2310 |
| | GBT | 31 | 0.6611 | 0.6564 ± 0.0037 | 0.6510 | 0.2297 |
| | **XGB** | 31 | **0.6622** | 0.6536 ± 0.0032 | **0.6504** | **0.2295** |
| | NN | 31 | 0.6492 | — | 0.6786 | 0.2424 |
| Truth-tag / ClusterFinder | RF | 31 | 0.6647 | 0.6680 ± 0.0044 | 0.6493 | 0.2287 |
| | **GBT** | 31 | **0.6717** | 0.6718 ± 0.0046 | **0.6455** | **0.2271** |
| | XGB | 31 | 0.6688 | 0.6692 ± 0.0054 | 0.6471 | 0.2278 |
| | NN | 31 | 0.6674 | — | 0.6691 | 0.2375 |
| Reco-tag / OPTICS | RF | 21 | 0.6844 | 0.6832 ± 0.0048 | 0.6391 | 0.2240 |
| | **GBT** | 21 | **0.6894** | 0.6850 ± 0.0042 | **0.6357** | **0.2225** |
| | XGB | 21 | 0.6879 | 0.6850 ± 0.0045 | 0.6365 | 0.2228 |
| | NN | 21 | 0.6644 | — | 0.6604 | 0.2337 |
| Reco-tag / ClusterFinder | RF | 21 | 0.6884 | 0.6851 ± 0.0038 | 0.6369 | 0.2228 |
| | **GBT** | 21 | **0.6936** | 0.6908 ± 0.0031 | **0.6332** | **0.2212** |
| | XGB | 21 | 0.6899 | 0.6883 ± 0.0038 | 0.6362 | 0.2226 |
| | NN | 21 | 0.6869 | — | 0.6547 | 0.2309 |

**The feature column is new and was not in the reports.** Both reco-tag merged trainings
used **21 of 31 features**, not 31: the world rows' fitted-vertex block is too sparse to
pass the non-NaN threshold. The reports note this for the *world-only* trainings; it is
true of the merged reco-tag ones too. So the two highest-AUC configurations are also the
ones with the fewest inputs — further evidence that the merged gain is background
statistics, not richer information.

**Is the lead real?** Only in one configuration.

| configuration | best | runner-up | margin | CV σ | resolvable? |
|---|---|---|---:|---:|:--:|
| Truth-tag / OPTICS | XGB | GBT | 0.0012 | 0.0032 | no |
| Truth-tag / ClusterFinder | GBT | XGB | 0.0029 | 0.0046 | no |
| Reco-tag / OPTICS | GBT | XGB | 0.0015 | 0.0042 | no |
| Reco-tag / ClusterFinder | GBT | XGB | 0.0038 | 0.0031 | **yes** |

GBT and XGBoost are a tie on AUC. What separates them is that **log-loss picks the same
winner as AUC in all four configurations**, and on loss GBT wins three of four. RF and the
NN are excluded on loss in every configuration — the NN decisively so: its log-loss is
0.02–0.03 worse than GBT's everywhere, while its AUC is competitive in three of four. It
ranks acceptably and calibrates badly, which is exactly the behaviour world report §5
describes from the training curves (validation loss turns up after ~5 epochs while
validation AUC keeps climbing to 0.66; early stopping monitors `val_auc` deliberately).

### 2.2 Working points

Full scan in `ccinc_v3_model_comparison.csv` (80 rows). At the reference 80% signal
efficiency:

| configuration | model | threshold | bkg rejection | purity |
|---|---|---:|---:|---:|
| Truth-tag / OPTICS | GBT | 0.439 | 0.425 | 0.580 |
| Truth-tag / ClusterFinder | **GBT** | 0.424 | 0.450 | **0.590** |
| Reco-tag / OPTICS | GBT | 0.430 | 0.466 | 0.597 |
| Reco-tag / ClusterFinder | **GBT** | 0.416 | 0.478 | **0.603** |

These reproduce `ccinc_v3_effpurity_by_config.csv` to |Δpurity| ≤ 1.9e-4 and |ΔAUC| = 0
(that CSV was built independently, so this is a real cross-check, not a tautology).

At 50% efficiency purity reaches 0.632 (tt/OPTICS GBT) to 0.670 (rt/CF GBT); at 90% it
falls to 0.551–0.561. **Max-purity tops out at 0.71–0.78 but only at 5% signal
efficiency** — and in most configurations that optimum is pinned against the 5%
efficiency floor imposed on the search, meaning purity is still rising monotonically into
the tail. There is no interior optimum: this is a smooth trade-off with no sweet spot.

**Max-significance is reported in the CSV but should not be used.** On a 1:1 test set
S/√(S+B) is maximised at 92–98% efficiency, i.e. at no selection at all. That is an
artifact of the balanced training sample, not a working point.

---

## 3. What limits the classifier — differential background lineage

§4.2 of the world report gives the integrated composition: 60% of the light in the
background class is real neutron capture, and of the non-neutron remainder μ⁻ 53.3%,
π⁺ 25.5%, π⁰ 7.2%, p 5.8%, π⁻ 4.6%, μ⁺ 3.4%. That cannot tell you what to do about it.
Binned along the features the models use, it can.

Reproduced integrated, truth-tag/OPTICS, as the check that the population and
denominators match the report: 36,008 background clusters, 116,968 traced non-neutron
hits, neutron-capture light **59.5%** of hits, and μ⁻ 53.2% / π⁺ 25.5% / π⁰ 7.2% /
p 5.8% / π⁻ 4.6% / μ⁺ 3.4%. **Exact agreement.**

Species axis is `origin_pdg` — the first non-EM ancestor, EM generations skipped
(world report §0.0). Not `bg_class`, never `ImmediateAncestorClass`. Deciles;
hit-weighted headline with the cluster-dominant view carried alongside; bins with fewer
than 200 traced hits dropped.

### 3.1 Along the discriminator — which background is removable

Ratio of a species' share in the lowest-score decile to its share in the highest.
**> 1 = the classifier pushes it down (removable); < 1 = it survives into the
signal-like tail (irreducible).**

| species | tt/OPTICS | tt/CF | rt/OPTICS | rt/CF |
|---|---:|---:|---:|---:|
| μ⁺ | **8.98** | **9.95** | **4.49** | **5.36** |
| μ⁻ | 1.15 | 1.13 | 1.15 | 1.12 |
| π⁰ | 0.63 | 0.50 | 0.57 | 1.09 |
| π⁺ | 0.54 | 0.46 | 0.57 | 0.51 |
| p | 0.54 | 0.35 | 0.48 | **0.28** |
| π⁻ | 0.46 | 0.29 | 0.65 | **0.30** |

The story is the same in all four configurations:

- **μ⁺ is what the classifier actually kills.** It is 16.3% of the traced non-neutron
  background light in the lowest-score decile of truth-tag/OPTICS and 1.8% in the
  highest — a factor of 9. It is also only 3.4% of the background overall, so killing it
  buys little.
- **π⁺, π⁻ and protons are the wall.** Each roughly *doubles to triples* its share
  going into the signal-like tail (π⁺ 15.9% → 29.5%, p 3.3% → 6.1% in truth-tag/OPTICS).
  These are the clusters the AUC stops on.
- **μ⁻ is flat (1.12–1.15) at 50–68% of the background everywhere.** The single largest
  background component is the one the discriminator has almost no handle on. This, not
  feature engineering, is why the merged AUC sits at 0.69.
- ClusterFinder pushes π⁻ and p further into the tail than OPTICS (0.29/0.35 vs
  0.46/0.54), i.e. its higher AUC is *not* bought by handling the hard species better.

**All four classifiers agree, which is the important control.** Repeating the ratio along
each model's own score (not just GBT's) gives, for truth-tag/ClusterFinder:

| species | RF | GBT | XGB | NN |
|---|---:|---:|---:|---:|
| μ⁺ | 7.76 | 9.95 | 10.09 | 9.86 |
| μ⁻ | 1.11 | 1.13 | 1.12 | 1.15 |
| π⁺ | 0.52 | 0.46 | 0.49 | 0.46 |
| p | 0.29 | 0.35 | 0.32 | 0.32 |
| π⁻ | 0.24 | 0.29 | 0.26 | 0.26 |

The spread across models is small everywhere except μ⁺ under RF (7.76 vs ~10), and the
ordering is identical in all four. The same holds in the other three configurations (the
one outlier is reco-tag/OPTICS RF, μ⁺ 2.49 vs ~4.5–5.1 for the rest). **So the
irreducible background is a property of the physics, not of the classifier** — changing
model does not change which species survive, which is why no classifier choice moves the
AUC by more than 0.03 and why §8 argues the ceiling is not a modelling problem.

### 3.2 Along total charge — the box-cut question, answered no

`pe_total` deciles, truth-tag/OPTICS, lowest → highest bin:

| species | low bin | high bin | low/high |
|---|---:|---:|---:|
| μ⁻ | 59.1% | 52.6% | 1.12 |
| π⁺ | 21.3% | 23.3% | 0.92 |
| π⁰ | 7.3% | 6.7% | 1.08 |
| p | 6.5% | 5.0% | 1.29 |
| π⁻ | 3.9% | 3.8% | 1.02 |
| μ⁺ | 1.9% | 8.3% | **0.22** |

Only μ⁺ has real structure — it concentrates at **high** charge (4.5× more of the
high-charge bin than the low). Everything else is flat to within 30% across a full
decade of charge. **A cut on total charge cannot separate these species**, so the
composition-cut approach that worked in the xi=0.02 campaign has no purchase on the
merged background. `n_hits` behaves the same way (all ratios 1.03–1.21 except μ⁺ at
0.39).

The one place a cut has leverage is the pairing of the two: μ⁺ is simultaneously
high-charge and low-score, so it is already being removed by the discriminator, and a
box cut would only duplicate that.

### 3.3 Geometry — open item 4 stays answered

Along `d_wall`, μ⁻ is flat (1.07) while π⁰ and π⁻ shift towards the interior
(0.65, 0.76). The species mix does change with geometry, but weakly, and the dominant
component does not change at all. This is consistent with world report §4.3 — separation
is led by multiplicity, geometry carries ~10–12% of the importance — and **does not
support the "the model is learning geometry" worry**. The differential view is the
stronger test and it agrees with the importances.

### 3.4 The background populations do not separate equally — see world report §4.4

**Added 2026-08-06.** §3.1–§3.3 ask which *species* the discriminator removes. The
complementary question — which *population* of the background it removes — is answered in
`REPORT_ccinc_v3_world_merged.md` §4.4 and changes how §2's AUCs should be read. GBT AUC
against each background population, common signal set:

| | BKG tank (muon/pion) | BKG dirt neutrons | BKG world non-neutron |
|---|---:|---:|---:|
| truth-tag / OPTICS | **0.584** | **0.689** | 0.661 |
| truth-tag / ClusterFinder | 0.595 | 0.701 | 0.643 |
| reco-tag / OPTICS | 0.579 | 0.707 | 0.681 |
| reco-tag / ClusterFinder | 0.604 | 0.711 | 0.659 |

The out-of-tank capture background — the one that "should" be inseparable from signal —
is the easiest part in every configuration, and the tank muon/pion background is the
hardest, at 0.584 against the tank-only model's own 0.605. Two consequences for this
report: the §2 model ranking is a ranking on a background that is 64% dirt neutrons by
cluster count, and §3.1's finding that μ⁻ is the irreducible component is reproduced from
a second direction. Reproduce with `python ccinc_v3_stats.py --do dirtn`.

### 3.5 Context that stops this being over-read

Across every axis, **52–68% of the light in the background class is genuine neutron
capture** and **63–84% of background clusters come from out-of-tank interactions**. The
species table above describes only the non-neutron remainder. The majority of this
background is real capture light that is background solely because no in-tank neutrino
made it — no feature of the *light* can separate that, which is the fundamental ceiling
on this discriminator and is not a modelling deficiency.

Full output: `ccinc_v3_bg_differential.csv` (8,640 rows), summary
`ccinc_v3_bg_dominance_summary.csv`, figures `V3BG__*` — 45 sheets: per configuration, nine
per-axis sheets (each with the stacked and the shape view side by side) plus a six-panel
stacked and a six-panel shape sheet over that configuration's top-six GBT features, and
one overall dominance table.

---

## 4. Truth-tag vs reco-tag — the discrepancy

Event-level overlap from streamlines §1.1 is **reproduced exactly** — truthtag 81,790,
recotag 41,418, both **40,442** — confirming the cluster-level join below did not change
the population.

At cluster level, joined on `(sample, eventID, cluster_id)` with the key verified before
comparing anything (`n_hits` agrees on 100.00% of shared clusters):

| | OPTICS | ClusterFinder |
|---|---:|---:|
| truth-tag clusters | 72,016 | 78,266 |
| reco-tag clusters | 50,988 | 52,090 |
| shared | 42,993 (84.3% of reco-tag) | 44,921 (86.2%) |
| truth-tag only | 29,023 | 33,345 |
| reco-tag only | 7,995 | 7,169 |
| **label agreement on shared** | **100.00%** | **100.00%** |
| score correlation, GBT | 0.942 | 0.969 |
| score correlation, RF | 0.878 | 0.895 |

**The two streamlines never disagree about a label.** Both apply the same vertex rule to
the same truth, so a cluster both streams see is classified identically, every time. The
entire discrepancy is *which clusters each stream includes*. That is a cleaner statement
than the event-level 97.6%-subset figure and it means the AUC difference between the
streams is a difference of training population, not of labelling.

Decomposing the populations (OPTICS; ClusterFinder is the same story):

| population | n | tank | world | signal | background |
|---|---:|---:|---:|---:|---:|
| shared | 42,993 | 19,645 | 23,348 | 17,595 | 25,398 |
| truth-tag only | 29,023 | 23,725 | 5,298 | 18,413 | 10,610 |
| reco-tag only | 7,995 | 7,780 | 215 | **7,899** | **96** |

**The reco-tag-only population is 98.8% signal.** Reco-tag never requires `trueCC` or a
muon final state, so a wrong-lepton event that is neutron-dominated, in-tank and in-FV
enters its **signal** class. The background composition of that population confirms it
independently: μ⁺ 51.1%, π⁻ 35.1%, π⁺ 7.5% (OPTICS) — i.e. antimuon CC events. This
reproduces streamlines §1.2's event-level decomposition (976 recotag-only events = 885 μ⁺
+ 48 e⁻ + 2 e⁺ + 41 NC) from a completely independent direction.

*Caveat on that composition: it rests on 96 (OPTICS) / 81 (CF) background clusters. The
direction is unambiguous, the percentages are not precise.*

So the reco-tag streamline buys its extra AUC by (a) 9.6× more background from the world
merge and (b) admitting μ⁺ events into signal. Neither is better physics. Combined with
the caveats already on record — its NC rejection is untested because the `trueVtx`
sentinel removed every NC event before tagging, and its FV and muon kinematics still read
truth branches — **the reco-tag streamline is not the deliverable for data**, whatever its
AUC.

---

## 5. AmBe closure — Stage 3 and 4, run for the first time

Scored with `run_ccinc_v3_ambe_stage3.sh`: 232,836 OPTICS + 93,809 ClusterFinder AmBe
clusters, 31 runs, 72,505 events, each method scored by its own method-matched model.
Zero features median-filled — all 31 (truth-tag) / 21 (reco-tag) inputs are present in
the data parquets.

**Two clobbering hazards were found and handled; do not undo the staging.**

1. The output name derives from the *input* name, and all four configurations share two
   input files. Scoring in place would have the four runs overwrite each other and would
   also have destroyed `ambe_all__data_features_optics_forscore__scored.parquet` from the
   June campaign.
2. **Every frozen `.pkl` points at the pre-rename NN filename.** `run_world_merged.sh`
   renamed the artifacts to `__merged__nn__{optics,cf}.keras`, but the stored
   `keras_file` still says `__merged__nn.keras`, which no longer exists — so
   `score_data()` would print one warning and silently return three scores instead of
   four, for all four configurations. Both methods' pkl want the same basename in the
   same directory, so this cannot be fixed in place; each configuration is staged in its
   own directory with symlinks. **The NN could not have been validated on data without
   this.**

Note also that `ambe_all__data_features_clusterfinder.parquet` (without `_forscore`) is
**empty** — picking it would score nothing and report success. The `_forscore` variants
are the ones to use; they also carry `clusterTime_earliest`.

### 5.1 Data/MC agreement — ClusterFinder transfers, OPTICS does not

KS distance from the AmBe score distribution to the MC signal and MC background shapes
(p-values not quoted: at 10⁵ clusters everything is "significant" and the number carries
no information — which side it is closer to does).

| configuration | model | data median | MC sig / bkg median | KS→sig | KS→bkg | closer to |
|---|---|---:|---:|---:|---:|---|
| Truth-tag / OPTICS | GBT | 0.262 | 0.565 / 0.476 | 0.642 | 0.472 | background |
| Truth-tag / ClusterFinder | GBT | 0.546 | 0.593 / 0.453 | **0.115** | 0.183 | **signal** |
| Reco-tag / OPTICS | GBT | 0.386 | 0.582 / 0.449 | 0.418 | 0.140 | background |
| Reco-tag / ClusterFinder | GBT | 0.543 | 0.601 / 0.430 | **0.127** | 0.205 | **signal** |

**This is the sharpest result in the report.** AmBe data is neutron-capture data, so a
model that transferred should place it near the MC *signal* shape. ClusterFinder does
exactly that in both streamlines, for RF/GBT/XGB. OPTICS places it *below the MC
background* — KS-to-signal 0.42–0.64, data median 0.21–0.39 against an MC background
median of 0.45. The OPTICS models have not transferred.

The likely cause is in the sample, not the model: OPTICS produces 232,836 AmBe clusters
of which only **31.9%** pass Stage 1, against 93,809 ClusterFinder clusters at **64.0%**.
The OPTICS data sample is dominated by low-quality clusters the MC training never saw.

**The NN is closer to background in all four configurations**, including the two where the
trees transfer. It transfers worst — consistent with its calibration deficit in §2.1.

Figure: `V3AMBE__score_data_mc_overlay`.

### 5.2 Physics closure — capture time

Fitted with the *imported* fitter from `fit_capture_time_fprompt_compare.py`, so the
binning, 10–67 µs fit window and functional form are the ones that produced the anchor
**τ = 30.53 ± 0.26 µs** (`REPORT_fprompt_waveform_cut.md` §7, 20 runs / 19 positions).
A τ fitted any other way would not be comparable. Capture time is `t_mean`/1000 µs for
both methods — the OPTICS parquet has `clusterTime_earliest` entirely NaN, and where the
CF parquet has both they agree to ~5 ns, negligible against 30 µs.

**28 of 71 fits are unusable and are excluded from every verdict**: pinned at a fit bound
(τ exactly 10.0 or 70.0), or χ²/ndof > 5, or στ > 10 µs. This gate is load-bearing — one
NN max-purity point pinned at τ = 10.0 ± 166 µs and passed a naive 2σ test before the
gate existed.

At the 80%-efficiency working point:

| configuration | model | τ [µs] | χ²/ndof | clusters kept | data pass | multiplicity |
|---|---|---:|---:|---:|---:|---:|
| Truth-tag / OPTICS | GBT | 30.65 ± 2.91 | 3.15 | 38,504 | 16.5% | 1.10 |
| Truth-tag / ClusterFinder | **GBT** | **31.30 ± 2.28** | 4.57 | 68,406 | 72.9% | 1.15 |
| Reco-tag / OPTICS | GBT | **70.0 (bound)** | **42.9** | 94,548 | 40.6% | 1.73 |
| Reco-tag / ClusterFinder | GBT | 31.47 ± 2.31 | 4.61 | 67,890 | 72.4% | 1.15 |

**Reco-tag/OPTICS GBT fails outright on data** — τ pinned at the 70 µs bound with
χ²/ndof = 21–55 at *every* efficiency working point, and a mean multiplicity of 1.73
against ~1.1 everywhere else. That is the configuration MC ranked second-best overall
(AUC 0.689). Its RF, XGB and NN in the same configuration are fine; only GBT collapses.
This is the clearest possible demonstration that MC AUC does not predict data behaviour.

**And the Stage-1 columns say exactly why.** `data_pass_frac` is the fraction of *all*
clusters kept, which is hard to compare across methods because the Stage-1 pass rates
differ so much (31.9% OPTICS vs 64.0% CF). Two derived columns fix that: `eff_of_stage1`
(what fraction of Stage-1 clusters the cut keeps) and `purity_stage1` (what fraction of
what it *selected* passes Stage 1).

| configuration | model | pass frac | eff of Stage-1 | purity_stage1 | τ |
|---|---|---:|---:|---:|---|
| Truth-tag / OPTICS | GBT | 0.165 | 0.374 | 0.720 | 30.65 ± 2.91 |
| Truth-tag / ClusterFinder | GBT | 0.729 | 0.973 | 0.854 | 31.30 ± 2.28 |
| Reco-tag / OPTICS | RF | 0.162 | 0.375 | 0.736 | 31.42 ± 3.21 |
| Reco-tag / OPTICS | **GBT** | 0.406 | 0.448 | **0.352** | **70.0 (bound)** |
| Reco-tag / OPTICS | XGB | 0.200 | 0.458 | 0.728 | 32.74 ± 2.72 |
| Reco-tag / ClusterFinder | GBT | 0.724 | 0.961 | 0.850 | 31.47 ± 2.31 |

Every model that closes on τ selects a sample that is **72–91% Stage-1 clusters**. The one
that fails selects a sample that is **65% *non*-Stage-1** — it is picking up precisely the
junk the MC training never contained. That is a mechanism, not just a symptom, and it is
the same mechanism §5.1 blames for the OPTICS transfer failure generally, appearing here
in its most extreme form.

**A caveat the CF numbers require.** At 80% efficiency the ClusterFinder models keep
**96–97% of all Stage-1 clusters** — they are very nearly inclusive on CF data. Their good
τ closure therefore largely inherits Stage 1's, and the MVA is adding little selection on
top of it. The comparison to make is against the Stage-1 baseline (τ = 30.73 ± 2.44 µs),
not against the unselected sample, and on that comparison the MVA holds τ rather than
improving it.

Baselines, for reference: the unselected ClusterFinder sample fits τ = 34.39 ± 2.36 µs
(χ²/ndof 6.7 — rejected), and its Stage-1 subset τ = 30.73 ± 2.44 µs. The unselected
OPTICS sample does not fit a capture-time shape at all (χ²/ndof 134). So on ClusterFinder
the MVA holds τ where Stage 1 already had it while keeping 72.9% of clusters; it does not
*improve* τ, and there is no evidence here that it should.

**Ranking models by τ is not possible in the ClusterFinder configurations.** The four
models' τ span 0.21 µs (tt/CF) and 0.32 µs (rt/CF) — below the anchor's own 2σ of
0.52 µs. All four close equally well. What τ closure does is **disqualify**
(reco-tag/OPTICS GBT), not rank.

Figures: `V3AMBE__capture_time_selected`, `V3AMBE__tau_summary`.

### 5.3 What this does and does not establish

- AmBe has no truth labels. There is **no efficiency and no purity** in §5, only
  agreement and physics closure.
- Only the **cluster-level MVA** is under test. The CC event selection cannot be applied
  to AmBe at all — no neutrino, no muon, no MRD track.
- The reco-tag numbers are a **model-transfer check**, not a validated selection: that
  streamline is not data-applicable in principle.
- Thresholds are MC calibrations applied to data. The realised data pass rate is
  reported next to the MC efficiency, and they diverge widely on OPTICS (16.5% data vs
  80% MC) — that divergence *is* the transfer failure of §5.1, restated.

---

## 6. Caveats

1. **Purity is on a 1:1 test set** and is a relative figure of merit only (§1).
2. **Max-significance is meaningless on a balanced sample** (§2.2) — it is in the CSV for
   completeness and should not be quoted.
3. **The reco-tag streamline is not data-applicable** and admits μ⁺ events into signal
   (§4). Its NC rejection remains untested.
4. **The reco-tag merged models use 21 of 31 features** (§2.1), not the 31 the reports
   imply for merged trainings.
5. **The reco-tag-only background composition rests on ~90 clusters** (§4).
6. **The truth-tag/OPTICS label is re-derived, not read** (§1). Guarded by the AUC
   assertion, but if that table is ever regenerated with `mva_label`, the guard should be
   re-run to confirm the two agree.
7. **The frozen `.pkl` NN-companion paths are stale** (§5). The staging directory works
   around it; the underlying artifacts are still wrong and any other consumer of those
   pkl files will silently lose the NN.
8. **OPTICS AmBe transfer failure is now partly diagnosed** (§5.2): the one model that
   fails τ closure outright selects a sample that is 65% non-Stage-1, against 72–91%
   Stage-1 for every model that closes. The general OPTICS/CF asymmetry (Stage-1 pass rate
   31.9% vs 64.0%) is still correlational, not demonstrated.
9. **The ClusterFinder MVA is nearly inclusive at 80% efficiency** — it keeps 96–97% of
   Stage-1 clusters, so its τ closure largely inherits Stage 1's and should be read against
   the Stage-1 baseline (30.73 ± 2.44 µs), not the unselected sample.
10. **Nothing here revisits the tank-only results or the world processing.** Both are
   frozen and quoted from their own reports.

---

## 7. Reproducing

```bash
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate

python ccinc_v3_stats.py --do guard          # the AUC assertion alone
python ccinc_v3_stats.py --do models         # §2  -> ccinc_v3_model_comparison.csv
python ccinc_v3_stats.py --do truthreco      # §4  -> ccinc_v3_truthreco_*.csv
python ccinc_v3_stats.py --do dirtn          # §3.4 + world report §4.4/§4.5
                                             #     -> ccinc_v3_dirtn_*.csv
python ccinc_v3_bg_differential.py --config all      # §3 -> ccinc_v3_bg_*.csv

bash run_ccinc_v3_ambe_stage3.sh all         # §5 scoring (~4 min, 8 passes)
python ccinc_v3_ambe_closure.py --do all     # §5 -> ccinc_v3_ambe_{agreement,closure}.csv
```

None of these has a silent default — every entry point requires you to name what you
want. Steps §2–§4 are read-only over existing parquet and take minutes; the AmBe scoring
is the only compute and trains nothing.

**Outputs.** CSVs in this directory: `ccinc_v3_model_comparison.csv`,
`ccinc_v3_truthreco_discrepancy.csv`, `ccinc_v3_truthreco_overlap_breakdown.csv`,
`ccinc_v3_truthreco_disagreement_composition.csv`, `ccinc_v3_bg_differential.csv`,
`ccinc_v3_bg_dominance_summary.csv`, `ccinc_v3_ambe_agreement.csv`,
`ccinc_v3_ambe_closure.csv`, and from `--do dirtn`: `ccinc_v3_dirtn_budget.csv`,
`ccinc_v3_dirtn_census.csv`, `ccinc_v3_dirtn_separability.csv`,
`ccinc_v3_dirtn_feature_separation.csv`, `ccinc_v3_dirtn_tankbkg_neutron_light.csv`.
Figures in `slide_plots_ccinc_v3_merged/` as
`V3STATS__*` (4), `V3BG__*` (45), `V3AMBE__*` (3) — same standalone PDF+PNG format and
palette as the existing `V3MERGED__*` set. Scored AmBe parquets under
`<BASE>/ccinc_v3_ambe_stage3/<stream>_<method>/`. Log: `logs_ccinc_v3_ambe_stage3.log`.

## 8. What is worth doing next

1. **Diagnose the OPTICS data transfer failure.** Compare the AmBe OPTICS feature
   distributions against the MC ones feature by feature; the Stage-1 pass-rate gap points
   at cluster quality, but that is a hypothesis. If it is cluster quality, requiring
   `passes_stage1` before scoring may recover OPTICS.
2. **Fix the frozen `.pkl` NN-companion paths at the source** so the artifacts are usable
   without staging.
3. **Do not pursue a total-charge box cut** (§3.2). If a cut is wanted, the target is
   π⁺/π⁻/p in the signal-like tail, and §3.1 says no single feature separates them.
4. **The ceiling is physics, not modelling.** 52–68% of the background light is real
   neutron capture from out-of-tank interactions. Raising the AUC materially needs
   information about *where the neutron came from*, not better classifiers — every one of
   the four is within 0.03 AUC of the others.
