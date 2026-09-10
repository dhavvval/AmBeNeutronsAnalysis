# CCinc v3 — Two Independent Selection Streamlines (truth-tag vs reco-tag)
## productionv3 tank/fmvmrd · Stages 0 → 2, both clustering methods

**Runs:** `cc_neutrino_v3_truthtag`, `cc_neutrino_v3_recotag`
**Date produced:** 2026-08-03 / 04 · **Report written:** 2026-08-04
**MC input:** `/pnfs/annie/persistent/users/dajana/output/genie_wcsim_tank/productionv3/tank/fmvmrd/ANNIEEvent_cc_neutrino_v3*.root`
**Output root:** `/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/`

This report is **standalone** — every number comes from these two runs. It documents the result of
the two-streamline study: running the *same* CC-inclusive analysis twice over the same files, once
tagging the muon with truth information and once with detector information, and comparing selection
efficiency, background composition and MVA performance.

The two streamlines are **alternatives, never ANDed together**. Their intersection is the
superseded "combined" stream from the earlier campaign.

---

## Provenance — what ran where

| Session | Stage(s) | Log | Finished | Exit |
|---|---|---|---|---|
| tmux `ccv3_streamlines` | CC event tables only | `logs_ccinc_v3_streamlines_{truthtag,recotag,ALL}.log` | 2026-08-03 16:08 | 0 |
| tmux `ccv3_pilots` | 5-file pilots, Stages 0–2 | `logs_pilot_v3_{truthtag,recotag}.log`, `logs_pilots_v3_BOTH.log` | 2026-08-03 16:34 | 0 |
| tmux `ccv3_full` | Stages 0, 1, 1b, 2 (× optics, clusterfinder) | `logs_full_v3_{truthtag,recotag}.log`, `logs_full_v3_BOTH.log` | 2026-08-04 03:38 | 0 |

Driver: `run_ccinc_xi02_compcut.sh` with `STOP_STAGE=2 METHODS='optics clusterfinder'`.
Stage 3 (AmBe data scoring) and Stage 4 (box cuts) were deliberately **not** run.

Each run directory holds 28 artifacts. The ones that matter:

```
csv/     *__cc_ccinc_<stream>_cutflow.{csv,md}      the two tables this study is about
         *__cc_ccinc_<stream>_{nminus1,trainstats}.csv
         *__bkg_{hit_ancestry,lineage_chains,cluster_yields,compcut_impact}.csv
         *__feature_separation.csv  *__spurious_composition.csv
         *__mva_summary__keepprompt__{optics,cf}.csv     (feature importances only)
parquet/ *__cc_ccinc_<stream>_events.parquet          one row/event, all 2.49M, per-cut flags
         *__pulses.parquet  (5.1 GB)  *__clusterfinder.parquet  *__cluster_features.parquet
         *__mva_frozen__keepprompt__{optics,cf}.pkl + __nn.keras + __mva_scores__*.parquet
plots/   *__bkg_composition.pdf  *__cluster_features.pdf
         *__feature_separation_plots.pdf  *__mva__keepprompt__{optics,cf}.pdf
```

**The AUCs are only in the logs** — `*__mva_summary__*.csv` contains feature importances, not
metrics. Grep `'^\[mva\] AUC'`.

### Sample

499 files globbed, **498 usable, 2,490,000 events**. `ANNIEEvent_cc_neutrino_v3_24.root` is an
empty ROOT file (zero keys) and is skipped by `filter_files_with_tree()` — loudly, by name. Index
158 is absent from the glob. The files came from an interrupted ToolAnalysis run.

GENIE truth on this sample is **sound**: `corr(trueMuonEnergy, trueFSLEnergy) = 1.0`, no
`trueCC & trueNC` overlap. (This is *not* true of the older hand-built world ntuple — see
`WORLD_v3_REPORT.md`.)

---

## 1. Selection cut-flows

Identical through the kinematic block by construction; divergence only in the tagging block.
Cut order is FV → kinematics → tagging, as specified.

| Cut | truthtag | % prev | recotag | % prev |
|---|---:|---:|---:|---:|
| Total | 2,490,000 | — | 2,490,000 | — |
| FV r < 100 cm | 771,588 | 31.0% | 771,588 | 31.0% |
| FV \|Y\| < 100 cm | 423,443 | 54.9% | 423,443 | 54.9% |
| p_mu ∈ [600,1200) MeV/c | 136,182 | 32.2% | 136,182 | 32.2% |
| cos θ > 0.8 | 84,226 | 61.8% | 84,226 | 61.8% |
| *tagging* → `trueCC==1` | 83,991 | 99.7% | — | — |
| `trueFSLPdg==13` | 81,790 | 97.4% | — | — |
| *tagging* → `NoVeto==1` | — | — | 83,187 | 98.8% |
| MRD-tagged (`MRDClusterNumber>0`) | — | — | 51,068 | **61.4%** |
| promptPE ∈ [500,3000) | — | — | 41,418 | 81.1% |
| nhits ≥ 4 | 81,790 | 100.0% | 41,418 | 100.0% |
| **FINAL `cc_pass`** | **81,790 (3.29%)** | | **41,418 (1.66%)** | |

**Detector tagging costs a factor 1.97 in statistics.** Attribution:

- **MRD −38.6%** — by far the dominant cost
- promptPE −18.9%
- FMV −1.2% — little to veto in a tank-only simulation
- `nhits >= 4` is a measured **no-op** (100.0% kept), as in every configuration so far

Selection-efficiency estimates for future planning are in `*__cc_*_trainstats.csv`: at 3.28%
efficiency, 10,000 selected truth-tag events need ~61 files; at 1.66%, ~121 files.

### 1.1 Overlap — the two streams at full scale

Computed directly from the two persisted event tables, with row alignment verified on
`_source_file` + `eventID` before comparing `cc_pass`:

| | events | of truthtag | of recotag |
|---|---:|---:|---:|
| truthtag | 81,790 | 100% | — |
| recotag | 41,418 | — | 100% |
| **both** | **40,442** | 49.4% | 97.6% |
| truthtag-only | 41,348 | 50.6% | — |
| recotag-only | 976 | — | 2.4% |
| union | 82,766 | | |

Two things worth noting:

1. The intersection is **exactly 40,442** — the superseded combined-stream count from the earlier
   campaign, reproduced independently. This is a strong check that the per-cut `cut_<id>` flags
   recombine correctly.
2. **Recotag is 97.6% a subset of truthtag.** Only 976 events pass detector tagging without being a
   true CC-with-muon-FSL event. But see §1.2 before reading that as a purity measurement — most of
   it is inherited from the FV cut, not earned by the detector tagging.

### 1.2 The FV cut is silently acting as a near-total NC veto

**This qualifies the purity claim above and was found on 2026-08-04, after the runs.**

The FV cut is evaluated on `trueVtxX/Y/Z`, the WCSim *primary start point*. For an NC event there is
no primary muon, so `trueVtx` is written as a sentinel `(0, 14.4602, -168.100)` — which has
r = 168.1 cm, comfortably outside the 100 cm FV radius. **Every NC event therefore fails the FV cut
for a bookkeeping reason, not a geometric one.**

Measured on the full sample:

| | events |
|---|---:|
| NC events total | 758,396 |
| NC events passing the FV cut | **1,159 (0.15%)** |
| recotag-selected events that are NC | **41 of 41,418** |

Cross-check on one file, comparing the FV cut evaluated on `trueVtx` against the same cut evaluated
on the true GENIE vertex (`trueNuIntxVtx` + offset): for **CC** events the two agree exactly (23.5%
pass rate both ways); for **NC** events `trueVtx` passes 0.0% while the GENIE vertex passes 22.2%.
The entire discrepancy is the sentinel.

Consequences:

- For **truthtag** this is harmless — `trueCC==1` is required anyway.
- For **recotag** it matters: the stream never applies a CC requirement, so its NC rejection is
  supplied *entirely by the FV cut*, before any detector tagging happens. The 976 "recotag-only"
  events decompose exactly as **885 μ⁺ (antimuon CC) + 48 e⁻ + 2 e⁺ + 41 NC**, i.e. they are
  overwhelmingly CC events with the wrong final-state lepton, not NC contamination.
- So **this study does not demonstrate that MRD+FMV+promptPE tagging rejects NC events.** That
  question is untested here. In real data there is no `trueVtx` sentinel to lean on, so a
  data-applicable version of this selection must expect NC contamination that the MC selection
  never saw. This is the most important caveat on the reco streamline.

The `cut_<id>` boolean columns in `*__cc_*_events.parquet` are evaluated **independently** (each cut
against the full sample, not against the previous cut's survivors). The sequential cut-flow is
recovered by AND-ing them in stream order. **Do not read a single `cut_` column as a sequential
efficiency.**

---

## 2. Delayed background composition

**How the lineage behind these numbers is tracked** — what `origin_pdg` means, why `e±`
and `γ` are skipped, the join-not-zip rule, and the `-5` sentinel — is documented in
`REPORT_ccinc_v3_world_merged.md` §0.0. Read that first if you are quoting any
composition percentage.

Delayed window `t > 10000 ns`, CC-passing events only. Stage 1b
(`report_background_composition.py`) reports three ancestry axes; the physics axis is **`origin`** =
first ancestor in the lineage chain that is neither e± nor γ.

Totals: **231,664 non-neutron hits over 81,790 CC events = 2.83 hits/event** (truthtag);
**99,436 over 41,418 = 2.40 hits/event** (recotag).

| origin (first non-EM ancestor) | truthtag | per CC evt | recotag | per CC evt |
|---|---:|---:|---:|---:|
| μ⁻ | 50.65% | 1.435 | 53.58% | 1.286 |
| π⁺ | 17.27% | 0.489 | 15.37% | 0.369 |
| pure EM (no non-EM ancestor) | 13.82% | 0.391 | 15.20% | 0.365 |
| π⁰ | 7.46% | 0.211 | **3.44%** | **0.083** |
| p | 5.39% | 0.153 | 5.10% | 0.122 |
| μ⁺ | 3.24% | 0.092 | **5.27%** | **0.127** |
| π⁻ | 2.10% | 0.060 | 1.99% | 0.048 |
| K⁺ | 0.05% | 0.001 | 0.02% | 0.001 |

Cross-check on the `root` axis (generator primary) gives the same picture with π⁰ decay products
folded into π⁺/γ, as expected: μ⁻ 50.64/53.58%, π⁺ 20.63/18.70%, γ 13.82/15.14%.

**The physics result here is the π⁰ suppression.** π⁰-origin background halves under detector
tagging (7.46% → 3.44%, and 0.211 → 0.083 hits per CC event, so it is a real reduction and not a
normalisation artifact), while μ⁺ roughly doubles in fraction. MRD + promptPE tagging preferentially
keeps events with a genuine penetrating muon, which suppresses the π⁰-rich, more NC-like tail. This
is the one true purity gain from the reco streamline.

> **Standing warning:** `origin_pdg == -5` means **pure-EM chain (a gamma is the generator
> primary)**, *not* untraced. Untraced is `lineage_status != 1`. Also never histogram
> `ImmediateAncestorClass` alone — it never says "neutron", and it reads "photon" for ~40% of all
> hits because of capture gammas.

### 2.1 Composition cut

`*__bkg_compcut_impact.csv` — dropping background-dominated species (OPTICS, truthtag):

| | clusters before | removed | % removed | signal clusters lost |
|---|---:|---:|---:|---:|
| ALL | 57,326 | 7,707 | 13.44% | 534 |
| μ⁻-origin | 36,315 | 3,798 | 10.46% | 336 |
| π⁺-origin | 4,727 | 2,174 | **45.99%** | 67 |
| π⁰-origin | 2,912 | 949 | 32.59% | 78 |
| π⁻-origin | 466 | 283 | **60.73%** | 5 |
| p-origin | 1,902 | 286 | 15.04% | 36 |

A cheap handle: 13.4% of clusters removed for 534 signal clusters. Efficiency is very uneven by
species. ClusterFinder behaves nearly identically (12.87% removed, 445 signal lost).

### 2.2 Spurious / prompt clusters

`*__spurious_composition.csv`:

| | truthtag optics | truthtag cf | recotag optics | recotag cf |
|---|---:|---:|---:|---:|
| spurious clusters | 23,819 | 20,746 | 10,673 | 9,009 |
| of which prompt | 9,818 | 8,984 | 4,109 | 3,631 |
| real spurious | 14,001 | 11,762 | 6,564 | 5,378 |
| mean frac neutron | 0.554 | 0.537 | 0.590 | 0.580 |
| mean frac untraced | 0.110 | 0.119 | 0.113 | 0.121 |
| contains ≥1 neutron hit | **93.4%** | 91.5% | **96.1%** | 94.8% |
| pure background | 6.6% | 8.5% | 3.9% | 5.2% |

Most "spurious" clusters are not background at all — they contain real neutron light that failed the
≥50% single-trackID purity match. Mean untraced ~11–12% in every configuration, consistent with the
earlier lineage validation.

---

## 3. Clusters and MVA performance

Signal = neutron-dominated cluster (`dominant_class ∈ {1,2,3,4}`); background = everything else,
prompt clusters kept (`--keep-prompt-bkg`), balanced 1:1 (`--max-ratio 1.0`), event-level
train/test split, 31 features.

| | truthtag OPTICS | truthtag CF | recotag OPTICS | recotag CF |
|---|---:|---:|---:|---:|
| clusters | 57,326 | 58,380 | 27,974 | 28,412 |
| signal | 48,934 | 50,128 | 24,796 | 25,335 |
| signal fraction | 85.4% | 85.9% | **88.6%** | **89.2%** |
| background | 8,392 | 8,252 | 3,178 | 3,077 |
| after 1:1 capping (per class) | 8,392 | 8,252 | 3,178 | 3,077 |
| train / test | 13,421 / 3,363 | 13,200 / 3,304 | 5,124 / 1,232 | 4,931 / 1,223 |
| **AUC Random Forest** | 0.602 | 0.591 | 0.578 | **0.623** |
| **AUC GBT** | **0.605** | 0.585 | 0.574 | 0.612 |
| **AUC XGBoost** | 0.584 | 0.583 | 0.559 | 0.599 |
| **AUC Neural Network** | 0.599 | **0.593** | 0.571 | 0.608 |
| NN early stop | epoch 36/100 | 23/100 | 36/100 | 44/100 |

Cluster yield per CC event, from `*__bkg_cluster_yields.csv` (`dominant_class` axis, all classes
summed) — stable across both streams and both methods, which is a useful consistency check since the
two streams select very different numbers of events:

| | total/CC evt | prompt | delayed/CC evt |
|---|---:|---:|---:|
| truthtag optics | 0.701 | 16,024 | 0.505 |
| truthtag cf | 0.714 | 16,273 | 0.515 |
| recotag optics | 0.675 | 7,320 | 0.499 |
| recotag cf | 0.686 | 7,365 | 0.508 |

**All eight AUCs sit in 0.56–0.62 — the MVA is barely separating**, and the ordering is not stable:
recotag is *worse* than truthtag under OPTICS (0.578 vs 0.602) and *better* under ClusterFinder
(0.623 vs 0.591). With only ~3.1k background clusters per class after balancing in the recotag runs,
that spread is consistent with statistical noise. **This is not a resolvable difference between the
streamlines**, and it should not be quoted as one.

### 3.1 Feature separation and importance

`*__feature_separation.csv` — best single-feature separations (truthtag):

| feature | method | separation | mean neutron | mean spurious |
|---|---|---:|---:|---:|
| `fit_goodness_init` | optics | **0.147σ** | 0.3815 | 0.3346 |
| `t_window_80pct` | optics | 0.105σ | 131.9 | 363.4 |
| `fit_goodness_reco` | optics | 0.102σ | 0.3731 | 0.3396 |
| `pe_total` | clusterfinder | 0.094σ | 20.47 | 21.57 |
| `n_hits` | clusterfinder | 0.093σ | 15.53 | 16.15 |
| `sigma_t` family | optics | 0.092σ | 195.3 | 303.2 |

Recotag is nearly identical (`fit_goodness_init` 0.150σ, `fit_goodness_reco` 0.118σ,
`t_window_80pct` 0.113σ).

Importances are **flat** — no feature carries more than ~6% in any of the four trainings. The top
two are `pe_total` and `charge_bal_legacy` in every configuration except recotag/OPTICS, where
`charge_bal_legacy` and `d_wall` lead. `fit_converged` is worthless everywhere (≈0.002).
`n_hits_early` is the one feature whose importance jumps for the boosted trees (xgb 0.05–0.077 vs
rf 0.024–0.032).

**A 0.15σ best single-feature separation with flat importances is the real finding of the MVA
section: the current 31-feature set does not discriminate a neutron-capture cluster from a
non-neutron cluster at the single-cluster level in this sample.**

---

## 4. Caveats — read before quoting anything above

- **The reco streamline is not data-applicable yet.** Its FV and muon-kinematic cuts still use
  **truth** branches (`trueVtx*`, `trueFSLMomentum_*`), per the instruction that those are shared
  between streams. So it isolates the *tagging* difference on a fixed truth phase space. Making it
  fully data-applicable needs reco replacements (`MRDTrackAngle`, MRD-derived momentum, a reco tank
  vertex) — new cut code, not a config change.
- **`mrd_source: cluster` = `MRDClusterNumber > 0`**, i.e. any MRD activity under a 100%-efficiency
  assumption. An optimistic proxy, not a reconstructed track. The real MRD tracking efficiency makes
  the factor-1.97 loss *worse*, not better.
- **The clusterTime 20 ns cut is deliberately NOT applied.** `clusterTime` in these files is not
  trigger-referenced (min 39.2 ns, first-cluster median 51.4 ns, 92.7% in 40–60 ns), so `|t| ≤ 20 ns`
  keeps 0.0% of events. Applying it needs an anchor decision (~51 ns offset, or "within 20 ns of the
  event's first cluster").
- The MVA is deliberately **binary**: all non-neutron clusters are one background class. The
  composition breakdown in §2 is diagnostic only — no truth variable is in `PHYSICS_FEATURES`.
- A known pre-existing bug survives here: `frac_bg_*_of_bg` can exceed 1 (up to 25.0) because
  `n_bg_photon` counts capture gammas. It was invisible on older MC. Quote `origin_*` fractions, not
  `bg_*`.

---

## 5. Conclusion

The detector-level tagging behaves as a well-behaved but lossy substitute for truth tagging:

- **nearly pure, but largely for the wrong reason** — only 976 of 41,418 selected events are not
  true CC-with-muon, and they are mostly μ⁺ CC (885), not NC. The NC rejection comes from the FV
  cut's sentinel artifact (§1.2), not from the detector tagging, so it will not carry over to data
- **costly** — a factor 1.97 in statistics, almost entirely MRD acceptance
- **modestly cleaner at cluster level** — signal fraction 85.4% → 88.6%, π⁰-origin background halved
- **no MVA benefit** — AUCs 0.56–0.62 with the truth/reco ordering flipping sign between clustering
  methods, i.e. unresolvable at these background statistics

**The lever for improving neutron identification is features and background statistics, not the
streamline choice.** Two concrete follow-ups:

1. **More background, from a wider volume.** The tank/fmvmrd sample only simulates interactions
   inside the tank, so the training has no examples of out-of-tank interactions depositing light in
   the tank. The world/fmvmrd productionv3 sample supplies exactly that — ~77% of its events
   interact in dirt/concrete/MRD steel. See the world-volume campaign.
2. **Better features.** At 0.15σ best separation, no amount of extra training data will move the
   AUC far. New feature families (per-hit topology, charge-weighted timing shape, PMT-level
   asymmetries) are the higher-leverage direction.
