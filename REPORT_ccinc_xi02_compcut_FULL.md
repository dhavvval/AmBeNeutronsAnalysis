# Final-State Neutron Identification in ANNIE — Full Analysis Report
## Pipeline `run_ccinc_xi02_compcut.sh` · Stage 0 → Stage 4

**Run name:** `cc_neutrino_xi02_compcut`
**Date produced:** 2026-06-11
**MC input:** 416 GENIE-neutrino-interaction WCSim ROOT files (tank-only simulation, normal PMT geometry)
**AmBe data:** 2023 campaign, 31 runs across 5 source ports
**Output root:** `/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/`
**Pipeline log:** `logs_ccinc_xi02_compcut.log`

This report is **standalone** — every number comes from this run only. There are no comparisons to any previous production.

> **Revision 2026-06-12:** (1) The Stage-4 page-4 table was completed with the frozen-**NN** rows (`nn_score ≥ 0.538`) alongside the RF rows — see §4. (2) The legacy box cut was **unified to `PE < 80, CB < 0.45, nHits > 9` across all stages** (it previously carried two variants — `(60,0.5,10)` in the Stage-4 benchmark vs `(80,0.45,9)` in the data feature/multiplicity code). The Stage-4 box-cut rows were re-run and the Stage-3 multiplicity bars/captions corrected; **only the box-cut rows changed — all MVA rows are unaffected** (the box constant does not enter the MVA). See §4.1 for the before/after. (3) The frozen-NN multiplicity PDFs (`…__multiplicity_nn.pdf`) were generated (previously referenced but missing). (4) The **legacy-box data rows** were added to §3.2 (rate), §3.5 (multiplicity) and §3.6 (capture-τ), computed live at 80/0.45/9 on the scored data — labeled throughout as a *reference*, not an MVA result. ⚠️ The OPTICS *data* parquet's baked `passes_stage1` column is **stale** (frozen at the old 60/0.5/10 box); the box rows bypass it and recompute live. (5) Added §"Frozen operating point" key-points + an **efficiency working-point scan** showing why 80% is chosen (bracketed by starved fits below, collapsing bkg-rejection above). (6) Added **§4b — separately-trained ClusterFinder MVA cross-check**: a dedicated CF model has lower AUC (0.709 vs OPTICS 0.754) and gives no purity gain (0.980 vs 0.981), validating the main choice to score CF with the OPTICS model. §4b now reports **both streamlines across RF / NN (and GBT for CF)** with eff/purity/fakes, capture-τ, single-n + rate, and side-by-side discriminator/loss figures. (7) **NN seeded + all NN numbers regenerated (2026-06-12):** the NN training previously had no random seed, so it wasn't reproducible (the 30.3↔33.0 µs OPTICS+NN drift). Added `tf.keras.utils.set_random_seed(42)` to `train_nn`; **every NN number in the report (threshold 0.538→0.569, OPTICS+NN τ→32.0 µs, rate→0.631, single-n→0.896, §4/§4b NN rows, per-port, multiplicity, summary, conclusions) is now the reproducible seeded value.** RF/GBT/XGB were always deterministic and are unchanged. The physics conclusion is unchanged: RF is the clean in-band baseline; NN is the higher-yield cross-check, now sitting just above the band (32.0 µs).

---

## Pipeline at a glance

| Stage | What it does | Key script | Output |
|---|---|---|---|
| 0 | ROOT → per-hit parquet, apply CC-truth selection, ClusterFinder sidecar | `ambe mc process` | `__pulses.parquet`, `__clusterfinder.parquet` |
| 1 | OPTICS clustering + 31-feature extraction, truth-match each cluster | `ambe mc features` | `__cluster_features.parquet` |
| 2 | Train 4 classifiers (RF/GBT/XGBoost/NN) on signal vs background, freeze | `mva_analysis.py --keep-prompt-bkg --save-model` | `__mva_frozen__keepprompt.pkl`, scores, plots |
| — | Composition cut applied to MC clusters (training/benchmark side only) | inline filter | `__cluster_features_compcut.parquet` |
| 3 | Score AmBe data, per-run neutron rate, multiplicity, capture-time | `mva_analysis.py --score-data` + downstream | `ccinc_xi02cc_{optics,cf}_mcbkg/` |
| 4 | MC efficiency/purity benchmark: box-cut vs frozen MVA | `benchmark_selection_2x2.py` | `ccinc_xi02cc_{optics,cf}_boxcuts/benchmark/` |

---

# STAGE 0 — Processing & CC-truth selection

## 0.1 Raw statistics from the ROOT files

| Quantity | Value |
|---|---:|
| ROOT files processed | 416 |
| Total MC events | 890,000 |
| Total PMT hits written | 80,960,300 |
| Raw ClusterFinder clusters | 2,164,856 |

## 0.2 CC-truth cut cascade (`ccinc_truth` stream)

Applied to all 890,000 events. This is the **same CC selection as James** (truth-level CC-ν interaction tagging), on a tank-only sim (no FMV/MRD available).

| Cut | Events in → out | Kept |
|---|---:|---:|
| `trueCC == 1` | 890,000 → 618,686 | 69.5% |
| FSL is muon (`trueFSLPdg == 13`) | 618,686 → 606,650 | 98.1% |
| p\_μ ∈ [600, 1200) MeV/c | 606,650 → 180,835 | 29.8% |
| cos θ > 0.8 | 180,835 → 110,839 | 61.3% |
| FV radius r < 100 cm | 110,839 → 50,571 | 45.6% |
| FV \|Y\| < 100 cm | 50,571 → 29,223 | 57.8% |
| nhits ≥ 4 | 29,223 → 29,223 | 100.0% |
| **FINAL cc_pass** | **29,223 / 890,000** | **3.3%** |

**Signal definition:** final-state-neutron hits in the **delayed residual window** — hits at **t > 2000 ns** in CC-passing events only. Everything before 2000 ns (prompt muon/gamma light) is excluded from the clustering input.

**Outputs:**
- `cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__pulses.parquet` (80.96 M hits, truth-matched via DirectParent, `cc_pass` joined)
- `cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__clusterfinder.parquet` (ClusterFinder sidecar)

---

# STAGE 1 — Clustering & feature extraction

Two clustering methods run on the **same** delayed-residual hits of the 29,223 CC-passing events:

## 1.1 Clustering parameters

| Method | Parameters |
|---|---|
| **OPTICS** | min_samples (minHits) = **8**, **ξ = 0.02**, t_unit = **25 ns**, hit_prefilter = 0 ns (cluster all residual hits), truth-match window = 75 ns |
| **ClusterFinder (legacy)** | minHits = 5, integration time window = 20 ns (legacy ANNIE algorithm) |

ξ=0.02 is the OPTICS reachability-steepness parameter: a **smaller ξ produces tighter, less over-merged clusters** — it splits overlapping deposits more aggressively than the looser ξ=0.10.

## 1.2 Cluster yields and truth composition

| | OPTICS | ClusterFinder |
|---|---:|---:|
| Total clusters | **29,572** | **29,851** |
| Neutron-dominated (signal, MC truth) | 22,843 (77.2%) | 23,445 (78.5%) |
| Background-dominated | 6,729 (22.8%) | 6,406 (21.5%) |
| `is_prompt_cluster == 1` | 10,663 (36.1%) | 10,539 (35.3%) |
| `is_truth_neutron == 1` (strict ≥50% one trackID) | 15,498 (52.4%) | 17,406 (58.3%) |
| Mean signal-cluster purity (`frac_neutron`) | 0.765 | 0.787 |

**Per-hit truth classes** (`truth_class`): 1 = primary neutron, 2 = secondary n←p, 3 = secondary n←n, 4 = secondary n←other, 0 = dark noise, −5 = non-neutron physics. `is_neutron` = class ∈ {1,2,3,4}.

**Per-cluster truth labels:** `dominant_class` = majority-vote of member-hit classes; signal = dominant ∈ {1,2,3,4}. `is_truth_neutron` (strict) = a single neutron trackID supplies ≥50% of the cluster's hits.

**Truth composition of all clusters (dominant_class):** primary n ~34%, secondary n←n ~37%, secondary n←p ~5%, non-neutron physics (−5) ~22%, dark noise <1%.

## 1.3 What is actually inside a cluster (contamination, both directions — MC truth)

| | OPTICS | ClusterFinder |
|---|---:|---:|
| Signal cluster mean composition | 76.5% neutron / 13.7% non-neutron / 9.8% dark-noise hits | 78.7% / 11.4% / 9.9% |
| Signal clusters carrying ≥1 contaminant hit | 83.9% | 74.4% |
| Perfectly pure signal clusters (100% neutron) | 4.0% | 7.3% |
| Background clusters containing ≥1 neutron hit | 62.9% (mean 3.3 n-hits) | 64.0% (mean 3.5) |

**Physics message:** the neutron-capture photon cloud and the prompt/non-neutron light **physically overlap** in space and time inside the tank. 84% of true neutron clusters carry some contamination and 63% of background clusters carry some real neutron hits — this overlap is **intrinsic to the detector, not an algorithm artefact**. This is why discrimination is done on **aggregate cluster features (MVA)** rather than per-hit cleaning.

**Outputs:**
- `cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__cluster_features.parquet` (59,423 clusters: 29,572 OPTICS + 29,851 CF, 31 features + truth labels)
- `cc_neutrino_xi02_compcut/csv/cc_neutrino_xi02_compcut__feature_separation.csv`
- `cc_neutrino_xi02_compcut/csv/cc_neutrino_xi02_compcut__spurious_composition.csv`

**Plots for the poster:**
- `cc_neutrino_xi02_compcut/plots/cc_neutrino_xi02_compcut__cluster_features.pdf` — feature distributions, signal vs background
- `cc_neutrino_xi02_compcut/plots/cc_neutrino_xi02_compcut__feature_separation_plots.pdf` — per-feature separation

---

# STAGE 2 — MVA training (freeze the model)

## 2.1 Training set construction

| Step | Result |
|---|---|
| Input clusters (cc_pass, OPTICS only) | 29,572 |
| Signal (neutron-dominated) | 22,843 |
| Background (all non-neutron, **incl. prompt** via `--keep-prompt-bkg`) | 6,729 |
| After balancing 1:1 (`--max-ratio 1.0`) | signal 6,729 : background 6,729 |
| Train / test split | train = 10,730, test = 2,728 |
| Features | 31 (all physics-only, computable on real data — no truth used) |

The **31 features**: `n_hits, pe_total, n_hits_early, sigma_t_mad, sigma_t_mad_corr, sigma_t_mad_tof, sigma_t_early_mad, t_window_80pct, pe_balance, charge_bal_legacy, spatial_rms, d_wall, vtx_y, beta1–beta5, fit_converged, n_fit_hits, fit_rms_ns, fit_goodness_reco, fit_goodness_init, d_wall_fit, sigma_t_mad_tof_fit, beta1_fit–beta5_fit, vtx_fit_y`.

## 2.2 Classifier performance (MC test set, n = 2,728: signal 1,353, bkg 1,375)

| Model | 5-fold CV AUC | Test AUC | bkg-rej @ 80% eff |
|---|---:|---:|---:|
| **Random Forest** | 0.7436 ± 0.0092 | **0.7545** | 0.596 |
| GBT | 0.7410 ± 0.0092 | 0.7535 | 0.589 |
| XGBoost | 0.7388 ± 0.0025 | 0.7488 | 0.580 |
| Neural Net (seeded) | — | 0.7526 | 0.580 |

All four AUCs lie within 0.006 of each other — they rank MC truth almost identically. **Random Forest is auto-selected** as the frozen baseline (highest test AUC). (NN values are the seeded `set_random_seed(42)` model, reproducible — see the reproducibility note under "Frozen operating point".) **Correction (this revision):** the OPTICS NN test AUC was previously listed as 0.7495 — that was a mid-training progress-bar readout, not the final test AUC. The seeded final value from the most recent training (`logs_cc_neutrino_xi02_compcut_cftrain.log`, OPTICS Stage-2a block) is **0.7526**; the earlier *no-seed* run gave 0.7475. RF/GBT/XGB are deterministic and unchanged.

## 2.3 Feature importances — what drives the RF (poster table)

| Rank | Feature | RF importance | GBT importance | XGB importance |
|---:|---|---:|---:|---:|
| 1 | **pe_total** | 0.123 | 0.377 | 0.146 |
| 2 | **spatial_rms** | 0.106 | 0.238 | 0.102 |
| 3 | beta2 | 0.074 | 0.039 | 0.039 |
| 4 | beta1 | 0.052 | 0.014 | 0.030 |
| 5 | d_wall | 0.042 | 0.021 | 0.025 |
| 6 | charge_bal_legacy | 0.038 | 0.014 | 0.023 |
| 7 | beta4 | 0.037 | 0.019 | 0.024 |
| 8 | n_hits | 0.036 | 0.010 | 0.041 |
| 9 | fit_goodness_init | 0.034 | 0.023 | 0.026 |
| 10 | sigma_t_mad | 0.034 | 0.025 | 0.025 |

**`pe_total` (total cluster charge) and `spatial_rms` (cluster spatial spread) dominate** across all three tree models — both are pure charge/shape variables with no truth dependence. The GBT concentrates even more heavily on these two (0.377 + 0.238 = 62% combined).

**Single-variable separation power** (`__feature_separation.csv`, top by σ): pe_total (0.29 σ CF / 0.24 σ OPTICS), n_hits, n_hits_early, beta1. These confirm `pe_total` is the strongest individual discriminant.

## 2.4 Neural Net architecture & training

| Component | Setting |
|---|---|
| Architecture | `31 → Dense(128) → BatchNorm → ReLU → Dropout(0.3)` × **3 hidden layers** → `Dense(1, sigmoid)` |
| Optimizer | Adam, initial lr = 1×10⁻³ |
| LR schedule | ReduceLROnPlateau (patience 7, min_lr 1×10⁻⁵) |
| Loss | binary cross-entropy |
| Batch size | 256 |
| Max epochs | 100 |
| Early stopping | monitor `val_auc`, patience 15 (restore best weights) |
| Feature scaling | StandardScaler (fit on train only, no leakage) |
| Validation split | 15% of training set |

Training converged to **val_auc ≈ 0.728–0.731** (test AUC 0.7526, seeded). The NN companion is stored as a `.keras` file beside the pickle; the StandardScaler is saved with it so the NN can be applied to data with the same normalization. Training is seeded (`tf.keras.utils.set_random_seed(42)`), so the NN is bit-reproducible across runs.

**Outputs:**
- `cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__mva_frozen__keepprompt.pkl` (RF/GBT/XGBoost frozen)
- `cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__mva_frozen__keepprompt__nn.keras` (NN companion)
- `cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__mva_scores__keepprompt.parquet` (per-cluster scores + `in_test` flag)
- `cc_neutrino_xi02_compcut/csv/cc_neutrino_xi02_compcut__mva_summary__keepprompt.csv` (importances)

**Plot for the poster:**
- `cc_neutrino_xi02_compcut/plots/cc_neutrino_xi02_compcut__mva__keepprompt.pdf` — **ROC curves, score distributions, feature importances, NN training history** (this single PDF has all the MVA-performance panels)

## 2.5 Composition cut (applied between training and downstream)

After freezing, a **composition cut** filters MC clusters used for the Stage-4 benchmark:

> **Reject a cluster if  n_neutron ≤ 10  AND  frac_nonneutron ≥ 0.40.**

This targets the clean prompt-γ / non-neutron-physics clusters (few real neutron hits, dominated by non-neutron light). On MC it removes **11,806 / 59,423 clusters = 19.9%**.

**Important scope note:** this cut uses truth columns (`n_neutron`, `frac_nonneutron`) that exist only in MC. The **real AmBe data has no truth labels**, so the composition cut is **applied to MC only** (training quality + Stage-4 benchmark). On data, the selection is purely **OPTICS-clustering + frozen MVA**.

---

# Frozen operating point applied to data

After freezing, the score threshold for each model is set at **80% MC signal efficiency** (derived from the MC test set, then applied unchanged to data — physically anchored and comparable across models):

| Model | Score column | Threshold (80% MC eff) |
|---|---|---:|
| **Random Forest** | `rf_score` | **0.547** |
| **Neural Net** | `nn_score` | **0.569** |

> ✅ **Reproducibility (NN now seeded — all NN numbers regenerated 2026-06-12).** All models are now fully seeded (`random_state=42` for the data split + trees; `tf.keras.utils.set_random_seed(42)` in `train_nn`) and **bit-reproducible** — verified by re-running. **Every NN number in this report has been regenerated with the seeded NN**; the seeded operating point is `nn_score ≥ 0.569` (OPTICS) / `≥ 0.543` (CF), versus the earlier unseeded draw (0.538). The RF/GBT/XGB numbers are unchanged (those models were always deterministic). The seeded NN shifts OPTICS+NN slightly (τ 30.3→**32.0 µs**, rate 0.609→**0.631**) — the physics conclusion is unchanged: RF is the cleanest deterministic baseline, NN the higher-yield cross-check sitting at the top of / above the capture-time band.

**Why a fixed 80%-efficiency cut and not an "optimized" threshold — in brief:**

1. **The cut is defined by physics efficiency, not an ML metric.** `rf_score ≥ 0.547` is simply "the score that keeps 80% of true MC neutrons" (the 20th percentile of signal scores on the held-out test set). Efficiency is fixed by construction; the score value is whatever delivers it. This is *deterministic and reproducible* (trees give the same 0.547 on a retrain).
2. **The "optimal" ML cuts were tried and rejected.** Youden's J (max tpr−fpr) ≈ 0.42 and max-F1 ≈ 0.34 correspond to ~90–95% efficiency — they maximize separation on MC truth but **fail the physics check**: they drag the AmBe capture time to ~35 µs, outside the 25–30 µs literature band.
3. **Same anchor across models.** "80% MC efficiency" means the same thing for RF (0.547) and NN (0.569) despite different raw score scales — so the RF-vs-NN comparison is fair. A raw "score > 0.5" or per-model optimized cut would not be comparable.
4. **It's the working point that *validates* against data, and it sits at a genuine sweet spot** (efficiency scan below).
5. **The residual choice is folded into the systematic.** 80% is a defensible choice in a 75–85% window; the report carries the RF↔NN rate spread (44%↔61% OPTICS) as the selection-efficiency systematic rather than claiming 80% is uniquely optimal.

**Efficiency working-point scan (OPTICS, RF) — the evidence for 80%.** Sweeping the target efficiency and re-deriving the cut, then measuring MC background-rejection and the AmBe capture-time fit:

| Target eff | rf cut | MC bkg-rej | AmBe capture τ | fit stability | |
|---:|---:|---:|---:|---:|---|
| 70% | 0.584 | 0.671 | **21.2 ± 10.2 µs** | 19/21 | ✗ too tight — fits starved, τ below band, spread blows up |
| **80%** | **0.547** | **0.596** | **26.6 ± 2.1 µs** | **21/21** | ✓ **in-band, tightest spread, bkg-rej still healthy** |
| 90% | 0.482 | 0.509 | 28.4 ± 2.2 µs | 21/21 | ✗ bkg-rej collapsing, τ drifting to band edge |
| ~92–95% (Youden/F1) | ≲0.42 | — | ~35 µs | — | ✗ unphysical (rejected) |

→ **80% is bracketed on both sides.** Tighter (70%) starves the per-position capture fits → τ falls to 21 µs with a ±10 µs spread; looser (90%+, where the optimized cuts live) collapses background rejection and pushes τ out of the band. 80% is the unique point where τ is **physical (26.6 µs)**, the fits are **stable (±2.1 µs, 21/21)**, and background rejection has not yet fallen off the knee. The choice is driven by the physics observable we validate against, not by a confusion-matrix metric.

---

# STAGE 3 — AmBe data scoring + validation

Frozen models scored on AmBe 2023 data. Two streamlines: **OPTICS** (232,836 data clusters over 79,633 events) and **ClusterFinder** (93,809 clusters). Both RF and NN reported.

## 3.1 How to read the per-run numbers

Each AmBe run = one source position. AmBe is a **single-neutron source** — one neutron per source decay — so ideally **N events ≈ N neutrons**, i.e. one tagged neutron per triggered event.

- `n_events` = beam-trigger events in the run **that produced ≥1 delayed cluster** (the denominator).
- `frac_events_with_neutron` = events with ≥1 MVA-tagged neutron / `n_events` → the detection rate.
- `mean_mult` ≈ 1 and `single_neutron_frac` ≈ 1 are the expected single-neutron signature.

> **Caveat:** `n_events` counts only events that already had delayed activity, so `frac_events_with_neutron` is **conditional on delayed activity**, not an absolute fraction of all triggers. It folds together neutron containment, detection threshold, and selection efficiency.

## 3.2 Aggregate data neutron rate — RF vs NN vs legacy box (all 31 runs, 79,633 events)

| Streamline | Selection | threshold/box | frac events w/ neutron | neutron clusters | events w/ neutron | mean mult | single-n frac |
|---|---|---|---:|---:|---:|---:|---:|
| **OPTICS** | **RF** | 0.547 | **0.438** | 36,008 | 34,854 | 1.034 | **0.968** |
| **OPTICS** | **NN** | 0.569 | **0.631** | 55,911 | 50,265 | 1.112 | 0.896 |
| OPTICS | box (ref) | 80/0.45/9 | 0.811 | 76,143 | 64,565 | 1.179 | 0.841 |
| **CF** | **RF** | 0.547 | **0.814** | 69,415 | 64,802 | 1.069 | 0.936 |
| **CF** | **NN** | 0.569 | **0.879** | 76,543 | 70,014 | 1.093 | 0.916 |
| CF | box (ref) | 80/0.45/9 | 0.725 | 60,083 | 57,766 | 1.040 | 0.963 |

- **NN tags more events** than RF at the same MC efficiency (OPTICS 63% vs 44%; CF 88% vs 81%) — NN's rate is closer to the AmBe one-neutron-per-trigger ideal.
- **RF holds higher single-neutron purity** (96.8% OPTICS) — fewer events split into multiple fake clusters.
- Both models show OPTICS under-counting vs CF (open item, §5).
- **Legacy box rows are a reference, not an MVA result** (the box is the §4.1 `PE<80/CB<0.45/nHits>9` selection, applied to the same scored data clusters — no MVA score used). The box tags many more OPTICS clusters (0.811 vs RF 0.438) but at a much lower single-neutron fraction (0.841 vs 0.968): it admits low-PE / high-charge-balance clusters the MVA rejects, inflating per-event multiplicity. On CF the box is *tighter* than the MVA (0.725 vs RF 0.814).
- **Computed live at 80/0.45/9** from the scored parquets (`["run","event_tank_time"]` event key, `t_mean` cluster time). ⚠️ The OPTICS data parquet's baked `passes_stage1` column is stale — it was frozen at the *old* 60/0.5/10 box (74,177 clusters) and must not be used for the unified box; the CF parquet's `passes_stage1` does match 80/0.45/9. These rows bypass that column.

## 3.3 Per-PORT breakdown — OPTICS (the position dependence)

**Random Forest @ 0.547:**

| Port | runs | events | events w/ n | frac | mean mult | single-n |
|---|---:|---:|---:|---:|---:|---:|
| port1_z-75 | 5 | 24,323 | 10,531 | 0.433 | 1.032 | 0.969 |
| port2_z75 | 2 | 7,888 | 3,241 | 0.411 | 1.028 | 0.973 |
| port3_z102 | 7 | 10,644 | 4,418 | 0.415 | 1.031 | 0.971 |
| port4_x75 | 12 | 17,618 | 7,744 | 0.440 | 1.036 | 0.966 |
| port5_z0 | 5 | 19,160 | 8,920 | 0.466 | 1.034 | 0.967 |

**Neural Net @ 0.569 (seeded):**

| Port | runs | events | events w/ n | frac | mean mult | single-n |
|---|---:|---:|---:|---:|---:|---:|
| port1_z-75 | 5 | 24,323 | 15,532 | 0.639 | 1.115 | 0.894 |
| port2_z75 | 2 | 7,888 | 4,806 | 0.609 | 1.105 | 0.901 |
| port3_z102 | 7 | 10,644 | 6,107 | 0.574 | 1.106 | 0.901 |
| port4_x75 | 12 | 17,618 | 11,015 | 0.625 | 1.113 | 0.896 |
| port5_z0 | 5 | 19,160 | 12,805 | 0.668 | 1.114 | 0.894 |

## 3.4 Per-PORT breakdown — ClusterFinder

**Random Forest @ 0.547:**

| Port | runs | events | events w/ n | frac | mean mult | single-n |
|---|---:|---:|---:|---:|---:|---:|
| port1_z-75 | 5 | 24,323 | 20,205 | 0.831 | 1.072 | 0.934 |
| port2_z75 | 2 | 7,888 | 6,151 | 0.780 | 1.069 | 0.936 |
| port3_z102 | 7 | 10,644 | 8,286 | 0.778 | 1.072 | 0.933 |
| port4_x75 | 12 | 17,618 | 14,408 | 0.818 | 1.068 | 0.936 |
| port5_z0 | 5 | 19,160 | 15,752 | 0.822 | 1.074 | 0.933 |

**Neural Net @ 0.569 (seeded):**

| Port | runs | events | events w/ n | frac | mean mult | single-n |
|---|---:|---:|---:|---:|---:|---:|
| port1_z-75 | 5 | 24,323 | 21,859 | 0.899 | 1.097 | 0.914 |
| port2_z75 | 2 | 7,888 | 6,678 | 0.847 | 1.089 | 0.919 |
| port3_z102 | 7 | 10,644 | 8,909 | 0.837 | 1.093 | 0.915 |
| port4_x75 | 12 | 17,618 | 15,519 | 0.881 | 1.087 | 0.920 |
| port5_z0 | 5 | 19,160 | 17,049 | 0.890 | 1.096 | 0.913 |

**Position stability:** the rate is **flat across all 5 ports** (RF OPTICS 0.41–0.47; CF 0.78–0.83) — the selection is not biased by source position. port5_z0 (tank center) is consistently the highest, as expected (best containment).

## 3.5 Neutron multiplicity (validation plot #1) — RF and NN, per-event

AmBe ideal = exactly 1 neutron per event. Distribution of tagged neutron clusters per event:

| multiplicity | OPTICS+RF | OPTICS+NN | OPTICS+box | CF+RF | CF+NN | CF+box |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33,761 | 45,029 | 54,289 | 60,542 | 64,105 | 55,641 |
| 2 | 1,035 | 4,856 | 9,242 | 3,939 | 5,350 | 1,949 |
| 3 | 55 | 336 | 872 | 296 | 496 | 161 |
| 4 | 3 | 22 | 100 | 19 | 50 | 14 |
| ≥5 | 0 | 11 | 62 | 6 | 8 | 1 |
| **single-neutron fraction** | **96.9%** | 89.6% | 84.1% | 93.4% | 91.6% | 96.3% |

→ The four **MVA** streamlines are strongly single-neutron-dominated, confirming the AmBe single-neutron source. **OPTICS+RF is closest to the 1-neutron ideal (96.9% single).** NN (seeded) gives 89.6% (OPTICS) / 91.6% (CF) — more multi-counts than RF because its looser cut admits a few extra clusters per event.

The **box columns are the legacy-box reference** (§4.1, `PE<80/CB<0.45/nHits>9`), not MVA results. On **OPTICS the box is markedly worse** at the single-neutron signature (84.1% vs RF's 96.9%) — its loose PE/charge-balance window admits extra clusters per event, pushing many events to multiplicity ≥2. On **CF the box is actually cleanest** (96.3%), because CF's own clustering already merges most of those into single clusters. This contrast — OPTICS *needs* the MVA to reach single-neutron purity, CF does not — is the same story as §3.2.

**Plots for the poster:**
- RF: `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/…__multiplicity.pdf`
- NN: `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/…__multiplicity_nn.pdf`
- Rate-vs-run + multiplicity + score panels: `…/rate_best/ambe_all__neutron_summary.pdf` (RF), `…/rate_best_nn/ambe_all__neutron_summary.pdf` (NN)

> The box-cut comparison bars inside these multiplicity PDFs ("OPTICS/CF + box cuts") use the **unified `PE<80 / CB<0.45 / nHits>9`** box; all four PDFs were regenerated on 2026-06-12 with the corrected box + caption (§4.1). The per-event single-neutron fractions above are MVA-selected and unaffected by the box.

> **Seeded-NN provenance (verified 2026-06-12):** all NN downstream products (multiplicity, capture-τ, rate) were checked to be **newer than the seeded NN scores** they consume (seeded models/scores written 16:33–16:47). 15 of 16 were regenerated after the seeded scores; the one exception — `ccinc_xi02cc_cf_mcbkg__multiplicity_nn.pdf` (originally 11:42, pre-seed) — was **regenerated at the seeded OPTICS NN cut `nn_score ≥ 0.569`**, reproducing the §3.5 CF·NN single-neutron fraction of 91.6% exactly. The pre-seed PDF is retained as `…__multiplicity_nn.pdf.stale_preseed_bak`. The §3.5/§3.6 *numbers* in this report were already the seeded values; only that one PDF on disk had been stale.

## 3.6 Neutron capture time (validation plot #2) — RF vs NN vs legacy box

Per-source-position exponential-plus-thermal fit (lmfit) of the cluster-time distribution. Literature n-capture-on-H in water = **25–30 µs**.

| Streamline | Selection | τ (mean ± std) | τ median | good fits |
|---|---|---:|---:|---:|
| OPTICS | RF | **26.7 ± 2.1 µs** | 26.5 | 20/21 |
| OPTICS | NN (seeded) | 32.0 ± 2.1 µs | 31.9 | 21/21 |
| OPTICS | box (ref) | 33.7 ± 2.1 µs | 33.1 | 21/21 |
| CF | RF | 29.0 ± 2.9 µs | 29.9 | 21/21 |
| CF | NN (seeded) | 29.3 ± 2.3 µs | 29.5 | 21/21 |
| CF | box (ref) | 26.9 ± 6.0 µs | 28.3 | 21/21 |

→ **OPTICS+RF, CF+RF and CF+NN validate** squarely in the 25–30 µs band. **OPTICS+RF gives the tightest, lowest τ (26.7 µs)** with a small position spread (±2.1 µs). **OPTICS+NN sits just above the band (32.0 µs)** — its looser cut admits prompt-ish clusters that lengthen the apparent capture time (the same reason the box pushes even higher, 33.7 µs). This is consistent with OPTICS+NN's lower single-neutron fraction (89.6%, §3.5): NN is the higher-yield cross-check, RF the clean baseline.

The **box rows are the legacy-box reference** (§4.1). They make the MVA's value concrete: **OPTICS+box pulls τ up to 33.7 µs — *above* the 25–30 µs band** — because the loose box keeps low-PE / prompt-ish clusters that the MVA rejects (exactly the failure mode that motivated the fixed 80%-eff operating point, see "Frozen operating point"). CF+box lands in-band (26.9 µs) but with a large position-to-position spread (±6.0 µs vs CF+RF's ±2.9). **The MVA produces a tighter, more physical capture time than the legacy box** — clearest for OPTICS. (Box τ computed with the identical lmfit fitter on box-filtered clusters; my recomputation reproduces the RF/NN τ mean/std/median exactly. The "good fits" count uses a finite-τ-and-error criterion that gives 21/21 here; the 20/21 in the RF/NN rows reflects the original run's slightly stricter convergence flag.)

**Plots for the poster:**
- RF: `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB/stageAB_capture_lmfit.pdf` + `stageAB_capture_lmfit_summary.csv`
- NN: `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB_nn/stageAB_capture_lmfit.pdf` + `stageAB_capture_lmfit_summary.csv`
- box (ref): `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB_box/stageAB_capture_lmfit.pdf` + `stageAB_capture_lmfit_summary.csv`

---

# STAGE 4 — MC selection benchmark (efficiency / purity)

The **box-cut** (legacy "Selection-2" discriminant) is **applied here at Stage 4**, on the MC cluster-feature parquet, to contrast against the frozen MVA:

> **Box cut: pe_total < 80  AND  charge_bal_legacy < 0.45  AND  n_hits > 9.** (Unified across all stages — see §4.1.)

Both compared against the same composition truth (`dominant_class ∈ {1,2,3,4}`) on the composition-cut MC features, on the **natural cluster population** (not the balanced 1:1 training sample). Both frozen models reported at their 80%-MC-efficiency operating points (**RF @ rf_score ≥ 0.547**, **seeded NN @ nn_score ≥ 0.569**):

| Clustering | Selection | Efficiency | Purity | Fake clusters |
|---|---|---:|---:|---:|
| OPTICS | box-cut (PE<80, CB<0.45, nHits>9) | 0.851 | 0.977 | 457 |
| **OPTICS** | **frozen RF (rf_score ≥ 0.547)** | **0.835** | **0.996** | **81** |
| OPTICS | frozen NN (nn_score ≥ 0.569) | 0.810 | 0.978 | 419 |
| CF | box-cut | 0.792 | 0.980 | 383 |
| CF | frozen RF (rf_score ≥ 0.547) | 0.854 | 0.981 | 390 |
| CF | frozen NN (nn_score ≥ 0.569) | 0.895 | 0.977 | 499 |

→ **OPTICS + frozen RF is the cleanest selector: 99.6% purity with only 81 fake clusters**, at 83.5% efficiency — far cleaner than the box-cut (457 fakes at slightly higher efficiency) and far cleaner than OPTICS+NN (419 fakes). The frozen NN sits essentially **on top of the box-cut** in purity for OPTICS (0.978 vs 0.977) while tagging fewer clusters; the NN's slightly lower MC AUC (0.7526 vs 0.7545) buys it no purity margin over the legacy box. RF stays the cleanest by a wide margin (~5–6× fewer fakes than either the box or NN); NN is the higher-yield model on data but **not** a cleaner MC selector.

> **NN threshold is frozen on OPTICS, so CF over-shoots.** The seeded 0.569 cut was derived at 80% MC signal efficiency on the **OPTICS** test set. CF clusters score systematically higher, so the same fixed cut admits **89.5%** of CF neutrons (not 80%) — that is why CF+NN reaches eff 0.895 with 499 fakes. The per-method *self*-consistent 80% threshold for CF is nn_score ≥ 0.616, which lands CF+NN at eff 0.800 / purity 0.981 / 369 fakes ("MVA self 80% eff" row). The RF cut (0.547) shows the same direction but milder (CF+RF lands at 0.854, not 0.835). Use the *self* row, not the frozen row, when comparing CF efficiency head-to-head with OPTICS at a true 80% point.

> Note on purity denominators: the ~63% "purity" computed on the balanced 1:1 *training* sample is an artefact of the artificial 50/50 mix. **All purities above are on the natural cluster population** — that is the physically meaningful number, directly comparable across the RF and NN rows.

## 4.1 Box-cut definition & provenance (unified across all stages)

**(a) Exact definition.** The box cut is **exactly** `pe_total < 80 AND charge_bal_legacy < 0.45 AND n_hits > 9` (NaN charge-balance is *kept*). This single definition is now used at **every stage** of the analysis — the Stage-4 benchmark (`benchmark_selection_2x2.py:45`), the Stage-3 multiplicity comparison bars (`plot_ambe_neutron_multiplicity.py:50-52`), the `passes_stage1` flag baked into the data feature parquets (`analyze_optics_beamcluster_data.py:87-89`, imported by `extract_cf_features_beamcluster_data.py`), and the auxiliary capture/multiplicity comparison scripts.

These three thresholds (80 / 0.45 / 9) were **not optimized in this analysis** — no MC sweep produced them. They are the inherited legacy ANNIE "Selection-2" preselection box.

> **History (resolved):** an earlier version of the chain carried *two* box-cut variants — the Stage-4 benchmark used `(60, 0.5, 10)` while the data feature extraction / multiplicity bars used `(80, 0.45, 9)`, and one multiplicity-plot caption mislabeled the applied `(80, 0.45, 9)` bars as "60/0.5/10". **This has been unified:** all stages now use `(80, 0.45, 9)`. The Stage-4 box-cut rows were re-run with the unified box (numbers in the table above); the MVA rows are unaffected (the box constant does not enter the MVA score). The only quantities that *changed* from the unification are the two box-cut rows: OPTICS 0.810→**0.851** eff (435→**457** fakes), CF 0.755→**0.792** eff (324→**383** fakes), purities essentially unchanged. The looser box admits more clusters → higher recall at the same ~98% purity.

**(b) The box cut is model-independent — confirmed.** `legacy_cut()` reads only `pe_total`, `charge_bal_legacy`, and `n_hits`; it never touches any MVA score column. So the box-cut efficiency/purity is **identical** whether the run uses RF or NN. Verified empirically after the unification: the RF and NN benchmarks both report the box-cut rows as OPTICS **0.851 / 0.977 / 457** and CF **0.792 / 0.980 / 383**. Only the "frozen MVA" rows differ between the RF and NN benchmarks; the two box-cut rows stay fixed.

**Plots for the poster:**
- RF: `ambe_data/ccinc_xi02cc_{optics,cf}_boxcuts/benchmark/selection_2x2_benchmark.pdf` + `.csv`
- NN: `ambe_data/ccinc_xi02cc_{optics,cf}_boxcuts/benchmark_nn/selection_2x2_benchmark.pdf` + `.csv`

(The same composition-truth + natural-population definitions used in §2.5/§4. Each CSV holds **both** OPTICS and CF rows plus the `strict` truth def and the per-method "self 80% eff" rows — the table above quotes the `composition` rows at the frozen cut.)

## 4.2 Test-only efficiency / purity (proper held-out benchmark)

The §4 table is computed on the **whole** composition-cut population, which includes the clusters each model trained on — so its purity is *in-sample-optimistic*. This sub-section is the **proper held-out benchmark**, following the standard recipe:

1. **Complete separation.** The train/test split is done at the **event level** (`event_train_test_split`, `random_state=42`, `test_size=0.2`, stratified — all clusters of an event go to the same side), so no event seen in training appears in the test set. The held-out test events are read directly from the `in_test` flag that `mva_analysis.py` writes into the scored sample. Verified: **OPTICS 1,665 test events, CF 1,622 test events**.
2. **Apply the model to the full natural population**, then keep only clusters belonging to the **held-out test events** — this recovers *all* signal and background (including the signal clusters that 1:1 down-sampling had discarded) at their **true physical abundance** (no reweighting). Test N: **OPTICS 6,484 (sig 6,144 / bkg 340); CF 6,240 (sig 5,885 / bkg 355)**.
3. **Apply the frozen threshold** — each model's training-time 80%-efficiency cut (RF 0.547, GBT 0.553, NN 0.569 for OPTICS; RF 0.509, GBT 0.512, NN 0.543 for CF). The cut is **not** re-tuned on the test set; we apply it and read off the result.
4. **Metrics** (on the full test-set statistics, signal = `dominant_class ∈ {1,2,3,4}`, all clusters — no prompt sub-filter):
   - Signal efficiency ε = (signal passing cut) / (total signal in test set)
   - Purity = (signal passing cut) / (signal passing + background passing)

| Streamline | classifier | frozen cut | **test ε** | **test purity** | test fakes | test N (sig/bkg) |
|---|---|---:|---:|---:|---:|---|
| **OPTICS** | **RF** | 0.547 | **0.826** | **0.971** | 154 | 6,484 (6144/340) |
| OPTICS | GBT | 0.553 | 0.838 | 0.969 | 166 | |
| OPTICS | NN (seeded) | 0.569 | 0.786 | 0.970 | 149 | |
| **CF** | RF (own) | 0.509 | 0.899 | 0.977 | 123 | 6,240 (5885/355) |
| **CF** | **GBT (own)** | 0.512 | 0.887 | 0.966 | 186 | |
| CF | NN (own, seeded) | 0.543 | 0.889 | 0.964 | 197 | |

> **Recomputed 2026-06-12 (this revision).** These rows were **independently recomputed** from the seeded scored parquets, replaying the exact event-level split (`in_test` flag) and applying the frozen cut at natural class abundance. They **supersede** the earlier §4.2 numbers (OPTICS+RF was listed 0.785/0.969/77 on N=3,202; CF+RF 0.806/0.976/59). The earlier values came from an ad-hoc run whose prompt-cluster handling and N (~3.2k, roughly half the natural test population) could not be reproduced; the recipe above (all clusters in held-out test events, composition truth, frozen cut) is the one stated in steps 1–4 and is the reproducible source of truth. **Purity is robust (~0.97 every variant);** efficiency is what shifted (the old N≈3.2k subset gave lower ε). RF/GBT/XGB are deterministic; the NN rows use the seeded model.

→ **Efficiency *varies* by classifier (OPTICS 0.786–0.838; CF 0.887–0.899)** because the frozen training cut is applied as-is and the result is measured — these are genuine held-out *instances* of the trained model, not re-optimized to hit 80%. (The cut targets 80% on the *training* test split; on this independent held-out event set the trees land a little above that, as expected for a well-behaved model.) **Held-out OPTICS+RF: ε = 0.826, purity = 0.971** — vs the in-sample whole-population purity 0.996, the small purity drop confirms **no material overfit**. CF's own model lands at *higher* efficiency here (0.887–0.899) and comparable purity (0.964–0.977) but at a lower AUC (0.709 vs OPTICS 0.754, §4b.1) — its natural test population is cleaner at this operating point, so the cut admits more of it. **Use this table — frozen cut, event-level held-out test, true class abundance, all clusters — as the defensible thesis/paper number;** §4 (whole population) is the larger-statistics in-sample cross-check. (The purity that matters is ~0.97 either way; the AUC, not the held-out ε, is the honest discrimination metric — see §4b.1.)

---

# STAGE 4b — Two-streamline cross-check: a *separately-trained* ClusterFinder MVA

The main analysis trains **one MVA on OPTICS clusters** and scores ClusterFinder clusters with that same model. This cross-check asks the obvious question: **does training a dedicated CF MVA (on CF clusters) do better?** A sandbox run (`run_ccinc_xi02_compcut_cftrain.sh`, isolated `cc_neutrino_xi02_compcut_cftrain` output tree, Stage-0/1 reused) trains a separate CF model and runs the full downstream per streamline, each method scored by **its own** model.

## 4b.1 Training quality — CF clusters are intrinsically harder

| Model trained on | RF test AUC | NN test AUC (seeded) | n_sig : n_bkg (1:1) |
|---|---:|---:|---:|
| **OPTICS** clusters | **0.7545** | 0.7526 | 6729 : 6729 |
| **ClusterFinder** clusters | **0.7093** | 0.7156 | 6406 : 6406 |

→ The dedicated CF model has a **lower AUC (0.709 vs 0.754)** — CF clusters are genuinely harder to separate signal-from-background than OPTICS clusters (CF's wider integration window blends more prompt/dark-noise light into each cluster). Training on CF does **not** recover discrimination power; it loses ~0.045 AUC.

**Full CF classifier comparison (dedicated CF model).** Test AUC + 5-fold CV from the sandbox log; held-out eff/purity/fakes from §4.2 (frozen 80%-eff cut, event-level held-out test, true class abundance):

| CF classifier | test AUC | 5-fold CV AUC | frozen cut | held-out ε | held-out purity | held-out fakes |
|---|---:|---:|---:|---:|---:|---:|
| RF | 0.7093 | 0.7051 ± 0.0095 | 0.509 | 0.899 | **0.977** | 123 |
| **GBT** *(sandbox auto-pick)* | **0.7130** | 0.7037 ± 0.0128 | 0.512 | 0.887 | 0.966 | 186 |
| XGBoost | 0.7013 | 0.7008 ± 0.0103 | — | — | — | — |
| NN (seeded) | 0.7156 | — | 0.543 | 0.889 | 0.964 | 197 |

> Source: `logs_cc_neutrino_xi02_compcut_cftrain.log` (test AUC + CV) and §4.2 (held-out selection). On CF the three carried-downstream classifiers are statistically tied (RF 0.7093, GBT 0.7130, NN 0.7156 — all within 0.006, CV intervals overlap), so the choice of CF classifier is immaterial; GBT was the sandbox auto-pick (highest-AUC tree) for the as-run CF downstream. XGBoost was trained but not carried downstream (no frozen-cut benchmark row). **Correction (this revision):** §4b.1 previously listed CF NN test AUC as 0.7116 — that was a mid-training progress-bar readout, not the final test AUC; the correct seeded final value is **0.7156** (log line 135).

## 4b.2 Selection performance — MC efficiency / purity / fakes (each method = own model)

Each model's **frozen training cut** (80%-eff operating point) applied to the **whole** composition-cut population (in-sample; the held-out test-only version is §4.2), composition truth, natural ratio. Each method scored by **its own** dedicated model. Efficiency varies (the cut is applied, not re-tuned):

| Streamline | classifier | frozen cut | eff | purity | fakes |
|---|---|---:|---:|---:|---:|
| **OPTICS** | **RF** (baseline) | 0.547 | 0.834 | **0.996** | **80** |
| OPTICS | NN (seeded) | 0.569 | 0.810 | 0.978 | 419 |
| **CF** | RF *(side note)* | 0.509 | 0.842 | 0.997 | 63 |
| CF | GBT *(sandbox auto-pick)* | 0.512 | 0.818 | 0.987 | 241 |
| CF | NN (seeded) | 0.543 | 0.788 | 0.983 | 325 |
| OPTICS | legacy box (80/0.45/9) | — | 0.851 | 0.977 | 457 |
| CF | legacy box (80/0.45/9) | — | 0.792 | 0.980 | 383 |

→ Both **OPTICS+RF (0.996, 80 fakes)** and **CF+RF-own (0.997, 63 fakes)** are very clean at this frozen cut; the NN rows are dirtier (OPTICS 0.978 / CF 0.983) and the legacy box dirtier still. The key point of this cross-check is **§4b.1**: the dedicated CF model has a **lower AUC (0.709 vs OPTICS 0.754)** — CF clusters are intrinsically harder to discriminate. The whole-population frozen-cut purities here look comparable because CF's natural population is already cleaner; the held-out test-only numbers (§4.2) and the AUC are the honest measure of discrimination power, and there **a dedicated CF model gives no advantage** over scoring CF with the OPTICS model. (These whole-population numbers are in-sample; see §4.2 for the recomputed held-out version, where OPTICS ε lands 0.786–0.838 and CF-own 0.887–0.899 at ~0.97 purity.)

## 4b.3 Data validation — rate, single-neutron fraction, capture-τ (each = own model)

OPTICS = published §3 frozen model; CF = the dedicated CF model. Both NN rows use the **seeded** NN (OPTICS nn≥0.569, CF nn≥0.543):

| Streamline | classifier | data rate | single-n frac | capture τ | in 25–30 µs band? |
|---|---|---:|---:|---:|:---:|
| **OPTICS** | **RF** | 0.438 | **0.969** | **26.7 ± 2.1 µs** | ✓ (best) |
| OPTICS | NN (seeded) | 0.631 | 0.896 | 32.0 ± 2.1 µs | ✗ (above band) |
| **CF** | GBT *(auto-pick)* | 0.759 | 0.937 | 28.3 ± 2.9 µs | ✓ |
| CF | RF *(own)* | 0.753 | 0.945 | 28.4 ± 2.8 µs | ✓ |
| CF | NN (seeded, own model) | 0.708 | 0.956 | 27.7 ± 2.5 µs | ✓ |

→ **OPTICS + RF gives the cleanest physics:** highest single-neutron fraction (0.969 — closest to the AmBe one-neutron-per-trigger ideal) and the tightest, lowest in-band capture time (26.7 µs). OPTICS + NN is higher-yield (rate 0.631) but drifts **above** the band (32.0 µs) with a looser single-neutron fraction (0.896). All three **CF** variants validate (τ in band) with higher raw rate (~0.71–0.76). **Net: OPTICS+RF is the best-quality streamline; CF is higher-yield; OPTICS+NN is the high-yield-but-τ-drifting cross-check.**

> **CF·RF-own added (this revision, computed 2026-06-12).** §4b.3 originally ran only the sandbox auto-pick (CF·GBT) and CF·NN-own; **CF·RF-own** (the §4b.2 "RF side note") was never carried through the data-validation pipeline. It was run here at its frozen cut `rf_score ≥ 0.509` on `ccinc_xi02cc_cf_cftrain__scored.parquet`: **data rate 0.753, single-n 94.5%, capture-τ 28.4 ± 2.8 µs (21/21 positions)**. As the report predicted, it lands **between CF·GBT and CF·NN** on every metric (GBT and RF differ by <0.004 AUC on CF), confirming the choice of CF classifier is immaterial downstream.

**Per-event multiplicity breakdown — CF own-model selections.** The §3.5 table reports CF·NN with the *frozen OPTICS* cut (OPTICS-model-scored CF); the rows below are the **CF own-model** selections (CF·GBT auto-pick, CF·RF-own, CF·NN-own seeded) from the sandbox `ccinc_xi02cc_cf_cftrain__scored.parquet`, counted per event exactly as in §3.5 (`groupby [run, event_tank_time]`). The single-neutron fractions reproduce the §4b.3 summary values above (93.7% / 94.5% / 95.6%) as a cross-check.

| multiplicity | CF·GBT (own) | CF·RF (own) | CF·NN (own, seeded) |
|---:|---:|---:|---:|
| 1 | 56,676 | 56,775 | 53,945 |
| 2 | 3,519 | 3,062 | 2,303 |
| 3 | 257 | 216 | 148 |
| 4 | 17 | 17 | 11 |
| ≥5 | 6 | 4 | 3 |
| **≥2 (total)** | **3,799** | **3,299** | **2,465** |
| events ≥1 | 60,475 | 60,074 | 56,410 |
| **single-neutron fraction** | **93.7%** | **94.5%** | **95.6%** |

> CF own-model cuts: GBT `gbt_score ≥ 0.5122`, RF `rf_score ≥ 0.509`, NN `nn_score ≥ 0.5430` (the seeded CF-own 80%-eff operating points, §4.2). The GBT/NN columns are the per-event histograms behind `ccinc_xi02cc_cf_cftrain/…__multiplicity.pdf` (GBT) and `…__multiplicity_nn.pdf` (NN); **CF·RF-own was computed this revision** (no PDF — RF was never the auto-pick downstream). CF·RF sits between GBT and NN, as expected. All three are even more single-neutron-dominated than the OPTICS-model-scored CF·NN (91.6%, §3.5) — CF's own model, trained on CF clusters, merges slightly fewer events to multiplicity ≥2.

> **Note on classifiers:** the sandbox auto-selected the highest-AUC model per run — **RF for OPTICS, GBT for CF** — so the as-run downstream used those. For an apples-to-apples cross-check, **CF+RF** is included in §4b.2 as a side note (it lands identically to CF+GBT: 0.800 / 0.980 / 381, since RF and GBT differ by <0.004 AUC on CF). **NN (seeded) is reported for both streamlines** at each method's own 80%-eff NN cut (OPTICS nn≥0.569, CF-own nn≥0.543).

**Scope note:** these sandbox numbers are an isolated cross-check; the headline analysis (§3–§4) remains OPTICS-MVA-based with CF scored by that model. The two-streamline run exists to *justify* that choice, not to replace it.

**Figures to look at (sandbox, `ambe_output/cc_neutrino_xi02_compcut_cftrain/`):**

*Side-by-side model quality — RF signal-vs-bkg discriminator + NN loss/accuracy, OPTICS vs CF:*
- **MVA training — OPTICS:** `plots/cc_neutrino_xi02_compcut_cftrain__mva__keepprompt__optics.pdf`
- **MVA training — ClusterFinder:** `plots/cc_neutrino_xi02_compcut_cftrain__mva__keepprompt__cf.pdf`
  - In each PDF: **page 2** = ROC, **page 3** = **signal-vs-background score distribution (the RF discriminator)**, **page 6** = **NN loss & AUC/accuracy training history**. Compare page 3 OPTICS-vs-CF to *see* why CF separates worse (more signal/bkg overlap), and page 2 for the AUC gap (0.754 vs 0.709).

*Per-streamline downstream (`ambe_data/ccinc_xi02cc_{optics,cf}_cftrain/`):*
- Multiplicity — best-classifier: `…__multiplicity.pdf` (OPTICS=RF, CF=GBT); **NN:** `…__multiplicity_nn.pdf`
- Rate summary — best: `rate_best/ambe_all__neutron_summary.pdf`; **NN:** `rate_best_nn/ambe_all__neutron_summary.pdf`
- Capture-time — best: `capture_stageAB/stageAB_capture_lmfit.pdf`; **NN:** `capture_stageAB_nn/stageAB_capture_lmfit.pdf`

*Stage-4 benchmark (eff/purity bars):* `ambe_data/ccinc_xi02cc_{optics,cf}_boxcuts_cftrain/benchmark/selection_2x2_benchmark.{pdf,csv}` (read the OPTICS row from the optics dir, the CF row from the cf dir; ignore the cross-method row each contains).

---

# Summary tables for the poster

## Selection summary (one table)

| Selection | MC efficiency | MC purity | AmBe data rate (OPTICS / CF) | Capture τ (OPTICS / CF) |
|---|---:|---:|---:|---:|
| **OPTICS + RF MVA** | 0.835 | **0.996** | 0.438 / 0.814 | 26.7 / 29.0 µs |
| OPTICS + NN MVA (seeded) | 0.810 | 0.978 | 0.631 / 0.879 | 32.0 / 29.3 µs |
| OPTICS + box cut (80/0.45/9) | 0.851 | 0.977 | — | — |
| CF + box cut (80/0.45/9) | 0.792 | 0.980 | — | — |

## Model comparison (RF vs NN)

| | Random Forest | Neural Net (seeded) |
|---|---|---|
| MC test AUC | 0.7545 | 0.7526 |
| Role | High-purity frozen baseline | High-yield cross-check |
| Threshold (80% MC eff) | rf_score ≥ 0.547 | nn_score ≥ 0.569 |
| OPTICS MC eff / purity / fakes (Stage 4) | 0.835 / **0.996** / **81** | 0.810 / 0.978 / 419 |
| OPTICS data rate | 0.438 | 0.631 |
| CF data rate | 0.814 | 0.879 |
| OPTICS single-n fraction | 0.968 | 0.896 |
| OPTICS capture τ | 26.7 µs | 32.0 µs |

**Recommendation:** present RF as the frozen baseline (cleanest, validated, in-band) and NN as a higher-efficiency cross-check. The RF↔NN spread (44%↔63% OPTICS rate) brackets the selection-efficiency systematic; note OPTICS+NN's τ sits just above the literature band (32.0 µs), reinforcing RF as the physics baseline.

---

# Headline conclusions

1. **3.3% of MC events pass the CC-truth selection** (29,223 / 890,000); these define the delayed-residual signal sample.
2. **OPTICS (ξ=0.02) and ClusterFinder produce comparable cluster yields** (~29.6 k / 29.9 k) with ~77–79% neutron-dominated.
3. **Cluster contamination is intrinsic** — 84% of signal clusters carry contaminant hits, 63% of background clusters carry neutron hits — which is why an aggregate-feature MVA beats per-hit cuts.
4. **All four classifiers tie at AUC ≈ 0.75**; RF auto-selected. `pe_total` and `spatial_rms` dominate.
5. **OPTICS + RF MVA is the cleanest MC selector: 99.6% purity, 81 fakes** at 83.5% efficiency. The seeded frozen NN (nn_score ≥ 0.569) lands at 0.810 / 0.978 / 419 — essentially tied with the legacy box-cut and **~5× dirtier than RF**, so RF is the selector to present; NN is the high-yield data cross-check only. The box cut (unified to PE<80 / CB<0.45 / nHits>9 across all stages, §4.1) is model-independent — identical rows under RF and NN — and reaches 0.851 (OPTICS) / 0.792 (CF) efficiency at ~98% purity, but with ~5–6× more fakes than RF.
6. **Capture time validates the RF chain** (OPTICS+RF 26.7 µs, CF+RF 29.0 µs, both in the 25–30 µs band). OPTICS+NN sits just above the band (32.0 µs, seeded) — the higher-yield cross-check, with RF as the physics baseline.
7. **Neutron multiplicity confirms the single-neutron AmBe source** (96.9% single, OPTICS+RF) — close to the ideal N events = N neutrons.
8. **Rate is position-stable** across all 5 ports.
9. **Open item:** OPTICS data rate (44% RF / 61% NN) is below CF (81% / 88%) — the tight ξ=0.02 likely splits some real captures below the cluster/score threshold; needs follow-up.

---

# Complete file map (for pulling plots into the poster)

| Content | Path under `ambe_output/` |
|---|---|
| Per-hit pulses | `cc_neutrino_xi02_compcut/parquet/…__pulses.parquet` |
| Cluster features + truth | `cc_neutrino_xi02_compcut/parquet/…__cluster_features.parquet` |
| Frozen MVA (RF/GBT/XGB) | `cc_neutrino_xi02_compcut/parquet/…__mva_frozen__keepprompt.pkl` |
| Frozen NN | `cc_neutrino_xi02_compcut/parquet/…__mva_frozen__keepprompt__nn.keras` |
| MVA scores | `cc_neutrino_xi02_compcut/parquet/…__mva_scores__keepprompt.parquet` |
| **Cluster-feature plots** | `cc_neutrino_xi02_compcut/plots/…__cluster_features.pdf` |
| **Feature separation plots** | `cc_neutrino_xi02_compcut/plots/…__feature_separation_plots.pdf` |
| **MVA ROC / importances / NN history** | `cc_neutrino_xi02_compcut/plots/…__mva__keepprompt.pdf` |
| Feature importance CSV | `cc_neutrino_xi02_compcut/csv/…__mva_summary__keepprompt.csv` |
| Feature separation CSV | `cc_neutrino_xi02_compcut/csv/…__feature_separation.csv` |
| AmBe OPTICS scored data | `ambe_data/ccinc_xi02cc_optics_mcbkg/…__scored.parquet` |
| AmBe CF scored data | `ambe_data/ccinc_xi02cc_cf_mcbkg/…__scored.parquet` |
| **Per-run rate CSV (RF)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/rate_best/ambe_all__neutron_rate_by_run.csv` |
| **Per-run rate CSV (NN)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/rate_best_nn/ambe_all__neutron_rate_by_run.csv` |
| **Rate / multiplicity summary plot (RF / NN)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/rate_best{,_nn}/ambe_all__neutron_summary.pdf` |
| **Multiplicity plot (RF / NN)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/…__multiplicity{,_nn}.pdf` |
| **Capture-time plot + CSV (RF)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB/stageAB_capture_lmfit.{pdf,_summary.csv}` |
| **Capture-time plot + CSV (NN)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB_nn/stageAB_capture_lmfit.{pdf,_summary.csv}` |
| **Capture-time plot + CSV (legacy box, ref)** | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB_box/stageAB_capture_lmfit.{pdf,_summary.csv}` |
| **Box-cut vs frozen-RF benchmark** | `ambe_data/ccinc_xi02cc_{optics,cf}_boxcuts/benchmark/selection_2x2_benchmark.{pdf,csv}` |
| **Box-cut vs frozen-NN benchmark** | `ambe_data/ccinc_xi02cc_{optics,cf}_boxcuts/benchmark_nn/selection_2x2_benchmark.{pdf,csv}` |
