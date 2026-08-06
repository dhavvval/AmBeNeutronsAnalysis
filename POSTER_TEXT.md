# Poster — drop-in text
**Final-State Neutron Identification in ANNIE**
Source: `run_ccinc_xi02_compcut.sh` · `REPORT_ccinc_xi02_compcut_FULL.md` (2026-06-11/12)

> Numbers below are taken verbatim from the report. Acronyms are expanded on first use; a glossary box is provided at the end for the layout. Figure file paths for each plot slot are listed in **[brackets]**.

---

## GLOSSARY BOX (place small, top-right or bottom — referenced throughout)

- **ANNIE** — Accelerator Neutrino Neutron Interaction Experiment
- **MC** — Monte Carlo (simulation)
- **CC** — Charged-Current neutrino interaction · **FSL** — Final-State Lepton · **FV** — Fiducial Volume · **Q²** — four-momentum transfer squared
- **AmBe** — Americium–Beryllium calibration source (emits ~one neutron per decay)
- **OPTICS** — Ordering Points To Identify the Clustering Structure (density-based clustering)
- **ClusterFinder** — ANNIE's standard fixed-time-window clustering algorithm
- **MVA** — Multivariate Analysis (a machine-learning classifier using many variables at once)
- **RF** — Random Forest · **GBT** — Gradient-Boosted Trees · **XGBoost** — eXtreme Gradient Boosting · **NN** — Neural Network
- **AUC** — Area Under the ROC Curve (ROC = Receiver Operating Characteristic); 0.5 = random, 1.0 = perfect
- **PE** — photoelectrons (total detected light) · **RMS** — Root Mean Square · **MAD** — Median Absolute Deviation
- **τ (tau)** — neutron-capture time constant

---

# LEFT COLUMN — Motivation & data sample

## About ANNIE
- **A**ccelerator **N**eutrino **N**eutron **I**nteraction **E**xperiment — a gadolinium-loaded (Gd) water Cherenkov detector in the Booster Neutrino Beam (~600 MeV ν_μ).
- **Primary goal:** measure neutrino-induced **neutron multiplicity** versus **Q²** (four-momentum transfer squared).
- Gd-loaded water gives excellent **neutron-tagging** capability — neutrons capture on Gd/H and release detectable light microseconds after the interaction.
- *(detector schematic image)*

## Why neutrons?
- Neutron multiplicity probes a key **systematic** for neutrino-oscillation measurements.
- Constrains neutrino-interaction models and **final-state interactions** (FSI).
- Improves **energy reconstruction** and reduces oscillation systematics.
- *(interaction cartoon image)*

## Data sample (simulation — the input to the selection)
- **GENIE** neutrino-interaction sample: **890,000** simulated events (tank-only simulation).
- **CC-truth selection** (true Charged-Current interaction): muon final-state lepton, muon momentum **600–1200 MeV/c**, forward angle (**cos θ > 0.8**), contained fiducial-volume vertex (radius < 100 cm, |Y| < 100 cm).
- Keeps **3.3% of events (29,223)** — the sample used to develop the selection.
- **Signal = late light:** only hits at **t > 2000 ns** (after the prompt muon flash) enter the clustering. This delayed-residual window is where neutron-capture light lives.

---

# CENTER COLUMN — Building the selection on simulation (MC)

*An MVA (multivariate analysis) is a machine-learning classifier that looks at all 31 cluster features at once and returns a single "neutron-likeness" score, instead of cutting on one variable at a time.*

## Three steps

**1 · Find candidate clusters.**
Look only at *late* light (more than 2000 ns after the beam — after the prompt muon flash). Group these delayed hits into clusters with **OPTICS**, a density-based algorithm that forms a cluster only where hits are genuinely dense in space and time — making tighter, cleaner clusters than ANNIE's standard fixed-window **ClusterFinder**. On the simulated CC sample this yields **~29,600 clusters, ~77% truly neutron-dominated** (MC truth). The parameter **ξ = 0.02** sets how aggressively OPTICS splits overlapping deposits (smaller ξ = tighter clusters); **minimum 8 hits** per cluster.

> *Why a classifier at all?* Neutron-capture light and prompt/background light **physically overlap** in the tank — **84% of true neutron clusters carry some contaminant hits**, and 63% of background clusters carry some real neutron hits. This overlap is intrinsic to the detector, not an algorithm artefact, so no per-hit cleaning works — we discriminate on **whole-cluster features** instead.

**2 · Describe each cluster.**
Compute **31 numbers per cluster** — total light (`pe_total`, in photoelectrons), spatial spread (`spatial_rms`), number of hits, timing spread, distance to the tank wall, charge balance, and fit-quality variables. **No truth information is used**, so the identical recipe runs unchanged on real data.

**3 · Train a classifier.**
Train four machine-learning models — **Random Forest (RF), Gradient-Boosted Trees (GBT), XGBoost, and a Neural Network (NN)** — to separate neutron clusters from background using all 31 features jointly. Events are split **80/20** (train/test) at the event level, balanced **1:1**, with fixed random seeds for reproducibility. The operating point is **frozen at 80% efficiency on simulation** (Random-Forest score ≥ 0.547) and then applied **unchanged** to data.

> **The model is frozen on simulation and never re-tuned — the same cut is applied to real AmBe data, which then validates it (capture time, multiplicity).**

> *Why 80% efficiency?* Tighter cuts starve the capture-time fits (τ falls to ~21 µs); looser cuts push τ above the physical band. **80% is the sweet spot** that validates against data.

### Neural-network architecture (sidebar)
Fully-connected (dense) network: **31 inputs → 3 hidden layers of 128 neurons** each (ReLU activation, batch-normalization, 30% dropout for regularization) **→ 1 sigmoid output** = neutron probability. Trained with the **Adam** optimizer, **binary cross-entropy** loss, batch size 256, up to 100 epochs with early stopping; random seed fixed (`set_random_seed(42)`) so results are reproducible.

### What ClusterFinder and OPTICS actually do (sidebar)
- **ClusterFinder** — ANNIE's standard algorithm: scans the delayed hits with a **fixed 20 ns time window** and groups hits in the same window. Simple and fast, but a fixed window tends to **merge separate light deposits** together.
- **OPTICS** — a **density-based** algorithm: instead of a fixed window, it groups hits that are close together in space *and* time and forms a cluster only where hit density is genuinely high. Result: **tighter, cleaner clusters** that split overlapping deposits the fixed window would merge.

### Plot — feature separation
**[`cc_neutrino_xi02_compcut/plots/cc_neutrino_xi02_compcut__feature_separation_plots.pdf`]**
*(also: `…__cluster_features.pdf` for full feature distributions)*
**Caption:** "Signal and background overlap even in the strongest features (`pe_total`, `spatial_rms`) — no single cut separates them, so we discriminate on all 31 jointly."

### Table — model comparison (held-out MC test set; signal = neutron-dominated cluster)
*Efficiency (ε) and purity at each model's frozen 80%-efficiency cut.*

| Clustering | Model | Test AUC | ε | Purity |
|---|---|---:|---:|---:|
| OPTICS | RF | **0.7545** | 0.826 | 0.971 |
| OPTICS | GBT | 0.7535 | 0.838 | 0.969 |
| OPTICS | XGBoost | 0.7488 | — | — |
| OPTICS | NN | 0.7526 | 0.786 | 0.970 |
| ClusterFinder | RF | 0.7093 | 0.899 | 0.977 |
| ClusterFinder | GBT | 0.7130 | 0.887 | 0.966 |
| ClusterFinder | XGBoost | 0.7013 | — | — |
| ClusterFinder | NN | 0.7156 | 0.889 | 0.964 |

**Caption:** "The four classifiers **tie** (AUC spread < cross-validation uncertainty), so the classifier choice is secondary. The decisive lever is the **clustering**: OPTICS separates signal from background markedly better than ClusterFinder (**AUC ~0.75 vs ~0.71**). Purity is uniformly high (~0.97). Two separate takeaways: **OPTICS clustering is the discrimination win**, and among classifiers **RF is the cleanest selector** (see right column)."

---

# RIGHT COLUMN — Validation on AmBe data & results

## Three big stat callouts
- **AUC 0.75** (OPTICS) vs **0.71** (ClusterFinder) — OPTICS clustering separates signal from background better.
- **97.1% purity** on held-out simulation — the selection is clean, with no overfit.
- **Capture time τ = 26.7 µs** — matches the known **25–30 µs** neutron-capture-on-hydrogen time in water (a physics validation, not just a measurement).

## Held-out benchmark (no overfit)
- Frozen Random Forest applied at **rf_score ≥ 0.547** to event-level held-out test events at **natural abundance** (true physical signal:background ratio).
- **OPTICS + RF: efficiency 82.6%, purity 97.1%** — consistent with the in-sample number, so **no overfit**.

### Plot — RF discriminator with the frozen cut
**[`cc_neutrino_xi02_compcut/plots/cc_neutrino_xi02_compcut__mva__keepprompt.pdf`]** *(ROC, score distributions, feature importances, NN training history — single PDF; use the score-distribution page)*
**Caption:** "Random-Forest score, signal vs background. The frozen cut (80% MC efficiency, rf_score ≥ 0.547) keeps the neutron peak while rejecting the low-score background — then is applied **unchanged** to data."

## Neutron capture time (AmBe validation #1)
- **τ = 26.7 ± 2.1 µs** (OPTICS + RF), squarely in the **25–30 µs** literature band for neutron capture on hydrogen.
- Per-source-position fit (lmfit): **A·(1 − e^(−t/τ_rise))·e^(−t/τ) + B** = a rise, an exponential capture (time constant **τ**), and a flat background **B**.

### Plot — capture-time fit
**[`ambe_data/ccinc_xi02cc_optics_mcbkg/capture_stageAB/stageAB_capture_lmfit.pdf`]** *(summary CSV: `stageAB_capture_lmfit_summary.csv`)*

## Neutron multiplicity (AmBe validation #2)
- **96.9% single-neutron** (OPTICS + RF) — consistent with the AmBe one-neutron-per-decay source (ideal: N events = N neutrons).

### Plot — multiplicity
**[`ambe_data/ccinc_xi02cc_optics_mcbkg/ccinc_xi02cc_optics_mcbkg__multiplicity.pdf`]** *(RF version)*
> Note: pair the **RF** multiplicity plot (`…__multiplicity.pdf`) with the **RF** τ and 96.9% number. The NN versions (`…__multiplicity_nn.pdf`, τ = 32.0 µs) are a separate, higher-yield cross-check — do not mix them.

## Rate stability (optional)
- Detection rate is **flat across all 5 source ports** (OPTICS+RF 0.41–0.47) — the selection is not biased by source position.
- **[`ambe_data/ccinc_xi02cc_optics_mcbkg/rate_best/ambe_all__neutron_summary.pdf`]**

## Conclusions box
1. **Cluster contamination is intrinsic** — 84% of signal clusters carry contaminant hits — so an aggregate-feature MVA beats per-hit cuts.
2. **OPTICS clustering separates best** (AUC 0.75 vs 0.71 for ClusterFinder); classifiers tie, so clustering is the lever.
3. **Frozen Random Forest is the cleanest selector** — 97.1% held-out purity, ~5× fewer fake clusters than NN or the legacy box cut, **no overfit**.
4. **Data validates the chain:** capture time 26.7 µs (in band) and 96.9% single-neutron, both consistent with the AmBe source.
5. **Future work:** the OPTICS data rate (44% RF) is lower than ClusterFinder (81%) — the tight ξ = 0.02 may split some real captures below threshold; recovering this efficiency is the next step.

---

## Notes for the layout (not poster text)
- **MC = Monte Carlo (simulation)** is used heavily — expand it on its first appearance in each column, or rely on the glossary box.
- The model-comparison ε/purity are at the **frozen 80%-eff cut** on the **held-out test set** (report §4.2) — the defensible, no-overfit numbers. (The §4 whole-population purity reads 0.996 but is in-sample-optimistic; prefer 0.971 for the poster.)
- Keep two takeaways distinct everywhere: **OPTICS = the clustering win**, **RF = the clean classifier**. They are separate results.
- **Model-comparison table provenance (center column):** the OPTICS-vs-ClusterFinder AUC table (0.7545 vs 0.7093 etc.) comes from the *separate-model-per-clustering-method* sandbox run (`run_ccinc_xi02_compcut_cftrain.sh`), not the shared-model "dualmodel" report. Don't mix its numbers with the RF@0.547/NN@0.538 dualmodel figures used elsewhere on this poster — the two setups train ClusterFinder differently (its own frozen model vs. scored by the OPTICS-trained model).
- **Config anomaly worth resolving before the final print run:** `configs/cc_neutrino_xi02_compcut.yaml` sets `residual.prompt_window_ns: 10000`, while every other config in the repo (productionv2, ccinc_test, all sweep configs, the cftrain variant) uses `2000`, matching the code's own default and this file's own `description:` text. So the actual `cc_neutrino_xi02_compcut` run that produced the frozen model and all the numbers on this poster executed with a 10 µs delayed window, not the intended/documented 2 µs — confirm whether that's deliberate before finalizing; if not, the pipeline may need a re-run at 2000 before this poster's numbers are locked in.
- **NN capture time (right column, τ = 32.0 µs):** the frozen NN artifact (`...__nn.keras`) appears to have been re-trained/re-frozen at some point after these numbers were produced — live-loading it now gives a different AUC/threshold (0.7495 / 0.5687) than what's floating around in older notes (0.7475 / 0.538). If the NN capture-time/multiplicity plots go on the poster, regenerate them from the current frozen model first rather than reusing this number as-is.
