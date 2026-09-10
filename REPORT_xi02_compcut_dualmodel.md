# Detailed Report — Final-State Neutron Identification (xi=0.02 + composition cut)
## Dual-model (Random Forest & Neural Net) AmBe results at the fixed 80%-efficiency threshold

**Run:** `cc_neutrino_xi02_compcut` · 412 GENIE-WCSim files · prompt-inclusive MVA
**Date:** 2026-06-11
**Base output:** `/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/`

---

## 0. Executive summary

- **Configuration:** OPTICS minHits=8, **ξ=0.02** (tighter than the 0.10 baseline), t_unit=25 ns; downstream **composition cut** (reject cluster if n_neutron ≤ 10 AND frac_nonneutron ≥ 0.40) applied on the MC training/benchmark side.
- **Operating point:** fixed **80% MC signal efficiency** per model → RF `rf_score ≥ 0.547`, NN `nn_score ≥ 0.538`.
- **Why this threshold and not an "optimized" one:** Youden's-J and max-F1 thresholds were tested and **rejected** — they pull the AmBe capture time to ~35 µs (outside the 25–30 µs literature band) and collapse the OPTICS fit quality to 10/21. The fixed 80%-eff cut is the only operating point where both models land in the physical capture-time band with stable fits. See §6.
- **Both classifiers are kept:** RF = high-purity baseline; NN = high-yield cross-check.

---

## 1. What "events" and the per-run rows mean (read this first)

The per-run CSV has **one row per AmBe run = one source position** (port + (x,y,z) cm). 31 runs total.

| Column | Definition |
|---|---|
| `n_events` | Beam-trigger events in that run **that produced ≥1 delayed cluster** (the denominator). |
| `n_clusters` | All reconstructed clusters (OPTICS or CF) before the MVA cut. |
| `n_neutron_clusters` | Clusters tagged neutron by the model (score ≥ threshold). |
| `n_events_with_neutron` | Events with ≥1 tagged neutron cluster (the numerator). |
| `frac_events_with_neutron` | = n_events_with_neutron / n_events → **the neutron detection rate**. |
| `mean_mult` | Mean tagged neutron clusters per neutron-event. |
| `single_neutron_frac` | Fraction of neutron-events with exactly one. |

**Physics interpretation (important):** AmBe is a **single-neutron source** — one neutron per source decay. So `mean_mult ≈ 1` and `single_neutron_frac ≈ 0.93–0.97` are the *expected* signature, and they confirm the selection is finding real single neutrons rather than fragmenting one capture into many fakes. The `frac_events_with_neutron` is a **combined containment × detection × selection efficiency**: of triggered events with delayed activity, how often we actually caught the neutron. It is not 100% because the neutron can (a) escape/capture outside the tank, (b) produce too few hits to cluster, or (c) form a cluster the MVA rejects.

> **Caveat to state explicitly to the collaboration:** the data parquet contains only events that already had ≥1 delayed cluster. So `frac_events_with_neutron` is **conditional on delayed activity existing**, not an absolute fraction of all AmBe triggers. The absolute trigger count lives in the AmBe processor bookkeeping, not in this CSV. Quote 44% (OPTICS-RF) as a *conditional* rate.

---

## 2. Stage 0–1 — Processing & clustering

| | OPTICS (ξ=0.02) | ClusterFinder | productionv2 (ξ=0.10) |
|---|---:|---:|---:|
| Total clusters | 29,572 | 29,851 | 26,585 (OPTICS) |
| Neutron-dominated (signal) | 22,843 (77.2%) | 23,445 (78.5%) | 21,285 |
| Background-dominated | 6,729 (22.8%) | 6,406 (21.5%) | 5,300 |
| `is_prompt_cluster` | 10,663 (36.1%) | 10,539 (35.3%) | 8,996 |
| `is_truth_neutron` (strict ≥50%) | 15,498 (52.4%) | 17,406 (58.3%) | 14,453 |
| Mean signal purity `frac_neutron` | 0.765 | 0.787 | 0.651 |

**Truth composition (dominant_class):** primary n ~34%, sec n←n ~37%, sec n←p ~5%, non-neutron (−5) ~22%, dark noise <1%.

**Composition cut on MC:** removes **11,806 / 59,423 clusters (19.9%)** — the prompt-γ / non-neutron-physics-dominated ones — before the MVA. This filter is **MC-side only** (training + benchmark); the real AmBe data parquet has no truth columns, so on data the selection is purely OPTICS-clustering + MVA.

---

## 3. Stage 2 — MVA training (all four classifiers)

Training set: signal 5,300 : background 5,300 (1:1), 31 physics features, prompt-inclusive background (`--keep-prompt-bkg`).

| Model | Test AUC | 80%-eff threshold | bkg-rej @80% |
|---|---:|---:|---:|
| **Random Forest** | **0.7545** | **0.547** | 0.596 |
| GBT | 0.7535 | 0.553 | 0.589 |
| XGBoost | 0.7488 | 0.537 | 0.580 |
| **Neural Net** | 0.7475 | 0.538 | 0.576 |

- All four AUCs within 0.007 — they rank MC truth almost identically.
- AUC up from **0.718** (June poster) → **0.754** (ξ=0.02 + prompt-inclusive fix).
- **Top RF features:** pe_total (0.123), spatial_rms (0.106), beta2 (0.074), beta1 (0.052), d_wall (0.042). All are pure charge/shape/timing variables computable on real data — this is what makes the MC→data transfer legitimate.

---

## 4. What is inside a cluster — contamination both directions (MC truth, OPTICS)

| Quantity | OPTICS | ClusterFinder |
|---|---:|---:|
| Signal cluster mean composition | 76.5% neutron / 13.7% non-neutron / 9.8% dark noise | 78.7% / 11.4% / 9.9% |
| Signal clusters carrying ≥1 contaminant hit | **83.9%** | 74.4% |
| Perfectly pure signal clusters | 4.0% | 7.3% |
| Background clusters containing ≥1 neutron hit | **62.9%** (mean 3.3 n-hits) | 64.0% (mean 3.5) |

**Message:** the capture-photon cloud and prompt/non-neutron light **physically overlap** in space-time. 84% of true neutron clusters are "dirty" and 63% of background clusters contain neutron hits — this is **intrinsic to the detector, not an algorithm bug**. That is exactly why we discriminate on **aggregate cluster features (MVA)** rather than per-hit purity, and why the composition cut only removes the one clean case (clearly prompt-γ-dominated clusters).

---

## 5. AmBe DATA neutron rate — RF vs NN, both streamlines (fixed 80%-eff)

**79,633 total events.**

| Streamline | Model | threshold | frac events w/ neutron | neutron clusters | events w/ neutron | mean mult | single-n frac |
|---|---|---:|---:|---:|---:|---:|---:|
| **OPTICS** | **RF** | 0.547 | **0.438** | 36,008 | 34,854 | 1.034 | **0.968** |
| **OPTICS** | **NN** | 0.538 | **0.609** | 52,849 | 48,522 | 1.091 | 0.918 |
| **CF** | **RF** | 0.547 | **0.814** | 69,415 | 64,802 | 1.069 | 0.936 |
| **CF** | **NN** | 0.538 | **0.878** | 75,661 | 69,890 | 1.080 | 0.928 |

**Reading it:**
- **NN tags substantially more events than RF** at the same MC efficiency (OPTICS 61% vs 44%; CF 88% vs 81%). NN's rate is **closer to the AmBe one-neutron-per-trigger physical expectation**.
- **RF holds the higher single-neutron fraction** (0.968 OPTICS) — fewer events fragmented into multiple fakes.
- The OPTICS↔CF gap persists in both models: OPTICS under-counts vs CF (open item, §8).

---

## 6. AmBe DATA capture time — RF vs NN (validation), and why the threshold is fixed

| Streamline | Model | threshold | τ (mean ± std) | median | good fits | in band? |
|---|---|---:|---:|---:|---:|:--:|
| OPTICS | RF | 0.547 | **26.7 ± 2.1 µs** | 26.5 | 20/21 | ✅ |
| OPTICS | NN | 0.538 | 30.3 ± 2.4 µs | 31.0 | 20/21 | ✅ (high edge) |
| CF | RF | 0.547 | 29.0 ± 2.9 µs | 29.9 | 21/21 | ✅ |
| CF | NN | 0.538 | 29.2 ± 2.5 µs | 29.6 | 21/21 | ✅ |

Literature n-capture-on-H in water = **25–30 µs**. All four configurations validate. **OPTICS+RF gives the tightest, lowest τ.**

### Why "optimized" thresholds were rejected (the evidence that killed the idea)

We tested Youden's-J (≈0.42) and max-F1 (≈0.34) thresholds. Both raise MC efficiency to 94–97% but **fail the physics cross-checks**:

| Configuration | τ (mean ± std) | good fits | single-n frac | verdict |
|---|---:|---:|---:|:--|
| OPTICS RF @0.547 (80% eff) | 26.7 ± 2.1 µs | 20/21 | 0.969 | ✅ keep |
| OPTICS RF @0.426 (Youden) | **34.6 ± 2.5 µs** | **10/21** | 0.825 | ❌ τ too long, fits collapse |
| OPTICS RF @0.344 (max-F1) | — | — | **0.349** | ❌ mean mult 1.94, over-tags fakes |
| OPTICS NN @0.420 (Youden) | 35.9 ± 2.4 µs | 17/21 | 0.804 | ❌ τ too long |

The looser cuts drag in fake clusters that **inflate τ to ~35 µs** (well outside the band) and crash fit quality / single-neutron purity. **Conclusion: the fixed 80%-MC-efficiency threshold is the physically correct operating point.** It is the only one where both models sit in the literature band with stable fits.

---

## 7. Stage 4 — MC selection benchmark (composition truth, natural population)

RF frozen model, composition-cut MC features:

| Clustering | Selection | Efficiency | Purity | Fakes |
|---|---|---:|---:|---:|
| OPTICS | box-cut (PE<60,CB<0.5,nHits>10) | 0.810 | 0.977 | 435 |
| **OPTICS** | **MVA RF @0.547** | **0.834** | **0.996** | **80** |
| CF | box-cut | 0.755 | 0.982 | 324 |
| CF | MVA RF @0.547 | 0.854 | 0.981 | 390 |

→ **OPTICS+RF MVA is the cleanest selector: 99.6% purity, 80 fakes** (vs 524 in the June poster — 6.5× fewer at equal efficiency). NN's MC AUC (0.747) is marginally below RF (0.754), so its purity at the same efficiency is slightly lower; RF remains the cleanest, NN the higher-yield.

> Note: the ~63–66% "purity" you may see computed on the balanced 1:1 *training* sample is a different denominator (artificial 50/50 mix). The 99.6% above is on the **natural cluster population** — that is the number to present.

---

## 8. RF vs NN — recommendation for the collaboration

- **MC:** AUC nearly identical (RF 0.754, NN 0.747) — both rank truth equally well.
- **On data:** NN is the higher-yield selector (OPTICS 61% vs 44%; CF 88% vs 81%), closer to the AmBe 1-n/trigger expectation. RF is the higher-purity selector (cleanest τ = 26.7 µs, single-n 97%, fewest MC fakes).
- **Recommendation:** present **RF as the frozen baseline** (defensible, clean, validated) and **NN as a high-efficiency cross-check**. The spread between them (44%↔61% OPTICS rate) is a useful **selection-efficiency systematic bracket**, not a contradiction. Do *not* swap to NN purely because its rate looks better — that would be tuning on data; report both side by side.

---

## 9. Open items / next steps

1. **OPTICS data rate (44% RF / 61% NN) is below CF (81% / 88%)** in both models — the tighter ξ=0.02 splits some real captures below `min_samples=8` or below the MVA cut. #1 thing to chase before final numbers.
2. **Absolute vs conditional rate:** wire the raw per-run AmBe trigger count into the rate CSV so `frac_events_with_neutron` can be quoted as an absolute capture efficiency, not conditional on delayed activity.
3. **Larger CC-candidate MC pool** (more GENIE files) to firm up the MVA training statistics.
4. **End-to-end check** of the data multiplicity construction.

---

## 10. File map

| Content | Path (under `ambe_output/`) |
|---|---|
| MC cluster features + truth | `cc_neutrino_xi02_compcut/parquet/…__cluster_features.parquet` |
| MVA model + scores (all 4 classifiers) | `cc_neutrino_xi02_compcut/parquet/…__mva_{frozen,scores}__keepprompt.{pkl,parquet}` |
| AmBe OPTICS scored data | `ambe_data/ccinc_xi02cc_optics_mcbkg/…__scored.parquet` |
| AmBe CF scored data | `ambe_data/ccinc_xi02cc_cf_mcbkg/…__scored.parquet` |
| Per-run rate (RF, fixed) | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/rate_best/ambe_all__neutron_rate_by_run.csv` |
| Capture-time fits (RF) | `ambe_data/ccinc_xi02cc_{optics,cf}_mcbkg/capture_stageAB/stageAB_capture_lmfit_summary.csv` |
| Capture-time fits (NN, dual-model) | `/tmp/ambe_view/capture_dualmodel/{optics,cf}_nn_eff80/stageAB_capture_lmfit_summary.csv` |
| MC eff/purity benchmark | `ambe_data/ccinc_xi02cc_{optics,cf}_boxcuts/benchmark/selection_2x2_benchmark.csv` |
| Collaboration slide deck | `collab_slides_xi02_compcut.pdf` |
| Dual-model rate JSON | `/tmp/ambe_dualmodel_rates.json` |
