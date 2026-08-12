# CC-Neutrino Neutron-Clustering — Results Bookkeeping

**Run:** productionv2 (376 GENIE-WCSim files) · prompt-inclusive MVA iteration
**Date:** 2026-06-11
**Base output path:** `/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/`

> ⚠️ **TWO model iterations coexist on disk.** Present only the NEW one.
> - **OLD (superseded):** `cc_neutrino_productionv2/…__mva_frozen.pkl` (13:31, AUC 0.587, prompt-EXCLUDED, starved) + `…__mva_frozen__vs_offbeam.pkl` (offbeam experiment, dropped) + `rate_rf/` & `rate_gbt/` (14:47).
> - **NEW (use this):** `cc_neutrino_productionv2_inclprompt/…__keepprompt.*` (14:59, **AUC 0.718**, prompt-INCLUDED) + `rate_best/`, `capture_stageAB/`, `benchmark/` (15:09–15:10).

---

## Pipeline overview

| Stage | Script | Input | Output |
|---|---|---|---|
| 0 | `ambe mc process` | 376 ROOT files | per-hit `__pulses.parquet` + `__clusterfinder.parquet` |
| 1 | `ambe mc features` | pulses parquet | per-cluster `__cluster_features.parquet` (OPTICS + CF) |
| 2 | `mva_analysis.py --keep-prompt-bkg --save-model` | features parquet | frozen model `.pkl` + scores + summary + ROC pdf |
| pick | `pick_best_model_cut.py` | scores parquet | best model + 80%-eff cut (→ rf, 0.52) |
| 3 | score data + rate + multiplicity + capture-time | data feature parquets | per-streamline scored parquet, rate CSV, capture CSV, multiplicity pdf |
| 4 | `benchmark_selection_2x2.py` | features + frozen model | box-cut vs MVA eff/purity CSV+pdf |

---

## STAGE 0 — Processing (ROOT → hits)

| File | Size | Contents |
|---|---|---|
| `cc_neutrino_productionv2/parquet/cc_neutrino_productionv2__pulses.parquet` | 1.4 GB | 78,060,262 per-hit rows, truth-matched (DirectParent), `cc_pass` joined |
| `cc_neutrino_productionv2/parquet/cc_neutrino_productionv2__clusterfinder.parquet` | 46 MB | ClusterFinder sidecar |

**Numbers:** 78.06M hits. CC-passing ≈ 3% of events (ccinc_truth stream: trueCC, FSLPdg==13, p_μ∈[600,1200) MeV/c, cosθ>0.8, FV r<100cm |Y|<100cm, nhits≥4). Delayed residual = CC-passing events, t > 2000 ns.

---

## STAGE 1 — Clustering & features

| File | Size | Contents |
|---|---|---|
| `cc_neutrino_productionv2/parquet/cc_neutrino_productionv2__cluster_features.parquet` | 16 MB | 26,585 OPTICS + 28,764 CF clusters, 31 features, truth labels |
| `cc_neutrino_productionv2/csv/cc_neutrino_productionv2__feature_separation.csv` | — | per-feature sig/bkg separation |
| `cc_neutrino_productionv2/csv/cc_neutrino_productionv2__spurious_composition.csv` | — | bkg cluster hit composition |
| `cc_neutrino_productionv2/plots/cc_neutrino_productionv2__cluster_features.pdf` | 188 KB | feature dists sig vs bkg |
| `cc_neutrino_productionv2/plots/cc_neutrino_productionv2__feature_separation_plots.pdf` | 109 KB | separation plots |

*(The `_inclprompt` run symlinks this same features parquet — no recompute.)*

**Cluster counts:**

| | OPTICS | ClusterFinder |
|---|---:|---:|
| Total clusters | 26,585 | 28,764 |
| Neutron-dominated (signal) | 21,285 | 22,570 |
| Background-dominated | 5,300 | 6,194 |
| `is_prompt_cluster==1` | 8,996 (33.8%) | 10,178 (35.4%) |
| Mean cluster purity (`frac_neutron`) | 0.651 | 0.658 |
| `is_truth_neutron==1` (strict ≥50% one-trackID) | 14,453 | 16,789 |

**Truth composition (dominant_class):** primary n ~35%, sec n←n ~38%, sec n←p ~5%, non-neutron(-5) ~20%, darknoise <1%.

**Truth-label definitions:**
- per-HIT `truth_class`: 1=primary n, 2=sec n←p, 3=sec n←n, 4=sec n←other, 0=darknoise, −5=non-neutron physics; `is_neutron` = class∈{1,2,3,4}.
- per-CLUSTER `dominant_class`: majority-vote truth_class (MVA signal = ∈{1,2,3,4}).
- per-CLUSTER `is_truth_neutron`: strict — dominant neutron trackID contributes ≥50% (MIN_MATCH_FRAC) of cluster hits.

---

## STAGE 2 — MVA training  ✅ USE inclprompt

| File | Size | Contents |
|---|---|---|
| **`cc_neutrino_productionv2_inclprompt/parquet/cc_neutrino_productionv2_inclprompt__mva_frozen__keepprompt.pkl`** | 33 MB | **NEW model — RF/GBT/XGB frozen** |
| `…_inclprompt/parquet/cc_neutrino_productionv2_inclprompt__mva_frozen__keepprompt__nn.keras` | 526 KB | NN companion |
| `…_inclprompt/parquet/cc_neutrino_productionv2_inclprompt__mva_scores__keepprompt.parquet` | 3.6 MB | per-cluster scores + `in_test` flag |
| `…_inclprompt/csv/cc_neutrino_productionv2_inclprompt__mva_summary__keepprompt.csv` | 2 KB | feature importances (RF/GBT/XGB) |
| `…_inclprompt/plots/cc_neutrino_productionv2_inclprompt__mva__keepprompt.pdf` | 153 KB | ROC, score dists, importances, NN history |

**Training set:** signal=5,300, background=5,300 (1:1), n_train=8,455, 31 features. Background = all non-neutron-dominated clusters INCLUDING prompt (the `--keep-prompt-bkg` fix; tripled bkg from 1,773 → 5,300).

**Performance — all classifiers (test set, n=2,145):**

| Model | Test AUC | CV AUC | 80%-eff cut | bkg-rej @80% |
|---|---:|---:|---:|---:|
| Random Forest | **0.718** | 0.711 ± 0.007 | 0.521 | 0.531 |
| XGBoost | 0.714 | 0.706 ± 0.006 | 0.502 | 0.504 |
| GBT | 0.713 | 0.714 ± 0.004 | 0.518 | 0.516 |
| Neural Net | 0.713 | — | 0.556 | 0.515 |

vs OLD starved model (prompt-excluded, 1,773/side): RF AUC 0.587. **The prompt-inclusive fix gave +0.13 AUC.**

**Top features (RF importance):** pe_total (0.124), n_hits (0.067), beta1 (0.061), spatial_rms (0.054), n_hits_early (0.050), charge_bal_legacy (0.043). `pe_total` dominates (GBT importance 0.45).

**Operating point (adaptive, via `pick_best_model_cut.py`):** BEST_SCORE_COL = **rf_score**, BEST_SCORE_CUT = **0.52** (80% MC signal efficiency). Replaces the stale hardcoded 0.643 (which gave only 35% eff on this model).

---

## STAGE 3 — AmBe data scoring + downstream

Subdir: `ambe_data/`. Data feature inputs: `ambe_all__data_features_optics.parquet` (232,836 clusters), `ambe_all__data_features_cf.parquet` (93,809 clusters).

### OPTICS streamline — `ambe_data/ccinc_optics_mcbkg/`
| File | Contents |
|---|---|
| `ccinc_optics_mcbkg__scored.parquet` (61 MB) | 232,836 data clusters, rf/gbt/xgb/nn scores |
| `rate_best/ambe_all__neutron_rate_by_run.csv` | **31-run neutron rate (rf @ 0.52)** |
| `rate_best/ambe_all__neutron_summary.pdf` | rate-vs-run plots |
| `ccinc_optics_mcbkg__multiplicity.pdf` | multiplicity distribution |
| `capture_stageAB/stageAB_capture_lmfit_summary.csv` | τ/thermal fit per source position |
| `capture_stageAB/stageAB_capture_lmfit.pdf` | capture-time fit plots |

### ClusterFinder streamline — `ambe_data/ccinc_cf_mcbkg/`
Same file set (scored 31 MB + rate_best + multiplicity + capture_stageAB).

### Stage 3b — neutron rate (MVA rf≥0.52, 79,633 events)

| | OPTICS | ClusterFinder |
|---|---:|---:|
| Neutron clusters tagged | 38,053 | 66,824 |
| Events with ≥1 neutron | 36,707 | 62,542 |
| **Fraction events w/ neutron** | **0.461** | **0.785** |
| Mean multiplicity | 1.035 | 1.068 |
| Single-neutron fraction | 0.966 | 0.937 |

**Box-cut vs MVA (ClusterFinder, computed from scored parquet, same EVENT_KEYS):**

| Metric | Box-cut (PE<60,CB<0.5,nHits>10) | MVA (rf≥0.52) |
|---|---:|---:|
| Frac events w/ neutron | 0.654 | **0.789** |
| Mean multiplicity | 1.034 | 1.069 |
| Single-neutron frac | 0.968 | 0.937 |

→ **MVA recovers ~13% more neutron events** than box-cut (matches AmBe one-n/trigger expectation); box-cut slightly cleaner per-event.

### Stage 3d — capture-time fit (per source position)

| | OPTICS | ClusterFinder |
|---|---:|---:|
| **τ (mean ± std, 21 positions)** | **25.9 ± 5.8 µs** | **28.9 ± 2.7 µs** |
| τ median | 26.5 µs | 29.5 µs |
| Thermal time (median) | 10.0 µs | 7.2 µs |
| Reduced χ² (median) | 1.54 | 1.29 |
| Good physical fits | 20/21 (1 railed @ P3 y0) | 21/21 |

**Literature n-capture-on-H in water ≈ 25–30 µs — both selections validate. CF gives the cleaner, more stable τ.**

---

## STAGE 4 — Selection benchmark (MC eff/purity)

| File | Contents |
|---|---|
| `ambe_data/ccinc_optics_boxcuts/benchmark/selection_2x2_benchmark.csv` | OPTICS: box-cut vs MVA |
| `ambe_data/ccinc_optics_boxcuts/benchmark/selection_2x2_benchmark.pdf` | plot |
| `ambe_data/ccinc_cf_boxcuts/benchmark/selection_2x2_benchmark.{csv,pdf}` | CF: box-cut vs MVA |

**Composition truth (dominant_class), MVA frozen @0.521:**

| Clustering | Selection | Efficiency | Purity | Fakes |
|---|---|---:|---:|---:|
| OPTICS | box-cut | 0.837 | 0.866 | 2,766 |
| OPTICS | **MVA @0.521** | **0.840** | **0.972** | **524** |
| CF | box-cut | 0.752 | 0.851 | 2,982 |
| CF | MVA @0.521 | 0.829 | 0.855 | 3,170 |

→ **OPTICS+MVA is the cleanest selector: 97% purity, 5× fewer fakes than box-cut at equal efficiency.**

---

## Headline conclusions

1. **Prompt-inclusive background fix** (`--keep-prompt-bkg`) lifted MVA AUC 0.587 → 0.718 by tripling background stats (1,773 → 5,300/side). This is the decisive change.
2. **All 4 classifiers tie at ~0.71**; RF auto-selected. Adaptive cut = 0.52 (not stale 0.643).
3. **`pe_total` is the dominant discriminator.**
4. **OPTICS+MVA = best purity/fewest fakes (MC, 97%).**
5. **CF+MVA = best data rate (78.9% events-w-neutron, matches AmBe physics) + cleanest capture-time (28.9±2.7 µs).**
6. **Open tension:** OPTICS data rate (46%) ≪ CF (79%) — OPTICS appears to lose real neutron clusters in the delayed window; needs follow-up.
7. **Capture-time validates the chain** — τ ≈ 26–29 µs, consistent with n-capture on H.

---

## "Which file for which plot/number" cheat-sheet

| To present… | File |
|---|---|
| MC cluster stats / truth composition | `cc_neutrino_productionv2/parquet/…__cluster_features.parquet` |
| Model AUC / ROC / importances | `…_inclprompt/plots/…__mva__keepprompt.pdf` + `csv/…__mva_summary__keepprompt.csv` |
| AmBe neutron rate (OPTICS) | `ambe_data/ccinc_optics_mcbkg/rate_best/ambe_all__neutron_rate_by_run.csv` |
| AmBe neutron rate (CF) | `ambe_data/ccinc_cf_mcbkg/rate_best/ambe_all__neutron_rate_by_run.csv` |
| Capture-time τ (OPTICS / CF) | `ambe_data/ccinc_{optics,cf}_mcbkg/capture_stageAB/stageAB_capture_lmfit_summary.csv` |
| Multiplicity plots | `ambe_data/ccinc_{optics,cf}_mcbkg/…__multiplicity.pdf` |
| Box-cut vs MVA eff/purity | `ambe_data/ccinc_{optics,cf}_boxcuts/benchmark/selection_2x2_benchmark.csv` |

## DO-NOT-PRESENT (superseded files)
- `cc_neutrino_productionv2/parquet/…__mva_frozen.pkl` (13:31) — old starved 0.587 model
- `cc_neutrino_productionv2/parquet/…__mva_frozen__vs_offbeam.pkl` — offbeam experiment (dropped)
- `ambe_data/ccinc_*_mcbkg/rate_rf/` and `rate_gbt/` (14:47) — old-model rates
- any `__allmethods_scored*` parquets — intermediate scoring artifacts
