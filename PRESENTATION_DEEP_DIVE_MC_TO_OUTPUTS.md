# Technical Deep-Dive: MC Samples → Final AmBe Neutron-Tagging Outputs

Presentation-support document for the ξ=0.02 + composition-cut analysis (`cc_neutrino_xi02_compcut`).
Every claim below is code- or artifact-cited. Where the underlying script and the actual frozen
output disagree, both are shown and the artifact is marked as ground truth.

---

## ⚠️ Things to check/fix before you present

1. **The NN model was re-frozen after the report was written.** Live-loading the frozen
   `...__mva_frozen__keepprompt__nn.keras` gives AUC **0.7495**, 80%-eff cut **0.5687**,
   bkg-rejection **0.5796** — all three disagree with `REPORT_xi02_compcut_dualmodel.md`
   (0.7475 / 0.538 / 0.576). RF/GBT/XGBoost reproduce the report to 3–4 sig figs (deterministic,
   seeded), so this isn't measurement noise — the NN artifact changed and the markdown wasn't
   regenerated. **Any NN-cut plot in `slide_plots_ccinc_xi02/` (`*_NN.pdf`) was made at the old
   cut (0.538) and is now stale relative to the live model.** The RF-only "presentable" set
   (`OPTICS_RF__presentable_*`) is unaffected — it never used the NN cut.
2. **The training set size in the report is wrong.** §3 of `REPORT_xi02_compcut_dualmodel.md`
   says "5,300 : 5,300" signal:background — that's actually the productionv2 (ξ=0.10) column
   from §2, pulled into the wrong row. The frozen ξ=0.02 model was actually trained on
   **6,729 : 6,729** (confirmed live from the `.pkl` metadata and cross-checked against the
   scored parquet).
3. **"412 GENIE-WCSim files" is a frozen snapshot from 2026-06-11**, hardcoded in the report and
   in `make_collab_slides_xi02_compcut.py` — not computed live. A live glob of the same input
   directory today (2026-07-20) returns **459** files. If you state a file count on a slide, say
   "as of the June analysis" or re-run the count.
4. **`cc_neutrino_xi02_compcut.yaml` has a one-off config anomaly, likely a bug.** Its `residual:`
   block sets `prompt_window_ns: 10000` (10 µs). Every other config in the repo — productionv2,
   ccinc_test, ccinc_truth, every OPTICS-sweep and `sweep*` variant, and even the `_cftrain` sibling
   of this same run — sets `prompt_window_ns: 2000` (2 µs), matching both the code's own default
   constant (`PROMPT_WINDOW_NS = 2000.0`, `cc_selection.py:92`) and this file's own `description:`
   text ("t > 2000 ns"). So the intended/designed delayed window is **2 µs**, and the
   `cc_neutrino_xi02_compcut` run that produced the frozen model and every number in this document
   actually executed with a **10 µs** window instead — confirmed directly (`apply_residual_filter`
   reads `prompt_window_ns` straight from the YAML, no other override). This is a real analysis
   question, not just stale prose: present the intended 2 µs value on slides, but flag internally
   that the underlying pipeline run may need to be redone at 2000 before the numbers are considered
   final, unless there's a deliberate reason for the 10000 that isn't recorded anywhere in the repo.
5. **Two different "box cuts" exist in the same slide folder.** Stage 4's benchmark table (below)
   uses PE<60 / CB<0.5 / nHits>10; the standalone multiplicity script's box cut uses PE<80 / CB<0.45
   / nHits>9. Both are legacy reference cuts (explicitly labeled "not an MVA result" where they
   appear), but if you show both on one slide, don't caption them with a single shared threshold.

Everything else below reconciled cleanly across code and artifacts.

---

## Stage 0 — Building the MC sample and defining "signal"

### Where the events come from
GENIE tune `G1810a0211a/standardv1.0/tank` (water target = the ANNIE tank), neutrino flux and
generation on the grid (`grid_wcsim/genie_samples/submit_wcsim_job.sh`), then propagated through
the full optical/PMT simulation with **WCSim** inside a singularity container
(`wcsim_container.sh`). Output per job: one `wcsim_<RUN>.root`, converted downstream (outside this
repo, via the ANNIE **ToolAnalysis** `ANNIEEventTreeMaker` chain) into the `ANNIEEvent_cc_neutrino_*.root`
files this pipeline actually reads (`configs/cc_neutrino_xi02_compcut.yaml`).

**Important:** GENIE generates *generic* neutrino interactions — there is no CC-only filter at
generation time. The CC selection described below is applied entirely offline, on truth branches,
after the fact.

### Stage 0 processing (`ambe mc process`)
`src/ambe/mc/processor.py`, dispatched from `src/ambe/cli.py`. For every ROOT file: reads the
`"Event"` tree via uproot, emits **one row per PMT hit** — every hit, whether or not truth-matched
— with columns `eventID, pmtID, t, x, y, z, pe, truth_class, is_darknoise, is_neutron, is_untraced,
bg_class, bg_pdg, ...`. Truth is attached by matching each hit's (chankey, time) against the
WCSim/BackTracker `DirectParent_*` branches, with an 8 ns match tolerance and a 15 ns SPE-waveform
offset. Two outputs per file, streamed to bound memory: `__pulses.parquet` (per-hit) and a
`__clusterfinder.parquet` **sidecar** (see Stage 1 — this is just copied-through ROOT branches, not
computed here).

**Hit-level truth classes** (BackTracker convention, `processor.py`):

| Class | Meaning |
|---|---|
| 0 | dark noise |
| 1 | primary neutron (capture γ from the original AmBe/FS neutron) |
| 2 | secondary, neutron ← proton |
| 3 | secondary, neutron ← neutron |
| 4 | secondary, neutron ← other |
| −5 | non-neutron physics background |

`is_neutron = (truth_class ∈ {1,2,3,4})` — this hit-level flag is the seed for every downstream
truth quantity (cluster composition, `dominant_class`, the composition cut, etc.).

### The CC event selection ("same selection as James")
Implemented as a registry of named cuts in `src/ambe/mc/cc_selection.py`; the stream used here
(`ccinc_truth`) chains: `cc → fsl_muon → mu_p → cos_theta → fv_radius → fv_y → fv_z → nhit`.

| Cut | Exact expression | Physical meaning |
|---|---|---|
| `cc` | `trueCC == 1` | charged-current interaction (vs. NC) |
| `fsl_muon` | `trueFSLPdg == 13` | the **final-state** lepton is a muon — an identity cut, not a track-count/topology cut (the code explicitly avoids the legacy `|truePrimaryPdg|==13`, which over-counts NC events) |
| `mu_p` | `600 ≤ \|p_μ\| < 1200` MeV/c | muon energetic enough to reconstruct, excludes high-energy outliers |
| `cos_theta` | `cosθ (vs. beam/Z) > 0.8` | forward-going muon (< ~37°), matches CC quasi-elastic-like kinematics |
| `fv_radius` | cylindrical `R = hypot(x,z) < 100 cm` | vertex fiducial cut: away from barrel PMTs/wall |
| `fv_y` | `\|y\| < 100 cm` | vertex fiducial cut: away from top/bottom |
| `fv_z` | off in this run (`fv_require_z_negative: false`) | (available but not applied here) |
| `nhit` | `nhits ≥ 4` | minimum activity to be reconstructible |

FV = fiducial volume: restricting the **true interaction vertex** to a sub-volume of the tank away
from PMT-covered surfaces, so light collection is well-behaved and wall/dead-material effects don't
contaminate the sample.

**Delayed-residual (signal) window:** among CC-passing events, keep only hits with
`t > prompt_window_ns` (after the prompt/beam flash) — this isolates the neutron thermalization +
capture-γ window from the prompt muon/interaction light. Intended/designed value across the whole
repo is **2000 ns (2 µs)**; the specific `cc_neutrino_xi02_compcut.yaml` config that produced this
run's frozen model instead has it set to 10,000 ns — see correction #4 above, this looks like a
one-off config bug rather than an intentional choice.

### The composition cut (MC-only)
`reject cluster if (n_neutron ≤ 10) AND (frac_nonneutron ≥ 0.40)` — removes prompt-γ /
non-neutron-physics-dominated clusters before MVA training/benchmarking. Both `n_neutron` and
`frac_nonneutron` are truth quantities (computed from hit-level `truth_class`), so **this cut only
exists on MC** — the real AmBe data parquet has no truth columns, and the pipeline explicitly
detects that and passes data through unfiltered.

---

## Stage 1a — Pre-selection / clustering: OPTICS vs. ClusterFinder

Both methods start from the *same* per-event hit collection (`df_ev`) and are run side-by-side in
one loop (`cluster_features.py`), producing two parallel labeled subsets — `method="optics"` and
`method="clusterfinder"` — that flow through identical downstream feature extraction.

### OPTICS
Confirmed to be literally `sklearn.cluster.OPTICS` (`src/ambe/mc/optics.py`):
```python
OPTICS(min_samples=min_samples, xi=xi, metric=metric).fit_predict(X_scaled)
```
**Feature space it clusters on** — 4 dimensions per hit: `x, y, z` (StandardScaler-normalized) plus
a time axis `t / t_unit_ns` (deliberately **not** re-standardized, so `t_unit_ns` stays a physically
interpretable knob — "1 OPTICS distance unit in time = `t_unit_ns` nanoseconds"). Optionally
ToF-corrects `t` for a known AmBe source position first.

**Parameters used for this run** (`run_ccinc_xi02_compcut.sh`): `min_samples=8`, `xi=0.02`,
`t_unit_ns=25`. `cluster_method` is never overridden, so sklearn's default **`cluster_method='xi'`**
applies: clusters are cut from the reachability plot wherever a point-to-successor reachability
ratio changes by more than `1 − xi` — i.e., **`xi` is the minimum steepness that counts as a cluster
boundary**. A smaller `xi` (0.02, tighter than the 0.10 baseline) demands a sharper density drop to
call something a new cluster, so it splits overlapping/merged deposits more aggressively — exactly
the tradeoff described in the June→June-11 changelog (cleaner clusters, but some real captures can
now fall below `min_samples` or get over-split).

**Output:** one integer label per hit (`-1` = noise); each label ≥0 becomes one candidate
neutron-capture cluster. Not guaranteed 1-true-capture-per-cluster — split/merge/spurious outcomes
are explicitly tracked as diagnostic counters in the code.

### ClusterFinder (CF)
**Not implemented in this repo.** There is no clustering algorithm for CF anywhere in the codebase.
What this repo does is:
1. Read `clusterTime`, `clusterPE`, `clusterHits` as **verbatim ROOT branches** produced upstream by
   the ANNIE ToolAnalysis chain (external, not in this repo) — stored as a "ClusterFinder sidecar"
   parquet at Stage 0.
2. `assign_clusterfinder_labels()` then does **membership lookup only**: each hit is assigned to the
   nearest pre-existing `clusterTime` value within a fixed **[−5, +20] ns window**. That's it — no
   density estimation, no clustering logic, just "which pre-made cluster is this hit close to in time."

So when a slide says "OPTICS vs. ClusterFinder," the honest framing is: *OPTICS is a clustering
algorithm this analysis runs and tunes; ClusterFinder is a fixed external/legacy clustering whose
result is simply re-associated with hits here.* This is an asymmetric comparison, not two competing
implementations of the same step.

### Why OPTICS was adopted alongside/over CF
Straight from the code's own rationale comment (`cluster_discrimination.py`): *"AmBe emits one
neutron per source event, so ~98–100% of triggered events should have exactly one neutron cluster.
In practice ClusterFinder returns multiple clusters in >19% of events, polluting the neutron
multiplicity distribution."* OPTICS was built as a tunable, density-based alternative to try to fix
that over-fragmentation, with `t_unit_ns`/`xi`/`min_samples` as the knobs to trade off
splitting-power vs. capture-efficiency.

---

## Stage 1b — Cluster feature extraction & truth labeling

`src/ambe/mc/cluster_features.py` defines a candidate list of **33 physics features**
(`PHYSICS_FEATURES`). For this run, 2 of them (`d_source`, `d_source_fit` — distance to the AmBe
source position) have 0% coverage because no `source_position_m` was configured in this YAML, so
the MVA pipeline's ≥50%-non-null filter drops them, leaving the **31 features** actually used and
quoted throughout the reports.

### Feature groups (one line each)

| Feature | What it measures |
|---|---|
| `n_hits`, `pe_total` | cluster size: hit count / total photoelectrons |
| `n_hits_early` | hits within ±10 ns of the (offset-corrected) cluster median time — the "direct light only" count |
| `sigma_t_mad*` (raw / PMT-offset-corrected / ToF-corrected / early-window) | robust (MAD-based) timing spread of the cluster, in a few different correction states |
| `t_window_80pct` | narrowest time window containing 80% of the cluster's hits — a vertex-independent compactness measure |
| `pe_balance` | quadrant (XZ) charge asymmetry, (max−min)/total, ∈[0,1] |
| `charge_bal_legacy` | the classic ANNIE box-cut charge-balance variable, √(ΣQ²/(ΣQ)² − 1/121) per tube — asymmetry of charge across PMTs |
| `spatial_rms` | RMS of the 3D distance from each hit PMT to the **PE-weighted centroid** vertex, in metres (no fitted-vertex analog exists for this one) |
| `d_wall`, `d_wall_fit` | distance from vertex (centroid / fitted) to the nearest tank surface — contained vs. near-wall |
| `vtx_y`, `vtx_fit_y` | vertical coordinate of the vertex (flagged in code as tracking a hit-count systematic from fewer bottom PMTs) |
| `beta1`…`beta5` (+ `_fit` variants) | see below |
| `fit_converged`, `n_fit_hits`, `fit_rms_ns` | diagnostics of the Gauss-Newton vertex fit (did it converge, on how many hits, residual RMS) |
| `fit_goodness_reco`, `fit_goodness_init` | Super-K–style "FitGoodness": weighted-Gaussian timing-consistency score, evaluated at the fitted vertex vs. the centroid — larger = timing pattern more consistent with a single point-like flash |

### β₁, β₂ — precise definition (this is a good slide to build out)
Not a decay-time-profile parameter. It's a purely **geometric/angular isotropy moment**, computed
once per cluster from static hit-PMT directions (Super-K convention, arXiv:2505.04409 eq. 3.4):

```
β_k = 2 / (N(N−1)) · Σ_{i≠j} P_k( cos θ_ij )
```

where the unit vectors point from the reconstructed vertex to each hit PMT, `cos θ_ij` is the
pairwise angle between them, and `P_k` is the degree-k Legendre polynomial. Interpretation:
- **β_k > 0** — hits cluster on one side of the vertex (a directional, Cherenkov-cone-like pattern —
  consistent with light from a moving relativistic charged particle).
- **β_k ≈ 0** — isotropic — consistent with a point-like, direction-less emitter (a neutron-capture
  γ cascade, or dark noise).
- **β_k < 0** — hits anti-correlated / on opposing sides.

RF's top features are `pe_total`, `spatial_rms`, `beta2`, `beta1`, `d_wall`, `charge_bal_legacy` —
i.e., the discriminator is mostly "how much light, how spread out spatially, and how isotropic is
the angular pattern" — all quantities computable on real data with no MC truth needed, which is
exactly what licenses the MC→data transfer.

### Truth/composition labels
Per cluster (`_hit_composition()`): `n_neutron`, `n_darknoise`, `n_nonneutron`, `n_untraced` (raw
hit counts by truth class) and their fractions `frac_neutron`, `frac_nonneutron`, `frac_darknoise`.
`dominant_class` = the truth class with a plurality of hits in the cluster.

There are **three different, easily-confused "is this cluster a neutron?" definitions** in the
codebase — worth stating explicitly in the talk since they produce different-looking percentages
for what sounds like the same question:
- `dominant_class ∈ {1,2,3,4}` (plurality across all 4 neutron sub-classes) → **77.2%** of OPTICS
  clusters, per the report.
- `frac_neutron ≥ 0.5` (majority of hits are *any* neutron class).
- `is_truth_neutron` (strict): the single dominant **trackID** (not just class) must itself be a
  genuine truth-neutron track, **and** that one trackID must own ≥50% of the cluster's hits
  (`MIN_MATCH_FRAC = 0.5`) → **52.4%**, the stricter number.

`is_prompt_cluster` = (cluster mean time is >500 ns *before* the event's median neutron-hit time)
**OR** (`frac_nonneutron > 0.5` and `n_neutron ≤ 2`) — i.e., either clearly early in time, or
clearly non-neutron-dominated with almost no neutron hits at all.

---

## Stage 2 — The four MVA classifiers

All four are trained on the **same 31-feature vector** per cluster, same labels
(signal = `dominant_class ∈ {1,2,3,4}`, background = everything else), same training set. Loaded
directly from the frozen `.pkl`/`.keras` artifacts (ground truth, not just the training script).

### Training set (ground truth from the frozen artifact — corrects the report)
Signal 6,729 : background 6,729 (**1:1**, not the "5,300:5,300" in §3 of the report — that number
belongs to the ξ=0.10 productionv2 column). `--keep-prompt-bkg` means prompt clusters are **kept**
as background rather than excluded — confirmed live: 71% (4,808/6,729) of background clusters are
prompt clusters. Background capped at `max_ratio=1.0`× signal. Train/test split is done at the
**event level** (all clusters from one event stay on the same side — stratified 80/20,
`random_state=42`), which avoids leaking information about the same physical event across the
train/test boundary.

### Random Forest (the frozen baseline)
`n_estimators=300`, `min_samples_leaf=5`, `max_features='sqrt'`, `bootstrap=True`,
`random_state=42`; imbalance handled via `sample_weight` (not `class_weight`). Top features:
`pe_total` (0.124), `spatial_rms` (0.106), `beta2` (0.074), `beta1` (0.052), `d_wall` (0.042),
`charge_bal_legacy` (0.038).

### Gradient-Boosted Trees
`n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`. Notably more concentrated
than RF — top 2 features (`pe_total`, `spatial_rms`) alone carry 62% of total importance (vs. 23%
for RF).

### XGBoost
`n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`,
`tree_method='hist'`. Feature ranking pulls in `n_hits_early` (0.10) more than the other two models.

### Neural Network — full architecture (live-loaded from `.keras`)
```
Input(31)
 → [Dense(128, linear) → BatchNorm(momentum=0.99) → ReLU → Dropout(0.3)]  ×3
 → Dense(1, sigmoid)
```
**38,017 trainable parameters** (768 more are BatchNorm running stats; the 114,821 "total param"
figure some tools report includes ~76k Adam optimizer state — don't quote that number as model
size). Loss: `binary_crossentropy`; optimizer: Adam, initial LR 1e-3. Training config (from source,
not separately serialized): up to 100 epochs, batch size 256, 15% validation split, `EarlyStopping`
on `val_auc` (patience 15, restores best weights), `ReduceLROnPlateau` on `val_auc` (factor 0.5,
patience 7, floor 1e-5) — the live artifact's saved optimizer state shows LR = 0.000125, i.e. three
factor-of-2 reductions from 1e-3, so plateaus were hit at least 3 times before the early stop.

**Feature scaling — an asymmetry worth calling out on a slide:** the NN's 31 inputs are
`StandardScaler`-normalized (fit on the training fold only); RF/GBT/XGBoost get the **raw,
NaN-median-imputed** features — trees don't need scaling, and the pipeline's own metadata literally
notes them as "scale-free."

### AUC / operating points (recomputed live from the scored parquet — ground truth)

| Model | Test AUC | 80%-eff cut | bkg-rejection @80% |
|---|---:|---:|---:|
| Random Forest | 0.7545 | 0.5474 | 0.5956 |
| GBT | 0.7535 | 0.5533 | 0.5891 |
| XGBoost | 0.7488 | 0.5373 | 0.5796 |
| **Neural Net** | **0.7495** | **0.5687** | 0.5796 |

(NN row differs from the previously-circulated report — see correction #1 at the top.)

### How "best model" + operating cut are chosen
`pick_best_model_cut.py`: best = highest test-set AUC (currently RF, by 0.001 over GBT — all four
are statistically close). Operating cut = the score below which only `100×(1−efficiency)`% of true
*test-set signal* falls — e.g., the 20th percentile of the signal-class test scores gives an 80%
signal-efficiency cut, by construction (not tuned against data).

---

## Stage 3 — Scoring real AmBe data + physics validation

The frozen model(s) score real AmBe data directly (no composition cut — no truth columns exist on
data, so that step is explicitly skipped and the data passed through). Two independent clustering
streamlines (OPTICS-scored, CF-scored) run through the same frozen classifier(s).

### Why the fixed 80%-efficiency point, not an "optimized" threshold
This is the strongest rationale slide available — it's a genuine ablation, not a claim:

| Config | τ (mean ± std) | good fits (of 21 positions) | single-neutron frac | verdict |
|---|---:|---:|---:|---|
| OPTICS RF @ 0.547 (80%-eff) | 26.7 ± 2.1 µs | 20/21 | 0.969 | ✅ keep |
| OPTICS RF @ 0.426 (Youden's-J) | 34.6 ± 2.5 µs | 10/21 | 0.825 | ❌ τ too long, fits collapse |
| OPTICS RF @ 0.344 (max-F1) | — | — | 0.349 | ❌ mean multiplicity 1.94 — over-tags fakes |
| OPTICS NN @ 0.420 (Youden's-J) | 35.9 ± 2.4 µs | 17/21 | 0.804 | ❌ τ too long |

Youden's-J and max-F1 raise MC efficiency to 94–97%, but the extra accepted clusters are enough fake
background to drag the fitted capture time to ~35 µs — outside the 25–30 µs literature band for
neutron capture on hydrogen in water — and collapse fit quality. **The fixed 80%-efficiency point is
the only one where both RF and NN land in the physically correct band with stable fits across all 21
AmBe source positions**, which is the actual justification for "why 80%" rather than a
data-optimized number.

### Per-run neutron rate (`summarize_ambe_neutrons.py`)
One row = one AmBe run = one source position (31 runs). Key columns: `n_events` (beam triggers with
≥1 delayed cluster — the denominator), `n_events_with_neutron`, `frac_events_with_neutron` (the
quoted "detection rate" — **conditional on delayed activity already existing**, not an absolute
fraction of all triggers), `mean_mult`, `single_neutron_frac`. Because AmBe emits exactly one
neutron per decay, `mean_mult ≈ 1` and `single_neutron_frac ≈ 0.93–0.97` is the expected signature —
it's a cross-check that the selection finds real single neutrons rather than fragmenting captures
into fakes, not just a nice-to-have number.

| Streamline | Model | frac. events w/ neutron | mean mult. | single-n frac. |
|---|---|---:|---:|---:|
| OPTICS | RF | 0.438 | 1.034 | 0.968 |
| OPTICS | NN | 0.609 | 1.091 | 0.918 |
| CF | RF | 0.814 | 1.069 | 0.936 |
| CF | NN | 0.878 | 1.080 | 0.928 |

NN tags more events (closer to the ideal "1 neutron per trigger"); RF holds the higher single-neutron
purity (fewer events fragmented into multiple fakes). OPTICS under-counts vs. CF in both models —
an open item, plausibly because the tighter ξ=0.02 splits some real captures below `min_samples=8`
or below the MVA cut.

### Capture-time fit (`capture_time_stageAB_from_parquet.py`, lmfit)
Fits a thermalization + exponential-capture + flat-background model (`NeutCapture`) to the cluster
time distribution, bounded to the 2–70 µs capture window.

| Streamline | Model | τ (mean ± std) | median | good fits |
|---|---|---:|---:|---:|
| OPTICS | RF | 26.7 ± 2.1 µs | 26.5 | 20/21 |
| OPTICS | NN | 30.3 ± 2.4 µs | 31.0 | 20/21 |
| CF | RF | 29.0 ± 2.9 µs | 29.9 | 21/21 |
| CF | NN | 29.2 ± 2.5 µs | 29.6 | 21/21 |

All four validate against the 25–30 µs literature value for n-capture-on-hydrogen in water; OPTICS+RF
gives the tightest, lowest τ.

---

## Stage 4 — MC selection benchmark (efficiency / purity / fakes, natural population)

`benchmark_selection_2x2.py`, evaluated on the *natural* cluster population (not the artificially
balanced 1:1 training sample — the report explicitly flags this distinction since a "63–66% purity"
number sometimes floats around from the training-sample denominator and is not the number to quote).

| Clustering | Selection | Efficiency | Purity | Fake clusters |
|---|---|---:|---:|---:|
| OPTICS | box-cut (PE<60, CB<0.5, nHits>10) | 0.810 | 0.977 | 435 |
| **OPTICS** | **MVA RF @ 0.547** | **0.834** | **0.996** | **80** |
| CF | box-cut | 0.755 | 0.982 | 324 |
| CF | MVA RF @ 0.547 | 0.854 | 0.981 | 390 |

OPTICS + RF MVA is the cleanest selector: 99.6% purity at 83% efficiency, only 80 fake clusters
(vs. 524 in the prior June-8 poster baseline — a 6.5× reduction in fakes at equal efficiency).

---

## Bottom-line framing for the talk

- **MC:** all four classifiers agree almost exactly on the truth ranking (AUC within ~0.006 of each
  other, ≈0.75) — the discriminating physics (mostly total light + spatial spread + angular
  isotropy) is robust to which algorithm you pick.
- **Data:** NN tags substantially more events (closer to the "1 neutron per trigger" physical
  expectation); RF is more conservative/pure (cleanest τ, highest single-neutron fraction, fewest MC
  fakes). This spread is a **selection-efficiency systematic bracket**, not a contradiction.
- **Recommendation:** present RF as the frozen, defensible baseline; NN as a high-efficiency
  cross-check. Don't switch to NN because its data rate "looks more physical" — that would be tuning
  on data rather than on MC truth.
- **Open items to mention:** the OPTICS-vs-CF data-rate gap (OPTICS under-counts in both models —
  plausibly the ξ=0.02 splitting real captures below the clustering floor), and that
  `frac_events_with_neutron` is conditional on pre-existing delayed activity, not an absolute
  trigger-level efficiency.

## Source map (for anyone who wants to double-check a number)

| Topic | File |
|---|---|
| MC generation | `grid_wcsim/genie_samples/submit_wcsim_job.sh`, `wcsim_container.sh` |
| Stage 0 (ROOT→pulses) | `src/ambe/mc/processor.py` |
| CC selection | `src/ambe/mc/cc_selection.py` |
| OPTICS clustering | `src/ambe/mc/optics.py` |
| ClusterFinder membership lookup | `src/ambe/mc/optics.py::assign_clusterfinder_labels`, sidecar written in `processor.py` |
| Feature extraction / truth labeling | `src/ambe/mc/cluster_features.py` |
| MVA training | `mva_analysis.py`; frozen artifacts under `ambe_output/cc_neutrino_xi02_compcut/parquet/` |
| Threshold selection | `pick_best_model_cut.py` |
| Data scoring pipeline | `run_ccinc_xi02_compcut.sh` |
| Per-run rate | `summarize_ambe_neutrons.py` |
| Capture-time fit | `capture_time_stageAB_from_parquet.py` |
| MC benchmark | `benchmark_selection_2x2.py` |
| Narrative numbers (partially stale — see corrections) | `REPORT_xi02_compcut_dualmodel.md` |
