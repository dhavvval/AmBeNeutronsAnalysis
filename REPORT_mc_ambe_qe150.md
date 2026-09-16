# AmBe MC productionv2 QE 1.50 — box and MVA decks, and the multi-cluster provenance question

> **See also — `REPORT_ccinc_v3_cluster_provenance_timewindows.md` (2026-09-11).**
> The §2 splitting result below does **not** transfer to CCinc: ClusterFinder splits only 0.40 %
> of neutrons there, ~40× rarer than the 17.9 % measured here. The AmBe effect is one capture's
> light straddling the clustering window; CCinc's several neutrons are microseconds apart and get
> correctly distinct clusters. On CCinc the dominant defect is the opposite one — *merging*
> several neutrons into one cluster.

Dataset: 1250 files × 1000 events = **1,250,000 events**, 25 source positions
(5 ports × 5 heights), `/pnfs/annie/persistent/users/dajana/output/AmBe_MC_toolchain/productionv2/QE_1.50`.
Every event has exactly **one primary neutron** (PDG 2112) and at most one true capture.

**No efficiency of any kind is reported here.** MC has no waveforms, so there is no
trigger denominator comparable to the data's IC-gated one. The efficiency and
residual-efficiency heatmaps are suppressed in all four decks.

**No classifier was trained.** The MVA decks score with the frozen CCinc-neutrino model
`cc_neutrino_v3_truthtag__mva_frozen__keepprompt__merged__cf.pkl`, GBT, eff80,
threshold **0.423621** — the same model and cut already used on the AmBe v4 data.
Only `src/ambe/mc/mva_analysis.py --score-data` was invoked.

## 1. The four decks

| Deck | total events | prompt-vetoed | single | multiple | unique neutron triggers | τ (25 pos, weighted) | χ²/ndof |
|---|---|---|---|---|---|---|---|
| box, veto ON  | 1,250,000 | 54,234 (4.34 %) | 902,786 | 3,318 | 906,104 | 23.810 ± 0.055 µs | 1.02 |
| box, veto OFF | 1,250,000 | 0 | 902,786 | 17,345 | 920,131 | 23.810 ± 0.055 µs | 1.07 |
| MVA, veto ON  | 1,250,000 | 54,234 (4.34 %) | 770,383 | 3,191 | 773,574 | 23.655 ± 0.061 µs | 1.03 |
| MVA, veto OFF | 1,250,000 | 0 | 770,507 | 15,275 | 785,782 | 23.651 ± 0.060 µs | 1.08 |

### 1.1 Capture time is 5.6 µs below the data

τ_MC = **23.810 ± 0.055 µs** against the frozen v4 data anchor **29.417 ± 0.221 µs**
(same recipe: 70 bins, 2–67 µs window, B fixed at 0, lmfit). That is a **≈25 σ**
difference, far outside the per-position spread (τ range 23.26–24.32 µs, χ²/ndof ≈ 1.0
across all 25 cells). It is a statement about the simulation's capture model, not a fit
artefact. The prompt veto does not move it; the MVA moves it by only −0.16 µs.

### 1.2 What the prompt veto costs

The **single**-neutron count is identical with the veto on and off, because a sub-2 µs
cluster fails the box's own `t ≥ 2 µs` requirement anyway. What the veto costs is
**multi-cluster events** (3,318 → 17,345): an event containing a prompt cluster is
discarded *entirely*, taking its genuine delayed capture cluster with it. Net yield
effect **+1.55 %** (box) / **+1.58 %** (MVA) with the veto off.

In this MC the veto is not a cosmic veto — there are no cosmics and no AmBe 4.4 MeV
gamma. It removes neutron-induced prompt light. `clusterPE` never exceeds 100, so the
PE half of the veto never fires.

### 1.3 MVA coverage — read the veto-OFF deck carefully

| Deck | coverage | keep rate (of SCORED clusters) |
|---|---|---|
| MVA veto ON  | 92.67 % | 83.88 % |
| MVA veto OFF | **87.94 %** | 83.89 % |

The veto-OFF deck falls below the 90 % rule, and the reason is structural and not
fixable: the feature extraction keeps only the delayed residual (t > 2 µs), so a cluster
with `clusterTime < 2000` can never be scored. With the veto off those clusters *are*
queried and read as rejected. Widening the window would be worse — a prompt cluster is
outside the frozen model's training domain. **Quote the veto-OFF MVA keep rate over
scored clusters, with the prompt-cluster count stated separately.** The veto-OFF MVA
deck recovers events that *contained* a prompt cluster, never the prompt cluster itself.

Scoring emitted **no "feature(s) absent" warning** — all 31 features present, nothing
median-filled.

## 2. Multi-cluster provenance

19,028 events (1.52 %) yield ≥2 ClusterFinder clusters. **The answer inverts depending on
whether you ask about all clusters or only the selected ones.**

| Variant | N events (≥2 accepted) | SPLIT | DISTINCT NEUTRONS | other |
|---|---|---|---|---|
| all CF clusters | 19,028 | **17.90 %** [17.62, 18.18] | **77.00 %** [76.70, 77.31] | 0.76 % both, 4.34 % unlabelled |
| box, veto ON | 2,770 | **99.71 %** [99.59, 99.80] | 0.11 % | 0.18 % |
| box, veto OFF | 2,813 | **99.64 %** [99.51, 99.74] | 0.14 % | 0.21 % |
| MVA, veto ON † | 316 | 99.37 % | 0.63 % | — |
| MVA, veto OFF † | 329 | 99.09 % | 0.91 % | — |

(68 % Wilson intervals. † MVA variants have 27.6 % / 59.8 % score coverage *within this
multi-cluster subsample*, because its clusters are disproportionately prompt and prompt
clusters cannot be scored. Direction is unambiguous; the precise fractions are not
quotable.)

**Physically most multi-cluster events are two different neutrons. Operationally,
essentially every multi-cluster event that survives the cuts is one neutron counted twice.**

### 2.1 The mechanism

| | SPLIT clusters (6,812) | DISTINCT-NEUTRON clusters (29,305) |
|---|---|---|
| median cluster time | 18.53 µs | **0.14 µs** |
| prompt (t < 2 µs) | 2.3 % | **50.9 %** |
| passes the box | 89.1 % | 44.8 % |
| dominant ancestor class | 99.6 % class 1 (primary) | 50 % class 3 (secondary from a neutron), 50 % class 1 |

A "different neutrons" event is one **prompt** cluster from a secondary knock-on neutron
at t ≈ 0 plus the **delayed** capture cluster of the primary. Exactly half its clusters
are prompt, and those fail `clusterTime ≥ 2 µs`, so the event drops to one accepted
cluster and never enters the selected multi-cluster sample.

### 2.2 Cross-checks (two use no track ID at all)

- **Pair Δt separates by three orders of magnitude.** Same-neutron pairs: median
  **15.0 ns**, p90 21.5 ns — the ClusterFinder window scale, i.e. one capture's light
  straggling. Different-neutron pairs: median **18.8 µs**, p90 47 µs — which is just the
  capture time, one cluster being at t ≈ 0 and the other the capture.
- **19,331 of 19,350 multi-cluster events have exactly ONE true capture** (11 have two).
  "Different neutrons" almost never means two captures; the second cluster is
  non-capture recoil light. Consistent with one primary neutron per event.
- **Every distinct-neutron event contains a secondary neutron** (class 2/3/4), which is
  required — one primary cannot supply two track IDs.
- **Threshold scan:** split fraction 18.97 % → 18.60 % as the dominance threshold goes
  0.4 → 0.7. Not threshold-driven.
- **Position spread:** split fraction 10.2–21.7 % across the 25 positions (all-cluster
  view), 98.7–100.0 % (box-accepted view).

### 2.3 Gate V1 failed, and what it actually means

Exact vs ±window membership agree on only **62.4 %** of cluster labels. The window pass
returns 16.76 % SPLIT, 0.03 % DISTINCT, **83.21 % AMBIGUOUS** — it cannot label 83 % of
clusters, because it drops every hit more than 20 ns after `clusterTime` and gives prompt
clusters no members at all.

So the disagreement is "window cannot label", not "window says something different": its
SPLIT count is 3,190 against the exact mode's 3,406, **agreeing to 6 %**. The
distinct-neutron population is exactly the prompt clusters the window rule is structurally
blind to. This is why exact membership is the right choice for this question, and the
window pass is reported as a cross-check rather than an alternative answer.

### 2.4 Other gates

| Gate | Result |
|---|---|
| V2 median untraced fraction ≤ 0.15 | PASS (0.083) |
| V3 `n_member_hits == clusterHits`, no unmapped hits, no key collisions | PASS (0 / 0 / 0 over 1.25 M events) |
| V4 control dt_cap IQR ≤ 50 ns, offset fallback ≤ 1 % | PASS (IQR **2 ns**, 0 fallbacks) |
| V5 same-neutron pairs with Δt > 1 µs ≤ 5 % | PASS (1.8 %) |
| V6 no DISTINCT event with two class-1 clusters | PASS (0) |
| V8 split fraction stable 0.5 → 0.7 | PASS (0.3 pts) |
| box cluster pass rate ≈ 88 % | 88.10 % on the control sample |

## 3. How to reproduce

```bash
# decks (Stage 2 with no Stage 1 -- MC has no waveforms)
ambe data processmc --config configs/mc_ambe_qe150_box_vetoon.yaml  --selection box --cosmic-veto on
ambe data processmc --config configs/mc_ambe_qe150_box_vetooff.yaml --selection box --cosmic-veto off
./run_mc_qe150_mva_chain.sh          # CF features -> frozen-model scoring -> MVA decks -> plots
./run_mc_qe150_plots.sh mc_ambe_qe150_box_vetoon mc_ambe_qe150_box_vetooff

# provenance
ambe mc match --config configs/mc_ambe_qe150_provenance.yaml --mode both
ambe mc match --config configs/mc_ambe_qe150_provenance.yaml --mode attach-mva
```

Outputs under `ambe_output/mc_ambe/<deck>/{plots,csv,parquet}`; trigger summaries and
capture-time fits in `TriggerSummary/`; candidate CSVs in
`EventAmBeNeutronCandidatesData/`.

## 4. Things a later session must not trip over

- **`eventTimeTank` is 0 in every MC event.** `eventNumber` is substituted in
  `load_event_data`; `runNumber = label_index*1000 + file_index` is unique per file, so
  `(run, eventNumber)` is globally unique. Downstream multiplicity and Δt code must group
  on `(run, eventTankTime)`, never `eventTankTime` alone — `plots/combined.py` and
  `plots/basic.py` were both fixed for this and would otherwise have merged one event
  from every run into a single group.
- **The MC label's `z` is the VERTICAL coordinate** and maps to `y` in the `(x, y, z)`
  source map, the same convention as runs 6273/6274/6275.
- **The 453/455 hitID swap in `plots/basic.py` is data-specific** and fires on any hitID
  string merely containing "453"/"455". MC configs set `apply_hitid_swap: false`.
- **The veto-ON scored table is derived, not extracted.** The prompt veto is a per-event
  `continue` taken before any feature is computed, so veto-ON is exactly veto-OFF minus
  rows from vetoed events; `scripts/workflows/derive_vetoon_scored.py` does that filter against the deck's
  own `is_cosmic` verdict and asserts the result. Extracting twice costs ~11 h for a row
  filter.
- **`ambe data processmc` is a separate command from `ambe data process`** and reads a
  `stage1_mc:` block, so neither config kind can be run through the wrong path.
