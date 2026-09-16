> Historical analysis note: results and commands describe the original run. See [the current workflow guide](../WORKFLOWS.md) and [output locations](../OUTPUTS.md) before rerunning. No numerical results were regenerated during cleanup.

# CCinc v3 — Cluster Provenance and Hit-Time Windows

**Date:** 2026-09-11
**Sample:** CCinc v3 truth-tag streamline — tank (498 files, 226,579,916 hits, 30,826 events)
and world (500 files, 12,626,193 hits, 48,372 events). Both clusterings.
**Code:** `ambe mc match --mode stage0|stage1 --method {clusterfinder,optics,both}`
(`src/ambe/mc/cluster_matching.py`) and `ambe mc timewindows --method {clusterfinder,optics}`
(`src/ambe/mc/hit_time_windows.py`).

---

## 0. Two headline results

1. **Splitting one neutron across clusters is a non-issue for ClusterFinder (0.40 %).**
   The AmBe MC finding of 17.9 % SPLIT does **not** carry over to CCinc. The real effect is
   the opposite one: *merging* two or three neutrons into a single cluster.
2. **61 % of the MVA's tank background class is neutron-majority light.** It is background
   only because no *single* neutron owned half the cluster. The classifier is substantially
   being asked to separate neutrons from neutrons, which is the most plausible explanation on
   the table for why AUC sits at 0.67 and will not move.

A third result is negative and closes a line of enquiry: **no hit-time window separates signal
from background inside a cluster**, in any of the four sample × method configurations.

---

## 1. Why this study exists

Two things were being taken on faith in the sample the MVA was trained on.

**The cluster ↔ neutron correspondence had never been checked on CCinc.** The AmBe MC QE1.50
provenance study (`REPORT_mc_ambe_qe150.md` §2) showed that on a one-neutron-per-event sample,
17.90 % of multi-cluster ClusterFinder events are one neutron split across clusters, rising to
99.71 % among box-accepted clusters. CCinc is messier — several true neutrons per event plus
prompt neutrino-interaction light — and nothing equivalent had been measured.

**The signal/background tag is a pure composition majority vote that looks at no timing.** The
production label (`cluster_features.py:_truth_label_cluster`, mirrored in
`optics.py:_optics_cluster_outcomes`) is:

> `is_truth_neutron = 1` iff the cluster's dominant neutron trackID is in the event's truth
> neutron set **and** `dom_frac = n_dom / n_all_cluster_hits >= 0.5`.
> Signal = `is_truth_neutron == 1`; background = `is_truth_neutron == 0 AND is_prompt_cluster == 0`.

Whether the *time structure* of the hits gives a cleaner separation had never been tested on
CCinc. The ~50 ns figure that motivated the question is a hypothesis inherited from
`ambe/diagnostics/analyze_truth_timing.py` (AmBe MC), not a result.

---

## 2. Method, and one deviation from the original plan

**Stage 0** reads `dominant_trackID` straight out of the persisted
`__cluster_features.parquet`: a neutron claimed by >1 cluster of the same method is a SPLIT.
Costs seconds, no reprocessing.

**Stage 1** recomputes per-hit cluster membership and records the full per-trackID decomposition
that the production label discards.

**Stage 2** bins per-hit `dt` against the cluster's own t0 by hit origin, and scans the window
edge.

### Deviation: parquet, not ROOT

The approved plan called for ROOT exact membership (`Cluster_HitChankey` + `Cluster_HitT`), as
the AmBe provenance study uses. **Stage 1 here reads parquet instead**, because the persisted
labels under test were produced by `assign_clusterfinder_labels` / `run_optics_on_event` running
on `__pulses.parquet` inside `cluster_features.extract_all_features`. Re-deriving membership
from ROOT would answer a different question ("what does CF really contain") and would not
reproduce the labels being diagnosed ("why is this cluster tagged background"). It is also far
cheaper. The ROOT path remains correct for the AmBe study and is untouched.

### Verification

| gate | result |
|---|---|
| Label reproduction — recomputed `is_truth_neutron` vs persisted | **100.0000 % (tank and world)** |
| Cluster count vs persisted `cluster_features.parquet` | exact (58,380 CF tank; 57,326 OPTICS tank) |
| Stage 0 ↔ Stage 1 split-rate closure | consistent |
| AmBe regression, full 1250 files after refactor | byte-identical headline; all 7 gates PASS; 17.9 % SPLIT / 77.0 % DISTINCT / 19,028 events |

---

## 3. Question 1a — are a cluster's hits from one neutron or several?

**Several, usually.** Fraction of clusters containing exactly one neutron trackID:

| sample | method | signal | background | mean trackIDs (sig / bkg) |
|---|---|---:|---:|---|
| tank | CF | **32.9 %** | 14.4 % | 2.29 / 3.26 |
| tank | OPTICS | 30.4 % | 12.2 % | 2.37 / 3.47 |
| world | CF | **50.9 %** | 20.3 % | 1.86 / 2.53 |
| world | OPTICS | 43.7 % | 16.7 % | 2.04 / 2.87 |

Hit-level composition, tank ClusterFinder:

| | signal | background |
|---|---:|---:|
| dominant neutron | 68.0 % | 29.7 % |
| **other** neutrons | 12.5 % | 24.9 % |
| non-neutron physics | 10.3 % | 34.1 % |
| dark noise + untraced | 9.2 % | 11.3 % |

Two-thirds of tank signal clusters are blends, and **12.5 % of the light in a signal cluster
comes from a neutron other than the one it is labelled with**. World is cleaner — half its
signal clusters are single-neutron — because it has fewer in-tank neutrons per event.

## 4. Question 1b — is one neutron split across several clusters?

**Essentially never for ClusterFinder; 8× worse for OPTICS.**

| sample | method | neutrons split | signal clusters on a split | SPLIT (multi-cluster events) | DISTINCT |
|---|---|---:|---:|---:|---:|
| tank | CF | **0.40 %** | 0.80 % | 0.47 % | 58.5 % |
| tank | OPTICS | **3.21 %** | 6.25 % | 3.73 % | 44.6 % |
| world | CF | 0.39 % | 0.79 % | 0.67 % | 61.7 % |
| world | OPTICS | 2.26 % | 4.44 % | 3.99 % | 46.9 % |

Against AmBe MC's 17.90 % SPLIT, ClusterFinder splitting on CCinc is ~40× rarer. The AmBe result
does not transfer, and the reason is physical: AmBe splitting was one capture's light straddling
the clustering window, whereas CCinc's multiple neutrons are separated by microseconds and are
correctly given distinct clusters.

> **This is a second, independent argument for ClusterFinder over OPTICS.**
> `REPORT_ccinc_v3_classifier_selection.md` chose CF purely on transfer (KS-to-MC-signal
> 0.11–0.16 for CF vs 0.42–0.64 for OPTICS). CF is also simply more faithful to the truth:
> 8× less splitting, and the only method with no merged-neutron tail beyond 50 ns (§6).

## 5. Question 1c — the same for background neutrons

This is the largest finding. **The MVA's background class is mostly neutron light.**

| sample | method | bkg with ≥1 neutron hit | `frac_neutron ≥ 0.5` | `dominant_class ∈ {1,2,3,4}` |
|---|---|---:|---:|---:|
| tank | CF | **99.9 %** | 74.9 % | 72.2 % |
| tank | OPTICS | 100.0 % | 75.7 % | 74.2 % |
| world | CF | 99.7 % | 66.8 % | 62.7 % |
| world | OPTICS | 99.9 % | 69.5 % | 66.4 % |

Why each background cluster failed the signal label — mutually exclusive and exhaustive, no
cluster fails for two reasons:

| sample | method | **merged neutrons** | diluted by non-neutron | no neutron hits | truth window |
|---|---|---:|---:|---:|---:|
| tank | CF | **60.7 %** | 25.9 % | 8.5 % | 4.9 % |
| tank | OPTICS | **64.4 %** | 25.0 % | 6.6 % | 4.1 % |
| world | CF | 44.1 % | 31.4 % | 19.4 % | 5.1 % |
| world | OPTICS | 50.9 % | 29.4 % | 15.6 % | 4.1 % |

### Root cause is a code behaviour, not physics

`_truth_label_cluster` (`cluster_features.py:1014-1016`) requires one single neutron trackID to
own ≥ 50 % of **all** cluster hits, then does `return 0, -1` on every failure path — it computes
the dominant trackID and throws it away. A cluster that is 60 % neutron light shared between
three neutrons is therefore recorded as background carrying **no neutron identity at all**.

On tank CF that describes 60.7 % of the background class: median neutron share 0.60, median
`dom_frac` 0.333 — three neutrons at a third each. Fewer than 9 % of tank background clusters
are genuinely neutron-free.

This is much worse than the 14.0 % recorded in `REPORT_ccinc_v3_world_merged.md` §4.5, which
measured a different population (merged tank+world) with a different criterion
(`frac_neutron > frac_nonneutron`). Both numbers are correct for what they measure; §4.5 should
not be read as bounding the effect.

**Consequence.** The MVA is substantially being trained to separate neutron light from neutron
light. That is the most plausible explanation currently available for AUC plateauing at 0.672
(truth-tag/CF GBT) despite 31 features. It is a **labelling** problem, not a features or model
problem, and is the most actionable result in this report.

---

## 6. Question 2 — hit-time windows

**Cleanly negative, in all four configurations.** Efficiency and purity vs window half-width,
t0 = cluster median:

| sample | method | eff dom @5 ns | eff **other** n @5 ns | eff dom @25 ns | eff other @25 ns | purity 5 ns → 1 µs |
|---|---|---:|---:|---:|---:|---:|
| tank | CF | 0.782 | 0.788 | 1.000 | 1.000 | 0.867 → 0.867 |
| tank | OPTICS | 0.720 | 0.672 | 0.986 | 0.927 | 0.869 → 0.850 |
| world | CF | 0.767 | 0.775 | 1.000 | 1.000 | 0.932 → 0.932 |
| world | OPTICS | 0.677 | 0.612 | 0.985 | 0.896 | 0.935 → 0.909 |

The dominant neutron and the other neutrons in the same cluster have **the same time profile** —
for ClusterFinder the two efficiency columns agree to three decimals at every edge. Purity is
flat, or slightly *falls* as the window widens. There is no edge at which a time cut helps.

The reason is the capture-time separation of the neutrons that share a cluster:

| sample | method | median | p90 | within 50 ns |
|---|---|---:|---:|---:|
| tank | CF | 2.9 ns | 8.3 ns | **100.0 %** |
| tank | OPTICS | 3.7 ns | 26.6 ns | 90.7 % |
| world | CF | 3.1 ns | 8.4 ns | 100.0 % |
| world | OPTICS | 4.3 ns | 297 ns | 87.0 % |

Merged neutrons are coincident to a few nanoseconds. **No window — 50 ns or any other — can
unmerge them.** ClusterFinder's own 50 ns clustering window has already collected everything,
which is why efficiency saturates at |dt| ≤ 25 ns.

OPTICS's tail beyond 50 ns (p90 26.6 ns tank, 297 ns world) is consistent with its 8× higher
split rate: it *is* cutting in time where CF does not, and that is a defect, not a feature.

### Where time separation does exist

At the microsecond scale, between genuinely distinct captures (median pair separation ~5 µs).
ClusterFinder already exploits it — that is the 58.5 % DISTINCT column in §4. The intuition that
timing matters was right; it operates at a scale the clusterer already handles, not at 50 ns.

---

## 7. What to do next

1. **Revisit the label, not the model.** The obvious candidate is to count *all* neutron light
   rather than requiring one trackID to win a plurality — e.g. signal iff
   `frac_neutron >= 0.5` (with the existing `origin_in_tank` override for world). That would move
   ~61 % of the tank background class into signal. Whether that *improves* the classifier is an
   open question — it changes what is being asked — and must be measured, not assumed.
2. **Do not pursue a time-window tag.** Excluded by §6 in all four configurations.
3. **Retain ClusterFinder.** Now justified on correctness as well as transfer.
4. Any re-labelling must be re-validated against the AmBe closure test before adoption, since
   the frozen model (`…truthtag…merged__cf.pkl`, GBT eff80, threshold 0.423621) is what the AmBe
   pipeline scores with.

---

## 8. Caveats

- **The microphysics of the 2.9 ns coincidence is unexplained.** Two class-1 (primary) neutrons
  should not capture 2.9 ns apart. This may be one capture's light attributed across several
  trackIDs rather than two captures. The operational conclusions hold either way — the light in
  one cluster does carry several trackIDs, and no window separates it — but the interpretation
  is unverified and is the first thing to check before quoting a physical neutron multiplicity.
- **`fail_truth_window` is 4–5 %, not 0 %.** `_apply_truth_window` keeps each trackID's own
  median hit by construction, so the `dom_tid in truth_tids` condition was expected never to
  fail. It does, because a hit can carry `ancestor_pdg == 2112` while `is_neutron == 0`. The
  column is emitted rather than assumed for exactly this reason.
- **Truth-tag streamline only.** Reco-tag is unmeasured; the CF-vs-OPTICS split gap may differ.
- **Tank and world are reported separately, never summed.** The merged training set applies the
  `origin_in_tank == 0 → background` override, which is not applied here.
- The persisted v3 `cluster_features.parquet` predates the `origin_in_tank` column; it is joined
  from the events table rather than read from the cluster table.

---

## 9. Outputs

Per sample, under `ambe_output/cc_neutrino_v3{,world}_truthtag/`:

| file | contents |
|---|---|
| `csv/…__stage0_split_census.csv` | split counts per method, background neutron-richness |
| `csv/…__stage0_event_categories.csv` | SPLIT / DISTINCT / SPLIT_AND_DISTINCT per method |
| `csv/…__stage1_summary.csv` | neutron multiplicity + failure-reason breakdown |
| `parquet/…__stage1_neutron_provenance.parquet` | per-cluster: `dom_tid`, `n_dom`, `dom_frac`, `tid2`, `n_contrib_neutrons`, `fail_reason` |
| `csv/…__htw_<method>_window_scan.csv` | efficiency/purity vs window edge |
| `csv/…__htw_<method>_neutron_pair_dt.csv` | merged-neutron capture-time separation (97,393 tank CF pairs) |
| `csv/…__htw_<method>_dt_histogram.csv` | binned dt by hit origin |
| `parquet/…__htw_<method>_clusters.parquet` | per-cluster timing + category counts |
| `plots/htw_*` | dt-by-category, window scan, pair separation |

Reproduce with `./run_ccinc_provenance_tmux.sh` (world) or the two CLI commands in the header.

### Performance note

Runtime is dominated by per-event OPTICS clustering and scales with **event count**, not data
volume: world Stage 1 took 2 h 36 m on 12.6 M hits while tank took ~2.5 h on 226 M hits, because
world has 1.57× more events. ClusterFinder is cheap by comparison (world CF time-windows: 2 min;
world OPTICS: 52 min). Stage 1 and Stage 2 each make an independent full pass and each re-cluster
every event — merging them into one pass would roughly halve the total.
