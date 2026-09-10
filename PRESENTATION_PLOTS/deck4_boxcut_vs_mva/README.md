# Deck 4 — Box cuts against the MVA neutron definition

## Selection used in this deck

Both. Every figure here puts the box cuts (PE <= 100, CB < 0.45, t >= 2 us, hits >= 5) and the MVA neutron (GBT eff80) on the same clusters or the same source positions. This is the only deck in which the two definitions appear together. CAPTURE-TIME FIT WINDOW: 2-67 us (was 10-67 before 2026-08-27). Every tau on these slides moved; the matched box-to-MVA shift is now -1.546 +- 0.311 us (4.98 sigma) against the old -0.989 +- 0.363 (2.73 sigma).


## Clustering statistics

Denominator: **370,389 AmBe triggers** (covered triggers) — events passing the IC gate 700–1200 and surviving the cosmic veto.  Identical for every column below, which is what makes them comparable.

| | box cuts | MVA neutron |
|---|---:|---:|
| events with **no** neutron cluster | 162,996 (44.01 %) | 199,442 (53.85 %) |
| events with **exactly one** neutron | 191,055 (51.58 %) | 160,829 (43.42 %) |
| events with **more than one** neutron | 16,338 (4.41 %) | 10,118 (2.73 %) |
| **events with ≥1 neutron** (signal) | 207,393 (55.99 %) | 170,947 (46.15 %) |

Of the signal events alone:

- **box cuts** — 92.12 % single, 7.88 % multiple
- **MVA neutron** — 94.08 % single, 5.92 % multiple

*Do not confuse the "more than one" row with the 19.74 % that appears in G7 and in the published tables.* **Both count events** — what differs is which. G7 counts events that had more than one cluster **in total before any selection** (`numberOfClusters != 1`, straight off the ntuple) and yielded at least one accepted neutron; because that condition does not depend on the selection, the number barely moves between them (19.74 % box, 17.23 % MVA). The rows above count events by how many neutrons the selection **accepted**, which is the quantity that responds to changing the neutron definition.

## Running order

| # | file | slot | original figure | what it shows |
|---|---|---|---|---|
| 1 | `01_residual_efficiency_by_position.pdf` | L1 | `V4RES__residual_efficiency_by_position` | **the deck-4 headline**: both efficiencies and their difference per position. On the same 370,389 covered triggers: box **55.99 %**, MVA **46.15 %**, **−9.84 pp** |
| 2 | `02_residual_efficiency_heatmap_boxvsmva.pdf` | L2 | `V4RES__residual_efficiency_heatmap_boxvsmva` | the same as a port × y map |
| 3 | `03_residual_capture_time_by_position.pdf` | L3 | `V4RES__residual_capture_time_by_position` | per-position τ both ways: box **30.444 ± 0.210**, MVA **29.329 ± 0.228**, Δ **−1.115 ± 0.310 (3.59σ)** — the MVA lands on the 29.42 anchor, the box sits 3.3σ above it. Per position, never pooled |
| 4 | `04_residual_cluster_overlap_table.pdf` | L4 | `V4RES__residual_cluster_overlap_table` | the disagreement, cluster by cluster: **178,404** both call a neutron, **47,406** box-only (the MVA rejects them), **3,922** MVA-only (outside the box, all on CB ≥ 0.45). Agreement, **not** purity |
| 5 | `05_residual_clusters_by_position.pdf` | L5 | `V4RES__residual_clusters_by_position` | clusters selected per position, both, plus the ratio |
| 6 | `06_residual_multiplicity_by_position.pdf` | L6 | `V4RES__residual_multiplicity_by_position` | share of neutron triggers with **more than one accepted neutron**, per position, both plus the residual: box **7.88 %**, MVA **5.92 %**. **NOT** G7's 19.74 % — see the note |
| 7 | `07_matched_boxcut_vs_mva_table.pdf` | G13 | `V4BOX__matched_boxcut_vs_mva_table` | **the strict comparison, and the best result in the AmBe chapter**: on the *same* 225,810 clusters the MVA moves τ **30.444 ± 0.210 → 28.898 ± 0.229** (−1.546 ± 0.311, **4.98σ**) and what it discards sits at **34.746 ± 0.533** |
| 8 | `08_validation_mva_vs_boxcut_table.pdf` | G8 | `V4BOX__validation_mva_vs_boxcut_table` | **the validation slide**: the frozen MVA definition and the box cuts on the *same* 26 AmBe positions under *one* fit recipe — agreement **0.11σ** (−0.188 ± 1.754 µs) |
| 9 | `09_validation_tau_scatter.pdf` | G9 | `V4BOX__validation_tau_scatter` | the same per position rather than pooled — MVA-selected τ against box-cut τ |
| 10 | `10_agreement_efficiency_by_position.pdf` | G15 | `V4BOX__agreement_efficiency_by_position` | **the agreement slide**: MVA-neutron vs box-cut efficiency, per position — **r = 0.993** over 26 positions. The two definitions rank the positions identically |
| 11 | `11_agreement_capture_time_table.pdf` | G14 | `V4BOX__agreement_capture_time_table` | the same comparison at **every** working point and **both** streamlines: the shift is monotonic (eff50 −1.76 µs / 4.0σ → eff90 −0.56 µs / 1.6σ) and the discarded sample gets later as the cut tightens |
| 12 | `12_agreement_kept_fraction.pdf` | G16 | `V4BOX__agreement_kept_fraction` | all four models × three working points × both streamlines on one page — the models agree to a few points everywhere |

---

Every file here is also in the flat `PRESENTATION_PLOTS/` folder under its slot-prefixed original name, and `MANIFEST.csv` maps between the two. Regenerate with `python -u collect_presentation_plots.py --do build`.
