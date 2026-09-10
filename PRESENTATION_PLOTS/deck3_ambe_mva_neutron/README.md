# Deck 3 — The same AmBe analysis with the MVA neutron definition (data, standalone)

## Selection used in this deck

MVA neutron. The frozen classifier from Deck 1 applied to AmBe data: a cluster is a neutron iff GBT score > 0.423621, the MC 80 % signal-efficiency point. There is NO PE, charge-balance, time or hits box -- the MVA REPLACES the box cuts rather than being applied on top of them. Shared upstream with Deck 2 is only the IC gate 700-1200 and the cosmic veto. Where a figure shows a split, 'MVA-rejected' means score <= 0.423621. CAVEAT: the scoring stage covered 225,810 of the 237,590 clusters Stage 1 finds (95.04 %), so efficiency here is quoted on COVERED triggers -- compare 46.15 % with Deck 2's 55.99 % on the same triggers, never with its full-sample 57.25 %. CAPTURE-TIME FIT WINDOW: 2-67 us (was 10-67 before 2026-08-27) -- see deck 2.


## Clustering statistics

Denominator: **370,389 AmBe triggers** (covered triggers) — events passing the IC gate 700–1200 and surviving the cosmic veto.

| | MVA neutron |
|---|---:|
| events with **no** neutron cluster | 199,442 (53.85 %) |
| events with **exactly one** neutron | 160,829 (43.42 %) |
| events with **more than one** neutron | 10,118 (2.73 %) |
| **events with ≥1 neutron** (signal) | 170,947 (46.15 %) |

Of the signal events alone:

- **MVA neutron** — 94.08 % single, 5.92 % multiple

*Do not confuse the "more than one" row with the 19.74 % that appears in G7 and in the published tables.* **Both count events** — what differs is which. G7 counts events that had more than one cluster **in total before any selection** (`numberOfClusters != 1`, straight off the ntuple) and yielded at least one accepted neutron; because that condition does not depend on the selection, the number barely moves between them (19.74 % box, 17.23 % MVA). The rows above count events by how many neutrons the selection **accepted**, which is the quantity that responds to changing the neutron definition.

## Running order

| # | file | slot | original figure | what it shows |
|---|---|---|---|---|
| 1 | `01_efficiency_heatmap.pdf` | K1 | `efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_mva` | **the deck-3 headline**, to be put beside J1. On covered triggers: **45.88 %** against the box's 55.99 % |
| 2 | `02_statistics_heatmap.pdf` | K2 | `statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J2's twin |
| 3 | `03_residual_efficiency_heatmap.pdf` | K3 | `residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J3's twin, same AmBe 1.0 reference |
| 4 | `04_capture_time_heatmap_mva.pdf` | K9 | `V4CAPHEAT__capture_time_heatmap_mva` | J9's twin: τ per port × y under the MVA neutron, 26 positions, weighted **29.33 ± 0.23 µs**, spread 25.93–35.74 — reproduces L3's MVA number |
| 5 | `05_thermal_time_heatmap_mva.pdf` | K10 | `V4CAPHEAT__thermal_time_heatmap_mva` | J10's twin: thermalisation time per port × y, weighted **6.87 ± 0.13 µs**, spread 4.61–9.93 |
| 6 | `06_neutron_capture_time_fit.pdf` | K4 | `neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J4's twin. Pooled — backup, not the result |
| 7 | `07_capture_time_fit_residuals.pdf` | K5 | `capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J5's twin |
| 8 | `08_neutron_multiplicity.pdf` | K6 | `neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J6's twin |
| 9 | `09_cluster_pe_vs_cb.pdf` | K7 | `cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J7's twin — **the honest one**: the MVA sample extends past CB = 0.45, which the box forbids |
| 10 | `10_delta_t_first_subsequent.pdf` | K8 | `delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28_mva` | J8's twin |
| 11 | `11_capture_time_selected.pdf` | E1 | `V3AMBE_AMBE6266__capture_time_selected` | **the closure**: 33.41 ± 3.11 all clusters → 30.40 ± 3.37 at GBT eff80, against the 29.42 ± 0.22 anchor (pull +0.29) |
| 12 | `12_tau_summary.pdf` | E2 | `V3AMBE_AMBE6266__tau_summary` | every model × working point against the anchor band; and that all 12 *gated* fits fail the quality gate on statistics |
| 13 | `13_score_data_mc_overlay.pdf` | E3 | `V3AMBE_AMBE6266__score_data_mc_overlay` | the honest one: 6266's scores sit closer to MC **background**, not signal — a source-position effect (§6.4) |

## Backup — `backup/`, not in the running order

| file | slot | original figure | what it shows |
|---|---|---|---|
| `backup/classified_fraction.pdf` | F1 | `V3AMBEN__classified_fraction` | **the answer to "what %"**: of 225,945 Stage-2 AmBe neutrons over all 28 source runs, the MVA calls 79.0% neutron and 21.0% other at the MC 80% point (42.7% at eff50, 91.6% at eff90); all four models agree |
| `backup/classified_by_position.pdf` | F2 | `V3AMBEN__classified_by_position` | the kept fraction runs **64.2–86.0% across 26 positions** for one nominal 80% threshold — driven by light yield (`n_hits` r = +0.91), not geometry (`d_wall` r = +0.01) |
| `backup/classified_capture_time.pdf` | F3 | `V3AMBEN__classified_capture_time` | kept vs rejected τ per position against the anchor band. **State the caveat with it:** the rejected fits fail on σ (10–33 µs at χ²/ndof ≈ 1.3), so the background claim rests on the model-free shape test, not on these |
| `backup/capture_time_selected__v3amben.pdf` | F6 | `V3AMBEN__capture_time_selected` | **the v4-native MVA validation**: the frozen definition applied to all 28 AmBe runs, capture time at each working point. Replaces June as the multi-position reference |
| `backup/tau_summary__v3amben.pdf` | F7 | `V3AMBEN__tau_summary` | every model × working point on the 28-run sample against the anchor band. **State the caveat with it:** the *pooled* fits fail on χ²/ndof (8–12) — see the note below |
| `backup/score_data_mc_overlay__v3ambe.pdf` | Z14 | `V3AMBE__score_data_mc_overlay` | the June score overlay — the "model transferred" baseline that 6266 reverses |
| `backup/tau_summary__v3ambe.pdf` | Z15 | `V3AMBE__tau_summary` | June τ per model × working point; **31.30 ± 2.28 µs** at truth-tag/CF GBT eff80 |
| `backup/capture_time_selected__v3ambe.pdf` | Z16 | `V3AMBE__capture_time_selected` | June capture time, the fit itself |
| `backup/classified_fraction__v3ambepipe.pdf` | Z18 | `V3AMBEPIPE__classified_fraction` | the superseded 20-run version of F1 — keep only to answer "did adding runs change it?" |

## Appendix — `appendix/`, multi-page PDFs

| file | pages |
|---|---:|
| `appendix/APPENDIX_perposition_mvaneutron.pdf` | 312 |
| `appendix/APPENDIX_IC_waveforms_all28runs.pdf` | 28 |

---

Every file here is also in the flat `PRESENTATION_PLOTS/` folder under its slot-prefixed original name, and `MANIFEST.csv` maps between the two. Regenerate with `python -u collect_presentation_plots.py --do build`.
