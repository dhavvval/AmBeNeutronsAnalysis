# Deck 2 — AmBe neutron analysis with the traditional box cuts (data, standalone)

## Selection used in this deck

Box cuts. IC waveform gate 700 < IC_adjusted < 1200 plus the second-pulse veto, then the cosmic veto (any cluster with clusterTime < 2 us or clusterPE > 100 drops the whole event), then a neutron is  0 < clusterPE <= 100,  0 < charge balance < 0.45,  clusterTime >= 2 us,  clusterHits >= 5. No MVA and no MC anywhere in this deck. CAPTURE-TIME FIT WINDOW: 2-67 us, changed 2026-08-27 from 10-67 so that one window is used everywhere and the fit sees the thermalisation rise the model exists to describe. The anchor is now 29.417 +- 0.221 us (19 positions) / 29.477 +- 0.193 us (26); the old 30.53 / 30.535 numbers were 10-67 results and are NOT comparable.


## Clustering statistics

Denominator: **381,263 AmBe triggers** (full sample) — events passing the IC gate 700–1200 and surviving the cosmic veto.

| | box cuts |
|---|---:|
| events with **no** neutron cluster | 162,996 (42.75 %) |
| events with **exactly one** neutron | 175,192 (45.95 %) |
| events with **more than one** neutron | 43,075 (11.30 %) |
| **events with ≥1 neutron** (signal) | 218,267 (57.25 %) |

Of the signal events alone:

- **box cuts** — 80.26 % single, 19.74 % multiple

*Do not confuse the "more than one" row with the 19.74 % that appears in G7 and in the published tables.* **Both count events** — what differs is which. G7 counts events that had more than one cluster **in total before any selection** (`numberOfClusters != 1`, straight off the ntuple) and yielded at least one accepted neutron; because that condition does not depend on the selection, the number barely moves between them (19.74 % box, 17.23 % MVA). The rows above count events by how many neutrons the selection **accepted**, which is the quantity that responds to changing the neutron definition.

## Running order

| # | file | slot | original figure | what it shows |
|---|---|---|---|---|
| 1 | `01_cutflow_table.pdf` | G1 | `V4BOX__cutflow_table` | the box cuts stage by stage: 474,192 IC-gated triggers → 19.85% cosmic-vetoed → 380,072 AmBe triggers → **217,532 neutron triggers** (45.9% of Stage 1) |
| 2 | `02_efficiency_heatmap.pdf` | J1 | `efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_box` | **the deck-2 headline**: port × y efficiency with per-cell binomial SE, 26 positions, one tag |
| 3 | `03_statistics_heatmap.pdf` | J2 | `statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_box` | exposure per cell, so a low-statistics cell is not read as a physics effect |
| 4 | `04_residual_efficiency_heatmap.pdf` | J3 | `residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28_box` | against the AmBe 1.0 reference map — the 40–72 % pattern is not new |
| 5 | `05_efficiency_by_position.pdf` | G2 | `V4BOX__efficiency_by_position` | **the headline**: campaign efficiency **57.23 ± 0.08 %** over 28 runs / 26 positions, range **40.3–71.8 %**; port 3 lowest, port 5 y=0 highest |
| 6 | `06_capture_time_by_position.pdf` | G4 | `V4BOX__capture_time_by_position` | τ per position, 26 positions, **all 26 fits converge**; weighted **29.477 ± 0.193 µs** |
| 7 | `07_capture_time_anchor_comparison.pdf` | G5 | `V4BOX__capture_time_anchor_comparison` | extending 19 → 26 positions moves τ by **+0.005 µs** and shrinks the error 12%. The anchor is stable |
| 8 | `08_capture_time_heatmap_box.pdf` | J9 | `V4CAPHEAT__capture_time_heatmap_box` | **the map that was missing**: τ per port × y on the same grid as J1, 26 positions, all converge, weighted **29.48 ± 0.19 µs**, spread 27.62–33.42 — reproduces G4's per-position anchor with lmfit |
| 9 | `09_thermal_time_heatmap_box.pdf` | J10 | `V4CAPHEAT__thermal_time_heatmap_box` | the *other* constant the same fit measures: thermalisation time per port × y, weighted **6.52 ± 0.11 µs**, spread 5.20–8.31, all 26 converged. On the 2–67 µs window therm is genuinely constrained — the fit now sees the rise it describes |
| 10 | `10_neutron_capture_time_fit.pdf` | J4 | `neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28_box` | the pooled capture-time fit. **Quote G4's per-position weighted τ, never this** — pooled χ²/ndof is 3.22 |
| 11 | `11_capture_time_fit_residuals.pdf` | J5 | `capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28_box` | residuals of J4, which is where the pooled χ² comes from |
| 12 | `12_neutron_multiplicity.pdf` | J6 | `neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28_box` | neutron multiplicity per trigger |
| 13 | `13_cluster_pe_vs_cb.pdf` | J7 | `cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28_box` | PE vs charge balance — the 2D the box is drawn in |
| 14 | `14_delta_t_first_subsequent.pdf` | J8 | `delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28_box` | Δt between first and subsequent clusters in multi-neutron events |
| 15 | `15_consistency_table.pdf` | G3 | `V4BOX__consistency_table` | **the QA statement**: the 8 new runs move the campaign efficiency by **−0.07 pp** (57.30 → 57.23). Adding 30% more data changed nothing |
| 16 | `16_cosmic_fraction.pdf` | G6 | `V4BOX__cosmic_fraction` | cosmic veto **19.85 %** campaign-wide, flat to 7 pp across 26 positions — a detector-stability check |
| 17 | `17_multiplicity_share.pdf` | G7 | `V4BOX__multiplicity_share` | multi-neutron share **19.74 %**, range 18.0–23.7 % |

## Backup — `backup/`, not in the running order

| file | slot | original figure | what it shows |
|---|---|---|---|
| `backup/efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | G10 | `efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28` | **the canonical heatmap**: port × y efficiency with per-cell binomial SE, all 26 positions |
| `backup/residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | G11 | `residual_efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28` | the same against the AmBe 1.0 reference map — shows the 40–72 % pattern is not new |
| `backup/statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | G12 | `statistics_heatmap__AmBe_2.0v4__AmBe2.0v4_all28` | exposure per cell, so nobody reads a low-statistics cell as a physics effect |
| `backup/cut_variables.pdf` | Z17 | `V4BOX__cut_variables` | what the box actually is: the four cut variables with the cut positions |
| `backup/neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | Z19 | `neutron_capture_time_fit__AmBe_2.0v4__AmBe2.0v4_all28` | the **pooled** 28-run box-cut capture-time fit, χ²/ndof 3.22. Backup only — quote the per-position weighted τ (G4), never this |
| `backup/capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | Z20 | `capture_time_fit_residuals__AmBe_2.0v4__AmBe2.0v4_all28` | residuals of Z19, which is where the pooled fit's χ² comes from |
| `backup/neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | Z21 | `neutron_multiplicity__AmBe_2.0v4__AmBe2.0v4_all28` | neutron multiplicity per trigger, campaign-pooled |
| `backup/cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | Z22 | `cluster_pe_vs_cb__AmBe_2.0v4__AmBe2.0v4_all28` | PE vs charge balance, the 2D the box cut is drawn in |
| `backup/delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28.pdf` | Z23 | `delta_t_first_subsequent__AmBe_2.0v4__AmBe2.0v4_all28` | Δt between first and subsequent clusters in multi-neutron events |

## Appendix — `appendix/`, multi-page PDFs

| file | pages |
|---|---:|
| `appendix/APPENDIX_perposition_boxcuts.pdf` | 312 |
| `appendix/APPENDIX_boxcut_v4_by_position.pdf` | 26 |
| `appendix/APPENDIX_IC_waveforms_all28runs.pdf` | 28 |

---

Every file here is also in the flat `PRESENTATION_PLOTS/` folder under its slot-prefixed original name, and `MANIFEST.csv` maps between the two. Regenerate with `python -u collect_presentation_plots.py --do build`.
