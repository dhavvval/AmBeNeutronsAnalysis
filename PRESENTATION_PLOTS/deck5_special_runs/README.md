# Deck 5 — The special runs, described stage by stage (data, standalone)

## Selection used in this deck

The SAME flow decks 2 and 3 use -- Stage-1 IC waveform gate 700 < IC_adjusted < 1200 plus the second-pulse veto, then the tank cluster stage, then the Stage-2 box PE <= 100, CB < 0.45, t >= 2 us, hits >= 5 -- but reporting the DISTRIBUTIONS at each stage rather than the survivor count, and reporting them UNGATED as well as gated. THREE DIFFERENT KINDS OF RUN, and the labels must not blur them: 6254/6256 are DUMMY (pulser) TRIGGERS with the LAPPD on and off -- the readout fires on a schedule, so each event is a random slice of the tank and they measure the ACCIDENTAL cluster rate (0.159 box-accepted clusters per trigger, the two agreeing to 0.3 %); 6264 is a genuine NO-SOURCE run with the real trigger logic live, so its 808 in-gate waveforms are FALSE STARTS of the IC trigger, which a dummy-trigger run can never measure because it never exercises the trigger logic. Ungated is not a relaxation: a scheduled readout carries no BGO gamma, so 6254 and 6256 have ZERO waveforms in the IC window (0 of 7,933 and 0 of 25,525) and every gated plot of those two is blank; what they contain is still a measurement. THE STAGE FIGURES USE TWO RUN SETS AND NEVER MIX THEM IN ONE FRAME. M13-M16 + M25 are 6264 (no source) against 6265 (source in), both at Port 5, (0,0,0) -- the same position to the centimetre, so a difference between them is the source and not the geometry -- through all three stages: before the AmBe waveform cut, after it, after the box (176,634 -> 11,210 -> 787 against 75,259 -> 31,081 -> 15,165). M21-M24 + M26 are the two DUMMY triggers alone, ungated, BOX ONLY: they have zero IC-passing waveforms, so the waveform-cut stage does not exist for them and cannot be a shared row. The earlier version showed all four runs in one frame and used 6266 (Port 5, y = 100) as 6264's contrast, which put a ~10 pp geometry difference inside the comparison. M1-M26 are these runs stage by stage; H1 is the campaign's own box-cut scoring of them; backup carries the figures that restate those numbers in bar form (H2/H3), the narrower-sample shape plot (H4), the IC 2D (M20) and the MVA neutron applied to these runs (F4/F5). NO efficiency and NO capture time is quoted for any run in this deck, by design -- these are not source runs and a campaign number computed on them would be wrong however it came out. ONE DENOMINATOR MISMATCH TO EXPECT, IT IS NOT A CONTRADICTION: M1 says 6256 has 25,525 waveforms and H1 says 4,254. The M slots read the PIPELINE's own WaveformFeatures parquet (every part file `ambe data process` saw); H1 reads the special-runs diagnostics join, which covers a narrower part range. Both report ZERO IC-passing triggers, which is the claim. Quote M1's denominator for anything about what the run contains.

## Running order

| # | file | slot | original figure | what it shows |
|---|---|---|---|---|
| 1 | `01_ic_cutflow_table.pdf` | M1 | `V4SPECDIAG__ic_cutflow_table` | **start here**: Stage-1 waveform cut flow per run. dummy trigger (LAPPD on) **0 of 7,933**, (LAPPD off) **0 of 25,525**, no source **808 in window → 589 accepted**, regular run 53,933 → 39,391 |
| 2 | `02_ic_adjusted_window.pdf` | M2 | `V4SPECDIAG__ic_adjusted_window` | the same as a spectrum, counts on log y. Both dummy triggers never leave a narrow peak at 0 — **no BGO gamma at all**, as a scheduled readout should not have; the regular run shows the AmBe BGO peak at 800–1000; the no-source run is a flat cosmic tail to 4000 that leaks **808 false starts** into the gate |
| 3 | `03_ic_features_1d.pdf` | M3 | `V4SPECDIAG__ic_features_1d` | the six IC waveform features ungated, area-normalised — where the false-trigger population differs in shape from a real one |
| 4 | `04_ic_features_1d_counts.pdf` | M4 | `V4SPECDIAG__ic_features_1d_counts` | M3 in raw counts on a log y, so the exposure of each run is visible rather than divided out |
| 5 | `05_ic_page_6254.pdf` | M5 | `V4SPECDIAG__ic_page_6254` | the basic IC page for the **dummy trigger (LAPPD on)** — one run, its own axes, its own counts. The special-run twin of the campaign's per-run IC book |
| 6 | `06_ic_page_6256.pdf` | M6 | `V4SPECDIAG__ic_page_6256` | the same for the **dummy trigger (LAPPD off)** |
| 7 | `07_ic_page_6264.pdf` | M7 | `V4SPECDIAG__ic_page_6264` | the same for the **no-source run** — this is the false-start IC distribution, and it is visibly a second population: prompt fraction spread to 1.0, width-over-threshold piled up near 850 bins, baseline σ to 50 ADC |
| 8 | `08_ic_page_6265.pdf` | M8 | `V4SPECDIAG__ic_page_6265` | the same for the **source-in run at the same position as 6264**, for comparison. Was 6266 (Port 5, y = 100); 6265 sits at (0,0,0) like 6264, so the whole deck now compares like with like |
| 9 | `09_tank_features_1d.pdf` | M9 | `V4SPECDIAG__tank_features_1d` | the tank side ungated, area-normalised: the four cut variables plus multiplicity and per-trigger charge, on ranges that run **past** every cut bound so what the box discards is visible. **9.55 clusters/trigger for the no-source run** against 1.95 for the regular run and 0.55 for the two dummy triggers |
| 10 | `10_tank_features_1d_counts.pdf` | M10 | `V4SPECDIAG__tank_features_1d_counts` | M9 in raw counts |
| 11 | `11_tank_features_1d_afterbox.pdf` | M11 | `V4SPECDIAG__tank_features_1d_afterbox` | the same six panels **after** the box, on the ranges the box admits, area-normalised. Multiplicity is recomputed as *accepted* clusters per trigger, which is the number that matters |
| 12 | `12_tank_features_1d_afterbox_counts.pdf` | M12 | `V4SPECDIAG__tank_features_1d_afterbox_counts` | M11 in raw counts |
| 13 | `13_pair_stages_clusterchargebalance_vs_clusterpe.pdf` | M13 | `V4SPECDIAG__pair_stages_clusterchargebalance_vs_clusterpe` | **the headline ask**: charge balance vs cluster PE for **6264 (no source) against 6265 (source in), both at Port 5, (0,0,0)** — rows are the streamline, ungated → AmBe waveform cut → box cut. Kept fractions **6.3 % → 7.0 % (787 clusters)** for no source against **41.3 % → 48.8 % (15,165)** for source in |
| 14 | `14_pair_stages_clustertime_vs_clusterpe.pdf` | M14 | `V4SPECDIAG__pair_stages_clustertime_vs_clusterpe` | the same three stages for cluster time vs PE. **This is the one to leave up**: after the box, 6265 shows the capture-time tail out to 67 µs and 6264 shows only a prompt clump near 8 µs |
| 15 | `15_pair_stages_clusterhits_vs_clusterpe.pdf` | M15 | `V4SPECDIAG__pair_stages_clusterhits_vs_clusterpe` | the same for cluster hits vs PE |
| 16 | `16_pair_stages_clustertime_vs_clusterchargebalance.pdf` | M16 | `V4SPECDIAG__pair_stages_clustertime_vs_clusterchargebalance` | the same for cluster time vs charge balance |
| 17 | `17_pair_stage_cutflow_table.pdf` | M25 | `V4SPECDIAG__pair_stage_cutflow_table` | the same streamline as numbers, so the panel counts are checkable: 176,634 → 11,210 → 787 for no source, 75,259 → 31,081 → 15,165 for source in |
| 18 | `18_dummy_stages_clusterchargebalance_vs_clusterpe.pdf` | M21 | `V4SPECDIAG__dummy_stages_clusterchargebalance_vs_clusterpe` | the **dummy triggers on their own**, 6254 (LAPPD on) vs 6256 (LAPPD off), before and after the box. **No AmBe waveform cut row exists**: both have 0 IC-passing waveforms, so that stage is empty by construction rather than by choice. Kept 28.6 % / 29.4 % |
| 19 | `19_dummy_stages_clustertime_vs_clusterpe.pdf` | M22 | `V4SPECDIAG__dummy_stages_clustertime_vs_clusterpe` | the same for cluster time vs PE |
| 20 | `20_dummy_stages_clusterhits_vs_clusterpe.pdf` | M23 | `V4SPECDIAG__dummy_stages_clusterhits_vs_clusterpe` | the same for cluster hits vs PE |
| 21 | `21_dummy_stages_clustertime_vs_clusterchargebalance.pdf` | M24 | `V4SPECDIAG__dummy_stages_clustertime_vs_clusterchargebalance` | the same for cluster time vs charge balance |
| 22 | `22_dummy_stage_cutflow_table.pdf` | M26 | `V4SPECDIAG__dummy_stage_cutflow_table` | the dummy streamline as numbers: 4,404 → 1,260 and 13,837 → 4,068 |
| 23 | `23_tank_cutflow_table.pdf` | M17 | `V4SPECDIAG__tank_cutflow_table` | Stage-2 cumulative cut flow on ungated clusters. The no-source run loses **94.2 %** of its 176,634 clusters, and it is charge balance and time that do it |
| 24 | `24_tank_nminus1_table.pdf` | M18 | `V4SPECDIAG__tank_nminus1_table` | the same leave-one-out, which is what identifies *which* cut is working. **`hits ≥ 5` is a strict no-op** at every run — ClusterFinder already requires 5 |
| 25 | `25_nosource_charge_6254_vs_6256.pdf` | M19 | `V4SPECDIAG__nosource_charge_6254_vs_6256` | **the nothing-to-detect charge answer, on the two DUMMY TRIGGERS, and it is two numbers not one**: hit-level charge differs by **+220.7 ± 13.9 PE (15.9σ)** and hit count by **+118.6 ± 1.2 (95.9σ)**, but CLUSTER-level charge **agrees** (−6.1 ± 17.7 PE, 0.35σ). The LAPPD leak raises the hit stream without producing clusters |
| 26 | `26_cutflow_table.pdf` | H1 | `V4BOXSPEC__cutflow_table` | all six runs on one page: 6254/6256 have **zero** IC-passing triggers (0 of 7,932 and 0 of 4,254); 6264/6265/6266/6270 gated at 149 / 2,150 / 2,171 / 2,312 |

## Backup — `backup/`, not in the running order

| file | slot | original figure | what it shows |
|---|---|---|---|
| `backup/ic_adjusted_vs_fprompt_2d.pdf` | M20 | `V4SPECDIAG__ic_adjusted_vs_fprompt_2d` | backup: IC_adjusted vs prompt fraction, one panel per run. Dropped from the running order — M2 and M3 already carry both of its projections. The figure to produce if anyone asks whether the gate could be moved |
| `backup/yield_per_trigger.pdf` | H2 | `V4BOXSPEC__yield_per_trigger` | **6264 gives zero Stage-2 candidates**; the other three sit at 0.64–0.74 candidates per AmBe trigger |
| `backup/cosmic_fraction.pdf` | H3 | `V4BOXSPEC__cosmic_fraction` | why: 6264 cosmic-vetoes at **98.0 %** against 20–25 % for the other three — ~4× higher |
| `backup/cluster_shapes.pdf` | H4 | `V4BOXSPEC__cluster_shapes` | the four box-cut variables per run, area-normalised. 6265/6266/6270 are indistinguishable from source-in runs |
| `backup/classified_fraction.pdf` | F4 | `V3AMBESPEC__classified_fraction` | the special runs, **separate and never merged**: 6265/6266/6270 at 79–86%; 6264 gives zero clusters (93.3% cosmic-vetoed); 6254/6256 have zero IC-passing triggers |
| `backup/classified_by_position.pdf` | F5 | `V3AMBESPEC__classified_by_position` | the same four runs placed against the campaign's per-position range — backup for F4 |

## Appendix — `appendix/`, multi-page PDFs

| file | pages |
|---|---:|
| `appendix/APPENDIX_specialruns_by_run.pdf` | 4 |
| `appendix/APPENDIX_IC_waveforms_specialruns.pdf` | 4 |

---

Every file here is also in the flat `PRESENTATION_PLOTS/` folder under its slot-prefixed original name, and `MANIFEST.csv` maps between the two. Regenerate with `python -u collect_presentation_plots.py --do build`.
