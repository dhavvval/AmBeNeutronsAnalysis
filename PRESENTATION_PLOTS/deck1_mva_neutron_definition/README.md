# Deck 1 — Defining a neutron cluster with the MVA (MC, Stage 0 to Stage 3/4)

## Selection used in this deck

MVA neutron. Trained on CC-inclusive MC; the deliverable is the GBT on ClusterFinder clusters, truth-tag streamline, taken at the MC 80 % signal-efficiency working point.

## Running order

| # | file | slot | original figure | what it shows |
|---|---|---|---|---|
| 1 | `01_cutflow_streamlines_table.pdf` | A1 | `V3MERGED__cutflow_streamlines_table` | the two selections, cut by cut |
| 2 | `02_cutflow_nminus1_table.pdf` | A2 | `V3MERGED__cutflow_nminus1_table` | which cut is actually doing the work |
| 3 | `03_streamline_overlap_table.pdf` | A3 | `V3MERGED__streamline_overlap_table` | the streamlines never disagree about a *label*, only about which events they include |
| 4 | `04_merged_training_sizes.pdf` | A4 | `V3MERGED__merged_training_sizes` | what merging the world sample bought: background 8,392 → 36,008 per class |
| 5 | `05_auc_tank_vs_merged.pdf` | B1 | `V3MERGED__auc_tank_vs_merged` | the +0.082 merged gain. **Say §4.4 out loud here** — it is a change of question, not a better answer |
| 6 | `06_auc_cv_boxplot.pdf` | B2 | `V3MERGED__auc_cv_boxplot` | the gain is 12–25× the CV scatter; the GBT/XGB gap is not |
| 7 | `07_roc_by_config.pdf` | B3 | `V3STATS__roc_by_config` | all four models, all four configurations, one scale |
| 8 | `08_loss_comparison.pdf` | B4 | `V3STATS__loss_comparison` | log-loss breaks the GBT/XGB tie, and shows the NN calibrates badly |
| 9 | `09_model_verdict_table.pdf` | B5 | `V3STATS__model_verdict_table` | **the verdict slide**: GBT on ClusterFinder |
| 10 | `10_effpurity_scan.pdf` | B6 | `V3STATS__effpurity_scan` | purity 0.590 at 80% efficiency — and that this is a 1:1 test set, not a physical purity |
| 11 | `11_selection_2x2_benchmark.pdf` | B7 | `selection_2x2_benchmark__ambe__gbt__eff` | **the validation B6 cannot give**: efficiency of the box cuts and of the MVA neutron on the *natural* MC cluster population, OPTICS vs ClusterFinder. CF: box **0.930**, MVA @0.424 **0.806** |
| 12 | `12_selection_2x2_benchmark__ambe__gbt__purity.pdf` | B8 | `selection_2x2_benchmark__ambe__gbt__purity` | the same for purity — CF: box **0.869**, MVA **0.883**, with **5,348** fakes against the box's **7,044** (−24 %). This is the physical purity; B6's is not |
| 13 | `13_bkg_containment_normalized__truthtag_cf.pdf` | C0 | `V3MERGED__bkg_containment_normalized__truthtag_cf` | the containment view: every component as a share of ALL delayed background light, tank vs outside the tank, out-of-tank neutron capture kept in the normalisation (61.1% of the total) |
| 14 | `14_bkg_containment_table__truthtag_cf.pdf` | C4 | `V3MERGED__bkg_containment_table__truthtag_cf` | the same budget as numbers, for the backup slide and for anyone who wants to read the percentages off |
| 15 | `15_bkg_composition_tank_vs_world.pdf` | C1 | `V3MERGED__bkg_composition_tank_vs_world` | 59.5% of the background light is real neutron capture, mostly from the world |
| 16 | `16_training_background_by_particle.pdf` | C2 | `V3MERGED__training_background_by_particle` | of the non-neutron remainder: μ⁻ 53%, π⁺ 26%, π⁰ 7%, p 6%, π⁻ 5%, μ⁺ 3% |
| 17 | `17_truthtag_cf__top6_shape.pdf` | C3 | `V3BG__truthtag_cf__top6_shape` | where each species lives along the leading features — *composition of a bin* |
| 18 | `18_light_sources_census.pdf` | D8 | `V3SPUR__light_sources_census` | what the background light IS: **neutron capture (background) is 61.1% of the background hits and dominates 76.8% of clusters**, then μ⁻ 11.3%, dark noise 10.9%, pure EM 6.6%, π⁺ 5.5%. Says "hits", not "light": the quantity is a hit count and "61.1% of the light" gets requoted as a PE fraction |
| 19 | `19_light_sources_table.pdf` | D9 | `V3SPUR__light_sources_table` | the all-light census as numbers: per source, share of the light, clusters it dominates, and how pure those are |
| 20 | `20_n_sources_alllight.pdf` | D6 | `V3SPUR__n_sources_alllight` | a cluster is never one thing: mean 2.33 sources in background, 2.83 in signal |
| 21 | `21_purity_alllight.pdf` | D7 | `V3SPUR__purity_alllight` | the leading source holds a median 0.83 of a background cluster and 0.80 of a signal one — composition purity is NOT what separates them |
| 22 | `22_light_purity_table.pdf` | D0 | `V3SPUR__light_purity_table` | signal against background on every mixture metric at once — the table that says purity is not the discriminant |
| 23 | `23_light_split_table.pdf` | D10 | `V3SPUR__light_split_table` | the same composition on the train half and the test half separately — they agree to 0.3 pp, so the block may be quoted on either |
| 24 | `24_n_species.pdf` | D1 | `V3SPUR__n_species` | the charged-particle zoom: where a cluster has charged-parent light, it is usually a mixture |
| 25 | `25_purity.pdf` | D2 | `V3SPUR__purity` | the leading particle holds a median 0.75 of a background cluster's traced light |
| 26 | `26_cooccurrence.pdf` | D3 | `V3SPUR__cooccurrence` | what the mixtures are, over **all** the background light and therefore over all 39,133 background clusters rather than the 30% with a traced charged parent: **neutron capture + dark noise in 78.4% of the 25,848 mixed clusters**, then capture + μ⁻ 27.4%, μ⁻ + dark noise 27.1%, μ⁻ + π⁺ 19.8% |
| 27 | `27_species_census.pdf` | D4 | `V3SPUR__species_census` | **the table to leave up**: per particle — share of background, purity, and GBT AUC against signal (0.552 π⁰ → 0.705 out-of-tank capture) |
| 28 | `28_presentable_features_byspecies_logy.pdf` | D5 | `TT_CF_GBT__presentable_features_byspecies_logy` | the six leading features with one curve per **source** instead of one grey background. The 27,393-cluster `no non-EM origin` class is split by its dominant light source into **neutron capture 25,724 / γ,e± 1,534 / dark noise 135** — the same partition D8 uses — so the largest background class is no longer one grey curve. All curves solid, separated by colour; the signal band is drawn at α=0.30 so the curves are readable over it. **LOG y**: dark noise is 135 clusters piled into the first bin or two, so on a linear axis its density there sets the y limit and squashes the other nine curves into the bottom fifth of every panel |

## Backup — `backup/`, not in the running order

| file | slot | original figure | what it shows |
|---|---|---|---|
| `backup/roc_merged.pdf` | Z1 | `V3MERGED__roc_merged` | the merged ROC on its own |
| `backup/feature_importances.pdf` | Z2 | `V3MERGED__feature_importances` | which features the GBT actually uses |
| `backup/feature_separation.pdf` | Z3 | `V3MERGED__feature_separation` | per-feature separation power |
| `backup/nn_loss_curves.pdf` | Z4 | `V3MERGED__nn_loss_curves` | why the NN calibrates badly |
| `backup/auc_by_campaign.pdf` | Z5 | `V3MERGED__auc_by_campaign` | the ξ=0.02 comparison — **only** with the "not comparable" caveat (§0) |
| `backup/bkg_composition_streams.pdf` | Z6 | `V3MERGED__bkg_composition_streams` | background composition per streamline |
| `backup/compcut_impact_table.pdf` | Z7 | `V3MERGED__compcut_impact_table` | what the composition cut removes |
| `backup/presentable_byspecies_top2_pe_total.pdf` | Z8 | `TT_CF_GBT__presentable_byspecies_top2_pe_total` | D5 zoom, full width, one feature |
| `backup/presentable_byspecies_top2_n_hits.pdf` | Z9 | `TT_CF_GBT__presentable_byspecies_top2_n_hits` | D5 zoom, full width, one feature |
| `backup/presentable_features_byspecies.pdf` | Z10 | `RT_CF_GBT__presentable_features_byspecies` | reco-tag / ClusterFinder, same page as D5 |
| `backup/presentable_features_byspecies__tt_optics_xgb.pdf` | Z11 | `TT_OPTICS_XGB__presentable_features_byspecies` | truth-tag / OPTICS, same page as D5 |
| `backup/presentable_features_byspecies__rt_optics_gbt.pdf` | Z12 | `RT_OPTICS_GBT__presentable_features_byspecies` | reco-tag / OPTICS, same page as D5 |
| `backup/truthtag_optics__top6_shape.pdf` | Z13 | `V3BG__truthtag_optics__top6_shape` | C3 for the OPTICS clustering |
| `backup/cooccurrence_matrix.pdf` | D11 | `V3SPUR__cooccurrence_matrix` | backup for D3: every source pair at once as a matrix, for the pair that is not in D3's top ten |
| `backup/presentable_features_byspecies__tt_cf_gbt.pdf` | D12 | `TT_CF_GBT__presentable_features_byspecies` | backup: the same page on a **linear** y axis, solid lines — the version to show if the log axis is more than the audience wants |
| `backup/presentable_features_byspecies_dashed.pdf` | D13 | `TT_CF_GBT__presentable_features_byspecies_dashed` | backup: linear y, **dashed** — a dash pattern per class as well as a colour, for print and for the panels where two curves overlap |
| `backup/presentable_features_byspecies_logy_dashed.pdf` | D14 | `TT_CF_GBT__presentable_features_byspecies_logy_dashed` | backup: log y **and** dashed, the two fixes together |
| `backup/presentable_features_byspecies_nodarknoise.pdf` | D15 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise` | backup: linear y, solid, **dark noise left off entirely** — the other way to stop 135 clusters setting the y limit, and the clearest of the eight on a linear axis. Every remaining curve is identical to D12's: each class is area-normalised on its own, so dropping one frees the axis without renormalising the rest |
| `backup/presentable_features_byspecies_nodarknoise_logy.pdf` | D16 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise_logy` | backup: log y, solid, no dark noise |
| `backup/presentable_features_byspecies_nodarknoise_dashed.pdf` | D17 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise_dashed` | backup: linear y, dashed, no dark noise |
| `backup/presentable_features_byspecies_nodarknoise_logy_dashed.pdf` | D18 | `TT_CF_GBT__presentable_features_byspecies_nodarknoise_logy_dashed` | backup: log y, dashed, no dark noise |

---

Every file here is also in the flat `PRESENTATION_PLOTS/` folder under its slot-prefixed original name, and `MANIFEST.csv` maps between the two. Regenerate with `python -u collect_presentation_plots.py --do build`.
