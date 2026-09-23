# AmBe v4: single cosmic veto, and efficiency at every source position

2026-09-22. Box cuts (PE ≤ 100, CB < 0.45, 2 ≤ t ≤ 67 µs, hits ≥ 5), IC 700–1200, all 28
v4 source runs on 26 positions. Tag `AmBe2.0v4_all28_box_veto`; config
`configs/data_ambe2v4_all28_box.yaml`. Outputs in
`ambe_output/ambe_data/AmBe2.0v4_all28_box_veto/{csv,plots}`.

```bash
ambe data process     --config configs/data_ambe2v4_effpair_box.yaml --selection box   # 6264 background index
ambe data process     --config configs/data_ambe2v4_all28_box.yaml   --selection box
ambe stats efficiency --config configs/data_ambe2v4_all28_box.yaml --jobs 4            # nominal, 4 bins
ambe stats efficiency --config configs/data_ambe2v4_all28_box.yaml --jobs 4 --n-bins 5 # sensitivity
ambe plots heatmap    --config configs/data_ambe2v4_all28_box.yaml
```

## 1. The cosmic veto is one path now

The two-pass veto (any cluster with t < 2 µs or PE > 100 vetoes the whole event, decided
before any candidate is banked) is the only code path. `stage1.legacy_ordering` and
`efficiency_fit`'s `numerator_mode` are gone; a config still setting the former is refused,
and `build_multiplicity` refuses an EventIndex in which a vetoed event carries candidates.

Reprocess checks against the 2026-08-26 `all28_box` summary: `total_events`,
`cosmic_events`, `ambe_triggers` identical in all 26 cells; `unique_neutron_triggers` lower
or equal in every cell; zero vetoed events with candidates over all 28 EventIndex files.
The effpair rerun lost exactly the documented 532 triggers on 6265 and its fit is
unchanged (0.605 Poisson / 0.640 thesis data-driven, histogram
[5704, 9192, 639, 82, 9, 1, 0, 1]).

The leak cost 1.8–3.7 pp per position, largest near the centre. The published 57.23 %
cannot be regenerated any more.

## 2. Efficiency definitions

- **counting**: unique_neutron_triggers / ambe_triggers (`ambe plots heatmap`).
- **Poisson ε**: multiplicity fit, M = Bernoulli(ε) + Pois(λ), λ free.
- **data-driven ε** (nominal): M = Bernoulli(ε) + B_s, B_s(k) ∝ B(k)·s^k. B is the true
  6264 template, whose bins are nuisance parameters constrained by 6264's own multinomial
  term (`background_uncertainty: profile`). The fit therefore minimises
  D_source(ε, B_s) + D_6264(B). s is free per position. λ_eff = mean of B_s.

Why s is free: 6264 is the only no-source run and it is cosmic-loaded (96 % vetoed, 22
AmBe triggers, 0.32 candidates/trigger). A no-source background has no geometry
dependence, so what has to be carried to other ports is its *rate*, set by detector
conditions. Its *shape* is what the data supply. With s fixed at 1 the fit is disfavoured
at 24/26 positions (p ≈ 0.02–0.07); with s free it is good everywhere.

Why a profile likelihood: the thesis χ² put the 22-trigger template error in the χ²
denominator (hence the old ±0.108 and a good-looking χ² for the wrong reason). A
Dirichlet-bootstrap marginal was tried and gave a jagged profile (Monte Carlo noise from a
few dominant replicas). The joint profile is smooth, exact, and 6264's statistics enter as
a likelihood.

## 3. Results (4 bins: 0, 1, 2, ≥3)

| port | y (cm) | runs | AmBe triggers | counting, pre-fix | counting | Poisson ε | data-driven ε | s | λ_eff | p (dd) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -100 | 6232 | 55,342 | 50.50 | 48.17 | 43.6 ± 0.3 | 44.6 ± 0.2 | 0.30 | 0.070 | 0.82 |
| 1 | -50 | 6234 | 8,646 | 61.80 | 58.78 | 55.0 ± 0.6 | 55.5 ± 0.6 | 0.32 | 0.075 | 0.67 |
| 1 | 0 | 6231 | 15,837 | 64.14 | 61.15 | 57.4 ± 0.5 | 57.9 ± 0.4 | 0.32 | 0.077 | 0.64 |
| 1 | 50 | 6235 | 12,790 | 62.58 | 59.70 | 55.5 ± 0.5 | 56.1 ± 0.5 | 0.33 | 0.080 | 0.52 |
| 1 | 100 | 6230 | 9,372 | 54.11 | 51.70 | 46.5 ± 0.6 | 47.6 ± 0.6 | 0.36 | 0.086 | 0.83 |
| 2 | -100 | 6237 | 13,920 | 52.41 | 50.19 | 45.9 ± 0.5 | 46.6 ± 0.5 | 0.29 | 0.067 | 0.65 |
| 2 | -50 | 6239 | 13,807 | 60.77 | 58.23 | 54.2 ± 0.5 | 54.7 ± 0.5 | 0.33 | 0.077 | 0.60 |
| 2 | 0 | 6241 | 13,268 | 63.17 | 60.39 | 56.3 ± 0.5 | 56.8 ± 0.5 | 0.33 | 0.079 | 0.49 |
| 2 | 50 | 6242 | 25,217 | 61.92 | 59.10 | 55.1 ± 0.4 | 55.5 ± 0.4 | 0.33 | 0.078 | 0.55 |
| 2 | 100 | 6188 6189 | 9,124 | 54.12 | 51.60 | 46.0 ± 0.7 | 47.0 ± 0.6 | 0.37 | 0.091 | 0.64 |
| 3 | -100 | 6243 | 11,547 | 40.31 | 38.50 | 32.4 ± 0.6 | 34.0 ± 0.6 | 0.32 | 0.074 | 0.83 |
| 3 | -60 | 6187 | 7,181 | 46.99 | 44.84 | 39.5 ± 0.7 | 40.7 ± 0.7 | 0.32 | 0.075 | 0.78 |
| 3 | -50 | 6244 | 15,991 | 48.96 | 46.86 | 42.3 ± 0.5 | 43.3 ± 0.5 | 0.30 | 0.069 | 0.85 |
| 3 | 0 | 6166 | 13,086 | 50.82 | 48.58 | 43.7 ± 0.5 | 44.6 ± 0.5 | 0.32 | 0.074 | 0.73 |
| 3 | 50 | 6246 | 13,418 | 49.36 | 47.05 | 41.7 ± 0.5 | 43.0 ± 0.5 | 0.33 | 0.079 | 0.84 |
| 3 | 100 | 6186 | 12,953 | 40.87 | 38.89 | 33.4 ± 0.6 | 34.9 ± 0.5 | 0.30 | 0.069 | 0.90 |
| 4 | -100 | 6060 | 11,714 | 49.70 | 47.51 | 42.8 ± 0.6 | 43.6 ± 0.5 | 0.30 | 0.068 | 0.61 |
| 4 | -50 | 6247 | 11,196 | 60.58 | 57.75 | 54.0 ± 0.5 | 54.5 ± 0.5 | 0.30 | 0.070 | 0.59 |
| 4 | 0 | 6056 | 11,577 | 65.29 | 62.15 | 58.0 ± 0.5 | 58.5 ± 0.5 | 0.35 | 0.085 | 0.49 |
| 4 | 50 | 6249 6250 | 11,900 | 62.84 | 60.03 | 56.3 ± 0.5 | 56.8 ± 0.5 | 0.31 | 0.073 | 0.61 |
| 4 | 100 | 6061 | 12,065 | 54.52 | 51.98 | 45.8 ± 0.6 | 47.0 ± 0.6 | 0.40 | 0.098 | 0.64 |
| 5 | -100 | 6062 | 12,315 | 62.97 | 60.11 | 56.5 ± 0.5 | 57.0 ± 0.5 | 0.31 | 0.072 | 0.62 |
| 5 | -50 | 6251 | 13,384 | 69.55 | 66.28 | 63.2 ± 0.5 | 63.5 ± 0.5 | 0.31 | 0.073 | 0.54 |
| 5 | 0 | 6046 | 21,676 | 71.82 | 68.48 | 65.1 ± 0.4 | 65.5 ± 0.4 | 0.35 | 0.085 | 0.54 |
| 5 | 50 | 6252 | 11,793 | 69.46 | 65.74 | 62.7 ± 0.5 | 63.2 ± 0.5 | 0.31 | 0.072 | 0.72 |
| 5 | 100 | 6165 | 12,144 | 63.04 | 60.24 | 56.7 ± 0.5 | 57.0 ± 0.5 | 0.29 | 0.068 | 0.46 |

All 26 positions: counting 57.25 % → 54.62 %.

± is the profile Δ(−2lnL) = 1 interval. p (dd) is χ²(joint deviance, 1 dof).

- Poisson fails at every position (deviance 23–219 on 1 dof): the tail is not Poisson.
- Data-driven − Poisson = +0.3 to +1.6 pp (mean +0.76): the background-model systematic.
- Both fits are 3–6 pp below counting — the background correction.
- λ_eff = 0.067–0.098 per trigger, against Poisson λ = 0.093 ± 0.009 across positions.
- ε barely depends on s (fixed s = 1 moves it ≤ 0.06 pp at 4 bins): the zero bin pins it.

## 4. Binning sensitivity

| n_bins | Δε data-driven vs 4 bins | Δε Poisson vs 4 bins | data-driven p |
|---|---|---|---|
| 5 | mean −0.04, range −0.10 … −0.01 pp | mean −0.03, range −0.20 … +0.10 pp | min 0.36, median 0.49 |
| 8 | mean −0.06, range −0.25 … −0.01 pp | mean −0.04, range −0.30 … +0.10 pp | min 0.49, median 0.91 |

**Known degeneracy.** The likelihood has a second, spurious solution at ε → 0 where the
free template becomes a copy of the source histogram, at a cost of only 6264's own
deviance (~9). With s free the physical minimum always wins clearly (position 6046:
Δ = 358 at 4 bins, 28 at 8 bins). With s fixed and 8 bins the two tie, and the fit had
drifted to ε = 0 at 22 of 54 groups. The optimum is now restricted to the configured
`eff_grid`; after that fix no fit in any binning sits at a grid edge (8 bins rerun). It is a statement that the 22-trigger background run is weak: a background
run with more AmBe triggers would remove it.

## 5. Caveats

- 6264 is the only background run and has 22 AmBe triggers after the gate and veto.
- 6264 has real cosmic-induced captures (four-run study); after the veto they are part of
  the template shape.
- δ_h (housing) is not evaluated. δ_bt still has no livetime.
- The Poisson model fails its goodness-of-fit everywhere. Do not quote it as nominal.
