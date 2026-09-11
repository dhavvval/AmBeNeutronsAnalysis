#!/usr/bin/env python
"""Diagnose why the data-driven multiplicity fit's chi2/ndof can look better than
the Poisson fit's even when it isn't a better description of the data.

Quantifies how much of a data-driven fit's chi2/ndof comes from the background
run's own statistical uncertainty swamping the residual, by recomputing chi2 with
sigma_bkg zeroed out (source-only denominator) and by scanning what background
trigger count would be needed for sigma_bkg to stop dominating.

Usage:
    python scripts/diagnose_datadriven_chi2.py --source 6046 --bkg 6264 --tag effpair_box
    python scripts/diagnose_datadriven_chi2.py --source 6265 --bkg 6264 --tag effpair_box
"""
import argparse
import numpy as np

from ambe.stats.efficiency_fit import (
    load_event_index, build_multiplicity, fit_datadriven, make_grid,
    analytic_profile_datadriven, chi2, delta_chi2_interval,
)


def refit_with_scaled_bkg_unc(source, bkg, eff_grid, bkg_unc_scale):
    """Re-run the data-driven scan with the background template's per-bin
    uncertainty scaled by `bkg_unc_scale` (0 = pretend sigma_bkg is negligible,
    1 = as measured, >1 = simulate a smaller/noisier background sample)."""
    n_bins = len(source.counts)
    data, data_unc = source.normed, source.normed_unc
    bkg_p, bkg_unc = bkg.normed, bkg.normed_unc * bkg_unc_scale

    chi2_vals = np.empty(len(eff_grid))
    profiles = []
    for i, eff in enumerate(eff_grid):
        prof = analytic_profile_datadriven(eff, bkg_p, n_bins)
        profiles.append(prof)
        chi2_vals[i] = chi2(prof, data, data_unc, bkg_unc)
    best = int(np.argmin(chi2_vals))
    lo, hi = delta_chi2_interval(eff_grid, chi2_vals, 1.0)
    return eff_grid[best], lo, hi, chi2_vals[best], n_bins - 1


def crossover_scan(source, bkg, eff_grid, target_ndof_ratio=1.0, n_points=25):
    """Find the background-uncertainty scale factor at which chi2/ndof crosses
    `target_ndof_ratio`, and translate that into an implied background trigger
    count (sigma ~ 1/sqrt(N), so scale = sqrt(N_measured / N_implied))."""
    scales = np.linspace(0.05, 1.0, n_points)
    results = []
    for s in scales:
        _, _, _, chi2_min, ndof = refit_with_scaled_bkg_unc(source, bkg, eff_grid, s)
        results.append((s, chi2_min, chi2_min / ndof))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--bkg", type=int, required=True)
    ap.add_argument("--tag", default="effpair_box")
    ap.add_argument("--nbins", type=int, default=8)
    ap.add_argument("--denominator", default="ambe_triggers")
    ap.add_argument("--numerator", default="noncosmic")
    args = ap.parse_args()

    ei = load_event_index([
        f"TriggerSummary/EventIndex_AmBe2.0v4_{args.tag}_{args.source}.parquet",
        f"TriggerSummary/EventIndex_AmBe2.0v4_{args.tag}_{args.bkg}.parquet",
    ])
    src = build_multiplicity(ei, args.source, args.nbins, args.denominator, args.numerator)
    bkg = build_multiplicity(ei, args.bkg, args.nbins, args.denominator, args.numerator)

    eg = make_grid({"min": 0.30, "max": 0.95, "step": 0.001})

    print(f"=== source {args.source}, background {args.bkg} ({args.tag}) ===")
    print(f"background: {bkg.denominator} ambe_triggers, {bkg.n_with_candidate} candidate-bearing events")

    # 1) baseline: as-measured data-driven fit
    rd = fit_datadriven(src, bkg, eg, engine="analytic")
    print(f"\n[as measured]     eff={rd.best_eff:.4f} +{rd.eff_hi-rd.best_eff:.4f}/"
          f"-{rd.best_eff-rd.eff_lo:.4f}  chi2/ndof={rd.chi2_min:.2f}/{rd.ndof}"
          f"={rd.chi2_min/rd.ndof:.3f}")

    # 2) sigma_bkg -> 0 (pretend background is perfectly known)
    eff0, lo0, hi0, chi2_0, ndof0 = refit_with_scaled_bkg_unc(src, bkg, eg, 0.0)
    print(f"[sigma_bkg -> 0]  eff={eff0:.4f} +{hi0-eff0:.4f}/-{eff0-lo0:.4f}  "
          f"chi2/ndof={chi2_0:.2f}/{ndof0}={chi2_0/ndof0:.3f}")
    print("  (this is the fit quality once the background's own thin statistics")
    print("   are no longer allowed to widen the denominator -- i.e. the honest")
    print("   comparison to the Poisson fit's chi2/ndof.)")

    # 3) crossover scan: at what bkg_unc scale does chi2/ndof cross 1?
    print("\n[sigma_bkg scale scan] scale -> chi2/ndof (scale=1.0 is as-measured)")
    for s, c2, ratio in crossover_scan(src, bkg, eg):
        implied_n = bkg.denominator / max(s, 1e-9) ** 2
        flag = "  <-- crosses 1" if ratio >= 1.0 else ""
        print(f"  scale={s:.3f}  (implied N_bkg~{implied_n:,.0f})  "
              f"chi2/ndof={ratio:.3f}{flag}")


if __name__ == "__main__":
    main()
