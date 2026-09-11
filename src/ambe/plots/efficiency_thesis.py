"""
Thesis chapter 8 efficiency figures.

The numeric work lives in ambe.stats.efficiency_fit; this module only draws it, so
the fit stays headless and testable. Figure numbering follows Pershing thesis ch. 8:

    8.18/8.19  multiplicity distributions, source and background
    8.20       both best-fit models overlaid on the data
    8.21       1D chi2 profile, data-driven model
    8.22       2D chi2 map over (eps_n, lambda_n)
    8.23/8.24  the two 1D projections through the 2D minimum
    8.29       efficiency vs position

Usage:  ambe plots efficiency --config configs/data_ambe2v4_effpair_box.yaml
"""

from __future__ import annotations

import numpy as np

from ..plotting import save_plot, set_style
from ..stats.efficiency_fit import (assert_interior, background_scale_factor,
                                    build_multiplicity, fit_datadriven, fit_poisson,
                                    make_grid, _resolve_event_index)


def _step(ax, counts, **kw):
    """Draw a histogram as a step, including the zero bin."""
    x = np.arange(len(counts) + 1) - 0.5
    y = np.append(counts, counts[-1])
    ax.step(x, y, where="post", **kw)


def plot_multiplicity(ctx, src, bkg):
    """thesis figs 8.18 / 8.19 -- the distributions the fit is performed on."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, h, name in ((axes[0], src, "source"), (axes[1], bkg, "background")):
        _step(ax, h.counts, color="C0" if name == "source" else "C3", lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("Neutron candidate multiplicity")
        ax.set_ylabel("Number of acquisitions")
        ax.set_title(f"Run {h.run} ({name})\n"
                     f"{h.denominator:,} triggers, naive eff "
                     f"{100*h.naive_efficiency:.2f} %")
        ax.grid(alpha=.3, ls="--")
    fig.suptitle(ctx.title("Neutron candidate multiplicity"))
    fig.tight_layout()
    return save_plot(fig, ctx, "eff_multiplicity")


def plot_bestfit_overlay(ctx, src, rp, rd):
    """thesis fig 8.20 -- both models against the data."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.arange(len(src.counts))
    ax.errorbar(x, src.normed, yerr=src.normed_unc, fmt="o", color="k",
                ms=7, label=f"Run {src.run} data", zorder=5)
    _step(ax, rp.best_profile, color="C0", lw=2,
          label=f"Uncorr. bkg. fit ($\\epsilon_n$={rp.best_eff:.3f}, "
                f"$\\lambda_n$={rp.best_lambda:.3f})")
    _step(ax, rd.best_profile, color="C3", lw=2, ls="--",
          label=f"Data-driven bkg. fit ($\\epsilon_n$={rd.best_eff:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("Neutron candidate multiplicity")
    ax.set_ylabel("Fraction of acquisitions")
    ax.set_title(ctx.title("Best fit multiplicity models"))
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, ls="--")
    fig.tight_layout()
    return save_plot(fig, ctx, "eff_bestfit_overlay")


def plot_chi2_1d(ctx, rd):
    """thesis fig 8.21 -- the data-driven profile, with the delta-chi2=1 band."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    d = rd.chi2_eff - rd.chi2_min
    ax.plot(rd.eff_grid, d, color="C3", lw=2, label="Data-driven model")
    ax.axhline(1.0, color="grey", ls=":", label="$\\Delta\\chi^2 = 1$")
    ax.axvspan(rd.eff_lo, rd.eff_hi, color="C3", alpha=.15)
    ax.axvline(rd.best_eff, color="C3", ls="--", lw=1)
    ax.set_ylim(0, 10)
    ax.set_xlim(max(rd.eff_grid[0], rd.best_eff - 0.15),
                min(rd.eff_grid[-1], rd.best_eff + 0.15))
    ax.set_xlabel("Neutron detection efficiency $\\epsilon_n$")
    ax.set_ylabel("$\\chi^2 - \\chi^2_{min}$")
    ax.set_title(ctx.title("Goodness of fit vs efficiency (data-driven)"))
    ax.legend()
    ax.grid(alpha=.3, ls="--")
    fig.tight_layout()
    return save_plot(fig, ctx, "eff_chi2_1d_datadriven")


def plot_chi2_2d(ctx, rp):
    """thesis fig 8.22 -- the 2D map, with the delta-chi2=2.30 contour drawn."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    d = rp.chi2_surface - rp.chi2_min
    # Zoom to the neighbourhood of the minimum. The scan grid is deliberately wide
    # so the fit cannot pin to an edge, but plotting all of it leaves the basin a
    # few pixels across (thesis fig 8.22 shows the zoomed region).
    ei = np.flatnonzero(d.min(axis=1) <= 80)
    li = np.flatnonzero(d.min(axis=0) <= 80)
    es, ls = slice(ei[0], ei[-1] + 1), slice(li[0], li[-1] + 1)
    eg, lg, dz = rp.eff_grid[es], rp.lam_grid[ls], d[es, ls]
    im = ax.pcolormesh(lg, eg, np.clip(dz, 0, 80), cmap="magma_r", shading="auto")
    ax.contour(lg, eg, dz, levels=[2.30], colors="cyan", linewidths=2)
    ax.plot(rp.best_lambda, rp.best_eff, "c*", ms=16,
            label=f"best fit ({rp.best_eff:.3f}, {rp.best_lambda:.3f})")
    fig.colorbar(im, ax=ax, label="$\\chi^2 - \\chi^2_{min}$")
    ax.set_xlabel("Background rate $\\lambda_n$ [candidates/trigger]")
    ax.set_ylabel("Neutron detection efficiency $\\epsilon_n$")
    ax.set_title(ctx.title("2D goodness-of-fit profile (cyan = 68.3 % contour)"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_plot(fig, ctx, "eff_chi2_2d")


def plot_chi2_projections(ctx, rp):
    """thesis figs 8.23 / 8.24 -- the profiled 1D slices."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, grid, vals, best, lo, hi, xlabel in (
            (axes[0], rp.lam_grid, rp.chi2_lam, rp.best_lambda, rp.lam_lo, rp.lam_hi,
             "Background mean $\\lambda_n$ [clusters/trigger]"),
            (axes[1], rp.eff_grid, rp.chi2_eff, rp.best_eff, rp.eff_lo, rp.eff_hi,
             "Neutron detection efficiency $\\epsilon_n$")):
        d = vals - np.min(vals)
        ax.plot(grid, d, color="C0", lw=2, label="Uncorr. bkg. model")
        ax.axhline(2.30, color="grey", ls=":", label="$\\Delta\\chi^2 = 2.30$")
        ax.axvspan(lo, hi, color="C0", alpha=.15)
        ax.axvline(best, color="C0", ls="--", lw=1)
        ax.set_ylim(0, 30)
        ax.set_xlim(max(grid[0], best - 6 * (hi - lo + 1e-9)),
                    min(grid[-1], best + 6 * (hi - lo + 1e-9)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$\\chi^2 - \\chi^2_{min}$")
        ax.legend(fontsize=9)
        ax.grid(alpha=.3, ls="--")
    fig.suptitle(ctx.title("Profiled goodness of fit (other parameter minimised)"))
    fig.tight_layout()
    return save_plot(fig, ctx, "eff_chi2_projections")


def plot_capture_time(ctx, candidate_csv, run, fit_lo_us=15.0, fit_hi_us=67.0):
    """Capture-time fit -- the evidence that a run is source-like.

    Fitted from 15 us, as the thesis does (section 8.5.2): AmBe neutrons need time to
    thermalise, so the turn-on below ~15 us is not exponential and including it biases
    tau upward.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.optimize import curve_fit

    d = pd.read_csv(candidate_csv)
    t = d["clusterTime"].to_numpy(float) / 1000.0
    sel = (t >= fit_lo_us) & (t <= fit_hi_us)
    if sel.sum() < 50:
        return None
    h, edges = np.histogram(t[sel], bins=30, range=(fit_lo_us, fit_hi_us))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    f = lambda x, A, tau, B: A * np.exp(-(x - fit_lo_us) / tau) + B
    try:
        p, cov = curve_fit(f, ctr, h, p0=[max(h[0], 1), 30.0, max(h[-1], 1.0)],
                           sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000)
    except Exception:
        return None
    tau, tau_e = p[1], float(np.sqrt(cov[1][1]))
    r = (h - f(ctr, *p)) / np.sqrt(np.maximum(h, 1))
    chi2ndof = float(np.sum(r ** 2) / (len(h) - 3))

    fig, ax = plt.subplots()
    all_h, all_e = np.histogram(t, bins=65, range=(2, fit_hi_us))
    ax.step(all_e[:-1], all_h, where="post", color="C7", alpha=.5,
            label="all delayed clusters")
    ax.errorbar(ctr, h, yerr=np.sqrt(np.maximum(h, 1)), fmt="o", ms=4, color="C0",
                label=f"fit range [{fit_lo_us:g}, {fit_hi_us:g}] $\\mu$s")
    xs = np.linspace(fit_lo_us, fit_hi_us, 300)
    ax.plot(xs, f(xs, *p), "k-", lw=2,
            label=f"$\\tau$ = {tau:.1f} $\\pm$ {tau_e:.1f} $\\mu$s, "
                  f"$\\chi^2$/ndof = {chi2ndof:.2f}")
    ax.set_xlabel("Neutron candidate time [$\\mu$s]")
    ax.set_ylabel("Number of candidates")
    ax.set_title(ctx.title(f"Capture time, run {run}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, ls="--")
    fig.tight_layout()
    save_plot(fig, ctx, f"eff_capture_time_{run}")
    return dict(run=run, tau_us=tau, tau_err_us=tau_e, chi2_per_ndof=chi2ndof)


def run(ctx, argv=None):
    import glob

    set_style()
    fp = dict(ctx.fit_params or {})
    if not fp:
        raise SystemExit("[plots.efficiency] config has no fit_params block")

    src_run, bkg_run = int(fp["source_run"]), int(fp["background_run"])
    n_bins = int(fp.get("n_bins", 8))
    engine = fp.get("engine", "analytic")
    rng = np.random.default_rng(int(fp.get("seed", 0)))
    eff_grid, lam_grid = make_grid(fp["eff_grid"]), make_grid(fp["lambda_grid"])
    scale = background_scale_factor(
        float(fp.get("acquisition_window_ns", 67000)),
        float(fp.get("signal_window_start_ns", 2000)),
        float(fp.get("bkg_window_start_ns", 2000)))

    ei = _resolve_event_index(ctx, ctx.run_name)
    denom = fp.get("denominator", "ambe_triggers")
    num_mode = fp.get("numerator_mode", "noncosmic")
    src = build_multiplicity(ei, src_run, n_bins, denom, num_mode)
    bkg = build_multiplicity(ei, bkg_run, n_bins, denom, num_mode)

    rp = fit_poisson(src, eff_grid, lam_grid, engine=engine,
                     n_throws=int(fp.get("n_throws", 1_000_000)), rng=rng)
    rd = fit_datadriven(src, bkg, eff_grid, engine=engine,
                        n_throws=int(fp.get("n_throws", 1_000_000)), rng=rng,
                        bkg_scale=scale)
    assert_interior(rp, "poisson")
    assert_interior(rd, "data-driven")

    print(f"[plots.efficiency] {rp.summary()}")
    print(f"[plots.efficiency] {rd.summary()}")

    plot_multiplicity(ctx, src, bkg)
    plot_bestfit_overlay(ctx, src, rp, rd)
    plot_chi2_1d(ctx, rd)
    plot_chi2_2d(ctx, rp)
    plot_chi2_projections(ctx, rp)

    pats = ctx.inputs.get("candidate_csvs") or []
    if isinstance(pats, str):
        pats = [pats]
    for pat in pats:
        for path in sorted(glob.glob(str(pat))):
            for r in (src_run, bkg_run, fp.get("crosscheck_run")):
                if r and f"_{r}." in path:
                    info = plot_capture_time(ctx, path, r)
                    if info:
                        print(f"[plots.efficiency] run {r}: tau = "
                              f"{info['tau_us']:.1f} +/- {info['tau_err_us']:.1f} us, "
                              f"chi2/ndof = {info['chi2_per_ndof']:.2f}")
    print(f"[plots.efficiency] figures in {ctx.plots_dir}")


def cli(ctx, argv=None):
    run(ctx, argv)
