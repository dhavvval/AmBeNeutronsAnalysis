"""
Thesis chapter 8 efficiency figures.

The numeric work lives in ambe.efficiency_extraction.efficiency_fit; this module only draws it, so
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
from .efficiency_fit import (assert_interior, background_scale_factor,
                                    build_multiplicity, fit_datadriven,
                                    fit_datadriven_scaled, fit_poisson,
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


def plot_chi2_2d(ctx, rp, xlabel="Background rate $\\lambda_n$ [candidates/trigger]",
                 name="eff_chi2_2d"):
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Neutron detection efficiency $\\epsilon_n$")
    ax.set_title(ctx.title("2D goodness-of-fit profile (cyan = 68.3 % contour)"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_plot(fig, ctx, name)


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
        # The shaded band is the PROFILE interval, so the reference line has to be
        # delta-chi2 = 1 to match it. It was 2.30 while the band was also built at
        # 2.30; both have moved together (see fit_poisson).
        ax.axhline(1.0, color="grey", ls=":", label="$\\Delta\\chi^2 = 1$")
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


MODEL_STYLE = {"counting": dict(color="0.35", marker="s", label="counting"),
               "poisson": dict(color="C0", marker="o", label="Poisson fit"),
               "dds": dict(color="C1", marker="D", label="data-driven fit")}


def _load_by_position(ctx, suffix=""):
    import pandas as pd
    path = ctx.csv_path("efficiency_fit_by_position" + suffix)
    if not path.exists():
        raise SystemExit(f"[plots.efficiency] missing {path}; run "
                         f"`ambe stats efficiency --config <cfg>` first")
    df = pd.read_csv(path)
    npz = path.with_suffix(".npz")
    arrays = np.load(npz) if npz.exists() else None
    from ..plots.heatmap import PORT_INFO
    df["port"] = [PORT_INFO.get((x, y, z), "Unknown")
                  for x, y, z in zip(df["x_pos"], df["y_pos"], df["z_pos"])]
    return df, arrays


def _ordered(df):
    """Position rows in the heatmap's physical port order, then y; index kept."""
    from ..plots.heatmap import PORT_ORDER
    d = df[df.level == "position"].copy()
    d["_p"] = d.port.map({p: k for k, p in enumerate(PORT_ORDER)})
    return d.sort_values(["_p", "y_pos"]).drop(columns="_p")


def _group(arrays, i, key):
    return arrays[f"g{i}__{key}"]


def _position_label(r):
    return f"{r.port}, y = {r.y_pos:g} cm (run {r.runs})"


def plot_bypos_fits(ctx, df, arrays):
    """One page per position: multiplicity data with both best-fit models."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = ctx.plot_path("eff_bypos_multiplicity_fits", "pdf")
    with PdfPages(out) as pdf:
        for i, r in _ordered(df).iterrows():
            counts = _group(arrays, i, "counts")
            n = counts.sum()
            x = np.arange(len(counts))
            unc = np.where(counts > 0, np.sqrt(counts), 1.15) / n
            fig, ax = plt.subplots()
            ax.errorbar(x, counts / n, yerr=unc, fmt="o", color="k", ms=6,
                        label="data")
            _step(ax, _group(arrays, i, "poisson_profile"), color="C0", lw=2,
                  label=f"Poisson fit, ε = {r.poisson_eff:.3f}, λ = {r.poisson_lambda:.3f}")
            _step(ax, _group(arrays, i, "dds_profile"), color="C1", lw=2, ls="--",
                  label=f"data-driven fit, ε = {r.dds_eff:.3f}, s = {r.dds_s:.2f}")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([str(k) for k in x[:-1]] + [f"≥{x[-1]}"])
            ax.set_xlabel("Neutron candidate multiplicity")
            ax.set_ylabel("Fraction of AmBe triggers")
            ax.set_title(f"{_position_label(r)}: {int(n):,} triggers, "
                         f"deviance {r.poisson_stat:.1f} (Poisson) vs "
                         f"{r.dds_stat:.2f} (data-driven)", fontsize=10)
            ax.legend(fontsize=9, frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[plots.efficiency] wrote {out}")


def plot_bypos_profiles(ctx, df, arrays):
    """One page per position: profiled Delta(-2 ln L) in eps, lambda and s."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = ctx.plot_path("eff_bypos_chi2_profiles", "pdf")
    with PdfPages(out) as pdf:
        for i, r in _ordered(df).iterrows():
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
            eg = _group(arrays, i, "eff_grid")
            span = max(r.poisson_eff_hi - r.poisson_eff_lo,
                       r.dds_eff_hi - r.dds_eff_lo)
            for tag in ("poisson", "dds"):
                c = _group(arrays, i, f"{tag}_chi2_eff")
                axes[0].plot(eg, c - c.min(), color=MODEL_STYLE[tag]["color"],
                             lw=2, label=MODEL_STYLE[tag]["label"])
            lo = min(r.poisson_eff, r.dds_eff) - 4 * span
            hi = max(r.poisson_eff, r.dds_eff) + 4 * span
            axes[0].set_xlim(lo, hi)
            axes[0].set_ylim(0, 10)
            axes[0].set_xlabel("Neutron detection efficiency ε")
            axes[0].set_ylabel("Δ(−2 ln L)")
            axes[0].set_title(f"ε: Poisson {r.poisson_eff:.3f} "
                              f"[{r.poisson_eff_lo:.3f}, {r.poisson_eff_hi:.3f}], "
                              f"data-driven {r.dds_eff:.3f} "
                              f"[{r.dds_eff_lo:.3f}, {r.dds_eff_hi:.3f}]", fontsize=9)
            axes[0].legend(fontsize=9, frameon=False)

            lg, cl = _group(arrays, i, "lam_grid"), _group(arrays, i, "poisson_chi2_lam")
            axes[1].plot(lg, cl - cl.min(), color="C0", lw=2)
            w = max(r.poisson_lambda_hi - r.poisson_lambda_lo, 0.005)
            axes[1].set_xlim(max(lg[0], r.poisson_lambda - 5 * w),
                             min(lg[-1], r.poisson_lambda + 5 * w))
            axes[1].set_ylim(0, 10)
            axes[1].set_xlabel("Poisson background rate λ [candidates/trigger]")
            axes[1].set_title(f"λ = {r.poisson_lambda:.3f} "
                              f"[{r.poisson_lambda_lo:.3f}, {r.poisson_lambda_hi:.3f}]",
                              fontsize=9)

            sg, cs = _group(arrays, i, "s_grid"), _group(arrays, i, "dds_chi2_s")
            axes[2].plot(sg, cs - cs.min(), color="C1", lw=2)
            axes[2].set_ylim(0, 10)
            axes[2].set_xlabel("Template rate scale s (6264 shape)")
            axes[2].set_title(f"s = {r.dds_s:.2f} [{r.dds_s_lo:.2f}, {r.dds_s_hi:.2f}], "
                              f"λ_eff = {r.dds_lambda_eff:.3f}", fontsize=9)
            fig.suptitle(f"Profiled likelihood, {_position_label(r)}; "
                         f"68 % interval at Δ(−2 ln L) = 1", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[plots.efficiency] wrote {out}")


def plot_bypos_vs_y(ctx, df):
    """Efficiency vs y, one panel per port: counting, Poisson and data-driven."""
    import matplotlib.pyplot as plt
    from ..plots.heatmap import PORT_ORDER
    d = df[df.level == "position"]
    fig, axes = plt.subplots(1, len(PORT_ORDER), figsize=(18, 4.8), sharey=True)
    for ax, port in zip(axes, PORT_ORDER):
        g = d[d.port == port].sort_values("y_pos")
        for k, (tag, off) in enumerate((("counting", -4), ("poisson", 0), ("dds", 4))):
            st = MODEL_STYLE[tag]
            if tag == "counting":
                y, lo, hi = g.naive_eff, g.naive_err, g.naive_err
            else:
                y = g[f"{tag}_eff"]
                lo, hi = y - g[f"{tag}_eff_lo"], g[f"{tag}_eff_hi"] - y
            ax.errorbar(g.y_pos + off, 100 * y, yerr=[100 * lo, 100 * hi],
                        fmt=st["marker"], color=st["color"], ms=6, capsize=2,
                        label=st["label"])
        ax.set_title(port)
        ax.set_xlabel("y position (cm)")
    axes[0].set_ylabel("Efficiency (%)")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(ctx.title("Neutron detection efficiency by port and height"))
    fig.tight_layout()
    save_plot(fig, ctx, "eff_bypos_vs_y")


def plot_bypos_gof(ctx, df):
    """Joint deviance per position for both models, with the ndof in the title."""
    import matplotlib.pyplot as plt
    d = _ordered(df).reset_index(drop=True)
    labels = [f"{r.port.replace('Port ', 'P')} y{r.y_pos:+g}" for r in d.itertuples()]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, d.poisson_stat, "o", color="C0", ms=7,
            label=f"Poisson fit ({int(d.poisson_ndof.iloc[0])} dof)")
    ax.plot(x, d.dds_stat, "D", color="C1", ms=7,
            label=f"data-driven fit ({int(d.dds_ndof.iloc[0])} dof)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_ylabel("Deviance at best fit")
    ax.set_title(ctx.title(
        f"Goodness of fit by position: data-driven p = {d.dds_p.min():.2f}–"
        f"{d.dds_p.max():.2f}, Poisson p < 0.001 everywhere"), fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_plot(fig, ctx, "eff_bypos_goodness_of_fit")


def plot_bypos_background(ctx, df):
    """Background rate per trigger: Poisson lambda vs the rescaled 6264 template mean."""
    import matplotlib.pyplot as plt
    d = _ordered(df).reset_index(drop=True)
    labels = [f"{r.port.replace('Port ', 'P')} y{r.y_pos:+g}" for r in d.itertuples()]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.errorbar(x - 0.15, d.poisson_lambda,
                yerr=[d.poisson_lambda - d.poisson_lambda_lo,
                      d.poisson_lambda_hi - d.poisson_lambda],
                fmt="o", color="C0", ms=6, capsize=2, label="Poisson λ")
    ax.plot(x + 0.15, d.dds_lambda_eff, "D", color="C1", ms=6,
            label="data-driven λ_eff (6264 shape × s^k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_ylabel("Background candidates per trigger")
    bkg_mean = float(np.sum(np.arange(len(d.background_hist.iloc[0].split()))
                            * np.array(d.background_hist.iloc[0].split(), float))
                     / np.sum(np.array(d.background_hist.iloc[0].split(), float)))
    ax.set_title(ctx.title(f"Background rate by position (6264 as measured: "
                           f"{bkg_mean:.2f} per trigger)"), fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_plot(fig, ctx, "eff_bypos_background_rate")


def plot_bypos_model_difference(ctx, df):
    """Data-driven minus Poisson, the background-model systematic, per position."""
    import matplotlib.pyplot as plt
    d = _ordered(df).reset_index(drop=True)
    labels = [f"{r.port.replace('Port ', 'P')} y{r.y_pos:+g}" for r in d.itertuples()]
    x = np.arange(len(d))
    stat = 50 * (d.dds_eff_hi - d.dds_eff_lo)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.errorbar(x, 100 * d.delta_model_signed, yerr=stat, fmt="D", color="C1",
                ms=6, capsize=2, label="data-driven − Poisson (bar: data-driven stat)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_ylabel("Δε (percentage points)")
    ax.set_title(ctx.title(f"Background-model difference: mean "
                           f"{100 * d.delta_model_signed.mean():+.2f} pp"), fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_plot(fig, ctx, "eff_bypos_model_difference")


def plot_bypos_counting_residual(ctx, df):
    """Counting minus each fitted efficiency, per position: the background correction."""
    import matplotlib.pyplot as plt
    d = _ordered(df).reset_index(drop=True)
    labels = [f"{r.port.replace('Port ', 'P')} y{r.y_pos:+g}" for r in d.itertuples()]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    means = {}
    for tag, off in (("poisson", -0.15), ("dds", 0.15)):
        st = MODEL_STYLE[tag]
        res = 100 * (d.naive_eff - d[f"{tag}_eff"])
        err = 50 * (d[f"{tag}_eff_hi"] - d[f"{tag}_eff_lo"])
        means[tag] = res.mean()
        ax.errorbar(x + off, res, yerr=err, fmt=st["marker"], color=st["color"],
                    ms=6, capsize=2, label=f"counting − {st['label']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_ylabel("Δε (percentage points)")
    ax.set_title(ctx.title(
        f"Counting minus fitted efficiency: mean {means['poisson']:.2f} pp (Poisson), "
        f"{means['dds']:.2f} pp (data-driven)"), fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_plot(fig, ctx, "eff_bypos_counting_residual")


def plot_bypos_binning(ctx, df):
    """epsilon at 5 and 8 bins minus the nominal, if those fits exist."""
    import matplotlib.pyplot as plt
    base = _ordered(df).reset_index(drop=True)
    alts = []
    for nb in (5, 8):
        try:
            alt, _ = _load_by_position(ctx, f"_nbins{nb}")
        except SystemExit:
            continue
        alts.append((nb, alt[alt.level == "position"].reset_index(drop=True)))
    if not alts:
        return
    labels = [f"{r.port.replace('Port ', 'P')} y{r.y_pos:+g}" for r in base.itertuples()]
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for (nb, alt), marker, off in zip(alts, ("o", "s"), (-0.15, 0.15)):
        m = base.merge(alt, on=["x_pos", "y_pos", "z_pos"], suffixes=("", "_a"))
        ax.plot(x + off, 100 * (m.dds_eff_a - m.dds_eff), marker, color="C1",
                ms=6, mfc="none" if nb == 8 else "C1",
                label=f"{nb} bins − 4 bins, data-driven")
    stat = 50 * (base.dds_eff_hi - base.dds_eff_lo)
    ax.fill_between(x, -stat, stat, color="0.85",
                    label="± data-driven stat (4 bins)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_ylabel("Δε (percentage points)")
    ax.set_title(ctx.title("Binning sensitivity of the data-driven efficiency"),
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_plot(fig, ctx, "eff_bypos_binning_sensitivity")


def run_by_position(ctx):
    df, arrays = _load_by_position(ctx)
    plot_bypos_vs_y(ctx, df)
    plot_bypos_gof(ctx, df)
    plot_bypos_background(ctx, df)
    plot_bypos_model_difference(ctx, df)
    plot_bypos_counting_residual(ctx, df)
    plot_bypos_binning(ctx, df)
    if arrays is None:
        print("[plots.efficiency] no profile .npz beside the CSV; rerun "
              "`ambe stats efficiency` to get the per-position fit and profile pages")
        return
    plot_bypos_fits(ctx, df, arrays)
    plot_bypos_profiles(ctx, df, arrays)
    print(f"[plots.efficiency] figures in {ctx.plots_dir}")


def run(ctx, argv=None):
    set_style()
    fp = dict(ctx.fit_params or {})
    if not fp:
        raise SystemExit("[plots.efficiency] config has no fit_params block")
    if "source_runs" in fp:
        return run_by_position(ctx)

    src_run, bkg_run = int(fp["source_run"]), int(fp["background_run"])
    n_bins = int(fp.get("n_bins", 8))
    engine = fp.get("engine", "analytic")
    rng = np.random.default_rng(int(fp.get("seed", 0)))
    eff_grid, lam_grid = make_grid(fp["eff_grid"]), make_grid(fp["lambda_grid"])
    scale = background_scale_factor(
        float(fp.get("acquisition_window_ns", 67000)),
        float(fp.get("signal_window_start_ns", 2000)),
        float(fp.get("bkg_window_start_ns", 2000)))

    ei = _resolve_event_index(ctx)
    denom = fp.get("denominator", "ambe_triggers")
    overflow_policy = fp.get("overflow_policy", "fail")
    likelihood = fp.get("likelihood", "thesis_chi2")
    src = build_multiplicity(ei, src_run, n_bins, denom,
                             overflow_policy=overflow_policy)
    bkg = build_multiplicity(ei, bkg_run, n_bins, denom,
                             overflow_policy=overflow_policy)

    rp = fit_poisson(src, eff_grid, lam_grid, engine=engine,
                     n_throws=int(fp.get("n_throws", 1_000_000)), rng=rng,
                     likelihood=likelihood)
    rd = fit_datadriven(src, bkg, eff_grid, engine=engine,
                        n_throws=int(fp.get("n_throws", 1_000_000)), rng=rng,
                        bkg_scale=scale, likelihood=likelihood,
                        bkg_unc_mode=fp.get("background_uncertainty", "thesis_diagonal"),
                        n_bootstrap=int(fp.get("n_bootstrap", 400)))
    assert_interior(rp, "poisson")
    assert_interior(rd, "data-driven")

    print(f"[plots.efficiency] {rp.summary()}")
    print(f"[plots.efficiency] {rd.summary()}")

    plot_multiplicity(ctx, src, bkg)
    plot_bestfit_overlay(ctx, src, rp, rd)
    plot_chi2_1d(ctx, rd)
    plot_chi2_2d(ctx, rp)
    plot_chi2_projections(ctx, rp)
    if "s_grid" in fp:
        rs = fit_datadriven_scaled(
            src, bkg, eff_grid, make_grid(fp["s_grid"]), likelihood=likelihood,
            bkg_unc_mode=fp.get("background_uncertainty", "thesis_diagonal"),
            n_bootstrap=int(fp.get("n_bootstrap", 400)), seed=int(fp.get("seed", 0)))
        print(f"[plots.efficiency] {rs.summary()}")
        plot_chi2_2d(ctx, rs, xlabel=f"Template rate scale $s$ (run {bkg_run} shape)",
                     name="eff_chi2_2d_datadriven_scaled")

    print(f"[plots.efficiency] figures in {ctx.plots_dir}")


def cli(ctx, argv=None):
    run(ctx, argv)
