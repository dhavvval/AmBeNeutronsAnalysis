"""
Combined plot: neutron multiplicity, PE vs CB, capture-time fit, fit residuals.

Ported to use RunContext for input paths, output paths, titles and cuts.

Entry:  ambe plots combined --config configs/<something>.yaml
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

from ..context import RunContext
from ..io import inputs_from_ctx, load_csvs
from ..plotting import save_plot, set_style


def neut_capture(t, A, therm, tau, B):
    """Neutron capture PDF: thermalisation * exponential decay + bg."""
    return A * (1 - np.exp(-t / therm)) * np.exp(-t / tau) + B


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots combined")
    p.parse_args(list(argv) if argv else [])
    set_style()

    csv_paths = inputs_from_ctx(ctx, "candidate_csvs")
    df = load_csvs(csv_paths)
    if df.empty:
        print("[plots.combined] no input rows; aborting")
        return

    pe = df["clusterPE"]
    ccb = df["clusterChargeBalance"]
    df["clusterTime"] = df["clusterTime"] / 1000.0  # ns -> us
    ct = df["clusterTime"]

    # Cuts -- can override via config
    pe_max = ctx.cuts.get("pe_max", np.inf)
    cb_max = ctx.cuts.get("charge_balance_max", np.inf)
    sel = (pe < pe_max) & (ccb < cb_max)

    event_counts = df.groupby("eventTankTime")["clusterTime"].transform("count")
    multi = df[event_counts > 1].copy()
    multi["first_cluster_time"] = multi.groupby("eventTankTime")["clusterTime"].transform("min")
    multi["delta_t"] = multi["clusterTime"] - multi["first_cluster_time"]
    delta_t_values = multi.loc[multi["delta_t"] > 0, "delta_t"]
    neutron_multiplicity = df["eventTankTime"].value_counts()

    # 1. Delta-t between first and subsequent clusters
    fig, ax = plt.subplots()
    ax.hist(delta_t_values, bins=50, color="coral", edgecolor="black")
    ax.set_title(ctx.title("Δt Between First and Subsequent Clusters"))
    ax.set_xlabel("Δt (μs)")
    ax.set_ylabel("Number of Subsequent Clusters")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    save_plot(fig, ctx, "delta_t_first_subsequent")

    # 2. Neutron multiplicity
    fig, ax = plt.subplots()
    ax.hist(neutron_multiplicity, bins=range(1, 10), edgecolor="blue",
            color="lightblue", linewidth=0.5, align="left")
    ax.set_xlabel("Neutron multiplicity")
    ax.set_ylabel("Counts")
    ax.set_title(ctx.title(f"AmBe Neutron Multiplicity (PE < {pe_max}, CCB < {cb_max})"))
    save_plot(fig, ctx, "neutron_multiplicity")

    # 3. Cluster PE vs Charge Balance
    fig, ax = plt.subplots()
    h = ax.hist2d(pe, ccb, bins=200, cmap="viridis",
                  range=[[-10, 500], [0.1, 1.0]], cmin=1)
    fig.colorbar(h[3], ax=ax, label="Counts")
    ax.set_title(ctx.title("Cluster PE vs Charge Balance"))
    ax.set_xlabel("Cluster PE")
    ax.set_ylabel("Cluster Charge Balance")
    save_plot(fig, ctx, "cluster_pe_vs_cb")

    # 4. Neutron capture-time fit (using fit_params from config)
    fp = ctx.fit_params
    bins = int(fp.get("time_bins", 200))
    trange = tuple(fp.get("time_range", (0, 70)))
    fit_lo = float(fp.get("fit_min_time", 2.0))
    fit_hi = float(fp.get("fit_max_time", 65.0))
    p0 = [
        float(fp.get("initial_amplitude", float(np.max(np.histogram(ct, bins=bins, range=trange)[0])))),
        float(fp.get("initial_thermal_time", 5.0)),
        float(fp.get("initial_capture_time", 25.0)),
        float(fp.get("initial_background", 0.0)),
    ]

    counts, bin_edges = np.histogram(ct, bins=bins, range=trange)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = (bin_centers > fit_lo) & (bin_centers < fit_hi)
    fx, fy = bin_centers[mask], counts[mask]
    fy_err = np.sqrt(np.maximum(fy, 1))

    popt, pcov = curve_fit(
        neut_capture, fx, fy, p0=p0, sigma=fy_err, absolute_sigma=True,
        bounds=([0, 0.1, 0.1, 0], [np.inf, 100, 100, np.inf]),
    )
    perr = np.sqrt(np.diag(pcov))
    fy_expected = neut_capture(fx, *popt)
    chi2_val = float(np.sum(((fy - fy_expected) ** 2) / (fy_err ** 2)))
    ndof = len(fy) - len(popt)
    chi2_ndof = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)

    fig, ax = plt.subplots()
    ax.hist(ct, bins=bins, range=trange, histtype="step", color="blue", label="Data")
    label = (
        fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
        fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
        fr"$\chi^2/\mathrm{{ndof}} = {chi2_ndof:.2f}$, p = {p_value:.3f}"
    )
    ax.plot(fx, fy_expected, "r-", linewidth=2, label=label)
    ax.set_xlabel(r"Cluster Time [$\mu s$]")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.set_title(ctx.title(f"Neutron Capture Time (PE < {pe_max}, CCB < {cb_max})"))
    save_plot(fig, ctx, "neutron_capture_time_fit")

    # 5. Fit residuals
    residuals = (fy - fy_expected) / fy_err
    fig, ax = plt.subplots()
    ax.plot(fx, residuals)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("Normalised Residual")
    ax.set_title(ctx.title("Capture-Time Fit Residuals"))
    save_plot(fig, ctx, "capture_time_fit_residuals")

    print(f"[plots.combined] chi2/ndof = {chi2_ndof:.2f}  p = {p_value:.3f}")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)


if __name__ == "__main__":
    raise SystemExit("Use: ambe plots combined --config <yaml>   (or python -m ambe plots combined ...)")
