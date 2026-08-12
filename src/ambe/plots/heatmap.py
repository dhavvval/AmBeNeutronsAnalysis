"""
AmBe neutron efficiency heatmaps.

Produces three plots:
  1. Statistics per port/y-position (normalised to max)
  2. Efficiency heatmap with per-cell SE
  3. Residual heatmap vs a reference campaign (default: AmBe 1.0)

Uses RunContext for input paths, output paths, titles and cuts.

Entry:  ambe plots heatmap --config configs/<something>.yaml
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..context import RunContext
from ..io import inputs_from_ctx, load_csvs
from ..plotting import save_plot, set_style


# Source-port lookup: (x,y,z) -> port label. Physics geometry, not per-run.
PORT_INFO = {
    (0, 100, 0): "Port 5", (0, 50, 0): "Port 5", (0, 0, 0): "Port 5",
    (0, -50, 0): "Port 5", (0, -100, 0): "Port 5", (0, 55.3, 0): "Port 5",

    (0, 100, -75): "Port 1", (0, 50, -75): "Port 1", (0, 0, -75): "Port 1",
    (0, -50, -75): "Port 1", (0, -100, -75): "Port 1",

    (75, 100, 0): "Port 4", (75, 50, 0): "Port 4", (75, 0, 0): "Port 4",
    (75, -50, 0): "Port 4", (75, -100, 0): "Port 4",

    (0, 100, 102): "Port 3", (0, 50, 102): "Port 3", (0, 0, 102): "Port 3",
    (0, -50, 102): "Port 3", (0, -100, 102): "Port 3", (0, -60, 102): "Port 3",

    (0, 100, 75): "Port 2", (0, 50, 75): "Port 2", (0, 0, 75): "Port 2",
    (0, -50, 75): "Port 2", (0, -100, 75): "Port 2",
}

PORT_ORDER = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

# Reference data for residuals. Keyed by campaign name.
REFERENCE_DATA = {
    "AmBe 1.0": {
        "y_positions": [100, 50, 0, -50, -100],
        "by_port": {
            "Port 1": [61, 71, 54, 58, 57],
            "Port 5": [71, 69, 69, 66, 65],
            "Port 2": [56, 67, 69, 64, 58],
            "Port 3": [57, 61, 61, 61, 54],
            "Port 4": [68, 67, None, 69, 67],
        },
    },
    # AmBe 2.0v2 was commented out in the legacy script; preserved here.
    "AmBe 2.0v2": {
        "y_positions": [100, 50, 0, -50, -100],
        "by_port": {
            "Port 1": [61.86, 64.44, 65.20, 59.64, 54.01],
            "Port 5": [70.34, 71.96, 70.02, 63.06, 47.16],
            "Port 2": [62.37, None, 65.02, 62.29, None],
            "Port 3": [None, 54.73, 53.18, 50.13, 38.32],
            "Port 4": [60.79, 60.30, 64.43, 60.32, 40.16],
        },
    },
}


def _load_and_prepare(ctx: RunContext) -> pd.DataFrame:
    csvs = inputs_from_ctx(ctx, "trigger_summary")
    df = load_csvs(csvs)

    # Historical position fix-up preserved from legacy code
    fix = (df["x_pos"] == 0) & (df["y_pos"] == -105.5) & (df["z_pos"] == 102)
    if fix.any():
        df.loc[fix, "y_pos"] = -100

    # Add missing rows so every (x,y,z) in PORT_INFO has a row
    for pos in PORT_INFO.keys():
        missing = not ((df["x_pos"] == pos[0]) &
                       (df["y_pos"] == pos[1]) &
                       (df["z_pos"] == pos[2])).any()
        if missing:
            df = pd.concat([df, pd.DataFrame([{
                "x_pos": pos[0], "y_pos": pos[1], "z_pos": pos[2],
                "neutron_candidates": 0, "ambe_triggers": 1, "total_events": 1,
                "unique_neutron_triggers": 0,
            }])], ignore_index=True)

    df["SourcePosition"] = list(zip(df["x_pos"], df["y_pos"], df["z_pos"]))
    df["port"] = df["SourcePosition"].map(lambda p: PORT_INFO.get(p, "Unknown"))

    df["efficiency"] = df["unique_neutron_triggers"] / df["ambe_triggers"]
    df["err_A"] = np.sqrt(df["unique_neutron_triggers"]) / df["ambe_triggers"]
    df["err_B"] = np.sqrt(df["efficiency"] * (1 - df["efficiency"]) / df["ambe_triggers"])

    for col in ("efficiency", "err_A", "err_B"):
        df[col] = df[col] * 100

    max_n = df["unique_neutron_triggers"].max() or 1
    df["unique_neutron_triggers"] = (df["unique_neutron_triggers"] / max_n) * 100

    return df


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots heatmap")
    p.parse_args(list(argv) if argv else [])
    set_style()

    df = _load_and_prepare(ctx)

    pivot_eff = df.pivot(index="y_pos", columns="port", values="efficiency").reindex(columns=PORT_ORDER)
    pivot_err = df.pivot(index="y_pos", columns="port", values="err_B").reindex(columns=PORT_ORDER)
    pivot_n = df.pivot(index="y_pos", columns="port", values="unique_neutron_triggers").reindex(columns=PORT_ORDER)
    pivot_counts = pivot_n.round(2).dropna(how="all")
    pivot_eff = pivot_eff.loc[pivot_counts.index]
    pivot_err = pivot_err.loc[pivot_counts.index]
    pivot_n = pivot_n.loc[pivot_counts.index]
    mask = (pivot_n == 0)

    pe_max = ctx.cuts.get("pe_max", "—")
    cb_max = ctx.cuts.get("charge_balance_max", "—")
    cut_note = f"(PE < {pe_max}, CB < {cb_max})"

    # 1. Statistics heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_counts, annot=True, fmt="", cmap="YlOrBr", cbar=True,
                annot_kws={"size": 12}, mask=mask, linecolor="black",
                linewidths=0.2, cbar_kws={"label": "Percentage (%)"}, ax=ax)
    ax.set_title(ctx.title(f"Neutron Statistics {cut_note}"))
    ax.set_xlabel("Ports"); ax.set_ylabel("Y Position (cm)")
    ax.invert_yaxis()
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    save_plot(fig, ctx, "statistics_heatmap")

    # 2. Efficiency heatmap
    def make_label(eff, err, n):
        if pd.isna(eff) or n == 0:
            return "empty"
        return f"{eff:.2f}${{\\pm{err:.2f}}}$"
    vlabel = np.vectorize(make_label)
    labels_se = vlabel(pivot_eff.values, pivot_err.values, pivot_n.values)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_eff, annot=labels_se, fmt="", cmap="YlOrBr", cbar=True,
                annot_kws={"size": 12}, mask=mask, linecolor="black",
                linewidths=0.2, cbar_kws={"label": "Efficiency (%)"}, ax=ax)
    ax.set_title(ctx.title(f"Neutron Efficiency {cut_note}"))
    ax.set_xlabel("Ports"); ax.set_ylabel("Y Position (cm)")
    ax.invert_yaxis()
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    save_plot(fig, ctx, "efficiency_heatmap")

    # 3. Residuals heatmap -- compare against configured reference campaign
    ref_name = ctx.extra.get("reference", {}).get("comparison_campaign", "AmBe 1.0")
    if ref_name not in REFERENCE_DATA:
        print(f"[plots.heatmap] reference '{ref_name}' unknown; "
              f"available: {list(REFERENCE_DATA)}; skipping residuals")
        return
    ref = REFERENCE_DATA[ref_name]
    ref_df = pd.DataFrame(ref["by_port"], index=ref["y_positions"])
    residuals = pivot_eff.reindex(index=ref_df.index, columns=PORT_ORDER) - ref_df

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(residuals, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                cbar_kws={"label": f"Residual ({ctx.campaign} − {ref_name})"},
                mask=residuals.isna(), linecolor="black", linewidths=0.2, ax=ax)
    ax.set_title(ctx.title(f"Residual Efficiency vs {ref_name} {cut_note}"))
    ax.set_xlabel("Ports"); ax.set_ylabel("Y Position (cm)")
    ax.invert_yaxis()
    plt.xticks(rotation=45)
    save_plot(fig, ctx, "residual_efficiency_heatmap")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)


if __name__ == "__main__":
    raise SystemExit("Use: ambe plots heatmap --config <yaml>")
