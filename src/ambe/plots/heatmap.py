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
import textwrap
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
    # AmBe 2.0v4 -- the most recent campaign, and so the right default comparison
    # for anything newer. Unlike the two blocks above, these are not transcribed
    # from a legacy script: they are unique_neutron_triggers / ambe_triggers read
    # straight out of the frozen
    #   TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_all28_box.csv
    # i.e. the 28-run box-cut reproduction -- SAME box cuts, SAME 700-1200 Stage-1
    # IC gate, same code path. That is what makes a residual against it a like-for-
    # like measurement of the campaign difference rather than a mix of campaign and
    # selection differences.
    # All 25 grid cells are covered. The 26th v4 position (Port 3, y = -60, run
    # 6187, 46.99 %) is off the standard y grid and so has no cell here.
    "AmBe 2.0v4": {
        "y_positions": [100, 50, 0, -50, -100],
        "by_port": {
            "Port 1": [54.1, 62.6, 64.1, 61.8, 50.5],
            "Port 5": [63.0, 69.5, 71.8, 69.5, 63.0],
            "Port 2": [54.1, 61.9, 63.2, 60.8, 52.4],
            "Port 3": [40.9, 49.4, 50.8, 49.0, 40.3],
            "Port 4": [54.5, 62.8, 65.3, 60.6, 49.7],
        },
    },
}


# ---------------------------------------------------------------------------
# One annotation contract for every port x y map in the campaign.
#
# There are two independent code paths that draw these maps -- this module (the
# efficiency / statistics / residual maps) and boxcut_v4_campaign._heat (the capture
# time and thermalisation time maps) -- and they are meant to be put side by side on a
# slide. They had drifted: this module printed "NN.NN${\pm N.NN}$" at size 12 while
# _heat printed "NN.NN$_{\pm N.NN}$", a real mathtext SUBSCRIPT, at size 11, so the
# error rendered smaller on one map than the other. These three names are the shared
# definition; _heat imports them rather than restating them.
#
# The literal "±" replaces the mathtext wrapper on purpose: mathtext width is not
# predictable from the character count, and the cells are now square and tight.
HEAT_ANNOT_SIZE = 14
# Helvetica metrics, matching make_plots_ccinc_v3_merged's rcParams. Kept to faces
# that are actually installed here -- the fuller fallback list in that module emits a
# findfont warning per missing name per text object, which is 25 warnings a map.
HEAT_ANNOT_FAMILY = ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
HEAT_TITLE_SIZE = 14
HEAT_LABEL_SIZE = 13      # axis labels
HEAT_TICK_SIZE = 12       # port names and y positions


def heat_label(value: float, err: float) -> str:
    """The cell string, one line, identical on every map."""
    return f"{value:.2f}±{err:.2f}"


def heat_annot_kws(**extra):
    return {"size": HEAT_ANNOT_SIZE, "fontfamily": HEAT_ANNOT_FAMILY, **extra}


def _figsize(pivot, per_col: float = 1.72, per_row: float = 1.34):
    """Figure size scaled to the grid, sized for a near-SQUARE OVERALL figure.

    Three iterations, and the difference between them matters:

      per_col=1.75-2.25 / per_row=0.78 -- the original. Cells more than twice as wide
        as tall, so the whole figure was a 14.7 x 5.9 in letterbox.
      per_col=per_row=1.30 + ax.set_box_aspect() -- forced square CELLS. That made the
        figure square-ish but squeezed the cell to ~1.3 in, which forced the label
        down to 9 pt to fit.
      what is here now -- the OVERALL FIGURE is near square (~1.1:1 for a 5 x 6 grid)
        and the cell is left a little wider than tall, which is what gives
        "NN.NN±N.NN" room at HEAT_ANNOT_SIZE. per_col is set by that label: it
        has to stay wider than the widest cell string at the current font, so
        raising HEAT_ANNOT_SIZE means raising per_col with it.

    Cells are NOT forced square any more, on purpose: cell squareness and a readable
    label are in direct competition here, and the label wins. The additive terms are
    the y-axis labels and the colourbar, which do not scale with the grid.
    """
    nrow, ncol = pivot.shape
    return (per_col * ncol + 2.7, per_row * nrow + 1.7)


def short_cut(ctx) -> str:
    """The selection in a few words, for a ONE-LINE title.

    ctx.cuts["selection_label"] is a full sentence -- it exists so a slide can be read
    without the talk -- but pasted into a title on a near-square canvas it wraps to
    three lines and crowds the map off the page. Decks 2 and 3 differ only in this
    string, so it cannot simply be dropped: without it the two maps are indis-
    tinguishable. Keep the discriminating part, drop the prose.
    """
    lbl = str(ctx.cuts.get("selection_label", ""))
    if lbl.lower().startswith("mva"):
        return lbl
    pe = ctx.cuts.get("pe_max", "—")
    cb = ctx.cuts.get("charge_balance_max", "—")
    return f"PE ≤ {pe}, CB < {cb}"


def heat_title(text: str, width: int = 78) -> str:
    """Wrap a title, for the rare one still long enough to need it."""
    return "\n".join(w for line in text.split("\n")
                     for w in textwrap.wrap(line, width) or [""])


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

    # Normalised to the BUSIEST CELL OF THIS SAMPLE, not to a fixed reference. So
    # "100 %" means a different absolute number in every campaign and under every
    # selection -- 27,948 neutron triggers at (0,-100,-75) under the box cuts,
    # 21,616 at the same position under the MVA -- and two statistics maps from
    # different selections are NOT comparable cell for cell even though both top out
    # at 100. The absolute count is carried out with the frame and printed into the
    # title by run(), because a percentage whose denominator is invisible is exactly
    # the sort of thing that gets read as an exposure ratio between decks.
    max_n = df["unique_neutron_triggers"].max() or 1
    imax = df["unique_neutron_triggers"].idxmax()
    df.attrs["stats_ref_n"] = int(max_n)
    df.attrs["stats_ref_pos"] = (df.loc[imax, "x_pos"], df.loc[imax, "y_pos"],
                                 df.loc[imax, "z_pos"])
    df.attrs["stats_ref_port"] = df.loc[imax, "port"]
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

    # Drop y rows that carry no data in ANY port. dropna() above is not enough:
    # _load_and_prepare() back-fills a row for every PORT_INFO position missing from
    # the data with unique_neutron_triggers=0, so an unvisited y (e.g. the y=55.3
    # Port 5 entry, which no run in any campaign has occupied) arrives as 0 rather
    # than NaN, survives dropna, and draws as a fully masked band of "empty" cells.
    # Test on n == 0 as well as NaN so any unvisited row disappears on its own.
    empty_row = pivot_counts.isna() | (pivot_counts == 0)
    keep = pivot_counts.index[~empty_row.all(axis=1)]
    dropped = [y for y in pivot_counts.index if y not in set(keep)]
    if dropped:
        print(f"[plots.heatmap] dropping y row(s) with no data in any port: "
              f"{', '.join(f'{y:g}' for y in dropped)}")
    pivot_counts = pivot_counts.loc[keep]
    pivot_eff = pivot_eff.loc[keep]
    pivot_err = pivot_err.loc[keep]
    pivot_n = pivot_n.loc[keep]
    mask = (pivot_n == 0)

    # Name the selection literally in the title, so a slide can be read without the
    # talk. A config may state it outright (selection_label) -- which is how the MVA
    # config does it, having no PE/CB box at all -- otherwise spell out the box from
    # the cuts block. Note the PE bound is inclusive: ambe_single_cut in
    # data/processor.py tests `pe_min < cpe <= pe_max`, so "PE < 100" was wrong.
    short = short_cut(ctx)
    cut_note = ctx.cuts.get("selection_label")
    if not cut_note:
        pe_max = ctx.cuts.get("pe_max", "—")
        cb_max = ctx.cuts.get("charge_balance_max", "—")
        cut_note = (f"box cuts PE ≤ {pe_max}, CB < {cb_max}, "
                    f"t ≥ 2 µs, hits ≥ 5")

    # 1. Statistics heatmap
    ref_n = df.attrs.get("stats_ref_n")
    ref_port = df.attrs.get("stats_ref_port")
    ref_pos = df.attrs.get("stats_ref_pos")
    ref_note = ""
    if ref_n:
        ref_note = (f" — 100 % = {ref_n:,} neutron triggers at {ref_port}, "
                    f"y = {ref_pos[1]:g} cm")
    fig, ax = plt.subplots(figsize=_figsize(pivot_counts))
    sns.heatmap(pivot_counts, annot=True, fmt="", cmap="YlOrBr", cbar=True,
                annot_kws=heat_annot_kws(), mask=mask, linecolor="black",
                linewidths=0.2,
                cbar_kws={"label": "Percentage of the busiest cell (%)"}, ax=ax)
    # The absolute reference used to be spelled into the title, because without it
    # this map cannot be compared with the same map from another selection. It is now
    # the colourbar label instead: the title has to stay one line, and "100 % =" is
    # a statement about the colour scale, which is where a reader looks for it.
    if ref_n:
        ax.collections[0].colorbar.set_label(
            f"% of the busiest cell ({ref_n:,} neutron triggers, "
            f"{ref_port}, y = {ref_pos[1]:g} cm)")
    ax.set_title(f"Neutron statistics for {ctx.campaign} by port position "
                 f"({short}) ", fontsize=HEAT_TITLE_SIZE)
    ax.set_xlabel("Ports", fontsize=HEAT_LABEL_SIZE)
    ax.set_ylabel("Y Position (cm)", fontsize=HEAT_LABEL_SIZE)
    ax.tick_params(labelsize=HEAT_TICK_SIZE)
    ax.invert_yaxis()
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    save_plot(fig, ctx, "statistics_heatmap")

    # 2. Efficiency heatmap
    def make_label(eff, err, n):
        if pd.isna(eff) or n == 0:
            return "empty"
        return heat_label(eff, err)
    vlabel = np.vectorize(make_label)
    labels_se = vlabel(pivot_eff.values, pivot_err.values, pivot_n.values)

    # Same geometry as the other two now. The old per_col=2.25 widened this map alone
    # because its cell text carries a value AND its error; with the shared, smaller
    # HEAT_ANNOT_SIZE the label fits a square cell, and all four maps in decks 2 and 3
    # can be laid out on one slide without one of them being twice as wide.
    fig, ax = plt.subplots(figsize=_figsize(pivot_eff))
    sns.heatmap(pivot_eff, annot=labels_se, fmt="", cmap="YlOrBr", cbar=True,
                annot_kws=heat_annot_kws(), mask=mask, linecolor="black",
                linewidths=0.2, cbar_kws={"label": "Efficiency (%)"}, ax=ax)
    ax.set_title(f"Neutron detection efficiency for {ctx.campaign} by port position "
                 f"({short})", fontsize=HEAT_TITLE_SIZE)
    ax.set_xlabel("Ports", fontsize=HEAT_LABEL_SIZE)
    ax.set_ylabel("Y Position (cm)", fontsize=HEAT_LABEL_SIZE)
    ax.tick_params(labelsize=HEAT_TICK_SIZE)
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

    fig, ax = plt.subplots(figsize=_figsize(residuals))
    sns.heatmap(residuals, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                annot_kws=heat_annot_kws(),
                cbar_kws={"label": f"Residual ({ctx.campaign} − {ref_name})"},
                mask=residuals.isna(), linecolor="black", linewidths=0.2, ax=ax)
    ax.set_title(f"Efficiency residual, {ctx.campaign} − {ref_name}, by port position "
                 f"({short})", fontsize=HEAT_TITLE_SIZE)
    ax.set_xlabel("Ports", fontsize=HEAT_LABEL_SIZE)
    ax.set_ylabel("Y Position (cm)", fontsize=HEAT_LABEL_SIZE)
    ax.tick_params(labelsize=HEAT_TICK_SIZE)
    ax.invert_yaxis()
    plt.xticks(rotation=45)
    save_plot(fig, ctx, "residual_efficiency_heatmap")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)


if __name__ == "__main__":
    raise SystemExit("Use: ambe plots heatmap --config <yaml>")
