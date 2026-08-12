#!/usr/bin/env python
"""
Side-by-side efficiency heatmaps by port x y-position: the IC-only baseline,
the fprompt2d widened window, and the per-cell difference.

`ambe plots heatmap` already draws the single-tag panels (and is what produced
efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_{gated,fprompt}), but its residual
panel can only compare against the hardcoded REFERENCE_DATA campaigns
(AmBe 1.0 / 2.0v2). The comparison that matters for the fprompt change is
fprompt2d minus the v4 IC baseline, cell by cell, which is what this adds.

Geometry (PORT_INFO / PORT_ORDER) and the pivot are imported from
ambe.plots.heatmap rather than restated, so the cells line up with the
single-tag plots exactly.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    MPLBACKEND=Agg python -u plot_efficiency_heatmap_fprompt_vs_baseline.py
"""
import sys

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ambe.plots.heatmap import PORT_INFO, PORT_ORDER
from ambe.plotting import set_style

SUMMARY = "TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_{tag}.csv"
OUTDIR = "verbose"
OUTBASE = "efficiency_heatmap_fprompt_vs_baseline"
# run 6242's input grew between the two passes (REPORT §7.5), so its cell is
# not a like-for-like comparison -- annotate it rather than silently drop it
GREW = (0, 50, 75)


def prepare(tag):
    """Efficiency (%) and binomial SE per (port, y), matching heatmap.py."""
    df = pd.read_csv(SUMMARY.format(tag=tag))
    fix = (df.x_pos == 0) & (df.y_pos == -105.5) & (df.z_pos == 102)
    df.loc[fix, "y_pos"] = -100
    df["SourcePosition"] = list(zip(df.x_pos, df.y_pos, df.z_pos))
    df["port"] = df.SourcePosition.map(lambda p: PORT_INFO.get(p, "Unknown"))
    df["efficiency"] = df.unique_neutron_triggers / df.ambe_triggers * 100
    df["err"] = np.sqrt(df.efficiency / 100 * (1 - df.efficiency / 100)
                        / df.ambe_triggers) * 100
    piv = lambda c: df.pivot(index="y_pos", columns="port", values=c).reindex(columns=PORT_ORDER)
    return piv("efficiency"), piv("err"), df


def main():
    set_style()
    eb, sb, dfb = prepare("gated")
    ef, sf, dff = prepare("fprompt")

    # keep only rows that carry data in both tags
    # ascending, then invert_yaxis() below -- same order heatmap.py produces,
    # so +100 ends up at the top as in every previous efficiency plot
    rows = sorted(set(eb.dropna(how="all").index) & set(ef.dropna(how="all").index))
    eb, sb, ef, sf = (t.loc[rows] for t in (eb, sb, ef, sf))
    delta = ef - eb
    mask = eb.isna() | ef.isna()

    grew_port = PORT_INFO.get(GREW)
    grew_y = GREW[1]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # --- panels 1 & 2: sequential, shared scale so the two are comparable ---
    vmin = float(min(eb.min().min(), ef.min().min()))
    vmax = float(max(eb.max().max(), ef.max().max()))
    for ax, eff, se, title in (
            (axes[0], eb, sb, "IC 700-1200 (baseline)"),
            (axes[1], ef, sf, "fprompt2d 500-2000 x 0.15-0.30")):
        def make_label(e, s):
            return "empty" if pd.isna(e) else f"{e:.2f}\n$\\pm${s:.2f}"
        labels = np.vectorize(make_label)(eff.values, se.values)
        sns.heatmap(eff, annot=labels, fmt="", cmap="YlOrBr", vmin=vmin, vmax=vmax,
                    annot_kws={"size": 11}, mask=mask, linecolor="black",
                    linewidths=0.2, cbar_kws={"label": "Efficiency (%)"}, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Ports")
        ax.set_ylabel("Y Position (cm)")

    # --- panel 3: diverging, two hues about a neutral zero ---
    # symmetric limits so 0 sits exactly on the neutral midpoint; every cell
    # here is negative, and a 0-centred diverging map is what shows that
    lim = float(np.nanmax(np.abs(delta.values)))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-lim, vmax=lim, annot_kws={"size": 11}, mask=mask,
                linecolor="black", linewidths=0.2,
                cbar_kws={"label": "$\\Delta$ efficiency (pp)"}, ax=axes[2])
    axes[2].set_title("fprompt2d $-$ baseline (pp)")
    axes[2].set_xlabel("Ports")
    axes[2].set_ylabel("Y Position (cm)")

    for ax in axes:
        ax.invert_yaxis()
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        # flag the one cell whose input data differs between the two passes
        if grew_port in PORT_ORDER and grew_y in list(eb.index):
            ax.add_patch(plt.Rectangle(
                (PORT_ORDER.index(grew_port), list(eb.index).index(grew_y)),
                1, 1, fill=False, edgecolor="#1a1a1a", lw=2.4, ls=":"))

    pooled_b = dfb.unique_neutron_triggers.sum() / dfb.ambe_triggers.sum() * 100
    pooled_f = dff.unique_neutron_triggers.sum() / dff.ambe_triggers.sum() * 100
    fig.suptitle(
        f"AmBe 2.0v4 neutron efficiency by port and Y position  (PE < 100, CB < 0.45)   "
        f"pooled {pooled_b:.2f}% $\\rightarrow$ {pooled_f:.2f}%   "
        f"[dotted cell: run 6242, input grew between passes]",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ("pdf", "png"):
        p = f"{OUTDIR}/{OUTBASE}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)

    # the table, for reading in a terminal
    print("\nEfficiency (%) by port x Y, and the change:")
    out = pd.concat({"baseline": eb.round(2), "fprompt2d": ef.round(2),
                     "delta_pp": delta.round(2)}, axis=1)
    print(out.to_string())


if __name__ == "__main__":
    sys.exit(main())
