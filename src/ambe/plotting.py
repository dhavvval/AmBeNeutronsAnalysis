"""
Plotting helpers shared by every plots.* script.

Key function is `save_plot` -- it uses the RunContext to derive the full
output path and consistent filename, so individual scripts never have to
construct paths manually.

Usage:
    from ambe.plotting import save_plot, set_style

    set_style()                                   # once per script
    fig, ax = plt.subplots()
    ax.plot(...)
    ax.set_title(ctx.title("Cluster PE"))
    save_plot(fig, ctx, "cluster_pe")              # writes PDF + PNG
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from .context import RunContext


DEFAULT_FORMATS: tuple[str, ...] = ("pdf", "png")


def set_style():
    """One-liner called at the top of a plotting script for consistency."""
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
    })


def save_plot(
    fig,
    ctx: RunContext,
    base_name: str,
    formats: Optional[Iterable[str]] = None,
    subdir: Optional[str] = None,
    close: bool = True,
) -> list[Path]:
    """
    Save `fig` under the context's plots dir, once per format in `formats`.

    Parameters
    ----------
    fig        : matplotlib Figure
    ctx        : RunContext
    base_name  : short slug; campaign/run are appended automatically
    formats    : iterable of extensions (default: pdf + png)
    subdir     : optional subdirectory under plots/ (e.g. "diagnostics/")
    close      : if True, close the figure after saving (default True)

    Returns the list of paths written.
    """
    formats = tuple(formats) if formats else DEFAULT_FORMATS
    out_paths = []
    for ext in formats:
        target_dir = ctx.plots_dir
        if subdir:
            target_dir = target_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / ctx.filename(base_name, ext)
        fig.savefig(path)
        out_paths.append(path)
    if close:
        plt.close(fig)
    return out_paths
