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
METHOD_COLORS = {"clusterfinder": "#2a78d6", "clusterfinder_nms": "#4a3aa7", "optics": "#eda100",
                 "optics_nms": "#eb6834", "peak": "#1baf7a", "peak_tof": "#008300"}


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


def two_titles(fig, simple: str, detail: str, rect=(0, 0, 1, 1), simple_size: float = 14,
               detail_size: float = 9) -> float:
    """Detailed context on top, a plain title under it; cropping the top strip leaves a clean figure.
    Lays the figure out below both and returns the top of the axes area (figure fraction)."""
    import textwrap
    from matplotlib.transforms import offset_copy
    w_in, h_in = fig.get_size_inches()
    width = max(40, int(w_in * 72 / (0.52 * detail_size)))
    lines = [x for line in detail.split("\n") for x in (textwrap.wrap(line, width) or [""])]
    d_pts = len(lines) * detail_size * 1.3
    tr = lambda dy: offset_copy(fig.transFigure, fig=fig, y=-dy, units="points")  # noqa: E731
    fig.text(0.5, 1.0, "\n".join(lines), ha="center", va="top", fontsize=detail_size, color="0.3", transform=tr(4))
    fig.text(0.5, 1.0, simple, ha="center", va="top", fontsize=simple_size, transform=tr(10 + d_pts))
    top = min(rect[3], 1 - (16 + d_pts + 1.35 * simple_size) / 72 / h_in)
    fig.tight_layout(rect=(rect[0], rect[1], rect[2], top))
    return top


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
