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


def main_resid_axes(nrows: int = 1, ncols: int = 1, figsize=None, ratio=(3, 1), hspace: float = 0.05):
    """Figure with a main axis and a residual axis under it in every cell (shared x).
    Returns fig, main, resid; main and resid are (nrows, ncols) object arrays."""
    import numpy as np
    fig = plt.figure(figsize=figsize or (6.4 * ncols, 5.0 * nrows))
    outer = fig.add_gridspec(nrows, ncols)
    main = np.empty((nrows, ncols), dtype=object)
    resid = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            inner = outer[i, j].subgridspec(2, 1, height_ratios=ratio, hspace=hspace)
            main[i, j] = fig.add_subplot(inner[0])
            resid[i, j] = fig.add_subplot(inner[1], sharex=main[i, j])
            main[i, j].tick_params(labelbottom=False)
    return fig, main, resid


def _sym_ylim(ax, r, floor: float):
    import numpy as np
    r = np.asarray(r, float)
    m = np.nanmax(np.abs(r[np.isfinite(r)])) if np.isfinite(r).any() else floor
    m = max(floor, 1.15 * m)
    ax.set_ylim(-m, m)


def draw_pull(ax, x, a, sa, b, sb, label: str = "pull", floor: float = 3.0, lim: Optional[float] = None, **kw):
    """Residual (a - b) / sqrt(sa^2 + sb^2) as points + line; bins with zero error are left out. With lim the axis is
    fixed to +-lim and points beyond it are drawn at the edge. Returns the (unclipped) pull array."""
    import numpy as np
    a, sa, b, sb = (np.asarray(v, float) for v in (a, sa, b, sb))
    s = np.sqrt(sa ** 2 + sb ** 2)
    pull = np.where(s > 0, (a - b) / np.where(s > 0, s, 1.0), np.nan)
    kw.setdefault("color", "k")
    kw.setdefault("marker", "o")
    kw.setdefault("ms", 3.5)
    kw.setdefault("lw", 1.0)
    ax.plot(x, pull if lim is None else np.clip(pull, -lim, lim), **kw)
    ax.set_ylabel(label)
    ax.set_gid("residual")
    if lim is None:
        _sym_ylim(ax, pull, floor)
    else:
        ax.set_ylim(-1.05 * lim, 1.05 * lim)
    return pull


def draw_diff(ax, x, a, sa, b, sb, label: str = "difference", floor: float = 0.0, center_zero: bool = True, **kw):
    """Residual a - b with error sqrt(sa^2 + sb^2) as points + line; center_zero=False lets the axis autoscale
    (for an offset that never crosses zero). Returns (diff, err)."""
    import numpy as np
    a, sa, b, sb = (np.asarray(v, float) for v in (a, sa, b, sb))
    d, e = a - b, np.sqrt(sa ** 2 + sb ** 2)
    kw.setdefault("color", "k")
    kw.setdefault("marker", "o")
    kw.setdefault("ms", 3.5)
    kw.setdefault("lw", 1.0)
    kw.setdefault("capsize", 0)
    ax.errorbar(x, d, yerr=e, **kw)
    ax.set_ylabel(label)
    ax.set_gid("residual")
    if center_zero:
        lim = np.concatenate([np.abs(d + e), np.abs(d - e)])
        _sym_ylim(ax, lim / 1.15, floor if floor > 0 else 1e-12)
    return d, e


def draw_ratio(ax, x, a, sa, b, sb=None, label: str = "data / MC", floor: float = 0.1, cap: float = 1.0,
               logy: bool = False, **kw):
    """Residual a / b with error sqrt((sa/b)^2 + (a sb/b^2)^2) as small points with error bars; bins with b <= 0 are
    left out. Linear axis: symmetric about 1, half-width between floor and cap (points beyond are drawn at the edge).
    Returns the ratio array."""
    import numpy as np
    a, sa, b = (np.asarray(v, float) for v in (a, sa, b))
    sb = np.zeros_like(b) if sb is None else np.asarray(sb, float)
    ok = b > 0
    bb = np.where(ok, b, 1.0)
    r = np.where(ok, a / bb, np.nan)
    e = np.where(ok, np.sqrt((sa / bb) ** 2 + (a * sb / bb ** 2) ** 2), np.nan)
    for k, v in (("color", "k"), ("fmt", "o"), ("ms", 2.2), ("lw", 0.8), ("capsize", 0)):
        kw.setdefault(k, v)
    ax.set_ylabel(label)
    ax.set_gid("residual")
    if logy:
        ax.errorbar(x, np.where(r > 0, r, np.nan), e, **kw)
        ax.set_yscale("log")
        return r
    fin = np.isfinite(r) & np.isfinite(e)
    m = float(np.clip(1.15 * np.max(np.abs(r[fin] - 1) + e[fin]), floor, cap)) if fin.any() else floor
    ax.errorbar(x, np.clip(r, 1 - m, 1 + m), e, **kw)
    ax.set_ylim(1 - 1.05 * m, 1 + 1.05 * m)
    return r


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
