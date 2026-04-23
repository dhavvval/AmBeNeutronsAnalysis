"""
One-shot pipeline orchestrators. Called by `ambe pipeline <name> --config ...`.

Each orchestrator runs the full DAG for one workflow:
  - pipeline mc    : MC ROOT -> parquet -> OPTICS training + ClusterFinder baseline
  - pipeline data  : data CSV -> plots + stats
"""

from __future__ import annotations

from typing import Iterable, Optional

from .context import RunContext


def run_mc(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    """Full MC pipeline: processor -> optics."""
    from .mc import processor as mc_processor
    from .mc import optics as mc_optics

    mc_processor.run(ctx)
    mc_optics.run(ctx)


def run_data(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    """Full data pipeline: (processor -> basic + combined + heatmap + stats).

    Individual stages can also be invoked on their own; this is just a driver.
    """
    # Stages imported lazily so a missing optional dep (e.g. pymc) only
    # breaks the stage that needs it, not the whole pipeline.
    from .data import processor as data_processor
    data_processor.run(ctx)

    try:
        from .plots import basic as basic_plots
        basic_plots.run(ctx)
    except ImportError as e:
        print(f"[pipeline] plots.basic skipped: {e}")

    try:
        from .plots import combined as combined_plots
        combined_plots.run(ctx)
    except ImportError as e:
        print(f"[pipeline] plots.combined skipped: {e}")

    try:
        from .plots import heatmap as heatmap_plots
        heatmap_plots.run(ctx)
    except ImportError as e:
        print(f"[pipeline] plots.heatmap skipped: {e}")

    try:
        from .stats import core as stats_core
        stats_core.run(ctx)
    except ImportError as e:
        print(f"[pipeline] stats.core skipped: {e}")
