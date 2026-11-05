"""
`ambe` command-line entry point.

After `pip install -e .`, a user can run:

    ambe --help
    ambe mc process --config configs/mc_lucho_full.yaml
    ambe mc optics  --config configs/mc_lucho_full.yaml
    ambe plots basic --config configs/data_ambe2v4_all28_box.yaml
    ambe leak features --config configs/data_ambe2v5sandi_box.yaml
    ambe pipeline mc --config configs/mc_lucho_full.yaml

Each subcommand dispatches to a function in the corresponding module that
accepts (ctx: RunContext, args: argparse.Namespace). Subcommands are registered
lazily so that importing ambe.cli does not import heavy dependencies for
commands the user isn't running.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable

from .context import RunContext, load_context


# --------------------------------------------------------------------------- #
# Dispatch table: (group, command) -> "module.path:function"
# --------------------------------------------------------------------------- #
COMMANDS: dict[tuple[str, str], str] = {
    # group, cmd                 where to find it
    ("mc", "process"):          "ambe.mc.processor:cli",
    ("mc", "cc"):               "ambe.mc.cc_selection:cli",
    ("mc", "optics"):           "ambe.mc.optics:cli",
    ("mc", "features"):         "ambe.mc.cluster_features:cli",
    ("mc", "discriminate"):     "ambe.mc.cluster_discrimination:cli",
    ("mc", "match"):            "ambe.mc.cluster_matching:cli",
    ("mc", "timespread"):       "ambe.mc.time_spread:cli",
    ("mc", "hitcomp"):          "ambe.mc.hit_accounting:cli",
    ("mc", "timewindows"):      "ambe.mc.hit_time_windows:cli",
    ("mc", "capture"):          "ambe.mc.capture_truth:cli",
    ("mc", "trackwindows"):     "ambe.mc.track_windows:cli",
    ("mc", "displays"):         "ambe.diagnostics.plot_cluster_event_displays:cli",
    ("data", "process"):        "ambe.data.processor:cli",
    # Same Stage 2 as `data process`, with no Stage 1. MC has no waveforms, so every
    # cluster is treated as having passed the good-waveform selection. Separate command
    # rather than a flag so neither config kind can be run through the wrong path.
    ("data", "processmc"):      "ambe.data.processor:cli_mc",
    ("data", "eff"):            "ambe.data.processor:cli",
    ("data", "icscan"):         "ambe.plots.icscan:cli",
    ("data", "runtable"):       "ambe.diagnostics.runtable:cli",
    ("data", "randoms"):        "ambe.data.randoms:cli",
    ("leak", "features"):       "ambe.leak.features:cli",
    ("leak", "pmtcharge"):      "ambe.leak.pmtcharge:cli",
    ("plots", "icscan"):        "ambe.plots.icscan:cli",
    ("diagnostics", "runtable"): "ambe.diagnostics.runtable:cli",
    ("mc", "mva"):             "ambe.mc.mva_analysis:cli",
    ("plots", "basic"):         "ambe.plots.basic:cli",
    ("plots", "combined"):      "ambe.plots.combined:cli",
    ("plots", "heatmap"):       "ambe.plots.heatmap:cli",
    ("plots", "provenance"):    "ambe.plots.provenance_deck:cli",
    ("plots", "clustertruth"):  "ambe.plots.cluster_truth:cli",
    ("plots", "windowcompare"): "ambe.plots.window_compare:cli",
    ("plots", "efficiency"):    "ambe.efficiency_extraction.efficiency_thesis:cli",
    # Waveform-stage companion to `plots efficiency`: the Stage-1 AmBe small-PMT
    # selection, and whether the no-source run's flat background can be subtracted
    # there rather than in the multiplicity fit.
    ("plots", "waveformcuts"):  "ambe.plots.waveform_cuts:cli",
    ("stats", "efficiency"):    "ambe.efficiency_extraction.efficiency_fit:cli",
    ("stats", "clusterindex"):  "ambe.efficiency_extraction.recluster_index:cli",
    ("stats", "clustereff"):    "ambe.efficiency_extraction.recluster_index:cli_fit",
    ("stats", "foldeff"):       "ambe.efficiency_extraction.recluster_index:cli_foldeff",
    ("plots", "datamc"):        "ambe.plots.datamc:cli",
    ("plots", "datamcclust"):   "ambe.plots.datamc_clustering:cli",
    ("plots", "datamcneutron"): "ambe.plots.datamc_clustering:cli_neutron",
    ("plots", "datamcwave"):    "ambe.plots.datamc_waveform:cli",
    ("plots", "datamczero"):    "ambe.plots.datamc_zerobin:cli",
    ("plots", "clustereff"):    "ambe.plots.clustereff:cli",
    ("pipeline", "mc"):         "ambe.pipeline:run_mc",
    ("pipeline", "data"):       "ambe.pipeline:run_data",
}


def _load(dotted: str) -> Callable:
    mod_name, func_name = dotted.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ambe",
        description="ANNIE AmBe neutron analysis toolkit.",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # Group -> command parsers, all accepting --config as the primary argument
    groups = {}
    for (g, c), _ in COMMANDS.items():
        if g not in groups:
            groups[g] = sub.add_parser(g, help=f"{g} commands").add_subparsers(
                dest="cmd", required=True
            )
        cp = groups[g].add_parser(c, help=f"ambe {g} {c}")
        cp.add_argument("--config", required=True, help="YAML config file")
        # Subcommand-specific flags handled downstream via parse_known_args
        cp.set_defaults(_target=f"{g}:{c}")

    args, extra = parser.parse_known_args(argv)

    ctx = load_context(args.config)
    print(ctx.summary(), flush=True)

    key = tuple(args._target.split(":"))
    if key not in COMMANDS:
        parser.error(f"unknown command: {key}")

    from .provenance import record_ambe_run
    record_ambe_run(ctx, *key, argv if argv is not None else sys.argv[1:])

    func = _load(COMMANDS[key])
    return func(ctx=ctx, argv=extra)


if __name__ == "__main__":
    sys.exit(main())
