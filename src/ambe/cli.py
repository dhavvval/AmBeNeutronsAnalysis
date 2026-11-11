"""
`ambe` command-line entry point.

After `pip install -e .`, a user can run:

    ambe --help
    ambe data process --config configs/data_ambe2v4_bkgnorm.yaml
    ambe mc process   --config configs/mc_ambe_v3fix_wf_dtho6256.yaml
    ambe mc capture   --config configs/mc_ambe_v3fix_wf_dtho6256.yaml
    ambe stats foldeff --config configs/step3_ambe2v4_bkgnorm.yaml --fold-model forward --stage data
    ambe plots basic  --config configs/data_ambe2v4_all28_box.yaml

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
    ("mc", "discriminate"):     "ambe.mc.cluster_discrimination:cli",
    ("mc", "timespread"):       "ambe.mc.time_spread:cli",
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
