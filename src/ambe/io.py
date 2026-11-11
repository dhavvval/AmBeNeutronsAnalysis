"""
Shared I/O helpers: resolve input paths declared in a config, read CSVs/Parquets.

The config's `inputs:` block can hold either absolute paths or glob patterns;
these helpers resolve them and give back the concrete files.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import pandas as pd

from .context import RunContext

PathLike = Union[str, Path]


def resolve_inputs(patterns: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    """
    Expand a single path-or-glob (or a list of them) into a sorted list of
    existing files. Raises FileNotFoundError if nothing matches.
    """
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    resolved: List[Path] = []
    for pat in patterns:
        pat = str(Path(pat).expanduser())
        matches = [Path(p) for p in sorted(glob.glob(pat))]
        if not matches and Path(pat).exists():
            matches = [Path(pat)]
        if not matches:
            raise FileNotFoundError(f"no files match: {pat}")
        resolved.extend(matches)
    return resolved


def filter_files_with_tree(files: Sequence[Path], tree_name: str = "Event",
                           verbose: bool = True) -> tuple[List[Path], List[Path]]:
    """
    Split `files` into (usable, unusable) by whether they hold a non-empty `tree_name`.

    Large productions run per-file on the grid and an interrupted or failed job can
    leave a zero-key ROOT file behind (observed: productionv3 tank/fmvmrd
    ANNIEEvent_cc_neutrino_v3_24.root, 1 of 499). Without this check the first such
    file aborts a multi-hour Stage 0 with a bare KeyError after everything before it
    has already been read.

    The skip is LOUD by design: every dropped file is named, and the count is
    repeated as a warning. A silently shorter sample would look like a real physics
    result. Raises FileNotFoundError if nothing is usable.
    """
    import uproot

    good: List[Path] = []
    bad: List[Path] = []
    for p in files:
        try:
            with uproot.open(str(p)) as f:
                if tree_name in f and f[tree_name].num_entries > 0:
                    good.append(Path(p))
                    continue
                reason = (f"tree {tree_name!r} absent" if tree_name not in f
                          else f"tree {tree_name!r} empty")
        except Exception as exc:                     # unreadable / truncated file
            reason = f"open failed ({type(exc).__name__})"
        bad.append(Path(p))
        if verbose:
            print(f"[io] SKIP {Path(p).name}: {reason}")

    if not good:
        raise FileNotFoundError(
            f"none of the {len(files)} input file(s) contain a non-empty "
            f"{tree_name!r} tree")
    if bad and verbose:
        print(f"[io] WARNING: skipped {len(bad)}/{len(files)} input file(s) with no "
              f"usable {tree_name!r} tree — the sample is that much smaller than the "
              f"glob suggests: {[p.name for p in bad]}")
    return good, bad


def inputs_from_ctx(ctx: RunContext, key: str) -> List[Path]:
    """Resolve `ctx.inputs[key]` (required). Convenient wrapper."""
    if key not in ctx.inputs:
        raise KeyError(f"config {ctx.run_name!r} is missing inputs.{key}")
    return resolve_inputs(ctx.inputs[key])


def event_key_columns(df: "pd.DataFrame", who: str = "ambe") -> List[str]:
    """The columns that identify one EVENT in a pooled candidate table.

    Always (run, eventTankTime) -- never eventTankTime alone.

    In real data eventTankTime is a globally unique ns timestamp, so adding `run` is a
    strict refinement that changes no result. In MC it is `eventNumber` (0-999, because
    the MC's own eventTimeTank is identically 0), so eventTankTime alone would collapse
    one event from EVERY run into a single group. That mis-grouping does not raise: it
    silently produces a multiplicity histogram peaking near the number of pooled runs
    and delta_t values taken between clusters of unrelated runs.

    `run` is written by data/processor.py. Older candidate CSVs predate that column, so
    it is recovered from the `_source_file` name (..._<run>.csv) that load_csvs attaches.
    Failing that we fall back to eventTankTime alone and say so loudly -- correct for
    legacy data, wrong for MC, and the caller deserves to know which it got.
    """
    if "run" in df.columns:
        return ["run", "eventTankTime"]
    if "_source_file" in df.columns:
        run = df["_source_file"].str.extract(r"_(\d+)\.csv$", expand=False)
        if run.notna().all():
            df["run"] = run.astype("int64")
            return ["run", "eventTankTime"]
    print(f"[{who}] WARNING: no `run` column and none recoverable from _source_file; "
          f"grouping events on eventTankTime alone. That is correct only if "
          f"eventTankTime is a globally unique timestamp (real data). For MC it merges "
          f"one event per run into each group.")
    return ["eventTankTime"]


def load_csvs(paths: Iterable[PathLike], **read_csv_kwargs) -> pd.DataFrame:
    """Concat-read a list of CSV files, adding a `_source_file` column."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, **read_csv_kwargs)
        df["_source_file"] = Path(p).name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
