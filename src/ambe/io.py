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


def inputs_from_ctx(ctx: RunContext, key: str) -> List[Path]:
    """Resolve `ctx.inputs[key]` (required). Convenient wrapper."""
    if key not in ctx.inputs:
        raise KeyError(f"config {ctx.run_name!r} is missing inputs.{key}")
    return resolve_inputs(ctx.inputs[key])


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
