"""
Output directory resolution.

Resolution order:
  1. explicit argument (e.g. from a config file)
  2. $AMBE_OUT environment variable
  3. ~/ambe_analysis_output/ (default)

All `*_dir` helpers guarantee the directory exists (mkdir parents).
"""

import os
from pathlib import Path
from typing import Optional

DEFAULT_OUTPUT = Path.home() / "ambe_analysis_output"


def output_root(explicit: Optional[str] = None) -> Path:
    """Base output dir. explicit > $AMBE_OUT > ~/ambe_analysis_output/."""
    if explicit:
        root = Path(explicit).expanduser()
    elif os.environ.get("AMBE_OUT"):
        root = Path(os.environ["AMBE_OUT"]).expanduser()
    else:
        root = DEFAULT_OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dir(run_name: str, explicit_root: Optional[str] = None) -> Path:
    """Directory for one named run: <root>/<run_name>/."""
    d = output_root(explicit_root) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def plots_dir(run_name: str, explicit_root: Optional[str] = None) -> Path:
    """<root>/<run_name>/plots/"""
    d = run_dir(run_name, explicit_root) / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parquet_dir(run_name: str, explicit_root: Optional[str] = None) -> Path:
    """<root>/<run_name>/parquet/"""
    d = run_dir(run_name, explicit_root) / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def csv_dir(run_name: str, explicit_root: Optional[str] = None) -> Path:
    """<root>/<run_name>/csv/"""
    d = run_dir(run_name, explicit_root) / "csv"
    d.mkdir(parents=True, exist_ok=True)
    return d
