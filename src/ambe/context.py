"""
RunContext -- the single object every script in the package accepts.

It bundles:
  - config-file contents (input paths, selection cuts, fit parameters, ...)
  - derived output directories (plots, parquet, csv) under <AMBE_OUT>/<run_name>/
  - title and filename helpers so scripts never have to construct them manually

Every script should take a RunContext and derive its behaviour from it rather
than from hardcoded constants.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import plots_dir, parquet_dir, csv_dir, run_dir, output_root


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sanitise(name: str) -> str:
    """Make `name` safe to embed in filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# --------------------------------------------------------------------------- #
# RunContext
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    """
    The single piece of state every analysis script receives.

    Fields:
      run_name       short identifier, used in dirs + filenames, e.g. "mc_optics_trial01"
      description    free-form, used in plot titles
      campaign       optional campaign tag (e.g. "AmBe2.0v4"); appears in titles/filenames
      inputs         dict of input paths (root_files, csv_files, etc.)
      cuts           dict of selection cuts (PE thresholds, etc.)
      fit_params     dict of fitting parameters
      output_root    override for AMBE_OUT; usually None
      extra          free-form dict for anything else the config declares
    """

    run_name: str
    description: str = ""
    campaign: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    cuts: Dict[str, Any] = field(default_factory=dict)
    fit_params: Dict[str, Any] = field(default_factory=dict)
    output_root: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -------- Directories (computed) ----------------------------------------
    @property
    def run_dir(self) -> Path:
        return run_dir(self.run_name, self.output_root)

    @property
    def plots_dir(self) -> Path:
        return plots_dir(self.run_name, self.output_root)

    @property
    def parquet_dir(self) -> Path:
        return parquet_dir(self.run_name, self.output_root)

    @property
    def csv_dir(self) -> Path:
        return csv_dir(self.run_name, self.output_root)

    # -------- Title / filename helpers --------------------------------------
    def title(self, base: str) -> str:
        """
        Compose a plot title. Scripts call ctx.title("Cluster PE vs Charge Balance")
        and get back e.g. "Cluster PE vs Charge Balance - AmBe2.0v4 (trial01)".
        """
        parts = [base]
        if self.campaign:
            parts.append(f" – {self.campaign}")
        if self.run_name:
            parts.append(f" ({self.run_name})")
        return "".join(parts)

    def filename(self, base: str, ext: str = "png") -> str:
        """
        Compose a filename. ctx.filename("cluster_pe_vs_cb", "pdf")
        -> "cluster_pe_vs_cb__AmBe2.0v4__trial01.pdf"
        """
        parts = [_sanitise(base)]
        if self.campaign:
            parts.append(_sanitise(self.campaign))
        if self.run_name:
            parts.append(_sanitise(self.run_name))
        return "__".join(parts) + "." + ext.lstrip(".")

    def plot_path(self, base: str, ext: str = "png") -> Path:
        """Full output path for a plot: <plots_dir>/<generated filename>."""
        return self.plots_dir / self.filename(base, ext)

    def parquet_path(self, base: str) -> Path:
        return self.parquet_dir / (_sanitise(base) + ".parquet")

    def csv_path(self, base: str) -> Path:
        return self.csv_dir / (_sanitise(base) + ".csv")

    # -------- Serialisation -------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"RunContext(run_name={self.run_name!r})",
            f"  campaign    : {self.campaign}",
            f"  description : {self.description}",
            f"  output_root : {output_root(self.output_root)}",
            f"  run_dir     : {self.run_dir}",
            f"  inputs      : {list(self.inputs.keys())}",
            f"  cuts        : {self.cuts}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# YAML loader
# --------------------------------------------------------------------------- #
REQUIRED = {"run_name"}


def load_context(config_path: str | os.PathLike) -> RunContext:
    """
    Parse a YAML config file into a RunContext.

    Minimal YAML:
        run_name: mc_optics_trial01

    Full example: see configs/example_mc_optics.yaml.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    missing = REQUIRED - raw.keys()
    if missing:
        raise KeyError(f"config {path} missing required fields: {missing}")

    # Pull out known fields; stash the rest in .extra so scripts can still reach them.
    known = {"run_name", "description", "campaign", "inputs", "cuts",
             "fit_params", "output_root"}
    ctx_kwargs = {k: raw[k] for k in known if k in raw}
    ctx_kwargs["extra"] = {k: v for k, v in raw.items() if k not in known}

    return RunContext(**ctx_kwargs)
