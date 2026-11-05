"""Data-vs-MC clock offset from the prompt tank light: `ambe data clockoffset`.

    clock_offset:
      data_files: [.../BeamCluster_6265.root, ...]    # globs allowed
      mc_files:   [.../ANNIEEvent_MC_AmBe_port5_z0QE_1.50_0_999.root, ...]
      max_entries: 8000      # entries read per file
      search_ns: [0, 3000]   # range searched for the prompt peak
      bin_ns: 10
      halfwidth_ns: 60       # prompt time = median of the hits within +-halfwidth of the peak bin

The offset median(data) - median(MC) is the mc_clock analysis_time_shift_ns of that trigger type
(ambe.mc.clock). Writes <run_name>__clock_offset.csv (one row per file) and .json (summary).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs

DEFAULTS = dict(data_files=[], mc_files=[], max_entries=8000, search_ns=[0.0, 3000.0],
                bin_ns=10.0, halfwidth_ns=60.0, tree="Event")


def prompt_time(path: Path, cfg: dict) -> dict:
    t = uproot.open(str(path))[cfg["tree"]]
    n = min(int(cfg["max_entries"]), t.num_entries)
    h = ak.to_numpy(ak.flatten(t["hitT"].array(entry_stop=n))).astype(float)
    lo, hi = (float(x) for x in cfg["search_ns"])
    h = h[(h >= lo) & (h < hi)]
    counts, edges = np.histogram(h, bins=np.arange(lo, hi + cfg["bin_ns"], cfg["bin_ns"]))
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
    core = h[np.abs(h - peak) <= float(cfg["halfwidth_ns"])]
    return dict(file=path.name, entries=n, peak_bin_ns=peak, n_core_hits=len(core),
                prompt_median_ns=float(np.median(core)) if len(core) else np.nan)


def run(ctx: RunContext) -> pd.DataFrame:
    cfg = {**DEFAULTS, **(ctx.extra.get("clock_offset") or {})}
    rows = []
    for kind in ("data", "mc"):
        for p in resolve_inputs(cfg[f"{kind}_files"]) if cfg[f"{kind}_files"] else []:
            r = dict(kind=kind, **prompt_time(p, cfg))
            rows.append(r)
            print(f"[clockoffset] {kind:4s} {r['file']:55s} prompt {r['prompt_median_ns']:8.1f} ns "
                  f"({r['n_core_hits']} hits, {r['entries']} entries)")
    df = pd.DataFrame(rows)
    out = ctx.csv_path(f"{ctx.run_name}__clock_offset")
    df.to_csv(out, index=False)
    summ = {k: dict(n_files=int((df.kind == k).sum()),
                    median_ns=float(df.prompt_median_ns[df.kind == k].median()),
                    min_ns=float(df.prompt_median_ns[df.kind == k].min()),
                    max_ns=float(df.prompt_median_ns[df.kind == k].max()))
            for k in ("data", "mc") if (df.kind == k).any()}
    if {"data", "mc"} <= set(summ):
        summ["offset_ns"] = summ["data"]["median_ns"] - summ["mc"]["median_ns"]
        print(f"[clockoffset] data {summ['data']['median_ns']:.1f} ns "
              f"({summ['data']['min_ns']:.1f}-{summ['data']['max_ns']:.1f}), MC {summ['mc']['median_ns']:.1f} ns "
              f"({summ['mc']['min_ns']:.1f}-{summ['mc']['max_ns']:.1f}) -> offset {summ['offset_ns']:.1f} ns")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=2))
    print(f"[clockoffset] wrote {out} and {out.with_suffix('.json').name}")
    return df


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    argparse.ArgumentParser(prog="ambe data clockoffset").parse_args(list(argv) if argv else [])
    run(ctx)
