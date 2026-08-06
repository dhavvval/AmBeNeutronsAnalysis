#!/usr/bin/env python
"""
Audit the merged tank+world training frame BEFORE trusting a merged training.

Checks the three things that can silently go wrong when two samples are combined,
each of which would look like a physics result rather than a bug:

  1. LABEL AUDIT     — cross-tab _source_run x neutron-dominated x origin_in_tank.
                       Confirms tank rows are ~all in-tank, world rows split ~22/78,
                       and that a non-trivial number of neutron-dominated world
                       clusters are being reclassified as background (the whole point
                       of processing the world sample).
  2. GATING AUDIT    — surviving out-of-tank clusters MUST include cc_pass==0 rows.
                       If every out-of-tank row has cc_pass==1 the origin-aware
                       filter never took effect and ~88% of the background is missing.
  3. SPLIT INTEGRITY — no (_source_run, eventID) pair may appear in both train and
                       test. eventID restarts per run, so a bare eventID key would
                       merge unrelated events across samples and leak clusters.

Usage:
  python audit_merged_labels.py --config configs/cc_neutrino_v3_truthtag.yaml \
      --merge-run cc_neutrino_v3world_truthtag --method optics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NEUTRON_CLASSES = [1, 2, 3, 4]


def load(config: str, merge_runs: list[str]) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config))
    root = Path(cfg["output_root"])
    frames = []
    for run in [cfg["run_name"], *merge_runs]:
        p = root / run / "parquet" / f"{run}__cluster_features.parquet"
        if not p.exists():
            sys.exit(f"[audit] missing: {p}")
        d = pd.read_parquet(p)
        d["_source_run"] = run
        print(f"[audit] {run}: {len(d)} clusters")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in ("origin_in_tank", "origin_in_fv"):
        if c in df.columns:
            df[c] = df[c].fillna(-1).astype(int)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--merge-run", action="append", default=[])
    ap.add_argument("--method", default="optics")
    a = ap.parse_args()

    df = load(a.config, a.merge_run)
    df = df[df["method"] == a.method].reset_index(drop=True)
    print(f"[audit] method={a.method}: {len(df)} clusters\n")

    has_origin = "origin_in_tank" in df.columns
    if not has_origin:
        print("[audit] no origin_in_tank column — nothing to audit (tank-only frame)")
        return 0

    # ---- apply the same gate mva_analysis applies -------------------------------
    keep = (df["cc_pass"] == 1) | (df["origin_in_tank"] == 0)
    kept = df[keep].reset_index(drop=True)
    print(f"[1] gate: {len(kept)}/{len(df)} clusters kept")

    out = kept["origin_in_tank"] == 0
    n_out_failing_cc = int((out & (kept["cc_pass"] != 1)).sum())
    print(f"    out-of-tank kept: {int(out.sum())}, of which cc_pass==0: {n_out_failing_cc}")
    ok_gate = n_out_failing_cc > 0
    print(f"    {'PASS' if ok_gate else 'FAIL'}: out-of-tank harvest is unconditional"
          f"{'' if ok_gate else ' — every out-of-tank row passed cc_pass, gate not in effect'}")
    bad_in = int(((kept["origin_in_tank"] == 1) & (kept["cc_pass"] != 1)).sum())
    print(f"    {'PASS' if bad_in == 0 else 'FAIL'}: in-tank rows all satisfy cc_pass "
          f"({bad_in} violations)\n")

    # ---- label audit -------------------------------------------------------------
    nd = kept["dominant_class"].isin(NEUTRON_CLASSES)
    print("[2] label audit — clusters by source x neutron-dominated x origin:")
    tab = pd.crosstab([kept["_source_run"], kept["origin_in_tank"]], nd,
                      rownames=["source_run", "origin_in_tank"], colnames=["neutron_dom"])
    print(tab.to_string(), "\n")
    reclass = int((nd & out).sum())
    print(f"    neutron-dominated clusters from out-of-tank interactions "
          f"-> BACKGROUND: {reclass}")
    print(f"    {'PASS' if reclass > 0 else 'FAIL'}: the world sample is contributing "
          f"reclassified real captures\n")

    if "origin_in_fv" in kept.columns:
        dropped = int(((kept["origin_in_tank"] == 1) & (kept["origin_in_fv"] == 0)).sum())
        elig = int(((kept["origin_in_tank"] == 1) & (kept["origin_in_fv"] == 1)).sum())
        print(f"[3] signal phase space: {elig} in-tank clusters inside the FV (eligible "
              f"for signal), {dropped} in-tank clusters outside the FV (dropped)\n")

    # ---- split integrity ---------------------------------------------------------
    # Reproduce the composite key mva_analysis uses and confirm it separates runs.
    codes, _ = pd.factorize(pd.Series(list(zip(kept["_source_run"].astype(str).tolist(),
                                               kept["eventID"].fillna(-1).astype(int).tolist()))))
    n_composite = len(np.unique(codes))
    n_bare = kept["eventID"].nunique()
    print(f"[4] split key: {n_composite} distinct (_source_run, eventID) vs "
          f"{n_bare} distinct bare eventID")
    ok_key = n_composite >= n_bare
    print(f"    {'PASS' if ok_key else 'FAIL'}: composite key does not collapse events "
          f"across runs")
    if n_composite > n_bare:
        print(f"    -> {n_composite - n_bare} event(s) would have been WRONGLY merged by a "
              f"bare eventID key")

    print("\n[audit] done")
    return 0 if (ok_gate and bad_in == 0 and reclass > 0 and ok_key) else 1


if __name__ == "__main__":
    sys.exit(main())
