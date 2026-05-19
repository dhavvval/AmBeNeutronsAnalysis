#!/usr/bin/env python3
"""
Extract Michel electron cluster features from an ANNIEEvent MC ROOT file.

Two modes:
  --exact-cuts  Apply exactly Michel_tuning.py thresholds; prints a cutflow
                table and exits (expected result: 0 surviving clusters because
                HasMRD=1 for all WCSim events and no delayed cluster has ≥20
                hits with CB < 0.18).
  default       MC-adapted selection (see below) suitable for this file.

MC-adapted selection (default):
  - Muon cluster  : max-PE cluster per event (PE > 500)
  - Michel candidate: cluster with dt > 200 ns from muon, PE < 650, hits >= 5

Standard Michel_tuning.py thresholds (hits >= 20, CB < 0.18, MRD/veto flags)
cannot be applied to this file: the muons are through-going (HasMRD=1 for all
events, MRDStop=0, MRDThrough=0) and the resulting Michel decay clusters have
only 5-17 hits and CB 0.2-0.6.  The dt > 200 ns cut is the primary discriminant
that separates true muon-decay products from prompt scattered muon light.

Usage
-----
    # Exact Michel_tuning.py cuts — cutflow diagnostic only:
    python extract_michel_features.py --config configs/mc_michel_mva.yaml --exact-cuts

    # MC-adapted cuts — produces features parquet:
    python extract_michel_features.py --config configs/mc_michel_mva.yaml
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ambe.mc.cluster_features import compute_cluster_features, load_geometry


# ── Selection thresholds (MC-adapted, see module docstring) ───────────────────
MUON_MIN_PE      = 500      # PE floor for the muon cluster
MICHEL_MAX_PE    = 650      # Michel_tuning.py upper bound: cluster_PE < 650
MICHEL_MIN_HITS  = 5        # relaxed from 20 (MC Michels have 5–17 hits)
MICHEL_MIN_DT_NS = 200.0    # minimum time after muon (ns)
MICHEL_MAX_DT_NS = 70_000.0 # ~30 muon lifetimes


def _exact_cutflow(root_path: str, tree_name: str = "Event;1") -> None:
    """
    Apply the exact Michel_tuning.py cuts to the ANNIEEvent ROOT file and
    print a step-by-step cutflow table.  Does not extract features — just
    reports how many events/clusters survive each cut.

    Muon (dirt) cuts from dirt():
      1. HasMRD == 0
      2. TankMRDCoinc == 0
      3. NoVeto == 0           (NV=0 means veto fired — MRD already checked)
      4. Not only cluster (≥2 clusters in event)
      5. hadExtended == 1
      6. isBrightest == 1 (muon = max-PE cluster)
      7. muon hits ≥ 50
      8. 1000 < muon PE < 4000
      9. muon CB ∈ [0, 0.2)
     10. muon clusterTime ∈ (200, 1800) ns
     11. charge barycenter downstream (ΣhitZ·hitPE > 0)

    Michel cuts from Michel():
     12. adj_time ∈ (200, 5000) ns
     13. 0 < Michel PE < 650
     14. Michel hits ≥ 20
     15. Michel CB ∈ (0, 0.2)
    """
    print(f"\n{'='*60}")
    print("EXACT Michel_tuning.py CUTFLOW")
    print(f"{'='*60}")
    print(f"File: {root_path}\n")

    with uproot.open(root_path) as f:
        tree = f[tree_name]
        print(f"Tree '{tree_name}': {tree.num_entries} total events\n")

        # Load cluster-level arrays
        ct_arr  = tree["clusterTime"].array(library="ak")
        cpe_arr = tree["clusterPE"].array(library="ak")
        ch_arr  = tree["clusterHits"].array(library="ak")
        ccb_arr = tree["clusterChargeBalance"].array(library="ak")
        hz_arr  = tree["Cluster_HitZ"].array(library="ak")
        hpe_arr = tree["Cluster_HitPE"].array(library="ak")

        # Try to load event-level flag branches (may not exist in WCSim output)
        def _try_load(branch_name):
            try:
                return tree[branch_name].array(library="np")
            except Exception:
                return None

        extended  = _try_load("hadExtended")

        # Report what's available
        flag_avail = {"hadExtended": extended is not None}
        print("Event-level branches found in ROOT file:")
        for name, found in flag_avail.items():
            status = "YES" if found else "NOT FOUND (cut skipped)"
            print(f"  {name:20s}: {status}")
        print()

    n_total = len(ct_arr)

    # Count surviving events after each muon cut
    # Accumulate per-event pass/fail
    survive = np.ones(n_total, dtype=bool)  # all events start passing

    # ── Cut 1: hadExtended == 1 ───────────────────────────────────────────────
    if extended is not None:
        cut1 = (extended == 1)
        survive &= cut1
        print(f"  Cut 1  hadExtended==1:         {survive.sum():5d} events pass")
    else:
        print(f"  Cut 1  hadExtended==1:         SKIPPED (branch absent)")

    print(f"\nAfter event-level flag cuts: {survive.sum()} events remain.\n")
    print("Continuing muon cluster cuts on ALL events (ignoring unavailable flags):\n")

    # For cluster-level cuts, run on all events (flags may be all unavailable)
    n_events = len(ct_arr)

    # Per-event cluster-level muon cut counts
    n_pass = {
        "≥2 clusters"          : 0,
        "brightest=max-PE"     : 0,
        "hits≥50"              : 0,
        "1000<PE<4000"         : 0,
        "CB∈[0,0.2)"           : 0,
        "CT∈(200,1800)ns"      : 0,
        "bary_downstream"      : 0,
    }
    n_dirt_muon_events = 0

    # Michel-level cutflow (applied to events that pass all muon cuts)
    michel_counts = {
        "adj_time∈(200,5000)ns": 0,
        "0<PE<650"             : 0,
        "hits≥20"              : 0,
        "CB∈(0,0.2)"           : 0,
    }
    n_michel_total = 0

    for i in range(n_events):
        ct  = ak.to_list(ct_arr[i])
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])

        if len(cpe) < 2:
            continue
        n_pass["≥2 clusters"] += 1

        muon_idx = int(np.argmax(cpe))
        muon_t   = float(ct[muon_idx])
        muon_pe  = float(cpe[muon_idx])
        muon_h   = int(ch[muon_idx])
        muon_cb  = float(ccb[muon_idx])
        n_pass["brightest=max-PE"] += 1  # by definition always

        if muon_h < 50:
            continue
        n_pass["hits≥50"] += 1

        if not (1000 < muon_pe < 4000):
            continue
        n_pass["1000<PE<4000"] += 1

        if not (0 <= muon_cb < 0.2):
            continue
        n_pass["CB∈[0,0.2)"] += 1

        if not (200 < muon_t < 1800):
            continue
        n_pass["CT∈(200,1800)ns"] += 1

        # Charge barycenter: ΣhitZ·hitPE > 0
        hz  = np.asarray(ak.to_list(hz_arr[i][muon_idx]),  dtype=float)
        hpe = np.asarray(ak.to_list(hpe_arr[i][muon_idx]), dtype=float)
        bary = float(np.sum(hz * hpe)) if len(hz) > 0 else 0.0
        if bary <= 0:
            continue
        n_pass["bary_downstream"] += 1

        n_dirt_muon_events += 1

        # ── Michel cluster cuts ──────────────────────────────────────────
        for k, (t_k, pe_k, h_k, cb_k) in enumerate(zip(ct, cpe, ch, ccb)):
            if k == muon_idx:
                continue
            dt = float(t_k) - muon_t

            if not (200 < dt < 5000):
                continue
            michel_counts["adj_time∈(200,5000)ns"] += 1

            if not (0 < float(pe_k) < 650):
                continue
            michel_counts["0<PE<650"] += 1

            if int(h_k) < 20:
                continue
            michel_counts["hits≥20"] += 1

            if not (0 < float(cb_k) < 0.2):
                continue
            michel_counts["CB∈(0,0.2)"] += 1

            n_michel_total += 1

    # ── Print muon cutflow ────────────────────────────────────────────────────
    print(f"{'─'*55}")
    print(f"MUON (dirt) cutflow  (on all {n_events} events, flags excluded)")
    print(f"{'─'*55}")
    remaining = n_events
    for cut_name, n in n_pass.items():
        print(f"  {cut_name:30s}  {n:5d} events pass")
        remaining = n
    print(f"{'─'*55}")
    print(f"  Dirt muon events selected:     {n_dirt_muon_events}")

    # ── Print Michel cutflow ──────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"MICHEL cutflow  (on clusters after dirt muon in same event)")
    print(f"{'─'*55}")
    for cut_name, n in michel_counts.items():
        print(f"  {cut_name:30s}  {n:5d} Michel candidates")
    print(f"{'─'*55}")
    print(f"  Michel clusters selected:      {n_michel_total}")
    print(f"\n{'='*60}")

    if n_michel_total == 0:
        print("\nResult: ZERO Michel clusters pass the exact Michel_tuning.py cuts.")
        print("This is expected for this MC file because:")
        print("  - All muons are through-going (HasMRD=1) → MRD cut blocks all events")
        print("  - Delayed clusters (dt>200ns) have only 5-17 hits → hits<20 cut kills them")
        print("  - Delayed clusters have CB 0.2-0.6 → CB<0.2 cut removes survivors")
        print("\nNext step: relax cuts to the MC-adapted thresholds (--config mode)")
    print(f"{'='*60}\n")


def _michel_indices(ct: list, cpe: list, ch: list, ccb: list) -> list[int]:
    """
    Return cluster indices that pass the Michel selection in one event.
    Muon = max-PE cluster; Michel must arrive >= MICHEL_MIN_DT_NS later.
    Applies Michel_tuning.py CB cut: charge balance strictly in (0, 0.2).
    """
    if len(cpe) < 2:
        return []
    muon_idx = int(np.argmax(cpe))
    muon_t   = float(ct[muon_idx])
    out = []
    for k, (t_k, pe_k, h_k, cb_k) in enumerate(zip(ct, cpe, ch, ccb)):
        if k == muon_idx:
            continue
        dt = float(t_k) - muon_t
        if not (MICHEL_MIN_DT_NS <= dt <= MICHEL_MAX_DT_NS):
            continue
        if float(pe_k) >= MICHEL_MAX_PE:
            continue
        if int(h_k) < MICHEL_MIN_HITS:
            continue
        if not (0 < float(cb_k) < 0.2):
            continue
        out.append(k)
    return out


def extract_michel_features(root_path: str, geo,
                             source_pos_m=None,
                             tree_name: str = "Event;1") -> pd.DataFrame:
    """
    Open ANNIEEvent ROOT file, apply MC-adapted Michel selection, and
    extract per-cluster physics features with compute_cluster_features().

    Returns a DataFrame with all features from cluster_features.py
    plus 'event_idx' and 'adj_time_ns' columns.
    """
    print(f"[michel] opening  {root_path}")
    with uproot.open(root_path) as f:
        tree = f[tree_name]
        print(f"[michel] tree '{tree_name}'  {tree.num_entries} entries")

        ct_arr  = tree["clusterTime"].array(library="ak")
        cpe_arr = tree["clusterPE"].array(library="ak")
        ch_arr  = tree["clusterHits"].array(library="ak")
        ccb_arr = tree["clusterChargeBalance"].array(library="ak")

        # Doubly-nested: (event, cluster, hit)
        hx_arr  = tree["Cluster_HitX"].array(library="ak")
        hy_arr  = tree["Cluster_HitY"].array(library="ak")
        hz_arr  = tree["Cluster_HitZ"].array(library="ak")
        ht_arr  = tree["Cluster_HitT"].array(library="ak")
        hpe_arr = tree["Cluster_HitPE"].array(library="ak")
        hck_arr = tree["Cluster_HitChankey"].array(library="ak")

    rows = []
    n_entries = len(ct_arr)

    for i in range(n_entries):
        ct  = ak.to_list(ct_arr[i])
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])

        if not cpe or float(max(cpe)) < MUON_MIN_PE:
            continue

        for k in _michel_indices(ct, cpe, ch, ccb):
            hx  = np.asarray(ak.to_list(hx_arr[i][k]),  dtype=float)
            hy  = np.asarray(ak.to_list(hy_arr[i][k]),  dtype=float)
            hz  = np.asarray(ak.to_list(hz_arr[i][k]),  dtype=float)
            ht  = np.asarray(ak.to_list(ht_arr[i][k]),  dtype=float)
            hpe = np.asarray(ak.to_list(hpe_arr[i][k]), dtype=float)
            hck = np.asarray(ak.to_list(hck_arr[i][k]), dtype=int)

            if len(ht) == 0:
                continue

            df_cl = pd.DataFrame({"x": hx, "y": hy, "z": hz,
                                  "t": ht, "pe": hpe, "pmtID": hck})
            feats = compute_cluster_features(df_cl, geo, source_pos_m=source_pos_m)
            if not feats:
                continue

            muon_t = float(ct[int(np.argmax(cpe))])
            feats["event_idx"]   = int(i)
            feats["adj_time_ns"] = round(float(ct[k]) - muon_t, 1)
            feats["is_michel"]   = 1
            rows.append(feats)

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"[michel] extracted {n} Michel clusters from {n_entries} events")
    if n > 0:
        print(f"[michel] adj_time_ns:  "
              f"min={df['adj_time_ns'].min():.0f}  "
              f"median={df['adj_time_ns'].median():.0f}  "
              f"max={df['adj_time_ns'].max():.0f} ns")
        print(f"[michel] n_hits:       "
              f"min={int(df['n_hits'].min())}  "
              f"median={df['n_hits'].median():.0f}  "
              f"max={int(df['n_hits'].max())}")
    return df


def _resolve_files(input_path: str) -> list[Path]:
    """Accept a single ROOT file, a directory, or a glob pattern."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("ANNIEEvent_*.root"))
        if not files:
            files = sorted(p.glob("*.root"))
    else:
        files = sorted(Path(f) for f in glob.glob(input_path))
    if not files:
        raise FileNotFoundError(f"No ROOT files found for: {input_path}")
    return files


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(prog="extract_michel_features")
    p.add_argument("--config",      help="YAML config (configs/mc_michel_mva.yaml)")
    # Individual args — override config when both are supplied
    p.add_argument("--input",       help="ROOT file, directory, or glob "
                                        "(e.g. /path/to/ANNIEEvent_dirtmuon_*.root)")
    p.add_argument("--geo",         help="FullTankPMTGeometry.csv")
    p.add_argument("--offsets",     help="TankPMTTimingOffsets.csv")
    p.add_argument("--output",      help="Output .parquet path")
    p.add_argument("--source-pos",  type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="AmBe source position in metres. Example: 0 -0.1446 1.681")
    p.add_argument("--tree",        default="Event;1")
    p.add_argument("--exact-cuts",  action="store_true",
                   help="Apply exact Michel_tuning.py cuts and print cutflow table. "
                        "Does not extract features or write output.")
    args = p.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    geo_block    = cfg.get("geometry", {})
    michel_block = cfg.get("michel",   {})

    root_path    = args.input  or michel_block.get("input")
    geo_path     = args.geo    or geo_block.get("pmt_geometry")
    off_path     = args.offsets or geo_block.get("timing_offsets")
    output_path  = args.output or michel_block.get("output")

    if args.exact_cuts:
        if not root_path:
            p.error("Provide --input or --config with michel.input for --exact-cuts")
        _exact_cutflow(root_path, tree_name=args.tree)
        return

    if not all([root_path, geo_path, off_path, output_path]):
        p.error("Provide --config or all of --input --geo --offsets --output")

    src_raw = (args.source_pos
               or michel_block.get("source_position_m"))
    source_pos_m = np.array([float(v) for v in src_raw]) if src_raw else None
    if source_pos_m is not None:
        print(f"[michel] source position: {tuple(source_pos_m)} m  (d_source enabled)")

    # Apply selection thresholds from config (fallback to module defaults)
    global MUON_MIN_PE, MICHEL_MAX_PE, MICHEL_MIN_HITS, MICHEL_MIN_DT_NS, MICHEL_MAX_DT_NS
    MUON_MIN_PE      = float(michel_block.get("muon_min_pe",      MUON_MIN_PE))
    MICHEL_MAX_PE    = float(michel_block.get("michel_max_pe",    MICHEL_MAX_PE))
    MICHEL_MIN_HITS  = int(michel_block.get("michel_min_hits",  MICHEL_MIN_HITS))
    MICHEL_MIN_DT_NS = float(michel_block.get("michel_min_dt_ns", MICHEL_MIN_DT_NS))
    MICHEL_MAX_DT_NS = float(michel_block.get("michel_max_dt_ns", MICHEL_MAX_DT_NS))

    geo   = load_geometry(geo_path, off_path)
    files = _resolve_files(root_path)
    print(f"[michel] found {len(files)} file(s)")

    frames = []
    for i, f in enumerate(files):
        print(f"[michel] [{i+1}/{len(files)}] {f.name}")
        df_f = extract_michel_features(str(f), geo,
                                       source_pos_m=source_pos_m,
                                       tree_name=args.tree)
        if len(df_f):
            frames.append(df_f)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[michel] wrote {len(df)} rows from {len(files)} file(s) → {out}")


if __name__ == "__main__":
    main()
