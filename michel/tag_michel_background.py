"""
tag_michel_background.py

Select Michel electron clusters from real ANNIE beam data ROOT files and
extract per-cluster physics features — same output schema as
extract_michel_features.py (MC version), so both feed directly into
mva_neutron_vs_michel.py without any extra processing step.

Selection mirrors Michel_tuning.py exactly (dirt muon + Michel cuts with
CB < 0.18 tightening).  Features are computed by compute_cluster_features()
from src/ambe/mc/cluster_features.py — identical to the MC pipeline.

Output
------
michel_data_features.parquet
    All physics features from cluster_features.py, plus:
      run, event_number, muon_event_idx, adj_time_ns, is_michel=1

Input format
------------
BeamClusterAnalysis ntuples ("data" tree), same files as Michel_tuning.py.
Cluster-level branches per event row:
  cluster_time, cluster_PE, cluster_Hits, cluster_Qb, cluster_Number,
  isBrightest, hadExtended, NoVeto, TankMRDCoinc, MRD_activity
Hit-level branches (one array per cluster row):
  hitX, hitY, hitZ, hitT, hitPE, hitID

Usage
-----
    python tag_michel_background.py \\
        /path/to/beam_data.root \\
        --geo    /path/to/FullTankPMTGeometry.csv \\
        --offsets /path/to/TankPMTTimingOffsets.csv \\
        --output michel/michel_data_features.parquet

    # Directory of beam files
    python tag_michel_background.py /path/to/BeamCluster/ \\
        --geo ... --offsets ... --output michel/michel_data_features.parquet
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ambe.mc.cluster_features import compute_cluster_features, load_geometry

# ── Michel_tuning.py cut thresholds ──────────────────────────────────────────
_DIRT_MIN_HITS  = 50
_DIRT_MIN_PE    = 1000.0
_DIRT_MAX_PE    = 4000.0
_DIRT_CB_MAX    = 0.2
_DIRT_CT_MIN    = 200.0
_DIRT_CT_MAX    = 1800.0

_MICHEL_DT_MIN  = 200.0
_MICHEL_DT_MAX  = 5000.0
_MICHEL_MAX_PE  = 650.0
_MICHEL_CB_MAX  = 0.18   # tightened from 0.2 after initial pass, per Michel_tuning.py line 207

_RUN_RE = re.compile(r"R(\d+)")


def _parse_args():
    p = argparse.ArgumentParser(prog="tag_michel_background")
    p.add_argument("input",      help="ROOT file, directory, or glob "
                                      "(e.g. /path/to/ANNIEEvent_dirtmuon_*.root)")
    p.add_argument("--geo",      required=True,
                   help="FullTankPMTGeometry.csv")
    p.add_argument("--offsets",  required=True,
                   help="TankPMTTimingOffsets.csv")
    p.add_argument("--output",   required=True,
                   help="Output .parquet path (e.g. michel/michel_data_features.parquet)")
    p.add_argument("--tree",     default="Event;1",
                   help="ROOT tree name (default: Event;1 for WCSim MC files)")
    p.add_argument("--source-pos", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Source position in metres for d_source feature (optional)")
    return p.parse_args()


def _resolve_files(input_path: str) -> list[Path]:
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


def _run_number(path: Path) -> int:
    m = _RUN_RE.search(path.stem)
    return int(m.group(1)) if m else -1


# ── Selection functions (Michel_tuning.py logic) ──────────────────────────────

def _is_dirt_muon(mrd_activity, tmrd_coinc, no_veto, extended,
                  is_brightest, hits, pe, cb, ct, hit_z, hit_pe) -> bool:
    """Dirt muon selection — mirrors Michel_tuning.py dirt()."""
    if mrd_activity == 1:
        return False
    if tmrd_coinc == 1:
        return False
    if no_veto == 1:
        return False
    if extended == 0:
        return False
    if is_brightest == 0:
        return False
    if hits < _DIRT_MIN_HITS:
        return False
    if not (_DIRT_MIN_PE < pe < _DIRT_MAX_PE):
        return False
    if cb > _DIRT_CB_MAX or cb < 0:
        return False
    if ct > _DIRT_CT_MAX or ct < _DIRT_CT_MIN:
        return False
    bary = float(np.dot(np.asarray(hit_z, dtype=float),
                        np.asarray(hit_pe, dtype=float)))
    return bary >= 0


def _is_michel(adj_time: float, pe: float, hits: int, cb: float) -> bool:
    """Michel candidate selection — mirrors Michel_tuning.py Michel() + CB<0.18 tightening."""
    if not (_MICHEL_DT_MIN < adj_time < _MICHEL_DT_MAX):
        return False
    if pe <= 0 or pe >= _MICHEL_MAX_PE:
        return False
    if hits < 20:
        return False
    if cb >= _MICHEL_CB_MAX or cb <= 0:
        return False
    return True


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(root_path: Path, geo, source_pos_m,
                     tree_name: str = "Event;1") -> list[dict]:
    """
    Apply strict Michel_tuning.py selection to one WCSim ANNIEEvent ROOT file
    and extract per-cluster physics features with compute_cluster_features().

    Reads WCSim Event;1 branch names (clusterTime, clusterPE, clusterChargeBalance,
    clusterHits, Cluster_HitX/Y/Z/T/PE/Chankey) and event-level flags
    (HasMRD, TankMRDCoinc, NoVeto, Extended).

    Returns a list of feature dicts (one per Michel cluster), each containing
    all cluster_features.py columns plus run, event_number, muon_event_idx,
    adj_time_ns, and is_michel=1.
    """
    run = _run_number(root_path)
    rows = []

    with uproot.open(str(root_path)) as f:
        # Use the latest cycle if tree_name not found directly
        tree = f[tree_name]
        n = tree.num_entries

        # Event-level flags (one value per event)
        has_mrd_arr  = tree["HasMRD"].array(library="np")
        tmrd_arr     = tree["TankMRDCoinc"].array(library="np")
        noveto_arr   = tree["NoVeto"].array(library="np")
        extended_arr = tree["Extended"].array(library="np")

        # Per-event cluster arrays (WCSim branch names)
        ct_arr   = tree["clusterTime"].array(library="ak")
        cpe_arr  = tree["clusterPE"].array(library="ak")
        ch_arr   = tree["clusterHits"].array(library="ak")
        ccb_arr  = tree["clusterChargeBalance"].array(library="ak")

        # Doubly-nested hit arrays: (event, cluster, hit)
        hx_arr   = tree["Cluster_HitX"].array(library="ak")
        hy_arr   = tree["Cluster_HitY"].array(library="ak")
        hz_arr   = tree["Cluster_HitZ"].array(library="ak")
        ht_arr   = tree["Cluster_HitT"].array(library="ak")
        hpe_arr  = tree["Cluster_HitPE"].array(library="ak")
        hid_arr  = tree["Cluster_HitChankey"].array(library="ak")

    n_dirt = 0
    n_michel = 0

    for i in range(n):
        # Event-level flag cuts (strict Michel_tuning.py dirt() requirements)
        if int(has_mrd_arr[i])  != 0: continue
        if int(tmrd_arr[i])     != 0: continue
        if int(noveto_arr[i])   != 0: continue
        if int(extended_arr[i]) != 1: continue

        ct  = ak.to_list(ct_arr[i])
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])

        if not cpe:
            continue

        # Find the dirt muon cluster — brightest cluster passing all muon cuts
        muon_idx = None
        for j in range(len(cpe)):
            hz_j  = ak.to_list(hz_arr[i][j])
            hpe_j = ak.to_list(hpe_arr[i][j])
            if _is_dirt_muon(
                mrd_activity=int(has_mrd_arr[i]),
                tmrd_coinc=int(tmrd_arr[i]),
                no_veto=int(noveto_arr[i]),
                extended=int(extended_arr[i]),
                is_brightest=1,   # already filtered at event level; use max-PE cluster
                hits=int(ch[j]),
                pe=float(cpe[j]),
                cb=float(ccb[j]),
                ct=float(ct[j]),
                hit_z=hz_j,
                hit_pe=hpe_j,
            ):
                muon_idx = j
                break

        if muon_idx is None:
            continue
        n_dirt += 1

        muon_t = float(ct[muon_idx])

        # Find Michel candidates in subsequent clusters of the same event
        for k in range(len(cpe)):
            if k == muon_idx:
                continue
            adj_time = float(ct[k]) - muon_t
            if not _is_michel(adj_time, float(cpe[k]), int(ch[k]), float(ccb[k])):
                continue

            hx  = np.asarray(ak.to_list(hx_arr[i][k]),  dtype=float)
            hy  = np.asarray(ak.to_list(hy_arr[i][k]),  dtype=float)
            hz  = np.asarray(ak.to_list(hz_arr[i][k]),  dtype=float)
            ht  = np.asarray(ak.to_list(ht_arr[i][k]),  dtype=float)
            hpe = np.asarray(ak.to_list(hpe_arr[i][k]), dtype=float)
            hid = np.asarray(ak.to_list(hid_arr[i][k]), dtype=int)

            if len(ht) == 0:
                continue

            df_cl = pd.DataFrame({"x": hx, "y": hy, "z": hz,
                                  "t": ht, "pe": hpe, "pmtID": hid})
            feats = compute_cluster_features(df_cl, geo, source_pos_m=source_pos_m)
            if not feats:
                continue

            feats["run"]            = run
            feats["event_number"]   = int(i)
            feats["muon_event_idx"] = int(muon_idx)
            feats["adj_time_ns"]    = round(adj_time, 1)
            feats["is_michel"]      = 1
            rows.append(feats)
            n_michel += 1

    print(f"  {root_path.name}: {n} events → {n_dirt} dirt muons → {n_michel} Michel clusters")
    return rows


def main():
    args = _parse_args()

    source_pos_m = (np.array([float(v) for v in args.source_pos])
                    if args.source_pos else None)
    if source_pos_m is not None:
        print(f"Source position: {tuple(source_pos_m)} m  (d_source enabled)")

    geo = load_geometry(args.geo, args.offsets)

    files = _resolve_files(args.input)
    print(f"Found {len(files)} ROOT file(s):")
    for f in files:
        print(f"  {f.name}")

    all_rows: list[dict] = []
    for i, f in enumerate(files):
        print(f"[michel] [{i+1}/{len(files)}] {f.name}")
        all_rows.extend(extract_features(f, geo, source_pos_m,
                                         tree_name=args.tree))

    df = pd.DataFrame(all_rows)
    n = len(df)
    print(f"\nTotal Michel clusters extracted: {n}")
    if n > 0:
        print(f"adj_time_ns:  min={df['adj_time_ns'].min():.0f}  "
              f"median={df['adj_time_ns'].median():.0f}  "
              f"max={df['adj_time_ns'].max():.0f} ns")
        print(f"n_hits:       min={int(df['n_hits'].min())}  "
              f"median={df['n_hits'].median():.0f}  "
              f"max={int(df['n_hits'].max())}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {n} rows → {out}")


if __name__ == "__main__":
    main()
