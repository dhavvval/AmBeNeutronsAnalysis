"""
tag_michel_background.py

Select Michel electron clusters from WCSim ANNIEEvent ROOT files using the
strict Michel_tuning.py cuts (dirt muon + Michel selection) and write a
background-tagged hits parquet.

Same two-step design as the off-beam pipeline:
    analyze_optics_data_rate.py  →  background_optics_hits.parquet  (hits)
    tag_michel_background.py     →  michel_background_hits.parquet   (hits)
                                        ↓
                          compute features downstream via
                          extract_features_from_background_hits()
                          in src/ambe/mc/cluster_features.py

No geometry files needed — this script only applies selection cuts and
saves raw hit coordinates.

Output
------
michel_background_hits.parquet
    run, event_number, muon_event_idx, cluster_id, is_background=1,
    x, y, z, t, pe, pmtID

Selection cuts (Michel_tuning.py)
----------------------------------
Dirt muon:  Extended==1,
            hits>=50, 1000<PE<4000, CB<0.2,
            charge barycenter downstream (no absolute CT window — WCSim convention)
Michel:     adj_time in (1000, 100000) ns, PE<650, hits>=5, CB<0.20

Usage
-----
    # Single file
    python3 michel/tag_michel_background.py /path/to/ANNIEEvent_dirtmuon_91500_91999.root

    # All files in a directory
    python3 michel/tag_michel_background.py /path/to/dirtmuon/
    
    # Glob
    python3 michel/tag_michel_background.py "/path/to/ANNIEEvent_dirtmuon_*.root"

    # Custom output location
    python3 michel/tag_michel_background.py /path/to/dirtmuon/ --output-dir ambe_output/
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

# ── Cut thresholds ────────────────────────────────────────────────────────────
# Dirt muon: no absolute CT window (WCSim t=0, no beam trigger offset)
_DIRT_MIN_HITS  = 50
_DIRT_MIN_PE    = 1000.0
_DIRT_MAX_PE    = 4000.0
_DIRT_CB_MAX    = 0.2

# Michel: relative timing only; lower hits threshold for WCSim (fewer hits than data)
# CB cut removed — WCSim small clusters (5–33 hits) naturally have CB 0.2–0.6
_MICHEL_DT_MIN  = 1000.0   # ns after muon cluster (µs-scale muon lifetime)
_MICHEL_DT_MAX  = 100000.0 # 100 µs — covers full muon lifetime tail
_MICHEL_MAX_PE  = 650.0
_MICHEL_MIN_HITS = 5        # WCSim produces ~5–33 hits for Michel; data cut was 20

_RUN_RE = re.compile(r"(\d+)")


def _parse_args():
    p = argparse.ArgumentParser(prog="tag_michel_background")
    p.add_argument("input",        help="ROOT file, directory, or glob "
                                        "(e.g. /path/to/ANNIEEvent_dirtmuon_*.root)")
    p.add_argument("--output-dir", default=".",
                   help="Output directory (default: current dir)")
    p.add_argument("--tree",       default="Event;1",
                   help="ROOT tree name (default: Event;1)")
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

def _is_dirt_muon(hits, pe, cb, hit_z, hit_pe) -> bool:
    """Cluster-level dirt muon cuts (no absolute CT window — WCSim has no beam trigger offset)."""
    if hits < _DIRT_MIN_HITS:
        return False
    if not (_DIRT_MIN_PE < pe < _DIRT_MAX_PE):
        return False
    if cb > _DIRT_CB_MAX or cb < 0:
        return False
    bary = float(np.dot(np.asarray(hit_z, dtype=float),
                        np.asarray(hit_pe, dtype=float)))
    return bary >= 0


def _is_michel(adj_time: float, pe: float, hits: int) -> bool:
    if not (_MICHEL_DT_MIN < adj_time < _MICHEL_DT_MAX):
        return False
    if pe <= 0 or pe >= _MICHEL_MAX_PE:
        return False
    if hits < _MICHEL_MIN_HITS:
        return False
    return True


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(root_path: Path, tree_name: str) -> list[pd.DataFrame]:
    """
    Apply Michel selection to one ROOT file.
    Returns a list of per-cluster hit DataFrames tagged with is_background=1.
    """
    run = _run_number(root_path)
    hit_frames = []

    with uproot.open(str(root_path)) as f:
        tree = f[tree_name]
        n = tree.num_entries

        extended_arr = tree["Extended"].array(library="np")
        ct_arr   = tree["clusterTime"].array(library="ak")
        cpe_arr  = tree["clusterPE"].array(library="ak")
        ch_arr   = tree["clusterHits"].array(library="ak")
        ccb_arr  = tree["clusterChargeBalance"].array(library="ak")
        hx_arr   = tree["Cluster_HitX"].array(library="ak")
        hy_arr   = tree["Cluster_HitY"].array(library="ak")
        hz_arr   = tree["Cluster_HitZ"].array(library="ak")
        ht_arr   = tree["Cluster_HitT"].array(library="ak")
        hpe_arr  = tree["Cluster_HitPE"].array(library="ak")
        hid_arr  = tree["Cluster_HitChankey"].array(library="ak")

    n_dirt = 0
    n_michel = 0

    for i in range(n):
        # Event-level flag cuts
        if int(extended_arr[i]) != 1: continue

        ct  = ak.to_list(ct_arr[i])
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])

        if not cpe:
            continue

        # Find dirt muon cluster
        muon_idx = None
        for j in range(len(cpe)):
            if _is_dirt_muon(
                hits=int(ch[j]),
                pe=float(cpe[j]),
                cb=float(ccb[j]),
                hit_z=ak.to_list(hz_arr[i][j]),
                hit_pe=ak.to_list(hpe_arr[i][j]),
            ):
                muon_idx = j
                break

        if muon_idx is None:
            continue
        n_dirt += 1

        muon_t = float(ct[muon_idx])

        # Find Michel candidates
        for k in range(len(cpe)):
            if k == muon_idx:
                continue
            adj_time = float(ct[k]) - muon_t
            if not _is_michel(adj_time, float(cpe[k]), int(ch[k])):
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
            df_cl["run"]            = run
            df_cl["event_number"]   = int(i)
            df_cl["muon_event_idx"] = int(muon_idx)
            df_cl["cluster_id"]     = n_michel
            df_cl["is_background"]  = 1
            hit_frames.append(df_cl)
            n_michel += 1

    print(f"  {root_path.name}: {n} events → {n_dirt} dirt muons → {n_michel} Michel clusters")
    return hit_frames


def main():
    args       = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _resolve_files(args.input)
    print(f"Found {len(files)} file(s)")

    all_frames: list[pd.DataFrame] = []
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] {f.name}")
        all_frames.extend(process_file(f, tree_name=args.tree))

    col_order = ["run", "event_number", "muon_event_idx", "cluster_id",
                 "is_background", "x", "y", "z", "t", "pe", "pmtID"]

    if all_frames:
        df = pd.concat(all_frames, ignore_index=True)[col_order]
    else:
        df = pd.DataFrame(columns=col_order)

    n_clusters = int(df.groupby(["run", "event_number", "cluster_id"]).ngroups) if len(df) else 0
    out = output_dir / "michel_background_hits.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {n_clusters} clusters ({len(df)} hits) → {out}")


if __name__ == "__main__":
    main()
