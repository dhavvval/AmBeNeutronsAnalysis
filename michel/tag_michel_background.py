"""
tag_michel_background.py

Select Michel electron clusters from WCSim ANNIEEvent ROOT files using the
Michel_tuning.py cut stream, adapted for WCSim timing convention.

Cut stream (Table B.1 from Michel analysis note)
-------------------------------------------------
Dirt muon selection:
    TankMRDCoinc == 0      (no beam-correlated tank-MRD coincidence → no MRD activity)
    NoVeto == 0            (FMV veto fired → FMV coincidence required; muon passed through FMV)
    Extended == 1          (extended readout window, needed to see Michel)
    numberOfClusters > 1   (has at least one secondary cluster)
    isBrightest            (highest-PE cluster in event — proxy for spill window
                            in data; in WCSim CT window is dropped since t=0 is
                            particle origin, not beam trigger)
    clusterHits > 100
    1000 < clusterPE < 4000
    clusterChargeBalance < 0.2
    charge barycenter downstream (sum_i Z_i * PE_i >= 0)

  Note on HasMRD: Michel_tuning.py uses HasMRD==0 on real data to select
  stopping muons. In WCSim the MRD reconstruction tool sets HasMRD=1 for ALL
  events regardless of physics — it is not a useful discriminant in MC. The
  equivalent physics cut in ANNIEEvent MC is TankMRDCoinc==0, which flags
  beam-correlated tank-MRD coincidences and is meaningful in both data and MC.

Michel candidate selection:
    0.2 µs < adj_time < 5 µs  (relative to muon cluster time)
    0 < clusterPE < 650
    clusterHits >= 20
    clusterChargeBalance < 0.2

Post-selection tightening (matching Michel_tuning.py post-processing):
    clusterChargeBalance < 0.18  (applied after primary Michel pass)

Output
------
michel_background_hits.parquet
    run, event_number, muon_event_idx, cluster_id, is_background=1,
    x, y, z, t, pe, pmtID

Usage
-----
    # Single file
    python3 michel/tag_michel_background.py /path/to/ANNIEEvent_dirtmuon_96500_96999.root

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

# ── Cut thresholds (Michel_tuning.py values) ─────────────────────────────────
_DIRT_MIN_HITS  = 100          # thesis Table B.1: "> 100 PMT hits"
_DIRT_MIN_PE    = 1000.0
_DIRT_MAX_PE    = 4000.0
_DIRT_CB_MAX    = 0.2

_MICHEL_DT_MIN  = 200.0     # ns  (0.2 µs)
_MICHEL_DT_MAX  = 5000.0    # ns  (5 µs)
_MICHEL_MAX_PE  = 650.0
_MICHEL_MIN_HITS = 20
_MICHEL_CB_MAX   = 0.2

_MICHEL_CB_TIGHT = 0.18     # post-selection tightening (Michel_tuning.py step 2)

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


# ── Selection functions ───────────────────────────────────────────────────────

def _is_dirt_muon(hits: int, pe: float, cb: float,
                  hit_z: list, hit_pe: list) -> bool:
    """Dirt muon cluster cuts (Michel_tuning.py logic, WCSim-adapted).

    CT window (200–1800 ns) is dropped because WCSim sets t=0 at the particle
    origin, not at the beam trigger. Brightest-cluster selection is applied
    upstream by choosing the highest-PE cluster in the event.
    """
    if hits <= _DIRT_MIN_HITS:
        return False
    if not (_DIRT_MIN_PE < pe < _DIRT_MAX_PE):
        return False
    if cb > _DIRT_CB_MAX or cb < 0:
        return False
    bary = sum(float(z) * float(p) for z, p in zip(hit_z, hit_pe))
    return bary >= 0


def _is_michel(adj_time: float, pe: float, hits: int, cb: float) -> bool:
    """Michel candidate cuts matching Michel_tuning.py primary selection."""
    if not (_MICHEL_DT_MIN < adj_time < _MICHEL_DT_MAX):
        return False
    if pe <= 0 or pe >= _MICHEL_MAX_PE:
        return False
    if hits < _MICHEL_MIN_HITS:
        return False
    if cb >= _MICHEL_CB_MAX or cb <= 0:
        return False
    return True


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(root_path: Path, tree_name: str) -> list[pd.DataFrame]:
    """Apply Michel selection to one ROOT file.

    Returns a list of per-cluster hit DataFrames tagged with is_background=1.
    Only Michel clusters passing the tight CB cut (< 0.18) are kept, matching
    the post-processing step in Michel_tuning.py.
    """
    run = _run_number(root_path)
    hit_frames = []

    with uproot.open(str(root_path)) as f:
        tree = f[tree_name]
        n = tree.num_entries

        tmrd_arr     = tree["TankMRDCoinc"].array(library="np").tolist()
        noveto_arr   = tree["NoVeto"].array(library="np").tolist()
        extended_arr = tree["Extended"].array(library="np").tolist()
        noc_arr      = tree["numberOfClusters"].array(library="np").tolist()

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
        # Event-level flag cuts (Michel_tuning.py: no MRD, no veto, extended,
        # has secondary cluster)
        if tmrd_arr[i] != 0:    continue  # TankMRDCoinc == 0
        if noveto_arr[i] != 0:  continue  # NoVeto == 0
        if extended_arr[i] != 1: continue  # Extended == 1
        if noc_arr[i] <= 1:     continue  # must have secondary cluster

        ct  = ak.to_list(ct_arr[i])
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])
        if not cpe:
            continue

        # isBrightest: pick highest-PE cluster as the muon candidate
        # (in data this is the brightest within the 200–1800 ns spill window;
        # in WCSim all prompt clusters are at ~10–40 ns so we drop the CT window
        # and select the globally brightest cluster)
        muon_idx = max(range(len(cpe)), key=lambda j: cpe[j])

        if not _is_dirt_muon(
            hits=ch[muon_idx],
            pe=cpe[muon_idx],
            cb=ccb[muon_idx],
            hit_z=ak.to_list(hz_arr[i][muon_idx]),
            hit_pe=ak.to_list(hpe_arr[i][muon_idx]),
        ):
            continue
        n_dirt += 1

        muon_t = ct[muon_idx]

        # Michel candidates
        for k in range(len(cpe)):
            if k == muon_idx:
                continue
            adj_time = ct[k] - muon_t
            if not _is_michel(adj_time, cpe[k], ch[k], ccb[k]):
                continue
            # Post-selection tightening (Michel_tuning.py step 2)
            if ccb[k] >= _MICHEL_CB_TIGHT:
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
