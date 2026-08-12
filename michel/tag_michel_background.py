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
    p.add_argument("--mode",       default="dirt", choices=["dirt", "electron"],
                   help="dirt: full Michel_tuning.py selection (find muon, then Michel "
                        "with Δt cut). electron: pure-electron MC — skip all muon cuts, "
                        "apply only Michel cluster cuts (PE, hits, CB) to every cluster.")
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

def process_file(root_path: Path, tree_name: str) -> tuple[list[pd.DataFrame], dict]:
    """Apply Michel selection to one ROOT file.

    Returns hit DataFrames and a cutflow dict for summary printing.
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

    # cutflow counters
    cf = dict(total=n, clustered=0, no_mrd_veto=0, pe_range=0,
              extended_secondary=0, bary=0, hits=0, cb_dirt=0,
              michel_dt=0, michel_cb_tight=0)

    n_michel = 0

    for i in range(n):
        ct  = ak.to_list(ct_arr[i])
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])

        if not any(h >= 5 for h in ch):
            continue
        cf["clustered"] += 1

        if tmrd_arr[i] != 0 or noveto_arr[i] != 0:
            continue
        cf["no_mrd_veto"] += 1

        if not any(_DIRT_MIN_PE < p < _DIRT_MAX_PE for p in cpe):
            continue
        cf["pe_range"] += 1

        if extended_arr[i] != 1 or noc_arr[i] <= 1:
            continue
        cf["extended_secondary"] += 1

        muon_idx = max(range(len(cpe)), key=lambda j: cpe[j])
        if not (_DIRT_MIN_PE < cpe[muon_idx] < _DIRT_MAX_PE):
            continue

        hz_list  = ak.to_list(hz_arr[i][muon_idx])
        hpe_list = ak.to_list(hpe_arr[i][muon_idx])
        if sum(float(z) * float(p) for z, p in zip(hz_list, hpe_list)) < 0:
            continue
        cf["bary"] += 1

        if ch[muon_idx] <= _DIRT_MIN_HITS:
            continue
        cf["hits"] += 1

        if ccb[muon_idx] > _DIRT_CB_MAX or ccb[muon_idx] < 0:
            continue
        cf["cb_dirt"] += 1  # dirt muon found

        muon_t = ct[muon_idx]

        # Michel candidates
        for k in range(len(cpe)):
            if k == muon_idx:
                continue
            adj_time = ct[k] - muon_t
            if not _is_michel(adj_time, cpe[k], ch[k], ccb[k]):
                continue
            cf["michel_dt"] += 1

            if ccb[k] >= _MICHEL_CB_TIGHT:
                continue
            cf["michel_cb_tight"] += 1

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

    print(f"  {root_path.name}: {n} events → {cf['cb_dirt']} dirt muons → {n_michel} Michel clusters")
    return hit_frames, cf


def process_file_electron(root_path: Path, tree_name: str) -> tuple[list[pd.DataFrame], dict]:
    """Apply Michel cluster cuts ONLY (no muon-finding) to a pure-electron MC file.

    Every cluster in every event is treated as a Michel candidate. The dirt-muon
    cuts and the Δt-relative-to-muon cut are skipped entirely. Cuts applied:
        0 < clusterPE < 650
        clusterHits >= 20
        0 < clusterChargeBalance < 0.18
    """
    run = _run_number(root_path)
    hit_frames = []

    with uproot.open(str(root_path)) as f:
        tree = f[tree_name]
        n = tree.num_entries

        cpe_arr  = tree["clusterPE"].array(library="ak")
        ch_arr   = tree["clusterHits"].array(library="ak")
        ccb_arr  = tree["clusterChargeBalance"].array(library="ak")
        hx_arr   = tree["Cluster_HitX"].array(library="ak")
        hy_arr   = tree["Cluster_HitY"].array(library="ak")
        hz_arr   = tree["Cluster_HitZ"].array(library="ak")
        ht_arr   = tree["Cluster_HitT"].array(library="ak")
        hpe_arr  = tree["Cluster_HitPE"].array(library="ak")
        hid_arr  = tree["Cluster_HitChankey"].array(library="ak")

    cf = dict(total=n, any_cluster=0, pe_range=0, hits=0, cb_tight=0)
    n_michel = 0

    for i in range(n):
        cpe = ak.to_list(cpe_arr[i])
        ch  = ak.to_list(ch_arr[i])
        ccb = ak.to_list(ccb_arr[i])

        if len(cpe) == 0:
            continue
        cf["any_cluster"] += 1

        for k in range(len(cpe)):
            pe, hits, cb = float(cpe[k]), int(ch[k]), float(ccb[k])

            if pe <= 0 or pe >= _MICHEL_MAX_PE:
                continue
            cf["pe_range"] += 1

            if hits < _MICHEL_MIN_HITS:
                continue
            cf["hits"] += 1

            if cb <= 0 or cb >= _MICHEL_CB_TIGHT:
                continue
            cf["cb_tight"] += 1

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
            df_cl["muon_event_idx"] = -1            # no muon in electron-only sample
            df_cl["cluster_id"]     = n_michel
            df_cl["is_background"]  = 0             # signal Michels, not background
            hit_frames.append(df_cl)
            n_michel += 1

    print(f"  {root_path.name}: {n} events → {n_michel} Michel clusters")
    return hit_frames, cf


def _print_cutflow_electron(cutflow: dict) -> None:
    total = cutflow["total"]
    print(f"\n{'Selection cut':<52} {'Clusters':>9}  {'% events':>9}")
    print("-" * 80)
    rows = [
        ("Total events",                                     total),
        ("Events with ≥1 cluster",                           cutflow["any_cluster"]),
        (f"0 < clusterPE < {_MICHEL_MAX_PE:.0f}",            cutflow["pe_range"]),
        (f"clusterHits >= {_MICHEL_MIN_HITS}",               cutflow["hits"]),
        (f"0 < CB < {_MICHEL_CB_TIGHT}  → Michel candidates", cutflow["cb_tight"]),
    ]
    for label, count in rows:
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {label:<50} {count:>9}  {pct:>8.1f}%")


def _print_cutflow(cutflow: dict) -> None:
    total = cutflow["total"]
    dirt  = cutflow["cb_dirt"]
    rows = [
        ("Selection cut",                                    "Events", "% of total", "% of prev"),
        ("-" * 50,                                           "-" * 8,  "-" * 10,     "-" * 10),
        ("Total",                                            total,    100.0,         100.0),
        ("Clustered (≥5 PMT hits)",                          cutflow["clustered"],        None, None),
        ("No MRD + FMV coincidence",                         cutflow["no_mrd_veto"],      None, None),
        ("1000 < cluster charge < 4000 pe",                  cutflow["pe_range"],         None, None),
        ("Prompt (brightest), extended, secondary cluster",   cutflow["extended_secondary"],None, None),
        ("Charge barycenter downstream",                     cutflow["bary"],             None, None),
        (f"> {_DIRT_MIN_HITS} PMT hits",                     cutflow["hits"],             None, None),
        ("Charge balance < 0.2  →  dirt muon candidates",   cutflow["cb_dirt"],          None, None),
        (f"0.2 µs < Δt < 5 µs  (Michel, CB<0.2, hits≥{_MICHEL_MIN_HITS})",
                                                             cutflow["michel_dt"],        None, None),
        ("Charge balance < 0.18  →  Michel candidates",     cutflow["michel_cb_tight"],  None, None),
    ]

    print(f"\n{'Selection cut':<52} {'Events':>7}  {'% total':>8}  {'% prev':>8}")
    print("-" * 80)
    prev = total
    for i, row in enumerate(rows):
        if i < 2:
            continue
        label, count = row[0], row[1]
        pct_total = 100.0 * count / total if total > 0 else 0.0
        pct_prev  = 100.0 * count / prev  if prev  > 0 else 0.0
        # Michel rows: % of prev relative to dirt muons, not previous Michel step
        if label.startswith("0.2 µs"):
            pct_prev = 100.0 * count / dirt if dirt > 0 else 0.0
        if label.startswith("Charge balance < 0.18"):
            pct_prev = 100.0 * count / max(cutflow["michel_dt"], 1)
        print(f"  {label:<50} {count:>7}  {pct_total:>7.1f}%  {pct_prev:>7.1f}%")
        prev = count


def main():
    args       = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _resolve_files(args.input)
    print(f"Found {len(files)} file(s)")

    proc_fn = process_file_electron if args.mode == "electron" else process_file

    all_frames: list[pd.DataFrame] = []
    total_cf: dict = {}
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] {f.name}")
        frames, cf = proc_fn(f, tree_name=args.tree)
        all_frames.extend(frames)
        for k, v in cf.items():
            total_cf[k] = total_cf.get(k, 0) + v

    col_order = ["run", "event_number", "muon_event_idx", "cluster_id",
                 "is_background", "x", "y", "z", "t", "pe", "pmtID"]

    if all_frames:
        df = pd.concat(all_frames, ignore_index=True)[col_order]
    else:
        df = pd.DataFrame(columns=col_order)

    n_clusters = int(df.groupby(["run", "event_number", "cluster_id"]).ngroups) if len(df) else 0
    out_name = "michel_signal_hits.parquet" if args.mode == "electron" else "michel_background_hits.parquet"
    out = output_dir / out_name
    df.to_parquet(out, index=False)

    print(f"\n{'='*80}")
    print(f"  CUTFLOW SUMMARY  ({len(files)} file(s), mode={args.mode})")
    print(f"{'='*80}")
    if args.mode == "electron":
        _print_cutflow_electron(total_cf)
    else:
        _print_cutflow(total_cf)
    print(f"\nWrote {n_clusters} Michel clusters ({len(df)} hits) → {out}")


if __name__ == "__main__":
    main()
