#!/usr/bin/env python3
"""
extract_cf_features_beamcluster_data.py
───────────────────────────────────────────────────────────────────────────────
Produce ClusterFinder-METHOD cluster features for AmBe real DATA, so the
ClusterFinder-trained frozen MVA (e.g. mc_lucho_100k_cf_frozen.pkl) has data to
score. The existing analyze_optics_beamcluster_data.py only emits OPTICS-method
features; this is its ClusterFinder twin.

WHY THIS MIRRORS THE MC CF PATH (correctness-critical)
──────────────────────────────────────────────────────
In MC (src/ambe/mc/cluster_features.py:extract_all_features), BOTH methods are
built from the SAME delayed-residual hit population:
  1. pulses are reduced to the delayed residual (apply_residual_filter, prompt
     window removed) BEFORE any clustering;
  2. OPTICS labels and ClusterFinder labels are both assigned on that residual
     population — CF labels via optics.assign_clusterfinder_labels(df_ev, df_cl),
     which maps each residual hit to the nearest CF cluster within (-5, +20) ns of
     its clusterTime;
  3. compute_cluster_features() runs on df_ev[mask] (the residual hits belonging
     to that CF cluster), with source_pos_m (None for the cc_neutrino / lucho
     training config).

So for DATA we reproduce EXACTLY the OPTICS data Stage-A (cosmic veto + prompt
window t>2000 ns), then assign the surviving residual hits to the file's
ClusterFinder clusters with the SAME assign_clusterfinder_labels, and extract the
SAME features. This guarantees CF data features sit on the same footing as the CF
frozen model's training distribution.

We deliberately do NOT read the raw nested Cluster_HitX/Y/Z/T branches: those are
the FULL (pre-residual) CF cluster membership, which the MC CF path never sees.
Re-deriving membership on the residual hits via assign_clusterfinder_labels is the
MC-matched choice.

OUTPUT
──────
Same per-cluster schema as the OPTICS data parquet (run, port, event_number,
event_tank_time, cluster_id, clusterTime_earliest, passes_stage1 + 47 features),
with method="clusterfinder". Per-run shards under <out>/_cf_feature_shards/,
concatenated to <out>/<run_name>__data_features_cf.parquet.

USAGE
─────
  python extract_cf_features_beamcluster_data.py /pnfs/.../AmBe2.0v1/ \
      --run-name ambe_all \
      --output-dir /exp/.../ambe_output/ambe_data \
      --gate-csv-dir /exp/.../AmBeNeutronsAnalysis/EventAmBeNeutronCandidatesData \
      --geometry /exp/.../FullTankPMTGeometry.csv \
      --offsets  /exp/.../TankPMTTimingOffsets.csv \
      --jobs 8
"""
import sys
import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Reuse the OPTICS data script's helpers verbatim so DATA conventions are identical
# (geometry build, per-run source/port maps, gating, cosmic-veto constants, the
# legacy charge balance, Stage-1 preselection, file resolution, run parsing).
from analyze_optics_beamcluster_data import (
    load_file, load_geometry, charge_balance_legacy, passes_preselection,
    port_name, run_number_from_path, build_gate_sets, resolve_files,
    SOURCE_POSITIONS, MIN_SAMPLES, COSMIC_CT_THRESHOLD, COSMIC_PE_THRESHOLD,
)
from ambe.mc.optics import assign_clusterfinder_labels
from ambe.mc.cluster_features import compute_cluster_features


def extract_cf_features_one_file(fp, geo_path, offsets_path, max_events: int = 0,
                                 max_hits: int = 1500, gate_set=None,
                                 prompt_window_ns: float = 2000.0,
                                 use_source_pos: bool = False) -> pd.DataFrame:
    """
    For every (waveform-gated, cosmic-vetoed) event, assign the delayed-residual
    hits (t > prompt_window_ns) to the file's ClusterFinder clusters and compute
    the full MVA feature set per CF cluster. Returns a per-cluster DataFrame with
    method="clusterfinder".

    Mirrors analyze_optics_beamcluster_data.extract_features_one_file step for
    step; the ONLY difference is the clustering step:
      OPTICS twin:  labels = run_optics_on_event(df_residual, ...)
      this (CF):    labels = assign_clusterfinder_labels(df_residual, df_cf)
    """
    geo  = load_geometry(geo_path, offsets_path)
    run  = run_number_from_path(Path(fp))
    port = port_name(run)
    src_cm = SOURCE_POSITIONS.get(run)
    # source_pos_m=None matches the MC training (cc_neutrino / lucho config used
    # source_pos_m=None → no source-ToF, d_source/d_source_fit are NaN → dropped).
    # Keep data identical unless use_source_pos is explicitly enabled.
    src_m = (np.asarray(src_cm, dtype=float) / 100.0
             if (use_source_pos and src_cm is not None) else None)

    rows = []
    n_cosmic = 0
    for ev in load_file(Path(fp), max_events=max_events, gate_set=gate_set):
        df_hits, df_cf = ev["df_hits"], ev["df_cf"]
        nh = len(df_hits)
        if max_hits and nh > max_hits:
            continue                          # cosmic shower — skip (as in OPTICS path)
        # ── Cosmic veto (per-event), identical to the OPTICS data worker ──
        # ANY CF cluster with clusterTime < 2000 ns OR clusterPE > 100 PE → drop event.
        if df_cf is not None and len(df_cf) > 0:
            ct_cf = df_cf["clusterTime"].to_numpy(float)
            pe_cf = df_cf["clusterPE"].to_numpy(float)
            if np.any((ct_cf < COSMIC_CT_THRESHOLD) | (pe_cf > COSMIC_PE_THRESHOLD)):
                n_cosmic += 1
                continue
        # MC-matched prompt-window cut: keep only the delayed residual (t > prompt).
        if prompt_window_ns > 0:
            df_hits = df_hits[df_hits["t"] > prompt_window_ns].reset_index(drop=True)
        nh = len(df_hits)
        if nh < MIN_SAMPLES:
            continue
        if df_cf is None or len(df_cf) == 0:
            continue                          # no CF clusters to map residual hits onto

        # ── CF clustering on the residual hits (the MC-matched step) ──
        # assign_clusterfinder_labels uses df_pulses["t"] and df_clusters["clusterTime"];
        # df_cf carries clusterTime from load_file. Returns one label per residual hit
        # (-1 = unassigned / outside any CF cluster's (-5,+20)ns window).
        labels = assign_clusterfinder_labels(df_hits, df_cf)

        # compute_cluster_features expects a 'pmtID' column (data hits carry 'chankey').
        df_hits = df_hits.rename(columns={"chankey": "pmtID"})
        for cid in (c for c in np.unique(labels) if c >= 0):
            mask = labels == cid
            df_clust = df_hits[mask].reset_index(drop=True)
            if len(df_clust) < 1:
                continue
            feats = compute_cluster_features(df_clust, geo, source_pos_m=src_m)
            if not feats:
                continue
            pe_tot = float(df_clust["pe"].sum())
            cb     = charge_balance_legacy(df_clust["pe"].to_numpy(float),
                                           df_clust["pmtID"].to_numpy(int))
            n_hits = int(mask.sum())
            t_earliest = float(df_clust["t"].min())
            row = {
                "run": run, "port": port,
                "event_number": ev["event_number"],
                "event_tank_time": ev["event_tank_time"],
                "cluster_id": int(cid),
                "clusterTime_earliest": t_earliest,
                "passes_stage1": bool(passes_preselection(pe_tot, cb, n_hits)),
                "method": "clusterfinder",
            }
            row.update(feats)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_cosmic_events_vetoed"] = n_cosmic
    print(f"  [R{run}] cosmic-vetoed events: {n_cosmic}", flush=True)
    return out


def run_cf_feature_extraction(files, outdir: Path, geo_path, offsets_path, gate_sets,
                              max_events: int, max_hits: int, run_name: str,
                              jobs: int = 1, prompt_window_ns: float = 2000.0,
                              use_source_pos: bool = False):
    """
    Extract CF features over all (gated) files and write one parquet. One process
    per file (the per-cluster vertex fit is CPU-bound), per-run shards for resume.
    Mirrors analyze_optics_beamcluster_data.run_feature_extraction.
    """
    shard_dir = outdir / "_cf_feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for fp in files:
        run = run_number_from_path(fp)
        if gate_sets is not None and run not in gate_sets:
            print(f"  R{run}: no gate set — skipped", flush=True)
            continue
        shard = shard_dir / f"{run_name}__cf_shard_{run}.parquet"
        if shard.exists():
            print(f"  R{run}: shard exists — resuming (skip)", flush=True)
            continue
        todo.append(fp)

    print(f"Dispatching {len(todo)} file(s) across {jobs} worker(s) ...", flush=True)
    t0 = time.time()
    n_done = 0

    def _submit(ex, fp):
        run = run_number_from_path(fp)
        gate_set = gate_sets.get(run) if gate_sets else None
        return ex.submit(extract_cf_features_one_file, str(fp), geo_path, offsets_path,
                         max_events, max_hits, gate_set, prompt_window_ns, use_source_pos)

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut2fp = {_submit(ex, fp): fp for fp in todo}
        for fut in as_completed(fut2fp):
            fp = fut2fp[fut]; run = run_number_from_path(fp)
            n_done += 1; el = (time.time() - t0) / 60
            try:
                df = fut.result()
            except Exception as e:
                print(f"  [{n_done}/{len(todo)} FAILED, {el:.1f} min] R{run} "
                      f"[{port_name(run)}]: {e!r} — skipped", flush=True)
                continue
            if len(df):
                df.to_parquet(shard_dir / f"{run_name}__cf_shard_{run}.parquet", index=False)
            print(f"  [{n_done}/{len(todo)} done, {el:.1f} min] R{run} [{port_name(run)}]: "
                  f"{len(df)} CF clusters"
                  + (f"  ({int(df['passes_stage1'].sum())} pass Stage-1)" if len(df) else ""),
                  flush=True)

    shards = sorted(shard_dir.glob(f"{run_name}__cf_shard_*.parquet"))
    frames = [pd.read_parquet(s) for s in shards]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    parquet_path = outdir / f"{run_name}__data_features_cf.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"\n[cf-features] {len(out)} CF clusters over "
          f"{out['run'].nunique() if len(out) else 0} run(s) -> {parquet_path}")
    if len(out):
        print(f"[cf-features] Stage-1 survivors: {int(out['passes_stage1'].sum())}/{len(out)} "
              f"({100*out['passes_stage1'].mean():.1f}%)")
    return parquet_path


def parse_args():
    p = argparse.ArgumentParser(description="Extract ClusterFinder-method cluster "
                                            "features for AmBe DATA from BeamCluster ntuples.")
    p.add_argument("input", help="BeamCluster file, glob, or directory of BeamCluster_*.root")
    p.add_argument("--run-name", default="ambe_all")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gate-csv-dir", default=None,
                   help="dir of Event/PromptAmBeNeutronCandidates_*.csv for waveform-gating "
                        "(must match the OPTICS run's gate dir for identical run coverage)")
    p.add_argument("--geometry", required=True, help="FullTankPMTGeometry.csv")
    p.add_argument("--offsets",  required=True, help="TankPMTTimingOffsets.csv")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--max-hits", type=int, default=1500)
    p.add_argument("--prompt-window-ns", type=float, default=2000.0)
    p.add_argument("--use-source-pos", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = resolve_files(args.input)
    print(f"Found {len(files)} BeamCluster file(s).")

    gate_sets = None
    if args.gate_csv_dir:
        gate_sets = build_gate_sets(args.gate_csv_dir)
        print(f"Waveform gate: loaded accepted eventTimeTank sets for "
              f"{len(gate_sets)} run(s) from {args.gate_csv_dir}")
        gated_files, ungated = [], []
        for fp in files:
            (gated_files if run_number_from_path(fp) in gate_sets else ungated).append(fp)
        if ungated:
            print(f"  NOTE: {len(ungated)} BeamCluster run(s) have NO gate CSV and "
                  f"will be SKIPPED: {[run_number_from_path(f) for f in ungated]}")
        files = gated_files
        if not files:
            raise SystemExit("No BeamCluster runs have a matching gate CSV — nothing to do.")

    # Build geometry once in the parent so a bad path fails fast.
    print(f"[cf-features] geometry: {args.geometry}")
    _ = load_geometry(args.geometry, args.offsets)
    gate_msg = "WAVEFORM-GATED" if gate_sets is not None else "ungated (all events)"
    print(f"[cf-features] extracting ClusterFinder features [{gate_msg}] "
          f"(CF labels via assign_clusterfinder_labels on residual hits; "
          f"prompt_window_ns={args.prompt_window_ns}; hit cap={args.max_hits}) "
          f"over {len(files)} file(s) ...")
    print(f"[cf-features] source_pos: "
          f"{'AmBe per-run (use_source_pos ON)' if args.use_source_pos else 'None (MC-matched)'}")
    run_cf_feature_extraction(files, outdir, args.geometry, args.offsets, gate_sets,
                              max_events=args.max_events, max_hits=args.max_hits,
                              run_name=args.run_name, jobs=args.jobs,
                              prompt_window_ns=args.prompt_window_ns,
                              use_source_pos=args.use_source_pos)


if __name__ == "__main__":
    main()
