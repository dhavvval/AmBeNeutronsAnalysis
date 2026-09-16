#!/usr/bin/env python3
"""Extract ClusterFinder features from waveform-gated real-data ROOT files.

Run from the repository root: python -m scripts.workflows.extract_cf_features_beamcluster_data
"""
import sys
import os
import re
import glob
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# Reuse the OPTICS data script's helpers verbatim so DATA conventions are identical
# (geometry build, per-run source/port maps, gating, cosmic-veto constants, the
# legacy charge balance, Stage-1 preselection, file resolution, run parsing).
from scripts.workflows.analyze_optics_beamcluster_data import (
    load_file, load_geometry, charge_balance_legacy, passes_preselection,
    port_name, run_number_from_path, build_gate_sets, resolve_files,
    SOURCE_POSITIONS, MIN_SAMPLES, COSMIC_CT_THRESHOLD, COSMIC_PE_THRESHOLD,
)
from ambe.mc.optics import assign_clusterfinder_labels
from ambe.mc.cluster_features import compute_cluster_features


MC_LABEL_RE = re.compile(r"^port(?P<port>\d+)_(?P<z>z(?:minus)?\d+)QE")


def mc_run_and_label(path: Path, label_index: dict) -> tuple:
    """(run, label) for an AmBe MC ROOT file. run = label_index*1000 + file_index."""
    label = path.parent.name
    if label not in label_index:
        raise SystemExit(f"[cf-features] {path}: label {label!r} not in the label map")
    m = re.search(r"_(\d+)_(\d+)\.root$", path.name)
    if not m:
        raise SystemExit(f"[cf-features] cannot parse file index from {path}")
    return label_index[label] * 1000 + int(m.group(1)) // 1000, label


def load_mc_label_index(label_map_path: str) -> dict:
    idx = {}
    with open(label_map_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            i, lab = line.split()
            idx[lab] = int(i)
    if not idx:
        raise SystemExit(f"[cf-features] no labels parsed from {label_map_path}")
    return idx


def _mc_events(fp, run, max_events):
    """load_file, with the MC run number and event key substituted in.

    Deliberately a wrapper here rather than a change to
    analyze_optics_beamcluster_data.load_file: that function is the DATA convention
    and is shared with the OPTICS extractor.
    """
    for ev in load_file(Path(fp), max_events=max_events, gate_set=None):
        if ev["event_tank_time"] != 0:
            raise SystemExit(
                f"[cf-features] {fp}: eventTimeTank is not 0, so substituting "
                f"eventNumber would discard a real event key. Not the AmBe MC.")
        ev["run"] = run
        ev["event_tank_time"] = ev["event_number"]
        yield ev


def extract_cf_features_one_file(fp, geo_path, offsets_path, max_events: int = 0,
                                 max_hits: int = 1500, gate_set=None,
                                 prompt_window_ns: float = 2000.0,
                                 use_source_pos: bool = False,
                                 input_kind: str = "data",
                                 mc_run: int = None, mc_label: str = None,
                                 cosmic_veto: bool = True) -> pd.DataFrame:
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
    if input_kind == "mc":
        run, port = int(mc_run), (mc_label or "mc")
        src_cm = None
        events = _mc_events(fp, run, max_events)
    else:
        run  = run_number_from_path(Path(fp))
        port = port_name(run)
        src_cm = SOURCE_POSITIONS.get(run)
        events = load_file(Path(fp), max_events=max_events, gate_set=gate_set)
    # source_pos_m=None matches the MC training (cc_neutrino / lucho config used
    # source_pos_m=None → no source-ToF, d_source/d_source_fit are NaN → dropped).
    # Keep data identical unless use_source_pos is explicitly enabled.
    src_m = (np.asarray(src_cm, dtype=float) / 100.0
             if (use_source_pos and src_cm is not None) else None)

    rows = []
    n_cosmic = 0
    for ev in events:
        df_hits, df_cf = ev["df_hits"], ev["df_cf"]
        nh = len(df_hits)
        if max_hits and nh > max_hits:
            continue                          # cosmic shower — skip (as in OPTICS path)
        if cosmic_veto and df_cf is not None and len(df_cf) > 0:
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
            cf = df_cf.iloc[int(cid)]
            row = {
                "run": run, "port": port,
                "event_number": ev["event_number"],
                "event_tank_time": ev["event_tank_time"],
                "cluster_id": int(cid),
                "clusterTime_earliest": t_earliest,
                "cf_clusterTime": float(cf["clusterTime"]),
                "cf_clusterPE": float(cf["clusterPE"]),
                "cf_clusterCB": float(cf["clusterCB"]),
                "cf_clusterHits": int(cf["clusterHits"]),
                # The pipeline's `clusterNumber`: for the BeamCluster tree it reads
                # `numberOfClusters`, the per-EVENT cluster count, and splits its
                # candidates into single (== 1) and multiple (!= 1). `n_cf_raw` is
                # that same branch, taken from load_file.
                "cf_clusterNumber": int(ev["n_cf_raw"]),
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
                              use_source_pos: bool = False,
                              input_kind: str = "data", mc_index: dict = None,
                              cosmic_veto: bool = True):
    """
    Extract CF features over all (gated) files and write one parquet. One process
    per file (the per-cluster vertex fit is CPU-bound), per-run shards for resume.
    Mirrors analyze_optics_beamcluster_data.run_feature_extraction.
    """
    shard_dir = outdir / "_cf_feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # One place that decides a file's run number and display label, so the shard name,
    # the submitted job and the log line can never disagree.
    def _run_label(fp):
        if input_kind == "mc":
            return mc_run_and_label(Path(fp), mc_index)
        r = run_number_from_path(fp)
        return r, port_name(r)

    todo = []
    for fp in files:
        run, _lab = _run_label(fp)
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
        run, lab = _run_label(fp)
        gate_set = gate_sets.get(run) if gate_sets else None
        return ex.submit(extract_cf_features_one_file, str(fp), geo_path, offsets_path,
                         max_events, max_hits, gate_set, prompt_window_ns, use_source_pos,
                         input_kind, run, lab, cosmic_veto)

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut2fp = {_submit(ex, fp): fp for fp in todo}
        for fut in as_completed(fut2fp):
            fp = fut2fp[fut]; run, port_lab = _run_label(fp)
            n_done += 1; el = (time.time() - t0) / 60
            try:
                df = fut.result()
            except Exception as e:
                print(f"  [{n_done}/{len(todo)} FAILED, {el:.1f} min] R{run} "
                      f"[{port_lab}]: {e!r} — skipped", flush=True)
                continue
            if len(df):
                df.to_parquet(shard_dir / f"{run_name}__cf_shard_{run}.parquet", index=False)
            print(f"  [{n_done}/{len(todo)} done, {el:.1f} min] R{run} [{port_lab}]: "
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
    p.add_argument("input", help="BeamCluster file, glob, or directory of BeamCluster_*.root; "
                                 "for --input-kind mc, the AmBe MC production root directory")
    # Required, no default. Getting this wrong is silent, not loud: on MC the data path
    # would give every file run == -1 (run_number_from_path's fallback), the parquet
    # would look correct, and the MVA would report 0 % coverage instead of an error.
    p.add_argument("--input-kind", required=True, choices=["data", "mc"],
                   help="'data' = BeamCluster real-data ntuples; 'mc' = AmBe MC production")
    p.add_argument("--mc-label-map", default=None,
                   help="label_map.txt from the MC production (required for --input-kind mc)")
    p.add_argument("--cosmic-veto", default=None, choices=["on", "off"],
                   help="MC only, required there: apply the <2us / >100PE prompt veto. "
                        "MUST match the --cosmic-veto the deck was processed with.")
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
    a = p.parse_args()
    if a.input_kind == "mc":
        if not a.mc_label_map:
            p.error("--input-kind mc requires --mc-label-map")
        if a.cosmic_veto is None:
            p.error("--input-kind mc requires --cosmic-veto {on,off}; it must match the "
                    "veto the deck was processed with")
        if a.gate_csv_dir:
            p.error("--gate-csv-dir is a waveform gate and MC has no waveforms")
        if a.use_source_pos:
            # The frozen model was fitted with source_pos_m=None, so d_source and
            # d_source_fit are NaN -> median-imputed. Supplying the (known) MC source
            # positions would push two features off the training distribution. We are
            # scoring with that model, not refitting one, so its convention rules.
            p.error("--use-source-pos is refused under --input-kind mc: the frozen "
                    "model was trained with source_pos_m=None and we score with it "
                    "unchanged")
    return a


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mc_index = None
    if args.input_kind == "mc":
        mc_index = load_mc_label_index(args.mc_label_map)
        files = sorted(Path(p) for p in
                       glob.glob(os.path.join(args.input, "*", "ANNIEEvent_MC_AmBe_*.root")))
        if not files:
            raise FileNotFoundError(f"No AmBe MC ROOT files under {args.input}")
        print(f"Found {len(files)} AmBe MC file(s) over {len(mc_index)} labels.")
        print(f"[cf-features] MC mode: event key = eventNumber (eventTimeTank is 0), "
              f"cosmic veto {args.cosmic_veto.upper()}")
        print("[cf-features] NO TRAINING HAPPENS HERE. These features exist only to be "
              "scored by the frozen CCinc-neutrino model.")
    else:
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
                              use_source_pos=args.use_source_pos,
                              input_kind=args.input_kind, mc_index=mc_index,
                              cosmic_veto=(args.cosmic_veto != "off"))


if __name__ == "__main__":
    main()
