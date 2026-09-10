#!/usr/bin/env python3
"""
reprocess_ambe_v1.py

Reprocess the AmBe v1 waveforms from scratch and write a FRESH, MINIMALLY-GATED
cluster table — the COMMON input shared by both selections. Per user decision the
ONLY gates applied here are:

  1. IC waveform gate (v1: pulse_gamma=400, pulse_max=575)  -> good_events
  2. Cosmic veto: an event with ANY cluster having clusterTime < 2000 ns OR
     clusterPE > 100 is dropped entirely (matches AmBeNeutronProcessing.cosmic_cut)

NO ambe_single_cut. NO PE/CB/nHits/clusterNumber cut. EVERY surviving ClusterFinder
cluster from every gated, non-cosmic event is written. The two selections are then
applied strictly downstream, separately:
  Selection 1 : OPTICS re-cluster -> frozen MVA
  Selection 2 : these ClusterFinder clusters -> legacy box-cut (PE<80/CB<0.45/nHits>9)

Output: one parquet per run + a combined parquet, columns:
  run, eventID, eventTankTime, sourceX/Y/Z, cluster_idx,
  clusterTime, clusterPE, clusterChargeBalance, clusterHits
(plus an event-count summary CSV).

Usage (myboy venv):
  python reprocess_ambe_v1.py --tag v1_raw \
      --waveform-dir /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1 \
      --beamcluster-dir /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1 \
      --out-dir /exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/ambe_data/cf_raw \
      --runs 4499
"""
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.data.processor import AmBeNeutronProcessing, WaveformConfig, CutCriteria

# The 31 runs present in ambe_all__data_features__scored.parquet (the clean OPTICS set).
TARGET_RUNS = [4499, 4505, 4506, 4507, 4508, 4589, 4590, 4593, 4596, 4598, 4603,
               4625, 4633, 4635, 4636, 4640, 4646, 4649, 4651, 4656, 4658, 4660,
               4662, 4664, 4665, 4666, 4667, 4670, 4678, 4683, 4687]


def process_run_raw(proc: AmBeNeutronProcessing, run: int, waveform_dir: str,
                    beamcluster_dir: str, campaign: int, which_tree: int) -> pd.DataFrame:
    """IC-gate + cosmic veto only; keep EVERY ClusterFinder cluster of surviving events."""
    src = proc.source_positions.get(run, (np.nan, np.nan, np.nan))

    # Stage 1 — canonical IC waveform gate (unchanged production code).
    wf = proc.process_run_waveforms(str(run), waveform_dir, campaign,
                                    save_waveform_samples=False)
    good = wf["good_events"]
    print(f"[run {run}] good_events (IC-gated) = {len(good)}")

    bc = Path(beamcluster_dir) / f"BeamCluster_{run}.root"
    if not bc.exists():
        print(f"[run {run}] no {bc.name}; skipping")
        return pd.DataFrame()
    ev = proc.load_event_data(str(bc), which_tree=which_tree)

    EN, ETT = ev["eventNumber"], ev["eventTimeTank"]
    CT, CPE, CCB, CH = ev["clusterTime"], ev["clusterPE"], ev["clusterChargeBalance"], ev["clusterHits"]
    # numberOfClusters per event (ClusterFinder), loaded as "clusterNumber" by
    # load_event_data when which_tree!=0. This is the per-event scalar that
    # ambe_single_cut/ambe_multiple_cut use (cn==1 single, cn!=1 multiple) — the
    # canonical single/multiple definition in the user's legacy stats table.
    NC = ev.get("clusterNumber")

    rows = []
    n_gated = n_cosmic = 0
    for i in range(len(EN)):
        if ETT[i] not in good:
            continue
        n_gated += 1
        cts, cpes, ccbs, chs = CT[i], CPE[i], CCB[i], CH[i]
        ncl = int(NC[i]) if NC is not None else len(cts)
        # cosmic veto: drop the WHOLE event if any cluster is cosmic-like
        if any(proc.cosmic_cut(float(ct), float(cpe)) for ct, cpe in zip(cts, cpes)):
            n_cosmic += 1
            continue
        for cidx in range(len(cts)):
            rows.append({
                "run": run, "eventID": int(EN[i]), "eventTankTime": int(ETT[i]),
                "sourceX": src[0], "sourceY": src[1], "sourceZ": src[2],
                "cluster_idx": cidx, "numberOfClusters": ncl,
                "clusterTime": float(cts[cidx]), "clusterPE": float(cpes[cidx]),
                "clusterChargeBalance": float(ccbs[cidx]), "clusterHits": int(chs[cidx]),
            })
    print(f"[run {run}] gated events={n_gated}  cosmic-vetoed={n_cosmic}  "
          f"clusters kept={len(rows)} (NO PE/CB/hits cut)")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1_raw")
    p.add_argument("--waveform-dir", required=True)
    p.add_argument("--beamcluster-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--campaign", type=int, default=1)
    p.add_argument("--which-tree", type=int, default=1)
    p.add_argument("--runs", type=int, nargs="*", default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip runs whose per-run parquet already exists (re-run safe)")
    args = p.parse_args()

    runs = args.runs if args.runs else TARGET_RUNS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = AmBeNeutronProcessing(config=WaveformConfig(), cuts=CutCriteria())
    print(f"[reprocess] IC gate v1: pulse_gamma={proc.config.pulse_gamma}, "
          f"pulse_max={proc.config.pulse_max}")
    print(f"[reprocess] cosmic veto: ct<{proc.cuts.cosmic_ct_threshold} ns OR "
          f"PE>{proc.cuts.cosmic_pe_threshold}  |  NO single-neutron cut")
    print(f"[reprocess] {len(runs)} run(s) -> {out_dir}/<tag>_<run>.parquet")

    summary = []
    t0 = time.time()
    for i, run in enumerate(runs, 1):
        run_path = out_dir / f"{args.tag}_{run}.parquet"
        if args.resume and run_path.exists():
            print(f"[reprocess] ({i}/{len(runs)}) run {run}: parquet exists, skipping (--resume)")
            continue
        print(f"[reprocess] ({i}/{len(runs)}) run {run} ...")
        try:
            df = process_run_raw(proc, run, args.waveform_dir, args.beamcluster_dir,
                                 args.campaign, args.which_tree)
        except Exception as e:
            print(f"[reprocess] ERROR on run {run}: {e!r} — continuing")
            continue
        if len(df):
            df.to_parquet(run_path, index=False)

    # (Re)build the combined parquet from ALL per-run files present, so an
    # interrupted+resumed job still produces a complete combined output.
    per_run_paths = sorted(out_dir.glob(f"{args.tag}_[0-9]*.parquet"))
    per_run = [pd.read_parquet(pp) for pp in per_run_paths]
    for df in per_run:
        r = int(df["run"].iloc[0])
        summary.append({"run": r, "n_clusters": len(df),
                        "n_events": df["eventTankTime"].nunique()})

    if per_run:
        comb = pd.concat(per_run, ignore_index=True)
        comb_path = out_dir / f"{args.tag}_all.parquet"
        comb.to_parquet(comb_path, index=False)
        pd.DataFrame(summary).to_csv(out_dir / f"{args.tag}_summary.csv", index=False)
        print(f"\n[reprocess] combined -> {comb_path}  "
              f"({len(comb)} clusters, {comb['eventTankTime'].nunique()} events, "
              f"{comb['run'].nunique()} runs)  in {time.time()-t0:.0f}s")
    else:
        print("[reprocess] no clusters written")


if __name__ == "__main__":
    main()
