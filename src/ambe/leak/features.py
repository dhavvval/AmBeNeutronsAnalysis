"""
LAPPD light-leak blind features -- `ambe leak features --config <cfg>`.

WHAT THIS MEASURES. A DC light leak shows up in the UNCLUSTERED hit population, not
the clustered one: ClusterFinder's time-coincidence requirement filters the diffuse
leak out before it reaches cluster-level charge. Two features carry the signal:

  leak_score  fraction of unclustered hit times below LEAK_HALF_WINDOW_NS, i.e. in the
              first half of the 0-70,000 ns acquisition window. 0.5 == perfectly flat
              in time == DC, which is the leak signature. A leak-free run sits
              measurably ABOVE 0.5, because its unclustered hits still track real
              (early, source- and cosmic-driven) tank activity.
  unclustered PE / hit rate per second of livetime, which the leak inflates while
              leaving the CLUSTERED PE rate essentially untouched. That contrast is
              the mechanism check: if clustered rate moves too, it is not the leak.

Validated on run 6249 (confirmed LAPPD ON, leak_score 0.5082, unclustered 61.1 Hz)
against 6250 (confirmed OFF, 0.5350, 21.7 Hz) -- see
/exp/annie/data/users/dajana/LAPPDAmBedebug_PMTcharge/lappd_report_ICgated_clustering.md
-- and re-used for the 22-run scan and the 6254/6256 background pair.

WHY THIS LIVES IN THE PACKAGE. It replaces a family of forked, hardcoded scripts
(blind_feature_extraction.py, blind_feature_extraction_new6.py,
clusters_and_tankcuts_*.py, light_leak_diagnosis.py, ...), each a copy of this
calculation with a different run list pasted in. Here the run list, dataset and
Stage-1 gate all come from the same YAML that drives `ambe data process`, so the leak
numbers and the box-cut numbers are guaranteed to describe the same sample.

EVENT-SELECTION BASIS. The reference tables restrict to Stage-1 IC-gated events, so
this does too. If a run has ZERO IC-passing acquisitions the features are computed
over ALL events instead and the row is flagged in the `basis` column -- that is what
was done for the pulser pair 6254/6256, and it is valid because the DC signature is a
whole-event property, but such a row must NOT be compared numerically against
IC-gated rows.

Output: TriggerSummary/LeakFeatures_<tag>.csv, appended one run at a time and
resumable -- a run already present is skipped, so an interrupted sweep continues
rather than restarting.

Usage:
    ambe leak features --config configs/data_ambe2v5sandi_box.yaml
    ambe leak features --config <cfg> --runs 6273 6275     # subset
    ambe leak features --config <cfg> --refit               # ignore existing rows
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import uproot

# The 0-70,000 ns acquisition window, split in half. A hit population flat in time
# puts half its hits below this, giving leak_score 0.5.
LEAK_HALF_WINDOW_NS = 35000.0

HIT_BRANCHES = ["eventTimeTank", "hitPE", "hitT", "hitChankey",
                "Cluster_HitChankey", "Cluster_HitT", "Cluster_HitPE"]

# Read the hit-level branches in batches. They are jagged and a whole-run read of them
# is several GB; batching keeps this usable on a shared node. Only running scalars and
# two counters are carried across batches, so memory does not grow with run length.
BATCH = 1500


def _runs_from_config(ctx):
    """Run numbers from the Stage-1 BeamCluster directory, same glob as Stage 1."""
    st = ctx.extra.get("stage1", {})
    data_dir = Path(st["data_directory"])
    pat = re.compile(st.get("file_pattern", r"BeamCluster_(\d+)\.root"))
    runs = sorted({m.group(1) for f in data_dir.iterdir()
                   for m in [pat.match(f.name)] if m}, key=int)
    if not runs:
        raise SystemExit(f"no files matching {pat.pattern} in {data_dir}")
    return runs


def _good_events(ctx, run):
    """Stage-1 IC-gated eventTimeTank set for one run.

    Prefers the dump that `ambe data process` writes when
    `stage1.dump_good_events: true`, because recomputing it means re-reading the whole
    waveform tree off /pnfs for a set of integers Stage 1 already had. Falls back to
    recomputing, so this works against a tag processed before that flag existed.
    """
    st = ctx.extra.get("stage1", {})
    tag = st.get("tag", ctx.run_name)
    dump = Path(f"TriggerSummary/GoodEvents_{tag}_{run}.pkl")
    if dump.exists():
        with open(dump, "rb") as fh:
            ge = pickle.load(fh)
        print(f"[leak] run {run}: reusing {dump} ({len(ge)} good events)", flush=True)
        return set(ge), "reused Stage-1 dump"

    print(f"[leak] run {run}: no {dump.name}, recomputing Stage 1 "
          f"(set stage1.dump_good_events: true to avoid this)", flush=True)
    from ..data.processor import (AmBeNeutronProcessing, CutCriteria,
                                 WaveformConfig)
    wf = WaveformConfig(
        pulse_gamma=int(st.get("ic_min", WaveformConfig.pulse_gamma)),
        pulse_max=int(st.get("ic_max", WaveformConfig.pulse_max)),
        selection_mode="ic",
    )
    extra = {int(k): tuple(float(x) for x in v)
             for k, v in (st.get("extra_source_positions") or {}).items()}
    proc = AmBeNeutronProcessing(config=wf, cuts=CutCriteria(), extra_positions=extra)
    res = proc.process_run_waveforms(run, st["waveform_dir"],
                                     campaign=int(st.get("campaign", 1)))
    return set(res["good_events"]), "recomputed Stage 1"


def blind_features(run, dataset, good_set):
    """Clustered/unclustered split and leak_score for one run."""
    f = uproot.open(f"{dataset}/BeamCluster_{run}.root")
    t = f["Event;3"] if "Event;3" in f else f["Event"]

    all_ett = t["eventTimeTank"].array(library="np")
    valid = all_ett[all_ett > 0]
    n_bad = int((all_ett <= 0).sum())
    # Zero-valued eventTimeTank entries are bad/epoch. A naive max-min over the raw
    # array is dominated by them and inflates the livetime to ~56 years.
    livetime = (valid.max() - valid.min()) / 1e9 if len(valid) else float("nan")

    if good_set:
        mask = np.array([int(e) in good_set for e in all_ett])
        basis = "IC-gated"
    else:
        mask = np.ones(len(all_ett), dtype=bool)
        basis = "ALL EVENTS (no IC-passing acquisitions)"

    raw_pe = clus_pe = unclus_pe = 0.0
    n_unclus = n_clus = n_unclus_early = 0

    start = 0
    for arrs in t.iterate(HIT_BRANCHES, library="np", step_size=BATCH):
        for j in range(len(arrs["eventTimeTank"])):
            if not mask[start + j]:
                continue
            hck, ht_, hpe = arrs["hitChankey"][j], arrs["hitT"][j], arrs["hitPE"][j]

            # A hit is "clustered" if its (channel, time, PE) triple appears in any
            # cluster of this event. Rounded to 6 dp because the cluster branches
            # carry the same float through a separate write.
            clus_keys = set()
            for c in range(len(arrs["Cluster_HitChankey"][j])):
                c_ck = np.asarray(arrs["Cluster_HitChankey"][j][c])
                c_t = np.asarray(arrs["Cluster_HitT"][j][c])
                c_pe = np.asarray(arrs["Cluster_HitPE"][j][c])
                clus_keys.update(zip(c_ck.tolist(), np.round(c_t, 6).tolist(),
                                     np.round(c_pe, 6).tolist()))

            raw_pe += hpe.sum()
            for k in range(len(hck)):
                key = (hck[k], round(float(ht_[k]), 6), round(float(hpe[k]), 6))
                if key in clus_keys:
                    clus_pe += hpe[k]
                    n_clus += 1
                else:
                    unclus_pe += hpe[k]
                    n_unclus += 1
                    if ht_[k] < LEAK_HALF_WINDOW_NS:
                        n_unclus_early += 1
        start += len(arrs["eventTimeTank"])
        del arrs

    return dict(
        run=run, basis=basis, n_events_used=int(mask.sum()),
        n_total_events=len(all_ett), n_bad_eventTimeTank=n_bad,
        livetime_sec=livetime,
        raw_PE_total=raw_pe, clustered_PE_total=clus_pe,
        unclustered_PE_total=unclus_pe,
        clustered_fraction=clus_pe / raw_pe if raw_pe > 0 else float("nan"),
        unclustered_fraction=unclus_pe / raw_pe if raw_pe > 0 else float("nan"),
        n_clustered_hits=n_clus, n_unclustered_hits=n_unclus,
        clustered_PE_rate_Hz=clus_pe / livetime if livetime > 0 else float("nan"),
        unclustered_PE_rate_Hz=unclus_pe / livetime if livetime > 0 else float("nan"),
        unclustered_hit_rate_Hz=n_unclus / livetime if livetime > 0 else float("nan"),
        leak_score=(n_unclus_early / n_unclus) if n_unclus else float("nan"),
    )


def run(ctx, argv=None):
    ap = argparse.ArgumentParser(prog="ambe leak features")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="subset of runs; default is every run Stage 1 covers")
    ap.add_argument("--refit", action="store_true",
                    help="recompute runs already present in the output CSV")
    args = ap.parse_args(argv or [])

    st = ctx.extra.get("stage1", {})
    tag = st.get("tag", ctx.run_name)
    dataset = st["waveform_dir"]
    out = Path(f"TriggerSummary/LeakFeatures_{tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = [str(r) for r in (args.runs or _runs_from_config(ctx))]

    done = set()
    if out.exists() and not args.refit:
        with open(out) as fh:
            done = {r["run"] for r in csv.DictReader(fh)}
    todo = [r for r in runs if r not in done]
    if done:
        print(f"[leak] {out.name} already has {sorted(done)}", flush=True)
    if not todo:
        print("[leak] nothing to do", flush=True)
        return 0

    print(f"[leak] tag {tag}, dataset {dataset}", flush=True)
    print(f"[leak] leak_score = fraction of unclustered hit times < "
          f"{LEAK_HALF_WINDOW_NS:.0f} ns; 0.5 == flat in time == DC leak", flush=True)

    for r in todo:
        good, how = _good_events(ctx, r)
        row = blind_features(r, dataset, good)
        row["good_events_source"] = how
        print(f"[leak] run {r} [{row['basis']}]: n_ev={row['n_events_used']}, "
              f"livetime={row['livetime_sec']:.0f}s, "
              f"unclus_frac={row['unclustered_fraction']:.4f}, "
              f"unclus_PE_rate={row['unclustered_PE_rate_Hz']:.2f}Hz, "
              f"clus_PE_rate={row['clustered_PE_rate_Hz']:.2f}Hz, "
              f"leak_score={row['leak_score']:.4f}", flush=True)

        write_header = not out.exists()
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    print(f"[leak] wrote {out}", flush=True)
    return 0


def cli(ctx, argv=None):
    return run(ctx, argv)
