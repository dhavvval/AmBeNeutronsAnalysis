"""
analyze_optics_beamcluster_data.py

Apply OPTICS Stage-1 pre-selection to the real AmBe BeamCluster DATA sample and
benchmark "neutron-like clusters per event" against the standard ClusterFinder
(CF) reconstruction.

This is the data-side counterpart to the MC validation done in the rest of the
AmBeNeutronsAnalysis toolkit: the MC study showed OPTICS + a Stage-1
pre-selection reproduces the CF neutron-candidate rate; this script confirms the
same techniques transfer to real data.

It mirrors analyze_optics_data_rate.py (per-event OPTICS, CF comparison, plots,
CSV outputs) but:
  * reads the BeamCluster_<run>.root schema (TTree "Event") instead of the
    off-beam ntuple schema (TTree "data"), and
  * applies the Stage-1 pre-selection cuts (clusterPE<80, clusterChargeBalance<0.45,
    clusterHits>9) used in src/ambe/clustering/optics_analysis.py, reporting BOTH
    raw cluster counts and post-pre-selection ("neutron-like") cluster counts.

Per-event procedure
-------------------
  * OPTICS input  : ALL raw tank hits for the event (hitX/Y/Z/T/PE), CF cluster
                    boundaries ignored.  OPTICS re-clusters them with the same
                    hyperparameters as the MC features config (ms=8, xi=0.10,
                    t_unit=25 ns).  Each label >=0 is one OPTICS cluster.
  * CF clusters   : taken directly from the file's per-cluster branches
                    (clusterPE / clusterChargeBalance / clusterHits).
  * Pre-selection : PE<80 & CB<0.45 & hits>=10, applied to BOTH methods.
      - CF      : uses the ready-made cluster branches.
      - OPTICS  : per cluster, PE  = sum of member hitPE,
                              hits = number of member hits,
                              CB   = legacy ANNIE charge balance computed from
                                     member hits grouped per PMT channel
                                     (charge_bal_legacy convention, see
                                     src/ambe/mc/cluster_features.py:606).

Outputs (--output-dir, default optics_beamcluster_benchmark/)
  optics_beamcluster_stats.csv     per-event: run, port, event_number,
                                   n_cf_raw, n_cf_presel, n_optics_raw,
                                   n_optics_presel, n_hits_total
  optics_beamcluster_perport.csv   per-port + combined: mean/median OPTICS vs CF
                                   (raw + pre-selected)
  optics_beamcluster_summary.txt   headline table
  optics_beamcluster_rate.pdf      distribution + scatter plots

Usage
-----
  source /exp/annie/app/users/dajana/myboy/bin/activate

  # single file (smoke test)
  python analyze_optics_beamcluster_data.py \
      /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/BeamCluster_4604.root

  # all runs in the dataset directory
  python analyze_optics_beamcluster_data.py \
      /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/
"""

import sys
import os
import re
import argparse
import glob
from pathlib import Path
from collections import Counter, defaultdict

import uproot
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.mc.optics import (run_optics_on_event, _prefilter_hits,
                            DEFAULT_PREFILTER)
from ambe.mc.cluster_features import load_geometry, compute_cluster_features
from ambe.data.processor import AmBeNeutronProcessing

# ── OPTICS hyperparameters — mirror mc_lucho_full features config ──────────────
MIN_SAMPLES = 8
XI          = 0.10
T_UNIT_NS   = 25.0

# ── Stage-1 pre-selection cuts (legacy box-cut reference) ─────────────────────
PRESEL_PE_MAX   = 80.0
PRESEL_CB_MAX   = 0.45
PRESEL_HITS_MIN = 9         # clusterHits > 9  → clusterHits >= 10

# ── Cosmic veto — MUST match AmBeNeutronProcessing.cosmic_cut ─────────────────
# (src/ambe/data/processor.py: cosmic_cut → ct < 2000 ns OR cpe > 100 PE; in
# process_events_efficient an event with ANY such ClusterFinder cluster is
# excluded entirely, `break`). We replicate this per-event veto so the data MVA
# sample matches the standard AmBe data processing (cosmic events fully removed).
COSMIC_CT_THRESHOLD = 2000.0   # ns
COSMIC_PE_THRESHOLD = 100.0    # PE

# ── Geometry for MVA feature extraction (--features-out) ──────────────────────
# Same FullTankPMTGeometry/TankPMTTimingOffsets the MC features run uses, so the
# data features land in the same coordinate/offset system as the training MC.
DEFAULT_PMT_GEOMETRY = "/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/FullTankPMTGeometry.csv"
DEFAULT_TIMING_OFFSETS = "/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry/TankPMTTimingOffsets.csv"

DEFAULT_DATASET = "/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v1/"

_RUN_RE = re.compile(r"BeamCluster_(\d+)\.root")

# Source-position → port-name map (positions are (x, y, z) in cm, from
# AmBeNeutronProcessing.source_positions).
SOURCE_POSITIONS = AmBeNeutronProcessing().source_positions


def port_name(run: int) -> str:
    """Human-readable port label for a run, derived from its source position."""
    pos = SOURCE_POSITIONS.get(run)
    if pos is None:
        return "unknown"
    x, y, z = pos
    if (x, y, z) == (0, 328, 0):
        return "outside_tank"
    if x == 75:
        return "port4_x75"
    if z == -75:
        return "port1_z-75"
    if z == 102:
        return "port3_z102"
    if z == 75:
        return "port2_z75"
    if z == 0:
        return "port5_z0"
    return f"x{x}_y{y}_z{z}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default=DEFAULT_DATASET,
                   help="BeamCluster ROOT file, directory, or glob pattern")
    p.add_argument("--output-dir", default="optics_beamcluster_benchmark",
                   help="Directory for outputs")
    p.add_argument("--max-events", type=int, default=0,
                   help="Cap events per file (0 = all); for quick tests")
    p.add_argument("--max-hits", type=int, default=1500,
                   help="Skip OPTICS on events with more than this many raw hits "
                        "(cosmic showers; flagged 'skipped_large', not silently dropped). "
                        "0 = no cap.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of files to process in parallel (process pool). "
                        "I/O is negligible; set near the core count.")
    p.add_argument("--no-resume", action="store_true",
                   help="Do not reuse an existing stats CSV; reprocess everything.")
    p.add_argument("--gate-csv-dir", default=None,
                   help="Waveform-gating: directory of EventAmBeNeutronCandidates_*_<run>.csv "
                        "files.  Only BeamCluster events whose eventTimeTank passed the "
                        "waveform IC cut (i.e. appears in the matching run's candidate CSV) "
                        "are processed.  Reuses ONLY the accepted-timestamp list, not the "
                        "old cluster processing.")
    # ── MVA feature-extraction mode ──────────────────────────────────────────
    p.add_argument("--features-out", action="store_true",
                   help="Instead of the count benchmark, extract the full MVA feature set "
                        "(compute_cluster_features) for every OPTICS cluster and write a "
                        "<run-name>__data_features.parquet for scoring with a frozen model.")
    p.add_argument("--geometry", default=DEFAULT_PMT_GEOMETRY,
                   help="FullTankPMTGeometry.csv for feature extraction.")
    p.add_argument("--offsets", default=DEFAULT_TIMING_OFFSETS,
                   help="TankPMTTimingOffsets.csv for feature extraction.")
    p.add_argument("--run-name", default="ambe_data",
                   help="Output parquet stem for --features-out (default 'ambe_data').")
    p.add_argument("--prompt-window-ns", type=float, default=2000.0,
                   help="Stage-A: drop hits with t <= this before OPTICS (delayed-residual "
                        "cut). MUST match the MC training value (apply_residual_filter "
                        "prompt_window_ns=2000). Default 2000.")
    p.add_argument("--prefilter-ns", type=float, default=0.0,
                   help="Stage-A: CF-proximity hit prefilter window. 0 = OPTICS on ALL "
                        "residual hits (MC-matched, hit_prefilter_ns=0). >0 reproduces the "
                        "legacy benchmark seeding. Default 0 (consistent with MC).")
    p.add_argument("--use-source-pos", action="store_true",
                   help="Use the AmBe per-run source position for OPTICS source-ToF + "
                        "d_source. OFF by default to match MC training (which set "
                        "source_pos_m=None → no ToF, d_source dropped from features). "
                        "Only enable if MC was retrained with WCSim-frame source positions "
                        "(NOT the AmBe data port convention — frames differ).")
    return p.parse_args()


def resolve_files(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("BeamCluster_*.root"))
    else:
        files = sorted(Path(f) for f in glob.glob(input_path))
    if not files:
        raise FileNotFoundError(f"No BeamCluster ROOT files found for: {input_path}")
    return files


def run_number_from_path(path: Path) -> int:
    m = _RUN_RE.search(path.name)
    return int(m.group(1)) if m else -1


def build_gate_sets(gate_csv_dir: str) -> dict:
    """
    Build {run -> set(accepted eventTimeTank timestamps)} from waveform-gated
    candidate CSVs (EventAmBeNeutronCandidates_*_<run>.csv).  Reuses ONLY the
    waveform-cut decision (which tank timestamps passed the IC cut) — none of the
    old cluster processing.  Run number is parsed from the trailing _<run>.csv.
    """
    run_re = re.compile(r"_(\d+)\.csv$")
    gate = {}
    for fp in sorted(glob.glob(os.path.join(gate_csv_dir,
                                            "EventAmBeNeutronCandidates_*.csv"))):
        if "_OPTICS" in fp:
            continue
        m = run_re.search(os.path.basename(fp))
        if not m:
            continue
        run = int(m.group(1))
        try:
            ts = pd.read_csv(fp, usecols=["eventTankTime"])["eventTankTime"]
        except Exception as e:
            print(f"  [gate] skip {os.path.basename(fp)}: {e}")
            continue
        s = set(int(x) for x in ts.dropna().astype("int64"))
        gate.setdefault(run, set()).update(s)
    return gate


# ── Legacy ANNIE charge balance for an OPTICS cluster ─────────────────────────

def charge_balance_legacy(pe: np.ndarray, chankey: np.ndarray) -> float:
    """
    Legacy ANNIE charge balance:  sqrt( ΣQ_i² / (ΣQ_i)² − 1/121 ),
    where Q_i is the total charge on PMT channel i (hits summed per tube).
    Mirrors charge_bal_legacy in src/ambe/mc/cluster_features.py:606.
    """
    if pe.sum() <= 0:
        return np.nan
    pmt_q: dict = {}
    for cid, p in zip(chankey, pe):
        pmt_q[int(cid)] = pmt_q.get(int(cid), 0.0) + float(p)
    q = np.array(list(pmt_q.values()), dtype=float)
    sq, sq2 = float(q.sum()), float((q ** 2).sum())
    arg = sq2 / (sq * sq) - 1.0 / 121.0
    return float(np.sqrt(max(arg, 0.0)))


def passes_preselection(pe_total: float, cb: float, n_hits: int) -> bool:
    return (pe_total < PRESEL_PE_MAX and
            (cb < PRESEL_CB_MAX or np.isnan(cb)) and
            n_hits > PRESEL_HITS_MIN)


# ── Data loading ──────────────────────────────────────────────────────────────

HIT_BR = ["hitX", "hitY", "hitZ", "hitT", "hitPE", "hitChankey"]
CL_BR  = ["clusterTime", "clusterPE", "clusterChargeBalance", "clusterHits"]
EV_BR  = ["eventNumber", "eventTimeTank", "numberOfClusters"]


# Events read into memory per uproot call. The nested hit branches expand by
# roughly an order of magnitude over the on-disk size, so reading a whole file at
# once costs several GB — a 75,625-event v4 BeamCluster file (run 6266) is an
# out-of-memory kill on an 11 GB node. Batching changes nothing observable: the
# same events are yielded, in the same order, with the same contents.
LOAD_BATCH_EVENTS = 5000


def load_file(root_file: Path, max_events: int = 0, gate_set=None,
              batch_events: int = LOAD_BATCH_EVENTS):
    """
    Yield per-event dicts from one BeamCluster file:
      run, event_number, n_cf_raw, df_cf (CF cluster table), df_hits (raw hits).
    Reads the latest 'Event' cycle (uproot default).

    gate_set: if not None, a set of accepted eventTimeTank timestamps; events
    whose eventTimeTank is not in the set are skipped (waveform-gating).  The
    timestamps come from the waveform IC cut (see --gate-csv-dir).

    batch_events: how many entries to hold in memory at a time. The gate is
    applied per event AFTER the read, so gating does not reduce the read cost and
    a gated run of a large file needs this just as much as an ungated one.
    """
    run  = run_number_from_path(root_file)
    f    = uproot.open(str(root_file))
    tree = f["Event"]
    n_total = tree.num_entries
    stop = min(max_events, n_total) if max_events > 0 else n_total
    step = max(int(batch_events), 1)

    for start in range(0, stop, step):
        a = tree.arrays(EV_BR + CL_BR + HIT_BR, library="np",
                        entry_start=start, entry_stop=min(start + step, stop))

        ett_all = a["eventTimeTank"]
        n = len(a["eventNumber"])
        for i in range(n):
            if gate_set is not None and int(ett_all[i]) not in gate_set:
                continue
            ncl = int(a["numberOfClusters"][i])
            df_cf = pd.DataFrame({
                "clusterTime": np.asarray(a["clusterTime"][i], dtype=float),
                "clusterPE":  np.asarray(a["clusterPE"][i], dtype=float),
                "clusterCB":  np.asarray(a["clusterChargeBalance"][i], dtype=float),
                "clusterHits": np.asarray(a["clusterHits"][i], dtype=float),
            })
            df_hits = pd.DataFrame({
                "x":  np.asarray(a["hitX"][i], dtype=float),
                "y":  np.asarray(a["hitY"][i], dtype=float),
                "z":  np.asarray(a["hitZ"][i], dtype=float),
                "t":  np.asarray(a["hitT"][i], dtype=float),
                "pe": np.asarray(a["hitPE"][i], dtype=float),
                "chankey": np.asarray(a["hitChankey"][i], dtype=int),
            })
            yield {
                "run": run,
                "event_number": int(a["eventNumber"][i]),
                "event_tank_time": int(ett_all[i]),
                "n_cf_raw": ncl,
                "df_cf": df_cf,
                "df_hits": df_hits,
            }
        del a


# ── Per-event counting ────────────────────────────────────────────────────────

def count_cf(df_cf: pd.DataFrame):
    raw = len(df_cf)
    if raw == 0:
        return 0, 0
    presel = int(((df_cf["clusterPE"] < PRESEL_PE_MAX) &
                  (df_cf["clusterCB"] < PRESEL_CB_MAX) &
                  (df_cf["clusterHits"] > PRESEL_HITS_MIN)).sum())
    return raw, presel


def count_optics(df_hits: pd.DataFrame, df_cf: pd.DataFrame,
                 prefilter_ns: float = DEFAULT_PREFILTER):
    """
    Return (n_raw, n_presel, list_of_member_hitcounts) for OPTICS clusters.

    OPTICS is run only on hits in the time proximity of ClusterFinder clusters:
    _prefilter_hits keeps hits within +/-(prefilter_ns/2) of any CF cluster time
    (default +/-1 us), exactly as the MC pipeline does.  This makes the data
    benchmark directly comparable to the MC validation and avoids clustering the
    dark-noise-filled rest of the ~70 us acquisition window.  If the event has no
    CF clusters, _prefilter_hits returns all hits unchanged (handled below).
    """
    if len(df_hits) < MIN_SAMPLES:
        return 0, 0, []
    # No CF clusters → no proximity windows to seed OPTICS; report zero.
    if df_cf is None or len(df_cf) == 0:
        return 0, 0, []
    df_pf = _prefilter_hits(df_hits, df_cf, prefilter_ns=prefilter_ns)
    df_pf = df_pf.reset_index(drop=True)
    if len(df_pf) < MIN_SAMPLES:
        return 0, 0, []
    labels = run_optics_on_event(df_pf, min_samples=MIN_SAMPLES, xi=XI,
                                 t_unit_ns=T_UNIT_NS, source_pos_m=None)
    cl_ids = [l for l in np.unique(labels) if l >= 0]
    n_raw  = len(cl_ids)
    n_presel = 0
    nhits_list = []
    pe  = df_pf["pe"].to_numpy(float)
    ck  = df_pf["chankey"].to_numpy(int)
    for cid in cl_ids:
        m = labels == cid
        nh = int(m.sum())
        nhits_list.append(nh)
        pe_tot = float(pe[m].sum())
        cb     = charge_balance_legacy(pe[m], ck[m])
        if passes_preselection(pe_tot, cb, nh):
            n_presel += 1
    return n_raw, n_presel, nhits_list


def process_one_file(fp, max_events: int = 0, max_hits: int = 1500, gate_set=None):
    """
    Worker: process a single ROOT file and return (df_run, optics_nhits, n_skip).
    Self-contained (no shared state) so it can run in a separate process.
    gate_set: optional set of accepted eventTimeTank timestamps (waveform gate).
    """
    run  = run_number_from_path(Path(fp))
    port = port_name(run)
    rows = []
    optics_nhits = []
    n_skip = 0
    for ev in load_file(Path(fp), max_events=max_events, gate_set=gate_set):
        nh = len(ev["df_hits"])
        cf_raw, cf_presel = count_cf(ev["df_cf"])
        if max_hits and nh > max_hits:
            opt_raw = opt_presel = np.nan
            skipped = 1
            n_skip += 1
        else:
            opt_raw, opt_presel, nhl = count_optics(ev["df_hits"], ev["df_cf"])
            optics_nhits.extend(nhl)
            skipped = 0
        rows.append({
            "run": run, "port": port,
            "event_number": ev["event_number"],
            "n_hits_total": nh,
            "n_cf_raw": cf_raw, "n_cf_presel": cf_presel,
            "n_optics_raw": opt_raw, "n_optics_presel": opt_presel,
            "skipped_large": skipped,
        })
    return pd.DataFrame(rows), optics_nhits, n_skip


def process_all(files, outdir: Path, max_events: int = 0, max_hits: int = 1500,
                jobs: int = 1, resume: bool = True, gate_sets: dict = None):
    """
    Process files in parallel (one process per file, up to `jobs` at once) and
    append each completed run to optics_beamcluster_stats.csv as it finishes, so
    partial progress is preserved/inspectable.  Events with > max_hits raw hits
    are flagged 'skipped_large' (cosmic showers), excluded from OPTICS, still
    recorded (n_optics_* = NaN) — never silently dropped.

    resume=True: runs already present in an existing CSV are skipped and their
    rows are reloaded, so an interrupted run can be continued without redoing work.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_path = outdir / "optics_beamcluster_stats.csv"
    all_rows = []
    optics_nhits = []
    done_runs = set()

    if resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        done_runs = set(prev["run"].unique())
        all_rows.append(prev)
        print(f"Resume: {len(done_runs)} run(s) already in CSV "
              f"({sorted(done_runs)}) — skipping them.", flush=True)
    else:
        if csv_path.exists():
            csv_path.unlink()

    todo = [fp for fp in files if run_number_from_path(fp) not in done_runs]
    header_written = csv_path.exists()
    t_start = time.time()
    n_done = 0
    print(f"Dispatching {len(todo)} file(s) across {jobs} worker(s) ...", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut2fp = {ex.submit(process_one_file, str(fp), max_events, max_hits,
                            (gate_sets or {}).get(run_number_from_path(fp))): fp
                  for fp in todo}
        for fut in as_completed(fut2fp):
            fp  = fut2fp[fut]
            run = run_number_from_path(fp)
            n_done += 1
            elapsed = time.time() - t_start
            # A worker may die (OOM on a huge file, etc.) — don't let one dead
            # run kill the whole job; log it and move on.
            try:
                df_run, nhl, n_skip = fut.result()
            except Exception as e:
                print(f"  [{n_done}/{len(todo)} FAILED, {elapsed/60:.1f} min] "
                      f"R{run} [{port_name(run)}]: worker error: {e!r} — skipped",
                      flush=True)
                continue

            if df_run is None or len(df_run) == 0:
                # Run produced no rows (e.g. every event gated out, or empty file).
                print(f"  [{n_done}/{len(todo)} done, {elapsed/60:.1f} min] "
                      f"R{run} [{port_name(run)}]: 0 events (nothing passed gate / "
                      f"empty file)", flush=True)
                continue

            df_run.to_csv(csv_path, mode="a", header=not header_written, index=False)
            header_written = True
            all_rows.append(df_run)
            optics_nhits.extend(nhl)

            cf_mean = df_run["n_cf_raw"].mean()
            opmean  = (df_run.loc[df_run["skipped_large"] == 0, "n_optics_raw"].mean()
                       if (df_run["skipped_large"] == 0).any() else float("nan"))
            print(f"  [{n_done}/{len(todo)} done, {elapsed/60:.1f} min] "
                  f"R{run} [{port_name(run)}]: {len(df_run)} events "
                  f"({n_skip} large skipped) → CF raw {cf_mean:.2f} / "
                  f"OPTICS raw {opmean:.2f}", flush=True)

    if not all_rows:
        raise SystemExit("No runs produced any rows.")
    return pd.concat(all_rows, ignore_index=True), optics_nhits


# ── Summary ───────────────────────────────────────────────────────────────────

def per_group_stats(df: pd.DataFrame, label: str) -> dict:
    # OPTICS stats and agreement use only events actually clustered (not the
    # cosmic-shower events skipped by the hit cap, which have NaN OPTICS counts).
    o = df[df["skipped_large"] == 0]
    return {
        "group": label,
        "n_events": len(df),
        "n_optics_events": len(o),
        "n_skipped_large": int((df["skipped_large"] == 1).sum()),
        "cf_raw_mean":      df["n_cf_raw"].mean(),
        "cf_raw_median":    df["n_cf_raw"].median(),
        "cf_presel_mean":   df["n_cf_presel"].mean(),
        "cf_presel_median": df["n_cf_presel"].median(),
        "optics_raw_mean":      o["n_optics_raw"].mean(),
        "optics_raw_median":    o["n_optics_raw"].median(),
        "optics_presel_mean":   o["n_optics_presel"].mean(),
        "optics_presel_median": o["n_optics_presel"].median(),
        "presel_agree_pct": (100.0 * (o["n_cf_presel"] == o["n_optics_presel"]).mean()
                             if len(o) else np.nan),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = [per_group_stats(df, "ALL_COMBINED")]
    for port, g in df.groupby("port"):
        out.append(per_group_stats(g, port))
    return pd.DataFrame(out)


def print_and_save_summary(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    lines = []
    lines.append("=" * 78)
    lines.append("OPTICS Stage-1 pre-selection vs ClusterFinder — AmBe DATA")
    lines.append(f"OPTICS params: ms={MIN_SAMPLES}, xi={XI}, t_unit={T_UNIT_NS} ns")
    lines.append("OPTICS hits: prefiltered to +/-1us of ClusterFinder cluster times "
                 "(MC-pipeline convention)")
    lines.append(f"Pre-selection: clusterPE<{PRESEL_PE_MAX}, "
                 f"CB<{PRESEL_CB_MAX}, clusterHits>{PRESEL_HITS_MIN}")
    n_skip_tot = int(summary.loc[summary['group'] == 'ALL_COMBINED',
                                 'n_skipped_large'].iloc[0])
    lines.append(f"Large events skipped (>hit cap, cosmic showers, excluded from "
                 f"OPTICS): {n_skip_tot}")
    lines.append("=" * 78)
    hdr = (f"{'group':<16}{'evts':>8}{'OPevts':>8}"
           f"{'CFraw':>8}{'OPraw':>8}"
           f"{'CFpre':>8}{'OPpre':>8}{'agree%':>9}")
    lines.append(hdr)
    lines.append("-" * 78)
    for _, r in summary.iterrows():
        lines.append(
            f"{r['group']:<16}{int(r['n_events']):>8}{int(r['n_optics_events']):>8}"
            f"{r['cf_raw_mean']:>8.2f}{r['optics_raw_mean']:>8.2f}"
            f"{r['cf_presel_mean']:>8.3f}{r['optics_presel_mean']:>8.3f}"
            f"{r['presel_agree_pct']:>8.1f}%")
    lines.append("-" * 78)
    lines.append("(means shown; CFpre/OPpre = neutron-like clusters per event; "
                 "OPevts = events OPTICS ran on)")
    text = "\n".join(lines)
    print(text)
    (outdir / "optics_beamcluster_summary.txt").write_text(text + "\n")
    summary.to_csv(outdir / "optics_beamcluster_perport.csv", index=False)


def make_plots(df_all: pd.DataFrame, optics_nhits, outdir: Path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # OPTICS histograms use only events OPTICS actually ran on (drop skipped).
    df = df_all[df_all["skipped_large"] == 0]

    # (0,0) raw clusters/event
    mx = int(min(max(df["n_cf_raw"].max(), df["n_optics_raw"].max()), 20))
    bins = np.arange(-0.5, mx + 1.5)
    ax[0, 0].hist(df["n_cf_raw"], bins=bins, alpha=0.6, color="#0077BB",
                  label=f"CF (mean={df['n_cf_raw'].mean():.2f})")
    ax[0, 0].hist(df["n_optics_raw"], bins=bins, alpha=0.6, color="#EE7733",
                  label=f"OPTICS (mean={df['n_optics_raw'].mean():.2f})")
    ax[0, 0].set(xlabel="Raw clusters / event", ylabel="Events",
                 title="Raw cluster count (no cuts)")
    ax[0, 0].legend(fontsize=8)

    # (0,1) pre-selected clusters/event
    mxp = int(min(max(df["n_cf_presel"].max(), df["n_optics_presel"].max()), 10))
    binsp = np.arange(-0.5, mxp + 1.5)
    ax[0, 1].hist(df["n_cf_presel"], bins=binsp, alpha=0.6, color="#0077BB",
                  label=f"CF (mean={df['n_cf_presel'].mean():.3f})")
    ax[0, 1].hist(df["n_optics_presel"], bins=binsp, alpha=0.6, color="#EE7733",
                  label=f"OPTICS (mean={df['n_optics_presel'].mean():.3f})")
    ax[0, 1].set(xlabel="Neutron-like clusters / event", ylabel="Events",
                 title="After Stage-1 pre-selection")
    ax[0, 1].legend(fontsize=8)

    # (1,0) per-port pre-selected means
    pp = df.groupby("port")[["n_cf_presel", "n_optics_presel"]].mean()
    xp = np.arange(len(pp))
    ax[1, 0].bar(xp - 0.2, pp["n_cf_presel"], width=0.4, color="#0077BB", label="CF")
    ax[1, 0].bar(xp + 0.2, pp["n_optics_presel"], width=0.4, color="#EE7733", label="OPTICS")
    ax[1, 0].set_xticks(xp)
    ax[1, 0].set_xticklabels(pp.index, rotation=45, ha="right", fontsize=7)
    ax[1, 0].set(ylabel="Mean neutron-like clusters/event",
                 title="Per-port pre-selected rate")
    ax[1, 0].legend(fontsize=8)

    # (1,1) hits per OPTICS cluster
    if optics_nhits:
        ax[1, 1].hist(optics_nhits, bins=40, color="darkorange", edgecolor="white")
        ax[1, 1].axvline(np.median(optics_nhits), color="k", ls="--",
                         label=f"median={np.median(optics_nhits):.0f}")
        ax[1, 1].set(xlabel="Hits per OPTICS cluster", ylabel="Clusters",
                     title="OPTICS cluster size")
        ax[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"OPTICS (ms={MIN_SAMPLES}, xi={XI}, t={T_UNIT_NS} ns) vs ClusterFinder — "
        f"AmBe DATA  |  {len(df_all)} events, {df_all['run'].nunique()} runs",
        fontsize=11)
    fig.tight_layout()
    out = outdir / "optics_beamcluster_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Plot  → {out}")


# ── MVA feature extraction on DATA (no truth) ─────────────────────────────────
# Produce, per OPTICS cluster, the SAME physics features the MC training used
# (compute_cluster_features is truth-independent — it reads only x/y/z/t/pe/pmtID).
# These features are the MVA inputs X; the data has no labels y, by design — the
# frozen MC model supplies the discrimination. Source position (known per run) is
# converted cm -> m so d_source/d_source_fit match the MC convention.

def extract_features_one_file(fp, geo_path, offsets_path, max_events: int = 0,
                              max_hits: int = 1500, gate_set=None,
                              prefilter_ns: float = 0.0,
                              prompt_window_ns: float = 2000.0,
                              use_source_pos: bool = False) -> pd.DataFrame:
    """
    Run OPTICS per (waveform-gated) event and compute the full MVA feature set for
    every OPTICS cluster. Returns a per-cluster DataFrame (one row per cluster)
    with the PHYSICS_FEATURES columns + bookkeeping (run, port, event_number,
    event_tank_time, cluster_id, passes_stage1). No truth columns — this is real
    data; the frozen MC MVA supplies the discrimination.

    STAGE-A CONSISTENCY WITH MC TRAINING (critical):
    The MC features pipeline (cluster_features.run) runs OPTICS on the DELAYED
    RESIDUAL — hits with t > prompt_window_ns (apply_residual_filter, prompt=2000ns)
    — with NO CF-proximity prefilter (hit_prefilter_ns=0 → OPTICS on all residual
    hits). To score data with that model on the SAME footing, we reproduce both:
      * prompt_window_ns (default 2000): drop hits with t <= this (post-muon/prompt).
      * prefilter_ns      (default 0):   0 = OPTICS on ALL surviving hits (MC-matched);
                                         >0 = legacy CF-proximity prefilter (benchmark).
    Defaults match MC. Override only to reproduce the old benchmark behavior.

    Geometry is built INSIDE the worker from CSV paths so this function is
    self-contained and picklable for a ProcessPoolExecutor (no large geo object
    crosses the process boundary).
    """
    geo  = load_geometry(geo_path, offsets_path)
    run  = run_number_from_path(Path(fp))
    port = port_name(run)
    src_cm = SOURCE_POSITIONS.get(run)
    # SOURCE POSITION — consistency with MC training (critical):
    # The MC training (cc_neutrino config) set source_pos_m=None, so MC OPTICS used
    # NO source-ToF correction and d_source/d_source_fit were NaN → dropped from the
    # 31 trained features. To keep Stage A identical, data must ALSO use source_pos_m
    # =None by default (use_source_pos=False). Otherwise the data OPTICS clustering
    # would get a source-ToF nudge the MC never had — an inconsistency even though
    # d_source isn't a model input.
    # NOTE: SOURCE_POSITIONS is the AmBe DATA port convention (cm). It is NOT the
    # WCSim port convention (Steven's all-ports config). They must NOT be mixed. If a
    # future MC retrain includes d_source, supply WCSim-frame coordinates there and
    # match the frame here before enabling use_source_pos.
    src_m = (np.asarray(src_cm, dtype=float) / 100.0
             if (use_source_pos and src_cm is not None) else None)

    rows = []
    n_cosmic = 0
    for ev in load_file(Path(fp), max_events=max_events, gate_set=gate_set):
        df_hits, df_cf = ev["df_hits"], ev["df_cf"]
        nh = len(df_hits)
        if max_hits and nh > max_hits:
            continue                          # cosmic shower — skip (as in counting path)
        # ── Cosmic veto (per-event), same as AmBeNeutronProcessing.cosmic_cut ──
        # If ANY ClusterFinder cluster in the event has clusterTime < 2000 ns OR
        # clusterPE > 100 PE, the WHOLE event is a cosmic event → exclude it
        # entirely (matches process_events_efficient's `break`).
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
        # prefilter_ns=0 (MC-matched): OPTICS on ALL residual hits, no CF seeding.
        # prefilter_ns>0: legacy benchmark behavior (needs CF clusters present).
        if prefilter_ns > 0 and df_cf is not None and len(df_cf) > 0:
            df_pf = _prefilter_hits(df_hits, df_cf, prefilter_ns=prefilter_ns).reset_index(drop=True)
        else:
            df_pf = df_hits
        if len(df_pf) < MIN_SAMPLES:
            continue
        labels = run_optics_on_event(df_pf, min_samples=MIN_SAMPLES, xi=XI,
                                     t_unit_ns=T_UNIT_NS, source_pos_m=src_m)
        # compute_cluster_features expects a 'pmtID' column (data has 'chankey').
        df_pf = df_pf.rename(columns={"chankey": "pmtID"})
        for cid in (c for c in np.unique(labels) if c >= 0):
            mask = labels == cid
            df_clust = df_pf[mask].reset_index(drop=True)
            feats = compute_cluster_features(df_clust, geo, source_pos_m=src_m)
            if not feats:
                continue
            pe_tot = float(df_clust["pe"].sum())
            cb     = charge_balance_legacy(df_clust["pe"].to_numpy(float),
                                           df_clust["pmtID"].to_numpy(int))
            n_hits = int(mask.sum())
            # Earliest-hit cluster time [ns] — the ClusterFinder cluster-time analogue,
            # so Selection 1 (OPTICS) and Selection 2 (CF) capture-time fits use the
            # SAME observable. (t_mean, the hit-time MEAN, is also kept by feats.)
            t_earliest = float(df_clust["t"].min())
            row = {
                "run": run, "port": port,
                "event_number": ev["event_number"],
                "event_tank_time": ev["event_tank_time"],
                "cluster_id": int(cid),
                "clusterTime_earliest": t_earliest,
                # Stage-1 "neutron-like" flag (does NOT gate the parquet — kept as a column).
                "passes_stage1": bool(passes_preselection(pe_tot, cb, n_hits)),
            }
            row.update(feats)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_cosmic_events_vetoed"] = n_cosmic
    print(f"  [R{run}] cosmic-vetoed events: {n_cosmic}", flush=True)
    return out


def run_feature_extraction(files, outdir: Path, geo_path, offsets_path, gate_sets,
                           max_events: int, max_hits: int, run_name: str,
                           jobs: int = 1, prefilter_ns: float = 0.0,
                           prompt_window_ns: float = 2000.0,
                           use_source_pos: bool = False):
    """
    Extract MVA features over all (gated) files and write one parquet.

    Parallelized one-process-per-file (up to `jobs`) like the counting path, since
    the per-cluster Gauss-Newton vertex fit makes this CPU-bound. Each worker
    writes a per-run parquet shard to outdir/_feature_shards/ as it finishes
    (crash-safe / resumable), then the shards are concatenated at the end.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    shard_dir = outdir / "_feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for fp in files:
        run = run_number_from_path(fp)
        if gate_sets is not None and run not in gate_sets:
            print(f"  R{run}: no gate set — skipped", flush=True)
            continue
        shard = shard_dir / f"{run_name}__shard_{run}.parquet"
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
        return ex.submit(extract_features_one_file, str(fp), geo_path, offsets_path,
                         max_events, max_hits, gate_set, prefilter_ns, prompt_window_ns,
                         use_source_pos)

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
                df.to_parquet(shard_dir / f"{run_name}__shard_{run}.parquet", index=False)
            print(f"  [{n_done}/{len(todo)} done, {el:.1f} min] R{run} [{port_name(run)}]: "
                  f"{len(df)} OPTICS clusters"
                  + (f"  ({int(df['passes_stage1'].sum())} pass Stage-1)" if len(df) else ""),
                  flush=True)

    # Concatenate all shards (including any from a previous resumed run).
    shards = sorted(shard_dir.glob(f"{run_name}__shard_*.parquet"))
    frames = [pd.read_parquet(s) for s in shards]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    parquet_path = outdir / f"{run_name}__data_features.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"\n[features] {len(out)} clusters over "
          f"{out['run'].nunique() if len(out) else 0} run(s) -> {parquet_path}")
    if len(out):
        print(f"[features] Stage-1 survivors: {int(out['passes_stage1'].sum())}/{len(out)} "
              f"({100*out['passes_stage1'].mean():.1f}%)")
    return parquet_path


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = resolve_files(args.input)
    print(f"Found {len(files)} BeamCluster file(s).")

    # Waveform-gating: build per-run accepted-timestamp sets and restrict to gated runs.
    gate_sets = None
    if args.gate_csv_dir:
        gate_sets = build_gate_sets(args.gate_csv_dir)
        print(f"Waveform gate: loaded accepted eventTimeTank sets for "
              f"{len(gate_sets)} run(s) from {args.gate_csv_dir}")
        for run, s in sorted(gate_sets.items()):
            print(f"  R{run}: {len(s)} accepted timestamps")
        gated_files, ungated = [], []
        for fp in files:
            (gated_files if run_number_from_path(fp) in gate_sets else ungated).append(fp)
        if ungated:
            print(f"  NOTE: {len(ungated)} BeamCluster run(s) have NO gate CSV and "
                  f"will be SKIPPED: {[run_number_from_path(f) for f in ungated]}")
        files = gated_files
        if not files:
            raise SystemExit("No BeamCluster runs have a matching gate CSV — nothing to do.")

    # ── MVA feature-extraction mode ──────────────────────────────────────────
    if args.features_out:
        # Sanity-build the geometry once in the parent so a bad path fails fast.
        print(f"[features] geometry: {args.geometry}")
        _ = load_geometry(args.geometry, args.offsets)
        gate_msg = "WAVEFORM-GATED" if gate_sets is not None else "ungated (all events)"
        print(f"[features] extracting MVA features [{gate_msg}] "
              f"(OPTICS ms={MIN_SAMPLES}, xi={XI}, t_unit={T_UNIT_NS} ns; "
              f"prefilter +/-1us of CF times; hit cap={args.max_hits}) "
              f"over {len(files)} file(s) ...")
        print(f"[features] Stage-A: prompt_window_ns={args.prompt_window_ns} "
              f"(t>prompt residual), prefilter_ns={args.prefilter_ns} "
              f"({'MC-matched: OPTICS on all residual hits' if args.prefilter_ns==0 else 'legacy CF-proximity seeding'})")
        print(f"[features] source_pos: {'AmBe per-run (use_source_pos ON)' if args.use_source_pos else 'None (MC-matched: no ToF, no d_source)'}")
        run_feature_extraction(files, outdir, args.geometry, args.offsets, gate_sets,
                               max_events=args.max_events, max_hits=args.max_hits,
                               run_name=args.run_name, jobs=args.jobs,
                               prefilter_ns=args.prefilter_ns,
                               prompt_window_ns=args.prompt_window_ns,
                               use_source_pos=args.use_source_pos)
        return

    # Unit sanity check on first file's first non-empty event.
    f0 = uproot.open(str(files[0]))["Event"]
    chk = f0.arrays(["hitT"], library="np", entry_stop=200)
    nonempty = [np.asarray(x, float) for x in chk["hitT"] if len(x)]
    allt = np.concatenate(nonempty) if nonempty else np.array([])
    if len(allt):
        print(f"[sanity] hitT (first file, 200 ev) min/median/max ns: "
              f"{allt.min():.1f} / {np.median(allt):.1f} / {allt.max():.1f}")

    gate_msg = "WAVEFORM-GATED" if gate_sets is not None else "ungated (all events)"
    print(f"\nProcessing [{gate_msg}] (OPTICS ms={MIN_SAMPLES}, xi={XI}, "
          f"t_unit={T_UNIT_NS} ns; OPTICS hits prefiltered to +/-1us of CF cluster "
          f"times; hit cap = {args.max_hits}) ...")
    df, optics_nhits = process_all(files, outdir, max_events=args.max_events,
                                   max_hits=args.max_hits, jobs=args.jobs,
                                   resume=not args.no_resume, gate_sets=gate_sets)

    print(f"\nTotal events: {len(df)}  over {df['run'].nunique()} runs")
    print(f"Stats → {outdir / 'optics_beamcluster_stats.csv'} (written incrementally)")

    summary = build_summary(df)
    print_and_save_summary(df, summary, outdir)
    make_plots(df, optics_nhits, outdir)


if __name__ == "__main__":
    main()
