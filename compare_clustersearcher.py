"""
compare_clustersearcher.py
--------------------------
Run a Python reimplementation of ClusterSearcher alongside OPTICS and
ClusterFinder on the cc_neutrino dataset and compare event-level metrics.

Usage (from repo root, annie venv):
    python compare_clustersearcher.py [--config configs/cc_neutrino_optics.yaml]
                                      [--max-events N]
                                      [--pe-thresh FLOAT]   # default 0 (no cut)
                                      [--out ambe_output/cc_neutrino/parquet/cc_neutrino__cs_comparison.parquet]

The script reuses the same residual filter, truth-matching, and metrics
functions from src/ambe/mc/optics.py so results are directly comparable.

ClusterSearcher parameters (from the ToolAnalysis Application-branch config):
    neighbour_radius : 0.60 m   (60 cm — same as their fPmtNeighbourRadius)
    cluster_radius   : 0.60 m   (60 cm — same as their fPmtClusterRadius)
    time_window_n_ns : 10 ns    (fPmtTimeWindowN)
    time_window_c_ns : 100 ns   (fPmtTimeWindowC)
    min_neighbour_digits : 4    (fPmtMinNeighbourDigits)
    min_cluster_digits   : 4    (fPmtMinClusterDigits)
    pe_thresh            : 0.0  (fPmtMinPulseHeight=20 ADC counts; not directly
                                 comparable to calibrated PE — set 0 to skip)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Locate the ambe package (run from repo root or any parent that contains src/)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from ambe.mc.optics import (
    run_optics_on_event,
    assign_clusterfinder_labels,
    event_metrics,
    _prefilter_hits,
    _apply_truth_window,
    DEFAULT_T_UNIT,
    DEFAULT_WINDOW,
    DEFAULT_PREFILTER,
)
from ambe.mc import cc_selection


# ---------------------------------------------------------------------------
# ClusterSearcher — Python reimplementation
# ---------------------------------------------------------------------------

def clustersearcher_labels(
    df_event: pd.DataFrame,
    pe_thresh: float = 0.0,
    neighbour_radius_m: float = 0.60,
    cluster_radius_m: float = 0.60,
    time_window_n_ns: float = 10.0,
    time_window_c_ns: float = 100.0,
    min_neighbour_digits: int = 4,
    min_cluster_digits: int = 4,
) -> np.ndarray:
    """
    Python reimplementation of ClusterSearcher::RecoClusters().

    Algorithm (mirrors the C++ exactly):
      1. Pulse-height filter  — keep hits with pe >= pe_thresh
      2. Neighbour filter     — keep hits with >= min_neighbour_digits neighbours
                                 within neighbour_radius_m AND time_window_n_ns
      3. Adjacency graph      — link pairs within cluster_radius_m AND
                                 time_window_c_ns
      4. Connected-components — iteratively expand clusters from each
                                 unclustered seed; discard clusters with
                                 fewer than min_cluster_digits hits

    Returns an integer label array of length len(df_event):
      -1  =>  noise / not clustered
      >=0 =>  cluster index
    """
    xyz = df_event[["x", "y", "z"]].to_numpy(float)
    t   = df_event["t"].to_numpy(float)
    pe  = df_event["pe"].to_numpy(float)

    # Step 1: pulse-height filter
    mask_pe = pe >= pe_thresh
    sel_pe  = np.where(mask_pe)[0]
    if len(sel_pe) == 0:
        return np.full(len(df_event), -1, dtype=int)

    xyz_pe = xyz[sel_pe]
    t_pe   = t[sel_pe]

    # Step 2: neighbour filter
    if len(xyz_pe) < 2:
        return np.full(len(df_event), -1, dtype=int)

    tree_pe = cKDTree(xyz_pe)
    pairs   = tree_pe.query_pairs(neighbour_radius_m)   # all spatial pairs within radius

    n_neighbours = np.zeros(len(sel_pe), dtype=int)
    for i, j in pairs:
        if abs(t_pe[i] - t_pe[j]) <= time_window_n_ns:
            n_neighbours[i] += 1
            n_neighbours[j] += 1

    mask_n  = n_neighbours >= min_neighbour_digits
    sel_n   = np.where(mask_n)[0]                       # indices into sel_pe
    if len(sel_n) == 0:
        return np.full(len(df_event), -1, dtype=int)

    xyz_sel = xyz_pe[sel_n]
    t_sel   = t_pe[sel_n]

    # Step 3: adjacency graph over selected hits
    tree_sel = cKDTree(xyz_sel)
    adj_spatial = tree_sel.query_pairs(cluster_radius_m)   # set of (i,j) pairs

    # Build adjacency list (spatial + time)
    n_sel = len(sel_n)
    adj: list[set] = [set() for _ in range(n_sel)]
    for i, j in adj_spatial:
        if abs(t_sel[i] - t_sel[j]) <= time_window_c_ns:
            adj[i].add(j)
            adj[j].add(i)

    # Step 4: connected-components expansion
    labels_sel = np.full(n_sel, -1, dtype=int)
    cluster_id = 0

    for start in range(n_sel):
        if labels_sel[start] != -1:
            continue
        if not adj[start]:                               # isolated hit — skip
            continue
        # BFS expansion
        component = [start]
        labels_sel[start] = cluster_id
        queue = list(adj[start])
        visited = {start}
        while queue:
            j = queue.pop()
            if j in visited:
                continue
            visited.add(j)
            labels_sel[j] = cluster_id
            component.append(j)
            queue.extend(n for n in adj[j] if n not in visited)

        if len(component) < min_cluster_digits:
            for idx in component:
                labels_sel[idx] = -1                    # discard small cluster
        else:
            cluster_id += 1

    # Map labels back to the full df_event length
    labels_full = np.full(len(df_event), -1, dtype=int)
    global_sel  = sel_pe[sel_n]                         # original df_event indices
    for local_i, global_i in enumerate(global_sel):
        labels_full[global_i] = labels_sel[local_i]

    return labels_full


# ---------------------------------------------------------------------------
# Main comparison driver
# ---------------------------------------------------------------------------

def run_comparison(
    pulses_path: Path,
    clusters_path: Path,
    out_path: Path,
    # OPTICS params (best config from your grid sweep for this dataset)
    min_samples: int   = 8,
    xi: float          = 0.10,
    t_unit_ns: float   = 25.0,
    # ClusterSearcher params
    cs_pe_thresh: float           = 0.0,
    cs_neighbour_radius_m: float  = 0.60,
    cs_cluster_radius_m: float    = 0.60,
    cs_time_window_n_ns: float    = 10.0,
    cs_time_window_c_ns: float    = 100.0,
    cs_min_neighbour_digits: int  = 4,
    cs_min_cluster_digits: int    = 4,
    # shared
    truth_window_ns: float  = DEFAULT_WINDOW,
    hit_prefilter_ns: float = 0.0,
    max_events: int | None  = None,
    # CC residual filter settings (matching cc_neutrino_optics.yaml)
    cc_only: bool           = True,
    prompt_window_ns: float = 2000.0,
) -> pd.DataFrame:

    print(f"[cs_compare] loading pulses  -> {pulses_path}")
    pulses   = pd.read_parquet(pulses_path)
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    # Apply the same residual filter as ambe mc optics
    print(f"[cs_compare] residual filter: cc_only={cc_only}, "
          f"prompt_window_ns={prompt_window_ns}")
    if cc_only and "cc_pass" in pulses.columns:
        pulses = pulses[pulses["cc_pass"].astype(bool)]
    if prompt_window_ns > 0 and "t" in pulses.columns:
        pulses = pulses[pulses["t"] > prompt_window_ns]
    pulses = pulses.reset_index(drop=True)
    print(f"[cs_compare] {len(pulses):,} hits in {pulses['eventID'].nunique():,} events "
          f"after residual filter")

    event_ids = pulses["eventID"].unique()
    if max_events is not None:
        event_ids = event_ids[:max_events]
        print(f"[cs_compare] truncating to {max_events} events for testing")

    rows = []
    t_start     = time.time()
    n_total     = len(event_ids)
    print_every = max(1, n_total // 20)

    print(f"[cs_compare] running 3 methods on {n_total} events ...")
    print(f"  OPTICS:          min_samples={min_samples}, xi={xi}, t_unit_ns={t_unit_ns}")
    print(f"  ClusterSearcher: pe_thresh={cs_pe_thresh}, "
          f"nb_r={cs_neighbour_radius_m}m, cl_r={cs_cluster_radius_m}m, "
          f"tn={cs_time_window_n_ns}ns, tc={cs_time_window_c_ns}ns, "
          f"min_nb={cs_min_neighbour_digits}, min_cl={cs_min_cluster_digits}")
    print()

    for i_ev, evid in enumerate(event_ids):
        df_ev = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        if len(df_ev) < 2:
            continue

        df_cl = (clusters[clusters["eventID"] == evid].reset_index(drop=True)
                 if clusters is not None else None)

        # Shared truth mask (computed once per event)
        truth_mask = _apply_truth_window(df_ev, truth_window_ns)

        # Optional hit prefilter for OPTICS (off by default for this dataset)
        if hit_prefilter_ns > 0:
            df_optics = _prefilter_hits(df_ev, df_cl, hit_prefilter_ns)
        else:
            df_optics = df_ev

        # --- OPTICS ---
        labels_optics = run_optics_on_event(
            df_optics, min_samples=min_samples, xi=xi, t_unit_ns=t_unit_ns)
        if len(df_optics) < len(df_ev):
            labels_full_optics = np.full(len(df_ev), -1, dtype=int)
            labels_full_optics[df_optics.index] = labels_optics
        else:
            labels_full_optics = labels_optics
        row_optics = event_metrics(df_ev, labels_full_optics, "optics",
                                   truth_window_ns=truth_window_ns,
                                   truth_neutron_mask=truth_mask)
        row_optics.update({"min_samples": min_samples, "xi": xi, "t_unit_ns": t_unit_ns})
        rows.append(row_optics)

        # --- ClusterSearcher ---
        labels_cs = clustersearcher_labels(
            df_ev,
            pe_thresh=cs_pe_thresh,
            neighbour_radius_m=cs_neighbour_radius_m,
            cluster_radius_m=cs_cluster_radius_m,
            time_window_n_ns=cs_time_window_n_ns,
            time_window_c_ns=cs_time_window_c_ns,
            min_neighbour_digits=cs_min_neighbour_digits,
            min_cluster_digits=cs_min_cluster_digits,
        )
        row_cs = event_metrics(df_ev, labels_cs, "clustersearcher",
                               truth_window_ns=truth_window_ns,
                               truth_neutron_mask=truth_mask)
        row_cs.update({"min_samples": cs_min_neighbour_digits,
                       "xi": float("nan"), "t_unit_ns": float("nan")})
        rows.append(row_cs)

        # --- ClusterFinder baseline ---
        if df_cl is not None:
            labels_cf = assign_clusterfinder_labels(df_ev, df_cl)
            row_cf = event_metrics(df_ev, labels_cf, "clusterfinder",
                                   truth_window_ns=truth_window_ns,
                                   truth_neutron_mask=truth_mask)
            row_cf.update({"min_samples": float("nan"), "xi": float("nan"),
                           "t_unit_ns": float("nan")})
            rows.append(row_cf)

        if (i_ev + 1) % print_every == 0 or (i_ev + 1) == n_total:
            elapsed   = time.time() - t_start
            rate      = (i_ev + 1) / elapsed
            remaining = (n_total - i_ev - 1) / rate if rate > 0 else 0
            print(f"  {i_ev+1:5d}/{n_total}  "
                  f"({100*(i_ev+1)/n_total:.0f}%)  "
                  f"rate={rate:.1f} ev/s  ETA={remaining:.0f}s",
                  flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out_path, index=False)
    print(f"\n[cs_compare] wrote {len(metrics):,} rows -> {out_path}")
    return metrics


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby("method", dropna=False).agg(
        events=("eventID",             "count"),
        mean_recall=("recall",         "mean"),
        mean_purity=("purity",         "mean"),
        mean_f1=("f1",                 "mean"),
        mean_ari=("ari",               "mean"),
        mean_n_clusters=("n_clusters", "mean"),
        mean_n_truth=("n_truth",       "mean"),
        mean_matched=("n_matched",     "mean"),
        mean_spurious=("n_spurious",   "mean"),
        mean_real_spurious=("n_real_spurious", "mean"),
        mean_missed=("n_missed",       "mean"),
        mean_split=("n_split",         "mean"),
        agreement_rate=("agreement",   "mean"),
        corrected_agreement_rate=("corrected_agreement", "mean"),
    ).reset_index()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/cc_neutrino_optics.yaml",
                   help="YAML config (used only to locate parquet dir + run_name)")
    p.add_argument("--parquet-dir",
                   default="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/"
                           "ambe_output/cc_neutrino/parquet",
                   help="Directory containing the _pulses and _clusterfinder parquets")
    p.add_argument("--run-name", default="cc_neutrino")
    p.add_argument("--out", default=None,
                   help="Output parquet path (default: <parquet-dir>/<run-name>__cs_comparison.parquet)")
    p.add_argument("--max-events", type=int, default=None,
                   help="Truncate to N events (for quick tests)")
    # OPTICS params
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--xi",          type=float, default=0.10)
    p.add_argument("--t-unit-ns",   type=float, default=25.0)
    # ClusterSearcher params
    p.add_argument("--pe-thresh",         type=float, default=0.0,
                   help="Min PE per hit for ClusterSearcher (0 = no cut)")
    p.add_argument("--nb-radius",         type=float, default=0.60,
                   help="Neighbour spatial radius [m] (default 0.60 = 60 cm)")
    p.add_argument("--cl-radius",         type=float, default=0.60,
                   help="Cluster spatial radius [m] (default 0.60 = 60 cm)")
    p.add_argument("--tw-n",              type=float, default=10.0,
                   help="Neighbour time window [ns] (default 10)")
    p.add_argument("--tw-c",              type=float, default=100.0,
                   help="Cluster time window [ns] (default 100)")
    p.add_argument("--min-nb-digits",     type=int,   default=4,
                   help="Min neighbours to keep a hit (default 4)")
    p.add_argument("--min-cl-digits",     type=int,   default=4,
                   help="Min hits per cluster (default 4)")
    args = p.parse_args()

    parquet_dir = Path(args.parquet_dir)
    pulses_path  = parquet_dir / f"{args.run_name}__pulses.parquet"
    clusters_path= parquet_dir / f"{args.run_name}__clusterfinder.parquet"
    out_path     = Path(args.out) if args.out else \
                   parquet_dir / f"{args.run_name}__cs_comparison.parquet"

    for p_path in (pulses_path, clusters_path):
        if not p_path.exists():
            print(f"ERROR: {p_path} not found. Run `ambe mc process` first.", file=sys.stderr)
            sys.exit(1)

    metrics = run_comparison(
        pulses_path   = pulses_path,
        clusters_path = clusters_path,
        out_path      = out_path,
        min_samples   = args.min_samples,
        xi            = args.xi,
        t_unit_ns     = args.t_unit_ns,
        cs_pe_thresh             = args.pe_thresh,
        cs_neighbour_radius_m    = args.nb_radius,
        cs_cluster_radius_m      = args.cl_radius,
        cs_time_window_n_ns      = args.tw_n,
        cs_time_window_c_ns      = args.tw_c,
        cs_min_neighbour_digits  = args.min_nb_digits,
        cs_min_cluster_digits    = args.min_cl_digits,
        max_events    = args.max_events,
    )

    summary = summarise(metrics)
    print("\n=== Summary (all events) ===")
    print(summary.to_string(index=False))

    csv_out = out_path.with_suffix(".csv")
    summary.to_csv(csv_out, index=False)
    print(f"\n[cs_compare] summary CSV -> {csv_out}")


if __name__ == "__main__":
    main()
