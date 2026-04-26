"""
MC OPTICS trainer + ClusterFinder baseline + per-event metrics.

Reads the Parquet files produced by ambe.mc.processor and writes:
    <parquet_dir>/<run_name>__metrics.parquet
    <plots_dir>/<run_name>__optics_summary.csv

The OPTICS hyperparameter grid is defined in the config's `optics:` block;
`ambe.mc.optics.run` sweeps it and also runs the ClusterFinder baseline so
the two can be compared apples-to-apples per event.

Metrics per (event, method, hyperparameters):
    purity, recall, f1, ari, n_clusters

--- Scaling design (why t_unit_ns, not t_scale) ---

The old `t_scale` parameter was a no-op: multiplying the time column by a
constant before StandardScaler() has no effect because StandardScaler
normalises each column to unit variance independently, cancelling the factor.

The correct approach is to:
  1. StandardScaler on x, y, z only  →  spatial axes: zero mean, unit std
  2. Divide t by a fixed physical time unit (t_unit_ns)  →  1 OPTICS distance
     unit in time = t_unit_ns nanoseconds

This makes t_unit_ns interpretable: choosing t_unit_ns = 50 means "50 ns of
timing difference looks the same as 1 spatial standard deviation to OPTICS."
The natural choice is the detector crossing time (~20 ns direct, ~40 ns with
reflections), so the recommended grid is [25, 40, 50, 75] ns.

--- Truth window (truth_window_ns) ---

Geant4 tracks every neutron scatter during thermalization.  A single neutron
capture event therefore has ~16 tight hits from the capture gamma (all within
~40 ns) plus ~1 outlier hit from a thermalization scatter that may be
thousands of nanoseconds away, still carrying the same neutron ancestor PDG.

From 10 k-event analysis of ANNIETree_MC.root:
  - median cluster delta_t  = 39.7 ns
  - at window = 75 ns: median frac_in = 1.000  (typical cluster fully inside)
  - at window = 75 ns: ~51 % of clusters have zero out-of-window hits
  - expanding beyond 75 ns yields < 2 % additional gain (bimodal gap confirmed)

truth_window_ns = 75 is therefore used to define what counts as a "recoverable
truth neutron hit" for purity/recall calculations.  Thermalization-outlier hits
(outside the window) are treated as acceptable noise; OPTICS labelling them -1
is not counted as a miss.  OPTICS still *sees* all hits including outliers —
the window only affects how we evaluate the result.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler

from ..context import RunContext


NEUTRON_CLASSES = {1, 2, 3, 4}
NEUTRON_PDG     = 2112
MIN_MATCH_FRAC  = 0.5     # fraction of cluster hits from dominant trackID to call it matched
DEFAULT_T_UNIT  = 50.0    # ns — default time-axis unit (see module docstring)
DEFAULT_WINDOW  = 75.0    # ns — default truth window (see module docstring)


# --------------------------------------------------------------------------- #
# Truth-window helper
# --------------------------------------------------------------------------- #

def _apply_truth_window(df_event: pd.DataFrame, truth_window_ns: float) -> np.ndarray:
    """
    For each neutron trackID present in df_event, compute the median hit time
    of that trackID's hits and mark hits within ±(truth_window_ns/2) as
    "in-window neutron hits".

    Returns a boolean array (length = len(df_event)) that is True only for
    hits that are both is_neutron==1 AND within the time window.

    Hits outside the window are genuine Geant4 tracks but represent
    thermalization scatters at microsecond timescales — they are treated as
    acceptable noise for evaluation purposes.
    """
    is_neutron  = df_event["is_neutron"].to_numpy(int).astype(bool)
    track_ids   = df_event["ancestor_trackID"].to_numpy(int)
    pdgs        = df_event["ancestor_pdg"].to_numpy(int)
    hit_t       = df_event["t"].to_numpy(float)
    half_win    = truth_window_ns / 2.0

    in_window = np.zeros(len(df_event), dtype=bool)
    neutron_mask = is_neutron & (pdgs == NEUTRON_PDG)

    for tid in np.unique(track_ids[neutron_mask]):
        tid_mask = neutron_mask & (track_ids == tid)
        if not tid_mask.any():
            continue
        median_t = np.median(hit_t[tid_mask])
        within   = np.abs(hit_t - median_t) <= half_win
        in_window |= (tid_mask & within)

    return in_window


# --------------------------------------------------------------------------- #
# OPTICS per event
# --------------------------------------------------------------------------- #

def run_optics_on_event(df_event: pd.DataFrame,
                        min_samples: int   = 5,
                        xi:          float = 0.05,
                        metric:      str   = "euclidean",
                        t_unit_ns:   float = DEFAULT_T_UNIT) -> np.ndarray:
    """
    Run OPTICS on one event's hits.

    Scaling:
      - x, y, z  →  StandardScaler (zero mean, unit variance per axis)
      - t         →  divided by t_unit_ns  (fixed physical unit, NOT re-standardised)

    This decoupling is critical: StandardScaler on t would erase any manual
    scaling choice.  See module docstring for the physical motivation.

    Parameters
    ----------
    t_unit_ns : float
        1 OPTICS distance unit in the time axis = t_unit_ns nanoseconds.
        Physically: choose ~detector-crossing time (~20-50 ns for ANNIE).
    """
    X = df_event[["x", "y", "z", "t"]].to_numpy(dtype=float, copy=True)

    # Step 1: scale xyz spatially
    X_xyz = StandardScaler().fit_transform(X[:, :3])

    # Step 2: scale time with a fixed physical unit (NOT standardised again)
    X_t = X[:, 3:4] / t_unit_ns

    X_scaled = np.hstack([X_xyz, X_t])

    if len(X_scaled) < min_samples:
        return np.full(len(X_scaled), -1, dtype=int)

    return OPTICS(min_samples=min_samples, xi=xi,
                  metric=metric).fit_predict(X_scaled)


# --------------------------------------------------------------------------- #
# ClusterFinder baseline: time-proximity membership
# --------------------------------------------------------------------------- #

def assign_clusterfinder_labels(df_pulses: pd.DataFrame,
                                df_clusters,
                                tmin_margin_ns: float = 5.0,
                                tmax_margin_ns: float = 20.0) -> np.ndarray:
    if df_clusters is None or len(df_clusters) == 0:
        return np.full(len(df_pulses), -1, dtype=int)
    ct     = df_clusters["clusterTime"].to_numpy(float)
    labels = np.full(len(df_pulses), -1, dtype=int)
    pulse_t = df_pulses["t"].to_numpy(float)
    for i, tp in enumerate(pulse_t):
        dt = tp - ct
        in_window = (dt > -tmin_margin_ns) & (dt < tmax_margin_ns)
        if not in_window.any():
            continue
        candidates = np.where(in_window)[0]
        labels[i]  = int(candidates[np.argmin(np.abs(dt[candidates]))])
    return labels


# --------------------------------------------------------------------------- #
# Cluster → neutron/non-neutron decision, and event-level metrics
# --------------------------------------------------------------------------- #

def classify_clusters_by_majority(labels: np.ndarray,
                                  truth_class: np.ndarray) -> np.ndarray:
    labels      = np.asarray(labels)
    truth_class = np.asarray(truth_class)
    predicted_is_neutron = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        if c == -1:
            continue
        mask     = labels == c
        members  = truth_class[mask]
        n_neutron = np.isin(members, list(NEUTRON_CLASSES)).sum()
        if n_neutron > 0 and n_neutron >= 0.5 * mask.sum():
            predicted_is_neutron[mask] = True
    return predicted_is_neutron


def _optics_cluster_outcomes(df_event: pd.DataFrame,
                             labels: np.ndarray,
                             truth_neutron_mask: np.ndarray) -> dict:
    """
    Match predicted clusters to truth neutron clusters.

    truth_neutron_mask : boolean array (len = df_event) marking which hits
        count as recoverable truth-neutron hits after applying the truth window.
        A trackID is considered a "truth neutron" only if it has at least one
        in-window hit.

    A predicted cluster is *matched* to a truth trackID if:
      - its dominant neutron trackID is a truth trackID, AND
      - that trackID's hits account for >= MIN_MATCH_FRAC of all cluster hits.

    Returns keys: n_truth, n_predicted, n_matched, n_spurious, n_missed,
                  n_split, agreement.
    """
    labels     = np.asarray(labels, dtype=int)
    track_ids  = df_event["ancestor_trackID"].to_numpy(int)
    pdgs       = df_event["ancestor_pdg"].to_numpy(int)

    # Truth neutron trackIDs: only those with at least one in-window hit
    truth_trackIDs = set(
        int(tid)
        for tid, pdg, in_win in zip(track_ids, pdgs, truth_neutron_mask)
        if pdg == NEUTRON_PDG and tid >= 0 and in_win
    )
    n_truth = len(truth_trackIDs)

    unique_labels = [c for c in np.unique(labels) if c >= 0]
    n_predicted   = len(unique_labels)

    matched_tids = set()
    split_tids   = set()
    n_spurious   = 0

    for c in unique_labels:
        mask    = labels == c
        n_total = int(mask.sum())

        neutron_mask_c = mask & (pdgs == NEUTRON_PDG) & (track_ids >= 0)
        if not neutron_mask_c.any():
            n_spurious += 1
            continue

        tids_in_cluster, counts = np.unique(track_ids[neutron_mask_c],
                                            return_counts=True)
        dom_tid  = int(tids_in_cluster[np.argmax(counts)])
        dom_frac = float(counts.max()) / n_total

        if dom_tid in truth_trackIDs and dom_frac >= MIN_MATCH_FRAC:
            if dom_tid in matched_tids:
                split_tids.add(dom_tid)
            matched_tids.add(dom_tid)
        else:
            n_spurious += 1

    n_matched = len(matched_tids)
    n_missed  = len(truth_trackIDs - matched_tids)
    n_split   = len(split_tids)

    return {
        "n_truth":     n_truth,
        "n_predicted": n_predicted,
        "n_matched":   n_matched,
        "n_spurious":  n_spurious,
        "n_missed":    n_missed,
        "n_split":     n_split,
        "agreement":   int(n_matched == n_truth and n_spurious == 0
                           and n_split == 0),
    }


def event_metrics(df_event: pd.DataFrame,
                  labels: np.ndarray,
                  method_name: str,
                  truth_window_ns: float = DEFAULT_WINDOW) -> dict:
    """
    Compute per-event metrics for one set of OPTICS (or CF) labels.

    truth_window_ns controls which neutron hits count as "recoverable signal"
    for purity/recall/F1.  Thermalization-outlier hits outside the window are
    excluded from the truth signal definition (treated as acceptable noise).
    """
    truth_class        = df_event["truth_class"].to_numpy(int)
    truth_neutron_mask = _apply_truth_window(df_event, truth_window_ns)

    predicted_is_neutron = classify_clusters_by_majority(labels, truth_class)

    total_n = int(truth_neutron_mask.sum())
    pred_n  = int(predicted_is_neutron.sum())
    tp      = int((predicted_is_neutron & truth_neutron_mask).sum())

    purity = tp / pred_n  if pred_n  else np.nan
    recall = tp / total_n if total_n else np.nan
    f1     = f1_score(truth_neutron_mask, predicted_is_neutron, zero_division=0)
    ari    = (adjusted_rand_score(truth_class, labels)
              if len(np.unique(labels)) > 1 else np.nan)

    row = {
        "eventID":             int(df_event["eventID"].iloc[0]),
        "method":              method_name,
        "n_pulses":            int(len(df_event)),
        "n_truth_neutron":     total_n,
        "n_predicted_neutron": pred_n,
        "purity":              purity,
        "recall":              recall,
        "f1":                  f1,
        "ari":                 ari,
        "n_clusters":          int((np.unique(labels) >= 0).sum()),
    }
    row.update(_optics_cluster_outcomes(df_event, labels, truth_neutron_mask))
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def train_and_evaluate(ctx: RunContext,
                       min_samples_list:  Sequence[int]   = (5,),
                       xi_list:           Sequence[float] = (0.05,),
                       t_unit_ns_list:    Sequence[float] = (DEFAULT_T_UNIT,),
                       truth_window_ns:   float           = DEFAULT_WINDOW,
                       min_pulses_per_event: int          = 3) -> pd.DataFrame:
    """
    Sweep the OPTICS hyperparameter grid and evaluate against truth.

    Parameters
    ----------
    t_unit_ns_list : sequence of floats
        Physical time units (ns) for the time axis scaling.
        Each value is swept independently; see run_optics_on_event docstring.
    truth_window_ns : float
        Single value (not swept) defining the truth signal window.
        Hits outside ±(truth_window_ns/2) of their cluster's median time
        are not counted as recoverable signal.  Default 75 ns.
    """
    pulses_path  = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if not pulses_path.exists():
        raise FileNotFoundError(
            f"{pulses_path} not found — run `ambe mc process` first"
        )

    pulses   = pd.read_parquet(pulses_path)
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    rows      = []
    event_ids = pulses["eventID"].unique()
    print(f"[mc.optics] {len(event_ids)} events, {len(pulses)} pulses")
    print(f"[mc.optics] truth_window_ns={truth_window_ns} ns  |  "
          f"t_unit_ns grid={list(t_unit_ns_list)}")

    for evid in event_ids:
        df_ev = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        if len(df_ev) < min_pulses_per_event:
            continue

        # --- OPTICS grid sweep ---
        for ms, xi, t_unit in itertools.product(min_samples_list, xi_list,
                                                t_unit_ns_list):
            labels = run_optics_on_event(df_ev, min_samples=ms, xi=xi,
                                         t_unit_ns=t_unit)
            row = event_metrics(df_ev, labels, "optics",
                                truth_window_ns=truth_window_ns)
            row.update({"min_samples": ms, "xi": xi, "t_unit_ns": t_unit,
                        "truth_window_ns": truth_window_ns})
            rows.append(row)

        # --- ClusterFinder baseline (same truth window for fair comparison) ---
        if clusters is not None:
            df_cl     = clusters[clusters["eventID"] == evid].reset_index(drop=True)
            labels_cf = assign_clusterfinder_labels(df_ev, df_cl)
            row = event_metrics(df_ev, labels_cf, "clusterfinder",
                                truth_window_ns=truth_window_ns)
            row.update({"min_samples": np.nan, "xi": np.nan,
                        "t_unit_ns": np.nan, "truth_window_ns": truth_window_ns})
            rows.append(row)

    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(
        ["method", "min_samples", "xi", "t_unit_ns"], dropna=False
    ).agg(
        events=("eventID",       "count"),
        mean_purity=("purity",   "mean"),
        mean_recall=("recall",   "mean"),
        mean_f1=("f1",           "mean"),
        mean_ari=("ari",         "mean"),
        mean_n_clusters=("n_clusters",   "mean"),
        mean_n_truth=("n_truth",         "mean"),
        mean_n_predicted=("n_predicted", "mean"),
        mean_matched=("n_matched",       "mean"),
        mean_spurious=("n_spurious",     "mean"),
        mean_missed=("n_missed",         "mean"),
        mean_split=("n_split",           "mean"),
        agreement_rate=("agreement",     "mean"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #

def _grid_from_ctx(ctx: RunContext):
    optics_block     = ctx.extra.get("optics", {}) or {}
    min_samples      = optics_block.get("min_samples",   [5])
    xi               = optics_block.get("xi",            [0.05])
    t_unit_ns        = optics_block.get("t_unit_ns",     [DEFAULT_T_UNIT])
    truth_window_ns  = float(optics_block.get("truth_window_ns", DEFAULT_WINDOW))

    t_unit_ns = [float(v) for v in t_unit_ns]
    return tuple(min_samples), tuple(xi), tuple(t_unit_ns), truth_window_ns


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None) -> Path:
    p = argparse.ArgumentParser(prog="ambe mc optics")
    p.add_argument("--min-pulses", type=int,
                   default=int(ctx.cuts.get("min_pulses_per_event", 3)))
    args = p.parse_args(list(argv) if argv else [])

    ms, xi, t_units, truth_win = _grid_from_ctx(ctx)
    metrics = train_and_evaluate(
        ctx,
        min_samples_list=ms,
        xi_list=xi,
        t_unit_ns_list=t_units,
        truth_window_ns=truth_win,
        min_pulses_per_event=args.min_pulses,
    )

    out = ctx.parquet_path(f"{ctx.run_name}__metrics")
    metrics.to_parquet(out, index=False)

    summary     = summarise(metrics)
    summary_csv = ctx.csv_path(f"{ctx.run_name}__optics_summary")
    summary.to_csv(summary_csv, index=False)

    print(f"[mc.optics] wrote metrics -> {out}")
    print(f"[mc.optics] wrote summary -> {summary_csv}")
    print(summary.to_string(index=False))
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)
