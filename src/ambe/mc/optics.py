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
MIN_MATCH_FRAC  = 0.5   # fraction of cluster hits that must come from dominant trackID


# --------------------------------------------------------------------------- #
# OPTICS per event
# --------------------------------------------------------------------------- #
def run_optics_on_event(df_event, min_samples=10, xi=0.05, metric="euclidean",
                        t_scale=None):
    X = df_event[["x", "y", "z", "t"]].to_numpy(dtype=float, copy=True)
    if t_scale is not None:
        X[:, 3] *= t_scale
    if len(X) < min_samples:
        return np.full(len(X), -1, dtype=int)
    X_scaled = StandardScaler().fit_transform(X)
    return OPTICS(min_samples=min_samples, xi=xi, metric=metric).fit_predict(X_scaled)


# --------------------------------------------------------------------------- #
# ClusterFinder baseline: time-proximity membership
# --------------------------------------------------------------------------- #
def assign_clusterfinder_labels(df_pulses, df_clusters,
                                tmin_margin_ns=5.0, tmax_margin_ns=20.0):
    if df_clusters is None or len(df_clusters) == 0:
        return np.full(len(df_pulses), -1, dtype=int)
    ct = df_clusters["clusterTime"].to_numpy(float)
    labels = np.full(len(df_pulses), -1, dtype=int)
    pulse_t = df_pulses["t"].to_numpy(float)
    for i, tp in enumerate(pulse_t):
        dt = tp - ct
        in_window = (dt > -tmin_margin_ns) & (dt < tmax_margin_ns)
        if not in_window.any():
            continue
        candidates = np.where(in_window)[0]
        labels[i] = int(candidates[np.argmin(np.abs(dt[candidates]))])
    return labels


# --------------------------------------------------------------------------- #
# Cluster -> neutron/non-neutron decision, and event-level metrics
# --------------------------------------------------------------------------- #
def classify_clusters_by_majority(labels, truth_class):
    labels = np.asarray(labels)
    truth_class = np.asarray(truth_class)
    predicted_is_neutron = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        if c == -1:
            continue
        mask = labels == c
        members = truth_class[mask]
        n_neutron = np.isin(members, list(NEUTRON_CLASSES)).sum()
        if n_neutron > 0 and n_neutron >= 0.5 * mask.sum():
            predicted_is_neutron[mask] = True
    return predicted_is_neutron


def _optics_cluster_outcomes(df_event, labels) -> dict:
    """
    Match predicted clusters (OPTICS or CF labels) to truth neutron clusters
    defined by unique ancestor_trackID where ancestor_pdg == 2112.

    A predicted cluster is *matched* if its dominant neutron trackID accounts
    for >= MIN_MATCH_FRAC of all its member hits.

    Returns keys: n_truth, n_predicted, n_matched, n_spurious, n_missed,
                  n_split, agreement (bool: n_matched==n_truth, no spurious, no split).
    """
    labels      = np.asarray(labels, dtype=int)
    track_ids   = df_event["ancestor_trackID"].to_numpy(int)
    pdgs        = df_event["ancestor_pdg"].to_numpy(int)

    truth_trackIDs = set(
        int(tid) for tid, pdg in zip(track_ids, pdgs)
        if pdg == NEUTRON_PDG and tid >= 0
    )
    n_truth = len(truth_trackIDs)

    unique_labels  = [c for c in np.unique(labels) if c >= 0]
    n_predicted    = len(unique_labels)

    matched_tids  = set()
    split_tids    = set()
    n_spurious    = 0

    for c in unique_labels:
        mask    = labels == c
        n_total = int(mask.sum())

        # dominant neutron trackID among member hits
        neutron_mask = mask & (pdgs == NEUTRON_PDG) & (track_ids >= 0)
        if not neutron_mask.any():
            n_spurious += 1
            continue

        tids_in_cluster, counts = np.unique(track_ids[neutron_mask], return_counts=True)
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
        "agreement":   int(n_matched == n_truth and n_spurious == 0 and n_split == 0),
    }


def event_metrics(df_event, labels, method_name):
    truth_class = df_event["truth_class"].to_numpy(int)
    truth_is_neutron = df_event["is_neutron"].to_numpy(int).astype(bool)
    predicted_is_neutron = classify_clusters_by_majority(labels, truth_class)

    total_n = int(truth_is_neutron.sum())
    pred_n = int(predicted_is_neutron.sum())
    tp = int((predicted_is_neutron & truth_is_neutron).sum())

    purity = tp / pred_n if pred_n else np.nan
    recall = tp / total_n if total_n else np.nan
    f1 = f1_score(truth_is_neutron, predicted_is_neutron, zero_division=0)
    ari = adjusted_rand_score(truth_class, labels) if len(np.unique(labels)) > 1 else np.nan

    row = {
        "eventID": int(df_event["eventID"].iloc[0]),
        "method": method_name,
        "n_pulses": int(len(df_event)),
        "n_truth_neutron": total_n,
        "n_predicted_neutron": pred_n,
        "purity": purity,
        "recall": recall,
        "f1": f1,
        "ari": ari,
        "n_clusters": int((np.unique(labels) >= 0).sum()),
    }
    row.update(_optics_cluster_outcomes(df_event, labels))
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def train_and_evaluate(ctx: RunContext,
                       min_samples_list: Sequence[int] = (10,),
                       xi_list: Sequence[float] = (0.05,),
                       t_scale_list: Sequence = (None,),
                       min_pulses_per_event: int = 3) -> pd.DataFrame:
    pulses_path = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if not pulses_path.exists():
        raise FileNotFoundError(
            f"{pulses_path} not found -- run `ambe mc process` first"
        )

    pulses = pd.read_parquet(pulses_path)
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    rows = []
    event_ids = pulses["eventID"].unique()
    print(f"[mc.optics] {len(event_ids)} events, {len(pulses)} pulses")

    for evid in event_ids:
        df_ev = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        if len(df_ev) < min_pulses_per_event:
            continue

        for ms, xi, ts in itertools.product(min_samples_list, xi_list, t_scale_list):
            labels = run_optics_on_event(df_ev, min_samples=ms, xi=xi, t_scale=ts)
            row = event_metrics(df_ev, labels, "optics")
            row.update({"min_samples": ms, "xi": xi, "t_scale": ts})
            rows.append(row)

        if clusters is not None:
            df_cl = clusters[clusters["eventID"] == evid].reset_index(drop=True)
            labels_cf = assign_clusterfinder_labels(df_ev, df_cl)
            row = event_metrics(df_ev, labels_cf, "clusterfinder")
            row.update({"min_samples": np.nan, "xi": np.nan, "t_scale": np.nan})
            rows.append(row)

    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(
        ["method", "min_samples", "xi", "t_scale"], dropna=False
    ).agg(
        events=("eventID", "count"),
        mean_purity=("purity", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_ari=("ari", "mean"),
        mean_n_clusters=("n_clusters", "mean"),
        mean_n_truth=("n_truth", "mean"),
        mean_n_predicted=("n_predicted", "mean"),
        mean_matched=("n_matched", "mean"),
        mean_spurious=("n_spurious", "mean"),
        mean_missed=("n_missed", "mean"),
        mean_split=("n_split", "mean"),
        agreement_rate=("agreement", "mean"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #
def _grid_from_ctx(ctx: RunContext):
    optics_block = ctx.extra.get("optics", {}) or {}
    min_samples = optics_block.get("min_samples", [10])
    xi = optics_block.get("xi", [0.05])
    t_scale = optics_block.get("t_scale", [None])
    t_scale = [None if v is None else float(v) for v in t_scale]
    return tuple(min_samples), tuple(xi), tuple(t_scale)


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None) -> Path:
    p = argparse.ArgumentParser(prog="ambe mc optics")
    p.add_argument("--min-pulses", type=int,
                   default=int(ctx.cuts.get("min_pulses_per_event", 3)))
    args = p.parse_args(list(argv) if argv else [])

    ms, xi, ts = _grid_from_ctx(ctx)
    metrics = train_and_evaluate(
        ctx, min_samples_list=ms, xi_list=xi, t_scale_list=ts,
        min_pulses_per_event=args.min_pulses,
    )
    out = ctx.parquet_path(f"{ctx.run_name}__metrics")
    metrics.to_parquet(out, index=False)

    summary = summarise(metrics)
    summary_csv = ctx.csv_path(f"{ctx.run_name}__optics_summary")
    summary.to_csv(summary_csv, index=False)

    print(f"[mc.optics] wrote metrics -> {out}")
    print(f"[mc.optics] wrote summary -> {summary_csv}")
    print(summary.to_string(index=False))
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)
