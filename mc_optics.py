"""
mc_optics.py
------------
Train and evaluate OPTICS space-time clustering on the per-pulse Parquet files
produced by mc_processor.py, using DirectParent_NeutronAncestorClass as the
ground-truth label.

Also evaluates the existing ClusterFinder output (from the *_clusterfinder.parquet
sidecar) against the same truth labels so OPTICS can be compared head-to-head.

Metrics reported per event (and averaged over events):
  - purity       : fraction of truth-neutron pulses inside predicted "neutron" clusters
  - recall       : fraction of truth-neutron pulses that end up in any predicted
                   "neutron" cluster (= efficiency)
  - f1           : binary F1 on the is_neutron flag
  - ari          : adjusted Rand index between predicted cluster ids and truth class

OPTICS feature vector: (x, y, z, t) scaled with StandardScaler per event.
Hyperparameters: min_samples, xi, metric (defaults below).
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler


NEUTRON_CLASSES = {1, 2, 3, 4}


# --------------------------------------------------------------------------- #
# OPTICS
# --------------------------------------------------------------------------- #
def run_optics_on_event(df_event, min_samples=10, xi=0.05, metric="euclidean",
                        t_scale=None):
    """
    Run OPTICS on one event's pulses. Returns an integer label per pulse
    (-1 means noise / unclustered).

    If t_scale is given, multiply t by that factor before StandardScaler.
    This lets you bias the scale ratio between time and space.
    """
    X = df_event[["x", "y", "z", "t"]].to_numpy(dtype=float, copy=True)
    if t_scale is not None:
        X[:, 3] *= t_scale
    if len(X) < min_samples:
        return np.full(len(X), -1, dtype=int)
    X_scaled = StandardScaler().fit_transform(X)
    return OPTICS(min_samples=min_samples, xi=xi, metric=metric).fit_predict(X_scaled)


# --------------------------------------------------------------------------- #
# ClusterFinder baseline
# --------------------------------------------------------------------------- #
def assign_clusterfinder_labels(df_pulses_event, df_clusters_event,
                                tmin_margin_ns=5.0, tmax_margin_ns=20.0):
    """
    Assign each pulse to a ClusterFinder cluster by time proximity.

    ClusterFinder exposes its clusters as (clusterTime, clusterHits). Without
    the per-cluster hit list we approximate membership: any pulse whose
    time falls in [clusterTime - tmin_margin, clusterTime + tmax_margin]
    is assigned to that cluster. Multiple clusters can claim the same pulse
    in principle; we break ties by taking the closest in time.

    Returns an integer label per pulse (-1 = not in any ClusterFinder cluster).
    """
    if df_clusters_event is None or len(df_clusters_event) == 0:
        return np.full(len(df_pulses_event), -1, dtype=int)

    ct = df_clusters_event["clusterTime"].to_numpy(float)
    labels = np.full(len(df_pulses_event), -1, dtype=int)
    pulse_t = df_pulses_event["t"].to_numpy(float)

    for i, tp in enumerate(pulse_t):
        dt = tp - ct
        in_window = (dt > -tmin_margin_ns) & (dt < tmax_margin_ns)
        if not in_window.any():
            continue
        candidates = np.where(in_window)[0]
        best = candidates[np.argmin(np.abs(dt[candidates]))]
        labels[i] = int(best)

    return labels


# --------------------------------------------------------------------------- #
# Cluster -> neutron/non-neutron mapping
# --------------------------------------------------------------------------- #
def classify_clusters_by_majority(labels, truth_class):
    """
    For each predicted cluster (labels >= 0), decide whether it is a
    "neutron" cluster based on majority-vote truth class.

    Returns a boolean array (len == len(labels)) indicating whether each
    pulse is inside a predicted-neutron cluster.
    """
    labels = np.asarray(labels)
    truth_class = np.asarray(truth_class)
    predicted_is_neutron = np.zeros(len(labels), dtype=bool)

    for c in np.unique(labels):
        if c == -1:  # OPTICS noise / unassigned
            continue
        mask = labels == c
        members = truth_class[mask]
        n_neutron = np.isin(members, list(NEUTRON_CLASSES)).sum()
        if n_neutron > 0 and n_neutron >= 0.5 * mask.sum():
            predicted_is_neutron[mask] = True

    return predicted_is_neutron


# --------------------------------------------------------------------------- #
# Per-event metrics
# --------------------------------------------------------------------------- #
def event_metrics(df_event, labels, method_name):
    """Return a dict of metrics for one event + one clustering method."""
    truth_class = df_event["truth_class"].to_numpy(int)
    truth_is_neutron = df_event["is_neutron"].to_numpy(int).astype(bool)

    predicted_is_neutron = classify_clusters_by_majority(labels, truth_class)

    total_neutron = int(truth_is_neutron.sum())
    predicted_neutron = int(predicted_is_neutron.sum())
    tp = int((predicted_is_neutron & truth_is_neutron).sum())

    purity = tp / predicted_neutron if predicted_neutron else np.nan
    recall = tp / total_neutron if total_neutron else np.nan
    f1 = f1_score(truth_is_neutron, predicted_is_neutron, zero_division=0)
    ari = adjusted_rand_score(truth_class, labels) if len(np.unique(labels)) > 1 else np.nan

    return {
        "eventID": int(df_event["eventID"].iloc[0]),
        "method": method_name,
        "n_pulses": int(len(df_event)),
        "n_truth_neutron": total_neutron,
        "n_predicted_neutron": predicted_neutron,
        "purity": purity,
        "recall": recall,
        "f1": f1,
        "ari": ari,
        "n_clusters": int((np.unique(labels) >= 0).sum()),
    }


# --------------------------------------------------------------------------- #
# Driver: process one parquet pair, possibly with a hyperparameter grid
# --------------------------------------------------------------------------- #
def train_and_evaluate(pulses_parquet, clusters_parquet=None,
                       min_samples_list=(10,), xi_list=(0.05,),
                       t_scale_list=(None,), verbose=True):
    """
    Run OPTICS over a grid of (min_samples, xi, t_scale) AND ClusterFinder
    baseline on the same events. Returns a DataFrame with one row per
    (event, method, hyperparameters).
    """
    pulses = pd.read_parquet(pulses_parquet)
    clusters = pd.read_parquet(clusters_parquet) if clusters_parquet else None

    rows = []
    event_ids = pulses["eventID"].unique()
    if verbose:
        print(f"[mc_optics] {len(event_ids)} events, {len(pulses)} pulses total")

    for evid in event_ids:
        df_ev = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        if len(df_ev) < 3:
            continue

        # OPTICS grid
        for min_samples, xi, t_scale in itertools.product(min_samples_list, xi_list, t_scale_list):
            labels = run_optics_on_event(df_ev, min_samples=min_samples, xi=xi, t_scale=t_scale)
            row = event_metrics(df_ev, labels, method_name="optics")
            row.update({"min_samples": min_samples, "xi": xi, "t_scale": t_scale})
            rows.append(row)

        # ClusterFinder baseline
        if clusters is not None:
            df_cl = clusters[clusters["eventID"] == evid].reset_index(drop=True)
            labels_cf = assign_clusterfinder_labels(df_ev, df_cl)
            row = event_metrics(df_ev, labels_cf, method_name="clusterfinder")
            row.update({"min_samples": np.nan, "xi": np.nan, "t_scale": np.nan})
            rows.append(row)

    return pd.DataFrame(rows)


def summarise(metrics_df):
    """Average metrics per (method, hyperparameters). Returns a DataFrame."""
    group_cols = ["method", "min_samples", "xi", "t_scale"]
    agg = metrics_df.groupby(group_cols, dropna=False).agg(
        events=("eventID", "count"),
        mean_purity=("purity", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_ari=("ari", "mean"),
        mean_n_clusters=("n_clusters", "mean"),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("pulses_parquet", help="per-pulse parquet file produced by mc_processor.py")
    p.add_argument("--clusters", default=None,
                   help="ClusterFinder sidecar parquet; inferred if omitted")
    p.add_argument("--out", default="metrics.parquet", help="Output metrics parquet")
    p.add_argument("--min-samples", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--xi", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--t-scale", type=float, nargs="*", default=[None],
                   help="Multiplicative pre-scale on t before StandardScaler. "
                        "None = equal-weight via StandardScaler alone.")
    args = p.parse_args()

    pulses_parquet = Path(args.pulses_parquet)
    clusters_parquet = Path(args.clusters) if args.clusters else \
        pulses_parquet.with_name(pulses_parquet.stem + "_clusterfinder.parquet")
    if not clusters_parquet.exists():
        print(f"[mc_optics] no ClusterFinder sidecar at {clusters_parquet}; baseline skipped")
        clusters_parquet = None

    t_scale_list = tuple(None if v is None else float(v) for v in args.t_scale)

    metrics = train_and_evaluate(
        pulses_parquet, clusters_parquet,
        min_samples_list=tuple(args.min_samples),
        xi_list=tuple(args.xi),
        t_scale_list=t_scale_list,
    )
    metrics.to_parquet(args.out, index=False)
    print(f"[mc_optics] wrote {len(metrics)} metric rows -> {args.out}")

    summary = summarise(metrics)
    print("\n[mc_optics] per-configuration averages:")
    print(summary.to_string(index=False))
    summary.to_csv(Path(args.out).with_suffix(".summary.csv"), index=False)


if __name__ == "__main__":
    main()
