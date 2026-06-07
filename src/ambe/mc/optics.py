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
import time
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler

from ..context import RunContext


NEUTRON_CLASSES   = {1, 2, 3, 4}
NEUTRON_PDG       = 2112
MIN_MATCH_FRAC    = 0.5     # fraction of cluster hits from dominant trackID to call it matched
DEFAULT_T_UNIT    = 50.0    # ns — default time-axis unit (see module docstring)
DEFAULT_WINDOW    = 75.0    # ns — default truth window (see module docstring)
DEFAULT_PREFILTER = 2000.0  # ns — default hit pre-filter window (see _prefilter_hits)

# Clusters are tagged as "prompt signal" (not real spurious) when EITHER:
#   1. Timing:     cluster mean time < (capture ref - PROMPT_SIGNAL_THRESHOLD_NS).
#      Catches AmBe gamma (~-18,000 ns), muon prompt light (~-30,000 ns), etc.
#   2. Composition: cluster is class-5 dominant (>50%) with only stray neutron
#      contamination (≤ PROMPT_SIGNAL_MAX_NEUTRON_HITS hits).
#      Catches near-capture class-5 clusters that picked up 1-2 stray neutron hits.
PROMPT_SIGNAL_THRESHOLD_NS     = -500.0
PROMPT_SIGNAL_MAX_NEUTRON_HITS = 2       # stray neutron hits ≤ this → prompt signal

# Speed of light in water (m/ns).  Used for source ToF correction.
SOL_WATER = 0.299792458 * 0.75


# --------------------------------------------------------------------------- #
# Hit pre-filter  (performance-critical)
# --------------------------------------------------------------------------- #

def _prefilter_hits(df_event: pd.DataFrame,
                    df_clusters,
                    prefilter_ns: float = DEFAULT_PREFILTER) -> pd.DataFrame:
    """
    Reduce the number of hits fed to OPTICS to those within a time window
    around any ClusterFinder cluster time.

    Why this matters
    ----------------
    The processor writes *every* detector hit (dark noise + all physics),
    so a typical WCSim event can contain 200-500+ hits spread across a long
    event window.  OPTICS is O(n² log n), so going from 200 → 40 hits is a
    25× speedup per fit — turning an overnight run into ~10 minutes.

    The filter keeps hits within ±(prefilter_ns/2) of any CF cluster time.
    This captures all neutron capture hits (within ~40 ns of the cluster)
    plus a small fraction of dark noise from the same time region.  OPTICS
    still needs to classify that dark noise — it just doesn't need to see the
    dark noise from the other 49 µs of the event.

    Fallback behaviour
    ------------------
    - If df_clusters is None or empty: no filter is applied (return all hits).
    - If the filter removes so many hits that fewer than 2 remain: return
      all hits (avoids pathological empty-event edge cases).
    """
    if df_clusters is None or len(df_clusters) == 0:
        return df_event

    ct      = df_clusters["clusterTime"].to_numpy(float)
    hit_t   = df_event["t"].to_numpy(float)
    half_w  = prefilter_ns / 2.0

    mask = np.zeros(len(df_event), dtype=bool)
    for c in ct:
        mask |= (np.abs(hit_t - c) <= half_w)

    # Keep original integer index — caller uses it to re-align labels back
    # onto the full df_event via np indexing.  Do NOT reset_index here.
    filtered = df_event[mask]
    return filtered if len(filtered) >= 2 else df_event


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
                        min_samples:  int   = 5,
                        xi:           float = 0.05,
                        metric:       str   = "euclidean",
                        t_unit_ns:    float = DEFAULT_T_UNIT,
                        source_pos_m: "Optional[np.ndarray]" = None) -> np.ndarray:
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
        With source ToF correction, signal hits cluster within ~2-5 ns,
        so a smaller value (5-10 ns) becomes appropriate.

    source_pos_m : np.ndarray of shape (3,) or None
        Known AmBe source position in **metres** (same coordinate system as
        hit x,y,z in the parquet).  When provided, the time axis is replaced
        by the ToF-corrected time:

            t_corr_i = t_i  -  |r_pmt_i - r_source| / c_water

        This collapses all Cherenkov hits from a genuine neutron capture
        (emitted nearly simultaneously at a point near the source) to within
        ~2-5 ns, regardless of which PMT they hit.  Dark noise and beam-
        induced background hits at random times are not collapsed and remain
        spread.  This makes OPTICS clusters much more compact in the time
        dimension, allowing a smaller t_unit_ns and lower min_samples.

        IMPORTANT: source_pos_m must be in metres.  The source_positions dict
        in data/processor.py stores positions in cm — divide by 100 before
        passing here.  Default (None) disables the correction.
    """
    X = df_event[["x", "y", "z", "t"]].to_numpy(dtype=float, copy=True)

    # Step 1: scale xyz spatially
    X_xyz = StandardScaler().fit_transform(X[:, :3])

    # Step 2: build the time axis
    t_raw = X[:, 3]
    if source_pos_m is not None:
        # Source ToF correction: subtract photon travel time from source to each PMT.
        # For AmBe analysis the source position is known, so this gives a much
        # tighter time representation for genuine Cherenkov clusters.
        src = np.asarray(source_pos_m, dtype=float)
        dists = np.linalg.norm(X[:, :3] - src[np.newaxis, :], axis=1)
        t_axis = t_raw - dists / SOL_WATER
    else:
        t_axis = t_raw

    X_t = t_axis[:, np.newaxis] / t_unit_ns

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
                  n_split, agreement,
                  n_prompt_clusters, n_real_spurious, corrected_agreement.

    n_prompt_clusters   — subset of spurious clusters identified as prompt signal
                          (AmBe gamma, muon light, etc.) by timing criterion
                          (offset < PROMPT_SIGNAL_THRESHOLD_NS) OR composition
                          (class-5 dominant with ≤ PROMPT_SIGNAL_MAX_NEUTRON_HITS
                          stray neutron hits).  These are not false positives.
    n_real_spurious     — n_spurious - n_prompt_clusters: true false-positive clusters.
    corrected_agreement — agreement counting only n_real_spurious (not prompt clusters).
    """
    labels      = np.asarray(labels, dtype=int)
    track_ids   = df_event["ancestor_trackID"].to_numpy(int)
    pdgs        = df_event["ancestor_pdg"].to_numpy(int)
    hit_times   = df_event["t"].to_numpy(float)
    truth_class = df_event["truth_class"].to_numpy(int)

    # Truth neutron capture reference time (median of in-window neutron hits)
    neu_times = hit_times[truth_neutron_mask] if truth_neutron_mask.any() else np.array([])
    neu_ref_t = float(np.median(neu_times)) if len(neu_times) > 0 else np.nan

    # Truth neutron trackIDs: only those with at least one in-window hit
    truth_trackIDs = set(
        int(tid)
        for tid, pdg, in_win in zip(track_ids, pdgs, truth_neutron_mask)
        if pdg == NEUTRON_PDG and tid >= 0 and in_win
    )
    n_truth = len(truth_trackIDs)

    unique_labels = [c for c in np.unique(labels) if c >= 0]
    n_predicted   = len(unique_labels)

    matched_tids     = set()
    split_tids       = set()
    n_spurious       = 0
    n_prompt_clusters = 0

    for c in unique_labels:
        mask    = labels == c
        n_total = int(mask.sum())

        n_neutron_c = int(np.isin(truth_class[mask], list(NEUTRON_CLASSES)).sum())
        n_class5_c  = int((truth_class[mask] == -5).sum())

        def _is_prompt(mean_t):
            timing = (not np.isnan(neu_ref_t) and
                      (mean_t - neu_ref_t) < PROMPT_SIGNAL_THRESHOLD_NS)
            comp   = (n_class5_c > n_total / 2 and
                      n_neutron_c <= PROMPT_SIGNAL_MAX_NEUTRON_HITS)
            return timing or comp

        neutron_mask_c = mask & (pdgs == NEUTRON_PDG) & (track_ids >= 0)
        if not neutron_mask_c.any():
            if _is_prompt(float(hit_times[mask].mean())):
                n_prompt_clusters += 1
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
            if _is_prompt(float(hit_times[mask].mean())):
                n_prompt_clusters += 1
            n_spurious += 1

    n_matched       = len(matched_tids)
    n_missed        = len(truth_trackIDs - matched_tids)
    n_split         = len(split_tids)
    n_real_spurious = n_spurious - n_prompt_clusters

    return {
        "n_truth":              n_truth,
        "n_predicted":          n_predicted,
        "n_matched":            n_matched,
        "n_spurious":           n_spurious,            # all non-matched (incl. prompt)
        "n_prompt_clusters":    n_prompt_clusters,     # subset: prompt signal clusters
        "n_real_spurious":      n_real_spurious,       # true false positives only
        "n_missed":             n_missed,
        "n_split":              n_split,
        "agreement":            int(n_matched == n_truth and n_spurious == 0
                                    and n_split == 0),
        "corrected_agreement":  int(n_matched == n_truth and n_real_spurious == 0
                                    and n_split == 0),
    }


def event_metrics(df_event: pd.DataFrame,
                  labels: np.ndarray,
                  method_name: str,
                  truth_window_ns: float = DEFAULT_WINDOW,
                  truth_neutron_mask: np.ndarray = None) -> dict:
    """
    Compute per-event metrics for one set of OPTICS (or CF) labels.

    truth_window_ns controls which neutron hits count as "recoverable signal"
    for purity/recall/F1.  Thermalization-outlier hits outside the window are
    excluded from the truth signal definition (treated as acceptable noise).

    truth_neutron_mask : optional pre-computed boolean array from
        _apply_truth_window().  Pass this when calling inside the grid loop to
        avoid recomputing the same mask for every hyperparameter combination.
    """
    truth_class        = df_event["truth_class"].to_numpy(int)
    if truth_neutron_mask is None:
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
                       min_samples_list:    Sequence[int]   = (5,),
                       xi_list:             Sequence[float] = (0.05,),
                       t_unit_ns_list:      Sequence[float] = (DEFAULT_T_UNIT,),
                       truth_window_ns:     float           = DEFAULT_WINDOW,
                       hit_prefilter_ns:    float           = DEFAULT_PREFILTER,
                       source_pos_m:        "Optional[np.ndarray]" = None,
                       min_pulses_per_event: int            = 3) -> pd.DataFrame:
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
    hit_prefilter_ns : float
        Time window (ns) around each CF cluster time to keep hits for OPTICS.
        Reduces the per-event hit count from ~200-500 (all detector hits) to
        ~30-50 (hits near the capture region), giving a ~25x OPTICS speedup.
        Set to 0 or np.inf to disable.  Default 2000 ns.
    source_pos_m : np.ndarray of shape (3,) or None
        Known AmBe source position in metres for ToF correction.
        See run_optics_on_event docstring.  Default None (no correction).
    """
    pulses_path   = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if not pulses_path.exists():
        raise FileNotFoundError(
            f"{pulses_path} not found — run `ambe mc process` first"
        )

    pulses   = pd.read_parquet(pulses_path)
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    # Reduce to the delayed residual (CC-passing events, prompt window removed)
    # so OPTICS clusters the post-muon hit population. See cc_selection.
    from . import cc_selection
    pulses = cc_selection.apply_residual_filter(pulses, ctx, verbose=True)

    rows      = []
    event_ids = pulses["eventID"].unique()
    n_hits_after_filter = []

    n_configs   = len(min_samples_list) * len(xi_list) * len(t_unit_ns_list)
    print(f"[mc.optics] {len(event_ids)} events, {len(pulses)} total hits")
    print(f"[mc.optics] truth_window_ns={truth_window_ns} ns  |  "
          f"hit_prefilter_ns={hit_prefilter_ns} ns  |  "
          f"t_unit_ns grid={list(t_unit_ns_list)}  |  "
          f"{n_configs} configs per event")

    t_start     = time.time()
    n_total     = len(event_ids)
    print_every = max(1, n_total // 20)

    for i_ev, evid in enumerate(event_ids):
        df_ev = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        if len(df_ev) < min_pulses_per_event:
            continue

        df_cl = (clusters[clusters["eventID"] == evid].reset_index(drop=True)
                 if clusters is not None else None)

        # --- Performance: pre-filter hits to CF cluster time window ---
        # Reduces n_hits from ~200-500 to ~30-50, giving ~25x OPTICS speedup.
        # Truth mask and CF baseline still use the FULL df_ev so truth
        # accounting is unaffected by the filter.
        if hit_prefilter_ns > 0 and not np.isinf(hit_prefilter_ns):
            df_optics = _prefilter_hits(df_ev, df_cl, hit_prefilter_ns)
        else:
            df_optics = df_ev
        n_hits_after_filter.append(len(df_optics))

        # --- Performance: compute truth mask ONCE per event (not per config) ---
        truth_mask = _apply_truth_window(df_ev, truth_window_ns)

        # --- OPTICS grid sweep ---
        for ms, xi, t_unit in itertools.product(min_samples_list, xi_list,
                                                t_unit_ns_list):
            labels_optics = run_optics_on_event(df_optics, min_samples=ms,
                                                xi=xi, t_unit_ns=t_unit,
                                                source_pos_m=source_pos_m)

            # Re-align OPTICS labels onto the full df_ev length.
            # _prefilter_hits preserves the original df_ev integer index, so
            # df_optics.index maps directly back into [0, len(df_ev)-1].
            # Hits excluded by the filter are assigned noise label (-1).
            if len(df_optics) < len(df_ev):
                labels_full = np.full(len(df_ev), -1, dtype=int)
                labels_full[df_optics.index] = labels_optics
            else:
                labels_full = labels_optics

            row = event_metrics(df_ev, labels_full, "optics",
                                truth_window_ns=truth_window_ns,
                                truth_neutron_mask=truth_mask)
            src_str = (f"{source_pos_m[0]:.3f},{source_pos_m[1]:.3f},"
                       f"{source_pos_m[2]:.3f}"
                       if source_pos_m is not None else "none")
            row.update({"min_samples": ms, "xi": xi, "t_unit_ns": t_unit,
                        "truth_window_ns": truth_window_ns,
                        "hit_prefilter_ns": hit_prefilter_ns,
                        "source_pos_m": src_str,
                        "n_hits_optics": len(df_optics)})
            rows.append(row)

        # --- ClusterFinder baseline (full event, same truth window) ---
        if df_cl is not None:
            labels_cf = assign_clusterfinder_labels(df_ev, df_cl)
            row = event_metrics(df_ev, labels_cf, "clusterfinder",
                                truth_window_ns=truth_window_ns,
                                truth_neutron_mask=truth_mask)
            row.update({"min_samples": np.nan, "xi": np.nan,
                        "t_unit_ns": np.nan,
                        "truth_window_ns": truth_window_ns,
                        "hit_prefilter_ns": np.nan,
                        "n_hits_optics": len(df_ev)})
            rows.append(row)

        if (i_ev + 1) % print_every == 0 or (i_ev + 1) == n_total:
            elapsed   = time.time() - t_start
            rate      = (i_ev + 1) / elapsed
            remaining = (n_total - i_ev - 1) / rate if rate > 0 else 0
            n_hits_ev = n_hits_after_filter[-1] if n_hits_after_filter else len(df_ev)
            print(f"[mc.optics]  {i_ev+1:5d}/{n_total}  "
                  f"({100*(i_ev+1)/n_total:.0f}%)  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.1f} ev/s  "
                  f"ETA={remaining:.0f}s  hits/ev={n_hits_ev}", flush=True)

    if n_hits_after_filter:
        print(f"[mc.optics] hits fed to OPTICS: "
              f"mean={np.mean(n_hits_after_filter):.0f}  "
              f"median={np.median(n_hits_after_filter):.0f}  "
              f"max={np.max(n_hits_after_filter)}")

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
        mean_matched=("n_matched",                   "mean"),
        mean_spurious=("n_spurious",                 "mean"),  # all non-matched incl. prompt
        mean_prompt_clusters=("n_prompt_clusters", "mean"),  # prompt signal clusters
        mean_real_spurious=("n_real_spurious",     "mean"),  # true false positives only
        mean_missed=("n_missed",                   "mean"),
        mean_split=("n_split",                     "mean"),
        agreement_rate=("agreement",               "mean"),
        corrected_agreement_rate=("corrected_agreement", "mean"),  # excl. gamma clusters
    ).reset_index()


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #

def _grid_from_ctx(ctx: RunContext):
    optics_block     = ctx.extra.get("optics", {}) or {}
    min_samples      = optics_block.get("min_samples",      [5])
    xi               = optics_block.get("xi",               [0.05])
    t_unit_ns        = optics_block.get("t_unit_ns",        [DEFAULT_T_UNIT])
    truth_window_ns  = float(optics_block.get("truth_window_ns",  DEFAULT_WINDOW))
    hit_prefilter_ns = float(optics_block.get("hit_prefilter_ns", DEFAULT_PREFILTER))

    # Source ToF correction: read source_position_m from config (metres).
    # Example YAML:
    #   optics:
    #     source_position_m: [0.0, 0.0, 0.0]   # Port 5 centre (AmBe default)
    # Leave unset or null to disable.  Data pipeline stores positions in cm —
    # convert before putting in config (divide by 100).
    src_raw = optics_block.get("source_position_m", None)
    if src_raw is not None:
        source_pos_m = np.array([float(v) for v in src_raw], dtype=float)
        print(f"[mc.optics] source ToF correction enabled: "
              f"source at ({source_pos_m[0]:.3f}, {source_pos_m[1]:.3f}, "
              f"{source_pos_m[2]:.3f}) m")
    else:
        source_pos_m = None

    t_unit_ns = [float(v) for v in t_unit_ns]
    return (tuple(min_samples), tuple(xi), tuple(t_unit_ns),
            truth_window_ns, hit_prefilter_ns, source_pos_m)


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None) -> Path:
    p = argparse.ArgumentParser(prog="ambe mc optics")
    p.add_argument("--min-pulses", type=int,
                   default=int(ctx.cuts.get("min_pulses_per_event", 3)))
    args = p.parse_args(list(argv) if argv else [])

    ms, xi, t_units, truth_win, prefilter, src_pos = _grid_from_ctx(ctx)
    metrics = train_and_evaluate(
        ctx,
        min_samples_list=ms,
        xi_list=xi,
        t_unit_ns_list=t_units,
        truth_window_ns=truth_win,
        hit_prefilter_ns=prefilter,
        source_pos_m=src_pos,
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
