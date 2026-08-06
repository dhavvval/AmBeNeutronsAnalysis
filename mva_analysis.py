#!/usr/bin/env python3
"""
Cluster-level MVA for neutron signal selection.

Trains Random Forest, Gradient Boosted Tree, XGBoost, and a feedforward
neural network on physics-only cluster features. The default NN is a
lightweight test architecture (see NN_* constants); swap those to restore
the thesis-sized model.

Signal     : is_truth_neutron == 1
Background : is_truth_neutron == 0  AND  is_prompt_cluster == 0

Usage
-----
    python mva_analysis.py --config configs/mc_lucho_full.yaml
    python mva_analysis.py --config configs/mc_oneneutrononemuon.yaml
    python mva_analysis.py --config configs/mc_twoneutron.yaml

Outputs (under <output_root>/<run_name>/)
-----------------------------------------
    plots/<run_name>__mva.pdf
        Page 1: ROC curves (all models, cluster-level test set)
        Page 2: Score distributions  signal vs background (one panel per model)
        Page 3: Feature importances  RF, GBT, XGBoost
        Page 4: Top-6 feature distributions (by RF importance)
        Page 5: Neural network training history (if tensorflow available)
        Page 6: Event-level efficiency vs score threshold

    parquet/<run_name>__mva_scores.parquet
        All columns from cluster_features + {rf,gbt,xgb,nn}_score

    csv/<run_name>__mva_summary.csv
        AUC values and per-feature importance table
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
# Force the non-interactive backend BEFORE pyplot is imported. With DISPLAY set this
# otherwise picks TkAgg, and Tk's teardown aborts the interpreter (exit 134,
# "RuntimeError: main thread is not in main loop") AFTER the plots and score table are
# written but BEFORE the importance CSV — which is exactly what happened to the merged
# truthtag/clusterfinder training on 2026-08-05.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    _HAS_KERAS = True
except ImportError:
    _HAS_KERAS = False


# Physics-only features — all computable from real data (no truth labels used).
# d_source / d_source_fit require source_position_m in the config; silently
# dropped when fewer than 50% of clusters have a non-NaN value (same applies to
# fit-vertex features when the Gauss-Newton fitter fails to converge).
PHYSICS_FEATURES = [
    # ---- Multiplicity ----
    "n_hits",
    "pe_total",
    "n_hits_early",
    # ---- Timing — centroid vertex ----
    "sigma_t_mad",
    "sigma_t_mad_corr",
    "sigma_t_mad_tof",
    "sigma_t_early_mad",
    "t_window_80pct",
    # ---- Charge / spatial — centroid vertex ----
    "pe_balance",
    "charge_bal_legacy",    # legacy ANNIE CB: sqrt(ΣQ²/ΣQ² − 1/121)
    "spatial_rms",
    "d_wall",
    "d_source",
    "vtx_y",               # bottom-of-tank (Y < −0.5 m) has 19 LUX PMTs vs 92 barrel — systematic hit-count drop
    # ---- Isotropy — centroid vertex ----
    "beta1",
    "beta2",
    "beta3",
    "beta4",
    "beta5",
    # ---- Gauss-Newton fitted vertex ----
    # NaN when fit_converged==0; residual NaNs are median-imputed (see prepare_data).
    # The ≥50% non-NaN filter below keeps these even when convergence rate is ~65%.
    "fit_converged",        # 1 if fitter found an in-tank solution (itself discriminating)
    "n_fit_hits",           # hits used after causal-compatibility filtering
    "fit_rms_ns",           # timing RMS at best-fit vertex (ns)
    "fit_goodness_reco",    # SK FitGoodness with fitted vertex — analog of thesis feature 3
    "fit_goodness_init",    # SK FitGoodness with centroid — always available, no NaN
    "d_wall_fit",           # distance to wall from fitted vertex
    "sigma_t_mad_tof_fit",  # MAD timing spread ToF-corrected with fitted vertex
    "beta1_fit",            # isotropy β₁ with fitted vertex
    "beta2_fit",
    "beta3_fit",
    "beta4_fit",
    "beta5_fit",
    "vtx_fit_y",           # fitted Y: more accurate than centroid for bottom-bias detection
    "d_source_fit",        # distance to AmBe source from fitted vertex (more accurate than centroid-based)
]

# Columns that are never valid MVA inputs regardless of source — truth labels,
# bookkeeping IDs, and source-specific geometry excluded from external background runs.
_EXTERNAL_BKG_EXCLUDE = {
    "is_truth_neutron", "dominant_trackID", "cluster_time_offset_ns", "is_prompt_cluster",
    "n_neutron", "n_darknoise", "n_nonneutron", "n_untraced",
    "frac_neutron", "frac_darknoise", "frac_nonneutron", "frac_untraced", "dominant_class",
    "d_source", "d_source_fit",          # source position not meaningful for background files
    "vtx_x", "vtx_y", "vtx_z",          # vertex coordinates — not discriminating quantities
    "vtx_fit_x", "vtx_fit_y", "vtx_fit_z",
    "eventID", "event_number", "run", "cluster_id",
    "cf_cluster_id", "optics_cluster_id", "n_optics_noise",
    "method", "is_background", "t_mean",
}

# Background parquet paths for named shorthand options
_NAMED_BACKGROUNDS = {
    "offbeam": Path("/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/beamoff/background_optics_cluster_features.parquet"),
    "michel":  Path("/Users/dajana/Documents/AmBe/michel-electron/michel_background_features.parquet"),
}

COLORS = {"signal": "#0077BB", "background": "#BBBBBB", "prompt": "#EE7733"}
_NAME  = {"rf": "Random Forest", "gbt": "GBT", "xgb": "XGBoost", "nn": "Neural Network"}
_LS    = {"rf": "-",             "gbt": "--",   "xgb": "-.",       "nn": ":"}

def _run_label(cfg: dict, run_name: str) -> str:
    """Presentation-friendly title from cfg['display_label'], falling back to run_name."""
    label = cfg.get("display_label") if isinstance(cfg, dict) else None
    return label if label else run_name.upper()


# ---------------------------------------------------------------------------
# Config + path helpers
# ---------------------------------------------------------------------------

def load_paths(config_path: str) -> tuple[str, Path, Path, Path, str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_name    = cfg["run_name"]
    # Remembered so --merge-run can resolve sibling runs under the same output_root
    # without each needing its own config file.
    load_paths_for_run._default_root = cfg["output_root"]  # type: ignore[attr-defined]
    root        = Path(cfg["output_root"]) / run_name
    parquet_dir = root / "parquet"
    plots_dir   = root / "plots"
    csv_dir     = root / "csv"
    for d in (plots_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)
    display_label = _run_label(cfg, run_name)
    return run_name, parquet_dir, plots_dir, csv_dir, display_label


def load_paths_for_run(run_name: str, output_root: Optional[str] = None) -> Path:
    """Return the parquet dir for a run named directly (no config file needed).

    Used by --merge-run.  Resolution order mirrors ambe.paths: explicit argument,
    then $AMBE_OUT, then the output_root the primary config already resolved to
    (set by main() so a merge run does not need its own config on disk).
    """
    root = (output_root or os.environ.get("AMBE_OUT")
            or getattr(load_paths_for_run, "_default_root", None))
    if not root:
        raise RuntimeError(
            f"cannot resolve output_root for --merge-run {run_name!r}; "
            "set $AMBE_OUT or pass a primary --config that defines output_root")
    return Path(root) / run_name / "parquet"


def _report_per_source(sub: pd.DataFrame, y: np.ndarray, when: str) -> None:
    """Signal/background counts broken down by source run and interaction origin.

    Only prints for merged trainings — this is the table that shows what the world
    sample actually contributed, and whether the 1:1 cap threw most of it away.
    """
    if "_source_run" not in sub.columns or sub["_source_run"].nunique() < 2:
        return
    if len(sub) != len(y):
        return
    print(f"[mva] per-source breakdown ({when}):")
    for rn, grp in sub.groupby("_source_run", sort=True):
        yy = y[grp.index.to_numpy()]
        line = (f"[mva]   {rn}: signal={int(yy.sum())}  "
                f"background={int((yy == 0).sum())}")
        if "origin_in_tank" in grp.columns:
            n_in = int((grp["origin_in_tank"] == 1).sum())
            n_out = int((grp["origin_in_tank"] == 0).sum())
            line += f"  (origin in-tank={n_in}, out-of-tank={n_out})"
        print(line)


def _apply_cc_filter(df: pd.DataFrame, cc_only: bool, run_name: str) -> pd.DataFrame:
    """Apply the CC selection to one run's clusters, respecting interaction origin.

    Out-of-tank interactions are harvested UNCONDITIONALLY: they are pure
    background, and the CC-inclusive cuts exist to select a muon-neutrino signal
    sample.  Gating the background on them would discard most of the contamination
    being modelled and would make the background population a function of the
    signal selection.  In-tank events still have to pass cc_pass.

    On a tank-only run (no origin_in_tank column) this reduces exactly to the
    previous behaviour: df[df.cc_pass == 1].
    """
    if not cc_only:
        return df.reset_index(drop=True)
    if "cc_pass" not in df.columns:
        print(f"[mva] WARN: --cc-only requested but no 'cc_pass' column in {run_name}; "
              "training on all clusters.")
        return df.reset_index(drop=True)

    n_before = len(df)
    keep = df["cc_pass"] == 1
    if "origin_in_tank" in df.columns:
        out_of_tank = df["origin_in_tank"] == 0
        keep = keep | out_of_tank
        n_out = int(out_of_tank.sum())
        n_out_failing_cc = int((out_of_tank & (df["cc_pass"] != 1)).sum())
        print(f"[mva] CC filter [{run_name}]: {int(keep.sum())}/{n_before} kept "
              f"(cc_pass==1: {int((df['cc_pass'] == 1).sum())}; "
              f"out-of-tank kept unconditionally: {n_out}, of which "
              f"{n_out_failing_cc} would have failed the CC selection)")
    else:
        print(f"[mva] CC filter [{run_name}] (cc_pass==1): "
              f"{int(keep.sum())}/{n_before} clusters kept")
    df = df[keep].reset_index(drop=True)
    if df.empty:
        sys.exit(f"[mva] No clusters survive the CC filter for {run_name} — check the "
                 "CC selection or pass --all-events.")
    return df


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame,
                 method: str = "optics",
                 bkg_mode: str = "all",
                 keep_prompt_bkg: bool = False,
                 fv_signal_only: bool = True) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Filter to one clustering method, build feature matrix and binary labels.

    bkg_mode:
      "all"       — signal vs all non-neutron non-prompt clusters (default)
      "darknoise" — signal vs clusters where dark noise (class 0) is the dominant component;
                    excludes class-5 (non-physics) dominated clusters from the background sample

    keep_prompt_bkg:
      If True, do NOT drop is_prompt_cluster==1 clusters from the background. The
      prompt-cluster exclusion was inherited from AmBe calibration (where the prompt
      gamma is a signal-side concern). In the CC-neutrino context those early-timing
      non-neutron-physics clusters ARE legitimate background, and excluding them
      shrinks the MC background pool ~3x (e.g. optics 5300 -> 1773), starving the
      1:1-capped training set. Set True to use the full non-neutron-dominated pool.

    Returns X, y, feature_names, filtered_sub_df.
    """
    sub = df[df["method"] == method].copy().reset_index(drop=True)

    # Signal = neutron-DOMINATED clusters (dominant_class in {1,2,3,4}). This is
    # consistent with the dominant_class-based background below, which is critical:
    # using is_truth_neutron==1 (a stricter trackID-purity match) for signal while
    # putting is_truth_neutron==0 into background dumps neutron-dominated clusters
    # that merely failed the trackID match INTO the background, contaminating it
    # (~70% of the old "all" background was actually neutron-dominated) and
    # collapsing the AUC to ~0.5. Composition-based labels for BOTH sides fixes it.
    NEUTRON_CLASSES = [1, 2, 3, 4]
    has_dc = "dominant_class" in sub.columns
    if has_dc:
        is_sig = sub["dominant_class"].isin(NEUTRON_CLASSES)
    else:
        # Fallback for parquets without dominant_class (older features runs)
        is_sig = sub["is_truth_neutron"] == 1

    # --- Interaction-origin term ------------------------------------------------
    # A cluster is signal only if a neutrino interacted INSIDE the tank. Neutron
    # light from an interaction in the dirt/concrete/MRD steel is background no
    # matter how clean the capture looks, because no in-tank neutrino produced it.
    # This is what makes the world-volume sample usable as extra background.
    #
    # origin_in_tank: 1 = inside, 0 = outside, -1 = unknown (event absent from the
    # CC table). Only an explicit 0 demotes a cluster; -1 and a missing column both
    # leave the label untouched, so the tank-only runs trained before this existed
    # reproduce their AUCs exactly.
    # Signal phase-space homogeneity. The tank sample's signal is FV-selected; world
    # in-tank events are not, because the world streams drop the FV cut (it reads
    # trueVtx*, which is not the interaction vertex there). Only 29.3% of world
    # in-tank cc_pass events are also inside the FV, so merging blind would make 71%
    # of the world signal near-wall events the tank signal never contains -- and
    # d_wall is a top-4 discriminator.
    #
    # These clusters are DROPPED, not moved to background: an in-tank neutron capture
    # is signal-like and we simply did not select it. Labeling it background would
    # teach the model that in-tank captures are background, which is the opposite of
    # the truth we are trying to encode.
    drop_outside_fv = pd.Series(False, index=sub.index)
    if (fv_signal_only and "origin_in_fv" in sub.columns
            and "origin_in_tank" in sub.columns):
        drop_outside_fv = (sub["origin_in_tank"] == 1) & (sub["origin_in_fv"] == 0)
        if int(drop_outside_fv.sum()):
            print(f"[mva] signal phase space: dropped {int(drop_outside_fv.sum())} "
                  "in-tank clusters outside the fiducial volume (kept the signal class "
                  "homogeneous with the tank sample; pass --no-fv-signal to keep them)")

    # Applied to the SIGNAL side here and to the background side below — these
    # clusters must leave BOTH classes, or they stay signal and the phase-space
    # inhomogeneity this flag exists to remove is still there.
    is_sig = is_sig & ~drop_outside_fv

    if "origin_in_tank" in sub.columns:
        outside = sub["origin_in_tank"] == 0
        n_demoted = int((is_sig & outside).sum())
        is_sig = is_sig & ~outside
        n_unknown = int((sub["origin_in_tank"] == -1).sum())
        print(f"[mva] interaction-origin term: {int(outside.sum())} clusters from "
              f"out-of-tank interactions -> background "
              f"({n_demoted} of them neutron-dominated, i.e. real captures "
              f"relabeled as background)")
        if n_unknown:
            print(f"[mva]   {n_unknown} clusters have origin_in_tank == -1 (unknown) "
                  "and keep their composition-based label")

    if bkg_mode in ("darknoise", "nonneutron", "pop2"):
        if not has_dc:
            raise ValueError(
                f"--bkg-mode {bkg_mode} requires 'dominant_class' column in the features parquet. "
                "Re-run: ambe mc features ..."
            )
        if bkg_mode == "darknoise":
            # Background = clusters dominated by class-0 dark noise (excludes class -5)
            is_bkg = sub["dominant_class"] == 0
        elif bkg_mode == "nonneutron":
            # Background = clusters dominated by class -5 (non-neutron physics)
            is_bkg = sub["dominant_class"] == -5
        else:
            # "pop2": class-5 clusters that are NOT the prompt gamma (is_prompt_cluster==0).
            if "is_prompt_cluster" not in sub.columns:
                raise ValueError(
                    "--bkg-mode pop2 requires 'is_prompt_cluster' column in the features parquet. "
                    "Re-run: ambe mc features ..."
                )
            is_bkg = (sub["dominant_class"] == -5) & (sub["is_prompt_cluster"] == 0)
    else:
        # "all": background = every NON-neutron-dominated cluster (class -5 physics +
        # class 0 dark noise), excluding the prompt cluster if flagged. Symmetric with
        # the neutron-dominated signal definition above — no neutron clusters leak in.
        if has_dc:
            is_bkg = ~sub["dominant_class"].isin(NEUTRON_CLASSES)
        else:
            is_bkg = sub["is_truth_neutron"] == 0
        # Out-of-tank neutron-dominated clusters were demoted out of the signal
        # above; they must be picked up HERE or `mask = is_sig | is_bkg` would drop
        # them from the training set altogether — silently discarding the exact
        # population the world sample was processed to provide.
        if "origin_in_tank" in sub.columns:
            is_bkg = is_bkg | (sub["origin_in_tank"] == 0)
        # In-tank/outside-FV clusters are excluded from BOTH classes (see above), so
        # `mask = is_sig | is_bkg` drops them rather than mislabeling them.
        is_bkg = is_bkg & ~drop_outside_fv
        # Prompt-cluster exclusion (default). With keep_prompt_bkg=True we KEEP the
        # early-timing non-neutron-physics clusters as background — ~3x more bkg
        # stats, which the 1:1 cap badly needs in the CC-neutrino context.
        if "is_prompt_cluster" in sub.columns and not keep_prompt_bkg:
            is_bkg = is_bkg & (sub["is_prompt_cluster"] == 0)

    mask = is_sig | is_bkg
    sub  = sub[mask].reset_index(drop=True)
    y    = is_sig[mask].astype(int).to_numpy()

    # Keep features present in the parquet with ≥50% non-NaN coverage.
    # Fitted-vertex features have ~65% convergence rate, so 80% would drop them;
    # residual NaNs are median-imputed below, making 50% safe.
    avail = [f for f in PHYSICS_FEATURES
             if f in sub.columns and sub[f].notna().mean() >= 0.5]
    X = sub[avail].to_numpy(dtype=float)

    # Median-impute residual NaNs (e.g. beta params for 1-hit clusters)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, j] = float(np.nanmedian(col))

    return X, y, avail, sub


def prepare_data_external_bkg(
        df_sig_full: pd.DataFrame,
        bkg_path: Path,
        method: str = "optics",
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix from MC signal + external real-data background parquet.

    Signal  : OPTICS true neutron clusters from df_sig_full (is_truth_neutron==1).
    Background : all rows from bkg_path (assumed pre-selected, label=0).

    Features used = intersection of PHYSICS_FEATURES present in both datasets,
    minus _EXTERNAL_BKG_EXCLUDE (truth columns, source geometry, bookkeeping IDs).
    Residual NaNs are median-imputed column-wise on the combined dataset.

    Returns X, y, feature_names, sig_df (signal rows only), bkg_df (background rows only).
    """
    sig = df_sig_full[(df_sig_full["method"] == method) &
                      (df_sig_full["is_truth_neutron"] == 1)].copy()
    bkg = pd.read_parquet(bkg_path)
    print(f"[external bkg] signal: {len(sig)} clusters  |  background: {len(bkg)} clusters "
          f"from {bkg_path.name}")

    # Features = PHYSICS_FEATURES that are (a) in both datasets, (b) not excluded,
    # (c) have ≥50% non-NaN coverage in both signal and background.
    candidate_feats = [
        f for f in PHYSICS_FEATURES
        if f not in _EXTERNAL_BKG_EXCLUDE
        and f in sig.columns
        and f in bkg.columns
        and sig[f].notna().mean() >= 0.5
        and bkg[f].notna().mean() >= 0.5
    ]
    print(f"[external bkg] {len(candidate_feats)} features: {candidate_feats}")

    sig_X = sig[candidate_feats].copy()
    bkg_X = bkg[candidate_feats].copy()
    sig_X["_label"] = 1
    bkg_X["_label"] = 0

    combined = pd.concat([sig_X, bkg_X], ignore_index=True)

    # Median-impute residual NaNs column-wise (fitted-vertex features ~36% NaN
    # in signal when Gauss-Newton did not converge — impute rather than drop rows)
    for f in candidate_feats:
        nan_mask = combined[f].isna()
        if nan_mask.any():
            med = float(combined[f].median())
            combined.loc[nan_mask, f] = med

    X = combined[candidate_feats].to_numpy(float)
    y = combined["_label"].to_numpy(int)
    print(f"[external bkg] final: {y.sum()} signal  +  {(y==0).sum()} background")
    return X, y, candidate_feats, sig, bkg


# ---------------------------------------------------------------------------
# Split utilities
# ---------------------------------------------------------------------------

def event_train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        sub: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Event-level stratified train/test split.

    Assigns whole events to train or test so that clusters from the same event
    never appear on both sides of the split — preventing information leakage
    when OPTICS finds multiple clusters per event.

    Strategy
    --------
    1. Collect unique eventIDs from the signal side (background is external
       real data with independent event numbering, so only signal events matter).
    2. For each event, determine its label: 1 if it contains ≥1 signal cluster,
       0 otherwise (pure-background events in internal MC mode).
    3. Stratified split of events -> assign all clusters of each event to the
       same partition.

    Falls back to a standard cluster-level stratified split if sub has no
    'eventID' column (external background mode where event structure is mixed).
    """
    # Fall back to cluster-level split when:
    # (a) no eventID column — external background mode, or
    # (b) eventID is present but only on the signal side (concat with external bkg
    #     leaves NaN eventIDs for background rows) — detect via NaN fraction.
    has_event_id = ("eventID" in sub.columns and
                    sub["eventID"].notna().mean() > 0.4)
    if not has_event_id:
        idx = np.arange(len(y))
        idx_tr, idx_te = train_test_split(idx, test_size=test_size,
                                          stratify=y, random_state=random_state)
        print(f"[split] cluster-level fallback (no shared eventID): "
              f"train={len(idx_tr)}  test={len(idx_te)}")
        return (X[idx_tr], X[idx_te],
                y[idx_tr], y[idx_te],
                idx_tr, idx_te)

    event_ids = sub["eventID"].fillna(-1).to_numpy(int)

    # With --merge-run, eventID restarts from 0 in every run. Keying the split on a
    # bare eventID would treat event 17 of the tank run and event 17 of the world run
    # as the SAME event, merging unrelated events and leaking clusters across the
    # train/test boundary. Key on (_source_run, eventID) instead.
    if "_source_run" in sub.columns and sub["_source_run"].nunique() > 1:
        codes, _ = pd.factorize(
            pd.Series(list(zip(sub["_source_run"].astype(str).tolist(),
                               event_ids.tolist()))))
        event_ids = np.where(event_ids < 0, -1, codes)
        print(f"[split] keyed on (_source_run, eventID): "
              f"{len(np.unique(event_ids[event_ids >= 0]))} distinct events across "
              f"{sub['_source_run'].nunique()} runs")

    unique_events = np.unique(event_ids[event_ids >= 0])

    # Label each event: 1 if it has any signal cluster, 0 if all background
    ev_label = np.array([
        int(y[event_ids == ev].max()) for ev in unique_events
    ])

    # Stratified split at event level
    ev_tr, ev_te = train_test_split(
        unique_events, test_size=test_size,
        stratify=ev_label, random_state=random_state,
    )
    ev_tr_set = set(ev_tr.tolist())
    ev_te_set = set(ev_te.tolist())

    idx_tr = np.where(np.isin(event_ids, list(ev_tr_set)))[0]
    idx_te = np.where(np.isin(event_ids, list(ev_te_set)))[0]

    # Rows with event_id == -1 are external background (no shared event structure).
    # Split them independently and append to both partitions so they aren't dropped.
    ext_idx = np.where(event_ids == -1)[0]
    if len(ext_idx) > 0:
        ext_tr, ext_te = train_test_split(ext_idx, test_size=test_size,
                                          random_state=random_state)
        idx_tr = np.concatenate([idx_tr, ext_tr])
        idx_te = np.concatenate([idx_te, ext_te])

    n_ev_tr = len(ev_tr); n_ev_te = len(ev_te)
    print(f"[split] event-level: {n_ev_tr} train events ({len(idx_tr)} clusters)  "
          f"| {n_ev_te} test events ({len(idx_te)} clusters)  "
          f"| test sig={y[idx_te].sum()}  test bkg={(y[idx_te]==0).sum()}")

    return (X[idx_tr], X[idx_te],
            y[idx_tr], y[idx_te],
            idx_tr, idx_te)


def cap_background(X: np.ndarray, y: np.ndarray,
                   max_ratio: float = 3.0,
                   random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Cap the background sample so it is at most max_ratio × n_signal.

    When signal > background (ratio inverted), caps signal to max_ratio × n_background
    to avoid the model simply learning "there are more signal examples".

    """
    rng   = np.random.default_rng(random_state)
    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())

    sig_idx = np.where(y == 1)[0]
    bkg_idx = np.where(y == 0)[0]

    cap_sig = int(max_ratio * n_bkg)   # max signal given background size
    cap_bkg = int(max_ratio * n_sig)   # max background given signal size

    if n_bkg > cap_bkg:
        # More background than allowed — subsample background
        keep = rng.choice(bkg_idx, size=cap_bkg, replace=False)
        keep = np.sort(np.concatenate([sig_idx, keep]))
        print(f"[cap] background capped: {n_bkg} -> {cap_bkg}  "
              f"(ratio was 1:{n_bkg/n_sig:.1f}, now 1:{cap_bkg/n_sig:.1f})")
    elif n_sig > cap_sig:
        # More signal than allowed — subsample signal
        keep = rng.choice(sig_idx, size=cap_sig, replace=False)
        keep = np.sort(np.concatenate([keep, bkg_idx]))
        print(f"[cap] signal capped: {n_sig} -> {cap_sig}  "
              f"(ratio was {n_sig/n_bkg:.1f}:1, now {cap_sig/n_bkg:.1f}:1)")
    else:
        print(f"[cap] ratio {n_sig}:{n_bkg} within max_ratio={max_ratio} — no capping")
        keep = np.arange(len(y))
        return X, y, keep

    return X[keep], y[keep], keep


def cv_roc_trees(
        X_tr: np.ndarray, y_tr: np.ndarray,
        n_splits: int = 5,
) -> tuple[dict, dict]:
    """
    Stratified k-fold cross-validation on the training set for tree models.

    Fits fresh RF, GBT and XGBoost on each fold's inner-train
    partition and evaluates on the inner-test partition.  Imputation medians and
    sample weights are computed inside each fold (no leakage).

    Returns
    -------
    cv_aucs  : dict  key -> list of per-fold AUC scores
    cv_curves: dict  key -> list of (fpr, tpr) per fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_aucs   = {"rf": [], "gbt": []}
    cv_curves = {"rf": [], "gbt": []}
    if _HAS_XGB:
        cv_aucs["xgb"]   = []
        cv_curves["xgb"] = []

    for fold, (inner_tr, inner_te) in enumerate(skf.split(X_tr, y_tr)):
        X_f_tr, X_f_te = X_tr[inner_tr], X_tr[inner_te]
        y_f_tr, y_f_te = y_tr[inner_tr], y_tr[inner_te]

        # Re-impute medians on fold training data only (no leakage across folds)
        medians = np.nanmedian(X_f_tr, axis=0)
        for j in range(X_f_tr.shape[1]):
            X_f_tr[:, j] = np.where(np.isnan(X_f_tr[:, j]), medians[j], X_f_tr[:, j])
            X_f_te[:, j] = np.where(np.isnan(X_f_te[:, j]), medians[j], X_f_te[:, j])

        fold_models = train_trees(X_f_tr, y_f_tr)
        for key, model in fold_models.items():
            sc = model.predict_proba(X_f_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_f_te, sc)
            fold_auc = float(auc(fpr, tpr))
            cv_aucs[key].append(fold_auc)
            cv_curves[key].append((fpr, tpr))

        fold_aucs_str = "  ".join(
            f"{k}={cv_aucs[k][-1]:.3f}" for k in cv_aucs)
        print(f"  [cv fold {fold+1}/{n_splits}]  {fold_aucs_str}")

    for key in cv_aucs:
        aucs_arr = cv_aucs[key]
        print(f"  [cv] {_NAME[key]:20s}  AUC = {np.mean(aucs_arr):.4f} ± {np.std(aucs_arr):.4f}")

    return cv_aucs, cv_curves


# ---------------------------------------------------------------------------
# Tree model training
# ---------------------------------------------------------------------------

def train_trees(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """
    Fit RF, GBT, and XGBoost.
    Returns dict keyed by 'rf', 'gbt', 'xgb'.
    """
    sw = compute_sample_weight("balanced", y_tr)

    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr, sample_weight=sw)

    gbt = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    )
    gbt.fit(X_tr, y_tr, sample_weight=sw)

    models = {"rf": rf, "gbt": gbt}

    if _HAS_XGB:
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        xgb = _XGBClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            # scale_pos_weight handles imbalance natively in XGBoost
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_tr, y_tr)
        models["xgb"] = xgb

    return models


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

NN_HIDDEN_UNITS  = 128    
NN_HIDDEN_LAYERS = 3      
NN_DROPOUT       = 0.3
NN_LR            = 1e-3   # initial Adam lr; ReduceLROnPlateau will lower it
NN_EPOCHS        = 100    
NN_BATCH_SIZE    = 256
NN_VERBOSE       = 1

def _build_nn(n_features: int):
    """
    input -> [Dense(128) -> BatchNorm -> ReLU -> Dropout(0.3)] × 3 -> Dense(1, sigmoid)

    BatchNorm before activation stabilises training across physics features with very
    different scales (ns timing, metres, PE counts, dimensionless β-parameters).
    """
    inp = tf.keras.Input(shape=(n_features,))
    x = inp
    for _ in range(NN_HIDDEN_LAYERS):
        x = tf.keras.layers.Dense(NN_HIDDEN_UNITS)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(NN_DROPOUT)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=NN_LR),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_nn(X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Scale features with StandardScaler (fit on train only — no leakage),
    then train the feedforward NN with early stopping.

    Returns (model, scaler, history) or (None, None, None) if tensorflow unavailable.
    """
    if not _HAS_KERAS:
        return None, None, None

    # Seed ALL of TF/Keras RNG (weight init, dropout masks, validation-split
    # shuffle) so the NN is bit-reproducible run-to-run — matching the
    # random_state=42 used for the data split and the tree models. Without this
    # the NN weight init is random each run, so retraining gives a slightly
    # different network (the 30.3<->33.0 us OPTICS+NN drift). Seeding fixes that.
    tf.keras.utils.set_random_seed(42)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr)
    sw = compute_sample_weight("balanced", y_tr)

    model = _build_nn(X_tr.shape[1])
    history = model.fit(
        X_sc, y_tr,
        sample_weight=sw,
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        validation_split=0.15,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=15,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5,
                patience=7, min_lr=1e-5, verbose=0,
            ),
        ],
        verbose=NN_VERBOSE,
    )
    return model, scaler, history


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _cv_roc_page(pdf, cv_aucs: dict, cv_curves: dict, run_name: str):
    """
    One panel per model showing all k fold ROC curves + mean ± std AUC annotation.
    Gives a visual sense of variance across folds — tight bands = stable model.
    """
    keys = list(cv_aucs.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5), squeeze=False)
    nfold = len(list(cv_curves.values())[0])
    fig.suptitle(f"Cross-Validation ROC  ({nfold}-fold)", fontsize=12)

    for ax, key in zip(axes[0], keys):
        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(cv_curves[key])))
        for i, (fpr, tpr) in enumerate(cv_curves[key]):
            ax.plot(fpr, tpr, color=colors[i], lw=1.2, alpha=0.8,
                    label=f"Fold {i+1}  (AUC = {cv_aucs[key][i]:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        mean_auc = float(np.mean(cv_aucs[key]))
        std_auc  = float(np.std(cv_aucs[key]))
        ax.set_title(f"{_NAME[key]}\nAUC = {mean_auc:.3f} ± {std_auc:.3f}", fontsize=11)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _roc_page(pdf, y_te: np.ndarray, model_scores: list, run_name: str, method: str):
    """model_scores: list of (display_name, test_scores, linestyle)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Neutron Classifier — ROC Curves (test set)", fontsize=12)

    ax = axes[0]
    for name, sc, ls in model_scores:
        fpr, tpr, _ = roc_curve(y_te, sc)
        auc = roc_auc_score(y_te, sc)
        ax.plot(fpr, tpr, lw=2, ls=ls, label=f"{name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Background Efficiency (False Positive Rate)", fontsize=10)
    ax.set_ylabel("Neutron Efficiency (True Positive Rate)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    ax = axes[1]
    for name, sc, ls in model_scores:
        fpr, tpr, _ = roc_curve(y_te, sc)
        ax.plot(tpr, 1 - fpr, lw=2, ls=ls, label=name)
    ax.set_xlabel("Neutron Efficiency", fontsize=10)
    ax.set_ylabel("Background Rejection", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _score_dist_page(pdf, y_te: np.ndarray, model_scores: list, run_name: str):
    """One panel per model in a 2-column grid."""
    n = len(model_scores)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
    fig.suptitle("Classifier Score Distributions (test set)", fontsize=12)
    bins = np.linspace(0, 1, 40)
    for idx, (name, sc, _) in enumerate(model_scores):
        ax = axes[idx // ncols][idx % ncols]
        auc = roc_auc_score(y_te, sc)
        kw = dict(bins=bins, density=True, alpha=0.6, edgecolor="none")
        ax.hist(sc[y_te == 1], color=COLORS["signal"],
                label=f"Neutron  (n = {int((y_te==1).sum())})", **kw)
        ax.hist(sc[y_te == 0], color=COLORS["background"],
                label=f"Background  (n = {int((y_te==0).sum())})", **kw)
        ax.set_xlabel(f"{name} Score", fontsize=10)
        ax.set_ylabel("Normalised Counts", fontsize=10)
        ax.set_title(f"{name}  —  AUC = {auc:.3f}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for idx in range(len(model_scores), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _importance_page(pdf, tree_models: dict, features: list, run_name: str):
    """Bar chart importances for all tree models that have feature_importances_."""
    has_imp = [(k, m) for k, m in tree_models.items() if hasattr(m, "feature_importances_")]
    if not has_imp:
        return
    fig, axes = plt.subplots(1, len(has_imp), figsize=(7 * len(has_imp), 5))
    if len(has_imp) == 1:
        axes = [axes]
    fig.suptitle("Feature Importances", fontsize=12)
    for ax, (key, model) in zip(axes, has_imp):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        ax.bar(range(len(imp)), imp[idx], color="steelblue", alpha=0.85)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([features[i] for i in idx],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Relative Importance", fontsize=10)
        ax.set_title(_NAME[key], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _top_features_page(pdf, X_te, y_te, rf, features, run_name):
    top6 = np.argsort(rf.feature_importances_)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Top 6 Discriminating Features (Random Forest)", fontsize=12)
    for ax, fi in zip(axes.flat, top6):
        feat = features[fi]
        sig_v = X_te[y_te == 1, fi]
        bkg_v = X_te[y_te == 0, fi]
        all_v = np.concatenate([sig_v, bkg_v])
        lo, hi = float(np.nanpercentile(all_v, 1)), float(np.nanpercentile(all_v, 99))
        bins = np.linspace(lo, hi, 35)
        kw = dict(bins=bins, density=True, alpha=0.6, edgecolor="none")
        ax.hist(sig_v, color=COLORS["signal"],     label="Neutron", **kw)
        ax.hist(bkg_v, color=COLORS["background"], label="Background", **kw)
        imp = rf.feature_importances_[fi]
        ax.set_title(f"{feat}  (importance = {imp:.3f})", fontsize=10)
        ax.set_xlabel(feat, fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _nn_history_page(pdf, history, run_name: str):
    has_auc = "val_auc" in history.history
    fig, axes = plt.subplots(1, 2 if has_auc else 1, figsize=(14 if has_auc else 8, 5), squeeze=False)
    fig.suptitle("Neural Network Training History", fontsize=12)

    ax = axes[0, 0]
    ax.plot(history.history["loss"],     lw=2, label="Training")
    ax.plot(history.history["val_loss"], lw=2, ls="--", label="Validation")
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    ax.axvline(best_epoch, color="gray", ls=":", lw=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Binary Cross-Entropy", fontsize=10)
    ax.set_title("Loss", fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    if has_auc:
        ax = axes[0, 1]
        ax.plot(history.history["auc"],     lw=2, label="Training")
        ax.plot(history.history["val_auc"], lw=2, ls="--", label="Validation")
        best_auc_epoch = int(np.argmax(history.history["val_auc"])) + 1
        ax.axvline(best_auc_epoch, color="gray", ls=":", lw=1, label=f"Best epoch ({best_auc_epoch})")
        ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("AUC", fontsize=10)
        ax.set_title("AUC", fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _event_level_page(pdf, sub, run_name, score_col="gbt_score"):
    """
    Event-level efficiency: for each event, take the highest cluster score.
    Show efficiency (fraction of truth-neutron events with max_score > threshold)
    vs threshold, and overlay background rate (fraction of no-truth events passing).
    """
    has_truth = sub.groupby("eventID")["is_truth_neutron"].max().rename("has_truth")
    max_score = sub.groupby("eventID")[score_col].max().rename("max_score")
    ev = pd.concat([has_truth, max_score], axis=1).dropna()
    if len(ev) == 0 or ev["has_truth"].nunique() < 2:
        return

    thresholds = np.linspace(0, 1, 100)
    sig_eff, bkg_eff = [], []
    for t in thresholds:
        sig = ev[ev["has_truth"] == 1]
        bkg = ev[ev["has_truth"] == 0]
        sig_eff.append(float((sig["max_score"] >= t).mean()) if len(sig) else np.nan)
        bkg_eff.append(float((bkg["max_score"] >= t).mean()) if len(bkg) else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Event-Level Neutron Selection Efficiency", fontsize=12)

    ax = axes[0]
    ax.plot(thresholds, sig_eff, lw=2, color=COLORS["signal"],
            label="Neutron events")
    ax.plot(thresholds, bkg_eff, lw=2, color=COLORS["background"],
            label="Background events")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="Score = 0.5")
    ax.set_xlabel("Score Threshold (per-event maximum)", fontsize=10)
    ax.set_ylabel("Fraction of Events Selected", fontsize=10)
    ax.set_title("Efficiency vs. Score Threshold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    ax = axes[1]
    sig_arr = np.array(sig_eff)
    bkg_arr = np.array(bkg_eff)
    valid   = ~(np.isnan(sig_arr) | np.isnan(bkg_arr))
    ax.plot(sig_arr[valid], 1 - bkg_arr[valid], lw=2, color="darkorchid")
    ax.set_xlabel("Neutron Efficiency", fontsize=10)
    ax.set_ylabel("Background Rejection", fontsize=10)
    ax.set_title("Event-Level ROC", fontsize=11)
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Application mode — score new data with a frozen model (no training)
# ---------------------------------------------------------------------------

def score_data(model_path: Path, data_path: Path) -> Path:
    """
    Load a frozen MVA artifact (from --save-model) and score a data features
    parquet. The data has NO truth labels — this is pure model application.

    The feature matrix is built EXACTLY as in training: select the stored
    feature columns in the stored order, NaN-impute with the stored (MC training)
    medians. Any feature missing from the data parquet is filled entirely with
    its training median (and reported), so the column vector the model expects is
    always present.

    Writes <data_stem>__scored.parquet (all original columns + rf_score/gbt_score)
    and returns its path.
    """
    art = joblib.load(model_path)
    features = art["features"]
    medians  = np.asarray(art["medians"], dtype=float)
    models   = art["models"]
    meta     = art.get("meta", {})
    print(f"[score] frozen model: {model_path.name}")
    print(f"[score]   trained on run '{meta.get('run_name','?')}', "
          f"bkg_mode={meta.get('bkg_mode','?')}, "
          f"n_sig={meta.get('n_sig','?')} n_bkg={meta.get('n_bkg','?')}")
    print(f"[score]   {len(features)} features, models={sorted(models.keys())}")

    df = pd.read_parquet(data_path)
    print(f"[score] data parquet: {data_path.name}  ({len(df)} clusters)")

    # Empty input (e.g. a method split with no rows — real data has no
    # ClusterFinder clusters). sklearn predict_proba rejects a 0-row array, so
    # write through the (empty) frame with empty score columns and return rather
    # than crashing the wrapper that scores both OPTICS and CF splits.
    if len(df) == 0:
        out = df.copy()
        for key in models.keys():
            out[f"{key}_score"] = pd.Series(dtype=float)
        score_path = data_path.with_name(data_path.stem + "__scored.parquet")
        out.to_parquet(score_path, index=False)
        print(f"[score] 0 clusters — wrote empty scored parquet -> {score_path}")
        return score_path

    # Build X in the trained feature order; fill missing features with training median.
    X = np.empty((len(df), len(features)), dtype=float)
    missing = []
    for j, f in enumerate(features):
        if f in df.columns:
            col = df[f].to_numpy(dtype=float)
        else:
            col = np.full(len(df), np.nan)
            missing.append(f)
        nanm = np.isnan(col)
        if nanm.any():
            col = np.where(nanm, medians[j], col)
        X[:, j] = col
    if missing:
        print(f"[score] WARN: {len(missing)} feature(s) absent in data parquet, "
              f"filled with training median: {missing}")

    out = df.copy()
    for key, model in models.items():
        out[f"{key}_score"] = model.predict_proba(X)[:, 1]
        s = out[f"{key}_score"]
        print(f"[score]   {key}_score: min={s.min():.3f} median={s.median():.3f} "
              f"max={s.max():.3f}  (frac>0.5: {(s>0.5).mean():.3f})")

    # Apply NN if a companion Keras file is present.
    nn_meta = art.get("nn")
    if nn_meta is not None:
        keras_file = model_path.with_name(nn_meta["keras_file"])
        if not keras_file.exists():
            print(f"[score]   WARN: NN companion file missing ({keras_file.name}) — skipping NN.")
        elif not _HAS_KERAS:
            print(f"[score]   WARN: tensorflow not installed — skipping NN. pip install tensorflow")
        else:
            try:
                nn_model = tf.keras.models.load_model(keras_file)
                X_sc = nn_meta["scaler"].transform(X)
                out["nn_score"] = nn_model.predict(X_sc, verbose=0).ravel()
                s = out["nn_score"]
                print(f"[score]   nn_score: min={s.min():.3f} median={s.median():.3f} "
                      f"max={s.max():.3f}  (frac>0.5: {(s>0.5).mean():.3f})")
            except Exception as exc:
                # Most common cause: the .keras file was saved by a newer Keras
                # than the scoring env (e.g. 3.14.1 writes 'quantization_config'
                # into Dense configs that older Keras rejects). Do NOT take down
                # the run — the tree scores above are already computed; just skip
                # nn_score so out.to_parquet() still persists them.
                print(f"[score]   WARN: NN load/predict failed — skipping nn_score. "
                      f"({type(exc).__name__}: {exc})")

    score_path = data_path.with_name(data_path.stem + "__scored.parquet")
    out.to_parquet(score_path, index=False)
    print(f"[score] wrote -> {score_path}")
    return score_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(prog="mva_analysis")
    p.add_argument("--config",    default=None,
                   help="Pipeline YAML config (required for training; for "
                        "--score-data only needed when --model is omitted)")
    p.add_argument("--method",    default="optics",
                   choices=["optics", "clusterfinder"])
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of clusters held out for testing (default 0.2)")
    p.add_argument("--no-nn", action="store_true",
                   help="Skip neural network training (faster — tree models only)")
    p.add_argument("--no-cv", action="store_true",
                   help="Skip 5-fold cross-validation (faster — single train/test split only)")
    p.add_argument("--bkg-mode", default="all",
                   choices=["all", "darknoise", "nonneutron", "pop2"],
                   help="Internal background sample (MC only): 'all' = all non-neutron non-prompt "
                        "clusters (default); 'darknoise' = class-0 dominated; "
                        "'nonneutron' = class-5 dominated; "
                        "'pop2' = Population 2 near-capture clusters only. "
                        "Ignored when --external-background is set.")
    p.add_argument("--keep-prompt-bkg", action="store_true",
                   help="Do NOT exclude is_prompt_cluster==1 clusters from the background "
                        "(only affects --bkg-mode all). In the CC-neutrino context the prompt "
                        "non-neutron-physics clusters are legitimate background; keeping them "
                        "~3x's the bkg pool so the 1:1 cap retains far more signal. Tagged "
                        "'__keepprompt' in output filenames + model meta.")
    p.add_argument("--external-background", default=None, metavar="PATH_OR_NAME",
                   help="Use real-data background instead of MC internal background. "
                        "Pass a parquet file path, or one of the named shorthands: "
                        + ", ".join(_NAMED_BACKGROUNDS.keys()) + ". "
                        "Output files are tagged with --background-label.")
    p.add_argument("--background-label", default=None, metavar="LABEL",
                   help="Short label for the background source, used in output filenames "
                        "(e.g. 'offbeam', 'michel'). Auto-derived from named shorthands if omitted.")
    p.add_argument("--max-ratio", type=float, default=3.0, metavar="R",
                   help="Maximum signal:background (or background:signal) ratio after capping. "
                        "Default 3.0. Use 1.0 for a physically balanced 1:1 dataset.")
    p.add_argument("--merge-run", action="append", default=None, metavar="RUN_NAME",
                   help="Merge another run's cluster_features.parquet into the training "
                        "set (repeatable). Resolved as <output_root>/<RUN_NAME>/parquet/. "
                        "Labels come from the same rule as the primary run, so the full "
                        "feature set is preserved — unlike --external-background, which "
                        "forces every external row to y=0 and drops vtx_*/d_source_*. "
                        "Intended for the world-volume background sample: "
                        "--merge-run cc_neutrino_v3world_truthtag")
    p.add_argument("--no-fv-signal", dest="fv_signal", action="store_false", default=True,
                   help="Keep in-tank clusters whose interaction vertex is OUTSIDE the "
                        "fiducial volume in the signal/background pool. Default is to "
                        "drop them, so the signal class stays in the same phase space as "
                        "the FV-selected tank sample. No effect on runs without an "
                        "origin_in_fv column (i.e. all tank-only runs).")
    p.add_argument("--cc-only", dest="cc_only", action="store_true", default=True,
                   help="Train only on CC-passing clusters (cc_pass==1), if the column exists. "
                        "Clusters from out-of-tank interactions (origin_in_tank==0) are kept "
                        "regardless, since they are background by construction. Default: on.")
    p.add_argument("--all-events", dest="cc_only", action="store_false",
                   help="Disable the CC filter — train on every cluster regardless of cc_pass.")
    p.add_argument("--save-model", nargs="?", const="__DEFAULT__", default=None,
                   metavar="PATH",
                   help="Freeze the fitted tree models to a joblib .pkl for later "
                        "application to real data (e.g. AmBe). With no path, writes "
                        "<parquet_dir>/<run_name>__mva_frozen{mode_tag}.pkl. The artifact "
                        "stores the fitted RF/GBT(/XGB), the exact feature list, and the "
                        "NaN-imputation medians, so the data-scoring step reproduces the inputs.")
    p.add_argument("--score-data", default=None, metavar="DATA_PARQUET",
                   help="APPLICATION mode: do NOT train. Load a frozen model (--model) and "
                        "score this data features parquet (e.g. ambe_4499__data_features.parquet). "
                        "Writes <stem>__scored.parquet with rf_score/gbt_score columns added. "
                        "No truth needed — pure model application to real data.")
    p.add_argument("--model", default=None, metavar="FROZEN_PKL",
                   help="Frozen model .pkl (from --save-model) to use with --score-data. "
                        "Defaults to <parquet_dir>/<run_name>__mva_frozen.pkl.")
    args = p.parse_args()

    # ── Application mode: score data with a frozen model, then exit ──────────
    if args.score_data is not None:
        # --config is only needed here to DERIVE the default model path when
        # --model is omitted. The streamlines always pass --model explicitly, so
        # don't force --config (and don't call load_paths(None)).
        if args.model:
            model_path = Path(args.model)
        else:
            if not args.config:
                sys.exit("[score] --score-data needs either --model <pkl> or "
                         "--config <yaml> (to find the default frozen model).")
            run_name, parquet_dir, _plots, _csv, _label = load_paths(args.config)
            model_path = parquet_dir / f"{run_name}__mva_frozen.pkl"
        if not model_path.exists():
            sys.exit(f"[score] Frozen model not found: {model_path}\n"
                     f"        Create it first with: python mva_analysis.py "
                     f"--config <yaml> --save-model")
        data_path = Path(args.score_data)
        if not data_path.exists():
            sys.exit(f"[score] Data features parquet not found: {data_path}")
        score_data(model_path, data_path)
        return

    # ── Training mode below: --config is required here. ──────────────────────
    if not args.config:
        sys.exit("[mva] --config is required for training mode "
                 "(only --score-data with --model can omit it).")

    if not _HAS_XGB:
        print("[mva] WARNING: xgboost not installed — skipping XGBoost.  pip install xgboost")
    if not _HAS_KERAS:
        print("[mva] WARNING: tensorflow not installed — skipping NN.  pip install tensorflow")
    if _HAS_KERAS:
        # TF auto-selects one GPU if CUDA is visible; no MirroredStrategy -> single-device only.
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[mva] TensorFlow: {len(gpus)} GPU(s) visible"
              + ("  (no MirroredStrategy — using GPU:0 only)" if len(gpus) > 1 else ""))

    run_name, parquet_dir, plots_dir, csv_dir, display_label = load_paths(args.config)

    feat_path = parquet_dir / f"{run_name}__cluster_features.parquet"
    if not feat_path.exists():
        sys.exit(f"[mva] Features parquet not found: {feat_path}\n"
                 f"      Run: ambe mc features --config {args.config}")

    print(f"[mva] loading {feat_path}")
    df = pd.read_parquet(feat_path)
    df["_source_run"] = run_name
    df = _apply_cc_filter(df, args.cc_only, run_name)

    # ── extra runs merged in (e.g. the world-volume background sample) ─────
    # Each run is CC-filtered on its OWN cc_pass before the concat: cc_pass encodes
    # a different selection in every run, so filtering after the merge would apply
    # one run's selection semantics to another's rows.
    for extra_run in (args.merge_run or []):
        extra_paths = load_paths_for_run(extra_run)
        extra_path = extra_paths / f"{extra_run}__cluster_features.parquet"
        if not extra_path.exists():
            sys.exit(f"[mva] --merge-run {extra_run}: features parquet not found: "
                     f"{extra_path}\n      Run: ambe mc features --config "
                     f"configs/{extra_run}.yaml")
        print(f"[mva] merging {extra_path}")
        extra_df = pd.read_parquet(extra_path)
        extra_df["_source_run"] = extra_run
        extra_df = _apply_cc_filter(extra_df, args.cc_only, extra_run)
        df = pd.concat([df, extra_df], ignore_index=True)

    if args.merge_run:
        # Runs processed before origin_in_tank existed contribute NaN here. Normalise
        # to -1 (unknown) so the label rule sees one dtype and one sentinel: NaN
        # would silently compare False against every test, which happens to be the
        # behaviour we want but for the wrong reason and only by luck.
        for _oc in ("origin_in_tank", "origin_in_fv"):
            if _oc in df.columns:
                df[_oc] = df[_oc].fillna(-1).astype(int)
        print(f"[mva] merged frame: {len(df)} clusters from "
              f"{df['_source_run'].nunique()} run(s)")
        for rn, cnt in df["_source_run"].value_counts().items():
            print(f"[mva]   {rn}: {cnt}")

    # ── external real-data background mode ────────────────────────────────
    external_bkg = args.external_background
    if external_bkg is not None:
        # Resolve named shorthands -> Path
        if external_bkg in _NAMED_BACKGROUNDS:
            bkg_path = _NAMED_BACKGROUNDS[external_bkg]
            bkg_tag  = args.background_label or external_bkg
        else:
            bkg_path = Path(external_bkg)
            bkg_tag  = args.background_label or bkg_path.stem
        if not bkg_path.exists():
            sys.exit(f"[mva] External background parquet not found: {bkg_path}")

        print(f"[mva] external background mode: {bkg_path.name}  (tag={bkg_tag})")
        X, y, features, sig_df, bkg_df = prepare_data_external_bkg(
            df, bkg_path, method=args.method)

        # For external mode, sub is a combined frame with a _label column for bookkeeping
        sig_df = sig_df.copy(); sig_df["_label"] = 1
        bkg_df = bkg_df.copy(); bkg_df["_label"] = 0
        sub = pd.concat([sig_df, bkg_df], ignore_index=True)

        plot_run_name = f"{display_label} vs {bkg_tag}"
        # Output tag: <run_name>__mva__vs_<bkg_tag>  e.g. mc_lucho_full__mva__vs_offbeam
        mode_tag = f"__vs_{bkg_tag}"
        # Event-level page requires eventID; not meaningful with external background
        skip_event_level = True

    # ── internal MC background mode (default) ────────────────────────────
    else:
        bkg_mode = args.bkg_mode
        keep_prompt = bool(args.keep_prompt_bkg)
        X, y, features, sub = prepare_data(df, method=args.method, bkg_mode=bkg_mode,
                                           keep_prompt_bkg=keep_prompt,
                                           fv_signal_only=bool(args.fv_signal))
        excl_str = "incl. prompt" if keep_prompt else "excl. prompt"
        bkg_label_str = {"darknoise":  "dark noise only (class 0)",
                         "nonneutron": "class-5 spurious (prompt gamma + near-capture)",
                         "pop2":       "Population 2 only (near-capture, offset ~-1 ns)",
                         "all":        f"all non-neutron ({excl_str})"}.get(bkg_mode, bkg_mode)
        print(f"[mva] bkg_mode={bkg_mode}  ({bkg_label_str})")
        kp_tag = "__keepprompt" if (keep_prompt and bkg_mode == "all") else ""
        # Merged trainings get their own tag so they never overwrite the tank-only
        # baseline artifacts — the two must stay comparable side by side.
        merge_tag = "__merged" if args.merge_run else ""
        mode_tag = ("" if bkg_mode == "all" else f"__{bkg_mode}") + kp_tag + merge_tag
        plot_run_name = (f"{display_label} [{bkg_mode} bkg{', +prompt' if kp_tag else ''}]"
                         if (bkg_mode != "all" or kp_tag) else display_label)
        skip_event_level = False

    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())
    print(f"[mva] {X.shape[0]} clusters  |  "
          f"signal={n_sig}  background={n_bkg}  |  {len(features)} features")
    print(f"[mva] features: {features}")
    _report_per_source(sub, y, "before capping")

    if n_sig < 10 or n_bkg < 10:
        sys.exit("[mva] Too few samples to train — check features parquet.")

    # Cap imbalanced background before splitting
    # max_ratio=args.max_ratio keeps training balanced: at most args.max_ratio × background per signal
    # (or args.max_ratio × signal per background if the ratio is inverted, e.g. lucho vs michel).
    X, y, _cap_keep = cap_background(X, y, max_ratio=args.max_ratio)
    sub = sub.iloc[_cap_keep].reset_index(drop=True)
    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())
    print(f"[mva] after capping: signal={n_sig}  background={n_bkg}")
    _report_per_source(sub, y, "after capping")

    # Event-level train/test split
    # Assigns whole events to train or test so no event leaks across the boundary.
    # Falls back to cluster-level for external background (no shared event structure).
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = event_train_test_split(
        X, y, sub.iloc[:len(y)].reset_index(drop=True),
        test_size=args.test_size,
    )
    print(f"[mva] train={len(y_tr)}  test={len(y_te)}")

    # Stratified k-fold CV on training set 
    if not args.no_cv:
        print(f"\n[mva] 5-fold cross-validation on training set …")
        cv_aucs, cv_curves = cv_roc_trees(X_tr, y_tr, n_splits=5)
        print()
    else:
        print("[mva] CV skipped (--no-cv)")
        cv_aucs, cv_curves = {}, {}

    tree_label = "RF, GBT" + (", XGBoost" if _HAS_XGB else "")
    print(f"[mva] training final models on full train set ({tree_label}) …")
    tree_models = train_trees(X_tr, y_tr)

    nn_model, nn_scaler, nn_history = None, None, None
    if _HAS_KERAS and not args.no_nn:
        hidden = "-".join([str(NN_HIDDEN_UNITS)] * NN_HIDDEN_LAYERS)
        arch = f"{len(features)}-{hidden}-1" if hidden else f"{len(features)}-1"
        print(f"[mva] training NN  ({arch}, ReLU, dropout={NN_DROPOUT})"
              f"  [{len(features)} features after NaN filter]")
        nn_model, nn_scaler, nn_history = train_nn(X_tr, y_tr)
        n_epochs = len(nn_history.history["loss"])
        print(f"[mva] NN stopped at epoch {n_epochs} / {NN_EPOCHS}")
    elif args.no_nn:
        print("[mva] NN skipped (--no-nn)")

    # ── Freeze fitted models for application to real data ─────────────────
    # Trees go into a joblib .pkl alongside the exact feature order, the
    # per-feature imputation medians, and (if trained) the NN StandardScaler.
    # The Keras NN itself is saved as a companion .keras file next to the .pkl
    # — score_data() picks it up automatically when present.
    if args.save_model is not None:
        if args.save_model == "__DEFAULT__":
            model_out = parquet_dir / f"{run_name}__mva_frozen{mode_tag}.pkl"
        else:
            model_out = Path(args.save_model)
        train_medians = np.nanmedian(X_tr, axis=0)
        artifact = {
            "features": list(features),
            "medians": np.asarray(train_medians, dtype=float),
            "models": tree_models,
            "meta": {
                "run_name": run_name,
                "display_label": display_label,
                "method": args.method,
                "bkg_mode": args.bkg_mode if external_bkg is None else f"external:{bkg_tag}",
                "keep_prompt_bkg": bool(args.keep_prompt_bkg) if external_bkg is None else False,
                "cc_only": args.cc_only,
                "max_ratio": args.max_ratio,
                "n_sig": int(y.sum()),
                "n_bkg": int((y == 0).sum()),
                "n_train": int(len(y_tr)),
                "note": "Trees are scale-free (no StandardScaler). Impute NaNs with stored medians, "
                        "order columns by 'features', then model.predict_proba(X)[:,1]. "
                        "NN (if present) requires the companion .keras file + the stored scaler.",
            },
        }
        nn_keras_path = None
        if nn_model is not None and nn_scaler is not None:
            nn_keras_path = model_out.with_name(model_out.stem + "__nn.keras")
            nn_model.save(nn_keras_path)
            artifact["nn"] = {
                "scaler": nn_scaler,
                "keras_file": nn_keras_path.name,  # relative to .pkl directory
            }
        joblib.dump(artifact, model_out)
        nn_msg = f" + NN -> {nn_keras_path.name}" if nn_keras_path else ""
        print(f"[mva] froze fitted models -> {model_out}  "
              f"({len(features)} features, models={sorted(tree_models.keys())}{nn_msg})")

    model_scores  = []   # (display_name, test_scores, linestyle) — for plots
    all_score_cols = {}  # col_name -> scores on full X, for parquet

    for key, model in tree_models.items():
        sc_te  = model.predict_proba(X_te)[:, 1]
        sc_all = model.predict_proba(X)[:, 1]
        model_scores.append((_NAME[key], sc_te, _LS[key]))
        all_score_cols[f"{key}_score"] = sc_all

    if nn_model is not None:
        X_te_sc  = nn_scaler.transform(X_te)
        X_all_sc = nn_scaler.transform(X)
        sc_te  = nn_model.predict(X_te_sc,  verbose=0).ravel()
        sc_all = nn_model.predict(X_all_sc, verbose=0).ravel()
        model_scores.append((_NAME["nn"], sc_te, _LS["nn"]))
        all_score_cols["nn_score"] = sc_all

    for name, sc, _ in model_scores:
        print(f"[mva] AUC  {name:20s} = {roc_auc_score(y_te, sc):.3f}")

    sub = sub.copy()
    for col, vals in all_score_cols.items():
        sub[col] = vals
    sub["in_test"] = False
    sub.loc[idx_te, "in_test"] = True
    # Persist the training target. Without it, anything reading the score table has
    # to re-derive the label, and for a merged frame dominant_class alone gives the
    # WRONG answer (out-of-tank captures are neutron-dominated but are background).
    sub["mva_label"] = y

    # Output paths — mode_tag encodes the background choice so all four combinations
    # land in the same run directory without overwriting each other:
    pdf_path   = plots_dir   / f"{run_name}__mva{mode_tag}.pdf"
    score_path = parquet_dir / f"{run_name}__mva_scores{mode_tag}.parquet"
    csv_path   = csv_dir     / f"{run_name}__mva_summary{mode_tag}.csv"

    print(f"[mva] writing plots -> {pdf_path}")
    best_event_col = next(
        (c for c in ["xgb_score", "nn_score", "gbt_score"] if c in sub.columns),
        "gbt_score",
    )
    with PdfPages(pdf_path) as pdf:
        if cv_aucs:
            _cv_roc_page(pdf, cv_aucs, cv_curves, plot_run_name)
        _roc_page(pdf, y_te, model_scores, plot_run_name, args.method)
        _score_dist_page(pdf, y_te, model_scores, plot_run_name)
        _importance_page(pdf, tree_models, features, plot_run_name)
        _top_features_page(pdf, X_te, y_te, tree_models["rf"], features, plot_run_name)
        if nn_history is not None:
            _nn_history_page(pdf, nn_history, plot_run_name)
        if not skip_event_level:
            _event_level_page(pdf, sub, plot_run_name, score_col=best_event_col)

    sub.to_parquet(score_path, index=False)
    print(f"[mva] wrote scores -> {score_path}")

    imp_df = pd.DataFrame({"feature": features})
    for key, model in tree_models.items():
        if hasattr(model, "feature_importances_"):
            imp_df[f"{key}_importance"] = model.feature_importances_
    if "rf_importance" in imp_df.columns:
        imp_df = imp_df.sort_values("rf_importance", ascending=False).reset_index(drop=True)
        imp_df.insert(0, "rf_rank", range(1, len(features) + 1))

    imp_df.to_csv(csv_path, index=False)

    print(f"\n[mva] ===== SUMMARY =====")
    for name, sc, _ in model_scores:
        key = next((k for k, n in _NAME.items() if n == name), None)
        cv_str = (f"  CV={np.mean(cv_aucs[key]):.4f}±{np.std(cv_aucs[key]):.4f}"
                  if key and key in cv_aucs else "")
        print(f"  AUC  {name:20s} : test={roc_auc_score(y_te, sc):.4f}{cv_str}")
    if "rf_importance" in imp_df.columns:
        imp_cols = [c for c in imp_df.columns if c.endswith("_importance")]
        print(f"\n  Feature importances (RF rank):")
        for _, row in imp_df.iterrows():
            parts = [f"{row['feature']:22s}"]
            for c in imp_cols:
                tag = c.replace("_importance", "").upper()
                parts.append(f"{tag}={row[c]:.4f}")
            print(f"    {int(row.get('rf_rank', 0)):2d}. {'  '.join(parts)}")
    print(f"\n[mva] wrote summary -> {csv_path}")
    print(f"[mva] wrote plots   -> {pdf_path}")


if __name__ == "__main__":
    main()
