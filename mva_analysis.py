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
import sys
from pathlib import Path

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
    "offbeam": Path("/Users/dajana/Documents/AmBe/off-beam/off-beam_background_features.parquet"),
    "michel":  Path("/Users/dajana/Documents/AmBe/michel-electron/michel_background_features.parquet"),
}

COLORS = {"signal": "#0077BB", "background": "#BBBBBB", "prompt": "#EE7733"}
_NAME  = {"rf": "Random Forest", "gbt": "GBT", "xgb": "XGBoost", "nn": "Neural Network"}
_LS    = {"rf": "-",             "gbt": "--",   "xgb": "-.",       "nn": ":"}


# ---------------------------------------------------------------------------
# Config + path helpers
# ---------------------------------------------------------------------------

def load_paths(config_path: str) -> tuple[str, Path, Path, Path]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_name    = cfg["run_name"]
    root        = Path(cfg["output_root"]) / run_name
    parquet_dir = root / "parquet"
    plots_dir   = root / "plots"
    csv_dir     = root / "csv"
    for d in (plots_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)
    return run_name, parquet_dir, plots_dir, csv_dir


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame,
                 method: str = "optics",
                 bkg_mode: str = "all") -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Filter to one clustering method, build feature matrix and binary labels.

    bkg_mode:
      "all"       — signal vs all non-neutron non-prompt clusters (default)
      "darknoise" — signal vs clusters where dark noise (class 0) is the dominant component;
                    excludes class-5 (non-physics) dominated clusters from the background sample

    Returns X, y, feature_names, filtered_sub_df.
    """
    sub = df[df["method"] == method].copy().reset_index(drop=True)

    is_sig = sub["is_truth_neutron"] == 1

    if bkg_mode in ("darknoise", "nonneutron", "pop2"):
        if "dominant_class" not in sub.columns:
            raise ValueError(
                f"--bkg-mode {bkg_mode} requires 'dominant_class' column in the features parquet. "
                "Re-run: ambe mc features ..."
            )
        if bkg_mode == "darknoise":
            # Background = clusters dominated by class-0 dark noise (excludes class -5)
            is_bkg = (sub["is_truth_neutron"] == 0) & (sub["dominant_class"] == 0)
        elif bkg_mode == "nonneutron":
            # Background = clusters dominated by class -5 (both prompt gamma + near-capture)
            is_bkg = (sub["is_truth_neutron"] == 0) & (sub["dominant_class"] == -5)
        else:
            # "pop2": Population 2 only — class-5 clusters that are NOT the prompt AmBe gamma.
            # Population 1 (offset ~-17,846 ns) is already flagged as is_prompt_cluster=1.
            # Population 2 (offset ~-1.4 ns) is the near-capture contamination that is
            # indistinguishable by timing alone — the genuinely hard spurious problem.
            if "is_prompt_cluster" not in sub.columns:
                raise ValueError(
                    "--bkg-mode pop2 requires 'is_prompt_cluster' column in the features parquet. "
                    "Re-run: ambe mc features ..."
                )
            is_bkg = (
                (sub["is_truth_neutron"] == 0) &
                (sub["dominant_class"] == -5) &
                (sub["is_prompt_cluster"] == 0)
            )
    else:
        # "all": exclude only the prompt AmBe gamma cluster (is_prompt_cluster) if flagged
        if "is_prompt_cluster" in sub.columns:
            is_bkg = (sub["is_truth_neutron"] == 0) & (sub["is_prompt_cluster"] == 0)
        else:
            is_bkg = sub["is_truth_neutron"] == 0

    mask = is_sig | is_bkg
    sub  = sub[mask].reset_index(drop=True)
    y    = (sub["is_truth_neutron"] == 1).astype(int).to_numpy()

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
    3. Stratified split of events → assign all clusters of each event to the
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
        print(f"[cap] background capped: {n_bkg} → {cap_bkg}  "
              f"(ratio was 1:{n_bkg/n_sig:.1f}, now 1:{cap_bkg/n_sig:.1f})")
    elif n_sig > cap_sig:
        # More signal than allowed — subsample signal
        keep = rng.choice(sig_idx, size=cap_sig, replace=False)
        keep = np.sort(np.concatenate([keep, bkg_idx]))
        print(f"[cap] signal capped: {n_sig} → {cap_sig}  "
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

    Fits fresh RF, GBT (and XGBoost if available) on each fold's inner-train
    partition and evaluates on the inner-test partition.  Imputation medians and
    sample weights are computed inside each fold (no leakage).

    Returns
    -------
    cv_aucs  : dict  key → list of per-fold AUC scores
    cv_curves: dict  key → list of (fpr, tpr) per fold
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

NN_HIDDEN_UNITS  = 128    # 64 was underfitting 30 features with class imbalance
NN_HIDDEN_LAYERS = 3      # third layer for nonlinear interactions between β-params and timing
NN_DROPOUT       = 0.3
NN_LR            = 1e-3   # initial Adam lr; ReduceLROnPlateau will lower it
NN_EPOCHS        = 100    # was 20 — too restrictive; EarlyStopping handles overfitting
NN_BATCH_SIZE    = 256
NN_VERBOSE       = 1

def _build_nn(n_features: int):
    """
    input → [Dense(128) → BatchNorm → ReLU → Dropout(0.3)] × 3 → Dense(1, sigmoid)

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
    fig.suptitle(f"{run_name.upper()}  —  Cross-validation ROC  ({len(list(cv_curves.values())[0])}-fold)",
                 fontsize=11)

    for ax, key in zip(axes[0], keys):
        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(cv_curves[key])))
        for i, (fpr, tpr) in enumerate(cv_curves[key]):
            ax.plot(fpr, tpr, color=colors[i], lw=1.2, alpha=0.8,
                    label=f"fold {i+1}  AUC={cv_aucs[key][i]:.3f}" if i == 0 else
                          f"fold {i+1}  AUC={cv_aucs[key][i]:.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        mean_auc = float(np.mean(cv_aucs[key]))
        std_auc  = float(np.std(cv_aucs[key]))
        ax.set_title(f"{_NAME[key]}\nAUC = {mean_auc:.4f} ± {std_auc:.4f}", fontsize=10)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _roc_page(pdf, y_te: np.ndarray, model_scores: list, run_name: str, method: str):
    """model_scores: list of (display_name, test_scores, linestyle)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{run_name.upper()}  [{method}]  —  Cluster-level ROC", fontsize=11)

    ax = axes[0]
    for name, sc, ls in model_scores:
        fpr, tpr, _ = roc_curve(y_te, sc)
        auc = roc_auc_score(y_te, sc)
        ax.plot(fpr, tpr, lw=2, ls=ls, label=f"{name}  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False positive rate  (background efficiency)")
    ax.set_ylabel("True positive rate  (signal efficiency)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    ax = axes[1]
    for name, sc, ls in model_scores:
        fpr, tpr, _ = roc_curve(y_te, sc)
        ax.plot(tpr, 1 - fpr, lw=2, ls=ls, label=name)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection  (1 − FPR)")
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
    fig.suptitle(f"{run_name.upper()}  —  Score distributions (test set)", fontsize=11)
    bins = np.linspace(0, 1, 40)
    for idx, (name, sc, _) in enumerate(model_scores):
        ax = axes[idx // ncols][idx % ncols]
        auc = roc_auc_score(y_te, sc)
        kw = dict(bins=bins, density=True, alpha=0.6, edgecolor="none")
        ax.hist(sc[y_te == 1], color=COLORS["signal"],
                label=f"Signal  n={int((y_te==1).sum())}", **kw)
        ax.hist(sc[y_te == 0], color=COLORS["background"],
                label=f"Background  n={int((y_te==0).sum())}", **kw)
        ax.set_xlabel(f"{name} score"); ax.set_ylabel("Density")
        ax.set_title(f"{name}  AUC = {auc:.3f}")
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
    fig.suptitle(f"{run_name.upper()}  —  Feature importances", fontsize=11)
    for ax, (key, model) in zip(axes, has_imp):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        ax.bar(range(len(imp)), imp[idx], color="steelblue", alpha=0.85)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([features[i] for i in idx],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        ax.set_title(_NAME[key])
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _top_features_page(pdf, X_te, y_te, rf, features, run_name):
    top6 = np.argsort(rf.feature_importances_)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{run_name.upper()}  —  Top 6 features by RF importance", fontsize=11)
    for ax, fi in zip(axes.flat, top6):
        feat = features[fi]
        sig_v = X_te[y_te == 1, fi]
        bkg_v = X_te[y_te == 0, fi]
        all_v = np.concatenate([sig_v, bkg_v])
        lo, hi = float(np.nanpercentile(all_v, 1)), float(np.nanpercentile(all_v, 99))
        bins = np.linspace(lo, hi, 35)
        kw = dict(bins=bins, density=True, alpha=0.6, edgecolor="none")
        ax.hist(sig_v, color=COLORS["signal"],     label="Signal", **kw)
        ax.hist(bkg_v, color=COLORS["background"], label="Background", **kw)
        imp = rf.feature_importances_[fi]
        ax.set_title(f"{feat}  (importance = {imp:.3f})", fontsize=9)
        ax.set_xlabel(feat, fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _nn_history_page(pdf, history, run_name: str):
    has_auc = "val_auc" in history.history
    fig, axes = plt.subplots(1, 2 if has_auc else 1, figsize=(14 if has_auc else 8, 5), squeeze=False)
    fig.suptitle(f"{run_name.upper()}  —  Neural network training history", fontsize=11)

    ax = axes[0, 0]
    ax.plot(history.history["loss"],     lw=2, label="Train loss")
    ax.plot(history.history["val_loss"], lw=2, ls="--", label="Val loss")
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    ax.axvline(best_epoch, color="gray", ls=":", lw=1, label=f"Best loss epoch {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    if has_auc:
        ax = axes[0, 1]
        ax.plot(history.history["auc"],     lw=2, label="Train AUC")
        ax.plot(history.history["val_auc"], lw=2, ls="--", label="Val AUC")
        best_auc_epoch = int(np.argmax(history.history["val_auc"])) + 1
        ax.axvline(best_auc_epoch, color="gray", ls=":", lw=1, label=f"Best AUC epoch {best_auc_epoch}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
        ax.set_title("AUC (early-stopping monitor)"); ax.legend(); ax.grid(alpha=0.3)

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
    fig.suptitle(f"{run_name.upper()}  —  Event-level  ({score_col})", fontsize=11)

    ax = axes[0]
    ax.plot(thresholds, sig_eff, lw=2, color=COLORS["signal"],
            label="Signal events (has truth neutron)")
    ax.plot(thresholds, bkg_eff, lw=2, color=COLORS["background"],
            label="Background events (no truth neutron)")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="Score = 0.5")
    ax.set_xlabel(f"Score threshold  (max {score_col} per event)")
    ax.set_ylabel("Fraction of events passing threshold")
    ax.set_title("Event-level efficiency vs threshold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    ax = axes[1]
    sig_arr = np.array(sig_eff)
    bkg_arr = np.array(bkg_eff)
    valid   = ~(np.isnan(sig_arr) | np.isnan(bkg_arr))
    ax.plot(sig_arr[valid], 1 - bkg_arr[valid], lw=2, color="darkorchid")
    ax.set_xlabel("Signal efficiency (event level)")
    ax.set_ylabel("Background rejection (event level)")
    ax.set_title("Event-level ROC")
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(prog="mva_analysis")
    p.add_argument("--config",    required=True,  help="Pipeline YAML config")
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
    p.add_argument("--external-background", default=None, metavar="PATH_OR_NAME",
                   help="Use real-data background instead of MC internal background. "
                        "Pass a parquet file path, or one of the named shorthands: "
                        + ", ".join(_NAMED_BACKGROUNDS.keys()) + ". "
                        "Output files are tagged with --background-label.")
    p.add_argument("--background-label", default=None, metavar="LABEL",
                   help="Short label for the background source, used in output filenames "
                        "(e.g. 'offbeam', 'michel'). Auto-derived from named shorthands if omitted.")
    args = p.parse_args()

    if not _HAS_XGB:
        print("[mva] WARNING: xgboost not installed — skipping XGBoost.  pip install xgboost")
    if not _HAS_KERAS:
        print("[mva] WARNING: tensorflow not installed — skipping NN.  pip install tensorflow")
    if _HAS_KERAS:
        # TF auto-selects one GPU if CUDA is visible; no MirroredStrategy → single-device only.
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[mva] TensorFlow: {len(gpus)} GPU(s) visible"
              + ("  (no MirroredStrategy — using GPU:0 only)" if len(gpus) > 1 else ""))

    run_name, parquet_dir, plots_dir, csv_dir = load_paths(args.config)

    feat_path = parquet_dir / f"{run_name}__cluster_features.parquet"
    if not feat_path.exists():
        sys.exit(f"[mva] Features parquet not found: {feat_path}\n"
                 f"      Run: ambe mc features --config {args.config}")

    print(f"[mva] loading {feat_path}")
    df = pd.read_parquet(feat_path)

    # ── external real-data background mode ────────────────────────────────
    external_bkg = args.external_background
    if external_bkg is not None:
        # Resolve named shorthands → Path
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

        plot_run_name = f"{run_name} vs {bkg_tag}"
        # Output tag: <run_name>__mva__vs_<bkg_tag>  e.g. mc_lucho_full__mva__vs_offbeam
        mode_tag = f"__vs_{bkg_tag}"
        # Event-level page requires eventID; not meaningful with external background
        skip_event_level = True

    # ── internal MC background mode (default) ────────────────────────────
    else:
        bkg_mode = args.bkg_mode
        X, y, features, sub = prepare_data(df, method=args.method, bkg_mode=bkg_mode)
        bkg_label_str = {"darknoise":  "dark noise only (class 0)",
                         "nonneutron": "class-5 spurious (prompt gamma + near-capture)",
                         "pop2":       "Population 2 only (near-capture, offset ~-1 ns)",
                         "all":        "all non-neutron (excl. prompt)"}.get(bkg_mode, bkg_mode)
        print(f"[mva] bkg_mode={bkg_mode}  ({bkg_label_str})")
        mode_tag = "" if bkg_mode == "all" else f"__{bkg_mode}"
        plot_run_name = f"{run_name} [{bkg_mode} bkg]" if bkg_mode != "all" else run_name
        skip_event_level = False

    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())
    print(f"[mva] {X.shape[0]} clusters  |  "
          f"signal={n_sig}  background={n_bkg}  |  {len(features)} features")
    print(f"[mva] features: {features}")

    if n_sig < 10 or n_bkg < 10:
        sys.exit("[mva] Too few samples to train — check features parquet.")

    # ── Issue 2: cap imbalanced background before splitting ───────────────
    # max_ratio=3 keeps training balanced: at most 3× background per signal
    # (or 3× signal per background if the ratio is inverted, e.g. lucho vs michel).
    X, y, _cap_keep = cap_background(X, y, max_ratio=3.0)
    sub = sub.iloc[_cap_keep].reset_index(drop=True)
    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())
    print(f"[mva] after capping: signal={n_sig}  background={n_bkg}")

    # ── Issue 1: event-level train/test split ────────────────────────────
    # Assigns whole events to train or test so no event leaks across the boundary.
    # Falls back to cluster-level for external background (no shared event structure).
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = event_train_test_split(
        X, y, sub.iloc[:len(y)].reset_index(drop=True),
        test_size=args.test_size,
    )
    print(f"[mva] train={len(y_tr)}  test={len(y_te)}")

    # ── Issue 3: stratified k-fold CV on training set ────────────────────
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

    # Output paths — mode_tag encodes the background choice so all four combinations
    # land in the same run directory without overwriting each other:
    pdf_path   = plots_dir   / f"{run_name}__mva{mode_tag}.pdf"
    score_path = parquet_dir / f"{run_name}__mva_scores{mode_tag}.parquet"
    csv_path   = csv_dir     / f"{run_name}__mva_summary{mode_tag}.csv"

    print(f"[mva] writing plots → {pdf_path}")
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
    print(f"[mva] wrote scores → {score_path}")

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
    print(f"\n[mva] wrote summary → {csv_path}")
    print(f"[mva] wrote plots   → {pdf_path}")


if __name__ == "__main__":
    main()
