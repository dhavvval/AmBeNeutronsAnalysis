#!/usr/bin/env python3
"""
MVA: neutron capture vs Michel electron discrimination.

Signal     : is_truth_neutron == 1  (from one-neutron OPTICS cluster features)
Background : Michel electron clusters (from extract_michel_features.py)

Because the Michel sample is small (~50 clusters), AUC is estimated via
5-fold stratified cross-validation (cross_val_predict out-of-fold scores)
rather than a single train/test split.  A final model trained on all data
provides feature importances.

Usage
-----
    python mva_neutron_vs_michel.py \\
        --neutron-parquet ambe_output/mc_oneneutron_no_darknoise/parquet/mc_oneneutron_no_darknoise__cluster_features.parquet \\
        --michel-parquet  ambe_output/michel_features.parquet \\
        --output-dir      ambe_output/neutron_vs_michel/ \\
        --method          optics

    python mva_neutron_vs_michel.py --config configs/mc_michel_mva.yaml
    python mva_neutron_vs_michel.py --config configs/mc_michel_mva.yaml --no-nn

Outputs
-------
    plots/neutron_vs_michel.pdf
        Page 1  ROC curve (5-fold CV out-of-fold)
        Page 2  Score distributions signal vs background
        Page 3  Feature importances RF, GBT, XGBoost
        Page 4  Top-6 features by RF importance
        Page 5  Neural network training history  (skipped if --no-nn)
        Page 6  Event-level efficiency vs score threshold
    parquet/neutron_vs_michel_scores.parquet
    csv/neutron_vs_michel_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

# Reuse PHYSICS_FEATURES list from mva_analysis.py
sys.path.insert(0, str(Path(__file__).parent))
from mva_analysis import PHYSICS_FEATURES

COLORS = {"neutron": "#0077BB", "michel": "#EE7733"}
_NAME  = {"rf": "Random Forest", "gbt": "GBT", "xgb": "XGBoost", "nn": "Neural Network"}
_LS    = {"rf": "-",             "gbt": "--",   "xgb": "-.",       "nn": ":"}

# Neural network architecture — same as mva_analysis.py
NN_HIDDEN_UNITS  = 128
NN_HIDDEN_LAYERS = 3
NN_DROPOUT       = 0.3
NN_LR            = 1e-3
NN_EPOCHS        = 100
NN_BATCH_SIZE    = 256


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(neutron_parquet: str, michel_parquet: str,
                 method: str = "optics") -> tuple:
    """
    Load and combine neutron (signal) and Michel (background) features.

    Returns X, y, feature_names, combined_df.
    """
    df_n = pd.read_parquet(neutron_parquet)
    # Keep only OPTICS neutron signal clusters
    df_n = df_n[(df_n["method"] == method) &
                (df_n["is_truth_neutron"] == 1)].copy().reset_index(drop=True)
    df_n["label"] = 1
    df_n["sample"] = "neutron"
    # Normalise event identifier column name
    if "eventID" in df_n.columns:
        df_n["ev_id"] = df_n["eventID"].astype(str) + "_n"

    df_m = pd.read_parquet(michel_parquet).copy().reset_index(drop=True)
    df_m["label"] = 0
    df_m["sample"] = "michel"
    if "event_idx" in df_m.columns:
        df_m["ev_id"] = df_m["event_idx"].astype(str) + "_m"

    df = pd.concat([df_n, df_m], ignore_index=True)
    y  = df["label"].to_numpy(int)

    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())
    print(f"[mva] signal (neutron): {n_sig}   background (Michel): {n_bkg}")
    if n_bkg < 20:
        print(f"[mva] WARNING: only {n_bkg} Michel clusters — "
              f"results may be unreliable.  Consider relaxing selection cuts "
              f"in extract_michel_features.py.")

    # Features present in the combined dataset with >=50% non-NaN coverage.
    avail = [f for f in PHYSICS_FEATURES
             if f in df.columns and df[f].notna().mean() >= 0.5]
    X = df[avail].to_numpy(dtype=float)

    # Median-impute residual NaNs
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, j] = float(np.nanmedian(col))

    return X, y, avail, df


# ---------------------------------------------------------------------------
# CV-based score estimation — tree models
# ---------------------------------------------------------------------------

def cv_scores(clf, X: np.ndarray, y: np.ndarray,
              n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predicted probabilities via StratifiedKFold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

def _build_nn(n_features: int):
    """input → [Dense(128) → BatchNorm → ReLU → Dropout(0.3)] × 3 → sigmoid."""
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


def cv_scores_nn(X: np.ndarray, y: np.ndarray,
                 n_splits: int = 5) -> tuple[np.ndarray, list]:
    """
    Manual stratified-CV for the NN: fit StandardScaler on train fold only
    (no leakage), build a fresh model each fold.

    Returns (oof_scores, list_of_histories).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    histories = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        print(f"[mva]   NN fold {fold + 1}/{n_splits} …")
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X[tr])
        X_va_sc  = scaler.transform(X[va])
        sw       = compute_sample_weight("balanced", y[tr])
        model    = _build_nn(X.shape[1])
        hist = model.fit(
            X_tr_sc, y[tr],
            sample_weight=sw,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_data=(X_va_sc, y[va]),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=15,
                    restore_best_weights=True, verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc", mode="max", factor=0.5,
                    patience=7, min_lr=1e-5, verbose=0,
                ),
            ],
            verbose=0,
        )
        oof[va] = model.predict(X_va_sc, verbose=0).ravel()
        histories.append(hist)
    return oof, histories


def train_nn_final(X: np.ndarray, y: np.ndarray):
    """Train a final NN on all data for the score parquet (no validation split)."""
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    sw      = compute_sample_weight("balanced", y)
    model   = _build_nn(X.shape[1])
    model.fit(X_sc, y, sample_weight=sw,
              epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, verbose=0)
    return model, scaler


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _roc_page(pdf, y, model_scores: list, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title}  —  ROC (5-fold CV, out-of-fold)", fontsize=11)

    ax = axes[0]
    for name, sc, ls in model_scores:
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, lw=2, ls=ls, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False positive rate  (Michel efficiency)")
    ax.set_ylabel("True positive rate  (neutron efficiency)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    ax = axes[1]
    for name, sc, ls in model_scores:
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(tpr, 1 - fpr, lw=2, ls=ls, label=name)
    ax.set_xlabel("Neutron signal efficiency")
    ax.set_ylabel("Michel rejection  (1 − FPR)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _score_dist_page(pdf, y, model_scores: list, title: str):
    n = len(model_scores)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
    fig.suptitle(f"{title}  —  Score distributions", fontsize=11)
    bins = np.linspace(0, 1, 40)
    for idx, (name, sc, _) in enumerate(model_scores):
        ax = axes[idx // ncols][idx % ncols]
        auc = roc_auc_score(y, sc)
        kw = dict(bins=bins, density=True, alpha=0.6, edgecolor="none")
        ax.hist(sc[y == 1], color=COLORS["neutron"],
                label=f"Neutron  n={int((y==1).sum())}", **kw)
        ax.hist(sc[y == 0], color=COLORS["michel"],
                label=f"Michel  n={int((y==0).sum())}", **kw)
        ax.set_xlabel(f"{name} score"); ax.set_ylabel("Density")
        ax.set_title(f"{name}  AUC={auc:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for idx in range(len(model_scores), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _importance_page(pdf, tree_models: dict, features: list, title: str):
    has_imp = [(k, m) for k, m in tree_models.items()
               if hasattr(m, "feature_importances_")]
    if not has_imp:
        return
    fig, axes = plt.subplots(1, len(has_imp), figsize=(7 * len(has_imp), 5))
    if len(has_imp) == 1:
        axes = [axes]
    fig.suptitle(f"{title}  —  Feature importances", fontsize=11)
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


def _top_features_page(pdf, X, y, rf, features, title):
    top6 = np.argsort(rf.feature_importances_)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{title}  —  Top 6 features (RF importance)", fontsize=11)
    for ax, fi in zip(axes.flat, top6):
        feat = features[fi]
        sig_v = X[y == 1, fi]
        bkg_v = X[y == 0, fi]
        lo = float(np.nanpercentile(np.concatenate([sig_v, bkg_v]), 1))
        hi = float(np.nanpercentile(np.concatenate([sig_v, bkg_v]), 99))
        bins = np.linspace(lo, hi, 35)
        kw = dict(bins=bins, density=True, alpha=0.6, edgecolor="none")
        ax.hist(sig_v, color=COLORS["neutron"], label="Neutron", **kw)
        ax.hist(bkg_v, color=COLORS["michel"],  label="Michel",  **kw)
        imp = rf.feature_importances_[fi]
        ax.set_title(f"{feat}  (imp={imp:.3f})", fontsize=9)
        ax.set_xlabel(feat, fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _nn_history_page(pdf, histories: list, title: str):
    """Overlay training curves from all CV folds."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title}  —  Neural network training (CV folds)", fontsize=11)

    ax = axes[0]
    for i, h in enumerate(histories):
        ax.plot(h.history["loss"],     lw=1.2, alpha=0.7, label=f"Fold {i+1} train")
        ax.plot(h.history["val_loss"], lw=1.2, alpha=0.7, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Loss  (solid=train, dashed=val)"); ax.grid(alpha=0.3)

    ax = axes[1]
    for i, h in enumerate(histories):
        if "auc" in h.history:
            ax.plot(h.history["auc"],     lw=1.2, alpha=0.7, label=f"Fold {i+1} train")
            ax.plot(h.history["val_auc"], lw=1.2, alpha=0.7, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    ax.set_title("AUC  (solid=train, dashed=val)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _event_level_page(pdf, df: pd.DataFrame, model_scores: list, title: str):
    """
    Event-level efficiency vs score threshold.

    For each event (grouped by ev_id), take the max cluster score.
    Neutron events (label==1): what fraction pass threshold? → signal efficiency.
    Michel events (label==0): what fraction pass threshold? → background rate.
    Also plots the event-level ROC curve (signal eff vs Michel rejection).
    """
    if "ev_id" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title}  —  Event-level efficiency", fontsize=11)
    thresholds = np.linspace(0, 1, 100)

    for name, sc, ls in model_scores:
        score_col = f"_tmp_{name}"
        df = df.copy()
        df[score_col] = sc

        ev_label = df.groupby("ev_id")["label"].max()
        ev_score = df.groupby("ev_id")[score_col].max()
        ev = pd.concat([ev_label, ev_score], axis=1).dropna()

        sig = ev[ev["label"] == 1]
        bkg = ev[ev["label"] == 0]

        sig_eff = [(sig[score_col] >= t).mean() for t in thresholds]
        bkg_eff = [(bkg[score_col] >= t).mean() for t in thresholds]

        axes[0].plot(thresholds, sig_eff, lw=2, ls=ls, label=f"{name} neutron eff")
        axes[0].plot(thresholds, bkg_eff, lw=1.5, ls=ls, alpha=0.5, label=f"{name} Michel rate")

        sig_arr = np.array(sig_eff)
        bkg_arr = np.array(bkg_eff)
        valid   = ~(np.isnan(sig_arr) | np.isnan(bkg_arr))
        axes[1].plot(sig_arr[valid], 1 - bkg_arr[valid], lw=2, ls=ls, label=name)

        df.drop(columns=[score_col], inplace=True)

    axes[0].axvline(0.5, color="gray", ls=":", lw=1, label="Score = 0.5")
    axes[0].set_xlabel("Score threshold"); axes[0].set_ylabel("Fraction of events passing")
    axes[0].set_title("Event-level efficiency vs threshold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)

    axes[1].set_xlabel("Signal efficiency (event level)")
    axes[1].set_ylabel("Michel rejection (event level)")
    axes[1].set_title("Event-level ROC")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(prog="mva_neutron_vs_michel")
    p.add_argument("--config",          help="YAML config (configs/mc_michel_mva.yaml)")
    # Individual args override config values when both supplied
    p.add_argument("--neutron-parquet",
                   help="cluster_features.parquet from one-neutron OPTICS run")
    p.add_argument("--michel-parquet",
                   help="michel_features.parquet from extract_michel_features.py")
    p.add_argument("--output-dir",
                   help="Directory for plots, parquet, csv outputs")
    p.add_argument("--method",  default=None,
                   choices=["optics", "clusterfinder"])
    p.add_argument("--cv-folds", type=int, default=None,
                   help="Number of CV folds for AUC estimation (default 5)")
    p.add_argument("--no-nn",   action="store_true",
                   help="Skip neural network training (faster — tree models only)")
    args = p.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    mva_block = cfg.get("mva", {})

    neutron_parquet = args.neutron_parquet or cfg.get("neutron_parquet")
    michel_parquet  = (args.michel_parquet
                       or cfg.get("michel", {}).get("output"))
    output_dir      = args.output_dir  or mva_block.get("output_dir")
    method          = args.method      or cfg.get("neutron_method", "optics")
    cv_folds        = args.cv_folds    or int(mva_block.get("cv_folds", 5))

    if not all([neutron_parquet, michel_parquet, output_dir]):
        p.error("Provide --config or all of --neutron-parquet "
                "--michel-parquet --output-dir")

    if not _HAS_XGB:
        print("[mva] WARNING: xgboost not installed — skipping XGBoost.")
    if not _HAS_KERAS:
        print("[mva] WARNING: tensorflow not installed — skipping NN.")
    if _HAS_KERAS and not args.no_nn:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[mva] TensorFlow: {len(gpus)} GPU(s) visible")

    out_dir = Path(output_dir)
    plots_dir   = out_dir / "plots";   plots_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = out_dir / "parquet"; parquet_dir.mkdir(parents=True, exist_ok=True)
    csv_dir     = out_dir / "csv";     csv_dir.mkdir(parents=True, exist_ok=True)

    X, y, features, df = prepare_data(neutron_parquet, michel_parquet, method=method)
    print(f"[mva] {X.shape[0]} clusters  |  {len(features)} features")
    print(f"[mva] features: {features}")

    n_sig = int(y.sum()); n_bkg = int((y == 0).sum())
    if n_sig < 5 or n_bkg < 5:
        sys.exit("[mva] Too few samples to train — check parquets.")

    # ── Define classifiers ──
    sw_bal = compute_sample_weight("balanced", y)

    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    gbt = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, subsample=0.8, random_state=42)
    tree_clfs = {"rf": rf, "gbt": gbt}

    if _HAS_XGB:
        xgb = _XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=n_bkg / max(n_sig, 1),
            eval_metric="logloss", tree_method="hist",
            random_state=42, verbosity=0)
        tree_clfs["xgb"] = xgb

    n_splits = min(cv_folds, n_bkg)
    if n_splits < cv_folds:
        print(f"[mva] WARNING: reduced CV folds to {n_splits} "
              f"(only {n_bkg} Michel clusters)")

    # ── CV out-of-fold scores — tree models ──
    print(f"[mva] {n_splits}-fold CV  (n_sig={n_sig}  n_bkg={n_bkg})")
    model_scores = []   # (display_name, oof_scores, linestyle)
    for key, clf in tree_clfs.items():
        print(f"[mva]   {_NAME[key]} …")
        oof = cv_scores(clf, X, y, n_splits=n_splits)
        auc = roc_auc_score(y, oof)
        print(f"[mva]   {_NAME[key]:20s}  CV AUC = {auc:.3f}")
        model_scores.append((_NAME[key], oof, _LS[key]))

    # ── CV out-of-fold scores — NN ──
    nn_histories  = None
    nn_final      = None
    nn_scaler     = None
    if _HAS_KERAS and not args.no_nn:
        print(f"[mva]   Neural Network ({n_splits}-fold CV) …")
        nn_oof, nn_histories = cv_scores_nn(X, y, n_splits=n_splits)
        nn_auc = roc_auc_score(y, nn_oof)
        print(f"[mva]   {'Neural Network':20s}  CV AUC = {nn_auc:.3f}")
        model_scores.append((_NAME["nn"], nn_oof, _LS["nn"]))
    elif args.no_nn:
        print("[mva] NN skipped (--no-nn)")

    # ── Train final models on all data (for importances + score parquet) ──
    print("[mva] training final models on full dataset …")
    for key, clf in tree_clfs.items():
        clf.fit(X, y, sample_weight=sw_bal)
    if _HAS_KERAS and not args.no_nn:
        print("[mva] training final NN …")
        nn_final, nn_scaler = train_nn_final(X, y)

    title = "Neutron capture vs Michel electron"

    # ── Plots ──
    pdf_path = plots_dir / "neutron_vs_michel.pdf"
    print(f"[mva] writing plots → {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        _roc_page(pdf, y, model_scores, title)
        _score_dist_page(pdf, y, model_scores, title)
        _importance_page(pdf, tree_clfs, features, title)
        _top_features_page(pdf, X, y, rf, features, title)
        if nn_histories is not None:
            _nn_history_page(pdf, nn_histories, title)
        _event_level_page(pdf, df, model_scores, title)

    # ── Save scores ──
    df = df.copy()
    for key, clf in tree_clfs.items():
        df[f"{key}_score"] = clf.predict_proba(X)[:, 1]
    if nn_final is not None:
        df["nn_score"] = nn_final.predict(nn_scaler.transform(X), verbose=0).ravel()
    score_path = parquet_dir / "neutron_vs_michel_scores.parquet"
    df.to_parquet(score_path, index=False)
    print(f"[mva] wrote scores → {score_path}")

    # ── Save summary ──
    imp_df = pd.DataFrame({"feature": features})
    for key, clf in tree_clfs.items():
        if hasattr(clf, "feature_importances_"):
            imp_df[f"{key}_importance"] = clf.feature_importances_
    if "rf_importance" in imp_df.columns:
        imp_df = imp_df.sort_values("rf_importance", ascending=False).reset_index(drop=True)
        imp_df.insert(0, "rf_rank", range(1, len(features) + 1))

    csv_path = csv_dir / "neutron_vs_michel_summary.csv"
    imp_df.to_csv(csv_path, index=False)

    print(f"\n[mva] ===== SUMMARY =====")
    for name, sc, _ in model_scores:
        print(f"  AUC  {name:20s} : {roc_auc_score(y, sc):.4f}")
    if "rf_importance" in imp_df.columns:
        imp_cols = [c for c in imp_df.columns if c.endswith("_importance")]
        print(f"\n  Feature importances (RF rank):")
        for _, row in imp_df.iterrows():
            parts = [f"{row['feature']:25s}"]
            for c in imp_cols:
                tag = c.replace("_importance", "").upper()
                parts.append(f"{tag}={row[c]:.4f}")
            print(f"    {int(row.get('rf_rank', 0)):2d}. {'  '.join(parts)}")
    print(f"\n[mva] wrote summary → {csv_path}")
    print(f"[mva] wrote plots   → {pdf_path}")


if __name__ == "__main__":
    main()
