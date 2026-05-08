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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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
]

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
                 method: str = "optics") -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Filter to one clustering method, build feature matrix and binary labels.

    Returns X, y, feature_names, filtered_sub_df.
    """
    sub = df[df["method"] == method].copy().reset_index(drop=True)

    is_sig = sub["is_truth_neutron"] == 1
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


# ---------------------------------------------------------------------------
# Tree model training
# ---------------------------------------------------------------------------

def train_trees(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """
    Fit RF, GBT, and (if xgboost installed) XGBoost.
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

NN_HIDDEN_UNITS = 64
NN_HIDDEN_LAYERS = 2
NN_DROPOUT = 0.3
NN_EPOCHS = 20
NN_BATCH_SIZE = 256
NN_VERBOSE = 1

def _build_nn(n_features: int):
    """
    Lightweight test architecture (easy to revise later):
        input → [Dense(64, ReLU) → Dropout(0.3)] × 2 → Dense(1, sigmoid)
    Binary cross-entropy loss, Adam optimiser.
    """
    inp = tf.keras.Input(shape=(n_features,))
    x = inp
    for _ in range(NN_HIDDEN_LAYERS):
        x = tf.keras.layers.Dense(NN_HIDDEN_UNITS, activation="relu")(x)
        x = tf.keras.layers.Dropout(NN_DROPOUT)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
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
                monitor="val_loss", patience=10,
                restore_best_weights=True, verbose=1,
            )
        ],
        verbose=NN_VERBOSE,  # show batch-level progress for responsiveness
    )
    return model, scaler, history


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

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
    """NN training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{run_name.upper()}  —  Neural network training history", fontsize=11)
    ax.plot(history.history["loss"],     lw=2, label="Train loss")
    ax.plot(history.history["val_loss"], lw=2, ls="--", label="Val loss")
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    ax.axvline(best_epoch, color="gray", ls=":", lw=1, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Loss (early stopping on val_loss)"); ax.legend(); ax.grid(alpha=0.3)
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
    args = p.parse_args()

    if not _HAS_XGB:
        print("[mva] WARNING: xgboost not installed — skipping XGBoost.  pip install xgboost")
    if not _HAS_KERAS:
        print("[mva] WARNING: tensorflow not installed — skipping NN.  pip install tensorflow")

    run_name, parquet_dir, plots_dir, csv_dir = load_paths(args.config)

    feat_path = parquet_dir / f"{run_name}__cluster_features.parquet"
    if not feat_path.exists():
        sys.exit(f"[mva] Features parquet not found: {feat_path}\n"
                 f"      Run: ambe mc features --config {args.config}")

    print(f"[mva] loading {feat_path}")
    df = pd.read_parquet(feat_path)

    X, y, features, sub = prepare_data(df, method=args.method)
    n_sig = int(y.sum())
    n_bkg = int((y == 0).sum())
    print(f"[mva] {X.shape[0]} clusters  |  "
          f"signal={n_sig}  background={n_bkg}  |  {len(features)} features")
    print(f"[mva] features: {features}")

    if n_sig < 10 or n_bkg < 10:
        sys.exit("[mva] Too few samples to train — check features parquet.")

    # Stratified train / test split
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)),
        test_size=args.test_size, stratify=y, random_state=42,
    )
    print(f"[mva] train={len(y_tr)}  test={len(y_te)}")

    # ---- Tree models ----
    tree_label = "RF, GBT" + (", XGBoost" if _HAS_XGB else "")
    print(f"[mva] training tree models ({tree_label}) …")
    tree_models = train_trees(X_tr, y_tr)

    # ---- Neural network ----
    nn_model, nn_scaler, nn_history = None, None, None
    if _HAS_KERAS and not args.no_nn:
        hidden = "-".join([str(NN_HIDDEN_UNITS)] * NN_HIDDEN_LAYERS)
        arch = f"{len(features)}-{hidden}-1" if hidden else f"{len(features)}-1"
        print(f"[mva] training NN  ({arch}, ReLU, dropout={NN_DROPOUT})"
              f"  [{len(features)} features after NaN filter]")
        nn_model, nn_scaler, nn_history = train_nn(X_tr, y_tr)
        n_epochs = len(nn_history.history["loss"])
        print(f"[mva] NN stopped at epoch {n_epochs} / 50")
    elif args.no_nn:
        print("[mva] NN skipped (--no-nn)")

    # ---- Collect scores on test set and full dataset ----
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

    # Attach scores to sub dataframe
    sub = sub.copy()
    for col, vals in all_score_cols.items():
        sub[col] = vals
    sub["in_test"] = False
    sub.loc[idx_te, "in_test"] = True

    # ---- Plots ----
    pdf_path = plots_dir / f"{run_name}__mva.pdf"
    print(f"[mva] writing plots → {pdf_path}")
    best_event_col = next(
        (c for c in ["xgb_score", "nn_score", "gbt_score"] if c in sub.columns),
        "gbt_score",
    )
    with PdfPages(pdf_path) as pdf:
        _roc_page(pdf, y_te, model_scores, run_name, args.method)
        _score_dist_page(pdf, y_te, model_scores, run_name)
        _importance_page(pdf, tree_models, features, run_name)
        _top_features_page(pdf, X_te, y_te, tree_models["rf"], features, run_name)
        if nn_history is not None:
            _nn_history_page(pdf, nn_history, run_name)
        _event_level_page(pdf, sub, run_name, score_col=best_event_col)

    # ---- Scored parquet ----
    score_path = parquet_dir / f"{run_name}__mva_scores.parquet"
    sub.to_parquet(score_path, index=False)
    print(f"[mva] wrote scores → {score_path}")

    # ---- Summary CSV ----
    imp_df = pd.DataFrame({"feature": features})
    for key, model in tree_models.items():
        if hasattr(model, "feature_importances_"):
            imp_df[f"{key}_importance"] = model.feature_importances_
    if "rf_importance" in imp_df.columns:
        imp_df = imp_df.sort_values("rf_importance", ascending=False).reset_index(drop=True)
        imp_df.insert(0, "rf_rank", range(1, len(features) + 1))

    csv_path = csv_dir / f"{run_name}__mva_summary.csv"
    imp_df.to_csv(csv_path, index=False)

    print(f"\n[mva] ===== SUMMARY =====")
    for name, sc, _ in model_scores:
        print(f"  AUC  {name:20s} : {roc_auc_score(y_te, sc):.4f}")
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
