"""
tune_mva.py

Honest hyperparameter search for the CC-neutron MVA. Selects configs by 5-fold
cross-validated AUC (NOT test-set AUC), so any reported improvement is real and
not test-set overfitting. Uses mva_analysis's own prepare_data / cap_background so
the signal/background definition and capping match the production model exactly.

Reports a ranked table per model. Does NOT refit/freeze — it only tells you which
hyperparameters to put into train_trees. Re-train + --save-model after deciding.

Usage:
  source /exp/annie/app/users/dajana/myboy/bin/activate
  python tune_mva.py --config configs/cc_neutrino_optics.yaml
"""

from __future__ import annotations
import argparse, itertools
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import mva_analysis as M

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def cv_auc(make_model, X, y, n_splits=5, seed=42):
    """Mean±std AUC over stratified folds; medians imputed within each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        med = np.nanmedian(Xtr, axis=0)
        for j in range(Xtr.shape[1]):
            Xtr[:, j] = np.where(np.isnan(Xtr[:, j]), med[j], Xtr[:, j])
            Xte[:, j] = np.where(np.isnan(Xte[:, j]), med[j], Xte[:, j])
        m = make_model()
        sw = compute_sample_weight("balanced", y[tr])
        try:
            m.fit(Xtr, y[tr], sample_weight=sw)
        except TypeError:
            m.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", default="optics")
    p.add_argument("--bkg-mode", default="all")
    p.add_argument("--max-ratio", type=float, default=3.0)
    p.add_argument("--top", type=int, default=8, help="show top-N configs per model")
    args = p.parse_args()

    run_name, parquet_dir, *_ = M.load_paths(args.config)
    df = pd.read_parquet(parquet_dir / f"{run_name}__cluster_features.parquet")
    if "cc_pass" in df.columns:
        df = df[df["cc_pass"] == 1].reset_index(drop=True)
    X, y, feats, _ = M.prepare_data(df, method=args.method, bkg_mode=args.bkg_mode)
    X, y, _ = M.cap_background(X, y, max_ratio=args.max_ratio)
    print(f"[tune] {len(y)} clusters | sig={int(y.sum())} bkg={int((y==0).sum())} | {len(feats)} features\n")

    results = []

    # ── Random Forest grid ──
    rf_grid = list(itertools.product(
        [200, 400, 600],          # n_estimators
        [2, 5, 10],               # min_samples_leaf
        [None, 6, 10],            # max_depth
        ["sqrt", 0.5],            # max_features
    ))
    print(f"[tune] Random Forest: {len(rf_grid)} configs")
    for ne, leaf, depth, mf in rf_grid:
        mk = lambda ne=ne, leaf=leaf, depth=depth, mf=mf: RandomForestClassifier(
            n_estimators=ne, min_samples_leaf=leaf, max_depth=depth,
            max_features=mf, random_state=42, n_jobs=-1)
        mu, sd = cv_auc(mk, X, y)
        results.append(("RF", mu, sd, dict(n_estimators=ne, min_samples_leaf=leaf,
                                           max_depth=depth, max_features=mf)))

    # ── GBT grid ──
    gbt_grid = list(itertools.product(
        [150, 300],               # n_estimators
        [0.03, 0.05, 0.1],        # learning_rate
        [3, 4, 5],                # max_depth
        [0.8, 1.0],               # subsample
    ))
    print(f"[tune] GBT: {len(gbt_grid)} configs")
    for ne, lr, depth, sub in gbt_grid:
        mk = lambda ne=ne, lr=lr, depth=depth, sub=sub: GradientBoostingClassifier(
            n_estimators=ne, learning_rate=lr, max_depth=depth,
            subsample=sub, random_state=42)
        mu, sd = cv_auc(mk, X, y)
        results.append(("GBT", mu, sd, dict(n_estimators=ne, learning_rate=lr,
                                            max_depth=depth, subsample=sub)))

    # ── XGB grid ──
    if _HAS_XGB:
        n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
        spw = n_neg / max(n_pos, 1)
        xgb_grid = list(itertools.product(
            [200, 400], [3, 5], [0.03, 0.05, 0.1], [0.8, 1.0]))
        print(f"[tune] XGBoost: {len(xgb_grid)} configs")
        for ne, depth, lr, sub in xgb_grid:
            mk = lambda ne=ne, depth=depth, lr=lr, sub=sub: XGBClassifier(
                n_estimators=ne, max_depth=depth, learning_rate=lr, subsample=sub,
                colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                tree_method="hist", random_state=42, verbosity=0)
            mu, sd = cv_auc(mk, X, y)
            results.append(("XGB", mu, sd, dict(n_estimators=ne, max_depth=depth,
                                                learning_rate=lr, subsample=sub)))

    # ── Report ──
    print("\n" + "="*70)
    print("RANKED BY 5-FOLD CV AUC (honest — not test-set)")
    print("="*70)
    for model in ["RF", "GBT", "XGB"]:
        rows = sorted([r for r in results if r[0] == model], key=lambda r: -r[1])
        if not rows:
            continue
        print(f"\n{model}  (current production CV AUC for reference: "
              f"RF~0.695 GBT~0.665 XGB~0.668)")
        for _, mu, sd, params in rows[:args.top]:
            print(f"  AUC {mu:.4f} ± {sd:.4f}   {params}")
    best = max(results, key=lambda r: r[1])
    print(f"\n[tune] BEST overall: {best[0]} AUC {best[1]:.4f} ± {best[2]:.4f}  {best[3]}")


if __name__ == "__main__":
    main()
