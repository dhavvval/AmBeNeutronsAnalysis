#!/usr/bin/env python3
"""Choose score thresholds from a saved MC score table.

Run from the repository root: python -m scripts.workflows.pick_best_model_cut
"""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NEUTRON_CLASSES = [1, 2, 3, 4]
SCORE_COLS = ["rf_score", "gbt_score", "xgb_score", "nn_score"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="MVA scores parquet")
    p.add_argument("--sig-eff", type=float, default=0.80,
                   help="target MC signal efficiency for the derived cut (default 0.80)")
    p.add_argument("--prefer", default=None,
                   help="optional: force this score col (e.g. rf_score) instead of auto-pick")
    args = p.parse_args()

    d = pd.read_parquet(args.scores)
    present = [c for c in SCORE_COLS if c in d.columns]
    if not present:
        sys.exit("[pick] no *_score columns in %s" % args.scores)

    # Held-out test rows for an honest AUC; fall back to all rows if no flag.
    te = d[d["in_test"]] if "in_test" in d.columns and d["in_test"].any() else d
    if "dominant_class" not in te.columns:
        sys.exit("[pick] scores parquet lacks dominant_class — cannot define signal.")
    y = te["dominant_class"].isin(NEUTRON_CLASSES).astype(int).to_numpy()
    if y.min() == y.max():
        sys.exit("[pick] test set has only one class — cannot pick.")

    pct = 100.0 * (1.0 - args.sig_eff)   # e.g. 80% eff -> 20th percentile of signal
    print("[pick] classifier comparison (test n=%d, sig=%d bkg=%d, target eff=%.0f%%):"
          % (len(te), int(y.sum()), int((y == 0).sum()), 100 * args.sig_eff),
          file=sys.stderr)
    print("[pick]   %-10s %8s %10s %12s" % ("model", "AUC", "cut", "bkg-rej"),
          file=sys.stderr)

    rows = []
    for col in present:
        sc = te[col].to_numpy(float)
        auc = roc_auc_score(y, sc)
        cut = float(np.percentile(sc[y == 1], pct))
        bkg_rej = float((sc[y == 0] < cut).mean())
        rows.append((col, auc, cut, bkg_rej))
        print("[pick]   %-10s %8.4f %10.4f %11.3f"
              % (col, auc, cut, bkg_rej), file=sys.stderr)

    if args.prefer and args.prefer in present:
        best = next(r for r in rows if r[0] == args.prefer)
        print("[pick] forced --prefer %s" % args.prefer, file=sys.stderr)
    else:
        best = max(rows, key=lambda r: r[1])   # highest AUC
    best_col, best_auc, best_cut, best_rej = best
    print("[pick] BEST = %s (AUC=%.4f, cut=%.4f @ %.0f%% eff, bkg-rej=%.3f)"
          % (best_col, best_auc, best_cut, 100 * args.sig_eff, best_rej), file=sys.stderr)

    # stdout: shell-evalable
    print("BEST_SCORE_COL=%s" % best_col)
    print("BEST_SCORE_CUT=%.4f" % best_cut)


if __name__ == "__main__":
    main()
