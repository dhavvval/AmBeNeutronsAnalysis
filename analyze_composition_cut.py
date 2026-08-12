"""
analyze_composition_cut.py
==========================
Post-OPTICS composition-based rejection cut study.

The spurious cluster composition analysis showed that ~98% of spurious OPTICS
clusters are dominated by class -5 (non-neutron physics) hits, with only
1-2 neutron MCHits on average. This script:

  1. Sweeps rejection thresholds based on cluster composition
     (n_neutron ≤ threshold  AND/OR  frac_nonneutron ≥ threshold)
  2. Recomputes per-event cluster counts and OPTICS metrics after each cut
  3. Produces neutron multiplicity comparison plots (truth vs predicted)
  4. Shows a ROC curve: signal efficiency vs spurious rejection rate
  5. Outputs a summary table with the "best" cut recommendation

Run:
    cd /Users/dajana/Documents/AmBe/AmBeNeutronsAnalysis
    source ~/venvs/annie/bin/activate
    python analyze_composition_cut.py

Reads (from output_root/run_name/parquet/):
    *__cluster_features.parquet   (per-cluster features + composition + truth label)
    *__metrics.parquet            (per-event n_truth from original OPTICS run)

Writes:
    composition_cut_analysis.pdf
    composition_cut_summary.csv
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path("/Users/dajana/Documents/ambe_output")
RUN_NAME    = "mc_twoneutron"
OUT_DIR     = Path(__file__).parent

# Threshold grids to sweep
N_NEUTRON_THRESHOLDS  = [1, 2, 3, 4, 5]        # reject if n_neutron <= this
FRAC_NONNEUTRON_CUTS  = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # reject if frac_nonneutron >= this


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(output_root: Path, run_name: str):
    parquet_dir = output_root / run_name / "parquet"
    feat_path    = parquet_dir / f"{run_name}__cluster_features.parquet"
    metrics_path = parquet_dir / f"{run_name}__metrics.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"cluster_features parquet not found at {feat_path}\n"
            "Run: ambe mc features --config configs/mc_local_trial.yaml"
        )

    feat = pd.read_parquet(feat_path)
    print(f"Loaded cluster features: {len(feat)} clusters  "
          f"({feat['method'].value_counts().to_dict()})")

    # n_truth per event comes from the metrics parquet (best source of truth)
    n_truth_per_event = {}
    if metrics_path.exists():
        met = pd.read_parquet(metrics_path)
        # Use one config row per event for n_truth (it's the same across configs)
        met_sub = met[met["method"] == "optics"].drop_duplicates("eventID")
        n_truth_per_event = dict(zip(met_sub["eventID"], met_sub["n_truth"]))
        print(f"Loaded metrics: {len(n_truth_per_event)} events with n_truth")
    else:
        # Fall back: infer n_truth from matched cluster count + missed estimate
        print("WARNING: metrics parquet not found — inferring n_truth from features")
        for evid, grp in feat[feat["method"] == "optics"].groupby("eventID"):
            n_truth_per_event[evid] = int(grp["is_truth_neutron"].sum())

    return feat, n_truth_per_event


# ---------------------------------------------------------------------------
# Apply composition cut to one DataFrame of clusters
# ---------------------------------------------------------------------------
def apply_cut(df: pd.DataFrame, n_neutron_max: int, frac_nonneutron_min: float) -> pd.Series:
    """
    Return a boolean mask: True = KEEP this cluster (passes cut).
    A cluster is REJECTED if it has:
      - n_neutron <= n_neutron_max  AND
      - frac_nonneutron >= frac_nonneutron_min
    This captures the "prompt-gamma background cluster" pattern:
    very few real neutron hits, dominated by non-neutron physics.
    """
    reject = (
        (df["n_neutron"]     <= n_neutron_max) &
        (df["frac_nonneutron"] >= frac_nonneutron_min)
    )
    return ~reject


# ---------------------------------------------------------------------------
# Recompute per-event metrics after cut
# ---------------------------------------------------------------------------
def recompute_metrics(feat: pd.DataFrame,
                      n_truth_per_event: dict,
                      n_neutron_max: int,
                      frac_nonneutron_min: float,
                      method: str = "optics") -> pd.DataFrame:
    """
    For each event, apply the cut and recompute:
      n_truth, n_predicted, n_matched_kept, n_spurious_remaining,
      n_missed (matched clusters removed by cut), agreement
    """
    sub = feat[feat["method"] == method].copy()
    keep = apply_cut(sub, n_neutron_max, frac_nonneutron_min)
    sub["kept"] = keep

    rows = []
    for evid, grp in sub.groupby("eventID"):
        n_truth   = n_truth_per_event.get(int(evid), 0)
        kept_grp  = grp[grp["kept"]]

        n_predicted        = int(len(kept_grp))
        n_matched_kept     = int((kept_grp["is_truth_neutron"] == 1).sum())
        n_spurious_kept    = int((kept_grp["is_truth_neutron"] == 0).sum())
        n_matched_original = int((grp["is_truth_neutron"] == 1).sum())
        n_missed_by_cut    = n_matched_original - n_matched_kept   # matched but removed by cut
        n_missed_total     = n_truth - n_matched_kept              # truth not found

        agreement = int(
            n_matched_kept == n_truth and
            n_spurious_kept == 0 and
            n_missed_by_cut == 0
        )

        rows.append({
            "eventID":           int(evid),
            "n_truth":           n_truth,
            "n_predicted":       n_predicted,
            "n_matched_kept":    n_matched_kept,
            "n_spurious_kept":   n_spurious_kept,
            "n_missed_by_cut":   n_missed_by_cut,
            "n_missed_total":    max(0, int(n_missed_total)),
            "agreement":         agreement,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def sweep_thresholds(feat: pd.DataFrame, n_truth_per_event: dict) -> pd.DataFrame:
    """Sweep all (n_neutron_max, frac_nonneutron_min) combinations."""
    results = []
    for method in ["optics", "clusterfinder"]:
        sub = feat[feat["method"] == method]
        n_total = len(sub)
        n_matched_orig  = int((sub["is_truth_neutron"] == 1).sum())
        n_spurious_orig = int((sub["is_truth_neutron"] == 0).sum())

        # Baseline (no cut)
        ev_base = recompute_metrics(feat, n_truth_per_event, 0, 0.0, method)
        agr_base = float(ev_base["agreement"].mean())

        for nn_max, fn_min in product(N_NEUTRON_THRESHOLDS, FRAC_NONNEUTRON_CUTS):
            keep = apply_cut(sub, nn_max, fn_min)

            kept     = sub[keep]
            rejected = sub[~keep]

            n_matched_kept     = int((kept["is_truth_neutron"] == 1).sum())
            n_spurious_kept    = int((kept["is_truth_neutron"] == 0).sum())
            n_matched_rejected = int((rejected["is_truth_neutron"] == 1).sum())

            sig_eff  = n_matched_kept  / n_matched_orig   if n_matched_orig  > 0 else 1.0
            bkg_rej  = 1 - (n_spurious_kept / n_spurious_orig) if n_spurious_orig > 0 else 1.0

            ev_df    = recompute_metrics(feat, n_truth_per_event, nn_max, fn_min, method)
            agr_rate = float(ev_df["agreement"].mean())

            results.append({
                "method":          method,
                "n_neutron_max":   nn_max,
                "frac_nonn_min":   fn_min,
                "n_matched_kept":  n_matched_kept,
                "n_spurious_kept": n_spurious_kept,
                "n_matched_lost":  n_matched_rejected,
                "signal_efficiency": round(sig_eff,  4),
                "spurious_rejection":round(bkg_rej,   4),
                "agreement_rate":    round(agr_rate,  4),
                "delta_agreement":   round(agr_rate - agr_base, 4),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Neutron multiplicity analysis
# ---------------------------------------------------------------------------
def multiplicity_analysis(feat: pd.DataFrame,
                          n_truth_per_event: dict,
                          n_neutron_max: int,
                          frac_nonneutron_min: float) -> dict:
    """
    For each event, compute n_truth vs n_predicted (before/after cut)
    for both methods. Returns dicts suitable for plotting.
    """
    out = {}
    for method in ["optics", "clusterfinder"]:
        before_rows, after_rows = [], []
        sub = feat[feat["method"] == method]

        for evid, grp in sub.groupby("eventID"):
            n_truth   = n_truth_per_event.get(int(evid), 0)
            # Before cut: all clusters
            n_pred_before = int(len(grp))
            # After cut
            keep = apply_cut(grp, n_neutron_max, frac_nonneutron_min)
            n_pred_after  = int(keep.sum())

            before_rows.append({"n_truth": n_truth, "n_predicted": n_pred_before})
            after_rows.append( {"n_truth": n_truth, "n_predicted": n_pred_after})

        out[method] = {
            "before": pd.DataFrame(before_rows),
            "after":  pd.DataFrame(after_rows),
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _confusion_matrix_data(df: pd.DataFrame, max_n: int = 5):
    """Build a confusion matrix: truth (rows) vs predicted (cols)."""
    mat = np.zeros((max_n + 1, max_n + 1), dtype=int)
    for _, row in df.iterrows():
        t = min(int(row["n_truth"]),     max_n)
        p = min(int(row["n_predicted"]), max_n)
        mat[t, p] += 1
    return mat


def make_plots(feat: pd.DataFrame,
               n_truth_per_event: dict,
               sweep: pd.DataFrame,
               best_nn_max: int,
               best_fn_min: float,
               out_dir: Path):

    pdf_path = out_dir / "composition_cut_analysis.pdf"
    with PdfPages(pdf_path) as pdf:

        # ------------------------------------------------------------------ #
        # Fig 1 — ROC: signal efficiency vs spurious rejection
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"optics": "tomato", "clusterfinder": "steelblue"}
        markers = {0.40: "o", 0.50: "s", 0.60: "^", 0.70: "D",
                   0.80: "v", 0.90: "P"}
        for method in ["optics", "clusterfinder"]:
            sub = sweep[sweep["method"] == method]
            for fn_min, grp in sub.groupby("frac_nonn_min"):
                grp_s = grp.sort_values("n_neutron_max")
                ax.plot(
                    grp_s["signal_efficiency"],
                    grp_s["spurious_rejection"],
                    color=colors[method],
                    alpha=0.6,
                    marker=markers.get(fn_min, "o"),
                    markersize=7,
                    label=f"{method} frac≥{fn_min:.2f}" if method == "optics" else None,
                )
        ax.set_xlabel("Signal efficiency  (fraction of matched clusters kept)")
        ax.set_ylabel("Spurious rejection rate  (fraction of spurious clusters removed)")
        ax.set_title("Fig 1 — ROC: composition cut signal efficiency vs spurious rejection")
        # Add "best cut" star
        for method in ["optics", "clusterfinder"]:
            best_row = sweep[(sweep["method"] == method) &
                             (sweep["n_neutron_max"] == best_nn_max) &
                             (sweep["frac_nonn_min"] == best_fn_min)]
            if len(best_row):
                ax.scatter(best_row["signal_efficiency"], best_row["spurious_rejection"],
                           s=200, marker="*", color=colors[method],
                           zorder=5, label=f"{method} best cut (n≤{best_nn_max}, fn≥{best_fn_min})")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.axvline(0.5, color="gray", ls=":", lw=0.8)
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ------------------------------------------------------------------ #
        # Fig 2 — Agreement rate heatmap vs (n_neutron_max, frac_nonneutron)
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, method in zip(axes, ["optics", "clusterfinder"]):
            sub = sweep[sweep["method"] == method].pivot(
                index="n_neutron_max",
                columns="frac_nonn_min",
                values="agreement_rate"
            )
            im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn",
                           vmin=0.35, vmax=0.65,
                           origin="lower",
                           extent=[sub.columns.min() - 0.05, sub.columns.max() + 0.05,
                                   sub.index.min() - 0.5,    sub.index.max() + 0.5])
            plt.colorbar(im, ax=ax, label="Agreement rate")
            ax.set_xlabel("frac_nonneutron ≥ threshold")
            ax.set_ylabel("n_neutron ≤ threshold")
            ax.set_xticks(sub.columns)
            ax.set_yticks(sub.index)
            # Annotate cells
            for i, nn in enumerate(sub.index):
                for j, fn in enumerate(sub.columns):
                    val = sub.loc[nn, fn]
                    ax.text(fn, nn, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="black")
            ax.set_title(f"Fig 2 — Agreement rate: {method.upper()}")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ------------------------------------------------------------------ #
        # Fig 3 — Neutron multiplicity: truth vs predicted, before/after cut
        # ------------------------------------------------------------------ #
        mult = multiplicity_analysis(feat, n_truth_per_event, best_nn_max, best_fn_min)
        max_n = 4
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Fig 3 — Neutron multiplicity: truth vs predicted\n"
            f"Best cut: n_neutron ≤ {best_nn_max}  AND  frac_nonneutron ≥ {best_fn_min}",
            fontsize=12
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4)

        for idx, method in enumerate(["optics", "clusterfinder"]):
            for jdx, (stage, title) in enumerate([
                ("before", "Before composition cut"),
                ("after",  f"After cut (n≤{best_nn_max}, fn≥{best_fn_min})")
            ]):
                ax = fig.add_subplot(gs[jdx, idx])
                df_m = mult[method][stage]

                mat = _confusion_matrix_data(df_m, max_n=max_n)
                im  = ax.imshow(mat, cmap="Blues", aspect="auto",
                                norm=LogNorm(vmin=0.5, vmax=mat.max() + 0.1))
                plt.colorbar(im, ax=ax, label="Events")

                # Annotate
                for r in range(max_n + 1):
                    for c in range(max_n + 1):
                        if mat[r, c] > 0:
                            ax.text(c, r, str(mat[r, c]), ha="center", va="center",
                                    fontsize=9 if mat[r, c] > 10 else 7,
                                    color="black" if mat[r, c] < mat.max()*0.7 else "white")

                # Diagonal = correct multiplicity
                for d in range(max_n + 1):
                    ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                                               fill=False, edgecolor="tomato", lw=2))

                ticks = list(range(max_n + 1))
                labels = [str(x) if x < max_n else f"≥{max_n}" for x in ticks]
                ax.set_xticks(ticks); ax.set_xticklabels(labels)
                ax.set_yticks(ticks); ax.set_yticklabels(labels)
                ax.set_xlabel("N predicted clusters")
                ax.set_ylabel("N truth neutrons")

                # Agreement on diagonal
                agr = sum(mat[d, d] for d in range(max_n + 1)) / mat.sum()
                ax.set_title(f"{method.upper()} — {title}\nAgreement={agr:.3f}",
                             fontsize=9)

        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ------------------------------------------------------------------ #
        # Fig 4 — Multiplicity marginal distributions
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle("Fig 4 — Predicted neutron multiplicity distributions before/after cut",
                     fontsize=11)

        for ax_row, method in zip(axes, ["optics", "clusterfinder"]):
            for ax, (stage, label, color) in zip(
                ax_row,
                [("before", "Before cut", "gray"), ("after",  "After cut",  "tomato")]
            ):
                df_m = mult[method][stage]
                n_truth_vals = df_m["n_truth"].clip(0, max_n)
                n_pred_vals  = df_m["n_predicted"].clip(0, max_n)
                bins = np.arange(-0.5, max_n + 1.5, 1)

                ax.hist(n_truth_vals,  bins=bins, alpha=0.5, color="steelblue",
                        label="Truth n_neutron")
                ax.hist(n_pred_vals,   bins=bins, alpha=0.5, color=color,
                        label=f"Predicted ({label})")

                # Fraction correct
                correct = (df_m["n_truth"].clip(0, max_n) ==
                           df_m["n_predicted"].clip(0, max_n)).mean()
                ax.set_title(f"{method.upper()} — {label}\n"
                             f"n_truth==n_pred: {correct:.3f}", fontsize=9)
                ax.set_xlabel("Number of neutrons")
                ax.set_ylabel("Events")
                ax.legend(fontsize=8)
                ax.set_xticks(range(max_n + 1))

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ------------------------------------------------------------------ #
        # Fig 5 — Delta agreement vs cut threshold (line plot)
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, method in zip(axes, ["optics", "clusterfinder"]):
            sub = sweep[sweep["method"] == method]
            for fn_min, grp in sub.groupby("frac_nonn_min"):
                grp_s = grp.sort_values("n_neutron_max")
                ax.plot(grp_s["n_neutron_max"],
                        grp_s["agreement_rate"],
                        "o-", label=f"frac_nonn ≥ {fn_min:.2f}")
            ax.set_xlabel("n_neutron ≤ threshold  (reject if ≤ this many neutron hits)")
            ax.set_ylabel("Agreement rate")
            ax.set_title(f"Fig 5 — {method.upper()}: agreement rate vs cut threshold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.axhline(sub[sub["n_neutron_max"] == 0]["agreement_rate"].mean() if len(sub) else 0.4,
                       color="black", ls="--", lw=1, label="No cut (baseline)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Recommendation: find best cut
# ---------------------------------------------------------------------------
def recommend_cut(sweep: pd.DataFrame, method: str = "optics") -> tuple:
    """
    Find the cut that maximises agreement_rate while keeping
    signal_efficiency ≥ 0.90 (don't sacrifice too many real neutrons).
    """
    sub = sweep[(sweep["method"] == method) &
                (sweep["signal_efficiency"] >= 0.90)]
    if len(sub) == 0:
        sub = sweep[sweep["method"] == method]
    best = sub.loc[sub["agreement_rate"].idxmax()]
    return int(best["n_neutron_max"]), float(best["frac_nonn_min"]), best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rfile = sys.argv[1] if len(sys.argv) > 1 else None
    if rfile:
        OUTPUT_ROOT = Path(rfile).parent.parent
        RUN_NAME    = Path(rfile).parent.name if not rfile else RUN_NAME

    feat, n_truth_per_event = load_data(OUTPUT_ROOT, RUN_NAME)

    # ---- Threshold sweep ----
    print("\nSweeping composition cut thresholds...")
    sweep = sweep_thresholds(feat, n_truth_per_event)

    # Save sweep CSV
    csv_path = OUT_DIR / "composition_cut_summary.csv"
    sweep.to_csv(csv_path, index=False)
    print(f"Saved sweep summary: {csv_path}")

    # ---- Print key table ----
    print("\n" + "=" * 80)
    print("COMPOSITION CUT SWEEP — OPTICS (sorted by agreement_rate desc)")
    print("=" * 80)
    optics_sweep = sweep[sweep["method"] == "optics"].sort_values(
        "agreement_rate", ascending=False
    )
    print(optics_sweep[["n_neutron_max", "frac_nonn_min",
                         "signal_efficiency", "spurious_rejection",
                         "n_spurious_kept", "agreement_rate",
                         "delta_agreement"]].head(10).to_string(index=False))

    # ---- Best cut recommendation ----
    best_nn, best_fn, best_row = recommend_cut(sweep, "optics")
    print(f"\n{'='*60}")
    print(f"RECOMMENDED CUT (OPTICS, sig_efficiency ≥ 0.90):")
    print(f"  Reject cluster if: n_neutron ≤ {best_nn}  AND  frac_nonneutron ≥ {best_fn}")
    print(f"  Signal efficiency:  {best_row['signal_efficiency']:.4f}")
    print(f"  Spurious rejection: {best_row['spurious_rejection']:.4f}")
    print(f"  Agreement rate:     {best_row['agreement_rate']:.4f}")
    print(f"  Delta agreement:    {best_row['delta_agreement']:+.4f} vs no-cut baseline")
    print(f"{'='*60}")

    # Also show n=1 truth events specifically
    print("\nNeutron multiplicity for n_truth=1 events (dominant case):")
    mult = multiplicity_analysis(feat, n_truth_per_event, best_nn, best_fn)
    for method in ["optics", "clusterfinder"]:
        for stage in ["before", "after"]:
            df_m = mult[method][stage]
            ev1  = df_m[df_m["n_truth"] == 1]
            if len(ev1):
                correct   = (ev1["n_predicted"] == 1).sum()
                too_many  = (ev1["n_predicted"] >  1).sum()
                too_few   = (ev1["n_predicted"] <  1).sum()
                print(f"  [{method:14s}] {stage:6s}: "
                      f"n_pred=1: {100*correct/len(ev1):.1f}%  "
                      f"n_pred>1: {100*too_many/len(ev1):.1f}%  "
                      f"n_pred=0: {100*too_few/len(ev1):.1f}%  "
                      f"(n={len(ev1)} events)")

    # ---- Plots ----
    make_plots(feat, n_truth_per_event, sweep, best_nn, best_fn, OUT_DIR)
    print("\nDone.")
