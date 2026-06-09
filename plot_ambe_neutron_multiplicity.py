"""
plot_ambe_neutron_multiplicity.py

Neutron-like-cluster multiplicity on AmBe DATA along the analysis pipeline:

  Stage A  pre-selection multiplicity per event
             - OPTICS pre-selection (our Stage-1 clusters; passes_stage1 flag)
             - ClusterFinder + (PE<60, CB<0.5, hits>10)   [from the counting CSV]
           overlaid, log-y — the analogue of the legacy CF neutron-multiplicity plot.

  Stage B  neutron multiplicity per event AFTER the frozen-MVA cut on top of the
           OPTICS clusters (count clusters with gbt_score >= --score-cut per event).

Plus transfer diagnostics:
  - data MVA score distribution (rf, gbt)
  - score vs d_source (real AmBe neutrons should cluster near the source)
  - optional MC-vs-data feature overlap if --mc-features is given.

Inputs
------
  --scored     <run>__data_features__scored.parquet  (from mva_analysis.py --score-data)
               must carry run/event_tank_time + gbt_score/rf_score + passes_stage1.
  --stats-csv  optics_beamcluster_stats.csv (optional) for the CF Stage-A comparison
               (columns n_cf_presel / n_optics_presel per event).
  --mc-features <cc_neutrino__cluster_features.parquet> (optional) for feature overlap.

Usage
-----
  source /exp/annie/app/users/dajana/myboy/bin/activate
  python plot_ambe_neutron_multiplicity.py \
      --scored .../ambe_4499__data_features__scored.parquet \
      --score-cut 0.5 --out .../ambe_4499__neutron_multiplicity.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

EVENT_KEYS = ["run", "event_tank_time"]   # unique per AmBe event


def _multiplicity_hist(counts: np.ndarray, max_n: int = 8):
    """Return (bin_centers 1..max_n, counts-per-multiplicity) for a log-y bar plot."""
    centers = np.arange(1, max_n + 1)
    h = np.array([(counts == n).sum() for n in centers], dtype=float)
    return centers, h


def stage_a_page(pdf, scored: pd.DataFrame, stats_csv: Path | None,
                 run_label: str, max_n: int):
    """
    Stage A: OUR pre-selection is purely the OPTICS clustering — EVERY OPTICS
    cluster counts (no PE / charge-balance / nHits cut applied). Multiplicity =
    number of OPTICS clusters per event.

    The ClusterFinder + (PE<60, CB<0.5, hits>10) bars are shown ONLY as the legacy
    reference method for comparison; they are NOT our selection. The passes_stage1
    column in the parquet is a stored side flag and is deliberately NOT used here.
    """
    # OPTICS pre-selection multiplicity = # OPTICS clusters per event (ALL clusters).
    opt_counts = scored.groupby(EVENT_KEYS).size().to_numpy()
    centers, opt_h = _multiplicity_hist(opt_counts, max_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.4
    ax.bar(centers - w/2, opt_h, width=w, color="#7fc7ff", edgecolor="navy",
           label=f"OPTICS pre-selection — all OPTICS clusters (Σ={int(opt_h.sum())} ev)")

    if stats_csv and Path(stats_csv).exists():
        st = pd.read_csv(stats_csv)
        if "n_cf_presel" in st.columns:
            cf_counts = st["n_cf_presel"].dropna().to_numpy()
            _, cf_h = _multiplicity_hist(cf_counts, max_n)
            ax.bar(centers + w/2, cf_h, width=w, color="#ffb27f", edgecolor="darkred",
                   label=f"[ref] ClusterFinder PE<60/CB<0.5/hits>10 (Σ={int(cf_h.sum())} ev)")

    ax.set_yscale("log")
    ax.set_xlabel("Cluster multiplicity / event")
    ax.set_ylabel("Counts")
    ax.set_title(f"Stage A — OPTICS Pre-Selection\n{run_label}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(centers)
    ax.legend(fontsize=8)
    fig.text(0.5, 0.005, "Pure OPTICS clustering, no PE/CB/nHits cuts applied",
             ha="center", fontsize=8, color="0.4")
    pdf.savefig(fig); plt.close(fig)


def optics_pecbhits_page(pdf, scored: pd.DataFrame, run_label: str, max_n: int):
    """
    Comparison: what do the legacy PE/CB/nHits cuts do to the OPTICS clusters?
    Two histograms — all OPTICS clusters/event vs OPTICS clusters that ALSO pass
    PE<60/CB<0.5/hits>10 (the passes_stage1 flag). This is the alternative to the
    MVA: applying the old cuts directly to OPTICS clusters instead of a model.
    """
    all_keys = scored.groupby(EVENT_KEYS).size().index
    all_mult = scored.groupby(EVENT_KEYS).size().reindex(all_keys, fill_value=0).to_numpy()
    pass_mult = (scored[scored["passes_stage1"]].groupby(EVENT_KEYS).size()
                 .reindex(all_keys, fill_value=0)).to_numpy()
    centers, h_all = _multiplicity_hist(all_mult, max_n)
    _, h_pass = _multiplicity_hist(pass_mult, max_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.4
    ax.bar(centers - w/2, h_all, width=w, color="#7fc7ff", edgecolor="navy",
           label=f"all OPTICS clusters (Σ={int(h_all.sum())} ev)")
    ax.bar(centers + w/2, h_pass, width=w, color="#1f6fb2", edgecolor="black",
           label=f"OPTICS + PE<60/CB<0.5/hits>10 (Σ={int(h_pass.sum())} ev)")
    ax.set_yscale("log"); ax.set_xticks(centers)
    ax.set_xlabel("Cluster multiplicity / event"); ax.set_ylabel("Counts")
    ax.set_title(f"OPTICS Clusters — Effect of Legacy Cuts\n{run_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(0.5, 0.005, "Reference only: legacy PE<60 / CB<0.5 / nHits>10 applied to OPTICS clusters (the MVA replaces these)",
             ha="center", fontsize=8, color="0.4")
    pdf.savefig(fig); plt.close(fig)


def cf_pecbhits_page(pdf, stats_csv: Path | None, run_label: str, max_n: int):
    """
    Comparison: what do PE/CB/nHits cuts do to ClusterFinder clusters?
    Raw CF multiplicity vs CF + (PE<60/CB<0.5/hits>10), from the counting CSV
    (n_cf_raw, n_cf_presel per event).
    """
    if not (stats_csv and Path(stats_csv).exists()):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "CF comparison unavailable\n(pass --stats-csv optics_beamcluster_stats.csv)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off"); pdf.savefig(fig); plt.close(fig); return
    st = pd.read_csv(stats_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.4
    if "n_cf_raw" in st.columns:
        _, h_raw = _multiplicity_hist(st["n_cf_raw"].dropna().to_numpy(), max_n)
        ax.bar(np.arange(1, max_n+1) - w/2, h_raw, width=w, color="#ffd9a0",
               edgecolor="peru", label=f"all CF clusters (Σ={int(h_raw.sum())} ev)")
    if "n_cf_presel" in st.columns:
        _, h_pre = _multiplicity_hist(st["n_cf_presel"].dropna().to_numpy(), max_n)
        ax.bar(np.arange(1, max_n+1) + w/2, h_pre, width=w, color="#d2691e",
               edgecolor="black", label=f"CF + PE<60/CB<0.5/hits>10 (Σ={int(h_pre.sum())} ev)")
    ax.set_yscale("log"); ax.set_xticks(np.arange(1, max_n+1))
    ax.set_xlabel("Cluster multiplicity / event"); ax.set_ylabel("Counts")
    ax.set_title(f"ClusterFinder Clusters — Effect of Legacy Cuts\n{run_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(0.5, 0.005, "Reference only: legacy PE<60 / CB<0.5 / nHits>10 applied to ClusterFinder clusters",
             ha="center", fontsize=8, color="0.4")
    pdf.savefig(fig); plt.close(fig)


def stage_b_page(pdf, scored: pd.DataFrame, score_col: str, score_cut: float,
                 run_label: str, max_n: int):
    """Neutron multiplicity per event after the frozen-MVA cut."""
    sel = scored[scored[score_col] >= score_cut]
    per_event_all = scored.groupby(EVENT_KEYS).size().index
    mult = (sel.groupby(EVENT_KEYS).size().reindex(per_event_all, fill_value=0)).to_numpy()
    centers, h = _multiplicity_hist(mult, max_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(centers, h, width=0.7, color="#9be8a0", edgecolor="darkgreen")
    ax.set_yscale("log")
    ax.set_xlabel("Neutron multiplicity / event  (MVA-selected)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Stage B — Neutron Multiplicity after MVA\n{run_label}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(centers)
    n_ev = int((mult >= 1).sum())
    ax.text(0.97, 0.95, f"events with ≥1 neutron: {n_ev}\n"
                        f"clusters selected: {len(sel)}/{len(scored)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.text(0.5, 0.005, f"Frozen MC-trained MVA · {score_col} ≥ {score_cut:.2f}",
             ha="center", fontsize=8, color="0.4")
    pdf.savefig(fig); plt.close(fig)


def score_dist_page(pdf, scored: pd.DataFrame, run_label: str):
    cols = [c for c in ["rf_score", "gbt_score"] if c in scored.columns]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in cols:
        ax.hist(scored[c], bins=50, range=(0, 1), histtype="step", linewidth=1.6,
                label=f"{c} (median {scored[c].median():.2f})")
    ax.set_xlabel("MVA neutron score"); ax.set_ylabel("Clusters")
    ax.set_title(f"Data MVA Score Distribution\n{run_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    pdf.savefig(fig); plt.close(fig)


def score_vs_dsource_page(pdf, scored: pd.DataFrame, score_col: str, run_label: str):
    if "d_source" not in scored.columns or scored["d_source"].notna().sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    s = scored.dropna(subset=["d_source", score_col])
    # Clip rare centroid-blowup outliers (degenerate single-PMT clusters) so the
    # axis reflects the physical bulk; report how many were clipped.
    dmax = float(np.nanpercentile(s["d_source"], 99.5))
    n_clip = int((s["d_source"] > dmax).sum())
    s = s[s["d_source"] <= dmax]
    hb = ax.hexbin(s["d_source"], s[score_col], gridsize=40, cmap="viridis", mincnt=1)
    if n_clip:
        ax.text(0.97, 0.03, f"{n_clip} outlier(s) >{dmax:.1f} m clipped",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="gray")
    fig.colorbar(hb, ax=ax, label="clusters")
    ax.set_xlabel("Distance from cluster vertex to AmBe source (m)")
    ax.set_ylabel(f"{score_col}")
    ax.set_title(f"MVA Score vs Distance to Source\n{run_label}",
                 fontsize=13, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)


def feature_overlap_page(pdf, scored: pd.DataFrame, mc_path: Path, run_label: str):
    mc = pd.read_parquet(mc_path)
    if "cc_pass" in mc.columns:
        mc = mc[mc["cc_pass"] == 1]
    feats = ["pe_total", "n_hits", "d_wall", "beta1", "sigma_t_mad_tof", "spatial_rms"]
    feats = [f for f in feats if f in scored.columns and f in mc.columns]
    ncol = 3; nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.2*nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(feats):
        ax = axes[i]
        d = scored[f].dropna().to_numpy(); m = mc[f].dropna().to_numpy()
        lo = np.nanpercentile(np.concatenate([d, m]), 1)
        hi = np.nanpercentile(np.concatenate([d, m]), 99)
        rng = (lo, hi) if hi > lo else None
        ax.hist(m, bins=40, range=rng, density=True, histtype="step",
                color="navy", label="MC (train)")
        ax.hist(d, bins=40, range=rng, density=True, histtype="step",
                color="crimson", label="AmBe data")
        ax.set_title(f, fontsize=9); ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"MC-vs-Data Feature Overlap (Domain Check)\n{run_label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True, help="scored data features parquet")
    p.add_argument("--stats-csv", default=None, help="optics_beamcluster_stats.csv (CF comparison)")
    p.add_argument("--mc-features", default=None, help="MC cluster_features parquet (overlap)")
    p.add_argument("--score-col", default="gbt_score", choices=["gbt_score", "rf_score"])
    p.add_argument("--score-cut", type=float, default=0.5)
    p.add_argument("--max-n", type=int, default=8, help="max multiplicity bin")
    p.add_argument("--out", required=True, help="output PDF")
    args = p.parse_args()

    scored = pd.read_parquet(args.scored)
    run_label = f"AmBe {sorted(scored['run'].unique().tolist())}" if "run" in scored else "AmBe data"
    print(f"[plot] {len(scored)} clusters | events={scored.groupby(EVENT_KEYS).ngroups} "
          f"| {args.score_col}≥{args.score_cut}")

    with PdfPages(args.out) as pdf:
        stage_a_page(pdf, scored, args.stats_csv, run_label, args.max_n)
        # Comparison: legacy PE/CB/nHits cuts applied AFTER Stage A, on each method.
        optics_pecbhits_page(pdf, scored, run_label, args.max_n)
        cf_pecbhits_page(pdf, args.stats_csv, run_label, args.max_n)
        stage_b_page(pdf, scored, args.score_col, args.score_cut, run_label, args.max_n)
        score_dist_page(pdf, scored, run_label)
        score_vs_dsource_page(pdf, scored, args.score_col, run_label)
        if args.mc_features and Path(args.mc_features).exists():
            feature_overlap_page(pdf, scored, Path(args.mc_features), run_label)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
