"""
Data / MC feature comparison for ANNIE AmBe neutron analysis.

Reads real-data cluster CSVs (EventAmBeNeutronCandidatesData/) and MC feature
parquets (ambe_output/<run>/parquet/*__cluster_features.parquet), computes the
same cluster-level features for both, then:

  1. Produces overlay normalised histograms (data vs MC) for each feature
     saved to data_mc_comparison_<run_name>.pdf

  2. Runs a Kolmogorov–Smirnov test per feature; outputs a summary CSV
     data_mc_comparison_<run_name>__ks.csv with (feature, ks_stat, p_value).
     Features with p < 0.01 are flagged as "mis-modelled" and need
     systematic uncertainty assigned before applying the MC-trained BDT to data.

  3. Prints a per-source-port efficiency table: observed candidates per
     event in data vs MC prediction; ratio = relative data efficiency.

Usage
-----
python analyze_data_mc_comparison.py \\
    --data-dir  EventAmBeNeutronCandidatesData/ \\
    --mc-parquet /Users/dajana/Documents/ambe_output/mc_local_trial/parquet/ \\
    --run-name  mc_local_trial \\
    --mc-method optics \\
    --output-dir /Users/dajana/Documents/ambe_output/

Arguments (also configurable at the top of this script if not using CLI):
  --data-dir    Folder containing EventAmBeNeutronCandidates_*.csv files
  --mc-parquet  Folder containing <run_name>__cluster_features.parquet
  --run-name    MC run name (used to locate the parquet and label plots)
  --mc-method   "optics" or "clusterfinder" (default: optics)
  --output-dir  Where to write the PDF and CSV (default: same as mc-parquet)

--- Unit note ---
Hit positions in data CSVs (hitX/Y/Z) are in METRES.
Source positions in CSVs (sourceX/Y/Z) are in CM → divide by 100 when
passing to compute_cluster_features().

--- Confidence level interpretation ---
After running:
  • Features with KS p > 0.05 : data and MC agree → apply BDT score to data,
    quote efficiency ± stat uncertainty only.
  • Features with KS p < 0.05 : MC may mis-model detector response for that
    feature → assign systematic from the mean shift between data and MC
    distributions.  Propagate via ±1σ shift in that feature's distribution
    before re-evaluating BDT score threshold.
  • Overall data efficiency = (n_candidates_data / n_events_data) /
    (MC expected per event at same BDT cut).
  • Confidence level on efficiency difference OPTICS vs CF: use binomial
    proportion test on n_matched_OPTICS vs n_matched_CF over same events.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ks_2samp

# Make the package importable without full install
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ambe.mc.cluster_features import (
    ANNIEGeometry,
    FEATURE_LABELS,
    compute_cluster_features,
    load_geometry,
)

# --------------------------------------------------------------------------- #
# Default paths (edit here or pass via CLI)
# --------------------------------------------------------------------------- #
DEFAULT_DATA_DIR   = "EventAmBeNeutronCandidatesData"
DEFAULT_MC_PARQUET = "/Users/dajana/Documents/ambe_output/mc_local_trial/parquet"
DEFAULT_RUN_NAME   = "mc_local_trial"
DEFAULT_GEO        = "/Users/dajana/Documents/ToolAnalysis/configfiles/LoadGeometry/FullTankPMTGeometry.csv"
DEFAULT_OFFSETS    = "/Users/dajana/Documents/ToolAnalysis/configfiles/LoadGeometry/TankPMTTimingOffsets.csv"

# KS p-value threshold: below this → flag feature as mis-modelled
KS_THRESHOLD = 0.05

# Source position unit note: data CSVs store sourceX/Y/Z in cm, hits in m.
SOURCE_POS_UNIT_FACTOR = 1.0 / 100.0   # cm → m

# --------------------------------------------------------------------------- #
# Data CSV parsing
# --------------------------------------------------------------------------- #

def _parse_float_list(s: str) -> List[float]:
    """Parse '[1.0, 2.5, ...]' or space-separated string → list of floats."""
    s = s.strip()
    if s.startswith("["):
        try:
            return [float(x) for x in ast.literal_eval(s)]
        except Exception:
            pass
    # Fallback: strip brackets and split
    s = re.sub(r"[\[\](){}]", "", s)
    parts = re.split(r"[,\s]+", s.strip())
    return [float(p) for p in parts if p]


def load_data_csvs(data_dir: str,
                   geo: ANNIEGeometry,
                   glob_pattern: str = "EventAmBeNeutronCandidates_*.csv",
                   verbose: bool = True) -> pd.DataFrame:
    """
    Load all data candidate CSVs, parse per-hit arrays, compute cluster features.

    Returns a DataFrame with one row per cluster (same feature columns as MC).

    Important: sourceX/Y/Z in CSV are in cm; hits are in metres.
    We convert source to metres before passing to compute_cluster_features.
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob(glob_pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{glob_pattern}' found in {data_dir}")

    rows = []
    total_clusters = 0
    skipped = 0

    for csv_file in csv_files:
        # Extract run number from filename
        m = re.search(r"_(\d+)\.csv$", csv_file.name)
        run_num = int(m.group(1)) if m else -1

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  WARNING: could not read {csv_file.name}: {e}")
            continue

        required = {"hitX", "hitY", "hitZ", "hitT", "hitPE",
                    "sourceX", "sourceY", "sourceZ", "clusterHits"}
        if not required.issubset(df.columns):
            print(f"  WARNING: {csv_file.name} missing columns: "
                  f"{required - set(df.columns)}")
            continue

        for idx, row in df.iterrows():
            try:
                hx  = _parse_float_list(str(row["hitX"]))
                hy  = _parse_float_list(str(row["hitY"]))
                hz  = _parse_float_list(str(row["hitZ"]))
                ht  = _parse_float_list(str(row["hitT"]))
                hpe = _parse_float_list(str(row["hitPE"]))
                n   = len(ht)
                if n < 2:
                    skipped += 1
                    continue

                # Determine PMT IDs from hitID if available, else sequential
                if "hitID" in df.columns:
                    hid_raw = str(row["hitID"])
                    try:
                        hid = _parse_float_list(hid_raw)
                    except Exception:
                        hid = list(range(n))
                else:
                    hid = list(range(n))

                # Source position: CSV stores in cm, hits in metres → convert
                src_x = float(row.get("sourceX", 0)) * SOURCE_POS_UNIT_FACTOR
                src_y = float(row.get("sourceY", 0)) * SOURCE_POS_UNIT_FACTOR
                src_z = float(row.get("sourceZ", 0)) * SOURCE_POS_UNIT_FACTOR

                # Pad/truncate arrays to consistent length
                hx  = hx[:n];  hy  = hy[:n];  hz  = hz[:n]
                hpe = hpe[:n]; hid = (hid[:n] if len(hid) >= n
                                      else hid + [0] * (n - len(hid)))

                df_cluster = pd.DataFrame({
                    "pmtID": [int(round(i)) for i in hid],
                    "x":     hx, "y": hy, "z": hz,
                    "t":     ht, "pe": hpe,
                })

                feats = compute_cluster_features(df_cluster, geo)
                feats["run"] = run_num
                feats["source_x_m"] = src_x
                feats["source_y_m"] = src_y
                feats["source_z_m"] = src_z
                feats["clusterTime"] = float(row.get("clusterTime", np.nan))
                feats["clusterPE"]   = float(row.get("clusterPE",   np.nan))
                feats["source"] = f"data_run{run_num}"
                feats["dataset"] = "data"

                # d_source: distance from estimated vertex to known source position
                vtx = np.array([feats["vtx_x"], feats["vtx_y"], feats["vtx_z"]])
                src = np.array([src_x, src_y, src_z])
                feats["d_source"] = float(np.linalg.norm(vtx - src))

                rows.append(feats)
                total_clusters += 1

            except Exception as e:
                skipped += 1
                if verbose:
                    print(f"  WARNING: row {idx} in {csv_file.name}: {e}")
                continue

    if verbose:
        print(f"[data] loaded {total_clusters} clusters from "
              f"{len(csv_files)} files  ({skipped} skipped)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MC parquet loading
# --------------------------------------------------------------------------- #

def load_mc_features(parquet_dir: str, run_name: str,
                     method: str = "optics") -> pd.DataFrame:
    """
    Load MC cluster-feature parquet for the given run and method.
    Adds 'd_source' if vtx and source position columns are present.
    """
    pq_path = Path(parquet_dir)
    # Try exact filename first
    candidate = pq_path / f"{run_name}__cluster_features.parquet"
    if not candidate.exists():
        # Fall back to glob
        matches = list(pq_path.glob("*cluster_features*.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"No cluster_features parquet found in {parquet_dir}")
        candidate = matches[0]
        print(f"[mc] using parquet: {candidate.name}")

    df = pd.read_parquet(candidate)
    df = df[df["method"] == method].copy()
    df["dataset"] = "mc"

    # Compute d_source if not already present
    if "d_source" not in df.columns:
        if all(c in df.columns for c in ["vtx_x", "vtx_y", "vtx_z"]):
            # For MC, source is at origin unless specified otherwise
            df["d_source"] = np.sqrt(
                df["vtx_x"]**2 + df["vtx_y"]**2 + df["vtx_z"]**2)

    print(f"[mc] loaded {len(df)} clusters  "
          f"(truth_neutron={df['is_truth_neutron'].sum() if 'is_truth_neutron' in df.columns else 'N/A'})")
    return df


# --------------------------------------------------------------------------- #
# KS test per feature
# --------------------------------------------------------------------------- #

def run_ks_tests(df_data: pd.DataFrame,
                 df_mc: pd.DataFrame,
                 features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Run a 2-sample Kolmogorov–Smirnov test for each feature.
    Returns a DataFrame with columns: feature, ks_stat, p_value, flag.
    """
    if features is None:
        features = [f for f in FEATURE_LABELS if f in df_data.columns
                    and f in df_mc.columns]
        # Also test the new features
        extra = ["sigma_t_mad", "sigma_t_mad_corr", "n_hits_early",
                 "sigma_t_early_mad", "t_window_80pct", "d_source"]
        features += [f for f in extra if f in df_data.columns
                     and f in df_mc.columns and f not in features]

    rows = []
    for feat in features:
        d_vals = df_data[feat].dropna().to_numpy()
        m_vals = df_mc[feat].dropna().to_numpy()
        if len(d_vals) < 5 or len(m_vals) < 5:
            continue
        stat, p = ks_2samp(d_vals, m_vals)
        rows.append({
            "feature":  feat,
            "ks_stat":  round(stat, 4),
            "p_value":  round(p, 6),
            "flag":     "MIS-MODELLED" if p < KS_THRESHOLD else "OK",
            "n_data":   len(d_vals),
            "n_mc":     len(m_vals),
            "mean_data": round(float(np.mean(d_vals)), 4),
            "mean_mc":   round(float(np.mean(m_vals)), 4),
            "mean_shift": round(float(np.mean(d_vals) - np.mean(m_vals)), 4),
        })

    return pd.DataFrame(rows).sort_values("p_value")


# --------------------------------------------------------------------------- #
# Overlay plots
# --------------------------------------------------------------------------- #

def make_comparison_plots(df_data: pd.DataFrame,
                          df_mc: pd.DataFrame,
                          ks_df: pd.DataFrame,
                          output_pdf: str,
                          run_name: str) -> None:
    """
    Produce one page per feature: normalised histograms data (black) vs MC
    signal (red) and MC background (blue).  KS p-value shown on each plot.
    """
    all_features = list(FEATURE_LABELS.keys()) + [
        "sigma_t_mad", "sigma_t_mad_corr", "sigma_t_mad_tof",
        "n_hits_early", "sigma_t_early_mad", "t_window_80pct", "d_source",
    ]
    plot_features = [f for f in all_features
                     if f in df_data.columns and f in df_mc.columns]

    ks_lookup = (ks_df.set_index("feature")[["ks_stat", "p_value", "flag"]]
                 if len(ks_df) else pd.DataFrame())

    mc_has_truth = "is_truth_neutron" in df_mc.columns

    with PdfPages(output_pdf) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        title_text = (f"Data / MC Comparison — {run_name}\n"
                      f"Data clusters: {len(df_data)}  |  MC clusters: {len(df_mc)}")
        ax.text(0.5, 0.9, title_text, ha="center", va="top",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
        if len(ks_df):
            n_ok   = (ks_df["flag"] == "OK").sum()
            n_bad  = (ks_df["flag"] == "MIS-MODELLED").sum()
            summary = (f"KS test summary: {n_ok} features OK  |  "
                       f"{n_bad} features mis-modelled (p < {KS_THRESHOLD})\n\n")
            summary += ks_df[["feature","ks_stat","p_value","flag",
                               "mean_data","mean_mc","mean_shift"]].to_string(index=False)
            ax.text(0.05, 0.75, summary, ha="left", va="top",
                    fontsize=8, fontfamily="monospace", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # One page per feature
        for feat in plot_features:
            feat_label = FEATURE_LABELS.get(feat, feat)
            d_vals = df_data[feat].dropna().to_numpy()
            if mc_has_truth:
                mc_sig = df_mc[df_mc["is_truth_neutron"] == 1][feat].dropna().to_numpy()
                mc_bkg = df_mc[df_mc["is_truth_neutron"] == 0][feat].dropna().to_numpy()
            else:
                mc_sig = df_mc[feat].dropna().to_numpy()
                mc_bkg = np.array([])

            if len(d_vals) < 3:
                continue

            all_vals = np.concatenate([d_vals, mc_sig,
                                        mc_bkg if len(mc_bkg) else np.array([])])
            lo = float(np.nanpercentile(all_vals, 1))
            hi = float(np.nanpercentile(all_vals, 99))
            if lo == hi:
                lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals)) + 1e-6
            bins = np.linspace(lo, hi, 40)

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.hist(d_vals,  bins=bins, density=True, histtype="step",
                    color="black",    linewidth=1.8, label=f"Data  (n={len(d_vals)})")
            if len(mc_sig) > 0:
                ax.hist(mc_sig,  bins=bins, density=True, alpha=0.5,
                        color="tomato",   label=f"MC signal  (n={len(mc_sig)})")
            if len(mc_bkg) > 0:
                ax.hist(mc_bkg,  bins=bins, density=True, alpha=0.4,
                        color="steelblue",label=f"MC bkg  (n={len(mc_bkg)})")

            ax.set_xlabel(feat_label, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"{feat}", fontsize=10)
            ax.legend(fontsize=8)

            # KS annotation
            if feat in ks_lookup.index:
                ks_row = ks_lookup.loc[feat]
                color = "red" if ks_row["flag"] == "MIS-MODELLED" else "darkgreen"
                ax.text(0.97, 0.95,
                        f"KS={ks_row['ks_stat']:.3f}  p={ks_row['p_value']:.4f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[output] plots → {output_pdf}")


# --------------------------------------------------------------------------- #
# Efficiency table
# --------------------------------------------------------------------------- #

def print_efficiency_table(df_data: pd.DataFrame,
                            df_mc: pd.DataFrame) -> None:
    """
    Print per-source-port summary:
      data: n_clusters / n_events (raw rate)
      MC:   n_matched / n_events (recall × truth rate)
    """
    print("\n" + "=" * 60)
    print("EFFICIENCY / CLUSTER RATE SUMMARY")
    print("=" * 60)

    # Data: group by source position
    if "run" in df_data.columns:
        print(f"\nData: {len(df_data)} total clusters from "
              f"{df_data['run'].nunique()} runs")
        if "source_x_m" in df_data.columns:
            grp = (df_data.groupby(["source_x_m", "source_y_m", "source_z_m"])
                          .agg(n_clusters=("n_hits", "count"))
                          .reset_index())
            print("\nData clusters by source position:")
            print(grp.to_string(index=False))

    # MC summary
    if "is_truth_neutron" in df_mc.columns:
        n_sig = int(df_mc["is_truth_neutron"].sum())
        n_tot = len(df_mc)
        print(f"\nMC: {n_tot} clusters  "
              f"({n_sig} truth-neutron, {n_tot-n_sig} spurious)")
        print(f"  MC purity: {n_sig/n_tot:.1%}  "
              f"(fraction of MC clusters that are true neutron captures)")

    print()
    print("Confidence level notes:")
    print("  • For BDT score application: requires KS p > 0.05 on MVA input features.")
    print("  • Efficiency = (data rate × MC efficiency) / MC rate.")
    print("  • Systematic uncertainty from feature shifts:")
    print("    propagate by varying feature means by ±(data_mean - MC_mean).")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Data / MC comparison for ANNIE AmBe neutron analysis")
    parser.add_argument("--data-dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--mc-parquet", default=DEFAULT_MC_PARQUET)
    parser.add_argument("--run-name",   default=DEFAULT_RUN_NAME)
    parser.add_argument("--mc-method",  default="optics",
                        choices=["optics", "clusterfinder"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--geo",        default=DEFAULT_GEO)
    parser.add_argument("--offsets",    default=DEFAULT_OFFSETS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.mc_parquet).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load geometry
    print("[setup] loading geometry...")
    try:
        geo = load_geometry(args.geo, args.offsets)
    except Exception as e:
        print(f"WARNING: could not load geometry ({e}). "
              f"Features requiring offsets will use fallback.")
        # Build minimal geometry
        from ambe.mc.cluster_features import ANNIEGeometry
        geo = ANNIEGeometry(
            pmt_pos={}, pmt_dir={}, pmt_offset={},
            tank_radius=1.12, tank_top=1.5, tank_bot=-1.5,
            tank_center=np.array([0., 0., 0.]), fallback_offset=10.0)

    # Load data
    print(f"[data] loading CSVs from {args.data_dir} ...")
    df_data = load_data_csvs(args.data_dir, geo)
    if len(df_data) == 0:
        print("ERROR: no data clusters loaded. Check data_dir path.")
        sys.exit(1)

    # Load MC
    print(f"[mc] loading parquet from {args.mc_parquet} ...")
    try:
        df_mc = load_mc_features(args.mc_parquet, args.run_name, args.mc_method)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("MC features not found — will output data-only plots.")
        df_mc = pd.DataFrame()

    # KS tests
    ks_df = pd.DataFrame()
    if len(df_mc) > 0:
        print("\n[ks] running KS tests...")
        ks_df = run_ks_tests(df_data, df_mc)
        ks_path = output_dir / f"data_mc_comparison_{args.run_name}__ks.csv"
        ks_df.to_csv(ks_path, index=False)
        print(f"[output] KS table → {ks_path}")
        print("\nKS TEST RESULTS (sorted by p-value):")
        print(ks_df[["feature", "ks_stat", "p_value", "flag",
                      "mean_data", "mean_mc", "mean_shift"]].to_string(index=False))

    # Plots
    pdf_path = str(output_dir / f"data_mc_comparison_{args.run_name}.pdf")
    make_comparison_plots(df_data, df_mc if len(df_mc) else df_data,
                          ks_df, pdf_path, args.run_name)

    # Efficiency summary
    print_efficiency_table(df_data, df_mc if len(df_mc) else pd.DataFrame())

    print(f"\nDone.")


if __name__ == "__main__":
    main()
