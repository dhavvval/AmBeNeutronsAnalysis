"""Fit scored OPTICS cluster times with the shared basic plotting analyzer.

Run from the repository root: python -m ambe.diagnostics.capture_time_stageAB_from_parquet
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ambe.plots.basic import AmBeNeutronAnalyzer
from ambe.data.processor import AmBeNeutronProcessing

SOURCE_POSITIONS = AmBeNeutronProcessing().source_positions   # run -> (x,y,z) cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--score-col", default="rf_score")
    p.add_argument("--score-cut", type=float, default=0.643)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--methods", nargs="+", default=["lmfit", "pymc"],
                   choices=["lmfit", "pymc", "scipy"])
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    d = pd.read_parquet(args.scored)
    sel = d[d[args.score_col] >= args.score_cut].copy()
    # run -> source position (cm); drop runs with no known position
    pos = sel["run"].map(lambda r: SOURCE_POSITIONS.get(int(r)))
    sel = sel[pos.notna()].copy()
    sel["src"] = pos[pos.notna()].map(lambda t: (round(float(t[0]), 3),
                                                 round(float(t[1]), 3),
                                                 round(float(t[2]), 3)))
    # CT in microseconds (analyzer convention), from OPTICS-cluster t_mean [ns]
    sel["CT_us"] = sel["t_mean"] / 1000.0

    print(f"Stage-A+B clusters: {len(sel):,} over {sel['src'].nunique()} source positions "
          f"({args.score_col} >= {args.score_cut})")

    analyzer = AmBeNeutronAnalyzer(
        data_directory="./EventAmBeNeutronCandidatesData/",
        output_pdf=str(out / "stageAB_capture.pdf"))
    # Use the shared fit window, optimizer and covariance convention.

    method_pdf = {m: PdfPages(out / f"stageAB_capture_{m}.pdf") for m in args.methods}

    # Fit each source position with the chosen method(s), reusing your fitters.
    for src, g in sorted(sel.groupby("src"), key=lambda kv: kv[0]):
        data_dict = {"CT": g["CT_us"], "EventTime": g["event_tank_time"].value_counts()}
        print(f"\n=== position {src}  ({len(g)} clusters) ===")
        if "lmfit" in args.methods:
            analyzer.lmfit_analysis(data_dict, src, method_pdf["lmfit"])
        if "scipy" in args.methods:
            analyzer.scipy_curve_fit(data_dict, src, method_pdf["scipy"])
        if "pymc" in args.methods:
            analyzer.pymc_analysis(data_dict, src, method_pdf["pymc"])

    for m, pdf in method_pdf.items():
        pdf.close()
        print(f"[{m}] -> {out / f'stageAB_capture_{m}.pdf'}")

    # therm + capture-time heatmaps (Y vs Port) from the lmfit results.
    if "lmfit" in args.methods and analyzer.lmfit_summary:
        analyzer.generate_summary_plots()
        # dump the per-position table too
        tab = pd.DataFrame(analyzer.lmfit_summary)
        tab.to_csv(out / "stageAB_capture_lmfit_summary.csv", index=False)
        print(f"[lmfit] summary table -> {out / 'stageAB_capture_lmfit_summary.csv'}")
        print(tab[["Coordination", "Thermal", "Thermal_err", "Tau", "Tau_err",
                   "reduced_chi2"]].to_string(index=False))


if __name__ == "__main__":
    main()
