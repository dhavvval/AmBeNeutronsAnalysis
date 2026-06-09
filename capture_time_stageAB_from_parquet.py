"""
capture_time_stageAB_from_parquet.py

Capture-time + thermal-time fits for the Stage-A (OPTICS) + Stage-B (MVA) selected
neutron clusters, using YOUR existing fitters in src/ambe/plots/basic.py
(AmBeNeutronAnalyzer.lmfit_analysis and .pymc_analysis) UNCHANGED.

Why an adapter: AmBeNeutronAnalyzer is built to read the EventAmBeNeutronCandidates
CSVs and fit data_dict['CT'] (cluster time, us) per source position.  The Stage-A+B
output is the scored feature parquet (one row per OPTICS cluster, with t_mean [ns]
and rf_score).  Both lmfit_analysis and pymc_analysis consume ONLY data_dict['CT']
and source_key, so this script:
  1. loads the scored parquet, applies the Stage-B MVA cut (rf_score >= cut),
  2. maps each cluster's run -> source (x,y,z) via AmBeNeutronProcessing,
  3. for each distinct source position builds data_dict = {'CT': t_mean/1000 (us)},
  4. calls analyzer.lmfit_analysis(...) and analyzer.pymc_analysis(...) per position,
  5. calls analyzer.generate_summary_plots() for the therm/tau heatmaps (Y vs Port).

Capture time here = the OPTICS-cluster time (t_mean) of MVA-selected clusters, the
Stage-A+B object — not the raw CF clusterTime.

Usage:
  python capture_time_stageAB_from_parquet.py \
      --scored .../ambe_all__data_features__scored.parquet \
      --score-col rf_score --score-cut 0.643 \
      --out-dir .../capture_stageAB
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import sys
# scipy>=1.13 removed scipy.signal.gaussian (moved to scipy.signal.windows). arviz
# (pulled in by `import pymc` at the top of ambe.plots.basic) still imports the old
# name, which otherwise breaks the whole basic.py import. Re-expose it before that.
import scipy.signal
import scipy.signal.windows as _scipy_windows
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = _scipy_windows.gaussian

sys.path.insert(0, str(Path(__file__).parent / "src"))
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
    # Per user request: lower the tau PARAMETER bound to (2, 70) us to match the
    # fit data window (fit_min_time=2, fit_max_time=67). Note: this lets the
    # capture-time parameter range down into the thermalization region, where tau
    # and therm can become degenerate on thin per-position stats.
    analyzer.update_fitting_config(capture_bounds=(2.0, 70.0))

    # lmfit_analysis hardcodes method="basinhopping", which does NOT populate
    # parameter stderr -> Tau_err/Thermal_err come back None. Force leastsq (via a
    # lmfit.Model.fit shim) so we get covariance-based per-position uncertainties.
    # Does not touch basic.py; only overrides the optimizer for the .fit() call.
    if "lmfit" in args.methods:
        import lmfit
        _orig_fit = lmfit.Model.fit
        def _fit_leastsq(self, *a, **kw):
            kw["method"] = "leastsq"
            return _orig_fit(self, *a, **kw)
        lmfit.Model.fit = _fit_leastsq

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
