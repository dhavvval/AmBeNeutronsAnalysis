"""
ccinc_v3_ambe_closure.py
========================
Stage 4 of the CCinc v3 campaign: does any of the four merged tank+world models
actually work on real AmBe data?

WHAT THIS CAN AND CANNOT MEASURE — read before quoting anything below.

AmBe data has no truth labels, so there is no efficiency and no purity here. Two
things can be measured, and only two:

  1. **Data/MC agreement.** Whether the score distribution the model produces on data
     resembles anything it produced on MC. A model whose data scores pile up outside
     the MC range has not transferred, whatever its AUC was.
  2. **Physics closure.** AmBe neutrons capture with a known lifetime. If a selection
     is really keeping neutron captures, the capture-time distribution of what it keeps
     must fit that lifetime. The anchor is **tau = 30.53 +- 0.26 us**
     (REPORT_fprompt_waveform_cut.md §7, 20 runs / 19 positions). The older 29.16 anchor
     is dead — the CSVs it was fitted on no longer exist.

Also: only the **cluster-level MVA** is under test. The CC event selection cannot be
applied to AmBe at all (no neutrino, no muon, no MRD track), and the reco-tag streamline
is not data-applicable even in principle — its FV and muon kinematics still read truth
branches. Reco-tag numbers here are a model-transfer check, not a validated selection.

Two deliberate design choices:

  * **Capture time comes from `t_mean`/1000 us, for both methods.** The OPTICS features
    parquet has `clusterTime_earliest` entirely NaN; the ClusterFinder one has it
    populated and it agrees with `t_mean` to about 5 ns — negligible against a 30 us
    lifetime. Using `t_mean` for both keeps the two methods on one axis.
  * **The fitter is imported, not rewritten.** `fit` / `fit_expflat` come from
    fit_capture_time_fprompt_compare.py, so the same binning, fit window (10-67 us) and
    functional form that produced the 30.53 anchor produce these numbers too. A tau
    fitted any other way would not be comparable to the anchor.

Thresholds are the MC working points from ccinc_v3_model_comparison.csv. They are MC
calibrations applied to data, so the realised data pass rate is reported next to the MC
signal efficiency — where the two diverge, the score calibration has not transferred and
the threshold, not the model, is what moved.

Run (no silent default):
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python ccinc_v3_ambe_closure.py --do all
    python ccinc_v3_ambe_closure.py --do agreement
    python ccinc_v3_ambe_closure.py --do closure
Output:
    ccinc_v3_ambe_agreement.csv
    ccinc_v3_ambe_closure.csv
    slide_plots_ccinc_v3_merged/V3AMBE__*.{pdf,png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from make_plots_ccinc_v3_merged import (
    BASE, MSFX, CONFIGS, CFG_LBL, OUTD, BLUE, GREY, ORANGE, GREEN,
    LINE_COLORS, bare, table_axes)
import ccinc_v3_stats as S
from fit_capture_time_fprompt_compare import (fit as fit_neutcapture, fit_expflat,
                                              BOUNDS_LO, BOUNDS_HI)

HERE = Path(__file__).parent
PREFIX = "V3AMBE__"
STAGE = BASE / "ccinc_v3_ambe_stage3"

TAU_ANCHOR, TAU_ANCHOR_ERR = 30.53, 0.26        # REPORT_fprompt_waveform_cut.md §7

SCORE_COLS = S.SCORE_COLS
SHORT = S.SHORT
# Working points to test on data. max_significance is excluded on purpose: on a 1:1
# balanced test set it always lands near full acceptance, so it is not a selection.
POINTS = ["eff50", "eff80", "eff90", "max_purity"]


def scored_path(stream: str, method: str) -> Path:
    m = MSFX[method]
    return STAGE / f"{stream}_{m}" / f"ambe_{stream}_{m}__scored.parquet"


def load_ambe(stream: str, method: str) -> pd.DataFrame:
    p = scored_path(stream, method)
    if not p.exists():
        sys.exit(f"[ambe] missing scored data: {p}\n"
                 f"       run: bash run_ccinc_v3_ambe_stage3.sh all")
    d = pd.read_parquet(p)
    d["t_us"] = pd.to_numeric(d["t_mean"], errors="coerce") / 1000.0
    d["_evt"] = d["run"].astype(str) + "#" + d["event_number"].astype(str)
    return d


# ════════════════════════════════════════════════════════════════════════════
# data/MC agreement
# ════════════════════════════════════════════════════════════════════════════
def agreement(mc: pd.DataFrame, data: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    KS distance from the AmBe score distribution to the MC signal and MC background
    score distributions, per model.

    AmBe is neutron-capture data, so a model that transferred should sit closer to the
    MC *signal* shape than to the MC background shape. `ks_to_bkg - ks_to_sig > 0` is
    that statement. The KS p-value is not quoted: with 10^5 clusters everything is
    "significantly different" and the number carries no information — the useful
    quantity is which side it is closer to, and by how much.
    """
    te = mc[mc["in_test"].astype(bool)]
    rows = []
    for col, model in SCORE_COLS.items():
        if col not in data.columns or col not in te.columns:
            continue
        a = data[col].dropna().to_numpy(float)
        sig = te.loc[te["y"] == 1, col].to_numpy(float)
        bkg = te.loc[te["y"] == 0, col].to_numpy(float)
        ks_s = ks_2samp(a, sig).statistic
        ks_b = ks_2samp(a, bkg).statistic
        rows.append(dict(
            config=CFG_LBL[cfg], streamline=cfg[0], method=cfg[1], model=model,
            model_short=SHORT[model], n_data=len(a),
            data_median=float(np.median(a)),
            mc_sig_median=float(np.median(sig)), mc_bkg_median=float(np.median(bkg)),
            data_min=float(a.min()), data_max=float(a.max()),
            mc_min=float(min(sig.min(), bkg.min())),
            mc_max=float(max(sig.max(), bkg.max())),
            ks_to_sig=ks_s, ks_to_bkg=ks_b, closer_to=("signal" if ks_s < ks_b
                                                       else "background"),
            in_mc_range=bool(a.min() >= min(sig.min(), bkg.min()) - 1e-9
                             and a.max() <= max(sig.max(), bkg.max()) + 1e-9)))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# physics closure
# ════════════════════════════════════════════════════════════════════════════
TAU_LO, TAU_HI = BOUNDS_LO["tau"], BOUNDS_HI["tau"]      # 10 and 70 us
MAX_REDCHI = 5.0
MAX_TAU_ERR = 10.0


def fit_quality(r: dict) -> tuple[bool, str]:
    """
    Whether a fitted tau means anything. Three ways it does not, all seen here:

      * **pinned at a bound.** curve_fit returns tau exactly 10.0 or 70.0 with a
        standard error of either 0.0 or something enormous. Both are non-fits. The 0.0
        case looks like an infinitely precise measurement and the huge-error case passes
        a 2-sigma test trivially — one NN max-purity point pinned at tau = 10.0 +- 166
        and was flagged "consistent" before this gate existed.
      * **bad chi2.** The OPTICS unselected sample fits with chi2/ndof = 134: the
        distribution is not a capture-time exponential at all, so no tau from it is a
        measurement of anything.
      * **useless precision.** tau +- 20 us is consistent with almost any anchor and
        excludes nothing.
    """
    if not np.isfinite(r["tau"]) or not np.isfinite(r["tau_err"]):
        return False, "non-finite"
    if abs(r["tau"] - TAU_LO) < 1e-6 or abs(r["tau"] - TAU_HI) < 1e-6:
        return False, "pinned at fit bound"
    if r["tau_err"] <= 0:
        return False, "zero error"
    if r["redchi"] > MAX_REDCHI:
        return False, f"chi2/ndof {r['redchi']:.1f}"
    if r["tau_err"] > MAX_TAU_ERR:
        return False, f"tau_err {r['tau_err']:.1f} us"
    return True, "ok"


def tau_row(t_us: np.ndarray, label: str, **extra) -> dict | None:
    """
    Both fit variants on one capture-time sample. None if too few clusters.

    `consistent` requires the fit to be usable AND within 2 sigma of the anchor — a
    failed fit is never "consistent", however small its nominal pull.
    """
    t = t_us[np.isfinite(t_us)]
    r = fit_neutcapture(t, float_B=True)
    if r is None:
        return None
    e = fit_expflat(t)
    ok, why = fit_quality(r)
    pull = (r["tau"] - TAU_ANCHOR) / np.hypot(r["tau_err"], TAU_ANCHOR_ERR)
    return dict(selection=label, n_clusters=int(len(t)),
                tau=r["tau"], tau_err=r["tau_err"],
                therm=r["therm"], therm_err=r["therm_err"],
                B=r["B"], redchi=r["redchi"], n_in_fit=r["N"],
                tau_expflat=(e["tau"] if e else np.nan),
                tau_expflat_err=(e["tau_err"] if e else np.nan),
                flat_frac=(e["flat_frac"] if e else np.nan),
                pull_vs_anchor=(pull if ok else np.nan),
                fit_ok=ok, fit_note=why,
                consistent=bool(ok and abs(pull) <= 2.0), **extra)


def closure(data: pd.DataFrame, cfg, wp: pd.DataFrame) -> pd.DataFrame:
    """
    Capture time and multiplicity for each model at each MC working point, against
    two within-pipeline baselines.

    The baselines matter: the anchor was fitted on a different pipeline (the Stage-A/B
    AmBe candidate CSVs), so an offset between "all clusters here" and 30.53 would be a
    pipeline difference, not a statement about the MVA. What the MVA is responsible for
    is the change from baseline to selected.
    """
    rows = []
    base = dict(config=CFG_LBL[cfg], streamline=cfg[0], method=cfg[1])
    r = tau_row(data["t_us"].to_numpy(float), "all clusters",
                model="—", model_short="—", point="—", threshold=np.nan,
                mc_eff=np.nan, data_pass_frac=1.0, **base)
    if r:
        rows.append(r)
    if "passes_stage1" in data.columns:
        s1 = data[data["passes_stage1"].astype(bool)]
        r = tau_row(s1["t_us"].to_numpy(float), "passes_stage1",
                    model="—", model_short="—", point="—", threshold=np.nan,
                    mc_eff=np.nan, data_pass_frac=len(s1) / len(data), **base)
        if r:
            rows.append(r)

    for col, model in SCORE_COLS.items():
        if col not in data.columns:
            continue
        for point in POINTS:
            q = wp[(wp.streamline == cfg[0]) & (wp.method == cfg[1])
                   & (wp.model == model) & (wp.point == point)]
            if q.empty:
                continue
            thr = float(q["threshold"].iloc[0])
            sel = data[data[col] >= thr]
            if len(sel) < 200:
                continue
            r = tau_row(sel["t_us"].to_numpy(float),
                        f"{SHORT[model]} @ {point}",
                        model=model, model_short=SHORT[model], point=point,
                        threshold=thr, mc_eff=float(q["eff_signal"].iloc[0]),
                        data_pass_frac=len(sel) / len(data), **base)
            if r:
                r["n_events_with_pass"] = int(sel["_evt"].nunique())
                r["mean_multiplicity"] = float(
                    sel.groupby("_evt").size().mean())
                # Two Stage-1-relative views. `data_pass_frac` alone is hard to read
                # because the two methods' denominators differ so much (Stage-1 pass
                # rate 31.9% OPTICS vs 64.0% CF). `purity_stage1` is the diagnostic
                # one: a model selecting mostly clusters that FAIL Stage 1 is picking
                # up junk the MC never contained, whatever its score distribution
                # looks like.
                if "passes_stage1" in data.columns:
                    s1 = data["passes_stage1"].astype(bool)
                    sel_s1 = s1.loc[sel.index]
                    r["eff_of_stage1"] = float(sel_s1.sum() / max(s1.sum(), 1))
                    r["purity_stage1"] = float(sel_s1.mean())
                rows.append(r)
    return pd.DataFrame(rows)


REF_POINT = "eff80"


def verdict(cl: pd.DataFrame, point: str = REF_POINT) -> pd.DataFrame:
    """
    Best model on data, per configuration, **at one fixed working point**.

    Comparing across working points would not compare models at all: eff90 is the
    loosest cut, so it always keeps the most clusters and always "wins" a
    yield-ordered ranking. Fixing the point isolates the model. Ranking is by |pull|
    against the anchor — closest capture-time closure first — with yield as context,
    not as the criterion.
    """
    rows = []
    d = cl[(cl["model_short"] != "—") & (cl["point"] == point)]
    for cfg in CONFIGS:
        sub = d[(d.streamline == cfg[0]) & (d.method == cfg[1])]
        if sub.empty:
            continue
        ok = sub[sub["consistent"]]
        pool = ok if len(ok) else sub
        best = pool.assign(_a=pool["pull_vs_anchor"].abs()) \
                   .sort_values("_a").iloc[0]
        allp = cl[(cl.streamline == cfg[0]) & (cl.method == cfg[1])
                  & (cl["model_short"] != "—")]
        # Is the ranking meaningful at all? If every usable model's tau sits inside the
        # anchor's own error band, tau closure has nothing left to say about which
        # model is better — it can only disqualify the ones that fail outright.
        spread = (float(ok["tau"].max() - ok["tau"].min()) if len(ok) > 1
                  else np.nan)
        rows.append(dict(
            config=CFG_LBL[cfg], point=point,
            best_model=best.model_short,
            tau_spread_consistent=spread,
            ranking_resolvable=(bool(spread > 2 * TAU_ANCHOR_ERR)
                                if np.isfinite(spread) else None),
            tau=best.tau, tau_err=best.tau_err, pull=best.pull_vs_anchor,
            redchi=best.redchi, n_selected=int(best.n_clusters),
            data_pass_frac=best.data_pass_frac,
            mean_multiplicity=best.get("mean_multiplicity", np.nan),
            n_consistent_at_point=int(sub["consistent"].sum()),
            n_models_at_point=len(sub),
            n_consistent_all_points=int(allp["consistent"].sum()),
            n_fits_all_points=len(allp),
            n_bad_fits_all_points=int((~allp["fit_ok"]).sum()),
            any_consistent=bool(len(ok) > 0)))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# figures
# ════════════════════════════════════════════════════════════════════════════
def save(fig, name):
    OUTD.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTD / f"{PREFIX}{name}.{ext}",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"    [fig] {PREFIX}{name}.pdf/.png")


def fig_score_overlay(frames_mc, frames_data):
    """AmBe scores over the MC signal and background shapes, GBT, all four configs."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        te = frames_mc[cfg]
        te = te[te["in_test"].astype(bool)]
        da = frames_data[cfg]
        b = np.linspace(0, 1, 61)
        ax.hist(te.loc[te["y"] == 0, "gbt_score"], bins=b, density=True,
                color=GREY, alpha=0.55, label="MC background")
        ax.hist(te.loc[te["y"] == 1, "gbt_score"], bins=b, density=True,
                histtype="step", lw=1.8, color=BLUE, label="MC signal")
        ax.hist(da["gbt_score"].dropna(), bins=b, density=True,
                histtype="step", lw=1.8, color=ORANGE, label="AmBe data")
        ax.set_xlabel("GBT score", fontsize=10)
        ax.set_ylabel("Normalised", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("AmBe Data vs MC Score Distributions — GBT, Merged Models",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "score_data_mc_overlay")


def fig_capture_time(frames_data, cl):
    """Capture time before and after the GBT 80%-efficiency cut, per configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        da = frames_data[cfg]
        sub = cl[(cl.streamline == cfg[0]) & (cl.method == cfg[1])
                 & (cl.model_short == "GBT") & (cl.point == "eff80")]
        b = np.linspace(0, 70, 71)
        ax.hist(da["t_us"].dropna(), bins=b, color=GREY, alpha=0.55,
                label="all clusters")
        if len(sub):
            thr = float(sub["threshold"].iloc[0])
            sel = da[da["gbt_score"] >= thr]
            ax.hist(sel["t_us"].dropna(), bins=b, histtype="step", lw=1.8,
                    color=BLUE,
                    label=(rf"GBT $\geq$ {thr:.3f}:  "
                           rf"$\tau$ = {float(sub['tau'].iloc[0]):.2f} $\pm$ "
                           rf"{float(sub['tau_err'].iloc[0]):.2f} $\mu$s"))
        ax.set_yscale("log")
        ax.set_xlabel(r"Capture time [$\mu$s]", fontsize=10)
        ax.set_ylabel("Clusters", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("AmBe Capture Time, MVA-Selected — anchor "
                 rf"$\tau$ = {TAU_ANCHOR} $\pm$ {TAU_ANCHOR_ERR} $\mu$s",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "capture_time_selected")


def fig_tau_summary(cl):
    """Every fitted tau against the anchor band — the closure result in one figure."""
    # Unusable fits (pinned at a bound, chi2/ndof > 5) are dropped rather than drawn:
    # a marker at tau = 10.0 or 70.0 reads as a measurement and is not one.
    d = cl[(cl["model_short"] != "—") & cl["fit_ok"]].copy()
    if d.empty:
        return
    # One fixed row order for all four panels. Panels do NOT share a row set — the
    # quality gate rejects different fits in each configuration — so deriving the rows
    # per panel while sharing the y axis would silently label every point wrong.
    rows = [(m, p) for m in ("RF", "GBT", "XGB", "NN") for p in POINTS]
    ypos = {k: i for i, k in enumerate(rows[::-1])}

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.2), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        sub = d[(d.streamline == cfg[0]) & (d.method == cfg[1])]
        ax.axvspan(TAU_ANCHOR - TAU_ANCHOR_ERR, TAU_ANCHOR + TAU_ANCHOR_ERR,
                   color=GREY, alpha=0.35)
        taus, errs, yy, cols = [], [], [], []
        for _, g in sub.iterrows():
            k = (g["model_short"], g["point"])
            if k not in ypos:
                continue
            taus.append(float(g["tau"]))
            errs.append(float(g["tau_err"]))
            yy.append(ypos[k])
            cols.append(BLUE if bool(g["consistent"]) else ORANGE)
        if taus:
            ax.errorbar(taus, yy, xerr=errs, fmt="none", lw=1.2,
                        ecolor="#666666")
            ax.scatter(taus, yy, s=26, c=cols, zorder=3)
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([f"{m} {p}" for m, p in ypos], fontsize=7.5)
        ax.set_xlabel(r"$\tau$ [$\mu$s]", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=10.5)
        ax.tick_params(labelsize=8.5)
        bare(ax)
    nbad = int((~cl["fit_ok"]).sum())
    fig.suptitle("Fitted AmBe Capture Lifetime by Model and Working Point\n"
                 rf"grey band = anchor {TAU_ANCHOR} $\pm$ {TAU_ANCHOR_ERR} "
                 r"$\mu$s;  blue = within 2$\sigma$;  "
                 f"{nbad} unusable fits omitted (pinned at a bound or "
                 r"$\chi^2$/ndof > 5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "tau_summary")


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(prog="ccinc_v3_ambe_closure")
    ap.add_argument("--do", required=True,
                    choices=["agreement", "closure", "all"],
                    help="Which section to run. No default on purpose.")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    wp_path = HERE / "ccinc_v3_model_comparison.csv"
    if not wp_path.exists():
        sys.exit("[ambe] ccinc_v3_model_comparison.csv missing — run "
                 "`python ccinc_v3_stats.py --do models` first")
    wp = pd.read_csv(wp_path)

    print("[ambe] loading MC score tables (with the label guard)")
    g = S.guard(verbose=False)
    frames_mc = g["frames"]
    frames_data = {cfg: load_ambe(*cfg) for cfg in CONFIGS}
    for cfg, d in frames_data.items():
        print(f"  [data] {CFG_LBL[cfg]}: {len(d):,} AmBe clusters, "
              f"{d['run'].nunique()} runs, {d['_evt'].nunique():,} events")

    if a.do in ("agreement", "all"):
        print("\n=== data/MC score agreement ===")
        ag = pd.concat([agreement(frames_mc[c], frames_data[c], c)
                        for c in CONFIGS], ignore_index=True)
        ag.to_csv(HERE / "ccinc_v3_ambe_agreement.csv", index=False)
        print(ag[["config", "model_short", "data_median", "mc_sig_median",
                  "mc_bkg_median", "ks_to_sig", "ks_to_bkg", "closer_to",
                  "in_mc_range"]].round(4).to_string(index=False))
        print("[csv] ccinc_v3_ambe_agreement.csv")
        if not a.no_figures:
            fig_score_overlay(frames_mc, frames_data)

    if a.do in ("closure", "all"):
        print("\n=== physics closure: capture time and multiplicity ===")
        cl = pd.concat([closure(frames_data[c], c, wp) for c in CONFIGS],
                       ignore_index=True)
        cl.to_csv(HERE / "ccinc_v3_ambe_closure.csv", index=False)
        print(cl[["config", "selection", "n_clusters", "data_pass_frac", "mc_eff",
                  "tau", "tau_err", "redchi", "pull_vs_anchor", "fit_ok",
                  "fit_note", "consistent"]]
              .round(3).to_string(index=False))
        nbad = int((~cl["fit_ok"]).sum())
        if nbad:
            print(f"\n{nbad} of {len(cl)} fits are unusable and excluded from every "
                  f"verdict below:")
            print(cl.loc[~cl["fit_ok"], ["config", "selection", "tau", "tau_err",
                                         "redchi", "fit_note"]]
                  .round(3).to_string(index=False))
        vd = verdict(cl)
        print(f"\n=== best model on DATA at a fixed {REF_POINT} working point "
              f"(ranked by closeness to the anchor) ===")
        print(vd.round(3).to_string(index=False))
        print("[csv] ccinc_v3_ambe_closure.csv")
        if not a.no_figures:
            fig_capture_time(frames_data, cl)
            fig_tau_summary(cl)


if __name__ == "__main__":
    main()
