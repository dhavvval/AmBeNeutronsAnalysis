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

A third thing became measurable once the extractor started carrying ClusterFinder's own
cluster summary (`cf_clusterTime/PE/CB/Hits/Number`):

  3. **How the new neutron definition compares to the AmBe pipeline's own** (`--do
     classify`, `--dataset ambepipe_v4gated`). The pipeline already defines a neutron —
     Stage 1 gates on the IC waveform cut, Stage 2 keeps clusters with
     `0 < clusterPE <= 100`, `0 < CB < 0.45`, `clusterTime >= 2000 ns`,
     `clusterHits >= 5` — and computes efficiency and capture time per source position.
     This asks what fraction of *those* clusters the MVA also calls neutron, per model,
     per working point and per position, and then tests whether what it rejects is
     actually background — model-free on the capture-time shape, because the
     per-position fits of the rejected sample are statistics-limited (tau_err 10-33 us
     at chi2/ndof ~1.3) and cannot answer it.

     The Stage-2 cut is applied to the SAME table the MVA scored, never by joining to
     `EventAmBeNeutronCandidates_*.csv`: those CSVs store ClusterFinder's FULL hit
     membership, while every MC cluster the model trained on was built from the
     delayed-residual hits within (-5,+20) ns of `clusterTime` — a window that keeps
     81.4% of the stored hits, against features whose top three are `n_hits`,
     `n_hits_early` and `pe_total`.

Run (no silent default):
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python ccinc_v3_ambe_closure.py --do all      --dataset june
    python ccinc_v3_ambe_closure.py --do all      --dataset ambe6266
    python ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4gated
Output (--dataset june):
    ccinc_v3_ambe_agreement.csv, ccinc_v3_ambe_closure.csv
    slide_plots_ccinc_v3_merged/V3AMBE__*.{pdf,png}
Output (--dataset ambe6266):
    ccinc_v3_ambe_agreement_ambe6266.csv, ccinc_v3_ambe_closure_ambe6266.csv
    slide_plots_ccinc_v3_merged/V3AMBE_AMBE6266__*.{pdf,png}
Output (--dataset ambepipe_v4gated):
    ccinc_v3_ambe_classification{,_byposition,_capturetime,_shapetest}_ambepipe_v4gated.csv
    slide_plots_ccinc_v3_merged/V3AMBEPIPE__*.{pdf,png}
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
STAGE = BASE / "ccinc_v3_ambe_stage3"

# RE-DERIVED on the 2-67 us window (was 30.53 +- 0.26 on 10-67), 20 runs / 19
# positions. See the FIT_MIN note in fit_capture_time_fprompt_compare.py.
TAU_ANCHOR, TAU_ANCHOR_ERR = 29.417, 0.221

SCORE_COLS = S.SCORE_COLS
SHORT = S.SHORT
# Working points to test on data. max_significance is excluded on purpose: on a 1:1
# balanced test set it always lands near full acceptance, so it is not a selection.
POINTS = ["eff50", "eff80", "eff90", "max_purity"]

# ════════════════════════════════════════════════════════════════════════════
# datasets (--dataset)
# ════════════════════════════════════════════════════════════════════════════
# `june` — the original Stage-3 sample: all four merged configurations scored on
#   the pooled AmBe2.0v1/v3 features built in June. This is what the report quotes.
#
# `ambe6266` — a single AmBe2.0v4 run, ClusterFinder only, scored twice: ungated
#   and IC-gated (run_ccinc_v3_ambe6266.sh). Its four "configurations" are
#   streamline x gate mode, not streamline x method, so a configuration key here is
#   a 3-tuple (streamline, method, mode) and the label carries the gate.
#
#   OPTICS is absent on purpose: it does not transfer to AmBe data (§4 of the
#   classifier-selection report). Reco-tag is present as a model-transfer check and
#   is NOT data-applicable.
#
#   One run is a PILOT. The point is whether the v4 extraction sits on the same
#   footing as the June v1/v3 campaign — compare the KS distances and the Stage-1
#   pass rate to the June numbers before reading anything into the tau.
# `ambepipe_v4gated` — the AmBe2.0v4_gated Stage-1+2 campaign, 20 runs, the sample
#   the pipeline's own efficiency and capture-time numbers are quoted on. Its
#   clusters carry the CF summary quantities (cf_clusterTime/PE/CB/Hits/Number), so
#   the pipeline's Stage-2 neutron definition can be applied exactly to the same
#   table the MVA scores — which is what `--do classify` is for.
JUNE_CONFIGS = list(CONFIGS)
AMBE6266_CONFIGS = [("truthtag", "clusterfinder", "ungated"),
                    ("truthtag", "clusterfinder", "gated"),
                    ("recotag", "clusterfinder", "ungated"),
                    ("recotag", "clusterfinder", "gated")]
V4GATED_CONFIGS = [("truthtag", "clusterfinder", "gated"),
                   ("recotag", "clusterfinder", "gated")]

DATASETS = {
    "june": dict(configs=JUNE_CONFIGS, prefix="V3AMBE__", suffix="",
                 stage=STAGE, file_prefix="ambe",
                 driver="run_ccinc_v3_ambe_stage3.sh all"),
    "ambe6266": dict(configs=AMBE6266_CONFIGS, prefix="V3AMBE_AMBE6266__",
                     suffix="_ambe6266",
                     stage=BASE / "ambe_data" / "ccinc_v3_ambe6266" / "scored",
                     file_prefix="ambe6266",
                     driver="run_ccinc_v3_ambe6266.sh --campaign run6266 all"),
    "ambepipe_v4gated": dict(
        configs=V4GATED_CONFIGS, prefix="V3AMBEPIPE__", suffix="_ambepipe_v4gated",
        stage=BASE / "ambe_data" / "ccinc_v3_ambe_v4gated" / "scored",
        file_prefix="ambev4gated", classify=True,
        driver="run_ccinc_v3_ambe6266.sh --campaign v4gated all"),
    # All 28 AmBe2.0v4 SOURCE runs — this supersedes ambepipe_v4gated, which is the
    # same study on the 20 that happened to be Stage-1 processed first.
    "ambepipe_v4neutron": dict(
        configs=V4GATED_CONFIGS, prefix="V3AMBEN__", suffix="_ambepipe_v4neutron",
        stage=BASE / "ambe_data" / "ccinc_v3_ambe_v4neutron" / "scored",
        file_prefix="ambev4neutron", classify=True,
        driver="run_ccinc_v3_ambe6266.sh --campaign v4neutron all"),
    # The four no-source runs, reported SEPARATELY and never merged into a capture
    # time or efficiency number.
    "ambepipe_v4special": dict(
        configs=V4GATED_CONFIGS, prefix="V3AMBESPEC__", suffix="_ambepipe_v4special",
        stage=BASE / "ambe_data" / "ccinc_v3_ambe_v4special" / "scored",
        file_prefix="ambev4special", classify=True, special=True,
        driver="run_ccinc_v3_ambe6266.sh --campaign v4special all"),
}

# Source positions for the four no-source special runs, supplied by the user
# (run / port / Z, where Z is the y scan axis). They are DELIBERATELY not added to
# src/ambe/data/processor.py `source_positions`: that map drives the pipeline's own
# efficiency heatmaps, and putting labelled-no-source runs into it would silently
# fold them into the campaign's efficiency numbers. Kept here, used only for
# labelling the separate special-run results.
SPECIAL_POSITIONS = {6264: (0.0, 0.0, 0.0),      # port 5, Z = 0
                     6265: (0.0, 0.0, 0.0),      # port 5, Z = 0
                     6266: (0.0, 100.0, 0.0),    # port 5, Z = 100
                     6270: (75.0, 0.0, 0.0)}     # port 4, Z = 0
SPECIAL_PORT = {6264: "port5_z0", 6265: "port5_z0",
                6266: "port5_z0", 6270: "port4_x75"}
# What the diagnostics' step7_source_presence.csv says about each, carried so the
# separate results are never read as ordinary source runs.
# Candidates / gated events in the pipeline's OWN Stage-1 CSVs, counted directly from
# EventAmBeNeutronCandidates_*.csv. The reference the closure check compares against.
PIPELINE_REFERENCE = {
    "ambepipe_v4gated": (181636, 166665),      # AmBe2.0v4_gated, 20 runs
    "ambepipe_v4neutron": (236792, 217532),    # + AmBe2.0v4_ext, 28 runs
}

SPECIAL_NOTE = {
    6264: "no-source label; cosmic-contaminated (10x prompt rate, tau 48 us)",
    6265: "no-source label, but behaves source-in (tau 35.5 us)",
    6266: "no-source label, but behaves source-in (tau 36.4 us)",
    6270: "no-source label, behaves source-in; BeamCluster is the CORRUPTED merge "
          "(11,195 duplicated events of 46,211 — FIX_6270.md)",
}

DATASET = "june"        # set by main() from --dataset; module-level for the fns
PREFIX = "V3AMBE__"     # figure prefix, likewise


def apply_dataset(name: str) -> None:
    global DATASET, PREFIX
    DATASET = name
    PREFIX = DATASETS[name]["prefix"]
    print(f"[ambe] dataset={name}  figures={PREFIX}*  "
          f"csv suffix={DATASETS[name]['suffix'] or '(none)'}")


def active_configs() -> list:
    return DATASETS[DATASET]["configs"]


def cfg_label(cfg) -> str:
    """Human label. 3-tuple configs carry the gate mode; 2-tuples are the June set."""
    if len(cfg) == 2:
        return CFG_LBL[cfg]
    return f"{CFG_LBL[(cfg[0], cfg[1])]} ({cfg[2]})"


def csv_name(stem: str) -> Path:
    """Per-dataset CSV name. June keeps its established filenames untouched."""
    return HERE / f"{stem}{DATASETS[DATASET]['suffix']}.csv"


def scored_path(cfg) -> Path:
    D = DATASETS[DATASET]
    m = MSFX[cfg[1]]
    if len(cfg) == 2:
        return D["stage"] / f"{cfg[0]}_{m}" / f"{D['file_prefix']}_{cfg[0]}_{m}__scored.parquet"
    stem = f"{D['file_prefix']}_{cfg[0]}_{m}_{cfg[2]}"
    return D["stage"] / f"{cfg[0]}_{m}_{cfg[2]}" / f"{stem}__scored.parquet"


def load_ambe(cfg) -> pd.DataFrame:
    p = scored_path(cfg)
    if not p.exists():
        sys.exit(f"[ambe] missing scored data: {p}\n"
                 f"       run: bash {DATASETS[DATASET]['driver']}")
    d = pd.read_parquet(p)
    d["t_us"] = pd.to_numeric(d["t_mean"], errors="coerce") / 1000.0
    # `event_tank_time` is the event key, NOT `event_number`. eventNumber restarts
    # per part file, so (run, event_number) merges distinct triggers: on run 6266
    # ungated it collapses 37,260 events into 15,032 (ratio 0.40), and on the June
    # v1 sample 79,633 into 72,505 (0.91). Every event-level quantity built on the
    # wrong key — event counts, mean cluster multiplicity — comes out inflated.
    # The June code used event_number; that is the bug, kept fixed here.
    key = "event_tank_time" if "event_tank_time" in d.columns else "event_number"
    if key != "event_tank_time":
        print(f"  [warn] {p.name} has no event_tank_time; falling back to "
              f"event_number, which merges triggers across part files")
    d["_evt"] = d["run"].astype(str) + "#" + d[key].astype(str)
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
            config=cfg_label(cfg), streamline=cfg[0], method=cfg[1], model=model,
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
    r = fit_neutcapture(t, float_B=True, backend="scipy")
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
    base = dict(config=cfg_label(cfg), streamline=cfg[0], method=cfg[1])
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
    for cfg in active_configs():
        # Select on the LABEL, not on (streamline, method): the ambe6266 dataset's
        # configurations differ only by gate mode, so a (streamline, method) filter
        # would silently pool the gated and ungated fits into one ranking.
        lbl = cfg_label(cfg)
        sub = d[d["config"] == lbl]
        if sub.empty:
            continue
        ok = sub[sub["consistent"]]
        pool = ok if len(ok) else sub
        best = pool.assign(_a=pool["pull_vs_anchor"].abs()) \
                   .sort_values("_a").iloc[0]
        allp = cl[(cl["config"] == lbl) & (cl["model_short"] != "—")]
        # Is the ranking meaningful at all? If every usable model's tau sits inside the
        # anchor's own error band, tau closure has nothing left to say about which
        # model is better — it can only disqualify the ones that fail outright.
        spread = (float(ok["tau"].max() - ok["tau"].min()) if len(ok) > 1
                  else np.nan)
        rows.append(dict(
            config=cfg_label(cfg), point=point,
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
# --do classify : the new neutron definition vs the AmBe pipeline's own
# ════════════════════════════════════════════════════════════════════════════
# The AmBe pipeline already defines a neutron: Stage 1 gates on the IC waveform
# cut, Stage 2 keeps clusters passing
#     0 < clusterPE <= 100,  0 < CB < 0.45,  clusterTime >= 2000 ns, clusterHits >= 5
# (src/ambe/data/processor.py: SelectionCuts defaults + ambe_single_cut), after a
# cosmic veto on clusterTime < 2000 ns or clusterPE > 100 PE. This section asks what
# the CCinc v3 MVA calls those clusters.
#
# It is applied to the SAME table the MVA scored, using the cf_* columns carried
# through by extract_cf_features_beamcluster_data.py — not by joining back to
# EventAmBeNeutronCandidates_*.csv. That matters: the candidate CSVs store
# ClusterFinder's FULL hit membership, while every MC cluster the model trained on
# was built from the delayed-residual hits within (-5,+20) ns of clusterTime. On run
# 6062 that window keeps 81.4% of the stored hits (median 0.816, exact agreement in
# 5.2% of clusters), and n_hits/n_hits_early/pe_total are the model's three
# highest-ranked features — so scoring the CSV hit lists would feed the model ~23%
# more light than it ever saw and bias every score upward.
AMBE_CUTS = dict(pe_min=0.0, pe_max=100.0, cb_min=0.0, cb_max=0.45,
                 ct_min=2000.0, hits_min=5)


def ambe_stage2_mask(d: pd.DataFrame) -> pd.Series:
    """The pipeline's Stage-2 neutron definition, on the extractor's cf_* columns."""
    need = ["cf_clusterPE", "cf_clusterCB", "cf_clusterTime", "cf_clusterHits"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        sys.exit(f"[classify] scored table lacks {missing} — it predates the "
                 f"cf_* columns in extract_cf_features_beamcluster_data.py. "
                 f"Re-extract; do not fall back to passes_stage1, which is a "
                 f"different cut (PE<80, CB<0.45, hits>9).")
    c = AMBE_CUTS
    return ((d["cf_clusterPE"] > c["pe_min"]) & (d["cf_clusterPE"] <= c["pe_max"]) &
            (d["cf_clusterCB"] > c["cb_min"]) & (d["cf_clusterCB"] < c["cb_max"]) &
            (d["cf_clusterTime"] >= c["ct_min"]) &
            (d["cf_clusterHits"] >= c["hits_min"]))


def attach_positions(d: pd.DataFrame) -> pd.DataFrame:
    """Source position per run, from the pipeline's own map. Exits on a gap."""
    sys.path.insert(0, str(HERE / "src"))
    from ambe.data.processor import AmBeNeutronProcessing      # noqa: E402
    sp = dict(AmBeNeutronProcessing().source_positions)
    if DATASETS[DATASET].get("special", False):
        sp.update(SPECIAL_POSITIONS)          # see SPECIAL_POSITIONS on why not upstream
    runs = sorted(d["run"].unique())
    missing = [int(r) for r in runs if int(r) not in sp]
    if missing:
        sys.exit(f"[classify] runs with no source_positions entry: {missing}. "
                 f"Every position-resolved number below would be silently wrong.")
    pos = {int(r): sp[int(r)] for r in runs}
    d = d.copy()
    d["src_x"] = d["run"].map(lambda r: pos[int(r)][0])
    d["src_y"] = d["run"].map(lambda r: pos[int(r)][1])
    d["src_z"] = d["run"].map(lambda r: pos[int(r)][2])
    # The position label the campaign uses: port from (x,z), y is the scan axis.
    if DATASETS[DATASET].get("special", False):
        # port_name() has no entry for these, so it returns "unknown" for all four
        # and they would collapse into one position bucket.
        d["port"] = d["run"].map(lambda r: SPECIAL_PORT.get(int(r), "unknown"))
    d["position"] = (d["port"].astype(str) + "  y=" +
                     d["src_y"].round(0).astype(int).astype(str))
    if DATASETS[DATASET].get("special", False):
        # Every special run is its own result. Two of them share port5 y=0, and
        # pooling 6264 (cosmic) with 6265 (source-like) would be exactly the merge
        # this dataset exists to avoid.
        d["position"] = d["run"].astype(str) + "  " + d["position"]
    return d


def closure_vs_pipeline(d: pd.DataFrame, cfg) -> dict:
    """
    Reproduce the pipeline's own candidate count from this table.

    A known, deliberate mismatch: the pipeline's cosmic cut `break`s out of its
    cluster loop, so candidates found BEFORE the cosmic cluster in loop order
    survive, whereas the extractor drops the whole event. That makes this table a
    slight UNDER-count, not an over-count. Reported, not silently corrected.
    """
    m = ambe_stage2_mask(d)
    return dict(config=cfg_label(cfg), clusters_scored=len(d),
                events_scored=int(d["_evt"].nunique()),
                stage2_candidates=int(m.sum()),
                stage2_events=int(d.loc[m, "_evt"].nunique()),
                mean_nhits_over_cf_hits=float(
                    (d["n_hits"] / d["cf_clusterHits"]).mean()))


def classify(d: pd.DataFrame, cfg, wp: pd.DataFrame) -> pd.DataFrame:
    """
    Of the pipeline's Stage-2 AmBe neutrons, what fraction does the MVA call neutron?

    Reported against the MVA's MC working points. These are MC calibrations applied
    to data, so `data_pass_frac` is quoted next to `mc_eff` — where they diverge, the
    threshold has moved, not the model.

    The rejected clusters are carried too, so the two definitions can be crossed:
    a cluster the pipeline rejects but the MVA calls neutron is a candidate the AmBe
    cuts are throwing away, and vice versa.
    """
    is_cand = ambe_stage2_mask(d).to_numpy()
    single = (d["cf_clusterNumber"].to_numpy(int) == 1) if \
        "cf_clusterNumber" in d.columns else np.zeros(len(d), bool)
    rows = []
    for col, model in SCORE_COLS.items():
        if col not in d.columns:
            continue
        s = d[col].to_numpy(float)
        for point in POINTS:
            q = wp[(wp.streamline == cfg[0]) & (wp.method == cfg[1])
                   & (wp.model == model) & (wp.point == point)]
            if q.empty:
                continue
            thr = float(q["threshold"].iloc[0])
            keep = s >= thr
            n_c, n_r = int(is_cand.sum()), int((~is_cand).sum())
            rows.append(dict(
                config=cfg_label(cfg), model=model, model_short=SHORT[model],
                point=point, threshold=thr,
                mc_eff=float(q["eff_signal"].iloc[0]),
                # the headline: the pipeline's neutrons, split by the new definition
                n_stage2=n_c,
                n_stage2_neutron=int((keep & is_cand).sum()),
                pct_stage2_neutron=100.0 * float((keep & is_cand).sum()) / max(n_c, 1),
                pct_stage2_other=100.0 * float((~keep & is_cand).sum()) / max(n_c, 1),
                # the same for what Stage 2 rejected, so the 2x2 closes
                n_rejected=n_r,
                pct_rejected_neutron=100.0 * float((keep & ~is_cand).sum()) / max(n_r, 1),
                # the 2x2 itself, as fractions of ALL scored clusters
                pct_agree_neutron=100.0 * float((keep & is_cand).mean()),
                pct_agree_reject=100.0 * float((~keep & ~is_cand).mean()),
                pct_mva_only=100.0 * float((keep & ~is_cand).mean()),
                pct_pipeline_only=100.0 * float((~keep & is_cand).mean()),
                # single vs multiple, among the pipeline's own candidates
                pct_single_neutron=(100.0 * float((keep & is_cand & single).sum())
                                    / max(int((is_cand & single).sum()), 1)),
                pct_multiple_neutron=(100.0 * float((keep & is_cand & ~single).sum())
                                      / max(int((is_cand & ~single).sum()), 1)),
                data_pass_frac=float(keep.mean()),
            ))
    return pd.DataFrame(rows)


def classify_by_position(d: pd.DataFrame, cfg, wp: pd.DataFrame,
                         model: str = "GBT", point: str = "eff80") -> pd.DataFrame:
    """
    The same split, per source position.

    This is the measurement §6.4 of ANALYSIS_ccinc_v3_FULL.md could only infer from
    one run: the score scale moves with where the source sits, because the leading
    features are light-yield features and the yield depends on the distance to the
    PMTs. 19 positions is the first sample that can actually measure it.
    """
    col = [c for c, m in SCORE_COLS.items() if m == model][0]
    q = wp[(wp.streamline == cfg[0]) & (wp.method == cfg[1])
           & (wp.model == model) & (wp.point == point)]
    if q.empty or col not in d.columns:
        return pd.DataFrame()
    thr = float(q["threshold"].iloc[0])
    cand = d[ambe_stage2_mask(d).to_numpy()].copy()
    cand["_keep"] = cand[col].to_numpy(float) >= thr
    g = cand.groupby(["port", "position"], observed=True)
    out = g.agg(n_stage2=("_keep", "size"),
                n_neutron=("_keep", "sum"),
                median_score=(col, "median"),
                median_n_hits=("n_hits", "median"),
                median_pe_total=("pe_total", "median"),
                median_d_wall=("d_wall", "median"),
                runs=("run", "nunique")).reset_index()
    out["pct_neutron"] = 100.0 * out["n_neutron"] / out["n_stage2"]
    out.insert(0, "point", point)
    out.insert(0, "model", model)
    out.insert(0, "config", cfg_label(cfg))
    return out.sort_values(["port", "position"])


def classify_capture_time(d: pd.DataFrame, cfg, wp: pd.DataFrame,
                          model: str = "GBT", point: str = "eff80") -> pd.DataFrame:
    """
    The check that makes the percentage physics rather than a number.

    Fit the capture time of the kept and the rejected halves of the pipeline's own
    neutron candidates, PER POSITION (pooled fits give chi2/ndof 9-14 — positions
    have genuinely different tau and amplitude). If the rejected half is background,
    its tau should be short, flat or unfittable while the kept half tightens toward
    the 30.53 +- 0.26 us anchor. If both halves fit the same tau, the new definition
    is discarding signal and the split means nothing on its own.
    """
    col = [c for c, m in SCORE_COLS.items() if m == model][0]
    q = wp[(wp.streamline == cfg[0]) & (wp.method == cfg[1])
           & (wp.model == model) & (wp.point == point)]
    if q.empty or col not in d.columns:
        return pd.DataFrame()
    thr = float(q["threshold"].iloc[0])
    cand = d[ambe_stage2_mask(d).to_numpy()].copy()
    cand["_keep"] = cand[col].to_numpy(float) >= thr
    rows = []
    for (port, pos), sub in cand.groupby(["port", "position"], observed=True):
        for lbl, part in (("all candidates", sub),
                          ("MVA neutron", sub[sub["_keep"]]),
                          ("MVA other", sub[~sub["_keep"]])):
            if len(part) < 200:
                continue
            r = tau_row(part["t_us"].to_numpy(float), lbl,
                        config=cfg_label(cfg), model=model, point=point,
                        port=port, position=pos)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def classify_shape_test(d: pd.DataFrame, cfg, wp: pd.DataFrame,
                        model: str = "GBT", point: str = "eff80") -> pd.DataFrame:
    """
    Model-free comparison of the kept and rejected capture-time shapes.

    THIS, not the per-position fits, is what decides whether the rejected fraction is
    background. Every "MVA other" fit fails the quality gate on tau_err (10-33 us)
    while its chi2/ndof sits near 1.3 — the fits are fine, the sample is just too
    small per position to pin a 30 us lifetime. Reading "no usable fit" as evidence of
    background would be wrong.

    The same trap is on record from the fprompt study: a flat pedestal on top of an
    exponential is degenerate with a long tau, and `lmfit_analysis` fixes B at 0, so a
    pedestal INFLATES tau rather than being absorbed. That study settled the analogous
    question model-free, and so does this.

    Three statistics, all on the anchor's own 10-67 us window, none needing a fit:
      * KS distance between the two shapes;
      * the difference of means, with its error;
      * the late/early ratio (30-67 us over 10-30 us) — a flat accidental component
        raises it, an exponential does not.
    """
    col = [c for c, m in SCORE_COLS.items() if m == model][0]
    q = wp[(wp.streamline == cfg[0]) & (wp.method == cfg[1])
           & (wp.model == model) & (wp.point == point)]
    if q.empty or col not in d.columns:
        return pd.DataFrame()
    thr = float(q["threshold"].iloc[0])
    cand_all = d[ambe_stage2_mask(d).to_numpy()]
    # The special dataset is tested PER RUN: pooling 6264 (cosmic-contaminated) with
    # 6265/6266/6270 (source-like) is exactly the merge that dataset exists to avoid.
    if DATASETS[DATASET].get("special", False):
        return pd.concat([_shape_one(sub, cfg, model, point, thr, col, run=int(r))
                          for r, sub in cand_all.groupby("run")],
                         ignore_index=True)
    return _shape_one(cand_all, cfg, model, point, thr, col)


def _shape_one(cand, cfg, model, point, thr, col, run=None) -> pd.DataFrame:
    keep = cand[col].to_numpy(float) >= thr
    t = cand["t_us"].to_numpy(float)
    lo, hi = 10.0, 67.0
    A = t[keep & (t > lo) & (t < hi)]
    B = t[~keep & (t > lo) & (t < hi)]
    if len(A) < 300 or len(B) < 300:
        return pd.DataFrame()

    def late_early(x):
        e = int(((x > lo) & (x < 30)).sum())
        l = int(((x >= 30) & (x < hi)).sum())
        r = l / max(e, 1)
        return r, r * np.sqrt(1 / max(l, 1) + 1 / max(e, 1))

    ra, ea = late_early(A)
    rb, eb = late_early(B)
    sa, sb = A.std() / np.sqrt(len(A)), B.std() / np.sqrt(len(B))
    dm, edm = B.mean() - A.mean(), float(np.hypot(sa, sb))
    ks = ks_2samp(A, B)
    return pd.DataFrame([dict(
        config=cfg_label(cfg), run=run, model=model, point=point, threshold=thr,
        n_kept=len(A), n_rejected=len(B),
        mean_kept=float(A.mean()), mean_kept_err=float(sa),
        mean_rejected=float(B.mean()), mean_rejected_err=float(sb),
        mean_diff=float(dm), mean_diff_err=edm,
        mean_diff_sigma=float(dm / edm) if edm > 0 else np.nan,
        ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
        late_early_kept=ra, late_early_kept_err=ea,
        late_early_rejected=rb, late_early_rejected_err=eb,
        late_early_ratio=rb / ra if ra else np.nan,
        # A flat accidental component raises the late/early ratio and pushes the mean
        # later. Both moving the same way is the signature; either alone is not.
        rejected_is_flatter=bool(rb - ra > 2 * np.hypot(ea, eb) and dm > 2 * edm),
    )])


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
    for ax, cfg in zip(axes.ravel(), active_configs()):
        te = frames_mc[cfg[:2]]
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
        ax.set_title(cfg_label(cfg), fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("MVA neutron — GBT score, AmBe data vs MC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "score_data_mc_overlay")


def fig_capture_time(frames_data, cl):
    """Capture time before and after the GBT 80%-efficiency cut, per configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    for ax, cfg in zip(axes.ravel(), active_configs()):
        da = frames_data[cfg]
        # Select on the config LABEL: the ambe6266 configurations differ only by
        # gate mode, so a (streamline, method) filter matches both and .iloc[0]
        # silently labels the gated panel with the ungated tau.
        sub = cl[(cl["config"] == cfg_label(cfg))
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
        ax.set_title(cfg_label(cfg), fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("MVA neutron — capture time; anchor "
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
    for ax, cfg in zip(axes, active_configs()):
        sub = d[d["config"] == cfg_label(cfg)]      # label, not (streamline, method)
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
        ax.set_title(cfg_label(cfg), fontsize=10.5)
        ax.tick_params(labelsize=8.5)
        bare(ax)
    nbad = int((~cl["fit_ok"]).sum())
    fig.suptitle("MVA neutron — fitted capture time by model and working point\n"
                 rf"grey band = anchor {TAU_ANCHOR} $\pm$ {TAU_ANCHOR_ERR} "
                 r"$\mu$s;  blue = within 2$\sigma$;  "
                 f"{nbad} unusable fits omitted (pinned at a bound or "
                 r"$\chi^2$/ndof > 5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "tau_summary")


def fig_classify_split(cl, frames_data, wp):
    """The headline: of the pipeline's AmBe neutrons, how many the MVA keeps."""
    cfgs = active_configs()
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.4 * len(cfgs), 4.8),
                             squeeze=False)
    for ax, cfg in zip(axes.ravel(), cfgs):
        sub = cl[cl["config"] == cfg_label(cfg)]
        if sub.empty:
            continue
        pts = [p for p in POINTS if p in set(sub["point"])]
        x = np.arange(len(pts))
        w = 0.2
        for i, (m, c) in enumerate(zip(["RF", "GBT", "XGB", "NN"], LINE_COLORS)):
            y = [float(sub[(sub.model_short == m) & (sub.point == p)]
                       ["pct_stage2_neutron"].iloc[0])
                 if len(sub[(sub.model_short == m) & (sub.point == p)]) else np.nan
                 for p in pts]
            ax.bar(x + (i - 1.5) * w, y, width=w, color=c, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(pts)
        ax.set_ylabel("Pipeline neutrons kept by the MVA [%]", fontsize=10)
        ax.set_xlabel("MC working point", fontsize=10)
        ax.set_title(cfg_label(cfg), fontsize=11)
        ax.legend(fontsize=9, frameon=False, ncol=4)
        ax.tick_params(labelsize=9)
        bare(ax)
    n = int(cl["n_stage2"].max()) if len(cl) else 0
    fig.suptitle("MVA neutron applied to box cuts PE ≤ 100, CB < 0.45, t ≥ 2 µs, "
                 f"hits ≥ 5\nFraction kept — {n:,} candidates", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "classified_fraction")


def fig_classify_by_position(bypos):
    """Per source position — the position-dependence of the score scale."""
    if bypos.empty:
        return
    t = bypos[bypos["config"].str.startswith("Truth-tag")]
    if t.empty:
        t = bypos
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ports = list(dict.fromkeys(t["port"]))
    # Five ports, and LINE_COLORS has four — cycling it silently paints port 5 the
    # same blue as port 1, which reads as one port with a huge spread.
    palette = [BLUE, ORANGE, GREEN, "#7b3294", "#8c564b", "#17becf", "#666666"]
    x = np.arange(len(t))
    for port, c in zip(ports, palette):
        m = (t["port"] == port).to_numpy()
        ax.scatter(x[m], t["pct_neutron"].to_numpy()[m], s=44, color=c, label=port)
    ax.set_xticks(x)
    ax.set_xticklabels(t["position"], rotation=90, fontsize=7.5)
    ax.set_ylabel("Pipeline neutrons kept by the MVA [%]", fontsize=11)
    ax.legend(fontsize=9, frameon=False, ncol=len(ports))
    ax.tick_params(axis="y", labelsize=9)
    bare(ax)
    lo, hi = t["pct_neutron"].min(), t["pct_neutron"].max()
    ax.set_title("MVA neutron, GBT eff80\n"
                 f"Fraction kept by source position — spread {lo:.0f}–{hi:.0f}%",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, "classified_by_position")


def fig_classify_capture_time(ct):
    """Kept vs rejected capture time, per position, against the anchor band."""
    d = ct[ct["fit_ok"]].copy()
    if d.empty:
        return
    t = d[d["config"].str.startswith("Truth-tag")]
    if t.empty:
        t = d
    order = sorted(set(t["position"]))
    ypos = {p: i for i, p in enumerate(order[::-1])}
    fig, ax = plt.subplots(figsize=(8.6, 0.30 * len(order) + 2.6))
    ax.axvspan(TAU_ANCHOR - TAU_ANCHOR_ERR, TAU_ANCHOR + TAU_ANCHOR_ERR,
               color=GREY, alpha=0.35)
    # (key, display label): the KEY must stay the value written into the CSV's
    # `selection` column, the label is only what the legend says. Renaming the label
    # alone made `selection == lbl` match nothing and drew an empty figure.
    for key, lbl, c, dy in (("MVA neutron", "MVA-selected", BLUE, +0.18),
                            ("MVA other", "MVA-rejected", ORANGE, -0.18)):
        s = t[t["selection"] == key]
        if s.empty:
            continue
        yy = [ypos[p] + dy for p in s["position"]]
        ax.errorbar(s["tau"], yy, xerr=s["tau_err"], fmt="o", ms=4, lw=1.1,
                    color=c, ecolor=c, label=lbl)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos), fontsize=7.5)
    ax.set_xlabel(r"$\tau$ [$\mu$s]", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(axis="x", labelsize=9)
    bare(ax)
    nbad = int((~ct["fit_ok"]).sum())
    ax.set_title("MVA-selected vs MVA-rejected — capture time by source position\n"
                 rf"grey band = anchor {TAU_ANCHOR} $\pm$ {TAU_ANCHOR_ERR} $\mu$s;  "
                 f"{nbad} unusable fits omitted", fontsize=11.5)
    fig.tight_layout()
    save(fig, "classified_capture_time")


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(prog="ccinc_v3_ambe_closure")
    ap.add_argument("--do", required=True,
                    choices=["agreement", "closure", "classify", "all"],
                    help="Which section to run. No default on purpose.")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                    help="june = the four merged configurations on the pooled "
                         "AmBe2.0v1/v3 features (what the report quotes). "
                         "ambe6266 = the single AmBe2.0v4 run 6266, ClusterFinder "
                         "only, ungated and IC-gated. ambepipe_v4gated = the 20-run "
                         "AmBe2.0v4_gated Stage-1+2 campaign, the one --do classify "
                         "is for. No default on purpose: they write different CSVs "
                         "and must not be confused.")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    apply_dataset(a.dataset)

    wp_path = HERE / "ccinc_v3_model_comparison.csv"
    if not wp_path.exists():
        sys.exit("[ambe] ccinc_v3_model_comparison.csv missing — run "
                 "`python ccinc_v3_stats.py --do models` first")
    wp = pd.read_csv(wp_path)

    print("[ambe] loading MC score tables (with the label guard)")
    g = S.guard(verbose=False)
    frames_mc = g["frames"]
    frames_data = {cfg: load_ambe(cfg) for cfg in active_configs()}
    for cfg, d in frames_data.items():
        print(f"  [data] {cfg_label(cfg)}: {len(d):,} AmBe clusters, "
              f"{d['run'].nunique()} runs, {d['_evt'].nunique():,} events")

    if a.do in ("agreement", "all"):
        print("\n=== data/MC score agreement ===")
        ag = pd.concat([agreement(frames_mc[c[:2]], frames_data[c], c)
                        for c in active_configs()], ignore_index=True)
        ag_path = csv_name("ccinc_v3_ambe_agreement")
        ag.to_csv(ag_path, index=False)
        print(ag[["config", "model_short", "data_median", "mc_sig_median",
                  "mc_bkg_median", "ks_to_sig", "ks_to_bkg", "closer_to",
                  "in_mc_range"]].round(4).to_string(index=False))
        print(f"[csv] {ag_path.name}")
        if not a.no_figures:
            fig_score_overlay(frames_mc, frames_data)

    if a.do in ("closure", "all"):
        print("\n=== physics closure: capture time and multiplicity ===")
        cl = pd.concat([closure(frames_data[c], c, wp) for c in active_configs()],
                       ignore_index=True)
        cl_path = csv_name("ccinc_v3_ambe_closure")
        cl.to_csv(cl_path, index=False)
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
        print(f"[csv] {cl_path.name}")
        if not a.no_figures:
            fig_capture_time(frames_data, cl)
            fig_tau_summary(cl)

    # `all` runs classify only where it is meaningful. Asking for it explicitly on a
    # dataset without the cf_* columns is an error, not a silent skip — `--do all
    # --dataset june` must keep working, but `--do classify --dataset june` must say
    # why it cannot.
    want_classify = a.do == "classify" or (
        a.do == "all" and DATASETS[DATASET].get("classify", False))
    if a.do == "classify" and not DATASETS[DATASET].get("classify", False):
        sys.exit(f"[classify] --dataset {DATASET} has no cf_* columns (it predates "
                 f"them in extract_cf_features_beamcluster_data.py). Use "
                 f"--dataset ambepipe_v4gated, or re-extract that campaign.")
    if want_classify:
        print("\n=== the new neutron definition vs the AmBe pipeline's own ===")
        frames_pos = {c: attach_positions(frames_data[c]) for c in active_configs()}

        clo = pd.DataFrame([closure_vs_pipeline(frames_pos[c], c)
                            for c in active_configs()])
        print("\n  closure against the pipeline's own Stage-2 count:")
        print(clo.round(4).to_string(index=False))
        ref = PIPELINE_REFERENCE.get(DATASET)
        if ref:
            print(f"  (the pipeline's own Stage-1 CSVs hold {ref[0]:,} candidates over "
                  f"{ref[1]:,} gated events;\n   this table is a slight UNDER-count by "
                  f"design — its cosmic veto drops the whole\n   event, the pipeline's "
                  f"breaks out of the cluster loop and keeps earlier candidates)")
            got_c = int(clo["stage2_candidates"].iloc[0])
            got_e = int(clo["stage2_events"].iloc[0])
            print(f"  candidates {got_c:,} vs {ref[0]:,} ({100*got_c/ref[0]-100:+.1f}%)   "
                  f"events {got_e:,} vs {ref[1]:,} ({100*got_e/ref[1]-100:+.1f}%)")
        r = float(clo["mean_nhits_over_cf_hits"].iloc[0])
        # Two different numbers, do not confuse them. Windowing the candidate CSV's
        # OWN hit list to (-5,+20) ns keeps 0.814 of it (measured on run 6062). The
        # extractor is slightly higher because it starts from ALL residual hits in
        # the event and assigns each to the NEAREST cluster in window, so a hit
        # outside CF's stored membership can still be assigned back — measured 0.873
        # on this campaign. What matters is that it is well below 1.0: a ratio at 1.0
        # means CF's raw membership leaked in and every score is biased high, because
        # n_hits / n_hits_early / pe_total are the model's three leading features.
        print(f"  mean n_hits / cf_clusterHits = {r:.3f}  "
              f"({'OK' if 0.75 < r < 0.95 else 'WRONG: raw membership leaked in'}"
              f" — MC-matched (-5,+20) ns assignment; 1.0 would mean raw membership)")

        cl = pd.concat([classify(frames_pos[c], c, wp) for c in active_configs()],
                       ignore_index=True)
        cl.to_csv(csv_name("ccinc_v3_ambe_classification"), index=False)
        print("\n  of the pipeline's Stage-2 AmBe neutrons, % the MVA also calls "
              "neutron:")
        print(cl[["config", "model_short", "point", "threshold", "mc_eff",
                  "n_stage2", "pct_stage2_neutron", "pct_rejected_neutron",
                  "pct_single_neutron", "pct_multiple_neutron"]]
              .round(2).to_string(index=False))

        bypos = pd.concat([classify_by_position(frames_pos[c], c, wp)
                           for c in active_configs()], ignore_index=True)
        bypos.to_csv(csv_name("ccinc_v3_ambe_classification_byposition"),
                     index=False)
        print("\n  per source position (GBT @ eff80):")
        print(bypos[["config", "port", "position", "runs", "n_stage2",
                     "pct_neutron", "median_score", "median_n_hits",
                     "median_d_wall"]].round(3).to_string(index=False))

        sh = pd.concat([classify_shape_test(frames_pos[c], c, wp)
                        for c in active_configs()], ignore_index=True)
        sh.to_csv(csv_name("ccinc_v3_ambe_classification_shapetest"), index=False)
        print("\n  is what the MVA rejects actually background? "
              "(model-free, 10-67 us window, GBT @ eff80)")
        for _, r in sh.iterrows():
            print(f"    {r.config}")
            print(f"      kept {int(r.n_kept):,} / rejected {int(r.n_rejected):,}   "
                  f"KS D={r.ks_stat:.4f} (p={r.ks_pvalue:.2g})")
            print(f"      mean capture time  kept {r.mean_kept:.3f}+-{r.mean_kept_err:.3f}"
                  f"   rejected {r.mean_rejected:.3f}+-{r.mean_rejected_err:.3f} us"
                  f"   diff {r.mean_diff:+.3f}+-{r.mean_diff_err:.3f} "
                  f"({r.mean_diff_sigma:+.1f} sigma)")
            print(f"      late/early ratio   kept {r.late_early_kept:.4f}+-{r.late_early_kept_err:.4f}"
                  f"   rejected {r.late_early_rejected:.4f}+-{r.late_early_rejected_err:.4f}"
                  f"   ({r.late_early_ratio:.3f}x)")
            print(f"      -> rejected sample is flatter (extra flat component): "
                  f"{'YES' if r.rejected_is_flatter else 'NO'}")

        ct = pd.concat([classify_capture_time(frames_pos[c], c, wp)
                        for c in active_configs()], ignore_index=True)
        ct.to_csv(csv_name("ccinc_v3_ambe_classification_capturetime"), index=False)
        ok = ct[ct["fit_ok"]]
        print("\n  capture time of what the MVA keeps vs rejects "
              f"({len(ok)} usable of {len(ct)} fits):")
        # "no usable fit" for MVA other is a STATISTICS statement, not evidence of
        # background: every failure below is tau_err > 10 us with chi2/ndof near 1.3.
        # The shape test above is what answers the background question.
        for lbl in ("all candidates", "MVA neutron", "MVA other"):
            s_ = ok[ok["selection"] == lbl]
            if not len(s_):
                print(f"    {lbl:16s} no usable fit")
                continue
            w = 1.0 / s_["tau_err"].to_numpy(float) ** 2
            tw = float((s_["tau"].to_numpy(float) * w).sum() / w.sum())
            print(f"    {lbl:16s} n={len(s_):3d} positions  "
                  f"weighted tau = {tw:6.2f} us  "
                  f"(median chi2/ndof {s_['redchi'].median():.2f})")
        print(f"[csv] ccinc_v3_ambe_classification{{,_byposition,_capturetime,"
              f"_shapetest}}{DATASETS[DATASET]['suffix']}.csv")
        if not a.no_figures:
            fig_classify_split(cl, frames_pos, wp)
            fig_classify_by_position(bypos)
            fig_classify_capture_time(ct)


if __name__ == "__main__":
    main()
