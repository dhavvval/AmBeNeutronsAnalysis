"""
make_presentable_optics_rf_plots.py

Poster-styled OPTICS + Random-Forest plots, appended as extra pages to the
existing MVA PDF. NOTHING is re-derived: every number reuses the same machinery
as run_ccinc_xi02_compcut.sh —
  * RF importances + sig/bkg feature distributions: frozen .pkl + MC scores parquet
  * capture-time fit: AmBeNeutronAnalyzer.NeutCapture / _prepare_fitting_data
    with the SAME fitting_config + capture_bounds=(2,70) the pipeline uses
  * multiplicity: clusters per [run, event_tank_time] passing rf_score >= cut

Output: a single new PDF (default beside the MVA PDF) with, in order:
  1. RF top-6 feature distributions, presentable names + units
  2. AmBe neutron multiplicity (OPTICS+RF) — clean histogram
  3. one capture-time page PER source position (styled)
  4. one combined all-positions capture-time page (styled)

Usage:
  python make_presentable_optics_rf_plots.py \
      --mc-scores .../cc_neutrino_xi02_compcut__mva_scores__keepprompt.parquet \
      --frozen    .../cc_neutrino_xi02_compcut__mva_frozen__keepprompt.pkl \
      --scored    .../ccinc_xi02cc_optics_mcbkg__scored.parquet \
      --score-cut 0.547 \
      --out       .../cc_neutrino_xi02_compcut__presentable_optics_rf.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.backends.backend_pdf import PdfPages

# --- expose scipy.signal.gaussian for arviz (pulled in via ambe.plots.basic) ---
import sys
import scipy.signal
import scipy.signal.windows as _w
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = _w.gaussian
sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.plots.basic import AmBeNeutronAnalyzer            # noqa: E402
from ambe.data.processor import AmBeNeutronProcessing       # noqa: E402

SOURCE_POSITIONS = AmBeNeutronProcessing().source_positions  # run -> (x,y,z) cm
NEUTRON_CLASSES = {1, 2, 3, 4}
EVENT_KEYS = ["run", "event_tank_time"]

# ---- uniform style for every page in the new PDF -------------------------------
# Two styles, selected by --style (no default: the choice is always explicit).
#   poster : the original June-16 look — DejaVu Sans, blue signal / grey background
#   bw     : black and white only, Helvetica metrics (Nimbus Sans), no hue anywhere
_BASE_RC = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "normal",       # no bold titles anywhere
    "font.weight": "normal",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.grid": False,            # plain background (no dotted gridlines)
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
# Nimbus Sans is the URW Helvetica clone and the only Helvetica-metric face on this
# machine. It must be set for mathtext too, or greek renders in DejaVu and the two
# faces don't match on the page. Check with `pdffonts` on the output.
_HELVETICA = ["Nimbus Sans", "Helvetica", "Nimbus Sans L", "Liberation Sans", "Arial"]
STYLE_RC = {
    "poster": {**_BASE_RC, "axes.edgecolor": "#333333"},
    "bw":     {**_BASE_RC, "axes.edgecolor": "black", "hatch.linewidth": 0.7},
}

# --font: the face, independent of the palette. helvetica = Nimbus Sans (the URW
# Helvetica clone, the only Helvetica-metric face here) for words AND mathtext, or
# greek renders in DejaVu and the two faces don't match. dejavu reproduces the
# June-16 figures exactly. Whichever is used is printed at startup.
FONT_RC = {
    "helvetica": {"font.family": "sans-serif",
                  "font.sans-serif": _HELVETICA,
                  "mathtext.fontset": "custom",
                  "mathtext.rm": "Nimbus Sans",
                  "mathtext.it": "Nimbus Sans:italic",
                  "mathtext.bf": "Nimbus Sans:bold",
                  "mathtext.cal": "Nimbus Sans"},
    "dejavu":    {"font.family": "DejaVu Sans",
                  "mathtext.fontset": "dejavusans"},
}

# Per-style drawing kwargs. In bw the two histogram classes are separated by fill
# level and outline, never by hue: signal is a black outline over a grey background
# fill, which stays readable in print and photocopy.
STYLE_ART = {
    "poster": {
        "sig":  dict(color="#1f77b4", alpha=0.85),
        "bkg":  dict(color="#999999", alpha=0.55),
        "step": dict(color="#1f77b4"),
        "fit":  dict(color="#2ca02c"),
    },
    "bw": {
        "sig":  dict(histtype="step", edgecolor="black", linewidth=1.7),
        "bkg":  dict(facecolor="0.78", edgecolor="black", linewidth=0.8),
        "step": dict(color="black"),
        "fit":  dict(color="black", linestyle="--"),
    },
}

STYLE = "poster"          # set by main() from --style; module-level for the page fns


def art(kind):
    """Drawing kwargs for the active style."""
    return STYLE_ART[STYLE][kind]


def apply_style(style: str, font: str = "helvetica") -> None:
    global STYLE, HIST_COLOR, FIT_COLOR
    STYLE = style
    plt.rcParams.update(STYLE_RC[style])
    plt.rcParams.update(FONT_RC[font])
    HIST_COLOR = STYLE_ART[style]["step"]["color"]
    FIT_COLOR = STYLE_ART[style]["fit"]["color"]
    print(f"[presentable] style={style}  font={font}")


HIST_COLOR = "#1f77b4"            # shared blue used by all three plot families
FIT_COLOR = "#2ca02c"            # green capture-fit curve (matches existing)

# Large-font sizes for the single-panel slides (multiplicity + capture time), to
# match the requested title>>axis>>tick ratio. The 6-panel feature pages keep the
# smaller default sizes (small multiples).
BIG_TITLE = 23
BIG_LABEL = 18
BIG_TICK = 16
BIG_LEGEND = 21          # multiplicity + top-2 feature legends (1.5x the old 14)
CAPTURE_LEGEND = 14      # capture-time legends: smaller, the fit-box overlaps data

# ---- "ANNIE Work In Progress" watermark ----------------------------------------
WATERMARK_TEXT = "ANNIE Work In Progress"


def _watermark(fig, x=0.012, y=0.985):
    """Watermark disabled (no-op). Re-enable by uncommenting the fig.text below."""
    # fig.text(x, y, WATERMARK_TEXT, ha="left", va="top", fontsize=11,
    #          color="#b22222", style="italic", alpha=0.85, zorder=1000)
    return

# ---- presentable feature names (β in greek, descriptive + units) ---------------
# spatial_rms confirmed from src/ambe/mc/cluster_features.py:576 = RMS of hit PMT
# positions around the PE-weighted centroid, in METRES.
# d_wall and vtx_y are METRES too — cluster_features.py:30 and its own FEATURE_LABELS
# map ("Distance to Nearest Wall (m)"), and the geometry is built in m. The "[cm]" that
# used to be here was wrong; it is on the June-16 poster figures as well.
FEATURE_LABELS = {
    "pe_total":          "Total PE [p.e.]",
    "n_hits":            "Cluster Hits",
    "n_hits_early":      "Early Hits",
    "n_fit_hits":        "Hits Used in the Vertex Fit",
    "sigma_t_mad":       r"$\sigma_t$ (MAD) [ns]",
    "sigma_t_mad_corr":  r"$\sigma_t$ (MAD, corrected) [ns]",
    "t_window_80pct":    "80% Time Window [ns]",
    "fit_rms_ns":        "Fit Timing RMS [ns]",
    "fit_goodness_init": "Fit Goodness (Centroid Vertex)",
    "vtx_y":             "Vertex Y [m]",
    "pe_balance":        "Charge Balance (PE)",
    "spatial_rms":       "Spatial RMS of Hit PMTs [m]",
    "beta2":             r"$\beta_2$",
    "beta1":             r"$\beta_1$",
    "d_wall":            "Distance to Wall [m]",
    "charge_bal_legacy": "Charge Balance",
}

# Title labels: same names without units (units stay on the x-axis).
FEATURE_TITLES = {
    "pe_total":          "Total PE",
    "n_hits":            "Cluster Hits",
    "n_hits_early":      "Early Hits",
    "n_fit_hits":        "Hits Used in the Vertex Fit",
    "sigma_t_mad":       r"$\sigma_t$ (MAD)",
    "sigma_t_mad_corr":  r"$\sigma_t$ (MAD, corrected)",
    "t_window_80pct":    "80% Time Window",
    "fit_rms_ns":        "Fit Timing RMS",
    "fit_goodness_init": "Fit Goodness (Centroid Vertex)",
    "vtx_y":             "Vertex Y",
    "pe_balance":        "Charge Balance (PE)",
    "spatial_rms":       "Spatial RMS of Hit PMTs",
    "beta2":             r"$\beta_2$",
    "beta1":             r"$\beta_1$",
    "d_wall":            "Distance to Wall",
    "charge_bal_legacy": "Charge Balance",
}


def _feat_title(feat):
    """Panel title: 'Discriminating Feature: <name>' (no units)."""
    return f"Discriminating Feature: {FEATURE_TITLES.get(feat, feat)}"


MODEL_NAMES = {"rf": "Random Forest", "gbt": "GBT", "xgb": "XGBoost"}


def top_features(frozen_path, model="rf", n=6):
    """
    Top-n features by the chosen model's importances, read from the FROZEN bundle.
    Nothing is refitted — the .pkl already holds the trained models.
    """
    bundle = joblib.load(frozen_path)
    feats = list(bundle["features"])
    models = bundle["models"]
    if model not in models:
        raise SystemExit(f"[presentable] frozen bundle has no '{model}' model "
                         f"(has: {sorted(models)})")
    imp = np.asarray(models[model].feature_importances_, float)
    order = np.argsort(imp)[::-1][:n]
    return [feats[i] for i in order], dict(zip(feats, imp))


def resolve_score_cut(d, score_col, spec):
    """
    'auto' = the 80%-signal-efficiency threshold, i.e. the 20th percentile of the
    signal scores on the test set — the same definition the June deck used. Read
    off the existing score table; nothing is retrained.
    """
    if spec != "auto":
        return float(spec)
    is_sig = signal_mask(d)
    m = is_sig & (d["in_test"].to_numpy(bool) if "in_test" in d.columns
                  else np.ones(len(d), bool))
    cut = float(np.quantile(d.loc[m, score_col].to_numpy(float), 0.20))
    print(f"      score cut (80% signal efficiency on {score_col}) = {cut:.3f}")
    return cut


def feature_bins(v_all, n=40):
    """
    1-99 percentile range. Integer-valued features get integer-centred bins —
    linspace over an integer variable splits one integer across two bars and the
    histogram comes out combed.
    """
    lo, hi = np.nanpercentile(v_all, [1, 99])
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1
    finite = v_all[np.isfinite(v_all)]
    if np.allclose(finite, np.round(finite)) and (hi - lo) <= 60:
        return np.arange(np.floor(lo), np.ceil(hi) + 2) - 0.5
    return np.linspace(lo, hi, n)


def signal_mask(d):
    """
    Truth signal mask for a scored frame: neutron-dominated cluster.

    On a MERGED tank+world frame dominant_class ALONE IS WRONG — an out-of-tank
    capture is neutron-dominated and is still background, because no in-tank
    neutrino made it (see REPORT_ccinc_v3_world_merged.md §0). mva_analysis.py
    persists the training target as `mva_label`; use it when present, else rebuild
    it from origin_in_tank the same way the training did.
    """
    if "mva_label" in d.columns:
        return d["mva_label"].to_numpy(int).astype(bool)
    is_ndom = d["dominant_class"].isin(NEUTRON_CLASSES).to_numpy()
    if "origin_in_tank" in d.columns:
        return np.where(d["origin_in_tank"].to_numpy(int) == 0, False, is_ndom)
    return is_ndom


def emit(pdf, fig, figures_dir, name):
    """Append the page to the multi-page PDF and, if asked, also save it alone."""
    _watermark(fig)
    pdf.savefig(fig)
    if figures_dir is not None:
        stem = Path(figures_dir) / name
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        print(f"      -> {stem.name}.pdf / .png")
    plt.close(fig)


def feature_page(pdf, mc_scores_path, frozen_path, figures_dir=None,
                 title="OPTICS + Random Forest — Top 6 Discriminating Features",
                 name="presentable_features", model="rf"):
    """Top-6 features of `model`, presentable names, signal vs background (MC)."""
    top6, imp_map = top_features(frozen_path, model, 6)

    d = pd.read_parquet(mc_scores_path)
    is_sig = signal_mask(d)

    # Plain features page only: rename pe_total -> "Total cluster charge"
    # (axis + title). The cut page keeps the shared FEATURE_LABELS/_TITLES maps.
    xlabel = dict(FEATURE_LABELS, pe_total="Total cluster charge [p.e.]")
    tlabel = dict(FEATURE_TITLES, pe_total="Total cluster charge")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=16)
    for ax, feat in zip(axes.ravel(), top6):
        v = d[feat].to_numpy(float)
        good = np.isfinite(v)
        bins = feature_bins(v[good])
        ax.hist(v[good & ~is_sig], bins=bins, density=True, label="Background",
                **art("bkg"))
        ax.hist(v[good & is_sig], bins=bins, density=True, label="Neutron",
                **art("sig"))
        ax.set_xlabel(xlabel.get(feat, feat))
        ax.set_ylabel("Normalised Counts")
        ax.set_title(f"Discriminating Feature: {tlabel.get(feat, feat)}", fontsize=12)
        ax.legend(frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    emit(pdf, fig, figures_dir, name)


def top2_feature_pages(pdf, mc_scores_path, frozen_path, figures_dir=None,
                       name_prefix="presentable_features_top2", model="rf"):
    """One full-width landscape page PER top-2 RF feature (wide aspect, big fonts,
    larger framed legend). Same Neutron/Background MC split as the 6-panel page;
    pe_total renamed to 'Total cluster charge' (this page is plain features)."""
    top2, _ = top_features(frozen_path, model, 2)

    d = pd.read_parquet(mc_scores_path)
    is_sig = signal_mask(d)
    xlabel = dict(FEATURE_LABELS, pe_total="Total cluster charge [p.e.]")
    tlabel = dict(FEATURE_TITLES, pe_total="Total cluster charge")

    for feat in top2:
        v = d[feat].to_numpy(float)
        good = np.isfinite(v)
        bins = feature_bins(v[good])

        fig, ax = plt.subplots(figsize=(11, 5.5))   # wide landscape, like attached
        ax.hist(v[good & ~is_sig], bins=bins, density=True, label="Background",
                **art("bkg"))
        ax.hist(v[good & is_sig], bins=bins, density=True, label="Neutron",
                **art("sig"))
        ax.set_xlabel(xlabel.get(feat, feat), fontsize=BIG_LABEL)
        ax.set_ylabel("Normalised Counts", fontsize=BIG_LABEL)
        ax.set_title(f"Discriminating Feature: {tlabel.get(feat, feat)}",
                     fontsize=BIG_TITLE)
        ax.tick_params(axis="both", labelsize=BIG_TICK)
        # bigger framed legend box (larger font + padding). Place it clear of the
        # data: charge peaks on the left -> upper-right; spatial_rms peaks on the
        # right -> upper-left.
        legend_loc = "upper left" if feat == "spatial_rms" else "upper right"
        ax.legend(frameon=True, edgecolor="#999999", fontsize=BIG_LEGEND,
                  borderpad=1.0, labelspacing=0.8, handlelength=2.0,
                  loc=legend_loc)
        fig.tight_layout()
        emit(pdf, fig, figures_dir, f"{name_prefix}_{feat}")


def feature_cut_page(pdf, mc_scores_path, frozen_path, score_cut,
                     figures_dir=None, name="presentable_features_cut",
                     model="rf", title=None):
    """Top-6 features split by the RF 80%-eff cut: clusters PASSING rf_score>=cut
    (kept) vs FAILING (rejected). The cut is a threshold on the RF *score*, not on
    any single feature, so this pass/fail overlay -- not a vertical line -- is the
    correct way to show what the cut does to each feature distribution."""
    top6, _ = top_features(frozen_path, model, 6)

    d = pd.read_parquet(mc_scores_path)
    score_col = f"{model}_score"
    score_cut = resolve_score_cut(d, score_col, score_cut)
    passes = (d[score_col] >= score_cut).to_numpy()
    is_sig = signal_mask(d)
    # Same Neutron/Background truth split as the plain feature page, but restricted
    # to clusters that PASS the RF cut -> shows what survives the 80%-eff selection.
    keep = passes

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title or ("OPTICS + Random Forest — Top 6 Features After the "
                           "80%-Efficiency Cut"), fontsize=16)
    for ax, feat in zip(axes.ravel(), top6):
        v = d[feat].to_numpy(float)
        good = np.isfinite(v)
        bins = feature_bins(v[good])
        ax.hist(v[good & keep & ~is_sig], bins=bins, density=True,
                label="Background", **art("bkg"))
        ax.hist(v[good & keep & is_sig], bins=bins, density=True, label="Neutron",
                **art("sig"))
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel("Normalised Counts")
        ax.set_title(_feat_title(feat), fontsize=12)
        ax.legend(frameon=False)
    fig.text(0.5, 0.005,
             fr"{MODEL_NAMES[model]} score $\geq$ {score_cut:.3f}   "
             fr"(80% neutron efficiency from CC-$\nu$ MC)",
             ha="center", fontsize=12,
             color="black" if STYLE == "bw" else "#b22222")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    emit(pdf, fig, figures_dir, name)


def multiplicity_page(pdf, scored, score_cut, max_n=6, logy=True):
    """OPTICS+RF neutron multiplicity histogram with Poisson sqrt(N) error bars.

    The sqrt(N) error per bar is the SAME Poisson error model the capture-time fit
    uses (basic.py _prepare_fitting_data: ydata_errors = sqrt(counts)). `logy`
    toggles log vs linear y so both versions can be produced."""
    sel = scored[scored["rf_score"] >= score_cut]
    ev_all = scored.groupby(EVENT_KEYS).size().index
    mult = sel.groupby(EVENT_KEYS).size().reindex(ev_all, fill_value=0).to_numpy()
    mult = mult[mult >= 1]                       # events with >=1 tagged neutron

    centers = np.arange(1, max_n + 1)
    heights = np.array([(mult == k).sum() for k in range(1, max_n)] +
                       [(mult >= max_n).sum()])  # last bin is ">= max_n"
    errors = np.sqrt(heights)                     # Poisson error = sqrt(N), as capture-time

    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Empty (step-outline) histogram in the same blue as the capture-time "Data"
    # histtype="step" -- bars centred on each integer, no fill, no top labels.
    edges = np.arange(0.5, max_n + 1.5)           # bin edges around integer centres
    ax.hist(centers, bins=edges, weights=heights, histtype="step",
            color=HIST_COLOR, linewidth=1.5, label="Number of tagged neutrons")
    # Poisson statistical error bar = sqrt(N), drawn in the same blue as the bars
    ax.errorbar(centers, heights, yerr=errors, fmt="none", ecolor=HIST_COLOR,
                elinewidth=1.2, capsize=4, label="Statistical error")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Tagged neutron clusters per event", fontsize=BIG_LABEL)
    ax.set_ylabel("Number of events", fontsize=BIG_LABEL)
    ax.set_title("AmBe neutron multiplicity", fontsize=BIG_TITLE)
    xticklabels = [str(k) for k in range(1, max_n)] + [fr"$\geq {max_n}$"]
    ax.set_xticks(centers)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="both", labelsize=BIG_TICK)
    ax.set_xlim(0.4, max_n + 0.6)
    ax.legend(frameon=True, edgecolor="#999999", loc="upper right",
              fontsize=BIG_LEGEND, borderpad=1.0, labelspacing=0.8,
              handlelength=2.0)
    fig.tight_layout()
    _watermark(fig)
    pdf.savefig(fig)
    plt.close(fig)


def _fit_and_plot_capture(pdf, analyzer, CT_us, title):
    """Fit one CT array with the analyzer's NeutCapture model and draw a styled page.

    Reuses analyzer._prepare_fitting_data + analyzer.NeutCapture + fitting_config so
    the binning/window/bounds are identical to capture_time_stageAB_from_parquet.py.
    Returns (tau, tau_err) or (nan, nan) on failure.
    """
    import lmfit
    cfg = analyzer.fitting_config
    xdata, ydata, yerr, counts, bin_centers = analyzer._prepare_fitting_data(CT_us)
    try:
        model = lmfit.Model(analyzer.NeutCapture)
        params = model.make_params(A=cfg["initial_amplitude"],
                                   therm=cfg["initial_thermal_time"],
                                   tau=cfg["initial_capture_time"],
                                   B=cfg["initial_background"])
        params["A"].min = cfg["amplitude_bounds"][0]
        params["therm"].min, params["therm"].max = cfg["thermal_bounds"]
        params["tau"].min, params["tau"].max = cfg["capture_bounds"]
        params["B"].vary = False
        # leastsq (not basinhopping) -> covariance-based stderr, same as the
        # pipeline's shim in capture_time_stageAB_from_parquet.py
        result = model.fit(ydata, params, t=xdata, weights=1 / yerr, method="leastsq")
        tau = result.params["tau"].value
        tau_err = result.params["tau"].stderr or float("nan")
        therm = result.params["therm"].value
        therm_err = result.params["therm"].stderr or float("nan")
        best = result.best_fit
        redchi = result.redchi
    except Exception as e:                       # pragma: no cover
        print(f"  fit failed for {title}: {e}")
        tau = tau_err = therm = therm_err = redchi = float("nan")
        best = None

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.hist(CT_us, bins=cfg["time_bins"], range=cfg["time_range"],
            histtype="step", color=HIST_COLOR, linewidth=1.3, label="Data")
    ax.errorbar(xdata, ydata, yerr=yerr, color=HIST_COLOR, linestyle="None",
                alpha=0.6, capsize=0)
    if best is not None:
        label = (fr"therm $= {therm:.2f} \pm {therm_err:.2f}\ \mu$s" + "\n"
                 fr"$\tau = {tau:.2f} \pm {tau_err:.2f}\ \mu$s" + "\n"
                 fr"$\chi^2/\mathrm{{ndof}} = {redchi:.2f}$")
        ax.plot(xdata, best, color=FIT_COLOR, linewidth=2.2, label=label)
    ax.set_xlabel(r"Cluster Time [$\mu$s]", fontsize=BIG_LABEL)
    ax.set_ylabel("Counts", fontsize=BIG_LABEL)
    ax.set_title(title, fontsize=BIG_TITLE)
    ax.tick_params(axis="both", labelsize=BIG_TICK)
    ax.legend(frameon=True, edgecolor="#999999", fontsize=CAPTURE_LEGEND,
              borderpad=0.5, labelspacing=0.4, handlelength=1.6)
    fig.tight_layout()
    _watermark(fig)
    pdf.savefig(fig)
    plt.close(fig)
    return {"tau": tau, "tau_err": tau_err, "therm": therm, "therm_err": therm_err,
            "redchi": redchi}


def capture_pages(pdf, scored, score_cut):
    """Per-position + combined capture-time pages, OPTICS+RF cut."""
    analyzer = AmBeNeutronAnalyzer(
        data_directory="./EventAmBeNeutronCandidatesData/",
        output_pdf="/tmp/_unused_presentable.pdf")
    analyzer.update_fitting_config(capture_bounds=(2.0, 70.0))   # same as pipeline

    sel = scored[scored[ "rf_score"] >= score_cut].copy()
    pos = sel["run"].map(lambda r: SOURCE_POSITIONS.get(int(r)))
    sel = sel[pos.notna()].copy()
    sel["src"] = pos[pos.notna()].map(
        lambda t: (round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)))
    sel["CT_us"] = sel["t_mean"] / 1000.0

    # per position — collect each fit for the inverse-variance combination
    per_pos = []
    for src, g in sorted(sel.groupby("src"), key=lambda kv: kv[0]):
        sx, sy, sz = (int(v) for v in src)
        title = "AmBe neutron thermal and capture time"
        r = _fit_and_plot_capture(pdf, analyzer, g["CT_us"], title)
        r["src"] = (sx, sy, sz)
        per_pos.append(r)
        print(f"  position ({sx},{sy},{sz})  n={len(g):5d}  tau={r['tau']:.2f} us")

    # ---- inverse-variance weighted mean of the 21 per-position tau (and therm) ----
    # Exactly basic.py:weighted_average: w_i = 1/sig_i^2 ;  abar = sum(a_i w_i)/sum(w_i)
    # ;  sig = sqrt(1/sum(w_i)).  Drops positions with non-finite / zero error.
    def _wmean(vals, errs):
        vals = np.asarray(vals, float); errs = np.asarray(errs, float)
        ok = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
        if not ok.any():
            return float("nan"), float("nan"), 0
        w = 1.0 / errs[ok] ** 2
        return float((vals[ok] * w).sum() / w.sum()), float(np.sqrt(1.0 / w.sum())), int(ok.sum())

    tau_w, tau_w_err, n_used = _wmean([r["tau"] for r in per_pos],
                                      [r["tau_err"] for r in per_pos])
    therm_w, therm_w_err, _ = _wmean([r["therm"] for r in per_pos],
                                     [r["therm_err"] for r in per_pos])
    print(f"  weighted mean over {n_used} positions: tau = {tau_w:.2f} +/- {tau_w_err:.2f} us, "
          f"therm = {therm_w:.2f} +/- {therm_w_err:.2f} us")

    # ---- Combined page A: pooled histogram + fit curve, weighted-mean tau in legend ----
    _combined_pooled_page(pdf, analyzer, sel["CT_us"], tau_w, tau_w_err,
                          therm_w, therm_w_err, n_used)
    print(f"  combined pooled  n={len(sel)}")

    # ---- Combined page B: tau-vs-position summary with weighted-mean band ----
    _tau_by_position_page(pdf, per_pos, tau_w, tau_w_err, n_used)


def _combined_pooled_page(pdf, analyzer, CT_us, tau_w, tau_w_err,
                          therm_w, therm_w_err, n_used):
    """Pooled-histogram combined page; legend tau = inverse-variance weighted mean
    of the 21 per-position fits (NOT the pooled-fit tau). The green curve is still
    the pooled-shape fit, drawn as a visual guide."""
    import lmfit
    cfg = analyzer.fitting_config
    xdata, ydata, yerr, counts, bc = analyzer._prepare_fitting_data(CT_us)
    model = lmfit.Model(analyzer.NeutCapture)
    params = model.make_params(A=cfg["initial_amplitude"], therm=cfg["initial_thermal_time"],
                               tau=cfg["initial_capture_time"], B=cfg["initial_background"])
    params["A"].min = cfg["amplitude_bounds"][0]
    params["therm"].min, params["therm"].max = cfg["thermal_bounds"]
    params["tau"].min, params["tau"].max = cfg["capture_bounds"]
    params["B"].vary = False
    result = model.fit(ydata, params, t=xdata, weights=1 / yerr, method="leastsq")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.hist(CT_us, bins=cfg["time_bins"], range=cfg["time_range"],
            histtype="step", color=HIST_COLOR, linewidth=1.3, label="Data (all positions)")
    ax.errorbar(xdata, ydata, yerr=yerr, color=HIST_COLOR, linestyle="None",
                alpha=0.6, capsize=0)
    label = (fr"therm $= {therm_w:.2f} \pm {therm_w_err:.2f}\ \mu$s" + "\n"
             fr"$\tau = {tau_w:.2f} \pm {tau_w_err:.2f}\ \mu$s")
    ax.plot(xdata, result.best_fit, color=FIT_COLOR, linewidth=2.2, label=label)
    ax.set_xlabel(r"Cluster Time [$\mu$s]", fontsize=BIG_LABEL)
    ax.set_ylabel("Counts", fontsize=BIG_LABEL)
    ax.set_title("AmBe neutron thermal and capture time", fontsize=BIG_TITLE)
    ax.tick_params(axis="both", labelsize=BIG_TICK)
    ax.legend(frameon=True, edgecolor="#999999", fontsize=CAPTURE_LEGEND,
              borderpad=0.5, labelspacing=0.4, handlelength=1.6)
    fig.tight_layout()
    _watermark(fig)
    pdf.savefig(fig)
    plt.close(fig)


def _tau_by_position_page(pdf, per_pos, tau_w, tau_w_err, n_used):
    """tau_i +/- sigma_i for each position + horizontal weighted-mean band."""
    pts = [(i, r) for i, r in enumerate(per_pos)
           if np.isfinite(r["tau"]) and np.isfinite(r["tau_err"]) and r["tau_err"] > 0]
    x = np.arange(len(pts))
    taus = [r["tau"] for _, r in pts]
    errs = [r["tau_err"] for _, r in pts]
    labels = [f"({r['src'][0]},{r['src'][1]},{r['src'][2]})" for _, r in pts]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.errorbar(x, taus, yerr=errs, fmt="o", color=HIST_COLOR, capsize=4,
                markersize=6, label=r"per-position $\tau_i \pm \sigma_i$")
    ax.axhline(tau_w, color=FIT_COLOR, linewidth=2.2,
               label=fr"weighted mean $\tau = {tau_w:.2f} \pm {tau_w_err:.2f}\ \mu$s "
                     fr"({n_used} positions)")
    ax.axhspan(tau_w - tau_w_err, tau_w + tau_w_err, color=FIT_COLOR, alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=11)
    ax.set_ylabel(r"Capture time $\tau$ [$\mu$s]", fontsize=BIG_LABEL)
    ax.set_xlabel("Source position (x, y, z) [cm]", fontsize=BIG_LABEL)
    ax.set_title("AmBe neutron capture time by position", fontsize=BIG_TITLE)
    ax.tick_params(axis="y", labelsize=BIG_TICK)
    ax.legend(frameon=True, edgecolor="#999999", fontsize=BIG_LEGEND,
              loc="upper right", borderpad=1.0, labelspacing=0.8, handlelength=2.0)
    fig.tight_layout()
    _watermark(fig)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mc-scores", required=True)
    p.add_argument("--frozen", required=True)
    p.add_argument("--style", required=True, choices=sorted(STYLE_RC),
                   help="poster = light blue signal over light grey background (the "
                        "established look); bw = black and white. Always explicit.")
    p.add_argument("--font", default="helvetica", choices=sorted(FONT_RC),
                   help="helvetica (default, Nimbus Sans incl. mathtext) or dejavu "
                        "to reproduce the June-16 figures exactly. Echoed at startup.")
    p.add_argument("--scored",
                   help="AmBe scored parquet. Needed for the multiplicity and "
                        "capture-time pages; omit it together with --mc-only.")
    p.add_argument("--mc-only", action="store_true",
                   help="Produce the MC feature pages only, no AmBe data pages. "
                        "Required when --scored is not given, so a missing input can "
                        "never silently drop pages.")
    p.add_argument("--rank-model", required=True, choices=sorted(MODEL_NAMES),
                   help="Which trained model's importances order the features, and "
                        "whose score the cut page uses. Pick the best-performing one "
                        "for the run; it is read from the frozen bundle, never refit.")
    p.add_argument("--score-cut", default="auto",
                   help="Score threshold for the cut page, or 'auto' (default) for "
                        "the 80%%-signal-efficiency threshold read off the score "
                        "table.")
    p.add_argument("--title",
                   help="Suptitle for the top-6 feature page. Defaults to the "
                        "OPTICS + Random Forest wording.")
    p.add_argument("--name-prefix", default="presentable",
                   help="Basename prefix for the per-figure files written into "
                        "--figures-dir.")
    p.add_argument("--figures-dir",
                   help="Also write every page as its own standalone PDF + PNG here.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.scored is None and not args.mc_only:
        p.error("--scored is missing: pass it, or pass --mc-only to say you want "
                "the MC feature pages only.")
    if args.mc_only and args.scored:
        p.error("--mc-only and --scored are contradictory; drop one.")

    apply_style(args.style, args.font)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figs = Path(args.figures_dir) if args.figures_dir else None
    if figs:
        figs.mkdir(parents=True, exist_ok=True)
    pfx = args.name_prefix
    mname = MODEL_NAMES[args.rank_model]
    title = args.title or f"OPTICS + {mname} — Top 6 Discriminating Features"
    cut_title = f"{title.split(' — ')[0]} — Top 6 Features After the 80%-Efficiency Cut"

    n_stages = 2 if args.mc_only else 4
    with PdfPages(out) as pdf:
        print(f"[1/{n_stages}] RF top-6 feature page ...")
        feature_page(pdf, args.mc_scores, args.frozen, figures_dir=figs,
                     title=title, name=f"{pfx}_features", model=args.rank_model)
        print("      + top-2 features, full-width landscape (1 per page) ...")
        top2_feature_pages(pdf, args.mc_scores, args.frozen, figures_dir=figs,
                           name_prefix=f"{pfx}_features_top2",
                           model=args.rank_model)
        print(f"[2/{n_stages}] RF cut-applied feature page ...")
        feature_cut_page(pdf, args.mc_scores, args.frozen, args.score_cut,
                         figures_dir=figs, name=f"{pfx}_features_cut",
                         model=args.rank_model, title=cut_title)
        if not args.mc_only:
            scored = pd.read_parquet(args.scored)
            print(f"[3/{n_stages}] multiplicity pages (log-y + linear-y) ...")
            data_cut = float(args.score_cut) if args.score_cut != "auto" else 0.547
            multiplicity_page(pdf, scored, data_cut, logy=True)
            multiplicity_page(pdf, scored, data_cut, logy=False)
            print(f"[4/{n_stages}] capture-time pages (per position + 2 combined) ...")
            capture_pages(pdf, scored, data_cut)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
