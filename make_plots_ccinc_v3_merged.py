"""
make_plots_ccinc_v3_merged.py
=============================
Figures for the CCinc v3 campaign — two selection streamlines (truth-tag vs
reco-tag), the world-volume background, and the merged tank+world MVA trainings.

Output format follows slide_plots_ccinc_xi02/: **plain standalone figures**, one PDF
(plus a PNG at 200 dpi) per plot, to be dropped into a presentation. Figure title,
panel titles, axis labels and legends only — no slide furniture, no gridlines, no
annotations. Helvetica metrics throughout (Nimbus Sans, mathtext included) and the
established palette: light blue for signal / neutron / merged, light grey for
background / the tank-only baseline. The selection cut-flow and the streamline
overlap are TABLES, not bar charts.

Every number is read live from
    /exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/<run>/{csv,parquet}
and from the training logs in this directory. The only hardcoded thing is the
labelling rule, which is a physics decision documented in
REPORT_ccinc_v3_world_merged.md §0.

The per-feature signal-vs-background distributions are NOT here: they come from
make_presentable_optics_rf_plots.py, which owns that plot (top-6 by the chosen model's
importances, "Neutron" vs "Background", one panel per feature). Run it with
--style bw --rank-model <best model for the run> --mc-only.

Merged artifacts may not exist yet (the four trainings are slow). Panels that depend
on them are drawn empty and labelled "pending", so the script can be re-run as each
training lands.

Run:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python make_plots_ccinc_v3_merged.py
Output:
    slide_plots_ccinc_v3_merged/
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Helvetica metrics: Nimbus Sans is the URW Helvetica clone, the only Helvetica-metric
# face on this machine. Set for words AND mathtext, or mathtext silently falls back to
# DejaVu and the two don't match. Verify with `pdffonts` on any output.
_HELV = ["Nimbus Sans", "Helvetica", "Nimbus Sans L", "Liberation Sans", "Arial"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _HELV,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans",
    "axes.grid": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
})

# One palette for the whole set, matching the established presentable figures:
# light blue = signal / neutron / the new (merged) thing, light grey = background /
# the old baseline. The other two hues only appear where a third and fourth series
# genuinely exist, so every figure stays readable and they stay distinguishable
# from each other.
BLUE   = "#1f77b4"     # signal, neutron, merged, truth-tag, tank
GREY   = "#999999"     # background, tank-only baseline
ORANGE = "#e08214"     # reco-tag, highlighted (geometry) features
GREEN  = "#2ca02c"     # world sample
PURPLE = "#7b3294"     # 4th model on the ROC
LBLUE  = "#cfe3f2"     # table header / box fills
LGREY  = "#f2f2f2"     # alternating table rows

# Two-series bars.
BAR_A = dict(color=BLUE, alpha=0.85, edgecolor="none")        # first series
BAR_B = dict(color=GREY, alpha=0.55, edgecolor="none")        # second series
BAR_W = dict(color=GREEN, alpha=0.70, edgecolor="none")       # world
BAR_C = dict(color=GREY, alpha=0.55, edgecolor="none")        # neutral / baseline
# Signal / background histograms: light blue over light grey, as in the past.
HIST_BKG = dict(color=GREY, alpha=0.55)
HIST_SIG = dict(color=BLUE, alpha=0.85)
# Line plots: colour per series, dashed for the validation twin.
LINE_COLORS = [BLUE, ORANGE, GREEN, PURPLE]

HERE = Path(__file__).parent
BASE = Path("/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output")
OUTD = HERE / "slide_plots_ccinc_v3_merged"
PREFIX = "V3MERGED__"

NEUTRON_CLASSES = [1, 2, 3, 4]
PROMPT_WINDOW_NS = 10000.0

STREAMS = ["truthtag", "recotag"]
TANK  = {s: f"cc_neutrino_v3_{s}" for s in STREAMS}
WORLD = {s: f"cc_neutrino_v3world_{s}" for s in STREAMS}
METHODS = {"optics": "OPTICS", "clusterfinder": "ClusterFinder"}
MSFX = {"optics": "optics", "clusterfinder": "cf"}
MODELS = ["Random Forest", "GBT", "XGBoost", "Neural Network"]
MODEL_SHORT = {"Random Forest": "RF", "GBT": "GBT", "XGBoost": "XGB",
               "Neural Network": "NN"}
CONFIGS = [("truthtag", "optics"), ("truthtag", "clusterfinder"),
           ("recotag", "optics"), ("recotag", "clusterfinder")]
CFG_LBL = {("truthtag", "optics"): "Truth-tag / OPTICS",
           ("truthtag", "clusterfinder"): "Truth-tag / ClusterFinder",
           ("recotag", "optics"): "Reco-tag / OPTICS",
           ("recotag", "clusterfinder"): "Reco-tag / ClusterFinder"}

# 31 features actually used by the merged trainings (d_source/d_source_fit fall
# below the 50% non-NaN threshold on the merged frame).
FEATURES = [
    "n_hits", "pe_total", "n_hits_early",
    "sigma_t_mad", "sigma_t_mad_corr", "sigma_t_mad_tof", "sigma_t_early_mad",
    "t_window_80pct", "pe_balance", "charge_bal_legacy", "spatial_rms",
    "d_wall", "vtx_y", "beta1", "beta2", "beta3", "beta4", "beta5",
    "fit_converged", "n_fit_hits", "fit_rms_ns", "fit_goodness_reco",
    "fit_goodness_init", "d_wall_fit", "sigma_t_mad_tof_fit",
    "beta1_fit", "beta2_fit", "beta3_fit", "beta4_fit", "beta5_fit",
    "vtx_fit_y",
]
GEOMETRY_FEATURES = ["d_wall", "d_wall_fit", "vtx_y", "vtx_fit_y"]

FEATURE_AXIS = {
    "n_hits": "Cluster hits", "n_hits_early": "Early hits",
    "n_fit_hits": "Hits used in the vertex fit",
    "pe_total": "Total cluster charge [p.e.]",
    "charge_bal_legacy": "Charge balance", "pe_balance": "Charge balance (PE)",
    "spatial_rms": "Spatial RMS of hit PMTs [m]",
    "beta1": r"$\beta_1$", "beta2": r"$\beta_2$", "beta3": r"$\beta_3$",
    "beta4": r"$\beta_4$", "beta5": r"$\beta_5$",
    "fit_rms_ns": "Fit timing RMS [ns]",
    "fit_goodness_init": "Fit goodness (centroid vertex)",
    "fit_goodness_reco": "Fit goodness (fitted vertex)",
    "t_window_80pct": "80% time window [ns]",
    "sigma_t_mad": r"$\sigma_t$ (MAD) [ns]",
    "d_wall": "Distance to wall [m]", "d_wall_fit": "Distance to wall, fitted [m]",
    "vtx_y": "Vertex Y [m]", "vtx_fit_y": "Fitted vertex Y [m]",
    "fit_converged": "Vertex fit converged",
}

# n_origin_<species>: BACKGROUND (non-neutron) hits only, complete chain,
# pure-EM excluded — see cluster_features.py, "first-non-EM-ancestor
# composition, background hits only".
ORIGIN_COLS = {
    "n_origin_muminus": r"$\mu^-$",
    "n_origin_muplus":  r"$\mu^+$",
    "n_origin_piplus":  r"$\pi^+$",
    "n_origin_piminus": r"$\pi^-$",
    "n_origin_pizero":  r"$\pi^0$",
    "n_origin_proton":  "p",
    "n_origin_neutron": "n",
    "n_origin_kplus":   r"$K^+$",
    "n_origin_kminus":  r"$K^-$",
    "n_origin_other":   "other",
}

PRETTY_ORIGIN = {
    "mu-": r"$\mu^-$", "mu+": r"$\mu^+$",
    "pi+": r"$\pi^+$", "pi-": r"$\pi^-$", "pi0": r"$\pi^0$",
    "p": "p", "n": "n", "K+": r"$K^+$", "K-": r"$K^-$",
    "K0L": r"$K^0_L$", "K0S": r"$K^0_S$", "gamma": r"$\gamma$",
    "e-": r"$e^-$", "e+": r"$e^+$", "d": "d", "t": "t",
    "pure_em (no non-EM ancestor)": r"pure EM ($\gamma$)",
}

# Canonical cut order across both streamlines (they share the kinematic block and
# diverge only in the tagging block, so a merged order is well defined).
CUT_ORDER = [
    "FV r < 100 cm", "FV |Y| < 100 cm", "p_mu in [600,1200) MeV/c",
    "cos(theta) > 0.8", "trueCC==1", "FSL is muon (trueFSLPdg==13)",
    "NoVeto == 1", "MRD-tagged (any MRD activity)", "promptPE in [500,3000)",
    "nhits >= 4",
]
CUT_SHORT = {
    "FV r < 100 cm": "FV  r < 100 cm",
    "FV |Y| < 100 cm": "FV  |Y| < 100 cm",
    "p_mu in [600,1200) MeV/c": r"$p_\mu\in[600,1200)$ MeV/c",
    "cos(theta) > 0.8": r"$\cos\theta > 0.8$",
    "trueCC==1": "trueCC == 1",
    "FSL is muon (trueFSLPdg==13)": "FSL is a muon",
    "NoVeto == 1": "FMV: NoVeto",
    "MRD-tagged (any MRD activity)": "MRD-tagged",
    "promptPE in [500,3000)": "promptPE 500-3000",
    "nhits >= 4": r"$n_{hits}\geq 4$",
}


# ════════════════════════════════════════════════════════════════════════════
# figure helpers
# ════════════════════════════════════════════════════════════════════════════
def bare(ax):
    """No grid, no top/right spines — the reference figures' look."""
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    """One standalone PDF + PNG per figure, ready to paste into a slide."""
    stem = OUTD / f"{PREFIX}{name}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  [fig] {stem.name}.pdf / .png")


def table_axes(ax, cell, cols, widths=None, fs=10.5, emphasise_last=True,
               fill=True):
    """
    A readable table as a figure: light blue header, alternating row shading, thin
    grey rules, and the final (total) row tinted so it reads as the bottom line.
    """
    ax.axis("off")
    kw = dict(cellText=cell, colLabels=cols, cellLoc="center", colWidths=widths)
    # bbox=[0,0,1,1] makes the table fill its axes exactly, which is what keeps the
    # figure from being mostly whitespace.
    t = ax.table(**kw, bbox=[0, 0, 1, 1]) if fill else ax.table(**kw, loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    if not fill:
        t.scale(1, 1.55)
    last = len(cell)
    for (r, c), cell_obj in t.get_celld().items():
        cell_obj.set_edgecolor("#cccccc")
        cell_obj.set_linewidth(0.6)
        if r == 0:
            cell_obj.set_facecolor(LBLUE)
            cell_obj.set_text_props(fontweight="bold", color="#1a1a1a")
        elif emphasise_last and r == last:
            cell_obj.set_facecolor(LBLUE)
            cell_obj.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell_obj.set_facecolor(LGREY)
        else:
            cell_obj.set_facecolor("white")
        if c == 0 and r > 0:
            cell_obj.set_text_props(ha="left")
            cell_obj.PAD = 0.04
    return t


def pending(ax, label):
    ax.set_title(label, fontsize=11)
    ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
            va="center", fontsize=12, color="0.6")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ════════════════════════════════════════════════════════════════════════════
# loaders
# ════════════════════════════════════════════════════════════════════════════
def cutflow(run: str) -> pd.DataFrame:
    hits = sorted((BASE / run / "csv").glob(f"{run}__cc_*_cutflow.csv"))
    if not hits:
        raise FileNotFoundError(f"no cut-flow csv for {run}")
    d = pd.read_csv(hits[0])
    d["cut"] = d["cut"].str.replace('"', "", regex=False)
    return d


def hit_ancestry(run: str, population="delayed", axis="origin") -> pd.DataFrame:
    d = pd.read_csv(BASE / run / "csv" / f"{run}__bkg_hit_ancestry.csv")
    d = d[(d["population"] == population) & (d["axis"] == axis)].copy()
    d["label"] = d["category"].map(lambda c: PRETTY_ORIGIN.get(c, c))
    return d.sort_values("pct_of_denominator", ascending=False)


def feature_separation(run: str) -> pd.DataFrame:
    return pd.read_csv(BASE / run / "csv" / f"{run}__feature_separation.csv")


def importances(run: str, method: str, merged: bool) -> pd.DataFrame | None:
    tag = "__merged" if merged else ""
    p = (BASE / run / "csv" /
         f"{run}__mva_summary__keepprompt{tag}__{MSFX[method]}.csv")
    return pd.read_csv(p) if p.exists() else None


def world_outoftank_origin(stream="truthtag") -> pd.DataFrame:
    """
    Delayed (t > 10 us) non-neutron hits from OUT-OF-TANK interactions, by origin
    species.  This is the population the merge adds and it is not in any pipeline
    CSV (those are scoped to cc_pass events), so it is computed here from the
    per-hit table and cached.

    Non-neutron background hit == truth_class == -5, exactly as
    report_background_composition.py defines it.  No cc_pass gating: the
    out-of-tank harvest is unconditional by design (report §0).
    """
    cache = OUTD / f"cache_world_outoftank_origin_{stream}.csv"
    if cache.exists():
        d = pd.read_csv(cache)
        d["label"] = d["category"].map(lambda c: PRETTY_ORIGIN.get(c, c))
        return d

    from collections import Counter
    from src.ambe.mc.processor import pdg_short
    path = BASE / WORLD[stream] / "parquet" / f"{WORLD[stream]}__pulses.parquet"
    pf = pq.ParquetFile(str(path))
    cols = ["t", "truth_class", "origin_pdg", "origin_in_tank"]
    cnt: Counter = Counter()
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=cols).to_pandas()
        m = ((d["t"].to_numpy(float) > PROMPT_WINDOW_NS)
             & (d["truth_class"].to_numpy(int) == -5)
             & (d["origin_in_tank"].to_numpy(int) == 0))
        if m.any():
            for o, n in d.loc[m, "origin_pdg"].value_counts().items():
                cnt[int(o)] += int(n)
    tot = sum(cnt.values())
    rows = [{"category": ("pure_em (no non-EM ancestor)" if p == -5 else pdg_short(p)),
             "n_hits": n,
             "pct_of_denominator": round(100.0 * n / max(tot, 1), 3)}
            for p, n in cnt.most_common()]
    d = pd.DataFrame(rows)
    d.to_csv(cache, index=False)
    print(f"  [cache] world out-of-tank delayed background hits: {tot:,}")
    d["label"] = d["category"].map(lambda c: PRETTY_ORIGIN.get(c, c))
    return d


def overlap() -> dict:
    """Event-level overlap of the two streamlines, from the persisted event tables."""
    cache = OUTD / "cache_stream_overlap.csv"
    if cache.exists():
        return pd.read_csv(cache).iloc[0].to_dict()

    def ev(run):
        p = next((BASE / run / "parquet").glob(f"{run}__cc_*_events.parquet"))
        d = pq.read_table(str(p), columns=["_source_file", "eventID", "cc_pass"]).to_pandas()
        d["key"] = d["_source_file"].astype(str) + "#" + d["eventID"].astype(str)
        return set(d.loc[d["cc_pass"].astype(bool), "key"])

    a, b = ev(TANK["truthtag"]), ev(TANK["recotag"])
    out = {"truthtag": len(a), "recotag": len(b), "both": len(a & b),
           "truthtag_only": len(a - b), "recotag_only": len(b - a),
           "union": len(a | b)}
    pd.DataFrame([out]).to_csv(cache, index=False)
    return out


def merged_frame(stream: str, method: str) -> pd.DataFrame:
    """
    Rebuild the merged training frame with the documented labelling rule
    (REPORT_ccinc_v3_world_merged.md §0).  Totals are printed so they can be
    checked against logs_merged_tankworld.log.
    """
    keep = (["method", "cc_pass", "dominant_class", "eventID", "is_prompt_cluster",
             "origin_dominant_species", "frac_neutron", "frac_untraced",
             "n_nonneutron", "n_neutron", "n_origin_traced"]
            + FEATURES + list(ORIGIN_COLS))

    def load(run):
        p = BASE / run / "parquet" / f"{run}__cluster_features.parquet"
        avail = set(pq.ParquetFile(str(p)).schema_arrow.names)
        cols = [c for c in keep + ["origin_in_tank", "origin_in_fv"] if c in avail]
        d = pq.read_table(str(p), columns=cols).to_pandas()
        d = d[d["method"] == method].copy()
        d["_source_run"] = run
        for c in ("origin_in_tank", "origin_in_fv"):
            if c not in d.columns:
                d[c] = -1
        return d

    t = load(TANK[stream])
    w = load(WORLD[stream])

    # cc filter: tank requires cc_pass; world keeps out-of-tank unconditionally
    t = t[t["cc_pass"] == 1]
    w = w[(w["cc_pass"] == 1) | (w["origin_in_tank"] == 0)]

    d = pd.concat([t, w], ignore_index=True)
    d["is_neutron_dom"] = d["dominant_class"].isin(NEUTRON_CLASSES)

    # drop in-tank-but-outside-FV (phase-space homogeneity with the tank signal)
    d = d[~((d["origin_in_tank"] == 1) & (d["origin_in_fv"] == 0))].reset_index(drop=True)

    in_tank = d["origin_in_tank"].to_numpy(int)
    d["label"] = np.where(in_tank == 0, 0,                  # out-of-tank -> background
                          d["is_neutron_dom"].astype(int))  # else composition
    d["is_world"] = d["_source_run"].str.contains("world")
    return d


def _sep_sigma(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(0.5 * (a.var() + b.var()))
    return abs(a.mean() - b.mean()) / s if s > 0 else np.nan


def separations(d: pd.DataFrame) -> pd.Series:
    sig = d[d["label"] == 1]
    bkg = d[d["label"] == 0]
    return pd.Series({f: _sep_sigma(sig[f], bkg[f])
                      for f in FEATURES if f in d.columns}).dropna() \
             .sort_values(ascending=False)


def dump_tables(frames: dict) -> None:
    """
    Write the two tables the report quotes: the per-particle composition of the
    training background class, and the feature separations. Same numbers the
    figures use, so the report and the plots can never drift apart.
    """
    rows = []
    for (s, meth), d in frames.items():
        b = d[d["label"] == 0]
        for part, sub in (("all", b), ("tank", b[~b["is_world"]]),
                          ("world_outoftank", b[b["is_world"]])):
            tot = sum(float(sub[c].sum()) for c in ORIGIN_COLS if c in sub.columns)
            nh = float(sub["n_hits"].sum())
            r = {"stream": s, "method": meth, "part": part, "clusters": len(sub),
                 "neutron_dominated": int(sub["is_neutron_dom"].sum()),
                 "hits": int(nh),
                 "pct_hits_neutron": round(100 * sub["n_neutron"].sum() / nh, 2),
                 "pct_hits_nonneutron": round(100 * sub["n_nonneutron"].sum() / nh, 2),
                 "traced_nonneutron_hits": int(tot)}
            for c in ORIGIN_COLS:
                if c in sub.columns:
                    r["pct_" + c.replace("n_origin_", "")] = \
                        round(100 * float(sub[c].sum()) / max(tot, 1.0), 2)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUTD / "table_merged_background_composition.csv",
                              index=False)
    pd.DataFrame({f"{s}_{meth}": separations(d)
                  for (s, meth), d in frames.items()}).round(3).to_csv(
        OUTD / "table_merged_feature_separation.csv")
    print("  [csv] table_merged_background_composition.csv, "
          "table_merged_feature_separation.csv")


# ════════════════════════════════════════════════════════════════════════════
# log parsing — AUCs and NN training histories live only in the logs
# ════════════════════════════════════════════════════════════════════════════
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_EPOCH = re.compile(
    r"auc:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*val_auc:\s*([0-9.]+)"
    r"\s*-\s*val_loss:\s*([0-9.]+)")
_AUC = re.compile(r"^\s*AUC\s+(\S.*?)\s*:\s*test=([0-9.]+)(?:\s+CV=([0-9.]+)"
                  r"±([0-9.]+))?")
# Fallback: the per-model line printed before the summary block. Needed because a
# training can die in matplotlib/Tk teardown after the AUCs are computed and the score
# table is written but before the summary prints (merged truthtag/clusterfinder,
# exit 134 on 2026-08-05).
_AUC_EARLY = re.compile(r"^\[mva\] AUC\s+(\S.*?)\s*=\s*([0-9.]+)")
_CV_FOLD = re.compile(r"\[cv fold (\d+)/(\d+)\]\s+(.*)")
_PIPE_METHOD = re.compile(r"\[pipeline\] scores -> .*__(optics|cf)\.parquet")
_MERGED_HDR = re.compile(r"#+ MERGED TRAIN (\w+) / (\w+)")


def parse_log_blocks(path: Path) -> list[dict]:
    """Split a training log into per-training blocks."""
    if not path.exists():
        return []
    text = _ANSI.sub("", path.read_text(errors="replace"))
    lines = text.splitlines()

    starts = [i for i, l in enumerate(lines) if l.startswith("[mva] loading")]
    if not starts:
        return []
    bounds = list(zip(starts, starts[1:] + [len(lines)]))

    blocks = []
    for bi, (s, e) in enumerate(bounds):
        chunk = lines[s:e]
        stream, method, explicit = None, None, False
        for l in lines[max(0, s - 40):s]:
            m = _MERGED_HDR.search(l)
            if m:
                stream, method, explicit = m.group(1), m.group(2), True
        for l in chunk:
            m = _PIPE_METHOD.search(l)
            if m:
                method = "clusterfinder" if m.group(1) == "cf" else "optics"
                explicit = True
        if method is None:
            method = ["optics", "clusterfinder"][bi % 2]
        for l in chunk:
            if "cc_neutrino_v3_truthtag" in l:
                stream = stream or "truthtag"
            if "cc_neutrino_v3_recotag" in l:
                stream = stream or "recotag"

        hist = [m.groups() for m in (_EPOCH.search(l) for l in chunk) if m]
        hdf = (pd.DataFrame([[float(x) for x in g] for g in hist],
                            columns=["auc", "loss", "val_auc", "val_loss"])
               if hist else pd.DataFrame())
        if len(hdf):
            hdf.insert(0, "epoch", np.arange(1, len(hdf) + 1))

        aucs, cvs = {}, {}
        for l in chunk:                     # 3-decimal fallback first …
            m = _AUC_EARLY.match(l)
            if m:
                aucs[m.group(1).strip()] = float(m.group(2))
        for l in chunk:                     # … then the 4-decimal summary wins
            m = _AUC.match(l)
            if m:
                aucs[m.group(1).strip()] = float(m.group(2))
                if m.group(3):
                    cvs[m.group(1).strip()] = (float(m.group(3)), float(m.group(4)))

        # per-fold cross-validation AUCs: the only place the spread is recorded
        folds = {}
        for l in chunk:
            m = _CV_FOLD.search(l)
            if m:
                for kv in m.group(3).split():
                    k, _, v = kv.partition("=")
                    if v:
                        folds.setdefault(k, []).append(float(v))

        # Pre-capping class counts. They identify the (stream, method) uniquely, which
        # is how a re-run log with no method header gets attributed correctly — a
        # direct `python mva_analysis.py` call prints neither the launcher header nor
        # the pipeline rename line.
        counts = None
        for l in chunk:
            m = re.search(r"\[mva\] (\d+) clusters\s+\|\s+signal=(\d+)\s+"
                          r"background=(\d+)", l)
            if m:
                counts = (int(m.group(2)), int(m.group(3)))

        best = stopped = None
        for l in chunk:
            m = re.search(r"best epoch:\s*(\d+)", l)
            if m:
                best = int(m.group(1))
            m = re.search(r"NN stopped at epoch (\d+)", l)
            if m:
                stopped = int(m.group(1))

        blocks.append(dict(stream=stream, method=method, method_explicit=explicit,
                           counts=counts, aucs=aucs, cv=cvs, folds=folds,
                           history=hdf, best_epoch=best, stopped_epoch=stopped,
                           complete=bool(aucs)))
    return blocks


def collect_trainings(frames: dict | None = None) -> dict:
    """
    key (scope, stream, method) -> block, scope in {'tank','merged'}.

    `frames` (the rebuilt merged frames) lets a block whose log does not state the
    method be attributed by its class counts, which are unique per configuration.
    """
    by_counts = {}
    for (st, meth), d in (frames or {}).items():
        by_counts[(int((d["label"] == 1).sum()), int((d["label"] == 0).sum()))] = \
            (st, meth)
    out = {}
    for stream in STREAMS:
        for b in parse_log_blocks(HERE / f"logs_full_v3_{stream}.log"):
            out[("tank", b["stream"] or stream, b["method"])] = b
    # Merged trainings can live in more than one log: a re-run gets its own file, and
    # the later log must win. logs_merged_tankworld_run1.log is deliberately excluded —
    # it predates the 2026-08-05 09:53 FV-drop fix and its labels are wrong.
    merged_logs = sorted(
        (f for f in HERE.glob("logs_merged_*.log")
         if f.name != "logs_merged_tankworld_run1.log"),
        key=lambda f: f.stat().st_mtime)
    for path in merged_logs:
        for b in parse_log_blocks(path):
            if not b["stream"]:
                continue
            b["log"] = path.name
            if not b["method_explicit"] and b["counts"] in by_counts:
                st, meth = by_counts[b["counts"]]
                if (st, meth) != (b["stream"], b["method"]):
                    print(f"   [log] {path.name}: no method header — attributed to "
                          f"{st}/{meth} by its class counts {b['counts']}")
                b["stream"], b["method"] = st, meth
            out[("merged", b["stream"], b["method"])] = b
    return out


def auc_of(tr, scope, stream, method, model):
    b = tr.get((scope, stream, method))
    return None if not b else b["aucs"].get(model)


# ════════════════════════════════════════════════════════════════════════════
# figures — selection
# ════════════════════════════════════════════════════════════════════════════
def _cutflow_rows(runs):
    """(rows, columns) for one sample's cut-flow, both streamlines side by side."""
    cfs = {st: cutflow(runs[st]).set_index("cut") for st in STREAMS}
    cuts = [c for c in CUT_ORDER if any(c in cfs[st].index for st in STREAMS)]
    rows = []
    for c in cuts:
        row = [CUT_SHORT.get(c, c)]
        for st in STREAMS:
            if c in cfs[st].index:
                r = cfs[st].loc[c]
                row += [f"{int(r['events_after']):,}",
                        f"{r['pct_of_previous']:.1f}%",
                        f"{r['pct_of_total']:.2f}%"]
            else:
                row += ["—", "—", "—"]
        rows.append(row)
    return rows


def fig_cutflow_table():
    """
    The selection as a TABLE, not a bar chart: one block per sample, truth-tag and
    reco-tag side by side, with both the step efficiency and the cumulative one.
    The last row is the final cc_pass.
    """
    cols = ["Cut", "Truth-tag", "% prev", "% total", "Reco-tag", "% prev", "% total"]
    widths = [0.28, 0.13, 0.09, 0.09, 0.13, 0.09, 0.09]
    tank_rows = _cutflow_rows(TANK)
    world_rows = _cutflow_rows(WORLD)
    n = len(tank_rows) + len(world_rows) + 2          # + the two header rows

    fig, axes = plt.subplots(
        2, 1, figsize=(12.0, 0.40 * n + 1.1),
        gridspec_kw=dict(height_ratios=[len(tank_rows) + 1, len(world_rows) + 1],
                         hspace=0.14, top=0.90, bottom=0.03, left=0.03, right=0.97))
    table_axes(axes[0], tank_rows, cols, widths)
    axes[0].set_title("tank/fmvmrd — 2,490,000 events", fontsize=12, pad=8)
    table_axes(axes[1], world_rows, cols, widths)
    axes[1].set_title("world/fmvmrd — 343,027 events", fontsize=12, pad=8)
    fig.suptitle("CC-Inclusive Selection Cut-Flows — Truth-Tag vs Reco-Tag "
                 "Streamlines", fontsize=14, y=0.985)
    save(fig, "cutflow_streamlines_table")


def _nminus1(run):
    hits = sorted((BASE / run / "csv").glob(f"{run}__cc_*_nminus1.csv"))
    return pd.read_csv(hits[0]) if hits else None


def fig_nminus1_table():
    """
    The N-1 cut-flow: for each cut, how many events it removes that no other cut
    removes. This is the 'which cut is actually costing me' view the sequential
    cut-flow cannot give, and it exists on disk for the tank runs only.
    """
    frames = {st: _nminus1(TANK[st]) for st in STREAMS}
    if all(v is None for v in frames.values()):
        print("  [skip] no N-1 cut-flow csv found")
        return
    order = [c for c in CUT_ORDER
             if any(v is not None and (v["cut_name"].str.replace('"', "", regex=False)
                                       == c).any() for v in frames.values())]
    rows = []
    for c in order:
        row = [CUT_SHORT.get(c, c)]
        for st in STREAMS:
            v = frames[st]
            if v is None:
                row += ["—", "—"]
                continue
            m = v["cut_name"].str.replace('"', "", regex=False) == c
            if not m.any():
                row += ["—", "—"]
            else:
                r = v[m].iloc[0]
                row += [f"{int(r['marginal_removed']):,}",
                        f"{100 * r['n1_efficiency']:.1f}%"]
        rows.append(row)

    fig = plt.figure(figsize=(11.0, 0.40 * (len(rows) + 1) + 1.0))
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.80])
    table_axes(ax, rows,
               ["Cut", "Truth-tag: only this cut removes",
                "N-1 efficiency", "Reco-tag: only this cut removes",
                "N-1 efficiency"],
               widths=[0.28, 0.20, 0.16, 0.20, 0.16], fs=10,
               emphasise_last=False)
    fig.suptitle("N-1 Cut-Flow, tank sample — events each cut removes on its own\n"
                 "N-1 efficiency = fraction kept when every other cut is already "
                 "applied", fontsize=12.5, y=0.98)
    save(fig, "cutflow_nminus1_table")


def fig_compcut_table():
    """
    The composition cut, per origin species: how many clusters it removes and how
    much signal that costs. A cluster-level cut-flow, from *__bkg_compcut_impact.csv.
    """
    rows, found = [], False
    for st in STREAMS:
        p = BASE / TANK[st] / "csv" / f"{TANK[st]}__bkg_compcut_impact.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[(d["method"] == "optics") &
              (d["split_axis"] == "origin_dominant_species")]
        if not len(d):
            continue
        found = True
        for _, r in d.iterrows():
            if r["category"] in ("none",):
                continue
            rows.append([("Truth-tag" if st == "truthtag" else "Reco-tag"),
                         PRETTY_ORIGIN.get(
                             {"muminus": "mu-", "muplus": "mu+", "piplus": "pi+",
                              "piminus": "pi-", "pizero": "pi0", "proton": "p",
                              "neutron": "n", "kplus": "K+", "kminus": "K-",
                              "ALL": "ALL"}.get(r["category"], r["category"]),
                             r["category"]),
                         f"{int(r['n_before']):,}", f"{int(r['n_removed']):,}",
                         f"{r['pct_removed']:.2f}%",
                         f"{int(r['n_signal_removed']):,}"])
    if not found:
        print("  [skip] no composition-cut csv found")
        return

    fig = plt.figure(figsize=(10.5, 0.38 * (len(rows) + 1) + 1.0))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.82])
    table_axes(ax, rows,
               ["Streamline", "Origin species", "Clusters before", "Removed",
                "% removed", "Signal lost"],
               widths=[0.16, 0.17, 0.18, 0.14, 0.15, 0.15], fs=10,
               emphasise_last=False)
    fig.suptitle("Composition Cut, OPTICS — clusters removed per origin species\n"
                 "ALL = the cut as applied; the species rows break down where it "
                 "acts", fontsize=12.5, y=0.98)
    save(fig, "compcut_impact_table")


def fig_overlap_table():
    """
    Streamline overlap as a table — same reason as the cut-flow: these are counts,
    and a table reads them out exactly.
    """
    ov = overlap()
    tt, rt = ov["truthtag"], ov["recotag"]
    rows = [
        ["Truth-tag selected",  f"{int(tt):,}",                 "100%",                      "—"],
        ["Reco-tag selected",   f"{int(rt):,}",                 "—",                         "100%"],
        ["Selected by both",    f"{int(ov['both']):,}",         f"{100*ov['both']/tt:.1f}%", f"{100*ov['both']/rt:.1f}%"],
        ["Truth-tag only",      f"{int(ov['truthtag_only']):,}", f"{100*ov['truthtag_only']/tt:.1f}%", "—"],
        ["Reco-tag only",       f"{int(ov['recotag_only']):,}",  "—",                        f"{100*ov['recotag_only']/rt:.1f}%"],
        ["Union",               f"{int(ov['union']):,}",         "—",                        "—"],
    ]
    fig = plt.figure(figsize=(9.0, 0.40 * (len(rows) + 1) + 0.9))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.78])
    table_axes(ax, rows, ["", "Events", "of truth-tag", "of reco-tag"],
               widths=[0.34, 0.20, 0.23, 0.23])
    fig.suptitle("Streamline Overlap, tank sample — reco-tag is {:.1f}% a subset of "
                 "truth-tag".format(100 * ov["both"] / rt), fontsize=13, y=0.97)
    save(fig, "streamline_overlap_table")


# ════════════════════════════════════════════════════════════════════════════
# figures — background composition
# ════════════════════════════════════════════════════════════════════════════
def fig_bkg_streams():
    tt_all = hit_ancestry(TANK["truthtag"])
    rt_all = hit_ancestry(TANK["recotag"]).set_index("category")
    tt = tt_all.head(8)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    yy = np.arange(len(tt))[::-1]
    ax.barh(yy + 0.19, tt["pct_of_denominator"], height=0.36, label="Truth-tag",
            **BAR_A)
    ax.barh(yy - 0.19,
            [float(rt_all["pct_of_denominator"].get(c, 0.0)) for c in tt["category"]],
            height=0.36, label="Reco-tag", **dict(color=ORANGE, alpha=0.75,
                                                   edgecolor="none"))
    ax.set_yticks(yy)
    ax.set_yticklabels(tt["label"], fontsize=12)
    ax.set_xlabel("% of delayed non-neutron hits", fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    bare(ax)
    ax.set_title("Delayed Background by Origin Particle — tank sample\n"
                 r"$t>10\,\mu$s, cc_pass events; {:,} hits (truth-tag), {:,} (reco-tag);"
                 r"  $\pi^0$: 7.46% / 3.44%".format(
                     int(tt_all["n_hits"].sum()), int(rt_all["n_hits"].sum())),
                 fontsize=12)
    fig.tight_layout()
    save(fig, "bkg_composition_streams")


def fig_bkg_world():
    tank = hit_ancestry(TANK["truthtag"]).set_index("category")
    wo = world_outoftank_origin("truthtag")
    wos = wo.set_index("category")
    keys = ["mu-", "pure_em (no non-EM ancestor)", "pi+", "p", "pi-", "mu+", "pi0"]
    labels = [PRETTY_ORIGIN.get(k, k) for k in keys]
    tv = [float(tank["pct_of_denominator"].get(k, 0.0)) for k in keys]
    wv = [float(wos["pct_of_denominator"].get(k, 0.0)) for k in keys]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    yy = np.arange(len(keys))[::-1]
    ax.barh(yy + 0.19, tv, height=0.36, label="Tank, in-tank CC", **BAR_A)
    ax.barh(yy - 0.19, wv, height=0.36, label="World, out-of-tank", **BAR_W)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("% of delayed non-neutron hits", fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    bare(ax)
    ax.set_title("Delayed Background by Origin Particle — tank vs world out-of-tank\n"
                 r"{:,} out-of-tank hits;  pure EM: 13.8% / 43.5%,  "
                 r"$\pi^0$: 7.5% / 0.43%".format(int(wo["n_hits"].sum())), fontsize=12)
    fig.tight_layout()
    save(fig, "bkg_composition_tank_vs_world")


def fig_bkg_of_training(frames):
    d = frames[("truthtag", "optics")]
    bkg = d[d["label"] == 0]
    parts = {"tank": bkg[~bkg["is_world"]],
             "world (out-of-tank)": bkg[bkg["is_world"]]}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # --- what kind of light is in the background clusters ---------------------
    ax = axes[0]
    cats = ["Neutron capture", "Non-neutron", "Dark noise / other"]
    share = {}
    for nm, p in parts.items():
        tot = p["n_hits"].sum()
        neu = 100 * p["n_neutron"].sum() / tot
        non = 100 * p["n_nonneutron"].sum() / tot
        share[nm] = [neu, non, 100 - neu - non]
    yy = np.arange(len(cats))[::-1]
    ax.barh(yy + 0.19, share["tank"], height=0.36, label="Tank", **BAR_A)
    ax.barh(yy - 0.19, share["world (out-of-tank)"], height=0.36,
            label="World, out-of-tank", **BAR_W)
    ax.set_yticks(yy)
    ax.set_yticklabels(cats, fontsize=11)
    ax.set_xlabel("% of hits in background-class clusters", fontsize=10.5)
    ax.set_title("Light inside the background clusters", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    bare(ax)

    # --- origin species of the non-neutron part -------------------------------
    def comp(sub):
        tot = sum(float(sub[c].sum()) for c in ORIGIN_COLS if c in sub.columns)
        return ({ORIGIN_COLS[c]: 100.0 * float(sub[c].sum()) / max(tot, 1.0)
                 for c in ORIGIN_COLS if c in sub.columns}, tot)

    all_c, _ = comp(bkg)
    tank_c, tank_n = comp(parts["tank"])
    wrld_c, wrld_n = comp(parts["world (out-of-tank)"])
    order = sorted(all_c, key=lambda k: -all_c[k])[:7]

    ax2 = axes[1]
    yy = np.arange(len(order))[::-1]
    ax2.barh(yy + 0.19, [tank_c.get(k, 0) for k in order], height=0.36,
             label=f"Tank ({int(tank_n):,} hits)", **BAR_A)
    ax2.barh(yy - 0.19, [wrld_c.get(k, 0) for k in order], height=0.36,
             label=f"World, out-of-tank ({int(wrld_n):,} hits)", **BAR_W)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(order, fontsize=12)
    ax2.set_xlabel("% of traced non-neutron background hits", fontsize=10.5)
    ax2.set_title("Non-neutron light by origin particle", fontsize=11)
    ax2.legend(fontsize=9.5, frameon=False, loc="lower right")
    bare(ax2)

    n_ndom = int(bkg["is_neutron_dom"].sum())
    fig.suptitle("Composition of the Merged Training Background — truth-tag / OPTICS\n"
                 "{:,} background clusters, {:,} neutron-dominated ({:.0f}%);  "
                 r"non-neutron light: $\mu^\pm$ {:.0f}%,  $\pi^\pm+\pi^0$ {:.0f}%".format(
                     len(bkg), n_ndom, 100 * n_ndom / len(bkg),
                     all_c.get(r"$\mu^-$", 0) + all_c.get(r"$\mu^+$", 0),
                     sum(all_c.get(k, 0) for k in (r"$\pi^+$", r"$\pi^-$", r"$\pi^0$"))),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "training_background_by_particle")


def fig_training_sizes(frames):
    d = frames[("truthtag", "optics")]
    n_sig, n_bkg = int((d["label"] == 1).sum()), int((d["label"] == 0).sum())
    bkg = d[d["label"] == 0]
    tank_bkg = int((~bkg["is_world"]).sum())
    world_sig = int(((d["label"] == 1) & d["is_world"]).sum())
    tank_sig = n_sig - world_sig

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    xx = np.arange(2)
    ax.bar(xx - 0.19, [tank_sig, tank_bkg], width=0.36, label="Tank only", **BAR_B)
    ax.bar(xx + 0.19, [n_sig, n_bkg], width=0.36, label="Tank + world", **BAR_A)
    for x, (a, b) in zip(xx, [(tank_sig, n_sig), (tank_bkg, n_bkg)]):
        ax.text(x - 0.19, a, f"{a:,}", ha="center", va="bottom", fontsize=9.5)
        ax.text(x + 0.19, b, f"{b:,}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(xx)
    ax.set_xticklabels(["Signal", "Background"], fontsize=12)
    ax.set_ylabel("Clusters", fontsize=11)
    ax.set_ylim(0, n_sig * 1.14)
    bare(ax)
    ax.set_title("Merged Training Set — truth-tag / OPTICS\n"
                 "background per class after the 1:1 cap: {:,} → {:,}  ({:.1f}×)".format(
                     tank_bkg, n_bkg, n_bkg / tank_bkg), fontsize=12)
    fig.tight_layout()
    save(fig, "merged_training_sizes")


# ════════════════════════════════════════════════════════════════════════════
# figures — features
# ════════════════════════════════════════════════════════════════════════════
def fig_feature_sep(frames):
    ser = separations(frames[("truthtag", "optics")])
    top = ser.head(12)
    tank_sep = feature_separation(TANK["truthtag"])
    tank_sep = tank_sep[tank_sep["method"] == "optics"].head(12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    yy = np.arange(len(top))[::-1]
    axes[0].barh(yy, top.to_numpy(), height=0.6, **BAR_A)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(top.index, fontsize=10)
    axes[0].set_xlabel(r"Separation [$\sigma$]", fontsize=10.5)
    axes[0].set_title("Merged tank+world, signal vs background\n"
                      "best {:.3f}".format(top.iloc[0]) + r"$\sigma$ (" +
                      f"{top.index[0]})", fontsize=11)
    bare(axes[0])

    yy2 = np.arange(len(tank_sep))[::-1]
    axes[1].barh(yy2, tank_sep["separation_sigma"], height=0.6, **BAR_C)
    axes[1].set_yticks(yy2)
    axes[1].set_yticklabels(tank_sep["feature"], fontsize=10)
    axes[1].set_xlabel(r"Separation [$\sigma$]", fontsize=10.5)
    axes[1].set_title("Tank only, neutron vs spurious clusters\n"
                      "best {:.3f}".format(tank_sep.iloc[0]["separation_sigma"]) +
                      r"$\sigma$ (" + f"{tank_sep.iloc[0]['feature']})", fontsize=11)
    bare(axes[1])

    ranks = {g: (list(ser.index).index(g) + 1, ser[g]) for g in GEOMETRY_FEATURES
             if g in ser.index}
    fig.suptitle("Feature Separation — truth-tag / OPTICS\n" + ",   ".join(
                     f"{g}: rank {r}/{len(ser)} ({v:.3f}" + r"$\sigma$)"
                     for g, (r, v) in ranks.items()), fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "feature_separation")


def fig_importances():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, merged, ttl in ((axes[0], False, "Tank only"),
                            (axes[1], True, "Tank + world (merged)")):
        imp = importances(TANK["truthtag"], "optics", merged)
        if imp is None:
            pending(ax, ttl)
            continue
        imp = imp.sort_values("rf_importance", ascending=False).head(12)
        yy = np.arange(len(imp))[::-1]
        style = BAR_A if merged else BAR_C
        bars = ax.barh(yy, imp["rf_importance"], height=0.6, **style)
        for bar, f in zip(bars, imp["feature"]):        # geometry rows stand out
            if f in GEOMETRY_FEATURES:
                bar.set_color(ORANGE)
                bar.set_alpha(0.85)
        ax.set_yticks(yy)
        ax.set_yticklabels(imp["feature"], fontsize=10)
        ax.set_xlabel("Random-forest importance", fontsize=10.5)
        ax.set_title(ttl, fontsize=11)
        bare(ax)

    m = importances(TANK["truthtag"], "optics", True)
    sub = ""
    if m is not None:
        geo = m[m["feature"].isin(GEOMETRY_FEATURES)]
        sub = ("\nGeometry features (orange) carry {:.1%} of the forest and {:.1%} of "
               "the GBT importance;  d_wall ranked {}, vtx_y {} of 31".format(
                   geo["rf_importance"].sum(), geo["gbt_importance"].sum(),
                   int(m.index[m["feature"] == "d_wall"][0]) + 1,
                   int(m.index[m["feature"] == "vtx_y"][0]) + 1))
    fig.suptitle("Feature Importances — truth-tag / OPTICS" + sub, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "feature_importances")


# ════════════════════════════════════════════════════════════════════════════
# figures — MVA performance
# ════════════════════════════════════════════════════════════════════════════
def fig_auc(tr):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        s, meth = cfg
        xx = np.arange(len(MODELS))
        base = [auc_of(tr, "tank", s, meth, m) or np.nan for m in MODELS]
        merg = [auc_of(tr, "merged", s, meth, m) or np.nan for m in MODELS]
        ax.bar(xx - 0.20, base, width=0.38, label="Tank only", **BAR_B)
        ax.bar(xx + 0.20, merg, width=0.38, label="Tank + world", **BAR_A)
        for x, v in zip(xx, base):
            if np.isfinite(v):
                ax.text(x - 0.20, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        for x, v in zip(xx, merg):
            if np.isfinite(v):
                ax.text(x + 0.20, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xx)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=10)
        ax.set_ylim(0.5, 0.72)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        bare(ax)
    axes[0].set_ylabel("Test AUC", fontsize=11)
    axes[0].legend(fontsize=9.5, frameon=False, loc="upper left")

    done = [c for c in CONFIGS if tr.get(("merged", *c), {}).get("complete")]
    gains = [g - b for c in done for m in MODELS
             if (b := auc_of(tr, "tank", *c, m)) and (g := auc_of(tr, "merged", *c, m))]
    sub = ("\nmean gain +{:.3f} over {} matched model/configuration pairs;  "
           "{} of 4 merged trainings done".format(np.mean(gains), len(gains),
                                                  len(done))) if gains else ""
    fig.suptitle("MVA Performance — Tank-Only Baseline vs Merged Tank+World" + sub,
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "auc_tank_vs_merged")


def fig_auc_box(tr):
    """
    Box plot of the 5-fold cross-validation AUCs, tank-only vs merged, with the
    held-out test AUC marked. The folds come straight out of the training logs —
    nothing is refitted, and this is the only place the fold-to-fold spread exists.
    """
    keys = ["rf", "gbt", "xgb"]                      # the NN is not cross-validated
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        s_, meth = cfg
        for off, scope, box_fc in ((-0.19, "tank", LGREY),
                                   (+0.19, "merged", LBLUE)):
            b = tr.get((scope, *cfg))
            data, pos = [], []
            for i, k in enumerate(keys):
                f = (b or {}).get("folds", {}).get(k)
                if f:
                    data.append(f)
                    pos.append(i + off)
            if not data:
                continue
            bp = ax.boxplot(data, positions=pos, widths=0.30, patch_artist=True,
                            medianprops=dict(color="black", lw=1.4),
                            whiskerprops=dict(color="black", lw=0.9),
                            capprops=dict(color="black", lw=0.9),
                            boxprops=dict(edgecolor="black", lw=0.9),
                            flierprops=dict(marker="+", markeredgecolor="black",
                                            markersize=4))
            for patch in bp["boxes"]:
                patch.set_facecolor(box_fc)
            # held-out test AUC for the same model
            for i, k in enumerate(keys):
                t = auc_of(tr, scope, s_, meth, {"rf": "Random Forest", "gbt": "GBT",
                                                 "xgb": "XGBoost"}[k])
                if t is not None:
                    ax.plot(i + off, t, marker="D", ms=4.5, color="black",
                            markerfacecolor="white", zorder=5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.upper() for k in keys], fontsize=10.5)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        bare(ax)
    axes[0].set_ylabel("AUC", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LGREY, edgecolor="black",
                             label="Tank only, 5-fold CV"),
               plt.Rectangle((0, 0), 1, 1, facecolor=LBLUE, edgecolor="black",
                             label="Tank + world, 5-fold CV"),
               plt.Line2D([], [], marker="D", ms=5, color="black", ls="none",
                          markerfacecolor="white", label="Held-out test AUC")]
    axes[0].legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("Cross-Validation AUC Spread — Tank-Only vs Merged Tank+World\n"
                 "box = 5 folds (median, quartiles, whiskers);  the neural network is "
                 "not cross-validated", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "auc_cv_boxplot")


# The earlier xi=0.02 + composition-cut campaign, kept as a reference baseline.
# Read-only: the AUCs are recomputed from the stored score tables, which reproduce
# the values published in REPORT_ccinc_xi02_compcut_FULL.md §2.2 / §4b.1 exactly for
# RF/GBT/XGB. The NN comes out 0.7495 (OPTICS) / 0.7116 (CF) because those score
# tables predate the seeded NN retrain; that report quotes the seeded 0.7526 / 0.7156.
# A 0.003 difference, immaterial here, but do not "correct" one against the other.
XI02_RUNS = {
    "xi02 + compcut, OPTICS": (
        "cc_neutrino_xi02_compcut",
        "cc_neutrino_xi02_compcut__mva_scores__keepprompt.parquet"),
    "xi02 + compcut, CF (own model)": (
        "cc_neutrino_xi02_compcut_cftrain",
        "cc_neutrino_xi02_compcut_cftrain__mva_scores__keepprompt__cf.parquet"),
}


def xi02_aucs():
    """{label: {model: auc}} for the xi02+compcut campaign, or {} if absent."""
    from sklearn.metrics import roc_auc_score
    cols = {"rf_score": "Random Forest", "gbt_score": "GBT",
            "xgb_score": "XGBoost", "nn_score": "Neural Network"}
    out = {}
    for label, (run, fname) in XI02_RUNS.items():
        p = BASE / run / "parquet" / fname
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        te = d[d["in_test"].astype(bool)]
        y = te["dominant_class"].isin(NEUTRON_CLASSES).astype(int)
        out[label] = ({cols[c]: float(roc_auc_score(y, te[c]))
                       for c in cols if c in te.columns},
                      len(te), int(y.sum()))
    return out


def fig_campaign_auc(tr):
    """
    Campaign comparison: the xi02+compcut baseline next to v3 tank-only and v3
    merged. The two campaigns are NOT interchangeable — different sample, different
    signal/background definition, composition cut applied, and a far smaller test
    set — so the figure states that rather than inviting the reading that the older
    campaign discriminates better.
    """
    xi = xi02_aucs()
    if not xi:
        print("  [skip] no xi02+compcut score tables found")
        return

    short = {("truthtag", "optics"): "tt / OPTICS",
             ("truthtag", "clusterfinder"): "tt / CF",
             ("recotag", "optics"): "rt / OPTICS",
             ("recotag", "clusterfinder"): "rt / CF"}
    groups = []                       # (two-line label, {model: auc})
    for label, (aucs, _n_te, _n_sig) in xi.items():
        groups.append((label.replace("xi02 + compcut, ", "xi=0.02 + compcut\n"),
                       aucs))
    for scope, tag in (("tank", "v3 tank only"), ("merged", "v3 tank + world")):
        for cfg in CONFIGS:
            a = {m: auc_of(tr, scope, *cfg, m) for m in MODELS}
            if any(v is not None for v in a.values()):
                groups.append((f"{tag}\n{short[cfg]}", a))

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    xx = np.arange(len(groups))
    w = 0.20
    for j, m in enumerate(MODELS):
        vals = [g[1].get(m) or np.nan for g in groups]
        ax.bar(xx + (j - 1.5) * w, vals, width=w * 0.92, label=MODEL_SHORT[m],
               color=LINE_COLORS[j], alpha=0.85, edgecolor="none")
    ax.set_xticks(xx)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax.set_ylim(0.5, 0.80)
    ax.set_ylabel("Test AUC", fontsize=11)
    ax.legend(fontsize=10, frameon=False, ncol=4, loc="upper right")
    bare(ax)
    fig.suptitle("Test AUC by Campaign — xi=0.02 + composition cut vs CCinc v3\n"
                 "Not one scale: the xi02 campaign uses a different sample and "
                 "signal definition, applies the composition cut, and its test set "
                 "is {:,} clusters against v3's 14,480".format(
                     list(xi.values())[0][1]), fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "auc_by_campaign")


def fig_loss(tr):
    panels = [("tank", "truthtag", "optics", GREY, "Tank only, truth-tag / OPTICS")]
    for cfg in CONFIGS:
        b = tr.get(("merged", *cfg))
        if b is not None and len(b["history"]):
            panels.append(("merged", cfg[0], cfg[1],
                           LINE_COLORS[min(len(panels) - 1, 3)],
                           "Tank + world, " + CFG_LBL[cfg]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    any_data = False
    for scope, s, meth, col, lbl in panels:
        b = tr.get((scope, s, meth))
        if not b or not len(b["history"]):
            continue
        any_data = True
        h = b["history"]
        axes[0].plot(h["epoch"], h["loss"], lw=1.7, color=col, label=lbl)
        axes[0].plot(h["epoch"], h["val_loss"], lw=1.5, ls="--", color=col)
        axes[1].plot(h["epoch"], h["auc"], lw=1.7, color=col, label=lbl)
        axes[1].plot(h["epoch"], h["val_auc"], lw=1.5, ls="--", color=col)

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary cross-entropy", fontsize=11)
    axes[0].set_title("Loss  (solid = training, dashed = validation)", fontsize=11)
    axes[0].legend(fontsize=8.5, frameon=False)
    bare(axes[0])
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("AUC", fontsize=11)
    axes[1].set_title("AUC  (solid = training, dashed = validation)", fontsize=11)
    bare(axes[1])
    if not any_data:
        axes[0].text(0.5, 0.5, "no training history in the logs yet",
                     transform=axes[0].transAxes, ha="center", color="0.6")

    fig.suptitle("Neural-Network Training — Loss and AUC per Epoch\n"
                 "Adam, early stopping on validation AUC (patience 15, best weights "
                 "restored)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "nn_loss_curves")


def fig_roc():
    from sklearn.metrics import roc_curve, roc_auc_score
    cols = {"rf_score": "Random Forest", "gbt_score": "GBT",
            "xgb_score": "XGBoost", "nn_score": "Neural Network"}
    colors = LINE_COLORS

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        s, meth = cfg
        p = (BASE / TANK[s] / "parquet" /
             f"{TANK[s]}__mva_scores__keepprompt__merged__{MSFX[meth]}.parquet")
        if not p.exists():
            pending(ax, CFG_LBL[cfg])
            continue
        d = pd.read_parquet(p)
        te = d[d["in_test"].astype(bool)]
        if "mva_label" in te.columns:            # written by mva_analysis.py
            y = te["mva_label"].to_numpy(int)
        else:
            # Re-derive the merged label. dominant_class ALONE is wrong here: an
            # out-of-tank capture is neutron-dominated and still background.
            y = np.where(te["origin_in_tank"].to_numpy(int) == 0, 0,
                         te["dominant_class"].isin(NEUTRON_CLASSES).astype(int))
        for c, col in zip(cols, colors):
            if c not in te.columns:
                continue
            fpr, tpr, _ = roc_curve(y, te[c].to_numpy(float))
            ax.plot(fpr, tpr, lw=1.8, color=col,
                    label=f"{cols[c]}   AUC = {roc_auc_score(y, te[c]):.3f}")
        ax.set_xlabel("False positive rate", fontsize=10)
        ax.set_ylabel("True positive rate", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("ROC Curves — Merged Tank+World Models (test set)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "roc_merged")


# ════════════════════════════════════════════════════════════════════════════
def main():
    OUTD.mkdir(exist_ok=True)
    print(f"[plots] output -> {OUTD}")

    print("[plots] rebuilding merged frames …")
    frames = {}
    for s_ in STREAMS:
        for meth in METHODS:
            d = merged_frame(s_, meth)
            frames[(s_, meth)] = d
            print(f"   {s_}/{meth}: {len(d):,} clusters  "
                  f"signal={int((d['label']==1).sum()):,}  "
                  f"background={int((d['label']==0).sum()):,}")

    tr = collect_trainings(frames)
    print("[plots] trainings found in logs:")
    for k in sorted(tr, key=str):
        b = tr[k]
        print(f"   {k}: complete={b['complete']}  epochs={len(b['history'])}  "
              f"aucs={ {m: round(v, 3) for m, v in b['aucs'].items()} }")

    dump_tables(frames)

    fig_cutflow_table()
    fig_nminus1_table()
    fig_compcut_table()
    fig_overlap_table()
    fig_bkg_streams()
    fig_bkg_world()
    fig_training_sizes(frames)
    fig_bkg_of_training(frames)
    fig_feature_sep(frames)
    fig_importances()
    fig_auc(tr)
    fig_auc_box(tr)
    fig_campaign_auc(tr)
    fig_loss(tr)
    fig_roc()
    print("[plots] done")


if __name__ == "__main__":
    main()
