"""
ccinc_v3_bg_differential.py
===========================
Where in the spectrum each background particle lives.

REPORT_ccinc_v3_world_merged.md §4.2 gives the *integrated* composition of the merged
training background: 60% of the light is real neutron capture, and of the non-neutron
remainder mu- 53.3%, pi+ 25.5%, pi0 7.2%, p 5.8%, pi- 4.6%, mu+ 3.4%. That single set
of numbers cannot tell you what to do about it. A species concentrated at low total
charge is removable by a box cut at almost no cost to the signal; one sitting under the
signal peak is not, and is the reason the AUC stops where it does.

This script bins the composition along the features the models actually use and reports,
per species, which end of each axis it owns.

Two views per axis, because they answer different questions:
  * STACKED  — what a bin is made of. Reading down a column tells you what dominates
               there.
  * SHAPE    — where one species lives, each species normalised to unit area over the
               axis. This is the view that identifies a species as low- or high-charge
               dominant; the stacked view hides it whenever one species is large
               everywhere.

Two composition axes, both reported:
  * HIT-WEIGHTED (the headline, and what §4.2 quotes) — sum of the per-cluster
    `n_origin_<species>` counts over the bin, divided by the traced non-neutron
    background hits in that bin. This is "which particle made the light".
  * CLUSTER-DOMINANT — `origin_dominant_species`, one vote per cluster. This is "which
    particle dominates a typical background cluster". They differ when a species makes
    a lot of light in a few clusters.

Neither uses `bg_class` or `ImmediateAncestorClass`; the species axis is the first
non-EM ancestor (`origin_pdg`), per world report §0.0.

The background population here is the merged training background class, taken from the
score tables. That is the complete background — the 1:1 cap downsamples the *signal*,
not the background — so summing these bins reproduces §4.2 exactly, which is asserted.

Run (no silent default):
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python ccinc_v3_bg_differential.py --config all
    python ccinc_v3_bg_differential.py --config truthtag/optics --no-figures
Output:
    ccinc_v3_bg_differential.csv          per bin, per species, both axes
    ccinc_v3_bg_dominance_summary.csv     which species owns which end of which axis
    slide_plots_ccinc_v3_merged/V3BG__*.{pdf,png}
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

from make_plots_ccinc_v3_merged import (
    BASE, TANK, MSFX, CONFIGS, CFG_LBL, ORIGIN_COLS, FEATURE_AXIS, OUTD,
    BLUE, GREY, ORANGE, GREEN, PURPLE, LBLUE, LGREY, bare, table_axes,
    importances)
import ccinc_v3_stats as S

HERE = Path(__file__).parent
PREFIX = "V3BG__"

N_BINS = 10                     # deciles: quantile bins, so every bin has statistics
MIN_TRACED_HITS = 200           # below this a bin's composition is noise, not a result

# The axes asked for by name, plus the two strongest merged separators. The top six
# features by that configuration's GBT importance are added per configuration.
#
# All FOUR classifier scores are binned, not just GBT: "which background survives into
# the signal-like tail" is a property of the model, and the four do not have to agree.
# Comparing them is what shows whether the irreducible background is the same for every
# classifier (it is) or an artifact of one of them (it is not).
SCORE_AXES = ["rf_score", "gbt_score", "xgb_score", "nn_score"]
PRIMARY_AXES = (["pe_total", "n_hits", "n_hits_early"] + SCORE_AXES
                + ["d_wall", "vtx_y"])

AXIS_LABEL = dict(FEATURE_AXIS)
AXIS_LABEL.update({"rf_score": "Random Forest score",
                   "gbt_score": "GBT discriminator score",
                   "xgb_score": "XGBoost score",
                   "nn_score": "Neural-network score"})

# Species order is fixed across every figure so colours mean the same thing everywhere.
SPECIES = ["muminus", "piplus", "pizero", "proton", "piminus", "muplus",
           "kplus", "kminus", "other"]
SPECIES_TEX = {"muminus": r"$\mu^-$", "muplus": r"$\mu^+$",
               "piplus": r"$\pi^+$", "piminus": r"$\pi^-$", "pizero": r"$\pi^0$",
               "proton": "p", "kplus": r"$K^+$", "kminus": r"$K^-$",
               "other": "other", "neutron": "n"}
SPECIES_COLOR = {"muminus": BLUE, "piplus": ORANGE, "pizero": GREEN,
                 "proton": PURPLE, "piminus": "#8c564b", "muplus": "#e377c2",
                 "kplus": "#7f7f7f", "kminus": "#bcbd22", "other": GREY}


# ════════════════════════════════════════════════════════════════════════════
# binning
# ════════════════════════════════════════════════════════════════════════════
def decile_bins(x: np.ndarray, n: int = N_BINS):
    """
    Quantile edges, de-duplicated. A heavily tied feature (fit_converged is 0/1,
    n_hits is small-integer) yields fewer than n usable bins; that is correct
    behaviour, not something to pad, so the caller gets whatever survives.
    """
    q = np.linspace(0, 1, n + 1)
    edges = np.unique(np.nanquantile(x, q))
    if len(edges) < 3:
        return None
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def differential(bkg: pd.DataFrame, axis: str) -> pd.DataFrame:
    """
    Per-bin composition along `axis`, both weighting schemes, with binomial errors.

    Also carries, per bin, the two context numbers without which the composition is
    easy to over-read: what fraction of the light in the bin is neutron capture at all
    (60% overall — most of the background is capture light), and what fraction of the
    clusters come from out-of-tank interactions.
    """
    if axis not in bkg.columns:
        return pd.DataFrame()
    x = bkg[axis].to_numpy(float)
    edges = decile_bins(x)
    if edges is None:
        print(f"    [skip] {axis}: too few distinct values to bin")
        return pd.DataFrame()

    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    origin_cols = [c for c in ORIGIN_COLS if c in bkg.columns]

    rows = []
    for b in range(len(edges) - 1):
        m = (idx == b) & np.isfinite(x)
        sub = bkg[m]
        if not len(sub):
            continue
        # hit-weighted: exactly the §4.2 recipe, restricted to this bin
        counts = {c.replace("n_origin_", ""): float(sub[c].sum())
                  for c in origin_cols}
        traced = sum(counts.values())
        nh = float(sub["n_hits"].sum())
        n_neu = float(sub["n_neutron"].sum()) if "n_neutron" in sub else np.nan
        dom = sub["origin_dominant_species"].value_counts(normalize=True) \
            if "origin_dominant_species" in sub else pd.Series(dtype=float)

        base = dict(axis=axis, bin=b, lo=float(edges[b]), hi=float(edges[b + 1]),
                    x_median=float(np.nanmedian(x[m])), n_clusters=int(len(sub)),
                    n_hits=int(nh), traced_nonneutron_hits=int(traced),
                    frac_hits_neutron=(n_neu / nh if nh else np.nan),
                    frac_clusters_outoftank=(
                        float((sub["origin_in_tank"] == 0).mean())
                        if "origin_in_tank" in sub else np.nan),
                    usable=bool(traced >= MIN_TRACED_HITS))
        for sp in SPECIES:
            f = counts.get(sp, 0.0) / traced if traced > 0 else np.nan
            err = (np.sqrt(max(f, 0.0) * max(1 - f, 0.0) / traced)
                   if traced > 0 else np.nan)
            rows.append(dict(base, species=sp, weighting="hit",
                             frac=f, frac_err=err, n=counts.get(sp, 0.0)))
            fd = float(dom.get(sp, 0.0))
            rows.append(dict(base, species=sp, weighting="cluster",
                             frac=fd,
                             frac_err=np.sqrt(fd * (1 - fd) / max(len(sub), 1)),
                             n=fd * len(sub)))
    return pd.DataFrame(rows)


def dominance_summary(diff: pd.DataFrame) -> pd.DataFrame:
    """
    Which end of each axis a species owns.

    `low_high_ratio` is the species' hit-weighted fraction in the lowest usable bin
    over the highest. > 1 means it concentrates at low values of the axis, < 1 at high.
    `peak_bin` is where its normalised shape is largest. Species that never reach 2% of
    a usable bin are dropped: their "peak" is a statistical accident.
    """
    d = diff[(diff.weighting == "hit") & diff.usable]
    rows = []
    for (cfg, axis), g in d.groupby(["config", "axis"]):
        for sp, s in g.groupby("species"):
            s = s.sort_values("bin")
            if s["frac"].max() < 0.02:
                continue
            lo, hi = s["frac"].iloc[0], s["frac"].iloc[-1]
            ratio = (lo / hi) if hi > 0 else np.inf
            pk = s.loc[s["frac"].idxmax()]
            rows.append(dict(
                config=cfg, axis=axis, species=sp,
                frac_lowest_bin=lo, frac_highest_bin=hi, low_high_ratio=ratio,
                peak_bin=int(pk["bin"]), peak_frac=float(pk["frac"]),
                peak_x_median=float(pk["x_median"]),
                trend=("low" if ratio > 1.25 else
                       "high" if ratio < 0.8 else "flat")))
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


def _pivot(diff, axis, weighting="hit"):
    d = diff[(diff.axis == axis) & (diff.weighting == weighting) & diff.usable]
    if d.empty:
        return None, None
    p = d.pivot_table(index="bin", columns="species", values="frac")
    xm = d.groupby("bin")["x_median"].first()
    order = [s for s in SPECIES if s in p.columns and p[s].max() >= 0.005]
    return p[order], xm


def panel_stacked(ax, diff, axis):
    p, xm = _pivot(diff, axis)
    if p is None:
        return False
    ax.stackplot(xm.to_numpy(), *[100 * p[s].to_numpy() for s in p.columns],
                 colors=[SPECIES_COLOR[s] for s in p.columns],
                 labels=[SPECIES_TEX[s] for s in p.columns], alpha=0.9)
    ax.set_xlim(xm.min(), xm.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel(AXIS_LABEL.get(axis, axis), fontsize=10)
    ax.set_ylabel("% of traced non-neutron hits", fontsize=10)
    ax.tick_params(labelsize=9)
    bare(ax)
    return True


def panel_shape(ax, diff, axis):
    """Each species normalised to unit sum across the axis."""
    p, xm = _pivot(diff, axis)
    if p is None:
        return False
    for s in p.columns:
        v = p[s].to_numpy(float)
        tot = np.nansum(v)
        if tot <= 0:
            continue
        ax.plot(xm.to_numpy(), v / tot, lw=1.7, color=SPECIES_COLOR[s],
                label=SPECIES_TEX[s])
    ax.set_xlabel(AXIS_LABEL.get(axis, axis), fontsize=10)
    ax.set_ylabel("Normalised species shape", fontsize=10)
    ax.tick_params(labelsize=9)
    bare(ax)
    return True


def fig_axis(diff, cfg, axis):
    """Stacked + shape for one axis, with the neutron-light context on a twin."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    if not panel_stacked(axes[0], diff, axis):
        plt.close(fig)
        return
    panel_shape(axes[1], diff, axis)
    axes[0].set_title("Composition of the bin", fontsize=11)
    axes[1].set_title("Where each species lives", fontsize=11)
    axes[1].legend(fontsize=9, frameon=False, ncol=3)

    d = diff[(diff.axis == axis) & (diff.weighting == "hit") & diff.usable]
    fneu = d.groupby("bin")["frac_hits_neutron"].first()
    fout = d.groupby("bin")["frac_clusters_outoftank"].first()
    fig.suptitle(
        f"Background Origin Species vs {AXIS_LABEL.get(axis, axis)} — "
        f"{CFG_LBL[cfg]}\n"
        f"neutron-capture light {100 * fneu.min():.0f}–{100 * fneu.max():.0f}% "
        f"of hits across the axis;  out-of-tank clusters "
        f"{100 * fout.min():.0f}–{100 * fout.max():.0f}%", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, f"{cfg[0]}_{MSFX[cfg[1]]}__{axis}")


def fig_top6(diff, cfg, feats, kind):
    """One 6-panel sheet over that configuration's six most important features."""
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    drew = False
    for ax, f in zip(axes.ravel(), feats):
        ok = (panel_stacked(ax, diff, f) if kind == "stacked"
              else panel_shape(ax, diff, f))
        drew = drew or ok
        if ok:
            ax.set_title(AXIS_LABEL.get(f, f), fontsize=10.5)
            ax.set_xlabel("")
    if not drew:
        plt.close(fig)
        return
    h, l = axes.ravel()[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, fontsize=9.5, frameon=False, ncol=9,
                   loc="lower center")
    what = ("Composition" if kind == "stacked" else "Species shapes")
    fig.suptitle(f"{what} Across the Six Most Important Features — "
                 f"{CFG_LBL[cfg]}\nranked by merged GBT importance", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    save(fig, f"{cfg[0]}_{MSFX[cfg[1]]}__top6_{kind}")


def fig_dominance_table(summary):
    """One table: which species owns which end of the named axes, per configuration."""
    axes_shown = ["pe_total", "n_hits"] + SCORE_AXES + ["d_wall"]
    d = summary[summary.axis.isin(axes_shown)]
    cols = ["Configuration", "Axis", "Low end", "High end", "Strongest trend"]
    cell = []
    for cfg in CONFIGS:
        lbl = CFG_LBL[cfg]
        first = True
        for axis in axes_shown:
            g = d[(d.config == lbl) & (d.axis == axis)]
            if g.empty:
                continue
            lo = g.sort_values("frac_lowest_bin", ascending=False).iloc[0]
            hi = g.sort_values("frac_highest_bin", ascending=False).iloc[0]
            tr = g.loc[(g["low_high_ratio"] - 1).abs().idxmax()]
            cell.append([
                lbl if first else "", AXIS_LABEL.get(axis, axis),
                f"{lo.species}  {100 * lo.frac_lowest_bin:.0f}%",
                f"{hi.species}  {100 * hi.frac_highest_bin:.0f}%",
                f"{tr.species}  x{tr.low_high_ratio:.2f} low/high"])
            first = False
    fig, ax = plt.subplots(figsize=(12.0, 0.42 * len(cell) + 1.6))
    table_axes(ax, cell, cols, fs=9.0, emphasise_last=False)
    fig.suptitle("Which Background Species Owns Which End of Each Axis\n"
                 "hit-weighted, deciles, bins with <"
                 f"{MIN_TRACED_HITS} traced hits excluded", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "dominance_summary")


# ════════════════════════════════════════════════════════════════════════════
def check_against_report(bkg: pd.DataFrame, cfg) -> None:
    """
    The integrated composition must reproduce world report §4.2 for truth-tag/OPTICS
    (mu- 53.3, pi+ 25.5, pi0 7.2, p 5.8, pi- 4.6, mu+ 3.4 % of the traced non-neutron
    remainder; 60% of the light is neutron capture). This is the check that the
    background population and the denominators here are the same ones the report used.
    """
    cols = [c for c in ORIGIN_COLS if c in bkg.columns]
    tot = sum(float(bkg[c].sum()) for c in cols)
    comp = {c.replace("n_origin_", ""): 100 * float(bkg[c].sum()) / tot
            for c in cols}
    nh = float(bkg["n_hits"].sum())
    fneu = 100 * float(bkg["n_neutron"].sum()) / nh
    print(f"    integrated: {len(bkg):,} background clusters, "
          f"{int(tot):,} traced non-neutron hits, "
          f"neutron-capture light {fneu:.1f}% of hits")
    print("    " + "  ".join(f"{k} {comp[k]:.1f}%" for k in
                             sorted(comp, key=lambda k: -comp[k])[:6]))
    if cfg != ("truthtag", "optics"):
        return
    want = {"muminus": 53.3, "piplus": 25.5, "pizero": 7.2,
            "proton": 5.8, "piminus": 4.6, "muplus": 3.4}
    bad = {k: (comp.get(k, 0), v) for k, v in want.items()
           if abs(comp.get(k, 0) - v) > 0.15}
    if bad or abs(fneu - 60) > 1.5:
        print("    WARNING — does not reproduce world report §4.2:")
        for k, (got, exp) in bad.items():
            print(f"       {k}: got {got:.1f}%, report says {exp:.1f}%")
        if abs(fneu - 60) > 1.5:
            print(f"       neutron-capture light: got {fneu:.1f}%, "
                  f"report says ~60%")
        print("    The binning below inherits whatever is wrong here — "
              "resolve before quoting.")
    else:
        print("    reproduces world report §4.2 (integrated cross-check PASS)")


def top6_features(cfg) -> list[str]:
    """Six most important features for this configuration, by merged GBT importance."""
    imp = importances(TANK[cfg[0]], cfg[1], merged=True)
    if imp is None or "gbt_importance" not in imp.columns:
        print(f"    [warn] no merged importance CSV for {cfg} — top-6 sheet skipped")
        return []
    return (imp.sort_values("gbt_importance", ascending=False)["feature"]
            .head(6).tolist())


def main():
    ap = argparse.ArgumentParser(prog="ccinc_v3_bg_differential")
    ap.add_argument("--config", required=True,
                    help="'all', or '<streamline>/<method>' e.g. truthtag/optics. "
                         "No default on purpose.")
    ap.add_argument("--no-figures", action="store_true",
                    help="CSVs only — skip every figure")
    a = ap.parse_args()

    if a.config == "all":
        cfgs = list(CONFIGS)
    else:
        try:
            s, m = a.config.split("/")
        except ValueError:
            sys.exit("[bg] --config must be 'all' or '<streamline>/<method>'")
        m = {"cf": "clusterfinder"}.get(m, m)
        if (s, m) not in CONFIGS:
            sys.exit(f"[bg] unknown configuration {a.config}; "
                     f"expected one of {CONFIGS}")
        cfgs = [(s, m)]

    all_diff = []
    for cfg in cfgs:
        print(f"\n[bg] {CFG_LBL[cfg]}")
        d = S.load_scores(cfg[0], cfg[1], verbose=False)
        bkg = d[d["y"] == 0].copy()
        check_against_report(bkg, cfg)

        feats = top6_features(cfg)
        axes = PRIMARY_AXES + [f for f in feats if f not in PRIMARY_AXES]
        print(f"    axes: {', '.join(axes)}")

        per_cfg = []
        for axis in axes:
            dd = differential(bkg, axis)
            if dd.empty:
                continue
            dd["config"] = CFG_LBL[cfg]
            dd["streamline"], dd["method"] = cfg
            per_cfg.append(dd)
        if not per_cfg:
            continue
        diff = pd.concat(per_cfg, ignore_index=True)
        all_diff.append(diff)

        if not a.no_figures:
            for axis in PRIMARY_AXES:
                if axis in diff["axis"].unique():
                    fig_axis(diff, cfg, axis)
            if feats:
                fig_top6(diff, cfg, feats, "stacked")
                fig_top6(diff, cfg, feats, "shape")

    if not all_diff:
        sys.exit("[bg] nothing computed")
    diff = pd.concat(all_diff, ignore_index=True)
    diff.to_csv(HERE / "ccinc_v3_bg_differential.csv", index=False)
    summary = dominance_summary(diff)
    summary.to_csv(HERE / "ccinc_v3_bg_dominance_summary.csv", index=False)
    print(f"\n[csv] ccinc_v3_bg_differential.csv ({len(diff)} rows), "
          f"ccinc_v3_bg_dominance_summary.csv ({len(summary)} rows)")

    print("\n=== dominance along the named axes (hit-weighted) ===")
    show = summary[summary.axis.isin(["pe_total", "n_hits", "d_wall"] + SCORE_AXES)]
    print(show[["config", "axis", "species", "frac_lowest_bin",
                "frac_highest_bin", "low_high_ratio", "trend"]]
          .round(4).to_string(index=False))

    # Does the irreducible background depend on which classifier you use?
    print("\n=== low/high ratio along EACH classifier's score "
          "(>1 removable, <1 survives into the signal-like tail) ===")
    sc = summary[summary.axis.isin(SCORE_AXES)]
    if len(sc):
        print(sc.pivot_table(index=["config", "species"], columns="axis",
                             values="low_high_ratio").round(2).to_string())

    if not a.no_figures and len(summary):
        fig_dominance_table(summary)


if __name__ == "__main__":
    main()
