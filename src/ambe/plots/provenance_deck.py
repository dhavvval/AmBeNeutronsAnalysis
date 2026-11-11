"""
Figures for the CCinc v3 cluster-provenance deck
(docs/presentations/ccinc_v3_cluster_provenance.tex).

    ambe plots provenance --deck provenance --config configs/cc_neutrino_v3_truthtag.yaml \
        --world-config configs/cc_neutrino_v3world_truthtag.yaml \
        --capture-config configs/cc_neutrino_v3_truthtag_capfix.yaml --capture-tag test2 \
        --out PRESENTATION_PLOTS/ccinc_provenance

Truth-fix deck (docs/presentations/ccinc_v3_truthfix.tex):

    ambe plots provenance --deck truthfix --config configs/cc_neutrino_v3fix_truthtag.yaml \
        --world-config configs/cc_neutrino_v3fixworld_truthtag.yaml \
        --old-config configs/cc_neutrino_v3oldtank19_truthtag.yaml \
        --old-world-config configs/cc_neutrino_v3oldworld20_truthtag.yaml \
        --tag v3fix --old-tag old --tids-tag tids --scan-tag v3fixscan3 \
        --out PRESENTATION_PLOTS/ccinc_truthfix

Cluster-purity deck (docs/presentations/ccinc_v3_clusterpurity.tex):

    ambe plots provenance --deck clusterpurity --config configs/cc_neutrino_v3fixsmear_truthtag.yaml \
        --world-config configs/cc_neutrino_v3fixsmearworld_truthtag.yaml --tag smear \
        --plot-methods clusterfinder,optics,peak --out PRESENTATION_PLOTS/ccinc_clusterpurity

Neutrino-interaction cluster deck (docs/presentations/ccinc_v3_neutrino_clusters.tex):

    ambe plots provenance --deck neutrino --config configs/cc_neutrino_v3fixsmear_truthtag.yaml \
        --world-config configs/cc_neutrino_v3fixsmearworld_truthtag.yaml --tag nu --clean-tag nuclean \
        --overlay-tag nuover --randoms-config configs/data_dummy_randoms_625x.yaml --overlay-run 6256 [--randoms-tag <tag>] \
        --plot-methods clusterfinder,clusterfinder_nms,optics_nms,peak --out PRESENTATION_PLOTS/ccinc_neutrino_clusters

Reads only products already on disk (`ambe mc match --mode stage0|stage1`,
`ambe mc timewindows`, `ambe mc capture`); computes nothing new.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..plotting import METHOD_COLORS, set_style

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
# data runs and MC versions, kept apart from the clusterer palette (METHOD_COLORS)
RUN_COLORS = {6256: "#111111", 6254: "#9a9a9a", "6264p6256": "#111111", 6264: "#d97a00"}
INK2 = "#52514e"
METHOD_LABEL = {"clusterfinder": "ClusterFinder", "optics": "OPTICS",
                "clusterfinder_nms": "CF + NMS", "optics_nms": "OPTICS + NMS",
                "peak": "time-peak", "peak_tof": "time-peak + ToF"}


def _style():
    set_style()
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.0), "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=180)
    plt.close(fig)
    print(f"[deck] {name}")


def _csv(ctx: RunContext, base: str, tag: str = "") -> pd.DataFrame:
    return pd.read_csv(ctx.csv_path(f"{ctx.run_name}__{base}" + (f"__{tag}" if tag else "")))


def _bars(ax, groups, series, values, colors, width=0.38):
    x = np.arange(len(groups))
    for k, (s, col) in enumerate(zip(series, colors)):
        off = (k - (len(series) - 1) / 2) * width
        ax.bar(x + off, values[k], width * 0.92, color=col, label=s)
    ax.set_xticks(x, [g.replace(" + ", "\n+ ") for g in groups])


def split_rates(tank, world, out):
    rows = []
    for name, ctx in (("tank", tank), ("world", world)):
        d = _csv(ctx, "stage0_split_census").set_index("method")
        rows.append((name, 100 * d.loc["clusterfinder", "frac_neutrons_split"],
                     100 * d.loc["optics", "frac_neutrons_split"]))
    fig, ax = plt.subplots()
    _bars(ax, [r[0] for r in rows], ["ClusterFinder", "OPTICS"],
          [[r[1] for r in rows], [r[2] for r in rows]], C[:2])
    ax.set_ylabel("neutrons split across >1 cluster [%]")
    ax.legend()
    ax.set_title(f"One neutron split across clusters: CF {rows[0][1]:.2f}% vs "
                 f"OPTICS {rows[0][2]:.2f}% (tank)")
    _save(fig, out, "p01_split_rate")


def trackids_per_cluster(tank, out):
    cl = pd.read_parquet(tank.parquet_path(f"{tank.run_name}__stage1_neutron_provenance"),
                         columns=["method", "is_truth_neutron", "n_contrib_neutrons"])
    cl = cl[cl.method == "clusterfinder"]
    edges = np.arange(-0.5, 8.5, 1.0)
    fig, ax = plt.subplots()
    for lab, sel, col in (("signal (trackID label)", cl.is_truth_neutron == 1, C[0]),
                          ("background (trackID label)", cl.is_truth_neutron == 0, C[1])):
        v = cl.loc[sel, "n_contrib_neutrons"].clip(upper=7)
        h, _ = np.histogram(v, bins=edges)
        ax.step(np.arange(0, 8), h / h.sum(), where="mid", color=col, label=lab)
    ax.set_xticks(range(8), [str(i) for i in range(7)] + ["7+"])
    ax.set_xlabel("neutron trackIDs contributing hits to the cluster")
    ax.set_ylabel("fraction of clusters")
    ax.legend()
    single = (cl.loc[cl.is_truth_neutron == 1, "n_contrib_neutrons"] == 1).mean()
    ax.set_title(f"ClusterFinder, tank: only {100*single:.1f}% of signal clusters "
                 "carry one trackID")
    _save(fig, out, "p02_trackids_per_cluster")


def failure_reasons(tank, world, out):
    order = [("fail_merged_neutrons", "several trackIDs, none >= 50%"),
             ("fail_diluted_by_nonneutron", "diluted by non-neutron light"),
             ("fail_no_neutron_hits", "no neutron hits"),
             ("fail_truth_window", "truth window")]
    labels, fr = [], []
    for name, ctx in (("tank", tank), ("world", world)):
        s = _csv(ctx, "stage1_summary")
        s = s[s.tag == "background"].set_index("method")
        for m in ("clusterfinder", "optics"):
            labels.append(f"{name} {METHOD_LABEL[m]}")
            fr.append([100 * s.loc[m, c] for c, _ in order])
    fr = np.array(fr)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for k, (_, lab) in enumerate(order):
        ax.barh(y, fr[:, k], left=left, color=C[k], label=lab, height=0.62,
                edgecolor="white", linewidth=1.5)
        left += fr[:, k]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("background-tagged clusters [%]")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title(f"Why background clusters fail the label: {fr[0,0]:.1f}% have several "
                 "trackIDs (tank CF)")
    _save(fig, out, "p03_failure_reasons")


def window_scan(tank, out):
    fig, ax = plt.subplots()
    for m, col in (("clusterfinder", C[0]), ("optics", C[1])):
        s = _csv(tank, f"htw_{m}_window_scan")
        s = s[(s.t0_ref == "median") & (s.tag == "signal")].sort_values("window_ns")
        ax.plot(s.window_ns, s.eff_dominant, color=col, marker="o", ms=4,
                label=f"{METHOD_LABEL[m]}: dominant trackID")
        ax.plot(s.window_ns, s.eff_other_neutron, color=col, ls="--", marker="s", ms=4,
                label=f"{METHOD_LABEL[m]}: other trackIDs")
    ax.set_xscale("log")
    ax.set_xlabel("window half-width |t - t$_{median}$| [ns]")
    ax.set_ylabel("fraction of hits inside the window")
    ax.legend(loc="lower right")
    ax.set_title("Signal clusters, tank: dominant and other trackIDs have the same "
                 "time profile")
    _save(fig, out, "p04_window_scan")


def window_purity(tank, world, out):
    fig, ax = plt.subplots()
    k = 0
    for name, ctx in (("tank", tank), ("world", world)):
        for m in ("clusterfinder", "optics"):
            s = _csv(ctx, f"htw_{m}_window_scan")
            s = s[(s.t0_ref == "median") & (s.tag == "signal")].sort_values("window_ns")
            ax.plot(s.window_ns, s.purity_vs_bg, color=C[k], marker="o", ms=4,
                    label=f"{name} {METHOD_LABEL[m]}")
            k += 1
    ax.set_xscale("log")
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel("window half-width |t - t$_{median}$| [ns]")
    ax.set_ylabel("purity: dominant / (dominant + non-neutron + noise)")
    ax.legend(loc="lower left")
    ax.set_title("Purity is flat from 5 ns to 1 $\\mu$s: no window edge helps")
    _save(fig, out, "p05_window_purity")


def pair_separation(tank, out):
    fig, ax = plt.subplots()
    b = np.logspace(-1, 4, 60)
    meds = []
    for m, col in (("clusterfinder", C[0]), ("optics", C[1])):
        p = _csv(tank, f"htw_{m}_neutron_pair_dt")
        v = np.clip(p.dt_median_ns.to_numpy(), 0.1, None)
        ax.hist(v, bins=b, histtype="step", color=col, lw=2,
                weights=np.full(len(v), 1 / len(v)),
                label=f"{METHOD_LABEL[m]} ({len(v):,} pairs)")
        meds.append(np.median(v))
    ax.set_xscale("log")
    ax.set_xlabel("|median time of trackID A - median time of trackID B| [ns]")
    ax.set_ylabel("fraction of trackID pairs")
    ax.legend()
    ax.set_title(f"TrackIDs sharing a cluster are {meds[0]:.1f} ns (CF) / "
                 f"{meds[1]:.1f} ns (OPTICS) apart")
    _save(fig, out, "p06_trackid_pair_dt")


def trackids_per_capture(cap, tag, out):
    t = _csv(cap, "capture_truth_tids_per_capture", tag)
    fig, ax = plt.subplots()
    ax.bar(t.n_trackids, 100 * t.fraction, color=C[0], width=0.7)
    ax.set_xticks(t.n_trackids, [str(int(i)) for i in t.n_trackids[:-1]] + ["5+"])
    ax.set_xlabel("neutron trackIDs whose hits belong to one true capture")
    ax.set_ylabel("true captures [%]")
    one = 100 * float(t.set_index("n_trackids").fraction.loc[1])
    ax.set_title(f"One capture, several trackIDs: only {one:.1f}% of captures carry "
                 "a single trackID")
    _save(fig, out, "p07_trackids_per_capture")


def merged_is_one_capture(cap, tag, out):
    mb = _csv(cap, "capture_truth_merged_breakdown", tag)
    mb = mb[mb.method == "clusterfinder"].set_index("category")
    cats = [("trackid_signal_single_tid", "signal,\n1 trackID"),
            ("trackid_signal_multi_tid", "signal,\nseveral trackIDs"),
            ("bkg_merged_neutrons", "bkg: several\ntrackIDs"),
            ("bkg_diluted", "bkg: diluted"),
            ("bkg_no_neutron", "bkg: no\nneutron")]
    fr = np.array([[100 * mb.loc[c, k] for c, _ in cats]
                   for k in ("frac_0_captures", "frac_1_capture", "frac_2plus_captures")])
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(cats))
    bottom = np.zeros(len(cats))
    for k, lab in enumerate(["0 true captures", "exactly 1 true capture",
                             ">= 2 true captures"]):
        ax.bar(x, fr[k], bottom=bottom, color=C[[4, 0, 1][k]], label=lab, width=0.62,
               edgecolor="white", linewidth=1.5)
        bottom += fr[k]
    ax.set_xticks(x, [c[1] for c in cats])
    ax.set_ylim(0, 100)
    ax.set_ylabel("clusters [%]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.set_title(f"'Merged neutrons' are one capture: {fr[1][2]:.1f}% hold exactly one "
                 f"(CF, {int(mb.loc['bkg_merged_neutrons','n_clusters'])} clusters)")
    _save(fig, out, "p08_merged_is_one_capture")


def background_composition(cap, tag, out):
    c = _csv(cap, "capture_truth_composition", tag)
    c = c[c.method == "clusterfinder"].set_index("sample")
    parts = [("capture light", ["cap_dom", "cap_other", "neutron_nocap"]),
             ("muon", ["bg_muon"]),
             ("charged pion", ["bg_pion_charged"]),
             ("other physics", ["bg_pion_zero", "bg_proton", "bg_pure_em", "bg_other"]),
             ("dark noise / untraced", ["darknoise", "untraced"])]
    samples = [("trackid_signal", "signal\n(trackID label)"),
               ("trackid_background", "background\n(trackID label)"),
               ("capture_signal", "signal\n(capture label)"),
               ("capture_background", "background\n(capture label)")]
    fr = np.array([[100 * c.loc[s, cols].sum() for s, _ in samples] for _, cols in parts])
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(samples))
    bottom = np.zeros(len(samples))
    for k, (lab, _) in enumerate(parts):
        ax.bar(x, fr[k], bottom=bottom, color=C[k], label=lab, width=0.62,
               edgecolor="white", linewidth=1.5)
        bottom += fr[k]
    ax.set_xticks(x, [s[1] for s in samples])
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of cluster hits [%]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.set_title(f"Background is {fr[0][1]:.0f}% capture light under the trackID label, "
                 f"{fr[0][3]:.0f}% under the capture label (CF)")
    _save(fig, out, "p09_background_composition")


def label_threshold_scan(cap, tag, out):
    s = _csv(cap, "capture_truth_tag_scan", tag)
    s = s[s.method == "clusterfinder"]
    fig, ax = plt.subplots()
    for sch, col, lab in (("capture", C[0], "dominant true capture"),
                          ("trackid", C[1], "dominant neutron trackID")):
        g = s[s.scheme == sch].sort_values("threshold")
        ax.plot(g.threshold, 100 * g.signal_fraction, color=col, marker="o", ms=5,
                label=lab)
    at = s[s.threshold == 0.5].set_index("scheme").signal_fraction
    ax.set_xlabel("required share of the cluster's hits")
    ax.set_ylabel("clusters labelled signal [%]")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title(f"Same 50% rule: {100*at['capture']:.1f}% signal by capture vs "
                 f"{100*at['trackid']:.1f}% by trackID (CF)")
    _save(fig, out, "p10_label_threshold_scan")


def clusterer_benchmark(cap, tag, out):
    s = _csv(cap, "capture_truth_summary", tag).set_index("method")
    methods = ["clusterfinder", "optics", "peak"]
    metrics = [("capture_split_rate", "split rate"), ("true_merge_rate", "merge rate")]
    fig, ax = plt.subplots()
    _bars(ax, [METHOD_LABEL[m] for m in methods], [m[1] for m in metrics],
          [[100 * s.loc[m, k] for m in methods] for k, _ in metrics], C[:2])
    ax.set_ylabel("true captures [%]")
    ax.legend()
    ax.set_title(f"Against true captures: OPTICS splits {100*s.loc['optics','capture_split_rate']:.1f}%"
                 f" and merges {100*s.loc['optics','true_merge_rate']:.1f}%; CF merges none")
    _save(fig, out, "p11_clusterer_benchmark")


OLD, NEW = "#eb6834", "#2a78d6"
OBJ_COL = {"n_capture": "#c62828", "michel": "#1b8a5a", "other": "#d97a00", "darknoise": "#b8b8b8"}
OBJ_LABEL = {"n_capture": "neutron capture", "michel": "Michel e\u00b1", "muon": "muon",
             "pion": "pion", "pion_charged": "\u03c0\u00b1", "pi0": "\u03c0\u2070", "proton": "proton", "em": "EM (e/\u03b3)", "n_other": "neutron, no capture",
             "other": "other", "noise": "dark noise"}
BASE3 = ["clusterfinder", "optics", "peak"]
METHOD_INK = [("#1f1f1f", "o", "-"), ("#5f5f5f", "s", "--"), ("#8a8a8a", "^", ":")]


def _pq(ctx: RunContext, base: str, tag: str, cols=None) -> pd.DataFrame:
    return pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__{base}__{tag}"), columns=cols)


def _part(src) -> int:
    import re
    return int(re.search(r"_(\d+)\.root$", str(src)).group(1))


def _matched_clusters(old: RunContext, new: RunContext, old_tag: str, new_tag: str) -> pd.DataFrame:
    cols = ["_source_file", "eventNumber", "method", "n_hits", "t_med", "tag", "owner_frac"]
    o = _pq(old, "truth_object_clusters", old_tag, cols)
    n = _pq(new, "truth_object_clusters", new_tag, cols)
    n = n[n.method.isin(BASE3)]
    o = o[o.method.isin(BASE3)]
    for d in (o, n):
        d["part"] = d["_source_file"].map(_part)
        d["t_med"] = d["t_med"].round(2)
    k = ["part", "eventNumber", "method", "n_hits", "t_med"]
    m = o.merge(n, on=k, how="outer", suffixes=("_old", "_new"), indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    if unmatched:
        raise SystemExit(f"[deck] {unmatched} clusters differ between old and fixed runs; "
                         "the before/after assumes identical reco clusters")
    return m.drop(columns="_merge")


def tids_before_after(tank_old, tank_new, world_old, world_new, tag, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    one = {}
    for ax, (name, o, n) in zip(axes, (("tank", tank_old, tank_new), ("world", world_old, world_new))):
        rows = []
        for ctx in (o, n):
            t = _csv(ctx, "capture_truth_tids_per_capture", tag).set_index("n_trackids").fraction
            rows.append([100 * t.get(k, 0.0) for k in (1, 2, 3, 4)] + [100 * t[t.index >= 5].sum()])
        _bars(ax, ["1", "2", "3", "4", "5+"], ["old labels", "both fixes"], rows, [OLD, NEW])
        ax.set_xlabel(f"neutron trackIDs in one true capture ({name})")
        one[name] = (rows[0][0], rows[1][0])
    axes[0].set_ylabel("true captures [%]")
    axes[0].legend()
    fig.suptitle(f"One capture = one trackID: {one['tank'][0]:.1f}% \u2192 {one['tank'][1]:.1f}% (tank), "
                 f"{one['world'][0]:.1f}% \u2192 {one['world'][1]:.1f}% (world)", fontsize=11)
    _save(fig, out, "t01_trackids_per_capture")


def delayed_objects(old, new, old_tag, new_tag, out, t_lo_ns: float, out_name: str, single: bool = False):
    order = ["n_capture", "michel", "em", "n_other", "muon", "pion", "pion_charged", "pi0", "proton"]
    obs = [_pq(c, "truth_objects", t, ["obj_class", "n_hits", "t_med"]) for c, t in ((old, old_tag), (new, new_tag))]
    obs = [o[o.t_med >= t_lo_ns] for o in obs]
    win = f"{t_lo_ns / 1000:g}–70 µs"
    cnt = [o.obj_class.value_counts() for o in obs]
    hits = [o.groupby("obj_class").n_hits.sum() for o in obs]
    order = [o for o in order if any(c.get(o, 0) > 0 for c in cnt)]
    if single:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        y = np.arange(len(order))
        for k, (c, col, name) in enumerate(zip(cnt, (OLD, NEW), ("old DirectParentID", "fixed DirectParentID"))):
            ax.barh(y + (k - 0.5) * 0.38, [max(c.get(o, 0), 0.5) for o in order], 0.35, color=col, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("number of true particles")
        ax.set_yticks(y, [OBJ_LABEL[o] for o in order])
        ax.invert_yaxis()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        ax.set_title(f"True parent particles of the delayed MCHits (tank, {win})", fontsize=11)
        fig.tight_layout()
        _save(fig, out, out_name)
        pd.DataFrame({"obj_class": order, "n_old": [int(cnt[0].get(o, 0)) for o in order],
                      "n_fixed": [int(cnt[1].get(o, 0)) for o in order],
                      "hits_old": [int(hits[0].get(o, 0)) for o in order],
                      "hits_fixed": [int(hits[1].get(o, 0)) for o in order]}).to_csv(out / f"{out_name}.csv", index=False)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    y = np.arange(len(order))
    for ax, vals, lab in ((axes[0], cnt, "number of true particles"), (axes[1], hits, "MCHits from those true particles")):
        for k, (c, col, name) in enumerate(zip(vals, (OLD, NEW), ("old labels", "both fixes"))):
            ax.barh(y + (k - 0.5) * 0.38, [max(c.get(o, 0), 0.5) for o in order], 0.35, color=col, label=name)
        ax.set_xscale("log")
        ax.set_xlabel(f"{lab}, {win}")
    axes[0].set_yticks(y, [OBJ_LABEL[o] for o in order])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    tot = [int(h.sum()) for h in hits]
    fake = sum(cnt[0].get(o, 0) for o in ("muon", "pion", "pion_charged", "pi0", "proton"))
    fake_n = sum(cnt[1].get(o, 0) for o in ("muon", "pion", "pion_charged", "pi0", "proton"))
    fig.suptitle(f"Tank, all events, {win}: delayed light grouped by the true particle that made it (truth, not clusters)\n"
                 f"old labels \u2192 both fixes: muon/pion/proton {fake:,} \u2192 {fake_n:,}; "
                 f"MCHits in true particles {tot[0]:,} \u2192 {tot[1]:,} (the rest is now dark noise)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out, out_name)


def michel_times(old, new, old_tag, new_tag, window_lo, min_hits, out):
    fig, ax = plt.subplots()
    bins = np.linspace(0, 20, 41)
    taus = []
    for c, t, col, lab in ((old, old_tag, OLD, "old labels"), (new, new_tag, NEW, "both fixes")):
        ob = _pq(c, "truth_objects", t, ["obj_class", "n_hits", "t_first"])
        dt = (ob[(ob.obj_class == "michel") & (ob.n_hits >= min_hits)].t_first - window_lo) / 1000.0
        tau, err = dt.mean(), dt.mean() / np.sqrt(len(dt))
        taus.append((tau, err))
        ax.hist(dt, bins=bins, density=True, histtype="step", lw=2, color=col,
                label=f"{lab}, n = {len(dt):,}")
    ax.set_yscale("log")
    ax.set_xlabel("Michel object first hit \u2212 2 \u00b5s [\u00b5s]")
    ax.set_ylabel("density [1/\u00b5s]")
    ax.legend()
    ax.set_title(f"Michel \u03c4: {taus[0][0]:.2f} \u00b1 {taus[0][1]:.2f} \u00b5s \u2192 "
                 f"{taus[1][0]:.3f} \u00b1 {taus[1][1]:.3f} \u00b5s (tank)")
    _save(fig, out, "t03_michel_time")


def cluster_retag(m, out):
    order = ["n_capture", "michel", "muon", "pion", "pion_charged", "pi0", "proton", "em", "n_other"]
    for meth in BASE3:
        g = m[m.method == meth]
        tot = np.array([int((g.tag_old == o).sum()) for o in order])
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        y = np.arange(len(order))
        left = np.zeros(len(order))
        for key in ("n_capture", "michel", "other"):
            sel = g.tag_new == key if key != "other" else ~g.tag_new.isin(["n_capture", "michel"])
            v = 100 * np.array([int((sel & (g.tag_old == o)).sum()) for o in order]) / np.maximum(tot, 1)
            ax.barh(y, v, left=left, color=OBJ_COL[key], edgecolor="white", linewidth=1.5,
                    label=f"now: {OBJ_LABEL[key]}", height=0.62)
            left += v
        ax.set_yticks(y, [f"was {OBJ_LABEL[o]} ({t:,})" for o, t in zip(order, tot)])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"share of the {METHOD_LABEL[meth]} clusters with that old tag [%]")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
        changed = 100 * (g.tag_old != g.tag_new).mean()
        ax.set_title(f"{METHOD_LABEL[meth]}: {changed:.1f}% of {len(g):,} identical clusters change truth tag (tank)")
        _save(fig, out, f"t04_cluster_retag_{meth}")


def capture_clusters_count(m, out):
    rows = [[int((m[m.method == k].tag_old == "n_capture").sum()) for k in BASE3],
            [int((m[m.method == k].tag_new == "n_capture").sum()) for k in BASE3]]
    fig, ax = plt.subplots()
    _bars(ax, [METHOD_LABEL[k] for k in BASE3], ["old labels", "both fixes"], rows, [OLD, NEW])
    ax.set_ylabel("clusters tagged neutron capture")
    ax.set_ylim(0, 1.18 * max(rows[1]))
    ax.legend(loc="upper left", ncol=2)
    gain = 100 * (rows[1][0] / rows[0][0] - 1)
    ax.set_title(f"Same clusters, more are captures: ClusterFinder {rows[0][0]:,} \u2192 {rows[1][0]:,} (+{gain:.0f}%)")
    _save(fig, out, "t05_capture_clusters")


def owner_fraction(m, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, meth in zip(axes, BASE3):
        g = m[m.method == meth]
        for col, c, lab in (("owner_frac_old", OLD, "old labels"), ("owner_frac_new", NEW, "both fixes")):
            ax.hist(g[col], bins=bins, histtype="step", lw=2, color=c, density=True,
                    label=f"{lab}, median {g[col].median():.2f}")
        ax.set_xlabel(f"owner fraction ({METHOD_LABEL[meth]})")
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("clusters (density)")
    fig.suptitle("Clusters looked mixed before; the largest truth object now owns ~90% of the hits (tank)",
                 fontsize=11)
    _save(fig, out, "t06_owner_fraction")


def metrics_before_after(old, new, old_tag, new_tag, out):
    def comb(ctx, tag):
        s = _csv(ctx, "truth_object_summary", tag)
        s = s[s.method.isin(BASE3)]
        r = {}
        for (meth, cls), g in s.groupby(["method", "obj_class"]):
            w = g.n_objects
            r[(meth, cls)] = {k: (g[k] * w).sum() / w.sum() for k in ("found", "split", "merged_into_other")}
        return r
    o, n = comb(old, old_tag), comb(new, new_tag)
    panels = [("n_capture", "found", "captures found"), ("n_capture", "merged_into_other", "captures merged into another object's cluster"),
              ("michel", "found", "Michels found")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, (cls, k, lab) in zip(axes, panels):
        _bars(ax, ["CF", "OPTICS", "time-peak"], ["old labels", "both fixes"],
              [[100 * o[(mm, cls)][k] for mm in BASE3], [100 * n[(mm, cls)][k] for mm in BASE3]], [OLD, NEW])
        ax.set_ylabel(f"{lab} [%]", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"The old labels made clustering look worse: CF Michels found "
                 f"{100*o[('clusterfinder','michel')]['found']:.0f}% \u2192 {100*n[('clusterfinder','michel')]['found']:.1f}%, "
                 f"captures merged {100*o[('clusterfinder','n_capture')]['merged_into_other']:.1f}% \u2192 "
                 f"{100*n[('clusterfinder','n_capture')]['merged_into_other']:.1f}% (tank)", fontsize=10.5)
    _save(fig, out, "t07_metrics_before_after")


def found_vs_size(new, new_tag, out):
    ob = _pq(new, "truth_objects", new_tag)
    c = ob[(ob.obj_class == "n_capture") & (ob.n_hits >= 5)]
    edges = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 60]
    mid = [(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots()
    lo = None
    for meth, (col, mk, ls) in zip(BASE3, METHOD_INK):
        f = [100 * (c[(c.n_hits >= a) & (c.n_hits < b)][f"n_owned_{meth}"] >= 1).mean()
             for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(mid, f, marker=mk, ms=6, color=col, ls=ls, label=METHOD_LABEL[meth])
        if meth == "optics":
            lo = f[0]
    ax.set_xscale("log")
    ax.set_xticks([5, 8, 10, 20, 50], ["5", "8", "10", "20", "50"])
    ax.set_xlabel("hits in the true capture")
    ax.set_ylabel("captures found [%]")
    ax.legend(loc="lower right")
    small = 100 * (c.n_hits < 8).mean()
    ax.set_title(f"Missed captures are the small ones: {small:.0f}% have 5\u20137 hits, OPTICS finds {lo:.0f}% of 5-hit ones")
    _save(fig, out, "t08_found_vs_size")


def michel_split(new, new_tag, out):
    s = _csv(new, "truth_object_summary", new_tag)
    ms = ["clusterfinder", "clusterfinder_nms", "optics", "optics_nms", "peak"]
    s = s[s.method.isin(ms) & (s.obj_class == "michel")]
    v = [100 * (s[s.method == k].split * s[s.method == k].n_objects).sum() / s[s.method == k].n_objects.sum() for k in ms]
    fig, ax = plt.subplots()
    ax.bar(range(len(ms)), v, color=INK2, width=0.62)
    ax.set_xticks(range(len(ms)), [METHOD_LABEL[k] for k in ms])
    ax.set_ylabel("Michel electrons split into \u2265 2 clusters [%]")
    ax.set_title(f"CF and OPTICS cut {v[0]:.0f}% / {v[2]:.0f}% of Michels in two; time-peak {v[4]:.1f}% (tank)")
    _save(fig, out, "t09_michel_split")


def han_scan(new, scan_tag, out):
    import re
    d = _csv(new, "truth_object_peak_scan", scan_tag)
    rx = d.method.str.extract(r"peak_tw(?P<tw>[\d.]+)_n(?P<n>\d+)_s(?P<s>[\d.]+)(?P<clean>_clean)?(?P<tof>_tof)?")
    d = pd.concat([d, rx], axis=1)
    d = d[d.clean.isna() & d.tof.isna() & (d.s.astype(float) == 50)]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={"wspace": 0.55})
    best = None
    for ax, (k, lab) in zip(axes, (("found", "captures found [%]"), ("hit_completeness", "capture hit completeness [%]"))):
        p = d.pivot_table(index=d.tw.astype(float), columns=d.n.astype(int), values=k) * 100
        im = ax.imshow(p.values, cmap="Blues", aspect="auto", origin="lower", vmin=p.values.min(), vmax=100)
        ax.set_xticks(range(p.shape[1]), p.columns)
        ax.set_yticks(range(p.shape[0]), [f"{v:g}" for v in p.index])
        ax.set_xlabel("NHITSTH (hits in the window)")
        ax.set_ylabel("TWIDTH [ns]")
        fig.colorbar(im, ax=ax, label=lab, fraction=0.05)
        if k == "found":
            best = (p.loc[30.0, 5], p.loc[20.0, 6])
    fig.suptitle(f"Han NTag Stage 1 scan: TWIDTH 30 ns, NHITSTH 5 finds {best[0]:.1f}% vs {best[1]:.1f}% at the default 20 ns / 6 (tank, 3 files)",
                 fontsize=10.5)
    _save(fig, out, "t10_han_scan")


def run_truthfix(ctx: RunContext, args) -> Path:
    _style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old, world, old_world = (load_context(args.old_config), load_context(args.world_config),
                             load_context(args.old_world_config))
    ot, nt = f"{args.old_tag}_plurality", f"{args.tag}_plurality"
    tids_before_after(old, ctx, old_world, world, args.tids_tag, out)
    prompt = float((ctx.extra.get("residual", {}) or {}).get("prompt_window_ns", 2000.0))
    delayed_objects(old, ctx, ot, nt, out, prompt, "t02_delayed_objects")
    delayed_objects(old, ctx, ot, nt, out, prompt, "t02_delayed_objects_single", single=True)
    delayed_objects(old, ctx, ot, nt, out, 10000.0, "t02b_delayed_objects_10us")
    michel_times(old, ctx, ot, nt, prompt, 5, out)
    m = _matched_clusters(old, ctx, ot, nt)
    cluster_retag(m, out)
    capture_clusters_count(m, out)
    owner_fraction(m, out)
    metrics_before_after(old, ctx, ot, nt, out)
    found_vs_size(ctx, nt, out)
    michel_split(ctx, nt, out)
    han_scan(ctx, f"{args.scan_tag}_plurality", out)
    return out


# --------------------------------------------------------------------------- #
# Cluster-purity deck (docs/presentations/ccinc_v3_clusterpurity.tex)
# --------------------------------------------------------------------------- #

SAMPLE_LS = {"tank": "-", "world": "--"}


def _purity_clusters(ctx: RunContext, tag: str) -> pd.DataFrame:
    d = _pq(ctx, "truth_object_clusters", tag)
    d = d[d.method.isin(BASE3)].copy()
    d["purity"] = d.owner_frac
    for k in ("nowf_lowpe_owner", "nowf_lowpe_other_obj", "nowf_lowpe_darknoise"):
        if k not in d:
            d[k] = 0
    if "nowf_lowpe_untraced" not in d:
        d["nowf_lowpe_untraced"] = d.comp_untraced_lowpe
    d["purity_clean"] = (d.n_owner_hits - d.nowf_lowpe_owner) / (d.n_hits - d.comp_untraced_lowpe).clip(lower=1)
    return d


def hit_match_vs_pe(tank: RunContext, world: RunContext, window_lo: float, out):
    import pyarrow.parquet as pq
    edges = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30, 1e4])
    xc = np.sqrt(np.maximum(edges[:-1], 0.1) * np.minimum(edges[1:], 60))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    parts = []
    for ax, (name, ctx) in zip(axes, (("tank", tank), ("world", world))):
        h = pq.read_table(ctx.parquet_path(f"{ctx.run_name}__pulses"),
                          columns=["t", "pe", "is_untraced", "match_failure_reason"]).to_pandas()
        h = h[h.t >= window_lo]
        reason = np.where(h.is_untraced.astype(bool), h.match_failure_reason.astype(str), "matched")
        idx = np.digitize(h.pe, edges) - 1
        for r, c, lab in (("matched", C[0], "has a waveform pulse"),
                          ("time_mismatch", C[1], "waveform pulses on the PMT, none within 8 ns"),
                          ("no_entry", C[3], "no waveform pulse on the PMT")):
            y = [np.mean(reason[idx == i] == r) if (idx == i).any() else np.nan for i in range(len(xc))]
            ax.plot(xc, y, marker="o", ms=4, color=c, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("MCHit pulse PE")
        ax.set_title(name)
        parts.append(f"{name} {np.mean(reason != 'matched'):.1%}")
    axes[0].set_ylabel("fraction of delayed MCHit pulses")
    axes[0].legend(loc="center right", fontsize=7.5)
    fig.suptitle("Unmatched MCHit pulses are sub-1-PE pulses the hit finder never sees ("
                 + ", ".join(parts) + " unmatched)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "c01_hit_match_vs_pe")


def purity_before_after(d: pd.DataFrame, out):
    fig, axes = plt.subplots(1, len(BASE3), figsize=(3.7 * len(BASE3), 3.6), sharey=True)
    for ax, meth in zip(axes, BASE3):
        parts = []
        for name in ("tank", "world"):
            g = d[(d["sample"] == name) & (d.method == meth) & (d.owner_class == "n_capture")]
            for col, c in (("purity", C[1]), ("purity_clean", C[0])):
                x = np.sort(g[col].to_numpy())
                ax.step(x, 1 - np.arange(len(x)) / len(x), where="post", color=c, ls=SAMPLE_LS[name],
                        lw=2 if name == "tank" else 1.4,
                        label=f"{name}, {'all hits' if col == 'purity' else 'unmatched < 1 PE removed'}")
            parts.append(f"{g.purity.mean():.3f}→{g.purity_clean.mean():.3f}")
        ax.set_xlim(0.5, 1.005)
        ax.set_xlabel("owner purity")
        ax.set_title(f"{METHOD_LABEL[meth]}: mean {parts[0]} (tank)", fontsize=10)
    axes[0].set_ylabel("capture clusters with purity ≥ x")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Removing unmatched hits below 1 PE: capture-cluster purity", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "c02_purity_before_after")


def composition_clean(d: pd.DataFrame, out):
    cats = [("owner", C[0]), ("other particles", C[1]), ("dark noise", "#8a8984"),
            ("unmatched ≥ 1 PE", C[3])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), sharey=True)
    for ax, name in zip(axes, ("tank", "world")):
        rows, labels = [], []
        for oc in ("n_capture", "michel"):
            for meth in BASE3:
                g = d[(d["sample"] == name) & (d.method == meth) & (d.owner_class == oc)]
                n = max((g.n_hits - g.comp_untraced_lowpe).sum(), 1)
                other = (g.n_hits - g.comp_untraced - g.comp_darknoise - g.n_owner_hits - g.nowf_lowpe_other_obj).sum()
                rows.append([(g.n_owner_hits - g.nowf_lowpe_owner).sum() / n, other / n,
                             (g.comp_darknoise - g.nowf_lowpe_darknoise).sum() / n,
                             (g.comp_untraced - g.nowf_lowpe_untraced).sum() / n])
                labels.append(f"{METHOD_LABEL[meth]}\n{'capture' if oc == 'n_capture' else 'Michel'}")
        rows = np.array(rows)
        x = np.arange(len(rows))
        b = np.zeros(len(rows))
        for j, (lab, col) in enumerate(cats):
            ax.bar(x, rows[:, j], bottom=b, width=0.72, color=col, label=lab, edgecolor="white", lw=1)
            b += rows[:, j]
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0.85, 1.0)
        ax.set_title(name)
    axes[0].set_ylabel("fraction of cluster hits (axis from 0.85)")
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("After removing unmatched hits below 1 PE, what is left in a cluster", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "c03_composition_clean")


def neutrons_per_cluster(d: pd.DataFrame, out, hits: str = "all hits"):
    groups = [METHOD_LABEL[m] for m in BASE3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, name in zip(axes, ("tank", "world")):
        vals = []
        for col, thr in (("n_neutrons", 2), ("n_neutrons_3", 2), ("n_captures", 2)):
            vals.append([(d[(d["sample"] == name) & (d.method == m) & (d.owner_class == "n_capture")][col] >= thr).mean()
                         for m in BASE3])
        _bars(ax, groups, ["2+ neutrons (any hit)", "2+ neutrons (≥ 3 hits each)", "2+ captures"],
              vals, [C[1], C[3], C[0]], width=0.26)
        ax.set_title(name)
    axes[0].set_ylabel("fraction of capture clusters")
    axes[0].legend(loc="upper left", fontsize=7.5)
    g = d[(d["sample"] == "tank") & (d.owner_class == "n_capture")]
    one = {m: (g[g.method == m].n_neutrons == 1).mean() for m in BASE3}
    fig.suptitle(f"One neutron per capture cluster (tank, all events, {hits}): " + ", ".join(
        f"{METHOD_LABEL[m]} {one[m]:.1%}" for m in BASE3), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "c04_neutrons_per_cluster")


def merged_capture_dt(d: pd.DataFrame, out, hits: str = "all hits"):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.logspace(0, 5, 31)
    parts = []
    for meth, c in ((m, METHOD_COLORS[m]) for m in BASE3):
        g = d[(d.method == meth) & (d.owner_class == "n_capture") & (d.n_captures >= 2)]
        ax.hist(g.dt_captures_min.dropna(), bins=bins, histtype="step", lw=2, color=c,
                label=f"{METHOD_LABEL[meth]}: {len(g)} clusters with 2+ captures")
        parts.append(f"{METHOD_LABEL[meth]} {g.dt_captures_min.median():.0f} ns")
    ax.set_xscale("log")
    ax.set_xlabel("time between the two true captures in the cluster [ns]")
    ax.set_ylabel("capture clusters holding hits of 2+ captures")
    ax.legend(loc="upper left")
    ax.set_title(f"One cluster, hits of two different neutron captures (tank + world, {hits})\nmedian "
                 + ", ".join(parts), fontsize=9)
    _save(fig, out, "c05_merged_capture_dt")


def merge_breakdown(d: pd.DataFrame, out) -> pd.DataFrame:
    """Capture clusters with hits of 2+ captures: how many hits the second capture brings, how far apart."""
    rows = []
    for name in ("tank", "world"):
        for meth in BASE3:
            g = d[(d["sample"] == name) & (d.method == meth) & (d.owner_class == "n_capture")]
            m = g[g.n_captures >= 2]
            o = m.n_hits_other_neutrons
            rows.append(dict(sample=name, clusterer=METHOD_LABEL[meth], capture_clusters=len(g), two_plus=len(m),
                             other_1hit=int((o == 1).sum()), other_2to4=int(o.between(2, 4).sum()),
                             other_5plus=int((o >= 5).sum()), dt_lt100=int((m.dt_captures_min < 100).sum()),
                             dt_100to1000=int(m.dt_captures_min.between(100, 1000).sum()),
                             dt_gt1000=int((m.dt_captures_min > 1000).sum()), median_dt=m.dt_captures_min.median()))
    t = pd.DataFrame(rows)
    t.to_csv(out / "merge_breakdown.csv", index=False)
    print(t.round(1).to_string(index=False))
    return t


def purity_vs_size(d: pd.DataFrame, out):
    bins = np.array([3, 6, 8, 10, 13, 17, 22, 30, 45, 70, 1000])
    xc = np.sqrt(bins[:-1] * np.minimum(bins[1:], 100))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for meth, c in ((m, METHOD_COLORS[m]) for m in BASE3):
        g = d[(d["sample"] == "tank") & (d.method == meth) & (d.owner_class == "n_capture")]
        idx = np.digitize(g.n_hits, bins) - 1
        for col, ls in (("purity", ":"), ("purity_clean", "-")):
            y = [g[col][idx == i].mean() if (idx == i).sum() >= 20 else np.nan for i in range(len(xc))]
            ax.plot(xc, y, marker="o", ms=3.5, color=c, ls=ls,
                    label=f"{METHOD_LABEL[meth]}{'' if col == 'purity_clean' else ', all hits'}")
    ax.set_xscale("log")
    ax.set_xlabel("cluster hits")
    ax.set_ylabel("mean owner purity")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.66), fontsize=7, ncol=2)
    ax.set_title("Capture-cluster purity vs size (tank); solid = unmatched < 1 PE removed", fontsize=10)
    _save(fig, out, "c06_purity_vs_size")


def purity_table(d: pd.DataFrame, out) -> pd.DataFrame:
    rows = []
    for (name, meth, oc), g in d.groupby(["sample", "method", "owner_class"]):
        rows.append(dict(sample=name, clusterer=METHOD_LABEL[meth], tag=oc, clusters=len(g),
                         purity_all=g.purity.mean(), purity_clean=g.purity_clean.mean(),
                         purity_clean_p10=g.purity_clean.quantile(0.1),
                         pure_100=(g.purity_clean >= 0.9999).mean(),
                         one_neutron=(g.n_neutrons <= 1).mean() if oc == "n_capture" else np.nan,
                         two_plus_captures=(g.n_captures >= 2).mean() if oc == "n_capture" else np.nan,
                         hits_removed=g.comp_untraced_lowpe.sum() / g.n_hits.sum()))
    t = pd.DataFrame(rows)
    t.to_csv(out / "clusterpurity_table.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


def _recall(ctx: RunContext, tag: str) -> pd.DataFrame:
    s = _csv(ctx, "truth_object_summary", tag)
    s = s[(s.obj_class == "n_capture") & s.method.isin(BASE3)]
    cols = ["found", "split", "merged_into_other", "hit_completeness"]
    w = s.n_objects
    g = s.assign(**{c: s[c] * w for c in cols}).groupby("method")[cols + ["n_objects"]].sum()
    return g[cols].div(g.n_objects, axis=0)


def cleaning_metrics(tank, world, tag, clean_tag, out) -> pd.DataFrame:
    rows = []
    for name, ctx in (("tank", tank), ("world", world)):
        for lab, tg in (("none", tag), ("unmatched_lowpe", clean_tag)):
            rc = _recall(ctx, tg)
            d = _purity_clusters(ctx, tg)
            for meth in BASE3:
                g = d[(d.method == meth) & (d.owner_class == "n_capture")]
                rows.append(dict(sample=name, cleaning=lab, clusterer=METHOD_LABEL[meth],
                                 capture_clusters=len(g), purity=g.purity.mean(),
                                 one_neutron=(g.n_neutrons <= 1).mean(),
                                 two_plus_captures=(g.n_captures >= 2).mean(),
                                 median_hits=g.n_hits.median(), **rc.loc[meth].to_dict()))
    t = pd.DataFrame(rows)
    t.to_csv(out / "cleanclustering_table.csv", index=False)
    print(t.round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    tk = t[t["sample"] == "tank"]
    for ax, (col, lab) in zip(axes, (("found", "captures found"), ("split", "captures split"),
                                     ("merged_into_other", "captures merged into another"),
                                     ("purity", "capture-cluster purity"))):
        vals = [tk[tk.cleaning == c].set_index("clusterer").loc[[METHOD_LABEL[m] for m in BASE3], col].to_numpy()
                for c in ("none", "unmatched_lowpe")]
        _bars(ax, [METHOD_LABEL[m] for m in BASE3], ["all MCHit pulses", "unmatched < 1 PE removed"],
              vals, [C[1], C[0]], width=0.36)
        lo = min(v.min() for v in vals)
        if col in ("found", "purity"):
            ax.set_ylim(max(0, lo - 0.05), 1.0)
        ax.set_title(lab, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Clustering without the sub-threshold MCHit pulses (tank, capture objects and clusters)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out, "c07_cleaning_recall_purity")
    return t


def cleaning_features(tank, tag, clean_tag, out):
    feats = [("n_hits", "cluster hits", np.arange(0, 61, 2)),
             ("f_pe_total", "cluster PE", np.arange(0, 121, 4)),
             ("f_charge_bal_legacy", "charge balance", np.linspace(0, 0.6, 31)),
             ("f_sigma_t", "hit-time spread [ns]", np.linspace(0, 20, 41))]
    a, b = _purity_clusters(tank, tag), _purity_clusters(tank, clean_tag)
    fig, axes = plt.subplots(len(BASE3), len(feats), figsize=(13, 7.2))
    for i, meth in enumerate(BASE3):
        ga = a[(a.method == meth) & (a.owner_class == "n_capture")]
        gb = b[(b.method == meth) & (b.owner_class == "n_capture")]
        for j, (col, lab, bins) in enumerate(feats):
            ax = axes[i, j]
            ax.hist(ga[col].dropna(), bins=bins, histtype="step", lw=1.8, color=C[1], density=True,
                    label=f"all, median {ga[col].median():.3g}")
            ax.hist(gb[col].dropna(), bins=bins, histtype="step", lw=1.8, color=C[0], density=True,
                    label=f"cleaned, median {gb[col].median():.3g}")
            ax.legend(loc="upper right", fontsize=6.5)
            ax.set_yticks([])
            if i == len(BASE3) - 1:
                ax.set_xlabel(lab)
            if j == 0:
                ax.set_ylabel(METHOD_LABEL[meth])
    fig.suptitle("Capture-cluster features with and without the sub-threshold MCHit pulses (tank)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "c08_cleaning_features")


def cleaning_truth_content(tank, tag, clean_tag, window_lo, out) -> pd.DataFrame:
    """What delayed clusters hold before (all MCHit pulses) and after the no-waveform < 1 PE cleaning."""
    cats = [("owner", "owner true particle", C[0]), ("owner_rm", "owner, no waveform pulse & < 1 PE", "#9cc3ef"),
            ("other", "another true particle", C[1]), ("other_rm", "another, no waveform pulse & < 1 PE", "#f5b89a"),
            ("dn", "dark noise", "#5f5e5a"), ("unres", "unresolved", NU_COL["noise"])]
    rows = []
    for lab, tg in (("before", tag), ("after", clean_tag)):
        d = _purity_clusters(tank, tg)
        d = d[d.t_med >= window_lo]
        for sel, g0 in (("capture clusters", d[d.owner_class == "n_capture"]), ("all delayed clusters", d)):
            for meth in BASE3:
                g = g0[g0.method == meth]
                n = max(g.n_hits.sum(), 1)
                other = g.n_hits - g.n_owner_hits - g.comp_darknoise - g.comp_untraced
                rows.append(dict(cleaning=lab, selection=sel, clusterer=METHOD_LABEL[meth], method=meth,
                                 clusters=len(g), hits=int(g.n_hits.sum()),
                                 owner=(g.n_owner_hits - g.nowf_lowpe_owner).sum() / n,
                                 owner_rm=g.nowf_lowpe_owner.sum() / n,
                                 other=(other - g.nowf_lowpe_other_obj).sum() / n,
                                 other_rm=g.nowf_lowpe_other_obj.sum() / n,
                                 dn=g.comp_darknoise.sum() / n, unres=g.comp_untraced.sum() / n,
                                 purity=g.owner_frac.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(out / "cleaning_truth_content.csv", index=False)
    print(t.round(4).to_string(index=False))
    short = {"clusterfinder": "CF", "clusterfinder_nms": "CF+NMS", "optics": "OPTICS", "optics_nms": "OPTICS+NMS",
             "peak": "time-peak"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, sel in zip(axes, ("capture clusters", "all delayed clusters")):
        x, labels = [], []
        for i, meth in enumerate(BASE3):
            for k, lab in enumerate(("before", "after")):
                r = t[(t.selection == sel) & (t.method == meth) & (t.cleaning == lab)].iloc[0]
                xx = i * 2.6 + k
                b = 0.0
                for key, name, col in cats:
                    ax.bar(xx, r[key], 0.85, bottom=b, color=col, edgecolor="white", lw=0.6,
                           label=name if (i == 0 and k == 0) else None)
                    b += r[key]
                x.append(xx)
                labels.append(f"{'all' if lab == 'before' else 'clean'}")
        ax.set_xticks(x, labels, fontsize=8)
        for i, meth in enumerate(BASE3):
            ax.text(i * 2.6 + 0.5, -0.12, short[meth], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9)
        ax.set_ylim(0.75, 1.0)
        pur = t[(t.selection == sel)].pivot(index="method", columns="cleaning", values="purity")
        pp = [f"{short[m]} {pur.loc[m, 'before']:.3f}\u2192{pur.loc[m, 'after']:.3f}" for m in BASE3]
        ax.set_title(f"{sel}, purity all hits \u2192 cleaned:\n" + ", ".join(pp[:2]) + "\n" + ", ".join(pp[2:]),
                     fontsize=9)
    axes[0].set_ylabel("share of cluster hits (axis from 0.75)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("Truth content of delayed clusters with all MCHits and after hit cleaning (tank, all events, t \u2265 2 \u00b5s)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save(fig, out, "c10_cleaning_truth_content")
    return t


NMS_PAIRS = (("clusterfinder", "clusterfinder_nms"), ("optics", "optics_nms"))


def nms_effect(tank: RunContext, world: RunContext, tag: str, out, hits: str = "all hits") -> Optional[pd.DataFrame]:
    rows = []
    meths = [m for p in NMS_PAIRS for m in p]
    for name, ctx in (("tank", tank), ("world", world)):
        s = _csv(ctx, "truth_object_summary", tag)
        cl = _pq(ctx, "truth_object_clusters", tag, ["method", "owner_class", "owner_frac", "n_neutrons"])
        if not set(meths) <= set(s.method):
            print(f"[deck] nms_effect skipped: {name} lacks {sorted(set(meths) - set(s.method))}")
            return None
        for m in meths:
            r = dict(sample=name, method=m, clusterer=METHOD_LABEL[m])
            for oc in ("n_capture", "michel", "muon"):
                g = s[(s.method == m) & (s.obj_class == oc)]
                w = g.n_objects.sum()
                for c in ("found", "split", "merged_into_other", "hit_completeness"):
                    r[f"{oc}_{c}"] = (g[c] * g.n_objects).sum() / max(w, 1)
                r[f"{oc}_objects"] = int(w)
            g = cl[(cl.method == m) & (cl.owner_class == "n_capture")]
            r.update(capture_clusters=len(g), capture_purity=g.owner_frac.mean(), one_neutron=(g.n_neutrons <= 1).mean())
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "nms_effect.csv", index=False)
    print(t.round(4).to_string(index=False))
    tk = t[t["sample"] == "tank"].set_index("method")
    order = [m for p in NMS_PAIRS for m in p]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw=dict(width_ratios=[3, 2]))
    groups = [("michel", "Michel e\u00b1 split"), ("muon", "muon split"), ("n_capture", "neutron capture split")]
    w = 0.8 / len(order)
    for i, (oc, _) in enumerate(groups):
        for k, m in enumerate(order):
            v = 100 * tk.loc[m, f"{oc}_split"]
            axes[0].bar(i + (k - (len(order) - 1) / 2) * w, v, w * 0.92, color=METHOD_COLORS[m],
                        label=METHOD_LABEL[m] if i == 0 else None)
    axes[0].set_xticks(range(len(groups)), [g[1] for g in groups])
    axes[0].set_ylabel("particles split into 2+ clusters [%]")
    cf, cfn, op, opn = (100 * tk.loc[m] for m in order)
    axes[0].set_title(f"What NMS fixes: Michel split CF {cf.michel_split:.1f} \u2192 {cfn.michel_split:.1f} %, "
                      f"OPTICS {op.michel_split:.1f} \u2192 {opn.michel_split:.1f} %", fontsize=10)
    cost = [("n_capture_found", "neutron captures\nnot found", True), ("n_capture_merged_into_other",
                                                                      "neutron captures merged\ninto another cluster", False)]
    for i, (col, _, inv) in enumerate(cost):
        for k, m in enumerate(order):
            v = 100 * ((1 - tk.loc[m, col]) if inv else tk.loc[m, col])
            axes[1].bar(i + (k - (len(order) - 1) / 2) * w, v, w * 0.92, color=METHOD_COLORS[m])
    axes[1].set_xticks(range(len(cost)), [c[1] for c in cost])
    axes[1].set_ylabel("neutron captures [%]")
    axes[1].set_title(f"Cost: captures not found CF {100 - cf.n_capture_found:.2f} \u2192 {100 - cfn.n_capture_found:.2f} %, "
                      f"merged {cf.n_capture_merged_into_other:.2f} \u2192 {cfn.n_capture_merged_into_other:.2f} %",
                      fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(order), fontsize=9)
    fig.suptitle(f"Time merge (NMS, 50 ns) after ClusterFinder and OPTICS: tank, all events, {hits}, "
                 "particles with \u2265 5 hits", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, out, "c09_nms_effect")
    return t


def run_clusterpurity(ctx: RunContext, args) -> Path:
    _style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = load_context(args.world_config)
    tag = f"{args.tag}_plurality"
    d = pd.concat([_purity_clusters(ctx, tag).assign(sample="tank"),
                   _purity_clusters(world, tag).assign(sample="world")], ignore_index=True)
    lo = float((ctx.extra.get("residual", {}) or {}).get("prompt_window_ns", 2000.0))
    # a run on already-cleaned hits has nothing left to remove: no before/after figures
    cleaned = d.comp_untraced_lowpe.sum() == 0
    hits = "cleaned hits" if cleaned else "all hits"
    hit_match_vs_pe(ctx, world, lo, out)
    if not cleaned:
        purity_before_after(d, out)
        composition_clean(d, out)
    neutrons_per_cluster(d, out, hits)
    merged_capture_dt(d, out, hits)
    merge_breakdown(d, out)
    if not cleaned:
        purity_vs_size(d, out)
    purity_table(d, out)
    nms_effect(ctx, world, tag, out, hits)
    if args.clean_tag:
        cleaning_metrics(ctx, world, tag, f"{args.clean_tag}_plurality", out)
        cleaning_features(ctx, tag, f"{args.clean_tag}_plurality", out)
        cleaning_truth_content(ctx, tag, f"{args.clean_tag}_plurality", lo, out)
    return out


# --------------------------------------------------------------------------- #
# Neutrino-interaction cluster deck (docs/presentations/ccinc_v3_neutrino_clusters.tex)
# --------------------------------------------------------------------------- #

NU_CLASSES = ["n_capture", "n_other", "muon", "michel", "pion_charged", "pi0", "proton", "em", "other"]
NU_COL = {"n_capture": "#2a78d6", "n_other": "#86b6ef", "muon": "#eb6834", "michel": "#eda100",
          "pion_charged": "#1baf7a", "pi0": "#008300", "proton": "#4a3aa7", "em": "#e87ba4",
          "other": "#898781", "noise": "#c3c2b7"}
NU_LABEL = {**OBJ_LABEL, "n_other": "neutron, no capture", "noise": "noise only"}
SELS = (("all", "all events"), ("cc", "CCinc truth cuts"))
MULT_DEFS = (("tagged", "capture-tagged clusters"), ("delayed", "delayed clusters (2–67 µs, ≥ 5 hits)"),
             ("box", "Box cuts"))
# World CCinc splits by where the neutrino interacted: outside the tank its neutrons are background.
MULT_SAMPLES = (("tank", "cc", "tank CCinc"), ("world", "cc_in", "world CCinc, vertex in tank"),
                ("world", "cc_dirt", "world CCinc, vertex outside"))


def _sel(d: pd.DataFrame, s: str) -> pd.DataFrame:
    if s == "all":
        return d
    d = d[d.cc_pass == 1]
    if s in ("cc_in", "cc_dirt"):
        return d[d.origin_in_tank == (1 if s == "cc_in" else 0)]
    return d


def _nu_clusters(ctx: RunContext, tag: str) -> pd.DataFrame:
    d = _pq(ctx, "truth_object_clusters", f"{tag}_plurality")
    d = d[d.method.isin(BASE3)].copy()
    d["window"] = np.where(d.t_med < float((ctx.extra.get("residual") or {}).get("prompt_window_ns", 2000.0)),
                           "prompt", "delayed")
    return d


def _hit_origin(h: pd.DataFrame) -> np.ndarray:
    cat = np.full(len(h), "other", dtype=object)
    op = h.origin_pdg.to_numpy(int)
    cat[np.abs(op) == 13] = "muon"
    cat[np.abs(op) == 211] = "pion_charged"
    cat[op == 111] = "pi0"
    cat[op == 2212] = "proton"
    cat[op == -5] = "em"
    cat[(h.ancestor_pdg.to_numpy(int) == 2112)] = "neutron"
    cat[(h.truth_class.to_numpy(int) == 0) & (h.is_untraced.to_numpy(int) == 0)] = "dark noise"
    cat[h.is_untraced.to_numpy(int) == 1] = "unresolved"
    return cat


def truth_crosscheck(tank: RunContext, world: RunContext, out) -> pd.DataFrame:
    """MCHit lineage vs the waveform-pulse match, and what the unmatched pulses really are."""
    import pyarrow.parquet as pq
    cols = ["t", "pe", "wf_matched", "truth_class", "is_untraced", "ancestor_pdg", "ancestor_trackID",
            "dp_truth_class", "dp_ancestor_trackID", "origin_pdg", "dp_origin_pdg"]
    edges = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 1e4])
    cats = [("neutron", NU_COL["n_capture"]), ("muon", NU_COL["muon"]), ("pion_charged", NU_COL["pion_charged"]),
            ("pi0", NU_COL["pi0"]), ("proton", NU_COL["proton"]), ("em", NU_COL["em"]),
            ("other", NU_COL["other"]), ("dark noise", "#5f5e5a"), ("unresolved", NU_COL["noise"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    rows = []
    for ax, (name, ctx) in zip(axes, (("tank", tank), ("world", world))):
        h = pq.read_table(ctx.parquet_path(f"{ctx.run_name}__pulses"), columns=cols).to_pandas()
        m = h[h.wf_matched == 1]
        phys = m.dp_truth_class != 0
        rows.append(dict(sample=name, hits=len(h), unmatched=float((h.wf_matched == 0).mean()),
                         agree_neutron_track=float((m.ancestor_trackID == m.dp_ancestor_trackID)[phys].mean()),
                         agree_origin_pdg=float((m.origin_pdg == m.dp_origin_pdg)[phys].mean()),
                         agree_noise=float(((m.truth_class == 0) == (m.dp_truth_class == 0)).mean())))
        u = h[h.wf_matched == 0]
        cat = _hit_origin(u)
        for k, v in pd.Series(cat).value_counts(normalize=True).items():
            rows[-1][f"unmatched_{k}"] = float(v)
        idx = np.digitize(u.pe, edges) - 1
        x = np.arange(len(edges) - 1)
        bottom = np.zeros(len(x))
        for cname, col in cats:
            y = np.array([np.mean(cat[idx == i] == cname) if (idx == i).any() else 0 for i in x])
            ax.bar(x, y, 0.8, bottom=bottom, color=col, label=NU_LABEL.get(cname, cname), edgecolor="white", lw=0.6)
            bottom += y
        ax.set_xticks(x, ["<.25", ".25–.5", ".5–.75", ".75–1", "1–1.5", "1.5–2", "2–3", "3–5", ">5"], fontsize=8)
        ax.set_xlabel("MCHit pulse PE")
        ax.set_title(f"{name}: {rows[-1]['unmatched']:.1%} of MCHit pulses have no waveform pulse", fontsize=10)
    axes[0].set_ylabel("share of unmatched pulses")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    fig.suptitle("True origin of the MCHit pulses the hit finder does not record (all hits, 0–70 µs)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "n01_unmatched_origin")
    return pd.DataFrame(rows)


def lowpe_in_clusters(d: pd.DataFrame, out) -> pd.DataFrame:
    """Where the removed (< 1 PE, no waveform pulse) hits inside clusters come from."""
    parts = [("nowf_lowpe_owner", "same particle as the cluster owner", C[0]),
             ("nowf_lowpe_other_obj", "another particle", C[1]),
             ("nowf_lowpe_darknoise", "dark noise", "#5f5e5a"),
             ("nowf_lowpe_untraced", "unresolved", NU_COL["noise"])]
    short = {"clusterfinder": "CF", "clusterfinder_nms": "CF\n+NMS", "optics": "OPTICS",
             "optics_nms": "OPTICS\n+NMS", "peak": "peak"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    rows = []
    for ax, name in zip(axes, ("tank", "world")):
        g0 = d[(d["sample"] == name) & (d.window == "delayed") & d.owner_class.isin(["n_capture", "michel"])]
        grp = list(g0.groupby(["owner_class", "method"]))
        vals = np.array([[g[k].sum() / max(g.comp_untraced_lowpe.sum(), 1) for k, _, _ in parts] for _, g in grp])
        x = np.arange(len(grp))
        bottom = np.zeros(len(x))
        for j, (k, lab, col) in enumerate(parts):
            ax.bar(x, vals[:, j], 0.75, bottom=bottom, color=col, label=lab, edgecolor="white", lw=0.6)
            bottom += vals[:, j]
        ax.set_xticks(x, [f"{short[m]}\n{'capture' if o == 'n_capture' else 'Michel'}" for (o, m), _ in grp],
                      fontsize=8)
        ax.set_title(name)
        for (o, m), g in grp:
            rows.append(dict(sample=name, owner=o, clusterer=METHOD_LABEL[m], clusters=len(g),
                             removed_share_of_hits=g.comp_untraced_lowpe.sum() / g.n_hits.sum(),
                             **{k.replace("nowf_lowpe_", "from_"): g[k].sum() / max(g.comp_untraced_lowpe.sum(), 1)
                                for k, _, _ in parts}))
    axes[0].set_ylabel("share of removed hits")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    t = pd.DataFrame(rows)
    cap = t[(t.owner == "n_capture") & (t["sample"] == "tank")]
    fig.suptitle("Unmatched < 1 PE hits inside delayed clusters: "
                 + ", ".join(f"{r.clusterer} {r.from_owner:.0%}" for r in cap.itertuples())
                 + " come from the capture itself (tank)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "n02_lowpe_in_clusters")
    return t


def multiplicity_truth(ev: dict, out):
    cols = [("trueNeutrons", "GENIE final-state neutrons"), ("n_true_captures", "captures in 2–70 µs"),
            ("n_visible_captures_1", "captures with ≥ 1 hit"), ("n_visible_captures_5", "captures with ≥ 5 hits")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, name in zip(axes, ("tank", "world")):
        e = _sel(ev[name], "cc")
        x = np.arange(0, 8)
        for k, ((col, lab), c) in enumerate(zip(cols, C)):
            v = e[col].clip(upper=7).value_counts(normalize=True).reindex(x, fill_value=0)
            ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=c, label=f"{lab} (mean {e[col].mean():.2f})")
        ax.set_xticks(x, [str(i) for i in range(7)] + ["7+"])
        ax.set_xlabel("per event")
        ax.set_title(f"{name}, CCinc truth cuts ({len(e):,} events)")
    axes[0].set_ylabel("fraction of events")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("True neutron multiplicity: produced, captured in the window, and visible", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "n03_true_multiplicity")


def multiplicity_overlay(ev: dict, title: str, name: str, out, nmax=5):
    """True and reco neutron counts per event, one step histogram each (integer bins, last = overflow)."""
    from ..efficiency_extraction.efficiency_thesis import _step
    series = [("n_true_captures", "true captures in 2–70 µs", INK, "-"),
              ("n_visible_captures_5", "true captures with ≥ 5 hits", INK2, "--")]
    series += [(f"n_reco_{d}", lab, col, "-") for (d, lab), col in zip(MULT_DEFS, C)]
    fig, axes = plt.subplots(len(MULT_SAMPLES), len(BASE3), figsize=(4.3 * len(BASE3), 3.2 * len(MULT_SAMPLES)),
                             sharex=True)
    for i, (smp, s, slab) in enumerate(MULT_SAMPLES):
        e = _sel(ev[smp], s)
        for j, m in enumerate(BASE3):
            ax = axes[i, j]
            for col, lab, colr, ls in series:
                key = col if col.startswith("n_true") or col.startswith("n_visible") else f"{col}_{m}"
                if key not in e.columns:
                    continue
                cnt = np.bincount(e[key].clip(upper=nmax).to_numpy(int), minlength=nmax + 1)
                _step(ax, cnt, color=colr, ls=ls, lw=1.8, label=f"{lab} (mean {e[key].mean():.2f})")
            ax.set_xticks(range(nmax + 1), [str(k) for k in range(nmax)] + [f"{nmax}+"])
            ax.set_title(f"{METHOD_LABEL[m]}: {slab}, {len(e)} events", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right")
            if j == 0:
                ax.set_ylabel("events")
            if i == len(MULT_SAMPLES) - 1:
                ax.set_xlabel("neutrons per event")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, name)


def mult_table(ev_by_tag: dict) -> pd.DataFrame:
    rows = []
    for (tag, sample), e0 in ev_by_tag.items():
        for s in ("all", "cc") if sample == "tank" else ("all", "cc", "cc_in", "cc_dirt"):
            e = _sel(e0, s)
            for m in BASE3:
                for dname, _ in MULT_DEFS:
                    col = f"n_reco_{dname}_{m}"
                    if col not in e.columns:
                        continue
                    rows.append(dict(run=tag, sample=sample, selection=s, clusterer=METHOD_LABEL[m], reco=dname,
                                     events=len(e), mean_true=e.n_true_captures.mean(),
                                     mean_visible5=e.n_visible_captures_5.mean(), mean_reco=e[col].mean(),
                                     equal_true=(e[col] == e.n_true_captures).mean(),
                                     equal_visible5=(e[col] == e.n_visible_captures_5).mean()))
    return pd.DataFrame(rows)


def objects_by_class(ob: dict, out) -> pd.DataFrame:
    """Truth objects (>= 5 hits) found / split / merged per particle class and clusterer."""
    rows = []
    for (name, s), o in ob.items():
        v = o[o.n_hits >= 5]
        for cls in NU_CLASSES:
            g = v[v.obj_class == cls]
            if len(g) < 20:
                continue
            for m in BASE3:
                rows.append(dict(sample=name, selection=s, obj_class=cls, clusterer=METHOD_LABEL[m], objects=len(g),
                                 per_event=len(g) / max(o.eventID.nunique(), 1),
                                 found=(g[f"n_owned_{m}"] >= 1).mean(), split=(g[f"n_owned_{m}"] >= 2).mean(),
                                 merged=(g[f"n_merged_into_{m}"] >= 1).mean(),
                                 completeness=g[f"best_hits_{m}"].sum() / max(g.n_hits.sum(), 1)))
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharey="row")
    for j, s in enumerate(("all", "cc")):
        tt = t[(t["sample"] == "tank") & (t.selection == s)]
        classes = [c for c in NU_CLASSES if c in set(tt.obj_class)]
        for i, (metric, lab) in enumerate((("found", "found (owns a cluster)"), ("merged", "merged into another particle's cluster"))):
            ax = axes[i, j]
            vals = [[tt[(tt.obj_class == c) & (tt.clusterer == METHOD_LABEL[m])][metric].squeeze() for c in classes] for m in BASE3]
            _bars(ax, [NU_LABEL[c] for c in classes], [METHOD_LABEL[m] for m in BASE3], vals, [METHOD_COLORS[m] for m in BASE3],
                  width=0.8 / len(BASE3))
            ax.tick_params(axis="x", labelsize=7.5, rotation=20)
            ax.set_ylabel(lab if j == 0 else "")
            ax.set_title(f"tank, {dict(SELS)[s]}", fontsize=10)
            ax.set_ylim(0, 1.05)
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    fig.suptitle("Is every particle found as its own cluster? Truth objects with ≥ 5 hits, full 0–70 µs window", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "n06_objects_by_class")
    return t


def clusters_by_owner(d: pd.DataFrame, out, hits_label: str) -> pd.DataFrame:
    rows = []
    for (name, s), g0 in ((("tank", "all"), d[d["sample"] == "tank"]), (("tank", "cc"), _sel(d[d["sample"] == "tank"], "cc")),
                          (("world", "all"), d[d["sample"] == "world"]), (("world", "cc"), _sel(d[d["sample"] == "world"], "cc"))):
        n_ev = max(g0.eventID.nunique(), 1)
        for (m, w, oc), g in g0.groupby(["method", "window", "owner_class"]):
            rows.append(dict(sample=name, selection=s, clusterer=METHOD_LABEL[m], window=w, owner=oc, clusters=len(g),
                             per_event=len(g) / n_ev, purity=g.owner_frac.mean(), pure_100=(g.owner_frac >= 0.9999).mean(),
                             purity_ge_90=(g.owner_frac >= 0.9).mean(), multi_particle=(g.n_objects_contrib >= 2).mean(),
                             median_hits=g.n_hits.median()))
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, len(BASE3), figsize=(4.3 * len(BASE3), 6.4), sharey="row")
    tt = t[(t["sample"] == "tank") & (t.selection == "cc")]
    for j, m in enumerate(BASE3):
        for i, w in enumerate(("prompt", "delayed")):
            ax = axes[i, j]
            g = tt[(tt.clusterer == METHOD_LABEL[m]) & (tt.window == w)].set_index("owner")
            classes = [c for c in NU_CLASSES + ["noise"] if c in g.index and g.loc[c, "clusters"] >= 10]
            x = np.arange(len(classes))
            ax.bar(x, [g.loc[c, "purity"] for c in classes], 0.7, color=[NU_COL[c] for c in classes])
            ax.set_xticks(x, [f"{NU_LABEL[c]}\n{g.loc[c, 'per_event']:.2f}/event" for c in classes], fontsize=7.5)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{METHOD_LABEL[m]}, {w} clusters", fontsize=10)
            if j == 0:
                ax.set_ylabel("mean owner purity")
    fig.suptitle(f"Cluster purity by the particle that owns it (tank, CCinc truth cuts, {hits_label})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "n07_clusters_by_owner")
    return t


def prompt_mixture(d: pd.DataFrame, out):
    comp = [f"comp_{c}" for c in NU_CLASSES] + ["comp_darknoise", "comp_untraced"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for ax, name in zip(axes, ("tank", "world")):
        g0 = _sel(d[(d["sample"] == name) & (d.window == "prompt")], "cc")
        x = np.arange(len(BASE3))
        bottom = np.zeros(len(x))
        for cc in comp:
            key = cc.replace("comp_", "")
            y = np.array([g0[g0.method == m][cc].sum() / max(g0[g0.method == m].n_hits.sum(), 1) for m in BASE3])
            col = NU_COL.get(key, "#5f5e5a" if key == "darknoise" else NU_COL["noise"])
            ax.bar(x, y, 0.6, bottom=bottom, color=col, label=NU_LABEL.get(key, key.replace("darknoise", "dark noise")),
                   edgecolor="white", lw=0.6)
            bottom += y
        mix = [f"{(g0[g0.method == m].n_objects_contrib >= 2).mean():.0%}" for m in BASE3]
        ax.set_xticks(x, [f"{METHOD_LABEL[m]}\n{k} multi-particle" for m, k in zip(BASE3, mix)], fontsize=8)
        ax.set_title(f"{name}: prompt clusters (< 2 µs), CCinc", fontsize=10)
    axes[0].set_ylabel("share of prompt-cluster hits")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    fig.suptitle("What the prompt clusters are made of", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "n08_prompt_mixture")


def randoms_summary(rctx: RunContext, out, tag: str = "", overlay_run="6256") -> pd.DataFrame:
    from ..data.randoms import pool_label, random_samples
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    rows = []
    samples = random_samples(rctx, overlay_run, tag)
    col = dict(RUN_COLORS, **({} if str(overlay_run) == "6256" else {6256: "#9a9a9a"}))
    for k, (run, lab, tr, h, rc, cf) in enumerate(samples):
        n = len(tr)
        axes[0].hist(tr.nhits.clip(upper=300), bins=np.arange(0, 305, 5), histtype="step", lw=2, color=col[run],
                     density=True, label=f"{lab}, median {tr.nhits.median():.0f}")
        tb = np.arange(0, 70001, 1000)
        rate = np.histogram(h.t, tb)[0] / n / 1.0
        axes[1].step(tb[:-1] / 1000, rate, where="post", color=col[run], label=lab)
        from ..data.selection import BoxSelection
        box = BoxSelection()
        for m in BASE3:
            g = rc[rc.method == m]
            if g.empty:
                raise SystemExit(f"[provenance] random_recoclusters_{run} has no {m} clusters; re-run "
                                 "`ambe data randoms --cluster` with that method")
            b = np.logical_and.reduce(box.cut_masks(g.f_pe_total.to_numpy(float), g.f_charge_bal_legacy.to_numpy(float),
                                                    g.t_med.to_numpy(float), g.n_hits.to_numpy(int)))
            dl = (g.t_med >= box.cuts.ct_min) & (g.t_med <= box.cuts.ct_max) & (g.n_hits >= box.cuts.chits_min)
            rows.append(dict(run=run, triggers=n, clusterer=METHOD_LABEL[m], clusters_per_trigger=len(g) / n,
                             delayed_per_trigger=dl.sum() / n, box_per_trigger=b.sum() / n,
                             hits_per_trigger=len(h) / n, median_hits=tr.nhits.median(),
                             khz_per_pmt=len(h) / n / h.pmtID.nunique() / 70.0 * 1e3,
                             tree_cf_per_trigger=len(cf) / n))
    t = pd.DataFrame(rows)
    axes[0].set_xlabel("tank hits per trigger")
    axes[0].set_ylabel("triggers (normalised)")
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("hits per random trigger")
    axes[1].set_xlabel("hit time [µs]")
    axes[1].set_ylabel("hits per trigger per µs")
    axes[1].set_title("hit rate vs time")
    x = np.arange(len(BASE3))
    ns = len(samples)
    for k, (run, lab, *_) in enumerate(samples):
        g = t[t.run == run]
        axes[2].bar(x + (k - (ns - 1) / 2) * 0.76 / ns, g.box_per_trigger, 0.72 / ns, color=col[run],
                    label=str(run) if str(overlay_run) == "6256" else lab)
    axes[2].set_xticks(x, [METHOD_LABEL[m] for m in BASE3])
    axes[2].set_ylabel("box-passing clusters per trigger")
    axes[2].set_title("noise clusters passing the box")
    axes[2].legend(fontsize=7.5)
    fig.suptitle("Dummy-trigger runs: random tank activity (6256 is the LAPPD-off noise sample)"
                 if str(overlay_run) == "6256" else
                 f"Background sample {pool_label(overlay_run)}, cosmic veto, no IC gate: random tank activity", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "n09_randoms")
    return t


def overlay_effect(cl: dict, ev: dict, out, randoms: Optional[pd.DataFrame] = None,
                   labels: Optional[dict] = None, overlay_run="6256") -> pd.DataFrame:
    from ..data.randoms import pool_label
    ok = 6256 if str(overlay_run) == "6256" else str(overlay_run)
    olab = pool_label(overlay_run)
    rows = []
    for (tag, name), d in cl.items():
        d = _sel(d[d.window == "delayed"], "cc")
        e = _sel(ev[(tag, name)], "cc")
        n_ev = max(len(e), 1)
        for m in BASE3:
            g = d[d.method == m]
            cap = g[g.owner_class == "n_capture"]
            rows.append(dict(run=tag, sample=name, clusterer=METHOD_LABEL[m],
                             capture_purity=cap.owner_frac.mean(), capture_pure_100=(cap.owner_frac >= 0.9999).mean(),
                             noise_share_capture=(cap.comp_darknoise.sum()) / max(cap.n_hits.sum(), 1),
                             overlay_share_capture=cap.comp_overlay.sum() / max(cap.n_hits.sum(), 1) if "comp_overlay" in cap else np.nan,
                             noise_clusters_per_event=(g.owner_class == "noise").sum() / n_ev,
                             delayed_per_event=e[f"n_reco_delayed_{m}"].mean(),
                             box_per_event=e[f"n_reco_box_{m}"].mean() if f"n_reco_box_{m}" in e else np.nan,
                             tagged_per_event=e[f"n_reco_tagged_{m}"].mean(), true_per_event=e.n_true_captures.mean()))
    t = pd.DataFrame(rows)
    if randoms is not None:
        # ClusterFinder clusters come from the tree, so overlaid hits never form new ones:
        # add the rate ClusterFinder (+ NMS) finds on the random sample alone.
        t[f"delayed_plus_{overlay_run}_cf"] = np.nan
        t[f"box_plus_{overlay_run}_cf"] = np.nan
        for m in ("clusterfinder", "clusterfinder_nms"):
            r = randoms[(randoms.run == ok) & (randoms.clusterer == METHOD_LABEL[m])]
            cf = (t.clusterer == METHOD_LABEL[m]).to_numpy()
            if r.empty or not cf.any():
                continue
            t.loc[cf, f"delayed_plus_{overlay_run}_cf"] = t.delayed_per_event[cf] + r.delayed_per_trigger.iloc[0]
            t.loc[cf, f"box_plus_{overlay_run}_cf"] = t.box_per_event[cf] + r.box_per_trigger.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    tt = t[t["sample"] == "tank"]
    runs = list(dict.fromkeys(tt.run))
    for ax, (col, lab) in zip(axes, (("capture_purity", "capture-cluster purity"), ("box_per_event", "box clusters per event"),
                                      ("noise_clusters_per_event", "noise-only clusters per event"))):
        vals = [[tt[(tt.run == r) & (tt.clusterer == METHOD_LABEL[m])][col].squeeze() for m in BASE3] for r in runs]
        _bars(ax, [METHOD_LABEL[m] for m in BASE3], [(labels or {}).get(r, r) for r in runs], vals,
              ["#b5b5b5", RUN_COLORS[ok]][:len(runs)], width=0.8 / len(runs))
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylim(0.8, 1.0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(runs), fontsize=9)
    fig.suptitle(f"Adding {olab} random-trigger hits to MC (tank, CCinc, delayed window; "
                 "ClusterFinder cannot make clusters from added hits)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    _save(fig, out, "n10_overlay_effect")
    return t


def run_neutrino(ctx: RunContext, args) -> Path:
    from ..data.randoms import pool_label
    _style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = load_context(args.world_config)
    tags = [t for t in (args.tag, args.clean_tag, args.overlay_tag) if t]
    cl, ev, ob = {}, {}, {}
    for tag in tags:
        for name, c in (("tank", ctx), ("world", world)):
            cl[(tag, name)] = _nu_clusters(c, tag).assign(sample=name)
            ev[(tag, name)] = _pq(c, "truth_event_multiplicity", f"{tag}_plurality")
    for name, c in (("tank", ctx), ("world", world)):
        o = _pq(c, "truth_objects", f"{args.clean_tag}_plurality")
        ob[(name, "all")], ob[(name, "cc")] = o, o[o.cc_pass == 1]
    tables = {}
    tables["crosscheck"] = truth_crosscheck(ctx, world, out)
    d_none = pd.concat([cl[(args.tag, n)] for n in ("tank", "world")], ignore_index=True)
    tables["lowpe"] = lowpe_in_clusters(d_none, out)
    multiplicity_truth({n: ev[(args.clean_tag, n)] for n in ("tank", "world")}, out)
    multiplicity_overlay({n: ev[(args.clean_tag, n)] for n in ("tank", "world")},
                         "Neutrons per CCinc event: truth vs clusters found on the same hits (cleaned hits)",
                         "n04_multiplicity", out)
    if args.overlay_tag:
        multiplicity_overlay({n: ev[(args.overlay_tag, n)] for n in ("tank", "world")},
                             f"Same with {pool_label(args.overlay_run)} random-trigger hits added "
                             "(ClusterFinder cannot form clusters from them)",
                             "n05_multiplicity_overlay", out)
    tables["multiplicity"] = mult_table({k: v for k, v in ev.items()})
    tables["objects"] = objects_by_class(ob, out)
    d_clean = pd.concat([cl[(args.clean_tag, n)] for n in ("tank", "world")], ignore_index=True)
    tables["clusters_by_owner"] = pd.concat(
        [clusters_by_owner(d_clean, out, "cleaned hits").assign(hits="cleaned"),
         clusters_by_owner(d_none, out, "all hits").assign(hits="all")], ignore_index=True)
    prompt_mixture(d_clean, out)
    if args.randoms_config:
        tables["randoms"] = randoms_summary(load_context(args.randoms_config), out, args.randoms_tag,
                                            args.overlay_run)
    tables["overlay"] = overlay_effect({k: v for k, v in cl.items() if k[0] != args.tag}, ev, out,
                                       tables.get("randoms"),
                                       {args.clean_tag: "MC, cleaned hits",
                                        args.overlay_tag: f"MC + {pool_label(args.overlay_run)} hits"},
                                       args.overlay_run)
    pd.set_option("display.width", 250)
    for k, t in tables.items():
        t.to_csv(out / f"neutrino_{k}.csv", index=False)
        print(f"\n== {k} ==\n{t.round(4).to_string(index=False)}")
    return out


def run(ctx: RunContext, args) -> Path:
    _style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = load_context(args.world_config)
    cap = load_context(args.capture_config)
    split_rates(ctx, world, out)
    trackids_per_cluster(ctx, out)
    failure_reasons(ctx, world, out)
    window_scan(ctx, out)
    window_purity(ctx, world, out)
    pair_separation(ctx, out)
    trackids_per_capture(cap, args.capture_tag, out)
    merged_is_one_capture(cap, args.capture_tag, out)
    background_composition(cap, args.capture_tag, out)
    label_threshold_scan(cap, args.capture_tag, out)
    clusterer_benchmark(cap, args.capture_tag, out)
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots provenance")
    p.add_argument("--deck", choices=["provenance", "truthfix", "clusterpurity", "neutrino"], required=True,
                   help="provenance = ccinc_v3_cluster_provenance.tex figures; truthfix = "
                        "ccinc_v3_truthfix.tex (old vs fixed DirectParent labels; --config is the fixed tank run); "
                        "clusterpurity = ccinc_v3_clusterpurity.tex (--config tank, --world-config world, --tag)")
    p.add_argument("--world-config", required=True)
    p.add_argument("--capture-config")
    p.add_argument("--capture-tag")
    p.add_argument("--old-config")
    p.add_argument("--old-world-config")
    p.add_argument("--tag", help="truthfix/clusterpurity: particle-mode tag of the runs")
    p.add_argument("--clean-tag", help="clusterpurity: tag of the --hit-cleaning unmatched_lowpe runs; "
                                        "adds the with/without-cleaning comparison (needs --cluster-features)")
    p.add_argument("--overlay-tag", help="neutrino: tag of the --overlay run")
    p.add_argument("--randoms-config", help="neutrino: `ambe data randoms` config (6254/6256)")
    p.add_argument("--randoms-tag", default="", help="neutrino: tag of the `ambe data randoms --cluster` run")
    p.add_argument("--overlay-run", help="neutrino: random sample of the --overlay run: 6256, or a --pool name "
                                         "such as 6264p6256")
    p.add_argument("--plot-methods", help="clusterpurity/neutrino: comma list of clusterers to plot, from "
                                          + ",".join(METHOD_LABEL))
    p.add_argument("--old-tag", help="truthfix: particle-mode tag of the old-label runs")
    p.add_argument("--tids-tag", help="truthfix: capture-unit tag with trackIDs per capture")
    p.add_argument("--scan-tag", help="truthfix: tag of the --scan-peak run on the fixed tank")
    p.add_argument("--out", required=True)
    a = p.parse_args(list(argv) if argv else [])
    need = {"provenance": ["capture_config", "capture_tag"],
            "truthfix": ["old_config", "old_world_config", "tag", "old_tag", "tids_tag", "scan_tag"],
            "clusterpurity": ["tag", "plot_methods"], "neutrino": ["tag", "clean_tag", "plot_methods"]}[a.deck]
    miss = [f"--{k.replace('_', '-')}" for k in need if getattr(a, k) is None]
    if miss:
        p.error(f"--deck {a.deck} needs {' '.join(miss)}")
    if a.deck == "neutrino" and (a.overlay_tag or a.randoms_config) and not a.overlay_run:
        p.error("--deck neutrino with --overlay-tag/--randoms-config needs --overlay-run")
    if a.plot_methods:
        bad = set(a.plot_methods.split(",")) - set(METHOD_LABEL)
        if bad:
            p.error(f"unknown --plot-methods {sorted(bad)}")
        BASE3[:] = a.plot_methods.split(",")
    {"truthfix": run_truthfix, "clusterpurity": run_clusterpurity, "neutrino": run_neutrino,
     "provenance": run}[a.deck](ctx, a)
