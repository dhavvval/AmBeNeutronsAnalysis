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
        --out PRESENTATION_PLOTS/ccinc_clusterpurity

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
from ..plotting import set_style

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
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
    ax.set_xticks(x, groups)


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
             "pion": "pion", "proton": "proton", "em": "EM (e/\u03b3)", "n_other": "neutron, no capture",
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


def delayed_objects(old, new, old_tag, new_tag, out):
    order = ["n_capture", "michel", "em", "n_other", "muon", "pion", "proton"]
    obs = [_pq(c, "truth_objects", t, ["obj_class", "n_hits"]) for c, t in ((old, old_tag), (new, new_tag))]
    cnt = [o.obj_class.value_counts() for o in obs]
    hits = [o.groupby("obj_class").n_hits.sum() for o in obs]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    y = np.arange(len(order))
    for ax, vals, lab in ((axes[0], cnt, "truth objects"), (axes[1], hits, "hits in those objects")):
        for k, (c, col, name) in enumerate(zip(vals, (OLD, NEW), ("old labels", "both fixes"))):
            ax.barh(y + (k - 0.5) * 0.38, [max(c.get(o, 0), 0.5) for o in order], 0.35, color=col, label=name)
        ax.set_xscale("log")
        ax.set_xlabel(f"{lab}, delayed window (2\u201370 \u00b5s)")
    axes[0].set_yticks(y, [OBJ_LABEL[o] for o in order])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    tot = [int(h.sum()) for h in hits]
    fake = sum(cnt[0].get(o, 0) for o in ("muon", "pion", "proton"))
    fake_n = sum(cnt[1].get(o, 0) for o in ("muon", "pion", "proton"))
    fig.suptitle(f"Same delayed hits, regrouped: muon/pion/proton objects {fake:,} \u2192 {fake_n:,}; "
                 f"hits in objects {tot[0]:,} \u2192 {tot[1]:,} (the rest is now dark noise) (tank)", fontsize=10.5)
    _save(fig, out, "t02_delayed_objects")


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
    order = ["n_capture", "michel", "muon", "pion", "proton", "em", "n_other"]
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
    delayed_objects(old, ctx, ot, nt, out)
    prompt = float((ctx.extra.get("residual", {}) or {}).get("prompt_window_ns", 2000.0))
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
    d["purity_clean"] = d.n_owner_hits / (d.n_hits - d.comp_untraced_lowpe).clip(lower=1)
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
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
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
                other = (g.n_hits - g.comp_untraced - g.comp_darknoise - g.n_owner_hits).sum()
                rows.append([g.n_owner_hits.sum() / n, other / n, g.comp_darknoise.sum() / n,
                             (g.comp_untraced - g.comp_untraced_lowpe).sum() / n])
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


def neutrons_per_cluster(d: pd.DataFrame, out):
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
    fig.suptitle("One neutron per capture cluster (tank): " + ", ".join(
        f"{METHOD_LABEL[m]} {one[m]:.1%}" for m in BASE3), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "c04_neutrons_per_cluster")


def merged_capture_dt(d: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.logspace(0, 5, 31)
    parts = []
    for meth, c in zip(BASE3, C):
        g = d[(d.method == meth) & (d.owner_class == "n_capture") & (d.n_captures >= 2)]
        ax.hist(g.dt_captures_min.dropna(), bins=bins, histtype="step", lw=2, color=c,
                label=f"{METHOD_LABEL[meth]} ({len(g)} clusters)")
        parts.append(f"{METHOD_LABEL[meth]} {g.dt_captures_min.median():.0f} ns")
    ax.set_xscale("log")
    ax.set_xlabel("time between the merged captures [ns]")
    ax.set_ylabel("capture clusters with 2+ captures")
    ax.legend(loc="upper left")
    ax.set_title("Merged captures, tank + world; median " + ", ".join(parts), fontsize=9)
    _save(fig, out, "c05_merged_capture_dt")


def purity_vs_size(d: pd.DataFrame, out):
    bins = np.array([3, 6, 8, 10, 13, 17, 22, 30, 45, 70, 1000])
    xc = np.sqrt(bins[:-1] * np.minimum(bins[1:], 100))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for meth, c in zip(BASE3, C):
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


def run_clusterpurity(ctx: RunContext, args) -> Path:
    _style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = load_context(args.world_config)
    tag = f"{args.tag}_plurality"
    d = pd.concat([_purity_clusters(ctx, tag).assign(sample="tank"),
                   _purity_clusters(world, tag).assign(sample="world")], ignore_index=True)
    lo = float((ctx.extra.get("residual", {}) or {}).get("prompt_window_ns", 2000.0))
    hit_match_vs_pe(ctx, world, lo, out)
    purity_before_after(d, out)
    composition_clean(d, out)
    neutrons_per_cluster(d, out)
    merged_capture_dt(d, out)
    purity_vs_size(d, out)
    purity_table(d, out)
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
    p.add_argument("--deck", choices=["provenance", "truthfix", "clusterpurity"], required=True,
                   help="provenance = ccinc_v3_cluster_provenance.tex figures; truthfix = "
                        "ccinc_v3_truthfix.tex (old vs fixed DirectParent labels; --config is the fixed tank run); "
                        "clusterpurity = ccinc_v3_clusterpurity.tex (--config tank, --world-config world, --tag)")
    p.add_argument("--world-config", required=True)
    p.add_argument("--capture-config")
    p.add_argument("--capture-tag")
    p.add_argument("--old-config")
    p.add_argument("--old-world-config")
    p.add_argument("--tag", help="truthfix/clusterpurity: particle-mode tag of the runs")
    p.add_argument("--old-tag", help="truthfix: particle-mode tag of the old-label runs")
    p.add_argument("--tids-tag", help="truthfix: capture-unit tag with trackIDs per capture")
    p.add_argument("--scan-tag", help="truthfix: tag of the --scan-peak run on the fixed tank")
    p.add_argument("--out", required=True)
    a = p.parse_args(list(argv) if argv else [])
    need = {"provenance": ["capture_config", "capture_tag"],
            "truthfix": ["old_config", "old_world_config", "tag", "old_tag", "tids_tag", "scan_tag"],
            "clusterpurity": ["tag"]}[a.deck]
    miss = [f"--{k.replace('_', '-')}" for k in need if getattr(a, k) is None]
    if miss:
        p.error(f"--deck {a.deck} needs {' '.join(miss)}")
    {"truthfix": run_truthfix, "clusterpurity": run_clusterpurity, "provenance": run}[a.deck](ctx, a)
