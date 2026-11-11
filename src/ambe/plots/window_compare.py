"""
Delayed window 2-70 us vs 10-70 us for the clusterers of the CCinc full deck, from the existing
`mc capture --hit-window full` outputs. Those runs cluster every hit of the event, so moving the start
of the delayed window only changes which clusters and true particles are counted; nothing is re-clustered.

    ambe plots windowcompare --config configs/cc_neutrino_v3fixnufull_truthtag.yaml \
        --world-config configs/cc_neutrino_v3fixnufullworld_truthtag.yaml --tag nuclean_all \
        --overlay-tag nuover_all --randoms-config configs/data_dummy_randoms_625x.yaml --randoms-tag nufull \
        --overlay-run 6256 \
        --windows 2000,10000 --plot-methods clusterfinder,clusterfinder_nms,peak \
        --out PRESENTATION_PLOTS/ccinc_window_compare_full_clean

Window [lo, 70 us] for true particles and capture/background clusters (cluster median time); the
"delayed >= 5 hits" and box counts keep their 67 us end (BoxSelection), with ct_min = lo.

  wc_objects.csv       true particles with >= 5 hits in the window: found, split, merged, completeness
  wc_clusters.csv      clusters in the window by owner: per event, share, purity, one neutron, features
  wc_multiplicity.csv  CCinc events: visible captures vs tagged / delayed / box clusters per event
  wc_randoms.csv       random-trigger clusters per trigger in the window (--overlay-run 6256: 6256 / 6254;
                       a pool: the pool and each of its runs)
  wc_overlay.csv       tank CCinc with the random hits added: capture purity and noise clusters per event
  wc_overlay_detail.csv  tank and world CCinc: noise hits inside capture clusters vs noise-only clusters
  wc_tankworld.csv     tank vs world CCinc: capture share and non-capture clusters by owner
  wc_noise_features.csv  noise-only clusters (random sample clustered alone) vs MC capture clusters
  w0*.pdf, tab_wc_*.tex, wc_numbers.tex (one macro per number quoted on the slides)
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..data.randoms import pool_label, random_samples
from ..data.selection import BoxSelection
from ..plotting import METHOD_COLORS, set_style
from .cluster_truth import METHOD_LABELS

HI_NS = 70000.0
KEY = ["_source_file", "eventNumber"]
OWNER_LABEL = {"n_capture": "neutron capture", "michel": "Michel e±", "muon": "muon", "n_other": "neutron, no capture",
               "pion_charged": "π±", "pi0": "π⁰", "proton": "proton", "em": "EM (e/γ)", "other": "other",
               "noise": "noise only"}
OWNER_TEX = {"n_capture": "neutron capture", "michel": "Michel e$^\\pm$", "muon": "muon",
             "n_other": "neutron, no capture", "pion_charged": "$\\pi^\\pm$", "pi0": "$\\pi^0$", "proton": "proton",
             "em": "EM (e/$\\gamma$)", "other": "other", "noise": "noise only"}
OWNER_COL = {"n_capture": "#2a78d6", "n_other": "#86b6ef", "muon": "#eb6834", "michel": "#eda100",
             "pion_charged": "#1baf7a", "pi0": "#008300", "proton": "#4a3aa7", "em": "#e87ba4", "other": "#898781",
             "noise": "#000000"}
METHOD_TEX = {"clusterfinder": "\\CF", "clusterfinder_nms": "\\CFN", "peak": "\\TP", "optics_nms": "\\OPN",
              "optics": "\\OP"}
SELS = (("all", "all events"), ("cc", "CCinc"))
MULT = (("tank", "cc", "tank CCinc"), ("world", "cc_in", "world CCinc, vertex in tank"),
        ("world", "cc_dirt", "world CCinc, vertex outside"))


def _wlab(lo: float) -> str:
    return f"{lo / 1000:g}–70 µs"


def _wtex(lo: float) -> str:
    return f"{lo / 1000:g}--70\\,$\\mu$s"


def _box(lo: float) -> BoxSelection:
    return BoxSelection(replace(BoxSelection().cuts, ct_min=lo))


def _load(ctx: RunContext, tag: str, methods: list) -> tuple:
    p = lambda b: ctx.parquet_path(f"{ctx.run_name}__{b}__{tag}_plurality")
    cl = pd.read_parquet(p("truth_object_clusters"))
    cl = cl[cl.method.isin(methods)].copy()
    for k in ("nowf_lowpe_owner",):
        if k not in cl:
            cl[k] = 0
    cl["purity_clean"] = (cl.n_owner_hits - cl.nowf_lowpe_owner) / (cl.n_hits - cl.comp_untraced_lowpe).clip(lower=1)
    ob = pd.read_parquet(p("truth_objects"))
    ev = pd.read_parquet(p("truth_event_multiplicity"), columns=KEY + ["cc_pass", "origin_in_tank"])
    org = ev[KEY + ["origin_in_tank"]]
    return cl.merge(org, on=KEY, how="left"), ob.merge(org, on=KEY, how="left"), ev


def _sel(d: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "all":
        return d
    d = d[d.cc_pass == 1]
    return d[d.origin_in_tank == 1] if which == "cc_in" else d[d.origin_in_tank == 0] if which == "cc_dirt" else d


def _inwin(t, lo, hi=HI_NS):
    return (t >= lo) & (t <= hi)


def objects_table(data: dict, methods: list, windows: list) -> pd.DataFrame:
    rows = []
    for smp, (cl, ob, ev) in data.items():
        for which, slab in SELS:
            o = _sel(ob, which)
            n_ev = len(_sel(ev, which))
            for lo in windows:
                for oc in ("n_capture", "michel", "muon"):
                    g = o[(o.obj_class == oc) & (o.n_hits >= 5) & _inwin(o.t_med, lo)]
                    for m in methods:
                        rows.append(dict(sample=smp, selection=which, window_lo_ns=lo, obj_class=oc, method=m,
                                         n_objects=len(g), per_event=len(g) / max(n_ev, 1),
                                         found=(g[f"n_owned_{m}"] >= 1).mean() if len(g) else np.nan,
                                         split=(g[f"n_owned_{m}"] >= 2).mean() if len(g) else np.nan,
                                         merged=(g[f"n_merged_into_{m}"] >= 1).mean() if len(g) else np.nan,
                                         completeness=g[f"best_hits_{m}"].sum() / max(g.n_hits.sum(), 1),
                                         small_found=(g[g.n_hits <= 7][f"n_owned_{m}"] >= 1).mean()
                                         if (g.n_hits <= 7).any() else np.nan))
    return pd.DataFrame(rows)


def clusters_table(data: dict, methods: list, windows: list) -> pd.DataFrame:
    rows = []
    for smp, (cl, ob, ev) in data.items():
        for which, slab in SELS:
            c = _sel(cl, which)
            n_ev = len(_sel(ev, which))
            for lo in windows:
                w = c[_inwin(c.t_med, lo)]
                for m in methods:
                    g0 = w[w.method == m]
                    for oc, g in g0.groupby("owner_class"):
                        rows.append(dict(sample=smp, selection=which, window_lo_ns=lo, method=m, owner_class=oc,
                                         n_clusters=len(g), per_event=len(g) / max(n_ev, 1), share=len(g) / len(g0),
                                         purity=g.purity_clean.mean(), one_neutron=(g.n_neutrons <= 1).mean(),
                                         median_hits=g.n_hits.median(), median_pe=g.pe_cluster.median(),
                                         median_cb=g.f_charge_bal_legacy.median(), median_t_us=g.t_med.median() / 1000,
                                         box=np.logical_and.reduce(_box(lo).cut_masks(
                                             g.f_pe_total.to_numpy(float), g.f_charge_bal_legacy.to_numpy(float),
                                             g.t_med.to_numpy(float), g.n_hits.to_numpy(int))).sum()))
    return pd.DataFrame(rows)


def multiplicity_table(data: dict, methods: list, windows: list) -> pd.DataFrame:
    rows = []
    for smp, which, slab in MULT:
        cl, ob, ev = data[smp]
        e0 = _sel(ev, which)[KEY].copy()
        c = _sel(cl, which)
        o = _sel(ob, which)
        for lo in windows:
            box = _box(lo)
            bc = box.cuts
            e = e0.copy()
            vis = o[(o.obj_class == "n_capture") & (o.n_hits >= 5) & _inwin(o.t_med, lo)]
            e = e.merge(vis.groupby(KEY).size().rename("n_visible").reset_index(), on=KEY, how="left")
            for m in methods:
                g = c[c.method == m]
                tm, nh = g.t_med.to_numpy(float), g.n_hits.to_numpy(int)
                sel = {"tagged": (g.tag == "n_capture").to_numpy() & _inwin(tm, lo),
                       "delayed": (tm >= bc.ct_min) & (tm <= bc.ct_max) & (nh >= bc.chits_min),
                       "box": np.logical_and.reduce(box.cut_masks(g.f_pe_total.to_numpy(float),
                                                                  g.f_charge_bal_legacy.to_numpy(float), tm, nh))}
                for k, s in sel.items():
                    e = e.merge(g[s].groupby(KEY).size().rename(f"n_{k}_{m}").reset_index(), on=KEY, how="left")
            e = e.fillna(0)
            for m in methods:
                rows.append(dict(sample=smp, selection=which, label=slab, window_lo_ns=lo, method=m, events=len(e),
                                 visible_per_event=e.n_visible.mean(), tagged_per_event=e[f"n_tagged_{m}"].mean(),
                                 delayed_per_event=e[f"n_delayed_{m}"].mean(), box_per_event=e[f"n_box_{m}"].mean(),
                                 tagged_eq_visible=(e[f"n_tagged_{m}"] == e.n_visible).mean(),
                                 delayed_eq_visible=(e[f"n_delayed_{m}"] == e.n_visible).mean(),
                                 box_eq_visible=(e[f"n_box_{m}"] == e.n_visible).mean(),
                                 tagged_minus_visible=e[f"n_tagged_{m}"].sub(e.n_visible).to_numpy().tolist()))
    return pd.DataFrame(rows)


def randoms_table(rctx: RunContext, tag: str, methods: list, windows: list, overlay_run="6256") -> pd.DataFrame:
    rows = []
    for run, _, tr, _, rc, _ in random_samples(rctx, overlay_run, tag):
        n = len(tr)
        for lo in windows:
            box = _box(lo)
            for m in methods:
                g = rc[rc.method == m]
                if g.empty:
                    raise SystemExit(f"[windowcompare] random_recoclusters_{run} has no {m} clusters")
                dl = (g.t_med >= lo) & (g.t_med <= box.cuts.ct_max) & (g.n_hits >= box.cuts.chits_min)
                b = np.logical_and.reduce(box.cut_masks(g.f_pe_total.to_numpy(float), g.f_charge_bal_legacy.to_numpy(float),
                                                        g.t_med.to_numpy(float), g.n_hits.to_numpy(int)))
                rows.append(dict(run=run, window_lo_ns=lo, method=m, triggers=n, delayed_per_trigger=dl.sum() / n,
                                 box_per_trigger=b.sum() / n))
    return pd.DataFrame(rows)


def overlay_table(octx: RunContext, tag: str, methods: list, windows: list) -> pd.DataFrame:
    cl, ob, ev = _load(octx, tag, methods)
    cl, ev = _sel(cl, "cc"), _sel(ev, "cc")
    rows = []
    for lo in windows:
        w = cl[_inwin(cl.t_med, lo)]
        for m in methods:
            g = w[w.method == m]
            cap = g[g.owner_class == "n_capture"]
            rows.append(dict(window_lo_ns=lo, method=m, events=len(ev), capture_purity=cap.owner_frac.mean(),
                             noise_clusters_per_event=(g.owner_class == "noise").sum() / max(len(ev), 1),
                             overlay_share_capture=cap.comp_overlay.sum() / max(cap.n_hits.sum(), 1)))
    return pd.DataFrame(rows)


def overlay_detail(octx: dict, tag: str, methods: list, windows: list) -> pd.DataFrame:
    """Per sample (tank, world CCinc): are the added hits inside capture clusters or in clusters of their own."""
    rows = []
    for smp, c in octx.items():
        cl, ob, ev = _load(c, tag, methods)
        cl, ev = _sel(cl, "cc"), _sel(ev, "cc")
        n_ev = max(len(ev), 1)
        for lo in windows:
            box = _box(lo)
            w = cl[_inwin(cl.t_med, lo)]
            for m in methods:
                g = w[w.method == m]
                cap, nz = g[g.owner_class == "n_capture"], g[g.owner_class == "noise"]
                nb = np.logical_and.reduce(box.cut_masks(nz.f_pe_total.to_numpy(float), nz.f_charge_bal_legacy.to_numpy(float),
                                                         nz.t_med.to_numpy(float), nz.n_hits.to_numpy(int))).sum()
                rows.append(dict(sample=smp, window_lo_ns=lo, method=m, events=len(ev), capture_clusters=len(cap),
                                 capture_purity=cap.owner_frac.mean(),
                                 capture_with_noise=(cap.comp_overlay >= 1).mean() if len(cap) else np.nan,
                                 noise_hits_per_capture=cap.comp_overlay.mean() if len(cap) else np.nan,
                                 overlay_share_capture=cap.comp_overlay.sum() / max(cap.n_hits.sum(), 1),
                                 capture_per_event=len(cap) / n_ev, noise_clusters_per_event=len(nz) / n_ev,
                                 noise_box_per_event=nb / n_ev,
                                 noise_median_hits=nz.n_hits.median() if len(nz) else np.nan,
                                 capture_median_hits=cap.n_hits.median() if len(cap) else np.nan))
    return pd.DataFrame(rows)


def tankworld_table(ct: pd.DataFrame, methods: list, windows: list) -> pd.DataFrame:
    rows = []
    for smp in ("tank", "world"):
        for lo in windows:
            for m in methods:
                d = ct[(ct["sample"] == smp) & (ct.selection == "cc") & (ct.method == m) & (ct.window_lo_ns == lo)]
                n = d.n_clusters.sum()
                r = dict(sample=smp, window_lo_ns=lo, method=m, clusters=int(n),
                         capture_share=d[d.owner_class == "n_capture"].n_clusters.sum() / max(n, 1),
                         noncapture_clusters=int(d[d.owner_class != "n_capture"].n_clusters.sum()))
                for oc in ("em", "michel", "muon", "n_other", "other"):
                    r[f"{oc}_per_event"] = d[d.owner_class == oc].per_event.sum()
                rows.append(r)
    return pd.DataFrame(rows)


NOISE_FEATS = (("n_hits", "hits", np.arange(5, 81, 3)), ("f_pe_total", "cluster PE", np.logspace(0, 3.3, 30)),
               ("f_charge_bal_legacy", "charge balance", np.linspace(0, 1, 26)),
               ("f_sigma_t", "hit-time spread σ_t [ns]", np.logspace(0, 3, 30)))


def noise_features(octx: RunContext, tag: str, rctx: RunContext, rtag: str, overlay_run, methods: list,
                   lo: float, out: Path) -> pd.DataFrame:
    """Noise-only clusters (the random sample clustered alone, lo-67 us, >= 5 hits) against the capture
    clusters of tank CCinc with the same random hits added."""
    cl, _, _ = _load(octx, tag, methods)
    cl = _sel(cl, "cc")
    rc = random_samples(rctx, overlay_run, rtag)[0][4]
    box = _box(lo)
    bc = box.cuts
    rows = []
    fig, axes = plt.subplots(len(methods), len(NOISE_FEATS), figsize=(4.0 * len(NOISE_FEATS), 3.2 * len(methods)))
    for r, m in enumerate(methods):
        cap = cl[(cl.method == m) & (cl.owner_class == "n_capture") & (cl.t_med >= lo) & (cl.t_med <= bc.ct_max)
                 & (cl.n_hits >= bc.chits_min)]
        nz = rc[(rc.method == m) & (rc.t_med >= lo) & (rc.t_med <= bc.ct_max) & (rc.n_hits >= bc.chits_min)]
        for lab, g, colr in (("neutron capture (MC + random hits)", cap, OWNER_COL["n_capture"]),
                             (f"noise only ({pool_label(overlay_run)} alone)", nz, OWNER_COL["noise"])):
            b = np.logical_and.reduce(box.cut_masks(g.f_pe_total.to_numpy(float), g.f_charge_bal_legacy.to_numpy(float),
                                                    g.t_med.to_numpy(float), g.n_hits.to_numpy(int)))
            rows.append(dict(method=m, kind="capture" if g is cap else "noise", clusters=len(g),
                             pass_box=b.mean() if len(g) else np.nan,
                             **{f"median_{f}": g[f].median() for f, _, _ in NOISE_FEATS},
                             **{f"q90_{f}": g[f].quantile(0.9) for f, _, _ in NOISE_FEATS}))
            for c, (f, xl, bins) in enumerate(NOISE_FEATS):
                ax = axes[r, c]
                x = g[f].dropna().to_numpy()
                h, e = np.histogram(x, bins)
                k = h > 0
                ax.plot((0.5 * (e[1:] + e[:-1]))[k], h[k] / max(len(x), 1), "o-", ms=2.5, lw=1.2, color=colr,
                        label=f"{lab} ({len(x):,})")
                if f in ("f_pe_total", "f_sigma_t"):
                    ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(xl)
                if c == 0:
                    ax.set_ylabel(f"{METHOD_LABELS[m]}\nfraction of clusters")
                    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Delayed clusters at {lo / 1000:g}–67 µs, ≥ 5 hits: noise only ({pool_label(overlay_run)}, cosmic veto) "
                 "vs neutron captures (tank CCinc)")
    _save(fig, out, "w05_noise_vs_capture")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- figures
def fig_time_owner(data, methods, out):
    cl = data["tank"][0]
    n_ev = len(data["tank"][2])
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.4), sharey=True)
    tb = np.arange(2.0, 70.01, 1.0)
    for ax, m in zip(np.atleast_1d(axes), methods):
        g = cl[(cl.method == m) & _inwin(cl.t_med, 2000.0)]
        for oc in [k for k in OWNER_LABEL if (g.owner_class == k).sum() >= 20]:
            h, _ = np.histogram(g[g.owner_class == oc].t_med / 1000, tb)
            k = h > 0
            ax.plot((0.5 * (tb[1:] + tb[:-1]))[k], h[k] / n_ev, "o-", ms=2.5, lw=1.2, color=OWNER_COL[oc],
                    label=OWNER_LABEL[oc])
        ax.set_yscale("log")
        ax.set_xlabel("cluster median time [µs]")
        ax.set_title(METHOD_LABELS[m])
    np.atleast_1d(axes)[0].set_ylabel("clusters per event per µs")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tank, all events, cleaned hits: delayed clusters by the true particle that owns them")
    _save(fig, out, "w01_cluster_time_owner")


def fig_background(ct, methods, windows, out, smp="tank", which="all"):
    d = ct[(ct["sample"] == smp) & (ct.selection == which)]
    owners = [k for k in OWNER_LABEL if k in set(d.owner_class)]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.4), sharey=True)
    y = np.arange(len(owners))
    for ax, m in zip(np.atleast_1d(axes), methods):
        for lo, mk, off in zip(windows, ("o", "D"), (-0.15, 0.15)):
            g = d[(d.method == m) & (d.window_lo_ns == lo)].set_index("owner_class").reindex(owners)
            v = g.per_event.to_numpy(float)
            k = np.isfinite(v) & (v > 0)
            ax.plot(v[k], y[k] + off, ls="none", marker=mk, ms=6, color=METHOD_COLORS[m] if lo == windows[-1] else "#9a9a9a",
                    label=_wlab(lo))
        ax.set_xscale("log")
        ax.set_yticks(y, [OWNER_LABEL[o] for o in owners])
        ax.invert_yaxis()
        ax.set_xlabel("clusters per event")
        ax.set_title(METHOD_LABELS[m])
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Tank, {dict(SELS)[which]}: clusters in the delayed window by owner, {' vs '.join(_wlab(w) for w in windows)}")
    _save(fig, out, f"w02_background_{smp}_{which}")


def fig_features(data, methods, lo, out):
    cl = data["tank"][0]
    feats = (("n_hits", "hits", np.arange(5, 81, 3)), ("pe_cluster", "cluster PE", np.logspace(0, 3.3, 30)),
             ("f_charge_bal_legacy", "charge balance", np.linspace(0, 1, 26)))
    fig, axes = plt.subplots(len(methods), 3, figsize=(13, 3.3 * len(methods)))
    for r, m in enumerate(methods):
        g = cl[(cl.method == m) & _inwin(cl.t_med, lo)]
        owners = [k for k in OWNER_LABEL if (g.owner_class == k).sum() >= 20]
        for c, (f, xl, bins) in enumerate(feats):
            ax = axes[r, c]
            for oc in owners:
                x = g[g.owner_class == oc][f].dropna().to_numpy()
                h, e = np.histogram(x, bins)
                k = h > 0
                ax.plot((0.5 * (e[1:] + e[:-1]))[k], h[k] / len(x), "o-", ms=2.5, lw=1.2, color=OWNER_COL[oc],
                        label=f"{OWNER_LABEL[oc]} ({len(x):,})")
            ax.set_yscale("log")
            if f == "pe_cluster":
                ax.set_xscale("log")
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(f"{METHOD_LABELS[m]}\nfraction of clusters")
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Tank, all events, cleaned hits: clusters at {_wlab(lo)} by owner")
    _save(fig, out, f"w03_features_{int(lo / 1000)}us")


def fig_multiplicity(mt, methods, windows, out):
    d = mt[mt.selection == "cc"]
    d = d[d["sample"] == "tank"]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.2), sharey=True)
    b = np.arange(-3, 4)
    for ax, m in zip(np.atleast_1d(axes), methods):
        for lo, mk in zip(windows, ("o", "D")):
            r = d[(d.method == m) & (d.window_lo_ns == lo)].iloc[0]
            x = np.clip(np.asarray(r.tagged_minus_visible), -3, 3)
            h = np.array([(x == k).mean() for k in b])
            k = h > 0
            ax.plot(b[k], h[k], marker=mk, ms=6, color=METHOD_COLORS[m] if lo == windows[-1] else "#9a9a9a",
                    label=f"{_wlab(lo)}: equal {r.tagged_eq_visible:.1%}")
        ax.set_yscale("log")
        ax.set_xticks(b, ["≤−3", "−2", "−1", "0", "1", "2", "≥3"])
        ax.set_xlabel("tagged capture clusters − visible true captures")
        ax.set_title(METHOD_LABELS[m])
        ax.legend(frameon=False, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("fraction of events")
    fig.suptitle("Tank CCinc: neutron multiplicity, clusters vs truth in the same window")
    _save(fig, out, "w04_multiplicity")


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------- tables
def _tab(path: Path, cols: str, header: list, rows: list, scale: str = "\\textwidth"):
    body = [" & ".join(header) + " \\\\", "\\midrule"]
    body += ["\\midrule" if r == "MID" else " & ".join(str(x) for x in r) + " \\\\" for r in rows]
    path.write_text(f"\\resizebox{{{scale}}}{{!}}{{\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n"
                    + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n")


def _f(v, fmt):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def tables_and_numbers(ot, ct, mt, rt, ovt, methods, windows, out, overlay_run="6256",
                       ovd=None, tw=None, nf=None) -> dict:
    N = {}
    ok = 6256 if str(overlay_run) == "6256" else str(overlay_run)
    olab = pool_label(overlay_run)
    lo2, lo10 = windows[0], windows[-1]
    q = lambda d, **k: d.loc[np.logical_and.reduce([d[a] == b for a, b in k.items()])]
    mk = {"clusterfinder": "Cf", "clusterfinder_nms": "Cfn", "peak": "Tp"}

    # summary: rows = metric, columns = method x window
    def obj(sm, sel, oc, m, lo, col):
        r = q(ot, sample=sm, selection=sel, obj_class=oc, method=m, window_lo_ns=lo)
        return r[col].iloc[0] if len(r) else np.nan

    def clu(sm, sel, oc, m, lo, col):
        r = q(ct, sample=sm, selection=sel, owner_class=oc, method=m, window_lo_ns=lo)
        return r[col].iloc[0] if len(r) else np.nan

    def bkg(sm, sel, m, lo, col="per_event"):
        r = q(ct, sample=sm, selection=sel, method=m, window_lo_ns=lo)
        r = r[r.owner_class != "n_capture"]
        return r[col].sum()

    def capshare(sm, sel, m, lo):
        r = q(ct, sample=sm, selection=sel, method=m, window_lo_ns=lo)
        return r[r.owner_class == "n_capture"].n_clusters.sum() / max(r.n_clusters.sum(), 1)

    def mul(m, lo, col, sm="tank", sel="cc"):
        r = q(mt, sample=sm, selection=sel, method=m, window_lo_ns=lo)
        return r[col].iloc[0]

    def rnd(m, lo, col, run=ok):
        r = q(rt, run=run, method=m, window_lo_ns=lo)
        return r[col].iloc[0]

    def ovl(m, lo, col):
        r = q(ovt, method=m, window_lo_ns=lo)
        return r[col].iloc[0] if len(r) else np.nan

    metrics = [
        ("visible captures ($\\geq$\\,5 hits), all events", lambda m, lo: obj("tank", "all", "n_capture", m, lo, "n_objects"), "{:,.0f}"),
        ("captures found", lambda m, lo: 100 * obj("tank", "all", "n_capture", m, lo, "found"), "{:.1f}\\,\\%"),
        ("5--7-hit captures found", lambda m, lo: 100 * obj("tank", "all", "n_capture", m, lo, "small_found"), "{:.1f}\\,\\%"),
        ("capture purity", lambda m, lo: clu("tank", "all", "n_capture", m, lo, "purity"), "{:.3f}"),
        ("one neutron per capture cluster", lambda m, lo: 100 * clu("tank", "all", "n_capture", m, lo, "one_neutron"), "{:.1f}\\,\\%"),
        ("capture-owned share of clusters in window", lambda m, lo: 100 * capshare("tank", "all", m, lo), "{:.1f}\\,\\%"),
        ("non-capture clusters per event", lambda m, lo: bkg("tank", "all", m, lo), "{:.3f}"),
        ("Michel e$^\\pm$ ($\\geq$\\,5 hits) per event", lambda m, lo: obj("tank", "all", "michel", m, lo, "per_event"), "{:.4f}"),
        ("Michel split", lambda m, lo: 100 * obj("tank", "all", "michel", m, lo, "split"), "{:.1f}\\,\\%"),
        ("CCinc: tagged = visible captures", lambda m, lo: 100 * mul(m, lo, "tagged_eq_visible"), "{:.1f}\\,\\%"),
        ("CCinc: box clusters per event", lambda m, lo: mul(m, lo, "box_per_event"), "{:.3f}"),
        (f"{olab}: delayed clusters per trigger", lambda m, lo: rnd(m, lo, "delayed_per_trigger"), "{:.3f}"),
        (f"{olab}: box clusters per trigger", lambda m, lo: rnd(m, lo, "box_per_trigger"), "{:.3f}"),
        (f"CCinc + {olab}: capture purity", lambda m, lo: ovl(m, lo, "capture_purity"), "{:.3f}"),
    ]
    head = ["tank (cleaned hits)"] + [f"{METHOD_TEX[m]} {_wtex(lo)}" for m in methods for lo in windows]
    rows = [[lab] + [_f(fn(m, lo), fmt) for m in methods for lo in windows] for lab, fn, fmt in metrics]
    _tab(out / "tab_wc_summary.tex", "l" + "rr" * len(methods), head, rows)
    pd.DataFrame([[lab] + [fn(m, lo) for m in methods for lo in windows] for lab, fn, _ in metrics],
                 columns=["metric"] + [f"{m}_{lo:g}" for m in methods for lo in windows]
                 ).to_csv(out / "wc_summary.csv", index=False)

    # background composition per window, tank all events and tank CCinc
    for sel in ("all", "cc"):
        d = ct[(ct["sample"] == "tank") & (ct.selection == sel)]
        owners = [o for o in OWNER_LABEL if o in set(d.owner_class)]
        rows = []
        for o in owners:
            r = [OWNER_TEX[o]]
            for m in methods:
                for lo in windows:
                    v = q(d, owner_class=o, method=m, window_lo_ns=lo)
                    r.append(_f(v.per_event.iloc[0] if len(v) else 0.0, "{:.4f}"))
            rows.append(r)
        _tab(out / f"tab_wc_owners_{sel}.tex", "l" + "rr" * len(methods),
             ["clusters per event, owner"] + [f"{METHOD_TEX[m]} {_wtex(lo)}" for m in methods for lo in windows], rows)

    # what the non-capture clusters at 10-70 look like
    d = ct[(ct["sample"] == "tank") & (ct.selection == "all") & (ct.window_lo_ns == lo10)]
    rows = []
    for m in methods:
        g = d[d.method == m].sort_values("n_clusters", ascending=False)
        for k, r in enumerate(g.itertuples()):
            rows.append([METHOD_TEX[m] if k == 0 else "", OWNER_TEX[r.owner_class], f"{r.n_clusters:,}",
                         _f(100 * r.share, "{:.2f}\\,\\%"), _f(r.median_hits, "{:.0f}"), _f(r.median_pe, "{:.0f}"),
                         _f(r.median_cb, "{:.2f}"), _f(r.median_t_us, "{:.1f}"), _f(100 * r.box / r.n_clusters, "{:.0f}\\,\\%")])
        rows.append("MID")
    _tab(out / "tab_wc_bkg_features.tex", "ll rr rrrr r",
         ["", "owner", "clusters", "share", "median hits", "median PE", "median CB", "median $t$ [$\\mu$s]",
          "pass box"], rows[:-1])

    # multiplicity, three samples
    rows = []
    for smp, sel, lab in MULT:
        for k, m in enumerate(methods):
            r = [lab if k == 0 else "", METHOD_TEX[m]]
            for lo in windows:
                x = q(mt, sample=smp, selection=sel, method=m, window_lo_ns=lo).iloc[0]
                r += [_f(x.visible_per_event, "{:.3f}"), _f(x.tagged_per_event, "{:.3f}"),
                      _f(100 * x.tagged_eq_visible, "{:.1f}")]
            rows.append(r)
        rows.append("MID")
    _tab(out / "tab_wc_multiplicity.tex", "ll" + "rrr" * len(windows),
         ["sample", ""] + [h for lo in windows for h in (f"visible/ev {_wtex(lo)}", "tagged/ev", "equal [\\%]")], rows[:-1])

    rows = []
    for run in dict.fromkeys(rt.run):
        for k, m in enumerate(methods):
            r = [(str(run) if str(overlay_run) == "6256" else pool_label(run)) if k == 0 else "", METHOD_TEX[m]]
            for lo in windows:
                r += [_f(rnd(m, lo, "delayed_per_trigger", run), "{:.3f}"), _f(rnd(m, lo, "box_per_trigger", run), "{:.3f}")]
            r.append(_f(100 * (1 - rnd(m, lo10, "delayed_per_trigger", run) / rnd(m, lo2, "delayed_per_trigger", run)), "{:.0f}\\,\\%"))
            rows.append(r)
        rows.append("MID")
    _tab(out / "tab_wc_randoms.tex", "ll" + "rr" * len(windows) + "r",
         ["run", ""] + [h for lo in windows for h in (f"delayed/trigger {_wtex(lo)}", "box/trigger")]
         + ["delayed removed"], rows[:-1], "0.85\\textwidth")

    for m in methods:
        k = mk[m]
        N[f"wcNoiseDrop{k}"] = _f(100 * (1 - rnd(m, lo10, "delayed_per_trigger") / rnd(m, lo2, "delayed_per_trigger")), "{:.0f}")
        N[f"wcNoiseRatio{k}"] = _f(rnd(m, lo10, "delayed_per_trigger") / max(bkg("tank", "all", m, lo10), 1e-12), "{:.0f}")
        for lo, w in ((lo2, "Two"), (lo10, "Ten")):
            N[f"wcFound{k}{w}"] = _f(100 * obj("tank", "all", "n_capture", m, lo, "found"), "{:.1f}")
            N[f"wcCapShare{k}{w}"] = _f(100 * capshare("tank", "all", m, lo), "{:.1f}")
            N[f"wcBkg{k}{w}"] = _f(bkg("tank", "all", m, lo), "{:.3f}")
            N[f"wcBkgCc{k}{w}"] = _f(bkg("tank", "cc", m, lo), "{:.3f}")
            N[f"wcEq{k}{w}"] = _f(100 * mul(m, lo, "tagged_eq_visible"), "{:.1f}")
            N[f"wcRnd{k}{w}"] = _f(rnd(m, lo, "box_per_trigger"), "{:.3f}")
            N[f"wcRndDel{k}{w}"] = _f(rnd(m, lo, "delayed_per_trigger"), "{:.3f}")
            N[f"wcMichel{k}{w}"] = _f(clu("tank", "all", "michel", m, lo, "per_event"), "{:.4f}")
            N[f"wcPur{k}{w}"] = _f(clu("tank", "all", "n_capture", m, lo, "purity"), "{:.3f}")
            N[f"wcOvPur{k}{w}"] = _f(ovl(m, lo, "capture_purity"), "{:.3f}")
        N[f"wcBkgDrop{k}"] = _f(100 * (1 - bkg("tank", "all", m, lo10) / max(bkg("tank", "all", m, lo2), 1e-12)), "{:.0f}")
        N[f"wcMichelDrop{k}"] = _f(100 * (1 - clu("tank", "all", "michel", m, lo10, "n_clusters")
                                          / clu("tank", "all", "michel", m, lo2, "n_clusters")), "{:.1f}")
        N[f"wcCapDrop{k}"] = _f(100 * (1 - clu("tank", "all", "n_capture", m, lo10, "n_clusters")
                                       / clu("tank", "all", "n_capture", m, lo2, "n_clusters")), "{:.1f}")
    c2 = obj("tank", "all", "n_capture", methods[0], lo2, "n_objects")
    c10 = obj("tank", "all", "n_capture", methods[0], lo10, "n_objects")
    N["wcVisTwo"], N["wcVisTen"] = f"{c2:,.0f}", f"{c10:,.0f}"
    N["wcVisLost"] = _f(100 * (1 - c10 / c2), "{:.1f}")
    m2 = obj("tank", "all", "michel", methods[0], lo2, "n_objects")
    m10 = obj("tank", "all", "michel", methods[0], lo10, "n_objects")
    N["wcMichelObjTwo"], N["wcMichelObjTen"] = f"{m2:,.0f}", f"{m10:,.0f}"
    N["wcMichelObjLost"] = _f(100 * (1 - m10 / m2), "{:.1f}")
    N["wcMichelSplitCfTwo"] = _f(100 * obj("tank", "all", "michel", "clusterfinder", lo2, "split"), "{:.1f}")
    for lo, w in ((lo2, "Two"), (lo10, "Ten")):
        N[f"wcCcVis{w}"] = _f(mul(methods[0], lo, "visible_per_event"), "{:.2f}")
        d = q(ct, sample="tank", selection="all", method="clusterfinder_nms", window_lo_ns=lo)
        for oc, nm in (("michel", "Michel"), ("muon", "Muon")):
            N[f"wcShare{nm}Cfn{w}"] = _f(100 * d[d.owner_class == oc].share.sum(), "{:.2f}")
    N["wcCcEvents"] = f"{int(q(mt, sample='tank', selection='cc', method=methods[0], window_lo_ns=lo2).events.iloc[0]):,}"
    if tw is not None:
        rows = []
        for smp, lab in (("tank", "tank CCinc"), ("world", "world CCinc")):
            for k, m in enumerate(methods):
                g = lambda lo: q(tw, sample=smp, method=m, window_lo_ns=lo).iloc[0]
                a, b = g(lo2), g(lo10)
                rows.append([lab if k == 0 else "", METHOD_TEX[m], _f(100 * a.capture_share, "{:.1f}\\,\\%"),
                             _f(100 * b.capture_share, "{:.1f}\\,\\%"), f"{b.noncapture_clusters:,} / {b.clusters:,}",
                             _f(b.em_per_event, "{:.4f}"), _f(b.michel_per_event, "{:.4f}"), _f(b.muon_per_event, "{:.4f}"),
                             _f(b.n_other_per_event + b.other_per_event, "{:.4f}")])
            rows.append("MID")
        _tab(out / "tab_wc_tankworld.tex", "ll rr r rrrr",
             ["sample", "", f"capture share {_wtex(lo2)}", f"capture share {_wtex(lo10)}",
              "non-capture / all", "e/$\\gamma$ per ev", "Michel per ev", "muon per ev", "other per ev"], rows[:-1])
        for smp, sk in (("tank", "Tank"), ("world", "World")):
            for m in methods:
                N[f"wcTw{sk}Share{mk[m]}"] = _f(100 * q(tw, sample=smp, method=m, window_lo_ns=lo10).capture_share.iloc[0], "{:.1f}")
            N[f"wcTw{sk}Events"] = f"{int(q(mt, sample=smp, selection='cc', method=methods[0], window_lo_ns=lo2).events.iloc[0]):,}"
    if ovd is not None:
        rows = []
        for smp, lab in (("tank", "tank CCinc"), ("world", "world CCinc")):
            for k, m in enumerate(methods):
                x = q(ovd, sample=smp, method=m, window_lo_ns=lo10).iloc[0]
                own = m not in ("clusterfinder", "clusterfinder_nms")
                rows.append([lab if k == 0 else "", METHOD_TEX[m], f"{x.capture_clusters:,}",
                             _f(100 * x.capture_with_noise, "{:.1f}\\,\\%"), _f(x.noise_hits_per_capture, "{:.3f}"),
                             _f(x.capture_purity, "{:.3f}"),
                             _f(x.noise_clusters_per_event if own else np.nan, "{:.3f}"),
                             _f(x.noise_box_per_event if own else np.nan, "{:.3f}"),
                             _f(rnd(m, lo10, "delayed_per_trigger"), "{:.3f}"), _f(rnd(m, lo10, "box_per_trigger"), "{:.3f}")])
            rows.append("MID")
        _tab(out / "tab_wc_noise_inside.tex", "ll r rrr rr rr",
             ["sample", "", "capture clusters", "with $\\geq$\\,1 noise hit", "noise hits / cluster", "purity",
              "noise-only / ev", "noise-only box / ev", f"{olab} alone: delayed / trig", "box / trig"], rows[:-1])
        for smp, sk in (("tank", "Tank"), ("world", "World")):
            for m in methods:
                x = q(ovd, sample=smp, method=m, window_lo_ns=lo10).iloc[0]
                N[f"wcOv{sk}WithNoise{mk[m]}"] = _f(100 * x.capture_with_noise, "{:.1f}")
                N[f"wcOv{sk}NoiseOnly{mk[m]}"] = _f(x.noise_clusters_per_event, "{:.2f}")
                N[f"wcOv{sk}Cap{mk[m]}"] = _f(x.capture_per_event, "{:.2f}")
                N[f"wcOv{sk}NoiseMedHits{mk[m]}"] = _f(x.noise_median_hits, "{:.0f}")
                N[f"wcOv{sk}CapMedHits{mk[m]}"] = _f(x.capture_median_hits, "{:.0f}")
    if nf is not None:
        rows = []
        for k0, m in enumerate(methods):
            for kind, lab in (("capture", "capture (MC)"), ("noise", "noise only")):
                x = q(nf, method=m, kind=kind).iloc[0]
                rows.append([METHOD_TEX[m] if kind == "capture" else "", lab, f"{x.clusters:,}",
                             _f(x.median_n_hits, "{:.0f}"), _f(x.median_f_pe_total, "{:.0f}"),
                             _f(x.median_f_charge_bal_legacy, "{:.2f}"), _f(x.median_f_sigma_t, "{:.0f}"),
                             _f(100 * x.pass_box, "{:.0f}\\,\\%")])
            rows.append("MID")
        _tab(out / "tab_wc_noise_features.tex", "ll r rrrr r",
             ["", "", "clusters", "median hits", "median PE", "median CB", "median $\\sigma_t$ [ns]", "pass box"],
             rows[:-1], "0.85\\textwidth")
        for m in methods:
            N[f"wcNfBoxNoise{mk[m]}"] = _f(100 * q(nf, method=m, kind="noise").pass_box.iloc[0], "{:.0f}")
            N[f"wcNfBoxCap{mk[m]}"] = _f(100 * q(nf, method=m, kind="capture").pass_box.iloc[0], "{:.0f}")
    N["wcOverlayRun"] = olab
    with open(out / "wc_numbers.tex", "w") as fh:
        fh.write("% generated by ambe.plots.window_compare\n")
        for k, v in N.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return N


def run(ctx: RunContext, a) -> Path:
    set_style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = a.plot_methods.split(",")
    windows = [float(x) for x in a.windows.split(",")]
    world = load_context(a.world_config)
    data = {"tank": _load(ctx, a.tag, methods), "world": _load(world, a.tag, methods)}
    rctx = load_context(a.randoms_config)
    ot = objects_table(data, methods, windows)
    ct = clusters_table(data, methods, windows)
    mt = multiplicity_table(data, methods, windows)
    rt = randoms_table(rctx, a.randoms_tag, methods, windows, a.overlay_run)
    ovt = overlay_table(ctx, a.overlay_tag, methods, windows)
    ovd = overlay_detail({"tank": ctx, "world": world}, a.overlay_tag, methods, windows)
    tw = tankworld_table(ct, methods, windows)
    nf = noise_features(ctx, a.overlay_tag, rctx, a.randoms_tag, a.overlay_run, methods, windows[-1], out)
    for name, t in (("objects", ot), ("clusters", ct), ("multiplicity", mt.drop(columns="tagged_minus_visible")),
                    ("randoms", rt), ("overlay", ovt), ("overlay_detail", ovd), ("tankworld", tw),
                    ("noise_features", nf)):
        t.to_csv(out / f"wc_{name}.csv", index=False)
    fig_time_owner(data, methods, out)
    for sel in ("all", "cc"):
        fig_background(ct, methods, windows, out, "tank", sel)
    for lo in windows:
        fig_features(data, methods, lo, out)
    fig_multiplicity(mt, methods, windows, out)
    N = tables_and_numbers(ot, ct, mt, rt, ovt, methods, windows, out, a.overlay_run, ovd, tw, nf)
    pd.set_option("display.width", 220)
    print(pd.read_csv(out / "wc_summary.csv").round(4).to_string(index=False))
    print(f"[windowcompare] {len(N)} numbers, figures and tables in {out}")
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots windowcompare")
    p.add_argument("--world-config", required=True)
    p.add_argument("--tag", required=True, help="cleaned-hit `mc capture` tag, e.g. nuclean_all")
    p.add_argument("--overlay-tag", required=True, help="tag with random hits added, e.g. nuover_all")
    p.add_argument("--overlay-run", required=True,
                   help="random sample of the --overlay-tag run: 6256, or a --pool name such as 6264p6256")
    p.add_argument("--randoms-config", required=True)
    p.add_argument("--randoms-tag", required=True, help="random_recoclusters tag, e.g. nufull")
    p.add_argument("--windows", required=True, help="delayed-window starts in ns, e.g. 2000,10000")
    p.add_argument("--plot-methods", required=True, help="e.g. clusterfinder,clusterfinder_nms,peak")
    p.add_argument("--out", required=True)
    run(ctx, p.parse_args(list(argv) if argv else []))
