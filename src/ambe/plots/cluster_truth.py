"""
Truth vs clustering for MC neutrino events (output of `ambe mc capture --truth-unit particle`).

    ambe plots clustertruth --config configs/cc_neutrino_v3fixnu_truthtag.yaml \
        --world-config configs/cc_neutrino_v3fixnuworld_truthtag.yaml --tag nuclean \
        --plot-methods clusterfinder,clusterfinder_nms,optics_nms,peak \
        [--overlay-tag nuover --randoms-config configs/data_dummy_randoms_625x.yaml --overlay-run 6256 [--randoms-tag nufull]] \
        --out PRESENTATION_PLOTS/ccinc_cluster_truth

One multi-page PDF per topic, one plot per page:
  cluster_scores       purity, recall and F1 of every cluster, by the particle that owns it
  capture_finding      are the true captures (with MCHits) found, and how pure are their clusters
  multiplicity         neutrons per event in 2-70 us: truth vs clusters vs box cuts
  hits_outside         share of each particle's hits left outside every cluster
  hits_per_cluster     number of hits in clusters, by owner
  overlay_multiplicity the multiplicity pages again with random-trigger hits (--overlay-run) added
Samples: tank CCinc, world CCinc with the vertex in the tank, world CCinc with the vertex
outside the tank (their neutrons are background). CCinc = the config's truth cut stream.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from ..data.randoms import pool_label
from ..context import RunContext, load_context
from ..data.selection import BoxSelection
from ..plotting import METHOD_COLORS, set_style

METHOD_LABELS = {"clusterfinder": "ClusterFinder", "clusterfinder_nms": "ClusterFinder + NMS",
                 "optics": "OPTICS", "optics_nms": "OPTICS + NMS", "peak": "Time-peak"}
OWNERS = {"n_capture": "Neutron capture", "michel": "Michel e±", "muon": "Muon",
          "pion_charged": "π±", "pi0": "π⁰", "proton": "Proton", "em": "EM (e/γ)",
          "n_other": "Neutron, no capture"}
SAMPLES = (("tank", "cc", "Tank CCinc"), ("world", "cc_in", "World CCinc, vertex in tank"),
           ("world", "cc_dirt", "World CCinc, vertex outside tank"))
DELAYED_NS = (2000.0, 70000.0)


def _events(ctx: RunContext, tag: str) -> pd.DataFrame:
    return pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__truth_event_multiplicity__{tag}_plurality"))


def _tables(ctx: RunContext, tag: str, methods: dict):
    cl = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__truth_object_clusters__{tag}_plurality"))
    ob = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__truth_objects__{tag}_plurality"))
    ev = _events(ctx, tag)
    key = ["_source_file", "eventNumber"]
    org = ev[key + ["origin_in_tank"]]
    cl = cl[cl.method.isin(methods)].merge(org, on=key, how="left")
    ob = ob.merge(org, on=key, how="left")
    return cl, ob, ev


def _select(d: pd.DataFrame, which: str) -> pd.DataFrame:
    d = d[d.cc_pass == 1]
    if which == "cc_in":
        return d[d.origin_in_tank == 1]
    if which == "cc_dirt":
        return d[d.origin_in_tank == 0]
    return d


def _page(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# 2. purity, recall, F1 per cluster -------------------------------------------------
def cluster_scores(data: dict, methods: dict, pdf) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0, 1, 21)
    for smp, which, slab in SAMPLES:
        cl = _select(data[smp][0], which)
        cl = cl.assign(recall=cl.owner_completeness,
                       f1=2 * cl.owner_frac * cl.owner_completeness / (cl.owner_frac + cl.owner_completeness))
        for owner in ("n_capture", "michel", "muon"):
            g0 = cl[cl.owner_class == owner]
            if len(g0) < 10:
                continue
            for col, lab in (("owner_frac", "Purity (owner hits / cluster hits)"),
                             ("recall", "Recall (owner hits in this cluster / all owner hits)"),
                             ("f1", "F1 = 2 P R / (P + R)")):
                fig = plt.figure(figsize=(10, 6))
                for m, mlab in methods.items():
                    g = g0[g0.method == m]
                    plt.hist(g[col].clip(0, 1), bins=bins, histtype="step", lw=2, color=METHOD_COLORS[m],
                             label=f"{mlab}: {len(g)} clusters, mean {g[col].mean():.3f}")
                plt.xlabel(lab)
                plt.ylabel("Clusters")
                plt.yscale("log")
                plt.legend(loc="upper left", frameon=False)
                plt.title(f"{slab}\n{lab.split(' (')[0]} of clusters owned by: {OWNERS[owner]}")
                _page(pdf, fig)
            for m, mlab in methods.items():
                g = g0[g0.method == m]
                rows.append(dict(sample=slab, owner=OWNERS[owner], clusterer=mlab, clusters=len(g),
                                 purity=g.owner_frac.mean(), purity_pe=g.owner_pe_frac.mean(),
                                 recall=g.recall.mean(), f1=g.f1.mean(),
                                 pure_100=(g.owner_frac >= 0.9999).mean(),
                                 other_particle_hits=(g.n_hits - g.n_owner_hits - g.comp_darknoise - g.comp_untraced).sum() / g.n_hits.sum(),
                                 darknoise_hits=g.comp_darknoise.sum() / g.n_hits.sum()))
    return pd.DataFrame(rows)


# 3. are the true captures found, and how pure are their clusters ---------------------
def capture_finding(data: dict, methods: dict, pdf) -> pd.DataFrame:
    rows = []
    bins = np.arange(0, 62, 2)
    for smp, which, slab in SAMPLES:
        ob = _select(data[smp][1], which)
        caps = ob[(ob.obj_class == "n_capture") & (ob.cap_j >= 0)]
        if len(caps) < 10:
            continue
        fig = plt.figure(figsize=(10, 6))
        plt.hist(caps.n_hits.clip(upper=60), bins=bins, color="lightblue", edgecolor="blue",
                 label=f"True captures with MCHits: {len(caps)}")
        for m, mlab in methods.items():
            f = caps[caps[f"n_owned_{m}"] >= 1]
            plt.hist(f.n_hits.clip(upper=60), bins=bins, histtype="step", lw=2, color=METHOD_COLORS[m],
                     label=f"found by {mlab}: {len(f) / len(caps):.1%}")
        plt.xlabel("MCHits from the capture (last bin = 60 or more)")
        plt.ylabel("True captures")
        plt.legend(frameon=False)
        plt.title(f"{slab}\nTrue neutron captures in 2–70 µs and the ones each clustering finds")
        _page(pdf, fig)
        for m, mlab in methods.items():
            for lo, hi in ((1, 4), (5, 7), (8, 10), (11, 20), (21, 10**6)):
                g = caps[(caps.n_hits >= lo) & (caps.n_hits <= hi)]
                rows.append(dict(sample=slab, clusterer=mlab, capture_hits=f"{lo}-{hi}" if hi < 10**6 else f"{lo}+",
                                 captures=len(g), found=(g[f"n_owned_{m}"] >= 1).mean() if len(g) else np.nan,
                                 split=(g[f"n_owned_{m}"] >= 2).mean() if len(g) else np.nan,
                                 merged=(g[f"n_merged_into_{m}"] >= 1).mean() if len(g) else np.nan))
    return pd.DataFrame(rows)


# 4 and 7. multiplicity -----------------------------------------------------------------
def _cf_noise(ev: pd.DataFrame, rctx: RunContext, seed: int, methods: dict, randoms_tag: str,
              run="6256") -> pd.DataFrame:
    """ClusterFinder clusters of the random trigger each MC event was given (same draw as
    capture_truth.add_overlay), counted as delayed / box clusters. ClusterFinder + NMS takes
    the trigger's re-clustered clusterfinder_nms clusters (`ambe data randoms --cluster`)."""
    trig = np.sort(pd.read_parquet(rctx.parquet_path(f"{rctx.run_name}__random_hits_{run}"),
                                   columns=["trigger"]).trigger.unique())
    pick = [trig[np.random.default_rng([seed, zlib.crc32(str(s).encode()), int(n)]).integers(len(trig))]
            for s, n in zip(ev._source_file, ev.eventNumber)]
    box = BoxSelection()
    c = box.cuts
    cf = pd.read_parquet(rctx.parquet_path(f"{rctx.run_name}__random_cfclusters_{run}"))
    noise = {"clusterfinder": cf.assign(
        delayed=(cf.clusterTime >= c.ct_min) & (cf.clusterTime <= c.ct_max) & (cf.clusterHits >= c.chits_min),
        box=np.logical_and.reduce(box.cut_masks(cf.clusterPE, cf.clusterCB, cf.clusterTime, cf.clusterHits)))}
    if "clusterfinder_nms" in methods:
        name = f"{rctx.run_name}__random_recoclusters_{run}" + (f"__{randoms_tag}" if randoms_tag else "")
        rc = pd.read_parquet(rctx.parquet_path(name))
        rc = rc[rc.method == "clusterfinder_nms"]
        if rc.empty:
            raise SystemExit(f"[clustertruth] {name} has no clusterfinder_nms clusters")
        noise["clusterfinder_nms"] = rc.assign(
            delayed=(rc.t_med >= c.ct_min) & (rc.t_med <= c.ct_max) & (rc.n_hits >= c.chits_min),
            box=np.logical_and.reduce(box.cut_masks(rc.f_pe_total, rc.f_charge_bal_legacy, rc.t_med, rc.n_hits)))
    out = {}
    for m, d in noise.items():
        got = d.groupby("trigger")[["delayed", "box"]].sum().reindex(pick).fillna(0).to_numpy(int)
        out[f"n_reco_delayed_{m}"] = ev[f"n_reco_delayed_{m}"] + got[:, 0]
        out[f"n_reco_box_{m}"] = ev[f"n_reco_box_{m}"] + got[:, 1]
    return ev.assign(**out)


def multiplicity(evs: dict, methods: dict, pdf, note: str = "") -> pd.DataFrame:
    rows = []
    bins = range(0, 9)
    truth = (("n_true_captures", "True captures (all, any position)", "black", "-"),
             ("n_visible_captures_1", "True captures with ≥ 1 MCHit", "black", "--"),
             ("n_visible_captures_5", "True captures with ≥ 5 MCHits", "gray", ":"))
    for smp, which, slab in SAMPLES:
        e = _select(evs[smp], which)
        for m, mlab in methods.items():
            fig = plt.figure(figsize=(10, 6))
            for col, lab, colr, ls in truth:
                plt.hist(e[col].clip(upper=7), bins=bins, histtype="step", lw=2, color=colr, ls=ls, align="left",
                         label=f"{lab}: mean {e[col].mean():.2f}")
            for col, reco, lab, colr in ((f"n_reco_delayed_{m}", "clusters (≥ 5 hits)",
                                          f"all {mlab} clusters (≥ 5 hits, 2–67 µs)", "blue"),
                                         (f"n_reco_box_{m}", "clusters passing box cuts",
                                          f"{mlab} clusters passing box cuts", "red")):
                plt.hist(e[col].clip(upper=7), bins=bins, histtype="step", lw=2, color=colr, align="left",
                         label=f"{lab}: mean {e[col].mean():.2f}")
                rows.append(dict(sample=slab, clusterer=mlab, reco=reco, events=len(e),
                                 true_all=e.n_true_captures.mean(), true_1hit=e.n_visible_captures_1.mean(),
                                 true_5hits=e.n_visible_captures_5.mean(), reco_mean=e[col].mean()))
            plt.xticks(range(0, 8), [str(k) for k in range(7)] + ["7+"])
            plt.xlabel("per event: true captures in 2–70 µs, clusters in 2–67 µs")
            plt.ylabel("Counts")
            plt.legend(frameon=False)
            plt.title(f"{slab}, {len(e)} events{note}\nNeutron multiplicity: truth vs {mlab}")
            _page(pdf, fig)
    return pd.DataFrame(rows)


# 5. hits left outside every cluster ----------------------------------------------------
def hits_outside(data: dict, methods: dict, pdf) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0, 1, 21)
    for smp, which, slab in SAMPLES:
        ob = _select(data[smp][1], which)
        ob = ob[ob.n_hits >= 5]
        for owner in ("n_capture", "michel", "muon", "pion_charged", "pi0", "proton", "em", "n_other"):
            g0 = ob[ob.obj_class == owner]
            if len(g0) < 10:
                continue
            if owner in ("n_capture", "michel", "muon"):
                fig = plt.figure(figsize=(10, 6))
            for m, mlab in methods.items():
                out = 1 - g0[f"hits_clustered_{m}"] / g0.n_hits
                own = g0[f"hits_own_{m}"] / g0.n_hits
                if owner in ("n_capture", "michel", "muon"):
                    plt.hist(out, bins=bins, histtype="step", lw=2, color=METHOD_COLORS[m],
                             label=f"{mlab}: {out.mean():.1%} of hits outside clusters "
                                   f"({(g0.n_hits - g0[f'hits_clustered_{m}']).sum() / g0.n_hits.sum():.1%} hit-weighted)")
                rows.append(dict(sample=slab, particle=OWNERS[owner], clusterer=mlab, particles=len(g0),
                                 hits=int(g0.n_hits.sum()),
                                 in_own_cluster=g0[f"hits_own_{m}"].sum() / g0.n_hits.sum(),
                                 in_other_cluster=(g0[f"hits_clustered_{m}"] - g0[f"hits_own_{m}"]).sum() / g0.n_hits.sum(),
                                 outside_clusters=1 - g0[f"hits_clustered_{m}"].sum() / g0.n_hits.sum(),
                                 pe_outside=1 - g0[f"pe_clustered_{m}"].sum() / g0.pe.sum(),
                                 particles_over_half_outside=(out > 0.5).mean()))
            if owner in ("n_capture", "michel", "muon"):
                plt.xlabel("Share of the particle's hits outside every cluster")
                plt.ylabel("Particles (≥ 5 hits)")
                plt.yscale("log")
                plt.legend(frameon=False)
                plt.title(f"{slab}\n{OWNERS[owner]}: hits the clustering leaves out ({len(g0)} particles)")
                _page(pdf, fig)
    return pd.DataFrame(rows)


# 6. hits per cluster by owner -------------------------------------------------------------
def hits_per_cluster(data: dict, methods: dict, pdf) -> pd.DataFrame:
    rows = []
    bins = np.arange(0, 102, 2)
    for smp, which, slab in SAMPLES:
        cl = _select(data[smp][0], which)
        for m, mlab in methods.items():
            g0 = cl[cl.method == m]
            fig = plt.figure(figsize=(10, 6))
            for owner, colr in (("n_capture", "blue"), ("michel", "green"), ("muon", "red"), ("pi0", "purple"),
                                ("pion_charged", "orange"), ("em", "magenta")):
                g = g0[g0.owner_class == owner]
                if len(g) < 5:
                    continue
                plt.hist(g.n_hits.clip(upper=100), bins=bins, histtype="step", lw=2, color=colr,
                         label=f"{OWNERS[owner]}: {len(g)} clusters, mean {g.n_hits.mean():.1f}, median {g.n_hits.median():.0f} hits")
            plt.xlabel("Hits in the cluster (last bin = 100 or more)")
            plt.ylabel("Clusters")
            plt.yscale("log")
            plt.legend(frameon=False)
            plt.title(f"{slab}\n{mlab}: hits per cluster by the particle that owns it")
            _page(pdf, fig)
            for owner, g in g0.groupby("owner_class"):
                rows.append(dict(sample=slab, clusterer=mlab, owner=OWNERS.get(owner, owner), clusters=len(g),
                                 per_event=len(g) / max(cl.eventID.nunique(), 1), mean_hits=g.n_hits.mean(),
                                 median_hits=g.n_hits.median(), mean_pe=g.pe_cluster.mean() if "pe_cluster" in g else np.nan,
                                 mean_owner_hits=g.n_owner_hits.mean()))
    return pd.DataFrame(rows)


def run(ctx: RunContext, args) -> Path:
    set_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = load_context(args.world_config)
    methods = {m: METHOD_LABELS[m] for m in args.plot_methods.split(",")}
    data = {"tank": _tables(ctx, args.tag, methods), "world": _tables(world, args.tag, methods)}
    tables = {}
    for name, fn in (("cluster_scores", cluster_scores), ("capture_finding", capture_finding),
                     ("hits_outside", hits_outside), ("hits_per_cluster", hits_per_cluster)):
        with PdfPages(out / f"{name}.pdf") as pdf:
            tables[name] = fn(data, methods, pdf)
    with PdfPages(out / "multiplicity.pdf") as pdf:
        tables["multiplicity"] = multiplicity({k: v[2] for k, v in data.items()}, methods, pdf)
    if args.overlay_tag:
        rctx = load_context(args.randoms_config)
        seed = int(((ctx.extra.get("capture_truth") or {}).get("overlay_seed", 6256)))
        evs = {k: _cf_noise(_events(c, args.overlay_tag), rctx, seed, methods, args.randoms_tag, args.overlay_run)
               for k, c in (("tank", ctx), ("world", world))}
        with PdfPages(out / "overlay_multiplicity.pdf") as pdf:
            tables["overlay_multiplicity"] = multiplicity(
                evs, methods, pdf, note=f", with {pool_label(args.overlay_run)} random-trigger hits")
    pd.set_option("display.width", 250)
    for k, t in tables.items():
        t.to_csv(out / f"{k}.csv", index=False)
        print(f"\n== {k} ==\n{t.round(4).to_string(index=False)}")
    print(f"\n[clustertruth] PDFs and CSVs in {out}")
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots clustertruth")
    p.add_argument("--world-config", required=True)
    p.add_argument("--tag", required=True, help="particle-mode tag of the `mc capture` runs")
    p.add_argument("--plot-methods", required=True,
                   help=f"comma list of clusterers to plot, from {','.join(METHOD_LABELS)}")
    p.add_argument("--overlay-tag", help="tag of the --overlay run; adds overlay_multiplicity.pdf")
    p.add_argument("--randoms-config", help="`ambe data randoms` config (needed with --overlay-tag)")
    p.add_argument("--randoms-tag", default="",
                   help="tag of the `ambe data randoms --cluster` run (ClusterFinder + NMS noise)")
    p.add_argument("--overlay-run", help="random sample of the --overlay run: 6256, or a --pool name such as 6264p6256")
    p.add_argument("--out", required=True)
    a = p.parse_args(list(argv) if argv else [])
    bad = set(a.plot_methods.split(",")) - set(METHOD_LABELS)
    if bad:
        p.error(f"unknown --plot-methods {sorted(bad)}")
    if a.overlay_tag and not (a.randoms_config and a.overlay_run):
        p.error("--overlay-tag needs --randoms-config and --overlay-run")
    run(ctx, a)
