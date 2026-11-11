"""
Why data have more triggers without a neutron cluster than MC (x) the no-source run.

    ambe plots datamczero --config <data cfg> --mc-config <mc cfg> --methods a,b,c --hitcorr-tag <tag>
        --parts budget,light,tag,readout,veto,energy --gd 0.084 --keep-scan 1,0.95,0.9,0.85,0.8
        --primary-energy-dir <dir of primary_<label>.txt> --dummy-randoms <6256 randoms cfg>

Parts (each writes <data csv>/datamczero_<part>.csv and pages into <data plots>/datamczero/):
  budget   per position: MC P(k=0) split by truth (no capture in window, dark, 1-4 hits, >=5 hits unclustered),
           the P(k=0) the data imply (P0_data / P0_6264), the extra loss f, and P(k=0) at the --gd loading.
  light    light-yield proxy: per capture, binomial thinning of its light hits through the MC's own
           P(found | hits) curve, keep fractions --keep-scan; hits per found capture vs data/MC hits per cluster.
  tag      data P(0) against IC_adjusted (all triggers), the second-pulse flag, time in run and time since
           the previous trigger, each relative to the run's IC-gated P(0).
  readout  per-trigger delayed hits and hit PMTs for empty vs non-empty IC triggers, against 6264 and 6256;
           hit rate vs time in the window.
  veto     neutron-like tree-CF clusters per trigger in passing, prompt-only-vetoed and PE>100-vetoed
           triggers, data and MC.
  energy   MC P(k>=1) against the primary neutron energy (WCSim truth), and for E below the
           9Be(a,n)12C*(4.44) kinematic limit.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..plotting import two_titles
from . import datamc_clustering as dmc
from .datamc import HEIGHTS, LABELS, PORTS

KEY = ["_source_file", "eventNumber"]
PARTS = ["budget", "light", "tag", "readout", "veto", "energy", "corrected", "efficiency", "foldmult", "activity", "cbcut", "bkgcombine"]
MC_FITS = [("nominal", "nominal MC"), ("hitcorr", "hit-corrected MC"), ("hitcorr_gd0p084", "+ Gd 0.084 %"),
           ("hitcorr_gd0p084_keep0p90", "+ Gd + light x0.9"), ("nominal_gd0p084", "+ Gd 0.084 %"),
           ("nominal_gd0p084_keep0p90", "+ Gd + light x0.9")]
VERSIONS = ["hit-corrected MC", "+ Gd", "+ Gd + light yield"]
VSTYLE = {"hit-corrected MC": ":", "+ Gd": "--", "+ Gd + light yield": "-"}
N1_EMAX_MEV = 6.36
IC_EDGES = np.arange(0.0, 2000.0 + 100.0, 100.0)
DT_EDGES = np.array([0, 2.5, 3, 4, 6, 10, 20, 60, 1e9])
WIN = (2000.0, 67000.0)


class Out:
    def __init__(self, ctx: RunContext, pdf, sub: str = "datamczero"):
        self.dir = ctx.plots_dir / sub
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pdf = pdf

    def save(self, fig, name: str):
        import matplotlib.pyplot as plt
        fig.savefig(self.dir / f"{name}.png", dpi=200)
        self.pdf.savefig(fig)
        plt.close(fig)


def _captures_path(mc: RunContext):
    tag = (mc.extra.get("capture_truth") or {}).get("overlay_tag", "ov6256")
    return mc.parquet_path(f"{mc.run_name}__capture_truth_captures__{tag}")


def _k(S) -> pd.DataFrame:
    """Non-vetoed MC events with k = capture-owned clusters (S.m_c.cap)."""
    nc = S.m_c[S.m_c.cap.to_numpy(bool)].groupby(KEY).size().rename("k").reset_index()
    e = S.m_e[KEY + ["pid", "run", "found"]].merge(nc, on=KEY, how="left")
    e["k"] = e.k.fillna(0).astype(int)
    return e


def _pos_label(p: int) -> str:
    i, j = divmod(p, len(HEIGHTS))
    return f"P{PORTS[i][0]} {HEIGHTS[j]:+.0f}"


# ---------------------------------------------------------------------------------------------- budget
def budget(S, mc: RunContext, m: str, gd: Optional[float]) -> pd.DataFrame:
    e = _k(S)
    ei = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__eventindex_mc_{m}"), columns=KEY + ["n_caps_window"])
    e = e.merge(ei, on=KEY, how="left")
    cp = pd.read_parquet(_captures_path(mc),
                         columns=KEY + ["n_light_hits"])
    nmax = cp.groupby(KEY).n_light_hits.max().rename("nmax").reset_index()
    e = e.merge(nmax, on=KEY, how="left")
    z = e.k == 0
    e["c_nocap"] = z & (e.n_caps_window.fillna(0) == 0)
    e["c_dark"] = z & ~e.c_nocap & (e.nmax.fillna(0) == 0)
    e["c_low"] = z & ~e.c_nocap & e.nmax.between(1, 4)
    e["c_unclustered"] = z & ~e.c_nocap & (e.nmax >= 5)
    if gd is not None:
        tc, model = dmc.gd_model(mc)
        w, w0, _ = dmc.gd_event_weights(tc, model, gd)
        e["wgd"] = dmc._attach(e, tc, w, w0)
    b0 = float((S.b_e.n_accepted_clusters == 0).mean())
    rows = []
    for p, g in e.groupby("pid"):
        d = S.d_e[S.d_e.pid == p]
        p0d = float((d.n_accepted_clusters == 0).mean())
        p0m = float((g.k == 0).mean())
        r = dict(method=m, pid=int(p), position=dmc.pos_title(p), n_data=len(d), n_mc=len(g),
                 p0_data=p0d, b0=b0, p0_implied=p0d / b0, p0_mc=p0m,
                 **{c: float(g[c].mean()) for c in ("c_nocap", "c_dark", "c_low", "c_unclustered")})
        r["extra"] = r["p0_implied"] - p0m
        r["f_loss"] = 1 - (1 - r["p0_implied"]) / (1 - p0m)
        if gd is not None:
            r["p0_mc_gd"] = float(np.average(g.k == 0, weights=g.wgd))
            r["f_loss_gd"] = 1 - (1 - r["p0_implied"]) / (1 - r["p0_mc_gd"])
        rows.append(r)
    return pd.DataFrame(rows)


def page_budget(out: Out, df: pd.DataFrame, gd: Optional[float]):
    import matplotlib.pyplot as plt
    ms = list(df.method.unique())
    fig, ax = plt.subplots(len(ms), 2, figsize=(16, 4.2 * len(ms)), squeeze=False,
                           gridspec_kw=dict(width_ratios=[2.2, 1]))
    for i, m in enumerate(ms):
        d = df[df.method == m].sort_values("pid")
        x = d.pid.to_numpy()
        a = ax[i, 0]
        a.plot(x, d.p0_implied, "ko-", label="data: P(0) / P0(6264)")
        a.plot(x, d.p0_mc, "s-", color="C0", label="MC P(k=0)")
        if gd is not None:
            a.plot(x, d.p0_mc_gd, "s--", color="C0", label=f"MC P(k=0), Gd {gd:g} %")
        for c, col, lab in (("c_nocap", "C1", "no capture in 2-67 us"), ("c_dark", "C2", "capture, 0 hits"),
                            ("c_low", "C3", "capture, 1-4 hits"), ("c_unclustered", "C4", "capture >=5 hits, no cluster")):
            a.plot(x, d[c], ".-", color=col, label=lab)
        a.set_xticks(x)
        a.set_xticklabels([_pos_label(p) for p in x], rotation=60, fontsize=8)
        a.set_ylabel("fraction of triggers / events")
        a.set_title(f"{LABELS.get(m, m)}: triggers without a neutron cluster, data against MC truth")
        a.legend(fontsize=8, ncol=2)
        b = ax[i, 1]
        lowf = (d.c_dark + d.c_low).to_numpy()
        b.plot(lowf, d.f_loss, "ko", label="Gd 0.10 % (WCSim)")
        if gd is not None:
            b.plot(lowf, d.f_loss_gd, "o", mfc="none", color="C0", label=f"Gd {gd:g} %")
        b.set_xlabel("MC fraction of events whose capture has 0-4 hits")
        b.set_ylabel("extra loss f the data need")
        b.set_title(f"f = 1 - (1 - P0_data/P0_6264) / (1 - P0_MC)")
        b.legend(fontsize=8)
    fig.tight_layout()
    out.save(fig, "budget")


# ---------------------------------------------------------------------------------------------- light
def _found_curve(cp: pd.DataFrame, m: str, nmax: int = 80) -> np.ndarray:
    n = np.minimum(cp.n_light_hits.to_numpy(int), nmax)
    f = (cp[f"n_owners_{m}"].to_numpy() >= 1).astype(float)
    num = np.bincount(n, weights=f, minlength=nmax + 1)
    den = np.bincount(n, minlength=nmax + 1)
    return np.divide(num, den, out=np.ones(nmax + 1), where=den > 0)


def _thinned(n: np.ndarray, keep: float, curve: np.ndarray) -> tuple:
    """Per capture: P(found) and E[kept hits x found] after binomial thinning of its n light hits."""
    from scipy.stats import binom
    nmax = len(curve) - 1
    nn = np.minimum(n, nmax)
    pf, eh = np.zeros(len(n)), np.zeros(len(n))
    for v in np.unique(nn):
        j = np.arange(v + 1)
        pm = binom.pmf(j, v, keep)
        sel = nn == v
        pf[sel] = (pm * curve[j]).sum()
        eh[sel] = (pm * curve[j] * j).sum()
    return pf, eh


def light_survival(mc: RunContext, m: str, keep: float, e: pd.DataFrame) -> np.ndarray:
    """Per event (rows of e): probability that its neutron cluster(s) survive keeping a fraction `keep` of
    the MC hits, from the MC's own P(found | light hits) curve; events without an in-window capture: 1."""
    cp = pd.read_parquet(_captures_path(mc),
                         columns=KEY + ["n_light_hits", f"n_owners_{m}"])
    curve = _found_curve(cp, m)
    n = cp.n_light_hits.to_numpy(int)
    pf1, _ = _thinned(n, 1.0, curve)
    pfk, _ = _thinned(n, keep, curve)
    ev = cp[KEY].assign(m1=1 - pf1, mk=1 - pfk).groupby(KEY).agg(m1=("m1", "prod"), mk=("mk", "prod")).reset_index()
    ev["s"] = np.clip(np.divide(1 - ev.mk, 1 - ev.m1, out=np.ones(len(ev)), where=(1 - ev.m1) > 0), 0, 1)
    return e[KEY].merge(ev[KEY + ["s"]], on=KEY, how="left").s.fillna(1.0).to_numpy(float)


def light(S, mc: RunContext, m: str, keeps: list, gd: Optional[float]) -> pd.DataFrame:
    cp = pd.read_parquet(_captures_path(mc),
                         columns=KEY + ["n_light_hits", f"n_owners_{m}"])
    curve = _found_curve(cp, m)
    e = _k(S)
    cp = cp.merge(e[KEY + ["pid"]], on=KEY)
    w = None
    if gd is not None:
        tc, model = dmc.gd_model(mc)
        gw, w0, _ = dmc.gd_event_weights(tc, model, gd)
        e["wgd"] = dmc._attach(e, tc, gw, w0)
    dc = S.d_c
    nb = len(S.b_e)
    rows = []
    base = {}
    for keep in keeps:
        pf, eh = _thinned(cp.n_light_hits.to_numpy(int), keep, curve)
        c = cp.assign(miss=1 - pf, eh=eh, pf=pf)
        ev = c.groupby(KEY).agg(miss=("miss", "prod")).reset_index()
        ee = e.merge(ev, on=KEY, how="left")
        ee["p_none"] = ee.miss.fillna(1.0)
        hits = c.eh.sum() / c.pf.sum()
        for p, g in ee.groupby("pid"):
            r = dict(method=m, keep=keep, pid=int(p), position=dmc.pos_title(p),
                     p0_proxy=float(g.p_none.mean()), p0_mc=float((g.k == 0).mean()))
            if gd is not None:
                r["p0_proxy_gd"] = float(np.average(g.p_none, weights=g.wgd))
            cc = c[c.pid == p]
            r["hits_per_found"] = float(cc.eh.sum() / cc.pf.sum())
            rows.append(r)
        print(f"[datamczero] light {m} keep {keep}: P(none) {ee.p_none.mean():.4f}, hits/found capture {hits:.2f}")
    df = pd.DataFrame(rows)
    b = df[df.keep == 1.0].set_index("pid")
    df["p0_pred"] = df.p0_mc + df.p0_proxy - df.pid.map(b.p0_proxy)
    if gd is not None:
        df["p0_pred_gd"] = df.p0_mc + df.p0_proxy_gd - df.pid.map(b.p0_proxy)
    df["hits_ratio"] = df.hits_per_found / df.pid.map(b.hits_per_found)
    mcap = S.m_c[S.m_c.cap.to_numpy(bool)]
    bc = S.b_c
    for p in df.pid.unique():
        d = dc[dc.pid == p]
        s = float((S.d_e.pid == p).sum()) / nb
        dh = (d.n_hits.sum() - s * bc.n_hits.sum()) / (len(d) - s * len(bc))
        mh = float(mcap[mcap.pid == p].n_hits.mean())
        df.loc[df.pid == p, "data_mc_hits_ratio"] = dh / mh
    return df


def page_light(out: Out, df: pd.DataFrame, bud: pd.DataFrame, gd: Optional[float]):
    import matplotlib.pyplot as plt
    ms = list(df.method.unique())
    fig, ax = plt.subplots(1, len(ms), figsize=(5.5 * len(ms), 4.6), squeeze=False)
    for i, m in enumerate(ms):
        a = ax[0, i]
        d = df[df.method == m]
        bb = bud[bud.method == m]
        g = d.groupby("keep").apply(lambda x: pd.Series(dict(
            p0=np.average(x.p0_pred, weights=x.pid.map(bb.set_index("pid").n_data)),
            p0gd=(np.average(x.p0_pred_gd, weights=x.pid.map(bb.set_index("pid").n_data)) if gd is not None else np.nan),
            hr=x.hits_ratio.mean())))
        a.plot(g.hr, g.p0, "o-", color="C0", label="MC P(k=0), thinned")
        if gd is not None:
            a.plot(g.hr, g.p0gd, "o--", color="C0", label=f"thinned, Gd {gd:g} %")
        imp = np.average(bb.p0_implied, weights=bb.n_data)
        hr = np.average(d[d.keep == 1.0].data_mc_hits_ratio, weights=bb.sort_values("pid").n_data)
        a.plot([hr], [imp], "ks", label="data")
        a.set_xlabel("hits per found capture, relative to nominal MC")
        a.set_ylabel("P(k=0)")
        a.set_title(f"{LABELS.get(m, m)}: light yield vs empty triggers (keep {', '.join(f'{k:g}' for k in g.index)})",
                    fontsize=9)
        a.legend(fontsize=8)
    fig.tight_layout()
    out.save(fig, "light")


# ---------------------------------------------------------------------------------------------- tag
FEATURES_TAG: Optional[str] = None


def _features(run: int, bkg_run: int) -> pd.DataFrame:
    if FEATURES_TAG:
        return pd.read_parquet(f"TriggerSummary/WaveformFeatures_{FEATURES_TAG}_{run}.parquet",
                               columns=["timestamp", "accepted", "IC_adjusted", "second_pulse"])
    tag = "effpair_box" if run == bkg_run else "all28_box_veto"
    f = Path(f"TriggerSummary/WaveformFeatures_AmBe2.0v4_{tag}_{run}.parquet")
    if not f.exists():
        f = Path(f"TriggerSummary/WaveformFeatures_AmBe2.0v4_effpair_box_{run}.parquet")
    if not f.exists():
        f = Path(f"TriggerSummary/WaveformFeatures_AmBe2.0v4_special_box_{run}.parquet")
    return pd.read_parquet(f, columns=["timestamp", "accepted", "IC_adjusted", "second_pulse"])


def tag(ctx: RunContext, methods: list, bkg_run: int) -> pd.DataFrame:
    rows = []
    for m in methods:
        ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_none_{m}"),
                             columns=["run", "eventTankTime", "is_cosmic", "n_accepted_clusters"])
        for run, g in ei.groupby("run"):
            w = _features(int(run), bkg_run)
            g = g.merge(w, left_on="eventTankTime", right_on="timestamp", how="left")
            g = g.sort_values("eventTankTime")
            g["dt_s"] = g.eventTankTime.diff().to_numpy() / 1e9
            g["t_frac"] = (g.eventTankTime - g.eventTankTime.min()) / max(1, g.eventTankTime.max() - g.eventTankTime.min())
            ok = g[~g.is_cosmic & g.IC_adjusted.notna()]
            z = ok.n_accepted_clusters == 0
            acc = ok[ok.accepted.astype(bool)]
            p0acc = float((acc.n_accepted_clusters == 0).mean()) if len(acc) else np.nan
            base = dict(method=m, run=int(run), p0_acc=p0acc, n_acc=len(acc))
            ic = np.digitize(ok.IC_adjusted, IC_EDGES) - 1
            for b in range(len(IC_EDGES) - 1):
                s = ic == b
                rows.append(dict(base, var="ic", lo=IC_EDGES[b], hi=IC_EDGES[b + 1], n=int(s.sum()),
                                 p0=float(z[s].mean()) if s.any() else np.nan))
            win = ok.IC_adjusted.between(700, 1200)
            for sp in (False, True):
                s = win & (ok.second_pulse.astype(bool) == sp)
                rows.append(dict(base, var="second_pulse", lo=float(sp), hi=float(sp), n=int(s.sum()),
                                 p0=float(z[s].mean()) if s.any() else np.nan))
            za = acc.n_accepted_clusters == 0
            for q in range(10):
                s = acc.t_frac.between(q / 10, (q + 1) / 10, inclusive="left" if q < 9 else "both")
                rows.append(dict(base, var="t_frac", lo=q / 10, hi=(q + 1) / 10, n=int(s.sum()),
                                 p0=float(za[s].mean()) if s.any() else np.nan))
            db = np.digitize(acc.dt_s.fillna(1e8), DT_EDGES) - 1
            for b in range(len(DT_EDGES) - 1):
                s = db == b
                rows.append(dict(base, var="dt_s", lo=DT_EDGES[b], hi=DT_EDGES[b + 1], n=int(s.sum()),
                                 p0=float(za[s].mean()) if s.any() else np.nan))
        print(f"[datamczero] tag {m}: {ei.run.nunique()} runs")
    df = pd.DataFrame(rows)
    df["rel"] = df.p0 / df.p0_acc
    return df


def _pooled(df: pd.DataFrame, src: bool, bkg_run: int) -> pd.DataFrame:
    d = df[(df.run != bkg_run) if src else (df.run == bkg_run)]
    d = d[d.n > 0]
    return d.groupby(["method", "var", "lo", "hi"])[["n", "p0", "p0_acc", "rel"]].apply(lambda x: pd.Series(dict(
        n=x.n.sum(), p0=np.average(x.p0, weights=x.n),
        p0_err=np.sqrt(np.average(x.p0, weights=x.n) * (1 - np.average(x.p0, weights=x.n)) / x.n.sum()),
        rel=np.average(x.rel, weights=x.n) if x.rel.notna().any() else np.nan,
        rel_err=np.sqrt(np.average(x.p0, weights=x.n) * (1 - np.average(x.p0, weights=x.n)) / x.n.sum())
        / max(1e-9, np.average(x.p0_acc, weights=x.n))))).reset_index()


def page_tag(out: Out, df: pd.DataFrame, bkg_run: int):
    import matplotlib.pyplot as plt
    src = _pooled(df, True, bkg_run)
    bkg = _pooled(df, False, bkg_run)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for m, g in src.groupby("method"):
        c = dict(clusterfinder="C0", clusterfinder_nms="C1", peak="C2").get(m, "C3")
        for a, var, xl in ((ax[0, 0], "ic", "IC_adjusted"), (ax[1, 0], "t_frac", "time in run (fraction)"),
                           (ax[1, 1], "dt_s", "time since previous trigger [s]")):
            h = g[g["var"] == var]
            h = h[h.n >= 200]
            x = 0.5 * (h.lo + h.hi) if var != "dt_s" else np.sqrt(np.maximum(h.lo, 0.5) * np.minimum(h.hi, 200))
            y, ye = (h.p0, h.p0_err) if var == "ic" else (h.rel, h.rel_err)
            a.errorbar(x, y, yerr=ye, fmt="o-", color=c, ms=4, label=LABELS.get(m, m))
            a.set_xlabel(xl)
        sp = g[g["var"] == "second_pulse"]
        ax[0, 1].plot(sp.lo, sp.rel, "o-", color=c, label=LABELS.get(m, m))
    b = bkg[(bkg["var"] == "ic") & (bkg.n >= 30) & (bkg.method == "clusterfinder_nms")]
    ax[0, 0].plot(0.5 * (b.lo + b.hi), b.p0, "k.--", label=f"{bkg_run} (no source), CF + NMS")
    ax[0, 0].set_ylabel("P(no cluster), non-vetoed triggers")
    ax[0, 0].set_title("empty triggers against the BGO charge (IC gate 700-1200)")
    ax[0, 1].set_xticks([0, 1])
    ax[0, 1].set_xticklabels(["no second pulse", "second pulse"])
    ax[0, 1].set_ylabel("P(0) / P(0) of the IC-gated triggers")
    ax[0, 1].set_title("IC 700-1200: second BGO pulse")
    ax[1, 0].set_ylabel("P(0) / run's IC-gated P(0)")
    ax[1, 0].set_title("IC-gated triggers: time in run")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_ylabel("P(0) / run's IC-gated P(0)")
    ax[1, 1].set_title("IC-gated triggers: time since the previous trigger")
    for a in ax.flat:
        a.legend(fontsize=8)
    fig.tight_layout()
    out.save(fig, "tag")


# ---------------------------------------------------------------------------------------------- readout
def _trigger_hits(path, keep: Optional[set] = None) -> pd.DataFrame:
    h = pd.read_parquet(path, columns=["trigger", "pmtID", "t"])
    h = h[(h.t >= WIN[0]) & (h.t <= WIN[1])]
    if keep is not None:
        h = h[h.trigger.isin(keep)]
    g = h.groupby("trigger")
    return pd.DataFrame({"n_hits": g.size(), "n_pmts": g.pmtID.nunique()}).reset_index(), h


def readout(ctx: RunContext, dummy: Optional[RunContext], m: str, bkg_run: int) -> tuple:
    ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_{m}"),
                         columns=["run", "trigger", "is_cosmic", "n_accepted_clusters"])
    ei = ei[~ei.is_cosmic]
    eb = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_none_{m}"),
                         columns=["run", "trigger", "is_cosmic", "n_accepted_clusters"])
    eb = eb[~eb.is_cosmic & (eb.run == bkg_run)]
    rows, prof = [], []
    tb = np.arange(WIN[0], WIN[1] + 1, 5000.0)
    for run in sorted(set(ei.run) | {bkg_run}):
        e = eb if run == bkg_run else ei[ei.run == run]
        if len(e) == 0:
            continue
        th, h = _trigger_hits(ctx.parquet_path(f"{ctx.run_name}__random_hits_{run}"), set(e.trigger))
        t = e.merge(th, on="trigger", how="left").fillna({"n_hits": 0, "n_pmts": 0})
        t["no_cluster"] = t.n_accepted_clusters == 0
        rows.append(t.assign(sample="no-source" if run == bkg_run else "data"))
        h = h.merge(t[["trigger", "no_cluster"]], on="trigger")
        for em, hh in h.groupby("no_cluster"):
            cnt = np.histogram(hh.t, tb)[0] / float((t["no_cluster"] == em).sum())
            prof.append(dict(run=run, no_cluster=bool(em), **{f"t{int(a/1000)}": v for a, v in zip(tb[:-1], cnt)}))
        print(f"[datamczero] readout run {run}: {len(t):,} triggers")
    if dummy is not None:
        f = dummy.parquet_path(f"{dummy.run_name}__random_hits_6256")
        th, _ = _trigger_hits(f)
        rows.append(th.assign(run=6256, sample="dummy 6256", no_cluster=np.nan, n_accepted_clusters=np.nan))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(prof)


def page_readout(out: Out, df: pd.DataFrame, prof: pd.DataFrame, m: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    e = np.arange(0, 141, 2)
    d = df[df["sample"] == "data"]
    for lab, s, st in (("data IC, no cluster", d[d["no_cluster"] == True], "-"), ("data IC, >=1 cluster", d[d["no_cluster"] == False], "--"),  # noqa: E712
                       ("no-source 6264", df[df["sample"] == "no-source"], ":"), ("dummy 6256", df[df["sample"] == "dummy 6256"], "-.")):
        if len(s):
            ax[0].hist(s.n_pmts, e, density=True, histtype="step", ls=st, label=f"{lab} ({len(s):,})")
            ax[1].hist(s.n_hits, np.arange(0, 401, 5), density=True, histtype="step", ls=st, label=lab)
    ax[0].set_xlabel("PMTs with a hit at 2-67 us")
    ax[1].set_xlabel("hits at 2-67 us")
    ax[1].set_yscale("log")
    ax[0].set_title(f"{LABELS.get(m, m)}: readout of empty vs non-empty triggers")
    cols = [c for c in prof.columns if c.startswith("t")]
    x = np.array([int(c[1:]) for c in cols]) + 2.5
    for em, st in ((True, "o-"), (False, "s--")):
        p = prof[(prof["no_cluster"] == em) & (prof.run != 6264)][cols].mean()
        ax[2].plot(x, p.to_numpy() / 5.0, st, label="no cluster" if em else ">=1 cluster")
    ax[2].set_xlabel("hit time [us]")
    ax[2].set_ylabel("hits per trigger per us")
    ax[2].set_title("data IC triggers: hit rate through the window")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    out.save(fig, f"readout_{m}")


# ---------------------------------------------------------------------------------------------- veto
def _veto_rates(cf: pd.DataFrame, key: list, trig: pd.DataFrame) -> pd.DataFrame:
    from ..data.selection import CutCriteria
    c = CutCriteria()
    cf = cf.assign(prompt=cf.clusterTime < c.cosmic_ct_threshold, big=cf.clusterPE > c.cosmic_pe_threshold,
                   neut=cf.clusterTime.between(*WIN) & (cf.clusterPE <= c.cosmic_pe_threshold))
    g = cf.groupby(key).agg(prompt=("prompt", "any"), big=("big", "any"), n_neut=("neut", "sum")).reset_index()
    t = trig.merge(g, on=key, how="left").fillna({"prompt": False, "big": False, "n_neut": 0})
    t["cat"] = np.where(t.big.astype(bool), "PE>100", np.where(t.prompt.astype(bool), "prompt only", "pass"))
    return t


def veto(ctx: RunContext, mc: RunContext, bkg_run: int) -> pd.DataFrame:
    ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_clusterfinder"), columns=["run", "trigger"])
    rows = []
    for run, g in ei.groupby("run"):
        cf = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_cfclusters_{run}"),
                             columns=["trigger", "clusterTime", "clusterPE"])
        t = _veto_rates(cf, ["trigger"], g[["trigger"]])
        rows.append(t.assign(sample="data" if run != bkg_run else "no-source", run=run))
    ev = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__events"), columns=["_source_file", "eventID", "eventNumber"])
    cf = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__clusterfinder"),
                         columns=["_source_file", "eventID", "clusterTime", "clusterPE"])
    t = _veto_rates(cf, ["_source_file", "eventID"], ev[["_source_file", "eventID"]])
    rows.append(t.assign(sample="MC", run=-1))
    t = pd.concat(rows, ignore_index=True)
    out = t.groupby(["sample", "cat"]).agg(n=("n_neut", "size"), neut_per_trig=("n_neut", "mean"),
                                           p_ge1=("n_neut", lambda x: float((x > 0).mean()))).reset_index()
    tot = out.groupby("sample").n.transform("sum")
    out["share"] = out.n / tot
    return out


# ---------------------------------------------------------------------------------------------- activity
def activity(ctx: RunContext, dummy: RunContext, bkg_run: int, methods: list, tag: str, src_run: int) -> tuple:
    """No-source run, the two dummy-trigger runs and one IC-gated source run under the same cosmic veto
    (production ClusterFinder cluster at t < 2 us or PE > 100): veto fraction, hits / PMTs per trigger and
    clusters per trigger at 2-67 us, and the hit rate through the window."""
    from ..data.selection import CutCriteria
    c = CutCriteria()
    ic = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_{methods[0]}"), columns=["run", "trigger"])
    sets = [(ctx, bkg_run, f"no-source {bkg_run}"), (dummy, 6254, "dummy 6254"), (dummy, 6256, "dummy 6256"),
            (ctx, src_run, f"AmBe {src_run}, IC gate")]
    rows, prof = [], []
    tb = np.arange(WIN[0], WIN[1] + 1, 5000.0)
    for cx, run, lab in sets:
        base = f"{cx.run_name}__random_%s_{run}"
        tr = pd.read_parquet(cx.parquet_path(base % "triggers"), columns=["trigger"])
        if run == src_run:
            tr = tr[tr.trigger.isin(ic[ic.run == run].trigger)]
        cf = pd.read_parquet(cx.parquet_path(base % "cfclusters"), columns=["trigger", "clusterTime", "clusterPE"])
        vet = set(cf[(cf.clusterTime < c.cosmic_ct_threshold) | (cf.clusterPE > c.cosmic_pe_threshold)].trigger)
        keep = tr[~tr.trigger.isin(vet)]
        th, h = _trigger_hits(cx.parquet_path(base % "hits"), set(keep.trigger))
        t = keep.merge(th, on="trigger", how="left").fillna({"n_hits": 0, "n_pmts": 0})
        rc = pd.read_parquet(cx.parquet_path(base % "recoclusters" + f"__{tag}"), columns=["trigger", "method", "t_med"])
        rc = rc[rc.trigger.isin(keep.trigger) & rc.t_med.between(*WIN)]
        row = dict(sample=lab, run=run, n_trig=len(tr), veto_frac=1 - len(keep) / len(tr), n_kept=len(keep),
                   hits_per_trig=t.n_hits.mean(), hits_per_us=t.n_hits.mean() / ((WIN[1] - WIN[0]) / 1000.0),
                   pmts_per_trig=t.n_pmts.mean(), frac_trig_no_hit=float((t.n_hits == 0).mean()))
        for m in methods:
            n = keep.trigger.map(rc[rc.method == m].groupby("trigger").size()).fillna(0)
            row[f"cl_per_trig_{m}"] = n.mean()
            row[f"p0_{m}"] = float((n == 0).mean())
        rows.append(row)
        cnt = np.histogram(h.t, tb)[0] / float(len(keep)) / 5.0
        prof.append(dict(sample=lab, **{f"t{int(a / 1000)}": v for a, v in zip(tb[:-1], cnt)}))
        print(f"[datamczero] activity {lab}: {len(tr):,} triggers, veto {row['veto_frac']:.3f}, "
              f"{row['hits_per_us']:.3f} hits/us")
    return pd.DataFrame(rows), pd.DataFrame(prof)


def page_activity(out: Out, df: pd.DataFrame, prof: pd.DataFrame, methods: list):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    cols = [c for c in prof.columns if c.startswith("t")]
    x = np.array([int(c[1:]) for c in cols]) + 2.5
    for i, r in prof.iterrows():
        ax[0].plot(x, r[cols].to_numpy(float), "o-", color=f"C{i}", label=r["sample"])
    ax[0].set_xlabel("hit time [us]")
    ax[0].set_ylabel("hits per trigger per us (after the cosmic veto)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8, frameon=False)
    xm = np.arange(len(methods))
    for i, r in df.iterrows():
        ax[1].plot(xm, [r[f"cl_per_trig_{m}"] for m in methods], "o-", color=f"C{i}", label=r["sample"])
    ax[1].set_xticks(xm)
    ax[1].set_xticklabels([LABELS.get(m, m) for m in methods])
    ax[1].set_ylabel("clusters per trigger at 2-67 us")
    ax[1].legend(fontsize=8, frameon=False)
    two_titles(fig, "Detector activity without the source",
               "Tank activity under the same cosmic veto: no-source run, dummy triggers, AmBe source run")
    out.save(fig, "activity")


# ---------------------------------------------------------------------------------------------- cbcut
CB_EDGES = np.arange(0.0, 1.0 + 0.02, 0.02)
T_EDGES_US = np.arange(2.0, 67.0 + 2.5, 2.5)


def cbcut(ctx: RunContext, bkg_run: int) -> tuple:
    """Production ClusterFinder clusters at 2-67 us in non-vetoed triggers (source runs: IC gate;
    no-source run: every trigger): charge balance and time, per trigger, split at the v4 CB cut."""
    from ..data.selection import CutCriteria
    c = CutCriteria()
    ic = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_clusterfinder"), columns=["run", "trigger", "is_cosmic"])
    nb = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_none_clusterfinder"), columns=["run", "trigger", "is_cosmic"])
    trig = pd.concat([ic[(ic.run != bkg_run) & ~ic.is_cosmic].assign(sample="source runs, IC gate"),
                      nb[(nb.run == bkg_run) & ~nb.is_cosmic].assign(sample=f"no-source {bkg_run}")])
    cl = []
    for run, g in trig.groupby("run"):
        cf = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_cfclusters_{run}"),
                             columns=["trigger", "clusterTime", "clusterPE", "clusterCB"])
        cf = cf[cf.clusterTime.between(*WIN) & cf.trigger.isin(g.trigger)]
        cl.append(cf.assign(run=run, sample=g["sample"].iloc[0]))
    cl = pd.concat(cl, ignore_index=True)
    cl["cb_pass"] = (cl.clusterCB > c.ccb_min) & (cl.clusterCB < c.ccb_max)
    ntrig = trig.groupby("sample").size()
    hist, summ = [], []
    for smp, g in cl.groupby("sample"):
        n = float(ntrig[smp])
        h = np.histogram(g.clusterCB, CB_EDGES)[0] / n
        hist += [dict(sample=smp, kind="cb", lo=a, hi=b, per_trig=v) for a, b, v in zip(CB_EDGES[:-1], CB_EDGES[1:], h)]
        for ok, gg in g.groupby("cb_pass"):
            ht = np.histogram(gg.clusterTime / 1000.0, T_EDGES_US)[0] / n
            hist += [dict(sample=smp, kind="t_pass" if ok else "t_fail", lo=a, hi=b, per_trig=v)
                     for a, b, v in zip(T_EDGES_US[:-1], T_EDGES_US[1:], ht)]
            late = gg[gg.clusterTime > 40000].shape[0] / n
            early = gg[gg.clusterTime < 20000].shape[0] / n
            summ.append(dict(sample=smp, cb_pass=bool(ok), n_trig=int(n), clusters_per_trig=len(gg) / n,
                             per_trig_2_20us=early, per_trig_40_67us=late))
    return pd.DataFrame(hist), pd.DataFrame(summ)


def page_cbcut(out: Out, hist: pd.DataFrame):
    import matplotlib.pyplot as plt
    from ..data.selection import CutCriteria
    cbmax = CutCriteria().ccb_max
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, (smp, g) in enumerate(hist.groupby("sample")):
        for a, kind in zip(ax, ("cb", "t_fail", "t_pass")):
            q = g[g.kind == kind]
            a.plot(0.5 * (q.lo + q.hi), q.per_trig, "o-", ms=3, color=f"C{i}", label=smp)
    ax[0].set_xlabel("cluster charge balance")
    ax[0].set_title(f"clusters per trigger, 2-67 us (v4 box keeps CB < {cbmax:g})")
    ax[1].set_title(f"clusters with CB >= {cbmax:g} (removed by the box)")
    ax[2].set_title(f"clusters with CB < {cbmax:g} (kept)")
    for a in ax[1:]:
        a.set_xlabel("cluster time [us]")
        a.set_yscale("log")
    for a in ax:
        a.set_ylabel("clusters per trigger per bin")
        a.legend(fontsize=8, frameon=False)
    two_titles(fig, "What the charge-balance cut removes",
               "Clusters per trigger after the cosmic veto, 2-67 us: source runs (IC gate) vs no-source run, split at the v4 box CB cut")
    out.save(fig, "cbcut")


# ---------------------------------------------------------------------------------------------- bkgcombine
BKG_T_EDGES = np.array([2, 10, 20, 30, 40, 50, 67]) * 1000.0


def _bkg_counts(cx: RunContext, run: int, m: str, tag: str, trig: Optional[pd.Series] = None):
    from ..data.selection import CutCriteria
    c = CutCriteria()
    b = f"{cx.run_name}__random_%s_{run}"
    if trig is None:
        tr = pd.read_parquet(cx.parquet_path(b % "triggers"), columns=["trigger"]).trigger
        cf = pd.read_parquet(cx.parquet_path(b % "cfclusters"), columns=["trigger", "clusterTime", "clusterPE"])
        vet = set(cf[(cf.clusterTime < c.cosmic_ct_threshold) | (cf.clusterPE > c.cosmic_pe_threshold)].trigger)
        trig = tr[~tr.isin(vet)]
    rc = pd.read_parquet(cx.parquet_path(b % "recoclusters" + f"__{tag}"), columns=["trigger", "method", "t_med", "f_pe_total"])
    rc = rc[(rc.method == m) & rc.trigger.isin(trig) & rc.t_med.between(*WIN)]
    return trig.map(rc.groupby("trigger").size()).fillna(0).astype(int).to_numpy(), rc


def _bkg_fit(hs, run, hb, s_max, fp):
    from scipy.stats import chi2 as chi2d
    from ..efficiency_extraction import efficiency_fit as ef

    def mh(h):
        return ef.multiplicity_from_counts(run, {k: int(v) for k, v in enumerate(h) if k > 0}, int(h.sum()),
                                           n_bins=len(h), overflow_policy="clip")
    r = ef.fit_datadriven_profile(mh(hs), mh(hb), ef.make_grid(fp["eff_grid"]), ef.make_grid({**fp["s_grid"], "max": s_max}))
    return r.best_eff, r.eff_lo, r.eff_hi, r.best_lambda, float(chi2d.sf(r.chi2_min, r.ndof_standard))


def bkgcombine(ctx: RunContext, dummy: RunContext, bkg_run: int, methods: list, tag: str, positions: list,
               s_max: float, n_jobs: int) -> tuple:
    """No-source run vs dummy run 6256 under the same veto (multiplicity, cluster time and charge), and the
    data-driven fit at the given positions with four templates: no-source ungated, dummy, both combined,
    no-source IC-gated."""
    from joblib import Parallel, delayed
    from scipy.stats import chi2_contingency, ks_2samp
    fp = dict(ctx.fit_params)
    nb = int(fp["n_bins"])
    comp, times, jobs = [], [], []
    for m in methods:
        e = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_none_{m}"))
        n64, c64 = _bkg_counts(ctx, bkg_run, m, tag, e[(e.run == bkg_run) & ~e.is_cosmic].trigger)
        n56, c56 = _bkg_counts(dummy, 6256, m, tag)
        g = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_{m}"))
        g64 = g[(g.run == bkg_run) & ~g.is_cosmic].n_accepted_clusters.to_numpy(int)
        h = {k: np.bincount(np.minimum(v, nb - 1), minlength=nb) for k, v in (("a", n64), ("b", n56), ("g", g64))}
        ct = np.vstack([h["a"], h["b"]])
        st, pv, _, _ = chi2_contingency(ct[:, ct.sum(0) > 0])
        comp.append(dict(method=m, n_nosource=len(n64), n_dummy=len(n56), per_trig_nosource=n64.mean(), per_trig_dummy=n56.mean(),
                         multi_nosource=float((n64 >= 2).mean()), multi_dummy=float((n56 >= 2).mean()),
                         hist_nosource=" ".join(map(str, h["a"])), hist_dummy=" ".join(map(str, h["b"])),
                         hist_nosource_ic=" ".join(map(str, h["g"])), mult_chi2=st, mult_p=pv,
                         time_ks_p=ks_2samp(c64.t_med, c56.t_med).pvalue, pe_ks_p=ks_2samp(c64.f_pe_total, c56.f_pe_total).pvalue))
        a, b = np.histogram(c64.t_med, BKG_T_EDGES)[0], np.histogram(c56.t_med, BKG_T_EDGES)[0]
        for lo, hi, x, y in zip(BKG_T_EDGES[:-1], BKG_T_EDGES[1:], a, b):
            times.append(dict(method=m, lo_us=lo / 1000, hi_us=hi / 1000, per_trig_nosource=x / len(n64), per_trig_dummy=y / len(n56),
                              diff=x / len(n64) - y / len(n56), diff_err=np.sqrt(x / len(n64) ** 2 + y / len(n56) ** 2)))
        tmpl = {f"{bkg_run} ungated": h["a"], "6256 dummy": h["b"], f"{bkg_run}+6256": h["a"] + h["b"], f"{bkg_run} IC-gated": h["g"]}
        src = g[(g.run != bkg_run) & ~g.is_cosmic]
        for pos in positions:
            q = src[np.isclose(src.sourceX, pos[0]) & np.isclose(src.sourceY, pos[1]) & np.isclose(src.sourceZ, pos[2])]
            if q.empty:
                raise SystemExit(f"[datamczero] bkgcombine: no source triggers at {pos}")
            hs = np.bincount(np.minimum(q.n_accepted_clusters.to_numpy(int), nb - 1), minlength=nb)
            for lab, hb in tmpl.items():
                jobs.append((m, pos, lab, len(q), hs, int(q.run.iloc[0]), hb))
    res = Parallel(n_jobs=n_jobs)(delayed(_bkg_fit)(hs, run, hb, s_max, fp) for _, _, _, _, hs, run, hb in jobs)
    fits = pd.DataFrame([dict(method=m, x_pos=p[0], y_pos=p[1], z_pos=p[2], template=lab, n_trig=n, eff=r[0], eff_lo=r[1],
                              eff_hi=r[2], s=r[3], p=r[4]) for (m, p, lab, n, *_), r in zip(jobs, res)])
    return pd.DataFrame(comp), pd.DataFrame(times), fits


def page_bkgcombine(out: Out, comp: pd.DataFrame, times: pd.DataFrame, bkg_run: int):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i, (m, q) in enumerate(times.groupby("method", sort=False)):
        x = 0.5 * (q.lo_us + q.hi_us)
        w = (q.hi_us - q.lo_us).to_numpy()
        ax[0].plot(x, 1000 * q.per_trig_nosource / w, "o-", color=f"C{i}", label=f"{LABELS.get(m, m)}, {bkg_run}")
        ax[0].plot(x, 1000 * q.per_trig_dummy / w, "s--", color=f"C{i}", label=f"{LABELS.get(m, m)}, 6256")
        ax[1].errorbar(x, 1000 * q["diff"] / w, yerr=1000 * q["diff_err"] / w, fmt="o-", color=f"C{i}", label=LABELS.get(m, m))
    ax[0].set_ylabel("clusters per 1000 triggers per us")
    ax[1].set_ylabel(f"{bkg_run} - 6256, clusters per 1000 triggers per us")
    for a in ax:
        a.set_xlabel("cluster time [us]")
        a.legend(fontsize=8, frameon=False)
    two_titles(fig, "Background with and without the source trigger",
               f"No-source run {bkg_run} and dummy run 6256 after the same cosmic veto; clusters per 1000 triggers per us, 2-67 us")
    out.save(fig, "bkgcombine")


# ---------------------------------------------------------------------------------------------- energy
def energy(S, edir: str, m: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(edir) / "primary_*.txt"))):
        lab = re.search(r"primary_(.+)\.txt$", f).group(1)
        for line in open(f):
            p = line.split()
            e = [float(x.split(":")[1]) for x in p[2:] if x.split(":")[0] == "0"]
            if e:
                rows.append((lab, p[0], int(p[1]), e[0]))
    pe = pd.DataFrame(rows, columns=["label", "wcsim", "eventNumber", "E_MeV"])
    pe["_source_file"] = "ANNIEEvent_MC_AmBe_" + pe.label + "_" + pe.wcsim.str.replace("wcsim_", "", regex=False)
    e = _k(S).merge(pe[KEY + ["E_MeV"]], on=KEY, how="left")
    miss = e.E_MeV.isna().mean()
    if miss > 0.001:
        raise SystemExit(f"[datamczero] energy: {miss:.4f} of MC events have no primary energy")
    e["eps"] = e.k > 0
    out = []
    edges = np.array([0, 1, 2, 3, 4, 5, 6, N1_EMAX_MEV, 8, 9, 11.5])
    e["eb"] = np.digitize(e.E_MeV, edges) - 1
    for p, g in e.groupby("pid"):
        r = dict(method=m, pid=int(p), position=dmc.pos_title(p), eps_all=float(g.eps.mean()),
                 eps_n1=float(g.eps[g.E_MeV <= N1_EMAX_MEV].mean()), f_above=float((g.E_MeV > N1_EMAX_MEV).mean()))
        for b in range(len(edges) - 1):
            s = g.eb == b
            r[f"eps_E{edges[b]:g}_{edges[b+1]:g}"] = float(g.eps[s].mean()) if s.any() else np.nan
        out.append(r)
    return pd.DataFrame(out), e[["E_MeV"]].describe()


def page_energy(out: Out, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    cols = [c for c in df.columns if c.startswith("eps_E")]
    mid = [0.5 * sum(float(v) for v in c[5:].split("_")) for c in cols]
    ms = list(df.method.unique())
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for m in ms:
        d = df[df.method == m]
        for p, st in ((0, "o-"), (14, "s--"), (24, "^:")):
            r = d[d.pid == p]
            if len(r):
                ax[0].plot(mid, r[cols].to_numpy()[0], st, label=f"{LABELS.get(m, m)}, {dmc.pos_title(p)}")
        ax[1].plot(d.pid, d.eps_n1 - d.eps_all, "o-", label=LABELS.get(m, m))
    ax[0].set_xlabel("primary neutron kinetic energy [MeV]")
    ax[0].set_ylabel("MC P(k>=1)")
    ax[0].set_title("MC neutron-cluster efficiency against the neutron energy")
    ax[1].set_xticks(range(len(PORTS) * len(HEIGHTS)))
    ax[1].set_xticklabels([_pos_label(p) for p in range(len(PORTS) * len(HEIGHTS))], rotation=60, fontsize=8)
    ax[1].set_ylabel("P(k>=1): E <= 6.36 MeV minus all")
    ax[1].set_title("tagged-group kinematic limit (12C* 4.44 MeV) against the full AmBe spectrum")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    out.save(fig, "energy")


# ---------------------------------------------------------------------------------------------- corrected
def _vpmf(k: np.ndarray, w: np.ndarray, sv: Optional[np.ndarray], kmax: int) -> np.ndarray:
    h = np.bincount(np.minimum(k, kmax), weights=w if sv is None else w * sv, minlength=kmax + 1).astype(float)
    if sv is not None:
        h[0] += float((w * (1 - sv)).sum())
    return h / h.sum()


def corrected(S, mc: RunContext, m: str, gd: float, keep: float, kmax: int) -> pd.DataFrame:
    """Clusters per trigger, data against MC (x) 6264 for the hit-corrected MC, + Gd reweight, + Gd and light-yield
    survival; per position and pooled (pid -1, positions weighted by data triggers)."""
    e = _k(S).merge(S.m_e[KEY + ["w"]], on=KEY)
    tc, model = dmc.gd_model(mc)
    gw, w0, _ = dmc.gd_event_weights(tc, model, gd)
    e["wgd"] = dmc._attach(e, tc, gw, w0)
    e["s"] = light_survival(mc, m, keep, e)
    pb = dmc.pmf(S.b_e.n_accepted_clusters.to_numpy(), kmax)
    rows = []
    nd = np.bincount(S.d_e.pid, minlength=dmc.NPOS).astype(float)
    acc = {v: np.zeros(kmax + 1) for v in VERSIONS}
    accm = {v: np.zeros(kmax + 1) for v in VERSIONS}
    for p, g in e.groupby("pid"):
        k = g.k.to_numpy(int)
        one = np.ones(len(g))
        pms = {VERSIONS[0]: _vpmf(k, one, None, kmax), VERSIONS[1]: _vpmf(k, g.wgd.to_numpy(), None, kmax),
               VERSIONS[2]: _vpmf(k, g.wgd.to_numpy(), g.s.to_numpy(), kmax)}
        pdat = dmc.pmf(S.d_e[S.d_e.pid == p].n_accepted_clusters.to_numpy(), kmax)
        for v, pm in pms.items():
            pf = dmc.convolve_pmf(pm, pb, kmax)
            acc[v] += nd[p] * pf
            accm[v] += nd[p] * pm
            rows.append(dict(method=m, pid=int(p), position=dmc.pos_title(p), version=v, n_data=nd[p],
                             eps_mc=1 - pm[0], **{f"mc{i}": pf[i] for i in range(kmax + 1)},
                             **{f"data{i}": pdat[i] for i in range(kmax + 1)}))
    pdat = dmc.pmf(S.d_e.n_accepted_clusters.to_numpy(), kmax)
    for v in VERSIONS:
        pf, pm = acc[v] / nd.sum(), accm[v] / nd.sum()
        rows.append(dict(method=m, pid=-1, position="all positions", version=v, n_data=nd.sum(), eps_mc=1 - pm[0],
                         **{f"mc{i}": pf[i] for i in range(kmax + 1)}, **{f"data{i}": pdat[i] for i in range(kmax + 1)}))
    df = pd.DataFrame(rows)
    df["b0"] = pb[0]
    df["p0_implied"] = df.data0 / df.b0
    df["eps_data_implied"] = 1 - df.p0_implied
    df["eps_ratio"] = df.eps_data_implied / df.eps_mc
    df["unexplained_p0"] = df.data0 - df.mc0
    return df


def page_corrected(out: Out, df: pd.DataFrame, bud: Optional[pd.DataFrame], gd: float, keep: float, kmax: int):
    import matplotlib.pyplot as plt
    tag = f"Gd {gd:g} %, light yield x{keep:g}"
    for m, d in df.groupby("method"):
        a0 = d[d.pid == -1]
        n = np.arange(kmax + 1)
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
        dd = a0.iloc[0]
        ax[0].plot(n, [dd[f"data{i}"] for i in n], "ko", label="data, IC triggers")
        for _, r in a0.iterrows():
            ax[0].plot(n, [r[f"mc{i}"] for i in n], VSTYLE[r.version], color="C0",
                       label=f"{r.version} (x) 6264: P(0) {r.mc0:.3f}")
            ax[1].plot(n, [dd[f"data{i}"] / r[f"mc{i}"] for i in n], "o" + VSTYLE[r.version], color="C0", label=r.version)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("clusters per trigger")
        ax[0].set_ylabel("fraction of triggers")
        ax[0].set_title(f"{LABELS.get(m, m)}: data P(0) {dd.data0:.3f}; MC corrected for {tag}", fontsize=10)
        ax[1].set_xlabel("clusters per trigger")
        ax[1].set_ylabel("data / MC")
        ax[1].set_title("data / prediction per bin")
        p = d[d.pid >= 0]
        for v in VERSIONS:
            q = p[p.version == v].sort_values("pid")
            ax[2].plot(q.pid, q.unexplained_p0, "o" + VSTYLE[v], color="C0", label=v)
        ax[2].set_xticks(range(dmc.NPOS))
        ax[2].set_xticklabels([_pos_label(i) for i in range(dmc.NPOS)], rotation=60, fontsize=8)
        ax[2].set_ylabel("P(0): data - prediction")
        ax[2].set_title("unexplained empty triggers per position")
        for a in ax:
            a.legend(fontsize=8)
        fig.tight_layout()
        out.save(fig, f"{m}_mult_corrected")

        fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(20, 17), sharex=True, sharey=True)
        for pid in range(dmc.NPOS):
            a = ax.flat[pid]
            q = p[p.pid == pid]
            if q.empty:
                a.set_visible(False)
                continue
            r0 = q.iloc[0]
            a.plot(n, [r0[f"data{i}"] for i in n], "ko", ms=4)
            for v in (VERSIONS[0], VERSIONS[2]):
                r = q[q.version == v].iloc[0]
                a.plot(n, [r[f"mc{i}"] for i in n], VSTYLE[v], color="C0")
            rc = q[q.version == VERSIONS[2]].iloc[0]
            a.set_yscale("log")
            a.set_title(f"{dmc.pos_title(pid)}: P(0) {r0.data0:.3f} / {q[q.version == VERSIONS[0]].iloc[0].mc0:.3f}"
                        f" / {rc.mc0:.3f}", fontsize=8)
        fig.suptitle(f"{LABELS.get(m, m)}: clusters per trigger; data (points), hit-corrected MC (x) 6264 (dotted), "
                     f"+ {tag} (solid); titles: P(0) data / MC / corrected MC", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out.save(fig, f"{m}_mult_grid_corrected")

        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
        for v in (VERSIONS[0], VERSIONS[2]):
            q = p[p.version == v].sort_values("pid")
            f = 1 - (1 - q.p0_implied) / q.eps_mc
            ax[0].plot(q.pid, q.eps_ratio, "o" + VSTYLE[v], color="C0", label=v)
            if bud is not None:
                lowf = q.pid.map(bud[bud.method == m].set_index("pid").eval("c_dark + c_low"))
                c = np.polyfit(lowf, f, 1)
                ax[1].plot(lowf, f, "o", mfc="none" if v == VERSIONS[0] else None, color="C0",
                           label=f"{v}: f = {c[1]:.3f} + {c[0]:.2f} x")
        ax[0].set_xticks(range(dmc.NPOS))
        ax[0].set_xticklabels([_pos_label(i) for i in range(dmc.NPOS)], rotation=60, fontsize=8)
        ax[0].set_ylabel("eps data (1 - P0/P0_6264) / eps MC")
        ax[0].set_title(f"{LABELS.get(m, m)}: data/MC neutron-cluster efficiency, before and after {tag}", fontsize=10)
        ax[1].set_xlabel("MC fraction of events whose capture has 0-4 hits")
        ax[1].set_ylabel("extra loss f still needed")
        ax[1].set_title("what is left: a constant term and a term that follows the dark captures")
        for a in ax:
            a.legend(fontsize=8)
        fig.tight_layout()
        out.save(fig, f"{m}_unexplained")


def page_corrected_eff(out: Out, ctx: RunContext, mc: RunContext, methods: list, gd: float, keep: float, tgate: str):
    import matplotlib.pyplot as plt
    ctag = f"_gd{gd:g}".replace(".", "p") + f"_keep{keep:.2f}".replace(".", "p")
    rows = []
    for m in methods:
        fd = ctx.csv_path(f"efficiency_fit_by_position_fold_data_{tgate}_{m}")
        fa = mc.csv_path(f"efficiency_fit_by_position_fold_mchitcorr_{tgate}_{m}")
        fc = mc.csv_path(f"efficiency_fit_by_position_fold_mchitcorr{ctag}_{tgate}_{m}")
        if not (fd.exists() and fa.exists() and fc.exists()):
            print(f"[datamczero] corrected efficiency {m}: fold fits missing ({fd.name}, {fa.name}, {fc.name}), skipped")
            continue
        d, a, c = (pd.read_csv(f) for f in (fd, fa, fc))
        d, a, c = (x[x.level == "position"].assign(pid=lambda y: dmc.pos_id(y.rename(columns={
            "x_pos": "sourceX", "y_pos": "sourceY", "z_pos": "sourceZ"}))) for x in (d, a, c))
        t = d[["pid", "naive_eff", "poisson_eff", "dds_eff", "dds_eff_lo", "dds_eff_hi"]].merge(
            a[["pid", "truth_eff", "dds_eff"]].rename(columns={"truth_eff": "truth_mc", "dds_eff": "fit_mc"}), on="pid").merge(
            c[["pid", "truth_eff", "dds_eff"]].rename(columns={"truth_eff": "truth_corr", "dds_eff": "fit_corr"}), on="pid")
        t["method"] = m
        t["ratio_mc"] = t.dds_eff / t.truth_mc
        t["ratio_corr"] = t.dds_eff / t.truth_corr
        rows.append(t)
    if not rows:
        return None
    t = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(len(set(t.method)), 2, figsize=(16, 4.4 * len(set(t.method))), squeeze=False)
    for i, (m, q) in enumerate(t.groupby("method")):
        q = q.sort_values("pid")
        ax[i, 0].errorbar(q.pid, q.dds_eff, yerr=[q.dds_eff - q.dds_eff_lo, q.dds_eff_hi - q.dds_eff], fmt="ko",
                          label="data, data-driven fit")
        ax[i, 0].plot(q.pid, q.truth_mc, ":", color="C0", label="MC truth, hit-corrected")
        ax[i, 0].plot(q.pid, q.truth_corr, "-", color="C0", label=f"MC truth, + Gd {gd:g} % + light x{keep:g}")
        ax[i, 0].plot(q.pid, q.fit_corr, "s", mfc="none", color="C0", label="MC (x) 6264, data-driven fit, corrected")
        ax[i, 1].plot(q.pid, q.ratio_mc, "o:", color="C0", label=f"hit-corrected MC: mean {q.ratio_mc.mean():.3f}")
        ax[i, 1].plot(q.pid, q.ratio_corr, "o-", color="C0", label=f"corrected MC: mean {q.ratio_corr.mean():.3f}")
        for a in ax[i]:
            a.set_xticks(range(dmc.NPOS))
            a.set_xticklabels([_pos_label(j) for j in range(dmc.NPOS)], rotation=60, fontsize=8)
            a.legend(fontsize=8)
        ax[i, 0].set_ylabel("neutron detection efficiency")
        ax[i, 0].set_title(f"{LABELS.get(m, m)}: data vs MC truth")
        ax[i, 1].set_ylabel("data / MC truth")
        ax[i, 1].set_title("what the corrections recover")
    fig.tight_layout()
    out.save(fig, "efficiency_corrected")
    return t


# ---------------------------------------------------------------------------------------------- efficiency
def _pos_rows(path) -> Optional[pd.DataFrame]:
    if not Path(path).exists():
        return None
    d = pd.read_csv(path)
    d = d[d.level == "position"].copy()
    d["pid"] = dmc.pos_id(d.rename(columns={"x_pos": "sourceX", "y_pos": "sourceY", "z_pos": "sourceZ"}))
    return d[d.pid >= 0]


def _mc_sources(mc: RunContext, cmp: Optional[RunContext], mc_label: str, cmp_label: Optional[str]) -> list:
    out = [(mc, "m", mc_label)]
    if cmp is not None:
        out.append((cmp, "c", cmp_label))
    return out


def efficiency(ctx: RunContext, srcs: list, methods: list, tgate: str) -> pd.DataFrame:
    """Counting / Poisson / data-driven efficiency per position from the `ambe stats foldeff` CSVs that exist:
    data, every MC source x version (truth and fit), the v4 box reference and the 10/5 IC-template fit."""
    ref = (ctx.extra.get("clusterindex") or {}).get("reference_efficiency")
    rows = []
    for m in methods:
        d = _pos_rows(ctx.csv_path(f"efficiency_fit_by_position_fold_data_{tgate}_{m}"))
        if d is None:
            print(f"[datamczero] efficiency {m}: no data fold fit yet, skipped")
            continue
        cols = ["pid", "n_trig", "naive_eff", "poisson_eff", "dd_eff", "dds_eff", "dds_eff_lo", "dds_eff_hi", "dds_s",
                "dds_s_lo", "dds_s_hi", "dds_p", "poisson_p"]
        t = d[cols].add_prefix("data_").rename(columns={"data_pid": "pid"})
        for mc, short, _ in srcs:
            for v, _ in MC_FITS:
                q = _pos_rows(mc.csv_path(f"efficiency_fit_by_position_fold_mc{v}_{tgate}_{m}"))
                if q is not None:
                    pre = f"{short}{v}_"
                    t = t.merge(q[["pid", "truth_eff", "truth_eff_err", "naive_eff", "poisson_eff", "dds_eff", "dds_eff_lo",
                                   "dds_eff_hi", "dds_s", "dds_p"]].add_prefix(pre).rename(columns={f"{pre}pid": "pid"}),
                                on="pid", how="left")
        o = _pos_rows(ctx.csv_path(f"efficiency_fit_by_position_ic_{m}"))
        if o is not None:
            t = t.merge(o[["pid", "dds_eff"]].rename(columns={"dds_eff": "ic1005_dds_eff"}), on="pid", how="left")
        if ref and Path(ref).exists():
            r = _pos_rows(ref)
            t = t.merge(r[["pid", "dds_eff"]].rename(columns={"dds_eff": "v4box_dds_eff"}), on="pid", how="left")
        t["method"] = m
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def page_efficiency(out: Out, t: pd.DataFrame, srcs: list, name: str = "efficiency", bkg: str = "6264"):
    import matplotlib.pyplot as plt
    for m, q in t.groupby("method"):
        q = q.sort_values("pid")
        x = q.pid.to_numpy()
        fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
        a = ax[0]
        a.plot(x, q.data_naive_eff, "o", mfc="none", color="k", label="data, counting")
        a.plot(x, q.data_poisson_eff, "x", color="k", label="data, Poisson")
        a.errorbar(x, q.data_dds_eff, yerr=[q.data_dds_eff - q.data_dds_eff_lo, q.data_dds_eff_hi - q.data_dds_eff],
                   fmt="ko-", ms=4, label=f"data, data-driven ({bkg} template)")
        if "v4box_dds_eff" in q:
            a.plot(x, q.v4box_dds_eff, "D:", color="0.5", ms=4, label="v4 box (ClusterFinder + cuts), data-driven")
        b = ax[1]
        i = 0
        for _, short, nice in srcs:
            for v, lab in MC_FITS:
                pre = f"{short}{v}_"
                if f"{pre}truth_eff" not in q or q[f"{pre}truth_eff"].isna().all():
                    continue
                c = f"C{i}"
                i += 1
                ll = f"{nice}, {lab}" if v != "nominal" else nice
                a.plot(x, q[f"{pre}truth_eff"], "-", color=c, label=f"MC truth, {ll}")
                a.plot(x, q[f"{pre}dds_eff"], "s", mfc="none", color=c, ms=4)
                r = q.data_dds_eff / q[f"{pre}truth_eff"]
                b.plot(x, r, "o-", color=c, label=f"data / MC truth, {ll}: mean {r.mean():.3f}")
                b.plot(x, 1 + 10 * (q[f"{pre}dds_eff"] - q[f"{pre}truth_eff"]), ".:", color=c,
                       label=f"1 + 10 x (MC fit - truth), {ll}")
        for aa in ax:
            aa.set_xticks(range(dmc.NPOS))
            aa.set_xticklabels([_pos_label(j) for j in range(dmc.NPOS)], rotation=60, fontsize=8)
            aa.legend(fontsize=7)
        a.set_ylabel("neutron detection efficiency")
        a.set_title(f"{LABELS.get(m, m)}: efficiency per position; open squares: MC (x) {bkg} fitted like data", fontsize=10)
        b.set_title("data / MC truth, and the MC closure (fit - truth, x10)", fontsize=10)
        two_titles(fig, "Efficiency per source position: data and simulation",
                   f"{LABELS.get(m, m)}: counting, Poisson and data-driven ({bkg} template) efficiency; MC truth = P(k >= 1)")
        out.save(fig, f"{name}_{m}")


def page_scenarios_vs_y(out: Out, t: pd.DataFrame, srcs: list, name: str = "scenarios_vs_y"):
    """Per method: efficiency (top) and data / MC truth (bottom) against height, one panel per port,
    one line per MC source x version present."""
    import matplotlib.pyplot as plt
    from .heatmap import PORT_ORDER
    order = [i for p in PORT_ORDER for i, (n, _) in enumerate(PORTS) if f"Port {n}" == p]
    series = [(f"{short}{v}_", f"{nice}, {lab}" if v != "nominal" else nice)
              for _, short, nice in srcs[::-1] for v, lab in MC_FITS if not v.startswith("hitcorr")]
    for m, q in t.groupby("method"):
        q = q.set_index("pid")
        cols = [(pre, lab) for pre, lab in series if f"{pre}truth_eff" in q and q[f"{pre}truth_eff"].notna().any()]
        fig, ax = plt.subplots(2, len(order), figsize=(20, 8), sharey="row", sharex=True)
        for k, i in enumerate(order):
            pids = [i * len(HEIGHTS) + j for j in range(len(HEIGHTS))]
            r = q.reindex(pids)
            y = np.array(HEIGHTS)
            ax[0, k].errorbar(y, r.data_dds_eff, yerr=[r.data_dds_eff - r.data_dds_eff_lo, r.data_dds_eff_hi - r.data_dds_eff],
                              fmt="ko-", ms=5, label="data, data-driven")
            for c, (pre, lab) in enumerate(cols):
                ax[0, k].plot(y, r[f"{pre}truth_eff"], "o-", color=f"C{c}", ms=4, label=f"MC truth, {lab}")
                ax[1, k].plot(y, r.data_dds_eff / r[f"{pre}truth_eff"], "o-", color=f"C{c}", ms=4, label=lab)
            ax[0, k].set_title(f"Port {PORTS[i][0]}")
            ax[1, k].set_xlabel("y position (cm)")
        ax[0, 0].set_ylabel("efficiency")
        ax[1, 0].set_ylabel("data / MC truth")
        ax[0, 0].legend(fontsize=8, frameon=False)
        ax[1, 0].legend(fontsize=8, frameon=False)
        two_titles(fig, "Efficiency in data and simulation, by source position",
                   f"{LABELS.get(m, m)}: efficiency and data / MC truth by port and height, per MC scenario")
        out.save(fig, f"{name}_{m}")


def _hist(path) -> Optional[pd.DataFrame]:
    d = _pos_rows(path)
    if d is None:
        return None
    h = np.array([[float(v) for v in s.split()] for s in d["hist"]])
    return pd.DataFrame(h / h.sum(1, keepdims=True), columns=[f"p{i}" for i in range(h.shape[1])]).assign(
        pid=d.pid.to_numpy(), n=h.sum(1))


def foldmult(ctx: RunContext, srcs: list, methods: list, tgate: str) -> pd.DataFrame:
    """Clusters per trigger exactly as the fits see them (the `hist` of the foldeff CSVs): data IC triggers and
    each MC source's nominal fold sample (MC neutron clusters + one 6264 trigger)."""
    rows = []
    for m in methods:
        d = _hist(ctx.csv_path(f"efficiency_fit_by_position_fold_data_{tgate}_{m}"))
        if d is None:
            continue
        rows.append(d.assign(method=m, sample="data"))
        for mc, short, nice in srcs:
            for v, lab in MC_FITS[:2]:
                q = _hist(mc.csv_path(f"efficiency_fit_by_position_fold_mc{v}_{tgate}_{m}"))
                if q is not None:
                    rows.append(q.assign(method=m, sample=f"{nice}" if v == "nominal" else f"{nice}, {lab}"))
    df = pd.concat(rows, ignore_index=True)
    nd = df[df["sample"] == "data"].set_index(["method", "pid"]).n
    df["w"] = [nd.get((m, p), 0.0) for m, p in zip(df.method, df.pid)]
    cols = [c for c in df.columns if re.match(r"p\d+$", c)]
    pooled = df.groupby(["method", "sample"]).apply(lambda g: pd.Series(np.average(g[cols], axis=0, weights=g.w), index=cols))
    pooled = pooled.reset_index().assign(pid=-1)
    return pd.concat([df, pooled], ignore_index=True)


def page_foldmult(out: Out, df: pd.DataFrame, name: str = "foldmult", bkg: str = "6264"):
    import matplotlib.pyplot as plt
    cols = sorted([c for c in df.columns if re.match(r"p\d+$", c)], key=lambda c: int(c[1:]))
    n = np.arange(len(cols))
    lab = [str(i) for i in n[:-1]] + [f">={n[-1]}"]
    ms = list(dict.fromkeys(df.method))
    fig, ax = plt.subplots(2, len(ms), figsize=(6 * len(ms), 8), squeeze=False)
    for i, m in enumerate(ms):
        q = df[(df.method == m) & (df.pid == -1)]
        d = q[q["sample"] == "data"].iloc[0]
        for j, (_, r) in enumerate(q[q["sample"] != "data"].iterrows()):
            st = ["-", "--", ":", "-."][j % 4]
            ax[0, i].plot(n, r[cols].astype(float), st, color=f"C{j}", label=f"{r['sample']} (x) {bkg}: P(0) {r.p0:.3f}")
            ax[1, i].plot(n, d[cols].astype(float).to_numpy() / r[cols].astype(float).to_numpy(), "o" + st, color=f"C{j}",
                          label=r["sample"])
        ax[0, i].plot(n, d[cols].astype(float), "ko", label=f"data: P(0) {d.p0:.3f}")
        ax[0, i].set_yscale("log")
        ax[0, i].set_title(f"{LABELS.get(m, m)}: clusters per trigger, all positions", fontsize=10)
        ax[1, i].set_ylabel("data / MC")
        for a in ax[:, i]:
            a.set_xticks(n)
            a.set_xticklabels(lab)
            a.set_xlabel("clusters per trigger")
            a.legend(fontsize=7)
    two_titles(fig, "Neutron candidates per trigger: data and simulation",
               f"All positions, weighted to the data triggers; data (points) vs MC (x) {bkg} (lines); bottom: data / MC")
    out.save(fig, name)
    for m in ms:
        q = df[(df.method == m) & (df.pid >= 0)]
        fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(20, 17), sharex=True, sharey=True)
        for pid in range(dmc.NPOS):
            a = ax.flat[pid]
            g = q[q.pid == pid]
            if g.empty:
                a.set_visible(False)
                continue
            d = g[g["sample"] == "data"]
            tt = []
            for j, (_, r) in enumerate(g[g["sample"] != "data"].iterrows()):
                a.plot(n, r[cols].astype(float), ["-", "--", ":", "-."][j % 4], color=f"C{j}")
                tt.append(f"{r.p0:.3f}")
            if len(d):
                a.plot(n, d.iloc[0][cols].astype(float), "ko", ms=4)
                tt.insert(0, f"{d.iloc[0].p0:.3f}")
            a.set_yscale("log")
            a.set_title(f"{dmc.pos_title(pid)}: P(0) {' / '.join(tt)}", fontsize=8)
        two_titles(fig, "Neutron candidates per trigger, by source position", f"{LABELS.get(m, m)}: clusters per trigger as fitted; data (points), MC (x) {bkg} (lines: "
                   f"{', '.join(dict.fromkeys(q[q['sample'] != 'data']['sample']))})", simple_size=18, detail_size=12)
        out.save(fig, f"{name}_grid_{m}")


# ---------------------------------------------------------------------------------------------- driver
def run_corrected(ctx: RunContext, mc: RunContext, S: dict, methods: list, gd: float, keep: float, tgate: str,
                  kmax: int = 4):
    """Separate output set: plots/datamczero_corrected/, datamczero_corrected.pdf, csv datamczero_corrected_*."""
    from matplotlib.backends.backend_pdf import PdfPages
    bpath = ctx.csv_path("datamczero_budget")
    bud = pd.read_csv(bpath) if bpath.exists() else None
    df = pd.concat([corrected(S[m], mc, m, gd, keep, kmax) for m in methods], ignore_index=True)
    df.to_csv(ctx.csv_path("datamczero_corrected_mult"), index=False)
    with PdfPages(ctx.plot_path("datamczero_corrected", "pdf")) as pdf:
        out = Out(ctx, pdf, "datamczero_corrected")
        page_corrected(out, df, bud, gd, keep, kmax)
        eff = page_corrected_eff(out, ctx, mc, methods, gd, keep, tgate)
    if eff is not None:
        eff.to_csv(ctx.csv_path("datamczero_corrected_efficiency"), index=False)
    print(f"[datamczero] corrected set: {ctx.plot_path('datamczero_corrected', 'pdf')}, csv datamczero_corrected_*")


def run(ctx: RunContext, mc: RunContext, methods: list, hc_tag: str, parts: list, gd: Optional[float],
        keeps: list, edir: Optional[str], dummy: Optional[RunContext], light_keep: Optional[float] = None,
        tgate: str = "none", cmp: Optional[RunContext] = None, mc_label: str = "MC", cmp_label: Optional[str] = None,
        out_name: str = "efficiency", recluster_tag: Optional[str] = None, bkg_positions: Optional[list] = None,
        bkg_s_max: Optional[float] = None, n_jobs: int = 1, from_csv: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    bkg_run = int((ctx.fit_params or {})["background_run"])
    win = tuple(float(v) for v in (mc.extra.get("clusterindex") or {})["window_ns"])
    need_mc = {"budget", "light", "energy", "corrected"} & set(parts)
    if set(parts) & {"cbcut", "activity", "bkgcombine"}:
        if not set(parts) <= {"cbcut", "activity", "bkgcombine"} or out_name == "efficiency":
            raise SystemExit("[datamczero] run cbcut/activity/bkgcombine on their own, with --out-name")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(ctx.plot_path(f"datamczero_{out_name}", "pdf")) as pdf:
            out = Out(ctx, pdf, f"datamczero_{out_name}")
            if from_csv:
                rd = lambda n: pd.read_csv(ctx.csv_path(f"datamczero_{n}"))  # noqa: E731
                if "cbcut" in parts:
                    page_cbcut(out, rd("cbcut_hist"))
                if "bkgcombine" in parts:
                    page_bkgcombine(out, rd("bkgcombine_compat"), rd("bkgcombine_time"), bkg_run)
                if "activity" in parts:
                    page_activity(out, rd("activity"), rd("activity_profile"), methods)
                print(f"[datamczero] redrew plots/datamczero_{out_name}/ from csv datamczero_*")
                return
            if "cbcut" in parts:
                ch, cs = cbcut(ctx, bkg_run)
                ch.to_csv(ctx.csv_path("datamczero_cbcut_hist"), index=False)
                cs.to_csv(ctx.csv_path("datamczero_cbcut"), index=False)
                print(cs.to_string())
                page_cbcut(out, ch)
            if "bkgcombine" in parts:
                if dummy is None or not recluster_tag or not bkg_positions or bkg_s_max is None:
                    raise SystemExit("[datamczero] bkgcombine needs --dummy-randoms, --recluster-tag, --bkg-positions and --bkg-s-max")
                bc, bt, bf = bkgcombine(ctx, dummy, bkg_run, methods, recluster_tag, bkg_positions, bkg_s_max, n_jobs)
                bc.to_csv(ctx.csv_path("datamczero_bkgcombine_compat"), index=False)
                bt.to_csv(ctx.csv_path("datamczero_bkgcombine_time"), index=False)
                bf.to_csv(ctx.csv_path("datamczero_bkgcombine_fits"), index=False)
                print(bc.to_string())
                page_bkgcombine(out, bc, bt, bkg_run)
            if "activity" in parts:
                if dummy is None or not recluster_tag:
                    raise SystemExit("[datamczero] activity needs --dummy-randoms and --recluster-tag")
                ac, ap = activity(ctx, dummy, bkg_run, methods, recluster_tag, 6046)
                ac.to_csv(ctx.csv_path("datamczero_activity"), index=False)
                ap.to_csv(ctx.csv_path("datamczero_activity_profile"), index=False)
                page_activity(out, ac, ap, methods)
        print(f"[datamczero] wrote plots/datamczero_{out_name}/ and csv datamczero_*")
        return
    if set(parts) <= {"efficiency", "foldmult"}:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        srcs = _mc_sources(mc, cmp, mc_label, cmp_label)
        bkg = f"{bkg_run}+6256" if tgate.endswith("_plus6256") else str(bkg_run)
        with PdfPages(ctx.plot_path(f"datamczero_{out_name}", "pdf")) as pdf:
            o = Out(ctx, pdf, f"datamczero_{out_name}")
            if "foldmult" in parts:
                fm = foldmult(ctx, srcs, methods, tgate)
                fm.to_csv(ctx.csv_path(f"datamczero_{out_name}_foldmult"), index=False)
                page_foldmult(o, fm, bkg=bkg)
            if "efficiency" in parts:
                ef = efficiency(ctx, srcs, methods, tgate)
                if len(ef):
                    ef.to_csv(ctx.csv_path(f"datamczero_{out_name}_efficiency"), index=False)
                    page_efficiency(o, ef, srcs, bkg=bkg)
                    page_scenarios_vs_y(o, ef, srcs)
        print(f"[datamczero] wrote plots/datamczero_{out_name}/ and csv datamczero_{out_name}_*")
        return
    S = {}
    if need_mc:
        hcl = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__candidates_hitcorr__{hc_tag}"))
        for m in methods:
            S[m] = dmc.hitcorr_sample(dmc.load(ctx, mc, m, False, bkg_run), hcl, win)
        del hcl
    if "corrected" in parts:
        if gd is None or light_keep is None:
            raise SystemExit("[datamczero] corrected needs --gd and --light-keep")
        run_corrected(ctx, mc, S, methods, gd, light_keep, tgate)
        parts = [x for x in parts if x != "corrected"]
        if not parts:
            return
    with PdfPages(ctx.plot_path("datamczero", "pdf")) as pdf:
        out = Out(ctx, pdf)
        bud = None
        if "budget" in parts:
            bud = pd.concat([budget(S[m], mc, m, gd) for m in methods], ignore_index=True)
            bud.to_csv(ctx.csv_path("datamczero_budget"), index=False)
            page_budget(out, bud, gd)
        if "light" in parts:
            if bud is None:
                raise SystemExit("[datamczero] light needs budget in --parts")
            li = pd.concat([light(S[m], mc, m, keeps, gd) for m in methods], ignore_index=True)
            li.to_csv(ctx.csv_path("datamczero_light"), index=False)
            page_light(out, li, bud, gd)
        if "tag" in parts:
            tg = tag(ctx, methods, bkg_run)
            tg.to_csv(ctx.csv_path("datamczero_tag"), index=False)
            _pooled(tg, True, bkg_run).to_csv(ctx.csv_path("datamczero_tag_pooled"), index=False)
            page_tag(out, tg, bkg_run)
        if "readout" in parts:
            m = "clusterfinder_nms" if "clusterfinder_nms" in methods else methods[0]
            ro, prof = readout(ctx, dummy, m, bkg_run)
            ro.groupby(["sample", "no_cluster"], dropna=False).agg(
                n=("n_hits", "size"), n_hits_med=("n_hits", "median"), n_pmts_med=("n_pmts", "median"),
                n_pmts_p01=("n_pmts", lambda x: float(np.quantile(x, 0.01))),
                n_pmts_p10=("n_pmts", lambda x: float(np.quantile(x, 0.10)))).reset_index().to_csv(
                ctx.csv_path("datamczero_readout"), index=False)
            prof.to_csv(ctx.csv_path("datamczero_readout_profile"), index=False)
            page_readout(out, ro, prof, m)
        if "veto" in parts:
            vt = veto(ctx, mc, bkg_run)
            vt.to_csv(ctx.csv_path("datamczero_veto"), index=False)
            print(vt.to_string())
        if set(parts) & {"efficiency", "foldmult"}:
            raise SystemExit("[datamczero] run efficiency/foldmult on their own (with --out-name)")
        if "energy" in parts:
            if not edir:
                raise SystemExit("[datamczero] energy needs --primary-energy-dir")
            en = []
            for m in methods:
                d, desc = energy(S[m], edir, m)
                en.append(d)
                print(f"[datamczero] energy {m}:\n{desc.T.to_string()}")
            en = pd.concat(en, ignore_index=True)
            en.to_csv(ctx.csv_path("datamczero_energy"), index=False)
            page_energy(out, en)
    print(f"[datamczero] wrote {ctx.plot_path('datamczero', 'pdf')} and csv/datamczero_*.csv")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots datamczero")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--hitcorr-tag", required=True, help="hit-corrected MC cache tag, e.g. ov6256_deaddoc")
    p.add_argument("--parts", required=True, help=f"comma list of {','.join(PARTS)}")
    p.add_argument("--gd", required=True, help="Gd loading [%%] for the reweighted MC, or none")
    p.add_argument("--keep-scan", required=True, help="hit keep fractions for the light part, e.g. 1,0.95,0.9")
    p.add_argument("--primary-energy-dir", help="directory with primary_<label>.txt (energy part)")
    p.add_argument("--dummy-randoms", help="randoms config holding random_hits_6256 (readout part)")
    p.add_argument("--light-keep", type=float, help="corrected part: MC hit keep fraction (data/MC hits per cluster)")
    p.add_argument("--template-gate", default="none", choices=["ic", "none", "none_plus6256"],
                   help="corrected/efficiency/foldmult parts: gate of the foldeff fits to read")
    p.add_argument("--mc-label", help="efficiency/foldmult: name of --mc-config in legends")
    p.add_argument("--compare-mc-config", help="efficiency/foldmult: a second MC whose foldeff fits are drawn too")
    p.add_argument("--compare-mc-label", help="name of --compare-mc-config in legends")
    p.add_argument("--out-name", help="efficiency/foldmult: output set plots/datamczero_<name>, csv datamczero_<name>_*")
    p.add_argument("--recluster-tag", help="activity part: recoclusters tag present in both the data and the dummy runs")
    p.add_argument("--bkg-positions", help="bkgcombine: source positions x:y:z,... to fit with each background template")
    p.add_argument("--bkg-s-max", type=float, help="bkgcombine: upper edge of the template scale grid")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--from-csv", action="store_true",
                   help="cbcut/activity/bkgcombine: redraw the figures from their saved csv, no recomputation")
    a = p.parse_args(list(argv) if argv else [])
    parts = a.parts.split(",")
    if set(parts) - set(PARTS):
        p.error(f"--parts takes {PARTS}")
    if set(parts) & {"efficiency", "foldmult"} and not (a.out_name and a.mc_label):
        p.error("efficiency/foldmult need --out-name and --mc-label")
    if bool(a.compare_mc_config) != bool(a.compare_mc_label):
        p.error("--compare-mc-config and --compare-mc-label go together")
    keeps = sorted({float(v) for v in a.keep_scan.split(",")} | {1.0}, reverse=True)
    run(ctx, load_context(a.mc_config), a.methods.split(","), a.hitcorr_tag, parts,
        None if a.gd == "none" else float(a.gd), keeps, a.primary_energy_dir,
        load_context(a.dummy_randoms) if a.dummy_randoms else None, a.light_keep, a.template_gate,
        load_context(a.compare_mc_config) if a.compare_mc_config else None, a.mc_label or "MC", a.compare_mc_label,
        a.out_name or "efficiency", a.recluster_tag,
        [tuple(float(v) for v in q.split(":")) for q in a.bkg_positions.split(",")] if a.bkg_positions else None,
        a.bkg_s_max, a.jobs, a.from_csv)
