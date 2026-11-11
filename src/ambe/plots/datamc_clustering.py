"""
AmBe Data/MC of the clusterers alone: IC 700-1200 data triggers against MC events.

    ambe plots datamcclust --config <data cfg> --mc-config <mc cfg> --methods a,b,c
        --backup-methods d,e --add-bkg-methods a,b

Reads the `ambe stats clusterindex` tables: data gate ic (no-source run excluded), MC gate mc,
and the no-source run (fit_params.background_run) under gate none as the per-trigger background.
Methods in --add-bkg-methods get that background added to the MC prediction; the others already
carry it through the MC overlay. MC is weighted per position to the data trigger counts.

Per method: cluster charge, hits and charge per hit, charge scale x resolution scan, charge grid,
capture time (summed, grid, tau maps), multiplicity (summed, grid); one per-hit charge page.
Writes <data plots>/datamcclust/*.png, datamcclust_<method>.pdf, datamcclust_{hits,backup}.pdf and
<data csv>/datamcclust_{summary,by_position,capturetime_fits,hit_charge}.csv.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..plotting import two_titles
from .datamc import HEIGHTS, LABELS, PORTS

NPOS = len(PORTS) * len(HEIGHTS)
CHI_EDGES = np.arange(5.0, 65.0 + 2.5, 5.0)
Q_EDGES = np.arange(0.0, 120.0 + 1.25, 2.5)
T_EDGES = np.arange(0.0, 70.0 + 0.5, 1.0)
T_FIT = (2.0, 67.0)
TAU_FIT = (10.0, 67.0)
T_PLOT = np.arange(2.0, 67.0 + 0.5, 1.0)
CHI0_EDGES = np.arange(0.0, 65.0 + 2.5, 5.0)
BKG_COLOR = "0.6"


def pos_id(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), -1, int)
    x, y, z = (df[c].to_numpy(float) for c in ("sourceX", "sourceY", "sourceZ"))
    for i, (_, (px, pz)) in enumerate(PORTS):
        for j, py in enumerate(HEIGHTS):
            out[np.isclose(x, px) & np.isclose(y, py) & np.isclose(z, pz)] = i * len(HEIGHTS) + j
    return out


def pos_title(p: int) -> str:
    i, j = divmod(p, len(HEIGHTS))
    return f"Port {PORTS[i][0]}, Y = {HEIGHTS[j]:+.0f} cm"


def convolve_pmf(a: np.ndarray, b: np.ndarray, kmax: int) -> np.ndarray:
    c = np.convolve(a, b)
    return np.append(c[:kmax], c[kmax:].sum())


def pmf(n: np.ndarray, kmax: int, w: Optional[np.ndarray] = None) -> np.ndarray:
    h = np.bincount(np.minimum(n, kmax), weights=w, minlength=kmax + 1).astype(float)
    return h / h.sum()


def shape_chi2(d: np.ndarray, p: np.ndarray, var: np.ndarray):
    """MC-like prediction p (variance var) scaled to the data entries; chi2 as datamc.chi2."""
    s = d.sum() / p.sum()
    v = d + var * s ** 2
    ok = v > 0
    return float(((d - s * p)[ok] ** 2 / v[ok]).sum()), int(ok.sum()) - 1, s


@dataclass
class Sample:
    method: str
    add_bkg: bool
    d_c: pd.DataFrame
    d_e: pd.DataFrame
    m_c: pd.DataFrame
    m_e: pd.DataFrame
    b_c: pd.DataFrame
    b_e: pd.DataFrame

    def __post_init__(self):
        self.n_d = np.bincount(self.d_e.pid, minlength=NPOS).astype(float)
        self.n_m = np.bincount(self.m_e.pid, minlength=NPOS).astype(float)
        self.n_b = float(len(self.b_e))
        w = np.divide(self.n_d, self.n_m, out=np.zeros(NPOS), where=self.n_m > 0)
        self.m_c["w"] = w[self.m_c.pid]
        self.m_e["w"] = w[self.m_e.pid]

    def parts(self, col: str, edges: np.ndarray, p: Optional[int] = None, mc_vals=None):
        """Expected clusters per bin for the data trigger count: (components, variance)."""
        mc = self.m_c if p is None else self.m_c[self.m_c.pid == p]
        x = mc[col].to_numpy(float) if mc_vals is None else (mc_vals if p is None else mc_vals[self.m_c.pid.to_numpy() == p])
        w = mc.w.to_numpy()
        cap = mc.cap.to_numpy()
        comp = {"MC capture": np.histogram(x[cap], edges, weights=w[cap])[0],
                "MC noise": np.histogram(x[~cap], edges, weights=w[~cap])[0]}
        var = np.histogram(x, edges, weights=w ** 2)[0]
        if self.add_bkg:
            nd = self.n_d.sum() if p is None else self.n_d[p]
            hb = np.histogram(self.b_c[col].to_numpy(float), edges)[0].astype(float)
            comp["no-source"] = hb * nd / self.n_b
            var = var + hb * (nd / self.n_b) ** 2
        return comp, var

    def data(self, col: str, edges: np.ndarray, p: Optional[int] = None):
        d = self.d_c if p is None else self.d_c[self.d_c.pid == p]
        return np.histogram(d[col].to_numpy(float), edges)[0].astype(float)

    def pred_mean(self, col: str, p: Optional[int] = None) -> float:
        mc = self.m_c if p is None else self.m_c[self.m_c.pid == p]
        s, n = (mc[col] * mc.w).sum(), mc.w.sum()
        if self.add_bkg:
            nd = self.n_d.sum() if p is None else self.n_d[p]
            s += self.b_c[col].sum() * nd / self.n_b
            n += len(self.b_c) * nd / self.n_b
        return float(s / n)

    def flat_data(self, p: Optional[int] = None) -> float:
        """Expected no-source clusters per us in the data sample (or position p)."""
        nd = self.n_d.sum() if p is None else self.n_d[p]
        return len(self.b_c) / self.n_b / (T_FIT[1] - T_FIT[0]) * nd

    def flat_mc(self, p: Optional[int] = None) -> float:
        """MC non-capture (overlay noise) clusters per us, unweighted like the MC fits."""
        mc = self.m_c if p is None else self.m_c[self.m_c.pid == p]
        return float((~mc.cap).sum()) / (T_FIT[1] - T_FIT[0])

    def mult(self, kmax: int, p: Optional[int] = None):
        d = self.d_e if p is None else self.d_e[self.d_e.pid == p]
        pd_ = pmf(d.n_accepted_clusters.to_numpy(), kmax)
        pb = pmf(self.b_e.n_accepted_clusters.to_numpy(), kmax)
        ids = range(NPOS) if p is None else [p]
        pm = np.zeros(kmax + 1)
        pc = np.zeros(kmax + 1)
        for q in ids:
            e = self.m_e[self.m_e.pid == q]
            if len(e) == 0 or self.n_d[q] == 0:
                continue
            wq = self.n_d[q] / sum(self.n_d[r] for r in ids)
            pm += wq * pmf(e.n_accepted_clusters.to_numpy(), kmax,
                           e.wgd.to_numpy() if "wgd" in e.columns else None)
            clean = (e.n_noise.to_numpy() == 0)
            pc += wq * np.bincount(np.minimum(e.n_accepted_clusters.to_numpy()[clean], kmax),
                                   minlength=kmax + 1) / len(e)
        return pd_, pm, convolve_pmf(pm, pb, kmax), pc, len(d)


def load(ctx: RunContext, mc: RunContext, method: str, add_bkg: bool, bkg_run: int) -> Sample:
    dp = lambda s: ctx.parquet_path(f"{ctx.run_name}__{s}")          # noqa: E731
    mp = lambda s: mc.parquet_path(f"{mc.run_name}__{s}")             # noqa: E731
    flt = [("method", "==", method)]
    d_c = pd.read_parquet(dp("candidates_ic"), filters=flt)
    d_e = pd.read_parquet(dp(f"eventindex_ic_{method}"))
    d_e = d_e[~d_e.is_cosmic & (d_e.run != bkg_run)].reset_index(drop=True)
    d_c = d_c[d_c.run != bkg_run].reset_index(drop=True)
    b_c = pd.read_parquet(dp("candidates_none"), filters=flt)
    b_c = b_c[b_c.run == bkg_run].reset_index(drop=True)
    b_e = pd.read_parquet(dp(f"eventindex_none_{method}"))
    b_e = b_e[~b_e.is_cosmic & (b_e.run == bkg_run)].reset_index(drop=True)
    m_c = pd.read_parquet(mp("candidates_mc"), filters=flt,
                          columns=["_source_file", "eventNumber", "run", "eventTankTime", "t_med", "pe", "n_hits",
                                   "cap_dom", "sourceX", "sourceY", "sourceZ"])
    m_e = pd.read_parquet(mp(f"eventindex_mc_{method}"),
                          columns=["run", "eventTankTime", "_source_file", "eventNumber", "is_cosmic",
                                   "n_accepted_clusters", "found", "sourceX", "sourceY", "sourceZ"])
    m_e = m_e[~m_e.is_cosmic].reset_index(drop=True)
    m_c["cap"] = m_c.cap_dom.to_numpy() >= 0
    nn = m_c[~m_c.cap].groupby(["run", "eventTankTime"]).size().rename("n_noise").reset_index()
    m_e = m_e.merge(nn, on=["run", "eventTankTime"], how="left")
    m_e["n_noise"] = m_e.n_noise.fillna(0).astype(int)
    for df in (d_c, b_c, m_c):
        df["t_us"] = df.t_med / 1000.0
        df["pe_per_hit"] = df.pe / df.n_hits
    for df in (d_c, d_e, m_c, m_e):
        df["pid"] = pos_id(df)
    for name, df in (("data candidates", d_c), ("data triggers", d_e), ("MC candidates", m_c),
                     ("MC events", m_e)):
        if (df.pid < 0).any():
            raise SystemExit(f"[datamcclust] {method}: {int((df.pid < 0).sum())} {name} off the 5x5 grid")
    print(f"[datamcclust] {method}: data {len(d_e):,} triggers / {len(d_c):,} clusters, "
          f"MC {len(m_e):,} events / {len(m_c):,} clusters, {bkg_run} {len(b_e):,} triggers / "
          f"{len(b_c):,} clusters, background added: {add_bkg}")
    return Sample(method, add_bkg, d_c, d_e, m_c, m_e, b_c, b_e)


class Out:
    def __init__(self, ctx: RunContext, pdf):
        self.dir = ctx.plots_dir / "datamcclust"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pdf = pdf

    def save(self, fig, name: str):
        import matplotlib.pyplot as plt
        fig.savefig(self.dir / f"{name}.png", dpi=200)
        fig.savefig(self.dir / f"{name}.pdf")
        self.pdf.savefig(fig)
        plt.close(fig)


def _colors(method: str):
    from ..plotting import METHOD_COLORS
    c = METHOD_COLORS.get(method, "#4a3aa7")
    return {"MC capture": c, "MC noise": c, "no-source": BKG_COLOR}


LINESTYLE = {"MC capture": "--", "MC noise": ":", "no-source": "-."}


def _stack(ax, edges, comp, scale, method):
    """Total prediction as a solid line, each non-empty component as its own unfilled line."""
    col = _colors(method)
    parts = {k: v * scale for k, v in comp.items() if v.sum() > 0}
    total = sum(parts.values())
    ax.stairs(total, edges, color=col["MC capture"], lw=1.8, baseline=None,
              label="prediction" if len(parts) > 1 else "MC")
    if len(parts) > 1:
        for k, v in parts.items():
            ax.stairs(v, edges, color=col[k], lw=1.2, ls=LINESTYLE[k], baseline=None, label=k)
    return total


def _datapts(ax, edges, d, **kw):
    cen = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(cen, d, yerr=np.sqrt(d), fmt="o", color="k", ms=3, lw=1, label="data, IC", **kw)


def _ratio(ax, edges, d, p, xlabel):
    cen = 0.5 * (edges[1:] + edges[:-1])
    ok = p > 0
    ax.errorbar(cen[ok], d[ok] / p[ok], yerr=np.sqrt(d[ok]) / p[ok], fmt="o", color="k", ms=2.5, lw=0.8)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("data / pred.")


def page_charge(S: Sample, out: Out, tag: str, edges=Q_EDGES, mc_vals=None, title_extra=""):
    import matplotlib.pyplot as plt
    comp, var = S.parts("pe", CHI_EDGES, mc_vals=mc_vals)
    c2, ndf, _ = shape_chi2(S.data("pe", CHI_EDGES), sum(comp.values()), var)
    comp, _ = S.parts("pe", edges, mc_vals=mc_vals)
    d = S.data("pe", edges)
    p = sum(comp.values())
    lo, hi = np.searchsorted(edges, CHI_EDGES[0]), np.searchsorted(edges, CHI_EDGES[-1])
    s = d[lo:hi].sum() / p[lo:hi].sum()
    fig, ax = plt.subplots(2, 3, figsize=(18, 7.5), gridspec_kw=dict(height_ratios=[3, 1]), sharex="col")
    for a, logy in ((ax[0, 0], False), (ax[0, 1], True)):
        _stack(a, edges, comp, s, S.method)
        _datapts(a, edges, d)
        a.set_ylabel(f"clusters / {edges[1] - edges[0]:g} p.e.")
        if logy:
            a.set_yscale("log")
            a.set_ylim(0.5, None)
        a.legend(frameon=False, fontsize=9)
    _ratio(ax[1, 0], edges, d, p * s, "cluster charge [p.e.]")
    _ratio(ax[1, 1], edges, d, p * s, "cluster charge [p.e.]")
    nt = S.n_d.sum()
    a = ax[0, 2]
    _stack(a, edges, {k: v / nt for k, v in comp.items()}, 1.0, S.method)
    a.errorbar(0.5 * (edges[1:] + edges[:-1]), d / nt, yerr=np.sqrt(d) / nt, fmt="o", color="k", ms=3, lw=1)
    a.set_ylabel("clusters per trigger per bin")
    a.set_title(f"per trigger: data {d.sum() / nt:.3f}, prediction {p.sum() / nt:.3f} clusters", fontsize=11)
    _ratio(ax[1, 2], edges, d, p, "cluster charge [p.e.]")
    md, mp = S.d_c.pe.mean(), S.pred_mean("pe") if mc_vals is None else np.nan
    ax[0, 0].set_title(f"shape, normalised in {CHI_EDGES[0]:g}–{CHI_EDGES[-1]:g} p.e.: "
                       f"χ²/ndf = {c2:.0f}/{ndf}", fontsize=11)
    ax[0, 1].set_title(f"mean charge: data {md:.1f}, prediction {mp:.1f} p.e.", fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: cluster charge, data (IC triggers) vs MC"
                 f"{' + no-source run' if S.add_bkg else ''}{title_extra}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, f"{S.method}_{tag}")
    return c2, ndf


def page_hits(S: Sample, out: Out, name: str, title_extra: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    rows = {}
    for a, col, edges, xl in ((ax[0, 0], "n_hits", np.arange(0.5, 60.5, 1.0), "hits per cluster"),
                              (ax[0, 1], "pe_per_hit", np.arange(0.0, 6.05, 0.1), "cluster charge / hits [p.e. per hit]")):
        comp, var = S.parts(col, edges)
        d = S.data(col, edges)
        c2, ndf, s = shape_chi2(d, sum(comp.values()), var)
        _stack(a, edges, comp, s, S.method)
        _datapts(a, edges, d)
        md, mp = S.d_c[col].mean(), S.pred_mean(col)
        rows[col] = (md, mp)
        a.set_xlabel(xl)
        a.set_ylabel("clusters")
        a.set_title(f"mean: data {md:.2f}, prediction {mp:.2f} (ratio {md / mp:.3f});  "
                    f"shape χ²/ndf = {c2:.0f}/{ndf}", fontsize=11)
        a.legend(frameon=False, fontsize=9)
    he, qe = np.arange(0.5, 50.5, 1.0), np.arange(0.0, 100.5, 2.0)
    hd = np.histogram2d(S.d_c.n_hits, S.d_c.pe, [he, qe])[0]
    hm = np.histogram2d(S.m_c.n_hits, S.m_c.pe, [he, qe], weights=S.m_c.w)[0]
    hd, hm = hd / hd.sum(), hm / hm.sum()
    vmax = max(hd.max(), hm.max())
    for a, h, lab in ((ax[1, 0], hd, "data, IC triggers"), (ax[1, 1], hm, "MC (all clusters)")):
        im = a.pcolormesh(he, qe, h.T, norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax), cmap="viridis")
        a.set_xlabel("hits per cluster")
        a.set_ylabel("cluster charge [p.e.]")
        a.set_title(lab, fontsize=11)
        fig.colorbar(im, ax=a, label="fraction of clusters")
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: charge = hits × charge per hit, data vs MC"
                 f"{' + no-source run' if S.add_bkg else ''}{title_extra}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, name)
    return rows


def _heat(ax, vals, title, cmap, vmin=None, vmax=None, fmt="{:.3f}"):
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(HEIGHTS)), [f"{h:+.0f}" for h in HEIGHTS])
    ax.set_yticks(range(len(PORTS)), [f"Port {p}" for p, _ in PORTS])
    ax.set_xlabel("Y [cm]")
    ax.set_title(title, fontsize=11)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                r, g, b, _ = im.cmap(im.norm(vals[i, j]))
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=9,
                        color="k" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "w")
    return im


def page_hits_map(S: Sample, out: Out):
    import matplotlib.pyplot as plt
    rows = []
    for p in range(NPOS):
        d = S.d_c[S.d_c.pid == p]
        rows.append(dict(pid=p, nh_data=d.n_hits.mean(), nh_pred=S.pred_mean("n_hits", p),
                         pph_data=d.pe_per_hit.mean(), pph_pred=S.pred_mean("pe_per_hit", p),
                         pe_data=d.pe.mean(), pe_pred=S.pred_mean("pe", p)))
    r = pd.DataFrame(rows)
    shape = (len(PORTS), len(HEIGHTS))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    for a, k, t in ((ax[0], "nh", "hits per cluster"), (ax[1], "pph", "charge per hit"),
                    (ax[2], "pe", "cluster charge")):
        v = (r[f"{k}_data"] / r[f"{k}_pred"]).to_numpy().reshape(shape)
        im = _heat(a, v, f"{t}: data / prediction (mean)", "coolwarm", 0.7, 1.3)
        fig.colorbar(im, ax=a)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: data/MC ratio of the cluster means per source position",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, f"{S.method}_hits_map")
    return r


def page_scan(S: Sample, out: Out, ks: np.ndarray, sig: np.ndarray, seed: int):
    """Diagnostic only: MC q -> k q + sigma sqrt(k q) N(0,1), chi2 in 5-65 p.e. (as 10/7) and in 0-65 p.e."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(len(S.m_c))
    pe = S.m_c.pe.to_numpy(float)
    grids = {}
    for key, edges in (("5", CHI_EDGES), ("0", CHI0_EDGES)):
        d = S.data("pe", edges)
        g = np.full((len(sig), len(ks)), np.nan)
        for i, sg in enumerate(sig):
            for j, k in enumerate(ks):
                v = np.clip(k * pe + sg * np.sqrt(np.clip(k * pe, 0, None)) * z, 0, None)
                comp, var = S.parts("pe", edges, mc_vals=v)
                c2, ndf, _ = shape_chi2(d, sum(comp.values()), var)
                g[i, j] = c2 / ndf
        grids[key] = g
    jk1 = int(np.argmin(np.abs(ks - 1.0)))
    res = {}
    for key, g in grids.items():
        i0, j0 = np.unravel_index(np.nanargmin(g), g.shape)
        res[key] = dict(nominal=g[0, jk1], scale_only=np.nanmin(g[0]), k_so=ks[int(np.nanargmin(g[0]))],
                        best=g[i0, j0], k=ks[j0], s=sig[i0], i0=i0, j0=j0)
    fig, ax = plt.subplots(1, 3, figsize=(25, 6.5))
    dk, ds = ks[1] - ks[0], sig[1] - sig[0]
    for a, key, lab in ((ax[0], "5", "5–65 p.e."), (ax[1], "0", "0–65 p.e.")):
        r = res[key]
        im = a.pcolormesh(np.append(ks - dk / 2, ks[-1] + dk / 2), np.append(sig - ds / 2, sig[-1] + ds / 2),
                          grids[key], norm=LogNorm(), cmap="viridis_r")
        a.plot(r["k"], r["s"], "w*", ms=14)
        fig.colorbar(im, ax=a, label="charge χ²/ndf")
        a.set_xlabel("MC charge scale k")
        a.set_ylabel("extra resolution σ [√p.e.]")
        a.set_title(f"χ² in {lab}: nominal {r['nominal']:.0f}, scale only {r['scale_only']:.0f} "
                    f"(k = {r['k_so']:.2f}), best {r['best']:.1f} (k = {r['k']:.2f}, σ = {r['s']:.2f})", fontsize=11)
    nom, _ = S.parts("pe", Q_EDGES)
    dq = S.data("pe", Q_EDGES)
    lo, hi = np.searchsorted(Q_EDGES, CHI_EDGES[0]), np.searchsorted(Q_EDGES, CHI_EDGES[-1])
    pn = sum(nom.values())
    ax[2].stairs(pn * dq[lo:hi].sum() / pn[lo:hi].sum(), Q_EDGES, color="0.5", ls="--", baseline=None,
                 label="MC nominal")
    for key, ls in (("5", "-"), ("0", ":")):
        r = res[key]
        v = np.clip(r["k"] * pe + r["s"] * np.sqrt(np.clip(r["k"] * pe, 0, None)) * z, 0, None)
        comp, _ = S.parts("pe", Q_EDGES, mc_vals=v)
        pb = sum(comp.values())
        ax[2].stairs(pb * dq[lo:hi].sum() / pb[lo:hi].sum(), Q_EDGES, color=_colors(S.method)["MC capture"],
                     lw=1.8, ls=ls, baseline=None,
                     label=f"best in {'5' if key == '5' else '0'}–65: × {r['k']:.2f}, σ {r['s']:.2f}")
    _datapts(ax[2], Q_EDGES, dq)
    ax[2].set_xlabel("cluster charge [p.e.]")
    ax[2].set_ylabel(f"clusters / {Q_EDGES[1] - Q_EDGES[0]:g} p.e.")
    ax[2].legend(frameon=False, fontsize=10)
    ax[2].set_title(f"data vs the best smeared MC (normalised in {CHI_EDGES[0]:g}–{CHI_EDGES[-1]:g} p.e.)",
                    fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: diagnostic scan, MC charge → k·q + σ·√(k·q)·N(0,1)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, f"{S.method}_charge_scan")
    r5, r0 = res["5"], res["0"]
    return dict(chi2ndf_nominal=r5["nominal"], chi2ndf_scale_only=r5["scale_only"], k_scale_only=r5["k_so"],
                chi2ndf_best=r5["best"], k_best=r5["k"], sigma_best=r5["s"],
                chi2ndf0_nominal=r0["nominal"], chi2ndf0_scale_only=r0["scale_only"], k0_scale_only=r0["k_so"],
                chi2ndf0_best=r0["best"], k0_best=r0["k"], sigma0_best=r0["s"])


def page_charge_grid(S: Sample, out: Out, name: str, mc_desc: str):
    import matplotlib.pyplot as plt
    edges = np.arange(0.0, 100.0 + 2.5, 5.0)
    fig, ax = plt.subplots(5, 5, figsize=(22, 16))
    rows = []
    for p in range(NPOS):
        a = ax.flat[p]
        comp, var = S.parts("pe", CHI_EDGES, p)
        d = S.data("pe", CHI_EDGES, p)
        if d.sum() == 0:
            a.set_title(f"{pos_title(p)}: no data", fontsize=10)
            continue
        c2, ndf, _ = shape_chi2(d, sum(comp.values()), var)
        comp, _ = S.parts("pe", edges, p)
        dq = S.data("pe", edges, p)
        pp = sum(comp.values())
        lo, hi = np.searchsorted(edges, CHI_EDGES[0]), np.searchsorted(edges, CHI_EDGES[-1])
        s = dq[lo:hi].sum() / pp[lo:hi].sum()
        _stack(a, edges, comp, s, S.method)
        _datapts(a, edges, dq)
        fb = comp.get("no-source", np.zeros(1)).sum() / pp.sum()
        a.set_title(f"{pos_title(p)}  χ²/ndf = {c2:.0f}/{ndf}", fontsize=10)
        a.set_xlabel("cluster charge [p.e.]")
        a.set_ylabel("clusters")
        rows.append(dict(pid=p, chi2_charge=c2, ndf_charge=ndf, bkg_share=fb))
    r = pd.DataFrame(rows)
    share = (f"; no-source share {r.bkg_share.min():.0%}–{r.bkg_share.max():.0%}"
             if S.add_bkg and len(r) else "")
    h, lab = ax.flat[int(r.pid.iloc[0])].get_legend_handles_labels() if len(r) else ([], [])
    fig.legend(h, lab, loc="upper right", ncol=len(lab), frameon=False, fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: cluster charge per source position, {mc_desc}, "
                 f"shape-normalised in {CHI_EDGES[0]:g}–{CHI_EDGES[-1]:g} p.e.{share}", fontsize=15, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, name)
    return r


def _tfit(t_us: np.ndarray, flat_per_us: float) -> dict:
    """A exp(-t/tau) + B on TAU_FIT (after the thermalisation rise), 1 us bins, B fixed to the sample's
    measured non-neutron cluster rate. tau is an effective slope: Gd + H captures are not one exponential."""
    from scipy.optimize import curve_fit
    t_us = np.asarray(t_us, float)
    nan = dict(tau=np.nan, tau_err=np.nan, B=flat_per_us, redchi=np.nan, N=len(t_us), flat_frac=np.nan)
    y, e = np.histogram(t_us, T_EDGES)
    x = 0.5 * (e[1:] + e[:-1])
    w = (x > TAU_FIT[0]) & (x < TAU_FIT[1])
    x, y = x[w], y[w].astype(float)
    if y.sum() < 100:
        return nan
    sd = np.sqrt(np.maximum(y, 1.0))
    f = lambda t, A, tau: A * np.exp(-t / tau) + flat_per_us        # noqa: E731
    try:
        p, c = curve_fit(f, x, y, p0=[max(y[0] - flat_per_us, 1.0) * np.exp(x[0] / 30.0), 30.0], sigma=sd,
                         absolute_sigma=True, bounds=([0, 5.0], [np.inf, 300.0]), maxfev=100000)
    except RuntimeError:
        return nan
    chi = float((((y - f(x, *p)) / sd) ** 2).sum())
    return dict(tau=float(p[1]), tau_err=float(np.sqrt(c[1, 1])), B=flat_per_us, redchi=chi / (len(x) - 2),
                N=int(y.sum()), flat_frac=flat_per_us * len(x) / y.sum())


NFIT_NAN = dict(A=np.nan, tau=np.nan, tau_err=np.nan, therm=np.nan, therm_err=np.nan, B=np.nan, redchi=np.nan,
                N=0, flat_frac=np.nan)


def _nfit(t_us: np.ndarray, b_per_us: float, w: Optional[np.ndarray] = None, seed: int = 0) -> dict:
    """basic.fit NeutCapture A(1-exp(-t/therm))exp(-t/tau) + B on 2-67 us (1 us bins, lmfit least_squares), B fixed
    to b_per_us. A weighted sample is unweighted first by Poisson replication (each entry Poisson(w / <w>) times)."""
    import contextlib
    import io
    from .basic import AmBeNeutronAnalyzer, fit
    t = np.asarray(t_us, float)
    if w is not None:
        w = np.asarray(w, float)
        t = np.repeat(t, np.random.default_rng(seed).poisson(w / w.mean()))
    an = AmBeNeutronAnalyzer()
    h = np.histogram(t, an.fitting_config["time_bins"], an.fitting_config["time_range"])[0]
    with contextlib.redirect_stdout(io.StringIO()):
        an.update_fitting_config(initial_background=float(b_per_us), background_bounds=(0.0, np.inf),
                                 initial_amplitude=float(max(h.max(), 1) * 2.0))
        r = fit(t, float_B=False, backend="lmfit", analyzer=an, min_counts=100)
    if r is None:
        return dict(NFIT_NAN, N=int(len(t)))
    nb = int(T_FIT[1] - T_FIT[0])
    return dict(A=r["A"], tau=r["tau"], tau_err=r["tau_err"], therm=r["therm"], therm_err=r["therm_err"],
                B=r["B"], redchi=r["redchi"], N=r["N"], flat_frac=r["B"] * nb / max(r["N"], 1))


def _curve(f: dict, edges: np.ndarray) -> tuple:
    from .basic import NeutCapture
    x = np.linspace(edges[0], edges[-1], 400)
    return x, NeutCapture(x, f["A"], f["therm"], f["tau"], f["B"]) * (edges[1] - edges[0])


def _flat_bkg(S: Sample, comp: dict, p: Optional[int], edges: np.ndarray) -> dict:
    """The no-source term in time is flat: its measured rate per us instead of the 6,004-trigger histogram."""
    if "no-source" in comp:
        comp = dict(comp)
        comp["no-source"] = np.full(len(edges) - 1, S.flat_data(p) * (edges[1] - edges[0]))
    return comp


def _ftxt(f: dict) -> str:
    return f"τ {f['tau']:.2f} ± {f['tau_err']:.2f}, therm {f['therm']:.2f} µs"


def page_time(S: Sample, out: Out, truth_t: pd.DataFrame, name: str = None, wgd_c=None, wgd_t=None, gd_label=""):
    import matplotlib.pyplot as plt
    E = T_PLOT
    m_c = S.m_c if wgd_c is None else S.m_c.assign(w=S.m_c.w * wgd_c)
    Sx = S if wgd_c is None else _with(S, m_c=m_c)
    comp = _flat_bkg(Sx, Sx.parts("t_us", E)[0], None, E)
    var = Sx.parts("t_us", E)[1]
    d = S.data("t_us", E)
    p = sum(comp.values())
    s = d.sum() / p.sum()
    fd = _nfit(S.d_c.t_us.to_numpy(), S.flat_data())
    fm = _nfit(S.m_c.t_us.to_numpy(), S.flat_mc(), w=wgd_c)
    ft = _nfit(truth_t.cap_us.to_numpy(), 0.0, w=wgd_t)
    e10 = {k: _tfit(x, b) for k, x, b in (("data", S.d_c.t_us.to_numpy(), S.flat_data()),
                                         ("mc_reco", S.m_c.t_us.to_numpy(), S.flat_mc()),
                                         ("mc_true", truth_t.cap_us.to_numpy(), 0.0))} if wgd_c is None else {}
    col = _colors(S.method)["MC capture"]
    fig, ax = plt.subplots(2, 3, figsize=(25, 8.5), gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
    cen = 0.5 * (E[1:] + E[:-1])
    sd = np.sqrt(d + var * s ** 2)
    ok = sd > 0
    for j, logy in ((0, False), (1, True)):
        a = ax[0, j]
        _stack(a, E, comp, s, S.method)
        _datapts(a, E, d)
        a.plot(*_curve(fd, E), color="k", lw=1.0, label="data fit")
        a.set_ylabel("clusters / µs")
        if logy:
            a.set_yscale("log")
        a.legend(frameon=False, fontsize=9)
        ax[1, j].errorbar(cen[ok], ((d - s * p) / np.where(ok, sd, 1))[ok], yerr=1, fmt="o", color="k", ms=2.5,
                          lw=0.8)
        ax[1, j].set_ylabel("(data − pred.) / σ")
        ax[1, j].set_xlabel("cluster time t_med [µs]")
    ax[0, 0].set_title(f"data {_ftxt(fd)} (flat {fd['flat_frac']:.1%})", fontsize=11)
    ax[0, 1].set_title(f"MC reco {_ftxt(fm)} (flat {fm['flat_frac']:.1%})", fontsize=11)
    wt = truth_t.w.to_numpy() * (1.0 if wgd_t is None else wgd_t)
    ht = np.histogram(truth_t.cap_us, E, weights=wt)[0]
    hr = Sx.parts("t_us", E)[0]["MC capture"]
    a = ax[0, 2]
    a.stairs(ht / ht.sum(), E, color="k", lw=1.5, baseline=None, label="MC true capture time")
    a.stairs(hr / hr.sum(), E, color=col, lw=1.5, baseline=None, label="MC capture-owned cluster t_med")
    a.set_yscale("log")
    a.set_ylabel("fraction / µs")
    a.legend(frameon=False, fontsize=9)
    a.set_title(f"MC truth {_ftxt(ft)}", fontsize=11)
    ok2 = ht > 0
    ax[1, 2].errorbar(cen[ok2], (hr / hr.sum())[ok2] / (ht / ht.sum())[ok2], fmt="o", color="k", ms=2.5)
    ax[1, 2].set_ylim(0.5, 1.5)
    ax[1, 2].set_ylabel("reco / true")
    ax[1, 2].set_xlabel("time [µs]")
    for a in ax.flat:
        a.set_xlim(E[0], E[-1])
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: capture time {E[0]:g}–{E[-1]:g} µs, data (IC triggers) vs MC"
                 f"{gd_label}{' + no-source run (flat)' if S.add_bkg else ''}; NeutCapture fit, B fixed to the "
                 f"measured non-neutron rate", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.save(fig, name or f"{S.method}_time")
    res = {"data": fd, "mc_reco": fm, "mc_true": ft}
    for k, f in e10.items():
        res[k] = dict(res[k], **{f"expflat10_{q}": f[q] for q in ("tau", "tau_err", "redchi", "flat_frac")})
    return res


def page_time_grid(S: Sample, out: Out, truth_t: pd.DataFrame):
    import matplotlib.pyplot as plt
    E = T_PLOT
    fig, ax = plt.subplots(5, 5, figsize=(22, 16), sharex=True)
    rows = []
    for p in range(NPOS):
        a = ax.flat[p]
        d = S.data("t_us", E, p)
        if d.sum() == 0:
            continue
        comp = _flat_bkg(S, S.parts("t_us", E, p)[0], p, E)
        pp = sum(comp.values())
        _stack(a, E, comp, d.sum() / pp.sum(), S.method)
        _datapts(a, E, d)
        fd = _nfit(S.d_c.t_us[S.d_c.pid == p].to_numpy(), S.flat_data(p))
        fm = _nfit(S.m_c.t_us[S.m_c.pid == p].to_numpy(), S.flat_mc(p))
        ft = _nfit(truth_t.cap_us[truth_t.pid == p].to_numpy(), 0.0)
        if np.isfinite(fd["tau"]):
            a.plot(*_curve(fd, E), color="k", lw=0.8)
        a.set_xlim(E[0], E[-1])
        a.set_title(f"{pos_title(p)}  τ data {fd['tau']:.1f}, MC {fm['tau']:.1f} µs", fontsize=10)
        a.set_xlabel("t_med [µs]")
        rows.append(dict(pid=p, **{f"tau_{k}_{q}": f[q] for k, f in (("data", fd), ("mc", fm), ("true", ft))
                                   for q in ("tau", "tau_err", "therm", "therm_err", "flat_frac", "redchi", "N")}))
    first = int(rows[0]["pid"]) if rows else 0
    h, lab = ax.flat[first].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=len(lab), frameon=False, fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: capture time per source position, data vs MC"
                 f"{' + no-source run (flat)' if S.add_bkg else ''}, normalised in {E[0]:g}–{E[-1]:g} µs; "
                 f"black line: data NeutCapture fit", fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, f"{S.method}_time_grid")
    r = pd.DataFrame(rows)
    shape = (len(PORTS), len(HEIGHTS))
    full = r.set_index("pid").reindex(range(NPOS))
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.6))
    lo = np.nanmin(full[["tau_true_tau", "tau_mc_tau", "tau_data_tau"]].to_numpy())
    hi = np.nanmax(full[["tau_true_tau", "tau_mc_tau", "tau_data_tau"]].to_numpy())
    for a, k, t in ((ax[0], "data", "data τ [µs]"), (ax[1], "mc", "MC reconstructed τ [µs]"),
                    (ax[2], "true", "MC true τ [µs]")):
        im = _heat(a, full[f"tau_{k}_tau"].to_numpy().reshape(shape), t, "viridis", lo, hi, "{:.1f}")
        fig.colorbar(im, ax=a)
    pull = ((full.tau_data_tau - full.tau_mc_tau)
            / np.hypot(full.tau_data_tau_err, full.tau_mc_tau_err)).to_numpy().reshape(shape)
    im = _heat(ax[3], pull, "(data − MC reco) / σ", "magma", None, None, "{:.0f}")
    fig.colorbar(im, ax=ax[3])
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: capture time τ per source position (NeutCapture, "
                 f"{T_FIT[0]:g}–{T_FIT[1]:g} µs)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.save(fig, f"{S.method}_tau_map")
    return r


def _with(S: Sample, **kw) -> Sample:
    """Copy of S with replaced tables; keeps S's position weights (no __post_init__)."""
    import copy
    T = copy.copy(S)
    for k, v in kw.items():
        setattr(T, k, v)
    return T


def page_mult(S: Sample, out: Out, kmax: int):
    import matplotlib.pyplot as plt
    pdat, pm, pmb, pclean, nd = S.mult(kmax)
    col = _colors(S.method)
    x = np.arange(kmax + 1)
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    for a, logy in ((ax[0], True), (ax[1], False)):
        a.stairs(pm, np.arange(kmax + 2) - 0.5, color=col["MC capture"], lw=1.8, label="MC")
        if S.add_bkg:
            a.stairs(pmb, np.arange(kmax + 2) - 0.5, color="k", lw=1.5, ls="--", label="MC ⊗ no-source run")
        a.errorbar(x, pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=5, label="data, IC")
        a.set_xticks(x, [str(k) for k in range(kmax)] + [f"≥{kmax}"])
        a.set_xlabel("candidate clusters per trigger (2–67 µs)")
        a.set_ylabel("fraction of triggers")
        if logy:
            a.set_yscale("log")
        a.legend(frameon=False, fontsize=9)
    ge = lambda q, k: q[k:].sum()                                    # noqa: E731
    t2 = (f"≥2: data {ge(pdat, 2):.4f}, MC {ge(pm, 2):.4f}"
          + (f", MC ⊗ bkg {ge(pmb, 2):.4f}" if S.add_bkg else ""))
    t1 = (f"≥1: data {ge(pdat, 1):.3f}, MC {ge(pm, 1):.3f}"
          + (f", MC ⊗ bkg {ge(pmb, 1):.3f}" if S.add_bkg else ""))
    ax[0].set_title(t2, fontsize=11)
    ax[1].set_title(t1, fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: candidates per trigger (data IC trigger / MC event)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, f"{S.method}_mult")
    return dict(ge1_data=ge(pdat, 1), ge1_mc=ge(pm, 1), ge1_mcbkg=ge(pmb, 1) if S.add_bkg else np.nan,
                ge2_data=ge(pdat, 2), ge2_mc=ge(pm, 2), ge2_mcbkg=ge(pmb, 2) if S.add_bkg else np.nan,
                mean_data=float(S.d_e.n_accepted_clusters.mean()),
                mean_mc=float(np.average(S.m_e.n_accepted_clusters, weights=S.m_e.w)))


def page_mult_grid(S: Sample, out: Out, kmax: int):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 5, figsize=(22, 16))
    col = _colors(S.method)
    rows = []
    for p in range(NPOS):
        a = ax.flat[p]
        pdat, pm, pmb, _, nd = S.mult(kmax, p)
        if nd == 0:
            continue
        e = np.arange(kmax + 2) - 0.5
        a.stairs(pm, e, color=col["MC capture"], lw=1.6)
        if S.add_bkg:
            a.stairs(pmb, e, color="k", lw=1.2, ls="--")
        a.errorbar(np.arange(kmax + 1), pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=4)
        a.set_yscale("log")
        a.set_xticks(range(kmax + 1), [str(k) for k in range(kmax)] + [f"≥{kmax}"])
        a.set_title(f"{pos_title(p)}  ≥1: {1 - pdat[0]:.3f} / {1 - pm[0]:.3f}"
                    + (f" / {1 - pmb[0]:.3f}" if S.add_bkg else ""), fontsize=10)
        rows.append(dict(pid=p, n_trig_data=nd, ge1_data=1 - pdat[0], ge1_mc=1 - pm[0],
                         ge1_mcbkg=1 - pmb[0] if S.add_bkg else np.nan,
                         ge2_data=pdat[2:].sum(), ge2_mc=pm[2:].sum(),
                         ge2_mcbkg=pmb[2:].sum() if S.add_bkg else np.nan))
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: candidates per trigger per position; data (points), MC "
                 f"(line){', MC ⊗ no-source run (dashed)' if S.add_bkg else ''}; titles ≥1 data / MC"
                 f"{' / MC ⊗ bkg' if S.add_bkg else ''}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.save(fig, f"{S.method}_mult_grid")
    return pd.DataFrame(rows)


def _mult_pred(S: Sample, kmax: int, p: Optional[int] = None):
    pdat, pm, pmb, _, nd = S.mult(kmax, p)
    return pdat, (pmb if S.add_bkg else pm), nd


def page_mult_versions(S: Sample, Sh: Sample, out: Out, kmax: int) -> dict:
    """Candidates per trigger: data against the nominal and the hit-corrected MC prediction."""
    import matplotlib.pyplot as plt
    pdat, p0, nd = _mult_pred(S, kmax)
    _, p1, _ = _mult_pred(Sh, kmax)
    col = _colors(S.method)["MC capture"]
    pred = "MC ⊗ no-source run" if S.add_bkg else "MC"
    x = np.arange(kmax + 1)
    e = np.arange(kmax + 2) - 0.5
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    for a, logy in ((ax[0], True), (ax[1], False)):
        a.stairs(p0, e, color=col, lw=1.5, ls="--", label=f"{pred}, nominal")
        a.stairs(p1, e, color=col, lw=1.8, label=f"{pred}, hit-corrected")
        a.errorbar(x, pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=5, label="data, IC")
        a.set_xticks(x, [str(k) for k in range(kmax)] + [f"≥{kmax}"])
        a.set_xlabel("candidate clusters per trigger (2–67 µs)")
        a.set_ylabel("fraction of triggers")
        if logy:
            a.set_yscale("log")
        a.legend(frameon=False, fontsize=9)
    ax[0].set_title(f"≥2: data {pdat[2:].sum():.4f}, nominal {p0[2:].sum():.4f}, hit-corrected {p1[2:].sum():.4f}",
                    fontsize=11)
    ax[1].set_title(f"0: data {pdat[0]:.3f}, nominal {p0[0]:.3f}, hit-corrected {p1[0]:.3f}", fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: candidates per trigger, nominal and hit-corrected MC",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, f"{S.method}_mult_versions")
    return dict(m0_pred_nominal=p0[0], m0_pred_hitcorr=p1[0], ge2_pred_hitcorr=p1[2:].sum())


def page_mult_grid_versions(S: Sample, Sh: Sample, out: Out, kmax: int) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 5, figsize=(22, 16), sharey=True)
    col = _colors(S.method)["MC capture"]
    e = np.arange(kmax + 2) - 0.5
    rows = []
    for p in range(NPOS):
        a = ax.flat[p]
        pdat, p0, nd = _mult_pred(S, kmax, p)
        if nd == 0:
            a.axis("off")
            continue
        _, p1, _ = _mult_pred(Sh, kmax, p)
        a.stairs(p0, e, color=col, lw=1.2, ls="--", label="nominal MC")
        a.stairs(p1, e, color=col, lw=1.6, label="hit-corrected MC")
        a.errorbar(np.arange(kmax + 1), pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=4,
                   label="data, IC")
        a.set_yscale("log")
        a.set_xticks(range(kmax + 1), [str(k) for k in range(kmax)] + [f"≥{kmax}"])
        a.set_title(f"{pos_title(p)}  0: {pdat[0]:.3f} / {p0[0]:.3f} / {p1[0]:.3f}", fontsize=10)
        rows.append(dict(pid=p, m0_pred_nominal=p0[0], m0_pred_hitcorr=p1[0]))
    h, lab = ax.flat[rows[0]["pid"]].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: candidates per trigger per position"
                 f"{' (MC ⊗ no-source run)' if S.add_bkg else ''}; titles: empty-trigger fraction data / nominal / "
                 f"hit-corrected", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, f"{S.method}_mult_grid_versions")
    return pd.DataFrame(rows)


def truth_times(mc: RunContext, tag: str, m_e: pd.DataFrame) -> pd.DataFrame:
    cp = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__capture_truth_captures__{tag}"),
                         columns=["_source_file", "eventNumber", "cap_t"])
    cp = cp.merge(m_e[["_source_file", "eventNumber", "pid", "w"]], on=["_source_file", "eventNumber"])
    return cp.assign(cap_us=cp.cap_t / 1000.0)


def page_rate(ctx: RunContext, mc: RunContext, out: Out, chi_rows: pd.DataFrame):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import spearmanr
    from ..data.processor import AmBeNeutronProcessing
    from ..data.randoms import run_number
    from ..plotting import METHOD_COLORS
    from .clustereff import _rate
    pos = AmBeNeutronProcessing().source_positions
    hits = {}
    for rf in ctx.inputs["root_files"]:
        run = run_number(Path(rf))
        if run not in pos:
            continue
        base = f"{ctx.run_name}__random_%s_{run}"
        n = len(pd.read_parquet(ctx.parquet_path(base % "triggers"), columns=["trigger"]))
        df = pd.DataFrame([dict(zip(("sourceX", "sourceY", "sourceZ"), map(float, pos[run])))])
        hits.setdefault(int(pos_id(df)[0]), []).append((_rate(ctx.parquet_path(base % "hits"), n), n))
    rate = {p: np.average([r for r, _ in v], weights=[n for _, n in v]) for p, v in hits.items()}
    spec = (mc.extra.get("clusterindex") or {}).get("overlay_veto") or {}
    ov = _rate(spec["random_hits"], len(pd.read_parquet(spec["random_hits"].replace("random_hits", "random_triggers"),
                                                        columns=["trigger"])))
    c = chi_rows.assign(rate=chi_rows.pid.map(rate), r=lambda d: d.chi2_charge / d.ndf_charge)
    fig, ax = plt.subplots(figsize=(10, 6))
    rho = {}
    for m, g in c.groupby("method", sort=False):
        rho[m] = spearmanr(g.rate, g.r).correlation
        ax.plot(g.rate, g.r, "o", ms=6, color=METHOD_COLORS.get(m), label=f"{LABELS.get(m, m)} (ρ = {rho[m]:+.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("data run noise level: hits per PMT per µs in the 70 µs window [kHz/PMT]")
    ax.set_ylabel("cluster charge χ²/ndf, data vs MC (per position)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Is the charge mismatch caused by the overlay ({ov:.1f} kHz/PMT) being quieter than the "
                 f"source runs? Spearman ρ per method", fontsize=11)
    fig.tight_layout()
    out.save(fig, "backup_chi2_vs_rate")
    return c[["method", "pid", "rate"]], rho, ov


def page_hit_charge(ctx: RunContext, mc: RunContext, out: Out, runs: list, method: str) -> dict:
    """Per-hit charge in the delayed window: data hits within 40 ns of a candidate's t_med, data hits
    more than 500 ns from any candidate (dark-noise reference), MC capture-light pulses."""
    import re
    import matplotlib.pyplot as plt
    from ..data.processor import AmBeNeutronProcessing, mc_label_position
    pos = AmBeNeutronProcessing().source_positions
    lo, hi = (T_FIT[0] * 1000, T_FIT[1] * 1000)
    cand = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_ic"),
                           filters=[("method", "in", ["clusterfinder", "peak", "optics"])],
                           columns=["run", "trigger", "method", "t_med"])
    inc, iso, xyz = [], [], set()
    for run in runs:
        ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_{method}"),
                             filters=[("run", "==", run)], columns=["trigger", "is_cosmic"])
        keep = set(ei.trigger[~ei.is_cosmic])
        h = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_hits_{run}"),
                            columns=["trigger", "pmtID", "t", "pe"])
        h = h[h.trigger.isin(keep) & (h.t >= lo) & (h.t <= hi)].sort_values("t")
        c = cand[cand.run == run]
        for m, tol, dst in ((method, 40.0, inc), (None, 500.0, iso)):
            cc = (c if m is None else c[c.method == m])[["trigger", "t_med"]].sort_values("t_med")
            j = pd.merge_asof(h, cc, left_on="t", right_on="t_med", by="trigger", direction="nearest")
            near = (j.t - j.t_med).abs() <= tol
            dst.append(j[near] if m is not None else j[~near])
        xyz.add(tuple(float(v) for v in pos[run]))
    inc, iso = pd.concat(inc), pd.concat(iso)
    rx = re.compile(r"ANNIEEvent_MC_AmBe_(?P<lab>.+QE_[\d.]+)_\d+_\d+\.root$")
    pf = mc.parquet_path(f"{mc.run_name}__pulses")
    files = pd.read_parquet(pf, columns=["_source_file"])._source_file.unique()
    sel = [f for f in files if (mm := rx.search(str(f))) and
           tuple(float(v) for v in mc_label_position(mm.group("lab"))) in xyz]
    if not sel:
        raise SystemExit(f"[datamcclust] no MC pulses at the positions of runs {runs}")
    mp = pd.read_parquet(pf, filters=[("_source_file", "in", sel)],
                         columns=["pmtID", "t", "pe", "is_neutron"])
    mp = mp[(mp.t >= lo) & (mp.t <= hi) & (mp.is_neutron == 1)]
    edges = np.arange(-0.5, 8.0 + 0.05, 0.1)
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    col = _colors(method)["MC capture"]
    for x, lab, c in ((inc.pe, "data, hits in candidates", "k"), (iso.pe, "data, isolated hits (dark noise)", "0.55"),
                      (mp.pe, "MC, capture-light hits", col)):
        hh = np.histogram(x, edges)[0]
        ax[0].stairs(hh / hh.sum(), edges, color=c, lw=1.6, label=f"{lab}: median {np.median(x):.2f} p.e.")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("hit charge [p.e.]")
    ax[0].set_ylabel("fraction of hits / 0.1 p.e.")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("per-hit charge, delayed window", fontsize=11)
    sd = inc.pmtID.value_counts(normalize=True)
    sm = mp.pmtID.value_counts(normalize=True)
    ids = sorted(set(sd.index) | set(sm.index))
    sd, sm = sd.reindex(ids, fill_value=0), sm.reindex(ids, fill_value=0)
    mc_only = [int(i) for i in ids if sd[i] == 0 and sm[i] > 0]
    ax[1].plot(sm.values, sd.values, "o", ms=4, color=col)
    ax[1].plot(sm[mc_only].values, sd[mc_only].values, "o", ms=7, color="#c0392b")
    ax[1].set_xlabel("MC: share of capture hits on the PMT")
    ax[1].set_ylabel("data: share of candidate hits on the PMT")
    ax[1].set_title(f"PMTs with MC hits but none in data (red): {', '.join(map(str, mc_only))}, "
                    f"{sm[mc_only].sum():.1%} of MC hits", fontsize=11)
    fig.suptitle(f"Hit level, runs {', '.join(map(str, runs))} vs MC at the same positions: data hits carry more "
                 f"charge, MC has more hits", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, "hit_charge")
    return dict(hit_runs=" ".join(map(str, runs)), median_pe_hit_data_cand=float(inc.pe.median()),
                median_pe_hit_data_iso=float(iso.pe.median()), median_pe_hit_mc=float(mp.pe.median()),
                mean_pe_hit_data_cand=float(inc.pe.mean()), mean_pe_hit_mc=float(mp.pe.mean()),
                n_pmt_data=int((sd > 0).sum()), n_pmt_mc=int((sm > 0).sum()),
                mc_only_pmts=" ".join(map(str, mc_only)), mc_only_hit_share=float(sm[mc_only].sum()))


def page_large_pulses(ctx: RunContext, out: Out, runs: list, method: str, q_big: float) -> pd.DataFrame:
    """Per-PMT share of delayed-window data hits above q_big p.e. (non-vetoed IC triggers of `runs`)."""
    import matplotlib.pyplot as plt
    lo, hi = (T_FIT[0] * 1000, T_FIT[1] * 1000)
    hs = []
    for run in runs:
        ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_{method}"),
                             filters=[("run", "==", run)], columns=["trigger", "is_cosmic"])
        h = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_hits_{run}"), columns=["trigger", "pmtID", "t", "pe"])
        hs.append(h[h.trigger.isin(set(ei.trigger[~ei.is_cosmic])) & (h.t >= lo) & (h.t <= hi)].assign(run=run))
    h = pd.concat(hs, ignore_index=True)
    g = h.groupby("pmtID").pe.agg(n="size", median="median", fbig=lambda x: float((x > q_big).mean()))
    g = g[g.n >= 500].sort_values("fbig", ascending=False)
    top = list(g.index[:5])
    fig, ax = plt.subplots(1, 3, figsize=(24, 6.5))
    ax[0].plot(np.arange(len(g)), g.fbig, "o", ms=4, color="k")
    ax[0].set_xlabel("PMTs, ranked by large-pulse share")
    ax[0].set_ylabel(f"fraction of hits above {q_big:g} p.e.")
    ax[0].set_yscale("log")
    ax[0].set_title(f"all data hits: {float((h.pe > q_big).mean()):.1%} above {q_big:g} p.e.; top PMTs "
                    f"{', '.join(str(int(i)) for i in top)}", fontsize=11)
    ax[1].plot(g["median"], g.fbig, "o", ms=4, color="k")
    ax[1].set_xlabel("PMT median hit charge [p.e.]")
    ax[1].set_ylabel(f"fraction of hits above {q_big:g} p.e.")
    ax[1].set_yscale("log")
    ax[1].set_title(f"median of PMT medians {g['median'].median():.2f} p.e.; "
                    f"{int((g['median'] > 2).sum())} PMTs above 2 p.e.", fontsize=11)
    edges = np.arange(-1.0, 60.0 + 0.5, 0.5)
    for pm in top[:3]:
        x = h.pe[h.pmtID == pm]
        hh = np.histogram(x, edges)[0]
        ax[2].stairs(hh / len(x), edges, baseline=None, lw=1.4, label=f"PMT {int(pm)}")
    x = h.pe[~h.pmtID.isin(top)]
    hh = np.histogram(x, edges)[0]
    ax[2].stairs(hh / len(x), edges, baseline=None, lw=1.6, color="k", label="all other PMTs")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("hit charge [p.e.]")
    ax[2].set_ylabel("fraction of hits / 0.5 p.e.")
    ax[2].legend(frameon=False, fontsize=10)
    ax[2].set_title("hit charge on the worst PMTs", fontsize=11)
    fig.suptitle(f"Data large pulses, runs {', '.join(map(str, runs))}, delayed window", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, "large_pulses")
    byrun = h[h.pmtID.isin(top)].groupby(["pmtID", "run"]).pe.apply(lambda x: float((x > q_big).mean()))
    return g.reset_index().merge(byrun.unstack().add_prefix("fbig_run").reset_index(), on="pmtID", how="left")


def spe_map(q_mc: np.ndarray, q_ref: np.ndarray, lo: float, hi: float, n: int = 2001):
    """Quantile map q -> F_ref^-1(F_mc(q)) between two per-hit charge samples, both restricted to the
    single-p.e. window [lo, hi]; charges outside the window are returned unchanged."""
    u = np.linspace(0.0, 1.0, n)
    a, b = q_mc[(q_mc >= lo) & (q_mc <= hi)], q_ref[(q_ref >= lo) & (q_ref <= hi)]
    xm, idx = np.unique(np.quantile(a, u), return_index=True)
    xr = np.quantile(b, u)[idx]
    return lambda q: np.where((q >= lo) & (q <= hi), np.interp(q, xm, xr), q)


def spe_reference(ctx: RunContext, mc: RunContext, runs: list, method: str):
    """Data isolated (dark-noise) hits and MC capture-light hits after the capture hit cleaning,
    delayed window, runs/positions of --hit-runs."""
    import re
    from ..data.processor import AmBeNeutronProcessing, mc_label_position
    from ..mc.capture_truth import DEFAULTS, _no_waveform
    pos = AmBeNeutronProcessing().source_positions
    lo, hi = (T_FIT[0] * 1000, T_FIT[1] * 1000)
    cand = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_ic"),
                           filters=[("method", "in", ["clusterfinder", "peak", "optics"])],
                           columns=["run", "trigger", "t_med"])
    iso, xyz = [], set()
    for run in runs:
        ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_{method}"),
                             filters=[("run", "==", run)], columns=["trigger", "is_cosmic"])
        keep = set(ei.trigger[~ei.is_cosmic])
        h = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_hits_{run}"), columns=["trigger", "t", "pe"])
        h = h[h.trigger.isin(keep) & (h.t >= lo) & (h.t <= hi)].sort_values("t")
        cc = cand[cand.run == run][["trigger", "t_med"]].sort_values("t_med")
        j = pd.merge_asof(h, cc, left_on="t", right_on="t_med", by="trigger", direction="nearest")
        iso.append(j[~((j.t - j.t_med).abs() <= 500.0)].pe.to_numpy(float))
        xyz.add(tuple(float(v) for v in pos[run]))
    rx = re.compile(r"ANNIEEvent_MC_AmBe_(?P<lab>.+QE_[\d.]+)_\d+_\d+\.root$")
    pf = mc.parquet_path(f"{mc.run_name}__pulses")
    files = pd.read_parquet(pf, columns=["_source_file"])._source_file.unique()
    sel = [f for f in files if (mm := rx.search(str(f))) and
           tuple(float(v) for v in mc_label_position(mm.group("lab"))) in xyz]
    mp = pd.read_parquet(pf, filters=[("_source_file", "in", sel)],
                         columns=["t", "pe", "is_neutron", "wf_matched", "is_untraced"])
    cfg = {**DEFAULTS, **(mc.extra.get("capture_truth", {}) or {})}
    mp = mp[(mp.t >= lo) & (mp.t <= hi) & (mp.is_neutron == 1)
            & ~(_no_waveform(mp) & (mp.pe.to_numpy() < cfg["lowpe_cut"]))]
    return np.concatenate(iso), mp.pe.to_numpy(float)


def geometry_off_pmts(path) -> list:
    """Channel keys whose detector_status is not ON in a ToolAnalysis FullTankPMTGeometry.csv."""
    g = pd.read_csv(path, skiprows=1)
    g = g[pd.to_numeric(g.channel_num, errors="coerce").notna()]
    return sorted(int(c) for c in g.channel_num[g.detector_status.astype(str).str.strip() != "ON"])


def mask_suffix(pmts) -> str:
    return "_mask" + "-".join(str(int(v)) for v in sorted(pmts))


def keep_suffix(keep: float) -> str:
    return f"_keep{keep:.2f}".replace(".", "p")


def thin_mask(src: str, evn: int, n: int, keep: float) -> np.ndarray:
    """Seeded per-event binomial thinning of n MC hits (True = kept)."""
    import zlib
    return np.random.default_rng([THIN_SEED, zlib.crc32(str(src).encode()), int(evn)]).random(n) < keep


_HC: dict = {}
THIN_SEED = 1014


def _hc_group(i_grp: int):
    """Production labelling (label_event, same cleaning + overlay) of one pulses row group with the
    hit correction applied to the MC hits; with _HC['nominal'] also the uncorrected labelling."""
    from ..mc import capture_truth as ct
    g, cfg = ct._G, ct._G["cfg"]
    chunk = ct._load_chunk(i_grp)
    lo, hi = g["window"]
    rows = []
    for evid, df_ev in chunk.groupby("eventID", sort=False):
        df_ev = df_ev.reset_index(drop=True)
        src, evn = df_ev["_source_file"].iloc[0], int(df_ev["eventNumber"].iloc[0])
        caps = g["caps"].get((src, evn))
        cap_t = (np.empty(0) if caps is None else
                 caps.cap_t[(caps.cap_t + g["offset_ns"] >= lo) & (caps.cap_t + g["offset_ns"] <= hi)].to_numpy(float))
        df_cl = g["clusters"].get(evid)
        mc = df_ev["is_overlay"].to_numpy(int) == 0
        dead = mc & df_ev["pmtID"].isin(_HC["dead"]).to_numpy()
        corr = df_ev[~dead].reset_index(drop=True)
        mck = corr["is_overlay"].to_numpy(int) == 0
        corr.loc[mck, "pe"] = _HC["map"](corr.loc[mck, "pe"].to_numpy(float))
        if _HC.get("keep") is not None:
            drop = np.zeros(len(corr), bool)
            drop[mck] = ~thin_mask(src, evn, int(mck.sum()), _HC["keep"])
            corr = corr[~drop].reset_index(drop=True)
        for variant, d in (("corrected", corr),) + ((("nominal", df_ev),) if _HC["nominal"] else ()):
            hc = ct.assign_hits_to_captures(d, cap_t, g["offset_ns"], cfg["hit_window_ns"])
            lbl = ct.label_event(d, df_cl, g["methods"], g["feat"], cfg, None)
            t, pe, ov = d["t"].to_numpy(float), d["pe"].to_numpy(float), d["is_overlay"].to_numpy(int)
            for method, lb in lbl.items():
                for cid in np.unique(lb[lb >= 0]):
                    m = lb == cid
                    rows.append((variant, src, evn, method, float(np.median(t[m])), float(pe[m].sum()),
                                 int(m.sum()), int(ov[m].sum()), bool((hc[m] >= 0).any())))
    return rows


def build_hitcorr(mc: RunContext, tag: str, methods: list, qmap, dead: list, validate_groups: int, jobs: int,
                  out_path, labels: Optional[list] = None, keep: Optional[float] = None) -> pd.DataFrame:
    """Re-label every MC row group with the hit correction; validates the uncorrected labelling of the
    first `validate_groups` groups against __capture_truth_clusters and stops on any mismatch."""
    import re
    import time
    from multiprocessing import Pool
    import pyarrow.parquet as pq
    from ..mc import capture_truth as ct
    cfg = {**ct.DEFAULTS, **(mc.extra.get("capture_truth", {}) or {})}
    cfg["peak"] = {**ct.DEFAULTS["peak"], **(cfg.get("peak") or {})}
    cfg["_full_window"] = False
    blk = mc.extra.get("clusterindex") or {}
    pulses = mc.parquet_path(f"{mc.run_name}__pulses")
    pf = pq.ParquetFile(str(pulses))
    gfile = {i: Path(str(pf.read_row_group(i, columns=["_source_file"]).column(0)[0].as_py())).name
             for i in range(pf.num_row_groups)}
    by_lab = {}
    for i, f in gfile.items():
        by_lab.setdefault(re.search(r"AmBe_(port\d_z\w+?)QE", f).group(1), []).append(i)
    caps = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__true_captures"))
    cf = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__clusterfinder"))
    ev = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__events"))
    overlay = ct.load_overlay(blk["overlay_veto"]["random_hits"])
    win = [float(v) for v in cfg["window_ns"]]
    rows, t0, done = [], time.time(), 0
    for lab, groups in sorted(by_lab.items()):
        if labels and lab not in labels:
            continue
        want = {gfile[i] for i in groups}
        c = caps[caps._source_file.isin(want)]
        cl = cf[cf._source_file.isin(want)]
        ct._G.clear()
        ct._G.update(ctx=mc, cfg=cfg, feat=mc.extra.get("features", {}) or {}, methods=methods,
                     pulses_path=str(pulses), caps={kk: gg for kk, gg in c.groupby(["_source_file", "eventNumber"])},
                     clusters={kk: gg.reset_index(drop=True) for kk, gg in cl.groupby("eventID")},
                     window=win, offset_ns=ct.mc_clock.truth_offset_ns(cfg["offset_ns"], mc.extra),
                     hit_cleaning="unmatched_lowpe",
                     hit_window="delayed", overlay=overlay, group_file=gfile,
                     events={f: g_[["eventNumber", "_source_file"]] for f, g_ in ev.groupby("_source_file")
                             if f in want})
        _HC.update(map=qmap, dead=list(dead), nominal=False, tag=tag, keep=keep)
        val = groups[:validate_groups] if done == 0 else []
        if val:
            _HC["nominal"] = True
            vr = [r for i in val for r in _hc_group(i)]
            _HC["nominal"] = False
            _validate(mc, vr, [gfile[i] for i in val], methods)
        with Pool(jobs) as pool:
            for r in pool.imap_unordered(_hc_group, groups, chunksize=2):
                rows.extend(r)
        done += 1
        print(f"[datamcclust] hitcorr {lab}: {len(groups)} files ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["variant", "_source_file", "eventNumber", "method", "t_med", "pe",
                                     "n_hits", "n_overlay", "cap"])
    df = df[df.variant == "corrected"].drop(columns="variant")
    df.to_parquet(out_path, index=False)
    return df


def _validate(mc: RunContext, rows: list, files: list, methods: list):
    """The uncorrected re-labelling must reproduce __capture_truth_clusters cluster by cluster."""
    r = pd.DataFrame(rows, columns=["variant", "_source_file", "eventNumber", "method", "t_med", "pe",
                                    "n_hits", "n_overlay", "cap"])
    r = r[r.variant == "nominal"]
    ref = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__capture_truth_clusters__{_HC['tag']}"),
                          filters=[("_source_file", "in", files), ("method", "in", methods)],
                          columns=["_source_file", "eventNumber", "method", "t_med", "pe_cluster", "n_hits",
                                   "n_overlay", "cap_dom"])
    key = ["_source_file", "eventNumber", "method"]
    a = r.sort_values(key + ["t_med"]).reset_index(drop=True)
    b = ref.sort_values(key + ["t_med"]).reset_index(drop=True)
    ok = (len(a) == len(b) and (a[key].values == b[key].values).all()
          and np.allclose(a.t_med, b.t_med) and np.allclose(a.pe, b.pe_cluster, rtol=1e-5, atol=1e-4)
          and (a.n_hits.values == b.n_hits.values).all() and (a.n_overlay.values == b.n_overlay.values).all()
          and (a.cap.values == (b.cap_dom.values >= 0)).all())
    print(f"[datamcclust] hitcorr validation on {len(files)} files: {len(a):,} re-labelled vs "
          f"{len(b):,} production clusters -> {'IDENTICAL' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit("[datamcclust] the re-labelling does not reproduce the production clusters; "
                         "the hit correction would not be comparable")


def page_template(ctx: RunContext, out: Out, method: str, bkg_run: int, kmax: int):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for gate, colr in (("ic", "#c0392b"), ("none", "k")):
        e = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_{gate}_{method}"))
        e = e[(e.run == bkg_run) & ~e.is_cosmic]
        c = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_{gate}"),
                            filters=[("method", "==", method)])
        c = c[c.run == bkg_run]
        q = pmf(e.n_accepted_clusters.to_numpy(), kmax)
        ax[0].errorbar(np.arange(kmax + 1), q, yerr=np.sqrt(q * len(e)) / len(e), fmt="o-", color=colr,
                       label=f"gate {gate}: {len(e):,} triggers, ≥1 {1 - q[0]:.3f}")
        h = np.histogram(c.pe, Q_EDGES)[0]
        ax[1].stairs(h / max(h.sum(), 1), Q_EDGES, color=colr, label=f"gate {gate}: {len(c):,} clusters")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(kmax + 1), [str(k) for k in range(kmax)] + [f"≥{kmax}"])
    ax[0].set_xlabel("candidates per trigger")
    ax[0].set_ylabel("fraction of triggers")
    ax[1].set_xlabel("cluster charge [p.e.]")
    ax[1].set_ylabel("fraction of clusters")
    for a in ax:
        a.legend(frameon=False, fontsize=9)
    fig.suptitle(f"No-source run {bkg_run}, {LABELS.get(method, method)}: IC-gated vs all non-vetoed triggers",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, f"backup_template_{bkg_run}_{method}")


GD_NOMINAL = 0.10      # % Gd by mass in WCSim "Doped Water" (WCSimConstructMaterials.cc)
GD_TAIL_US = 20.0      # thermal regime: the MC capture-time tail is a single exponential from here on


def gd_model(mc: RunContext) -> tuple:
    """Per-event first true capture (all times) and the thermal capture rates of the nominal MC:
    1/tau0 = lambda_Gd + lambda_other, tau0 = mean(t - t0 | t > t0) (memoryless), Gd share of the tail."""
    tc = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__true_captures"),
                         columns=["_source_file", "eventNumber", "cap_idx", "cap_t", "cap_medium"])
    tc = tc[tc.cap_idx == 0].drop(columns="cap_idx")
    t = tc.cap_t.to_numpy(float) / 1000.0
    th = t > GD_TAIL_US
    tau0 = float((t[th] - GD_TAIL_US).mean())
    f_gd = float((tc.cap_medium.to_numpy()[th] == "Gd").mean())
    n_ev = len(pd.read_parquet(mc.parquet_path(f"{mc.run_name}__events"), columns=["eventNumber"]))
    model = dict(tau0=tau0, tau0_err=tau0 / np.sqrt(th.sum()), f_gd=f_gd, lam_gd=f_gd / tau0,
                 lam_other=(1 - f_gd) / tau0, n_events=n_ev, n_captured=len(tc))
    return tc.assign(cap_us=t, is_gd=tc.cap_medium.to_numpy() == "Gd"), model


def gd_event_weights(tc: pd.DataFrame, model: dict, gd: float) -> tuple:
    """Weight per captured event for a Gd loading gd [%]: lambda_Gd -> r lambda_Gd (r = gd / nominal),
    density lambda_m S(t) -> (r on Gd, 1 otherwise) x exp((1 - r) lambda_Gd t). Normalised so that the
    mean over ALL MC events (uncaptured ones weigh 1) is 1. Returns (weights, uncaptured weight, raw norm)."""
    r = gd / GD_NOMINAL
    w = np.where(tc.is_gd.to_numpy(), r, 1.0) * np.exp((1.0 - r) * model["lam_gd"] * tc.cap_us.to_numpy())
    norm = (w.sum() + (model["n_events"] - len(w))) / model["n_events"]
    return w / norm, 1.0 / norm, norm


def _attach(df: pd.DataFrame, tc: pd.DataFrame, w: np.ndarray, w_none: float) -> np.ndarray:
    key = ["_source_file", "eventNumber"]
    m = df[key].merge(tc[key].assign(_w=w), on=key, how="left")
    return m._w.fillna(w_none).to_numpy(float)


def _invert(tau: np.ndarray, gd: np.ndarray, tau0: float) -> tuple:
    """Gd loading at tau0 and d(gd)/d(tau) there, from a quadratic gd(tau) through the scan points."""
    c = np.polyfit(tau, gd, 2 if len(tau) >= 3 else 1)
    return float(np.polyval(c, tau0)), float(np.polyval(np.polyder(c), tau0))


def page_gd(S_by: dict, out: Out, truths: dict, tc: pd.DataFrame, model: dict, gds: list, data_fits: dict):
    """tau(Gd) calibration from the reweighted MC, the Gd loading the data prefer, the Gd share, and per run."""
    import matplotlib.pyplot as plt
    from ..plotting import METHOD_COLORS
    rows, share = [], []
    for gd in gds:
        w, w0, norm = gd_event_weights(tc, model, gd)
        share.append(dict(gd=gd, f_gd_all=float(w[tc.is_gd.to_numpy()].sum() / w.sum()),
                          f_gd_thermal=gd / GD_NOMINAL * model["lam_gd"]
                          / (gd / GD_NOMINAL * model["lam_gd"] + model["lam_other"]),
                          tau_tail_pred=1.0 / (gd / GD_NOMINAL * model["lam_gd"] + model["lam_other"]), norm=norm))
        for m, S in S_by.items():
            tt = truths[m]
            ft = _nfit(tt.cap_us.to_numpy(), 0.0, w=_attach(tt, tc, w, w0), seed=1)
            fm = _nfit(S.m_c.t_us.to_numpy(), S.flat_mc(), w=_attach(S.m_c, tc, w, w0), seed=1)
            rows.append(dict(method=m, gd=gd, **{f"true_{k}": ft[k] for k in ("tau", "tau_err", "therm")},
                             **{f"reco_{k}": fm[k] for k in ("tau", "tau_err", "therm")}))
            for p in range(NPOS):
                sel = (S.m_c.pid == p).to_numpy()
                if sel.sum() < 1000:
                    continue
                fp = _nfit(S.m_c.t_us.to_numpy()[sel], S.flat_mc(p), w=_attach(S.m_c[sel], tc, w, w0), seed=1)
                rows.append(dict(method=m, gd=gd, pid=p, reco_tau=fp["tau"], reco_tau_err=fp["tau_err"]))
    cal = pd.DataFrame(rows)
    sh = pd.DataFrame(share)
    pooled = cal[cal.pid.isna()] if "pid" in cal.columns else cal
    pref = []
    for m, g in pooled.groupby("method", sort=False):
        g = g.sort_values("reco_tau")
        fd = data_fits[m]
        x, sl = _invert(g.reco_tau.to_numpy(), g.gd.to_numpy(), fd["tau"])
        inside = g.reco_tau.min() <= fd["tau"] <= g.reco_tau.max()
        pref.append(dict(method=m, tau_data=fd["tau"], tau_data_err=fd["tau_err"], gd_pref=x if inside else np.nan,
                         gd_pref_err=abs(sl) * fd["tau_err"] if inside else np.nan, in_scan_range=inside))
    pref = pd.DataFrame(pref)
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    g0 = pooled[pooled.method == pooled.method.iloc[0]].sort_values("gd")
    ax[0].plot(g0.gd, g0.true_tau, "o-", color="k", label="MC true capture time")
    for m, g in pooled.groupby("method", sort=False):
        g = g.sort_values("gd")
        c = METHOD_COLORS.get(m)
        ax[0].errorbar(g.gd, g.reco_tau, yerr=g.reco_tau_err, fmt="s--", color=c, ms=5,
                       label=f"MC reco, {LABELS.get(m, m)}")
        r = pref[pref.method == m].iloc[0]
        if r.in_scan_range:
            ax[0].errorbar(r.gd_pref, r.tau_data, xerr=r.gd_pref_err, yerr=r.tau_data_err, fmt="o", color=c, ms=9,
                           mfc="white", mew=2, label=f"data {LABELS.get(m, m)}: Gd {r.gd_pref:.4f} ± {r.gd_pref_err:.4f} %")
    ax[0].set_xlabel("Gd loading in the reweighted MC [% by mass]")
    ax[0].set_ylabel("τ, NeutCapture 2–67 µs [µs]")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title(f"τ vs Gd: WCSim nominal {GD_NOMINAL:.2f} %, data placed at the loading where MC τ = data τ",
                    fontsize=11)
    ax[1].plot(sh.gd, sh.f_gd_thermal, "o-", color="k", label="thermal (t > 20 µs) captures")
    ax[1].plot(sh.gd, sh.f_gd_all, "s--", color="0.5", label="all captures")
    ax[1].set_xlabel("Gd loading [% by mass]")
    ax[1].set_ylabel("fraction of captures on Gd")
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title(f"Gd share; nominal MC: tail τ {model['tau0']:.2f} µs, {model['f_gd']:.1%} on Gd in the tail",
                    fontsize=11)
    fig.suptitle("Reweighting the existing MC to another Gd loading (capture rate on Gd ∝ [Gd], 1/v cross-sections)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, "gd_calibration")
    return cal, sh, pref


def page_gd_runs(S: Sample, out: Out, cal: pd.DataFrame, bkg_rate_per_trig_us: float):
    """Data tau per run (NeutCapture, B fixed to the no-source rate) turned into a Gd loading with the
    reweighted MC at the run's source position."""
    import matplotlib.pyplot as plt
    rows = []
    for run, g in S.d_c.groupby("run"):
        e = S.d_e[S.d_e.run == run]
        p = int(g.pid.iloc[0])
        fd = _nfit(g.t_us.to_numpy(), bkg_rate_per_trig_us * len(e))
        c = cal[(cal.method == S.method) & (cal.pid == p)].sort_values("reco_tau")
        gd = gd_err = np.nan
        if len(c) >= 3 and np.isfinite(fd["tau"]) and c.reco_tau.min() <= fd["tau"] <= c.reco_tau.max():
            gd, sl = _invert(c.reco_tau.to_numpy(), c.gd.to_numpy(), fd["tau"])
            gd_err = abs(sl) * fd["tau_err"]
        rows.append(dict(run=int(run), pid=p, position=pos_title(p), n_trig=len(e),
                         date=pd.to_datetime(float(e.eventTankTime.median()), unit="ns"),
                         tau=fd["tau"], tau_err=fd["tau_err"], therm=fd["therm"], redchi=fd["redchi"],
                         gd=gd, gd_err=gd_err))
    r = pd.DataFrame(rows).sort_values("date")
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5))
    ports = [PORTS[int(p) // len(HEIGHTS)][0] for p in r.pid]
    r["port"] = ports
    for port, g in r.groupby("port"):
        ax[0].errorbar(g.date, g.tau, yerr=g.tau_err, fmt="o", ms=5, label=f"Port {port}")
        ax[1].errorbar(g.date, g.gd, yerr=g.gd_err, fmt="o", ms=5, label=f"Port {port}")
    ax[0].set_ylabel("data τ, NeutCapture 2–67 µs [µs]")
    ax[1].set_ylabel("Gd loading from τ [% by mass]")
    for a in ax:
        a.set_xlabel("run date")
        a.legend(frameon=False, fontsize=9)
        a.tick_params(axis="x", rotation=30)
    ok = np.isfinite(r.gd)
    wm = np.average(r.gd[ok], weights=1 / r.gd_err[ok] ** 2) if ok.any() else np.nan
    ax[0].set_title(f"{LABELS.get(S.method, S.method)}, {len(r)} runs", fontsize=11)
    ax[1].set_title(f"each run against reweighted MC at its own position; weighted mean {wm:.4f} %", fontsize=11)
    fig.suptitle("Gd loading over time from the AmBe capture time", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, "gd_vs_time")
    return r


def hitcorr_sample(S: Sample, hc: pd.DataFrame, window: tuple) -> Sample:
    """S with the MC clusters replaced by the hit-corrected re-labelling (same events, same weights)."""
    key = ["_source_file", "eventNumber"]
    h = hc[(hc.method == S.method) & (hc.t_med >= window[0]) & (hc.t_med <= window[1])]
    ev = S.m_e[key + ["run", "eventTankTime", "sourceX", "sourceY", "sourceZ", "pid", "w"]]
    mc = h.merge(ev, on=key).reset_index(drop=True)
    mc["t_us"] = mc.t_med / 1000.0
    mc["pe_per_hit"] = mc.pe / mc.n_hits
    n = mc.groupby(key).agg(n_accepted_clusters=("cap", "size"), n_noise=("cap", lambda c: int((~c).sum())))
    me = S.m_e.drop(columns=["n_accepted_clusters", "n_noise"]).merge(n.reset_index(), on=key, how="left")
    me["n_accepted_clusters"] = me.n_accepted_clusters.fillna(0).astype(int)
    me["n_noise"] = me.n_noise.fillna(0).astype(int)
    return _with(S, m_c=mc, m_e=me)


def page_hitcorr(out: Out, iso: np.ndarray, mcref: np.ndarray, qmap, dead: list, mean_q_mc: float,
                 spe: tuple, slide: bool = False) -> dict:
    import matplotlib.pyplot as plt
    edges = np.arange(-0.5, 8.0 + 0.05, 0.1)
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    x = np.linspace(0, 8, 400)
    ax[0].plot(x, qmap(x), color="k", lw=1.6)
    ax[0].set_xlabel("MC hit charge [p.e.]")
    ax[0].set_ylabel("corrected hit charge [p.e.]")
    ax[0].set_title(f"map q → F⁻¹_data,isolated(F_MC(q)) inside {spe[0]:g}–{spe[1]:g} p.e., identity outside", fontsize=11)
    for v, lab, c in ((iso, "data isolated hits (dark noise)", "0.55"), (mcref, "MC capture hits", "#2a7ab9"),
                      (qmap(mcref), "MC capture hits, mapped", "k")):
        h = np.histogram(v, edges)[0]
        if slide and v is iso:
            ax[1].plot(0.5 * (edges[1:] + edges[:-1]), h / h.sum(), "o", color=c, ms=4, zorder=3,
                       label=f"{lab}: median {np.median(v):.2f} p.e.")
            continue
        ax[1].stairs(h / h.sum(), edges, color=c, lw=1.6, baseline=None,
                     label=f"{lab}: median {np.median(v):.2f}" + (" p.e." if slide else
                                                                  f", mean {v.mean():.2f}, rms {v.std():.2f} p.e."))
    ax[1].set_yscale("log")
    ax[1].set_xlabel("hit charge [p.e.]")
    ax[1].set_ylabel("fraction of hits / 0.1 p.e.")
    ax[1].legend(frameon=False, fontsize=9)
    inw = lambda v: (v >= spe[0]) & (v <= spe[1])                     # noqa: E731
    dvar = float(iso[inw(iso)].var() - mcref[inw(mcref)].var())
    sp = np.sqrt(max(dvar, 0) / mean_q_mc)
    out_iso, out_mc = float(1 - inw(iso).mean()), float(1 - inw(mcref).mean())
    if slide:
        ax[0].set_title(f"MC → data charge map (quantile matching, {spe[0]:g} to {spe[1]:g} p.e.)", fontsize=13)
        ax[1].set_title("hit charge: data dark noise vs MC capture light", fontsize=13)
        fig.suptitle("MC hit charge mapped onto the data single-p.e. response", fontsize=15)
    else:
        ax[1].set_title(f"in {spe[0]:g}–{spe[1]:g} p.e.: extra per-hit variance {dvar:.2f} p.e.² → cluster σ = "
                        f"√(Δvar/⟨q_hit⟩) = {sp:.2f} √p.e.; outside: data {out_iso:.2%}, MC {out_mc:.2%} of hits",
                        fontsize=11)
        fig.suptitle(f"Hit-level MC correction: data single-p.e. response, and MC hits on PMTs "
                     f"{', '.join(map(str, dead))} removed (silent in data)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, "hitcorr_spe_map_slide" if slide else "hitcorr_spe_map")
    return dict(spe_lo=spe[0], spe_hi=spe[1], dvar_hit_window=dvar, mean_q_hit_mc=mean_q_mc, sigma_pred=sp,
                frac_outside_data=out_iso, frac_outside_mc=out_mc,
                iso_mean_window=float(iso[inw(iso)].mean()), iso_rms_window=float(iso[inw(iso)].std()),
                mcref_mean_window=float(mcref[inw(mcref)].mean()), mcref_rms_window=float(mcref[inw(mcref)].std()),
                iso_mean_all=float(iso.mean()), iso_rms_all=float(iso.std()), dead_pmts=" ".join(map(str, dead)))


def truth_table(mc: RunContext, tag: str, S_by: dict, out: Out) -> pd.DataFrame:
    """The per-label capture_truth summaries pooled with the data trigger weights, plus the truth efficiency."""
    import glob
    import re
    import matplotlib.pyplot as plt
    from ..data.processor import mc_label_position
    S0 = next(iter(S_by.values()))
    parts = []
    for f in sorted(glob.glob(str(mc.csv_dir / f"{mc.run_name}__capture_truth_summary__{tag}_port*.csv"))):
        lab = re.search(rf"__{tag}_(port\d_z\w+?)\.csv$", f).group(1)
        x, y, z = mc_label_position(f"{lab}QE")
        pid = int(pos_id(pd.DataFrame(dict(sourceX=[x], sourceY=[y], sourceZ=[z])))[0])
        parts.append(pd.read_csv(f).assign(pid=pid))
    t = pd.concat(parts, ignore_index=True)
    t["wd"] = S0.n_d[t.pid]
    cols = ["capture_recall", "capture_split_rate", "true_merge_rate", "hit_purity", "spurious_per_event",
            "clusters_per_event"]
    rows = []
    for m, g in t.groupby("method", sort=False):
        r = dict(method=m, **{c: float(np.average(g[c], weights=g.wd)) for c in cols})
        if m in S_by and "found" in S_by[m].m_e.columns:
            e = S_by[m].m_e
            r["truth_eff"] = float(np.average(e.found.astype(float), weights=e.w))
        rows.append(r)
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(15, 0.6 + 0.45 * len(tab)))
    ax.axis("off")
    hdr = ["method", "capture found", "capture split", "captures merged", "hit purity", "spurious / event",
           "clusters / event", "event truth ε"]
    cell = [[LABELS.get(r.method, r.method)] + [f"{r[c]:.4f}" for c in cols] + [f"{r.get('truth_eff', np.nan):.4f}"]
            for _, r in tab.iterrows()]
    tb = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.5)
    ax.set_title("MC reco–truth matching, 25 positions weighted like the data (captures with ≥ 5 hits; "
                 "owner = ≥ 3 of its hits)", fontsize=12)
    fig.tight_layout()
    out.save(fig, "truth_matching")
    return tab


def run(ctx: RunContext, mc: RunContext, methods: list, backup: list, add_bkg: set, tag: str,
        kmax: int, ks: np.ndarray, sig: np.ndarray, seed: int, hit_runs: list, hit_correction: str,
        gds: list, gd_show: float, jobs: int, validate_files: int, spe: tuple,
        dead_geometry: Optional[str] = None, hitcorr_only: bool = False, mask_pmts: Optional[list] = None,
        thin_keep: Optional[float] = None):
    import json
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    bkg_run = int((ctx.fit_params or {})["background_run"])
    win = tuple(float(v) for v in (mc.extra.get("clusterindex") or {})["window_ns"])
    if hitcorr_only:
        dead = None
    else:
        with PdfPages(ctx.plot_path("datamcclust_hits", "pdf")) as pdf:
            hc = page_hit_charge(ctx, mc, Out(ctx, pdf), hit_runs, methods[0])
            lp = page_large_pulses(ctx, Out(ctx, pdf), hit_runs, methods[0], spe[1])
        lp.to_csv(ctx.csv_path("datamcclust_large_pulses"), index=False)
        dead = [int(v) for v in hc["mc_only_pmts"].split()]
    hc_tag = tag
    if dead_geometry:
        dead = geometry_off_pmts(dead_geometry)
        hc_tag = f"{tag}_deaddoc"
        print(f"[datamcclust] dead PMTs from {dead_geometry}: {dead}")
    if mask_pmts:
        dead = sorted(set(dead) | set(mask_pmts))
        hc_tag = f"{hc_tag}{mask_suffix(mask_pmts)}"
        print(f"[datamcclust] also masking PMTs {sorted(mask_pmts)} in the MC (tag {hc_tag})")
    if thin_keep is not None:
        hc_tag = f"{hc_tag}{keep_suffix(thin_keep)}"
        print(f"[datamcclust] keeping a fraction {thin_keep} of the MC hits (tag {hc_tag})")
    hcl = None
    if hit_correction == "spe_deadpmt":
        iso, mcref = spe_reference(ctx, mc, hit_runs, methods[0])
        qmap = spe_map(mcref, iso, *spe)
        hpath = mc.parquet_path(f"{mc.run_name}__candidates_hitcorr__{hc_tag}")
        meta = dict(hit_runs=hit_runs, dead=dead, methods=methods, tag=tag, spe=list(spe))
        if dead_geometry:
            meta["dead_geometry"] = str(dead_geometry)
        if mask_pmts:
            meta["mask_pmts"] = sorted(mask_pmts)
        if thin_keep is not None:
            meta["thin_keep"] = thin_keep
        side = hpath.with_suffix(".json")
        if hpath.exists() and side.exists() and json.loads(side.read_text()) == meta:
            print(f"[datamcclust] hit correction: reusing {hpath.name} ({meta})")
            hcl = pd.read_parquet(hpath)
        else:
            hcl = build_hitcorr(mc, tag, methods, qmap, dead, validate_files, jobs, hpath, keep=thin_keep)
            side.write_text(json.dumps(meta))
    if hitcorr_only:
        print(f"[datamcclust] hit-corrected candidates: {hpath} ({len(hcl):,} clusters)")
        return
    tc, gmodel = gd_model(mc)
    summary, bypos, fits, chi_rows, S_main, truths, dfits = [], [], [], [], {}, {}, {}
    for m in methods + backup:
        main = m in methods
        S = load(ctx, mc, m, m in add_bkg, bkg_run)
        S.m_c = S.m_c.reset_index(drop=True)
        name = f"datamcclust_{m}" if main else f"datamcclust_backup_{m}"
        with PdfPages(ctx.plot_path(name, "pdf")) as pdf:
            out = Out(ctx, pdf)
            c2, ndf = page_charge(S, out, "charge")
            row = dict(method=m, background_added=S.add_bkg, n_trig_data=int(S.n_d.sum()),
                       n_trig_mc=len(S.m_e), n_trig_bkg=int(S.n_b), n_clu_data=len(S.d_c), n_clu_mc=len(S.m_c),
                       mean_pe_data=S.d_c.pe.mean(), mean_pe_pred=S.pred_mean("pe"),
                       mean_pe_mc_only=float(np.average(S.m_c.pe, weights=S.m_c.w)),
                       chi2_charge=c2, ndf_charge=ndf)
            row.update(page_mult(S, out, kmax))
            g = page_charge_grid(S, out, f"{m}_charge_grid", "MC uncorrected").assign(method=m)
            chi_rows.append(g)
            if main:
                h = page_hits(S, out, f"{m}_hits", "")
                row.update(nh_data=h["n_hits"][0], nh_pred=h["n_hits"][1],
                           pph_data=h["pe_per_hit"][0], pph_pred=h["pe_per_hit"][1],
                           k_decomposed=h["n_hits"][0] / h["n_hits"][1] * h["pe_per_hit"][0] / h["pe_per_hit"][1])
                hm = page_hits_map(S, out)
                row.update(page_scan(S, out, ks, sig, seed))
                tt = truth_times(mc, tag, S.m_e)
                f = page_time(S, out, tt)
                dfits[m] = f["data"]
                for k, v in f.items():
                    fits.append(dict(method=m, sample=k, gd=GD_NOMINAL, position="all", **v))
                w, w0, _ = gd_event_weights(tc, gmodel, gd_show)
                fg = page_time(S, out, tt, name=f"{m}_time_gd{round(gd_show * 1000):03d}",
                               wgd_c=_attach(S.m_c, tc, w, w0), wgd_t=_attach(tt, tc, w, w0),
                               gd_label=f" reweighted to {gd_show:.3f} % Gd")
                for k in ("mc_reco", "mc_true"):
                    fits.append(dict(method=m, sample=k, gd=gd_show, position="all", **fg[k]))
                Sg = _with(S, m_c=S.m_c.assign(w=S.m_c.w * _attach(S.m_c, tc, w, w0)),
                           m_e=S.m_e.assign(wgd=_attach(S.m_e, tc, w, w0)))
                c2g, _ = page_charge(Sg, out, f"charge_gd{round(gd_show * 1000):03d}",
                                     title_extra=f", MC reweighted to {gd_show:.3f} % Gd")
                _, pmg, pmbg, _, _ = Sg.mult(kmax)
                row.update(chi2_charge_gd=c2g, ge1_mc_gd=1 - pmg[0], ge1_mcbkg_gd=(1 - pmbg[0]) if S.add_bkg else np.nan)
                tg = page_time_grid(S, out, tt)
                mg = page_mult_grid(S, out, kmax)
                bp = g.merge(hm, on="pid").merge(tg, on="pid", how="left").merge(mg, on="pid", how="left")
                if hcl is not None:
                    Sh = hitcorr_sample(S, hcl, win)
                    c2h, ndfh = page_charge(Sh, out, "charge_hitcorr",
                                            title_extra=", MC hit-corrected (dead PMTs + data single-p.e.)")
                    hh = page_hits(Sh, out, f"{m}_hits_hitcorr", ", MC hit-corrected")
                    gh = page_charge_grid(Sh, out, f"{m}_charge_grid_hitcorr", "MC hit-corrected")
                    pdh, pmh, pmbh, _, _ = Sh.mult(kmax)
                    row.update(chi2_charge_hitcorr=c2h, ndf_charge_hitcorr=ndfh,
                               mean_pe_pred_hitcorr=Sh.pred_mean("pe"), nh_pred_hitcorr=hh["n_hits"][1],
                               pph_pred_hitcorr=hh["pe_per_hit"][1], ge1_mc_hitcorr=1 - pmh[0],
                               ge1_mcbkg_hitcorr=(1 - pmbh[0]) if S.add_bkg else np.nan,
                               n_clu_mc_hitcorr=len(Sh.m_c))
                    row.update(page_mult_versions(S, Sh, out, kmax))
                    bp = bp.merge(gh[["pid", "chi2_charge"]].rename(columns={"chi2_charge": "chi2_charge_hitcorr"}),
                                  on="pid", how="left").merge(page_mult_grid_versions(S, Sh, out, kmax), on="pid",
                                                              how="left")
                bypos.append(bp.assign(method=m))
                S_main[m], truths[m] = S, tt
        summary.append(row)
        print(f"[datamcclust] wrote {ctx.plot_path(name, 'pdf')}")
    pd.Series(hc).to_csv(ctx.csv_path("datamcclust_hit_charge"), header=["value"])
    with PdfPages(ctx.plot_path("datamcclust_physics", "pdf")) as pdf:
        out = Out(ctx, pdf)
        if hcl is not None:
            mq = float(sum(S.m_c.pe.sum() for S in S_main.values()) / sum(S.m_c.n_hits.sum() for S in S_main.values()))
            hrow = page_hitcorr(out, iso, mcref, qmap, dead, mq, spe)
            pd.Series(hrow).to_csv(ctx.csv_path("datamcclust_hitcorr"), header=["value"])
        tab = truth_table(mc, tag, S_main, out)
        tab.to_csv(ctx.csv_path("datamcclust_truthmatch"), index=False)
        cal, sh, pref = page_gd(S_main, out, truths, tc, gmodel, gds, dfits)
        mrun = "clusterfinder_nms" if "clusterfinder_nms" in S_main else methods[0]
        Sr = S_main[mrun]
        runs = page_gd_runs(Sr, out, cal, len(Sr.b_c) / Sr.n_b / (T_FIT[1] - T_FIT[0]))
    cal.to_csv(ctx.csv_path("datamcclust_gd_calibration"), index=False)
    sh.assign(**{k: v for k, v in gmodel.items()}).to_csv(ctx.csv_path("datamcclust_gd_share"), index=False)
    pref.to_csv(ctx.csv_path("datamcclust_gd_preferred"), index=False)
    runs.to_csv(ctx.csv_path("datamcclust_gd_by_run"), index=False)
    with PdfPages(ctx.plot_path("datamcclust_backup", "pdf")) as pdf:
        out = Out(ctx, pdf)
        rates, rho, ov = page_rate(ctx, mc, out, pd.concat(chi_rows, ignore_index=True))
        page_template(ctx, out, "clusterfinder_nms" if "clusterfinder_nms" in methods else methods[0],
                      bkg_run, kmax)
    sm = pd.DataFrame(summary).assign(spearman_rate=lambda d: d.method.map(rho), overlay_rate_khz=ov)
    sm.to_csv(ctx.csv_path("datamcclust_summary"), index=False)
    bp = pd.concat(bypos, ignore_index=True).merge(rates, on=["method", "pid"], how="left")
    bp.insert(1, "position", bp.pid.map(pos_title))
    bp.to_csv(ctx.csv_path("datamcclust_by_position"), index=False)
    pd.DataFrame(fits).to_csv(ctx.csv_path("datamcclust_capturetime_fits"), index=False)
    pd.set_option("display.width", 250)
    print(sm.round(4).T.to_string())
    print(tab.round(4).to_string(index=False))
    print(pref.round(4).to_string(index=False))


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots datamcclust")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--methods", required=True, help="main methods: every page")
    p.add_argument("--backup-methods", required=True, help="charge and multiplicity pages only")
    p.add_argument("--add-bkg-methods", required=True,
                   help="methods whose MC prediction gets the no-source run's per-trigger clusters "
                        "(clusterers that never see the MC overlay); 'none' for no method")
    p.add_argument("--mc-tag", required=True, help="capture_truth tag of the MC captures (true capture times)")
    p.add_argument("--hit-runs", required=True,
                   help="data runs for the per-hit charge page and the single-p.e. reference")
    p.add_argument("--hit-correction", required=True, choices=["none", "spe_deadpmt"],
                   help="spe_deadpmt: re-label the MC with hits on the data-silent PMTs removed and the MC hit "
                        "charge mapped onto the data isolated-hit (single-p.e.) response")
    p.add_argument("--spe-range", required=True,
                   help="lo,hi [p.e.] single-p.e. window of the hit charge map (outside: unchanged), e.g. -0.5,8")
    p.add_argument("--gd-fractions", required=True,
                   help="Gd loadings [%% by mass] for the reweighted-MC calibration, e.g. 0.06,0.07,0.08,0.09,0.10")
    p.add_argument("--gd-show", required=True, type=float,
                   help="Gd loading [%%] of the reweighted capture-time and charge pages, e.g. 0.08")
    p.add_argument("--dead-pmts-geometry",
                   help="FullTankPMTGeometry.csv: mask the PMTs whose detector_status is not ON instead of the "
                        "PMTs silent in the --hit-runs data; the cache gets the tag suffix _deaddoc")
    p.add_argument("--hitcorr-only", action="store_true",
                   help="build (or reuse) the hit-corrected MC candidates and stop; needs --dead-pmts-geometry")
    p.add_argument("--mask-pmts",
                   help="comma list of PMTs whose MC hits are also removed (with --hitcorr-only), for the "
                        "mask-in-data-and-MC test of `ambe plots datamcneutron --mask-pmts`; tag suffix _mask<ids>")
    p.add_argument("--thin-keep", type=float,
                   help="with --hitcorr-only: keep this fraction of the MC hits (seeded per event) before "
                        "re-labelling, for the light-yield test; tag suffix _keep<f>")
    p.add_argument("--jobs", type=int, default=3, help="processes for the hit-corrected re-labelling")
    p.add_argument("--validate-files", type=int, default=3,
                   help="MC files whose uncorrected re-labelling must reproduce the production clusters")
    p.add_argument("--kmax", type=int, default=4)
    p.add_argument("--scan-k", default="0.90,1.40,0.02", help="lo,hi,step of the MC charge scale scan")
    p.add_argument("--scan-sigma", default="0.0,3.0,0.1", help="lo,hi,step of the extra resolution scan")
    p.add_argument("--seed", type=int, default=20261007)
    a = p.parse_args(list(argv) if argv else [])
    def rng(spec):
        lo, hi, step = (float(v) for v in spec.split(","))
        return np.round(np.arange(lo, hi + step / 2, step), 6)
    methods = a.methods.split(",")
    backup = [m for m in a.backup_methods.split(",") if m and m != "none"]
    add = set() if a.add_bkg_methods == "none" else set(a.add_bkg_methods.split(","))
    if add - set(methods + backup):
        p.error(f"--add-bkg-methods {sorted(add - set(methods + backup))} not in --methods/--backup-methods")
    gds = sorted(float(v) for v in a.gd_fractions.split(","))
    if a.hitcorr_only and not (a.dead_pmts_geometry and a.hit_correction == "spe_deadpmt"):
        p.error("--hitcorr-only needs --dead-pmts-geometry and --hit-correction spe_deadpmt")
    if a.mask_pmts and not a.hitcorr_only:
        p.error("--mask-pmts only with --hitcorr-only")
    if a.thin_keep is not None and not (a.hitcorr_only and 0 < a.thin_keep <= 1):
        p.error("--thin-keep only with --hitcorr-only, in (0, 1]")
    run(ctx, load_context(a.mc_config), methods, backup, add, a.mc_tag, a.kmax,
        rng(a.scan_k), rng(a.scan_sigma), a.seed, [int(r) for r in a.hit_runs.split(",")], a.hit_correction,
        gds, a.gd_show, a.jobs, a.validate_files, tuple(float(v) for v in a.spe_range.split(",")),
        a.dead_pmts_geometry, a.hitcorr_only, [int(v) for v in a.mask_pmts.split(",")] if a.mask_pmts else None,
        a.thin_keep)


# --------------------------------------------------------------------------------------------------
# `ambe plots datamcneutron`: MC neutron clusters against data minus the no-source run.

def _sel(df: pd.DataFrame, pids) -> pd.DataFrame:
    return df if pids is None else df[df.pid.isin(pids)]


def _nd(S: Sample, pids) -> float:
    return float(S.n_d.sum() if pids is None else S.n_d[list(pids)].sum())


def data_minus_bkg(S: Sample, col: str, edges: np.ndarray, pids=None):
    """Data clusters minus the no-source run scaled to the data trigger count: (values, variance)."""
    d = np.histogram(_sel(S.d_c, pids)[col].to_numpy(float), edges)[0].astype(float)
    f = _nd(S, pids) / S.n_b
    hb = np.histogram(S.b_c[col].to_numpy(float), edges)[0].astype(float)
    return d - hb * f, d + hb * f ** 2


def mc_neutron(S: Sample, col: str, edges: np.ndarray, pids=None):
    """MC capture-owned clusters, weighted to the data trigger count per position: (values, variance)."""
    mc = _sel(S.m_c, pids)
    mc = mc[mc.cap.to_numpy(bool)]
    x, w = mc[col].to_numpy(float), mc.w.to_numpy(float)
    return np.histogram(x, edges, weights=w)[0], np.histogram(x, edges, weights=w ** 2)[0]


def neutron_means(S: Sample, col: str, pids=None) -> tuple:
    d = _sel(S.d_c, pids)[col].to_numpy(float)
    f = _nd(S, pids) / S.n_b
    b = S.b_c[col].to_numpy(float)
    mc = _sel(S.m_c, pids)
    mc = mc[mc.cap.to_numpy(bool)]
    return (float((d.sum() - b.sum() * f) / (len(d) - len(b) * f)),
            float(np.average(mc[col], weights=mc.w)) if len(mc) else np.nan)


def neutron_chi2(S: Sample, col: str = "pe", pids=None, edges: np.ndarray = CHI_EDGES):
    d, vd = data_minus_bkg(S, col, edges, pids)
    m, vm = mc_neutron(S, col, edges, pids)
    s = d.sum() / m.sum()
    v = vd + vm * s ** 2
    ok = v > 0
    return float(((d - s * m)[ok] ** 2 / v[ok]).sum()), int(ok.sum()) - 1


def _npanel(a, r, S: Sample, edges, pids, label, logy=False, ls="-", data=True, color=None, zorder=None):
    d, vd = data_minus_bkg(S, "pe", edges, pids)
    m, _ = mc_neutron(S, "pe", edges, pids)
    lo, hi = np.searchsorted(edges, CHI_EDGES[0]), np.searchsorted(edges, CHI_EDGES[-1])
    s = d[lo:hi].sum() / m[lo:hi].sum()
    cen = 0.5 * (edges[1:] + edges[:-1])
    a.stairs(m * s, edges, color=color or METHOD_COLOR(S.method), lw=1.6, ls=ls, baseline=None, label=label,
             zorder=zorder)
    if data:
        a.errorbar(cen, d, yerr=np.sqrt(vd), fmt="o", color="k", ms=3, lw=1, label="data − no-source run")
    if logy:
        a.set_yscale("log")
        a.set_ylim(0.5, None)
    ok = m > 0
    if r is not None:
        r.errorbar(cen[ok], d[ok] / (m[ok] * s), yerr=np.sqrt(vd[ok]) / (m[ok] * s), fmt="o", color="k", ms=2.5, lw=0.8)
        r.set_ylim(0, 2.5)
        r.set_ylabel("data / MC")
        r.set_xlabel("cluster charge [p.e.]")
    a.set_ylabel(f"clusters / {edges[1] - edges[0]:g} p.e.")


def METHOD_COLOR(method: str) -> str:
    from ..plotting import METHOD_COLORS
    return METHOD_COLORS.get(method, "#4a3aa7")


def page_ncharge(S0: Sample, S1: Sample, out: Out, edges=Q_EDGES) -> dict:
    """Nominal and hit-corrected MC neutron clusters against data minus the no-source run."""
    import matplotlib.pyplot as plt
    lab = LABELS.get(S0.method, S0.method)
    c0, n0 = neutron_chi2(S0)
    c1, n1 = neutron_chi2(S1)
    fig, ax = plt.subplots(2, 3, figsize=(18, 7.5), gridspec_kw=dict(height_ratios=[3, 1]), sharex="col")
    _npanel(ax[0, 0], ax[1, 0], S0, edges, None, "MC neutrons, nominal")
    _npanel(ax[0, 1], ax[1, 1], S1, edges, None, "MC neutrons, hit-corrected")
    _npanel(ax[0, 2], ax[1, 2], S1, edges, None, "MC neutrons, hit-corrected", logy=True)
    md, mm0 = neutron_means(S0, "pe")
    _, mm1 = neutron_means(S1, "pe")
    ax[0, 0].set_title(f"nominal MC: χ²/ndf {c0:.0f}/{n0}, mean {mm0:.1f} p.e. (data {md:.1f})", fontsize=11)
    ax[0, 1].set_title(f"hit-corrected MC: χ²/ndf {c1:.0f}/{n1}, mean {mm1:.1f} p.e.", fontsize=11)
    ax[0, 2].set_title("hit-corrected MC, log scale", fontsize=11)
    for a in ax[0]:
        a.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{lab}: cluster charge, MC neutron clusters vs data minus the no-source run "
                 f"(shape-normalised in {CHI_EDGES[0]:g}–{CHI_EDGES[-1]:g} p.e.)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.save(fig, f"{S0.method}_charge")
    return dict(chi2_nominal=c0, ndf_nominal=n0, chi2_hitcorr=c1, ndf_hitcorr=n1, mean_pe_data_sub=md,
                mean_pe_mc_nominal=mm0, mean_pe_mc_hitcorr=mm1)


def page_nports(S: Sample, out: Out, name: str, desc: str) -> pd.DataFrame:
    """One panel per port (its five heights summed)."""
    import matplotlib.pyplot as plt
    edges = np.arange(0.0, 100.0 + 2.5, 2.5)
    fig, ax = plt.subplots(4, 3, figsize=(18, 11), gridspec_kw=dict(height_ratios=[3, 1, 3, 1]))
    rows = []
    for i, (port, _) in enumerate(PORTS):
        pids = [i * len(HEIGHTS) + j for j in range(len(HEIGHTS))]
        a, r = ax[2 * (i // 3), i % 3], ax[2 * (i // 3) + 1, i % 3]
        _npanel(a, r, S, edges, pids, "MC neutrons")
        c2, ndf = neutron_chi2(S, pids=pids)
        md, mm = neutron_means(S, "pe", pids)
        a.set_title(f"Port {port}: χ²/ndf {c2:.0f}/{ndf}, mean data {md:.1f}, MC {mm:.1f} p.e.", fontsize=11)
        rows.append(dict(port=port, chi2=c2, ndf=ndf, mean_pe_data_sub=md, mean_pe_mc=mm, n_trig=_nd(S, pids)))
    h, lab = ax[0, 0].get_legend_handles_labels()
    ax[2, 2].axis("off")
    ax[3, 2].axis("off")
    ax[2, 2].legend(h, lab, loc="center", frameon=False, fontsize=13)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: cluster charge per port (five heights summed), {desc}",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.save(fig, name)
    return pd.DataFrame(rows)


def page_ngrid(S: Sample, out: Out, name: str, desc: str) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    edges = np.arange(0.0, 100.0 + 2.5, 5.0)
    fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(22, 16))
    rows = []
    for p in range(NPOS):
        a = ax.flat[p]
        if S.n_d[p] == 0:
            a.set_title(f"{pos_title(p)}: no data", fontsize=10)
            a.axis("off")
            continue
        _npanel(a, None, S, edges, [p], "MC neutrons")
        c2, ndf = neutron_chi2(S, pids=[p])
        md, mm = neutron_means(S, "pe", [p])
        a.set_title(f"{pos_title(p)}: χ²/ndf {c2:.0f}/{ndf}", fontsize=10)
        a.set_xlabel("cluster charge [p.e.]")
        rows.append(dict(pid=p, position=pos_title(p), chi2=c2, ndf=ndf, mean_pe_data_sub=md, mean_pe_mc=mm,
                         n_trig=float(S.n_d[p])))
    h, lab = ax.flat[rows[0]["pid"]].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: cluster charge per source position, {desc}",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, name)
    return pd.DataFrame(rows)


def page_ngrid_both(S0: Sample, S1: Sample, out: Out, name: str):
    """Charge per source position: data minus 6264 against both MC versions in the same panel."""
    import matplotlib.pyplot as plt
    edges = np.arange(0.0, 100.0 + 2.5, 5.0)
    fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(22, 16))
    first = None
    for p in range(NPOS):
        a = ax.flat[p]
        if S0.n_d[p] == 0:
            a.set_title(f"{pos_title(p)}: no data", fontsize=10)
            a.axis("off")
            continue
        first = p if first is None else first
        _npanel(a, None, S1, edges, [p], "MC neutrons, hit-corrected")
        _npanel(a, None, S0, edges, [p], "MC neutrons, nominal", ls="--", data=False, color="0.45", zorder=3)
        c0, ndf = neutron_chi2(S0, pids=[p])
        c1, _ = neutron_chi2(S1, pids=[p])
        a.set_title(f"{pos_title(p)}: χ²/ndf {c0:.0f} → {c1:.0f} /{ndf}", fontsize=10)
        a.set_xlabel("cluster charge [p.e.]")
    h, lab = ax.flat[first].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{LABELS.get(S0.method, S0.method)}: cluster charge per source position, nominal and hit-corrected MC",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, name)


def page_nmult_grid_both(S0: Sample, S1: Sample, out: Out, kmax: int, name: str):
    """Clusters per trigger per source position: data, nominal MC neutron clusters alone, and hit-corrected MC
    neutron clusters folded with 6264."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(22, 16), sharey=True)
    e = np.arange(kmax + 2) - 0.5
    first = None
    for p in range(NPOS):
        a = ax.flat[p]
        if S0.n_d[p] == 0:
            a.set_title(f"{pos_title(p)}: no data", fontsize=10)
            a.axis("off")
            continue
        first = p if first is None else first
        pdat, pm0, _, nd = neutron_mult(S0, kmax, [p])
        _, _, pb1, _ = neutron_mult(S1, kmax, [p])
        a.stairs(pm0, e, color=METHOD_COLOR(S0.method), lw=1.2, ls="--", baseline=None,
                 label="MC neutron clusters, nominal")
        a.stairs(pb1, e, color=METHOD_COLOR(S0.method), lw=1.8, baseline=None,
                 label="MC neutron clusters ⊗ no-source run, hit-corrected")
        a.errorbar(np.arange(kmax + 1), pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=4,
                   label="data, IC triggers")
        a.set_yscale("log")
        a.set_ylim(5e-5, 1.5)
        a.set_xticks(range(kmax + 1), [str(k) for k in range(kmax)] + [f"≥{kmax}"])
        a.set_title(f"{pos_title(p)}: no cluster {pdat[0]:.3f} / {pb1[0]:.3f}", fontsize=10)
    h, lab = ax.flat[first].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=12)
    two_titles(fig, "Neutron candidates per trigger, by source position",
               f"{LABELS.get(S0.method, S0.method)}: clusters per trigger per source position; panel titles: no-cluster "
               f"fraction, data / MC ⊗ no-source run", rect=(0, 0.03, 1, 1), simple_size=18, detail_size=12)
    out.save(fig, name)


def page_fold_explain(S: Sample, out: Out, kmax: int) -> dict:
    """The fold in three steps: 6264 clusters per trigger, MC neutron clusters per event, and their sum
    against the data."""
    import matplotlib.pyplot as plt
    pdat, pm, pmb, nd = neutron_mult(S, kmax)
    nb = np.bincount(np.minimum(S.b_e.n_accepted_clusters.to_numpy(), kmax), minlength=kmax + 1)
    pb = nb / nb.sum()
    e = np.arange(kmax + 2) - 0.5
    k = np.arange(kmax + 1)
    ticks = [str(i) for i in range(kmax)] + [f"≥{kmax}"]
    col = METHOD_COLOR(S.method)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), sharey=True)
    ax[0].stairs(pb, e, color="0.35", lw=1.8, baseline=None)
    ax[0].errorbar(k, pb, yerr=np.sqrt(nb) / nb.sum(), fmt="none", color="0.35", lw=1)
    ax[0].set_title(f"no-source run 6264, {int(nb.sum()):,} triggers: " + " / ".join(f"{v:,}" for v in nb), fontsize=11)
    ax[0].set_xlabel("no-source clusters per trigger, j")
    ax[0].set_ylabel("fraction")
    ax[1].stairs(pm, e, color=col, lw=1.8, ls="--", baseline=None)
    ax[1].set_title("MC neutron clusters per event, k: " + " / ".join(f"{v:.3f}" for v in pm[:3]), fontsize=11)
    ax[1].set_xlabel("MC neutron clusters per event, k")
    ax[2].stairs(pmb, e, color=col, lw=1.8, baseline=None, label="P(n) = Σ_k P_MC(k) P_6264(n − k)")
    ax[2].errorbar(k, pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=5, label="data, IC triggers")
    ax[2].set_title(f"n = k + j: no cluster {pmb[0]:.3f} (data {pdat[0]:.3f}), one {pmb[1]:.3f} ({pdat[1]:.3f})",
                    fontsize=11)
    ax[2].set_xlabel("clusters per trigger, n")
    ax[2].legend(frameon=False, fontsize=10)
    for a in ax:
        a.set_yscale("log")
        a.set_ylim(5e-5, 1.5)
        a.set_xticks(k, ticks)
    two_titles(fig, "Adding the detector background to the simulation",
               f"{LABELS.get(S.method, S.method)}: clusters per trigger, the no-source count added to the MC",
               simple_size=16, detail_size=11)
    out.save(fig, f"{S.method}_explain_fold")
    return dict(method=S.method, **{f"nb{i}": int(v) for i, v in enumerate(nb)},
                **{f"pb{i}": float(v) for i, v in enumerate(pb)}, **{f"pm{i}": float(v) for i, v in enumerate(pm)},
                **{f"pfold{i}": float(v) for i, v in enumerate(pmb)}, **{f"pdata{i}": float(v) for i, v in enumerate(pdat)},
                mean_mc=float((k * pm).sum()), mean_6264=float((k * pb).sum()), mean_data=float((k * pdat).sum()))


def page_nmaps(tabs: dict, out: Out):
    """Per position: chi2/ndf nominal and hit-corrected, mean-charge ratio data/MC hit-corrected."""
    import matplotlib.pyplot as plt
    meths = list(tabs)
    fig, ax = plt.subplots(len(meths), 3, figsize=(18, 5.6 * len(meths)), squeeze=False)
    for r_, m in enumerate(meths):
        t0, t1 = tabs[m]
        grid = lambda t, f: np.array([[f(t[t.pid == i * len(HEIGHTS) + j]) for j in range(len(HEIGHTS))]  # noqa: E731
                                      for i in range(len(PORTS))])
        val = lambda t: (t.chi2 / t.ndf).iloc[0] if len(t) else np.nan                                    # noqa: E731
        rat = lambda t: (t.mean_pe_data_sub / t.mean_pe_mc).iloc[0] if len(t) else np.nan                 # noqa: E731
        g0, g1, g2 = grid(t0, val), grid(t1, val), grid(t1, rat)
        vmax = np.nanmax([g0, g1])
        for a, g, ttl, cm, vr, fmt in ((ax[r_, 0], g0, "χ²/ndf, nominal MC", "viridis", (0, vmax), "{:.1f}"),
                                       (ax[r_, 1], g1, "χ²/ndf, hit-corrected MC", "viridis", (0, vmax), "{:.1f}"),
                                       (ax[r_, 2], g2, "mean charge data / MC, hit-corrected", "coolwarm",
                                        (0.8, 1.3), "{:.3f}")):
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(1.0, vr[0], vr[1]) if cm == "coolwarm" else None
            im = a.imshow(g, cmap=cm, norm=norm) if norm else a.imshow(g, cmap=cm, vmin=vr[0], vmax=vr[1])
            for i in range(len(PORTS)):
                for j in range(len(HEIGHTS)):
                    if np.isfinite(g[i, j]):
                        a.text(j, i, fmt.format(g[i, j]), ha="center", va="center", fontsize=10,
                               color="w" if cm == "viridis" and g[i, j] < 0.6 * vr[1] else "k")
            a.set_xticks(range(len(HEIGHTS)), [f"{h:+.0f}" for h in HEIGHTS])
            a.set_yticks(range(len(PORTS)), [f"Port {p}" for p, _ in PORTS])
            a.set_xlabel("source Y [cm]")
            a.set_title(f"{LABELS.get(m, m)}: {ttl}", fontsize=12)
            fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    out.save(fig, "position_maps")


def page_nhits(S0: Sample, S1: Sample, out: Out) -> dict:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    res = {}
    for a, col, edges, xl in ((ax[0], "n_hits", np.arange(4.5, 40.5, 1.0), "hits per cluster"),
                              (ax[1], "pe_per_hit", np.arange(0.5, 4.0 + 0.05, 0.1), "charge per hit [p.e.]")):
        d, vd = data_minus_bkg(S0, col, edges)
        cen = 0.5 * (edges[1:] + edges[:-1])
        a.errorbar(cen, d / d.sum(), yerr=np.sqrt(vd) / d.sum(), fmt="o", color="k", ms=3, lw=1,
                   label="data − no-source run")
        md, _ = neutron_means(S0, col)
        res[f"{col}_data_sub"] = md
        for S, ls, tag in ((S0, "--", "nominal"), (S1, "-", "hit-corrected")):
            m, _ = mc_neutron(S, col, edges)
            mm = neutron_means(S, col)[1]
            res[f"{col}_mc_{tag.replace('-', '')}"] = mm
            a.stairs(m / m.sum(), edges, color=METHOD_COLOR(S.method), lw=1.6, ls=ls, baseline=None,
                     label=f"MC neutrons, {tag}: mean {mm:.2f}")
        a.set_xlabel(xl)
        a.set_ylabel("fraction of clusters")
        a.set_title(f"{xl}: data mean {md:.2f}", fontsize=11)
        a.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{LABELS.get(S0.method, S0.method)}: charge = hits × charge per hit, "
                 f"MC neutron clusters vs data minus the no-source run", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, f"{S0.method}_hits")
    return res


def neutron_mult(S: Sample, kmax: int, pids=None):
    """Per-trigger cluster counts: data, MC neutron clusters, MC neutron clusters ⊗ no-source run."""
    key = ["_source_file", "eventNumber"]
    e = _sel(S.m_e, pids)
    nc = S.m_c[S.m_c.cap.to_numpy(bool)].groupby(key).size().rename("n_cap").reset_index()
    n = e[key + ["w"]].merge(nc, on=key, how="left").n_cap.fillna(0).to_numpy(int)
    pm = pmf(n, kmax, e.w.to_numpy(float))
    pb = pmf(S.b_e.n_accepted_clusters.to_numpy(), kmax)
    d = _sel(S.d_e, pids)
    return pmf(d.n_accepted_clusters.to_numpy(), kmax), pm, convolve_pmf(pm, pb, kmax), len(d)


def _mpanel(a, S: Sample, kmax: int, pids, ms=5):
    pdat, pm, pmb, nd = neutron_mult(S, kmax, pids)
    e = np.arange(kmax + 2) - 0.5
    a.stairs(pmb, e, color=METHOD_COLOR(S.method), lw=1.8, baseline=None, label="MC neutron clusters ⊗ no-source run")
    a.stairs(pm, e, color=METHOD_COLOR(S.method), lw=1.2, ls="--", baseline=None, label="MC neutron clusters")
    a.errorbar(np.arange(kmax + 1), pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=ms,
               label="data, IC triggers")
    a.set_yscale("log")
    a.set_ylim(5e-5, 1.5)
    a.set_xticks(range(kmax + 1), [str(k) for k in range(kmax)] + [f"≥{kmax}"])
    return pdat, pm, pmb, nd


def page_nmult(S0: Sample, S1: Sample, out: Out, kmax: int) -> dict:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5), sharey=True)
    res = {}
    for a, S, tag in ((ax[0], S0, "nominal"), (ax[1], S1, "hitcorr")):
        pdat, pm, pmb, _ = _mpanel(a, S, kmax, None)
        k = np.arange(kmax + 1)
        res.update({"m0_data": pdat[0], f"m0_mc_{tag}": pm[0], f"m0_mcbkg_{tag}": pmb[0],
                    "ge2_data": pdat[2:].sum(), f"ge2_mcbkg_{tag}": pmb[2:].sum(),
                    "mean_mult_data": float(S0.d_e.n_accepted_clusters.mean()),
                    f"mean_mult_mcbkg_{tag}": float((k * pmb).sum())})
        a.set_title(f"{'nominal' if tag == 'nominal' else 'hit-corrected'} MC: no cluster, data {pdat[0]:.3f} / "
                    f"MC {pmb[0]:.3f}; ≥2, data {pdat[2:].sum():.3f} / MC {pmb[2:].sum():.3f}", fontsize=11)
        a.set_xlabel("clusters per trigger (2–67 µs)")
        a.legend(frameon=False, fontsize=9)
    ax[0].set_ylabel("fraction of triggers")
    two_titles(fig, "Neutron candidates per trigger: data and simulation",
               f"{LABELS.get(S0.method, S0.method)}: clusters per trigger, data vs MC neutron clusters with the "
               f"no-source run folded in", simple_size=16, detail_size=11)
    out.save(fig, f"{S0.method}_mult")
    return res


def page_nmult_grid(S: Sample, out: Out, kmax: int, name: str, desc: str) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(22, 16), sharey=True)
    rows = []
    for p in range(NPOS):
        a = ax.flat[p]
        if S.n_d[p] == 0:
            a.set_title(f"{pos_title(p)}: no data", fontsize=10)
            a.axis("off")
            continue
        pdat, pm, pmb, nd = _mpanel(a, S, kmax, [p], ms=4)
        a.set_title(f"{pos_title(p)}: no cluster {pdat[0]:.3f} / {pmb[0]:.3f}", fontsize=10)
        rows.append(dict(pid=p, position=pos_title(p), n_trig=nd, m0_data=pdat[0], m0_mc=pm[0], m0_mcbkg=pmb[0],
                         ge2_data=pdat[2:].sum(), ge2_mcbkg=pmb[2:].sum()))
    h, lab = ax.flat[rows[0]["pid"]].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: clusters per trigger per source position, {desc}; "
                 f"titles: no-cluster fraction data / MC", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, name)
    return pd.DataFrame(rows)


def _random_clusters(rctx: RunContext, method: str, win: tuple, tag: str):
    """6256 dummy-trigger clusters of one clusterer in the cluster window, and the trigger count."""
    n = len(pd.read_parquet(rctx.parquet_path(f"{rctx.run_name}__random_triggers_6256")))
    r = pd.read_parquet(rctx.parquet_path(f"{rctx.run_name}__random_recoclusters_6256__{tag}"),
                        filters=[("method", "==", method)], columns=["trigger", "t_med", "f_pe_total", "n_hits"])
    if r.empty:
        raise SystemExit(f"[datamcneutron] random_recoclusters_6256__{tag} has no {method} clusters")
    r = r[(r.t_med >= win[0]) & (r.t_med <= win[1])].rename(columns={"f_pe_total": "pe"})
    return r, n


def page_nosource_what(S: Sample, out: Out, rctx: RunContext, rtag: str, win: tuple, kmax: int) -> dict:
    """Run 6264 (no source, same trigger, cosmic veto) against 6256 dummy triggers, per trigger."""
    import matplotlib.pyplot as plt
    r, n_r = _random_clusters(rctx, S.method, win, rtag)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    te = np.arange(win[0] / 1000, win[1] / 1000 + 2.5, 2.5)
    qe = np.arange(0.0, 100.0 + 5.0, 5.0)
    for (c, n, lab, col) in ((S.b_c, S.n_b, f"6264 no-source run, {int(S.n_b):,} triggers", "k"),
                              (r, n_r, f"6256 dummy triggers, {n_r:,} triggers", "#9a9a9a")):
        ax[0].stairs(np.histogram(c.t_med / 1000.0, te)[0] / n / (te[1] - te[0]), te, color=col, lw=1.8,
                     baseline=None, label=f"{lab}: {len(c) / n:.3f} clusters per trigger")
        ax[1].stairs(np.histogram(c.pe, qe)[0] / n, qe, color=col, lw=1.8, baseline=None, label=lab)
        if c is r:
            k = np.bincount(np.minimum(r.groupby("trigger").size().to_numpy(), kmax), minlength=kmax + 1) / n_r
            k[0] = 1 - k[1:].sum()
        else:
            k = pmf(S.b_e.n_accepted_clusters.to_numpy(), kmax)
        ax[2].plot(range(kmax + 1), k, "o-", color=col, label=lab)
    ax[0].set_xlabel("cluster time [µs]")
    ax[0].set_ylabel("clusters per trigger per µs")
    ax[1].set_xlabel("cluster charge [p.e.]")
    ax[1].set_ylabel("clusters per trigger / 5 p.e.")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(kmax + 1), [str(i) for i in range(kmax)] + [f"≥{kmax}"])
    ax[2].set_xlabel("clusters per trigger")
    ax[2].set_ylabel("fraction of triggers")
    for a, t in zip(ax, ("cluster time", "cluster charge", "clusters per trigger")):
        a.set_title(t, fontsize=12)
        a.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: what run 6264 adds per trigger, against random 6256 triggers "
                 f"({win[0] / 1000:g}–{win[1] / 1000:g} µs)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, f"{S.method}_nosource_what")
    return dict(nosource_per_trig=len(S.b_c) / S.n_b, random6256_per_trig=len(r) / n_r, n_trig_6256=n_r)


def page_nosource_subtraction(S: Sample, out: Out, edges=np.arange(0.0, 100.0 + 5.0, 5.0)) -> dict:
    """Data clusters, the scaled no-source run and their difference."""
    import matplotlib.pyplot as plt
    f = S.n_d.sum() / S.n_b
    d = np.histogram(S.d_c.pe, edges)[0].astype(float)
    hb = np.histogram(S.b_c.pe, edges)[0].astype(float)
    sub, vsub = data_minus_bkg(S, "pe", edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.8))
    for a, logy in zip(ax, (False, True)):
        a.errorbar(cen, d, yerr=np.sqrt(d), fmt="o", color="k", ms=4, lw=1, label=f"data, {int(S.n_d.sum()):,} IC triggers")
        a.stairs(hb * f, edges, color="#555555", lw=1.8, baseline=None,
                 label=f"6264 × {f:.1f} ({int(S.n_b):,} triggers scaled to the data)")
        a.errorbar(cen, sub, yerr=np.sqrt(vsub), fmt="o", mfc="white", color="k", ms=4, lw=1,
                   label="data − scaled 6264")
        a.set_xlabel("cluster charge [p.e.]")
        a.set_ylabel(f"clusters / {edges[1] - edges[0]:g} p.e.")
        if logy:
            a.set_yscale("log")
            a.set_ylim(1, None)
        a.legend(frameon=False, fontsize=9)
    j = int(np.searchsorted(edges, 15.0))
    ax[0].set_title(f"bin {edges[j]:g}–{edges[j + 1]:g} p.e.: {d[j]:,.0f} − {hb[j]:,.0f} × {f:.1f} = {sub[j]:,.0f} "
                    f"± {np.sqrt(vsub[j]):,.0f}", fontsize=11)
    ax[1].set_title(f"no-source share of all data clusters: {hb.sum() * f / d.sum():.1%}", fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: data minus the no-source run, bin by bin "
                 f"(error² = data + 6264 × scale²)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, f"{S.method}_nosource_subtraction")
    return dict(scale=f, worked_bin_lo=edges[j], worked_data=d[j], worked_6264=hb[j], worked_sub=sub[j],
                worked_err=float(np.sqrt(vsub[j])), nosource_share=float(hb.sum() * f / d.sum()))


def page_nosource_fold(S: Sample, out: Out, kmax: int, desc: str) -> dict:
    """P(n) = sum_k P_MC(k) P_6264(n - k): the MC neutron-cluster count folded with the no-source count."""
    import matplotlib.pyplot as plt
    pdat, pm, pmb, nd = neutron_mult(S, kmax)
    pb = pmf(S.b_e.n_accepted_clusters.to_numpy(), kmax)
    k = np.arange(kmax + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.8), sharey=True)
    ax[0].plot(k, pm, "s--", color=METHOD_COLOR(S.method), label="P_MC(k): MC neutron clusters per event")
    ax[0].plot(k, pb, "^:", color="#555555", label="P_6264(j): no-source clusters per trigger")
    ax[1].plot(k, pmb, "o-", color=METHOD_COLOR(S.method), label="P(n) = Σ_k P_MC(k) P_6264(n−k)")
    ax[1].errorbar(k, pdat, yerr=np.sqrt(pdat * nd) / nd, fmt="o", color="k", ms=5, label="data, IC triggers")
    for a in ax:
        a.set_yscale("log")
        a.set_ylim(5e-5, 1.5)
        a.set_xticks(k, [str(i) for i in range(kmax)] + [f"≥{kmax}"])
        a.set_xlabel("clusters per trigger / event")
        a.legend(frameon=False, fontsize=9)
    ax[0].set_ylabel("fraction")
    ax[0].set_title(f"inputs ({desc}): P_MC(0) {pm[0]:.3f}, P_MC(1) {pm[1]:.3f}; "
                    f"P_6264(0) {pb[0]:.3f}, P_6264(1) {pb[1]:.3f}", fontsize=11)
    ax[1].set_title(f"P(0) = {pm[0]:.3f}×{pb[0]:.3f} = {pmb[0]:.3f} (data {pdat[0]:.3f}); "
                    f"P(1) = {pm[1]:.3f}×{pb[0]:.3f} + {pm[0]:.3f}×{pb[1]:.3f} = {pmb[1]:.3f} (data {pdat[1]:.3f})",
                    fontsize=11)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: clusters per trigger, the no-source count folded into the MC",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, f"{S.method}_nosource_fold")
    return dict(p_mc0=pm[0], p_mc1=pm[1], p_b0=pb[0], p_b1=pb[1], p_fold0=pmb[0], p_fold1=pmb[1],
                p_data0=pdat[0], p_data1=pdat[1])


def cf_labels_fast(h: pd.DataFrame, cf: pd.DataFrame, lo_ns: float = 5.0, hi_ns: float = 20.0) -> np.ndarray:
    """assign_clusterfinder_labels for many triggers at once: per hit, the index (within its trigger's tree
    clusters, file order) of the nearest cluster with -lo < t - clusterTime < hi, else -1."""
    c = cf[["trigger", "clusterTime"]].copy()
    c["cid"] = c.groupby("trigger").cumcount()
    c = c.sort_values("clusterTime", kind="stable").rename(columns={"clusterTime": "ct"})
    x = h[["trigger", "t"]].reset_index(drop=True).assign(_i=lambda d: np.arange(len(d))).sort_values("t", kind="stable")
    near = {}
    for d in ("backward", "forward"):
        m = pd.merge_asof(x, c, left_on="t", right_on="ct", by="trigger", direction=d).sort_values("_i")
        dt = (m.t - m.ct).to_numpy()
        ok = (dt > -lo_ns) & (dt < hi_ns)
        near[d] = (np.where(ok, np.abs(dt), np.inf), np.nan_to_num(m.cid.to_numpy(float), nan=1e9), ok)
    (ab, cb, okb), (af, cf_, okf) = near["backward"], near["forward"]
    lab = np.where((af < ab) | ((af == ab) & (cf_ < cb)), cf_, cb)
    return np.where(okb | okf, lab, -1).astype(int)


def nms_fast(trigger: np.ndarray, lab: np.ndarray, t: np.ndarray, sep_ns: float) -> np.ndarray:
    """merge_clusters_in_time for many triggers at once (group ids are global, -1 stays -1)."""
    h = pd.DataFrame({"trigger": trigger, "lab": lab, "t": t})
    m = h[h.lab >= 0].groupby(["trigger", "lab"]).t.median().rename("med").reset_index()
    m = m.sort_values(["trigger", "med"], kind="stable")
    new = (m.trigger.diff().fillna(1).to_numpy() != 0) | (m.med.diff().fillna(np.inf).to_numpy() >= sep_ns)
    m["grp"] = np.cumsum(new) - 1
    g = h.merge(m[["trigger", "lab", "grp"]], on=["trigger", "lab"], how="left").grp
    return np.where(lab >= 0, g.fillna(-1).to_numpy(), -1).astype(int)


def masked_candidates(ctx: RunContext, mc: RunContext, gate: str, runs: list, mask: list, win: tuple,
                      hits_out: Optional[list] = None) -> pd.DataFrame:
    """ClusterFinder and CF + NMS candidates of the non-vetoed `gate` triggers of `runs`, rebuilt from the data
    hits with the hits on `mask` removed (empty mask: the production candidates_<gate>). hits_out collects the
    member hits (run, trigger, pmtID, pe, CF + NMS group) for the large-pulse page."""
    from ..mc.capture_truth import DEFAULTS
    t_min = float((mc.extra.get("residual") or {})["prompt_window_ns"])
    sep = float({**DEFAULTS, **(mc.extra.get("capture_truth", {}) or {})}["nms_ns"])
    ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_{gate}_clusterfinder"),
                         columns=["run", "trigger", "eventTankTime", "is_cosmic", "sourceX", "sourceY", "sourceZ"])
    rows = []
    for run in runs:
        e = ei[(ei.run == run) & ~ei.is_cosmic]
        h = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_hits_{run}"), columns=["trigger", "pmtID", "t", "pe"])
        cf = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_cfclusters_{run}"), columns=["trigger", "clusterTime"])
        keep = set(e.trigger)
        h = h[h.trigger.isin(keep) & (h.t > t_min)].reset_index(drop=True)
        cf = cf[cf.trigger.isin(keep)]
        h["cf"] = cf_labels_fast(h, cf)
        h = h[~h.pmtID.isin(mask)].reset_index(drop=True)
        h["cfn"] = nms_fast(h.trigger.to_numpy(), h.cf.to_numpy(), h.t.to_numpy(float), sep)
        for method, col in (("clusterfinder", "cf"), ("clusterfinder_nms", "cfn")):
            c = h[h[col] >= 0].groupby(["trigger", col]).agg(t_med=("t", "median"), pe=("pe", "sum"),
                                                            n_hits=("t", "size")).reset_index()
            c = c[(c.t_med >= win[0]) & (c.t_med <= win[1])]
            rows.append(c.drop(columns=col).assign(method=method, run=run))
            if hits_out is not None and method == "clusterfinder_nms":
                ok = h[col] >= 0
                hits_out.append(h.loc[ok, ["trigger", "pmtID", "pe", "t", col]].merge(
                    c[["trigger", col]], on=["trigger", col]).assign(run=run))
    out = pd.concat(rows, ignore_index=True)
    return out.merge(ei[["run", "trigger", "eventTankTime", "sourceX", "sourceY", "sourceZ"]], on=["run", "trigger"],
                     how="left")


def check_masked_candidates(ctx: RunContext, mc: RunContext, gate: str, runs: list, win: tuple):
    """The unmasked rebuild must reproduce candidates_<gate> cluster by cluster; stops otherwise."""
    a = masked_candidates(ctx, mc, gate, runs, [], win)
    b = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_{gate}"),
                        filters=[("method", "in", ["clusterfinder", "clusterfinder_nms"]), ("run", "in", list(runs))])
    key = ["method", "run", "trigger", "t_med"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    ok = (len(a) == len(b) and (a[["method", "run", "trigger"]].values == b[["method", "run", "trigger"]].values).all()
          and np.allclose(a.t_med, b.t_med) and np.allclose(a.pe, b.pe, rtol=1e-5, atol=1e-3)
          and (a.n_hits.values == b.n_hits.values).all())
    print(f"[datamcneutron] masked-candidate rebuild check, runs {runs}: {len(a):,} rebuilt vs {len(b):,} "
          f"production clusters -> {'IDENTICAL' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("[datamcneutron] the unmasked rebuild does not reproduce the production candidates")


def page_mask(S0: Sample, S1: Sample, out: Out, mask: list, edges=Q_EDGES) -> dict:
    """Hit-corrected MC vs data minus 6264, without and with the hits of `mask` removed in data, 6264 and MC."""
    import matplotlib.pyplot as plt
    lab = LABELS.get(S0.method, S0.method)
    fig, ax = plt.subplots(2, 3, figsize=(18, 7.5), gridspec_kw=dict(height_ratios=[3, 1]), sharex="col")
    res = {}
    for j, (S, tag) in enumerate(((S0, "all PMTs"), (S1, f"PMTs {', '.join(map(str, mask))} removed"))):
        _npanel(ax[0, j], ax[1, j], S, edges, None, "MC neutrons, hit-corrected")
        c2, ndf = neutron_chi2(S)
        md, mm = neutron_means(S, "pe")
        lo = np.searchsorted(edges, 40.0)
        d, _ = data_minus_bkg(S, "pe", edges)
        m, _ = mc_neutron(S, "pe", edges)
        k = slice(np.searchsorted(edges, CHI_EDGES[0]), np.searchsorted(edges, CHI_EDGES[-1]))
        tail = float(d[lo:].sum() / (m[lo:].sum() * d[k].sum() / m[k].sum()))
        ax[0, j].set_title(f"{tag}\nχ²/ndf {c2:.0f}/{ndf}; mean data {md:.2f}, MC {mm:.2f} p.e.; "
                           f"data/MC above 40 p.e. {tail:.2f}", fontsize=10)
        ax[0, j].legend(frameon=False, fontsize=9)
        key = "ref" if j == 0 else "mask"
        res.update({f"chi2_{key}": c2, f"ndf_{key}": ndf, f"mean_data_{key}": md, f"mean_mc_{key}": mm,
                    f"tail40_ratio_{key}": tail,
                    f"neutron_per_trig_data_{key}": len(S.d_c) / S.n_d.sum() - len(S.b_c) / S.n_b})
    _npanel(ax[0, 2], ax[1, 2], S1, edges, None, "MC neutrons, hit-corrected", logy=True)
    ax[0, 2].set_title("PMTs removed, log scale\n", fontsize=10)
    ax[0, 2].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{lab}: cluster charge with the large-pulse PMTs removed in data, no-source run and MC",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.save(fig, f"{S0.method}_charge_mask")
    return res


def page_sub_explain(S: Sample, Sh: Sample, out: Out, col: str, edges: np.ndarray, xlabel: str, unit: str) -> dict:
    """Before / after / per-bin share of the no-source subtraction for one cluster variable."""
    import matplotlib.pyplot as plt
    f = S.n_d.sum() / S.n_b
    d = np.histogram(S.d_c[col].to_numpy(float), edges)[0].astype(float)
    hb = np.histogram(S.b_c[col].to_numpy(float), edges)[0].astype(float)
    sub, vsub = data_minus_bkg(S, col, edges)
    m, _ = mc_neutron(Sh, col, edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].errorbar(cen, d, yerr=np.sqrt(d), fmt="o", color="k", ms=4, lw=1, label=f"data, {int(S.n_d.sum()):,} triggers")
    ax[0].errorbar(cen, hb * f, yerr=np.sqrt(hb) * f, fmt="s", color="0.55", ms=4, lw=1,
                   label=f"6264 × {f:.1f} ({int(S.n_b):,} triggers)")
    ax[0].set_title(f"before: data {len(S.d_c):,} clusters; 6264 {len(S.b_c)} clusters × {f:.1f} = "
                    f"{len(S.b_c) * f:,.0f}", fontsize=11)
    ax[1].errorbar(cen, sub, yerr=np.sqrt(vsub), fmt="o", color="k", ms=4, lw=1, label="data − 6264 × scale")
    ax[1].stairs(m * sub.sum() / m.sum(), edges, color=METHOD_COLOR(S.method), lw=1.6, baseline=None,
                 label="MC neutrons, hit-corrected (same area)")
    ax[1].set_title(f"after: {sub.sum():,.0f} clusters = {sub.sum() / S.n_d.sum():.3f} per trigger", fontsize=11)
    ok = d >= 200
    ax[2].errorbar(cen[ok], (hb * f / d)[ok], yerr=(np.sqrt(hb) * f / d)[ok], fmt="o-", color="k", ms=4, lw=1)
    ax[2].set_ylim(0, None)
    ax[2].set_ylabel("subtracted share of the data bin")
    ax[2].set_title(f"share removed per bin, bins with ≥ 200 data clusters (all: {hb.sum() * f / d.sum():.1%})", fontsize=11)
    for a in ax[:2]:
        a.set_ylabel(f"clusters / {w:g} {unit}".strip())
        a.legend(frameon=False, fontsize=9)
    for a in ax:
        a.set_xlabel(xlabel)
    fig.suptitle(f"{LABELS.get(S.method, S.method)}: data minus the no-source run, {xlabel}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, f"{S.method}_explain_sub_{col}")
    share = np.divide(hb * f, d, out=np.full(len(d), np.nan), where=d > 0)
    return pd.DataFrame(dict(method=S.method, var=col, lo=edges[:-1], hi=edges[1:], data=d, nosource_raw=hb,
                             nosource_scaled=hb * f, share=share, sub=sub, sub_err=np.sqrt(vsub)))


def nosource_time_checks(S: Sample, tau_us: float = 29.4) -> dict:
    """Is 6264 random in time (flat), and does data minus 6264 leave a flat remainder? Poisson fits."""
    from scipy.optimize import curve_fit, minimize
    e = np.arange(TAU_FIT[0], TAU_FIT[1] + 1e-9, 1.0)
    c = 0.5 * (e[1:] + e[:-1])
    b = np.histogram(S.b_c.t_us, e)[0].astype(float)

    def nll(q, x, h, t0, tau):
        mu = np.maximum(q[0] + q[1] * np.exp(-(x - t0) / tau), 1e-12)
        return -(h * np.log(mu) - mu).sum()
    r1 = minimize(nll, [b.mean(), 1.0], args=(c, b, 0.0, tau_us), method="Nelder-Mead")
    r0 = minimize(lambda q: nll([q[0], 0.0], c, b, 0.0, tau_us), [b.mean()], method="Nelder-Mead")
    capt = r1.x[1] * tau_us * (np.exp(-TAU_FIT[0] / tau_us) - np.exp(-TAU_FIT[1] / tau_us))
    ee = np.arange(T_FIT[0], T_FIT[1] + 1e-9, 1.0)
    cc = 0.5 * (ee[1:] + ee[:-1])
    bb = np.histogram(S.b_c.t_us, ee)[0].astype(float)
    s1 = minimize(nll, [bb.mean(), 5.0], args=(cc, bb, T_FIT[0], 2.2), method="Nelder-Mead")
    s0 = minimize(lambda q: nll([q[0], 0.0], cc, bb, T_FIT[0], 2.2), [bb.mean()], method="Nelder-Mead")
    f = S.n_d.sum() / S.n_b
    d = np.histogram(S.d_c.t_us, e)[0].astype(float)
    s, vs = d - b * f, d + b * f * f
    mod = lambda t, a, tau, k: a * np.exp(-t / tau) + k                 # noqa: E731
    p, cov = curve_fit(mod, c, s, p0=[s[0] * 1.4, 30.0, 0.0], sigma=np.sqrt(vs), absolute_sigma=True)
    return dict(method=S.method, n_6264=len(S.b_c), flat_6264_per_trig_us=float(b.sum() / S.n_b / (e[-1] - e[0])),
                capture_like_6264=float(capt), capture_like_6264_of=float(b.sum()),
                capture_like_2dnll=float(2 * (r0.fun - r1.fun)),
                early_6264=float(s1.x[1] * 2.2 * (1 - np.exp(-(T_FIT[1] - T_FIT[0]) / 2.2))),
                early_2dnll=float(2 * (s0.fun - s1.fun)), early_scaled=float(s1.x[1] * 2.2 * f),
                sub_tau=float(p[1]), sub_tau_err=float(np.sqrt(cov[1, 1])), sub_flat=float(p[2]),
                sub_flat_err=float(np.sqrt(cov[2, 2])), subtracted_flat=float(b.sum() * f / (e[-1] - e[0])),
                norm_rel_err=float(1 / np.sqrt(len(S.b_c))))


def page_spe_explain(out: Out, iso: np.ndarray, mcref: np.ndarray, qmap, spe: tuple) -> dict:
    """The single-p.e. quantile map in three steps: the two hit-charge shapes, their cumulative
    distributions, and the map they define."""
    import matplotlib.pyplot as plt
    inw = lambda v: v[(v >= spe[0]) & (v <= spe[1])]                    # noqa: E731
    a, b = inw(iso), inw(mcref)
    edges = np.arange(-0.5, 5.0 + 0.05, 0.1)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    for v, lab, col, ls in ((a, "data, isolated hits (one photon)", "k", "-"), (b, "MC, capture hits", "#2a7ab9", "-"),
                            (qmap(b), "MC, capture hits after the map", "#2a7ab9", "--")):
        h = np.histogram(v, edges)[0]
        ax[0].stairs(h / len(v), edges, color=col, ls=ls, lw=1.7, baseline=None,
                     label=f"{lab}: median {np.median(v):.2f}, mean {v.mean():.2f}")
    ax[0].set_xlabel("hit charge [p.e.]")
    ax[0].set_ylabel("fraction of hits / 0.1 p.e.")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("hit charge, data and MC", fontsize=11)
    x = np.linspace(-0.5, 5.0, 600)
    Fa = np.searchsorted(np.sort(a), x) / len(a)
    Fb = np.searchsorted(np.sort(b), x) / len(b)
    ax[1].plot(x, Fa, color="k", lw=1.7, label="data, isolated hits")
    ax[1].plot(x, Fb, color="#2a7ab9", lw=1.7, label="MC, capture hits")
    us = (0.25, 0.50, 0.75)
    qb, qa = np.quantile(b, us), np.quantile(a, us)
    for u, xb, xa in zip(us, qb, qa):
        ax[1].plot([xb, xa], [u, u], "o-", color="0.45", ms=5, lw=1.2)
    ax[1].set_xlabel("hit charge [p.e.]")
    ax[1].set_ylabel("fraction of hits below")
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    ax[1].set_title("same fraction below → same rank: " + "; ".join(
        f"{int(u * 100)} %: {xb:.2f} → {xa:.2f}" for u, xb, xa in zip(us, qb, qa)), fontsize=10)
    xx = np.linspace(spe[0], spe[1], 400)
    ax[2].plot(xx, qmap(xx), color="k", lw=1.7)
    ax[2].plot(qb, qa, "o", color="0.45", ms=5)
    ax[2].set_xlabel("MC hit charge [p.e.]")
    ax[2].set_ylabel("corrected MC hit charge [p.e.]")
    ax[2].set_title(f"the map, applied to every MC hit in {spe[0]:g}–{spe[1]:g} p.e.", fontsize=11)
    fig.suptitle("Hit-charge correction of the MC: give each MC hit the data charge of the same rank", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, "explain_spe_map")
    return dict(q25_mc=qb[0], q25_data=qa[0], q50_mc=qb[1], q50_data=qa[1], q75_mc=qb[2], q75_data=qa[2],
                mean_mc=float(b.mean()), mean_mc_mapped=float(qmap(b).mean()), mean_data=float(a.mean()),
                n_iso=len(iso), n_mc=len(mcref))


def page_lp_explain(hits: pd.DataFrame, out: Out, mask: list, q_big: float = 8.0) -> pd.DataFrame:
    """Large pulses inside data CF + NMS clusters: hit charge on each masked PMT against all other PMTs, per run."""
    import matplotlib.pyplot as plt
    oth = hits[~hits.pmtID.isin(mask)]
    rows = []
    for run, g in hits.groupby("run"):
        ncl = g.groupby(["trigger", "cfn"]).ngroups
        o = g[~g.pmtID.isin(mask)]
        for p in list(mask) + ["others"]:
            x = o if p == "others" else g[g.pmtID == p]
            rows.append(dict(run=run, pmt=str(p), n_hits=len(x), median_pe=float(x.pe.median()),
                             mean_pe=float(x.pe.mean()), frac_big=float((x.pe > q_big).mean()),
                             hits_per_cluster=len(x) / ncl / (o.pmtID.nunique() if p == "others" else 1)))
        t = g[g.pmtID.isin(mask)]
        rows.append(dict(run=run, pmt="per cluster", n_hits=ncl, hits_per_cluster=len(t) / ncl,
                         pe_mask_per_cluster=float(t.pe.sum() / ncl),
                         extra_pe_per_cluster=float((t.pe.sum() - len(t) * o.pe.mean()) / ncl),
                         mean_cluster_pe=float(g.pe.sum() / ncl)))
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
    edges = np.arange(0.0, 30.0 + 0.5, 0.5)
    for p in mask:
        x = hits.pe[hits.pmtID == p]
        ax[0].stairs(np.histogram(x, edges)[0] / len(x), edges, lw=1.4, baseline=None,
                     label=f"PMT {p}: median {x.median():.2f}, above {q_big:g} p.e. {(x > q_big).mean():.1%}")
    ax[0].stairs(np.histogram(oth.pe, edges)[0] / len(oth), edges, color="k", lw=1.8, baseline=None,
                 label=f"all other PMTs: median {oth.pe.median():.2f}, above {q_big:g} p.e. {(oth.pe > q_big).mean():.1%}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("hit charge in a CF + NMS cluster [p.e.]")
    ax[0].set_ylabel("fraction of hits / 0.5 p.e.")
    ax[0].legend(frameon=False, fontsize=8.5)
    ax[0].set_title("hits inside data clusters", fontsize=11)
    runs = sorted(tab.run.unique())
    for p in list(mask) + ["others"]:
        r = tab[tab.pmt == str(p)].set_index("run").reindex(runs)
        ax[1].plot([str(v) for v in runs], r.frac_big, "o-", lw=1.4 if p != "others" else 2.0,
                   color=None if p != "others" else "k", label=f"PMT {p}" if p != "others" else "all other PMTs")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("data run")
    ax[1].set_ylabel(f"fraction of cluster hits above {q_big:g} p.e.")
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title("per run", fontsize=11)
    fig.suptitle("Data large pulses inside neutron-like clusters (CF + NMS, runs of the hit-charge reference)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, "explain_large_pulses")
    return tab


def page_mc_anatomy(mc: RunContext, methods: list, tag: str, out: Out, win: tuple) -> pd.DataFrame:
    """AmBe MC neutrons (nominal MC, one neutron per event): hits per capture vs hits in its cluster,
    where the neutrons capture, and what the MC clusters hold."""
    import matplotlib.pyplot as plt
    caps = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__capture_truth_captures__{tag}"))
    tc = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__true_captures"),
                         columns=["_source_file", "eventNumber", "cap_idx", "cap_x", "cap_y", "cap_z"])
    caps = caps.merge(tc, on=["_source_file", "eventNumber", "cap_idx"], how="left")
    m0 = "clusterfinder_nms" if "clusterfinder_nms" in methods else methods[0]
    ev = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__eventindex_mc_{m0}"),
                         columns=["_source_file", "eventNumber", "is_cosmic", "sourceX", "sourceY", "sourceZ"])
    ev = ev[~ev.is_cosmic].drop_duplicates(["_source_file", "eventNumber"]).drop(columns="is_cosmic")
    caps = caps.merge(ev, on=["_source_file", "eventNumber"], how="inner")
    caps["r"] = np.hypot(caps.cap_x, caps.cap_z)
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.8))
    e = np.arange(-0.5, 50.5, 1.0)
    ax[0].stairs(np.histogram(caps.n_light_hits, e)[0] / len(caps), e, color="k", lw=1.8, baseline=None,
                 label=f"MCHits made by the capture: mean {caps.n_light_hits.mean():.1f}")
    parts = []
    for m in methods:
        h = caps[f"hits_in_owners_{m}"]
        none = (caps[f"n_owners_{m}"] == 0).mean()
        ax[0].stairs(np.histogram(h, e)[0] / len(caps), e, color=METHOD_COLOR(m), lw=1.6, baseline=None,
                     label=f"hits in its {LABELS.get(m, m)} cluster: mean {h.mean():.1f}")
        parts.append(f"{LABELS.get(m, m)} {none:.1%}")
        rows.append(dict(method=m, captures=len(caps), no_light=float((caps.n_light_hits == 0).mean()),
                         visible=float(caps.visible.mean()), no_cluster=float(none),
                         visible_no_cluster=float(((caps[f"n_owners_{m}"] == 0) & (caps.visible == 1)).mean()),
                         mean_capture_hits=float(caps.n_light_hits.mean()), mean_cluster_hits=float(h.mean()),
                         completeness=float(h[caps.n_light_hits > 0].sum() / caps.n_light_hits.sum())))
    ax[0].set_xlabel("hits")
    ax[0].set_ylabel("fraction of captures")
    ax[0].set_title("captures with no cluster: " + ", ".join(parts), fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)
    hh = ax[1].hist2d(caps.n_light_hits, caps[f"hits_in_owners_{m0}"], bins=[e, e], cmin=1, cmap="viridis",
                      norm=__import__("matplotlib.colors", fromlist=["LogNorm"]).LogNorm())
    fig.colorbar(hh[3], ax=ax[1], label="captures")
    ax[1].set_xlabel("MCHits made by the capture")
    ax[1].set_ylabel(f"hits in its {LABELS.get(m0, m0)} cluster")
    ax[1].set_title(f"{LABELS.get(m0, m0)}: hits kept {rows[methods.index(m0)]['completeness']:.1%}", fontsize=11)
    fig.suptitle(f"AmBe MC neutrons (nominal MC, {len(caps):,} captures in {win[0] / 1000:g}–{win[1] / 1000:g} µs): "
                 f"capture light vs what the cluster keeps", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, "mc_anatomy_hits")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    cl = caps[f"n_owners_{m0}"] > 0
    groups = (("all captures", np.ones(len(caps), bool), "k", "-"),
              (f"with a {LABELS.get(m0, m0)} cluster", cl.to_numpy(), METHOD_COLOR(m0), "-"),
              ("≥ 5 hits but no cluster", ((caps.visible == 1) & ~cl).to_numpy(), "#d97a00", "--"),
              ("1–4 hits, no cluster", ((caps.n_light_hits > 0) & (caps.visible == 0) & ~cl).to_numpy(), "#e87ba4", "-."),
              ("no light (0 hits)", (caps.n_light_hits == 0).to_numpy(), "#9a9a9a", ":"))
    for a, col, edges, xl in ((ax[0], "r", np.arange(0, 1.62, 0.05), "capture radius √(x² + z²) [m]"),
                              (ax[1], "cap_y", np.arange(-2.0, 2.05, 0.1), "capture height y [m]")):
        for lab, msk, c, ls in groups:
            a.stairs(np.histogram(caps[col][msk], edges)[0], edges, color=c, ls=ls, lw=1.6, baseline=None,
                     label=f"{lab}: {msk.mean():.1%}")
        a.set_xlabel(xl)
        a.set_ylabel("captures")
        a.set_yscale("log")
        a.legend(frameon=False, fontsize=9)
    med = caps.cap_medium.fillna("other")
    lab_m = ["Gd", "H", "other"]
    x = np.arange(len(lab_m))
    for k_, (lab, msk, c, _) in enumerate(groups):
        v = [((med == mm) & msk).sum() / max(msk.sum(), 1) for mm in lab_m]
        ax[2].bar(x + (k_ - (len(groups) - 1) / 2) * 0.16, v, 0.15, color=c, label=lab)
    ax[2].set_xticks(x, lab_m)
    ax[2].set_ylabel("fraction of the group")
    ax[2].set_title("capture nucleus", fontsize=12)
    ax[2].legend(frameon=False, fontsize=9)
    ax[0].set_title("radius (tank wall at 1.52 m)", fontsize=12)
    ax[1].set_title("height", fontsize=12)
    fig.suptitle(f"Where AmBe MC neutrons capture, and which captures leave no {LABELS.get(m0, m0)} cluster "
                 f"(nominal MC, all 25 positions)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, "mc_anatomy_where")

    fig, ax = plt.subplots(len(PORTS), len(HEIGHTS), figsize=(22, 16), sharex=True)
    caps["pid"] = pos_id(caps)
    edges = np.arange(0, 1.62, 0.1)
    for p in range(NPOS):
        a = ax.flat[p]
        g = caps[caps.pid == p]
        clp = g[f"n_owners_{m0}"] > 0
        a.stairs(np.histogram(g.r, edges)[0], edges, color="k", lw=1.4, baseline=None, label="all captures")
        a.stairs(np.histogram(g.r[~clp], edges)[0], edges, color="#d97a00", lw=1.4, ls="--", baseline=None,
                 label=f"no {LABELS.get(m0, m0)} cluster")
        a.set_title(f"{pos_title(p)}: no cluster {(~clp).mean():.1%}", fontsize=10)
        a.set_yscale("log")
    for a in ax[-1]:
        a.set_xlabel("capture radius [m]")
    h, lab = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Capture radius per source position, AmBe MC neutrons (nominal MC)", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.save(fig, "mc_anatomy_where_grid")

    cand = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__candidates_mc"),
                           filters=[("method", "in", methods)],
                           columns=["_source_file", "eventNumber", "method", "pe", "n_hits", "n_overlay", "cap_dom"])
    cand = cand.merge(ev[["_source_file", "eventNumber"]], on=["_source_file", "eventNumber"], how="inner")
    nev = len(ev)
    kind = np.where(cand.cap_dom >= 0, "neutron capture",
                    np.where(cand.n_overlay * 2 >= cand.n_hits, "6256 noise", "other MC light"))
    cand["kind"] = kind
    kinds = (("neutron capture", "k", "-"), ("6256 noise", "#9a9a9a", "--"), ("other MC light", "#d97a00", ":"))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    x = np.arange(len(methods))
    for k_, (kd, c, ls) in enumerate(kinds):
        v = [((cand.method == m) & (cand.kind == kd)).sum() / nev for m in methods]
        ax[0].bar(x + (k_ - 1) * 0.27, v, 0.25, color=c, label=kd)
        for m in methods:
            rows.append(dict(method=m, kind=kd, clusters_per_event=float(((cand.method == m) & (cand.kind == kd)).sum() / nev)))
    ax[0].set_xticks(x, [LABELS.get(m, m) for m in methods])
    ax[0].set_ylabel("clusters per MC event")
    ax[0].set_title("clusters per event, by what they hold", fontsize=12)
    ax[0].legend(frameon=False, fontsize=9)
    m1 = "peak" if "peak" in methods else m0
    g = cand[cand.method == m1]
    for a, col, edges, xl in ((ax[1], "n_hits", np.arange(-0.5, 50.5, 1.0), "hits per cluster"),
                              (ax[2], "pe", np.arange(0, 100.0 + 2.5, 2.5), "cluster charge [p.e.]")):
        for kd, c, ls in kinds:
            v = g[g.kind == kd][col]
            if not len(v):
                continue
            a.stairs(np.histogram(v, edges)[0], edges, color=c, ls=ls, lw=1.6, baseline=None,
                     label=f"{kd}: mean {v.mean():.1f}")
        a.set_xlabel(xl)
        a.set_ylabel("clusters")
        a.set_yscale("log")
        a.set_title(f"{LABELS.get(m1, m1)} (the only clusterer here that can build 6256-noise clusters)", fontsize=11)
        a.legend(frameon=False, fontsize=9)
    fig.suptitle(f"What the AmBe MC clusters hold ({win[0] / 1000:g}–{win[1] / 1000:g} µs, one 6256 trigger "
                 f"overlaid per event, nominal MC)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.save(fig, "mc_anatomy_clusters")
    return pd.DataFrame(rows)


def run_neutron(ctx: RunContext, mc: RunContext, methods: list, hc_tag: str, kmax: int = 4, out_suffix: str = "",
                nosource_rctx: Optional[RunContext] = None, randoms_tag: str = "", anatomy: bool = False,
                explain: bool = False, mask_pmts: Optional[list] = None, both_grids: bool = False):
    import json
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    bkg_run = int((ctx.fit_params or {})["background_run"])
    win = tuple(float(v) for v in (mc.extra.get("clusterindex") or {})["window_ns"])
    hpath = mc.parquet_path(f"{mc.run_name}__candidates_hitcorr__{hc_tag}")
    side = hpath.with_suffix(".json")
    if not (hpath.exists() and side.exists()):
        raise SystemExit(f"[datamcneutron] {hpath.name} missing: run `ambe plots datamcclust --hit-correction "
                         f"spe_deadpmt` first")
    meta = json.loads(side.read_text())
    if set(methods) - set(meta["methods"]):
        raise SystemExit(f"[datamcneutron] {sorted(set(methods) - set(meta['methods']))} not in {side.name}")
    hcl = pd.read_parquet(hpath)
    print(f"[datamcneutron] hit correction {hpath.name}: {meta}")
    base = "datamcneutron" + (f"_{out_suffix}" if out_suffix else "")
    odir = ctx.plots_dir / base
    summary, ports, grids, mults, maps, nos = [], [], [], [], {}, []
    subs, tchecks, masks, folds = [], [], [], []
    q_sum = h_sum = 0.0
    hcm = dmask = None
    if mask_pmts:
        mpath = mc.parquet_path(f"{mc.run_name}__candidates_hitcorr__{hc_tag}{mask_suffix(mask_pmts)}")
        mside = mpath.with_suffix(".json")
        if not (mpath.exists() and mside.exists()):
            raise SystemExit(f"[datamcneutron] {mpath.name} missing: run `ambe plots datamcclust --hitcorr-only "
                             f"--mask-pmts {','.join(map(str, mask_pmts))}` first")
        mmeta = json.loads(mside.read_text())
        if sorted(mmeta.get("mask_pmts", [])) != sorted(mask_pmts):
            raise SystemExit(f"[datamcneutron] {mside.name} masks {mmeta.get('mask_pmts')}, not {sorted(mask_pmts)}")
        hcm = pd.read_parquet(mpath)
        ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_ic_clusterfinder"), columns=["run"])
        druns = sorted(r for r in ei.run.unique() if r != bkg_run)
        check_masked_candidates(ctx, mc, "ic", druns, win)
        check_masked_candidates(ctx, mc, "none", [bkg_run], win)
        dmask = (masked_candidates(ctx, mc, "ic", druns, list(mask_pmts), win),
                 masked_candidates(ctx, mc, "none", [bkg_run], list(mask_pmts), win))
    with PdfPages(ctx.plot_path(base, "pdf")) as pdf:
        out = Out(ctx, pdf)
        out.dir = odir
        odir.mkdir(parents=True, exist_ok=True)
        for m in methods:
            S0 = load(ctx, mc, m, False, bkg_run)
            S0.m_c = S0.m_c.reset_index(drop=True)
            S1 = hitcorr_sample(S0, hcl, win)
            nd = S0.n_d.sum()
            row = dict(method=m, n_trig_data=int(nd), n_trig_bkg=int(S0.n_b), n_events_mc=len(S0.m_e),
                       clusters_per_trig_data=len(S0.d_c) / nd,
                       clusters_per_trig_bkg=len(S0.b_c) / S0.n_b,
                       clusters_per_trig_data_sub=len(S0.d_c) / nd - len(S0.b_c) / S0.n_b,
                       neutron_clusters_per_event_mc=float(S0.m_c.w[S0.m_c.cap].sum() / nd),
                       neutron_clusters_per_event_mc_hitcorr=float(S1.m_c.w[S1.m_c.cap.astype(bool)].sum() / nd),
                       bkg_share=(len(S0.b_c) / S0.n_b) / (len(S0.d_c) / nd),
                       mc_noncapture_share=float(S0.m_c.w[~S0.m_c.cap].sum() / S0.m_c.w.sum()))
            row.update(page_ncharge(S0, S1, out))
            row.update(page_nhits(S0, S1, out))
            ports.append(page_nports(S1, out, f"{m}_ports_hitcorr", "hit-corrected MC").assign(method=m, mc="hitcorr"))
            ports.append(page_nports(S0, out, f"{m}_ports_nominal", "nominal MC").assign(method=m, mc="nominal"))
            g1 = page_ngrid(S1, out, f"{m}_grid_hitcorr", "hit-corrected MC").assign(method=m, mc="hitcorr")
            g0 = page_ngrid(S0, out, f"{m}_grid_nominal", "nominal MC").assign(method=m, mc="nominal")
            grids += [g1, g0]
            maps[m] = (g0, g1)
            row.update(page_nmult(S0, S1, out, kmax))
            if both_grids:
                page_ngrid_both(S0, S1, out, f"{m}_grid_both")
                page_nmult_grid_both(S0, S1, out, kmax, f"{m}_mult_grid_both")
            for S, tag, desc in ((S1, "hitcorr", "hit-corrected MC"), (S0, "nominal", "nominal MC")):
                mults.append(page_nmult_grid(S, out, kmax, f"{m}_mult_grid_{tag}", desc).assign(method=m, mc=tag))
            if explain:
                for col, edges, xl, unit in (("pe", np.arange(0.0, 120.0 + 5.0, 5.0), "cluster charge [p.e.]", "p.e."),
                                             ("n_hits", np.arange(4.5, 40.5 + 1.0, 1.0), "hits per cluster", ""),
                                             ("t_us", np.arange(2.0, 67.0 + 2.5, 2.5), "cluster time [µs]", "µs")):
                    subs.append(page_sub_explain(S0, S1, out, col, edges, xl, unit))
                tchecks.append(nosource_time_checks(S0))
                folds.append(page_fold_explain(S1, out, kmax))
                q_sum, h_sum = q_sum + float(S0.m_c.pe.sum()), h_sum + float(S0.m_c.n_hits.sum())
            if dmask is not None and m in ("clusterfinder", "clusterfinder_nms"):
                dm, bm = (x[x.method == m].reset_index(drop=True) for x in dmask)
                for df in (dm, bm):
                    df["t_us"] = df.t_med / 1000.0
                    df["pe_per_hit"] = df.pe / df.n_hits
                dm["pid"] = pos_id(dm)
                Sm = hitcorr_sample(_with(S0, d_c=dm, b_c=bm), hcm, win)
                masks.append(dict(method=m, mask=" ".join(map(str, sorted(mask_pmts))), **page_mask(S1, Sm, out, mask_pmts)))
            if nosource_rctx is not None:
                nr = dict(method=m)
                nr.update(page_nosource_what(S0, out, nosource_rctx, randoms_tag, win, kmax))
                nr.update(page_nosource_subtraction(S0, out))
                nr.update(page_nosource_fold(S1, out, kmax, "hit-corrected MC"))
                nos.append(nr)
            summary.append(row)
        page_nmaps(maps, out)
        if explain:
            iso, mcref = spe_reference(ctx, mc, meta["hit_runs"], methods[0])
            sp = tuple(meta["spe"])
            ex = page_spe_explain(out, iso, mcref, spe_map(mcref, iso, *sp), sp)
            page_hitcorr(out, iso, mcref, spe_map(mcref, iso, *sp), meta["dead"], q_sum / h_sum, sp)
            page_hitcorr(out, iso, mcref, spe_map(mcref, iso, *sp), meta["dead"], q_sum / h_sum, sp, slide=True)
            pd.Series(ex).to_csv(ctx.csv_path(f"{base}_explain_spe"), header=["value"])
            if mask_pmts:
                hh = []
                masked_candidates(ctx, mc, "ic", [int(r) for r in meta["hit_runs"]], [], win, hits_out=hh)
                page_lp_explain(pd.concat(hh, ignore_index=True), out, list(mask_pmts)).to_csv(
                    ctx.csv_path(f"{base}_explain_large_pulses"), index=False)
        if anatomy:
            page_mc_anatomy(mc, methods, meta["tag"], out, win).to_csv(ctx.csv_path(f"{base}_mc_anatomy"), index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(ctx.csv_path(f"{base}_summary"), index=False)
    pd.concat(ports, ignore_index=True).to_csv(ctx.csv_path(f"{base}_by_port"), index=False)
    pd.concat(grids, ignore_index=True).to_csv(ctx.csv_path(f"{base}_by_position"), index=False)
    pd.concat(mults, ignore_index=True).to_csv(ctx.csv_path(f"{base}_mult_by_position"), index=False)
    if nos:
        pd.DataFrame(nos).to_csv(ctx.csv_path(f"{base}_nosource"), index=False)
    if subs:
        pd.concat(subs, ignore_index=True).to_csv(ctx.csv_path(f"{base}_explain_subtraction"), index=False)
        pd.DataFrame(tchecks).to_csv(ctx.csv_path(f"{base}_explain_nosource_time"), index=False)
        pd.DataFrame(folds).to_csv(ctx.csv_path(f"{base}_explain_fold"), index=False)
        print(pd.DataFrame(tchecks).round(4).T.to_string())
    if masks:
        pd.DataFrame(masks).to_csv(ctx.csv_path(f"{base}_mask"), index=False)
        print(pd.DataFrame(masks).round(4).T.to_string())
    pd.set_option("display.width", 250)
    print(sm.round(4).T.to_string())
    if nos:
        print(pd.DataFrame(nos).round(4).T.to_string())
    print(f"[datamcneutron] wrote {ctx.plot_path(base, 'pdf')} and {odir}")


def cli_neutron(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots datamcneutron")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--methods", required=True, help="clusterers to compare, e.g. clusterfinder_nms,peak")
    p.add_argument("--hitcorr-tag", required=True,
                   help="tag of the cached hit-corrected MC from `ambe plots datamcclust --hit-correction spe_deadpmt`")
    p.add_argument("--kmax", type=int, default=4)
    p.add_argument("--out-suffix", default="", help="write plots/datamcneutron_<suffix> and csv datamcneutron_<suffix>_*")
    p.add_argument("--nosource-randoms", help="`ambe data randoms` config with the 6256 dummy triggers: adds the "
                                              "pages on how the no-source run enters (with --randoms-tag)")
    p.add_argument("--randoms-tag", help="tag of the `ambe data randoms --cluster` run of --nosource-randoms")
    p.add_argument("--anatomy", action="store_true", help="add the AmBe MC neutron pages (hits, position, contents)")
    p.add_argument("--explain", action="store_true",
                   help="add the explanation pages: the no-source subtraction bin by bin (charge, hits, time) with "
                        "its time checks, the single-p.e. map, and (with --mask-pmts) the data large pulses")
    p.add_argument("--mask-pmts", help="comma list: also compare with these PMTs removed in data, 6264 and MC "
                                       "(ClusterFinder and CF + NMS; needs the matching datamcclust --mask-pmts cache)")
    p.add_argument("--both-grids", action="store_true",
                   help="add 25-position charge and clusters-per-trigger grids with both MC versions in each panel")
    a = p.parse_args(list(argv) if argv else [])
    if a.nosource_randoms and not a.randoms_tag:
        p.error("--nosource-randoms needs --randoms-tag")
    run_neutron(ctx, load_context(a.mc_config), a.methods.split(","), a.hitcorr_tag, a.kmax, a.out_suffix,
                load_context(a.nosource_randoms) if a.nosource_randoms else None, a.randoms_tag or "", a.anatomy,
                a.explain, [int(v) for v in a.mask_pmts.split(",")] if a.mask_pmts else None, a.both_grids)
