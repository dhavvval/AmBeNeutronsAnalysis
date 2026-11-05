"""
AmBe Data/MC comparison of re-clustered candidates, every port x height.

    ambe plots datamc --config <data cfg> --mc-config <mc cfg> --gate ic|none --methods a,b,...

Reads the `ambe stats clusterindex` tables of both configs. Per method: one page of cluster
charge (MC scaled to the data entries in the plotted range, chi2 = sum (d-m)^2/(sd^2+sm^2),
ndf = filled bins - 1, as MC_tuning/Michel_tuning.py) and one page of candidate multiplicity
per trigger. Writes <data plots>/datamc_<gate>.pdf and <data csv>/datamc_chi2_<gate>.csv.
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context

PORTS = [(5, (0.0, 0.0)), (1, (0.0, -75.0)), (4, (75.0, 0.0)), (2, (0.0, 75.0)), (3, (0.0, 102.0))]
HEIGHTS = [100.0, 50.0, 0.0, -50.0, -100.0]
LABELS = {"clusterfinder": "ClusterFinder", "clusterfinder_nms": "ClusterFinder + NMS",
          "peak": "time-peak", "optics": "OPTICS", "optics_nms": "OPTICS + NMS"}


def chi2(d: np.ndarray, m_raw: np.ndarray, scale: float):
    m = m_raw * scale
    var = d + m_raw * scale ** 2
    ok = var > 0
    return float(((d - m)[ok] ** 2 / var[ok]).sum()), int(ok.sum()) - 1


def _at(df: pd.DataFrame, x: float, y: float, z: float) -> pd.DataFrame:
    return df[np.isclose(df.sourceX, x) & np.isclose(df.sourceY, y) & np.isclose(df.sourceZ, z)]


def run(ctx: RunContext, mc: RunContext, gate: str, methods: list, edges: np.ndarray, kmax: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ..plotting import METHOD_COLORS

    cd = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_{gate}"))
    cm = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__candidates_mc"))
    rows = []
    pdf_path = ctx.plot_path(f"datamc_{gate}", "pdf")
    with PdfPages(pdf_path) as pdf:
        for m in methods:
            ed = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_{gate}_{m}"))
            em = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__eventindex_mc_{m}"))
            ed, em = ed[~ed.is_cosmic], em[~em.is_cosmic]
            col = METHOD_COLORS.get(m, "#4a3aa7")
            fq, aq = plt.subplots(5, 5, figsize=(22, 16))
            fk, ak = plt.subplots(5, 5, figsize=(22, 16))
            for i, (port, (x, z)) in enumerate(PORTS):
                for j, y in enumerate(HEIGHTS):
                    title = f"Port {port}, Y = {y:+.0f} cm"
                    d = _at(cd[cd.method == m], x, y, z).pe.to_numpy()
                    s = _at(cm[cm.method == m], x, y, z).pe.to_numpy()
                    nd, _ = np.histogram(d, edges)
                    ns, _ = np.histogram(s, edges)
                    ax = aq[i, j]
                    if nd.sum() == 0 or ns.sum() == 0:
                        ax.set_title(f"{title}: no {'data' if nd.sum() == 0 else 'MC'}", fontsize=10)
                        continue
                    sc = nd.sum() / ns.sum()
                    c2, ndf = chi2(nd.astype(float), ns.astype(float), sc)
                    ax.stairs(ns * sc, edges, color=col, lw=1.5)
                    cen = 0.5 * (edges[1:] + edges[:-1])
                    ax.errorbar(cen, nd, yerr=np.sqrt(nd), fmt="o", color="k", ms=4, lw=1)
                    ax.set_title(f"{title}  χ²/ndf = {c2:.0f}/{ndf}, N = {nd.sum():,}", fontsize=10)
                    ax.set_xlabel("Cluster charge [p.e.]")
                    ax.set_ylabel(f"clusters / {edges[1] - edges[0]:g} p.e. bin")
                    ax.set_xlim(0, edges[-1] + 10)
                    ax.set_ylim(0, None)

                    kd = np.bincount(np.minimum(_at(ed, x, y, z).n_accepted_clusters, kmax), minlength=kmax + 1)
                    km = np.bincount(np.minimum(_at(em, x, y, z).n_accepted_clusters, kmax), minlength=kmax + 1)
                    fd, fm = kd / kd.sum(), km / km.sum()
                    a2 = ak[i, j]
                    a2.stairs(fm, np.arange(kmax + 2) - 0.5, color=col, lw=1.5)
                    a2.errorbar(np.arange(kmax + 1), fd, yerr=np.sqrt(kd) / kd.sum(), fmt="o", color="k", ms=4, lw=1)
                    a2.set_yscale("log")
                    a2.set_xticks(range(kmax + 1), [str(k) for k in range(kmax)] + [f"≥{kmax}"])
                    a2.set_title(f"{title}  ≥1: data {1 - fd[0]:.3f}, MC {1 - fm[0]:.3f}", fontsize=10)
                    a2.set_xlabel("candidates per trigger")
                    a2.set_ylabel("fraction of triggers")
                    rows.append(dict(method=m, gate=gate, port=port, x=x, y=y, z=z,
                                     n_data=int(nd.sum()), n_mc=int(ns.sum()), chi2=c2, ndf=ndf,
                                     mean_pe_data=float(d.mean()), mean_pe_mc=float(s.mean()),
                                     median_pe_data=float(np.median(d)), median_pe_mc=float(np.median(s)),
                                     n_trig_data=int(kd.sum()), n_trig_mc=int(km.sum()),
                                     frac_ge1_data=float(1 - fd[0]), frac_ge1_mc=float(1 - fm[0]),
                                     mean_mult_data=float(_at(ed, x, y, z).n_accepted_clusters.mean()),
                                     mean_mult_mc=float(_at(em, x, y, z).n_accepted_clusters.mean())))
            g = "IC 700-1200 triggers" if gate == "ic" else "all triggers"
            fq.suptitle(f"{LABELS.get(m, m)}: cluster charge, data (points, {g}) vs MC (line, scaled to data)",
                        fontsize=15)
            fk.suptitle(f"{LABELS.get(m, m)}: candidates per trigger, data (points, {g}) vs MC (line)",
                        fontsize=15)
            for f in (fq, fk):
                f.tight_layout(rect=(0, 0, 1, 0.97))
                pdf.savefig(f)
                plt.close(f)
    out = pd.DataFrame(rows)
    out.to_csv(ctx.csv_path(f"datamc_chi2_{gate}"), index=False)
    print(f"[datamc] wrote {pdf_path}")
    print(out.groupby("method")[["chi2", "ndf"]].sum().assign(
        chi2_per_ndf=lambda d: d.chi2 / d.ndf).round(2).to_string())


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots datamc")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--gate", choices=["ic", "none"], required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--pe-bins", default="5,65,5", help="lo,hi,width of the charge bins (p.e.)")
    p.add_argument("--kmax", type=int, default=4, help="top multiplicity bin (>= kmax)")
    a = p.parse_args(list(argv) if argv else [])
    lo, hi, w = (float(v) for v in a.pe_bins.split(","))
    run(ctx, load_context(a.mc_config), a.gate, a.methods.split(","),
        np.arange(lo, hi + w / 2, w), a.kmax)
