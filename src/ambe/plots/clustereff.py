"""
Figures across the `ambe stats clustereff` fit sets: data vs MC truth vs a reference efficiency,
the MC closure, and the Data/MC charge chi2 against each run's hit rate.

    ambe plots clustereff --config <data cfg> --mc-config <mc cfg> --methods a,b [--overlay-page]

Reads efficiency_fit_by_position_{ic,none}_<m> (data), efficiency_fit_by_position_mc_<m> (MC),
datamc_chi2_<gate> (`ambe plots datamc`) and clusterindex.reference_efficiency (data config).
--overlay-page also redraws the MC charge with and without clusterindex.overlay_veto (slow).
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..plotting import METHOD_COLORS, save_plot, set_style
from .datamc import LABELS
from .heatmap import PORT_INFO, PORT_ORDER

DUR_US = 70.0


def _positions(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.level == "position"].copy()
    d["port"] = [PORT_INFO.get((x, y, z), "Unknown") for x, y, z in zip(d.x_pos, d.y_pos, d.z_pos)]
    return d


def _err(d, tag):
    return [d[f"{tag}_eff"] - d[f"{tag}_eff_lo"], d[f"{tag}_eff_hi"] - d[f"{tag}_eff"]]


def _ports(d):
    return [p for p in PORT_ORDER if (d.port == p).any()]


def _reference(ctx, data):
    ref = (ctx.extra.get("clusterindex") or {}).get("reference_efficiency")
    if not ref:
        return None, None
    r = pd.read_csv(ref)
    runs = {" ".join(sorted(str(row.runs).split())): row for _, row in r.iterrows()}
    rows = [runs.get(" ".join(sorted(str(x).split()))) for x in data.runs]
    out = data[["port", "y_pos"]].copy()
    out["eff"] = [np.nan if row is None else row.dds_eff for row in rows]
    return out, (ctx.extra.get("clusterindex") or {}).get("reference_label", "reference")


def vs_y(ctx, mc, m):
    import matplotlib.pyplot as plt
    ic = _positions(ctx.csv_path(f"efficiency_fit_by_position_ic_{m}"))
    no = _positions(ctx.csv_path(f"efficiency_fit_by_position_none_{m}"))
    sim = _positions(mc.csv_path(f"efficiency_fit_by_position_mc_{m}"))
    ref, ref_label = _reference(ctx, ic)
    ports = _ports(ic)
    fig, axes = plt.subplots(1, len(ports), figsize=(3.7 * len(ports), 4.8), sharey=True)
    col = METHOD_COLORS.get(m, "C0")
    for ax, port in zip(axes, ports):
        a, b, s = (d[d.port == port].sort_values("y_pos") for d in (ic, no, sim))
        ax.plot(s.y_pos, 100 * s.truth_eff, "-", color=col, lw=2, label="MC truth")
        ax.errorbar(a.y_pos - 3, 100 * a.dds_eff, yerr=100 * np.array(_err(a, "dds")), fmt="o",
                    color="k", ms=6, capsize=2, label="data, IC triggers")
        ax.errorbar(b.y_pos + 3, 100 * b.dds_eff, yerr=100 * np.array(_err(b, "dds")), fmt="o",
                    mfc="none", color="0.45", ms=6, capsize=2, label="data, all triggers")
        if ref is not None:
            r = ref[ref.port == port].sort_values("y_pos")
            ax.plot(r.y_pos, 100 * r.eff, "s", color="C3", ms=6, label=ref_label)
        ax.set_title(port)
        ax.set_xlabel("y position (cm)")
    axes[0].set_ylabel("Efficiency (%)")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"{LABELS.get(m, m)}: neutron detection efficiency, data-driven fit")
    fig.tight_layout()
    save_plot(fig, ctx, f"clustereff_vs_y_{m}")


def ratio(ctx, mc, methods):
    import matplotlib.pyplot as plt
    frames = {}
    for m in methods:
        a = _positions(ctx.csv_path(f"efficiency_fit_by_position_ic_{m}"))
        s = _positions(mc.csv_path(f"efficiency_fit_by_position_mc_{m}"))
        frames[m] = a.merge(s[["x_pos", "y_pos", "z_pos", "truth_eff", "truth_eff_err"]],
                            on=["x_pos", "y_pos", "z_pos"])
    ports = _ports(next(iter(frames.values())))
    fig, axes = plt.subplots(1, len(ports), figsize=(3.7 * len(ports), 6.0), sharey=True)
    for ax, port in zip(axes, ports):
        for k, (m, d) in enumerate(frames.items()):
            g = d[d.port == port].sort_values("y_pos")
            r = g.dds_eff / g.truth_eff
            e = r * np.hypot(0.5 * (g.dds_eff_hi - g.dds_eff_lo) / g.dds_eff, g.truth_eff_err / g.truth_eff)
            ax.errorbar(g.y_pos + 6 * (k - 0.5), r, yerr=e, fmt="o", ms=6, capsize=2,
                        color=METHOD_COLORS.get(m, f"C{k}"), label=LABELS.get(m, m))
        ax.set_title(port)
        ax.set_xlabel("y position (cm)")
    axes[0].set_ylabel("data ε (IC triggers) / MC truth ε")
    axes[0].legend(fontsize=9, frameon=False)
    allr = pd.concat([d.dds_eff / d.truth_eff for d in frames.values()])
    fig.suptitle(f"Data over MC efficiency: mean {allr.mean():.3f}, range {allr.min():.3f}–{allr.max():.3f}")
    fig.tight_layout()
    save_plot(fig, ctx, "clustereff_data_over_mc")


def closure(ctx, mc, methods):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 3.6 * len(methods)), sharex=True,
                             squeeze=False)
    for ax, m in zip(axes[:, 0], methods):
        s = _positions(mc.csv_path(f"efficiency_fit_by_position_mc_{m}"))
        s["_p"] = s.port.map({p: k for k, p in enumerate(PORT_ORDER)})
        s = s.sort_values(["_p", "y_pos"]).reset_index(drop=True)
        x = np.arange(len(s))
        ax.errorbar(x - 0.15, 100 * (s.naive_eff - s.truth_eff), yerr=100 * s.truth_eff_err, fmt="s",
                    color="0.45", ms=5, capsize=2, label="counting − truth")
        ax.errorbar(x + 0.15, 100 * (s.dds_eff - s.truth_eff),
                    yerr=100 * np.hypot(0.5 * (s.dds_eff_hi - s.dds_eff_lo), s.truth_eff_err),
                    fmt="o", color=METHOD_COLORS.get(m, "C1"), ms=6, capsize=2, label="data-driven fit − truth")
        ax.set_ylabel("Δε (percentage points)")
        ax.set_title(f"{LABELS.get(m, m)}: fit − truth {100 * (s.dds_eff - s.truth_eff).min():+.2f} to "
                     f"{100 * (s.dds_eff - s.truth_eff).max():+.2f} pp, fitted s "
                     f"{s.dds_s.min():.2f}–{s.dds_s.max():.2f}", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p.replace('Port ', 'P')} y{y:+g}" for p, y in zip(s.port, s.y_pos)],
                           rotation=70, fontsize=8)
    fig.suptitle("MC: fitted efficiency against truth (6264 template, IC triggers)")
    fig.tight_layout()
    save_plot(fig, ctx, "clustereff_mc_closure")


def _rate(hits_path, n_trig: int) -> float:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(hits_path)
    n_pmt = pd.read_parquet(hits_path, columns=["pmtID"]).pmtID.nunique()
    return pf.metadata.num_rows / n_trig / n_pmt / DUR_US * 1e3


def chi2_vs_rate(ctx, mc, gate):
    """chi2/ndf of the Data/MC charge per position against the run's hit rate (randoms estimator)."""
    import matplotlib.pyplot as plt
    from ..data.processor import AmBeNeutronProcessing
    from ..data.randoms import run_number
    from pathlib import Path
    pos = AmBeNeutronProcessing().source_positions
    hits, trig = {}, {}
    for rf in ctx.inputs["root_files"]:
        run = run_number(Path(rf))
        if run not in pos:
            continue
        base = f"{ctx.run_name}__random_%s_{run}"
        t = pd.read_parquet(ctx.parquet_path(base % "triggers"), columns=["trigger"])
        key = tuple(float(v) for v in pos[run])
        hits.setdefault(key, []).append((ctx.parquet_path(base % "hits"), len(t)))
    rates = {k: np.average([_rate(p, n) for p, n in v], weights=[n for _, n in v]) for k, v in hits.items()}
    spec = (mc.extra.get("clusterindex") or {}).get("overlay_veto") or {}
    ov = None
    if spec.get("random_hits"):
        tp = spec["random_hits"].replace("random_hits", "random_triggers")
        ov = _rate(spec["random_hits"], len(pd.read_parquet(tp, columns=["trigger"])))
    c = pd.read_csv(ctx.csv_path(f"datamc_chi2_{gate}"))
    c["rate"] = [rates.get((float(x), float(y), float(z)), np.nan) for x, y, z in zip(c.x, c.y, c.z)]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m, g in c.groupby("method", sort=False):
        ax.plot(g.rate, g.chi2 / g.ndf, "o", ms=6, color=METHOD_COLORS.get(m, None), label=LABELS.get(m, m))
    ax.set_yscale("log")
    ax.set_xlabel("data run hit rate (kHz/PMT, all hits in 70 µs)")
    ax.set_ylabel("cluster charge χ²/ndf, data vs MC")
    ax.legend(fontsize=9, frameon=False)
    ttl = "Data/MC charge χ² against the data run's hit rate"
    if ov is not None:
        ttl += f"; MC overlay (6256) {ov:.1f} kHz/PMT"
    ax.set_title(ttl, fontsize=11)
    fig.tight_layout()
    save_plot(fig, ctx, f"clustereff_chi2_vs_rate_{gate}")


def overlay_page(ctx, mc, method, tag):
    """MC candidate charge with the MC-only veto and with the overlay veto added, against data (ic)."""
    import matplotlib.pyplot as plt
    from ..efficiency_extraction.recluster_index import _mc_tables, _overlay_veto
    blk = dict(mc.extra.get("clusterindex") or {})
    spec = blk.pop("overlay_veto", None)
    if not spec:
        raise SystemExit("[clustereff] --overlay-page needs clusterindex.overlay_veto in the MC config")
    lo, hi = (float(v) for v in blk["window_ns"])
    tr, cand, key = _mc_tables(mc, blk, tag)
    ov = pd.Series(_overlay_veto(mc, spec, tr), index=tr.index)
    tr = tr.assign(ov=ov)
    cand = cand[(cand.method == method) & (cand.t_med >= lo) & (cand.t_med <= hi)]
    cand = cand.merge(tr[key + ["is_cosmic", "ov"]], on=key)
    data = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_ic"))
    data = data[data.method == method].pe.to_numpy()
    edges = np.arange(0, 305, 5)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for sel, lab, st in ((~cand.is_cosmic, "MC, MC-only veto", dict(color="0.55", ls="--")),
                         (~cand.is_cosmic & ~cand.ov, "MC, + overlay veto",
                          dict(color=METHOD_COLORS.get(method, "C1")))):
        pe = cand.pe[sel].to_numpy()
        h, _ = np.histogram(pe, edges)
        ax.stairs(h / len(pe), edges, lw=2, label=f"{lab}: {100 * (pe > 100).mean():.2f} % > 100 p.e.", **st)
    h, _ = np.histogram(data, edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(cen, h / len(data), yerr=np.sqrt(h) / len(data), fmt="o", color="k", ms=3,
                label=f"data, IC triggers: {100 * (data > 100).mean():.2f} % > 100 p.e.")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster charge [p.e.]")
    ax.set_ylabel("fraction of candidates")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(f"{LABELS.get(method, method)} candidates; overlay veto removes "
                 f"{100 * tr.ov.mean():.2f} % of MC events", fontsize=11)
    fig.tight_layout()
    save_plot(fig, ctx, f"clustereff_overlay_veto_{method}")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots clustereff")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--overlay-page", default=None, metavar="METHOD",
                   help="also draw the MC charge of METHOD with and without the overlay veto (slow)")
    p.add_argument("--overlay-tag", default="ov6256", help="capture_truth tag of the overlaid MC")
    a = p.parse_args(list(argv) if argv else [])
    set_style()
    mc = load_context(a.mc_config)
    methods = a.methods.split(",")
    for m in methods:
        vs_y(ctx, mc, m)
    ratio(ctx, mc, methods)
    closure(ctx, mc, methods)
    for g in ("ic", "none"):
        chi2_vs_rate(ctx, mc, g)
    if a.overlay_page:
        overlay_page(ctx, mc, a.overlay_page, a.overlay_tag)
    print(f"[clustereff] figures in {ctx.plots_dir}")
