"""
One AmBe source position, data against two MC hit models: 35 ns-merged MCHits (parametric) and
PMTWaveformSim waveform hits from the data hit finder.

    ambe plots datamcwave --config <data cfg> --mc-config <waveform MC cfg> --compare-mc-config <35 ns MC cfg>
        --mc-label L --compare-mc-label L --methods a,b,c --position 2 --dummy-config <dummy cfg> --dummy-tag reclust

Same samples as datamcclust/datamcneutron (data IC 700-1200 triggers, cosmic veto; MC capture-owned clusters).
--backgrounds picks the no-source background used for the subtraction, the MC fold and the efficiency rows:
ic = run 6264 IC 700-1200 gated with the cosmic veto (foldeff --template-gate ic --fold-gate ic);
pool = 6264 + dummy 6256, no IC gate (foldeff --template-dummy); ungated = 6264 with the cosmic veto, no IC gate. Fold efficiencies are read from `ambe stats foldeff`.
Writes <data plots>/datamcwave/*.png, datamcwave.pdf and <data csv>/datamcwave_{summary,efficiency}.csv;
--out-name X writes datamcwave_X instead, leaving the default set untouched.
"""
from __future__ import annotations

import argparse
import dataclasses
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..plotting import two_titles
from . import datamc_clustering as dmc
from .datamc import LABELS

DIST = {
    "pe": (np.arange(0.0, 80.0 + 2.5, 2.5), "cluster charge [p.e.]", (5.0, 65.0)),
    "n_hits": (np.arange(4.5, 40.5 + 1.0, 1.0), "hits per cluster", (5.0, 35.0)),
    "pe_per_hit": (np.arange(0.5, 4.0 + 0.1, 0.1), "charge per hit [p.e.]", (0.8, 3.0)),
}
KMAX = 4


def pool_background(ctx: RunContext, dummy: RunContext, m: str, tag: str, bkg_run: int, S: dmc.Sample):
    """6264 (cosmic veto, no IC gate) plus the dummy 6256 triggers surviving the same veto."""
    from .datamc_zerobin import WIN, _bkg_counts
    n6256, rc = _bkg_counts(dummy, 6256, m, tag)
    c6256 = pd.read_parquet(dummy.parquet_path(f"{dummy.run_name}__random_recoclusters_6256__{tag}"),
                            columns=["trigger", "method", "t_med", "f_pe_total", "n_hits"])
    c6256 = c6256[(c6256.method == m) & c6256.t_med.between(*WIN) & c6256.trigger.isin(rc.trigger.unique())]
    c6256 = c6256.rename(columns={"f_pe_total": "pe"})[["t_med", "pe", "n_hits"]].assign(run=6256)
    b_c = pd.concat([S.b_c[["t_med", "pe", "n_hits", "run"]], c6256], ignore_index=True)
    b_c["t_us"] = b_c.t_med / 1000.0
    b_c["pe_per_hit"] = b_c.pe / b_c.n_hits
    b_e = pd.concat([S.b_e[["run", "n_accepted_clusters"]],
                     pd.DataFrame(dict(run=6256, n_accepted_clusters=n6256))], ignore_index=True)
    return b_c, b_e


def ic_background(ctx: RunContext, m: str, bkg_run: int):
    dp = lambda s: ctx.parquet_path(f"{ctx.run_name}__{s}")          # noqa: E731
    b_c = pd.read_parquet(dp("candidates_ic"), filters=[("method", "==", m)])
    b_c = b_c[b_c.run == bkg_run].reset_index(drop=True)
    b_e = pd.read_parquet(dp(f"eventindex_ic_{m}"))
    b_e = b_e[~b_e.is_cosmic & (b_e.run == bkg_run)].reset_index(drop=True)
    b_c["t_us"] = b_c.t_med / 1000.0
    b_c["pe_per_hit"] = b_c.pe / b_c.n_hits
    return b_c, b_e


def bkey(label: str) -> str:
    return next(k for k, v in BKG.items() if v[0] == label)


def with_bkg(S: dmc.Sample, b_c: pd.DataFrame, b_e: pd.DataFrame) -> dmc.Sample:
    return dataclasses.replace(S, b_c=b_c, b_e=b_e)


def shapes(S: dmc.Sample, col: str, edges: np.ndarray, p: int, rng: tuple):
    d, vd = dmc.data_minus_bkg(S, col, edges, [p])
    m, vm = dmc.mc_neutron(S, col, edges, [p])
    lo, hi = np.searchsorted(edges, rng[0]), np.searchsorted(edges, rng[1])
    s = d[lo:hi].sum() / m[lo:hi].sum() if m[lo:hi].sum() > 0 else np.nan
    v = vd[lo:hi] + vm[lo:hi] * s ** 2
    ok = v > 0
    chi = float(((d[lo:hi] - s * m[lo:hi])[ok] ** 2 / v[ok]).sum())
    return d, vd, m * s, chi, int(ok.sum()) - 1


def means(S: dmc.Sample, col: str, p: int) -> tuple:
    d = S.d_c[S.d_c.pid == p][col].to_numpy(float)
    f = S.n_d[p] / S.n_b
    b = S.b_c[col].to_numpy(float)
    mc = S.m_c[(S.m_c.pid == p) & S.m_c.cap.to_numpy(bool)]
    return float((d.sum() - b.sum() * f) / (len(d) - len(b) * f)), float(mc[col].mean()) if len(mc) else np.nan


def page_dist(out, Ss: dict, data_S: dmc.Sample, col: str, p: int, method: str, bkg_label: str, rows: list):
    import matplotlib.pyplot as plt
    edges, xlabel, rng = DIST[col]
    cen = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
    d = None
    stats = []
    for (lab, S), c in zip(Ss.items(), ("#2a6fbb", "#c23b22")):
        d, vd, m, chi, ndf = shapes(S, col, edges, p, rng)
        md, mm = means(S, col, p)
        ax[0].stairs(m, edges, color=c, lw=1.8, baseline=None, label=f"{lab}: mean {mm:.2f}, χ²/ndf {chi:.0f}/{ndf}")
        ok = m > 0
        ax[1].errorbar(cen[ok], d[ok] / m[ok], yerr=np.sqrt(vd[ok]) / m[ok], fmt="o", ms=2.5, lw=0.8, color=c)
        stats.append((lab, mm, chi, ndf))
        rows.append(dict(method=method, background=bkg_label, quantity=col, sample=lab, mean=mm, data_mean=md,
                         chi2=chi, ndf=ndf, shape_range=f"{rng[0]:g}-{rng[1]:g}"))
    ax[0].errorbar(cen, d, yerr=np.sqrt(vd), fmt="o", color="k", ms=3, lw=1, label=f"data − background: mean {md:.2f}")
    ax[0].set_ylabel("clusters / bin")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_ylim(0, 2.5)
    ax[1].set_ylabel("data / MC")
    ax[1].set_xlabel(xlabel)
    top = two_titles(fig, f"{LABELS.get(method, method)}: {xlabel.split(' [')[0]}",
                     f"{dmc.pos_title(p)}, data IC 700-1200 minus {bkg_label}; MC capture clusters shape-normalised "
                     f"in {rng[0]:g}-{rng[1]:g}")
    fig.tight_layout(rect=(0, 0, 1, top))
    out.save(fig, f"{method}_{col}_{bkey(bkg_label)}")


def nfit_free(t_us: np.ndarray) -> dict:
    """dmc._nfit with the flat term B free."""
    import contextlib
    import io
    from .basic import AmBeNeutronAnalyzer, fit
    an = AmBeNeutronAnalyzer()
    with contextlib.redirect_stdout(io.StringIO()):
        r = fit(np.asarray(t_us, float), float_B=True, backend="lmfit", analyzer=an, min_counts=100)
    if r is None:
        return dict(dmc.NFIT_NAN)
    return dict(tau=r["tau"], tau_err=r["tau_err"], therm=r["therm"], therm_err=r["therm_err"], B=r["B"],
                redchi=r["redchi"], N=r["N"])


def page_time(out, Ss: dict, p: int, method: str, bkg_label: str, rows: list):
    import matplotlib.pyplot as plt
    E = dmc.T_PLOT
    cen = 0.5 * (E[1:] + E[:-1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    S0 = next(iter(Ss.values()))
    d = S0.d_c[S0.d_c.pid == p].t_us.to_numpy(float)
    b = S0.flat_data(p) * (E[1] - E[0])
    hd = np.histogram(d, E)[0]
    fd = dmc._nfit(d, S0.flat_data(p))
    ax.errorbar(cen, hd, yerr=np.sqrt(hd), fmt="o", color="k", ms=3, lw=1,
                label=f"data: τ {fd['tau']:.2f} ± {fd['tau_err']:.2f}, therm {fd['therm']:.2f} µs")
    ax.stairs(np.full(len(E) - 1, b), E, color=dmc.BKG_COLOR, lw=1.2, baseline=None, label=f"{bkg_label} (flat rate)")
    rows.append(dict(method=method, background=bkg_label, sample="data", **{k: fd[k] for k in ("tau", "tau_err", "therm", "therm_err", "redchi", "N")}))
    ff = nfit_free(d)
    rows.append(dict(method=method, background=bkg_label, sample="data, B free", **{k: ff[k] for k in ("tau", "tau_err", "therm", "therm_err", "redchi", "N")}))
    for (lab, S), c in zip(Ss.items(), ("#2a6fbb", "#c23b22")):
        mc = S.m_c[(S.m_c.pid == p) & S.m_c.cap.to_numpy(bool)]
        t = mc.t_us.to_numpy(float)
        f = dmc._nfit(t, 1e-6)
        hm = np.histogram(t, E)[0].astype(float)
        scale = (hd.sum() - b * len(cen)) / hm.sum() if hm.sum() else np.nan
        ax.stairs(hm * scale + b, E, color=c, lw=1.6, baseline=None,
                  label=f"{lab} + background: τ {f['tau']:.2f} ± {f['tau_err']:.2f}, therm {f['therm']:.2f} µs")
        rows.append(dict(method=method, background=bkg_label, sample=lab, **{k: f[k] for k in ("tau", "tau_err", "therm", "therm_err", "redchi", "N")}))
    ax.set_xlabel("cluster time [µs]")
    ax.set_ylabel("clusters / µs")
    ax.legend(frameon=False, fontsize=8)
    top = two_titles(fig, f"{LABELS.get(method, method)}: capture time",
                     f"{dmc.pos_title(p)}; fit A(1−e^(−t/therm))e^(−t/τ) + B on 2-67 µs, B fixed to the {bkg_label} rate "
                     "(data) or 0 (MC capture clusters)")
    fig.tight_layout(rect=(0, 0, 1, top))
    out.save(fig, f"{method}_time_{bkey(bkg_label)}")


def page_mult(out, Ss: dict, p: int, method: str, bkg_label: str, rows: list):
    import matplotlib.pyplot as plt
    k = np.arange(KMAX + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    pd_, n = None, 0
    for (lab, S), c, dx in zip(Ss.items(), ("#2a6fbb", "#c23b22"), (-0.12, 0.12)):
        pd_, pm, pf, _, n = S.mult(KMAX, p)
        ax.plot(k + dx, pf, "-s", color=c, ms=5, lw=1.4, label=f"{lab} ⊗ {bkg_label}: P(0) {pf[0]:.3f}")
        rows.append(dict(method=method, background=bkg_label, sample=lab, p0=pf[0], p1=pf[1], p2plus=pf[2:].sum(),
                         mean=float((k * pf).sum()), mc_only_p0=pm[0]))
    e = np.sqrt(pd_ * (1 - pd_) / n)
    ax.errorbar(k, pd_, yerr=e, fmt="o", color="k", ms=5, label=f"data, {n:,} triggers: P(0) {pd_[0]:.3f}")
    rows.append(dict(method=method, background=bkg_label, sample="data", p0=pd_[0], p1=pd_[1], p2plus=pd_[2:].sum(),
                     mean=float((k * pd_).sum()), mc_only_p0=np.nan))
    ax.set_xticks(k)
    ax.set_xticklabels([str(x) for x in k[:-1]] + [f"≥{KMAX}"])
    ax.set_xlabel("clusters per trigger (2-67 µs)")
    ax.set_ylabel("fraction of triggers")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    top = two_titles(fig, f"{LABELS.get(method, method)}: clusters per trigger",
                     f"{dmc.pos_title(p)}; MC capture clusters per event folded with the {bkg_label} cluster count")
    fig.tight_layout(rect=(0, 0, 1, top))
    out.save(fig, f"{method}_mult_{bkey(bkg_label)}")


BKG = {"ic": ("6264 IC-gated", "ic_icfold"), "pool": ("6264+6256 pool", "none_plus6256"),
       "ungated": ("6264 ungated", "none")}


def efficiency_table(ctx: RunContext, mcs: dict, methods: list, p: int, backgrounds: list) -> pd.DataFrame:
    """Port p rows of the foldeff CSVs: data and every MC, per background."""
    rows = []
    for m in methods:
        for tl, tg in (BKG[b] for b in backgrounds):
            srcs = [("data", ctx, f"_fold_data_{tg}_{m}")] + [(lab, c, f"_fold_mcnominal_{tg}_{m}") for lab, c in mcs.items()]
            for lab, cx, sfx in srcs:
                path = cx.csv_path("efficiency_fit_by_position" + sfx)
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df = df[df.level == "position"]
                df = df[dmc.pos_id(df.rename(columns={"x_pos": "sourceX", "y_pos": "sourceY", "z_pos": "sourceZ"})) == p]
                if df.empty:
                    continue
                r = df.iloc[0]
                rows.append(dict(method=m, template=tl, sample=lab, n_trig=r.n_trig, counting=r.naive_eff,
                                 counting_err=r.naive_err, poisson=r.poisson_eff, poisson_lo=r.poisson_eff_lo,
                                 poisson_hi=r.poisson_eff_hi, poisson_p=r.poisson_p, datadriven=r.dds_eff,
                                 datadriven_lo=r.dds_eff_lo, datadriven_hi=r.dds_eff_hi, datadriven_p=r.dds_p,
                                 s=r.dds_s, truth=r.get("truth_eff", np.nan), truth_err=r.get("truth_eff_err", np.nan),
                                 background_hist=r.background_hist))
    return pd.DataFrame(rows)


def page_eff(out, eff: pd.DataFrame, method: str, p: int):
    import matplotlib.pyplot as plt
    e = eff[eff.method == method]
    if e.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = [("counting", None, None), ("poisson", "poisson_lo", "poisson_hi"), ("datadriven", "datadriven_lo", "datadriven_hi")]
    groups = list(dict.fromkeys(zip(e.template, e["sample"])))
    truth_labelled = False
    for i, (tl, lab) in enumerate(groups):
        r = e[(e.template == tl) & (e["sample"] == lab)].iloc[0]
        for j, (c, lo, hi) in enumerate(cols):
            x = i + (j - 1) * 0.22
            y = r[c]
            yerr = None if lo is None else [[y - r[lo]], [r[hi] - y]]
            ax.errorbar(x, y, yerr=yerr if yerr is not None else r.counting_err, fmt="osD"[j], color=("0.4", "#2a6fbb", "#c23b22")[j],
                        ms=6, label=("counting", "Poisson", "data-driven")[j] if i == 0 else None)
        if np.isfinite(r.truth):
            ax.plot([i - 0.35, i + 0.35], [r.truth, r.truth], color="k", lw=1.2, label=None if truth_labelled else "MC truth")
            truth_labelled = True
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lab}\n{tl}" for tl, lab in groups], fontsize=8)
    ax.set_ylabel("neutron detection efficiency")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    top = two_titles(fig, f"{LABELS.get(method, method)}: efficiency", f"{dmc.pos_title(p)}; fold fits from ambe stats foldeff")
    fig.tight_layout(rect=(0, 0, 1, top))
    out.save(fig, f"{method}_efficiency")


def nms_check(mcs: dict, methods: list) -> pd.DataFrame:
    """Capture truth of every method in every MC, 2-67 us: what the NMS merge does to splits and merges."""
    from .datamc_zerobin import WIN
    rows = []
    for lab, mc in mcs.items():
        tag = (mc.extra.get("capture_truth") or {})["overlay_tag"]
        cl = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__capture_truth_clusters__{tag}"),
                             columns=["_source_file", "eventNumber", "method", "t_med", "cap_dom", "cap_dom_frac", "n_caps_contrib"])
        cp = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__capture_truth_captures__{tag}"))
        ev = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__events"), columns=["_source_file", "eventNumber"])
        nev = ev[ev._source_file.isin(set(cl._source_file) | set(cp._source_file))].drop_duplicates().shape[0]
        vis = cp[cp.visible.astype(bool) & cp.cap_t.between(WIN[0], WIN[1])]
        for m in methods:
            c = cl[(cl.method == m) & cl.t_med.between(*WIN)]
            cap = c[c.cap_dom >= 0]
            own = vis[f"n_owners_{m}"].to_numpy()
            rows.append(dict(sample=lab, method=LABELS.get(m, m), events=nev, clusters_per_event=len(c) / nev,
                             capture_share=float((c.cap_dom >= 0).mean()), one_neutron=float((cap.n_caps_contrib <= 1).mean()),
                             purity=float(cap.cap_dom_frac.mean()), found=float((own >= 1).mean()),
                             split_per_capture=float(np.clip(own - 1, 0, None).sum() / max((own >= 1).sum(), 1)),
                             merged_share=float((cap.n_caps_contrib >= 2).mean())))
    return pd.DataFrame(rows)


class Out:
    def __init__(self, ctx: RunContext, pdf, name: str = "datamcwave"):
        self.dir = ctx.plots_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pdf = pdf

    def save(self, fig, name: str):
        import matplotlib.pyplot as plt
        fig.savefig(self.dir / f"{name}.png", dpi=200)
        fig.savefig(self.dir / f"{name}.pdf")
        self.pdf.savefig(fig)
        plt.close(fig)


def run(ctx: RunContext, mcs: dict, methods: list, p: int, backgrounds: list, dummy: Optional[RunContext],
        dummy_tag: Optional[str], out_name: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    bkg_run = int(ctx.fit_params["background_run"])
    dist_rows, time_rows, mult_rows = [], [], []
    pre = f"datamcwave_{out_name}" if out_name else "datamcwave"
    (ctx.plots_dir / pre).mkdir(parents=True, exist_ok=True)
    with PdfPages(ctx.plots_dir / pre / f"{pre}.pdf") as pdf:
        out = Out(ctx, pdf, pre)
        for m in methods:
            S = {lab: dmc.load(ctx, mc, m, False, bkg_run) for lab, mc in mcs.items()}
            first = next(iter(S.values()))
            bk = {"ic": lambda: ic_background(ctx, m, bkg_run),
                  "pool": lambda: pool_background(ctx, dummy, m, dummy_tag, bkg_run, first),
                  "ungated": lambda: (first.b_c, first.b_e)}
            for bl, (bc, be) in ((BKG[b][0], bk[b]()) for b in backgrounds):
                Sb = {lab: with_bkg(s, bc, be) for lab, s in S.items()}
                print(f"[datamcwave] {m}: background {bl}, {len(be):,} triggers, {len(bc):,} clusters")
                for col in DIST:
                    page_dist(out, Sb, first, col, p, m, bl, dist_rows)
                page_time(out, Sb, p, m, bl, time_rows)
                page_mult(out, Sb, p, m, bl, mult_rows)
        eff = efficiency_table(ctx, mcs, methods, p, backgrounds)
        for m in methods:
            page_eff(out, eff, m, p)
    pd.DataFrame(dist_rows).to_csv(ctx.csv_path(f"{pre}_distributions"), index=False)
    pd.DataFrame(time_rows).to_csv(ctx.csv_path(f"{pre}_capturetime"), index=False)
    pd.DataFrame(mult_rows).to_csv(ctx.csv_path(f"{pre}_multiplicity"), index=False)
    eff.to_csv(ctx.csv_path(f"{pre}_efficiency"), index=False)
    nms_check(mcs, methods).to_csv(ctx.csv_path(f"{pre}_nmscheck"), index=False)
    print(f"[datamcwave] wrote {out.dir} and {pre}_*.csv")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots datamcwave")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--mc-label", required=True)
    p.add_argument("--compare-mc-config", required=True)
    p.add_argument("--compare-mc-label", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--position", type=int, required=True, help="position id (Port 5, Y = 0 is 2)")
    p.add_argument("--backgrounds", required=True, help="comma list of ic, pool, ungated")
    p.add_argument("--dummy-config", help="pool only: dummy-run randoms config holding 6256")
    p.add_argument("--dummy-tag", help="pool only: recoclusters tag of the dummy run, e.g. reclust")
    p.add_argument("--out-name", help="write datamcwave_<name> plots and csv instead of the default datamcwave set")
    a = p.parse_args(list(argv) if argv is not None else None)
    bks = a.backgrounds.split(",")
    if set(bks) - set(BKG):
        p.error("--backgrounds takes ic, pool, ungated")
    if ("pool" in bks) != bool(a.dummy_config and a.dummy_tag):
        p.error("--dummy-config and --dummy-tag go with pool, and only with it")
    mcs = {a.compare_mc_label: load_context(a.compare_mc_config), a.mc_label: load_context(a.mc_config)}
    run(ctx, mcs, a.methods.split(","), a.position, bks, load_context(a.dummy_config) if a.dummy_config else None,
        a.dummy_tag, a.out_name)
