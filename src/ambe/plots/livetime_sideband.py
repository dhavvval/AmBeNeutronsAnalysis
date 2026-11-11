"""
Live-time normalisation of source and no-source runs, tested in the BGO IC sidebands.

    ambe plots livetime --config configs/data_ambe2v4_reclust.yaml
        --mc-config configs/mc_ambe_v3fix_dead15_ovbkg.yaml --runs-db configs/run_db/ambe_v4_sql_runs.yaml
        --source-runs 6046,6252 --nosource-run 6264 --extra-runs 6265
        --methods clusterfinder,clusterfinder_nms --template-gate none_plus6256
        --activity-nps 1.6667,100 --gap-s 30 --out-name livetime_port5

The DAQ has a non-paralyzable dead time tau (~2 s) after every readout, so source runs (true rate ~9 Hz) are
live ~5 % of the time and 6264 (~0.6 Hz) ~45 %. Exposure is T_live = sum(dt - tau) over intervals shorter
than --gap-s; the readout composition is unchanged by the dead time, so a class of triggers has rate
N_class / T_live. Outputs (csv prefix livetime_<out-name>_):
  livetime      per run: readouts, tau, T_sql / T_span / T_gapcut / T_live (tau, tau +- 20 ms, gap 10/30/60 s), lambda
  sideband      per run x IC class: counts, rate per live / wall second, cosmic-veto split, excess over no-source,
                and k = (source / no-source) rate of IC > 3000 vetoed triggers, the empirical live-time check
  tank          per run x IC class x veto x method: clusters per trigger, P(n), capture-like fraction (fixed tau);
                the re-clustered candidates exist only for non-vetoed triggers, so the no-veto set uses tree_cf,
                the BeamCluster ClusterFinder clusters of every trigger in the same 2-67 us window
  tspec         cluster-time histograms behind `tank`
  expected      gate-rate budget, false starts, implied tag probability per neutron for --activity-nps, MC equivalent
  foldfs        data IC-trigger multiplicity vs MC (x) pool (the foldeff fold) and vs the false-start mixture
  shapes        (opt-in, --shape-mc) per source run x IC class x method x background, vetoed triggers: data cluster
                means (n_hits, p.e./hit, p.e.) raw and minus the no-source clusters scaled per trigger, data P0,
                and each MC's capture-cluster means and P0 (MC only, MC (x) background); the gate row uses the
                datamcwave definitions
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from ..context import RunContext, load_context
from ..plotting import two_titles
from . import datamc_zerobin as dz

TAU_Q = 0.001
TAU_SHIFT_S = 0.020
GAPS_S = (10.0, 30.0, 60.0)
BR_444 = 0.57
TAU_CAP_US = 29.4
T_EDGES_US = np.arange(2.0, 67.0 + 2.5, 2.5)
CLASSES = [("IC<0", -np.inf, 0.0), ("0-300", 0.0, 300.0), ("300-700", 300.0, 700.0),
           ("700-1200", 700.0, 1200.0), ("1200-3000", 1200.0, 3000.0), ("IC>3000", 3000.0, np.inf)]
GATE = "gate (700-1200, no 2nd pulse)"
COLORS = ["#1f5fbf", "#e07b00", "#7a3e9d", "0.45"]
MARKERS = ["o", "s", "D", "^"]
TREE = "tree_cf"
SHAPE_CLASSES = ("300-700", "700-1200", "1200-3000", GATE)
SHAPE_COLS = ("n_hits", "pe_per_hit", "pe")


def sql_seconds(row: dict) -> float:
    f = "%Y-%m-%d %H:%M:%S.%f"
    a, b = (datetime.strptime(row[k] if "." in row[k] else row[k] + ".0", f) for k in ("start", "stop"))
    return (b - a).total_seconds()


def livetime(ts_ns: np.ndarray, gap_s: float, tau: Optional[float] = None) -> dict:
    """Non-paralyzable dead-time accounting from readout timestamps [ns]."""
    t = np.unique(np.asarray(ts_ns, dtype=np.int64))
    t = t[t > 0] / 1e9
    dt = np.diff(t)
    ok = (dt > 0) & (dt < gap_s)
    d = dt[ok]
    if tau is None:
        tau = float(np.quantile(d, TAU_Q))
    t_live = float(np.clip(d - tau, 0.0, None).sum())
    return dict(n_readouts=len(t), n_intervals=int(ok.sum()), tau_s=tau, span_s=float(t[-1] - t[0]),
                gapcut_s=float(d.sum()), n_gaps=int((dt >= gap_s).sum()), live_s=t_live,
                lambda_hz=ok.sum() / t_live, r_rec_hz=ok.sum() / d.sum())


def mixture(p_fold: np.ndarray, p_fs: np.ndarray, f: float) -> np.ndarray:
    return (1.0 - f) * np.asarray(p_fold, float) + f * np.asarray(p_fs, float)


def capture_fraction(counts: np.ndarray, edges_us: np.ndarray, tau_us: float = TAU_CAP_US) -> tuple:
    """Poisson fit of A exp(-t/tau) + C over the bins; returns (capture share of the window, error, A, C)."""
    from scipy.optimize import minimize
    n = np.asarray(counts, float)
    if n.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    e = tau_us * (np.exp(-edges_us[:-1] / tau_us) - np.exp(-edges_us[1:] / tau_us))
    w = np.diff(edges_us)
    nll = lambda q: float(np.sum(q[0] * e + q[1] * w - n * np.log(np.maximum(q[0] * e + q[1] * w, 1e-300))))  # noqa: E731
    x0 = np.array([n.sum() / e.sum() / 2, n.sum() / w.sum() / 2])
    r = minimize(nll, x0, method="L-BFGS-B", bounds=[(0, None), (0, None)])
    a, c = r.x
    mu = a * e + c * w
    J = np.stack([e, w], 1)
    cov = np.linalg.pinv(J.T @ (J / np.maximum(mu, 1e-12)[:, None]))
    tot = mu.sum()
    g = np.array([e.sum() / tot - a * e.sum() * e.sum() / tot ** 2, -a * e.sum() * w.sum() / tot ** 2])
    return a * e.sum() / tot, float(np.sqrt(max(g @ cov @ g, 0.0))), a, c


def _pmf(n: np.ndarray, nb: int) -> np.ndarray:
    h = np.bincount(np.minimum(np.asarray(n, int), nb - 1), minlength=nb).astype(float)
    return h


def _chi2(d: np.ndarray, p: np.ndarray, p_err: np.ndarray) -> tuple:
    from scipy.stats import chi2
    N = d.sum()
    var = d + (N * p_err) ** 2
    ok = var > 0
    x = float(np.sum((d[ok] - N * p[ok]) ** 2 / var[ok]))
    dof = int(ok.sum()) - 1
    return x, dof, float(chi2.sf(x, dof)) if dof > 0 else np.nan


# ---------------------------------------------------------------------------------------------- inputs
def _features(run: int, bkg_run: int, extra: bool) -> pd.DataFrame:
    if extra:
        f = Path(f"TriggerSummary/WaveformFeatures_AmBe2.0v4_special_box_{run}.parquet")
        return pd.read_parquet(f, columns=["timestamp", "accepted", "IC_adjusted", "second_pulse"])
    return dz._features(run, bkg_run)


def tree_clusters(ctx: RunContext, run: int) -> pd.DataFrame:
    cf = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_cfclusters_{run}"),
                         columns=["trigger", "clusterTime", "clusterPE", "clusterHits"])
    return cf


def triggers(ctx: RunContext, run: int, bkg_run: int, methods: list, cand: pd.DataFrame) -> pd.DataFrame:
    """Every readout of a run with IC, second pulse, the cosmic veto and its two halves, and per-method
    cluster counts in the window."""
    ei = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_none_{methods[0]}"),
                         columns=["run", "trigger", "eventTankTime", "is_cosmic", "sourceX", "sourceY", "sourceZ"],
                         filters=[("run", "==", run)])
    w = _features(run, bkg_run, False)
    w = w[w.timestamp > 0].drop_duplicates("timestamp")
    g = ei.merge(w, left_on="eventTankTime", right_on="timestamp", how="left", validate="one_to_one")
    if g.IC_adjusted.isna().any():
        raise SystemExit(f"[livetime] run {run}: {int(g.IC_adjusted.isna().sum())} triggers without a waveform")
    cf = tree_clusters(ctx, run)
    from ..data.selection import CutCriteria
    cc = CutCriteria()
    vp = cf.assign(v=cf.clusterTime < cc.cosmic_ct_threshold).groupby("trigger").v.any()
    ve = cf.assign(v=cf.clusterPE > cc.cosmic_pe_threshold).groupby("trigger").v.any()
    g["veto_prompt"] = g.trigger.map(vp).fillna(False).astype(bool)
    g["veto_pe"] = g.trigger.map(ve).fillna(False).astype(bool)
    if ((g.veto_prompt | g.veto_pe) != g.is_cosmic.astype(bool)).any():
        raise SystemExit(f"[livetime] run {run}: veto halves do not reproduce is_cosmic")
    for m in methods + [TREE]:
        n = cand[(cand.run == run) & (cand.method == m)].groupby("trigger").size()
        g[f"n_{m}"] = g.trigger.map(n).fillna(0).astype(int)
    g["cls"] = pd.cut(g.IC_adjusted, [k[1] for k in CLASSES] + [np.inf], labels=[k[0] for k in CLASSES],
                      right=False).astype(str)
    g["gate"] = g.accepted.astype(bool)
    return g


# ---------------------------------------------------------------------------------------------- parts
def livetime_table(runs: dict, sql: dict, gap_s: float) -> pd.DataFrame:
    rows = []
    for run, (kind, ts) in runs.items():
        base = livetime(ts, gap_s)
        r = dict(run=run, kind=kind, **base)
        for sh, lab in ((-TAU_SHIFT_S, "m20ms"), (TAU_SHIFT_S, "p20ms")):
            q = livetime(ts, gap_s, base["tau_s"] + sh)
            r[f"live_s_tau{lab}"], r[f"lambda_hz_tau{lab}"] = q["live_s"], q["lambda_hz"]
        for gs in GAPS_S:
            q = livetime(ts, gs, base["tau_s"])
            r[f"live_s_gap{gs:g}"] = q["live_s"]
        s = sql.get(run)
        r["sql_s"] = sql_seconds(s) if s else np.nan
        r["sql_numevents"] = s["numevents"] if s else np.nan
        r["numevents_per_readout"] = r["sql_numevents"] / r["n_readouts"]
        r["span_over_sql"] = r["span_s"] / r["sql_s"]
        r["live_fraction"] = r["live_s"] / r["gapcut_s"]
        rows.append(r)
    return pd.DataFrame(rows)


def implied_tau(lt: pd.DataFrame, sb: pd.DataFrame, ts: dict, gap_s: float) -> pd.DataFrame:
    """Dead time that makes the vetoed IC > 3000 rate of each run equal to the no-source run's."""
    from scipy.optimize import brentq
    k = sb.drop_duplicates("run").set_index("run")
    lt = lt.copy()
    lt["k_cosmic"] = lt.run.map(k.k_cosmic)
    lt["k_cosmic_err"] = lt.run.map(k.k_cosmic_err)
    lt["live_s_kcorr"] = lt.live_s * lt.k_cosmic
    tau = []
    for r in lt.itertuples():
        if np.isnan(r.k_cosmic):
            tau.append(np.nan)
            continue
        f = lambda t: livetime(ts[r.run][1], gap_s, t)["live_s"] - r.live_s_kcorr  # noqa: E731
        tau.append(brentq(f, r.tau_s - 0.2, r.tau_s + 0.2) if r.k_cosmic != 1 else r.tau_s)
    lt["tau_implied_s"] = tau
    return lt


def sideband(T: dict, lt: pd.DataFrame, bkg_run: int) -> pd.DataFrame:
    L = lt.set_index("run")
    rows = []
    for run, g in T.items():
        tl, tw = L.live_s[run], L.sql_s[run]
        sets = [(c[0], g.cls == c[0]) for c in CLASSES] + [(GATE, g.gate)]
        for name, s in sets:
            q = g[s]
            n, nv = len(q), int((~q.is_cosmic).sum())
            rows.append(dict(run=run, kind=L.kind[run], cls=name, n=n, frac_of_run=n / len(g),
                             rate_live=n / tl, rate_live_err=np.sqrt(n) / tl, rate_sql=n / tw,
                             rate_gapcut=n / L.gapcut_s[run], n_kept=nv, rate_live_kept=nv / tl,
                             rate_live_kept_err=np.sqrt(nv) / tl,
                             frac_cosmic=q.is_cosmic.mean() if n else np.nan,
                             frac_veto_prompt=q.veto_prompt.mean() if n else np.nan,
                             frac_veto_pe=q.veto_pe.mean() if n else np.nan,
                             frac_veto_prompt_only=(q.veto_prompt & ~q.veto_pe).mean() if n else np.nan,
                             frac_veto_pe_only=(q.veto_pe & ~q.veto_prompt).mean() if n else np.nan,
                             frac_second_pulse=q.second_pulse.astype(bool).mean() if n else np.nan))
    df = pd.DataFrame(rows)
    b = df[df.run == bkg_run].set_index("cls")
    for col in ("rate_live", "rate_live_kept", "rate_sql"):
        df[f"{col}_bkg"] = df.cls.map(b[col])
        df[f"{col}_excess"] = df[col] - df[f"{col}_bkg"]
        df[f"{col}_ratio"] = df[col] / df[f"{col}_bkg"]
    df["rate_live_ratio_err"] = df.rate_live_ratio * np.sqrt(1 / df.n.clip(lower=1) + 1 / df.cls.map(b.n).clip(lower=1))
    # empirical live-time check: tagged (vetoed) IC > 3000 triggers are cosmic muons through the BGO in both runs
    tag = {r: int((T[r][(T[r].cls == "IC>3000")].is_cosmic).sum()) for r in T}
    k = {r: (tag[r] / L.live_s[r]) / (tag[bkg_run] / L.live_s[bkg_run]) for r in T}
    kerr = {r: k[r] * np.sqrt(1 / max(tag[r], 1) + 1 / max(tag[bkg_run], 1)) for r in T}
    df["k_cosmic"] = df.run.map(k)
    df["k_cosmic_err"] = df.run.map(kerr)
    df["n_ic3000_tagged"] = df.run.map(tag)
    df["rate_live_kcorr"] = df.rate_live / df.k_cosmic
    df["rate_live_kcorr_excess"] = df.rate_live_kcorr - df.rate_live_bkg
    return df


def tank(T: dict, lt: pd.DataFrame, methods: list, cand: pd.DataFrame, bkg_run: int) -> tuple:
    L = lt.set_index("run")
    rows, spec = [], []
    for run, g in T.items():
        sets = [(c[0], g.cls == c[0]) for c in CLASSES] + [(GATE, g.gate)]
        cr = cand[cand.run == run]
        for name, s in sets:
            for veto in ("noveto", "veto"):
                q = g[s & (~g.is_cosmic if veto == "veto" else True)]
                for m in (methods + [TREE] if veto == "veto" else [TREE]):
                    n = q[f"n_{m}"].to_numpy()
                    cl = cr[(cr.method == m) & cr.trigger.isin(q.trigger)]
                    h, _ = np.histogram(cl.t_med.to_numpy() / 1000.0, T_EDGES_US)
                    fc, fce, a, c = capture_fraction(h, T_EDGES_US)
                    nt = len(q)
                    rows.append(dict(run=run, kind=L.kind[run], cls=name, veto=veto, method=m, n_trig=nt,
                                     clusters=int(n.sum()), clusters_per_trig=n.mean() if nt else np.nan,
                                     p0=(n == 0).mean() if nt else np.nan, p1=(n == 1).mean() if nt else np.nan,
                                     p2plus=(n >= 2).mean() if nt else np.nan,
                                     trig_rate_live=nt / L.live_s[run], cluster_rate_live=n.sum() / L.live_s[run],
                                     capture_frac=fc, capture_frac_err=fce,
                                     pe_median=float(cl.pe.median()) if len(cl) else np.nan,
                                     hits_mean=float(cl.n_hits.mean()) if len(cl) else np.nan))
                    spec += [dict(run=run, cls=name, veto=veto, method=m, lo=T_EDGES_US[i], hi=T_EDGES_US[i + 1],
                                  n=int(h[i])) for i in range(len(h))]
    df = pd.DataFrame(rows)
    b = df[df.run == bkg_run].set_index(["cls", "veto", "method"])
    key = list(zip(df.cls, df.veto, df.method))
    for col in ("trig_rate_live", "cluster_rate_live"):
        df[f"{col}_bkg"] = [b[col].get(x, np.nan) for x in key]
        df[f"{col}_excess"] = df[col] - df[f"{col}_bkg"]
    df["clusters_per_excess_trig"] = df.cluster_rate_live_excess / df.trig_rate_live_excess
    return df, pd.DataFrame(spec)


def shapes(ctx: RunContext, mcs: dict, T: dict, cand: pd.DataFrame, methods: list, src_runs: list, bkg_run: int,
           backgrounds: list) -> pd.DataFrame:
    from . import datamc_clustering as dmc
    from .datamc_waveform import KMAX, ic_background, with_bkg
    rows = []
    for m in methods:
        S = {lab: dmc.load(ctx, mc, m, False, bkg_run) for lab, mc in mcs.items()}
        first = next(iter(S.values()))
        bk = {"ic": lambda: ic_background(ctx, m, bkg_run), "ungated": lambda: (first.b_c, first.b_e)}
        for b in backgrounds:
            b_c, b_e = bk[b]()
            Sb = {lab: with_bkg(s, b_c, b_e) for lab, s in S.items()}
            for run in src_runs:
                g = T[run]
                p = int(dmc.pos_id(g.head(1))[0])
                cr = cand[(cand.run == run) & (cand.method == m)]
                for name in SHAPE_CLASSES:
                    q = g[(g.gate if name == GATE else g.cls == name) & ~g.is_cosmic]
                    cl = cr[cr.trigger.isin(q.trigger)].assign(pe_per_hit=lambda x: x.pe / x.n_hits)
                    f = len(q) / len(b_e)
                    row = dict(run=run, pid=p, method=m, background=b, cls=name, n_trig=len(q), n_clusters=len(cl),
                               n_bkg_trig=len(b_e), n_bkg_clusters=len(b_c), bkg_share=len(b_c) * f / max(len(cl), 1),
                               data_p0=float((q[f"n_{m}"] == 0).mean()) if len(q) else np.nan)
                    for col in SHAPE_COLS:
                        d, bb = cl[col].to_numpy(float), b_c[col].to_numpy(float)
                        row[f"data_raw_{col}"] = float(d.mean()) if len(d) else np.nan
                        row[f"data_sub_{col}"] = float((d.sum() - bb.sum() * f) / (len(d) - len(bb) * f))
                    for lab, s in Sb.items():
                        mc = s.m_c[(s.m_c.pid == p) & s.m_c.cap.to_numpy(bool)]
                        for col in SHAPE_COLS:
                            row[f"mc_{lab}_{col}"] = float(mc[col].mean())
                        _, pm, pf, _, _ = s.mult(KMAX, p)
                        row[f"mc_{lab}_p0_only"], row[f"mc_{lab}_p0_fold"] = float(pm[0]), float(pf[0])
                    rows.append(row)
    return pd.DataFrame(rows)


def _fold_rows(ctx: RunContext, mc: RunContext, m: str, tgate: str) -> tuple:
    d = dz._pos_rows(ctx.csv_path(f"efficiency_fit_by_position_fold_data_{tgate}_{m}"))
    q = dz._pos_rows(mc.csv_path(f"efficiency_fit_by_position_fold_mcnominal_{tgate}_{m}"))
    if d is None or q is None:
        raise SystemExit(f"[livetime] {m}: foldeff CSVs for gate {tgate} missing (data or MC)")
    key = lambda r: (float(r.x_pos), float(r.y_pos), float(r.z_pos))  # noqa: E731
    h = lambda s: np.array([float(v) for v in s.split()])  # noqa: E731
    return ({key(r): h(r.hist) for r in d.itertuples()}, {key(r): h(r.hist) for r in q.itertuples()})


def expected_and_fold(ctx: RunContext, mc: RunContext, T: dict, lt: pd.DataFrame, sb: pd.DataFrame,
                      methods: list, src_runs: list, bkg_run: int, tgate: str, activities: list) -> tuple:
    L = lt.set_index("run")
    k = sb.drop_duplicates("run").set_index("run").k_cosmic
    b = T[bkg_run]
    ex, fo = [], []
    for run in src_runs:
        g = T[run]
        pos = (float(g.sourceX.iloc[0]), float(g.sourceY.iloc[0]), float(g.sourceZ.iloc[0]))
        for veto in ("noveto", "veto"):
            kept = lambda x: x[~x.is_cosmic] if veto == "veto" else x  # noqa: E731
            gs, bs = kept(g[g.gate]), kept(b[b.gate])
            r_fs = len(bs) / L.live_s[bkg_run]
            n_fs = r_fs * L.live_s[run]
            n_fs_k = n_fs * k[run]
            r_gate = len(gs) / L.live_s[run]
            row = dict(run=run, pos=str(pos), veto=veto, n_gate=len(gs), live_s=L.live_s[run],
                       rate_gate_live=r_gate, n_gate_bkg=len(bs), rate_fs_live=r_fs,
                       n_fs_expected=n_fs, n_fs_expected_kcorr=n_fs_k, f_fs=n_fs / len(gs), f_fs_kcorr=n_fs_k / len(gs),
                       rate_src_gate=r_gate - r_fs, rate_all_excess=L.lambda_hz[run] - L.lambda_hz[bkg_run],
                       n_wall_naive_fs=len(bs) * L.sql_s[run] / L.sql_s[bkg_run],
                       n_mc_equiv=len(gs) - n_fs)
            for a in activities:
                cap = a * BR_444
                row[f"p_tag_gate_A{a:g}"] = (r_gate - r_fs) / cap
                row[f"max_gate_trig_A{a:g}"] = cap * L.live_s[run]
            ex.append(row)
        for m in methods:
            dh, mh = _fold_rows(ctx, mc, m, tgate)
            if pos not in dh:
                raise SystemExit(f"[livetime] position {pos} of run {run} not in the {m} fold CSV")
            d, mcounts = dh[pos], mh[pos]
            nb = len(d)
            mine = _pmf(g[g.gate & ~g.is_cosmic][f"n_{m}"], nb)
            p0 = mcounts / mcounts.sum()
            p0e = np.sqrt(p0 * (1 - p0) / mcounts.sum())
            e = {x["veto"]: x for x in ex if x["run"] == run}
            sets = {"veto": (d, b[b.gate & ~b.is_cosmic], b[(b.IC_adjusted < 700) & ~b.is_cosmic])}
            for veto, (dd, fs, prox) in sets.items():
                pfs = _pmf(fs[f"n_{m}"], nb)
                ppx = _pmf(prox[f"n_{m}"], nb)
                compat = _chi2(pfs, ppx / ppx.sum(), np.sqrt(ppx / ppx.sum() * (1 - ppx / ppx.sum()) / ppx.sum()))
                models = [("M0 MC x pool", p0, p0e),
                          ("M1 + false starts (6264 gate)", mixture(p0, pfs / pfs.sum(), e[veto]["f_fs"]), p0e),
                          ("M1k + false starts, k-corrected", mixture(p0, pfs / pfs.sum(), e[veto]["f_fs_kcorr"]), p0e),
                          ("M1p + false starts (6264 IC<700 proxy)", mixture(p0, ppx / ppx.sum(), e[veto]["f_fs"]), p0e)]
                for name, p, pe in models:
                    x, dof, pv = _chi2(dd, p, pe)
                    fo.append(dict(run=run, pos=str(pos), method=m, veto=veto, model=name, n_data=dd.sum(),
                                   n_mc=mcounts.sum(), chi2=x, dof=dof, p=pv, mean_pred=float(np.arange(nb) @ p),
                                   mean_data=float(np.arange(nb) @ dd / dd.sum()),
                                   n_fs_pmf=pfs.sum(), n_proxy_pmf=ppx.sum(), fs_vs_proxy_chi2=compat[0],
                                   fs_vs_proxy_p=compat[2],
                                   data_hist_matches_fold_csv=bool(np.array_equal(mine, d)) if veto == "veto" else np.nan,
                                   **{f"data_p{i}": dd[i] / dd.sum() for i in range(nb)},
                                   **{f"pred_p{i}": p[i] for i in range(nb)},
                                   **{f"fs_p{i}": pfs[i] / max(pfs.sum(), 1) for i in range(nb)}))
    return pd.DataFrame(ex), pd.DataFrame(fo)


# ---------------------------------------------------------------------------------------------- pages
def _style(runs: list, bkg_run: int) -> dict:
    src = [r for r in runs if r != bkg_run]
    return {r: dict(color=COLORS[i], marker=MARKERS[i]) for i, r in enumerate(src + [bkg_run])}


def _label(run: int, kind: str) -> str:
    return f"{kind} ({run})"


def page_rates(out, sb: pd.DataFrame, bkg_run: int):
    import matplotlib.pyplot as plt
    st = _style(list(sb.run.unique()), bkg_run)
    names = [c[0] for c in CLASSES]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for run, q in sb[sb.cls.isin(names)].groupby("run"):
        q = q.set_index("cls").loc[names]
        lab = _label(run, q.kind.iloc[0])
        ax[0].errorbar(x, q.rate_live, q.rate_live_err, lw=2, ms=8, label=lab, **st[run])
        ax[1].plot(x, q.rate_sql, lw=2, ms=8, label=lab, **st[run])
        ax[2].errorbar(x, q.rate_live_kept, q.rate_live_kept_err, lw=2, ms=8, label=lab, **st[run])
    ttl = ["triggers per live second", "triggers per wall-clock second (run database)",
           "triggers per live second, after the cosmic veto"]
    for a, t in zip(ax, ttl):
        a.set_xticks(x, names, rotation=30, fontsize=9)
        a.set_yscale("log")
        a.set_xlabel("BGO IC class")
        a.set_title(t, fontsize=11)
        a.legend(fontsize=8)
    ax[0].set_ylabel("rate [Hz]")
    k = sb.drop_duplicates("run").set_index("run")
    ks = ", ".join(f"{r}: {k.k_cosmic[r]:.3f} ± {k.k_cosmic_err[r]:.3f}" for r in k.index if r != bkg_run)
    two_titles(fig, "Trigger rate in each IC class",
               f"live time = sum(dt − tau), tau = 0.1 % quantile of dt; IC > 3000 vetoed rate, source / no-source: {ks}")
    out.save(fig, "rates_by_ic")


def page_tank(out, tk: pd.DataFrame, bkg_run: int, method: str, vetos: tuple = ("noveto", "veto")):
    import matplotlib.pyplot as plt
    names = [c[0] for c in CLASSES] + [GATE]
    short = [c[0] for c in CLASSES] + ["gate"]
    x = np.arange(len(names))
    st = _style(list(tk.run.unique()), bkg_run)
    fig, ax = plt.subplots(len(vetos), 3, figsize=(16, 4.25 * len(vetos)), squeeze=False)
    for i, veto in enumerate(vetos):
        for run, q in tk[(tk.veto == veto) & (tk.method == method)].groupby("run"):
            q = q.set_index("cls").reindex(names)
            lab = _label(run, q.kind.dropna().iloc[0])
            ax[i, 0].plot(x, q.p0, lw=2, ms=8, label=lab, **st[run])
            ax[i, 1].plot(x, q.clusters_per_trig, lw=2, ms=8, label=lab, **st[run])
            ok = q.clusters >= 50
            ax[i, 2].errorbar(x[ok], q.capture_frac[ok], q.capture_frac_err[ok], lw=2, ms=8, label=lab, **st[run])
        tag = "no cosmic veto" if veto == "noveto" else "cosmic veto applied"
        for a, t in zip(ax[i], ("P(no cluster)", "clusters per trigger, 2–67 µs",
                                f"capture-like share, ≥ 50 clusters (tau {TAU_CAP_US} µs fixed)")):
            a.set_xticks(x, short, rotation=30, fontsize=9)
            a.set_title(f"{t}, {tag}", fontsize=10)
            a.legend(fontsize=8)
    two_titles(fig, f"Tank activity in each IC class ({method})",
               "clusters in 2–67 µs; gate = 700 < IC < 1200 and no second pulse")
    out.save(fig, f"tank_by_ic_{method}")


def page_tspec(out, sp: pd.DataFrame, lt: pd.DataFrame, bkg_run: int, method: str, veto: str):
    import matplotlib.pyplot as plt
    L = lt.set_index("run")
    names = [c[0] for c in CLASSES] + [GATE]
    st = _style(list(sp.run.unique()), bkg_run)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    ax = ax.ravel()
    src = [r for r in sp.run.unique() if r != bkg_run]
    for a, name in zip(ax, names):
        q = sp[(sp.cls == name) & (sp.method == method) & (sp.veto == veto)]
        for run in src:
            r = q[q.run == run]
            c = 0.5 * (r.lo + r.hi)
            a.plot(c, r.n / L.live_s[run], lw=2, ms=6, label=_label(run, L.kind[run]), **st[run])
        r = q[q.run == bkg_run]
        a.plot(0.5 * (r.lo + r.hi), r.n / L.live_s[bkg_run], lw=2, ms=6, label=_label(bkg_run, L.kind[bkg_run]),
               **st[bkg_run])
        a.set_yscale("log")
        a.set_title(name, fontsize=10)
        a.set_xlabel("cluster time [µs]")
        a.legend(fontsize=7)
    ax[-1].axis("off")
    ax[0].set_ylabel("clusters per live second per 2.5 µs")
    tag = "no cosmic veto" if veto == "noveto" else "cosmic veto applied"
    two_titles(fig, f"Cluster time in each IC class, per live second ({method}, {tag})",
               "both runs divided by their own live time, so the no-source curve is what false starts and cosmics give")
    out.save(fig, f"tspec_{method}_{veto}")


def page_fold(out, fo: pd.DataFrame):
    import matplotlib.pyplot as plt
    q0 = fo[fo.veto == "veto"]
    groups = list(q0.groupby(["run", "method"]))
    fig, ax = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.8), squeeze=False)
    for a, ((run, m), q) in zip(ax[0], groups):
        cols = sorted([c for c in q.columns if c.startswith("pred_p")], key=lambda s: int(s[6:]))
        x = np.arange(len(cols))
        r0 = q.iloc[0]
        a.plot(x, [r0[f"data_p{i}"] for i in x], color="black", marker="o", lw=2, ms=8, label="data, IC gate")
        for (_, r), c, mk in zip(q.iterrows(), COLORS, MARKERS[1:] + ["v"]):
            a.plot(x, [r[c_] for c_ in cols], color=c, marker=mk, lw=1.5, ms=7, ls="--",
                   label=f"{r.model}: chi2/dof {r.chi2:.0f}/{r.dof}")
        a.set_yscale("log")
        a.set_xticks(x, [str(i) if i < len(x) - 1 else f"≥{i}" for i in x])
        a.set_xlabel("clusters per trigger")
        a.set_title(f"run {run} {r0.pos}, {m}", fontsize=10)
        a.legend(fontsize=7)
    two_titles(fig, "Clusters per trigger: MC ⊗ pool and the false-start mixture",
               "cosmic veto applied; false-start weight f from the live-time-normalised 6264 gate rate")
    out.save(fig, "fold_falsestarts")


# ---------------------------------------------------------------------------------------------- driver
def page_shapes(out, sh: pd.DataFrame, labels: list, method: str, background: str):
    import matplotlib.pyplot as plt
    s = sh[(sh.method == method) & (sh.background == background)]
    x = np.arange(len(SHAPE_CLASSES))
    names = ["IC 300-700", "IC 700-1200", "IC 1200-3000", "gate"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [("n_hits", "hits per cluster"), ("pe_per_hit", "charge per hit [p.e.]"), ("p0", "P(0 clusters)")]
    for a, (col, yl) in zip(ax, panels):
        for i, run in enumerate(sorted(s.run.unique())):
            r = s[s.run == run].set_index("cls").loc[list(SHAPE_CLASSES)]
            a.plot(x, r[f"data_sub_{col}" if col != "p0" else "data_p0"], "-o", color="k", ms=5, lw=1.2,
                   label=f"data {run} − background" if col != "p0" else f"data {run}")
            for j, lab in enumerate(labels):
                v = r[f"mc_{lab}_{col}" if col != "p0" else f"mc_{lab}_p0_fold"]
                a.plot(x, v, "-" + MARKERS[j + 1], color=COLORS[j], ms=4, lw=1.0,
                       label=lab if col != "p0" else f"{lab} ⊗ background")
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=8)
        a.set_ylabel(yl)
        a.legend(frameon=False, fontsize=7)
    top = two_titles(fig, f"{method}: cluster shape and P(0) by IC class",
                     f"vetoed triggers, clusters 2-67 µs; background {background} 6264, scaled per trigger; "
                     "MC capture clusters (no IC tag in MC)")
    fig.tight_layout(rect=(0, 0, 1, top))
    out.save(fig, f"shapes_{method}_{background}")


def run(ctx: RunContext, mc: RunContext, runs_db: str, src_runs: list, bkg_run: int, extra_runs: list,
        methods: list, tgate: str, activities: list, gap_s: float, out_name: str, shape_mcs: Optional[dict] = None,
        shape_bkgs: tuple = ("ic", "ungated")):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    if bkg_run != int((ctx.fit_params or {})["background_run"]):
        raise SystemExit("[livetime] --nosource-run differs from fit_params.background_run of the data config")
    sql = {int(k): v for k, v in yaml.safe_load(Path(runs_db).read_text())["runs"].items()}
    cand = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_none"),
                           columns=["trigger", "method", "t_med", "pe", "n_hits", "run"],
                           filters=[("run", "in", src_runs + [bkg_run]), ("method", "in", methods)])
    tree = []
    for r in src_runs + [bkg_run]:
        cf = tree_clusters(ctx, r)
        cf = cf[(cf.clusterTime >= dz.WIN[0]) & (cf.clusterTime <= dz.WIN[1])]
        tree.append(pd.DataFrame(dict(trigger=cf.trigger, method=TREE, t_med=cf.clusterTime, pe=cf.clusterPE,
                                      n_hits=cf.clusterHits, run=r)))
    cand = pd.concat([cand] + tree, ignore_index=True)
    T = {r: triggers(ctx, r, bkg_run, methods, cand) for r in src_runs + [bkg_run]}
    kinds = {r: ("no source" if r == bkg_run else
                 f"source P5 y={float(T[r].sourceY.iloc[0]):+.0f}") for r in T}
    ts = {r: (kinds[r], T[r].eventTankTime.to_numpy()) for r in T}
    for r in extra_runs:
        w = _features(r, bkg_run, True)
        ts[r] = ("extra (waveforms only)", w.timestamp.to_numpy())
    lt = livetime_table(ts, sql, gap_s)
    sb = sideband(T, lt, bkg_run)
    tk, sp = tank(T, lt, methods, cand, bkg_run)
    lt = implied_tau(lt, sb, ts, gap_s)
    ex, fo = expected_and_fold(ctx, mc, T, lt, sb, methods, src_runs, bkg_run, tgate, activities)
    sh = shapes(ctx, shape_mcs, T, cand, methods, src_runs, bkg_run, list(shape_bkgs)) if shape_mcs else None
    pre = f"livetime_{out_name}"
    for name, df in (("livetime", lt), ("sideband", sb), ("tank", tk), ("tspec", sp), ("expected", ex), ("foldfs", fo),
                     ("shapes", sh)):
        if df is not None:
            df.to_csv(ctx.csv_path(f"{pre}_{name}"), index=False)
    with PdfPages(ctx.plot_path(pre, "pdf")) as pdf:
        out = dz.Out(ctx, pdf, pre)
        page_rates(out, sb, bkg_run)
        page_tank(out, tk, bkg_run, TREE)
        for v in ("noveto", "veto"):
            page_tspec(out, sp, lt, bkg_run, TREE, v)
        for m in methods:
            page_tank(out, tk, bkg_run, m, vetos=("veto",))
            page_tspec(out, sp, lt, bkg_run, m, "veto")
        page_fold(out, fo)
        if sh is not None:
            for m in methods:
                for b in shape_bkgs:
                    page_shapes(out, sh, list(shape_mcs), m, b)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(lt[["run", "kind", "n_readouts", "tau_s", "sql_s", "span_s", "gapcut_s", "live_s", "live_fraction",
                  "lambda_hz", "numevents_per_readout"]].to_string(index=False))
        print(sb[["run", "cls", "n", "rate_live", "rate_sql", "rate_live_ratio", "frac_cosmic", "frac_veto_prompt_only",
                  "k_cosmic"]].to_string(index=False))
        print(ex.to_string(index=False))
        print(fo[["run", "method", "veto", "model", "chi2", "dof", "p", "data_p0", "pred_p0"]].to_string(index=False))
    print(f"[livetime] wrote csv {pre}_* and {ctx.plot_path(pre, 'pdf')}")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots livetime")
    p.add_argument("--mc-config", required=True, help="MC whose foldeff fold CSVs define MC x pool")
    p.add_argument("--runs-db", required=True, help="YAML with run-database start/stop per run")
    p.add_argument("--source-runs", required=True, help="comma list of source-in runs in the data event index")
    p.add_argument("--nosource-run", required=True, type=int)
    p.add_argument("--extra-runs", default="", help="runs for the live-time table only (waveform timestamps)")
    p.add_argument("--methods", required=True)
    p.add_argument("--template-gate", required=True, help="gate tag of the foldeff CSVs, e.g. none_plus6256")
    p.add_argument("--activity-nps", required=True, help="source neutron activities [n/s] for the rate budget")
    p.add_argument("--gap-s", required=True, type=float, help="intervals at least this long are DAQ gaps")
    p.add_argument("--out-name", required=True)
    p.add_argument("--shape-mc", help="shapes part: comma list of LABEL:mc-config, e.g. wf:configs/mc_x.yaml")
    p.add_argument("--shape-backgrounds", default="ic,ungated", help="shapes part: comma list of ic, ungated")
    a = p.parse_args(list(argv) if argv else [])
    smc = {k: load_context(v) for k, v in (x.split(":", 1) for x in a.shape_mc.split(","))} if a.shape_mc else None
    run(ctx, load_context(a.mc_config), a.runs_db, [int(x) for x in a.source_runs.split(",")], a.nosource_run,
        [int(x) for x in a.extra_runs.split(",") if x], a.methods.split(","), a.template_gate,
        [float(x) for x in a.activity_nps.split(",")], a.gap_s, a.out_name, smc, tuple(a.shape_backgrounds.split(",")))
