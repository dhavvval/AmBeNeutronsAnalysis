"""
Deck stages of the step-3 forward fold (opt-in, existing stages unchanged):

    ambe stats foldeff --config configs/step3_ambe2v4_bkgnorm.yaml --fold-model forward --stage datamc
    ambe stats foldeff --config configs/step3_ambe2v4_bkgnorm.yaml --fold-model forward --stage deckplots

datamc writes forwardfold_dm_*.csv:
- clmult: clusters per trigger of the source runs vs F_r x 6264 (gain-matched IC) and alpha_r x 6264, four trigger
  classes (the 10/29 side plot with the frozen factors);
- clusters: per-cluster time / PE / hits per position (MC label) and port: data - F x 6264 per AmBe trigger, the
  setup-A MC with each capture cluster kept with probability s_port, the MC non-capture clusters and the 6256 dummy;
- capturetime: lmfit NeutCapture fits (ambe.plots.basic.fit) of data and MC per position and port;
- icspec, multpos, cutflow, tail_* tables.
deckplots draws them into plots/deck_chain/, every comparison with a residual panel.
Data and MC use the ToolAnalysis tree quantities (clusterTime, clusterPE, clusterHits); MC clusters take their
capture truth (cap_dom) from candidates_mc by the nearest median hit time within 50 ns.
"""
from __future__ import annotations

import contextlib
import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..context import RunContext
from . import forward_fold as ff

KMAX = 8
EDGES = {"time": np.arange(0.0, 71.0, 1.0), "pe": np.arange(0.0, 102.0, 2.0), "hits": np.arange(4.5, 72.5, 2.0),
         "dt": np.arange(0.0, 66.0, 1.0)}
COL = {"time": "clusterTime", "pe": "clusterPE", "hits": "clusterHits"}
XLAB = {"time": "cluster time [µs]", "pe": "cluster charge [p.e.]", "hits": "hits per cluster",
        "dt": "time between consecutive clusters [µs]", "mult": "clusters per trigger"}
IC_EDGES = np.arange(-1000.0, 18000.0 + 250.0, 250.0)
SELECTIONS = (("ic3000_noveto", "IC > 3000, no veto"), ("ic3000_veto", "IC > 3000, veto"),
              ("gate_veto", "IC gate 700–1200, veto"), ("all_noveto", "all triggers, no veto"))
PAIRS = ((6046, 6265), (6056, 6270), (6165, 6266))
FIT_MAX_US = 66.0   # last full 1 us bin: the readout ends at 66.5 us in data and MC, so 66-67 is half empty
PLAIN_FIT = ("fit A·(1 − e^(−t/t_th))·e^(−t/τ) + B: rise, capture decay τ, flat level B; 1 µs bins, 2–66 µs; "
             "lower panel = (data − fit) / error")
FITTER = ("lmfit least_squares (ambe.plots.basic.fit): NeutCapture A(1−e^(−t/therm))e^(−t/τ)+B, χ², 1 µs bins, "
          "2–66 µs (66–67 half empty: readout ends at 66.5 µs), B free")
TZ = ZoneInfo("America/Chicago")


def _hist(v, var: str) -> np.ndarray:
    e = EDGES[var]
    v = np.asarray(v, float)
    if var in ("pe", "hits"):
        v = np.clip(v, e[0], e[-1] - 1e-9)
    return np.histogram(v, e)[0].astype(float)


def _vals(df: pd.DataFrame, var: str) -> np.ndarray:
    v = df[COL[var]].to_numpy(float)
    return v / 1000.0 if var == "time" else v


# ---------------------------------------------------------------------------------------------- loaders
class Data:
    """Every trigger of a run (IC, gate, veto, n) and its tree clusters in the fold window."""

    def __init__(self, su: ff.Setup):
        self.su = su
        m = su.blk["data_method"]
        self.tn = ff.gated_triggers(su.dctx, m, "none")[["run", "trigger", "eventTankTime", "is_cosmic"]]
        self.ic = ff.gated_triggers(su.dctx, m, "ic")[["run", "trigger"]]
        self.tag = su.blk["features_tag"]
        self._c = {}

    def features(self, run: int) -> pd.DataFrame:
        f = Path(f"TriggerSummary/WaveformFeatures_{self.tag}_{run}.parquet")
        w = pd.read_parquet(f, columns=["timestamp", "accepted", "IC_adjusted"])
        return w[w.timestamp > 0].drop_duplicates("timestamp")

    def run(self, run: int) -> tuple:
        if run in self._c:
            return self._c[run]
        t = self.tn[self.tn.run == run].reset_index(drop=True)
        n0 = len(t)
        t = t.merge(self.features(run), left_on="eventTankTime", right_on="timestamp", how="left")
        if len(t) != n0:
            raise SystemExit(f"[forwardfold datamc] run {run}: the feature join changed the trigger count")
        t["gate"] = t.trigger.isin(self.ic.trigger[self.ic.run == run])
        cf = pd.read_parquet(self.su.dctx.parquet_path(f"{self.su.dctx.run_name}__random_cfclusters_{run}"),
                             columns=["trigger", "clusterTime", "clusterPE", "clusterHits"])
        lo, hi = self.su.window
        cf = cf[(cf.clusterTime >= lo) & (cf.clusterTime <= hi)]
        t["n"] = t.trigger.map(cf.groupby("trigger").size()).fillna(0).astype(int)
        cf = cf.merge(t[["trigger", "gate", "is_cosmic", "n"]], on="trigger")
        self._c[run] = (t, cf)
        return t, cf

    def gv(self, run: int) -> tuple:
        """Gated veto-passing triggers and their clusters (the step-3 sample)."""
        t, cf = self.run(run)
        return t[t.gate & ~t.is_cosmic], cf[cf.gate & ~cf.is_cosmic]


class MC:
    """Veto-passing setup-A MC events and their tree clusters in the window, with capture truth."""

    def __init__(self, su: ff.Setup):
        mctx, m = su.mctx, su.blk["mc_method"]
        rn = mctx.run_name
        ei = pd.read_parquet(mctx.parquet_path(f"{rn}__eventindex_mc_{m}"),
                             columns=["_source_file", "eventNumber", "eventID", "label", "is_cosmic"])
        self.n_events_all = len(ei)
        ei = ei[~ei.is_cosmic].drop(columns="is_cosmic")
        self.N = ei.groupby("label").size()
        cf = pd.read_parquet(mctx.parquet_path(f"{rn}__clusterfinder"))
        lo, hi = su.window
        cf = cf[(cf.clusterTime >= lo) & (cf.clusterTime <= hi)]
        cf = cf.merge(ei, on=["_source_file", "eventID"])
        c = pd.read_parquet(mctx.parquet_path(f"{rn}__candidates_mc"),
                            columns=["_source_file", "eventNumber", "method", "t_med", "cap_dom"])
        c = c[c.method == m].drop(columns="method")
        a = cf.assign(_t=cf.clusterTime).sort_values("_t")
        b = c.assign(_t=c.t_med).sort_values("_t")
        j = pd.merge_asof(a, b, on="_t", by=["_source_file", "eventNumber"], tolerance=50.0, direction="nearest")
        self.n_clusters, self.n_unmatched = len(j), int(j.cap_dom.isna().sum())
        self.n_candidates = len(c.merge(ei[["_source_file", "eventNumber"]], on=["_source_file", "eventNumber"]))
        j["cap"] = j.cap_dom.fillna(-1).to_numpy() >= 0
        j["n"] = j.groupby(["_source_file", "eventNumber"]).clusterTime.transform("size")
        self.cf = j[["label", "_source_file", "eventNumber", "clusterTime", "clusterPE", "clusterHits", "cap", "n"]]
        key = ["_source_file", "eventNumber"]
        cnt = j.groupby(key).size().rename("n").reset_index()
        self.ev = ei.merge(cnt, on=key, how="left").fillna({"n": 0}).astype({"n": int})
        print(f"[forwardfold datamc] MC: {self.n_clusters} tree clusters in the window on {int(self.N.sum())} "
              f"veto-passing events; {self.n_candidates} candidates; unmatched {self.n_unmatched}", flush=True)


# ---------------------------------------------------------------------------------------------- tables
def _mask(t: pd.DataFrame, key: str, g: float, ic_cut: float) -> np.ndarray:
    ic = t.IC_adjusted.to_numpy(float) * g
    veto = t.is_cosmic.to_numpy(bool)
    if key == "ic3000_noveto":
        return ic > ic_cut
    if key == "ic3000_veto":
        return (ic > ic_cut) & ~veto
    if key == "gate_veto":
        return t.gate.to_numpy(bool) & ~veto
    return np.ones(len(t), bool)


def clmult(D: Data, su: ff.Setup) -> pd.DataFrame:
    ic_cut = float(su.blk.get("ic_cosmic", 3000.0))
    B, _ = D.run(su.bkg_run)
    rows = []
    for r in su.fac.itertuples():
        S, _ = D.run(r.run)
        for key, _lab in SELECTIONS:
            hs = ff.hist(S.n[_mask(S, key, 1.0, ic_cut)], KMAX)
            hb = ff.hist(B.n[_mask(B, key, float(r.g), ic_cut)], KMAX)
            hr = ff.hist(B.n[_mask(B, key, 1.0, ic_cut)], KMAX)
            for k in range(KMAX + 1):
                rows.append(dict(run=r.run, port=int(r.port), x=r.x, y=r.y, z=r.z, label=su.labels[r.run], selection=key,
                                 k=k, n_source=hs[k], n_6264_g=hb[k], n_6264_raw=hr[k], F=r.F_rsb_g,
                                 F_err=r.F_rsb_g_err, alpha=r.alpha, g=r.g))
    return pd.DataFrame(rows)


def cutflow(D: Data, su: ff.Setup) -> pd.DataFrame:
    rows = []
    runs = list(su.fac.run) + [su.bkg_run, int(su.blk["dummy_run"])]
    nfold = su.counts.groupby("run").size()
    for run in runs:
        t, _ = D.run(run)
        gv = int((t.gate & ~t.is_cosmic).sum())
        if run in nfold.index and gv != int(nfold[run]):
            raise SystemExit(f"[forwardfold datamc] run {run}: {gv} gated veto-passing triggers vs {int(nfold[run])} in the fold")
        rows.append(dict(run=run, kind="source" if run in set(su.fac.run) else
                         ("no source" if run == su.bkg_run else "dummy"),
                         date=datetime.fromtimestamp(float(np.median(t.eventTankTime)) / 1e9, TZ).strftime("%Y-%m-%d"),
                         n_triggers=len(t), n_with_ic=int(t.IC_adjusted.notna().sum()), n_gate=int(t.gate.sum()),
                         n_veto_pass=int((~t.is_cosmic).sum()), n_gate_veto=gv,
                         veto_frac_gate=float(t.is_cosmic[t.gate].mean()) if t.gate.any() else np.nan,
                         mean_n_gate_veto=float(t.n[t.gate & ~t.is_cosmic].mean()) if gv else np.nan))
    return pd.DataFrame(rows)


def _s_of(prm: pd.DataFrame) -> dict:
    q = prm[prm.variant == "nominal"].set_index("group")
    return {g: (float(q.s[g]), float(q.s_err[g])) for g in q.index}


def clusters(D: Data, M: MC, su: ff.Setup, prm: pd.DataFrame) -> tuple:
    """Per-cluster histograms (time / pe / hits) per position (MC label) and per port, and the lmfit capture fits."""
    s_of = _s_of(prm)
    fac = su.fac.assign(group=ff.group_of(su.fac, "port"), label=su.fac.run.map(su.labels))
    Bt, Bc = D.gv(su.bkg_run)
    dm_t, dm_c = D.run(int(su.blk["dummy_run"]))
    dm_c = dm_c[~dm_c.is_cosmic]
    Nd = int((~dm_t.is_cosmic).sum())
    flat_d = float(((dm_c.clusterTime >= 2000) & (dm_c.clusterTime < 67000)).sum()) / Nd / 65.0
    rows, fits, fsfrac = [], [], []
    port_acc = {}
    for lab, f in fac.groupby("label"):
        g = f.group.iloc[0]
        s, s_err = s_of[g]
        parts = [D.gv(int(r)) for r in f.run]
        N = sum(len(p[0]) for p in parts)
        cl = pd.concat([p[1] for p in parts], ignore_index=True)
        Fsum = float(f.F_rsb_g.sum())
        NA = N - Fsum * len(Bt)
        mc = M.cf[M.cf.label == lab]
        Nmc = int(M.N[lab])
        y = float(f.y.iloc[0] + su.label_dy[int(f.run.iloc[0])])
        meta = dict(port=int(f.port.iloc[0]), y=y, runs=" ".join(str(int(r)) for r in f.run), s=s, s_err=s_err)
        for var in ("time", "pe", "hits"):
            e = EDGES[var]
            src, bk = _hist(_vals(cl, var), var), _hist(_vals(Bc, var), var)
            cap, non = _hist(_vals(mc[mc.cap], var), var), _hist(_vals(mc[~mc.cap], var), var)
            dmh = _hist(_vals(dm_c, var), var)
            acc = port_acc.setdefault((g, var), dict(src=0.0, fs=0.0, fs_var=0.0, N=0.0, NA=0.0, mc=0.0, mc_var=0.0,
                                                     non=0.0, non_var=0.0, cap_raw=0.0, non_raw=0.0))
            acc["src"] = acc["src"] + src
            acc["fs"] = acc["fs"] + Fsum * bk
            acc["fs_var"] = acc["fs_var"] + Fsum ** 2 * bk
            acc["N"] += N
            acc["NA"] += NA
            # MC per AmBe trigger at this position, weighted by the position's share of the port's AmBe triggers
            acc["mc"] = acc["mc"] + NA * (s * cap + non) / Nmc
            acc["mc_var"] = acc["mc_var"] + NA ** 2 * (s * s * cap + non) / Nmc ** 2
            acc["non"] = acc["non"] + NA * non / Nmc
            acc["non_var"] = acc["non_var"] + NA ** 2 * non / Nmc ** 2
            acc["cap_raw"] = acc["cap_raw"] + cap
            acc["non_raw"] = acc["non_raw"] + non
            for i in range(len(e) - 1):
                rows.append(dict(level="position", group=lab, var=var, lo=e[i], hi=e[i + 1], **meta, N_trig=N, N_A=NA,
                                 src=src[i], fs=Fsum * bk[i], data_pt=(src[i] - Fsum * bk[i]) / NA,
                                 data_pt_err=np.sqrt(src[i] + Fsum ** 2 * bk[i]) / NA,
                                 mc_pt=(s * cap[i] + non[i]) / Nmc, mc_pt_err=np.sqrt(s * s * cap[i] + non[i]) / Nmc,
                                 mc_non_pt=non[i] / Nmc, mc_non_pt_err=np.sqrt(non[i]) / Nmc, mc_cap_raw=cap[i],
                                 mc_non_raw=non[i], N_mc=Nmc, dummy_pt=dmh[i] / Nd, dummy_pt_err=np.sqrt(dmh[i]) / Nd,
                                 N_dummy=Nd))
        fsfrac.append(dict(label=lab, clusters_source=len(cl), clusters_falsestart=Fsum * len(Bc),
                           frac=Fsum * len(Bc) / max(len(cl), 1)))
        fits += _fits("position", lab, meta, cl, mc, B6256=flat_d * N)
    for (g, var), a in port_acc.items():
        e = EDGES[var]
        f = fac[fac.group == g]
        s, s_err = s_of[g]
        meta = dict(port=int(f.port.iloc[0]), y=np.nan, runs=" ".join(str(int(r)) for r in f.run), s=s, s_err=s_err)
        dmh = _hist(_vals(dm_c, var), var)
        for i in range(len(e) - 1):
            rows.append(dict(level="port", group=g, var=var, lo=e[i], hi=e[i + 1], **meta, N_trig=a["N"], N_A=a["NA"],
                             src=a["src"][i], fs=a["fs"][i], data_pt=(a["src"][i] - a["fs"][i]) / a["NA"],
                             data_pt_err=np.sqrt(a["src"][i] + a["fs_var"][i]) / a["NA"],
                             mc_pt=a["mc"][i] / a["NA"], mc_pt_err=np.sqrt(a["mc_var"][i]) / a["NA"],
                             mc_non_pt=a["non"][i] / a["NA"], mc_non_pt_err=np.sqrt(a["non_var"][i]) / a["NA"],
                             mc_cap_raw=a["cap_raw"][i], mc_non_raw=a["non_raw"][i], N_mc=np.nan,
                             dummy_pt=dmh[i] / Nd, dummy_pt_err=np.sqrt(dmh[i]) / Nd, N_dummy=Nd))
    for g, f in fac.groupby("group"):
        s, s_err = s_of[g]
        cl = pd.concat([D.gv(int(r))[1] for r in f.run], ignore_index=True)
        mc = M.cf[M.cf.label.isin(set(f.label))]
        meta = dict(port=int(f.port.iloc[0]), y=np.nan, runs=" ".join(str(int(r)) for r in f.run), s=s, s_err=s_err)
        N = sum(len(D.gv(int(r))[0]) for r in f.run)
        fits += _fits("port", g, meta, cl, mc, B6256=flat_d * N)
    return pd.DataFrame(rows), pd.DataFrame(fits), pd.DataFrame(fsfrac)


def _lmfit(t_us: np.ndarray, B_fixed=None):
    from ..plots.basic import AmBeNeutronAnalyzer, fit
    an = AmBeNeutronAnalyzer()
    h = np.histogram(t_us, an.fitting_config["time_bins"], an.fitting_config["time_range"])[0]
    with contextlib.redirect_stdout(io.StringIO()):
        an.update_fitting_config(background_bounds=(0.0, np.inf), initial_amplitude=float(max(h.max(), 1) * 2.0),
                                 fit_max_time=FIT_MAX_US)
        if B_fixed is not None:
            an.update_fitting_config(initial_background=float(B_fixed))
            return fit(t_us, float_B=False, backend="lmfit", analyzer=an, min_counts=100)
        return fit(t_us, float_B=True, backend="lmfit", analyzer=an, min_counts=100,
                   initial_background=float(max(np.mean(h[60:66]), 0.1)))


def _fits(level: str, group: str, meta: dict, cl: pd.DataFrame, mc: pd.DataFrame, B6256=None) -> list:
    """lmfit capture fits: data and MC with B free; data with B fixed to the 6256 flat level (B6256 per 1 us bin)."""
    out = []
    samples = [("data", cl.clusterTime, None), ("mc_all", mc.clusterTime, None),
               ("mc_capture", mc.clusterTime[mc.cap], None)]
    if B6256 is not None:
        samples.append(("data_B6256", cl.clusterTime, B6256))
    for sample, t, bfix in samples:
        r = _lmfit(t.to_numpy(float) / 1000.0, bfix)
        row = dict(level=level, group=group, sample=sample, **meta, n_clusters=len(t),
                   fitter=FITTER if bfix is None else FITTER.replace("B free", "B fixed to the 6256 flat level"),
                   B_fixed=bfix if bfix is not None else np.nan, fit_ok=r is not None)
        if r is not None:
            row.update({k: r[k] for k in ("A", "A_err", "therm", "therm_err", "tau", "tau_err", "B", "B_err", "chi2",
                                          "ndof", "redchi", "N")})
        out.append(row)
    return out


def ic_spectra(D: Data, su: ff.Setup) -> pd.DataFrame:
    """BGO IC of every trigger (no cuts): source runs at a position vs sum_r F_r x 6264(g_r IC) and alpha x 6264."""
    icb = D.features(su.bkg_run).IC_adjusted.to_numpy(float)
    e = IC_EDGES
    rows = []
    fac = su.fac.assign(label=su.fac.run.map(su.labels))
    for lab, f in fac.groupby("label"):
        hs = sum(np.histogram(D.features(int(r)).IC_adjusted.to_numpy(float), e)[0].astype(float) for r in f.run)
        hb = [np.histogram(icb * float(g), e)[0].astype(float) for g in f.g]
        bF = sum(F * h for F, h in zip(f.F_rsb_g, hb))
        bF_err = sum(F * np.sqrt(h) for F, h in zip(f.F_rsb_g, hb))
        hraw = np.histogram(icb, e)[0].astype(float)
        a = float(f.alpha.sum())
        for i in range(len(e) - 1):
            rows.append(dict(label=lab, port=int(f.port.iloc[0]), y=float(f.y.iloc[0] + su.label_dy[int(f.run.iloc[0])]),
                             runs=" ".join(str(int(r)) for r in f.run), lo=e[i], hi=e[i + 1], source=hs[i],
                             bkg_F=bF[i], bkg_F_err=bF_err[i], bkg_alpha=a * hraw[i], bkg_alpha_err=a * np.sqrt(hraw[i]),
                             F_sum=float(f.F_rsb_g.sum()), alpha_sum=a))
    return pd.DataFrame(rows)


def multpos(ctx: RunContext, su: ff.Setup) -> pd.DataFrame:
    b = pd.read_csv(ctx.csv_path("forwardfold_bins"))
    b = b[b.group != "background"].copy()
    b["label"] = b.run.map(su.labels)
    fac = su.fac.set_index("run")
    q = b.groupby(["label", "group", "n"]).agg(data=("data", "sum"), pred=("pred", "sum")).reset_index()
    first = b.drop_duplicates("label").set_index("label").run
    q["port"] = q.label.map(lambda lab: int(fac.port[first[lab]]))
    q["y"] = q.label.map(lambda lab: float(fac.y[first[lab]] + su.label_dy[int(first[lab])]))
    q["runs"] = q.label.map(b.groupby("label").run.apply(lambda v: " ".join(str(int(x)) for x in sorted(set(v)))))
    return q


def tail(D: Data, M: MC, su: ff.Setup, ctx: RunContext, cut: pd.DataFrame) -> dict:
    out = {}
    b = pd.read_csv(ctx.csv_path("forwardfold_bins"))
    b = b[b.group != "background"]
    date = cut.set_index("run").date
    rows = []
    for (run, g), q in b.groupby(["run", "group"]):
        q = q.set_index("n")
        N = float(q.data.sum())
        d = str(date[run])
        rows.append(dict(run=run, group=g, date=d, period=datetime.strptime(d, "%Y-%m-%d").strftime("%B"), N_trig=N,
                         **{f"data_n{n}": float(q.data[n]) for n in q.index},
                         **{f"pred_n{n}": float(q.pred[n]) for n in q.index}))
    out["tail_runs"] = pd.DataFrame(rows)
    fac = su.fac.assign(group=ff.group_of(su.fac, "port"), label=su.fac.run.map(su.labels))
    rows = []
    for g, f in fac.groupby("group"):
        cl = pd.concat([D.gv(int(r))[1].assign(run=int(r)) for r in f.run], ignore_index=True)
        trig = pd.concat([D.gv(int(r))[0] for r in f.run], ignore_index=True)
        mc = M.cf[M.cf.label.isin(set(f.label))]
        mev = M.ev[M.ev.label.isin(set(f.label))]
        for sample, c, ntr in (("data", cl, trig.n), ("mc", mc, mev.n)):
            for cls, sel, nsel in (("n12", c.n.isin([1, 2]), ntr.isin([1, 2])), ("n3p", c.n >= 3, ntr >= 3)):
                cc = c[sel]
                for var in ("time", "pe", "hits"):
                    h = _hist(_vals(cc, var), var)
                    e = EDGES[var]
                    rows += [dict(group=g, sample=sample, cls=cls, var=var, lo=e[i], hi=e[i + 1], count=h[i],
                                  n_clusters=len(cc), n_trig=int(nsel.sum())) for i in range(len(h))]
                key = ["run", "trigger"] if sample == "data" else ["_source_file", "eventNumber"]
                for dcls, dc, dn in (("n2", c[c.n == 2], int((ntr == 2).sum())), ("n3p", cc, int(nsel.sum()))):
                    if cls != "n3p":
                        continue
                    ss = dc.sort_values(key + ["clusterTime"])
                    dt = ss.groupby(key)["clusterTime"].diff().dropna().to_numpy(float) / 1000.0
                    h = _hist(dt, "dt")
                    e = EDGES["dt"]
                    rows += [dict(group=g, sample=sample, cls=dcls, var="dt", lo=e[i], hi=e[i + 1], count=h[i],
                                  n_clusters=len(dt), n_trig=dn) for i in range(len(h))]
    out["tail_shapes"] = pd.DataFrame(rows)
    rows = []
    for a, b2 in PAIRS:
        for run in (a, b2):
            t, c = D.gv(run)
            pair = f"{a}_{b2}"
            h = ff.hist(t.n, KMAX)
            rows += [dict(pair=pair, run=run, var="mult", lo=k - 0.5, hi=k + 0.5, count=h[k], N=len(t))
                     for k in range(KMAX + 1)]
            for var in ("time", "pe", "hits"):
                hh = _hist(_vals(c, var), var)
                e = EDGES[var]
                rows += [dict(pair=pair, run=run, var=var, lo=e[i], hi=e[i + 1], count=hh[i], N=len(t))
                         for i in range(len(hh))]
    out["tail_pairs"] = pd.DataFrame(rows)
    return out


def overlay_reuse(M: MC) -> pd.DataFrame:
    """Capture share per MC multiplicity class and how often the same overlay (non-capture) cluster recurs."""
    rows = []
    for cls, sel in (("n1", M.cf.n == 1), ("n2", M.cf.n == 2), ("n3p", M.cf.n >= 3), ("all", M.cf.n >= 1)):
        q = M.cf[sel]
        non = q[~q.cap]
        key = non.clusterTime.round(1).astype(str) + "_" + non.clusterPE.round(3).astype(str)
        rows.append(dict(cls=cls, clusters=len(q), capture_share=float(q.cap.mean()) if len(q) else np.nan,
                         noncapture_clusters=len(non), noncapture_distinct=int(key.nunique()),
                         mean_reuse=len(non) / max(int(key.nunique()), 1),
                         max_reuse=int(key.value_counts().max()) if len(key) else 0))
    return pd.DataFrame(rows)


def datamc(ctx: RunContext, su: ff.Setup) -> None:
    prm_p = ctx.csv_path("forwardfold_params")
    if not Path(prm_p).exists():
        raise SystemExit("[forwardfold datamc] run --stage data first (needs forwardfold_params.csv)")
    prm = pd.read_csv(prm_p)
    D, M = Data(su), MC(su)
    cut = cutflow(D, su)
    cut.to_csv(ctx.csv_path("forwardfold_dm_cutflow"), index=False)
    print("[forwardfold datamc] cutflow: gated veto-passing counts match the fold for every run", flush=True)
    cm = clmult(D, su)
    cm.to_csv(ctx.csv_path("forwardfold_dm_clmult"), index=False)
    cl, fits, fsf = clusters(D, M, su, prm)
    cl.to_csv(ctx.csv_path("forwardfold_dm_clusters"), index=False)
    fits.to_csv(ctx.csv_path("forwardfold_dm_capturetime"), index=False)
    fsf.to_csv(ctx.csv_path("forwardfold_dm_falsestart_clusters"), index=False)
    pd.DataFrame([dict(mc_clusters=M.n_clusters, mc_candidates=M.n_candidates, mc_unmatched=M.n_unmatched,
                       mc_events=int(M.N.sum()), mc_events_all=M.n_events_all)]).to_csv(ctx.csv_path("forwardfold_dm_mcjoin"), index=False)
    overlay_reuse(M).to_csv(ctx.csv_path("forwardfold_dm_overlay_reuse"), index=False)
    ic_spectra(D, su).to_csv(ctx.csv_path("forwardfold_dm_icspec"), index=False)
    multpos(ctx, su).to_csv(ctx.csv_path("forwardfold_dm_multpos"), index=False)
    for k, v in tail(D, M, su, ctx, cut).items():
        v.to_csv(ctx.csv_path(f"forwardfold_dm_{k}"), index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        q = fits[fits.level == "port"][["group", "sample", "n_clusters", "tau", "tau_err", "therm", "therm_err", "redchi"]]
        print(q.to_string(index=False))
    print(f"[forwardfold datamc] wrote forwardfold_dm_*.csv in {Path(ctx.csv_path('x')).parent}")


# ---------------------------------------------------------------------------------------------- figures
PORTS = (1, 2, 3, 4, 5)
YS = (-100.0, -50.0, 0.0, 50.0, 100.0)
RESID_NOTE = "residual = A / B (ratio)"


class Out:
    """PNG + PDF per figure in plots/deck_chain/, an index with both titles, and a residual-axis check."""

    def __init__(self, ctx: RunContext):
        self.d = Path(ctx.plots_dir) / "deck_chain"
        self.d.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def save(self, fig, name: str, simple: str, detail: str, compare: bool = True):
        import matplotlib.pyplot as plt
        from ..plotting import two_titles
        if compare and not any(a.get_gid() == "residual" for a in fig.axes):
            raise SystemExit(f"[forwardfold deckplots] {name}: comparison figure without a residual panel")
        two_titles(fig, simple, detail)
        for ext, kw in (("png", dict(dpi=150)), ("pdf", {})):
            fig.savefig(self.d / f"{name}.{ext}", **kw)
        plt.close(fig)
        self.rows.append(dict(name=name, simple=simple, detail=detail, compare=compare))

    def index(self):
        pd.DataFrame(self.rows).to_csv(self.d / "index.csv", index=False)
        print(f"[forwardfold deckplots] {len(self.rows)} figures in {self.d}")


DATA_C, PRED_C, THIRD_C, VAR_C, FIT_C = "k", "#1f77b4", "#ff7f0e", "#999999", "#d62728"
MS = 2.6


def _edges_of(x) -> np.ndarray:
    x = np.asarray(x, float)
    h = 0.5 * (x[1] - x[0]) if len(x) > 1 else 0.5
    return np.concatenate([x - h, [x[-1] + h]])


def _hstep(ax, edges, y, err, color, label, ls="-", lw=1.4):
    """Step histogram (a prediction or a reference sample), optional same-colour error bars at the bin centres."""
    edges, y = np.asarray(edges, float), np.asarray(y, float)
    ax.stairs(y, edges, color=color, ls=ls, lw=lw, label=label, baseline=None)
    if err is not None:
        ax.errorbar(0.5 * (edges[1:] + edges[:-1]), y, np.asarray(err, float), fmt="none", ecolor=color, lw=0.8)


def _pts(ax, x, y, e, label, color=DATA_C, marker="o"):
    """Data: small black points with error bars."""
    ax.errorbar(np.asarray(x, float), np.asarray(y, float), np.asarray(e, float), fmt=marker, ms=MS, lw=0.8,
                capsize=0, color=color, label=label, zorder=3)


def _cmp(am, ar, x, a, sa, b, sb, la: str, lb: str, logy: bool = False, lim=None, extra=None, rlabel: str = "ratio",
         resid: str = "ratio", hist: bool = True, rlogy: bool = False):
    """A (data, black points) vs B (prediction, blue step or line); residual A/B (resid="ratio") or (A − B)/σ
    (resid="pull"). Returns the pull array either way."""
    from ..plotting import draw_pull, draw_ratio
    x, a, sa, b, sb = (np.asarray(v, float) for v in (x, a, sa, b, sb))
    if hist:
        e = _edges_of(x)
        _hstep(am, e, b, None, PRED_C, lb)
        for lab, yv, st in (extra or ()):
            _hstep(am, e, yv, None, VAR_C, lab, ls=st)
    else:
        am.plot(x, b, "-", lw=1.4, color=PRED_C, label=lb)
    _pts(am, x, a, sa, la)
    if logy:
        am.set_yscale("log")
    s = np.sqrt(sa ** 2 + sb ** 2)
    pull = np.where(s > 0, (a - b) / np.where(s > 0, s, 1.0), np.nan)
    if resid == "pull":
        draw_pull(ar, x, a, sa, b, sb, lim=lim, label=rlabel, ms=MS, lw=0.6, color=DATA_C)
    else:
        draw_ratio(ar, x, a, sa, b, sb, label=rlabel, logy=rlogy, color=DATA_C)
    return pull


def _cmp_diff(am, ar, x, a, sa, b, sb, la: str, lb: str, rlabel: str, center_zero: bool = True):
    from ..plotting import draw_diff
    am.errorbar(x, b, sb, fmt="s-", ms=3, lw=1.1, capsize=0, color=PRED_C, label=lb)
    am.errorbar(x, a, sa, fmt="o-", ms=3, lw=1.0, capsize=0, color=DATA_C, label=la)
    return draw_diff(ar, x, a, sa, b, sb, label=rlabel, center_zero=center_zero, color=DATA_C, ms=3)


def _grid_axes(rows_cols=(5, 5), size=(3.3, 3.1)):
    from ..plotting import main_resid_axes
    r, c = rows_cols
    return main_resid_axes(r, c, figsize=(size[0] * c, size[1] * r))


def _centres(q):
    return 0.5 * (q.lo.to_numpy(float) + q.hi.to_numpy(float))


def _neut(t, A, therm, tau, B):
    return A * (1 - np.exp(-t / therm)) * np.exp(-t / tau) + B


def _chi2(p: np.ndarray) -> tuple:
    p = p[np.isfinite(p)]
    return float((p ** 2).sum()), int(len(p))


def deckplots(ctx: RunContext) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from ..plotting import set_style
    set_style()
    blk = dict((ctx.extra or {}).get("fold") or {})
    rd = lambda b: pd.read_csv(ctx.csv_path(b))  # noqa: E731
    O = Out(ctx)
    tagline = ("clusters 2–67 µs after the trigger, same cluster finder and settings in data and MC, no cuts on the clusters; "
               "data = AmBe source runs; MC = simulated AmBe neutrons with the hits of one random trigger of run 6256 added")
    fdir = Path(blk["factors_csv"]).parent
    _normalisation(O, rd, fdir, tagline, ctx.csv_path("forwardfold_dm_clmult_chi2"))
    _multiplicity(O, rd, tagline)
    _clusters(O, rd, tagline)
    _capture(O, rd, tagline)
    _efficiency(O, rd, tagline)
    _tail(O, rd, tagline)
    _steps(O, ctx, rd, tagline)
    O.index()


def _normalisation(O: Out, rd, fdir: Path, tagline: str, chi2_csv):
    from ..plotting import draw_pull, main_resid_axes
    ic = rd("forwardfold_dm_icspec")
    for p in PORTS:
        q = ic[ic.port == p]
        labs = q.drop_duplicates("label").sort_values("y")
        n = len(labs)
        fig, M, R = main_resid_axes(1, n, figsize=(3.9 * n, 4.8))
        for j, (_, l) in enumerate(labs.iterrows()):
            z = q[q.label == l.label]
            x = _centres(z)
            _cmp(M[0, j], R[0, j], x, z.source, np.sqrt(z.source), z.bkg_F, z.bkg_F_err, f"source ({l.runs})",
                 "6264 × F (gain-matched)", logy=True, rlabel="source / (6264×F)", rlogy=True)
            M[0, j].set_title(f"y = {l.y:+.0f} cm", fontsize=10)
            R[0, j].set_xlabel("BGO IC (IC_adjusted)")
        M[0, 0].set_ylabel("triggers per 250 IC")
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"icspec_port{p}", f"Port {p}: BGO trigger charge, source runs vs the no-source run scaled",
               "every trigger, no cuts; 6264 scaled by the frozen F_rsb_g = N(IC>3000) ratio with the 6264 IC × g per run "
               f"(nominal); residual = source / (6264 × F)")
    icr = pd.read_csv(fdir / "nosourcenorm_icrun_icrun.csv").sort_values(["y", "port"])
    dmod = pd.read_csv(fdir / "nosourcenorm_icrun_depthmodel.csv").iloc[0]
    fig, M, R = main_resid_axes(1, 1, figsize=(8, 5.6))
    x = icr.y.to_numpy(float) + (icr.port.to_numpy(float) - 3) * 3.0
    model = dmod.a + dmod.b_per_cm * icr.y.to_numpy(float)
    M[0, 0].errorbar(x, icr.closure_g, icr.closure_g_err, fmt="o", ms=3, capsize=0, color=DATA_C, label="source runs")
    yy = np.array([-110.0, 110.0])
    M[0, 0].plot(yy, dmod.a + dmod.b_per_cm * yy, "-", color=PRED_C,
                 label=f"linear depth model, χ²/ndf {dmod.chi2:.0f}/{int(dmod.ndf)}")
    M[0, 0].set_ylabel("N(IC>3000) / (α_live · N_6264(IC·g>3000))")
    M[0, 0].legend(frameon=False, fontsize=8)
    draw_pull(R[0, 0], x, icr.closure_g, icr.closure_g_err, model, np.zeros(len(model)), label="(run − model)/σ",
              ms=MS, lw=0.6, color=DATA_C)
    R[0, 0].set_xlabel("source height y [cm] (points offset by port)")
    O.save(fig, "icclosure_y", "False-start rate per live second depends on source depth",
           "IC>3000 closure of α_live per run (6264 gain-matched); 6264 sits at Port 5 y = 0; this is why the nominal "
           "false-start factor is the measured count ratio F_rsb_g, not α_live")
    mr = pd.read_csv(fdir / "nosourcenorm_muonrate_muonrate.csv")
    src = mr[mr.kind == "source"].sort_values(["date", "run"]).reset_index(drop=True)
    ref = mr[mr.run == 6256].iloc[0]
    fig, M, R = main_resid_axes(1, 1, figsize=(12, 5.2))
    x = np.arange(len(src))
    k = 1000.0
    _cmp(M[0, 0], R[0, 0], x, src.rate_1000 * k, src.rate_1000_err * k, np.full(len(x), ref.rate_1000 * k),
         np.full(len(x), ref.rate_1000_err * k), "source runs", "6256 dummy run", hist=False, rlabel="source / 6256")
    M[0, 0].set_ylabel("clusters > 1000 p.e. per ms of window")
    M[0, 0].legend(frameon=False, fontsize=8)
    R[0, 0].set_xticks(x, [f"{int(r)}\n{d[5:]}" for r, d in zip(src.run, src.date)], fontsize=7)
    O.save(fig, "muonrate_runs", "Tank muon rate per window: every source run vs the dummy run",
           "tree clusters > 1000 p.e. in 10–66 µs per trigger, all triggers; one 6256 accidental template serves every run "
           "if these agree")
    cm = rd("forwardfold_dm_clmult")
    chi = []
    labels = dict(SELECTIONS)
    for level, keys in (("port", PORTS), ("run", sorted(cm.run.unique()))):
        for key in keys:
            q = cm[cm.port == key] if level == "port" else cm[cm.run == key]
            fig, M, R = main_resid_axes(2, 2, figsize=(11, 9))
            for ax_i, (sel, slab) in enumerate(SELECTIONS):
                z = q[q.selection == sel].assign(Fn=lambda d: d.F * d.n_6264_g, an=lambda d: d.alpha * d.n_6264_raw,
                                                  e6264=lambda d: d.F * np.sqrt(d.n_6264_g),
                                                  eF2=lambda d: (d.n_6264_g * d.F_err) ** 2)
                g = z.groupby("k")[["n_source", "Fn", "an", "e6264", "eF2"]].sum()
                S = g.n_source.to_numpy(float)
                Fn = g.Fn.to_numpy(float)
                Fe = np.sqrt(g.e6264.to_numpy(float) ** 2 + g.eF2.to_numpy(float))   # 6264 counts shared by all runs
                An = g.an.to_numpy(float)
                kk = np.arange(KMAX + 1)
                am, ar = M.flat[ax_i], R.flat[ax_i]
                pull = _cmp(am, ar, kk, S, np.sqrt(S), Fn, Fe, "source", "6264 × F (nominal)", logy=True,
                            rlabel="source / (6264×F)", rlogy=sel != "ic3000_noveto")
                ar.set_xticks(kk, [str(i) for i in kk[:-1]] + [f"{KMAX}+"])
                ar.set_xlabel("clusters per trigger (2–67 µs)")
                am.set_title(slab, fontsize=10)
                am.set_ylabel("triggers")
                c2, nb = _chi2(pull)
                chi.append(dict(level=level, key=key, selection=sel, n_source=S.sum(), n_bkg_F=Fn.sum(),
                                n_bkg_alpha=An.sum(), mean_source=float((kk * S).sum() / max(S.sum(), 1)),
                                mean_bkg=float((kk * Fn).sum() / max(Fn.sum(), 1e-12)), chi2=c2, nbins=nb))
            M[0, 0].legend(frameon=False, fontsize=8)
            what = f"Port {key}, runs pooled" if level == "port" else f"Run {key}"
            O.save(fig, f"clmult_{level}{key}", f"{what}: clusters per trigger, source vs the scaled no-source run",
                   f"{tagline}; 6264 × frozen F_rsb_g per run (IC>3000 panels: 6264 IC × g); residual = source / "
                   "(6264 × F)")
    pd.DataFrame(chi).to_csv(chi2_csv, index=False)


def _multiplicity(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    b = rd("forwardfold_bins")
    b = b[b.group != "background"]
    xt = ["0", "1", "2", "3", "≥4"]
    q = b.groupby(["group", "n"])[["data", "pred"]].sum().reset_index()
    fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
    for j, (g, z) in enumerate(q.groupby("group")):
        _cmp(M[0, j], R[0, j], z.n, z.data, np.sqrt(z.data), z.pred, np.zeros(len(z)), "data", "forward-fold fit",
             logy=True, rlabel="data / fit")
        M[0, j].set_title(g.replace("port", "Port "), fontsize=10)
        R[0, j].set_xticks(range(5), xt)
        R[0, j].set_xlabel("clusters per trigger")
    M[0, 0].set_ylabel("veto-passing IC-gated triggers")
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "mult_port", "Clusters per trigger per port: data vs the forward-fold fit",
           f"{tagline}; fit = N_A·[MC(s)⊛Poisson(μ_ind)] + F·6264, bins 0–3, ≥4; residual = (data − fit)/√data")
    mp = rd("forwardfold_dm_multpos")
    fig, M, R = _grid_axes()
    for i, p in enumerate(PORTS):
        for j, y in enumerate(YS):
            z = mp[(mp.port == p) & (np.isclose(mp.y, y))].sort_values("n")
            if z.empty:
                M[i, j].set_visible(False); R[i, j].set_visible(False)
                continue
            _cmp(M[i, j], R[i, j], z.n, z.data, np.sqrt(z.data), z.pred, np.zeros(len(z)), "data", "fit", logy=True, rlabel="data / fit")
            M[i, j].set_title(f"P{p} y={y:+.0f} ({z.runs.iloc[0]})", fontsize=8)
            R[i, j].set_xticks(range(5), xt)
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "mult_grid", "Clusters per trigger at every source position: data vs fit",
           f"{tagline}; per-port fit summed over the runs at each MC position (P3 y=−50 holds 6244 and 6187 at y=−60); "
           "residual = data / fit")
    for g, z in b.groupby("group"):
        runs = list(dict.fromkeys(z.run))
        fig, M, R = main_resid_axes(1, len(runs), figsize=(3.6 * len(runs), 4.6))
        for j, r in enumerate(runs):
            w = z[z.run == r]
            _cmp(M[0, j], R[0, j], w.n, w.data, np.sqrt(w.data), w.pred, np.zeros(len(w)), "data", "fit", logy=True, rlabel="data / fit")
            M[0, j].set_title(f"{r}: dev {w.run_deviance.iloc[0]:.1f}", fontsize=9)
            R[0, j].set_xticks(range(5), xt)
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"mult_runs_{g}", f"{g.replace('port', 'Port ')}: clusters per trigger per run vs the fit",
               f"{tagline}; run deviance in the panel title; residual = (data − fit)/√data")


def _clusters(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    cl = rd("forwardfold_dm_clusters")
    tmask = lambda z: z[(z["var"] != "time") | ((z.lo >= 2) & (z.hi <= 67))]  # noqa: E731
    for var in ("time", "pe", "hits"):
        z0 = tmask(cl[(cl["var"] == var)])
        fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
        for j, p in enumerate(PORTS):
            z = z0[(z0.level == "port") & (z0.port == p)]
            _cmp(M[0, j], R[0, j], _centres(z), z.data_pt, z.data_pt_err, z.mc_pt, z.mc_pt_err, "data − F·6264",
                 f"MC, capture clusters × s = {z.s.iloc[0]:.3f}", logy=var == "time", rlabel="data / MC")
            M[0, j].set_title(f"Port {p}", fontsize=10)
            R[0, j].set_xlabel(XLAB[var])
        M[0, 0].set_ylabel("clusters per AmBe trigger per bin")
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"cl_{var}_port", f"Data vs MC per port: {XLAB[var]}",
               f"{tagline}; per veto-passing gated AmBe trigger (N_A = N − F·N_6264); MC weighted by each position's share "
               f"of AmBe triggers; the MC has no μ_ind; {RESID_NOTE}")
        fig, M, R = _grid_axes()
        for i, p in enumerate(PORTS):
            for j, y in enumerate(YS):
                z = z0[(z0.level == "position") & (z0.port == p) & np.isclose(z0.y, y)]
                if z.empty:
                    M[i, j].set_visible(False); R[i, j].set_visible(False)
                    continue
                _cmp(M[i, j], R[i, j], _centres(z), z.data_pt, z.data_pt_err, z.mc_pt, z.mc_pt_err, "data − F·6264",
                     "MC (s)", logy=var == "time", rlabel="data / MC")
                M[i, j].set_title(f"P{p} y={y:+.0f} ({z.runs.iloc[0]})", fontsize=8)
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"cl_{var}_grid", f"Data vs MC at every source position: {XLAB[var]}",
               f"{tagline}; per AmBe trigger; MC capture clusters × s_port; {RESID_NOTE}")
        if var in ("time", "pe"):
            fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
            for j, p in enumerate(PORTS):
                z = z0[(z0.level == "port") & (z0.port == p)]
                _cmp(M[0, j], R[0, j], _centres(z), z.dummy_pt, z.dummy_pt_err, z.mc_non_pt, z.mc_non_pt_err,
                     "6256 dummy triggers", "MC non-capture clusters", logy=var == "time", rlabel="6256 / MC")
                M[0, j].set_title(f"Port {p}", fontsize=10)
                R[0, j].set_xlabel(XLAB[var])
            M[0, 0].set_ylabel("clusters per trigger per bin")
            M[0, 0].legend(frameon=False, fontsize=7)
            O.save(fig, f"accid_{var}_port", f"Accidental clusters: dummy-run data vs the MC overlay, {XLAB[var]}",
                   "6256 = dummy (random) triggers, veto-passing; MC non-capture = clusters not owned by a true capture "
                   f"(mostly the overlaid 6256 hits) per veto-passing MC event; {RESID_NOTE}")


def _capture(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    cl = rd("forwardfold_dm_clusters")
    ft = rd("forwardfold_dm_capturetime")
    t = cl[(cl["var"] == "time") & (cl.lo >= 2) & (cl.hi <= FIT_MAX_US)]
    for sample, col, lab in (("data", "src", "data (source runs)"), ("mc_all", None, "MC, all clusters")):
        for level in ("port", "position"):
            fig, M, R = (main_resid_axes(1, 5, figsize=(18, 4.8)) if level == "port" else _grid_axes())
            cells = [(0, j, p, None) for j, p in enumerate(PORTS)] if level == "port" else \
                [(i, j, p, y) for i, p in enumerate(PORTS) for j, y in enumerate(YS)]
            for i, j, p, y in cells:
                z = t[(t.level == level) & (t.port == p)]
                f = ft[(ft.level == level) & (ft.port == p) & (ft["sample"] == sample)]
                if y is not None:
                    z, f = z[np.isclose(z.y, y)], f[np.isclose(f.y, y)]
                if z.empty or f.empty or not bool(f.fit_ok.iloc[0]):
                    M[i, j].set_visible(False); R[i, j].set_visible(False)
                    continue
                f = f.iloc[0]
                x = _centres(z)
                h = z.src.to_numpy(float) if col else (z.mc_cap_raw + z.mc_non_raw).to_numpy(float)
                mdl = _neut(x, f.A, f.therm, f.tau, f.B)
                _cmp(M[i, j], R[i, j], x, h, np.sqrt(h), mdl, np.zeros(len(x)), lab,
                     f"fit τ = {f.tau:.2f} ± {f.tau_err:.2f} µs, therm {f.therm:.2f}", logy=True,
                     resid="pull", rlabel="(data − fit)/σ")
                M[i, j].set_title(f"Port {p}" if y is None else f"P{p} y={y:+.0f}", fontsize=10 if y is None else 8)
                M[i, j].legend(frameon=False, fontsize=6 if y is not None else 7)
                if y is None:
                    R[i, j].set_xlabel("cluster time [µs]")
            M[0, 0].set_ylabel("clusters per µs")
            nm = f"ct_{'data' if sample == 'data' else 'mc'}_{'port' if level == 'port' else 'grid'}"
            where = "per port" if level == "port" else "at every source position"
            O.save(fig, nm, f"Neutron capture time {where}: {lab}",
                   f"{FITTER}; {tagline}; residual = (counts − fit)/√counts")
    pos = ft[ft.level == "position"]
    for par, ylab in (("tau", "capture time τ [µs]"), ("therm", "thermalisation time [µs]")):
        fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
        for j, p in enumerate(PORTS):
            d = pos[(pos.port == p) & (pos["sample"] == "data")].sort_values("y")
            m = pos[(pos.port == p) & (pos["sample"] == "mc_all")].set_index("y").reindex(d.y).reset_index()
            _cmp_diff(M[0, j], R[0, j], d.y, d[par], d[f"{par}_err"], m[par], m[f"{par}_err"], "data", "MC",
                      "data − MC [µs]")
            M[0, j].set_title(f"Port {p}", fontsize=10)
            R[0, j].set_xlabel("source height y [cm]")
        M[0, 0].set_ylabel(ylab)
        M[0, 0].legend(frameon=False, fontsize=8)
        O.save(fig, f"{par}_vs_y", f"{'Capture' if par == 'tau' else 'Thermalisation'} time per position: data vs MC",
               f"{FITTER}; data = source runs at the position, MC = all setup-A clusters at that position")
    fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
    for j, p in enumerate(PORTS):
        d = pos[(pos.port == p) & (pos["sample"] == "data")].sort_values("y")
        m = pos[(pos.port == p) & (pos["sample"] == "data_B6256")].set_index("y").reindex(d.y).reset_index()
        _cmp_diff(M[0, j], R[0, j], d.y, d.tau, d.tau_err, m.tau, m.tau_err, "B free", "B = 6256 flat level",
                  "free − fixed [µs]")
        M[0, j].set_title(f"Port {p}", fontsize=10)
        R[0, j].set_xlabel("source height y [cm]")
    M[0, 0].set_ylabel("data capture time τ [µs]")
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "tau_bfix_vs_y", "Data capture time: flat term free vs fixed to the dummy-run level",
           f"{FITTER}; fixed variant: B = 6256 clusters per trigger per µs × AmBe triggers (accidentals only, no μ_ind); "
           "the two are not independent (same clusters), so the residual error overstates the spread")
    ts = rd("forwardfold_timespectrum")
    tb = rd("forwardfold_timespectrum_bins")
    fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
    for j, (g, q) in enumerate(tb.groupby("group")):
        r = ts[ts.group == g].iloc[0]
        x = 0.5 * (q.t_lo_us + q.t_hi_us).to_numpy(float)
        NA = float(q.N_A.iloc[0])
        y = (q.source - q.background_scaled).to_numpy(float) / NA
        e = np.sqrt(q.source.to_numpy(float)) / NA
        mdl = _neut(x, r.A_fit, r.therm_us, r.tau_us, r.C_fit) / NA
        _cmp(M[0, j], R[0, j], x, y, e, mdl, np.zeros(len(x)), "source − F·6264", "capture + flat fit", logy=True,
             resid="pull", rlabel="(data − fit)/σ")
        M[0, j].set_title(f"{g.replace('port', 'Port ')}: μ_ind(time) {r.mu_ind_time:+.3f} ± {r.mu_ind_time_err:.3f}",
                          fontsize=9)
        R[0, j].set_xlabel("cluster time [µs]")
    M[0, 0].set_ylabel("clusters per AmBe trigger per µs")
    M[0, 0].legend(frameon=False, fontsize=7)
    fitter = str(ts.fitter.iloc[0]) if "fitter" in ts else "scipy.optimize.least_squares"
    O.save(fig, "timecheck_port", "Source-induced clusters from the time spectrum (cross-check of μ_ind)",
           f"{fitter}: A(1−e^(−t/therm))e^(−t/τ) + C, 2–66 µs, 1 µs bins; μ_ind(time) = (C/N_A − 6256 flat) × 64.5 µs; "
           "residual = (data − fit)/σ, σ = √source/N_A")


def _efficiency(O: Out, rd, tagline: str):
    from ..plotting import draw_pull, main_resid_axes
    ef = rd("forwardfold_efficiency")
    e = ef[ef.variant == "nominal"]
    fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
    for j, (g, q) in enumerate(e.groupby("group")):
        q = q.sort_values("y")
        mc_err = np.sqrt(q.eps_mc * (1 - q.eps_mc) / q.mc_events)
        _cmp_diff(M[0, j], R[0, j], q.y, q.eps_data, q.eps_data_err, q.eps_mc, mc_err, "data (MC thinned by s)",
                  "MC truth", "data − MC", center_zero=False)
        M[0, j].set_title(f"{g.replace('port', 'Port ')}: s = {q.s.iloc[0]:.4f} ± {q.s_err.iloc[0]:.4f}", fontsize=9)
        R[0, j].set_xlabel("source height y [cm]")
    M[0, 0].set_ylabel("neutron detection efficiency")
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "eff_port", "Neutron detection efficiency per source position: data vs MC",
           "ε = fraction of veto-passing AmBe triggers with ≥ 1 capture cluster in 2–67 µs; ε_data = ε_MC(s) with each "
           "true capture's cluster kept with probability s_port; data errors stat only")
    pm = rd("forwardfold_params")
    bp = pm[pm.variant == "by_position"].copy()
    nom = pm[pm.variant == "nominal"].set_index("group")
    if len(bp):
        bp["port"] = bp.group.str.extract(r"port(\d)").astype(int)
        bp["y"] = bp.group.str.extract(r"_y([+-]?\d+)").astype(float)
        fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
        for j, p in enumerate(PORTS):
            q = bp[bp.port == p].sort_values("y")
            sp, se = float(nom.s[f"port{p}"]), float(nom.s_err[f"port{p}"])
            _cmp_diff(M[0, j], R[0, j], q.y, q.s, q.s_err, np.full(len(q), sp), np.zeros(len(q)), "fit per position",
                      "fit per port", "position − port")
            M[0, j].set_title(f"Port {p}", fontsize=10)
            R[0, j].set_xlabel("source height y [cm]")
        M[0, 0].set_ylabel("efficiency scale s (data/MC)")
        M[0, 0].legend(frameon=False, fontsize=8)
        O.save(fig, "s_bypos", "Data/MC efficiency scale fitted per position vs per port",
               "variant by_position of the forward fold (one s and μ_ind per position); residual error = position fit only")
    toys = rd("forwardfold_closure_toys")
    toys = toys[toys.fit_ok]
    pts = toys.drop_duplicates(["s_true", "mu_true"])[["s_true", "mu_true"]].sort_values(["s_true", "mu_true"],
                                                                                           ascending=[False, True])
    from math import erf, sqrt
    edges = np.arange(-4.0, 4.01, 0.5)
    cdf = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in edges])
    fig, M, R = main_resid_axes(2, len(pts), figsize=(3.3 * len(pts), 8.4))
    for j, (_, pt) in enumerate(pts.iterrows()):
        q = toys[(toys.s_true == pt.s_true) & (toys.mu_true == pt.mu_true)]
        for i, (col, lab) in enumerate((("s_pull", "s"), ("mu_pull", "μ_ind"))):
            h = np.histogram(np.clip(q[col], -3.999, 3.999), edges)[0].astype(float)
            ex = len(q) * np.diff(cdf)
            x = 0.5 * (edges[1:] + edges[:-1])
            _cmp(M[i, j], R[i, j], x, h, np.sqrt(h), ex, np.zeros(len(x)), "toys (5 ports pooled)", "unit Gaussian",
                 rlabel="toys / Gaussian")
            M[i, j].set_title(f"{lab}: s={pt.s_true:g}, μ={pt.mu_true:g}", fontsize=9)
            R[i, j].set_xlabel(f"pull of {lab}")
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "closure_pulls", "MC closure: fit pulls in 1,000 pseudo-experiments per point",
           "closure v3 (template term shared per template, full covariance); template files 0–24, pseudo-data drawn "
           "from the full MC per toy; residual = toys / Gaussian")


def _tail(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    tr = rd("forwardfold_dm_tail_runs").sort_values(["date", "run"]).reset_index(drop=True)
    fig, M, R = main_resid_axes(2, 1, figsize=(13, 8.6))
    x = np.arange(len(tr))
    for i, (cols, lab) in enumerate(((["n2"], "exactly 2 clusters"), (["n3", "n4"], "3 or more clusters"))):
        d = sum(tr[f"data_{c}"] for c in cols) / tr.N_trig
        p = sum(tr[f"pred_{c}"] for c in cols) / tr.N_trig
        _cmp(M[i, 0], R[i, 0], x, d, np.sqrt(d * tr.N_trig) / tr.N_trig, p, np.zeros(len(x)), "data", "forward-fold fit",
             hist=False, rlabel="data / fit")
        M[i, 0].set_ylabel(f"fraction with {lab}")
        R[i, 0].set_xticks(x, [f"{int(r)}\n{dt[5:]}" for r, dt in zip(tr.run, tr.date)], fontsize=7)
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "tail_runs", "Multi-cluster triggers per run, in date order: data vs fit",
           f"{tagline}; per-port fit; residual = (data − fit)/√data")
    sh = rd("forwardfold_dm_tail_shapes")
    for var in ("time", "pe", "hits", "dt"):
        z0 = sh[sh["var"] == var]
        if var == "time":
            z0 = z0[(z0.lo >= 2) & (z0.hi <= 67)]
        combos = [("data_cls", "data", "n3p", "data", "n12", "3+ clusters", "1–2 clusters")] if var != "dt" else \
            [("datamc_n2", "data", "n2", "mc", "n2", "data, exactly 2 clusters", "MC, exactly 2 clusters")]
        combos.append(("datamc", "data", "n3p", "mc", "n3p", "data, 3+ clusters", "MC, 3+ clusters"))
        for tag, sa, ca, sb, cb, la, lb in combos:
            fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
            for j, p in enumerate(PORTS):
                g = f"port{p}"
                A = z0[(z0.group == g) & (z0["sample"] == sa) & (z0.cls == ca)]
                B = z0[(z0.group == g) & (z0["sample"] == sb) & (z0.cls == cb)]
                ha, hb = A["count"].to_numpy(float), B["count"].to_numpy(float)
                na, nb = max(ha.sum(), 1), max(hb.sum(), 1)
                _cmp(M[0, j], R[0, j], _centres(A), ha / na, np.sqrt(ha) / na, hb / nb, np.sqrt(hb) / nb,
                     f"{la} ({int(A.n_trig.iloc[0])} trig.)", f"{lb} ({int(B.n_trig.iloc[0])} trig.)",
                     logy=var in ("time", "dt"), rlabel={"data_cls": "3+ / 1–2"}.get(tag, "data / MC"))
                M[0, j].set_title(f"Port {p}", fontsize=10)
                R[0, j].set_xlabel(XLAB[var])
            M[0, 0].set_ylabel("fraction of clusters per bin" if var != "dt" else "fraction of intervals per bin")
            M[0, 0].legend(frameon=False, fontsize=7)
            what = {"data_cls": "data triggers with 3+ clusters vs 1–2 clusters",
                    "datamc": "triggers with 3+ clusters, data vs MC",
                    "datamc_n2": "triggers with exactly 2 clusters, data vs MC (is one capture split in two?)"}[tag]
            O.save(fig, f"tail_{tag}_{var}", f"What the multi-cluster triggers look like: {XLAB[var]}",
                   f"{what}; unit area; data = veto-passing gated source triggers, MC = veto-passing setup-A events "
                   f"(not thinned); {RESID_NOTE}")
    pr = rd("forwardfold_dm_tail_pairs")
    for pair, z0 in pr.groupby("pair"):
        a, b = (int(v) for v in pair.split("_"))
        fig, M, R = main_resid_axes(1, 4, figsize=(16, 4.8))
        for j, var in enumerate(("mult", "time", "pe", "hits")):
            z = z0[z0["var"] == var]
            if var == "time":
                z = z[(z.lo >= 2) & (z.hi <= 67)]
            A, B = z[z.run == a], z[z.run == b]
            Na, Nb = float(A.N.iloc[0]), float(B.N.iloc[0])
            ha, hb = A["count"].to_numpy(float), B["count"].to_numpy(float)
            _cmp(M[0, j], R[0, j], _centres(A), ha / Na, np.sqrt(ha) / Na, hb / Nb, np.sqrt(hb) / Nb,
                 f"{a} ({int(Na)} trig.)", f"{b} ({int(Nb)} trig.)", logy=var in ("mult", "time"),
                 rlabel=f"{a} / {b}")
            R[0, j].set_xlabel(XLAB[var] if var != "mult" else "clusters per trigger (8 = 8+)")
        M[0, 0].set_ylabel("per veto-passing gated trigger")
        M[0, 0].legend(frameon=False, fontsize=8)
        O.save(fig, f"pair_{pair}", f"Same position, different period: run {a} vs run {b}",
               f"{tagline}; both runs at one source position; per trigger, no MC; {RESID_NOTE}")


# ---------------------------------------------------------------------------------------------- focus position
def focus(ctx: RunContext, su: ff.Setup, spec: str) -> None:
    """--stage focus: one source position in detail (spec "port,y", default Port 5, y = 0), written to
    forwardfold_focus_*.csv: the model build-up per run for every run (MC as simulated, captures thinned by s,
    Poisson(mu_ind) added, the position fit), the MC composition per label, per-run IC spectra and capture fits."""
    port, y0 = (float(v) for v in spec.split(","))
    prm = pd.read_csv(ctx.csv_path("forwardfold_params"))
    nom = prm[prm.variant == "nominal"].set_index("group")
    bpos = prm[prm.variant == "by_position"].set_index("group")
    b = pd.read_csv(ctx.csv_path("forwardfold_bins"))
    b = b[b.group != "background"]
    fac = su.fac.assign(group=ff.group_of(su.fac, "port"), pgroup=ff.group_of(su.fac, "position"),
                        label=su.fac.run.map(su.labels))
    tm = {lab: ff.template(su.sig, lab, su.K) for lab in sorted(set(fac.label))}
    rows = []
    for r in fac.itertuples():
        q = b[b.run == r.run].set_index("n")
        NA = float(q.N_A.iloc[0])
        s, mu = float(nom.s[r.group]), float(nom.mu_ind[r.group])
        curves = {"mc_s1": (1.0, 0.0), "mc_s": (s, 0.0), "mc_s_mu": (s, mu)}
        if r.pgroup in bpos.index:
            curves["mc_pos"] = (float(bpos.s[r.pgroup]), float(bpos.mu_ind[r.pgroup]))
        pmf = {k: tm[r.label].pmf(*v)[0] for k, v in curves.items()}
        for k in q.index:
            rows.append(dict(run=int(r.run), group=r.group, pgroup=r.pgroup, label=r.label, port=int(r.port),
                             y=float(r.y), y_mc=float(r.y + su.label_dy[r.run]), n=int(k), data=float(q.data[k]),
                             pred=float(q.pred[k]), pred_ambe=float(q.pred_ambe[k]),
                             pred_falsestart=float(q.pred_falsestart[k]), N_trig=float(q.N_trig.iloc[0]), N_A=NA,
                             s=s, mu_ind=mu, s_pos=curves.get("mc_pos", (np.nan,))[0],
                             mu_pos=curves.get("mc_pos", (np.nan, np.nan))[1], eps_mc=tm[r.label].eff(1.0),
                             eps_s=tm[r.label].eff(s), **{c: NA * pmf[c][k] for c in pmf}))
    pd.DataFrame(rows).to_csv(ctx.csv_path("forwardfold_focus_templates"), index=False)
    sig = su.sig.assign(n_capcl=su.sig.m.map(lambda m: int(sum(m))), n_cap=su.sig.m.map(len))
    sig.groupby(["label", "sourceY", "n_cap", "n_capcl", "ko"]).w.sum().reset_index().to_csv(
        ctx.csv_path("forwardfold_focus_mccomp"), index=False)
    frun = fac[(fac.port == port) & np.isclose(fac.y, y0)]
    if frun.empty:
        raise SystemExit(f"[forwardfold focus] no source run at port {port:g}, y {y0:g}")
    D = Data(su)
    dm_t, dm_c = D.run(int(su.blk["dummy_run"]))
    dm_c = dm_c[~dm_c.is_cosmic]
    Nd = int((~dm_t.is_cosmic).sum())
    flat_d = float(((dm_c.clusterTime >= 2000) & (dm_c.clusterTime < 67000)).sum()) / Nd / 65.0
    icb = D.features(su.bkg_run).IC_adjusted.to_numpy(float)
    hraw = np.histogram(icb, IC_EDGES)[0].astype(float)
    fits, ics = [], []
    for r in frun.itertuples():
        t, cl = D.gv(int(r.run))
        for sample, bfix in (("data", None), ("data_B6256", flat_d * len(t))):
            f = _lmfit(cl.clusterTime.to_numpy(float) / 1000.0, bfix)
            row = dict(run=int(r.run), sample=sample, n_trig=len(t), n_clusters=len(cl), B_fixed=bfix,
                       fitter=FITTER if bfix is None else FITTER.replace("B free", "B fixed to the 6256 flat level"),
                       fit_ok=f is not None)
            if f is not None:
                row.update({k: f[k] for k in ("A", "A_err", "therm", "therm_err", "tau", "tau_err", "B", "B_err", "chi2",
                                              "ndof", "redchi", "N")})
            fits.append(row)
        hs = np.histogram(D.features(int(r.run)).IC_adjusted.to_numpy(float), IC_EDGES)[0].astype(float)
        hb = np.histogram(icb * float(r.g), IC_EDGES)[0].astype(float)
        ics += [dict(run=int(r.run), lo=IC_EDGES[i], hi=IC_EDGES[i + 1], source=hs[i], bkg_F=r.F_rsb_g * hb[i],
                     bkg_F_err=r.F_rsb_g * np.sqrt(hb[i]), bkg_alpha=r.alpha * hraw[i],
                     bkg_alpha_err=r.alpha * np.sqrt(hraw[i]), F=r.F_rsb_g, alpha=r.alpha, g=r.g)
                for i in range(len(IC_EDGES) - 1)]
    pd.DataFrame(fits).to_csv(ctx.csv_path("forwardfold_focus_capturetime"), index=False)
    pd.DataFrame(ics).to_csv(ctx.csv_path("forwardfold_focus_icspec"), index=False)
    pd.DataFrame([dict(port=int(port), y=y0, label=frun.label.iloc[0], group=frun.group.iloc[0],
                       pgroup=frun.pgroup.iloc[0], runs=" ".join(str(int(v)) for v in frun.run),
                       flat_6256_per_trigger_us=flat_d, n_dummy=Nd)]).to_csv(ctx.csv_path("forwardfold_focus_meta"),
                                                                              index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(fits)[["run", "sample", "n_clusters", "tau", "tau_err", "therm", "redchi"]].to_string(index=False))
    print(f"[forwardfold focus] port {port:g}, y {y0:g}: wrote forwardfold_focus_*.csv")


# ---------------------------------------------------------------------------------------------- figures, step style
MEDGES = np.arange(-0.5, 5.5, 1.0)
XT5 = ["0", "1", "2", "3", "≥4"]
METHOD_STYLE = {"poisson": ("C3", "o", "--", "Poisson fit"), "dd": ("C2", "^", "-.", "data-driven fit (6256 template)"),
                "dds": ("C4", "v", ":", "data-driven, scaled template")}


def _edges(q) -> np.ndarray:
    return np.append(q.lo.to_numpy(float), q.hi.to_numpy(float)[-1])


def _opt(ctx: RunContext, name: str):
    p = Path(ctx.csv_path(name))
    return pd.read_csv(p) if p.exists() else None


def _ct_panel(am, ar, z_edges, h, f, lab):
    """Data as black points, the lmfit curve (red) over the fit range, linear y."""
    from ..plotting import draw_pull
    x = 0.5 * (z_edges[1:] + z_edges[:-1])
    _pts(am, x, h, np.sqrt(h), lab)
    tt = np.linspace(2.0, FIT_MAX_US, 400)
    am.plot(tt, _neut(tt, f.A, f.therm, f.tau, f.B), color=FIT_C, lw=1.6,
            label=f"τ = {f.tau:.2f} ± {f.tau_err:.2f} µs\ntherm = {f.therm:.2f} ± {f.therm_err:.2f} µs\n"
                  f"χ²/ndf = {f.chi2:.0f}/{int(f.ndof)}")
    m = (x > 2.0) & (x < FIT_MAX_US) & (h > 0)
    draw_pull(ar, x[m], h[m], np.sqrt(h[m]), _neut(x[m], f.A, f.therm, f.tau, f.B), np.zeros(int(m.sum())),
              label="(data − fit)/σ", ms=MS, lw=0.6, color=DATA_C)
    am.set_ylim(bottom=0)


def _steps(O: Out, ctx: RunContext, rd, tagline: str):
    """Deck figures in one style: data = small black points; what it is compared with (6264 × F, MC, fit) = blue
    steps; model variants grey or orange by line style; fit curves red; F only (no α_live line); ratio residuals for
    Data/MC, pulls only for fit residuals."""
    _s_clmult(O, rd, tagline)
    _s_mult(O, ctx, rd, tagline)
    _s_clusters(O, rd, tagline)
    _s_capture(O, rd, tagline)
    _s_maps(O, rd)
    _s_focus(O, ctx, rd, tagline)
    _s_methods(O, ctx, rd)


def _s_clmult(O: Out, rd, tagline: str):
    from ..plotting import draw_ratio, main_resid_axes
    cm = rd("forwardfold_dm_clmult")
    edges = np.arange(-0.5, KMAX + 1.5, 1.0)
    kk = np.arange(KMAX + 1)
    for level, keys in (("port", PORTS), ("run", sorted(cm.run.unique()))):
        for key in keys:
            q = cm[cm.port == key] if level == "port" else cm[cm.run == key]
            fig, M, R = main_resid_axes(2, 2, figsize=(11, 9))
            for i, (sel, slab) in enumerate(SELECTIONS):
                z = q[q.selection == sel].assign(Fn=lambda d: d.F * d.n_6264_g, e6264=lambda d: d.F * np.sqrt(d.n_6264_g),
                                                  eF2=lambda d: (d.n_6264_g * d.F_err) ** 2)
                g = z.groupby("k")[["n_source", "Fn", "e6264", "eF2"]].sum()
                S, Fn = g.n_source.to_numpy(float), g.Fn.to_numpy(float)
                Fe = np.sqrt(g.e6264.to_numpy(float) ** 2 + g.eF2.to_numpy(float))
                am, ar = M.flat[i], R.flat[i]
                _hstep(am, edges, Fn, Fe, PRED_C, "6264 × F")
                _pts(am, kk, S, np.sqrt(S), "source run(s)")
                am.set_yscale("log")
                draw_ratio(ar, kk, S, np.sqrt(S), Fn, Fe, label="source / (6264×F)", logy=sel != "ic3000_noveto",
                           color=DATA_C)
                ar.set_xticks(kk, [str(i) for i in kk[:-1]] + [f"{KMAX}+"])
                ar.set_xlabel("clusters per trigger (2–67 µs)")
                ms_, mb_ = (kk * S).sum() / max(S.sum(), 1), (kk * Fn).sum() / max(Fn.sum(), 1e-12)
                am.set_title(f"{slab}: ⟨n⟩ source {ms_:.2f}, 6264×F {mb_:.2f}", fontsize=10)
                am.set_ylabel("triggers")
            M[0, 0].legend(frameon=False, fontsize=8)
            what = f"Port {key}, runs pooled" if level == "port" else f"Run {key}"
            O.save(fig, f"clmultS_{level}{key}", f"{what}: clusters per trigger, source vs the scaled no-source run",
                   f"{tagline}; black = source run, blue = no-source run 6264 × F (F = ratio of triggers above IC 3000, "
                   "where the source adds nothing); lower panel = source / (6264 × F)")


def _mult_panel(am, ar, z, title, mc_s1=None, extra=()):
    from ..plotting import draw_ratio
    d = z.groupby("n")[["data", "pred"]].sum()
    N = float(d.data.sum())
    x = d.index.to_numpy(float)
    _hstep(am, MEDGES, d.pred / N, None, PRED_C, "forward-fold fit")
    if mc_s1 is not None:
        _hstep(am, MEDGES, mc_s1.reindex(d.index).to_numpy(float) / N, None, VAR_C,
               "MC as simulated (s = 1, μ_ind = 0)", ls="--", lw=1.2)
    for lab, v, col, ls in extra:
        _hstep(am, MEDGES, v.reindex(d.index).to_numpy(float) / N, None, col, lab, ls=ls, lw=1.2)
    _pts(am, x, d.data / N, np.sqrt(d.data) / N, "data")
    am.set_yscale("log")
    draw_ratio(ar, x, d.data, np.sqrt(d.data), d.pred, None, label="data / fit", color=DATA_C)
    ar.set_xticks(range(5), XT5)
    ar.set_xlabel("clusters per trigger (2–67 µs)")
    am.set_title(title, fontsize=9)


def _s_mult(O: Out, ctx: RunContext, rd, tagline: str):
    from ..plotting import main_resid_axes
    ft = _opt(ctx, "forwardfold_focus_templates")
    if ft is None:
        print("[forwardfold deckplots] no forwardfold_focus_templates.csv (run --stage focus): step multiplicity "
              "figures skipped")
        return
    note = (f"{tagline}; fraction of selected triggers with n clusters; black = data, blue = forward-fold fit (MC with "
            "capture clusters kept with probability s, plus extra source clusters μ_ind, plus false starts), grey dashed "
            "= MC as simulated (every capture cluster kept, nothing added); lower panel = data / fit")
    fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
    for j, p in enumerate(PORTS):
        z = ft[ft.port == p]
        _mult_panel(M[0, j], R[0, j], z, f"Port {p}: s = {z.s.iloc[0]:.4f}, μ_ind = {z.mu_ind.iloc[0]:.4f}",
                    z.groupby("n").mc_s1.sum())
    M[0, 0].set_ylabel("fraction of triggers")
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "multS_port", "Clusters per trigger per port: data vs the forward-fold fit", note)
    for p in PORTS:
        z0 = ft[ft.port == p]
        ys = sorted(z0.y_mc.unique())
        fig, M, R = main_resid_axes(1, len(ys), figsize=(3.7 * len(ys), 4.8))
        for j, y in enumerate(ys):
            z = z0[np.isclose(z0.y_mc, y)]
            runs = " ".join(str(r) for r in dict.fromkeys(z.run))
            _mult_panel(M[0, j], R[0, j], z, f"y = {y:+.0f} cm ({runs})", z.groupby("n").mc_s1.sum())
        M[0, 0].set_ylabel("fraction of triggers")
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"multS_pos_port{p}", f"Port {p}: clusters per trigger at each source position, data vs fit",
               note + "; runs at one MC position summed (P3 y=−50 holds 6187 at y=−60)")
        runs = list(dict.fromkeys(z0.run))
        fig, M, R = main_resid_axes(1, len(runs), figsize=(3.4 * len(runs), 4.6))
        for j, r in enumerate(runs):
            z = z0[z0.run == r]
            _mult_panel(M[0, j], R[0, j], z, f"{r} (y = {z.y.iloc[0]:+.0f})", z.groupby("n").mc_s1.sum())
        M[0, 0].set_ylabel("fraction of triggers")
        M[0, 0].legend(frameon=False, fontsize=6)
        O.save(fig, f"multS_runs_port{p}", f"Port {p}: clusters per trigger per run, data vs fit", note)


def _cl_panel(am, ar, z, title, xlab):
    from ..plotting import draw_ratio
    e, x = _edges(z), _centres(z)
    _hstep(am, e, z.mc_pt, None, PRED_C, "MC total = capture + non-capture")
    _hstep(am, e, z.mc_pt - z.mc_non_pt, None, PRED_C, f"MC capture clusters (× s = {z.s.iloc[0]:.3f})", ls="--", lw=1.1)
    _hstep(am, e, z.mc_non_pt, None, VAR_C, "MC non-capture clusters (6256 overlay)", ls=":", lw=1.5)
    _pts(am, x, z.data_pt, z.data_pt_err, "data (false starts subtracted)")
    am.set_ylim(bottom=0)
    draw_ratio(ar, x, z.data_pt, z.data_pt_err, z.mc_pt, z.mc_pt_err, label="data / MC", color=DATA_C)
    ar.set_xlabel(xlab)
    am.set_title(title, fontsize=9)


def _s_clusters(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    cl = rd("forwardfold_dm_clusters")
    for var in ("time", "pe", "hits"):
        z0 = cl[cl["var"] == var]
        if var == "time":
            z0 = z0[(z0.lo >= 2) & (z0.hi <= 67)]
        note = (f"{tagline}; data per AmBe trigger (false starts subtracted); MC per simulated event, capture clusters "
                "× the port's fitted s, no extra source clusters; not a fit; lower panel = data / MC")
        fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
        for j, p in enumerate(PORTS):
            z = z0[(z0.level == "port") & (z0.port == p)]
            _cl_panel(M[0, j], R[0, j], z, f"Port {p}: data {z.data_pt.sum():.4f}, MC {z.mc_pt.sum():.4f} per trigger",
                      XLAB[var])
        M[0, 0].set_ylabel("clusters per AmBe trigger per bin")
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"clS_{var}_port", f"Data vs MC per port: {XLAB[var]}", note)
        for p in PORTS:
            zp = z0[(z0.level == "position") & (z0.port == p)]
            ys = sorted(zp.y.unique())
            fig, M, R = main_resid_axes(1, len(ys), figsize=(3.7 * len(ys), 4.8))
            for j, y in enumerate(ys):
                z = zp[np.isclose(zp.y, y)]
                _cl_panel(M[0, j], R[0, j], z, f"y = {y:+.0f} ({z.runs.iloc[0]})", XLAB[var])
            M[0, 0].set_ylabel("clusters per AmBe trigger per bin")
            M[0, 0].legend(frameon=False, fontsize=6)
            O.save(fig, f"clS_{var}_pos_port{p}", f"Port {p}: data vs MC at each source position, {XLAB[var]}", note)


def _s_capture(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    cl = rd("forwardfold_dm_clusters")
    ft = rd("forwardfold_dm_capturetime")
    t = cl[(cl["var"] == "time") & (cl.lo >= 2) & (cl.hi <= 67)]
    for sample, lab in (("data", "data (source runs)"), ("mc_all", "MC, all clusters")):
        tag = "data" if sample == "data" else "mc"
        fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
        for j, p in enumerate(PORTS):
            z = t[(t.level == "port") & (t.port == p)]
            f = ft[(ft.level == "port") & (ft.port == p) & (ft["sample"] == sample)].iloc[0]
            h = z.src.to_numpy(float) if sample == "data" else (z.mc_cap_raw + z.mc_non_raw).to_numpy(float)
            _ct_panel(M[0, j], R[0, j], _edges(z), h, f, lab)
            M[0, j].set_title(f"Port {p}", fontsize=10)
            M[0, j].legend(frameon=False, fontsize=7)
            R[0, j].set_xlabel("cluster time [µs]")
        M[0, 0].set_ylabel("clusters per µs")
        O.save(fig, f"ctS_{tag}_port", f"Neutron capture time per port: {lab}",
               f"{PLAIN_FIT}; {tagline}")
        for p in PORTS:
            zp = t[(t.level == "position") & (t.port == p)]
            ys = sorted(zp.y.unique())
            fig, M, R = main_resid_axes(1, len(ys), figsize=(3.7 * len(ys), 4.8))
            for j, y in enumerate(ys):
                z = zp[np.isclose(zp.y, y)]
                f = ft[(ft.level == "position") & (ft.port == p) & np.isclose(ft.y, y) & (ft["sample"] == sample)]
                if f.empty or not bool(f.fit_ok.iloc[0]):
                    M[0, j].set_visible(False); R[0, j].set_visible(False)
                    continue
                h = z.src.to_numpy(float) if sample == "data" else (z.mc_cap_raw + z.mc_non_raw).to_numpy(float)
                _ct_panel(M[0, j], R[0, j], _edges(z), h, f.iloc[0], lab)
                M[0, j].set_title(f"y = {y:+.0f} cm ({z.runs.iloc[0]})", fontsize=9)
                M[0, j].legend(frameon=False, fontsize=6)
                R[0, j].set_xlabel("cluster time [µs]")
            M[0, 0].set_ylabel("clusters per µs")
            O.save(fig, f"ctS_{tag}_pos_port{p}", f"Port {p}: neutron capture time at each source position, {lab}",
                   f"{PLAIN_FIT}; {tagline}")


def _s_maps(O: Out, rd):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ft = rd("forwardfold_dm_capturetime")
    pos = ft[(ft.level == "position") & ft.fit_ok]
    order = [1, 5, 2, 3, 4]

    def piv(sample, col):
        return pos[pos["sample"] == sample].pivot_table(index="y", columns="port", values=col).reindex(columns=order)

    def draw(name, V, E, simple, detail, cbl):
        lab = np.vectorize(lambda v, e: "" if not np.isfinite(v) else f"{v:.2f}±{e:.2f}")(V.to_numpy(float),
                                                                                           E.to_numpy(float))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(V, annot=lab, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 11}, linecolor="black",
                    linewidths=0.2, cbar_kws={"label": cbl}, ax=ax)
        ax.set_xticklabels([f"Port {p}" for p in order])
        ax.set_yticklabels([f"{v:+.0f}" for v in V.index], rotation=0)
        ax.set_xlabel("Port")
        ax.set_ylabel("source height y [cm]")
        ax.invert_yaxis()
        O.save(fig, name, simple, detail, compare=False)

    base = ("fit A·(1 − e^(−t/t_th))·e^(−t/τ) + B, 1 µs bins, 2–66 µs; one cell per source position; Port 3 y = −50 "
            "includes run 6187 at y = −60; ports ordered by z (Port 4 is at x = 75 cm)")
    for par, nm in (("tau", "Capture time"), ("therm", "Thermalisation time")):
        for sample, lab in (("data", "data"), ("mc_all", "MC")):
            draw(f"{par}S_map_{'data' if sample == 'data' else 'mc'}", piv(sample, par), piv(sample, f"{par}_err"),
                 f"{nm} per source position: {lab}", base, f"{nm} (µs)")
    draw("tauS_map_bfix", piv("data_B6256", "tau"), piv("data_B6256", "tau_err"),
         "Capture time per source position: data, flat term fixed to the dummy run",
         base.replace("B free", "B fixed to the 6256 flat level"), "Capture Time (µs)")
    d = piv("data", "tau") - piv("mc_all", "tau")
    e = np.sqrt(piv("data", "tau_err") ** 2 + piv("mc_all", "tau_err") ** 2)
    draw("tauS_map_diff", d, e, "Capture time per source position: data − MC", base, "data − MC (µs)")


def _s_focus(O: Out, ctx: RunContext, rd, tagline: str):
    from math import erf, sqrt
    from ..plotting import draw_ratio, main_resid_axes
    meta = _opt(ctx, "forwardfold_focus_meta")
    if meta is None:
        print("[forwardfold deckplots] no forwardfold_focus_meta.csv (run --stage focus): focus figures skipped")
        return
    m = meta.iloc[0]
    port, y0, label = int(m.port), float(m.y), str(m.label)
    runs = [int(v) for v in str(m.runs).split()]
    where = f"Port {port}, y = {y0:+.0f} cm"
    ic = rd("forwardfold_focus_icspec")
    cols = runs + ([0] if len(runs) > 1 else [])
    fig, M, R = main_resid_axes(1, len(cols), figsize=(6.2 * len(cols), 5.2))
    for j, r in enumerate(cols):
        z = (ic[ic.run == r] if r else ic.groupby(["lo", "hi"], as_index=False)[["source", "bkg_F", "bkg_F_err"]].sum())
        e, x = _edges(z), _centres(z)
        _hstep(M[0, j], e, z.bkg_F, z.bkg_F_err, PRED_C, "6264 × F (gain-matched)")
        _pts(M[0, j], x, z.source, np.sqrt(z.source), "source run")
        M[0, j].set_yscale("log")
        draw_ratio(R[0, j], x, z.source, np.sqrt(z.source), z.bkg_F, z.bkg_F_err, label="source / (6264×F)", logy=True,
                   color=DATA_C)
        R[0, j].set_xlabel("BGO trigger charge IC")
        M[0, j].set_title(f"run {r}" if r else "both runs", fontsize=10)
    M[0, 0].set_ylabel("triggers per 250 IC")
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "focus_icspec", f"{where}: BGO trigger charge, source runs vs the scaled no-source run",
           "every trigger, no cuts; black = source run, blue = no-source run 6264 × F (gain-matched); lower panel = "
           "source / (6264 × F)")
    ft = rd("forwardfold_focus_templates")
    fz = ft[ft.run.isin(runs)]
    fig, M, R = main_resid_axes(1, len(cols), figsize=(5.2 * len(cols), 5.0))
    for j, r in enumerate(cols):
        z = fz[fz.run == r] if r else fz
        _mult_panel(M[0, j], R[0, j], z, (f"run {r}" if r else "both runs") + f": {int(z.data.sum()):,} triggers",
                    z.groupby("n").mc_s1.sum())
    M[0, 0].set_ylabel("fraction of triggers")
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "focus_mult", f"{where}: clusters per trigger, data vs the forward-fold fit",
           f"{tagline}; Port {port} fit: s = {fz.s.iloc[0]:.4f}, μ_ind = {fz.mu_ind.iloc[0]:.4f}; black = data, blue = fit, "
           "grey dashed = MC as simulated; lower panel = data / fit")
    fig, M, R = main_resid_axes(1, 2, figsize=(13, 5.4))
    g = fz.groupby("n")
    _mult_panel(M[0, 0], R[0, 0], fz, "how the fit builds the prediction", g.mc_s1.sum(),
                extra=[(f"captures thinned, s = {fz.s.iloc[0]:.3f}", g.mc_s.sum(), THIRD_C, ":"),
                       (f"+ Poisson(μ_ind = {fz.mu_ind.iloc[0]:.4f})", g.mc_s_mu.sum(), THIRD_C, "-.")])
    if "mc_pos" in fz:
        _mult_panel(M[0, 1], R[0, 1], fz, "port fit vs a fit of this position alone", None,
                    extra=[(f"position fit: s = {fz.s_pos.iloc[0]:.3f}, μ_ind = {fz.mu_pos.iloc[0]:.4f}",
                            g.mc_pos.sum(), THIRD_C, "--")])
    M[0, 0].set_ylabel("fraction of triggers")
    for a in M[0]:
        a.legend(frameon=False, fontsize=7)
    O.save(fig, "focus_mult_build", f"{where}: from the MC to the fitted prediction, step by step",
           f"{tagline}; both runs summed; orange lines = the MC after each step (no false starts added); solid blue = "
           "the full fit; lower panel = data / fit")
    cl = rd("forwardfold_dm_clusters")
    z0 = cl[(cl.level == "position") & (cl.group == label)]
    fig, M, R = main_resid_axes(1, 3, figsize=(16, 5.0))
    for j, var in enumerate(("time", "pe", "hits")):
        z = z0[z0["var"] == var]
        if var == "time":
            z = z[(z.lo >= 2) & (z.hi <= 67)]
        _cl_panel(M[0, j], R[0, j], z, f"data {z.data_pt.sum():.4f}, MC {z.mc_pt.sum():.4f} clusters per trigger",
                  XLAB[var])
    M[0, 0].set_ylabel("clusters per AmBe trigger per bin")
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "focus_cl", f"{where}: clusters in data and MC (time, charge, hits)",
           f"{tagline}; data per AmBe trigger (false starts subtracted); MC capture clusters × the port's s, no extra "
           "source clusters; not a fit; lower panel = data / MC")
    fig, M, R = main_resid_axes(1, 2, figsize=(12, 5.0))
    for j, var in enumerate(("time", "pe")):
        z = z0[z0["var"] == var]
        if var == "time":
            z = z[(z.lo >= 2) & (z.hi <= 67)]
        e, x = _edges(z), _centres(z)
        _hstep(M[0, j], e, z.mc_non_pt, None, PRED_C, "MC non-capture clusters")
        _pts(M[0, j], x, z.dummy_pt, z.dummy_pt_err, "6256 dummy triggers")
        M[0, j].set_ylim(bottom=0)
        draw_ratio(R[0, j], x, z.dummy_pt, z.dummy_pt_err, z.mc_non_pt, z.mc_non_pt_err, label="6256 / MC", color=DATA_C)
        R[0, j].set_xlabel(XLAB[var])
        M[0, j].set_title(f"6256 {z.dummy_pt.sum():.4f}, MC {z.mc_non_pt.sum():.4f} clusters per trigger", fontsize=9)
    M[0, 0].set_ylabel("clusters per trigger per bin")
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "focus_accid", f"{where}: accidental clusters, dummy run vs the MC overlay",
           "black = clusters per random trigger of the dummy run 6256; blue = MC clusters that are not a neutron capture, "
           "per simulated event; the MC takes these from the same 6256 triggers, so the two are not independent")
    t = cl[(cl["var"] == "time") & (cl.lo >= 2) & (cl.hi <= 67) & (cl.level == "position") & (cl.group == label)]
    cf = rd("forwardfold_dm_capturetime")
    cf = cf[(cf.level == "position") & (cf.group == label)].set_index("sample")
    fig, M, R = main_resid_axes(1, 3, figsize=(16, 5.0))
    for j, (sample, lab) in enumerate((("data", "data, flat term B free"), ("data_B6256", "data, B = 6256 level"),
                                       ("mc_all", "MC, all clusters"))):
        h = t.src.to_numpy(float) if sample != "mc_all" else (t.mc_cap_raw + t.mc_non_raw).to_numpy(float)
        _ct_panel(M[0, j], R[0, j], _edges(t), h, cf.loc[sample], lab)
        M[0, j].legend(frameon=False, fontsize=7)
        R[0, j].set_xlabel("cluster time [µs]")
    M[0, 0].set_ylabel("clusters per µs")
    O.save(fig, "focus_ct", f"{where}: neutron capture time, data and MC", f"{PLAIN_FIT}; {tagline}")
    pr = rd("forwardfold_dm_tail_pairs")
    fc = rd("forwardfold_focus_capturetime")
    if len(runs) == 2:
        pair = f"{runs[0]}_{runs[1]}" if f"{runs[0]}_{runs[1]}" in set(pr.pair) else f"{runs[1]}_{runs[0]}"
        z0 = pr[pr.pair == pair]
        fig, M, R = main_resid_axes(1, 2, figsize=(12, 5.0))
        for j, r in enumerate(runs):
            z = z0[(z0.run == r) & (z0["var"] == "time") & (z0.lo >= 2) & (z0.hi <= 67)]
            f = fc[(fc.run == r) & (fc["sample"] == "data")].iloc[0]
            _ct_panel(M[0, j], R[0, j], _edges(z), z["count"].to_numpy(float), f, f"run {r}")
            M[0, j].legend(frameon=False, fontsize=7)
            R[0, j].set_xlabel("cluster time [µs]")
        M[0, 0].set_ylabel("clusters per µs")
        O.save(fig, "focus_ct_runs", f"{where}: neutron capture time of each run", f"{PLAIN_FIT}; selected triggers "
               "of each run")
        a, b2 = (int(v) for v in pair.split("_"))
        fig, M, R = main_resid_axes(1, 4, figsize=(17, 4.8))
        for j, var in enumerate(("mult", "time", "pe", "hits")):
            z = z0[z0["var"] == var]
            if var == "time":
                z = z[(z.lo >= 2) & (z.hi <= 67)]
            A, B = z[z.run == a], z[z.run == b2]
            Na, Nb = float(A.N.iloc[0]), float(B.N.iloc[0])
            ha, hb = A["count"].to_numpy(float), B["count"].to_numpy(float)
            _hstep(M[0, j], _edges(B), hb / Nb, np.sqrt(hb) / Nb, PRED_C, f"{b2} ({int(Nb):,} trig.)")
            _pts(M[0, j], _centres(A), ha / Na, np.sqrt(ha) / Na, f"{a} ({int(Na):,} trig.)")
            if var == "mult":
                M[0, j].set_yscale("log")
            else:
                M[0, j].set_ylim(bottom=0)
            draw_ratio(R[0, j], _centres(A), ha / Na, np.sqrt(ha) / Na, hb / Nb, np.sqrt(hb) / Nb, label=f"{a} / {b2}",
                       color=DATA_C)
            R[0, j].set_xlabel(XLAB[var] if var != "mult" else "clusters per trigger (8 = 8+)")
        M[0, 0].set_ylabel("per veto-passing gated trigger")
        M[0, 0].legend(frameon=False, fontsize=8)
        O.save(fig, "focus_pair", f"{where}: the two runs compared (Data/Data)",
               f"{tagline}; per selected trigger, no MC; black = {a}, blue = {b2}; lower panel = {a} / {b2}")
    ef = rd("forwardfold_efficiency")
    nomv = ef[(ef.variant == "nominal") & (ef.group == f"port{port}")].sort_values("y")
    bpv = ef[(ef.variant == "by_position") & ef.group.str.startswith(f"port{port}_")].copy()
    fig, M, R = main_resid_axes(1, 1, figsize=(8, 6))
    mc_err = np.sqrt(nomv.eps_mc * (1 - nomv.eps_mc) / nomv.mc_events)
    M[0, 0].errorbar(nomv.y, nomv.eps_mc, mc_err, fmt="s-", ms=3.5, lw=1.2, capsize=0, color=PRED_C,
                     label="MC truth (s = 1)")
    M[0, 0].errorbar(nomv.y, nomv.eps_data, nomv.eps_data_err, fmt="o-", ms=3.5, lw=1.2, capsize=0, color=DATA_C,
                     label=f"data, port fit (s = {nomv.s.iloc[0]:.4f})")
    if len(bpv):
        bpv = bpv.sort_values("y")
        M[0, 0].errorbar(bpv.y + 2.0, bpv.eps_data, bpv.eps_data_err, fmt="D", ms=3.5, capsize=0, color=THIRD_C,
                         label="data, one fit per position")
    M[0, 0].set_ylabel("neutron detection efficiency")
    M[0, 0].legend(frameon=False, fontsize=8)
    draw_ratio(R[0, 0], nomv.y, nomv.eps_data, nomv.eps_data_err, nomv.eps_mc, mc_err, label="data / MC", floor=0.05,
               cap=0.5, color=DATA_C)
    R[0, 0].set_xlabel("source height y [cm]")
    O.save(fig, "focus_eff", f"Port {port}: neutron detection efficiency per source height, data vs MC",
           "efficiency = fraction of AmBe triggers with at least one neutron-capture cluster in 2–67 µs; data value "
           "= MC value after keeping each capture cluster with probability s; orange points shifted by +2 cm so they "
           "do not overlap; lower panel = data (port fit) / MC")
    toys = rd("forwardfold_closure_toys")
    toys = toys[toys.fit_ok & (toys.group == f"port{port}")]
    edges = np.arange(-4.0, 4.01, 0.5)
    cdf = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in edges])
    x = 0.5 * (edges[1:] + edges[:-1])
    fig, M, R = main_resid_axes(1, 2, figsize=(12, 5.0))
    for j, (col, lab) in enumerate((("s_pull", "s"), ("mu_pull", "μ_ind"))):
        h = np.histogram(np.clip(toys[col], -3.999, 3.999), edges)[0].astype(float)
        ex = len(toys) * np.diff(cdf)
        _hstep(M[0, j], edges, ex, None, PRED_C, "unit Gaussian")
        _pts(M[0, j], x, h, np.sqrt(h), f"{len(toys)} toys, 6 injection points")
        M[0, j].set_ylim(bottom=0)
        draw_ratio(R[0, j], x, h, np.sqrt(h), ex, None, label="toys / Gaussian", cap=1.0, color=DATA_C)
        R[0, j].set_xlabel(f"pull of {lab} = (fit − true) / fitted error")
        M[0, j].set_title(f"mean {toys[col].mean():+.3f}, width {toys[col].std():.3f}", fontsize=10)
    M[0, 0].set_ylabel("toys per 0.5")
    M[0, 0].legend(frameon=False, fontsize=8)
    O.save(fig, "focus_closure", f"Port {port}: does the fit return what was put in? (MC closure)",
           "fake data made from the MC with known s and μ_ind (six settings pooled), fitted like the real data; pull = "
           "(fitted − true) / fitted error; lower panel = toys / Gaussian")


def _at_mc_y(s: pd.Series, y) -> np.ndarray:
    """Value of a per-MC-height series at each data height (nearest MC height: Port 3 y = -60 uses MC -50)."""
    idx = s.index.to_numpy(float)
    return np.array([float(s.iloc[int(np.argmin(np.abs(idx - v)))]) for v in np.asarray(y, float)])


def _port_positions(em: pd.DataFrame, fac: pd.DataFrame, port: int) -> pd.DataFrame:
    nomv = fac[(fac.variant == "nominal") & (fac.group == f"port{port}")]
    x0, z0 = float(nomv.x.iloc[0]), float(nomv.z.iloc[0])
    pos = em[em.level == "position"]
    return pos[np.isclose(pos.x_pos, x0) & np.isclose(pos.z_pos, z0)].sort_values("y_pos")


def _grid(n: int, ncols: int = 3):
    """2-row (or more) grid of main+residual cells for n panels; spare cells hold the legend or are hidden.
    Returns fig, the n (main, resid) pairs and the spare main axis (or None)."""
    from ..plotting import main_resid_axes
    nrows = -(-n // ncols)
    fig, M, R = main_resid_axes(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    cells = [(M[i, j], R[i, j]) for i in range(nrows) for j in range(ncols)]
    for am, ar in cells[n:]:
        am.set_axis_off()
        ar.set_axis_off()
    return fig, cells[:n], (cells[n][0] if len(cells) > n else None)


def _legend_cell(spare, ref_ax):
    h, lab = ref_ax.get_legend_handles_labels()
    if spare is not None:
        spare.legend(h, lab, frameon=False, fontsize=10, loc="center")
    else:
        ref_ax.legend(frameon=False, fontsize=7.5)


def _methods_every_position(O: Out, rd, em, arr, fac, ports, note):
    """Per port, one panel per height: data and the four models (fold = port fit, from forwardfold_bins)."""
    bins = rd("forwardfold_bins")
    for port in ports:
        q = _port_positions(em, fac, port)
        if q.empty:
            continue
        fig, cells, spare = _grid(len(q))
        for (am, ar), (i, r) in zip(cells, q.iterrows()):
            cnt = arr[f"g{i}__counts"].astype(float)
            N, K = cnt.sum(), len(cnt)
            runs = [int(v) for v in str(r.runs).split()]
            fz = (bins[(bins.group == f"port{port}") & bins.run.isin(runs)].groupby("n").pred.sum()
                  .sort_index().to_numpy(float))
            fold = np.append(fz[:K - 1], fz[K - 1:].sum()) / fz.sum()
            e, xk = np.arange(-0.5, K, 1.0), np.arange(K, dtype=float)
            _hstep(am, e, fold, None, PRED_C, "forward fold (port fit)")
            for k, key in enumerate(("poisson", "dd", "dds")):
                col, mk, ls, lab = METHOD_STYLE[key]
                prof = arr[f"g{i}__{key}_profile"].astype(float)
                _hstep(am, e, prof, None, col, lab, ls=ls, lw=1.3)
                ar.errorbar(xk + 0.1 * k - 0.1, cnt / N / prof, np.sqrt(cnt) / N / prof, fmt=mk, ms=3, capsize=0,
                            color=col)
            _pts(am, xk, cnt / N, np.sqrt(cnt) / N, "data")
            ar.errorbar(xk + 0.2, cnt / N / fold, np.sqrt(cnt) / N / fold, fmt="s", ms=3, capsize=0, color=PRED_C)
            am.set_yscale("log")
            am.set_title(f"y = {float(r.y_pos):+.0f} cm, {int(N):,} triggers", fontsize=11)
            am.set_ylabel("fraction of triggers")
            ar.set_ylabel("data / model")
            ar.set_gid("residual")
            ar.set_xticks(xk, [str(k) for k in range(K - 1)] + [f"≥{K - 1}"])
            ar.set_xlabel("clusters per trigger (2–67 µs)")
        _legend_cell(spare, cells[0][0])
        O.save(fig, f"methods_fit_port{port}", f"Port {port}: the data and the four efficiency models at every height",
               note + "; black = data; lower panels = data / each model")


def _methods_allports(O: Out, em, fac, ports, note):
    """All ports: data efficiency per method vs height, and each method − forward fold (port fit)."""
    fig, cells, spare = _grid(len(ports))
    for (am, ar), port in zip(cells, ports):
        q = _port_positions(em, fac, port)
        nomv = fac[(fac.variant == "nominal") & (fac.group == f"port{port}")].sort_values("y")
        bpv = fac[(fac.variant == "by_position") & fac.group.str.startswith(f"port{port}_")]
        bpv = bpv.set_index(bpv.data_y.astype(str).str.split().str[0].astype(float))
        y = q.y_pos.to_numpy(float)
        ref = _at_mc_y(nomv.set_index("y").eps_data, y)
        am.errorbar(nomv.y, nomv.eps_data, nomv.eps_data_err, fmt="s-", ms=3, lw=1.3, capsize=0, color=PRED_C,
                    label="forward fold, port fit")
        if len(bpv):
            b = bpv.reindex(y)
            am.errorbar(y - 3.0, b.eps_data, b.eps_data_err, fmt="D", ms=3, capsize=0, color=PRED_C, mfc="none",
                        label="forward fold, each position alone")
            ar.errorbar(y - 3.0, 100 * (b.eps_data.to_numpy(float) - ref), 100 * b.eps_data_err.to_numpy(float),
                        fmt="D", ms=3, capsize=0, color=PRED_C, mfc="none")
        for dx, key in ((3.0, "poisson"), (6.0, "dd"), (9.0, "dds")):
            col, mk, ls, lab = METHOD_STYLE[key]
            e = q[f"{key}_eff"].to_numpy(float)
            lo, hi = e - q[f"{key}_eff_lo"].to_numpy(float), q[f"{key}_eff_hi"].to_numpy(float) - e
            am.errorbar(y + dx, e, [lo, hi], fmt=mk, ls=ls, ms=3, lw=1.0, capsize=0, color=col, label=lab)
            ar.errorbar(y + dx, 100 * (e - ref), [100 * lo, 100 * hi], fmt=mk, ms=3, capsize=0, color=col)
        am.errorbar(y + 12.0, q.naive_eff, q.naive_err, fmt="x", ms=3.5, capsize=0, color="0.4",
                    label="counting: 1 − P(0 clusters)")
        ar.errorbar(y + 12.0, 100 * (q.naive_eff.to_numpy(float) - ref), 100 * q.naive_err, fmt="x", ms=3.5,
                    capsize=0, color="0.4")
        am.set_title(f"Port {port}", fontsize=11)
        am.set_ylabel("efficiency (data)")
        ar.set_ylabel("method − fold [pp]")
        ar.set_gid("residual")
        ar.set_xlabel("source height y [cm]")
    for k in (0, 1):
        lo_ = min(c[k].get_ylim()[0] for c in cells)
        hi_ = max(c[k].get_ylim()[1] for c in cells)
        for c in cells:
            c[k].set_ylim(lo_, hi_)
    _legend_cell(spare, cells[0][0])
    O.save(fig, "methods_eff_allports", "Every port: data efficiency from four methods",
           note + "; lower panels = each method − forward fold (port fit), in percentage points")


def _s_methods(O: Out, ctx: RunContext, rd):
    """Counting, Poisson and data-driven efficiency (ambe stats efficiency on the step-3 data config) next to the
    forward fold, for the positions that fit ran on."""
    from ..context import load_context
    from ..plotting import main_resid_axes
    blk = dict((ctx.extra or {}).get("fold") or {})
    dctx = load_context(blk["data_config"])
    p = Path(dctx.csv_path("efficiency_fit_by_position"))
    if not p.exists():
        print("[forwardfold deckplots] no efficiency_fit_by_position.csv for the data config: method figures skipped")
        return
    em = pd.read_csv(p)
    arr = np.load(p.with_suffix(".npz"))
    pos = em[em.level == "position"].copy()
    fac = rd("forwardfold_efficiency")
    meta = _opt(ctx, "forwardfold_focus_meta")
    nom_all = fac[fac.variant == "nominal"]
    ports = [p for p in PORTS if any(np.isclose(r.x, pos.x_pos).any() and np.isclose(r.z, pos.z_pos).any()
                                     for r in nom_all[nom_all.group == f"port{p}"].itertuples())]
    note = ("same triggers and clusters as the forward fold; Poisson fit: background clusters Poisson, neutron adds one "
            "cluster with probability ε; data-driven fit: background clusters drawn from the dummy run 6256; scaled: the "
            "6256 background allowed to be larger")
    for port in ports:
        nomv = fac[(fac.variant == "nominal") & (fac.group == f"port{port}")].sort_values("y")
        bpv = fac[(fac.variant == "by_position") & fac.group.str.startswith(f"port{port}_")]
        bpv = bpv.set_index(bpv.data_y.astype(str).str.split().str[0].astype(float))
        x0, z0 = float(nomv.x.iloc[0]), float(nomv.z.iloc[0])
        q = pos[np.isclose(pos.x_pos, x0) & np.isclose(pos.z_pos, z0)].sort_values("y_pos")
        if q.empty:
            continue
        ref = pd.Series(_at_mc_y(nomv.set_index("y").eps_data, q.y_pos), index=q.y_pos.to_numpy())
        fig, M, R = main_resid_axes(1, 1, figsize=(9, 6.2))
        am, ar = M[0, 0], R[0, 0]
        y = q.y_pos.to_numpy(float)
        am.errorbar(nomv.y, nomv.eps_data, nomv.eps_data_err, fmt="s-", ms=3.5, lw=1.4, capsize=0, color=PRED_C,
                    label="forward fold, port fit")
        if len(bpv):
            b = bpv.reindex(y)
            am.errorbar(y - 2.0, b.eps_data, b.eps_data_err, fmt="D", ms=3.5, capsize=0, color=PRED_C, mfc="none",
                        label="forward fold, each position alone")
            ar.errorbar(y - 2.0, 100 * (b.eps_data.to_numpy(float) - ref.to_numpy(float)),
                        100 * b.eps_data_err.to_numpy(float), fmt="D", ms=3, capsize=0, color=PRED_C, mfc="none")
        for k, (dx, key) in enumerate(((2.0, "poisson"), (4.0, "dd"), (6.0, "dds"))):
            col, mk, ls, lab = METHOD_STYLE[key]
            e = q[f"{key}_eff"].to_numpy(float)
            lo, hi = e - q[f"{key}_eff_lo"].to_numpy(float), q[f"{key}_eff_hi"].to_numpy(float) - e
            am.errorbar(y + dx, e, [lo, hi], fmt=mk, ls=ls, ms=3.5, lw=1.1, capsize=0, color=col, label=lab)
            ar.errorbar(y + dx, 100 * (e - ref.to_numpy(float)), [100 * lo, 100 * hi], fmt=mk, ms=3, capsize=0,
                        color=col)
        am.errorbar(y + 8.0, q.naive_eff, q.naive_err, fmt="x", ms=4, capsize=0, color="0.4",
                    label="counting: 1 − P(0 clusters)")
        ar.errorbar(y + 8.0, 100 * (q.naive_eff.to_numpy(float) - ref.to_numpy(float)), 100 * q.naive_err, fmt="x",
                    ms=4, capsize=0, color="0.4")
        ar.set_ylabel("method − fold [pp]")
        ar.set_gid("residual")
        ar.set_xlabel("source height y [cm] (points offset for visibility)")
        am.set_ylabel("neutron detection efficiency (data)")
        am.legend(frameon=False, fontsize=8)
        O.save(fig, f"methods_eff_port{port}", f"Port {port}: data efficiency from four methods",
               note + "; lower panel = each method − forward fold (port fit), in percentage points")
    ntoys = int((dctx.fit_params or {}).get("n_gof_toys", 2000))
    _methods_every_position(O, rd, em, arr, fac, ports, note)
    if len(ports) > 1:
        _methods_allports(O, em, fac, ports, note)
    if meta is None:
        return
    mm = meta.iloc[0]
    fx, fy, fzz = (float(v) for v in fac[fac.mc_label == mm.label][["x", "y", "z"]].iloc[0])
    idx = [i for i, r in enumerate(em.itertuples()) if r.level == "position" and np.isclose(r.x_pos, fx)
           and np.isclose(r.y_pos, fy) and np.isclose(r.z_pos, fzz)]
    if not idx:
        return
    i = idx[0]
    r = em.iloc[i]
    cnt = arr[f"g{i}__counts"].astype(float)
    N = cnt.sum()
    K = len(cnt)
    ft = rd("forwardfold_focus_templates")
    fz = ft[ft.run.isin([int(v) for v in str(mm.runs).split()])].groupby("n").pred.sum().to_numpy(float)
    fold = np.append(fz[:K - 1], fz[K - 1:].sum()) / fz.sum()
    e = np.arange(-0.5, K, 1.0)
    xk = np.arange(K, dtype=float)
    fig, M, R = main_resid_axes(1, 1, figsize=(9, 6.2))
    am, ar = M[0, 0], R[0, 0]
    _hstep(am, e, fold, None, PRED_C, "forward fold (port fit)")
    for key in ("poisson", "dd", "dds"):
        col, mk, ls, lab = METHOD_STYLE[key]
        prof = arr[f"g{i}__{key}_profile"].astype(float)
        pv = float(r[f"{key}_p"])
        pstr = f"< {1.0 / ntoys:.0e}" if (key == "poisson" and pv == 0) else f"= {pv:.2g}"
        _hstep(am, e, prof, None, col, f"{lab}: ε = {r[f'{key}_eff']:.4f}, p {pstr}", ls=ls, lw=1.3)
        ar.errorbar(xk + 0.1 * ("poisson", "dd", "dds").index(key) - 0.1, cnt / N / prof, np.sqrt(cnt) / N / prof,
                    fmt=mk, ms=3, capsize=0, color=col)
    _pts(am, xk, cnt / N, np.sqrt(cnt) / N, f"data ({int(N):,} triggers)")
    ar.errorbar(xk + 0.2, cnt / N / fold, np.sqrt(cnt) / N / fold, fmt="s", ms=3, capsize=0, color=PRED_C)
    am.set_yscale("log")
    ar.set_ylabel("data / model")
    ar.set_gid("residual")
    ar.set_xticks(xk, [str(k) for k in range(K - 1)] + [f"≥{K - 1}"])
    ar.set_xlabel("clusters per trigger (2–67 µs)")
    am.set_ylabel("fraction of triggers")
    am.legend(frameon=False, fontsize=8)
    O.save(fig, "methods_fit_focus", f"Port {int(mm.port)}, y = {float(mm.y):+.0f} cm: the data and the four efficiency "
           "models", note + "; black = data; lower panel = data / each model")
