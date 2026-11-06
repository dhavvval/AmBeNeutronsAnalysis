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
RESID_NOTE = "residual = (A − B)/√(σ_A² + σ_B²)"


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


def _cmp(am, ar, x, a, sa, b, sb, la: str, lb: str, logy: bool = False, lim=None, extra=None, rlabel: str = "pull"):
    from ..plotting import draw_pull
    am.errorbar(x, a, sa, fmt="o", ms=3, lw=1, color="k", label=la)
    am.plot(x, b, "-", lw=1.4, color="C0", label=lb)
    for lab, yv, st in (extra or ()):
        am.plot(x, yv, st, lw=1.1, color="C1", label=lab)
    if logy:
        am.set_yscale("log")
    return draw_pull(ar, x, a, sa, b, sb, lim=lim, label=rlabel)


def _cmp_diff(am, ar, x, a, sa, b, sb, la: str, lb: str, rlabel: str, center_zero: bool = True):
    from ..plotting import draw_diff
    am.errorbar(x, a, sa, fmt="o-", ms=4, lw=1, color="k", label=la)
    am.errorbar(x, b, sb, fmt="s-", ms=4, lw=1, color="C0", label=lb)
    return draw_diff(ar, x, a, sa, b, sb, label=rlabel, center_zero=center_zero)


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
    tagline = "TA ClusterFinder 20 ns/5 hits + C++ merge 50/2000, clusters in 2–67 µs, no box; AmBe2.0v4_bkgnorm vs setup-A MC (6256 overlay)"
    fdir = Path(blk["factors_csv"]).parent
    _normalisation(O, rd, fdir, tagline, ctx.csv_path("forwardfold_dm_clmult_chi2"))
    _multiplicity(O, rd, tagline)
    _clusters(O, rd, tagline)
    _capture(O, rd, tagline)
    _efficiency(O, rd, tagline)
    _tail(O, rd, tagline)
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
                 "6264 × F (gain-matched)", logy=True, lim=10,
                 extra=[("6264 × α_live", z.bkg_alpha, "--")])
            M[0, j].set_title(f"y = {l.y:+.0f} cm", fontsize=10)
            R[0, j].set_xlabel("BGO IC (IC_adjusted)")
        M[0, 0].set_ylabel("triggers per 250 IC")
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"icspec_port{p}", f"Port {p}: BGO trigger charge, source runs vs the no-source run scaled",
               "every trigger, no cuts; 6264 scaled by the frozen F_rsb_g = N(IC>3000) ratio with the 6264 IC × g per run "
               f"(nominal) and by α_live (dashed); residual = pull vs F·6264, clipped at ±10")
    icr = pd.read_csv(fdir / "nosourcenorm_icrun_icrun.csv").sort_values(["y", "port"])
    dmod = pd.read_csv(fdir / "nosourcenorm_icrun_depthmodel.csv").iloc[0]
    fig, M, R = main_resid_axes(1, 1, figsize=(8, 5.6))
    x = icr.y.to_numpy(float) + (icr.port.to_numpy(float) - 3) * 3.0
    model = dmod.a + dmod.b_per_cm * icr.y.to_numpy(float)
    M[0, 0].errorbar(x, icr.closure_g, icr.closure_g_err, fmt="o", ms=4, color="k", label="source runs")
    yy = np.array([-110.0, 110.0])
    M[0, 0].plot(yy, dmod.a + dmod.b_per_cm * yy, "-", color="C0",
                 label=f"linear depth model, χ²/ndf {dmod.chi2:.0f}/{int(dmod.ndf)}")
    M[0, 0].set_ylabel("N(IC>3000) / (α_live · N_6264(IC·g>3000))")
    M[0, 0].legend(frameon=False, fontsize=8)
    draw_pull(R[0, 0], x, icr.closure_g, icr.closure_g_err, model, np.zeros(len(model)))
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
         np.full(len(x), ref.rate_1000_err * k), "source runs", "6256 dummy run")
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
                            extra=[("6264 × α_live", An, "--")])
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
                   f"{tagline}; 6264 × frozen F_rsb_g per run (IC>3000 panels: 6264 IC × g), α_live dashed; "
                   f"{RESID_NOTE} vs F·6264")
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
             logy=True)
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
            _cmp(M[i, j], R[i, j], z.n, z.data, np.sqrt(z.data), z.pred, np.zeros(len(z)), "data", "fit", logy=True)
            M[i, j].set_title(f"P{p} y={y:+.0f} ({z.runs.iloc[0]})", fontsize=8)
            R[i, j].set_xticks(range(5), xt)
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "mult_grid", "Clusters per trigger at every source position: data vs fit",
           f"{tagline}; per-port fit summed over the runs at each MC position (P3 y=−50 holds 6244 and 6187 at y=−60); "
           "residual = (data − fit)/√data")
    for g, z in b.groupby("group"):
        runs = list(dict.fromkeys(z.run))
        fig, M, R = main_resid_axes(1, len(runs), figsize=(3.6 * len(runs), 4.6))
        for j, r in enumerate(runs):
            w = z[z.run == r]
            _cmp(M[0, j], R[0, j], w.n, w.data, np.sqrt(w.data), w.pred, np.zeros(len(w)), "data", "fit", logy=True)
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
                 f"MC, capture clusters × s = {z.s.iloc[0]:.3f}", logy=var == "time")
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
                     "MC (s)", logy=var == "time")
                M[i, j].set_title(f"P{p} y={y:+.0f} ({z.runs.iloc[0]})", fontsize=8)
        M[0, 0].legend(frameon=False, fontsize=7)
        O.save(fig, f"cl_{var}_grid", f"Data vs MC at every source position: {XLAB[var]}",
               f"{tagline}; per AmBe trigger; MC capture clusters × s_port; {RESID_NOTE}")
        if var in ("time", "pe"):
            fig, M, R = main_resid_axes(1, 5, figsize=(18, 4.8))
            for j, p in enumerate(PORTS):
                z = z0[(z0.level == "port") & (z0.port == p)]
                _cmp(M[0, j], R[0, j], _centres(z), z.dummy_pt, z.dummy_pt_err, z.mc_non_pt, z.mc_non_pt_err,
                     "6256 dummy triggers", "MC non-capture clusters", logy=var == "time")
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
                     f"fit τ = {f.tau:.2f} ± {f.tau_err:.2f} µs, therm {f.therm:.2f}", logy=True)
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
        _cmp(M[0, j], R[0, j], x, y, e, mdl, np.zeros(len(x)), "source − F·6264", "capture + flat fit", logy=True)
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
            _cmp(M[i, j], R[i, j], x, h, np.sqrt(h), ex, np.zeros(len(x)), "toys (5 ports pooled)", "unit Gaussian")
            M[i, j].set_title(f"{lab}: s={pt.s_true:g}, μ={pt.mu_true:g}", fontsize=9)
            R[i, j].set_xlabel(f"pull of {lab}")
    M[0, 0].legend(frameon=False, fontsize=7)
    O.save(fig, "closure_pulls", "MC closure: fit pulls in 1,000 pseudo-experiments per point",
           "closure v3 (template term shared per template, full covariance); template files 0–24, pseudo-data drawn "
           "from the full MC per toy; residual = (toys − Gaussian)/√toys")


def _tail(O: Out, rd, tagline: str):
    from ..plotting import main_resid_axes
    tr = rd("forwardfold_dm_tail_runs").sort_values(["date", "run"]).reset_index(drop=True)
    fig, M, R = main_resid_axes(2, 1, figsize=(13, 8.6))
    x = np.arange(len(tr))
    for i, (cols, lab) in enumerate(((["n2"], "exactly 2 clusters"), (["n3", "n4"], "3 or more clusters"))):
        d = sum(tr[f"data_{c}"] for c in cols) / tr.N_trig
        p = sum(tr[f"pred_{c}"] for c in cols) / tr.N_trig
        _cmp(M[i, 0], R[i, 0], x, d, np.sqrt(d * tr.N_trig) / tr.N_trig, p, np.zeros(len(x)), "data", "forward-fold fit")
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
                     logy=var in ("time", "dt"))
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
                 f"{a} ({int(Na)} trig.)", f"{b} ({int(Nb)} trig.)", logy=var in ("mult", "time"))
            R[0, j].set_xlabel(XLAB[var] if var != "mult" else "clusters per trigger (8 = 8+)")
        M[0, 0].set_ylabel("per veto-passing gated trigger")
        M[0, 0].legend(frameon=False, fontsize=8)
        O.save(fig, f"pair_{pair}", f"Same position, different period: run {a} vs run {b}",
               f"{tagline}; both runs at one source position; per trigger, no MC; {RESID_NOTE}")
