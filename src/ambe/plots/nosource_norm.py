"""
No-source normalisation tests (TC-3..TC-10 of campaigns/ambe-efficiency-fit/METHODS_nosource_normalisation.md).

    ambe plots nosourcenorm --config configs/data_ambe2v4_reclust.yaml --source-runs 6046,6252 --nosource-run 6264
        --dummy-config configs/data_dummy_randoms_625x.yaml --runs-db configs/run_db/ambe_v4_sql_runs.yaml
        --method clusterfinder_nms --gap-s 30 --parts deadtime,stability,rsb,leftsb,insitu,convfit,icfit
        --foldfs livetime_port5_y0y50_foldfs --out-name port5

Parts (csv nosourcenorm_<out-name>_<table>.csv, pages in plots/nosourcenorm_<out-name>/ and one pdf):
  deadtime   interval spectrum per run fitted with dt = tau + Exp(lambda), tau ~ N(mu, sigma) (exponentially
             modified Gaussian, truncated unbinned likelihood); live time N / lambda; DAQ stalls (dt beyond the fit
             window) count as dead; window and model variations; mu by the previous readout's IC class, veto, second
             pulse and tank hits; pulser runs: share of 2.000 s pulses that were missed; run-DB scaler comparison
  stability  per run and wall-clock block: lambda, rates per live second (cosmic-tagged IC > 3000, gate, untagged
             IC < 0 and 0-300), 4.4 MeV peak position, cosmic IC median
  rsb        right-sideband factor F = N_src / N_nosrc of cosmic-tagged IC > 3000 triggers: BGO gain g from the
             cosmic IC shape, F in IC sub-bands, F from a tank muon tag (prompt cluster > 1000 p.e.), k against the
             fitted live time; extrapolation into every IC class with stat, gain and non-closure errors
  leftsb     no-source triggers split into cosmic-tagged (scaled by F) and untagged (scaled by live time, or by the
             IC < 0 control region); observed / predicted per class; tank P0 per class and the false-start share
             it implies
  insitu     accidental level inside the source triggers: pre-prompt tank hit rate, flat term of the hit-time and
             cluster-time spectra, against the no-source run and the dummy runs
  convfit    multiplicity fit with the accidental template convolved with Poisson(mu_ind) against the s^k tilt,
             template counts profiled (Barlow-Beeston); Asimov closure on MC (x) template from --foldfs
  icfit      simultaneous fit over IC classes: one efficiency per class, shared mu_ind, false-start shares fixed
  ctc        (--ctc-dir) trigger-card words decoded from the raw files (EventBuild StoreDecodedTimestamps, CTC only):
             accepted triggers (word 19 groups) against BeamCluster readouts, the dead-time fit on the complete
             accepted list, exposure of the BeamCluster counts N_BC / lambda, and k with that exposure
  icnorm     (--ic-runs, --ic-ref) BGO IC of every recorded waveform (processor.py Stage-1 features, no cuts) for each
             run, weighted by span(ref) / span(run) and by the non-paralyzable dead-time factor n / m = 1 / (1 - m tau)
             (Knoll eq. 4.24), m = recorded rate, tau = fitted mean dead time; m over the full span (as asked) and over
             the processed-data time (span minus intervals >= --gap-s, i.e. part files missing from the processed data
             while the DAQ ran); every rate variant in the csv; an unweighted page first
  muonrate   (--muon-runs) tank muon-like clusters (BeamCluster tree clusters above --muon-pe) per us of open window
             (--muon-window-us) in every run, and the ratio to the dummy run --muon-ref; time profile 2-66 us.
             Late cosmic products are accidentals and scale per window, so equal rates are what lets one dummy
             template serve every run. Thresholds stay above the neutron-capture PE range (blind-safe)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy import optimize, stats

from ..context import RunContext, load_context
from ..plotting import two_titles
from . import datamc_zerobin as dz
from . import livetime_sideband as ls

PARTS = ["deadtime", "stability", "rsb", "leftsb", "insitu", "convfit", "icfit", "ctc", "icnorm", "icrun", "muonrate"]
DUMMY_RUNS = (6254, 6256)
PORT_OF = {(0.0, 0.0): 5, (75.0, 0.0): 4, (0.0, -75.0): 1, (0.0, 75.0): 2, (0.0, 102.0): 3}
NFOLD = 10.0
NFOLD_VARIANTS = (6.0, 15.0)
IC_COSMIC = 3000.0
SUBBANDS = [(3000.0, 4500.0), (4500.0, 6000.0), (6000.0, 9000.0), (9000.0, np.inf)]
TANK_MU_PE = 1000.0
PRE_NS = (100.0, 900.0)
TAUS_FIX_US = (27.5, 29.4, 32.5)
KMAX = 5
PULSER_RUNS = (6254, 6256)
GATE = ls.GATE
CLASS_NAMES = [c[0] for c in ls.CLASSES] + [GATE]
TZ = ZoneInfo("America/Chicago")


# ---------------------------------------------------------------------------------------------- loading
def load(ctx: RunContext, src_runs: list, bkg_run: int, method: str) -> tuple:
    runs = src_runs + [bkg_run]
    cand = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_none"),
                           columns=["trigger", "method", "t_med", "pe", "n_hits", "run"],
                           filters=[("run", "in", runs), ("method", "in", [method])])
    tree = []
    for r in runs:
        cf = ls.tree_clusters(ctx, r)
        cf = cf[cf.clusterTime.between(*dz.WIN)]
        tree.append(pd.DataFrame(dict(trigger=cf.trigger, method=ls.TREE, t_med=cf.clusterTime, pe=cf.clusterPE,
                                      n_hits=cf.clusterHits, run=r)))
    cand = pd.concat([cand] + tree, ignore_index=True)
    T = {}
    for r in runs:
        g = ls.triggers(ctx, r, bkg_run, [method], cand)
        rt = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_triggers_{r}"), columns=["trigger", "nhits", "pe_sum"])
        cf = ls.tree_clusters(ctx, r)
        prompt = cf[cf.clusterTime < 2000.0].groupby("trigger").clusterPE.max()
        g = g.merge(rt, on="trigger", how="left")
        g["prompt_pe"] = g.trigger.map(prompt).fillna(0.0)
        g["tank_mu"] = g.prompt_pe > TANK_MU_PE
        g["tagged"] = g.is_cosmic.astype(bool)
        g["sp"] = g.second_pulse.astype(bool)
        g["n"] = g[f"n_{method}"]
        T[r] = g.sort_values("eventTankTime").reset_index(drop=True)
        print(f"[nosourcenorm] run {r}: {len(g):,} readouts")
    return T, cand


def class_of(ic: np.ndarray, sp: np.ndarray, g: float = 1.0) -> tuple:
    x = np.asarray(ic) * g
    cls = pd.cut(x, [c[1] for c in ls.CLASSES] + [np.inf], labels=[c[0] for c in ls.CLASSES], right=False).astype(str)
    gate = (x > 700.0) & (x < 1200.0) & ~np.asarray(sp, bool)
    return np.asarray(cls), gate


def counts(df: pd.DataFrame, g: float = 1.0) -> dict:
    cls, gate = class_of(df.IC_adjusted.to_numpy(), df.sp.to_numpy(), g)
    out = {c: int((cls == c).sum()) for c in CLASS_NAMES[:-1]}
    out[GATE] = int(gate.sum())
    return out


def scalers(runs_db: str) -> dict:
    sl = yaml.safe_load(Path(runs_db).read_text()).get("shift_log", {})
    return {int(r): (v["scaler_triggers"] / v["scaler_s"], np.sqrt(v["scaler_triggers"]) / v["scaler_s"], v["scaler_s"])
            for r, v in sl.items() if "scaler_triggers" in v}


# ---------------------------------------------------------------------------------------------- dead time
def _hess(f, p, h):
    p = np.asarray(p, float)
    n = len(p)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i] * h[i], np.eye(n)[j] * h[j]
            H[i, j] = H[j, i] = (f(p + e_i + e_j) - f(p + e_i - e_j) - f(p - e_i + e_j) + f(p - e_i - e_j)) / (4 * h[i] * h[j])
    return H


def _errors(f, x, steps, bounds) -> np.ndarray:
    x = np.asarray(x, float)
    free = np.array([not (lo is not None and abs(v - lo) < 1e-6) and not (hi is not None and abs(v - hi) < 1e-6)
                     for v, (lo, hi) in zip(x, bounds)])
    idx = np.where(free)[0]
    err = np.full(len(x), np.nan)
    for scale in (1.0, 3.0, 0.3, 10.0):
        def g(y):
            z = x.copy()
            z[idx] = y
            return f(z)
        try:
            H = _hess(g, x[idx], [steps[i] * scale for i in idx])
            c = np.diag(np.linalg.inv(H))
        except np.linalg.LinAlgError:
            continue
        if np.all(c[:min(len(c), 8)] > 0):
            err[idx] = np.sqrt(np.clip(c, 0, None))
            return err
    return err


def _emg(mu, s, lam):
    return stats.exponnorm(1.0 / (s * lam), loc=mu, scale=s)


def _nll_emg(x, lo, hi):
    def f(p):
        D = _emg(p[0], np.exp(p[1]), np.exp(p[2]))
        z = D.cdf(hi) - D.cdf(lo)
        return -(D.logpdf(x).sum() - len(x) * np.log(z))
    return f


def window(d: np.ndarray, nfold: float) -> tuple:
    q0 = float(np.quantile(d, 0.001))
    lam0 = np.log(2.0) / (float(np.median(d)) - q0)
    return q0 - 0.05, q0 + nfold / lam0


def emg_fit(d: np.ndarray, lo: float, hi: float) -> dict:
    x = d[(d >= lo) & (d < hi)]
    f = _nll_emg(x, lo, hi)
    q0 = float(np.quantile(d, 0.001))
    lam0 = 1.0 / max(float(np.mean(x)) - q0 - 0.015, 1e-3)
    r = optimize.minimize(f, [q0 + 0.015, np.log(0.01), np.log(lam0)], method="Nelder-Mead",
                          options=dict(maxiter=8000, xatol=1e-8, fatol=1e-6))
    p = r.x
    try:
        err = np.sqrt(np.diag(np.linalg.inv(_hess(f, p, [2e-4, 2e-2, 2e-3]))))
    except np.linalg.LinAlgError:
        err = np.full(3, np.nan)
    mu, s, lam = p[0], np.exp(p[1]), np.exp(p[2])
    D = _emg(mu, s, lam)
    edges = np.linspace(lo, hi, 101)
    obs = np.histogram(x, edges)[0]
    exp_ = len(x) * np.diff(D.cdf(edges)) / (D.cdf(hi) - D.cdf(lo))
    o, e = _merge(obs, exp_)
    chi2 = float(((o - e) ** 2 / e).sum())
    ndf = len(o) - 3
    return dict(mu=mu, sigma=s, lam=lam, mu_err=err[0], sigma_err=s * err[1], lam_err=lam * err[2], n_win=len(x),
                lo=lo, hi=hi, chi2=chi2, ndf=ndf, p=float(stats.chi2.sf(chi2, ndf)), ok=bool(r.success))


def _merge(o, e, emin=5.0):
    oo, ee, co, ce = [], [], 0.0, 0.0
    for a, b in zip(o, e):
        co, ce = co + a, ce + b
        if ce >= emin:
            oo.append(co)
            ee.append(ce)
            co = ce = 0.0
    if ce > 0 and ee:
        oo[-1] += co
        ee[-1] += ce
    return np.array(oo), np.array(ee)


def mu_given(x: np.ndarray, lo: float, hi: float, s: float, lam: float) -> tuple:
    f = lambda m: _nll_emg(x, lo, hi)([m[0], np.log(s), np.log(lam)])  # noqa: E731
    r = optimize.minimize(f, [float(np.mean(x)) - 1.0 / lam], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-6))
    h = 2e-4
    c = (f(r.x + h) - 2 * f(r.x) + f(r.x - h)) / h ** 2
    return float(r.x[0]), float(1.0 / np.sqrt(c)) if c > 0 else np.nan


def intervals(df: pd.DataFrame, gap_s: float) -> pd.DataFrame:
    t = df.eventTankTime.to_numpy(np.float64) / 1e9
    d = np.append(np.diff(t), np.nan)
    out = df.assign(t_s=t, dt=d)
    return out[(out.dt > 0) & (out.dt < gap_s)]


def pulser(run: int, gap_s: float) -> dict:
    w = pd.read_parquet(f"TriggerSummary/WaveformFeatures_AmBe2.0v4_special_box_{run}.parquet", columns=["timestamp"])
    t = np.sort(w.timestamp[w.timestamp > 0].unique()) / 1e9
    d = np.diff(t)
    d = d[(d > 0) & (d < gap_s)]
    n2 = int((np.abs(d - 2.0) < 0.002).sum())
    n4 = int((np.abs(d - 4.0) < 0.002).sum())
    p = n4 / (n2 + n4)
    return dict(run=run, kind="pulser", n_intervals=len(d), n_2s=n2, n_4s=n4, n_other=len(d) - n2 - n4,
                p_tau_gt_2s=p, p_tau_gt_2s_err=np.sqrt(p * (1 - p) / (n2 + n4)),
                period_s=float(np.median(d[np.abs(d - 2.0) < 0.002])), dt_min_s=float(d.min()))


def deadtime(T: dict, bkg_run: int, gap_s: float, sc: dict) -> tuple:
    rows, grp = [], []
    for r, df in T.items():
        iv = intervals(df, gap_s)
        d = iv.dt.to_numpy()
        lo, hi = window(d, NFOLD)
        f = emg_fit(d, lo, hi)
        n = len(d)
        q = float(np.quantile(d, ls.TAU_Q))
        stall = d >= hi
        lq = float(np.clip(d - q, 0, None).sum())
        row = dict(run=r, kind="no source" if r == bkg_run else "source", n_intervals=n, tau_q001=q,
                   live_s_quantile=lq, lam_quantile=n / lq, **f,
                   live_s=n / f["lam"], live_s_err=n * f["lam_err"] / f["lam"] ** 2,
                   n_stall=int(stall.sum()), stall_s=float((d[stall] - f["mu"]).sum()),
                   live_s_quantile_minus_stall=float(np.clip(d[~stall] - q, 0, None).sum()))
        for nf in NFOLD_VARIANTS:
            v = emg_fit(d, *window(d, nf))
            row[f"lam_nfold{nf:g}"] = v["lam"]
            row[f"mu_nfold{nf:g}"] = v["mu"]
        x = d[(d >= lo) & (d < hi)]
        row["lam_shiftexp"] = 1.0 / (float(x.mean()) - float(x.min()))
        if r in sc:
            row["lam_scaler"], row["lam_scaler_err"], row["scaler_s"] = sc[r]
            row["lam_over_scaler"] = f["lam"] / sc[r][0]
            row["live_s_scaler"] = n / sc[r][0]
        rows.append(row)
        win = iv[(iv.dt >= lo) & (iv.dt < hi)].copy()
        cls, _ = class_of(win.IC_adjusted.to_numpy(), win.sp.to_numpy())
        win["cls"] = cls
        win["nhits_tercile"] = pd.qcut(win.nhits.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
        for feat in ("cls", "tagged", "sp", "nhits_tercile"):
            for key, sub in win.groupby(feat):
                if len(sub) < 50:
                    continue
                m, e = mu_given(sub.dt.to_numpy(), lo, hi, f["sigma"], f["lam"])
                grp.append(dict(run=r, feature=feat, group=str(key), n=len(sub), mu=m, mu_err=e,
                                nhits_median=float(sub.nhits.median())))
        print(f"[nosourcenorm] deadtime {r}: mu {f['mu']:.4f} s, sigma {1e3 * f['sigma']:.1f} ms, "
              f"lambda {f['lam']:.3f} Hz, live {row['live_s']:.0f} s (quantile {row['live_s_quantile']:.0f} s)")
    for r in PULSER_RUNS:
        rows.append(pulser(r, gap_s))
    return pd.DataFrame(rows), pd.DataFrame(grp)


def page_deadtime(out, T: dict, dt: pd.DataFrame, grp: pd.DataFrame, bkg_run: int, gap_s: float):
    import matplotlib.pyplot as plt
    runs = list(T)
    fig, ax = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6))
    for a, r in zip(np.atleast_1d(ax), runs):
        d = intervals(T[r], gap_s).dt.to_numpy()
        f = dt[dt.run == r].iloc[0]
        span = min(f.hi, f.mu + 6.0 / f.lam)
        edges = np.linspace(f.lo, span, 121)
        h = np.histogram(d, edges)[0]
        c = 0.5 * (edges[1:] + edges[:-1])
        D = _emg(f.mu, f.sigma, f.lam)
        e = f.n_win * np.diff(D.cdf(edges)) / (D.cdf(f.hi) - D.cdf(f.lo))
        a.errorbar(c, h, np.sqrt(h), fmt="o", ms=3, color="k", label="data")
        a.plot(c, e, color="#1f5fbf", lw=2, label="fit")
        a.set_yscale("log")
        a.set_ylim(0.5, None)
        a.set_xlabel("time to next readout [s]")
        a.set_title(f"{r}: dead time {f.mu:.4f} s (0.1 % quantile {f.tau_q001:.4f}), jitter {1e3 * f.sigma:.1f} ms, "
                    f"rate {f.lam:.2f} Hz", fontsize=9)
    np.atleast_1d(ax)[0].set_ylabel("intervals per bin")
    np.atleast_1d(ax)[0].legend(fontsize=8)
    two_titles(fig, "Time between readouts: dead time plus waiting time",
               "fit dt = tau + Exp(lambda), tau ~ Gaussian(mu, sigma), truncated likelihood; the 0.1 % quantile "
               "is the tau used before")
    out.save(fig, "deadtime_spectra")
    rr = dt[dt.kind != "pulser"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(rr))
    ax.plot(x, rr.lam_quantile, "s",
            color="0.5", ms=9, label="0.1 % quantile tau")
    ax.errorbar(x, rr.lam, rr.lam_err, fmt="o", color="#1f5fbf", ms=9, label="dead-time fit")
    if "lam_scaler" in rr:
        m = rr.lam_scaler.notna()
        ax.errorbar(x[m.to_numpy()], rr.lam_scaler[m], rr.lam_scaler_err[m], fmt="D", color="#e07b00", ms=9,
                    label="run-DB scaler")
    ax.set_xticks(x, [str(r) for r in rr.run])
    ax.set_yscale("log")
    ax.set_ylabel("presented trigger rate [Hz]")
    ax.legend(fontsize=9)
    two_titles(fig, "True trigger rate from three methods",
               "lambda = N / live time; scaler = shift-log count (6264: 300 s snapshot after the double-pulse veto; "
               "6265: whole run)")
    out.save(fig, "deadtime_lambda")
    if len(grp):
        g = grp[grp.feature == "cls"]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for i, (r, q) in enumerate(g.groupby("run")):
            q = q.set_index("group").reindex([c for c in CLASS_NAMES[:-1] if c in set(q.group)])
            ax.errorbar(np.arange(len(q)), 1e3 * (q.mu - q.mu.mean()), 1e3 * q.mu_err, fmt=ls.MARKERS[i % 4] + "-",
                        color=ls.COLORS[i % 4], ms=7, label=str(r))
            ax.set_xticks(np.arange(len(q)), q.index, rotation=20)
        ax.set_ylabel("mean dead time minus run mean [ms]")
        ax.set_xlabel("IC class of the readout before the interval")
        ax.legend(fontsize=8)
        two_titles(fig, "Does the dead time depend on what was read out?",
                   "mu fitted per group with sigma and lambda fixed to the run fit")
        out.save(fig, "deadtime_by_class")


# ---------------------------------------------------------------------------------------------- stability
def _peak44(ic: np.ndarray) -> float:
    x = ic[(ic > 400) & (ic < 1600)]
    if len(x) < 200:
        return np.nan
    h, e = np.histogram(x, np.arange(400, 1601, 25))
    c = 0.5 * (e[1:] + e[:-1])
    m = c[np.argmax(np.convolve(h, np.ones(3) / 3, "same"))]
    y = x[np.abs(x - m) < 200]
    try:
        p, _ = optimize.curve_fit(lambda z, a, mu, s: a * np.exp(-0.5 * ((z - mu) / s) ** 2),
                                  c[np.abs(c - m) < 200], h[np.abs(c - m) < 200], p0=[h.max(), m, 150])
        return float(p[1])
    except RuntimeError:
        return float(np.median(y))


def stability(T: dict, dt: pd.DataFrame, bkg_run: int, gap_s: float, block_h: float) -> tuple:
    rows, summ = [], []
    for r, df in T.items():
        f = dt[dt.run == r].iloc[0]
        iv = intervals(df, gap_s)
        t0 = iv.t_s.min()
        iv = iv.assign(block=((iv.t_s - t0) // (3600.0 * block_h)).astype(int))
        cls, gate = class_of(iv.IC_adjusted.to_numpy(), iv.sp.to_numpy())
        iv["cls"], iv["gate"] = cls, gate
        for b, q in iv.groupby("block"):
            w = q[(q.dt >= f.lo) & (q.dt < f.hi)]
            if len(w) < 20:
                continue
            lam = 1.0 / (w.dt.mean() - f.mu)
            live = len(q) / lam
            tc = datetime.fromtimestamp(float(q.t_s.median()), tz=timezone.utc)
            row = dict(run=r, block=int(b), t_mid=tc.isoformat(), local_hour=tc.astimezone(TZ).hour, n=len(q),
                       lam=lam, lam_err=lam / np.sqrt(len(w)), live_s=live)
            sel = {"cosmic": (q.cls == "IC>3000") & q.tagged, "gate": q.gate,
                   "icneg_untagged": (q.cls == "IC<0") & ~q.tagged, "ic0300_untagged": (q.cls == "0-300") & ~q.tagged}
            for k, m in sel.items():
                row[f"n_{k}"] = int(m.sum())
                row[f"rate_{k}"] = m.sum() / live
                row[f"rate_{k}_err"] = np.sqrt(m.sum()) / live
            row["peak44"] = _peak44(q.IC_adjusted.to_numpy()) if r != bkg_run else np.nan
            cz = q.IC_adjusted[(q.cls == "IC>3000") & q.tagged]
            row["cosmic_median"] = float(cz.median()) if len(cz) >= 10 else np.nan
            rows.append(row)
        s = pd.DataFrame([x for x in rows if x["run"] == r])
        for k in ("lam", "rate_cosmic", "rate_gate", "rate_icneg_untagged", "rate_ic0300_untagged"):
            e = s[f"{k}_err"] if k != "lam" else s.lam_err
            ok = e > 0
            w = 1 / e[ok] ** 2
            m = float((s[k][ok] * w).sum() / w.sum())
            chi2 = float((((s[k][ok] - m) / e[ok]) ** 2).sum())
            summ.append(dict(run=r, quantity=k, mean=m, chi2=chi2, ndf=int(ok.sum()) - 1,
                             p=float(stats.chi2.sf(chi2, int(ok.sum()) - 1)), min=float(s[k].min()), max=float(s[k].max())))
        for k in ("peak44", "cosmic_median"):
            v = s[k].dropna()
            if len(v):
                summ.append(dict(run=r, quantity=k, mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                                 rel_spread=float(v.std() / v.mean())))
    return pd.DataFrame(rows), pd.DataFrame(summ)


def page_stability(out, st: pd.DataFrame, bkg_run: int, block_h: float):
    import matplotlib.pyplot as plt
    runs = list(st.run.unique())
    panels = [("lam", "presented rate λ [Hz]"), ("rate_cosmic", "cosmic IC > 3000 per live s [Hz]"),
              ("rate_icneg_untagged", "untagged IC < 0 per live s [Hz]"), ("cosmic_median", "cosmic IC median"),
              ("peak44", "4.4 MeV peak IC")]
    fig, ax = plt.subplots(len(panels), len(runs), figsize=(4.6 * len(runs), 2.6 * len(panels)), squeeze=False)
    for j, r in enumerate(runs):
        s = st[st.run == r]
        hrs = s.block * block_h
        for i, (k, lab) in enumerate(panels):
            a = ax[i, j]
            if s[k].notna().any():
                e = s[f"{k}_err"] if f"{k}_err" in s else (s.lam_err if k == "lam" else None)
                a.errorbar(hrs, s[k], e, fmt="o-", ms=4, color=ls.COLORS[j % 4])
            else:
                a.axis("off")
            if j == 0:
                a.set_ylabel(lab, fontsize=9)
            if i == 0:
                a.set_title(f"run {r}", fontsize=10)
            if i == len(panels) - 1:
                a.set_xlabel("hours into run")
    two_titles(fig, "Is each run stable in time?",
               f"{block_h:g} h blocks; live time per block = N / λ_block, λ_block from the mean interval minus the run's "
               "fitted dead time")
    out.save(fig, "stability")


# ---------------------------------------------------------------------------------------------- right sideband
def gain_fit(src: np.ndarray, bkg: np.ndarray, lo: float = 4000.0) -> dict:
    hi = float(min(np.quantile(src, 0.995), np.quantile(bkg, 0.995)))
    edges = np.geomspace(lo, hi, 21)
    hs = np.histogram(src, edges)[0].astype(float)
    grid = np.arange(0.85, 1.20, 0.0025)
    chi = []
    for g in grid:
        hb = np.histogram(bkg * g, edges)[0].astype(float)
        a, b = hs / hs.sum(), hb / hb.sum()
        v = hs / hs.sum() ** 2 + hb / hb.sum() ** 2
        m = v > 0
        chi.append(float(((a - b)[m] ** 2 / v[m]).sum()))
    chi = np.array(chi)
    i = int(np.argmin(chi))
    sl = slice(max(i - 8, 0), min(i + 9, len(grid)))
    c = np.polyfit(grid[sl], chi[sl], 2)
    g = -c[1] / (2 * c[0]) if c[0] > 0 else grid[i]
    err = 1.0 / np.sqrt(c[0]) if c[0] > 0 else np.nan
    hb = np.histogram(bkg, edges)[0].astype(float)
    a, b = hs / hs.sum(), hb / hb.sum()
    v = hs / hs.sum() ** 2 + hb / hb.sum() ** 2
    chi1 = float(((a - b)[v > 0] ** 2 / v[v > 0]).sum())
    return dict(g=float(g), g_err=float(err), chi2_min=float(chi.min()), chi2_g1=chi1, ndf=int((v > 0).sum()) - 2,
                median_ratio=float(np.median(src) / np.median(bkg)), lo=lo, hi=hi)


def rsb(T: dict, dt: pd.DataFrame, src_runs: list, bkg_run: int) -> tuple:
    B = T[bkg_run]
    Tl = dt.set_index("run").live_s
    bc = B[B.tagged & (B.IC_adjusted > 0)]
    fac, ext = [], []
    for r in src_runs:
        S = T[r]
        sc = S[S.tagged & (S.IC_adjusted > IC_COSMIC)]
        gf = gain_fit(sc.IC_adjusted.to_numpy(), bc[bc.IC_adjusted > IC_COSMIC * 0.8].IC_adjusted.to_numpy())
        g, ge = gf["g"], gf["g_err"]
        ns = len(sc)
        nb = {gg: int((bc.IC_adjusted * gg > IC_COSMIC).sum()) for gg in (1.0, g, g - ge, g + ge)}
        F = {gg: ns / v for gg, v in nb.items()}
        f_live = Tl[r] / Tl[bkg_run]
        row = dict(run=r, n_src_cosmic=ns, n_bkg_cosmic=nb[1.0], n_bkg_cosmic_g=nb[g], **gf,
                   F_rsb=F[1.0], F_rsb_err=F[1.0] * np.sqrt(1 / ns + 1 / nb[1.0]), F_rsb_g=F[g],
                   F_rsb_g_err=F[g] * np.sqrt(1 / ns + 1 / nb[g]), F_rsb_gain_err=abs(F[g + ge] - F[g - ge]) / 2,
                   F_live=f_live, F_live_err=f_live * np.hypot(dt.set_index("run").live_s_err[r] / Tl[r],
                                                               dt.set_index("run").live_s_err[bkg_run] / Tl[bkg_run]))
        row["k"] = row["F_rsb"] / f_live
        row["k_g"] = row["F_rsb_g"] / f_live
        row["k_g_err"] = row["k_g"] * np.hypot(row["F_rsb_g_err"] / row["F_rsb_g"], row["F_live_err"] / f_live)
        sub = []
        for lo_, hi_ in SUBBANDS:
            a = int(((sc.IC_adjusted >= lo_) & (sc.IC_adjusted < hi_)).sum())
            b = int(((bc.IC_adjusted * g >= lo_) & (bc.IC_adjusted * g < hi_)).sum())
            fb = a / b if b else np.nan
            sub.append(fb)
            row[f"F_band_{lo_:g}"] = fb
            row[f"F_band_{lo_:g}_err"] = fb * np.sqrt(1 / max(a, 1) + 1 / max(b, 1))
        sub = np.array(sub)
        row["F_band_rel_spread"] = float(np.nanstd(sub) / np.nanmean(sub))
        tm_s, tm_b = int(S.tank_mu.sum()), int(B.tank_mu.sum())
        row.update(n_src_tankmu=tm_s, n_bkg_tankmu=tm_b, F_tankmu=tm_s / tm_b,
                   F_tankmu_err=tm_s / tm_b * np.sqrt(1 / tm_s + 1 / tm_b),
                   frac_cosmic_with_tankmu_src=float(sc.tank_mu.mean()),
                   frac_cosmic_with_tankmu_bkg=float(bc[bc.IC_adjusted > IC_COSMIC].tank_mu.mean()))
        row["k_tankmu"] = row["F_tankmu"] / f_live
        ds = dt.set_index("run")
        if "live_s_scaler" in ds and pd.notna(ds.live_s_scaler.get(r)):
            fs = ds.live_s_scaler[r] / Tl[bkg_run]
            row["F_scaler"] = fs
            row["k_scaler"] = row["F_rsb_g"] / fs
        fac.append(row)
        Fn, Fe = row["F_rsb_g"], row["F_rsb_g_err"]
        for veto in ("noveto", "veto"):
            s_ = S if veto == "noveto" else S[~S.tagged]
            b_ = B if veto == "noveto" else B[~B.tagged]
            cs, cb = counts(s_), counts(b_, g)
            cbp, cbm = counts(b_, g + ge), counts(b_, g - ge)
            for c in CLASS_NAMES:
                pred = Fn * cb[c]
                stat = pred * np.sqrt(1 / max(cb[c], 1) + (Fe / Fn) ** 2) if cb[c] else Fn * 1.0
                gain = abs(Fn * cbp[c] - Fn * cbm[c]) / 2
                ncl = pred * row["F_band_rel_spread"]
                tot = np.sqrt(stat ** 2 + gain ** 2 + ncl ** 2)
                ext.append(dict(run=r, cls=c, veto=veto, n_src=cs[c], n_bkg=counts(b_)[c], n_bkg_g=cb[c], F=Fn,
                                pred=pred, err_stat=stat, err_gain=gain, err_nonclosure=ncl, err_tot=tot,
                                share=pred / cs[c] if cs[c] else np.nan,
                                share_err=tot / cs[c] if cs[c] else np.nan,
                                pred_live=f_live * cb[c], pred_tankmu=row["F_tankmu"] * cb[c]))
        print(f"[nosourcenorm] rsb {r}: g {g:.4f} ± {ge:.4f}, F {row['F_rsb']:.4f} -> {Fn:.4f}, "
              f"k {row['k']:.3f} -> {row['k_g']:.3f}, sub-band spread {row['F_band_rel_spread']:.3f}, "
              f"F_tankmu {row['F_tankmu']:.4f}")
    return pd.DataFrame(fac), pd.DataFrame(ext)


def page_rsb(out, T: dict, fac: pd.DataFrame, ext: pd.DataFrame, bkg_run: int):
    import matplotlib.pyplot as plt
    B = T[bkg_run]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    edges = np.geomspace(3000, 20000, 31)
    c = np.sqrt(edges[1:] * edges[:-1])
    hb = np.histogram(B.IC_adjusted[B.tagged], edges)[0]
    ax[0].errorbar(c, hb / hb.sum(), np.sqrt(hb) / hb.sum(), fmt="o-", ms=4, color="0.45", label=f"no source ({bkg_run})")
    for i, f in fac.reset_index(drop=True).iterrows():
        S = T[f.run]
        hs = np.histogram(S.IC_adjusted[S.tagged], edges)[0]
        ax[0].errorbar(c, hs / hs.sum(), np.sqrt(hs) / hs.sum(), fmt=ls.MARKERS[i] + "-", ms=4, color=ls.COLORS[i],
                       label=f"{int(f.run)}, gain {f.g:.3f}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("BGO IC, cosmic-tagged")
    ax[0].set_ylabel("fraction per bin")
    ax[0].legend(fontsize=8)
    ax[0].set_title("cosmic IC shape", fontsize=10)
    for i, f in fac.reset_index(drop=True).iterrows():
        x = np.arange(len(SUBBANDS))
        y = [f[f"F_band_{lo:g}"] for lo, _ in SUBBANDS]
        e = [f[f"F_band_{lo:g}_err"] for lo, _ in SUBBANDS]
        ax[1].errorbar(x + 0.08 * i, y, e, fmt=ls.MARKERS[i], ms=7, color=ls.COLORS[i], label=str(int(f.run)))
    ax[1].set_xticks(np.arange(len(SUBBANDS)), [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}" for lo, hi in SUBBANDS])
    ax[1].set_xlabel("IC band (no-source IC × gain)")
    ax[1].set_ylabel("source / no-source count ratio F")
    ax[1].legend(fontsize=8)
    ax[1].set_title("factor in sub-bands", fontsize=10)
    q = ext[ext.veto == "noveto"]
    names = CLASS_NAMES
    for i, (r, g) in enumerate(q.groupby("run")):
        g = g.set_index("cls").loc[names]
        ax[2].errorbar(np.arange(len(names)) + 0.08 * i, 100 * g.share, 100 * g.share_err, fmt=ls.MARKERS[i], ms=7,
                       color=ls.COLORS[i], label=str(r))
    ax[2].set_xticks(np.arange(len(names)), ["IC<0", "0-300", "300-700", "700-1200", "1200-3000", ">3000", "gate"],
                     rotation=20)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("no-source share of the class [%]")
    ax[2].legend(fontsize=8)
    ax[2].set_title("extrapolated background, no veto", fontsize=10)
    two_titles(fig, "Right-sideband normalisation",
               "F = source / no-source count of cosmic-tagged IC > 3000 after matching the BGO gain on the cosmic IC "
               "shape; errors: statistics, gain, sub-band spread")
    out.save(fig, "rsb")


# ---------------------------------------------------------------------------------------------- left sideband / P0
def leftsb(T: dict, dt: pd.DataFrame, fac: pd.DataFrame, src_runs: list, bkg_run: int) -> pd.DataFrame:
    B = T[bkg_run]
    Tl = dt.set_index("run").live_s
    rows = []
    for r in src_runs:
        S = T[r]
        f = fac.set_index("run").loc[r]
        g, F = f.g, f.F_rsb_g
        Fl = Tl[r] / Tl[bkg_run]
        ct, cu = counts(B[B.tagged], g), counts(B[~B.tagged], g)
        cs, csv = counts(S), counts(S[~S.tagged])
        Fleft = (cs["IC<0"] - F * ct["IC<0"]) / cu["IC<0"]
        p0s, p0b = _p0(S[~S.tagged]), _p0(B[~B.tagged], g)
        p0b_all = float((B[~B.tagged].n == 0).mean())
        p0_gate = p0s[GATE][0]
        for c in CLASS_NAMES:
            one = F * (ct[c] + cu[c])
            two = F * ct[c] + Fl * cu[c]
            lft = F * ct[c] + Fleft * cu[c]
            fv = Fl * cu[c] / csv[c] if csv[c] else np.nan
            fvl = Fleft * cu[c] / csv[c] if csv[c] else np.nan
            ps, ns_ = p0s[c]
            pb = p0b[c][0] if p0b[c][1] >= 20 else p0b_all
            imp = (ps - p0_gate) / (pb - p0_gate) if c != GATE and pb != p0_gate else np.nan
            rows.append(dict(run=r, cls=c, n_src=cs[c], n_src_veto=csv[c], n_bkg_tagged=ct[c], n_bkg_untagged=cu[c],
                             F_rsb=F, F_live=Fl, F_left=Fleft, pred_one=one, pred_two=two, pred_left=lft,
                             obs_over_one=cs[c] / one if one else np.nan, obs_over_two=cs[c] / two if two else np.nan,
                             obs_over_one_err=cs[c] / one * np.sqrt(1 / max(cs[c], 1) + 1 / max(ct[c] + cu[c], 1)) if one else np.nan,
                             obs_over_two_err=cs[c] / two * np.sqrt(1 / max(cs[c], 1) + 1 / max(ct[c] + cu[c], 1)) if two else np.nan,
                             f_fs_veto_live=fv, f_fs_veto_left=fvl, p0_src=ps, n_p0_src=ns_, p0_bkg=pb,
                             n_p0_bkg=p0b[c][1], f_fs_from_p0=imp,
                             p0_corrected=(ps - fv * pb) / (1 - fv) if fv == fv and fv < 1 else np.nan,
                             dp0_from_fs=fv * (pb - ps) if fv == fv else np.nan))
        print(f"[nosourcenorm] leftsb {r}: F_left {Fleft:.4f} (F_rsb {F:.4f}, F_live {Fl:.4f})")
    return pd.DataFrame(rows)


def _p0(df: pd.DataFrame, g: float = 1.0) -> dict:
    cls, gate = class_of(df.IC_adjusted.to_numpy(), df.sp.to_numpy(), g)
    z = (df.n == 0).to_numpy()
    out = {c: (float(z[cls == c].mean()) if (cls == c).any() else np.nan, int((cls == c).sum())) for c in CLASS_NAMES[:-1]}
    out[GATE] = (float(z[gate].mean()), int(gate.sum()))
    return out


def page_leftsb(out, lf: pd.DataFrame):
    import matplotlib.pyplot as plt
    names = CLASS_NAMES
    lab = ["IC<0", "0-300", "300-700", "700-1200", "1200-3000", ">3000", "gate"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, (r, g) in enumerate(lf.groupby("run")):
        g = g.set_index("cls").loc[names]
        x = np.arange(len(names)) + 0.1 * i
        ax[0].errorbar(x, g.obs_over_one, g.obs_over_one_err, fmt=ls.MARKERS[i], mfc="none", ms=7, color=ls.COLORS[i])
        ax[0].errorbar(x + 0.04, g.obs_over_two, g.obs_over_two_err, fmt=ls.MARKERS[i], ms=7, color=ls.COLORS[i],
                       label=str(r))
        ok = (g.n_src_veto >= 20).to_numpy()
        ax[1].plot(x[ok], g.p0_src[ok], ls.MARKERS[i] + "-", ms=7, color=ls.COLORS[i], label=f"{r} observed")
        ax[1].plot(x[ok], g.p0_corrected[ok], ls.MARKERS[i] + ":", ms=5, mfc="none", color=ls.COLORS[i],
                   label=f"{r} minus false starts")
    for a in ax:
        a.set_xticks(np.arange(len(names)), lab, rotation=20)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("observed / predicted no-source-like triggers")
    ax[0].legend(fontsize=8)
    ax[0].set_title("open: one factor; filled: cosmic by F, rest by live time", fontsize=10)
    ax[1].set_ylabel("P(no neutron cluster), vetoed triggers")
    ax[1].legend(fontsize=7)
    ax[1].set_title("tank P0 per class", fontsize=10)
    two_titles(fig, "Left sideband: what the no-source run can and cannot explain",
               "classes in source-run IC; no-source counts after the BGO gain match; P0 corrected with the "
               "live-time false-start share after the veto")
    out.save(fig, "leftsb")


# ---------------------------------------------------------------------------------------------- in situ
def _veto_dummy(dummy: RunContext, run: int) -> pd.Series:
    from ..data.selection import CutCriteria
    c = CutCriteria()
    b = f"{dummy.run_name}__random_%s_{run}"
    tr = pd.read_parquet(dummy.parquet_path(b % "triggers"), columns=["trigger"]).trigger
    cf = pd.read_parquet(dummy.parquet_path(b % "cfclusters"), columns=["trigger", "clusterTime", "clusterPE"])
    vet = set(cf[(cf.clusterTime < c.cosmic_ct_threshold) | (cf.clusterPE > c.cosmic_pe_threshold)].trigger)
    return tr[~tr.isin(vet)]


def _expfit(t: np.ndarray, y: np.ndarray, n: np.ndarray, tau_fixed: Optional[float] = None) -> dict:
    scale = max(float(y.sum() / n.sum()), 1e-12)

    def nll(p):
        a, tau, cc = p[0] * scale, (tau_fixed if tau_fixed else p[1]), p[-1] * scale
        mu = n * (a * np.exp(-t / tau) + cc)
        mu = np.clip(mu, 1e-300, None)
        return float((mu - y * np.log(mu)).sum())
    p0 = [1.0] + ([28.0] if not tau_fixed else []) + [0.5]
    bnds = [(0.0, 1e3)] + ([(5.0, 200.0)] if not tau_fixed else []) + [(0.0, 1e3)]
    r = optimize.minimize(nll, p0, method="L-BFGS-B", bounds=bnds)
    r = optimize.minimize(nll, r.x, method="L-BFGS-B", bounds=bnds, options=dict(ftol=1e-14, gtol=1e-10))
    try:
        h = [1e-4 * max(abs(v), 1e-2) for v in r.x]
        err = np.sqrt(np.diag(np.linalg.inv(_hess(nll, r.x, h))))
    except (np.linalg.LinAlgError, ValueError):
        err = np.full(len(r.x), np.nan)
    if not np.isfinite(err[-1]):
        err[-1] = np.sqrt(max(y[-len(y) // 3:].sum(), 1.0)) / n[-len(y) // 3:].sum() / scale
    return dict(A=r.x[0] * scale, tau=tau_fixed if tau_fixed else r.x[1],
                tau_err=np.nan if tau_fixed else err[1], C=r.x[-1] * scale, C_err=err[-1] * scale)


def pairs(T: dict, cand: pd.DataFrame, src_runs: list, bkg_run: int, method: str, seed: int = 6264) -> pd.DataFrame:
    """Time difference of the two clusters in vetoed gate triggers with exactly two clusters, against one capture-like
    time (from one-cluster triggers) paired with one accidental time (no-source cluster times)."""
    rng = np.random.default_rng(seed)
    q = cand[cand.method == method]
    q = q[q.t_med.between(*dz.WIN)]
    B = T[bkg_run]
    acc = q[(q.run == bkg_run) & q.trigger.isin(B.trigger[~B.tagged])].t_med.to_numpy() / 1000.0
    cuts = (0.5, 1.0, 2.0, 5.0)
    rows = []
    for r in src_runs:
        S = T[r]
        keep = S.trigger[S.gate & ~S.tagged]
        c = q[(q.run == r) & q.trigger.isin(keep)]
        n = c.groupby("trigger").size()
        one = c[c.trigger.isin(n.index[n == 1])].t_med.to_numpy() / 1000.0
        two = c[c.trigger.isin(n.index[n == 2])].sort_values(["trigger", "t_med"])
        tt = two.t_med.to_numpy().reshape(-1, 2) / 1000.0
        d = np.abs(tt[:, 1] - tt[:, 0])
        ref_ca = np.abs(rng.choice(one, 200000) - rng.choice(acc, 200000))
        ref_cc = np.abs(rng.choice(one, 200000) - rng.choice(one, 200000))
        row = dict(run=r, n_two=len(d), n_one=len(one), median_dt=float(np.median(d)),
                   median_dt_capture_acc=float(np.median(ref_ca)), median_dt_capture_capture=float(np.median(ref_cc)))
        for x in cuts:
            row[f"frac_lt{x:g}"] = float((d < x).mean())
            row[f"frac_lt{x:g}_err"] = float(np.sqrt((d < x).mean() * (1 - (d < x).mean()) / len(d)))
            row[f"ref_capture_acc_lt{x:g}"] = float((ref_ca < x).mean())
            row[f"ref_capture_capture_lt{x:g}"] = float((ref_cc < x).mean())
        rows.append(row)
        print(f"[nosourcenorm] pairs {r}: {len(d)} two-cluster triggers, median dt {row['median_dt']:.1f} us "
              f"(capture+accidental {row['median_dt_capture_acc']:.1f}, capture+capture {row['median_dt_capture_capture']:.1f}), "
              f"dt<2 us {row['frac_lt2']:.3f} vs {row['ref_capture_acc_lt2']:.3f}")
    return pd.DataFrame(rows)


def insitu(ctx: RunContext, dummy: Optional[RunContext], T: dict, cand: pd.DataFrame, src_runs: list, bkg_run: int,
           method: str) -> tuple:
    samples = [(r, "source, gate", ctx, T[r].trigger[T[r].gate & ~T[r].tagged]) for r in src_runs]
    samples.append((bkg_run, "no source", ctx, T[bkg_run].trigger[~T[bkg_run].tagged]))
    if dummy is not None:
        samples += [(r, "dummy", dummy, _veto_dummy(dummy, r)) for r in PULSER_RUNS]
    rows, prof = [], []
    tb = np.arange(0.0, 70.0 + 1.0, 1.0)
    tc_edges = np.arange(2.0, 67.0 + 2.5, 2.5)
    for run, kind, cx, trig in samples:
        keep = set(trig)
        n = len(keep)
        h = pd.read_parquet(cx.parquet_path(f"{cx.run_name}__random_hits_{run}"), columns=["trigger", "t"])
        h = h[h.trigger.isin(keep)]
        t_us = h.t.to_numpy() / 1000.0
        pre = int(((h.t >= PRE_NS[0]) & (h.t < PRE_NS[1])).sum())
        cnt = np.histogram(t_us, tb)[0]
        prof.append(dict(run=run, kind=kind, n_trig=n, **{f"t{int(a)}": v / n for a, v in zip(tb[:-1], cnt)}))
        c = 0.5 * (tb[1:] + tb[:-1])
        m = (c >= 10.0) & (c < 67.0)
        hf = _expfit(c[m], cnt[m].astype(float), np.full(m.sum(), float(n)))
        if run in PULSER_RUNS:
            rc = pd.read_parquet(cx.parquet_path(f"{cx.run_name}__random_recoclusters_{run}__reclust"),
                                 columns=["trigger", "method", "t_med"])
            tt = rc[(rc.method == method) & rc.trigger.isin(keep)].t_med.to_numpy()
        else:
            q = cand[(cand.run == run) & (cand.method == method)]
            tt = q[q.trigger.isin(keep)].t_med.to_numpy()
        tt = tt[(tt >= dz.WIN[0]) & (tt <= dz.WIN[1])] / 1000.0
        ccnt = np.histogram(tt, tc_edges)[0].astype(float)
        cc = 0.5 * (tc_edges[1:] + tc_edges[:-1])
        nn = np.full(len(cc), float(n) * 2.5)
        cf = _expfit(cc, ccnt, nn)
        win_us = (dz.WIN[1] - dz.WIN[0]) / 1000.0
        row = dict(run=run, kind=kind, n_trig=n, pre_hits_per_us=pre / n / ((PRE_NS[1] - PRE_NS[0]) / 1000.0),
                   pre_hits_per_us_err=np.sqrt(pre) / n / ((PRE_NS[1] - PRE_NS[0]) / 1000.0),
                   hits_flat_per_us=hf["C"], hits_flat_per_us_err=hf["C_err"], hits_tau_us=hf["tau"],
                   hits_late_per_us=float(cnt[60:67].sum()) / n / 7.0,
                   clusters_per_trig=len(tt) / n, clus_flat_per_trig_taufree=cf["C"] * win_us,
                   clus_flat_per_trig_taufree_err=cf["C_err"] * win_us, clus_tau_us=cf["tau"], clus_tau_err=cf["tau_err"])
        for tf in TAUS_FIX_US:
            cfx = _expfit(cc, ccnt, nn, tau_fixed=tf)
            row[f"clus_flat_per_trig_tau{tf:g}"] = cfx["C"] * win_us
            row[f"clus_flat_per_trig_tau{tf:g}_err"] = cfx["C_err"] * win_us
        row["clus_flat_per_trig"] = row[f"clus_flat_per_trig_tau{ls.TAU_CAP_US:g}"]
        row["clus_flat_per_trig_err"] = row[f"clus_flat_per_trig_tau{ls.TAU_CAP_US:g}_err"]
        rows.append(row)
        print(f"[nosourcenorm] insitu {run} {kind}: {n:,} triggers, pre-prompt {rows[-1]['pre_hits_per_us']:.3f} hits/us, "
              f"flat {rows[-1]['clus_flat_per_trig']:.4f} clusters/trigger")
    df = pd.DataFrame(rows)
    df.attrs["pairs"] = pairs(T, cand, src_runs, bkg_run, method)
    for ref, lab in ((bkg_run, "nosource"), (PULSER_RUNS[1], "dummy")):
        if ref in set(df.run):
            b = df[df.run == ref].iloc[0]
            for k in ("pre_hits_per_us", "hits_late_per_us", "clus_flat_per_trig"):
                df[f"{k}_minus_{lab}"] = df[k] - b[k]
            df[f"mu_ind_vs_{lab}"] = df.clus_flat_per_trig - b.clusters_per_trig
            df[f"mu_ind_vs_{lab}_err"] = np.hypot(df.clus_flat_per_trig_err, np.sqrt(b.clusters_per_trig / b.n_trig))
    return df, pd.DataFrame(prof)


def page_insitu(out, df: pd.DataFrame, prof: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    cols = [c for c in prof.columns if c.startswith("t")]
    x = np.array([float(c[1:]) + 0.5 for c in cols])
    for i, (_, p) in enumerate(prof.iterrows()):
        ax[0].plot(x, p[cols].to_numpy(float), "-", lw=1.8, color=ls.COLORS[i % 4] if i < 4 else "0.7",
                   label=f"{p.run} {p.kind}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("hit time in the readout [µs]")
    ax[0].set_ylabel("hits per µs per trigger")
    ax[0].legend(fontsize=8)
    ax[0].set_title("tank hit rate through the window", fontsize=10)
    xx = np.arange(len(df))
    ax[1].errorbar(xx, df.clus_flat_per_trig, df.clus_flat_per_trig_err, fmt="o", ms=8, color="#1f5fbf",
                   label=f"flat term, capture τ fixed {ls.TAU_CAP_US:g} µs")
    ax[1].plot(xx, df.clusters_per_trig, "s", ms=7, mfc="none", color="0.4", label="all clusters per trigger")
    ax[1].set_xticks(xx, [f"{r}\n{k}" for r, k in zip(df.run, df.kind)], fontsize=8)
    ax[1].set_ylabel("clusters per trigger, 2-67 µs")
    ax[1].legend(fontsize=8)
    ax[1].set_title("flat term of the cluster-time fit", fontsize=10)
    two_titles(fig, "Accidental activity measured inside the source triggers",
               "source: vetoed gate triggers; no source and dummy: all vetoed triggers; flat term from "
               "A·exp(−t/τ) + C, Poisson likelihood; with τ free the 65 µs window cannot separate C from the tail")
    out.save(fig, "insitu")


# ---------------------------------------------------------------------------------------------- multiplicity fits
def hist(n: np.ndarray, K: int = KMAX) -> np.ndarray:
    return np.bincount(np.minimum(np.asarray(n, int), K), minlength=K + 1).astype(float)


def accidental(b: np.ndarray, kind: str, par: float) -> np.ndarray:
    K = len(b) - 1
    if kind == "tilt":
        w = b * par ** np.arange(K + 1)
        return w / w.sum()
    if kind == "conv":
        a = np.convolve(b, stats.poisson.pmf(np.arange(K + 1), par))[:K + 1]
        a[K] += max(1.0 - a.sum(), 0.0)
        return a
    return b


def model(b: np.ndarray, eps: float, kind: str, par: float) -> np.ndarray:
    a = accidental(b, kind, par)
    s = np.zeros_like(a)
    s[1:] = a[:-1]
    s[-1] += a[-1]
    return (1 - eps) * a + eps * s


def _profile_err(f, x, i, bnds, fmin) -> float:
    others = [j for j in range(len(x)) if j != i]

    def prof(v):
        def g(y):
            z = np.array(x, float)
            z[others] = y
            z[i] = v
            return f(z)
        r = optimize.minimize(g, np.array(x, float)[others], method="L-BFGS-B", bounds=[bnds[j] for j in others])
        return r.fun - fmin - 0.5
    sides = []
    for sgn in (-1, 1):
        lim = bnds[i][0] if sgn < 0 else bnds[i][1]
        step = 1e-3 * max(abs(x[i]), 1e-2)
        a, b = x[i], x[i] + sgn * step
        while prof(b) < 0:
            a, b = b, x[i] + 2 * (b - x[i])
            if (sgn < 0 and b <= lim) or (sgn > 0 and b >= lim):
                b = lim
                break
        try:
            sides.append(abs(optimize.brentq(prof, min(a, b), max(a, b), xtol=1e-6) - x[i]))
        except ValueError:
            pass
    return float(np.mean(sides)) if sides else np.nan


def _softmax(beta):
    z = np.concatenate([[0.0], beta])
    z = np.exp(z - z.max())
    return z / z.sum()


def fit_mult(d: np.ndarray, t: np.ndarray, kind: str, par_fixed: Optional[float] = None) -> dict:
    K = len(d) - 1
    beta0 = np.log((t[1:] + 0.5) / (t[0] + 0.5))
    free_par = kind != "none" and par_fixed is None
    par0 = 1.0 if kind == "tilt" else 0.02

    def unpack(p):
        eps = p[0]
        par = p[1] if free_par else (par_fixed if par_fixed is not None else (1.0 if kind == "tilt" else 0.0))
        return eps, par, _softmax(p[2:] if free_par else p[1:])

    def nll(p):
        eps, par, b = unpack(p)
        P = np.clip(model(b, eps, kind, par), 1e-300, None)
        return float(-(d * np.log(P)).sum() - (t * np.log(np.clip(b, 1e-300, None))).sum())

    x0 = [0.7] + ([par0] if free_par else []) + list(beta0)
    bnds = [(1e-4, 1 - 1e-4)] + ([(1e-4, 5.0)] if free_par else []) + [(-30, 30)] * K
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bnds)
    r = optimize.minimize(nll, r.x, method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-9, fatol=1e-9))
    eps, par, b = unpack(r.x)
    P = model(b, eps, kind, par)

    def sat(o, e):
        m = o > 0
        return 2.0 * float((o[m] * np.log(o[m] / e[m])).sum())
    dev = sat(d, d.sum() * P) + sat(t, t.sum() * b)
    npar = 1 + (1 if free_par else 0)
    ndf = 2 * K - (K + npar)
    e_eps = _profile_err(nll, r.x, 0, bnds, r.fun)
    e_par = _profile_err(nll, r.x, 1, bnds, r.fun) if free_par else np.nan
    return dict(kind=kind, eps=eps, eps_err=e_eps, par=par, par_err=e_par, deviance=dev, ndf=ndf,
                p=float(stats.chi2.sf(dev, ndf)) if ndf > 0 else np.nan, nll=r.fun,
                acc_mean=float((np.arange(K + 1) * accidental(b, kind, par)).sum()), pred=P, b=b)


def convfit(T: dict, dummy: Optional[RunContext], insit: Optional[pd.DataFrame], src_runs: list, bkg_run: int,
            method: str, ctx: RunContext, foldfs: Optional[str], foldfs_template: str = "ungated") -> tuple:
    B = T[bkg_run]
    tmpl = {f"{bkg_run}": hist(B[~B.tagged].n)}
    if dummy is not None:
        n6256, _ = dz._bkg_counts(dummy, PULSER_RUNS[1], method, "reclust")
        tmpl[f"{PULSER_RUNS[1]}"] = hist(n6256)
    rows, preds = [], []
    for r in src_runs:
        S = T[r]
        d = hist(S[S.gate & ~S.tagged].n)
        for tn, t in tmpl.items():
            fits = [fit_mult(d, t, "none"), fit_mult(d, t, "tilt"), fit_mult(d, t, "conv")]
            if insit is not None and r in set(insit.run) and f"mu_ind_vs_nosource" in insit:
                mu = float(insit.set_index("run").loc[r, "mu_ind_vs_nosource"])
                if np.isfinite(mu) and mu >= 0:
                    fits.append({**fit_mult(d, t, "conv", par_fixed=mu), "kind": "conv, mu in situ"})
            for f in fits:
                rows.append(dict(run=r, template=tn, n_data=int(d.sum()), n_template=int(t.sum()), sample="data",
                                 template_mean=float((np.arange(len(t)) * t).sum() / t.sum()),
                                 **{k: v for k, v in f.items() if k not in ("pred", "b")}))
                preds.append(dict(run=r, template=tn, kind=f["kind"], **{f"data_p{k}": d[k] / d.sum() for k in range(len(d))},
                                  **{f"pred_p{k}": f["pred"][k] for k in range(len(d))}))
            print(f"[nosourcenorm] convfit {r} template {tn}: " + ", ".join(
                f"{f['kind']} eps {f['eps']:.4f} p {f['p']:.3f}" for f in fits))
        if foldfs:
            fo = pd.read_csv(ctx.csv_path(foldfs))
            q = fo[(fo.run == r) & (fo.method == method) & fo.model.str.startswith("M0")]
            if len(q):
                q = q.iloc[0]
                d4 = q.n_mc * np.array([q[f"pred_p{k}"] for k in range(4)])
                t4 = hist(B[B.gate & ~B.tagged].n if foldfs_template == "ic" else B[~B.tagged].n, 3)
                for kind in ("tilt", "conv"):
                    f = fit_mult(d4, t4, kind)
                    rows.append(dict(run=r, template=f"{bkg_run} {foldfs_template}", n_data=int(round(d4.sum())),
                                     n_template=int(t4.sum()),
                                     sample="MC (x) template, Asimov",
                                     template_mean=float((np.arange(4) * t4).sum() / t4.sum()),
                                     **{k: v for k, v in f.items() if k not in ("pred", "b")}))
    return pd.DataFrame(rows), pd.DataFrame(preds)


def page_convfit(out, cf: pd.DataFrame, pr: pd.DataFrame, bkg_run: int):
    import matplotlib.pyplot as plt
    runs = list(pr.run.unique())
    fig, ax = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.8), squeeze=False)
    for a, r in zip(ax[0], runs):
        q = pr[(pr.run == r) & (pr.template == str(bkg_run))]
        k = np.arange(KMAX + 1)
        d = q.iloc[0][[f"data_p{i}" for i in k]].to_numpy(float)
        a.plot(k, d, "ko", ms=8, label="data")
        for j, (_, p) in enumerate(q.iterrows()):
            a.plot(k, p[[f"pred_p{i}" for i in k]].to_numpy(float), ls.MARKERS[j % 4] + "-", mfc="none", ms=6,
                   color=ls.COLORS[j % 4], label=p.kind)
        a.set_yscale("log")
        a.set_xlabel(f"clusters per trigger (last bin ≥ {KMAX})")
        a.set_title(str(r), fontsize=10)
        a.legend(fontsize=8)
    ax[0, 0].set_ylabel("fraction of vetoed gate triggers")
    two_titles(fig, "Accidentals as an extra Poisson term instead of a free scale",
               f"template = vetoed no-source triggers ({bkg_run}), counts profiled in the likelihood; none: s = 1; "
               "tilt: b_k s^k; conv: b ⊗ Poisson(μ)")
    out.save(fig, "convfit")


def icfit(T: dict, lf: pd.DataFrame, src_runs: list, bkg_run: int) -> pd.DataFrame:
    B = T[bkg_run]
    t = hist(B[~B.tagged].n)
    K = len(t) - 1
    rows = []
    for r in src_runs:
        S = T[r][~T[r].tagged]
        f = lf[lf.run == r].set_index("cls")
        cls, gate = class_of(S.IC_adjusted.to_numpy(), S.sp.to_numpy())
        cats = [("0-300", (cls == "0-300") & ~S.sp.to_numpy(), "0-300"),
                ("300-700", (cls == "300-700") & ~S.sp.to_numpy(), "300-700"),
                ("gate", gate, GATE),
                ("1200-3000", (cls == "1200-3000") & ~S.sp.to_numpy(), "1200-3000"),
                ("700-1200, 2nd pulse", (cls == "700-1200") & S.sp.to_numpy(), "700-1200")]
        D = [hist(S.n[m]) for _, m, _ in cats]
        fs = [float(f.loc[ref, "f_fs_veto_live"]) for _, _, ref in cats]
        nc = len(cats)

        def unpack(p):
            return p[:nc], p[nc], _softmax(p[nc + 1:])

        def nll(p):
            eps, mu, b = unpack(p)
            acc = accidental(b, "conv", mu)
            s = -float((t * np.log(np.clip(b, 1e-300, None))).sum())
            for d, e, fc in zip(D, eps, fs):
                P = (1 - fc) * model(b, e, "conv", mu) + fc * acc
                s -= float((d * np.log(np.clip(P, 1e-300, None))).sum())
            return s
        x0 = [0.7] * nc + [0.02] + list(np.log((t[1:] + 0.5) / (t[0] + 0.5)))
        bnds = [(1e-4, 1 - 1e-4)] * nc + [(0.0, 1.0)] + [(-30, 30)] * K
        res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bnds)
        res = optimize.minimize(nll, res.x, method="Nelder-Mead", options=dict(maxiter=80000, xatol=1e-9, fatol=1e-9))
        err = _errors(nll, res.x, [1e-4] * (nc + 1) + [1e-3] * K, bnds)
        eps, mu, b = unpack(res.x)
        dev = 0.0
        acc = accidental(b, "conv", mu)
        for d, e, fc in zip(D, eps, fs):
            P = (1 - fc) * model(b, e, "conv", mu) + fc * acc
            m = d > 0
            dev += 2 * float((d[m] * np.log(d[m] / (d.sum() * P[m]))).sum())
        m = t > 0
        dev += 2 * float((t[m] * np.log(t[m] / (t.sum() * b[m]))).sum())
        ndf = (nc + 1) * K - (nc + 1 + K)
        for (name, msk, _), e, ee, d, fc in zip(cats, eps, err[:nc], D, fs):
            rows.append(dict(run=r, category=name, n=int(d.sum()), p0=float(d[0] / d.sum()), f_fs=fc, eps=e, eps_err=ee,
                             mu_ind=mu, mu_ind_err=err[nc], deviance=dev, ndf=ndf, p=float(stats.chi2.sf(dev, ndf))))
        print(f"[nosourcenorm] icfit {r}: mu_ind {mu:.4f}; " + ", ".join(f"{c[0]} {e:.3f}" for c, e in zip(cats, eps)))
    return pd.DataFrame(rows)


def ctc(T: dict, dt: pd.DataFrame, fac: Optional[pd.DataFrame], ctc_dir: str, src_runs: list, bkg_run: int,
        gap_s: float, sc: dict) -> pd.DataFrame:
    import uproot
    rows = []
    for r in src_runs + [bkg_run]:
        files = sorted(Path(ctc_dir).glob(f"R{r}S0p*_ctc.root"))
        if not files:
            continue
        ts, bad = [], 0
        for f in files:
            try:
                a = uproot.open(f)["tree_timestamps_ctc"].arrays(["t_ctc", "triggerword_ctc"], library="np")
            except (uproot.KeyInFileError, OSError, ValueError):
                bad += 1
                continue
            ts.append(a["t_ctc"][a["triggerword_ctc"] == 19].astype(np.int64))
        t = np.unique(np.concatenate(ts))
        t = t[np.append(True, np.diff(t) > 10_000)]
        bc = np.sort(T[r].eventTankTime.to_numpy(np.int64))
        i = np.clip(np.searchsorted(bc, t), 1, len(bc) - 1)
        matched = np.minimum(np.abs(bc[i] - t), np.abs(bc[i - 1] - t)) < 1000
        j = np.clip(np.searchsorted(t, bc), 1, len(t) - 1)
        bc_in = np.minimum(np.abs(t[j] - bc), np.abs(t[j - 1] - bc)) < 1000
        d = np.diff(t) / 1e9
        d = d[(d > 0) & (d < gap_s)]
        lo, hi = window(d, NFOLD)
        f = emg_fit(d, lo, hi)
        n_bc_span = int(((bc >= t[0]) & (bc <= t[-1])).sum())
        row = dict(run=r, n_files=len(files) - bad, n_files_unreadable=bad, n_ctc=len(t), n_bc=len(bc), n_bc_in_span=n_bc_span,
                   n_ctc_matched=int(matched.sum()), n_ctc_missing_in_bc=int((~matched).sum()),
                   frac_missing=float((~matched).mean()), n_bc_not_in_ctc=int((~bc_in).sum()),
                   mu=f["mu"], sigma=f["sigma"], lam=f["lam"], lam_err=f["lam_err"], chi2=f["chi2"], ndf=f["ndf"],
                   p=f["p"], live_s_ctc=len(d) / f["lam"], exposure_bc=n_bc_span / f["lam"],
                   exposure_bc_err=n_bc_span * f["lam_err"] / f["lam"] ** 2,
                   lam_bc=float(dt.set_index("run").lam[r]), live_s_bc=float(dt.set_index("run").live_s[r]))
        if r in sc:
            row["lam_scaler"], row["lam_scaler_err"] = sc[r][0], sc[r][1]
            row["lam_over_scaler"] = f["lam"] / sc[r][0]
        rows.append(row)
        print(f"[nosourcenorm] ctc {r}: {len(t):,} accepted triggers in {len(files)} files, {int((~matched).sum()):,} "
              f"not in BeamCluster; lambda {f['lam']:.3f} Hz (BeamCluster {row['lam_bc']:.3f}); exposure "
              f"{row['exposure_bc']:.0f} s (BeamCluster fit {row['live_s_bc']:.0f} s)")
    df = pd.DataFrame(rows)
    if fac is not None and bkg_run in set(df.run):
        b = df.set_index("run").loc[bkg_run]
        for r in src_runs:
            if r not in set(df.run):
                continue
            f = fac.set_index("run").loc[r]
            s = df.set_index("run").loc[r]
            k = (f.n_src_cosmic / s.exposure_bc) / (f.n_bkg_cosmic_g / b.exposure_bc)
            df.loc[df.run == r, "k_ctc"] = k
            df.loc[df.run == r, "k_ctc_err"] = k * np.sqrt(1 / f.n_src_cosmic + 1 / f.n_bkg_cosmic_g)
            df.loc[df.run == r, "F_ctc"] = s.exposure_bc / b.exposure_bc
    return df


def icnorm(ic_runs: list, ref: int, bkg_run: int, gap_s: float, sc: dict) -> tuple:
    rows, data = [], {}
    for r in ic_runs:
        w = dz._features(r, bkg_run)
        w = w[w.timestamp > 0].drop_duplicates("timestamp").sort_values("timestamp")
        t = w.timestamp.to_numpy(np.float64) / 1e9
        d = np.diff(t)
        ok = (d > 0) & (d < gap_s)
        fit = emg_fit(d[ok], *window(d[ok], NFOLD))
        q = float(np.quantile(d[ok], ls.TAU_Q))
        n, span, proc = len(w), float(t[-1] - t[0]), float(d[ok].sum())
        m_span, m_proc = n / span, int(ok.sum()) / proc
        row = dict(run=r, n_waveforms=n, span_s=span, processed_s=proc, n_gaps=int((d >= gap_s).sum()), tau_fit_s=fit["mu"],
                   tau_q001_s=q, m_span_hz=m_span, m_proc_hz=m_proc)
        for lab, m in (("span", m_span), ("proc", m_proc)):
            for tl, tau in (("taufit", fit["mu"]), ("tauq", q)):
                row[f"mtau_{lab}_{tl}"] = m * tau
                row[f"n_{lab}_{tl}_hz"] = m / (1 - m * tau)
        row["lambda_fit_hz"] = fit["lam"]
        row["lambda_scaler_hz"] = sc[r][0] if r in sc else np.nan
        rows.append(row)
        data[r] = w.IC_adjusted.to_numpy()
    df = pd.DataFrame(rows)
    span_ref = float(df.set_index("run").span_s[ref])
    df["w_raw"] = 1.0
    df["w_span"] = span_ref / df.span_s
    df["c_deadtime_span"] = 1.0 / (1.0 - df.mtau_span_taufit)
    df["c_deadtime_proc"] = 1.0 / (1.0 - df.mtau_proc_taufit)
    df["w_total_span"] = df.w_span * df.c_deadtime_span
    df["w_total_proc"] = df.w_span * df.c_deadtime_proc
    df["n_presented_in_ref_span"] = df.n_waveforms * df.w_total_span
    proc_ref = float(df.set_index("run").processed_s[ref])
    df["live_s_proc"] = df.processed_s * (1.0 - df.mtau_proc_taufit)
    df["w_proc"] = proc_ref / df.processed_s
    df["w_live"] = df.w_proc * df.c_deadtime_proc
    for _, r in df.iterrows():
        print(f"[nosourcenorm] icnorm {int(r.run)}: N {int(r.n_waveforms):,}, span {r.span_s:.0f} s, w_span {r.w_span:.4f}, "
              f"m {r.m_span_hz:.4f} Hz, tau {r.tau_fit_s:.4f} s, m*tau {r.mtau_span_taufit:.4f}, "
              f"1/(1-m*tau) {r.c_deadtime_span:.3f} (processed-data time: {r.c_deadtime_proc:.3f})")
    return df, data


IC_BANDS = [(3000, np.inf), (5000, 5500), (5500, 6000), (6000, 6500), (6500, 7000), (7000, 7500), (5000, 7500)]


def icbands(df: pd.DataFrame, data: dict, bkg_run: int) -> pd.DataFrame:
    rows = []
    d = df.set_index("run")
    for lo, hi in IC_BANDS:
        nb = int(((data[bkg_run] >= lo) & (data[bkg_run] < hi)).sum())
        for r in d.index:
            n = int(((data[r] >= lo) & (data[r] < hi)).sum())
            row = dict(ic_lo=lo, ic_hi=hi, run=r, n_raw=n, n_bkg_raw=nb)
            for lab, wc in (("raw", "w_raw"), ("step12", "w_total_span"), ("step123", "w_total_proc"), ("time", "w_proc"),
                            ("live", "w_live")):
                q = n * d[wc][r] / (nb * d[wc][bkg_run]) if nb else np.nan
                row[f"n_{lab}"] = n * d[wc][r]
                row[f"ratio_{lab}"] = q
                row[f"ratio_{lab}_err"] = q * np.sqrt(1 / max(n, 1) + 1 / max(nb, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def icrun(ic_runs: list, bkg_run: int, gap_s: float, sc: dict, extra_pos: dict) -> tuple:
    """Per source run: live time, alpha = T_live(run) / T_live(bkg), IC>3000 closure (raw and gain-matched), IC>3000
    sub-band ratios, gate false starts; then pooled per source position. Triggers and BGO IC only."""
    from ..data.processor import AmBeNeutronProcessing
    pos = {**AmBeNeutronProcessing().source_positions, **extra_pos}
    df, data = icnorm(list(ic_runs) + [bkg_run], bkg_run, bkg_run, gap_s, sc)
    d = df.set_index("run")
    t0, b = float(d.live_s_proc[bkg_run]), data[bkg_run]
    nb = int((b > IC_COSMIC).sum())
    gate = {r: int(dz._features(r, bkg_run).drop_duplicates("timestamp").query("timestamp > 0").accepted.sum())
            for r in list(ic_runs) + [bkg_run]}
    rows = []
    for r in ic_runs:
        s, a = data[r], float(d.live_s_proc[r]) / t0
        ns = int((s > IC_COSMIC).sum())
        gf = gain_fit(s[s > IC_COSMIC], b[b > 0.8 * IC_COSMIC])
        nbg = int((b * gf["g"] > IC_COSMIC).sum())
        x, y, z = (float(v) for v in pos[r])
        row = dict(run=r, x=x, y=y, z=z, port=PORT_OF.get((x, z), -1), n_waveforms=int(d.n_waveforms[r]),
                   processed_s=float(d.processed_s[r]), n_gaps=int(d.n_gaps[r]), tau_fit_s=float(d.tau_fit_s[r]),
                   m_proc_hz=float(d.m_proc_hz[r]), mtau=float(d.mtau_proc_taufit[r]), live_s=float(d.live_s_proc[r]),
                   alpha=a, n_ic3000=ns, n_ic3000_bkg=nb, closure=ns / (a * nb),
                   closure_err=ns / (a * nb) * np.sqrt(1 / max(ns, 1) + 1 / nb), g=gf["g"], g_err=gf["g_err"],
                   n_ic3000_bkg_g=nbg, closure_g=ns / (a * nbg), closure_g_err=ns / (a * nbg) * np.sqrt(1 / max(ns, 1) + 1 / nbg),
                   n_gate=gate[r], n_gate_bkg=gate[bkg_run], fs_gate=a * gate[bkg_run], f_fs_gate=a * gate[bkg_run] / gate[r],
                   F_rsb_raw=ns / nb, F_rsb_raw_err=ns / nb * np.sqrt(1 / max(ns, 1) + 1 / nb),
                   F_rsb_g=ns / nbg, F_rsb_g_err=ns / nbg * np.sqrt(1 / max(ns, 1) + 1 / nbg))
        br = []
        for lo, hi in SUBBANDS:
            n1, n0 = int(((s >= lo) & (s < hi)).sum()), int(((b >= lo) & (b < hi)).sum())
            q = n1 / (a * n0) if n0 else np.nan
            row[f"closure_{lo:g}"], row[f"closure_{lo:g}_err"] = q, q * np.sqrt(1 / max(n1, 1) + 1 / max(n0, 1))
            br.append(q)
        row["band_rel_spread"] = float(np.nanstd(br) / np.nanmean(br))
        rows.append(row)
    runs = pd.DataFrame(rows)
    prow = []
    for (x, y, z), g in runs.groupby(["x", "y", "z"]):
        a, ns, ng = g.live_s.sum() / t0, int(g.n_ic3000.sum()), int(g.n_gate.sum())
        prow.append(dict(x=x, y=y, z=z, port=int(g.port.iloc[0]), runs=" ".join(str(int(r)) for r in g.run),
                         live_s=float(g.live_s.sum()), alpha=a, n_ic3000=ns, closure=ns / (a * nb),
                         closure_err=ns / (a * nb) * np.sqrt(1 / max(ns, 1) + 1 / nb), n_gate=ng,
                         fs_gate=a * gate[bkg_run], n_ambe_gate=ng - a * gate[bkg_run],
                         F_rsb_g=float(g.F_rsb_g.sum()), F_rsb_raw=float(g.F_rsb_raw.sum()),
                         F_rsb_g_err=float(g.F_rsb_g.sum()) * np.sqrt(1 / max(ns, 1) + 1 / nb),
                         fs_gate_rsb_g=float(g.F_rsb_g.sum()) * gate[bkg_run]))
    return runs, pd.DataFrame(prow), data


def depth_model(runs: pd.DataFrame) -> pd.DataFrame:
    """IC>3000 closure against source height y: weighted straight line per closure variant, plus the weighted
    mean per y. Rows with model = 'linear' carry the fit, rows with model = 'mean_y' the per-y means."""
    rows = []
    for col in ("closure_g", "closure"):
        y, c, e = runs.y.to_numpy(float), runs[col].to_numpy(float), runs[f"{col}_err"].to_numpy(float)
        w = 1.0 / e ** 2
        A = np.vstack([np.ones_like(y), y]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        p = cov @ (A.T @ (w * c))
        chi = float(np.sum(w * (c - A @ p) ** 2))
        cm = float(np.sum(w * c) / np.sum(w))
        chi0 = float(np.sum(w * (c - cm) ** 2))
        rows.append(dict(variant=col, model="linear", y=np.nan, a=p[0], a_err=np.sqrt(cov[0, 0]), b_per_cm=p[1],
                         b_per_cm_err=np.sqrt(cov[1, 1]), chi2=chi, ndf=len(y) - 2, chi2_const=chi0, ndf_const=len(y) - 1,
                         n_runs=len(y)))
        for yy, g in runs.groupby("y"):
            ww = 1.0 / g[f"{col}_err"] ** 2
            m = float(np.sum(ww * g[col]) / np.sum(ww))
            rows.append(dict(variant=col, model="mean_y", y=yy, a=m, a_err=float(np.sum(ww) ** -0.5),
                             chi2=float(np.sum(ww * (g[col] - m) ** 2)), ndf=len(g) - 1, n_runs=len(g)))
    return pd.DataFrame(rows)


def page_icrun(out, runs: pd.DataFrame, ports: pd.DataFrame, data: dict, bkg_run: int):
    import matplotlib.pyplot as plt
    r = runs.sort_values("run").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(r))
    ax.errorbar(x - 0.12, r.closure, r.closure_err, fmt="o", ms=5, label="IC > 3000")
    ax.errorbar(x + 0.12, r.closure_g, r.closure_g_err, fmt="s", ms=5, label="IC > 3000, 6264 gain-matched (g)")
    ax.set_xticks(x, [f"{int(v)}\nP{int(p)} y{int(y)}" for v, p, y in zip(r.run, r.port, r.y)], fontsize=7)
    ax.set_ylabel(f"N(IC>3000) / (α × N_{bkg_run}(IC>3000))")
    ax.legend(fontsize=8, frameon=False)
    two_titles(fig, "Live-time α closure in the source-free IC > 3000 region, every source run",
               f"α = T_live(run) / T_live({bkg_run}), T_live = T_rec (1 − m τ), τ = fitted mean dead time; "
               "no cuts, BGO IC only")
    out.save(fig, "icrun_closure")
    edges = np.arange(-1000.0, 18000.0 + 250.0, 250.0)
    hb = np.histogram(data[bkg_run], edges)[0].astype(float)
    for port, pg in ports.groupby("port"):
        pg = pg.sort_values("y")
        nc = len(pg)
        fig, axs = plt.subplots(1, nc, figsize=(4.2 * nc, 4.2), squeeze=False)
        for a_, (_, p) in zip(axs[0], pg.iterrows()):
            hs = sum(np.histogram(data[int(v)], edges)[0].astype(float) for v in p.runs.split())
            a_.stairs(hs, edges, color="C0", lw=1.2, label=f"source ({p.runs})")
            a_.stairs(p.alpha * hb, edges, color="C1", lw=1.2, label=f"{bkg_run} × α ({p.alpha:.3f})")
            a_.set_yscale("log")
            a_.set_xlabel("BGO IC (IC_adjusted)")
            a_.set_ylabel("triggers per 250 IC")
            a_.set_title(f"y = {p.y:+.0f} cm, closure {p.closure:.3f} ± {p.closure_err:.3f}", fontsize=9)
            a_.legend(fontsize=7, frameon=False)
        two_titles(fig, f"Port {int(port)}: BGO IC of the source runs and the no-source run × α",
                   f"runs at one position pooled, α = Σ T_live(runs) / T_live({bkg_run}); closure = N(IC>3000) / "
                   f"(α N_{bkg_run}(IC>3000)); no cuts")
        out.save(fig, f"icrun_port{int(port)}")


def muonrate(runs: list, bc_dir: str, pe_list: list, win_us: tuple, ref: int, bkg_run: int,
             extra_pos: dict) -> tuple:
    import awkward as ak
    import uproot
    from ..data.processor import AmBeNeutronProcessing
    pos = {**AmBeNeutronProcessing().source_positions, **extra_pos}
    lo, hi = win_us[0] * 1e3, win_us[1] * 1e3
    dt = float(win_us[1] - win_us[0])
    edges = np.arange(2000.0, 66000.0 + 1.0, 4000.0)
    tz = ZoneInfo("America/Chicago")
    rows, prof = [], []
    for r in runs:
        t = uproot.open(str(Path(bc_dir) / f"BeamCluster_{r}.root"))["Event"].arrays(
            ["eventTimeTank", "clusterTime", "clusterPE"], library="ak")
        ett = ak.to_numpy(t.eventTimeTank).astype(np.int64)
        sel = np.zeros(len(ett), bool)
        sel[np.unique(ett, return_index=True)[1]] = True
        sel &= ett != 0
        ct, pe, n = t.clusterTime[sel], t.clusterPE[sel], int(sel.sum())
        x, y, z = (float(v) for v in pos.get(r, (np.nan, np.nan, np.nan)))
        kind = "dummy" if r in DUMMY_RUNS else "no source" if r == bkg_run else "source"
        row = dict(run=r, kind=kind, x=x, y=y, z=z, port=PORT_OF.get((x, z), -1) if kind == "source" else -1,
                   date=datetime.fromtimestamp(np.median(ett[sel]) / 1e9, tz).strftime("%Y-%m-%d"),
                   n_triggers=n, window_us=dt)
        for q in pe_list:
            m = (ct >= lo) & (ct < hi) & (pe > q)
            nc, nw = int(ak.sum(m)), int(ak.sum(ak.any(m, axis=1)))
            row.update({f"n_cl_{q:g}": nc, f"n_win_{q:g}": nw, f"rate_{q:g}": nc / (n * dt),
                        f"rate_{q:g}_err": np.sqrt(nc) / (n * dt), f"rate_win_{q:g}": -np.log1p(-nw / n) / dt})
        h = np.histogram(ak.to_numpy(ak.flatten(ct[pe > pe_list[0]])), edges)[0]
        w = n * np.diff(edges) / 1e3
        prof += [dict(run=r, kind=kind, t_lo_us=a / 1e3, t_hi_us=b / 1e3, n=int(k), rate=k / ww, rate_err=np.sqrt(k) / ww)
                 for a, b, k, ww in zip(edges[:-1], edges[1:], h, w)]
        rows.append(row)
    df = pd.DataFrame(rows)
    d = df.set_index("run")
    for q in pe_list:
        r0, e0 = d.loc[ref, f"rate_{q:g}"], d.loc[ref, f"rate_{q:g}_err"]
        df[f"ratio_{q:g}"] = df[f"rate_{q:g}"] / r0
        df[f"ratio_{q:g}_err"] = df[f"ratio_{q:g}"] * np.sqrt((df[f"rate_{q:g}_err"] / df[f"rate_{q:g}"]) ** 2
                                                              + (e0 / r0) ** 2)
    prow = []
    for (kind, port, y), g in df.groupby(["kind", "port", "y"], dropna=False):
        nt, ncl = int(g.n_triggers.sum()), int(g[f"n_cl_{pe_list[0]:g}"].sum())
        rate, err = ncl / (nt * dt), np.sqrt(ncl) / (nt * dt)
        prow.append(dict(kind=kind, port=port, y=y, runs=" ".join(str(int(v)) for v in g.run), n_triggers=nt,
                         n_cl=ncl, rate=rate, rate_err=err, ratio=rate / d.loc[ref, f"rate_{pe_list[0]:g}"],
                         ratio_err=rate / d.loc[ref, f"rate_{pe_list[0]:g}"]
                         * np.sqrt((err / rate) ** 2 + (d.loc[ref, f"rate_{pe_list[0]:g}_err"]
                                                        / d.loc[ref, f"rate_{pe_list[0]:g}"]) ** 2)))
    return df, pd.DataFrame(prow), pd.DataFrame(prof)


def page_muonrate(out, df: pd.DataFrame, prof: pd.DataFrame, pe_list: list, win_us: tuple, ref: int):
    import matplotlib.pyplot as plt
    r = df.sort_values(["date", "run"]).reset_index(drop=True)
    x = np.arange(len(r))
    lab = [f"{int(v)}\n{'P%d y%d' % (p, y) if k == 'source' else k}" for v, p, y, k in zip(r.run, r.port, r.y, r.kind)]
    q0 = pe_list[0]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, kind in enumerate(["source", "dummy", "no source"]):
        m = (r.kind == kind).to_numpy()
        ax.errorbar(x[m], 1e3 * r[f"rate_{q0:g}"][m], 1e3 * r[f"rate_{q0:g}_err"][m], fmt=ls.MARKERS[i],
                    ms=5, color=ls.COLORS[i], label=kind)
    ax.set_xticks(x, lab, fontsize=6)
    ax.set_ylabel(f"clusters > {q0:g} p.e. per ms of window")
    ax.legend(fontsize=8, frameon=False)
    two_titles(fig, f"Tank muon rate per window, {win_us[0]:g}–{win_us[1]:g} µs, cluster PE > {q0:g}, every run by date",
               "BeamCluster tree clusters (ClusterFinder + C++ merge 50/2000); one entry per tank timestamp")
    out.save(fig, "muonrate_runs")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, q in enumerate(pe_list):
        ax.errorbar(x + 0.15 * (i - 1), r[f"ratio_{q:g}"], r[f"ratio_{q:g}_err"], fmt=ls.MARKERS[i] + "-", lw=0.6,
                    ms=4, color=ls.COLORS[i], label=f"PE > {q:g}")
    ax.set_xticks(x, lab, fontsize=6)
    ax.set_ylabel(f"rate / rate({ref})")
    ax.legend(fontsize=8, frameon=False)
    two_titles(fig, f"Tank muon rate per window relative to dummy run {ref}",
               f"window {win_us[0]:g}–{win_us[1]:g} µs; equal rates mean equal late cosmic products per window")
    out.save(fig, "muonrate_ratio")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, kind in enumerate(["source", "dummy", "no source"]):
        g = prof[prof.kind == kind].groupby(["t_lo_us", "t_hi_us"], as_index=False).n.sum()
        nt = df[df.kind == kind].n_triggers.sum()
        w = nt * (g.t_hi_us - g.t_lo_us)
        c = 0.5 * (g.t_lo_us + g.t_hi_us)
        ax.errorbar(c, 1e3 * g.n / w, 1e3 * np.sqrt(g.n) / w, fmt=ls.MARKERS[i] + "-", ms=5, color=ls.COLORS[i],
                    label=f"{kind} ({nt:,} triggers)")
    ax.set_xlabel("cluster time [µs]")
    ax.set_ylabel(f"clusters > {q0:g} p.e. per ms of window")
    ax.legend(fontsize=8, frameon=False)
    two_titles(fig, f"Time profile of clusters > {q0:g} p.e., runs pooled by kind",
               "a flat profile means the late muon-like clusters are uncorrelated with the trigger")
    out.save(fig, "muonrate_profile")


def page_icnorm(out, df: pd.DataFrame, data: dict, ref: int, bkg_run: int):
    import matplotlib.pyplot as plt
    zoom, full = np.arange(-500.0, 3000.0 + 25.0, 25.0), np.arange(-1000.0, 18000.0 + 200.0, 200.0)
    st = {r: dict(color=ls.COLORS[i % 4], marker=ls.MARKERS[i % 4]) for i, r in enumerate(df.run)}
    lab = {r: f"{int(r)} ({'no source' if r == bkg_run else 'source'})" for r in df.run}

    def panel(a, edges, wcol, logy):
        c = 0.5 * (edges[1:] + edges[:-1])
        for _, r in df.iterrows():
            h = np.histogram(data[r.run], edges)[0] * r[wcol]
            a.plot(c, h, "-", lw=1.4, ms=2.5, label=lab[r.run], **st[r.run])
        if logy:
            a.set_yscale("log")
        a.set_xlabel("BGO IC (IC_adjusted)")
        a.set_ylabel(f"triggers per {edges[1] - edges[0]:g} IC bin")
        a.legend(fontsize=8)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    panel(ax[0], zoom, "w_raw", False)
    panel(ax[1], full, "w_raw", True)
    ax[0].set_title("recorded waveforms, no normalisation", fontsize=10)
    ax[1].set_title("same, full IC range", fontsize=10)
    w = ", ".join(f"{int(r.run)}: {int(r.n_waveforms):,} waveforms in {r.span_s:,.0f} s" for _, r in df.iterrows())
    two_titles(fig, "BGO IC of every recorded waveform, before any normalisation",
               f"no cuts; processor.py Stage-1 IC; raw counts per bin; {w}")
    out.save(fig, "icnorm_raw")
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    panel(ax[0, 0], zoom, "w_span", False)
    panel(ax[0, 1], full, "w_span", True)
    panel(ax[1, 0], zoom, "w_total_span", False)
    panel(ax[1, 1], full, "w_total_span", True)
    ax[0, 0].set_title(f"recorded waveforms × span({ref}) / span(run)", fontsize=10)
    ax[0, 1].set_title("same, full IC range", fontsize=10)
    ax[1, 0].set_title("× 1 / (1 − m τ): presented triggers in the reference span", fontsize=10)
    ax[1, 1].set_title("same, full IC range", fontsize=10)
    w = ", ".join(f"{int(r.run)}: ×{r.w_span:.3f} ×{r.c_deadtime_span:.2f}" for _, r in df.iterrows())
    two_titles(fig, f"BGO IC of every recorded waveform, normalised to the span of run {ref}",
               f"no cuts; processor.py Stage-1 IC; m = N / span, τ = fitted mean dead time, n = m / (1 − m τ) "
               f"(Knoll 4.24); weights {w}")
    out.save(fig, "icnorm_span")
    for wc, key, t1, t2 in (("w_proc", "icnorm_time", f"BGO IC, every run scaled to the recorded time of {ref}",
                             "weight = recorded time(" + str(ref) + ") / recorded time(run); recorded time = span minus "
                             "unprocessed part files; "),
                            ("w_live", "icnorm_live", f"BGO IC per live time, scaled to the recorded time of {ref}",
                             f"weight = recorded time({ref}) / live time(run), live time = recorded time × (1 − m τ), "
                             "m = recorded rate; ")):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        panel(ax[0], zoom, wc, False)
        panel(ax[1], full, wc, True)
        two_titles(fig, t1, t2 + "weights " + ", ".join(f"{int(r.run)}: ×{r[wc]:.3f}" for _, r in df.iterrows()))
        out.save(fig, key)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    panel(ax[0], zoom, "w_span", False)
    panel(ax[1], full, "w_span", True)
    two_titles(fig, f"Step 1: span({ref}) / span(run)",
               "no cuts; processor.py Stage-1 IC; weights " + ", ".join(f"{int(r.run)}: ×{r.w_span:.3f}" for _, r in df.iterrows()))
    out.save(fig, "icnorm_step1")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    panel(ax[0], zoom, "w_total_span", False)
    panel(ax[1], full, "w_total_span", True)
    two_titles(fig, f"BGO IC with the dead-time rate taken over the full span (unprocessed part files counted as live)",
               f"no cuts; processor.py Stage-1 IC; τ = fitted mean dead time (Knoll 4.24); weights {w}")
    out.save(fig, "icnorm_step12")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    panel(ax[0], zoom, "w_total_proc", False)
    panel(ax[1], full, "w_total_proc", True)
    w = ", ".join(f"{int(r.run)}: ×{r.w_span:.3f} ×{r.c_deadtime_proc:.2f}" for _, r in df.iterrows())
    two_titles(fig, "Same, with the recorded rate over the time our processed data cover",
               f"m = N / (span − intervals ≥ 30 s, which are part files missing from the processed data while the DAQ ran), "
               f"τ = fitted mean dead time, n = m / (1 − m τ); weights {w}")
    out.save(fig, "icnorm_processed")


def page_icfit(out, ic: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (r, g) in enumerate(ic.groupby("run")):
        x = np.arange(len(g)) + 0.1 * i
        ax.errorbar(x, g.eps, g.eps_err, fmt=ls.MARKERS[i] + "-", ms=8, color=ls.COLORS[i],
                    label=f"{r}, shared μ_ind {g.mu_ind.iloc[0]:.3f}")
        ax.set_xticks(np.arange(len(g)), g.category, rotation=15)
    ax.set_ylabel("neutron detection efficiency")
    ax.set_xlabel("BGO IC class (no second pulse unless stated)")
    ax.legend(fontsize=8)
    two_titles(fig, "Efficiency in each IC class from one joint fit",
               "vetoed triggers; accidentals = no-source template ⊗ Poisson(μ_ind), shared; false-start shares fixed "
               "from the live-time prediction")
    out.save(fig, "icfit")


# ---------------------------------------------------------------------------------------------- driver
def run(ctx: RunContext, dummy: Optional[RunContext], runs_db: str, src_runs: list, bkg_run: int, method: str,
        gap_s: float, parts: list, out_name: str, block_h: float, foldfs: Optional[str], foldfs_template: str,
        ctc_dir: Optional[str] = None, ic_runs: Optional[list] = None, ic_ref: Optional[int] = None,
        ic_positions: Optional[dict] = None, muon: Optional[dict] = None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    pre = f"nosourcenorm_{out_name}"
    if parts == ["muonrate"]:
        bc_dir = muon["bc_dir"] or (ctx.extra.get("stage1") or {}).get("data_directory")
        if not bc_dir:
            raise SystemExit("[nosourcenorm] muonrate needs --bc-dir or stage1.data_directory in the config")
        df, ports, prof = muonrate(muon["runs"], bc_dir, muon["pe"], muon["window_us"], muon["ref"], bkg_run,
                                   ic_positions or {})
        df.to_csv(ctx.csv_path(f"{pre}_muonrate"), index=False)
        ports.to_csv(ctx.csv_path(f"{pre}_muonrate_positions"), index=False)
        prof.to_csv(ctx.csv_path(f"{pre}_muonrate_profile"), index=False)
        with PdfPages(ctx.plot_path(pre, "pdf")) as pdf:
            page_muonrate(dz.Out(ctx, pdf, pre), df, prof, muon["pe"], muon["window_us"], muon["ref"])
        print(f"[nosourcenorm] wrote csv {pre}_muonrate(_positions, _profile) and {ctx.plot_path(pre, 'pdf')}")
        return
    if parts == ["icnorm"]:
        df, data = icnorm(ic_runs, ic_ref, bkg_run, gap_s, scalers(runs_db))
        df.to_csv(ctx.csv_path(f"{pre}_icnorm"), index=False)
        icbands(df, data, bkg_run).to_csv(ctx.csv_path(f"{pre}_icbands"), index=False)
        with PdfPages(ctx.plot_path(pre, "pdf")) as pdf:
            page_icnorm(dz.Out(ctx, pdf, pre), df, data, ic_ref, bkg_run)
        print(f"[nosourcenorm] wrote csv {pre}_icnorm and {ctx.plot_path(pre, 'pdf')}")
        return
    if parts == ["icrun"]:
        runs, ports, data = icrun(ic_runs, bkg_run, gap_s, scalers(runs_db), ic_positions or {})
        runs.to_csv(ctx.csv_path(f"{pre}_icrun"), index=False)
        ports.to_csv(ctx.csv_path(f"{pre}_icport"), index=False)
        fac = runs[["run", "port", "x", "y", "z", "F_rsb_g", "F_rsb_g_err", "alpha", "F_rsb_raw", "F_rsb_raw_err",
                    "g", "g_err", "closure_g", "closure_g_err", "n_ic3000", "n_ic3000_bkg_g", "live_s"]].copy()
        fac["alpha_minus_F_rel"] = (fac.alpha - fac.F_rsb_g) / fac.F_rsb_g
        fac.sort_values(["port", "y", "run"]).to_csv(ctx.csv_path(f"{pre}_factors"), index=False)
        depth_model(runs).to_csv(ctx.csv_path(f"{pre}_depthmodel"), index=False)
        with PdfPages(ctx.plot_path(pre, "pdf")) as pdf:
            page_icrun(dz.Out(ctx, pdf, pre), runs, ports, data, bkg_run)
        print(f"[nosourcenorm] wrote csv {pre}_icrun, {pre}_icport and {ctx.plot_path(pre, 'pdf')}")
        return
    T, cand = load(ctx, src_runs, bkg_run, method)
    tabs = {}
    dtab, grp = deadtime(T, bkg_run, gap_s, scalers(runs_db))
    tabs.update(deadtime=dtab, deadtime_groups=grp)
    if "stability" in parts:
        tabs["stability"], tabs["stability_summary"] = stability(T, dtab, bkg_run, gap_s, block_h)
    fac = ext = lf = ins = None
    if any(p in parts for p in ("rsb", "leftsb", "icfit")):
        fac, ext = rsb(T, dtab, src_runs, bkg_run)
        tabs.update(rsb_factors=fac, rsb_extrapolation=ext)
    if any(p in parts for p in ("leftsb", "icfit")):
        lf = leftsb(T, dtab, fac, src_runs, bkg_run)
        tabs["leftsb"] = lf
    if any(p in parts for p in ("insitu", "convfit")):
        ins, prof = insitu(ctx, dummy, T, cand, src_runs, bkg_run, method)
        tabs.update(insitu=ins, insitu_profile=prof, insitu_pairs=ins.attrs.pop("pairs"))
    if "convfit" in parts:
        tabs["convfit"], tabs["convfit_pred"] = convfit(T, dummy, ins, src_runs, bkg_run, method, ctx, foldfs,
                                                        foldfs_template)
    if "icfit" in parts:
        tabs["icfit"] = icfit(T, lf, src_runs, bkg_run)
    if "ctc" in parts:
        if not ctc_dir:
            raise SystemExit("[nosourcenorm] part ctc needs --ctc-dir")
        if fac is None:
            fac, ext = rsb(T, dtab, src_runs, bkg_run)
        tabs["ctc"] = ctc(T, dtab, fac, ctc_dir, src_runs, bkg_run, gap_s, scalers(runs_db))
    for name, df in tabs.items():
        df.to_csv(ctx.csv_path(f"{pre}_{name}"), index=False)
    with PdfPages(ctx.plot_path(pre, "pdf")) as pdf:
        out = dz.Out(ctx, pdf, pre)
        page_deadtime(out, T, dtab, grp, bkg_run, gap_s)
        if "stability" in tabs:
            page_stability(out, tabs["stability"], bkg_run, block_h)
        if fac is not None and "rsb" in parts:
            page_rsb(out, T, fac, ext, bkg_run)
        if lf is not None and "leftsb" in parts:
            page_leftsb(out, lf)
        if ins is not None and "insitu" in parts:
            page_insitu(out, ins, tabs["insitu_profile"])
        if "convfit" in tabs:
            page_convfit(out, tabs["convfit"], tabs["convfit_pred"], bkg_run)
        if "icfit" in tabs:
            page_icfit(out, tabs["icfit"])
    print(f"[nosourcenorm] wrote csv {pre}_* and {ctx.plot_path(pre, 'pdf')}")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe plots nosourcenorm")
    p.add_argument("--source-runs", required=True, help="comma list of source-in runs in the data event index")
    p.add_argument("--nosource-run", required=True, type=int)
    p.add_argument("--dummy-config", help="randoms config of the dummy-trigger runs 6254/6256")
    p.add_argument("--runs-db", required=True, help="run-database YAML; its shift_log scalers are compared")
    p.add_argument("--method", required=True)
    p.add_argument("--gap-s", required=True, type=float, help="intervals at least this long are DAQ gaps")
    p.add_argument("--parts", default=",".join(PARTS))
    p.add_argument("--block-h", type=float, default=2.0, help="stability block length [h]")
    p.add_argument("--foldfs", help="livetime foldfs csv base name for the MC Asimov closure of convfit")
    p.add_argument("--foldfs-template", choices=["ungated", "ic"], default="ungated",
                   help="no-source template the --foldfs fold used (livetime --template-gate none -> ungated, ic_icfold -> ic)")
    p.add_argument("--ctc-dir", help="ctc part: directory of R<run>S0p<part>_ctc.root (StoreDecodedTimestamps, CTC only)")
    p.add_argument("--ic-runs", help="icnorm part: comma list of runs whose IC spectra are normalised")
    p.add_argument("--ic-ref", type=int, help="icnorm part: run whose span is the reference")
    p.add_argument("--ic-positions", help="icrun part: source positions [cm] missing from processor.py, "
                                          "'run:x,y,z run:x,y,z'")
    p.add_argument("--features-tag", help="icnorm/icrun: read TriggerSummary/WaveformFeatures_<tag>_<run>.parquet "
                                          "(e.g. AmBe2.0v4_bkgnorm) instead of the old-tag fallbacks")
    p.add_argument("--muon-runs", help="muonrate part: comma list of runs (source, dummy and no-source)")
    p.add_argument("--bc-dir", help="muonrate part: folder of BeamCluster_<run>.root (default stage1.data_directory)")
    p.add_argument("--muon-pe", default="1000,2000,3000",
                   help="muonrate part: cluster PE thresholds, first = nominal; >= 1000 (300 already shows a "
                        "source-run excess, a step-3 quantity)")
    p.add_argument("--muon-window-us", default="10,66", help="muonrate part: cluster-time window [us]")
    p.add_argument("--muon-ref", type=int, default=6256, help="muonrate part: reference (dummy) run of the ratios")
    p.add_argument("--out-name", required=True)
    a = p.parse_args(list(argv) if argv else [])
    parts = a.parts.split(",")
    if a.features_tag:
        dz.FEATURES_TAG = a.features_tag
    bad = set(parts) - set(PARTS)
    if bad:
        raise SystemExit(f"[nosourcenorm] unknown parts {sorted(bad)}")
    run(ctx, load_context(a.dummy_config) if a.dummy_config else None, a.runs_db,
        [int(x) for x in a.source_runs.split(",")], a.nosource_run, a.method, a.gap_s, parts, a.out_name, a.block_h,
        a.foldfs, a.foldfs_template, a.ctc_dir, [int(x) for x in a.ic_runs.split(",")] if a.ic_runs else None, a.ic_ref,
        {int(k): tuple(float(v) for v in xyz.split(",")) for k, xyz in (s.split(":") for s in a.ic_positions.split())}
        if a.ic_positions else None,
        dict(runs=[int(x) for x in a.muon_runs.split(",")] if a.muon_runs else [], bc_dir=a.bc_dir,
             pe=[float(x) for x in a.muon_pe.split(",")], window_us=tuple(float(x) for x in a.muon_window_us.split(",")),
             ref=a.muon_ref))
