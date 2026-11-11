"""PMTWaveformSim pulses as PhaseIIADCHitFinder finds them, read from the PUL/MCH/RAW lines of a debug build.

ambe plots pulseshape --config configs/pr6_pulses.yaml --log ambe=<analyse log> --log tank=<analyse log>
  PUL ev ch j start end peak pe qtrue nmembers allnoise | peak:q ... | trace ADC (baseline-subtracted, 2 ns samples)
  MCH ev ch t0 peak q darknoise directparent0 assigned_pulse(-1 = in no pulse)
  RAW ev ch peak q t_first | raw ADC samples (2 ns) around an MCHit that is in no pulse
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext
from ..plotting import set_style, two_titles

NS = 2.0
WINDOW_PRE_NS, WINDOW_POST_NS = 160.0, 400.0
Q_BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 1e9]


def read_log(path: Path):
    pul, mch, raw = [], [], []
    for line in open(path, errors="replace"):
        line = line.replace("[1]: ", "").replace("[0]: ", "")  # ToolDAQ log prefix on every cerr insertion
        if line.startswith("PUL "):
            head, mem, tr = line.split("|")
            f = head.split()
            members = [tuple(map(float, m.split(":"))) for m in mem.split()]
            pul.append(dict(ev=int(f[1]), ch=int(f[2]), j=int(f[3]), start=float(f[4]), end=float(f[5]),
                            peak=float(f[6]), pe=float(f[7]), qtrue=float(f[8]), nmem=int(f[9]), noise=int(f[10]),
                            mem_peak=[m[0] for m in members], mem_q=[m[1] for m in members],
                            trace=np.array(tr.split(), dtype=float)))
        elif line.startswith("MCH "):
            f = line.split()
            mch.append(dict(ev=int(f[1]), ch=int(f[2]), t0=float(f[3]), peak=float(f[4]), q=float(f[5]),
                            dn=int(f[6]), dp0=int(f[7]), pulse=int(f[8])))
        elif line.startswith("RAW "):
            head, s = line.split("|")
            f = head.split()
            raw.append(dict(ev=int(f[1]), ch=int(f[2]), peak=float(f[3]), q=float(f[4]), t_first=float(f[5]),
                            samples=np.array(s.split(), dtype=float)))
    return pd.DataFrame(pul), pd.DataFrame(mch), pd.DataFrame(raw)


def pmt_types(geometry_csv: str) -> dict:
    g = pd.read_csv(geometry_csv, skiprows=1)
    g.columns = [c.strip() for c in g.columns]
    g = g.dropna(subset=["channel_num"])
    return dict(zip(g["channel_num"].astype(int), g["PMT_type"].astype(str).str.strip()))


def template(params_csv: str, chans: Iterable[int], q: float, t: np.ndarray, t0_offset_ticks: float = 25) -> np.ndarray:
    """Mean over PMTs of the PMTWaveformSim lognormal + 2 reflections (the data fit), q p.e., peak at t = 0."""
    p = pd.read_csv(params_csv, skipinitialspace=True)
    p.columns = [c.strip() for c in p.columns]
    p = p[p["PMT"].astype(int).isin(set(chans))]
    out = []
    for _, r in p.iterrows():
        x = t + r["p1"]

        def lg(z):
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.exp(-0.5 * (np.log(z / r["p1"]) / r["p2"]) ** 2)
            return np.where(z > 0, v, 0.0)
        out.append(r["p0"] * q * (lg(x) + r["r1"] * lg(x - r["T1"]) + r["r2"] * lg(x - 2 * r["T2"])))
    return np.mean(out, axis=0) if out else np.full_like(t, np.nan)


def aligned_average(traces, lo=-10.0, hi=30.0):
    """Mean trace aligned on the highest sample (same for MC and data)."""
    grid = np.arange(lo, hi + NS, NS)
    rows = []
    for tr in traces:
        x = NS * (np.arange(len(tr)) - int(np.argmax(tr)))
        rows.append(np.interp(grid, x, tr, left=np.nan, right=np.nan))
    a = np.array(rows) if rows else np.full((0, len(grid)), np.nan)
    with np.errstate(invalid="ignore"):
        return grid, np.nanmean(a, axis=0), a.shape[0]


def read_data_traces(path: str) -> pd.DataFrame:
    """PrintADCTraces output: one TGraph per pulse (ns from pulse start, ADC above baseline) + TraceSummary."""
    import uproot
    f = uproot.open(path)
    rows = []
    for k, cls in f.classnames().items():
        if cls != "TGraph":
            continue
        ch, name = k.split(";")[0].split("/")
        pe = float(name.split("_")[-1])
        rows.append(dict(ch=int(ch), pe=pe, trace=np.asarray(f[k].values()[1], dtype=float)))
    return pd.DataFrame(rows)


def window_vs_peak(pul: pd.DataFrame, mch: pd.DataFrame) -> pd.DataFrame:
    """Per pulse: MCHits the #392 window gives it (t0 in [start - 160, start + 400] ns) vs the peak rule."""
    by = {k: g for k, g in mch.groupby(["ev", "ch"])}
    rows = []
    for r in pul.itertuples():
        g = by.get((r.ev, r.ch))
        if g is None:
            continue
        win = g[(g.t0 > r.start - WINDOW_PRE_NS) & (g.t0 < r.start + WINDOW_POST_NS)].sort_values("t0")
        pk = g[(g.peak >= r.start) & (g.peak <= r.end)].sort_values("q", ascending=False, kind="stable")
        rows.append(dict(ev=r.ev, ch=r.ch, n_window=len(win), n_peak=len(pk),
                         first_window=int(win.dp0.iloc[0]) if len(win) else -99,
                         first_peak=int(pk.dp0.iloc[0]) if len(pk) else -99,
                         q_window=win.q.sum(), q_peak=pk.q.sum(), pe=r.pe))
    return pd.DataFrame(rows)


def _bin_frac(x, sel, bins):
    idx = np.digitize(x, bins) - 1
    return [(bins[i], bins[i + 1], int((idx == i).sum()), float(sel[idx == i].mean()) if (idx == i).any() else np.nan)
            for i in range(len(bins) - 1)]


def run(ctx: RunContext, args):
    set_style()
    out = ctx.plots_dir / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    types = pmt_types(args.geometry)
    samples = {}
    for spec in args.log:
        name, path = spec.split("=", 1)
        pul, mch, raw = read_log(Path(path))
        pul["type"] = pul.ch.map(types).fillna("?")
        samples[name] = (pul, mch, raw)
        print(f"[pulseshape] {name}: {len(pul)} pulses, {len(mch)} simulated MCHits, {len(raw)} raw snippets")

    summary = []
    for name, (pul, mch, raw) in samples.items():
        nsamp_ok = np.mean([len(t) == round((e - s) / NS) + 1 for t, s, e in zip(pul.trace, pul.start, pul.end)])
        lost = mch.pulse == -1
        summary.append(dict(sample=name, n_pulses=len(pul), n_mchits=len(mch), frac_mchits_in_no_pulse=lost.mean(),
                            frac_pe_in_no_pulse=mch.q[lost].sum() / mch.q.sum(), mean_q_lost=mch.q[lost].mean(),
                            mean_q_kept=mch.q[~lost].mean(), frac_orphan_pulses=(pul.nmem == 0).mean(),
                            median_pe_over_true=float(np.median((pul.pe / pul.qtrue)[pul.qtrue > 0])),
                            trace_length_matches_window=nsamp_ok))
    pd.DataFrame(summary).to_csv(ctx.csv_path(f"{args.out_name}_summary"), index=False)

    # 1. average 0.5-2 p.e. pulse by PMT type: MC, AmBe data, and the data-fit pulse model
    sel_types = [t for t in ["LUX", "ETEL", "Hamamatsu", "Watchboy", "Watchman"] if any(
        (s[0].type == t).any() for s in samples.values())]
    name0 = args.shape_sample if args.shape_sample in samples else next(iter(samples))
    pul = samples[name0][0]
    spe = pul[(pul.pe >= 0.5) & (pul.pe <= 2.0)]
    data = read_data_traces(args.data_traces) if args.data_traces else pd.DataFrame(columns=["ch", "pe", "trace"])
    if len(data):
        data["type"] = data.ch.map(types).fillna("?")
        data = data[(data.pe >= 0.5) & (data.pe <= 2.0)]
    rows = []
    n = len(sel_types)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False, sharey=True)
    tf = np.linspace(-10, 30, 401)
    for ax, t in zip(axes[0], sel_types):
        s = spe[spe.type == t]
        grid, avg, k = aligned_average(s.trace)
        ax.plot(grid, avg, "o-", ms=3, color="C3", label=f"MC ({k} pulses)")
        for g, v in zip(grid, avg):
            rows.append(dict(sample=name0, pmt_type=t, t_ns=g, avg_adc=v, n_pulses=k, source="MC"))
        if len(data):
            d = data[data.type == t]
            gd, ad, kd = aligned_average(d.trace)
            ax.plot(gd, ad, "o-", ms=3, color="k", label=f"data {args.data_label} ({kd} pulses)")
            for g, v in zip(gd, ad):
                rows.append(dict(sample=args.data_label, pmt_type=t, t_ns=g, avg_adc=v, n_pulses=kd, source="data"))
        q = float(s.qtrue[s.nmem == 1].mean()) if (s.nmem == 1).any() else 1.0
        ax.plot(tf, template(args.params, s.ch.unique(), q, tf), "-", color="0.55", lw=1.2, label="pulse model (fit)")
        ax.set_title(f"{t}", fontsize=11)
        ax.set_xlabel("time from highest sample [ns]")
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("ADC above baseline (mean)")
    top = two_titles(fig, "Average 0.5-2 p.e. pulse by PMT type",
                     f"MC {name0}: PMTWaveformSim + PhaseIIADCHitFinder (Fixed_2023_Gains, 7 ADC); data: PrintADCTraces "
                     f"on {args.data_label}; samples inside the found pulse window only; model = mean lognormal + "
                     "2 reflections from PMTWaveformLognormFit.csv")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "average_spe_by_type.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ctx.csv_path(f"{args.out_name}_average_spe"), index=False)

    # 2. which true photons make no pulse, by MCHit charge
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (name, (pul, mch, raw)) in enumerate(samples.items()):
        b = _bin_frac(mch.q.values, (mch.pulse == -1).values, Q_BINS)
        c = [0.5 * (lo + min(hi, 6.0)) for lo, hi, _, _ in b]
        ax.plot(c, [f for *_, f in b], "o-", color=f"C{i}", label=name)
        rows += [dict(sample=name, q_lo=lo, q_hi=hi, n=nn, frac_in_no_pulse=f) for lo, hi, nn, f in b]
    ax.set_yscale("log")
    ax.set_xlabel("true MCHit charge [p.e.]")
    ax.set_ylabel("fraction in no pulse")
    ax.legend(frameon=False)
    top = two_titles(fig, "True photons that make no pulse", "MCHit simulated, but no pulse crosses the 7-ADC threshold "
                     "around its peak")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "lost_vs_charge.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ctx.csv_path(f"{args.out_name}_lost_vs_charge"), index=False)

    # 3. pulse charge over true charge
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (name, (pul, mch, raw)) in enumerate(samples.items()):
        p = pul[pul.qtrue > 0]
        b = np.digitize(p.qtrue, Q_BINS) - 1
        for k in range(len(Q_BINS) - 1):
            r = (p.pe / p.qtrue)[b == k]
            if len(r):
                rows.append(dict(sample=name, q_lo=Q_BINS[k], q_hi=Q_BINS[k + 1], n=len(r), median=float(np.median(r)),
                                 q16=float(np.quantile(r, 0.16)), q84=float(np.quantile(r, 0.84))))
        d = pd.DataFrame([r for r in rows if r["sample"] == name])
        c = 0.5 * (d.q_lo + np.minimum(d.q_hi, 10.0))
        ax.errorbar(c, d["median"], yerr=[d["median"] - d.q16, d.q84 - d["median"]], fmt="o-", color=f"C{i}",
                    capsize=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("true charge in the pulse [p.e.]")
    ax.set_ylabel("pulse charge / true charge (median, 16-84 %)")
    ax.legend(frameon=False)
    top = two_titles(fig, "Reconstructed over true charge per pulse", "pulse charge / SPE gain of the data channel "
                     "vs the summed charge of the true MCHits whose peak lies in the pulse")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "charge_response.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ctx.csv_path(f"{args.out_name}_charge_response"), index=False)

    # 4. what each pulse inherits: #392 window vs simulated-peak rule
    rows, wv = [], {}
    for name, (pul, mch, raw) in samples.items():
        w = window_vs_peak(pul, mch)
        wv[name] = w
        lab = w[(w.first_peak != -99)]
        rows.append(dict(sample=name, n_pulses=len(w), mean_mchits_window=w.n_window.mean(),
                         mean_mchits_peak=w.n_peak.mean(), frac_window_more=(w.n_window > w.n_peak).mean(),
                         frac_first_parent_differs=(lab.first_window != lab.first_peak).mean(),
                         mean_true_pe_window=w.q_window.mean(), mean_true_pe_peak=w.q_peak.mean(),
                         mean_pulse_pe=w.pe.mean()))
    pd.DataFrame(rows).to_csv(ctx.csv_path(f"{args.out_name}_window_vs_peak"), index=False)
    fig, axes = plt.subplots(1, len(wv), figsize=(5.2 * len(wv), 4.2), squeeze=False)
    for ax, (name, w) in zip(axes[0], wv.items()):
        hi = int(min(max(w.n_window.quantile(0.99), w.n_peak.max()), 60))
        edges = np.arange(-0.5, hi + 1.5)
        for col, lab, c in [("n_window", "#392 window (t0 in start-160 .. start+400 ns)", "C0"),
                            ("n_peak", "simulated peak inside the pulse", "C3")]:
            h, _ = np.histogram(w[col].clip(upper=hi), edges)
            ax.plot(edges[:-1] + 0.5, h / max(h.sum(), 1), "o-", ms=3, color=c, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("true MCHits attached to the pulse")
        ax.set_title(name, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("fraction of pulses")
    top = two_titles(fig, "True photons each pulse inherits", "same pulses, two matching rules")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "window_vs_peak.png", dpi=130)
    plt.close(fig)

    # 5. example waveforms
    name0 = args.example_sample if args.example_sample in samples else next(iter(samples))
    pul, mch, raw = samples[name0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    rng = np.random.default_rng(1)

    def draw(ax, ps, title):
        for k, r in enumerate(ps.itertuples()):
            x = r.start + NS * np.arange(len(r.trace))
            ax.plot(x, r.trace, "o-", ms=3, color=f"C{k}", label=f"pulse {k + 1}: {r.pe:.2f} p.e. (true {r.qtrue:.2f})")
            ax.plot(r.mem_peak, np.zeros(len(r.mem_peak)) - 1.5 - 1.5 * k, "v", color=f"C{k}", ms=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("ADC above baseline")
        ax.legend(frameon=False, fontsize=8)

    one = pul[(pul.nmem == 1) & (pul.pe.between(0.8, 1.2))]
    if len(one):
        draw(axes[0, 0], one.iloc[[rng.integers(len(one))]], "one photon")
    pile = pul[pul.nmem >= 2].copy()
    pile["spread"] = [max(m) - min(m) for m in pile.mem_peak]
    pile = pile[pile.spread.between(4, 12)]
    if len(pile):
        draw(axes[0, 1], pile.iloc[[rng.integers(len(pile))]], "two or more photons in one pulse")
    ps = pul.sort_values(["ev", "ch", "start"])
    prev_end = ps.groupby(["ev", "ch"]).end.shift()
    ov = ps[ps.start <= prev_end]
    if len(ov):
        r = ov.iloc[rng.integers(len(ov))]
        pair = ps[(ps.ev == r.ev) & (ps.ch == r.ch) & (ps.end >= r.start - 40) & (ps.start <= r.end)]
        draw(axes[1, 0], pair, "overlapping windows: second pulse re-integrates the first")
    if len(raw):
        r = raw.iloc[rng.integers(len(raw))]
        x = r.t_first + NS * np.arange(len(r.samples))
        base = np.median(r.samples[:6])
        axes[1, 1].plot(x, r.samples - base, "o-", ms=3, color="C7", label=f"raw waveform, true {r.q:.2f} p.e.")
        axes[1, 1].plot([r.peak], [-1.5], "v", color="C3", ms=7)
        axes[1, 1].set_title("photon below the 7-ADC threshold: no pulse", fontsize=10)
        axes[1, 1].set_xlabel("time [ns]")
        axes[1, 1].legend(frameon=False, fontsize=8)
    top = two_titles(fig, "Example MC pulses", f"{name0}: markers = simulated peak times of the true MCHits")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "examples.png", dpi=130)
    plt.close(fig)
    print(f"[pulseshape] plots in {out}")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(prog="ambe plots pulseshape")
    ap.add_argument("--log", action="append", required=True, help="name=path of an Analyse log with PUL/MCH/RAW lines")
    ap.add_argument("--geometry", default="/exp/annie/app/users/dajana/EB_BC_TA_pr/full/configfiles/LoadGeometry/"
                    "FullTankPMTGeometry.csv")
    ap.add_argument("--params", default="/exp/annie/app/users/dajana/EB_BC_TA_pr/full/configfiles/PMTWaveformSim/"
                    "PMTWaveformLognormFit.csv")
    ap.add_argument("--data-traces", help="PrintADCTraces root file (data pulses)")
    ap.add_argument("--data-label", default="AmBe data")
    ap.add_argument("--shape-sample", default="ambe")
    ap.add_argument("--example-sample", default="ambe")
    ap.add_argument("--out-name", default="pulseshape")
    run(ctx, ap.parse_args(list(argv) if argv else []))
