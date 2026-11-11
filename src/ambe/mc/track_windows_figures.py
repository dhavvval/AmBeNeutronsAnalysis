"""
Figures, LaTeX tables and deck numbers for `ambe mc trackwindows`, drawn only from its CSVs.

    ambe mc trackwindows --config <tank.yaml> --figures-only \
        --out ambe_output/cc_neutrino_v3fixnufull_truthtag/g4trackwindows \
        --fig-out PRESENTATION_PLOTS/ccinc_g4track_windows

Writes fNN_*.pdf, tab_*.tex and deck_numbers.tex (one LaTeX macro per number quoted on the slides).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotting import set_style

SAMPLE_LABEL = {"tank": "tank, all events", "world": "world, all events"}
TEX_SAMPLE = {"tank": "tank", "world": "world"}
SAMPLE_COL = {"tank": "#2A78D6", "world": "#1BAF7A"}
MARK = {"tank": "o", "world": "D"}
SAMPLES = list(SAMPLE_LABEL)
PARTICLES = ["n", "p", "mu-", "mu+", "pi+", "pi-", "pi0", "K", "gamma", "e-", "e+", "nucleus", "nu", "other"]
PLOT_LABEL = {"n": "n", "p": "p", "mu-": "μ⁻", "mu+": "μ⁺", "pi+": "π⁺", "pi-": "π⁻", "pi0": "π⁰", "K": "K",
              "gamma": "γ", "e-": "e⁻", "e+": "e⁺", "nucleus": "nucleus", "nu": "ν", "other": "other",
              "own light": "own light", "not saved": "not saved", "primary": "primary"}
TEX_LABEL = {"n": "n", "p": "p", "mu-": "$\\mu^-$", "mu+": "$\\mu^+$", "pi+": "$\\pi^+$", "pi-": "$\\pi^-$",
             "pi0": "$\\pi^0$", "K": "K", "gamma": "$\\gamma$", "e-": "e$^-$", "e+": "e$^+$", "nucleus": "nucleus",
             "nu": "$\\nu$", "other": "other", "own light": "own light", "not saved": "not saved",
             "primary": "primary"}
WIN = ["<2", "2-10", "10-70"]


def _save(fig, d: Path, name: str):
    fig.tight_layout()
    fig.savefig(d / f"{name}.pdf")
    plt.close(fig)


def _dot(ax, labels, series, log=False, xlabel=""):
    y = np.arange(len(labels))
    off = np.linspace(-0.22, 0.22, len(series)) if len(series) > 1 else [0.0]
    for (name, vals, kw), o in zip(series, off):
        v = np.asarray(vals, float)
        m = np.isfinite(v) & ((v > 0) if log else True)
        ax.plot(v[m], y[m] + o, ls="none", ms=7, label=name, **kw)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)


def _num(v, fmt="{:,.0f}"):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def _pct(a, b, fmt="{:.1f}"):
    return "--" if not b else fmt.format(100.0 * a / b)


def _tab(path: Path, cols: str, header: list, rows: list, scale: str = "\\textwidth"):
    body = [" & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        body.append("\\midrule" if r == "MID" else " & ".join(str(x) for x in r) + " \\\\")
    path.write_text(f"\\resizebox{{{scale}}}{{!}}{{\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n"
                    + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n")


def loss_table(cr: pd.DataFrame, s: str) -> pd.DataFrame:
    d = cr[cr["sample"] == s].pivot_table(index="particle", columns="window", values="n", aggfunc="sum").fillna(0)
    d = d.reindex(columns=WIN + [">=70"], fill_value=0)
    d = d.reindex([p for p in PARTICLES if p in d.index])
    d.loc["all tracks"] = d.sum()
    d["2-70"] = d["2-10"] + d["10-70"]
    d["lost"] = d["2-10"]
    d["loss_pct"] = 100 * d["2-10"] / d["2-70"].replace(0, np.nan)
    tot = d.loc["all tracks"]
    d["share_2_70"] = 100 * d["2-70"] / tot["2-70"]
    d["share_10_70"] = 100 * d["10-70"] / tot["10-70"]
    d["share_change"] = d.share_10_70 - d.share_2_70
    return d


def light_loss_table(li: pd.DataFrame, s: str) -> pd.DataFrame:
    d = li[(li["sample"] == s) & (li.light_mask > 0)]
    rows = {}
    for p, g in d.groupby("particle"):
        m = g.light_mask.to_numpy()
        n = g.n.to_numpy()
        rows[p] = {"<2": n[(m & 1) > 0].sum(), "2-10": n[(m & 2) > 0].sum(), "10-70": n[(m & 4) > 0].sum(),
                   "2-70": n[(m & 6) > 0].sum(), "lost": n[((m & 2) > 0) & ((m & 4) == 0)].sum(),
                   "with light": n.sum(), "prompt_late": n[((m & 1) > 0) & ((m & 4) > 0)].sum()}
    t = pd.DataFrame(rows).T.reindex([p for p in PARTICLES if p in rows])
    t.loc["all tracks"] = t.sum()
    t["loss_pct"] = 100 * t.lost / t["2-70"].replace(0, np.nan)
    return t


def make_figures(csv_dir: Path, fig_dir: Path) -> Path:
    set_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda n: pd.read_csv(csv_dir / f"g4tw_{n}.csv")
    ev, cr, cp, ch = rd("events"), rd("creation"), rd("creation_process"), rd("creation_hist")
    lf, li, ca, cap, dg = rd("lifetime"), rd("light"), rd("carriers"), rd("captures"), rd("digits")
    nev = ev.set_index("sample").n_events
    L = {s: loss_table(cr, s) for s in SAMPLES}
    LL = {s: light_loss_table(li, s) for s in SAMPLES}
    parts = [p for p in PARTICLES if p in L["tank"].index and L["tank"].loc[p, "2-70"] > 0]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    _dot(ax, [PLOT_LABEL[p] for p in parts] + ["all tracks"],
         [(SAMPLE_LABEL[s], L[s].reindex(parts + ["all tracks"]).loss_pct, dict(marker=MARK[s], color=SAMPLE_COL[s]))
          for s in SAMPLES], xlabel="tracks created at 2–10 µs, as % of those created at 2–70 µs")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Geant4 tracks lost when the window starts at 10 µs instead of 2 µs")
    _save(fig, fig_dir, "f01_loss_by_particle")

    for s in SAMPLES:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        lt = L[s].reindex(parts)
        _dot(ax, [PLOT_LABEL[p] for p in parts],
             [(f"created {w} µs", lt[w] / nev[s], dict(marker=m, color=c))
              for w, m, c in (("<2", "o", "#555555"), ("2-10", "s", "#EDA100"), ("10-70", "D", "#2A78D6"))],
             log=True, xlabel="saved Geant4 tracks per event")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"{SAMPLE_LABEL[s].capitalize()}: saved Geant4 tracks per event by creation window")
        _save(fig, fig_dir, f"f02_tracks_per_event_{s}")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    _dot(ax, [PLOT_LABEL[p] for p in parts],
         [(SAMPLE_LABEL[s], L[s].reindex(parts).share_change, dict(marker=MARK[s], color=SAMPLE_COL[s]))
          for s in SAMPLES], xlabel="share of all tracks at 10–70 µs minus share at 2–70 µs [percentage points]")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Which particles gain or lose share when the window starts at 10 µs")
    _save(fig, fig_dir, "f03_share_change")

    cols = {"e-": "#2A78D6", "gamma": "#EDA100", "nucleus": "#1BAF7A", "e+": "#EB6834", "nu": "#4A3AA7",
            "n": "#C62828", "p": "#555555"}
    for s in SAMPLES:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for p, c in cols.items():
            g = ch[(ch["sample"] == s) & (ch.particle == p)].sort_values("lo")
            g = g[g.lo >= 1.0]
            if g.n.sum() == 0:
                continue
            ax.plot(0.5 * (g.lo + g.hi), g.n / nev[s], "o-", ms=3, color=c, label=PLOT_LABEL[p])
        ax.set_yscale("log")
        ax.set_xlabel("creation time t₀ [µs] (1 µs bins, from 1 µs)")
        ax.set_ylabel("tracks per event per µs")
        ax.legend(frameon=False, fontsize=9, ncol=2)
        ax.set_title(f"{SAMPLE_LABEL[s].capitalize()}: when the delayed Geant4 tracks are created")
        _save(fig, fig_dir, f"f04_creation_time_{s}")

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        for ax, w in zip(axes, ("2-10", "10-70")):
            d = cp[(cp["sample"] == s) & (cp.window == w)].sort_values("n", ascending=False)
            tot = d.n.sum()
            d = d.head(12)
            lab = [f"{PLOT_LABEL.get(r.particle, r.particle)} ← {r.sproc} ({PLOT_LABEL.get(r.parent_particle, r.parent_particle)})"
                   for r in d.itertuples()]
            _dot(ax, lab, [("", 100 * d.n / tot, dict(marker="o", color=SAMPLE_COL[s]))],
                 xlabel=f"% of tracks created at {w} µs")
            ax.set_title(f"created at {w} µs ({tot / nev[s]:.1f} per event)")
        fig.suptitle(f"{SAMPLE_LABEL[s].capitalize()}: what the delayed tracks are — particle ← creator process (parent)")
        _save(fig, fig_dir, f"f05_what_is_lost_{s}")

    lparts = [p for p in parts if p in LL["tank"].index and LL["tank"].loc[p, "2-70"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5.0))
    _dot(ax, [PLOT_LABEL[p] for p in lparts] + ["all tracks"],
         [(SAMPLE_LABEL[s], LL[s].reindex(lparts + ["all tracks"]).loss_pct, dict(marker=MARK[s], color=SAMPLE_COL[s]))
          for s in SAMPLES], xlabel="tracks whose light is only at 2–10 µs, as % of tracks with light at 2–70 µs")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Tracks whose light (own or descendants') is lost with a 10 µs start")
    _save(fig, fig_dir, "f06_light_loss")

    for s in SAMPLES:
        fig, ax = plt.subplots(figsize=(8, 5.0))
        life = lf[lf["sample"] == s]
        tot = life.groupby("particle").n.sum()
        pp = [p for p in parts if p in tot.index]
        sel = lambda a, b: (life[(life.start_window == a) & (life.end_window == b)].groupby("particle").n.sum()
                            .reindex(pp).fillna(0))
        _dot(ax, [PLOT_LABEL[p] for p in pp],
             [("created <2 µs, ends 2–10 µs", 100 * sel("<2", "2-10") / tot[pp], dict(marker="s", color="#EDA100")),
              ("created <2 µs, ends 10–70 µs", 100 * sel("<2", "10-70") / tot[pp], dict(marker="D", color="#2A78D6")),
              ("created <2 µs, ends ≥70 µs", 100 * sel("<2", ">=70") / tot[pp], dict(marker="o", color="#555555"))],
             log=True, xlabel="% of all tracks of this particle")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"{SAMPLE_LABEL[s].capitalize()}: one Geant4 track alive from the prompt window into a later one")
        _save(fig, fig_dir, f"f07_lifetime_{s}")

    fig, ax = plt.subplots(figsize=(8, 5.0))
    _dot(ax, [PLOT_LABEL[p] for p in lparts],
         [(SAMPLE_LABEL[s], 100 * LL[s].reindex(lparts).prompt_late / LL[s].reindex(lparts)["with light"],
           dict(marker=MARK[s], color=SAMPLE_COL[s])) for s in SAMPLES], log=True,
         xlabel="% of tracks with light that have light both <2 µs and at 10–70 µs")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Same track, light early and late (own light or its descendants')")
    _save(fig, fig_dir, "f08_light_prompt_late")

    for s in SAMPLES:
        c = ca[(ca["sample"] == s) & (ca.group == "prompt + late")]
        cparts = [p for p in lparts if c[c.particle == p].n.sum() >= (10 if s == "tank" else 3)]
        carriers = list(c.groupby("carrier").n.sum().sort_values(ascending=False).index[:6])
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        tot = c.groupby("particle").n.sum()
        _dot(ax, [PLOT_LABEL[p] for p in cparts],
             [(PLOT_LABEL.get(k, k), 100 * c[c.carrier == k].groupby("particle").n.sum().reindex(cparts).fillna(0) / tot[cparts],
               dict(marker=m, color=col)) for k, m, col in
              zip(carriers, ["o", "s", "D", "^", "v", "P"], ["#C62828", "#2A78D6", "#1BAF7A", "#EDA100", "#4A3AA7", "#555555"])],
             xlabel="% of the tracks with light <2 µs and at 10–70 µs")
        ax.legend(frameon=False, fontsize=9, title="10–70 µs light carried by")
        ax.set_title(f"{SAMPLE_LABEL[s].capitalize()}: which daughter carries a track's 10–70 µs light")
        _save(fig, fig_dir, f"f09_late_carriers_{s}")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = ["<2", "2-10", "10-66.5", "66.5-70", ">=70"]
    for s in SAMPLES:
        d = dg[dg["sample"] == s].set_index("window").reindex(bins).n.fillna(0)
        ax.plot(range(len(bins)), d / nev[s], ls="none", marker=MARK[s], ms=8, color=SAMPLE_COL[s], label=SAMPLE_LABEL[s])
    ax.set_yscale("log")
    ax.set_xticks(range(len(bins)), [f"{b} µs" for b in bins])
    ax.set_ylabel("true photon digits per event")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("True tank photons per time bin; the tree's MCHits stop at 66.5 µs")
    _save(fig, fig_dir, "f10_true_photons")

    write_tables(fig_dir, L, LL, cp, lf, ca, cap, dg, nev, ev)
    N = numbers(L, LL, cp, lf, ca, cap, dg, nev, ev)
    if (csv_dir / "g4tw_units.parquet").exists():
        N.update(background_section(csv_dir, fig_dir, nev))
    with open(fig_dir / "deck_numbers.tex", "w") as fh:
        fh.write(f"% generated by ambe.mc.track_windows_figures from {csv_dir}\n")
        for k, v in N.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"[trackwindows] {len(list(fig_dir.glob('f*.pdf')))} figures, {len(list(fig_dir.glob('tab_*.tex')))} tables "
          f"and deck_numbers.tex in {fig_dir}")
    return fig_dir


def numbers(L, LL, cp, lf, ca, cap, dg, nev, ev) -> dict:
    N = {}
    e = ev.set_index("sample")
    for s, k in (("tank", "Tank"), ("world", "World")):
        t, ll = L[s], LL[s]
        N[f"twEvents{k}"] = f"{int(nev[s]):,}"
        N[f"twLossAll{k}"] = f"{t.loc['all tracks', 'loss_pct']:.1f}"
        N[f"twLightLossAll{k}"] = f"{ll.loc['all tracks', 'loss_pct']:.1f}"
        c = cap[cap["sample"] == s].set_index("window")
        N[f"twCapLoss{k}"] = _pct(c.loc["2-10", "n_capture_nuclei"],
                                  c.loc["2-10", "n_capture_nuclei"] + c.loc["10-70", "n_capture_nuclei"])
        for p, q in (("e-", "Em"), ("gamma", "Gamma"), ("nucleus", "Nucleus"), ("e+", "Ep"), ("nu", "Nu"),
                     ("n", "N"), ("p", "P")):
            N[f"twLoss{q}{k}"] = f"{t.loc[p, 'loss_pct']:.1f}" if p in t.index else "--"
            N[f"twShareChange{q}{k}"] = f"{t.loc[p, 'share_change']:+.1f}" if p in t.index else "--"
        nr = t.loc["n"]
        N[f"twNBornLate{k}"] = _pct(nr["2-70"], nr[["<2", "2-10", "10-70", ">=70"]].sum())
        dn = cp[(cp["sample"] == s) & (cp.particle == "n")]
        N[f"twNLatePrimary{k}"] = _pct(dn[dn.parent_particle == "primary"].n.sum(), dn.n.sum(), "{:.0f}")
        N[f"twNLateMuCap{k}"] = _pct(dn[dn.sproc == "muMinusCaptureAtRest"].n.sum(), dn.n.sum(), "{:.0f}")
        c10 = cap[cap["sample"] == s].set_index("window").loc["10-70", "n_capture_nuclei"]
        d10 = cp[(cp["sample"] == s) & (cp.window == "10-70")]
        N[f"twTracksPerCap{k}"] = f"{d10.n.sum() / c10:.0f}"
        N[f"twComptPerCap{k}"] = f"{d10[(d10.particle == 'e-') & (d10.sproc == 'compt')].n.sum() / c10:.0f}"
        N[f"twGammaPerCap{k}"] = f"{d10[(d10.particle == 'gamma') & (d10.sproc == 'nCapture')].n.sum() / c10:.1f}"
        N[f"twTracksPerEvent{k}"] = f"{t.loc['all tracks', ['<2', '2-10', '10-70', '>=70']].sum() / nev[s]:.0f}"
        N[f"twLateTracksPerEvent{k}"] = f"{t.loc['all tracks', '10-70'] / nev[s]:.1f}"
        N[f"twMidTracksPerEvent{k}"] = f"{t.loc['all tracks', '2-10'] / nev[s]:.1f}"
        d = cp[(cp["sample"] == s) & (cp.window == "2-10")]
        N[f"twLostComptonFrac{k}"] = _pct(d[(d.particle == "e-") & (d.sproc == "compt")].n.sum(), d.n.sum(), "{:.0f}")
        life = lf[lf["sample"] == s]
        nn, mm, mp = (life[life.particle == x] for x in ("n", "mu-", "mu+"))
        N[f"twNLifeLate{k}"] = _pct(nn[(nn.start_window == "<2") & (nn.end_window == "10-70")].n.sum(), nn.n.sum())
        N[f"twNLifeMid{k}"] = _pct(nn[(nn.start_window == "<2") & (nn.end_window == "2-10")].n.sum(), nn.n.sum())
        N[f"twMumEndPrompt{k}"] = _pct(mm[mm.end_window == "<2"].n.sum(), mm.n.sum())
        N[f"twMupEndMid{k}"] = _pct(mp[mp.end_window == "2-10"].n.sum(), mp.n.sum())
        for p, q in (("n", "N"), ("p", "P"), ("mu-", "Mum"), ("pi+", "Pip"), ("pi-", "Pim"), ("e-", "Em"),
                     ("gamma", "Gamma")):
            N[f"twPromptLate{q}{k}"] = _pct(ll.loc[p, "prompt_late"], ll.loc[p, "with light"], "{:.2f}") if p in ll.index else "--"
        c = ca[(ca["sample"] == s) & (ca.group == "prompt + late")]
        for p, q in (("mu-", "Mum"), ("p", "P"), ("pi+", "Pip"), ("n", "N")):
            g = c[c.particle == p]
            N[f"twCarrierN{q}{k}"] = _pct(g[g.carrier == "n"].n.sum(), g.n.sum(), "{:.0f}")
        d = dg[dg["sample"] == s].set_index("window").n
        N[f"twPhotonsCut{k}"] = f"{int(d.get('66.5-70', 0)):,}"
        N[f"twPhotonsCutFrac{k}"] = _pct(d.get("66.5-70", 0), d.get("10-66.5", 0) + d.get("66.5-70", 0))
        N[f"twNeutronEndFixed{k}"] = f"{int(e.loc[s, 'capturing_neutrons_end_time_replaced']):,}"
    N["twLineageMissing"] = f"{int(e.lineage_steps_not_in_tracks.sum()):,}"
    N["twLineageSteps"] = f"{int(e.lineage_steps.sum()):,}"
    return N


def write_tables(out, L, LL, cp, lf, ca, cap, dg, nev, ev):
    for s in SAMPLES:
        rows = []
        for p, r in L[s].iterrows():
            if p != "all tracks" and r[WIN + [">=70"]].sum() == 0:
                continue
            if p == "all tracks":
                rows.append("MID")
            rows.append([TEX_LABEL.get(p, p)] + [_num(r[w]) for w in WIN] + [_num(r["2-70"]), _num(r["lost"]),
                        _num(r.loss_pct, "{:.1f}\\,\\%"), _num(r.share_2_70, "{:.1f}"), _num(r.share_10_70, "{:.1f}"),
                        _num(r.share_change, "{:+.1f}")])
        _tab(out / f"tab_loss_{s}.tex", "l rrr rr r rrr",
             ["particle", "$<$\\,2", "2--10", "10--70", "2--70", "lost", "lost / 2--70", "share 2--70 [\\%]",
              "share 10--70 [\\%]", "change [pp]"], rows)

    rows = []
    for s in SAMPLES:
        c = cap[cap["sample"] == s].set_index("window")
        for col, lab in (("n_capture_nuclei", "Geant4 tracks (nucleus made by nCapture)"),
                         ("n_trueNeutCap", "\\texttt{trueNeutCap} branch")):
            v = {w: (c.loc[w, col] if w in c.index else 0) for w in WIN}
            rows.append([TEX_SAMPLE[s] if col == "n_capture_nuclei" else "", lab] + [_num(v[w]) for w in WIN]
                        + [_num(v["2-10"] + v["10-70"]), _pct(v["2-10"], v["2-10"] + v["10-70"]) + "\\,\\%"])
    _tab(out / "tab_captures.tex", "ll rrr r r", ["sample", "counted from", "$<$\\,2", "2--10", "10--70", "2--70",
                                                   "lost by 10\\,$\\mu$s"], rows)

    for s in SAMPLES:
        for w in ("2-10", "10-70"):
            d = cp[(cp["sample"] == s) & (cp.window == w)].sort_values("n", ascending=False)
            tot = d.n.sum()
            rows = [[TEX_LABEL.get(r.particle, r.particle), f"\\texttt{{{r.sproc}}}",
                     TEX_LABEL.get(r.parent_particle, r.parent_particle), _num(r.n), _num(r.n / nev[s], "{:.3f}"),
                     _pct(r.n, tot) + "\\,\\%"] for r in d.head(10).itertuples()]
            rows += ["MID", ["all", "", "", _num(tot), _num(tot / nev[s], "{:.2f}"), "100\\,\\%"]]
            _tab(out / f"tab_what_{s}_{w.replace('-', '_')}.tex", "lllrrr",
                 ["particle", "creator process", "parent", "tracks", "per event", "share"], rows)

    for s in SAMPLES:
        rows = []
        for p, r in LL[s].iterrows():
            if p == "all tracks":
                rows.append("MID")
            rows.append([TEX_LABEL.get(p, p), _num(r["with light"])] + [_num(r[w]) for w in WIN]
                        + [_num(r["2-70"]), _num(r.lost), _num(r.loss_pct, "{:.1f}\\,\\%"),
                           _num(r.prompt_late), _pct(r.prompt_late, r["with light"], "{:.2f}")])
        _tab(out / f"tab_light_{s}.tex", "l r rrr rr r rr",
             ["particle", "with light", "light $<$\\,2", "2--10", "10--70", "2--70", "lost", "lost / 2--70",
              "$<$\\,2 and 10--70", "\\%"], rows)

    for s in SAMPLES:
        life = lf[lf["sample"] == s]
        rows = []
        for p in [p for p in PARTICLES if p in set(life.particle)]:
            g = life[life.particle == p]
            tot = g.n.sum()
            v = lambda a, b: g[(g.start_window == a) & (g.end_window == b)].n.sum()
            rows.append([TEX_LABEL[p], _num(tot), _pct(v("<2", "<2"), tot), _pct(v("<2", "2-10"), tot),
                         _pct(v("<2", "10-70"), tot), _pct(v("<2", ">=70"), tot), _pct(v("2-10", "2-10"), tot),
                         _pct(v("2-10", "10-70") + v("2-10", ">=70"), tot), _pct(v("10-70", "10-70"), tot),
                         _pct(v("10-70", ">=70") + v(">=70", ">=70"), tot)])
        _tab(out / f"tab_lifetime_{s}.tex", "lr rrrr rr rr",
             ["particle", "tracks", "$<$2$\\to<$2", "$<$2$\\to$2--10", "$<$2$\\to$10--70", "$<$2$\\to\\geq$70",
              "2--10$\\to$2--10", "2--10$\\to$later", "10--70$\\to$10--70", "later"], rows)

    for s in SAMPLES:
        d = ca[(ca["sample"] == s) & (ca.group == "prompt + late")]
        tot = d.groupby("particle").n.sum()
        rows = []
        for p in [p for p in PARTICLES if p in tot.index and tot[p] >= (10 if s == "tank" else 3)]:
            g = d[d.particle == p].sort_values("n", ascending=False)
            rows.append([TEX_LABEL[p], _num(tot[p]),
                         ", ".join(f"{TEX_LABEL.get(r.carrier, r.carrier)} {100 * r.n / tot[p]:.0f}\\,\\%"
                                   for r in g.head(4).itertuples())])
        _tab(out / f"tab_carriers_{s}.tex", "lrl", ["particle", "tracks with light $<$\\,2 and 10--70",
                                                     "10--70\\,$\\mu$s light carried by (daughter)"], rows)

    bins = ["<2", "2-10", "10-66.5", "66.5-70", ">=70"]
    rows = []
    for s in SAMPLES:
        d = dg[dg["sample"] == s].set_index("window").reindex(bins).n.fillna(0)
        rows.append([TEX_SAMPLE[s]] + [_num(d[b] / nev[s], "{:.2f}") for b in bins]
                    + [_pct(d["66.5-70"], d["10-66.5"] + d["66.5-70"], "{:.2f}") + "\\,\\%"])
    _tab(out / "tab_photons.tex", "l rrrrr r", ["photons per event", "$<$\\,2", "2--10", "10--66.5", "66.5--70",
                                                 "$\\geq$\\,70", "66.5--70 / 10--70"], rows)

    e = ev.set_index("sample")
    _tab(out / "tab_samples.tex", "lrrr", ["sample", "events", "saved Geant4 tracks", "per event"],
         [[TEX_SAMPLE[s], _num(e.loc[s, "n_events"]), _num(e.loc[s, "n_tracks"]),
           _num(e.loc[s, "n_tracks"] / e.loc[s, "n_events"], "{:.0f}")] for s in SAMPLES], "0.8\\textwidth")


# ---------------------------------------------------------------------------------------------------------
# Background to neutron captures: sources, light units and their separation (g4tw_origin_*, g4tw_units)

UNIT_LABEL = {"ncap": "n capture, ν-interaction n", "ncap_mu": "n capture, μ⁻-capture n",
              "ncap_late": "n capture, n entering late", "ncap_out": "n capture outside tank",
              "mu_decay": "μ± decay", "mu_stop": "μ⁻ stop, no n", "gamma_late": "γ entering late",
              "other_late": "other particle entering late", "n_scatter": "n scatter, no capture",
              "noise": "dark noise", "unsaved": "parent not saved"}
UNIT_TEX = {"ncap": "n capture, $\\nu$-interaction n", "ncap_mu": "n capture, $\\mu^-$-capture n",
            "ncap_late": "n capture, n entering late", "ncap_out": "n capture outside tank",
            "mu_decay": "$\\mu^\\pm$ decay", "mu_stop": "$\\mu^-$ stop, no n", "gamma_late": "$\\gamma$ entering late",
            "other_late": "other particle entering late", "n_scatter": "n scatter, no capture",
            "noise": "dark noise", "unsaved": "parent not saved"}
UNIT_COL = {"ncap": "#2A78D6", "ncap_mu": "#4A3AA7", "ncap_late": "#1BAF7A", "ncap_out": "#555555",
            "mu_decay": "#C62828", "mu_stop": "#EB6834", "gamma_late": "#EDA100", "other_late": "#8C6D31",
            "n_scatter": "#999999", "noise": "#000000", "unsaved": "#BBBBBB"}
SIGNAL = "ncap"
FEATURES = [("n_photons", "true photons", True), ("n_tubes_cf", "PMTs within ±10 ns", True),
            ("charge_balance", "charge balance", False), ("t_rms_ns", "photon time rms [ns]", False),
            ("t_unit", "source time [µs]", False), ("d_wall_cm", "vertex distance to tank wall [cm]", False)]
FEATURE_TEX = {"n_photons": "photons", "n_tubes_cf": "PMTs $\\pm$10\\,ns", "charge_balance": "charge bal.",
               "t_rms_ns": "time rms", "t_unit": "time", "d_wall_cm": "dist.\\ to wall"}


def light_class(origin: pd.Series) -> pd.Series:
    src = origin.str.slice(len("n capture: "))
    cap = origin.str.startswith("n capture")
    return pd.Series(np.select(
        [cap & (src == "mu- stop"), cap & (src == "late primary"), cap, origin == "mu decay", origin == "mu- stop",
         origin == "late primary", origin == "n scatter", origin == "dark noise"],
        ["ncap_mu", "ncap_late", "ncap", "mu_decay", "mu_stop", "other_late", "n_scatter", "noise"], "unsaved"),
        index=origin.index)


def unit_class(u: pd.DataFrame) -> pd.Series:
    c = light_class(u.origin)
    cap = c.isin(["ncap", "ncap_mu", "ncap_late"])
    c[cap & (u.d_wall_cm <= 0)] = "ncap_out"
    c[(u.origin == "late primary") & (u.pdg == 22)] = "gamma_late"
    return c


def _auc(sig: np.ndarray, bkg: np.ndarray) -> float:
    from scipy.stats import rankdata
    sig, bkg = sig[np.isfinite(sig)], bkg[np.isfinite(bkg)]
    if len(sig) == 0 or len(bkg) == 0:
        return np.nan
    r = rankdata(np.concatenate([sig, bkg]))
    return (r[len(sig):].sum() - len(bkg) * (len(bkg) + 1) / 2) / (len(sig) * len(bkg))


def _kept_at(sig: np.ndarray, bkg: np.ndarray, eff: float = 0.99) -> float:
    """Background fraction kept by the better one-sided cut that keeps `eff` of the signal."""
    sig, bkg = sig[np.isfinite(sig)], bkg[np.isfinite(bkg)]
    if len(sig) == 0 or len(bkg) == 0:
        return np.nan
    hi, lo = np.quantile(sig, eff), np.quantile(sig, 1 - eff)
    return min((bkg <= hi).mean(), (bkg >= lo).mean())


def background_section(csv_dir: Path, out: Path, nev) -> dict:
    N = {}
    ol = pd.read_csv(csv_dir / "g4tw_origin_light.csv")
    ne = pd.read_csv(csv_dir / "g4tw_neutron_end.csv")
    ot = pd.read_csv(csv_dir / "g4tw_origin_tracks.csv")
    u = pd.read_parquet(csv_dir / "g4tw_units.parquet")
    u["cls"] = unit_class(u)
    u["t_unit"] = u.t_unit / 1000.0
    u["vis"] = u.n_tubes_cf >= 5
    ol["cls"] = light_class(ol.origin)

    # photons by source
    for s in SAMPLES:
        d = ol[ol["sample"] == s].pivot_table(index="cls", columns="window", values="n_photons", aggfunc="sum").fillna(0)
        d = d.reindex(columns=["<2", "2-10", "10-66.5", "66.5-70", ">=70"], fill_value=0)
        d = d[(d["2-10"] + d["10-66.5"]) > 0].sort_values("10-66.5", ascending=False)
        late = d["10-66.5"] + d["66.5-70"]
        rows = [[UNIT_TEX[c], _num(r["2-10"] / nev[s], "{:.2f}"), _pct(r["2-10"], d["2-10"].sum()) + "\\,\\%",
                 _num(r["10-66.5"] / nev[s], "{:.2f}"), _pct(r["10-66.5"], d["10-66.5"].sum()) + "\\,\\%",
                 _pct(r["2-10"], r["2-10"] + late[c]) + "\\,\\%"] for c, r in d.iterrows()]
        _tab(out / f"tab_bkg_light_{s}.tex", "l rr rr r",
             ["source of the photon", "2--10: per event", "share", "10--66.5: per event", "share",
              "lost by 10\\,$\\mu$s"], rows)
        k = TEX_SAMPLE[s].capitalize()
        caps = d.loc[[c for c in ("ncap", "ncap_mu", "ncap_late") if c in d.index], "10-66.5"].sum()
        N[f"twBkgCapLightShare{k}"] = _pct(caps, d["10-66.5"].sum())
        mu = d.loc[[c for c in ("mu_decay", "mu_stop") if c in d.index]]
        N[f"twBkgMuLightShareMid{k}"] = _pct(mu["2-10"].sum(), d["2-10"].sum(), "{:.0f}")
        N[f"twBkgMuLightLost{k}"] = _pct(mu["2-10"].sum(), mu["2-10"].sum() + mu["10-66.5"].sum() + mu["66.5-70"].sum())
        N[f"twBkgCapLightLost{k}"] = _pct(d.loc["ncap", "2-10"], d.loc["ncap", ["2-10", "10-66.5", "66.5-70"]].sum())
        N[f"twBkgNoiseLightShare{k}"] = _pct(d.loc["noise", "10-66.5"], d["10-66.5"].sum()) if "noise" in d.index else "0"
    t = ot[ot.window == "10-70"].assign(cls=light_class(ot[ot.window == "10-70"].origin))
    for s in SAMPLES:
        g = t[t["sample"] == s]
        N[f"twTracksFromCapture{TEX_SAMPLE[s].capitalize()}"] = _pct(
            g[g.cls.isin(["ncap", "ncap_mu", "ncap_late"])].n.sum(), g.n.sum())

    # neutron end reasons
    lab = {"capture": "captured", "neutronInelastic": "neutronInelastic (new n tracks)",
           "stops outside tank": "leaves the tank, not captured", "other": "other"}
    for s in SAMPLES:
        d = ne[ne["sample"] == s].pivot_table(index="reason", columns="end_window", values="n", aggfunc="sum").fillna(0)
        d = d.reindex(columns=LIFE_ORDER, fill_value=0)
        tot = d.values.sum()
        rows = [[lab.get(r, r)] + [_pct(d.loc[r, w], tot) for w in LIFE_ORDER] + [_pct(d.loc[r].sum(), tot)]
                for r in ["capture", "neutronInelastic", "stops outside tank", "other"] if r in d.index]
        rows += ["MID", ["all neutron tracks"] + [_pct(d[w].sum(), tot) for w in LIFE_ORDER] + ["100.0"]]
        _tab(out / f"tab_neutron_end_{s}.tex", "l rrrr r",
             ["how the neutron track ends [\\% of all n tracks]", "$<$\\,2", "2--10", "10--70", "$\\geq$\\,70", "all"],
             rows, "0.9\\textwidth")
        k = TEX_SAMPLE[s].capitalize()
        e2 = d["<2"]
        N[f"twNEndInel{k}"] = _pct(e2.get("neutronInelastic", 0), tot)
        N[f"twNEndOut{k}"] = _pct(e2.get("stops outside tank", 0), tot)
        N[f"twNEndCap{k}"] = _pct(e2.get("capture", 0), tot)
        N[f"twNEndPrompt{k}"] = _pct(e2.sum(), tot)
        N[f"twNCapAll{k}"] = _pct(d.loc["capture"].sum(), tot) if "capture" in d.index else "0"

    # units per event and visibility
    order = [c for c in UNIT_LABEL if c in set(u.cls)]
    for s in SAMPLES:
        d = u[u["sample"] == s]
        g = d.groupby(["cls", "w"]).agg(n=("vis", "size"), v=("vis", "sum")).unstack("w", fill_value=0)
        g = g.reindex(index=[c for c in order if c in g.index])
        n2, v1, v2 = g[("n", 2)], g[("v", 1)], g[("v", 2)]
        rows = [[UNIT_TEX[c], _num(n2[c] / nev[s], "{:.4f}"), _num(v2[c] / nev[s], "{:.4f}"),
                 _pct(v2[c], v2.sum(), "{:.2f}") + "\\,\\%", _pct(v1[c], v1[c] + v2[c]) + ("\\,\\%" if v1[c] + v2[c] else "")]
                for c in g.index if n2[c] + v1[c] + v2[c] > 0]
        rows += ["MID", ["all", _num(n2.sum() / nev[s], "{:.4f}"), _num(v2.sum() / nev[s], "{:.4f}"), "100\\,\\%",
                         _pct(v1.sum(), v1.sum() + v2.sum()) + "\\,\\%"]]
        _tab(out / f"tab_units_{s}.tex", "l rr r r",
             ["source unit at 10--70\\,$\\mu$s", "per event", "visible per event", "share of visible",
              "visible lost by 10\\,$\\mu$s"], rows)
        k = TEX_SAMPLE[s].capitalize()
        capv = v2.reindex(["ncap", "ncap_mu", "ncap_late", "ncap_out"]).fillna(0).sum()
        N[f"twVisCapShare{k}"] = _pct(capv, v2.sum())
        N[f"twVisMuDecayShare{k}"] = _pct(v2.get("mu_decay", 0), v2.sum(), "{:.2f}")
        N[f"twVisGammaLateShare{k}"] = _pct(v2.get("gamma_late", 0), v2.sum(), "{:.2f}")
        N[f"twVisOtherCapShare{k}"] = _pct(v2.reindex(["ncap_mu", "ncap_late", "ncap_out"]).fillna(0).sum(), v2.sum())
        N[f"twVisMuStopShare{k}"] = _pct(v2.get("mu_stop", 0), v2.sum(), "{:.2f}")
        N[f"twVisMuDecayLost{k}"] = _pct(v1.get("mu_decay", 0), v1.get("mu_decay", 0) + v2.get("mu_decay", 0))
        N[f"twVisCapLost{k}"] = _pct(v1.get("ncap", 0), v1.get("ncap", 0) + v2.get("ncap", 0))
        nz = d[d.cls == "noise"]
        N[f"twNoiseMaxPMTs{k}"] = f"{int(nz.n_tubes_cf.max()) if len(nz) else 0}"

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    vis2 = u[(u.w == 2) & u.vis]
    lab_order = [c for c in order if c != "noise" and (vis2.cls == c).any()]
    _dot(ax, [UNIT_LABEL[c] for c in lab_order],
         [(SAMPLE_LABEL[s], u[(u["sample"] == s) & (u.w == 2) & u.vis].groupby("cls").size().reindex(lab_order) / nev[s],
           dict(marker=MARK[s], color=SAMPLE_COL[s])) for s in SAMPLES], log=True,
         xlabel="visible source units at 10–70 µs per event (≥ 5 PMTs within ±10 ns)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("What makes visible light at 10–70 µs, per event")
    _save(fig, out, "f11_units_per_event")

    # visibility of captures by nucleus and position
    cap = u[u.cls.isin(["ncap", "ncap_mu", "ncap_late", "ncap_out"])].copy()
    cap["nucleus"] = np.select([cap.capture_Z == 64, cap.capture_Z == 1], ["Gd", "H"], "other")
    cap["where"] = np.where(cap.d_wall_cm > 0, "inside tank", "outside tank")
    rows = []
    for s in SAMPLES:
        d = cap[(cap["sample"] == s) & (cap.w == 2)]
        tot, first = len(d), True
        for (w, nuc), g in d.groupby(["where", "nucleus"]):
            if len(g) < 0.005 * tot:
                continue
            rows.append([TEX_SAMPLE[s] if first else "", w, nuc, _num(len(g)), _pct(len(g), tot),
                         _num(g.n_photons.median(), "{:.0f}"), _pct(g.vis.sum(), len(g))])
            first = False
        rows.append("MID")
    _tab(out / "tab_capture_visibility.tex", "lll rr rr",
         ["sample", "capture vertex", "nucleus", "captures at 10--70", "\\%", "median photons", "visible [\\%]"],
         rows[:-1], "0.9\\textwidth")
    for s in SAMPLES:
        d = cap[(cap["sample"] == s) & (cap.w == 2) & (cap.cls == "ncap")]
        k = TEX_SAMPLE[s].capitalize()
        N[f"twCapVisible{k}"] = _pct(d.vis.sum(), len(d))
        gd, h = d[d.nucleus == "Gd"], d[d.nucleus == "H"]
        N[f"twCapVisibleGd{k}"] = _pct(gd.vis.sum(), len(gd), "{:.0f}")
        N[f"twCapVisibleH{k}"] = _pct(h.vis.sum(), len(h), "{:.0f}")
        z0 = d[d.n_photons == 0]
        N[f"twCapZeroPhot{k}"] = _pct(len(z0), len(d), "{:.0f}")
        N[f"twCapZeroPhotWall{k}"] = _num(z0.d_wall_cm.median(), "{:.0f}")
        N[f"twCapLightWall{k}"] = _num(d[d.n_photons > 0].d_wall_cm.median(), "{:.0f}")

    # feature distributions, visible units at 10-70 us (time panel: 2-70 us)
    bins = {"n_photons": np.logspace(np.log10(5), np.log10(3000), 30), "n_tubes_cf": np.arange(5, 125, 4),
            "charge_balance": np.linspace(0, 1, 26), "t_rms_ns": np.linspace(0, 40, 26),
            "t_unit": np.linspace(2, 70, 35), "d_wall_cm": np.linspace(-50, 200, 26)}
    for s in SAMPLES:
        d = u[(u["sample"] == s) & u.vis]
        show = [c for c in order if c != "noise" and ((d.cls == c) & (d.w == 2)).sum() >= 20]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
        for ax, (f, xl, logx) in zip(axes.flat, FEATURES):
            for c in show:
                x = d[(d.cls == c) & ((d.w >= 1) if f == "t_unit" else (d.w == 2))][f].dropna().to_numpy()
                h, e = np.histogram(x, bins=bins[f])
                if h.sum() == 0:
                    continue
                m = h > 0
                ax.plot(0.5 * (e[1:] + e[:-1])[m], h[m] / len(x), "o-", ms=3, color=UNIT_COL[c],
                        label=f"{UNIT_LABEL[c]} ({len(x):,})")
            if logx:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("fraction of units")
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{SAMPLE_LABEL[s].capitalize()}: visible source units at 10–70 µs (time: 2–70 µs)")
        _save(fig, out, f"f12_unit_features_{s}")

    # separation from the signal (visible, 10-70 us)
    med_rows, kept_rows = [], []
    for s in SAMPLES:
        d = u[(u["sample"] == s) & u.vis & (u.w == 2)]
        sig = d[d.cls == SIGNAL]
        cl = [c for c in order if c != "noise" and (d.cls == c).sum() >= 20]
        for c in cl:
            g = d[d.cls == c]
            med_rows.append([TEX_SAMPLE[s] if c == cl[0] else "", UNIT_TEX[c], _num(len(g))]
                            + [_num(g[f].median(), "{:.2f}" if f == "charge_balance" else "{:.1f}" if f in ("t_rms_ns", "t_unit") else "{:.0f}")
                               for f, _, _ in FEATURES])
            if c == SIGNAL:
                continue
            kept = {f: _kept_at(sig[f].to_numpy(), g[f].to_numpy()) for f, _, _ in FEATURES}
            auc = {f: _auc(sig[f].to_numpy(), g[f].to_numpy()) for f, _, _ in FEATURES}
            kept_rows.append([TEX_SAMPLE[s], UNIT_TEX[c], _num(len(g))]
                             + [f"{100 * kept[f]:.0f} ({max(auc[f], 1 - auc[f]):.2f})" for f, _, _ in FEATURES])
            k = TEX_SAMPLE[s].capitalize() + {"mu_decay": "MuDecay", "mu_stop": "MuStop", "gamma_late": "GammaLate"}.get(c, "")
            if c in ("mu_decay", "mu_stop", "gamma_late"):
                for f, q in (("n_photons", "Phot"), ("n_tubes_cf", "Tubes"), ("t_unit", "Time"), ("d_wall_cm", "Wall")):
                    N[f"twKept{q}{k}"] = f"{100 * kept[f]:.0f}"
                    N[f"twMed{q}{k}"] = _num(g[f].median(), "{:.1f}" if f == "t_unit" else "{:.0f}")
        med_rows.append("MID")
        k = TEX_SAMPLE[s].capitalize()
        for f, q in (("n_photons", "Phot"), ("n_tubes_cf", "Tubes"), ("t_unit", "Time"), ("d_wall_cm", "Wall")):
            N[f"twMed{q}Sig{k}"] = _num(sig[f].median(), "{:.1f}" if f == "t_unit" else "{:.0f}")
    _tab(out / "tab_unit_medians.tex", "ll r rrrrrr",
         ["sample", "visible units at 10--70\\,$\\mu$s", "units"] + [FEATURE_TEX[f] for f, _, _ in FEATURES], med_rows[:-1])
    _tab(out / "tab_unit_separation.tex", "ll r rrrrrr",
         ["sample", "background", "units"] + [FEATURE_TEX[f] for f, _, _ in FEATURES], kept_rows)

    # what enters the world tank late
    g = u[(u["sample"] == "world") & (u.cls == "gamma_late") & (u.w == 2)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    e = np.arange(0, 10.25, 0.25)
    h, _ = np.histogram(g.e0, bins=e)
    m = h > 0
    ax.plot(0.5 * (e[1:] + e[:-1])[m], h[m], "o-", ms=3, color=UNIT_COL["gamma_late"])
    ax.set_yscale("log")
    ax.set_xlabel("energy of the γ entering the tank [MeV]")
    ax.set_ylabel("γ per 0.25 MeV")
    ax.set_title(f"World: energy of the γ entering the tank at 10–70 µs ({len(g):,} γ)")
    _save(fig, out, "f13_gamma_entering_world")
    N["twGammaLateWorld"] = f"{len(g):,}"
    N["twGammaLateHWorld"] = _pct(((g.e0 > 2.1) & (g.e0 < 2.35)).sum(), len(g), "{:.0f}")
    N["twGammaLateFeWorld"] = _pct(((g.e0 > 7.5) & (g.e0 < 7.8)).sum(), len(g), "{:.0f}")
    N["twGammaLateVisWorld"] = _pct(g.vis.sum(), len(g), "{:.0f}")
    N["twGammaLateLowWorld"] = _pct((g.e0 < 1.0).sum(), len(g), "{:.0f}")
    return N


LIFE_ORDER = ["<2", "2-10", "10-70", ">=70"]
