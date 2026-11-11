"""
Figures, tables and numbers for the upstream-PR validation deck (docs/presentations/pr_validation.tex):
WCSim #27 (+ wcsimlib), ToolAnalysis #392, the ClusterFinder MC pulse-parent fix and the ClusterFinder
time merge (NMS). Reads only the outputs of the local test runs named in the config (`pr_inputs`).

    ambe plots prvalidation --config configs/pr_validation.yaml --out PRESENTATION_PLOTS/pr_validation

  pr_review/out   split_*  (#392: ref = a627624, T1 = fix reverted + PMTWaveformSim first, T2 = fix reverted)
                  app_bt_* (Application / + ClusterFinder fix / + fix + BackTracker distinct parents)
                  integ_*  (#392 TA on the WCSim benchmark files: final #27 and upstream)
  pr_review/logs  analyse_app_bt_*.log (BackTracker best particle per cluster, verbosity 3)
  nms_port/out    nms2_* (10/20), nms3_{par,wf}_{off,on,on0}_*, devwf_* (dev waveform-hit chain, exact truth);
  nms_port/res    t0_output.txt, t2_*, t5_*, nms3_wf_*, devwf_* csv
  wcsim_bench     results.csv, sortfix_changes.csv

  p*.pdf/png, tab_*.tex, prv_numbers.tex (one macro per number quoted on the slides), prv_*.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..plotting import set_style, two_titles
from .window_compare import _f, _tab

NEW, OLD, THIRD, WORLD = "#1f77b4", "#999999", "#eb6834", "#1baf7a"
SAMPLES = ("tank", "world", "ambe")
SLAB = {"tank": "tank", "world": "world", "ambe": "AmBe"}
NEV = {"tank": 500, "world": 700, "ambe": 1000}
MUPI = {13, 211}
RUNWORD = {"6265": "Source", "6256": "Dummy"}


def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=180)
    plt.close(fig)
    print(f"[prv] {name}")


def _tree(path: Path, branches=None):
    t = uproot.open(path)["Event"]
    return t.arrays(branches) if branches else t


def _equal(a, b) -> bool:
    try:
        fa, fb = ak.to_numpy(ak.flatten(a, axis=None)), ak.to_numpy(ak.flatten(b, axis=None))
        if ak.to_list(ak.num(a, axis=1)) != ak.to_list(ak.num(b, axis=1)):
            return False
    except Exception:
        try:
            fa, fb = ak.to_numpy(a), ak.to_numpy(b)
        except Exception:
            return ak.to_list(a) == ak.to_list(b)
    if fa.shape != fb.shape:
        return False
    return bool(np.array_equal(fa, fb, equal_nan=fa.dtype.kind == "f"))


def identity(ref: Path, new: Path) -> dict:
    """Branch-by-branch comparison of two ANNIEEvent trees (NaN-aware)."""
    r, n = _tree(ref), _tree(new)
    common = sorted(set(r.keys()) & set(n.keys()))
    changed = [k for k in common if r.num_entries != n.num_entries or not _equal(r[k].array(), n[k].array())]
    return dict(ref=ref.stem, new=new.stem, common=len(common), identical=len(common) - len(changed),
                changed=";".join(changed), only_ref=len(set(r.keys()) - set(n.keys())),
                only_new=len(set(n.keys()) - set(r.keys())))


# ---------------------------------------------------------------------------- WCSim #27
def wcsim(bench: Path, prr: Path, out: Path, N: dict):
    b = pd.read_csv(bench / "results.csv")
    up, fin, off = (b[b.variant == v].set_index("sample") for v in ("upstream_up", "final_on", "final_off"))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(3)
    for ax, col, scale, lab in zip(axs, ("tracks_per_event", "bytes_per_event", "cpu_user_s_per_event"),
                                   (1, 1e-3, 1), ("saved tracks / event", "file size / event [kB]",
                                                  "CPU / event [s]")):
        for d, c, m, name in ((up, OLD, "o", "upstream"), (fin, NEW, "s", "#27"),
                              (off, THIRD, "^", "#27, SaveTracksOnDemand false")):
            ax.plot(x, [d.loc[s, col] * scale for s in SAMPLES], m, color=c, ms=8, label=name)
        ax.set_xticks(x, [SLAB[s] for s in SAMPLES])
        ax.set_ylabel(lab)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(bottom=0)
    axs[0].legend(loc="upper right")
    two_titles(fig, "WCSim #27: extra saved tracks cost file size, not CPU",
               "WCSim benchmark, one macro and seed per sample: tank GENIE 50 events, world 100, AmBe GPS 200; "
               "upstream annie e88314b vs #27 7193757, same wcsim container")
    _save(fig, out, "p01_wcsim_cost")
    rows = []
    for s in SAMPLES:
        rows.append([SLAB[s], f"{up.loc[s,'bytes_per_event']/1e3:.1f} $\\to$ {fin.loc[s,'bytes_per_event']/1e3:.1f}",
                     f"{up.loc[s,'tracks_per_event']:.1f} $\\to$ {fin.loc[s,'tracks_per_event']:.1f}",
                     f"{100*up.loc[s,'dp_track_saved_frac']:.1f} $\\to$ {100*fin.loc[s,'dp_track_saved_frac']:.1f}",
                     f"{up.loc[s,'cpu_user_s_per_event']:.3g} $\\to$ {fin.loc[s,'cpu_user_s_per_event']:.3g}",
                     "yes" if (off.loc[s, "bytes"] == up.loc[s, "bytes"]
                               and off.loc[s, "tracks_per_event"] == up.loc[s, "tracks_per_event"]) else "no"])
        N[f"wcSize{s.capitalize()}"] = f"{100*(fin.loc[s,'bytes_per_event']/up.loc[s,'bytes_per_event']-1):+.0f}"
    _tab(out / "tab_wcsim_cost.tex", "l rrrr c",
         ["sample", "kB / event", "saved tracks / event", "photons with saved direct parent [\\%]",
          "CPU / event [s]", "switch off = upstream"], rows)

    s = pd.read_csv(bench / "sortfix_changes.csv")
    s = s[s.switch == "on"].set_index("sample")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for col, n, c, m, lab in (("prompt_dp_changed", "prompt_photons", OLD, "o", "prompt photons"),
                              ("delayed_dp_changed", "delayed_photons", NEW, "s", "delayed photons")):
        ax.plot(x, [100 * s.loc[q, col] / s.loc[q, n] for q in SAMPLES], m, color=c, ms=9, label=lab)
    ax.set_xticks(x, [SLAB[q] for q in SAMPLES])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("direct parent changed [%]")
    ax.legend(loc="upper right")
    other = int(s.other_columns_changed.sum())
    two_titles(fig, f"Sort fix changes only the direct parent ({other} other values change)",
               "Same seed and macro, #27 before (7f8970a) and after (7193757) the SortDigiMapsByHitTime fix; "
               "every photon of every digit compared; delayed = photon time > 2 us")
    _save(fig, out, "p03_wcsim_sortfix")
    _tab(out / "tab_wcsim_sortfix.tex", "l rr rr r",
         ["sample", "prompt photons", "direct parent changed", "delayed photons", "direct parent changed",
          "other values changed"],
         [[SLAB[q], f"{s.loc[q,'prompt_photons']:,}", f"{s.loc[q,'prompt_dp_changed']:,}",
           f"{s.loc[q,'delayed_photons']:,}", f"{s.loc[q,'delayed_dp_changed']:,}",
           f"{s.loc[q,'other_columns_changed']}"] for q in SAMPLES], "0.8\\textwidth")
    for q in SAMPLES:
        N[f"wcSortDel{q.capitalize()}"] = f"{100*s.loc[q,'delayed_dp_changed']/s.loc[q,'delayed_photons']:.1f}"

    rows, lin = [], {}
    for q in SAMPLES:
        for v in ("up", "final"):
            p = prr / "out" / f"integ_{v}_{q}.root"
            if not p.exists():
                continue
            a = _tree(p, ["DirectParent_IsDarknoise", "DirectParent_LineageStatus"])
            phys = ak.flatten(a.DirectParent_IsDarknoise) == 0
            st = ak.to_numpy(ak.flatten(a.DirectParent_LineageStatus)[phys])
            lin[q, v] = dict(n=len(st), complete=float(np.mean(st == 1)), stopped=float(np.mean(st == 0)))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for v, c, m, lab in (("up", OLD, "o", "upstream WCSim"), ("final", NEW, "s", "#27")):
        xs = [k for k, q in enumerate(SAMPLES) if (q, v) in lin]
        ax.plot(xs, [100 * lin[SAMPLES[k], v]["complete"] for k in xs], m, color=c, ms=9, label=lab)
    ax.set_xticks(x, [SLAB[q] for q in SAMPLES])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("complete lineage [%]")
    ax.legend(loc="lower right")
    two_titles(fig, "With #27 every PMT hit traces back to its primary",
               "#392 ToolAnalysis (a627624, lib from 7193757) on the WCSim benchmark files; physics hit = "
               "DirectParent_IsDarknoise 0; complete = DirectParent_LineageStatus 1 (0 = walk stopped at a track WCSim did not save)")
    _save(fig, out, "p02_wcsim_lineage")
    for q in SAMPLES:
        r = [SLAB[q]]
        for v in ("up", "final"):
            d = lin.get((q, v))
            r += ["--", "--"] if d is None else [f"{d['n']:,}", f"{100*d['complete']:.2f}"]
            if d is not None:
                N[f"wcLin{v.capitalize()}{q.capitalize()}"] = f"{100*d['complete']:.1f}"
        rows.append(r)
    _tab(out / "tab_wcsim_lineage.tex", "l rr rr", ["sample", "upstream: hits", "complete [\\%]",
                                                    "\\#27: hits", "complete [\\%]"], rows, "0.7\\textwidth")


# ---------------------------------------------------------------------------- ToolAnalysis #392
BR = ["DirectParent_HitTime", "DirectParent_NeutronAncestorTrackID", "DirectParent_IsDarknoise",
      "DirectParent_PDGs", "DirectParent_LineageStatus"]


def truth_metrics(path: Path) -> dict:
    a = _tree(path, BR)
    n_del = n_del_neut = n_late = n_late_mupi = n_groups = n_one = n_phys = n_complete = 0
    purity = []
    for t, nat, dark, pdgs, ls in zip(a.DirectParent_HitTime, a.DirectParent_NeutronAncestorTrackID,
                                      a.DirectParent_IsDarknoise, a.DirectParent_PDGs, a.DirectParent_LineageStatus):
        t, nat, dark, ls = (ak.to_numpy(v) for v in (t, nat, dark, ls))
        phys = dark == 0
        n_phys += phys.sum()
        n_complete += (ls[phys] == 1).sum()
        sel = phys & (t > 2000)
        n_del += sel.sum()
        n_del_neut += (sel & (nat > 0)).sum()
        late = np.flatnonzero(phys & (t > 10000))
        n_late += len(late)
        n_late_mupi += sum(bool(set(np.abs(ak.to_list(pdgs[i]))) & MUPI) for i in late)
        idx = np.flatnonzero(sel)
        if not len(idx):
            continue
        o = idx[np.argsort(t[idx], kind="stable")]
        for g in np.split(o, np.flatnonzero(np.diff(t[o]) > 50) + 1):
            if len(g) < 3 or (nat[g] > 0).mean() <= 0.5:
                continue
            n_groups += 1
            vals, cnt = np.unique(nat[g][nat[g] > 0], return_counts=True)
            n_one += len(vals) == 1
            purity.append(cnt.max() / len(g))
    return dict(phys_hits=int(n_phys), lineage_complete=n_complete / max(n_phys, 1), delayed_hits=int(n_del),
                delayed_neutron_frac=n_del_neut / max(n_del, 1), late_hits=int(n_late),
                late_mupi_frac=n_late_mupi / max(n_late, 1), capture_groups=n_groups,
                one_neutron_frac=n_one / max(n_groups, 1), capture_purity=float(np.mean(purity)) if purity else np.nan)


VARS392 = (("split_ref", "old tool order, with the fix", OLD, "o"), ("split_T1", "#392 as submitted", NEW, "s"),
           ("split_T2", "old tool order, without the fix", THIRD, "^"))


def pr392(prr: Path, out: Path, N: dict):
    rows = []
    for v, lab, *_ in VARS392:
        for s in SAMPLES:
            rows.append(dict(variant=v, label=lab, sample=s, **truth_metrics(prr / "out" / f"{v}_{s}.root")))
    m = pd.DataFrame(rows)
    m.to_csv(out / "prv_392_truth.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(3)
    for k, (v, lab, c, mk) in enumerate(VARS392):
        d = m[m.variant == v].set_index("sample")
        dx = (k - 1) * 0.12
        axs[0].plot(x + dx, [d.loc[s, "one_neutron_frac"] for s in SAMPLES], mk, color=c, ms=8, label=lab)
        axs[1].plot(x + dx, [100 * d.loc[s, "late_mupi_frac"] for s in SAMPLES], mk, color=c, ms=8, label=lab)
    for ax, yl in zip(axs, ("capture groups with one neutron track ID", "hits after 10 µs with a μ/π parent [%]")):
        ax.set_xticks(x, [SLAB[s] for s in SAMPLES])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(yl)
    axs[0].set_ylim(0, 1.05)
    axs[1].set_ylim(bottom=0)
    axs[0].legend(loc="lower right")
    two_titles(fig, "Per-hit neutron truth of #392 with and without the ClusterFinder fix",
               "DirectParent_* tree branches; capture group = non-dark hits after 2 us split at 50 ns gaps, >= 3 hits, "
               "majority with a neutron ancestor; tank 500 / world 700 / AmBe 1000 events, productionv3fix; lib 7193757")
    _save(fig, out, "p05_392_truth")
    _tab(out / "tab_392_truth.tex", "ll rrrr",
         ["sample", "variant", "one neutron / capture", "capture purity", "delayed hits with neutron ancestor",
          "$\\mu/\\pi$ parent after 10\\,$\\mu$s"],
         [r for s in SAMPLES for r in
          [[SLAB[s] if k == 0 else "", lab.replace("#", "\\#"), f"{d.one_neutron_frac:.3f}", f"{d.capture_purity:.3f}",
            f"{d.delayed_neutron_frac:.3f}", f"{d.late_mupi_frac:.3f}"]
           for k, (v, lab, *_) in enumerate(VARS392) for d in [m[(m.variant == v) & (m["sample"] == s)].iloc[0]]]
          + ["MID"]][:-1])
    for _, d in m.iterrows():
        key = {"split_ref": "Ref", "split_T1": "Sub", "split_T2": "Nofix"}[d.variant] + d["sample"].capitalize()
        N[f"one{key}"] = f"{d.one_neutron_frac:.3f}"
        N[f"mupi{key}"] = f"{100*d.late_mupi_frac:.1f}"

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(2, 67, 66)
    for ax, (v, lab) in zip(axs, (("split_T1", "#392 as submitted"), ("split_T2", "old tool order, without the fix"))):
        a = _tree(prr / "out" / f"{v}_tank.root", BR)
        t = ak.to_numpy(ak.flatten(a.DirectParent_HitTime)) / 1000
        nat = ak.to_numpy(ak.flatten(a.DirectParent_NeutronAncestorTrackID))
        dark = ak.to_numpy(ak.flatten(a.DirectParent_IsDarknoise))
        mupi = np.array([bool(set(np.abs(p)) & MUPI) for p in ak.to_list(ak.flatten(a.DirectParent_PDGs, axis=1))])
        keep = (dark == 0) & (t > 2)
        for sel, c, ls, name in ((nat > 0, NEW, "-", "neutron ancestor"), (mupi & (nat <= 0), THIRD, "--", "μ/π parent"),
                                 ((nat <= 0) & ~mupi, OLD, ":", "other")):
            h, _ = np.histogram(t[keep & sel], bins)
            ax.step(bins[:-1], h, where="post", color=c, ls=ls, lw=2, label=f"{name} ({int(h.sum()):,})")
        ax.set_yscale("log")
        ax.set_xlabel("hit time [µs]")
        ax.set_title(lab)
        ax.legend(loc="upper right")
    axs[0].set_ylabel("hits / µs")
    two_titles(fig, "Delayed tank light by true origin",
               "Tank productionv3fix part 12, 500 events; DirectParent_* hits after 2 us, dark noise removed; "
               "mu/pi parent = a mu or pi among the hit's direct parents; neutron ancestor = DirectParent_NeutronAncestorTrackID > 0")
    _save(fig, out, "p04_392_delayed_origin")

    pairs = [("split_ref", "split_T1", "old tool order vs \\#392 as submitted (new order)"),
             ("split_ref", "split_T2", "old tool order: with vs without the ClusterFinder fix"),
             ("split_T1", "split_reord", "new tool order: without vs with the ClusterFinder fix")]
    ids = []
    for a_, b_, lab in pairs:
        for s in SAMPLES:
            r = identity(prr / "out" / f"{a_}_{s}.root", prr / "out" / f"{b_}_{s}.root")
            ids.append(dict(pair=lab, sample=s, **r))
    idf = pd.DataFrame(ids)
    idf.to_csv(out / "prv_392_identity.csv", index=False)
    _tab(out / "tab_392_identity.tex", "l ccc l",
         ["comparison", "tank", "world", "AmBe", "branches that change"],
         [[lab] + [f"{g.loc[g['sample']==s,'identical'].iloc[0]}/{g.loc[g['sample']==s,'common'].iloc[0]}"
                   for s in SAMPLES] + [_kinds(g)] for lab, g in idf.groupby("pair", sort=False)])
    return m


def _kinds(g: pd.DataFrame) -> str:
    ch = set(";".join(g.changed).split(";")) - {""}
    if not ch:
        return "none"
    if all(c.startswith("DirectParent_") for c in ch):
        return f"only the per-hit truth branches ({len(ch)})"
    return ", ".join(sorted(c.replace("_", "\\_") for c in ch))


# ---------------------------------------------------------------------------- ClusterFinder fix
BT = {"app_bt_ref": ("Application", OLD, "o"), "app_bt_cf": ("+ ClusterFinder fix", THIRD, "^"),
      "app_bt_both": ("+ fix + BackTracker distinct parents", NEW, "s")}
TBINS = [0, 2, 10, 30, 67]


def _bt(prr: Path, v: str, s: str) -> pd.DataFrame:
    txt = (prr / "logs" / f"analyse_{v}_{s}.log").read_text(errors="replace")
    pdg = [int(p) for p in re.findall(r"best particleId is : -?\d+ which has PDG: (-?\d+)", txt)]
    a = _tree(prr / "out" / f"{v}_{s}.root", ["clusterTime"])
    ct = ak.to_numpy(ak.flatten(a.clusterTime))
    if len(ct) != len(pdg):
        raise ValueError(f"{v}_{s}: {len(ct)} clusters in the tree, {len(pdg)} BackTracker matches in the log")
    skipped = len(re.findall(r"this MCHit has \d+ parents!", txt))
    return pd.DataFrame(dict(t_us=ct / 1000, pdg=np.abs(pdg), skipped=skipped))


def cffix(prr: Path, out: Path, N: dict):
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, s in zip(axs, ("tank", "world")):
        for v, (lab, c, mk) in BT.items():
            d = _bt(prr, v, s)
            d["bin"] = pd.cut(d.t_us, TBINS, right=False)
            g = d.groupby("bin", observed=False)
            frac = g.pdg.apply(lambda p: np.mean(p == 2112) if len(p) else np.nan)
            ax.plot(np.arange(len(frac)), frac.values, mk + "-", color=c, ms=7, label=lab)
            prompt = d[d.t_us < 2]
            rows.append(dict(variant=v, label=lab, sample=s, clusters=len(d), neutron=int((d.pdg == 2112).sum()),
                             muon=int((d.pdg == 13).sum()), delayed=int((d.t_us >= 2).sum()),
                             delayed_neutron=int(((d.t_us >= 2) & (d.pdg == 2112)).sum()),
                             prompt_mupi_frac=float(np.isin(prompt.pdg, list(MUPI)).mean()) if len(prompt) else np.nan,
                             skipped_hits=int(d.skipped.iloc[0])))
        ax.set_xticks(np.arange(len(TBINS) - 1), [f"{a}-{b}" for a, b in zip(TBINS[:-1], TBINS[1:])])
        ax.set_xlabel("cluster time [µs]")
        ax.set_title(SLAB[s])
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("clusters best-matched to a neutron")
    axs[0].legend(loc="lower right")
    two_titles(fig, "Upstream BackTracker: which particle made each MC cluster",
               "ANNIEsoft Application 10dcfa3, stock WCSimLib; ClusterFinder HitStore MCHits, 50 ns / >= 5 hits; "
               "BackTracker best particle (primary parent, MCParticle index) per cluster, aligned with the tree clusters in time order")
    _save(fig, out, "p06_cf_bt")
    b = pd.DataFrame(rows)
    b.to_csv(out / "prv_cf_bt.csv", index=False)
    _tab(out / "tab_cf_bt.tex", "ll rrrrr r",
         ["sample", "variant", "clusters", "$\\to$ neutron", "$\\to$ muon", "delayed $\\to$ neutron",
          "prompt $\\to \\mu/\\pi$", "hits skipped"],
         [r for s in ("tank", "world", "ambe") for r in
          [[SLAB[s] if k == 0 else "", d.label, f"{d.clusters}", f"{d.neutron}", f"{d.muon}",
            f"{d.delayed_neutron}/{d.delayed}", _f(d.prompt_mupi_frac, "{:.2f}"), f"{d.skipped_hits:,}"]
           for k, v in enumerate(BT) for d in [_bt_row(prr, v, s, b)]] + ["MID"]][:-1])
    for _, d in b.iterrows():
        key = {"app_bt_ref": "App", "app_bt_cf": "Cf", "app_bt_both": "Both"}[d.variant] + d["sample"].capitalize()
        N[f"btN{key}"], N[f"btMu{key}"], N[f"btSkip{key}"] = str(d.neutron), str(d.muon), f"{d.skipped_hits:,}"
    ids = [dict(sample=s, **identity(prr / "out" / f"app_ref_{s}.root", prr / "out" / f"app_fix_{s}.root"))
           for s in SAMPLES]
    pd.DataFrame(ids).to_csv(out / "prv_cf_identity.csv", index=False)
    for r in ids:
        N[f"cfId{r['sample'].capitalize()}"] = f"{r['identical']}/{r['common']}"


def _bt_row(prr, v, s, b):
    r = b[(b.variant == v) & (b["sample"] == s)]
    if len(r):
        return r.iloc[0]
    d = _bt(prr, v, s)
    return pd.Series(dict(label=BT[v][0], clusters=len(d), neutron=int((d.pdg == 2112).sum()),
                          muon=int((d.pdg == 13).sum()), delayed=int((d.t_us >= 2).sum()),
                          delayed_neutron=int(((d.t_us >= 2) & (d.pdg == 2112)).sum()),
                          prompt_mupi_frac=float(np.isin(d[d.t_us < 2].pdg, list(MUPI)).mean())
                          if (d.t_us < 2).any() else np.nan, skipped_hits=int(d.skipped.iloc[0])))


# ---------------------------------------------------------------------------- ClusterFinder time merge (NMS)
def _gaps(path: Path, tmin_us=2.0) -> np.ndarray:
    a = _tree(path, ["clusterTime"])
    g = []
    for ct in a.clusterTime:
        c = np.sort(ak.to_numpy(ct))
        c = c[c > tmin_us * 1000]
        if len(c) > 1:
            g.append(np.diff(c))
    return np.concatenate(g) if g else np.array([])


TRUTH = ("parametric MC", "waveform hits + truth")


def nms(nmsd: Path, tag: str, out: Path, N: dict, devtag: str = "devwf", pilot: Optional[Path] = None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0, 500, 51)
    for ax, (mdl, lab) in zip(axs, (("par", "parametric (MCHits merged into MC pulses)"),
                                    ("wf", "waveform (PMTWaveformSim + hit finder)"))):
        for v, c, ls, name in (("off", OLD, "-", "ClusterFinder"), ("on", NEW, "-", "+ time merge 50 ns")):
            g = np.concatenate([_gaps(nmsd / "out" / f"{tag}_{mdl}_{v}_{s}.root") for s in ("tank", "world")])
            h, _ = np.histogram(g, bins)
            ax.step(bins[:-1], h, where="post", color=c, ls=ls, lw=2,
                    label=f"{name}: {int((g < 50).sum())} gaps < 50 ns")
        ax.set_xlabel("time between consecutive delayed clusters [ns]")
        ax.set_title(lab)
        ax.legend(loc="upper right")
    axs[0].set_ylabel("cluster pairs / 10 ns")
    two_titles(fig, "Time merge removes the close cluster pairs in both MC pulse models",
               f"tank 500 + world 700 events, productionv3fix; ClusterFinder 20 ns / >= 5 hits; delayed = cluster time > 2 us; "
               f"TimeMergeWindow 50, TimeMergeMinTime 2000; build {tag}")
    _save(fig, out, "p07_nms_gaps")

    rows = []
    for mdl, mlab in (("on", "parametric MC"), (f"{tag}_wf_on", "waveform MC"), (f"{devtag}_on", "waveform hits + truth")):
        for s in SAMPLES:
            f = nmsd / "res" / (f"t2_{mdl}_{s}.csv" if mdl == "on" else f"{mdl}_{s}.csv")
            if f.exists():
                rows.append((mlab, SLAB[s], pd.read_csv(f).set_index("method")))
    for run in ("6265", "6256"):
        f = nmsd / "res" / (f"t5_{run}_fix.csv" if (nmsd / "res" / f"t5_{run}_fix.csv").exists() else f"t5_{run}.csv")
        rows.append(("data", run, pd.read_csv(f).set_index("method")))
    tr = []
    for mlab, s, d in rows:
        tr.append([mlab, s, f"{d.loc['CF','clusters']}", f"{d.loc['PyNMS','clusters']}", f"{d.loc['CppNMS','clusters']}",
                   f"{100*d.loc['CppNMS','same_count_frac']:.2f}", f"{100*d.loc['CppNMS','within5ns']:.1f}",
                   f"{d.loc['CF','split_objects']} $\\to$ {d.loc['CppNMS','split_objects']}" if mlab in TRUTH else "--",
                   f"{d.loc['PyNMS','purity']:.3f} / {d.loc['CppNMS','purity']:.3f}" if mlab in TRUTH else "--"])
    _tab(out / "tab_nms_cpp_py.tex", "ll rrr rr cc",
         ["input", "sample", "CF", "Python NMS", "C++ NMS", "same count [\\%]", "within 5\\,ns [\\%]",
          "split objects", "purity Py / C++"], tr)
    ids = []
    for v in ("off", "on", "on0"):
        for s in SAMPLES:
            ids.append(dict(variant=v, sample=s, **identity(nmsd / "out" / f"nms2_{v}_{s}.root",
                                                         nmsd / "out" / f"{tag}_par_{v}_{s}.root")))
    idf = pd.DataFrame(ids)
    idf.to_csv(out / "prv_nms_identity.csv", index=False)
    N["nmsParIdentical"] = str(int((idf.identical == idf.common).sum()))
    N["nmsParPairs"] = str(len(idf))
    if pilot is not None:
        r = identity(pilot, nmsd / "out" / f"{devtag}_off_ambe.root")
        pd.DataFrame([r]).to_csv(out / "prv_nms_dev_identity.csv", index=False)
        N["nmsDevPilot"] = f"{r['identical']}/{r['common']}"
    for mlab, s, d in rows:
        k = {"parametric MC": "Par", "waveform MC": "Wf", "waveform hits + truth": "Dev", "data": "Data"}[mlab] \
            + RUNWORD.get(s, s.capitalize())
        if mlab in TRUTH:
            N[f"nmsSplit{k}"] = f"{d.loc['CF','split_objects']}"
            N[f"nmsOne{k}"] = f"{d.loc['CppNMS','one_neutron']:.3f}"
        N[f"nmsSame{k}"] = f"{100*d.loc['CppNMS','same_count_frac']:.1f}"
        N[f"nmsCf{k}"], N[f"nmsPy{k}"], N[f"nmsCpp{k}"] = (str(d.loc[m, "clusters"]) for m in ("CF", "PyNMS", "CppNMS"))
    t0 = (nmsd / "res" / "t0_output.txt").read_text()
    m0 = re.search(r"python parity: (\d+) cases, (\d+) hits", t0)
    N["nmsTzeroCases"], N["nmsTzeroHits"] = f"{int(m0.group(1)):,}", f"{int(m0.group(2)):,}"
    N["nmsTzeroResult"] = re.search(r"T0 (PASS|FAIL) (\d+) failures", t0).group(0)
    wall = dict(re.findall(r"^\S+ \S+ +(\d+ (?:off|on)) rc=0 wall=(\d+)s", (nmsd / "t5" / "STATUS.txt").read_text(), re.M))
    for run in ("6265", "6256"):
        N[f"nmsWall{RUNWORD[run]}"] = f"{wall.get(run + ' off', '--')} / {wall.get(run + ' on', '--')}"


# ---------------------------------------------------------------------------- explanation figures
PCOL = {"neutron": NEW, "muon / pion": THIRD, "other": OLD}


def _ptype(pdg: int, neutron_ancestor: bool = False) -> str:
    if neutron_ancestor or abs(pdg) == 2112:
        return "neutron"
    return "muon / pion" if abs(pdg) in MUPI else "other"


def _dumps(run: Path):
    trk = pd.read_csv(run / "dump.trk", sep=" ", header=None,
                      names=["ev", "trig", "id", "pdg", "ptype", "flag", "prim", "dir"])
    trk = trk[trk.flag >= 0]
    pdg = {(e, i): p for e, i, p in zip(trk.ev, trk.id, trk.pdg)}
    ph = pd.read_csv(run / "dump.tph", sep=" ", header=None, names=["ev", "trig", "tube", "t", "prim", "dir"])
    ph["dir_pdg"] = [abs(pdg.get((e, d), 0)) for e, d in zip(ph.ev, ph.dir)]
    ph["prim_pdg"] = [abs(pdg.get((e, d), 0)) for e, d in zip(ph.ev, ph.prim)]
    return ph


def sortfix_photons(bench: Path, out: Path, N: dict):
    bins = [0, 0.5, 1, 2, 5, 10, 20, 40, 70]
    lab = [f"{a:g}-{b:g}" for a, b in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for v, name, c, mk in (("corr_on_tank", "before the sort fix", OLD, "o"), ("final_on_tank", "after", NEW, "s")):
        ph = _dumps(bench / "run" / v)
        ph["bin"] = pd.cut(ph.t / 1000, bins, right=False)
        f = ph.groupby("bin", observed=False).dir_pdg.apply(lambda p: 100 * np.isin(p, list(MUPI)).mean())
        ax.plot(np.arange(len(lab)), f.values, mk + "-", color=c, ms=7, label=name)
        late = ph[ph.t > 10000]
        N[f"sfLate{'Before' if v.startswith('corr') else 'After'}"] = f"{100*np.isin(late.dir_pdg, list(MUPI)).mean():.1f}"
    ax.set_xticks(np.arange(len(lab)), lab)
    ax.set_xlabel("photon time [µs]")
    ax.set_ylabel("photons with a μ/π direct parent [%]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    two_titles(fig, "Before the fix, late light was labelled with the muon",
               "WCSim tank benchmark (GENIE, 50 events, same seed), every detected photon; direct parent = the track that "
               "emitted it; muons stop and decay within a few us, so a mu/pi direct parent after ~10 us is a wrong label")
    _save(fig, out, "e01_sortfix_photons")


def _cf_pulses(t, window=10.0):
    groups, cur = [], []
    for i in np.argsort(t, kind="stable"):
        if cur and abs(t[cur[0]] - t[i]) >= window:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


CF_TMAX = 0.95 * 70000


def pulse_illustration(bench: Path, out: Path, N: dict):
    ph = _dumps(bench / "run" / "final_on_tank")
    best = None
    for (e, tube), g in ph.groupby(["ev", "tube"]):
        g = g[g.t < CF_TMAX]
        pr, dl = g[g.t < 200], g[g.t > 2000]
        if len(pr) >= 3 and (pr.prim_pdg == 13).any() and (dl.prim_pdg == 2112).sum() >= 2 and len(g) < 40:
            score = (dl.prim_pdg == 2112).sum()
            if best is None or score > best[0]:
                best = (score, e, tube, g.sort_values("t", kind="stable"))
    _, ev, tube, g = best
    g = g[g.t < CF_TMAX]
    t, prim = g.t.to_numpy(), g.prim_pdg.to_numpy()
    first = _ptype(prim[0])
    groups = _cf_pulses(t)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True, gridspec_kw=dict(width_ratios=[1, 1]))
    pr_lo, pr_hi = t.min() - 5, t[t < 200].max() + 5
    dl = np.sort(t[(t > 2000) & (prim == 2112)])
    dl_lo, dl_hi = dl[0] - 0.05 * (dl[-1] - dl[0] + 100), dl[-1] + 0.05 * (dl[-1] - dl[0] + 100)
    for ax, (lo, hi) in zip(axs, ((pr_lo, pr_hi), (dl_lo, dl_hi))):
        sel = (t >= lo) & (t <= hi)
        for typ, c in PCOL.items():
            m = sel & np.array([_ptype(p) == typ for p in prim])
            ax.plot(t[m], np.full(m.sum(), 2), "|", color=c, ms=22, mew=2)
        for grp in groups:
            tm = float(np.median(t[grp]))
            if not lo <= tm <= hi:
                continue
            own = pd.Series([_ptype(p) for p in prim[grp]]).mode().iloc[0]
            ax.plot(tm, 1, "o", color=PCOL[own], ms=11)
            ax.plot(tm, 0, "o", color=PCOL[first], ms=11)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("time [ns]")
    axs[0].set_yticks([0, 1, 2], ["pulse label, before the fix", "pulse label, fixed", "photons"])
    axs[0].set_ylim(-0.6, 2.6)
    axs[0].set_title("prompt light")
    axs[1].set_title(f"same PMT, {dl[0]/1000:.1f}-{dl[-1]/1000:.1f} µs")
    for typ, c in PCOL.items():
        axs[1].plot([], [], "o", color=c, label=typ)
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="true primary particle")
    two_titles(fig, "How ClusterFinder labels MC pulses: one tank PMT",
               f"WCSim tank benchmark event {ev}, PMT {tube}, after the WCSim sort fix; photons merged as ClusterFinder does "
               "(10 ns from a pulse's first photon, pulse time = median); before the fix every pulse took the PMT's first photon's parents")
    _save(fig, out, "e02_pulse_labels")
    N["illusEvent"], N["illusTube"] = str(ev), str(tube)


def _dp_hits(path: Path, e: int):
    a = uproot.open(path)["Event"].arrays(["DirectParent_PMTID", "DirectParent_HitTime", "DirectParent_IsDarknoise",
                                           "DirectParent_NeutronAncestorTrackID", "DirectParent_PDGs"],
                                          entry_start=e, entry_stop=e + 1)
    d = pd.DataFrame(dict(pmt=ak.to_numpy(a.DirectParent_PMTID[0]), t=ak.to_numpy(a.DirectParent_HitTime[0]) / 1000,
                          dark=ak.to_numpy(a.DirectParent_IsDarknoise[0]),
                          nat=ak.to_numpy(a.DirectParent_NeutronAncestorTrackID[0])))
    pd_ = ak.to_list(a.DirectParent_PDGs[0])
    d["origin"] = [("neutron" if n > 0 else "muon / pion" if set(map(abs, p)) & MUPI else "other")
                   for n, p in zip(d.nat, pd_)]
    return d[(d.dark == 0) & (d.t > 2)]


def event_display_392(prr: Path, out: Path, N: dict):
    a = uproot.open(prr / "out" / "split_T1_tank.root")["Event"].arrays(
        ["DirectParent_HitTime", "DirectParent_NeutronAncestorTrackID", "DirectParent_IsDarknoise"])
    cnt = [int(((ak.to_numpy(t) > 2000) & (ak.to_numpy(n) > 0) & (ak.to_numpy(d) == 0)).sum())
           for t, n, d in zip(a.DirectParent_HitTime, a.DirectParent_NeutronAncestorTrackID, a.DirectParent_IsDarknoise)]
    e = int(np.argsort(cnt)[len(cnt) - 1 - len(cnt) // 20])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (v, lab) in zip(axs, (("split_T1", "#392 as submitted"), ("split_T2", "old tool order, without the ClusterFinder fix"))):
        d = _dp_hits(prr / "out" / f"{v}_tank.root", e)
        for typ, c in PCOL.items():
            s = d[d.origin == typ]
            ax.plot(s.t, s.pmt, "o", color=c, ms=4, label=f"{typ} ({len(s)})")
        ax.set_xlabel("hit time [µs]")
        ax.set_title(lab)
        ax.legend(loc="center", title="true origin of the hit")
    axs[0].set_ylabel("PMT ID")
    two_titles(fig, "One tank event: delayed hits by true origin",
               f"Tank productionv3fix part 12, event {e}; DirectParent_* hits after 2 us, dark noise removed; same WCSim event in both panels")
    _save(fig, out, "e03_event_392")
    N["dispEvent"] = str(e)


def event_display_nms(nmsd: Path, devtag: str, out: Path, N: dict):
    br = ["clusterTime", "Cluster_HitT", "Cluster_HitChankey", "DirectParent_PMTID", "DirectParent_HitTime",
          "DirectParent_NeutronAncestorTrackID", "DirectParent_ImmediateAncestorTrackID",
          "DirectParent_ImmediateAncestorPDG"]
    off = uproot.open(nmsd / "out" / f"{devtag}_off_tank.root")["Event"].arrays(br, library="np")
    on = uproot.open(nmsd / "out" / f"{devtag}_on_tank.root")["Event"].arrays(br[:3], library="np")
    pick = None
    for e in range(len(off["clusterTime"])):
        med = [np.median(h) for h in off["Cluster_HitT"][e]]
        dl = sorted((m, k) for k, m in enumerate(med) if m > 2000)
        for (m1, k1), (m2, k2) in zip(dl, dl[1:]):
            if m2 - m1 < 50 and len(off["Cluster_HitT"][e][k1]) + len(off["Cluster_HitT"][e][k2]) >= 14:
                pick = (e, k1, k2, m1)
                break
        if pick:
            break
    e, k1, k2, m1 = pick
    own = {}
    for ck, t, n, ia, ip in zip(off["DirectParent_PMTID"][e], off["DirectParent_HitTime"][e],
                                off["DirectParent_NeutronAncestorTrackID"][e], off["DirectParent_ImmediateAncestorTrackID"][e],
                                off["DirectParent_ImmediateAncestorPDG"][e]):
        own[(int(ck), round(float(t), 3))] = ("neutron" if n > 0 else f"{ip}") + f" #{n if n > 0 else ia}"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    cols = [NEW, THIRD, WORLD, "#7b5ea7"]
    owners = set()
    for k, c in zip((k1, k2), cols):
        h, ck = off["Cluster_HitT"][e][k], off["Cluster_HitChankey"][e][k]
        axs[0].plot(h - m1, ck, "o", color=c, ms=5, label=f"cluster {1 + (k == k2)}: {len(h)} hits, median {np.median(h)-m1:+.0f} ns")
        owners |= {own.get((int(a), round(float(b), 3)), "?") for a, b in zip(ck, h)}
    on_med = [np.median(h) for h in on["Cluster_HitT"][e]]
    j = int(np.argmin([abs(m - m1) for m in on_med]))
    h, ck = on["Cluster_HitT"][e][j], on["Cluster_HitChankey"][e][j]
    axs[1].plot(h - m1, ck, "o", color=NEW, ms=5, label=f"merged: {len(h)} hits")
    for ax, lab in zip(axs, ("ClusterFinder (20 ns window)", "+ time merge (50 ns)")):
        ax.set_xlabel("hit time - first cluster median [ns]")
        ax.set_title(lab)
        ax.legend(loc="upper right")
    axs[0].set_ylabel("PMT channel")
    tp = sorted(o for o in owners if o != "?")
    words = {"13": "a Michel electron from a μ⁻ decay", "-13": "a Michel positron from a μ⁺ decay",
             "neutron": "a neutron capture"}
    who = ", ".join(sorted({words.get(o.split(" #")[0], o) for o in tp})) or "n/a"
    two_titles(fig, f"One true particle ({who}) in two clusters: the merge joins them",
               f"Dev waveform-hit chain (PMTWaveformSim hits with exact parents), tank event {e}; true owner(s) of the hits: "
               f"{', '.join(tp) if tp else 'n/a'} (neutron #trackID or PDG #trackID)")
    _save(fig, out, "e04_event_nms")
    N["nmsEvent"] = str(e)


def lineage_vs_time(prr: Path, out: Path):
    bins = [0, 2, 10, 30, 67]
    lab = [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, s in zip(axs, ("tank", "ambe")):
        for v, name, c, mk in (("up", "upstream WCSim", OLD, "o"), ("final", "#27", NEW, "s")):
            a = _tree(prr / "out" / f"integ_{v}_{s}.root", ["DirectParent_HitTime", "DirectParent_IsDarknoise",
                                                            "DirectParent_LineageStatus"])
            t = ak.to_numpy(ak.flatten(a.DirectParent_HitTime)) / 1000
            ok = ak.to_numpy(ak.flatten(a.DirectParent_IsDarknoise)) == 0
            st = ak.to_numpy(ak.flatten(a.DirectParent_LineageStatus))
            f = [100 * np.mean(st[ok & (t >= lo) & (t < hi)] == 1) if (ok & (t >= lo) & (t < hi)).any() else np.nan
                 for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(np.arange(len(lab)), f, mk + "-", color=c, ms=8, label=name)
        ax.set_xticks(np.arange(len(lab)), lab)
        ax.set_xlabel("hit time [µs]")
        ax.set_title(SLAB[s])
        ax.set_ylim(-3, 105)
    axs[0].set_ylabel("hits traced back to their primary [%]")
    axs[1].legend(loc="center right")
    two_titles(fig, "Hits traced back to their primary particle, by time",
               "ToolAnalysis #392 on the WCSim benchmark files (tank GENIE 50 events, AmBe GPS 200); physics hits only; "
               "upstream WCSim does not save most low-energy secondaries, so the walk stops at an unsaved track; "
               "for AmBe it saves none of the tracks behind the capture light")
    _save(fig, out, "e05_lineage_vs_time")


def bt_times(prr: Path, out: Path):
    bins = np.arange(2, 70, 5)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (pdg, lab) in zip(axs, ((2112, "clusters matched to a neutron"), (13, "clusters matched to a muon"))):
        for v, name, c in (("app_bt_ref", "Application", OLD), ("app_bt_both", "with the fix", NEW)):
            d = _bt(prr, v, "tank")
            h, _ = np.histogram(d.t_us[d.pdg == pdg], bins)
            ax.step(bins[:-1], h, where="post", color=c, lw=2, label=f"{name} ({int(h.sum())})")
        ax.set_xlabel("cluster time [µs]")
        ax.set_title(lab)
        ax.legend(loc="upper right")
    axs[0].set_ylabel("delayed clusters / 5 µs")
    two_titles(fig, "Delayed tank clusters: muon labels turn into neutron labels",
               "Upstream Application, tank productionv3fix part 12, 500 events; ClusterFinder 50 ns / >= 5 hits; BackTracker best "
               "particle per cluster (primary parent); 'with the fix' = ClusterFinder pulse parents + BackTracker distinct parents")
    _save(fig, out, "e06_bt_times")


def split_counts(nmsd: Path, devtag: str, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(3)
    for ax, (pre, lab) in zip(axs, (("t2_on", "parametric MC pulses"), (f"{devtag}_on", "waveform hits (exact truth)"))):
        d = [pd.read_csv(nmsd / "res" / f"{pre}_{s}.csv").set_index("method") for s in SAMPLES]
        for k, (m, name, c, mk) in enumerate((("CF", "ClusterFinder", OLD, "o"), ("PyNMS", "+ merge, Python", THIRD, "^"),
                                              ("CppNMS", "+ merge, ToolAnalysis", NEW, "s"))):
            ax.plot(x + (k - 1) * 0.12, [q.loc[m, "split_objects"] for q in d], mk, color=c, ms=9, label=name)
        ax.set_xticks(x, [SLAB[s] for s in SAMPLES])
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(lab)
    axs[0].set_ylabel("true particles split into several clusters")
    axs[0].set_ylim(bottom=-1)
    axs[1].legend(loc="upper right")
    two_titles(fig, "True particles that the fixed window cuts into pieces",
               "Delayed window (cluster median > 2 us); a true particle counts as split when >= 3 of its hits sit in each of >= 2 "
               "clusters; tank 500 / world 700 / AmBe 1000 events; merge window 50 ns from 2 us")
    _save(fig, out, "e07_split_counts")


def explain(prr: Path, nmsd: Path, bench: Path, devtag: str, out: Path, N: dict):
    sortfix_photons(bench, out, N)
    pulse_illustration(bench, out, N)
    event_display_392(prr, out, N)
    event_display_nms(nmsd, devtag, out, N)
    lineage_vs_time(prr, out)
    bt_times(prr, out)
    split_counts(nmsd, devtag, out)


def run(ctx: RunContext, a) -> Path:
    set_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                         "legend.frameon": False, "legend.fontsize": 9})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = ctx.extra["pr_inputs"]
    prr, nmsd, bench = Path(p["pr_review"]), Path(p["nms_port"]), Path(p["wcsim_bench"])
    N: dict = {}
    wcsim(bench, prr, out, N)
    pr392(prr, out, N)
    cffix(prr, out, N)
    nms(nmsd, a.nms_tag, out, N, a.dev_tag, Path(p["pilot_wf_tree"]) if "pilot_wf_tree" in p else None)
    explain(prr, nmsd, bench, a.dev_tag, out, N)
    with open(out / "prv_numbers.tex", "w") as fh:
        fh.write("% generated by ambe.plots.pr_validation\n")
        for k, v in N.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(prog="ambe plots prvalidation")
    ap.add_argument("--nms-tag", required=True, help="tag of the nms_port run_nms_paths.sh outputs, e.g. nms3")
    ap.add_argument("--dev-tag", default="devwf", help="tag of the nms_port run_nms_devwf.sh outputs (dev waveform-hit chain)")
    ap.add_argument("--out", required=True)
    run(ctx, ap.parse_args(list(argv) if argv else []))
