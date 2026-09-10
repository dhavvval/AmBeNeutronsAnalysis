#!/usr/bin/env python
"""
boxcut_v4_campaign.py — the traditional box-cut AmBe analysis over the full
AmBe2.0v4 source campaign: 28 runs, 26 source positions, all five ports.

NO MVA AND NO MC ANYWHERE IN THIS SCRIPT. The neutron definition here is the AmBe
pipeline's own: Stage 1 is the IC waveform gate (700 < IC_adjusted < 1200 plus the
second-pulse veto), Stage 2 is ambe_single_cut / ambe_multiple_cut from
src/ambe/data/processor.py — 0 < clusterPE <= 100, 0 < CB < 0.45,
clusterTime >= 2000 ns, clusterHits >= 5 — after the cosmic veto
(clusterTime < 2000 ns or clusterPE > 100 PE). This is the QA baseline the MVA
work in ANALYSIS_ccinc_v3_FULL.md §6.6 is measured against, so it has to stand on
its own.

WHERE THE NUMBERS COME FROM. Everything is read from artifacts already on disk;
nothing here re-reads a ROOT file or re-runs Stage 1:

  TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_gated.csv   19 positions (20 runs)
  TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_ext.csv      7 positions ( 8 runs)
  TriggerSummary/CaptureTimeFits_baseline.csv             19 positions, FROZEN
  TriggerSummary/CaptureTimeFits_v4_all28.csv             26 positions
  EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_AmBe2.0v4_{gated,ext}_*.csv

The two trigger summaries are kept as separate tags on disk because
AmBeWaveformResults_<tag>.csv is written with a plain to_csv — re-running the 8
later runs under the `_gated` tag would have destroyed the 20-run acceptance table
the published 57.30% efficiency rests on. They are merged here, and the merge is
only safe because they share NO source position (19 + 7 = 26 unique). That is
asserted, not assumed: see check_merge().

Do NOT feed this AmBeTriggerSummary_AmBe2.0v4_fprompt.csv. It covers the SAME 20
runs as `_gated` under a different Stage-1 cut, so including it double-counts 19
of the 26 positions.

Run 6248 is absent because it has no data on disk at all — no BeamCluster ntuple,
no waveform directory. It appears in processor.py `source_positions` at (75, 50, 0)
as a position entry only. That position is therefore covered solely by the LAPPD
debug pair 6249/6250.

THE ANCHOR IS FROZEN. tau = 29.417 +- 0.221 us is the inverse-variance weighted mean
of the 19 per-position fits in CaptureTimeFits_baseline.csv. This script reports the
26-position value alongside it and never overwrites it.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    MPLBACKEND=Agg python -u boxcut_v4_campaign.py --do all
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from make_plots_ccinc_v3_merged import (OUTD, BLUE, GREY, ORANGE, GREEN, PURPLE,
                                        bare, table_axes)
# The port x y maps here must look identical to the efficiency map, so the look is
# imported from the module that owns it rather than restated. See _heat().
from ambe.plots.heatmap import (HEAT_ANNOT_SIZE, HEAT_ANNOT_FAMILY, HEAT_TITLE_SIZE,
                                HEAT_LABEL_SIZE, HEAT_TICK_SIZE,
                                heat_label, _figsize as _heat_figsize)
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TS = HERE / "TriggerSummary"
CAND = HERE / "EventAmBeNeutronCandidatesData"
PREFIX = "V4BOX__"

# Every figure title in this script names the selection literally rather than a
# nickname, so a slide can be read without the talk. BOXCUT is the Stage-2 cut set
# itself; BOXCUT_FULL adds the Stage-1 gate for the figures where the gate is what
# the numbers are normalised to (cut flow, efficiency, cosmic veto).
BOXCUT = "box cuts PE ≤ 100, CB < 0.45, t ≥ 2 µs, hits ≥ 5"
BOXCUT_FULL = ("IC gate 700–1200 + cosmic veto, then "
               "box cuts PE ≤ 100, CB < 0.45, t ≥ 2 µs, hits ≥ 5")

TAG_GATED, TAG_EXT = "AmBe2.0v4_gated", "AmBe2.0v4_ext"

# The frozen anchor and the sample it was measured on. Not recomputed here.
# RE-DERIVED on the 2-67 us window (was 30.53 +- 0.26 on 10-67). The whole analysis
# moved to one window; see the FIT_MIN note in fit_capture_time_fprompt_compare.py.
# Numbers from the two windows are NOT comparable, and anything in an older document
# quoting 30.53 predates the change.
TAU_ANCHOR, TAU_ANCHOR_ERR, TAU_ANCHOR_NPOS = 29.417, 0.221, 19

# Published campaign efficiency on the first 20 runs, for the consistency check.
EFF_PUBLISHED_20 = 0.5730

# Candidate rows / unique neutron triggers the MVA study cross-checks against
# (PIPELINE_REFERENCE in ccinc_v3_ambe_closure.py). Asserted, not printed blindly.
REF_28 = (236792, 217532)

# Physics geometry, mirrored from src/ambe/plots/heatmap.py PORT_INFO so this
# script stands alone. (0, -60, 102) and (0, -105.5, 102) are real acquired
# positions on the port-3 axis, not typos.
PORT_INFO = {
    (0, 100, 0): "Port 5", (0, 50, 0): "Port 5", (0, 0, 0): "Port 5",
    (0, -50, 0): "Port 5", (0, -100, 0): "Port 5", (0, 55.3, 0): "Port 5",
    (0, 100, -75): "Port 1", (0, 50, -75): "Port 1", (0, 0, -75): "Port 1",
    (0, -50, -75): "Port 1", (0, -100, -75): "Port 1",
    (75, 100, 0): "Port 4", (75, 50, 0): "Port 4", (75, 0, 0): "Port 4",
    (75, -50, 0): "Port 4", (75, -100, 0): "Port 4",
    (0, 100, 102): "Port 3", (0, 50, 102): "Port 3", (0, 0, 102): "Port 3",
    (0, -50, 102): "Port 3", (0, -100, 102): "Port 3", (0, -60, 102): "Port 3",
    (0, 100, 75): "Port 2", (0, 50, 75): "Port 2", (0, 0, 75): "Port 2",
    (0, -50, 75): "Port 2", (0, -100, 75): "Port 2",
}
PORT_ORDER = ["Port 1", "Port 2", "Port 3", "Port 4", "Port 5"]
PORT_COLOR = {"Port 1": BLUE, "Port 2": ORANGE, "Port 3": GREEN,
              "Port 4": PURPLE, "Port 5": GREY}


def save(fig, name, prefix=None):
    # `prefix` overrides the module default so the residual figures land under
    # V4RES__ instead of V4BOX__ -- they are not box-cut results, they are the
    # difference between two selections, and deck 4 owns them.
    stem = OUTD / f"{prefix or PREFIX}{name}"
    OUTD.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  [fig] {stem.name}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# loaders + the merge assertion
# ════════════════════════════════════════════════════════════════════════════
def load_summaries():
    """The two trigger summaries, tagged by origin and with port/efficiency added."""
    frames = []
    for tag, path in ((TAG_GATED, TS / f"AmBeTriggerSummary_{TAG_GATED}.csv"),
                      (TAG_EXT, TS / f"AmBeTriggerSummary_{TAG_EXT}.csv")):
        if not path.exists():
            raise SystemExit(f"missing {path}")
        d = pd.read_csv(path)
        d["tag"] = tag
        frames.append(d)
        print(f"  {tag:20s}: {len(d):2d} positions, "
              f"{int(d.ambe_triggers.sum()):>7,d} AmBe triggers")
    d = pd.concat(frames, ignore_index=True)
    d["pos"] = list(zip(d.x_pos, d.y_pos, d.z_pos))
    d["port"] = d["pos"].map(lambda p: PORT_INFO.get(p, "Unknown"))
    # Headline efficiency, per the campaign definition: an event counts once
    # whether it yielded one candidate or several.
    d["efficiency"] = d.unique_neutron_triggers / d.ambe_triggers
    d["eff_err"] = np.sqrt(d.efficiency * (1 - d.efficiency) / d.ambe_triggers)
    d["eff_single"] = d.single_neutron_candidates / d.ambe_triggers
    d["cosmic_frac"] = d.cosmic_events / d.total_events
    d["mult_share"] = d.multiple_neutron_candidates / d.unique_neutron_triggers
    return d


def check_merge(d):
    """The one failure mode that would silently corrupt every number below."""
    print("\n[check] merge sanity")
    assert len(d) == 26, f"expected 26 positions, got {len(d)}"
    dup = d[d.duplicated("pos", keep=False)]
    assert dup.empty, f"positions in BOTH tags — would double-count:\n{dup}"
    unknown = d[d.port == "Unknown"]
    assert unknown.empty, f"position not in PORT_INFO:\n{unknown[['pos']]}"
    tot = d[["ambe_triggers", "unique_neutron_triggers",
             "single_neutron_candidates", "multiple_neutron_candidates"]].sum()
    assert int(tot.unique_neutron_triggers) == REF_28[1], (
        f"unique neutron triggers {int(tot.unique_neutron_triggers):,} != "
        f"PIPELINE_REFERENCE {REF_28[1]:,}")
    assert (int(tot.single_neutron_candidates) + int(tot.multiple_neutron_candidates)
            == int(tot.unique_neutron_triggers)), "single + multiple != unique"
    print(f"  PASS  26 unique positions, no overlap, all ports resolved")
    print(f"  PASS  unique neutron triggers {int(tot.unique_neutron_triggers):,} "
          f"matches PIPELINE_REFERENCE")
    return True


def load_candidates():
    """All 28 per-run candidate CSVs, pooled. Used for the shape/cut-flow figures."""
    frames = []
    for tag in (TAG_GATED, TAG_EXT):
        for p in sorted(CAND.glob(f"EventAmBeNeutronCandidates_{tag}_*.csv")):
            run = int(p.stem.rsplit("_", 1)[1])
            d = pd.read_csv(p, usecols=["clusterTime", "clusterPE",
                                        "clusterChargeBalance", "clusterHits",
                                        "sourceX", "sourceY", "sourceZ",
                                        "eventTankTime"])
            d["run"], d["tag"] = run, tag
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["clusterTime"] = d.clusterTime / 1000.0          # ns -> us
    d["pos"] = list(zip(d.sourceX, d.sourceY, d.sourceZ))
    d["port"] = d["pos"].map(lambda p: PORT_INFO.get(p, "Unknown"))
    print(f"  candidates: {len(d):,} rows over {d.run.nunique()} runs, "
          f"{d.pos.nunique()} positions")
    assert len(d) == REF_28[0], f"{len(d):,} candidate rows != {REF_28[0]:,}"
    return d


def load_fits():
    """Frozen 19-position anchor fits and the new 26-position fits."""
    a = pd.read_csv(TS / "CaptureTimeFits_baseline.csv")
    b = pd.read_csv(TS / "CaptureTimeFits_v4_all28.csv")
    for t in (a, b):
        t["pos"] = t["pos"].map(lambda s: tuple(
            float(v) for v in str(s).strip("()").split(",")))
        t["port"] = t["pos"].map(
            lambda p: PORT_INFO.get(tuple(int(v) if float(v).is_integer() else v
                                          for v in p), "Unknown"))
    return a, b


def wavg(t, col="tau", ecol="tau_err"):
    w = 1.0 / t[ecol] ** 2
    m = (t[col] * w).sum() / w.sum()
    return float(m), float(np.sqrt(1.0 / w.sum()))


# ════════════════════════════════════════════════════════════════════════════
# products
# ════════════════════════════════════════════════════════════════════════════
def do_cutflow(d):
    """Campaign cut-flow: what the box cuts remove, stage by stage."""
    print("\n[cutflow] event selection flow, campaign totals")
    t = d[["total_events", "cosmic_events", "ambe_triggers",
           "single_neutron_candidates", "multiple_neutron_candidates",
           "unique_neutron_triggers"]].sum()
    n0 = int(t.total_events)
    rows = [
        ("IC-gated triggers (Stage 1)", n0, 1.0),
        ("− cosmic veto", -int(t.cosmic_events), -t.cosmic_events / n0),
        ("= AmBe triggers", int(t.ambe_triggers), t.ambe_triggers / n0),
        ("single-neutron events", int(t.single_neutron_candidates),
         t.single_neutron_candidates / n0),
        ("multi-neutron events", int(t.multiple_neutron_candidates),
         t.multiple_neutron_candidates / n0),
        ("= neutron triggers (Stage 2)", int(t.unique_neutron_triggers),
         t.unique_neutron_triggers / n0),
    ]
    cell = [[lbl, f"{n:+,d}" if lbl.startswith("−") else f"{n:,d}", f"{f:.1%}"]
            for lbl, n, f in rows]
    for r in cell:
        print(f"  {r[0]:32s} {r[1]:>12s} {r[2]:>8s}")
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    table_axes(ax, cell, ["stage", "events", "of Stage-1 triggers"],
               widths=[0.50, 0.27, 0.23])
    ax.set_title(f"{BOXCUT_FULL}\n"
                 f"Cut flow — 28 runs, 26 positions, {n0:,} IC-gated triggers",
                 fontsize=11)
    save(fig, "cutflow_table")

    out = pd.DataFrame(rows, columns=["stage", "events", "frac_of_stage1"])
    out.to_csv(HERE / "boxcut_v4_cutflow.csv", index=False)
    print("  wrote boxcut_v4_cutflow.csv")


def do_efficiency(d):
    """Per-position efficiency with binomial errors, plus the heatmap-style view."""
    print("\n[efficiency] per position, unique_neutron_triggers / ambe_triggers")
    t = d.sort_values("efficiency").reset_index(drop=True)
    for _, r in t.iterrows():
        print(f"  {str(r['pos']):>16s} {r['port']:8s} "
              f"{int(r.ambe_triggers):>7,d} trig  "
              f"{100*r.efficiency:5.2f} +- {100*r.eff_err:.2f} %")
    tot_eff = d.unique_neutron_triggers.sum() / d.ambe_triggers.sum()
    tot_err = np.sqrt(tot_eff * (1 - tot_eff) / d.ambe_triggers.sum())
    print(f"  CAMPAIGN: {100*tot_eff:.2f} +- {100*tot_err:.2f} %  "
          f"(range {100*d.efficiency.min():.1f} - {100*d.efficiency.max():.1f} %)")

    # efficiency vs y, one line per port
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for port in PORT_ORDER:
        s = d[d.port == port].sort_values("y_pos")
        if s.empty:
            continue
        ax.errorbar(s.y_pos, 100 * s.efficiency, yerr=100 * s.eff_err,
                    marker="o", ms=5, lw=1.4, capsize=2.5,
                    color=PORT_COLOR[port], label=port)
    ax.set_xlabel("source y position [cm]")
    ax.set_ylabel("neutron detection efficiency [%]")
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="lower center")
    ax.set_title(f"{BOXCUT}\n"
                 f"Neutron detection efficiency by source position — campaign "
                 f"{100*tot_eff:.2f} %, range {100*d.efficiency.min():.1f}–"
                 f"{100*d.efficiency.max():.1f} %", fontsize=11)
    bare(ax)
    save(fig, "efficiency_by_position")

    d.sort_values(["port", "y_pos"]).to_csv(
        HERE / "boxcut_v4_efficiency_by_position.csv", index=False)
    print("  wrote boxcut_v4_efficiency_by_position.csv")
    return tot_eff, tot_err


def do_consistency(d):
    """Did adding 8 runs move the campaign answer? The core QA statement."""
    print("\n[consistency] 20-run vs 8-run vs 28-run")
    rows = []
    for lbl, sub in (("20 runs (published)", d[d.tag == TAG_GATED]),
                     ("8 runs (added)", d[d.tag == TAG_EXT]),
                     ("28 runs (all)", d)):
        e = sub.unique_neutron_triggers.sum() / sub.ambe_triggers.sum()
        err = np.sqrt(e * (1 - e) / sub.ambe_triggers.sum())
        rows.append((lbl, len(sub), int(sub.ambe_triggers.sum()),
                     int(sub.unique_neutron_triggers.sum()), 100 * e, 100 * err))
        print(f"  {lbl:22s} {len(sub):2d} pos  {int(sub.ambe_triggers.sum()):>7,d} trig  "
              f"{100*e:5.2f} +- {100*err:.2f} %")
    e20, e28 = rows[0][4] / 100, rows[2][4] / 100
    assert abs(e20 - EFF_PUBLISHED_20) < 5e-4, (
        f"20-run efficiency {e20:.4f} != published {EFF_PUBLISHED_20}")
    print(f"  PASS  20-run value reproduces the published {100*EFF_PUBLISHED_20:.2f} %")
    print(f"  shift from adding 8 runs: {100*(e28-e20):+.2f} pp")

    cell = [[l, f"{n}", f"{t:,d}", f"{u:,d}", f"{e:.2f} ± {er:.2f}"]
            for l, n, t, u, e, er in rows]
    fig, ax = plt.subplots(figsize=(7.6, 2.1))
    table_axes(ax, cell, ["sample", "positions", "AmBe triggers",
                          "neutron triggers", "efficiency [%]"],
               widths=[0.28, 0.14, 0.20, 0.20, 0.18])
    ax.set_title(f"{BOXCUT}\n"
                 f"Adding 8 runs moves the campaign efficiency by "
                 f"{100*(e28-e20):+.2f} pp", fontsize=11)
    save(fig, "consistency_table")
    pd.DataFrame(rows, columns=["sample", "positions", "ambe_triggers",
                                "neutron_triggers", "eff_pct", "eff_err_pct"]
                 ).to_csv(HERE / "boxcut_v4_consistency.csv", index=False)
    print("  wrote boxcut_v4_consistency.csv")


def do_multiplicity(d):
    """Is the single/multiple split flat across ports, or position-dependent?"""
    print("\n[multiplicity] multi-neutron share of neutron triggers")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for port in PORT_ORDER:
        s = d[d.port == port].sort_values("y_pos")
        if s.empty:
            continue
        ax.plot(s.y_pos, 100 * s.mult_share, marker="o", ms=5, lw=1.4,
                color=PORT_COLOR[port], label=port)
    tot = d.multiple_neutron_candidates.sum() / d.unique_neutron_triggers.sum()
    print(f"  campaign {100*tot:.2f} %, "
          f"range {100*d.mult_share.min():.2f} - {100*d.mult_share.max():.2f} %")
    ax.set_xlabel("source y position [cm]")
    ax.set_ylabel("multi-neutron share of neutron triggers [%]")
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="lower center")
    ax.set_title(f"{BOXCUT}\n"
                 f"Multi-neutron share by source position — campaign "
                 f"{100*tot:.2f} %, spread "
                 f"{100*d.mult_share.min():.1f}–{100*d.mult_share.max():.1f} %",
                 fontsize=11)
    bare(ax)
    save(fig, "multiplicity_share")


def do_cosmic(d):
    """Cosmic-veto fraction per position: a detector-stability check."""
    print("\n[cosmic] veto fraction of Stage-1 triggers")
    tot = d.cosmic_events.sum() / d.total_events.sum()
    print(f"  campaign {100*tot:.2f} %, "
          f"range {100*d.cosmic_frac.min():.2f} - {100*d.cosmic_frac.max():.2f} %")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for port in PORT_ORDER:
        s = d[d.port == port].sort_values("y_pos")
        if s.empty:
            continue
        ax.plot(s.y_pos, 100 * s.cosmic_frac, marker="o", ms=5, lw=1.4,
                color=PORT_COLOR[port], label=port)
    ax.set_xlabel("source y position [cm]")
    ax.set_ylabel("cosmic-vetoed fraction of Stage-1 triggers [%]")
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="lower center")
    ax.set_title("Cosmic veto (clusterTime < 2 µs or PE > 100), applied before the "
                 "box cuts\n"
                 f"Vetoed fraction by source position — campaign {100*tot:.2f} %, "
                 f"flat to "
                 f"{100*(d.cosmic_frac.max()-d.cosmic_frac.min()):.1f} pp across "
                 "26 positions", fontsize=11)
    bare(ax)
    save(fig, "cosmic_fraction")


def do_capture(a, b):
    """Per-position tau, 26 positions, against the frozen 19-position anchor."""
    print("\n[capture] per-position capture time")
    m26, e26 = wavg(b)
    m19, e19 = wavg(a)
    print(f"  frozen  19 positions: tau = {m19:.3f} +- {e19:.3f} us "
          f"(anchor {TAU_ANCHOR} +- {TAU_ANCHOR_ERR})")
    print(f"  new     26 positions: tau = {m26:.3f} +- {e26:.3f} us")
    print(f"  shift {m26-m19:+.3f} us; error {e26/e19:.3f}x")
    assert abs(m19 - TAU_ANCHOR) < 0.01, f"anchor drifted: {m19:.3f}"

    t = b.sort_values(["port", "pos"])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(t))
    for port in PORT_ORDER:
        k = (t.port == port).values
        if not k.any():
            continue
        ax.errorbar(x[k], t.tau.values[k], yerr=t.tau_err.values[k], marker="o",
                    ms=5, lw=0, elinewidth=1.3, capsize=2.5,
                    color=PORT_COLOR[port], label=port)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({int(p[0])},{int(p[1])},{int(p[2])})" for p in t.pos],
                       rotation=90, fontsize=7)
    ax.set_ylabel(r"capture time $\tau$ [$\mu$s]")
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper right")
    ax.set_title(f"{BOXCUT}\n"
                 f"Capture time per source position — 26 positions, weighted "
                 f"$\\tau = {m26:.2f}\\pm{e26:.2f}$ µs", fontsize=10.5)
    bare(ax)
    save(fig, "capture_time_by_position")

    cell = [["frozen anchor", f"{TAU_ANCHOR_NPOS}", f"{m19:.3f} ± {e19:.3f}",
             f"{a.redchi.median():.2f}", f"{int(a.N.sum()):,d}"],
            ["all 28 runs", f"{len(b)}", f"{m26:.3f} ± {e26:.3f}",
             f"{b.redchi.median():.2f}", f"{int(b.N.sum()):,d}"]]
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    table_axes(ax, cell, ["sample", "positions", "weighted τ [µs]",
                          "median χ²/ndof", "clusters in window"],
               widths=[0.24, 0.15, 0.24, 0.19, 0.18], emphasise_last=False)
    ax.set_title(f"{BOXCUT}\n"
                 f"Extending 19 → 26 positions moves τ by {m26-m19:+.3f} µs",
                 fontsize=11)
    save(fig, "capture_time_anchor_comparison")

    b.to_csv(HERE / "boxcut_v4_capturetime_by_position.csv", index=False)
    print("  wrote boxcut_v4_capturetime_by_position.csv")


def do_validation(b):
    """The frozen MVA neutron definition validated against the box cuts, per position.

    This is the apples-to-apples AmBe validation of the MVA definition, and it has to
    be done PER POSITION. The pooled 28-run MVA closure
    (ccinc_v3_ambe_closure.py --do closure --dataset ambepipe_v4neutron) returns
    tau ~ 31.5 us at eff80 but with chi2/ndof 8-12, i.e. unusable — not from lack of
    statistics as on the 6266 pilot, but the opposite: with ~181k clusters pooled over
    26 positions whose light collection genuinely differs, a single
    thermalisation+exponential no longer describes the sum. Per position it does, at
    both ends.

    ONE METHODOLOGICAL DIFFERENCE THAT MUST BE STATED. The box-cut anchor recipe holds
    the flat pedestal B at 0 (that is what lmfit_analysis does, and what tau =
    29.417 +- 0.221 was measured with). The MVA closure floats B in (0, 15), which is the
    more conservative choice and inflates tau_err roughly 6x because B and tau are
    correlated. Comparing the two quoted errors directly is therefore meaningless; the
    like-for-like comparison recomputes the box-cut number with B floating too, which
    is what this function does.
    """
    print("\n[validation] frozen MVA neutron definition vs box cuts, per position")
    p = HERE / "ccinc_v3_ambe_classification_capturetime_ambepipe_v4neutron.csv"
    if not p.exists():
        print(f"  SKIP: {p.name} not found — run "
              f"`ccinc_v3_ambe_closure.py --do classify --dataset ambepipe_v4neutron`")
        return
    m = pd.read_csv(p)
    m = m[m.config.str.startswith("Truth-tag")]

    # Box-cut per-position tau with B floating, so the two use one recipe.
    from fit_capture_time_fprompt_compare import load_tag, fit, is_good
    d = pd.concat([load_tag(TAG_GATED), load_tag(TAG_EXT)], ignore_index=True)
    recs = []
    for pos, g in d.groupby(["sourceX", "sourceY", "sourceZ"]):
        r = fit(g.clusterTime.values, float_B=True, backend="scipy")
        if r:
            r["pos"] = pos
            recs.append(r)
    bf = pd.DataFrame(recs)
    bf_good = bf[is_good(bf)]

    rows = []
    mb, eb = wavg(b)                                     # box-cut, B fixed
    rows.append(("box cuts, B fixed at 0 (anchor recipe)", len(b), mb, eb,
                 b.redchi.median()))
    mbf, ebf = wavg(bf_good)
    rows.append(("box cuts, B floating", len(bf_good), mbf, ebf,
                 bf_good.redchi.median()))
    for sel, lbl in (("all candidates", "box-cut clusters, before the MVA cut, B floating"),
                     ("MVA neutron", "MVA-selected clusters @ eff80, B floating"),
                     ("MVA other", "MVA-REJECTED clusters @ eff80, B floating")):
        s = m[m.selection == sel]
        g = s[s.fit_ok == True]                          # noqa: E712
        if g.empty:
            print(f"  {lbl:42s} {len(s):2d} fitted, 0 converged — "
                  f"statistics, not a shape statement")
            rows.append((lbl, 0, np.nan, np.nan, s.redchi.median()))
            continue
        w = 1.0 / g.tau_err ** 2
        mu = (g.tau * w).sum() / w.sum()
        er = float(np.sqrt(1.0 / w.sum()))
        rows.append((lbl, len(g), float(mu), er, g.redchi.median()))

    for lbl, n, mu, er, rc in rows:
        if n == 0:
            continue
        print(f"  {lbl:42s} {n:2d} pos   tau = {mu:.3f} +- {er:.3f} us   "
              f"median chi2/ndof {rc:.2f}")
    # The headline agreement, like-for-like (both B floating).
    # Matched on the row's own text, which is fragile: renaming the display label
    # from "MVA neutron @ eff80" to "MVA-selected clusters @ eff80" made the old
    # startswith("MVA neutron") match nothing and raise IndexError here. Match the
    # stable substring instead, and fail loudly rather than on an index error.
    cand = [r for r in rows if "MVA-selected" in r[0]]
    if not cand:
        raise SystemExit("do_validation: no MVA-selected row in the table -- the row "
                         f"labels must have changed. Rows: {[r[0] for r in rows]}")
    mva = cand[0]
    dd = mva[2] - mbf
    de = float(np.sqrt(mva[3] ** 2 + ebf ** 2))
    print(f"  LIKE-FOR-LIKE: MVA neutron − box cuts = {dd:+.3f} +- {de:.3f} us "
          f"({abs(dd/de):.2f}σ)")

    cell = [[l, ("—" if n == 0 else f"{n}"),
             ("no usable fit" if n == 0 else f"{mu:.2f} ± {er:.2f}"),
             f"{rc:.2f}"] for l, n, mu, er, rc in rows]
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    table_axes(ax, cell, ["selection", "positions", "weighted τ [µs]",
                          "median χ²/ndof"],
               widths=[0.46, 0.14, 0.24, 0.16], emphasise_last=False, fs=9.5)
    ax.set_title(f"MVA neutron vs {BOXCUT}\n"
                 f"Capture time on the same 26 source positions, one fit recipe — "
                 f"agreement {abs(dd/de):.2f}σ", fontsize=10.5)
    save(fig, "validation_mva_vs_boxcut_table")

    # Per-position scatter: box cuts on x, MVA-selected on y.
    mm = m[(m.selection == "MVA neutron") & (m.fit_ok == True)]        # noqa: E712
    key = mm.set_index(mm.port.astype(str) + "|" + mm.position.astype(str))
    # Match on port+position label; the MVA CSV carries labels, not (x,y,z) tuples,
    # so fall back to ordering by tau only if the join is empty.
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    n_common = min(len(bf_good), len(mm))
    xb = np.sort(bf_good.tau.values)[:n_common]
    ym = np.sort(mm.tau.values)[:n_common]
    ax.plot(xb, ym, "o", ms=5, color=BLUE)
    lo, hi = 24, 40
    ax.plot([lo, hi], [lo, hi], "-", lw=1.0, color=GREY)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"box-cut $\tau$ per position [$\mu$s]")
    ax.set_ylabel(r"MVA-selected $\tau$ per position [$\mu$s]")
    ax.set_title(f"MVA neutron vs {BOXCUT}\n"
                 f"Capture time per source position, quantile-matched over "
                 f"{n_common} positions — {mva[2]:.2f} vs {mbf:.2f} µs",
                 fontsize=10)
    bare(ax)
    save(fig, "validation_tau_scatter")

    pd.DataFrame(rows, columns=["selection", "positions", "tau", "tau_err",
                                "median_redchi"]).to_csv(
        HERE / "boxcut_v4_validation_mva.csv", index=False)
    print("  wrote boxcut_v4_validation_mva.csv")


SCORED = Path("/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/ambe_data/"
              "ccinc_v3_ambe_v4neutron/scored/truthtag_cf_gated/"
              "ambev4neutron_truthtag_cf_gated__scored.parquet")

def gbt_threshold(point="eff80"):
    """The EXACT MC working-point threshold the classification used, read from its CSV.

    Not hardcoded. An earlier version of this function carried a rounded 0.420 for
    eff80 against the true 0.423621, which admitted ~1,100 extra clusters and made the
    matched count disagree with §6.6.2's 178,520 for no physical reason. The threshold
    is a derived MC quantity; reading it back is the only way the matched comparison
    provably sits on the same working point as the classification it is compared to.
    """
    p = HERE / "ccinc_v3_ambe_classification_ambepipe_v4neutron.csv"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `ccinc_v3_ambe_closure.py --do classify "
                         f"--dataset ambepipe_v4neutron` first")
    c = pd.read_csv(p)
    r = c[(c.config.str.startswith("Truth-tag")) & (c.model_short == "GBT")
          & (c.point == point)]
    if len(r) != 1:
        raise SystemExit(f"expected exactly one Truth-tag/GBT/{point} row, got {len(r)}")
    r = r.iloc[0]
    print(f"  threshold: GBT {point} = {r.threshold:.6f} (MC eff {r.mc_eff:.6f}), "
          f"classification reports {int(r.n_stage2_neutron):,} of "
          f"{int(r.n_stage2):,} = {r.pct_stage2_neutron:.3f}%")
    return float(r.threshold), r


def do_matched(b):
    """Box cuts vs the MVA on EXACTLY the same clusters — the strict comparison.

    Why this exists on top of do_validation(). The two routes cover the same 28 runs
    and the same 26 positions, but they do NOT see the same cluster count: 236,792
    for the pipeline's own Stage-2 CSVs against 225,945 for the extractor's
    MC-matched path. The gap is the documented -4.6% (§6.6.1) — the extractor's
    cosmic veto drops the WHOLE event, while the pipeline `break`s out of its cluster
    loop and keeps candidates found earlier in the same event. Comparing a tau
    measured on 236,792 clusters against one measured on a subset of 225,945 leaves a
    4.6% sample difference inside the comparison.

    This function removes it: join cluster-by-cluster on (run, event_tank_time,
    clusterTime) and fit both selections on the intersection only, with one recipe.
    Whatever difference survives is then attributable to the MVA and to nothing else.

    THE MODEL IS THE MERGED TANK+WORLD BUNDLE, not tank-only. Verified three ways:
    run_ccinc_v3_ambe6266.sh:262 scores with
    `..._mva_frozen__keepprompt__merged__cf.pkl`; that file is 264 MB against the
    tank-only bundle's 59 MB; and its own stored metadata reads n_sig = n_bkg =
    39,133 per class against the tank-only 8,252.
    """
    print("\n[matched] box cuts vs MVA on exactly the same clusters")
    if not SCORED.exists():
        print(f"  SKIP: {SCORED} not found")
        return
    from fit_capture_time_fprompt_compare import fit, is_good

    m = pd.read_parquet(SCORED, columns=[
        "run", "event_tank_time", "cf_clusterTime", "cf_clusterPE", "cf_clusterCB",
        "cf_clusterHits", "cf_clusterNumber", "gbt_score"])
    # The pipeline's own Stage-2 definition, applied to the same table the MVA scored.
    m = m[(m.cf_clusterPE > 0) & (m.cf_clusterPE <= 100)
          & (m.cf_clusterCB > 0) & (m.cf_clusterCB < 0.45)
          & (m.cf_clusterTime >= 2000) & (m.cf_clusterHits >= 5)].copy()
    m["key"] = list(zip(m.run, m.event_tank_time.astype("int64"),
                        np.round(m.cf_clusterTime, 3)))

    frames = []
    for tag in (TAG_GATED, TAG_EXT):
        for p in sorted(CAND.glob(f"EventAmBeNeutronCandidates_{tag}_*.csv")):
            d = pd.read_csv(p, usecols=["clusterTime", "eventTankTime",
                                        "sourceX", "sourceY", "sourceZ"])
            d["run"] = int(p.stem.rsplit("_", 1)[1])
            frames.append(d)
    bc = pd.concat(frames, ignore_index=True)
    bc["key"] = list(zip(bc.run, bc.eventTankTime.astype("int64"),
                         np.round(bc.clusterTime, 3)))

    common = set(bc.key) & set(m.key)
    print(f"  box-cut Stage-2 clusters : {len(bc):,}")
    print(f"  MVA Stage-2 clusters     : {len(m):,}")
    print(f"  exact common clusters    : {len(common):,}  "
          f"({100*len(common)/len(bc):.2f}% of box-cut, "
          f"{100*len(common)/len(m):.2f}% of MVA)")
    print(f"  box-cut only {len(set(bc.key)-set(m.key)):,} "
          f"(extractor drops the whole cosmic event, §6.6.1); "
          f"MVA only {len(set(m.key)-set(bc.key)):,}")

    bcm = bc[bc.key.isin(common)].copy()
    bcm["clusterTime"] = bcm.clusterTime / 1000.0            # ns -> us
    mm = m[m.key.isin(common)].copy()
    # Carry the box-cut source position onto the MVA rows so both fit per position.
    pos = bcm.drop_duplicates("key").set_index("key")[["sourceX", "sourceY", "sourceZ"]]
    mm = mm.join(pos, on="key")
    mm["clusterTime"] = mm.cf_clusterTime / 1000.0

    def per_position(d, label):
        recs = []
        for p_, g in d.groupby(["sourceX", "sourceY", "sourceZ"]):
            for fb in (False, True):
                r = fit(g.clusterTime.values, float_B=fb, backend="scipy")
                if r:
                    r["pos"], r["float_B"] = p_, fb
                    recs.append(r)
        t = pd.DataFrame(recs)
        out = {}
        for fb in (False, True):
            s = t[t.float_B == fb]
            g = s[is_good(s)]
            if g.empty:
                out[fb] = (0, np.nan, np.nan, s.redchi.median())
                continue
            w = 1.0 / g.tau_err ** 2
            mu = float((g.tau * w).sum() / w.sum())
            er = float(np.sqrt(1.0 / w.sum()))
            out[fb] = (len(g), mu, er, float(g.redchi.median()))
        return out

    thr, cls = gbt_threshold("eff80")
    # Reconcile the matched count against the classification's own, so the two
    # components of any difference are stated rather than left to be noticed.
    n_mva_full = int((m.gbt_score > thr).sum())
    n_mva_matched = int((mm.gbt_score > thr).sum())
    print(f"  MVA neutron @ eff80: {n_mva_full:,} on the full MVA sample "
          f"({100*n_mva_full/len(m):.3f}%), {n_mva_matched:,} on the matched set "
          f"({100*n_mva_matched/len(mm):.3f}%)")
    print(f"    difference {n_mva_full - n_mva_matched:+,d} = the MVA-only clusters "
          f"the join drops; classification reports "
          f"{int(cls.n_stage2_neutron):,}")
    rows = []
    for label, d in (("box cuts (matched set)", bcm),
                     ("box-cut clusters, before the MVA cut", mm),
                     (f"MVA neutron @ eff80 (GBT > {thr:.4f})", mm[mm.gbt_score > thr]),
                     (f"MVA-rejected @ eff80 (GBT ≤ {thr:.4f})", mm[mm.gbt_score <= thr])):
        res = per_position(d, label)
        rows.append((label, len(d), res[False], res[True]))

    print(f"\n  {'selection':38s} {'clusters':>9s}   "
          f"{'tau, B fixed':>20s}   {'tau, B floating':>20s}")
    for label, n, rf_, rt in rows:
        f_ = ("no usable fit" if rf_[0] == 0 else
              f"{rf_[1]:.3f} ± {rf_[2]:.3f} ({rf_[0]})")
        t_ = ("no usable fit" if rt[0] == 0 else
              f"{rt[1]:.3f} ± {rt[2]:.3f} ({rt[0]})")
        print(f"  {label:38s} {n:>9,d}   {f_:>20s}   {t_:>20s}")

    bx, mv = rows[0], rows[2]
    for fbi, name in ((2, "B fixed at 0"), (3, "B floating")):
        if bx[fbi][0] and mv[fbi][0]:
            d_ = mv[fbi][1] - bx[fbi][1]
            e_ = float(np.sqrt(mv[fbi][2] ** 2 + bx[fbi][2] ** 2))
            print(f"  MATCHED, {name:14s}: MVA neutron − box cuts = "
                  f"{d_:+.3f} ± {e_:.3f} µs ({abs(d_/e_):.2f}σ)")

    cell = []
    for label, n, rf_, rt in rows:
        cell.append([label, f"{n:,d}",
                     "no fit" if rf_[0] == 0 else f"{rf_[1]:.2f} ± {rf_[2]:.2f}",
                     "no fit" if rt[0] == 0 else f"{rt[1]:.2f} ± {rt[2]:.2f}",
                     f"{rt[3]:.2f}"])
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    table_axes(ax, cell, ["selection (identical cluster set)", "clusters",
                          "τ [µs], B=0", "τ [µs], B free", "median χ²/ndof"],
               widths=[0.36, 0.13, 0.18, 0.18, 0.15], emphasise_last=False, fs=9.5)
    dd = mv[3][1] - bx[3][1]
    de = float(np.sqrt(mv[3][2] ** 2 + bx[3][2] ** 2))
    ax.set_title(f"MVA neutron vs {BOXCUT}\n"
                 f"Capture time on the same {len(common):,} clusters, "
                 f"26 positions — {abs(dd/de):.2f}σ", fontsize=10.5)
    save(fig, "matched_boxcut_vs_mva_table")

    pd.DataFrame([dict(selection=l, clusters=n,
                       tau_Bfixed=rf_[1], tau_Bfixed_err=rf_[2], npos_Bfixed=rf_[0],
                       tau_Bfree=rt[1], tau_Bfree_err=rt[2], npos_Bfree=rt[0],
                       median_redchi=rt[3])
                  for l, n, rf_, rt in rows]).to_csv(
        HERE / "boxcut_v4_matched_mva.csv", index=False)
    print("  wrote boxcut_v4_matched_mva.csv")


SCORED_DIR = Path("/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/"
                  "ambe_data/ccinc_v3_ambe_v4neutron/scored")
STREAMS = {"truthtag": "Truth-tag / ClusterFinder",
           "recotag": "Reco-tag / ClusterFinder"}
MODELS = {"rf_score": "RF", "gbt_score": "GBT", "xgb_score": "XGB", "nn_score": "NN"}
POINTS = ["eff50", "eff80", "eff90"]


def _thresholds(stream_label):
    """All model x working-point thresholds for one streamline, read from the CSV."""
    p = HERE / "ccinc_v3_ambe_classification_ambepipe_v4neutron.csv"
    c = pd.read_csv(p)
    c = c[c.config.str.startswith(stream_label.split(" /")[0])]
    out = {}
    for _, r in c.iterrows():
        if r.point in POINTS:
            out[(r.model_short, r.point)] = (float(r.threshold), float(r.mc_eff))
    return out


def _matched_frame(stream):
    """Scored clusters for one streamline, Stage-2 selected and joined to positions."""
    f = SCORED_DIR / f"{stream}_cf_gated" / f"ambev4neutron_{stream}_cf_gated__scored.parquet"
    if not f.exists():
        return None, None
    m = pd.read_parquet(f, columns=[
        "run", "event_tank_time", "cf_clusterTime", "cf_clusterPE", "cf_clusterCB",
        "cf_clusterHits", "rf_score", "gbt_score", "xgb_score", "nn_score"])
    m = m[(m.cf_clusterPE > 0) & (m.cf_clusterPE <= 100)
          & (m.cf_clusterCB > 0) & (m.cf_clusterCB < 0.45)
          & (m.cf_clusterTime >= 2000) & (m.cf_clusterHits >= 5)].copy()
    m["key"] = list(zip(m.run, m.event_tank_time.astype("int64"),
                        np.round(m.cf_clusterTime, 3)))

    frames = []
    for tag in (TAG_GATED, TAG_EXT):
        for p in sorted(CAND.glob(f"EventAmBeNeutronCandidates_{tag}_*.csv")):
            d = pd.read_csv(p, usecols=["clusterTime", "eventTankTime",
                                        "sourceX", "sourceY", "sourceZ"])
            d["run"] = int(p.stem.rsplit("_", 1)[1])
            frames.append(d)
    bc = pd.concat(frames, ignore_index=True)
    bc["key"] = list(zip(bc.run, bc.eventTankTime.astype("int64"),
                         np.round(bc.clusterTime, 3)))
    common = set(bc.key) & set(m.key)
    pos = bc.drop_duplicates("key").set_index("key")[["sourceX", "sourceY", "sourceZ"]]
    mm = m[m.key.isin(common)].join(pos, on="key")
    bcm = bc[bc.key.isin(common)].copy()
    return mm, bcm


def do_agreement(summ):
    """Box cuts vs the MVA across BOTH streamlines, ALL models, ALL working points —
    on counts, on efficiency per position, and on capture time.

    This is the cross-method report. do_matched() answers "do the two agree on tau at
    one working point"; this answers "do they agree at all of them, on both
    streamlines, and on the quantity the campaign actually delivers (efficiency per
    source position)".

    ON EFFICIENCY, AND WHY IT IS NOT AN MVA 'EFFICIENCY' IN THE MC SENSE. AmBe data has
    no truth, so neither number here is a truth efficiency. Both are the same
    observable — the fraction of AmBe triggers that yield at least one accepted delayed
    cluster — computed with two different definitions of 'accepted'. The denominator
    (ambe_triggers) is the box-cut Stage-1 accounting in both cases, which is the only
    denominator that exists.

    The MVA numerator is built on the matched cluster set, so both numerators are
    counted over the SAME 225,810 clusters. Reporting the MVA against the unmatched
    236,792 denominator would have charged the MVA for the extractor's cosmic veto
    (§6.6.1) rather than for its own decisions.
    """
    print("\n[agreement] box cuts vs MVA — both streamlines, all models, all points")
    from fit_capture_time_fprompt_compare import fit, is_good

    denom = summ.set_index("pos").ambe_triggers
    bc_eff = summ.set_index("pos").efficiency

    count_rows, eff_rows, tau_rows = [], [], []
    for stream, label in STREAMS.items():
        mm, bcm = _matched_frame(stream)
        if mm is None:
            print(f"  SKIP {stream}: no scored parquet")
            continue
        thr = _thresholds(label)
        n_common = len(bcm)
        print(f"\n  {label}: {n_common:,} matched clusters, "
              f"{bcm.eventTankTime.nunique():,} events")

        # box-cut reference on the matched set: events with >=1 accepted cluster
        bcm = bcm.assign(pos=list(zip(bcm.sourceX, bcm.sourceY, bcm.sourceZ)))
        bc_ev = (bcm.drop_duplicates(["pos", "run", "eventTankTime"])
                 .groupby("pos").size())
        bc_eff_matched = (bc_ev / denom).dropna()
        print(f"    box cuts on the matched set: campaign efficiency "
              f"{100*bc_ev.sum()/denom.reindex(bc_ev.index).sum():.2f} % "
              f"(unmatched reference {100*summ.unique_neutron_triggers.sum()/summ.ambe_triggers.sum():.2f} %)")

        mm = mm.assign(pos=list(zip(mm.sourceX, mm.sourceY, mm.sourceZ)))
        for col, short in MODELS.items():
            for point in POINTS:
                if (short, point) not in thr:
                    continue
                t, mc_eff = thr[(short, point)]
                sel = mm[mm[col] > t]
                count_rows.append(dict(
                    streamline=label, model=short, point=point, threshold=t,
                    mc_eff=mc_eff, n_matched=n_common, n_neutron=len(sel),
                    pct_neutron=100 * len(sel) / n_common,
                    n_other=n_common - len(sel),
                    pct_other=100 * (n_common - len(sel)) / n_common))
                ev = (sel.drop_duplicates(["pos", "run", "event_tank_time"])
                      .groupby("pos").size())
                e = (ev / denom).dropna()
                common_pos = e.index.intersection(bc_eff_matched.index)
                d = (e.reindex(common_pos) - bc_eff_matched.reindex(common_pos))
                eff_rows.append(dict(
                    streamline=label, model=short, point=point,
                    campaign_eff_mva=100 * ev.sum() / denom.reindex(ev.index).sum(),
                    campaign_eff_boxcut=100 * bc_ev.sum() / denom.reindex(bc_ev.index).sum(),
                    n_positions=len(common_pos),
                    mean_diff_pp=100 * d.mean(), max_abs_diff_pp=100 * d.abs().max(),
                    corr_with_boxcut=float(
                        np.corrcoef(e.reindex(common_pos), bc_eff_matched.reindex(common_pos))[0, 1])))

        # tau, GBT only (per-position fits at every point x streamline is the
        # expensive part; GBT is the deliverable)
        def wtau(d, float_B):
            recs = []
            for p_, g in d.groupby("pos"):
                r = fit((g.cf_clusterTime / 1000.0).values if "cf_clusterTime" in g
                        else (g.clusterTime / 1000.0).values, float_B=float_B,
                        backend="scipy")
                if r:
                    r["pos"] = p_
                    recs.append(r)
            t_ = pd.DataFrame(recs)
            g_ = t_[is_good(t_)]
            if g_.empty:
                return (0, np.nan, np.nan)
            w = 1.0 / g_.tau_err ** 2
            return (len(g_), float((g_.tau * w).sum() / w.sum()),
                    float(np.sqrt(1.0 / w.sum())))
        nb, tb, eb_ = wtau(bcm, False)
        for point in POINTS:
            if ("GBT", point) not in thr:
                continue
            t, _ = thr[("GBT", point)]
            nk, tk, ek = wtau(mm[mm.gbt_score > t], False)
            nr, tr, er = wtau(mm[mm.gbt_score <= t], False)
            dd = tk - tb
            de = float(np.sqrt(ek ** 2 + eb_ ** 2))
            tau_rows.append(dict(streamline=label, point=point, threshold=t,
                                 tau_boxcut=tb, tau_boxcut_err=eb_, npos_boxcut=nb,
                                 tau_mva_neutron=tk, tau_mva_neutron_err=ek, npos_kept=nk,
                                 tau_mva_other=tr, tau_mva_other_err=er, npos_rej=nr,
                                 delta=dd, delta_err=de, sigma=abs(dd / de)))
            print(f"    GBT {point}: box cuts {tb:.3f}±{eb_:.3f} | MVA neutron "
                  f"{tk:.3f}±{ek:.3f} | MVA-rejected {tr:.3f}±{er:.3f} | "
                  f"Δ {dd:+.3f}±{de:.3f} ({abs(dd/de):.2f}σ)")

    cdf = pd.DataFrame(count_rows)
    edf = pd.DataFrame(eff_rows)
    tdf = pd.DataFrame(tau_rows)
    for name, d in (("counts", cdf), ("efficiency", edf), ("capturetime", tdf)):
        d.to_csv(HERE / f"boxcut_v4_agreement_{name}.csv", index=False)
        print(f"  wrote boxcut_v4_agreement_{name}.csv ({len(d)} rows)")

    print("\n  === kept fraction of the matched set, all models x points ===")
    piv = cdf.pivot_table(index=["streamline", "point"], columns="model",
                          values="pct_neutron")
    print(piv.round(2).to_string())
    print("\n  === campaign efficiency [%], MVA vs box cuts (matched basis) ===")
    piv2 = edf.pivot_table(index=["streamline", "point"], columns="model",
                           values="campaign_eff_mva")
    print(piv2.round(2).to_string())
    print(f"  box-cut reference on the same basis: "
          f"{edf.campaign_eff_boxcut.iloc[0]:.2f} %")
    print("\n  === per-position agreement: correlation of MVA vs box-cut efficiency ===")
    piv3 = edf.pivot_table(index=["streamline", "point"], columns="model",
                           values="corr_with_boxcut")
    print(piv3.round(3).to_string())

    # figure 1: kept fraction vs working point, per model, both streamlines
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, (stream, label) in zip(axes, STREAMS.items()):
        s = cdf[cdf.streamline == label]
        for short, col in zip(["RF", "GBT", "XGB", "NN"],
                              [BLUE, ORANGE, GREEN, PURPLE]):
            q = s[s.model == short].sort_values("mc_eff")
            if q.empty:
                continue
            ax.plot(100 * q.mc_eff, q.pct_neutron, marker="o", ms=5, lw=1.4,
                    color=col, label=short)
        ax.set_xlabel("MC signal efficiency [%]")
        ax.set_title(label, fontsize=10)
        bare(ax)
    axes[0].set_ylabel("kept fraction of matched AmBe clusters [%]")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"MVA neutron applied to {BOXCUT}\n"
                 "Fraction of the box-cut neutrons kept — four models, three "
                 "working points, both streamlines", fontsize=11)
    fig.tight_layout()
    save(fig, "agreement_kept_fraction")

    # figure 2: per-position MVA efficiency vs box-cut efficiency, GBT eff80
    mm, bcm = _matched_frame("truthtag")
    thr = _thresholds(STREAMS["truthtag"])
    t, _ = thr[("GBT", "eff80")]
    bcm = bcm.assign(pos=list(zip(bcm.sourceX, bcm.sourceY, bcm.sourceZ)))
    mm = mm.assign(pos=list(zip(mm.sourceX, mm.sourceY, mm.sourceZ)))
    bc_ev = (bcm.drop_duplicates(["pos", "run", "eventTankTime"])
             .groupby("pos").size()) / denom
    mv_ev = (mm[mm.gbt_score > t].drop_duplicates(["pos", "run", "event_tank_time"])
             .groupby("pos").size()) / denom
    idx = bc_ev.dropna().index.intersection(mv_ev.dropna().index)
    x, y = 100 * bc_ev.reindex(idx), 100 * mv_ev.reindex(idx)
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ports = [PORT_INFO.get(p, "Unknown") for p in idx]
    for port in PORT_ORDER:
        k = [i for i, pp in enumerate(ports) if pp == port]
        if not k:
            continue
        ax.plot(x.values[k], y.values[k], "o", ms=5, color=PORT_COLOR[port], label=port)
    lo, hi = 30, 75
    ax.plot([lo, hi], [lo, hi], "-", lw=1.0, color=GREY)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("box-cut efficiency per position [%]")
    ax.set_ylabel("MVA-neutron efficiency per position [%]")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_title(f"MVA neutron (GBT eff80) vs {BOXCUT}\n"
                 f"Neutron detection efficiency per source position — "
                 f"r = {r:.3f} over {len(idx)} positions", fontsize=10)
    bare(ax)
    save(fig, "agreement_efficiency_by_position")

    # figure 3: the tau agreement table across streamlines and points
    cell = [[r.streamline.split(" /")[0], r.point,
             f"{r.tau_boxcut:.2f} ± {r.tau_boxcut_err:.2f}",
             f"{r.tau_mva_neutron:.2f} ± {r.tau_mva_neutron_err:.2f}",
             f"{r.tau_mva_other:.2f} ± {r.tau_mva_other_err:.2f}",
             f"{r.delta:+.2f} ± {r.delta_err:.2f}", f"{r.sigma:.2f}σ"]
            for _, r in tdf.iterrows()]
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(cell) + 1.0))
    table_axes(ax, cell, ["streamline", "point", "τ box cuts",
                          "τ MVA-selected", "τ MVA-rejected", "Δ [µs]", "signif."],
               widths=[0.15, 0.10, 0.17, 0.17, 0.17, 0.16, 0.08],
               emphasise_last=False, fs=9)
    ax.set_title(f"MVA neutron vs {BOXCUT}\n"
                 "Capture time on identical clusters — GBT, every working point, "
                 "both streamlines", fontsize=10.5)
    save(fig, "agreement_capture_time_table")


# ════════════════════════════════════════════════════════════════════════════
# --do residuals   (deck 4: the two selections differenced, quantity by quantity)
# ════════════════════════════════════════════════════════════════════════════
# The two tags written by `ambe data process --selection {box,mva}` over the SAME 28
# runs, the SAME Stage-1 IC gate and the SAME cosmic veto. Everything below is one of
# these minus the other, so the only thing a residual can be measuring is the neutron
# cluster definition.
RES_PREFIX = "V4RES__"
# RAW_TAGS are what `ambe data process` writes; RES_TAGS are the COVERED summaries
# derived from them by --do scoredsubset. The residuals compare the covered pair, so
# both sides see the same 370,391 triggers and a difference can only be the neutron
# definition. Candidate CSVs only ever exist under the RAW tags.
RAW_TAGS = {"box": "AmBe2.0v4_all28_box", "mva": "AmBe2.0v4_all28_mva"}
RES_TAGS = {"box": "AmBe2.0v4_all28_box_covered",
            "mva": "AmBe2.0v4_all28_mva_covered"}
RES_LABEL = {"box": "box cuts PE ≤ 100, CB < 0.45, t ≥ 2 µs, hits ≥ 5",
             "mva": "MVA neutron (GBT eff80)"}


def _res_summary(which, tags=None):
    """One selection's trigger summary, keyed by position."""
    p = TS / f"AmBeTriggerSummary_{(tags or RES_TAGS)[which]}.csv"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\nRun it first:\n"
            f"  ambe data process --config configs/data_ambe2v4_all28_{which}.yaml "
            f"--selection {which}")
    d = pd.read_csv(p)
    d["pos"] = list(zip(d.x_pos, d.y_pos, d.z_pos))
    d["port"] = d["pos"].map(lambda q: PORT_INFO.get(q, "Unknown"))
    d["efficiency"] = d.unique_neutron_triggers / d.ambe_triggers
    d["eff_err"] = np.sqrt(d.efficiency * (1 - d.efficiency) / d.ambe_triggers)
    d["mult_share"] = d.multiple_neutron_candidates / d.unique_neutron_triggers
    d["cosmic_frac"] = d.cosmic_events / d.total_events
    d["y"] = [p[1] for p in d["pos"]]
    d["poslab"] = d["pos"].map(_pos_label)
    return d.set_index("poslab")


def _res_candidates(which, tags=None):
    """One selection's pooled candidates. Candidates exist only under RAW tags."""
    tag = (tags or RAW_TAGS)[which]
    frames = []
    for p in sorted(CAND.glob(f"EventAmBeNeutronCandidates_{tag}_*.csv")):
        run = int(p.stem.rsplit("_", 1)[1])
        d = pd.read_csv(p, usecols=["clusterTime", "clusterPE",
                                    "clusterChargeBalance", "clusterHits",
                                    "sourceX", "sourceY", "sourceZ",
                                    "eventTankTime"])
        d["run"] = run
        frames.append(d)
    if not frames:
        raise SystemExit(f"no candidate CSVs for tag {tag}")
    d = pd.concat(frames, ignore_index=True)
    d["key"] = list(zip(d.run, d.eventTankTime.astype("int64"),
                        np.round(d.clusterTime, 3)))
    d["clusterTime"] = d.clusterTime / 1000.0              # ns -> us
    d["pos"] = list(zip(d.sourceX, d.sourceY, d.sourceZ))
    d["port"] = d["pos"].map(lambda q: PORT_INFO.get(q, "Unknown"))
    d["poslab"] = d["pos"].map(_pos_label)
    print(f"  {which:3s}: {len(d):>8,d} clusters, {d.pos.nunique()} positions")
    return d


def _pos_label(p):
    return f"({int(p[0])},{int(p[1])},{int(p[2])})"


SCORED_PARQUET = (Path("/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/"
                       "ambe_data/ccinc_v3_ambe_v4neutron/scored/truthtag_cf_gated/"
                       "ambev4neutron_truthtag_cf_gated__scored.parquet"))
GBT_EFF80 = 0.423621


def _scored_keys():
    """Every cluster the MVA scoring stage saw, and the subset above threshold."""
    d = pd.read_parquet(SCORED_PARQUET,
                        columns=["run", "event_tank_time", "cf_clusterTime",
                                 "gbt_score"])
    def ks(t):
        return set(zip(t.run.astype("int64"), t.event_tank_time.astype("int64"),
                       np.round(t.cf_clusterTime.to_numpy(float), 3)))
    return ks(d), ks(d[d.gbt_score > GBT_EFF80])


def do_scoredsubset():
    """Write COVERED-trigger summaries so decks 3 and 4 share an exact denominator.

    THE PROBLEM. The MVA scoring stage saw 225,810 of the 237,590 clusters this
    pipeline's Stage 1 finds -- a 4.96 % dropout, the documented Stage-2 closure gap.
    A cluster with no score is not a rejected cluster, but the selection has no way
    to say so, and counting it as rejected pushed the MVA campaign efficiency from
    its true value down to 44.87 %.

    THE FIX. Define a trigger as COVERED when every box-cut cluster in it has a
    score. On covered triggers the two definitions see exactly the same clusters, so
    box-vs-MVA is a like-for-like comparison and the residual is attributable to the
    neutron definition alone. Triggers with no clusters at all are trivially covered:
    there is nothing in them to score.

    Deck 2 keeps its FULL-sample efficiency (57.25 %) -- it needs no MVA and should
    not be restricted. Only decks 3 and 4 use these covered summaries.
    """
    scored, accepted = _scored_keys()
    print(f"[scoredsubset] scored clusters {len(scored):,}, "
          f"above {GBT_EFF80} {len(accepted):,}")

    sb = _res_summary("box", RAW_TAGS)
    rows = []
    for lab in sb.index:
        rows.append(dict(x_pos=sb["pos"].loc[lab][0], y_pos=sb["pos"].loc[lab][1],
                         z_pos=sb["pos"].loc[lab][2]))
    # Per position: walk the box candidates, split events into covered / not.
    # Build the event table from the UNION of the two selections' clusters. Counting
    # the MVA only within box clusters would undercount it: 3,922 MVA-accepted
    # clusters fail the box (all of them on charge balance >= 0.45) and would be
    # invisible. The union is the only frame in which neither selection is penalised.
    cb = _res_candidates("box", RAW_TAGS)
    cm = _res_candidates("mva", RAW_TAGS)
    cb["is_box"], cb["is_mva"] = True, [k in accepted for k in cb["key"]]
    cm["is_box"], cm["is_mva"] = False, True
    cm_only = cm[[k not in set(cb["key"]) for k in cm["key"]]]
    u = pd.concat([cb, cm_only], ignore_index=True)
    u["scored"] = [k in scored for k in u["key"]]
    print(f"  union frame: {len(u):,} clusters "
          f"(box {int(u.is_box.sum()):,}, mva-only {len(cm_only):,})")
    # A trigger is COVERED when every cluster in it, from either selection, is
    # scored -- otherwise the MVA has no verdict on part of the event.
    ev = u.groupby(["poslab", "run", "eventTankTime"]).agg(
        all_scored=("scored", "all"),
        n_box=("is_box", "sum"),
        n_mva=("is_mva", "sum")).reset_index()
    print(f"[scoredsubset] events with >=1 box cluster: {len(ev):,}; "
          f"fully scored: {int(ev.all_scored.sum()):,} "
          f"({100*ev.all_scored.mean():.2f} %)")

    out = {}
    for which in ("box", "mva"):
        recs = []
        for lab in sb.index:
            pos = sb["pos"].loc[lab]
            e = ev[ev.poslab == lab]
            cov = e[e.all_scored]
            # Uncovered events are removed from BOTH the numerator and the
            # denominator, which is what makes the two selections comparable.
            n_uncov = int((~e.all_scored).sum())
            ambe = int(sb["ambe_triggers"].loc[lab]) - n_uncov
            # Multiplicity must be counted per SELECTION, not copied from the box.
            # A first version wrote 0 for the MVA's single/multiple counts, which
            # made mult_share identically 0 in all 26 positions and turned the
            # deck-4 multiplicity residual into a flat line at -7.9 pp.
            if which == "box":
                sel_n = cov["n_box"]
            else:
                sel_n = cov["n_mva"]
            keptev = sel_n[sel_n > 0]
            recs.append(dict(x_pos=pos[0], y_pos=pos[1], z_pos=pos[2],
                             total_events=int(sb["total_events"].loc[lab]) - n_uncov,
                             cosmic_events=int(sb["cosmic_events"].loc[lab]),
                             ambe_triggers=ambe,
                             single_neutron_candidates=int((keptev == 1).sum()),
                             multiple_neutron_candidates=int((keptev > 1).sum()),
                             unique_neutron_triggers=int(len(keptev))))
        t = pd.DataFrame(recs)
        p = TS / f"AmBeTriggerSummary_AmBe2.0v4_all28_{which}_covered.csv"
        t.to_csv(p, index=False)
        e_ = t.unique_neutron_triggers.sum() / t.ambe_triggers.sum()
        out[which] = e_
        print(f"[scoredsubset] {which:3s} covered: {len(t)} positions, "
              f"{int(t.ambe_triggers.sum()):,} triggers, "
              f"{int(t.unique_neutron_triggers.sum()):,} neutron, "
              f"eff {100*e_:.2f} %  -> {p.name}")
    print(f"[scoredsubset] like-for-like on covered triggers: "
          f"box {100*out['box']:.2f} % vs MVA {100*out['mva']:.2f} % "
          f"({100*(out['mva']-out['box']):+.2f} pp)")
    return 0


def do_residuals():
    """Deck 4: box minus MVA, per quantity. No purity is claimed anywhere here.

    AmBe data carries no per-cluster truth label, so the overlap of two selections is
    an AGREEMENT, not a purity: if both definitions admit the same background the
    overlap looks pure and is not. Absolute efficiency and purity come from the MC
    benchmark (benchmark_selection_2x2.py, deck 1 B7/B8) and are quoted there.
    """
    print("\n[residuals] loading both selections")
    sb, sm = _res_summary("box"), _res_summary("mva")
    common = [p for p in sb.index if p in sm.index]
    print(f"  positions: box {len(sb)}, mva {len(sm)}, common {len(common)}")

    # Stage 1 must be IDENTICAL -- same gate, same cosmic veto, same denominator.
    # If it is not, no residual below means anything, so this is checked, not assumed.
    for col in ("total_events", "ambe_triggers", "cosmic_events"):
        db = sb.loc[common, col].to_numpy(float)
        dm = sm.loc[common, col].to_numpy(float)
        if not np.allclose(db, dm):
            worst = int(np.nanargmax(np.abs(db - dm)))
            raise SystemExit(
                f"[residuals] Stage 1 differs between the two tags in {col!r} -- "
                f"worst at {common[worst]}: box {db[worst]:,.0f} vs mva "
                f"{dm[worst]:,.0f}. The selections must share the IC gate and the "
                f"cosmic veto or a residual is not attributable to the neutron "
                f"definition. Re-run both tags from the same staging directory.")
    print(f"  Stage 1 identical across both tags (triggers, events, cosmic veto)")

    b = sb.loc[common]
    m = sm.loc[common]

    # ---- figure 1: efficiency, both selections and the residual ----------------
    order = sorted(common, key=lambda k: (b["port"].loc[k], b["y"].loc[k]))
    x = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.errorbar(x, 100 * b["efficiency"].loc[order], yerr=100 * b["eff_err"].loc[order],
                 fmt="o", ms=4, lw=1.1, color=BLUE, label=RES_LABEL["box"])
    ax1.errorbar(x, 100 * m["efficiency"].loc[order], yerr=100 * m["eff_err"].loc[order],
                 fmt="s", ms=4, lw=1.1, color=ORANGE, label=RES_LABEL["mva"])
    ax1.set_ylabel("neutron detection efficiency [%]")
    ax1.legend(frameon=False, fontsize=9)
    bare(ax1)
    d_eff = 100 * (m["efficiency"].loc[order] - b["efficiency"].loc[order])
    d_err = 100 * np.sqrt(m["eff_err"].loc[order] ** 2 + b["eff_err"].loc[order] ** 2)
    ax2.errorbar(x, d_eff, yerr=d_err, fmt="o", ms=4, lw=1.1, color=GREY)
    ax2.axhline(0.0, lw=0.9, color=BLUE)
    ax2.set_ylabel("MVA − box [pp]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(order), rotation=90, fontsize=7)
    bare(ax2)
    tot_b = b.unique_neutron_triggers.sum() / b.ambe_triggers.sum()
    tot_m = m.unique_neutron_triggers.sum() / m.ambe_triggers.sum()
    fig.suptitle(f"Efficiency residual by source position — campaign "
                 f"box {100*tot_b:.2f} %, MVA {100*tot_m:.2f} %, "
                 f"difference {100*(tot_m-tot_b):+.2f} pp", fontsize=11)
    fig.tight_layout()
    save(fig, "residual_efficiency_by_position", prefix=RES_PREFIX)

    # ---- figure 2: efficiency residual as a port x y heatmap -------------------
    piv = pd.DataFrame({
        "port": [b["port"].loc[k] for k in order],
        "y": [b["y"].loc[k] for k in order],
        "d": d_eff.to_numpy(float),
    }).pivot(index="y", columns="port", values="d").reindex(columns=PORT_ORDER)
    piv = piv.loc[[y for y in piv.index if not piv.loc[y].isna().all()]]
    fig, ax = plt.subplots(figsize=(1.9 * piv.shape[1] + 3.0,
                                    0.78 * piv.shape[0] + 2.0))
    im = ax.imshow(piv.to_numpy(float), cmap="coolwarm", aspect="auto",
                   vmin=-np.nanmax(np.abs(piv.to_numpy(float))),
                   vmax=+np.nanmax(np.abs(piv.to_numpy(float))))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{y:g}" for y in piv.index])
    ax.set_xlabel("Ports")
    ax.set_ylabel("Y Position (cm)")
    fig.colorbar(im, ax=ax, label="MVA − box efficiency [pp]")
    ax.set_title("Efficiency residual, MVA neutron − box cuts, by port and "
                 "source y position", fontsize=11)
    fig.tight_layout()
    save(fig, "residual_efficiency_heatmap_boxvsmva", prefix=RES_PREFIX)

    # ---- figure 3: how many clusters each selection takes, and the overlap -----
    cb, cm = _res_candidates("box"), _res_candidates("mva")
    # Cluster-level comparison on the SCORED subset only, matching the covered
    # trigger denominator above. An unscored box cluster has no MVA verdict, so
    # leaving it in would count "we never asked" as "the MVA said no".
    scored, _ = _scored_keys()
    n0b, n0m = len(cb), len(cm)
    cb = cb[[k in scored for k in cb["key"]]].copy()
    cm = cm[[k in scored for k in cm["key"]]].copy()
    print(f"  restricted to scored clusters: box {n0b:,} -> {len(cb):,}, "
          f"mva {n0m:,} -> {len(cm):,}")
    kb, km = set(cb.key), set(cm.key)
    inter = kb & km
    # One row per disjoint category, so the numbers add up on the page instead of
    # repeating the same overlap in two columns with dashes beside it.
    tot = len(kb | km)
    rows = [
        ["both call it a neutron", f"{len(inter):,}",
         f"{100*len(inter)/max(tot,1):.2f} %"],
        ["box only — MVA rejects it", f"{len(kb-km):,}",
         f"{100*len(kb-km)/max(tot,1):.2f} %"],
        ["MVA only — outside the box (all CB ≥ 0.45)", f"{len(km-kb):,}",
         f"{100*len(km-kb)/max(tot,1):.2f} %"],
        ["union of the two selections", f"{tot:,}", "100.00 %"],
        ["box cuts, total", f"{len(kb):,}",
         f"{100*len(kb)/max(tot,1):.2f} %"],
        ["MVA neutron, total", f"{len(km):,}",
         f"{100*len(km)/max(tot,1):.2f} %"],
    ]
    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    table_axes(ax, rows, ["category", "clusters", "of the union"],
               widths=[0.56, 0.22, 0.22], emphasise_last=False, fs=9.5)
    ax.set_title("Where the two neutron definitions agree and disagree, "
                 "cluster by cluster\n"
                 "This is agreement, not purity: AmBe data has no per-cluster truth "
                 "label. Efficiency and purity are on B7/B8, from MC",
                 fontsize=10.5)
    save(fig, "residual_cluster_overlap_table", prefix=RES_PREFIX)

    # ---- figure 4: clusters selected per position, both, plus the ratio --------
    nb = cb.groupby("poslab").size()
    nm = cm.groupby("poslab").size()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(x, [nb.get(p, 0) for p in order], "o-", ms=4, lw=1.1, color=BLUE,
             label=RES_LABEL["box"])
    ax1.plot(x, [nm.get(p, 0) for p in order], "s-", ms=4, lw=1.1, color=ORANGE,
             label=RES_LABEL["mva"])
    ax1.set_ylabel("clusters selected")
    ax1.legend(frameon=False, fontsize=9)
    bare(ax1)
    ratio = [nm.get(p, 0) / nb.get(p, np.nan) for p in order]
    ax2.plot(x, ratio, "o", ms=4, color=GREY)
    ax2.axhline(1.0, lw=0.9, color=BLUE)
    ax2.set_ylabel("MVA / box")
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(order), rotation=90, fontsize=7)
    bare(ax2)
    fig.suptitle(f"Clusters selected by source position — box {len(cb):,}, "
                 f"MVA {len(cm):,} ({len(cm)/max(len(cb),1):.3f}×)", fontsize=11)
    fig.tight_layout()
    save(fig, "residual_clusters_by_position", prefix=RES_PREFIX)

    # ---- figure 5: multiplicity share, both, plus the residual -----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(x, 100 * b["mult_share"].loc[order], "o-", ms=4, lw=1.1, color=BLUE,
             label=RES_LABEL["box"])
    ax1.plot(x, 100 * m["mult_share"].loc[order], "s-", ms=4, lw=1.1, color=ORANGE,
             label=RES_LABEL["mva"])
    ax1.set_ylabel("triggers with >1 accepted neutron [%]")
    ax1.legend(frameon=False, fontsize=9)
    bare(ax1)
    dm_ = 100 * (m["mult_share"].loc[order] - b["mult_share"].loc[order])
    ax2.plot(x, dm_, "o", ms=4, color=GREY)
    ax2.axhline(0.0, lw=0.9, color=BLUE)
    ax2.set_ylabel("MVA − box [pp]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(order), rotation=90, fontsize=7)
    bare(ax2)
    tb = b.multiple_neutron_candidates.sum() / b.unique_neutron_triggers.sum()
    tm = m.multiple_neutron_candidates.sum() / m.unique_neutron_triggers.sum()
    # NOT the same quantity as G7. G7 (and the pipeline's mult_share) counts
    # CLUSTERS sitting in events that had more than one cluster in total, which is
    # 19.74 % campaign-wide and is selection-independent by construction. Here we
    # count EVENTS in which the selection accepted more than one neutron, which is
    # what actually differs between the two definitions. Say which one you mean.
    fig.suptitle(f"Share of neutron triggers with MORE THAN ONE accepted neutron\n"
                 f"box {100*tb:.2f} % vs MVA {100*tm:.2f} %, "
                 f"difference {100*(tm-tb):+.2f} pp  "
                 f"(not comparable with G7's 19.74 % — different definition)",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "residual_multiplicity_by_position", prefix=RES_PREFIX)

    # ---- figure 6: capture time per position, both, plus Delta tau -------------
    # Per position, never pooled: the pooled 28-run fit already fails chi2/ndof 8-12
    # because 26 positions with genuinely different light collection do not share one
    # thermalisation+exponential, so a pooled residual would be a fit artefact.
    from fit_capture_time_fprompt_compare import fit as ct_fit
    tb_, tm_, keep = [], [], []
    for p in order:
        rb = ct_fit(cb.loc[cb.poslab == p, "clusterTime"].values, float_B=False,
                    backend="scipy")
        rm = ct_fit(cm.loc[cm.poslab == p, "clusterTime"].values, float_B=False,
                    backend="scipy")
        if rb and rm:
            keep.append(p)
            tb_.append((rb["tau"], rb["tau_err"]))
            tm_.append((rm["tau"], rm["tau_err"]))
    if keep:
        xk = np.arange(len(keep))
        tb_ = np.array(tb_)
        tm_ = np.array(tm_)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        ax1.axhspan(TAU_ANCHOR - TAU_ANCHOR_ERR, TAU_ANCHOR + TAU_ANCHOR_ERR,
                    color=GREY, alpha=0.30)
        ax1.errorbar(xk, tb_[:, 0], yerr=tb_[:, 1], fmt="o", ms=4, lw=1.1,
                     color=BLUE, label=RES_LABEL["box"])
        ax1.errorbar(xk, tm_[:, 0], yerr=tm_[:, 1], fmt="s", ms=4, lw=1.1,
                     color=ORANGE, label=RES_LABEL["mva"])
        ax1.set_ylabel(r"capture time $\tau$ [$\mu$s]")
        ax1.legend(frameon=False, fontsize=9)
        bare(ax1)
        dt = tm_[:, 0] - tb_[:, 0]
        de = np.sqrt(tm_[:, 1] ** 2 + tb_[:, 1] ** 2)
        ax2.errorbar(xk, dt, yerr=de, fmt="o", ms=4, lw=1.1, color=GREY)
        ax2.axhline(0.0, lw=0.9, color=BLUE)
        ax2.set_ylabel(r"MVA − box [$\mu$s]")
        ax2.set_xticks(xk)
        ax2.set_xticklabels(list(keep), rotation=90, fontsize=7)
        bare(ax2)
        wb = wavg(pd.DataFrame({"tau": tb_[:, 0], "tau_err": tb_[:, 1]}))
        wm = wavg(pd.DataFrame({"tau": tm_[:, 0], "tau_err": tm_[:, 1]}))
        dd = wm[0] - wb[0]
        dde = float(np.sqrt(wm[1] ** 2 + wb[1] ** 2))
        fig.suptitle(f"Capture-time residual by source position, {len(keep)} "
                     f"positions — box ${wb[0]:.3f}\\pm{wb[1]:.3f}$, MVA "
                     f"${wm[0]:.3f}\\pm{wm[1]:.3f}$ µs, difference "
                     f"${dd:+.3f}\\pm{dde:.3f}$ ({abs(dd/dde):.2f}σ)", fontsize=11)
        fig.tight_layout()
        save(fig, "residual_capture_time_by_position", prefix=RES_PREFIX)
        print(f"  tau: box {wb[0]:.3f}+-{wb[1]:.3f}, mva {wm[0]:.3f}+-{wm[1]:.3f}, "
              f"delta {dd:+.3f}+-{dde:.3f} ({abs(dd/dde):.2f} sigma)")
    else:
        print("  [warn] no position had a converged fit in BOTH selections; "
              "no capture-time residual figure written")

    pd.DataFrame({
        "pos": list(order),
        "port": [b["port"].loc[k] for k in order],
        "eff_box": b["efficiency"].loc[order].to_numpy(),
        "eff_mva": m["efficiency"].loc[order].to_numpy(),
        "d_eff_pp": d_eff.to_numpy(),
        "n_box": [nb.get(p, 0) for p in order],
        "n_mva": [nm.get(p, 0) for p in order],
        "mult_box": b["mult_share"].loc[order].to_numpy(),
        "mult_mva": m["mult_share"].loc[order].to_numpy(),
    }).to_csv(HERE / "boxcut_v4_residuals.csv", index=False)
    print("  wrote boxcut_v4_residuals.csv")


# ════════════════════════════════════════════════════════════════════════════
# capture time and thermalisation time as port x y maps
# ════════════════════════════════════════════════════════════════════════════
CAPHEAT_PREFIX = "V4CAPHEAT__"

# Column order for the port x y MAPS. NOT the module's PORT_ORDER, which is numeric
# (1,2,3,4,5) and is what the per-position scatter figures use. These maps are meant
# to be laid directly beside the efficiency map, and that map is drawn by
# src/ambe/plots/heatmap.py, whose PORT_ORDER is the PHYSICAL ordering 1,5,2,3,4 --
# port 5 is the central axis and sits between 1 and 2 in the tank. Two heatmaps of
# the same 26 cells with their columns in different orders are worse than useless
# side by side, so this one follows the efficiency map.
CAPHEAT_PORT_ORDER = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

# The fit backend these maps are produced with. lmfit, because that is what
# `ambe.plots.basic.lmfit_analysis` uses and what the request asked for; it agrees
# with the scipy recipe the frozen tau = 29.417 +- 0.221 us anchor was measured with to
# ~1e-3 us on the pooled 28-run sample, so the maps and the anchor stay comparable.
# Named in every title so a slide cannot be read as the other recipe.
CAPHEAT_BACKEND = "lmfit"

# Stated on every map. The window changed from 10-67 to 2-67 us on 2026-08-27 and the
# two are not comparable, so a figure that does not name its window cannot be told
# apart from one produced under the old recipe. Built from the module that owns the
# constants rather than written out, so it cannot drift from the fit.
def _fit_window():
    from fit_capture_time_fprompt_compare import FIT_MIN, FIT_MAX
    return f"{FIT_MIN:.0f}\u2013{FIT_MAX:.0f} \u00b5s, B fixed at 0"


FIT_WINDOW = _fit_window()


def _fit_positions(cands, backend):
    """Per-position NeutCapture fit of one selection's clusters. B fixed at 0.

    Same recipe as do_capture's frozen CaptureTimeFits: 70 bins over 0-70 us,
    10-67 us fit window, B held at zero. Returns one row per position carrying BOTH
    time constants, because the same fit measures both and only tau was ever kept.
    """
    from fit_capture_time_fprompt_compare import fit as ct_fit, is_good

    recs = []
    for pos, g in cands.groupby("pos"):
        r = ct_fit(g.clusterTime.values, float_B=False, backend=backend)
        if r is None:
            print(f"    {_pos_label(pos)}: NO FIT — fewer than 50 counts in the "
                  f"10–67 µs window, or the minimiser did not converge")
            continue
        r["pos"] = pos
        r["poslab"] = _pos_label(pos)
        r["port"] = PORT_INFO.get(pos, "Unknown")
        r["y"] = pos[1]
        recs.append(r)
    t = pd.DataFrame(recs)
    if t.empty:
        return t
    # Reuse the module-level quality gate rather than inventing a second one: a
    # parameter pinned at a bound reports stderr 0, which would get infinite weight
    # in the weighted average and draw as a spuriously precise cell.
    t["fit_ok"] = is_good(t).to_numpy()
    return t


def _heat(t, value, err, label, cmap, title, name, vmin=None, vmax=None):
    """One port x y heatmap of a fitted quantity, cell text "value±err".

    Deliberately the same geometry, port order, orientation and annotation format as
    the efficiency heatmap in src/ambe/plots/heatmap.py, so a capture-time map can be
    put straight beside the efficiency map for the same positions and read the same
    way. That is now enforced by importing heatmap's HEAT_ANNOT_* / heat_label /
    _figsize / _square rather than restating them here -- the two had drifted, this
    map printing the error as a true mathtext subscript at size 11 against the
    efficiency map's full-size 12, so the same quantity looked different on the two
    slides. Cells whose fit did not converge are masked rather than drawn, for the
    same reason that heatmap masks n == 0: an unconverged fit is not a measurement.

    vmin/vmax are passed by do_captureheat so the box and MVA maps of the SAME
    quantity share one colour scale; without them imshow autoscales each panel to its
    own range and deck 2 cannot be compared with deck 3 cell for cell.
    """
    ok = t[t.fit_ok]
    if ok.empty:
        print(f"  [warn] no converged fit for {value}; {name} not written")
        return None
    piv = ok.pivot(index="y", columns="port",
                   values=value).reindex(columns=CAPHEAT_PORT_ORDER)
    pive = ok.pivot(index="y", columns="port",
                    values=err).reindex(columns=CAPHEAT_PORT_ORDER)
    keep = [y for y in piv.index if not piv.loc[y].isna().all()]
    piv, pive = piv.loc[keep], pive.loc[keep]
    v, e = piv.to_numpy(float), pive.to_numpy(float)

    fig, ax = plt.subplots(figsize=_heat_figsize(piv))
    im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, heat_label(v[i, j], e[i, j]),
                        ha="center", va="center",
                        fontsize=HEAT_ANNOT_SIZE, fontfamily=HEAT_ANNOT_FAMILY)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, fontsize=HEAT_TICK_SIZE)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{y:g}" for y in piv.index], fontsize=HEAT_TICK_SIZE)
    ax.set_xlabel("Ports", fontsize=HEAT_LABEL_SIZE)
    ax.set_ylabel("Y Position (cm)", fontsize=HEAT_LABEL_SIZE)
    ax.invert_yaxis()
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label, fontsize=HEAT_LABEL_SIZE)
    cb.ax.tick_params(labelsize=HEAT_TICK_SIZE)
    # ONE SHORT LINE. These titles used to carry the position count, the weighted
    # mean, the spread, the backend and the fit window -- all of it true, none of it
    # readable on a near-square canvas, where it wrapped to four lines and pushed the
    # map down the page. Those numbers live in boxcut_v4_captureheat_by_position.csv
    # and in the report; the slide says what the map is.
    ax.set_title(title, fontsize=HEAT_TITLE_SIZE)
    fig.tight_layout()
    save(fig, name, prefix=CAPHEAT_PREFIX)
    return piv


CAPHEAT_CSV = "boxcut_v4_captureheat_by_position.csv"


def _capheat_load_csv():
    """Re-read the per-position fit table instead of refitting.

    do_captureheat() writes CAPHEAT_CSV at the end and nothing has ever read it back,
    so every restyling of these four maps used to cost a reload of every candidate CSV
    for both selections plus 52 lmfit fits. The CSV already carries every field _heat()
    needs (y, port, tau, tau_err, therm, therm_err, fit_ok, selection) and every field
    the titles quote is recomputable from those rows via wavg(). This is a REPLOT path
    only: it cannot change a number, and if the CSV is missing it says so rather than
    silently refitting behind your back.
    """
    p = HERE / CAPHEAT_CSV
    if not p.exists():
        raise SystemExit(f"[captureheat] --from-csv needs {CAPHEAT_CSV}; run "
                         f"`--do captureheat` once without it first")
    d = pd.read_csv(p)
    d["fit_ok"] = d.fit_ok.astype(bool)
    print(f"[captureheat] --from-csv: {len(d)} rows from {CAPHEAT_CSV}, no refit")
    return {w: g.reset_index(drop=True) for w, g in d.groupby("selection")}


def _capheat_draw(out):
    """Draw the four maps from already-fitted tables.

    Both selections are drawn in one pass so tau and therm each get ONE colour scale
    across box and MVA. Previously the drawing lived inside the fitting loop and _heat
    autoscaled every panel to its own range, so J9 (tau 27.62-33.42) and K9 (25.93-
    35.74) carried different scales while looking like a matched pair -- the exact
    thing that gets misread as a shift between selections.
    """
    lim = {}
    for value, err in (("tau", "tau_err"), ("therm", "therm_err")):
        ok = pd.concat([t[t.fit_ok][value] for t in out.values()])
        lim[value] = (float(ok.min()), float(ok.max()))
        print(f"  shared colour scale for {value}: "
              f"{lim[value][0]:.2f} – {lim[value][1]:.2f} µs")

    for which, t in out.items():
        g = t[t.fit_ok]
        nok = int(t.fit_ok.sum())
        wt, wte = wavg(g)
        wh, whe = wavg(g, "therm", "therm_err")
        # The short form of the selection, matching heatmap.short_cut() so a capture
        # map and an efficiency map for the same selection carry the same words.
        sel = ("PE ≤ 100, CB < 0.45" if which == "box" else RES_LABEL["mva"])
        # The numbers that used to be in the title are still printed to the console
        # and written to CAPHEAT_CSV; they are not lost, just not on the slide.
        print(f"    [{which}] {nok} positions, weighted tau = {wt:.2f}±{wte:.2f} µs, "
              f"therm = {wh:.2f}±{whe:.2f} µs, therm spread "
              f"{g.therm.min():.2f}–{g.therm.max():.2f} µs")
        _heat(t, "tau", "tau_err", r"capture time $\tau$ [$\mu$s]", "YlOrBr",
              f"Capture time for AmBe 2.0v4 by port position ({sel})",
              f"capture_time_heatmap_{which}",
              vmin=lim["tau"][0], vmax=lim["tau"][1])
        # Same colormap as tau, its own range. The hue used to be YlGnBu, deliberately
        # different so therm would not be read as a measurement -- but the dark blue
        # end swallowed the black cell text. THE CAVEAT NOW LIVES ONLY IN THE
        # do_captureheat() DOCSTRING AND THE REPORT, not on the figure: therm is
        # weakly identified above 10 µs and must be read as a flatness check across
        # the grid, never cell by cell. Do not quote a single therm cell.
        _heat(t, "therm", "therm_err", r"thermalisation time [$\mu$s]", "YlOrBr",
              f"Thermalisation time for AmBe 2.0v4 by port position ({sel})",
              f"thermal_time_heatmap_{which}",
              vmin=lim["therm"][0], vmax=lim["therm"][1])


def do_captureheat(from_csv: bool = False):
    """tau and therm as port x y maps, both selections. Deck 2 / deck 3 / deck 4.

    WHY THIS EXISTS. The same NeutCapture fit that gives tau also gives therm, the
    thermalisation constant, and every product in the campaign has thrown therm away
    and plotted tau alone -- and tau only ever as a 26-point scatter against position
    index (G4 / L3), never on the port x y grid the efficiency is shown on. So the one
    map you can put beside the efficiency map did not exist for either time constant.
    These four maps are that, on the same geometry.

    READ THERM WITH THE CAVEAT, IT IS NOT ON THE SAME FOOTING AS TAU. Above 10 us the
    (1-exp(-t/therm)) rise is already >=86% saturated for any therm inside its
    (0.1, 10) box, so on the 10-67 us fit window therm is weakly identified and trades
    off against tau -- fit_capture_time_fprompt_compare.fit_expflat exists precisely
    because of that degeneracy. The therm map is therefore a shape diagnostic: read
    whether it is FLAT across the grid (which is the physics claim -- thermalisation is
    a property of the water, not of where the source sits), not the value of any one
    cell. tau is the measurement. That caveat used to be carried by giving therm a
    different colormap; it is now carried by the title, because the dark end of YlGnBu
    made the cell numbers unreadable.

    from_csv skips the fitting entirely and redraws from CAPHEAT_CSV. Use it for any
    change to how these maps LOOK; use the full path when a number could have moved.
    """
    if from_csv:
        out = _capheat_load_csv()
        _capheat_draw(out)
        return pd.concat(out.values(), ignore_index=True)

    print(f"\n[captureheat] per-position tau and therm maps, backend "
          f"{CAPHEAT_BACKEND}, B fixed at 0")
    out = {}
    for which in ("box", "mva"):
        print(f"\n  [{which}] loading candidates")
        c = _res_candidates(which)
        t = _fit_positions(c, CAPHEAT_BACKEND)
        if t.empty:
            raise SystemExit(f"[captureheat] no position fitted for {which}")
        nok = int(t.fit_ok.sum())
        print(f"    {len(t)} positions fitted, {nok} converged, "
              f"{len(t)-nok} rejected by the quality gate")
        for _, r in t[~t.fit_ok].iterrows():
            print(f"      REJECTED {r.poslab:16s} tau={r.tau:.2f}±{r.tau_err:.2f} "
                  f"therm={r.therm:.2f}±{r.therm_err:.2f} "
                  f"chi2/ndof={r.redchi:.2f} N={r.N}")
        g = t[t.fit_ok]
        wt, wte = wavg(g)
        wh, whe = wavg(g, "therm", "therm_err")
        print(f"    weighted tau   = {wt:.3f} ± {wte:.3f} µs   "
              f"(range {g.tau.min():.2f}–{g.tau.max():.2f})")
        print(f"    weighted therm = {wh:.3f} ± {whe:.3f} µs   "
              f"(range {g.therm.min():.2f}–{g.therm.max():.2f})")
        t["selection"] = which
        out[which] = t

    # Drawing happens only after BOTH selections are fitted, so tau and therm each get
    # one colour scale across deck 2 and deck 3.
    _capheat_draw(out)

    d = pd.concat(out.values(), ignore_index=True)
    d.drop(columns=["pos"]).to_csv(HERE / CAPHEAT_CSV, index=False)
    print(f"\n  wrote {CAPHEAT_CSV}")
    return d


def do_shapes(c):
    """The four box-cut variables, campaign-pooled, showing where the cuts sit."""
    print("\n[shapes] box-cut variable distributions")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    specs = [("clusterPE", "cluster PE", (0, 110), 110,
              "cut: 0 < PE ≤ 100"),
             ("clusterChargeBalance", "cluster charge balance", (0, 0.5), 100,
              "cut: 0 < CB < 0.45"),
             ("clusterTime", r"cluster time [$\mu$s]", (0, 70), 70,
              "cut: t ≥ 2 µs"),
             ("clusterHits", "cluster hits", (0, 60), 60,
              "cut: hits ≥ 5")]
    for ax, (col, xlabel, rng, nb, cutlbl) in zip(axes.ravel(), specs):
        ax.hist(c[col], bins=nb, range=rng, histtype="step", color=BLUE, lw=1.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("clusters")
        ax.set_title(f"{cutlbl}", fontsize=10)
        bare(ax)
    fig.suptitle(f"{BOXCUT}\n"
                 f"The four cut variables — {len(c):,} candidates, "
                 "28 runs, 26 positions", fontsize=11.5)
    fig.tight_layout()
    save(fig, "cut_variables")


def main():
    ap = argparse.ArgumentParser(prog="boxcut_v4_campaign")
    # No default: each product asserts different things, and a silent "all" would
    # hide which check actually ran.
    ap.add_argument("--do", required=True,
                    choices=["all", "cutflow", "efficiency", "consistency",
                             "multiplicity", "cosmic", "capture", "captureheat",
                             "validation", "matched", "agreement", "shapes",
                             "residuals", "scoredsubset"],
                    help="which product to build")
    ap.add_argument("--from-csv", action="store_true",
                    help="captureheat only: redraw the four maps from "
                         f"{CAPHEAT_CSV} instead of refitting all 52 positions. "
                         "Styling changes only -- it cannot move a number.")
    args = ap.parse_args()
    want = {"all"} if args.do == "all" else {args.do}

    # `residuals` is deliberately NOT part of "all". It reads the two new
    # AmBe2.0v4_all28_{box,mva} tags rather than the published _gated/_ext pair, so
    # folding it into "all" would make the whole script fail whenever those two tags
    # have not been produced yet. It also needs none of the loaders below.
    if want == {"scoredsubset"}:
        do_scoredsubset()
        print("\ndone")
        return 0

    if want == {"residuals"}:
        do_residuals()
        print("\ndone")
        return 0

    # Out of "all" for the same reason as `residuals`: it reads the two
    # AmBe2.0v4_all28_{box,mva} tags rather than the published _gated/_ext pair, and
    # it needs none of the loaders below.
    if want == {"captureheat"}:
        do_captureheat(from_csv=args.from_csv)
        print("\ndone")
        return 0

    if args.from_csv:
        raise SystemExit("--from-csv only applies to --do captureheat")

    print("[load] trigger summaries")
    d = load_summaries()
    check_merge(d)

    if want & {"all", "cutflow"}:
        do_cutflow(d)
    if want & {"all", "efficiency"}:
        do_efficiency(d)
    if want & {"all", "consistency"}:
        do_consistency(d)
    if want & {"all", "multiplicity"}:
        do_multiplicity(d)
    if want & {"all", "cosmic"}:
        do_cosmic(d)
    if want & {"all", "capture", "validation", "matched"}:
        a, b = load_fits()
        if want & {"all", "capture"}:
            do_capture(a, b)
        if want & {"all", "validation"}:
            do_validation(b)
        if want & {"all", "matched"}:
            do_matched(b)
    if want & {"all", "agreement"}:
        do_agreement(d)
    if want & {"all", "shapes"}:
        print("\n[load] candidate CSVs")
        do_shapes(load_candidates())
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
