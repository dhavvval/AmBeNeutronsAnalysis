#!/usr/bin/env python
"""
boxcut_v4_special.py — the traditional box-cut analysis on the AmBe v4 SPECIAL runs,
reported entirely separately from the 28-run source campaign.

WHAT IS DELIBERATELY NOT PRODUCED HERE: efficiency and capture time. These six runs
are not ordinary source runs, so folding them into a campaign efficiency or a tau
would be wrong regardless of how the numbers came out. The split follows the
diagnostics' own step7_source_presence.csv, not a judgement call:

  6254, 6256  kind=pulser, no source, and ZERO IC-passing triggers. With no AmBe
              source there is nothing for the gate to select. That is the result.
  6264        labelled no-source, cosmic-dominated (10x the prompt rate, tau 48 us
              in the diagnostics). Expected to survive the cosmic veto barely at all.
  6265, 6270  labelled no-source but behave source-in (diagnostics tau 35.5 / 34.7 us).
  6266        labelled no-source but behaves source-in (diagnostics tau 36.4 us).

WHY THE GATE COMES FROM A PARQUET. None of these runs has an
EventAmBeNeutronCandidates_*.csv — that directory stops at 6242 — so Stage 1 is taken
from the special-runs diagnostics join, step4_joined_<run>.parquet, whose `timestamp`
column IS eventTimeTank and whose `passes_ic` column is the same 700-1200 IC window
plus second-pulse veto the pipeline applies. make_gate_csvs_from_parquet.py owns that
conversion and warns that a clock mismatch silently produces an EMPTY gate, which
downstream reads as "success" on zero events. So this script re-verifies the
timestamp match per run and refuses to continue on a gate that does not overlap.

Stage 2 is the pipeline's own box cuts, imported from src/ambe/data/processor.py so
they cannot drift from the campaign: cosmic_cut, ambe_single_cut, ambe_multiple_cut.

Source positions come from SPECIAL_POSITIONS in ccinc_v3_ambe_closure.py and are
NOT added to processor.py `source_positions` — that map drives the campaign
efficiency heatmaps, and putting labelled-no-source runs into it would silently fold
them into the campaign numbers.

Run 6270 caveat, carried on every output: BeamCluster_6270.root is still the
corrupted merge — 11,195 duplicated events of 46,211 (24%), see FIX_6270.md.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    MPLBACKEND=Agg python -u boxcut_v4_special.py --do all
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.data.processor import AmBeNeutronProcessing, CutCriteria

from make_plots_ccinc_v3_merged import (OUTD, BLUE, GREY, ORANGE, GREEN, PURPLE,
                                        bare, table_axes)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).parent
PREFIX = "V4BOXSPEC__"
# Same wording as boxcut_v4_campaign.BOXCUT, so the special-run slides and the
# campaign slides name the selection identically.
BOXCUT = "box cuts PE ≤ 100, CB < 0.45, t ≥ 2 µs, hits ≥ 5"
DIAG = Path("/exp/annie/data/users/dajana/AmBe_special_runs_diagnostics/outputs")
BC_DIR = Path("/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4")

# Runs with no source and no IC-passing triggers. Reported, never processed.
PULSER_RUNS = [6254, 6256]
# The four labelled-no-source runs that do have gated triggers.
SPECIAL_RUNS = [6264, 6265, 6266, 6270]

# Mirrored from ccinc_v3_ambe_closure.py SPECIAL_POSITIONS / SPECIAL_PORT.
SPECIAL_POSITIONS = {6264: (0.0, 0.0, 0.0), 6265: (0.0, 0.0, 0.0),
                     6266: (0.0, 100.0, 0.0), 6270: (75.0, 0.0, 0.0)}
SPECIAL_PORT = {6264: "Port 5", 6265: "Port 5", 6266: "Port 5", 6270: "Port 4"}
SPECIAL_NOTE = {
    6264: "cosmic-dominated (diagnostics: 10x prompt rate, tau 48 us)",
    6265: "behaves source-in (diagnostics tau 35.5 us)",
    6266: "behaves source-in (diagnostics tau 36.4 us)",
    6270: "behaves source-in; BeamCluster is the CORRUPTED merge (FIX_6270.md)",
}
# Independently established in ANALYSIS_ccinc_v3_FULL.md §6.7 via the MVA path.
# Asserted here so the two routes cannot silently disagree.
EXPECT_GATED = {6264: 149, 6265: 2150, 6266: 2171, 6270: 2312}


def save(fig, name, prefix=None):
    # `prefix` overrides the module default so the deck-5 diagnostic figures land
    # under V4SPECDIAG__ instead of V4BOXSPEC__. They are not box-cut results on the
    # special runs -- they are what the special runs look like BEFORE and AFTER the
    # box, which is a different question and a different deck.
    stem = OUTD / f"{prefix or PREFIX}{name}"
    OUTD.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  [fig] {stem.name}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — the gate, from the diagnostics join
# ════════════════════════════════════════════════════════════════════════════
def check_pulsers():
    """6254/6256 must have zero IC-passing triggers. That claim is the result."""
    print("\n[pulser] dummy-trigger runs — expect zero IC-passing triggers")
    rows = []
    for run in PULSER_RUNS:
        p = DIAG / f"step4_joined_{run}.parquet"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        d = pd.read_parquet(p, columns=["run", "timestamp", "passes_ic"])
        n_pass = int(d.passes_ic.astype(bool).sum())
        print(f"  run {run}: {n_pass} of {len(d):,} IC candidates pass")
        assert n_pass == 0, (
            f"run {run} has {n_pass} IC-passing triggers — it is not a null run "
            f"and must not be reported as one")
        rows.append(dict(run=run, kind="pulser", ic_candidates=len(d),
                         ic_passing=0, gated_triggers=0, cosmic_vetoed=0,
                         ambe_triggers=0, single=0, multiple=0, candidates=0))
    print("  PASS  both pulser runs have exactly zero IC-passing triggers")
    return pd.DataFrame(rows)


def build_gate(run):
    """IC-passing eventTimeTank set for one run, verified against the ntuple clock."""
    p = DIAG / f"step4_joined_{run}.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = pd.read_parquet(p, columns=["run", "timestamp", "passes_ic"])
    n_all = len(d)
    gate = set(d.loc[d.passes_ic.astype(bool), "timestamp"].astype("int64"))
    if not gate:
        raise SystemExit(f"run {run}: empty gate — refusing to continue, an empty "
                         f"gate silently yields zero clusters and reads as success")
    return gate, n_all, set(d.timestamp.astype("int64"))


def load_bc(run):
    """BeamCluster event data for one run (ANNIEEventTreeMaker layout)."""
    f = BC_DIR / f"BeamCluster_{run}.root"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    with uproot.open(f) as fh:
        t = fh["Event"]
        return {k: t[k].array(library="np") for k in
                ("eventNumber", "eventTimeTank", "numberOfClusters",
                 "clusterTime", "clusterPE", "clusterChargeBalance", "clusterHits")}


def process_run(run, proc):
    """Stage 1 gate + Stage 2 box cuts for one special run.

    The cluster loop mirrors AmBeNeutronProcessing.process_events_efficient
    exactly, including the `break` on the cosmic veto — which drops the whole
    event, not just the offending cluster.
    """
    gate, n_ic, all_ts = build_gate(run)
    ev = load_bc(run)
    ett = ev["eventTimeTank"].astype("int64")

    # The check make_gate_csvs_from_parquet.py explicitly asks for: a different
    # clock produces an empty intersection and a silently empty analysis.
    bc_ts = set(ett.tolist())
    matched = len(all_ts & bc_ts)
    frac = matched / max(len(all_ts), 1)
    print(f"  clock check: {matched:,}/{len(all_ts):,} diagnostics timestamps "
          f"({frac:.1%}) match a BeamCluster eventTimeTank")
    assert frac > 0.5, (f"run {run}: only {frac:.1%} of diagnostics timestamps match "
                        f"the ntuple clock — the gate is not on the same clock")

    # DE-DUPLICATE BY eventTimeTank. A repeated eventTimeTank is the same physical
    # trigger written twice, so counting it twice inflates every yield. This is not
    # hypothetical: run 6270's BeamCluster is the corrupted merge and carries 11,279
    # duplicated events of 46,211 (24.4%), which is exactly the excess an undeduped
    # loop reports. 6265 carries 337 (0.8%) from the same class of merge fault. The
    # event key is eventTimeTank, never eventNumber — eventNumber restarts per part
    # file and cannot identify a trigger.
    n_dup_total = len(ett) - len(set(ett.tolist()))
    n_gated = n_cosmic = n_single = n_multi = 0
    n_dup_gated = 0
    seen_events = set()
    seen_multi = set()
    rows = []
    for i in range(len(ett)):
        if ett[i] not in gate:
            continue
        if int(ett[i]) in seen_events:
            n_dup_gated += 1
            continue
        seen_events.add(int(ett[i]))
        n_gated += 1
        CT, CPE = ev["clusterTime"][i], ev["clusterPE"][i]
        CCB, CH = ev["clusterChargeBalance"][i], ev["clusterHits"][i]
        CN = ev["numberOfClusters"][i]
        vetoed = False
        for k in range(len(CT)):
            if proc.cosmic_cut(CT[k], CPE[k]):
                n_cosmic += 1
                vetoed = True
                break
            single = proc.ambe_single_cut(CPE[k], CCB[k], CT[k], CN, CH[k])
            multi = proc.ambe_multiple_cut(CPE[k], CCB[k], CT[k], CN, CH[k])
            if single or multi:
                rows.append(dict(run=run, eventTankTime=int(ett[i]),
                                 clusterTime=CT[k] / 1000.0, clusterPE=CPE[k],
                                 clusterChargeBalance=CCB[k], clusterHits=int(CH[k]),
                                 clusterNumber=int(CN)))
            if single:
                n_single += 1
            if multi and ett[i] not in seen_multi:
                n_multi += 1
                seen_multi.add(int(ett[i]))
        del vetoed

    x, y, z = SPECIAL_POSITIONS[run]
    summary = dict(run=run, kind="nosource", port=SPECIAL_PORT[run],
                   x_pos=x, y_pos=y, z_pos=z,
                   ic_candidates=n_ic, ic_passing=len(gate),
                   gated_triggers=n_gated, cosmic_vetoed=n_cosmic,
                   cosmic_frac=n_cosmic / max(n_gated, 1),
                   ambe_triggers=n_gated - n_cosmic,
                   single=n_single, multiple=n_multi,
                   candidates=len(rows),
                   dup_events_in_file=n_dup_total, dup_gated_dropped=n_dup_gated,
                   note=SPECIAL_NOTE[run])
    if n_dup_gated:
        print(f"  dropped {n_dup_gated:,} duplicated gated trigger(s) "
              f"({n_dup_total:,} duplicated events in the file overall)")
    print(f"  gated {n_gated:,} | cosmic-vetoed {n_cosmic:,} "
          f"({100*summary['cosmic_frac']:.1f}%) | AmBe triggers "
          f"{summary['ambe_triggers']:,} | Stage-2 candidates {len(rows):,} "
          f"(single {n_single:,}, multi-events {n_multi:,})")
    # Cross-check against the MVA route's gated-trigger counts (§6.7), REPORTED not
    # asserted: that route re-derives cluster features in a (-5,+20) ns delayed
    # residual window, so its cosmic veto sees different clusterPE values and can
    # legitimately disagree at the few-event level. A large disagreement still means
    # something is wrong and is flagged.
    if run in EXPECT_GATED:
        exp = EXPECT_GATED[run]
        d = n_gated - exp
        flag = "" if abs(d) <= max(5, 0.02 * exp) else "   <-- LARGE, investigate"
        print(f"  cross-check vs MVA route (§6.7): {n_gated:,} here vs {exp:,} "
              f"there ({d:+d}){flag}")
        summary["mva_route_gated"] = exp
        summary["mva_route_delta"] = d
    return summary, pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# products
# ════════════════════════════════════════════════════════════════════════════
def fig_table(summ):
    print("\n[table] special-run cut flow")
    cell = []
    for _, r in summ.iterrows():
        if r.kind == "pulser":
            cell.append([f"{int(r.run)}", "pulser", f"{int(r.ic_candidates):,d}",
                         "0", "—", "—", "0"])
        else:
            cell.append([f"{int(r.run)}", f"{r.port}, y={r.y_pos:g}",
                         f"{int(r.ic_candidates):,d}", f"{int(r.gated_triggers):,d}",
                         f"{100*r.cosmic_frac:.1f}%",
                         f"{int(r.ambe_triggers):,d}", f"{int(r.candidates):,d}"])
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    table_axes(ax, cell, ["run", "position", "IC candidates", "gated triggers",
                          "cosmic-vetoed", "AmBe triggers", "Stage-2 candidates"],
               widths=[0.10, 0.19, 0.16, 0.15, 0.14, 0.14, 0.17],
               emphasise_last=False, fs=9.5)
    ax.set_title(f"Special runs, {BOXCUT}\n"
                 "Cut flow — reported separately, never merged; no efficiency and "
                 "no τ quoted", fontsize=11)
    save(fig, "cutflow_table")


def fig_yields(summ):
    """Stage-2 candidates per gated trigger — a yield, explicitly not an efficiency."""
    print("\n[yield] Stage-2 candidates per AmBe trigger (a yield, NOT an efficiency)")
    s = summ[summ.kind == "nosource"].copy()
    s["yield_per_trig"] = s.candidates / s.ambe_triggers.replace(0, np.nan)
    for _, r in s.iterrows():
        v = r.yield_per_trig
        print(f"  run {int(r.run)}: "
              + ("no AmBe triggers survive the cosmic veto" if not np.isfinite(v)
                 else f"{v:.3f} candidates/trigger") + f"   [{r.note}]")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    lbl = [f"{int(r.run)}" for _, r in s.iterrows()]
    val = [0.0 if not np.isfinite(v) else v for v in s.yield_per_trig]
    col = [ORANGE if int(r.run) == 6264 else BLUE for _, r in s.iterrows()]
    ax.bar(lbl, val, color=col, width=0.6)
    ax.set_xlabel("run")
    ax.set_ylabel("Stage-2 candidates per AmBe trigger")
    ax.set_title(f"Special runs, {BOXCUT}\n"
                 "Candidates per AmBe trigger — 6264 gives zero clusters "
                 f"({100*s.set_index('run').loc[6264,'cosmic_frac']:.1f}% "
                 "cosmic-vetoed)", fontsize=10.5)
    bare(ax)
    save(fig, "yield_per_trigger")


def fig_cosmic(summ):
    print("\n[cosmic] veto fraction — the 6264 discriminator")
    s = summ[summ.kind == "nosource"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    col = [ORANGE if int(r.run) == 6264 else BLUE for _, r in s.iterrows()]
    ax.bar([f"{int(r.run)}" for _, r in s.iterrows()],
           [100 * r.cosmic_frac for _, r in s.iterrows()], color=col, width=0.6)
    ax.set_xlabel("run")
    ax.set_ylabel("cosmic-vetoed fraction of gated triggers [%]")
    lo = 100 * s[s.run != 6264].cosmic_frac.min()
    hi = 100 * s[s.run != 6264].cosmic_frac.max()
    ax.set_title("Special runs, cosmic veto (clusterTime < 2 µs or PE > 100)\n"
                 f"6264 vetoes at {100*s.set_index('run').loc[6264,'cosmic_frac']:.1f}% "
                 f"against {lo:.0f}–{hi:.0f}% for the other three",
                 fontsize=10.5)
    bare(ax)
    save(fig, "cosmic_fraction")


def fig_shapes(cands, summ):
    """Special-run cluster shapes, per run. Shape only — no fit, no tau."""
    print("\n[shapes] cluster distributions per special run")
    runs = [r for r in SPECIAL_RUNS if len(cands[cands.run == r])]
    if not runs:
        print("  no run has any Stage-2 candidate; nothing to plot")
        return
    cols = [("clusterTime", r"cluster time [$\mu$s]", (0, 70), 35),
            ("clusterPE", "cluster PE", (0, 110), 44),
            ("clusterChargeBalance", "cluster charge balance", (0, 0.5), 25),
            ("clusterHits", "cluster hits", (0, 60), 30)]
    palette = {6265: BLUE, 6266: ORANGE, 6270: GREEN, 6264: GREY}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    for ax, (c, xlabel, rng, nb) in zip(axes.ravel(), cols):
        for r in runs:
            v = cands.loc[cands.run == r, c]
            ax.hist(v, bins=nb, range=rng, histtype="step", lw=1.4,
                    density=True, color=palette[r], label=f"{r} (n={len(v):,})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalised")
        ax.legend(frameon=False, fontsize=8)
        bare(ax)
    fig.suptitle(f"Special runs, {BOXCUT}\n"
                 "The four cut variables, area-normalised; no capture-time fit is "
                 "quoted (per-run statistics are insufficient)", fontsize=11)
    fig.tight_layout()
    save(fig, "cluster_shapes")


# ════════════════════════════════════════════════════════════════════════════
# DECK 5 — the special runs as their own analysis, not as four bars beside the
#          campaign
# ════════════════════════════════════════════════════════════════════════════
# WHAT THIS SECTION IS FOR, AND WHY IT IS NOT THE SECTION ABOVE.
#
# Everything above answers "what does the campaign selection do to these runs?", and
# for the runs that matter most the answer is a single number: 6254 and 6256 give ZERO
# IC-passing triggers, 6264 gives ZERO Stage-2 candidates. Three bars at zero is a
# true summary and a useless one -- it says nothing about what those runs actually
# contain, which is the whole reason they were taken.
#
# So this section runs the SAME flow the campaign runs (IC waveform stage, then the
# tank cluster stage, then the Stage-2 box) but reports the distributions at each
# stage instead of the survivor count, and reports them WITHOUT the IC gate as well as
# with it. That is not a relaxation of the analysis; for 6254/6256 it is the only
# option that exists, because the gate keeps nothing and a gated plot of nothing is
# blank.
#
# THE THREE QUESTIONS, ONE PER RUN:
#
#   6254 / 6256   DUMMY TRIGGERS -- `kind=pulser` in the diagnostics, LAPPD on and
#                 off respectively (established two independent ways in
#                 LAPPD_PMTcharge_6254_6256). The readout fires on a schedule, not on
#                 anything in the detector, so each event is a RANDOM SLICE of the
#                 tank. That is what makes them the right tool for two questions:
#                 what charge does the tank register with nothing to detect, and what
#                 is the ACCIDENTAL cluster rate the box would accept anyway. Shown
#                 ungated because a scheduled trigger carries no BGO gamma and so
#                 puts 0 of 7,933 and 0 of 25,525 waveforms in the 700-1200 window.
#
#   6264          NO-SOURCE RUN -- the real trigger logic is running, the source is
#                 simply not there. 808 waveforms still land in the IC window and 589
#                 pass the second-pulse veto. With no source those are FALSE STARTS of
#                 the IC trigger by construction, so this run measures the false-start
#                 rate directly; that is a systematic of the AmBe trigger and it
#                 cannot be measured on a dummy-trigger run, which never exercises the
#                 trigger logic at all. The tank side is the matching study: 9.6
#                 clusters per trigger against 1.9 for a source-in run. Note it is
#                 also cosmic-loaded (66.6 % cosmic-vetoed against 13.6 % for the
#                 source-in specials), so the rate must be scaled before it is applied
#                 to a normal run.
#
#   6266          the contrast, and the reason it is in these plots at all: it is
#                 labelled no-source but behaves source-in (tau 36.4 us). Without a
#                 source-in shape on the same axes there is nothing to call 6254,
#                 6256 and 6264 anomalous WITH RESPECT TO. It is never merged and no
#                 efficiency or tau is quoted for it here.
DIAG_PREFIX = "V4SPECDIAG__"

# The Stage-1 IC waveform gate, stated once and reused in every title. Same numbers
# as ambe.stats.waveform applies; named here so a deck-5 slide can be read alone.
IC_LO, IC_HI = 700.0, 1200.0
IC_GATE = f"IC gate {IC_LO:.0f} < IC_adjusted < {IC_HI:.0f} + second-pulse veto"

# 6266 was here and is deliberately gone. It is a source-in run at Port 5 but at
# y = 100 cm, while 6264 sits at y = 0, so every overlay that used it as the
# source-in reference carried a ~10 pp geometry difference inside the comparison.
# 6265 is source-in at (0, 0, 0), the SAME position as 6264 to the centimetre, so
# the whole deck -- the 1D overlays, the per-run IC pages, the cut flows and the
# stage grids -- now compares like with like. 6266 keeps its entries below and is
# still in SPECIAL_RUNS for the H-block provenance table; it is simply not drawn.
DIAG_RUNS = [6254, 6256, 6264, 6265]

# WHAT A CURVE IS, not which run it came from. A legend entry reading "6254" makes
# every one of these figures unreadable without the run table in front of you, and the
# whole point of the deck is what the runs ARE. The run number stays on the per-run
# panel titles and in every table, so nothing becomes untraceable.
#
# THESE THREE ARE NOT THE SAME KIND OF RUN AND THE LABELS MUST NOT BLUR THEM.
# 6254/6256 are DUMMY (forced/pulser) triggers -- `kind=pulser` in the diagnostics'
# step8_cosmic.csv, "dummy trigger, LAPPD ON/OFF" in stepG_charge_summary.csv. The
# readout fires on a schedule, not on anything in the detector, so each event is a
# random slice of the tank. 6264 is a genuine NO-SOURCE run: the real trigger logic
# is running, the source is simply not there. That distinction carries the physics:
# a dummy trigger measures the ACCIDENTAL cluster rate, while a no-source run with
# live trigger logic measures the FALSE-START rate of the IC trigger. Calling them
# both "no source" loses exactly the difference that makes them worth having.
DIAG_LABEL = {
    6254: "dummy trigger (LAPPD on)",
    6256: "dummy trigger (LAPPD off)",
    6264: "no source",
    6266: "regular run (Port 5, y = 100 cm)",
}
DIAG_LABEL[6265] = "source in (Port 5, y = 0 cm)"
DIAG_ROLE = DIAG_LABEL          # kept: the CSV writers and older prints use this name
# 6265 takes the green that used to mean 6266: green is "the source-in reference"
# in this deck, and it is the curve every other one is read against.
DIAG_COLOR = {6254: BLUE, 6256: GREY, 6264: ORANGE, 6265: GREEN, 6266: PURPLE}

# ── the two run sets the stage figures are built on ─────────────────────────
# THE POINT OF SPLITTING THEM. The four-run frames put 6254/6256 (dummy triggers) in
# the same figure as 6264/6266 (real trigger logic), and used 6266 -- Port 5 at
# y = 100 cm -- as the contrast for 6264 at y = 0. Two problems. The dummy triggers
# have ZERO IC-passing waveforms, so any figure that shows the AmBe waveform cut is
# blank for them and the cut cannot be part of a shared frame. And a y = 100 run is
# not the control for a y = 0 run: efficiency varies by ~10 pp over that span on the
# campaign's own map, so a difference between 6264 and 6266 is partly geometry.
#
# 6265 is the apple-to-apple partner. It sits at Port 5, (0, 0, 0) -- the SAME
# position as 6264, to the centimetre -- and is the run 6264 should be read against.
# It is labelled no-source in the run database and behaves source-in (tau = 35.5 us,
# 2,150 of 4,124 IC candidates passing); the label here says what it does, and
# REPORT_capturetime_maps_and_deck5.md carries the provenance.
PAIR_RUNS = [6264, 6265]                 # both Port 5, (0, 0, 0)
DUMMY_RUNS = [6254, 6256]                # scheduled readout, no trigger logic
PAIR_LABEL = {6264: "no source (Port 5, y = 0 cm)",
              6265: "source in (Port 5, y = 0 cm)"}
PAIR_POSITION = "Port 5, (x, y, z) = (0, 0, 0) cm — identical for both runs"

# Every run any deck-5 product needs loaded. Kept separate from DIAG_RUNS so adding
# 6265 for the pair figures does not put a fifth curve on the four-run overlays.
LOAD_RUNS = sorted(set(DIAG_RUNS) | set(PAIR_RUNS) | set(DUMMY_RUNS))


def plab(run, with_run=True):
    """Legend/title text for a run in the apple-to-apple pair figures."""
    lbl = PAIR_LABEL.get(run, DIAG_LABEL[run])
    return f"{lbl} — run {run}" if with_run else lbl


def dlab(run, with_run=False):
    """Legend/title text for a run. `with_run` appends the number, for panel titles."""
    return f"{DIAG_LABEL[run]} — run {run}" if with_run else DIAG_LABEL[run]

# The Stage-2 box, as (label, mask-builder) pairs so the cut flow, the N-1 table and
# the before/after 2D panels all apply literally the same four cuts in the same order
# and cannot drift apart. Mirrors ambe_single_cut in src/ambe/data/processor.py,
# including the INCLUSIVE PE bound (`pe_min < cpe <= pe_max`).
BOX_CUTS = [
    ("0 < PE ≤ 100", lambda d: (d.clusterPE > 0) & (d.clusterPE <= 100)),
    ("0 < CB < 0.45", lambda d: (d.clusterChargeBalance > 0)
                                & (d.clusterChargeBalance < 0.45)),
    ("t ≥ 2 µs", lambda d: d.clusterTime >= 2.0),
    ("hits ≥ 5", lambda d: d.clusterHits >= 5),
]


def box_mask(d):
    """All four Stage-2 cuts together, as one boolean mask."""
    m = np.ones(len(d), dtype=bool)
    for _, f in BOX_CUTS:
        m &= f(d).to_numpy()
    return m


def load_wf(run):
    """Per-waveform IC features for one run, from the pipeline's own parquet.

    TriggerSummary/WaveformFeatures_AmBe2.0v4_special_box_<run>.parquet is what
    `ambe data process` wrote for these runs, so `accepted` here IS the Stage-1
    decision the campaign made -- not a re-derivation of it. The diagnostics
    step4_joined_<run>.parquet carries the same columns but over a different part
    range (7,932 vs 7,933 rows for 6254, 4,624 vs 18,497 for 6264), so mixing the two
    would silently change the denominator. This section uses the pipeline file only.
    """
    p = HERE / "TriggerSummary" / f"WaveformFeatures_AmBe2.0v4_special_box_{run}.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = pd.read_parquet(p)
    d["run"] = run
    d["in_window"] = (d.IC_adjusted > IC_LO) & (d.IC_adjusted < IC_HI)
    d["accepted"] = d.accepted.astype(bool)
    d["has_second_pulse"] = d.second_pulse.astype(bool)
    print(f"  run {run} ({DIAG_ROLE[run]}): {len(d):,} waveforms, "
          f"{int(d.in_window.sum()):,} in the IC window, "
          f"{int(d.accepted.sum()):,} accepted")
    return d


def load_clusters_ungated(run):
    """Every tank cluster in one run, NO IC gate, de-duplicated by eventTimeTank.

    Returns (clusters, triggers). `clusters` is one row per cluster; `triggers` is one
    row per trigger, carrying the cluster multiplicity and TWO different charge
    estimators, because the "constant charge with no source" question turns out to
    have different answers in the two:

      trigger_pe  summed clusterPE over the clusters in the trigger -- charge that
                  the clustering actually reconstructed into something.
      hit_pe_sum  summed hitPE over ALL tank hits in the trigger, clustered or not.

    They are NOT interchangeable and must never be quoted as one number. On the
    dummy-trigger LAPPD pair they disagree in both size and significance (see
    fig_charge_constancy), which is the finding rather than a nuisance.

    TWO DE-DUPLICATIONS, BOTH NECESSARY AND DIFFERENT:

      * repeated eventTimeTank -- the same physical trigger written twice by a bad
        merge. Counting it twice inflates every yield. Same rule as process_run():
        the event key is eventTimeTank, NEVER eventNumber, which restarts per part
        file.
      * eventTimeTank == 0 -- not a duplicate trigger but a null timestamp, ~300 per
        run in all four of these files. They collapse onto one another under the rule
        above, so they would be counted once rather than dropped; they are counted and
        reported separately instead, because a null clock is a data-quality statement
        and not a trigger.
    """
    f = BC_DIR / f"BeamCluster_{run}.root"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    with uproot.open(f) as fh:
        t = fh["Event"]
        ev = {k: t[k].array(library="np") for k in
              ("eventTimeTank", "numberOfClusters", "clusterTime", "clusterPE",
               "clusterChargeBalance", "clusterHits", "nhits")}
        # hitPE is jagged and one entry per PMT hit, so on 6266 it is ~15M numbers.
        # Summed per event with awkward rather than converted to a per-event object
        # array, which is what would actually blow up.
        hit_pe = ak.to_numpy(ak.sum(t["hitPE"].array(library="ak"), axis=1))
    ett = ev["eventTimeTank"].astype("int64")

    n_null = int((ett == 0).sum())
    seen = set()
    keep = []
    n_dup = 0
    for i in range(len(ett)):
        if ett[i] == 0:
            continue
        if int(ett[i]) in seen:
            n_dup += 1
            continue
        seen.add(int(ett[i]))
        keep.append(i)

    crow, trow = [], []
    for i in keep:
        CT, CPE = ev["clusterTime"][i], ev["clusterPE"][i]
        CCB, CH = ev["clusterChargeBalance"][i], ev["clusterHits"][i]
        for k in range(len(CT)):
            crow.append((run, int(ett[i]), CT[k] / 1000.0, CPE[k], CCB[k],
                         int(CH[k])))
        trow.append((run, int(ett[i]), int(len(CT)), float(np.sum(CPE)),
                     int(ev["nhits"][i]), float(hit_pe[i])))

    cl = pd.DataFrame(crow, columns=["run", "eventTankTime", "clusterTime",
                                     "clusterPE", "clusterChargeBalance",
                                     "clusterHits"])
    tr = pd.DataFrame(trow, columns=["run", "eventTankTime", "n_clusters",
                                     "trigger_pe", "n_hits", "hit_pe_sum"])
    print(f"  run {run} ({DIAG_ROLE[run]}): {len(tr):,} triggers "
          f"({n_dup:,} duplicated, {n_null:,} null-timestamp dropped), "
          f"{len(cl):,} clusters, {len(cl)/max(len(tr),1):.2f} clusters/trigger")
    return cl, tr


# ---- Stage 1: the IC waveform stage ---------------------------------------------
IC_SPECS = [
    ("IC_adjusted", "IC_adjusted", (-500, 3000), 70),
    ("fprompt", "prompt fraction", (0, 1), 50),
    ("peak_amp", "peak amplitude [ADC]", (0, 800), 50),
    ("total_integral", "total integral [ADC·bin]", (-2000, 40000), 50),
    ("width_over_thresh", "width over threshold [bins]", (0, 1000), 50),
    ("baseline_sigma", "baseline σ [ADC]", (0, 60), 50),
]


def _overlay(ax, frames, col, rng, nb, density, logy=False):
    """One panel: every run overlaid, either area-normalised or raw counts.

    Factored out because every 1D figure in this section is emitted TWICE -- once
    normalised so the SHAPES can be compared across runs whose exposures differ by
    10x, once in counts so the size of each sample is visible. Neither answers the
    other's question: a normalised plot cannot show that 6254 contributes 4,404
    clusters against the regular run's 146,470, and a counts plot buries the two dummy
    triggers.
    """
    for run, v in frames:
        if not len(v):
            continue
        ax.hist(v, bins=nb, range=rng, histtype="step", lw=1.4, density=density,
                color=DIAG_COLOR[run],
                label=dlab(run) + ("" if density else f"  (n={len(v):,})"))
    ax.set_ylabel("normalised" if density else "entries")
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    bare(ax)


def fig_ic_1d(wf):
    """The six IC waveform features, all runs. Normalised page and counts page."""
    print("\n[ic-1d] IC waveform features per run")
    for density, tag, word in ((True, "ic_features_1d", "area-normalised"),
                               (False, "ic_features_1d_counts", "raw counts")):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
        for ax, (col, xlabel, rng, nb) in zip(axes.ravel(), IC_SPECS):
            frames = [(r, wf.loc[wf.run == r, col].dropna()) for r in DIAG_RUNS]
            _overlay(ax, frames, col, rng, nb, density, logy=not density)
            ax.set_xlabel(xlabel)
        fig.suptitle(
            f"IC waveform features for the special runs, ungated ({word})",
            fontsize=12)
        fig.tight_layout()
        save(fig, tag, prefix=DIAG_PREFIX)


def fig_ic_window(wf):
    """IC_adjusted on a log count axis: where the gate is, and who reaches it.

    Counts, not density, and log y on purpose. This is the figure that has to carry
    "0 of 7,933" and "0 of 25,525" as a visible statement rather than a footnote, and
    an area-normalised curve cannot say zero.
    """
    print("\n[ic-window] IC_adjusted, counts, log scale")
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for run in DIAG_RUNS:
        v = wf.loc[wf.run == run, "IC_adjusted"].dropna()
        ax.hist(v, bins=90, range=(-500, 4000), histtype="step", lw=1.5,
                color=DIAG_COLOR[run],
                label=f"{dlab(run)} — "
                      f"{int(wf.loc[wf.run == run, 'in_window'].sum()):,} in window")
    ax.set_yscale("log")
    ax.set_xlabel("IC_adjusted")
    ax.set_ylabel("waveforms")
    ax.legend(frameon=False, fontsize=8.5)
    bare(ax)
    ax.set_title(f"IC_adjusted for the special runs, ungated "
                 f"(gate {IC_LO:.0f}–{IC_HI:.0f})", fontsize=12)
    fig.tight_layout()
    save(fig, "ic_adjusted_window", prefix=DIAG_PREFIX)


def fig_ic_2d(wf):
    """IC_adjusted vs prompt fraction, one panel per run.

    NOT in the deck-5 running order -- it was dropped as not adding to M2/M3, which
    already carry both projections. Kept reachable as `--do ic2d` because the 2D
    plane is the honest picture of what the gate cuts in, and it is the figure to
    produce if anyone asks whether the gate could be moved.
    """
    print("\n[ic-2d] IC_adjusted vs prompt fraction per run")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for ax, run in zip(axes.ravel(), DIAG_RUNS):
        d = wf[(wf.run == run)].dropna(subset=["IC_adjusted", "fprompt"])
        h = ax.hist2d(d.IC_adjusted, d.fprompt, bins=(60, 50),
                      range=((-500, 3000), (0, 1)), cmap="viridis", cmin=1)
        fig.colorbar(h[3], ax=ax, label="waveforms")
        ax.set_xlabel("IC_adjusted")
        ax.set_ylabel("prompt fraction")
        ax.set_title(f"{dlab(run, with_run=True)}, "
                     f"{int(wf.loc[wf.run == run, 'in_window'].sum()):,} of "
                     f"{len(d):,} in the gate window", fontsize=10)
        bare(ax)
    fig.suptitle(f"Stage 1, {IC_GATE} — the 2D plane the gate is drawn in, ungated",
                 fontsize=11.5)
    fig.tight_layout()
    save(fig, "ic_adjusted_vs_fprompt_2d", prefix=DIAG_PREFIX)


def fig_ic_perrun(wf):
    """The basic IC page, ONE RUN PER PAGE — the special-run twin of the campaign's
    per-run IC waveform appendix.

    The overlay figures (M2/M3) answer "how do these runs differ from each other".
    They cannot answer "what does this run's IC look like", because four curves on
    one axis with exposures differing 10x hide everything but the loudest. This is
    the same content laid out the way the 28-run campaign book lays it out: one run,
    its own axes, its own counts, the gate window stated in the page title.

    Written as a multi-page PDF (the appendix) AND as one standalone figure per run,
    because a slide needs the single page and a QA reader needs the book.
    """
    print("\n[ic-perrun] one IC page per run")
    out = HERE / "PRESENTATION_PLOTS" / "APPENDIX_IC_waveforms_specialruns.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for run in DIAG_RUNS:
            d = wf[wf.run == run]
            n_in = int(d.in_window.sum())
            n_acc = int(d.accepted.sum())
            fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
            for ax, (col, xlabel, rng, nb) in zip(axes.ravel(), IC_SPECS):
                v = d[col].dropna()
                ax.hist(v, bins=nb, range=rng, histtype="step", lw=1.5,
                        color=DIAG_COLOR[run])
                ax.set_yscale("log")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("waveforms")
                ax.set_title(f"median {v.median():.4g}" if len(v) else "no entries",
                             fontsize=9)
                bare(ax)
            fig.suptitle(
                f"{dlab(run, with_run=True)} — IC waveform features, ungated\n"
                f"{n_in:,} of {len(d):,} in the gate window, {n_acc:,} accepted",
                fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            stem = OUTD / f"{DIAG_PREFIX}ic_page_{run}"
            fig.savefig(stem.with_suffix(".pdf"))
            fig.savefig(stem.with_suffix(".png"), dpi=200)
            plt.close(fig)
            print(f"  [page] run {run}: {len(d):,} waveforms, {n_in:,} in window")
    print(f"  wrote {out.name} ({len(DIAG_RUNS)} pages)")


def fig_ic_cutflow(wf):
    """Stage-1 cut flow per run: waveforms → in window → veto → accepted."""
    print("\n[ic-cutflow] Stage-1 waveform cut flow")
    cell = []
    for run in DIAG_RUNS:
        d = wf[wf.run == run]
        n0 = len(d)
        n1 = int(d.in_window.sum())
        n2 = int((d.in_window & ~d.has_second_pulse).sum())
        n3 = int(d.accepted.sum())
        cell.append([f"{run}", DIAG_LABEL[run], f"{n0:,d}",
                     f"{n1:,d} ({100*n1/n0:.2f}%)",
                     f"{n2:,d}", f"{n3:,d} ({100*n3/n0:.2f}%)"])
    fig, ax = plt.subplots(figsize=(12.0, 2.3))
    table_axes(ax, cell, ["run", "what it is", "waveforms",
                          f"in {IC_LO:.0f}–{IC_HI:.0f}", "and no second pulse",
                          "accepted (Stage 1)"],
               widths=[0.07, 0.26, 0.13, 0.19, 0.17, 0.18],
               emphasise_last=False, fs=9.5)
    ax.set_title("AmBe waveform cut flow for the special runs", fontsize=12)
    save(fig, "ic_cutflow_table", prefix=DIAG_PREFIX)


# ---- Stage 2: the tank cluster stage, ungated -----------------------------------
# (column, axis label, range BEFORE cuts, bins, range AFTER cuts, box-cut wording)
#
# The two ranges are the point. Before the box, the range has to run well past every
# cut bound or the figure cannot show what the box discards. After the box, keeping
# that range leaves the survivors crushed into a fraction of an otherwise empty axis
# -- PE survivors occupy 0-100 of a 0-300 axis, CB survivors 0-0.45 of 0-1. The after
# panels are therefore drawn on the range the cut actually admits.
TANK_SPECS = [
    ("clusterTime", r"cluster time [$\mu$s]", (0, 70), 70, (0, 70), "t ≥ 2 µs"),
    ("clusterPE", "cluster PE", (0, 300), 60, (0, 100), "0 < PE ≤ 100"),
    ("clusterChargeBalance", "cluster charge balance", (0, 1), 50, (0, 0.45),
     "0 < CB < 0.45"),
    ("clusterHits", "cluster hits", (0, 120), 60, (0, 60), "hits ≥ 5"),
]


def _upper(series_list, pct=99.9, floor=1.0):
    """Upper axis bound from the data: the widest 99.9th percentile, padded 10 %.

    Used only for the AFTER-the-box panels whose cut has no upper bound of its own
    (cluster hits, multiplicity, per-trigger charge). Those three would otherwise keep
    the before-cut range and leave the survivors in the leftmost tenth of the frame.
    The percentile rather than the max so one outlier trigger cannot re-stretch the
    axis back to where it started.
    """
    vals = [float(np.percentile(s, pct)) for s in series_list if len(s)]
    return max(floor, (max(vals) if vals else floor) * 1.1)


def fig_tank_1d(cl, tr, after):
    """Tank cluster distributions and trigger multiplicity, ungated, per run.

    `after` selects which sample and therefore which axis ranges: False draws every
    ungated cluster on ranges that run past the cut bounds, True draws only the
    clusters the box keeps, on the ranges the box admits. Four figures in total, since
    each is emitted normalised and in counts.
    """
    lbl = "after the box cuts" if after else "no IC gate and no box cuts"
    stem = "tank_features_1d_afterbox" if after else "tank_features_1d"
    print(f"\n[tank-1d] cluster distributions and multiplicity — {lbl}")

    if after:
        cl = cl[box_mask(cl)].copy()
        # Multiplicity and per-trigger charge have to be RECOMPUTED from the
        # surviving clusters, not carried over from tr: after the box, "clusters per
        # trigger" means accepted neutrons per trigger, which is a different number
        # and the one that matters. Triggers with nothing left are kept at zero so
        # the denominator stays the trigger count, not the survivor count.
        g = cl.groupby(["run", "eventTankTime"])
        acc = g.size().rename("n_clusters").reset_index()
        pe = g.clusterPE.sum().rename("trigger_pe").reset_index()
        tr = (tr[["run", "eventTankTime"]]
              .merge(acc, on=["run", "eventTankTime"], how="left")
              .merge(pe, on=["run", "eventTankTime"], how="left")
              .fillna({"n_clusters": 0, "trigger_pe": 0.0}))

    for density, word, suffix in ((True, "area-normalised", ""),
                                  (False, "raw counts", "_counts")):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
        axf = axes.ravel()
        for ax, (col, xlabel, rng, nb, rng_a, cutlbl) in zip(axf, TANK_SPECS):
            r = rng_a if after else rng
            _overlay(ax, [(run, cl.loc[cl.run == run, col]) for run in DIAG_RUNS],
                     col, r, nb, density)
            ax.set_xlabel(xlabel)
            ax.set_title(("kept: " if after else "box cut: ") + cutlbl, fontsize=10)

        ax = axf[4]
        # Log y, always. The no-source run really does put up to 112 box-accepted
        # clusters in ONE trigger (99.9th percentile 66) while the other three never
        # exceed 7-16, so the axis has to reach 66 and on a linear y everything except
        # the first two bins then sits on the floor. That tail is the finding, not a
        # nuisance, so it is shown rather than clipped.
        mult = [tr.loc[tr.run == run, "n_clusters"] for run in DIAG_RUNS]
        mx = (int(np.ceil(_upper(mult, floor=3))) if after
              else int(max(3, min(120, tr.n_clusters.max() + 1))))
        _overlay(ax, list(zip(DIAG_RUNS, mult)), "n_clusters", (0, mx), mx, density,
                 logy=True)
        ax.set_xlabel("accepted clusters per trigger" if after
                      else "clusters per trigger")
        ax.set_title("cluster multiplicity (log y — note the no-source tail)",
                     fontsize=10)

        ax = axf[5]
        # Log x: before the box the per-trigger charge means span 279 PE (the null
        # triggers) to 6,006 PE (the no-source run), a factor of 21, and any linear range
        # wide enough for one end buries the other. The upper edge is taken from the
        # data so the after-the-box panel does not keep four decades of empty axis
        # that only the before-the-box sample ever filled. Zeros are dropped for the
        # log axis and their share is stated in the legend instead.
        charge = [tr.loc[tr.run == run, "trigger_pe"] for run in DIAG_RUNS]
        lbins = np.logspace(0, np.log10(_upper(charge, floor=100.0)), 60)
        for run in DIAG_RUNS:
            v = tr.loc[tr.run == run, "trigger_pe"]
            pos = v[v > 0]
            ax.hist(pos, bins=lbins, histtype="step", lw=1.4, density=density,
                    color=DIAG_COLOR[run],
                    label=f"{dlab(run)} (mean {v.mean():.0f} PE, "
                          f"{100*(v <= 0).mean():.0f}% at zero)")
        ax.set_xscale("log")
        ax.set_xlabel("summed cluster PE per trigger")
        ax.set_ylabel("normalised" if density else "entries")
        ax.set_title("total charge per trigger (zeros excluded, log axis)",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
        bare(ax)

        fig.suptitle(f"Tank cluster variables for the special runs, {lbl} ({word})",
                     fontsize=12)
        fig.tight_layout()
        save(fig, stem + suffix, prefix=DIAG_PREFIX)


def _panel_2d(fig, ax, d, xa, xr, ya, yr, title):
    if len(d):
        h = ax.hist2d(d[xa], d[ya], bins=(60, 50), range=(xr, yr),
                      cmap="viridis", cmin=1)
        fig.colorbar(h[3], ax=ax, label="clusters")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_title(title, fontsize=9.5)
    bare(ax)


def fig_pe_vs_cb(cl):
    """PE vs charge balance, per run, BEFORE and AFTER the box. The headline ask.

    Two rows of four panels: top row every ungated cluster on a range that runs past
    both cut bounds, bottom row the survivors on the range the box admits (PE 0-100,
    CB 0-0.45). The after row is re-ranged rather than left on the before row's axes:
    on the wide axes the survivors occupy one ninth of the frame and the other eight
    ninths are empty, which reads as "almost nothing survived" rather than showing the
    shape of what did. The kept fraction is on every after-panel title instead, which
    is the quantity the shared axes were there to convey.
    """
    print("\n[pe-vs-cb] PE vs charge balance, before and after the box")
    fig, axes = plt.subplots(2, len(DIAG_RUNS), figsize=(4.2 * len(DIAG_RUNS), 8.4))
    for j, run in enumerate(DIAG_RUNS):
        d = cl[cl.run == run]
        keep = d[box_mask(d)]
        frac = len(keep) / max(len(d), 1)
        _panel_2d(fig, axes[0, j], d, "clusterPE", (0, 300),
                  "clusterChargeBalance", (0, 1),
                  f"{dlab(run, with_run=True)}\nbefore cuts, {len(d):,} clusters")
        _panel_2d(fig, axes[1, j], keep, "clusterPE", (0, 100),
                  "clusterChargeBalance", (0, 0.45),
                  f"after cuts, {len(keep):,} clusters ({100*frac:.1f} % kept)")
        for i in (0, 1):
            axes[i, j].set_xlabel("cluster PE")
            axes[i, j].set_ylabel("cluster charge balance")
    fig.suptitle(f"Charge balance vs cluster PE, ungated, per run — top row before "
                 f"the box, bottom row after {BOXCUT}\n"
                 "the after row is drawn on the range the box admits, so the "
                 "survivors fill the frame; the kept fraction is on each panel",
                 fontsize=11.5)
    fig.tight_layout()
    save(fig, "pe_vs_cb_before_after", prefix=DIAG_PREFIX)


def attach_gate(cl, tr, wf):
    """Add the per-cluster / per-trigger Stage-1 decision, joined on the tank clock.

    The AmBe waveform cut is a decision about a TRIGGER; every cluster in that trigger
    inherits it. The join key is the pipeline parquet's `timestamp` against the
    ntuple's `eventTimeTank`: verified 100% of the box-passing candidates in
    boxcut_v4_special_candidates.csv for 6265 and 6266 are present in their run's
    waveform parquet under that key, so this is the same clock and not a coincidence
    of magnitude.

    `accepted` in the pipeline parquet is exactly (in_window & ~second_pulse) -- the
    IC gate as ambe.stats.waveform applies it -- checked for all five runs. It is NOT
    the diagnostics' step4_joined `passes_ic`: that file covers a tenth of the part
    range (4,124 of 41,249 waveforms for 6265), so its absolute counts are smaller
    while its ACCEPTANCE FRACTION agrees to a percent (52.1% vs 50.8%). Deck 5 uses
    the pipeline file throughout, for the reason load_wf() gives.
    """
    ok = {r: set(g.loc[g.accepted, "timestamp"].astype("int64"))
          for r, g in wf.groupby("run")}

    def flag(d):
        return np.fromiter(
            (int(t) in ok.get(int(r), ()) for r, t in
             zip(d["run"].to_numpy(), d["eventTankTime"].to_numpy())),
            dtype=bool, count=len(d))

    cl = cl.assign(ic_accepted=flag(cl))
    tr = tr.assign(ic_accepted=flag(tr))
    for r in sorted(cl.run.unique()):
        c, t = cl[cl.run == r], tr[tr.run == r]
        print(f"  run {r} ({DIAG_ROLE[r]}): {int(t.ic_accepted.sum()):,}/{len(t):,} "
              f"triggers and {int(c.ic_accepted.sum()):,}/{len(c):,} clusters pass "
              f"the AmBe waveform cut")
    return cl, tr


# The three streamline stages, as (row label, mask-builder) pairs, so the 2D panels
# and the cut-flow table cannot describe different selections. `after` says whether
# the panel is drawn on the range the box admits.
def _stage_before(d):
    return np.ones(len(d), dtype=bool)


def _stage_gated(d):
    return d.ic_accepted.to_numpy()


def _stage_box(d):
    return d.ic_accepted.to_numpy() & box_mask(d)


# Stage labels are SHORT. They are column headers in a four-column table and panel
# titles two to a row, and the cut definitions are long enough that spelling them out
# here made adjacent labels overlap and become unreadable. The definitions are stated
# once in the figure title instead, which is where IC_GATE and BOXCUT exist for.
PAIR_STAGES = [
    ("before AmBe waveform cut", _stage_before, False),
    ("after AmBe waveform cut", _stage_gated, False),
    ("after box cut", _stage_box, True),
]
PAIR_STAGE_NOTE = f"AmBe waveform cut = {IC_GATE}.  Box cut = {BOXCUT}"

# The dummy triggers have NO waveform-cut row. It is not omitted for space: a
# scheduled readout carries no BGO gamma, 0 of 7,933 and 0 of 25,525 waveforms land
# in the IC window, so the stage is empty by construction and three blank panels
# would read as a processing failure rather than as the result. The box here is
# therefore applied to ungated clusters, which is the only sample those runs have.
DUMMY_STAGES = [
    ("before box cut", lambda d: np.ones(len(d), dtype=bool), False),
    ("after box cut", lambda d: box_mask(d), True),
]


# (x column, x label, x range before, x range after,
#  y column, y label, y range before, y range after)
PAIR_SPECS = [
    ("clusterPE", "cluster PE", (0, 300), (0, 100),
     "clusterTime", r"cluster time [$\mu$s]", (0, 70), (0, 70)),
    ("clusterPE", "cluster PE", (0, 300), (0, 100),
     "clusterHits", "cluster hits", (0, 120), (0, 120)),
    ("clusterChargeBalance", "cluster charge balance", (0, 1), (0, 0.45),
     "clusterTime", r"cluster time [$\mu$s]", (0, 70), (0, 70)),
]

# All four cut-variable planes in one list. PAIR_SPECS is the older three; the PE vs
# charge-balance plane was hard-coded in fig_pe_vs_cb and is folded in here so the
# stage figures below cover the whole box in one loop and cannot miss a plane.
ALL_PLANES = [
    ("clusterPE", "cluster PE", (0, 300), (0, 100),
     "clusterChargeBalance", "cluster charge balance", (0, 1), (0, 0.45)),
] + PAIR_SPECS


def _stage_grid(cl, runs, stages, labeller, name_fmt, suptitle_fmt, note):
    """One figure per cut-variable plane: rows = streamline stage, cols = run.

    LANDSCAPE: ROWS ARE RUNS, COLUMNS ARE STAGES. This was rows = stage, columns =
    run, which for the three-stage pair figures is a 3 x 2 portrait page -- taller
    than it is wide, so it has to be shrunk to fit a slide and the panels come out
    small. 2 x 3 fills the slide instead.

    Reading order is unchanged in meaning, only in direction. ACROSS a row is the
    selection tightening on one run; DOWN a column is the comparison at a fixed
    stage. That only means anything if the two rows are the same kind of measurement,
    which is why the run sets are split.

    Each stage keeps its own count and its kept fraction OF THE STAGE TO ITS LEFT,
    not of the ungated total: on a three-stage chain a fraction of the original is the
    less useful of the two, because it hides which stage did the cutting.
    """
    for xa, xl, xr, xr_a, ya, yl, yr, yr_a in ALL_PLANES:
        fig, axes = plt.subplots(len(runs), len(stages),
                                 figsize=(4.7 * len(stages), 4.5 * len(runs)),
                                 squeeze=False)
        for i, run in enumerate(runs):
            d = cl[cl.run == run]
            prev = len(d)
            for j, (slabel, mask_of, after) in enumerate(stages):
                sel = d[mask_of(d)]
                rx, ry = (xr_a, yr_a) if after else (xr, yr)
                # Run label on the leftmost panel of its row; the stage label on every
                # panel, because the stages no longer share a column and a reader
                # should not have to count across to know which one they are looking
                # at. Both are short enough now not to need wrapping.
                head = labeller(run) + "\n" if j == 0 else ""
                frac = ("" if j == 0 else
                        f"\n({100 * len(sel) / max(prev, 1):.1f} % of the panel "
                        f"to its left)")
                title = head + slabel + f"\n{len(sel):,} clusters{frac}"
                ax = axes[i, j]
                _panel_2d(fig, ax, sel, xa, rx, ya, ry, title)
                if not len(sel):
                    # An empty panel with nothing written in it reads as a bug. Say
                    # that zero IS the measurement.
                    ax.text(0.5, 0.5, "0 clusters", transform=ax.transAxes,
                            ha="center", va="center", fontsize=13, color="#b22222")
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                prev = len(sel)
        fig.suptitle(suptitle_fmt.format(yl=yl, xl=xl) + "\n" + note, fontsize=11.5)
        fig.tight_layout()
        save(fig, name_fmt.format(y=ya, x=xa).lower(), prefix=DIAG_PREFIX)


def fig_pair_stages_2d(cl):
    """6264 vs 6265 through the full streamline. Same port, same y, source out/in."""
    print("\n[pair-2d] no source vs source in at Port 5 y = 0, stage by stage")
    _stage_grid(
        cl, PAIR_RUNS, PAIR_STAGES, plab,
        name_fmt="pair_stages_{y}_vs_{x}",
        suptitle_fmt="{yl} vs {xl} — no source vs source in, the same position",
        note=PAIR_POSITION)


def fig_dummy_stages_2d(cl):
    """The two dummy triggers on their own, box only. No waveform-cut row exists."""
    print("\n[dummy-2d] dummy triggers, ungated, before and after the box")
    _stage_grid(
        cl, DUMMY_RUNS, DUMMY_STAGES, lambda r: dlab(r, with_run=True),
        name_fmt="dummy_stages_{y}_vs_{x}",
        suptitle_fmt="{yl} vs {xl} — the two dummy triggers, LAPPD on and off",
        note="Ungated: neither run has an IC-passing waveform, so there is no "
             "AmBe waveform cut stage")


def fig_stage_cutflow(cl, tr, runs, stages, labeller, name, csv, title):
    """The same streamline as a table, so the panel counts are checkable.

    Written next to the 2D grids rather than folded into the existing
    tank_cutflow_table: that table is a four-run box-only flow on ungated clusters
    and stays as it is, while this one has a stage axis the other does not have.
    """
    rows, cell = [], []
    for run in runs:
        d = cl[cl.run == run]
        t = tr[tr.run == run]
        prev, cols = len(d), []
        for slabel, mask_of, _ in stages:
            n = int(mask_of(d).sum())
            cols.append(f"{n:,d} ({100 * n / max(prev, 1):.1f} %)")
            rows.append(dict(run=run, role=labeller(run), stage=slabel,
                             triggers=len(t),
                             triggers_gated=int(t.ic_accepted.sum())
                             if "ic_accepted" in t else 0,
                             clusters=n,
                             pct_of_previous_stage=100 * n / max(prev, 1)))
            prev = n
        cell.append([f"{run}", labeller(run).split(" — ")[0], f"{len(t):,d}"] + cols)
    ncol = 3 + len(stages)
    fig, ax = plt.subplots(figsize=(2.2 * ncol, 1.4 + 0.5 * len(runs)))
    table_axes(ax, cell,
               ["run", "what it is", "triggers"] + [s for s, _, _ in stages],
               widths=[0.09, 0.25] + [(0.66 / (ncol - 2))] * (ncol - 2),
               emphasise_last=False, fs=9.0)
    ax.set_title(title, fontsize=11)
    save(fig, name, prefix=DIAG_PREFIX)
    pd.DataFrame(rows).to_csv(HERE / csv, index=False)
    print(f"  wrote {csv}")


def fig_pairs_2d(cl):
    """The other three cut-variable planes, before and after the box, per run.

    One figure per plane rather than one giant grid: three planes x four runs x
    before/after is 24 panels, which is unreadable on a slide and unavoidable in a
    single figure. Same re-ranging rule as fig_pe_vs_cb.
    """
    print("\n[pairs-2d] the other cut-variable planes, before and after the box")
    for xa, xl, xr, xr_a, ya, yl, yr, yr_a in PAIR_SPECS:
        fig, axes = plt.subplots(2, len(DIAG_RUNS),
                                 figsize=(4.2 * len(DIAG_RUNS), 8.4))
        for j, run in enumerate(DIAG_RUNS):
            d = cl[cl.run == run]
            keep = d[box_mask(d)]
            frac = len(keep) / max(len(d), 1)
            _panel_2d(fig, axes[0, j], d, xa, xr, ya, yr,
                      f"{dlab(run, with_run=True)}\nbefore cuts, "
                      f"{len(d):,} clusters")
            _panel_2d(fig, axes[1, j], keep, xa, xr_a, ya, yr_a,
                      f"after cuts, {len(keep):,} clusters ({100*frac:.1f} % kept)")
            for i in (0, 1):
                axes[i, j].set_xlabel(xl)
                axes[i, j].set_ylabel(yl)
        fig.suptitle(f"{yl} vs {xl}, ungated, per run — top row before the box, "
                     f"bottom row after {BOXCUT}\n"
                     "the after row is drawn on the range the box admits",
                     fontsize=11.5)
        fig.tight_layout()
        save(fig, f"{ya}_vs_{xa}_before_after".lower(), prefix=DIAG_PREFIX)


def fig_tank_cutflow(cl, tr):
    """Stage-2 cut flow on ungated clusters, cumulative and leave-one-out, per run.

    Both columns are given because they answer different questions and are routinely
    confused. Cumulative says how many clusters survive the whole box. Leave-one-out
    (N-1) says how many survive if THAT ONE cut is dropped, which is what identifies
    which cut is doing the work -- for a run whose clusters are all cosmic-like, one
    cut removes nearly everything and the other three are then nearly free.
    """
    print("\n[tank-cutflow] Stage-2 cut flow on ungated clusters")
    rows = []
    for run in DIAG_RUNS:
        d = cl[cl.run == run]
        n0 = len(d)
        cum = np.ones(n0, dtype=bool)
        per = []
        for lbl, f in BOX_CUTS:
            m = f(d).to_numpy()
            cum = cum & m
            # N-1: every cut EXCEPT this one
            other = np.ones(n0, dtype=bool)
            for lbl2, f2 in BOX_CUTS:
                if lbl2 != lbl:
                    other &= f2(d).to_numpy()
            per.append((lbl, int(m.sum()), int(cum.sum()), int(other.sum())))
        rows.append((run, n0, len(tr[tr.run == run]), per, int(cum.sum())))

    cell = []
    for run, n0, ntrig, per, nfinal in rows:
        cell.append([f"{run}", DIAG_LABEL[run], f"{ntrig:,d}", f"{n0:,d}",
                     " → ".join(f"{c:,d}" for _, _, c, _ in per),
                     f"{nfinal:,d} ({100*nfinal/max(n0,1):.2f}%)"])
    fig, ax = plt.subplots(figsize=(13.0, 2.3))
    table_axes(ax, cell, ["run", "what it is", "triggers", "clusters (ungated)",
                          "cumulative: PE → CB → t → hits", "survive the box"],
               widths=[0.07, 0.24, 0.10, 0.14, 0.27, 0.18],
               emphasise_last=False, fs=9.0)
    ax.set_title("Box cut flow on ungated clusters, cumulative", fontsize=12)
    save(fig, "tank_cutflow_table", prefix=DIAG_PREFIX)

    cell = []
    for run, n0, _, per, nfinal in rows:
        cell.append([DIAG_LABEL[run]] + [f"{nm1:,d}" for _, _, _, nm1 in per]
                    + [f"{nfinal:,d}"])
    fig, ax = plt.subplots(figsize=(11.6, 2.3))
    table_axes(ax, cell, ["what it is"] + [f"drop {lbl}" for lbl, _ in BOX_CUTS]
               + ["all four"],
               widths=[0.26, 0.16, 0.16, 0.14, 0.14, 0.14],
               emphasise_last=False, fs=9.5)
    ax.set_title("Box cut, leave one out, on ungated clusters", fontsize=12)
    save(fig, "tank_nminus1_table", prefix=DIAG_PREFIX)

    out = []
    for run, n0, ntrig, per, nfinal in rows:
        for lbl, alone, cum, nm1 in per:
            out.append(dict(run=run, role=DIAG_LABEL[run], triggers=ntrig,
                            clusters_ungated=n0, cut=lbl, pass_alone=alone,
                            pass_cumulative=cum, pass_without_this_cut=nm1,
                            pass_all_four=nfinal))
    pd.DataFrame(out).to_csv(HERE / "boxcut_v4_specialdiag_cutflow.csv", index=False)
    print("  wrote boxcut_v4_specialdiag_cutflow.csv")


def fig_charge_constancy(cl, tr):
    """The 6254-vs-6256 question: what charge does the tank register with nothing to
    detect?

    This is the one figure in the section that is a comparison of two runs rather than
    a description of four, because the two DUMMY TRIGGERS are a matched LAPPD ON/OFF
    pair with no source in either and a readout that fires on a schedule -- so every
    event is a random slice of the tank, any difference between the two runs is
    instrumental, and any charge either registers is not neutron light.

    THE RESULT, AND WHY BOTH ESTIMATORS ARE ON THE FIGURE. The charge the tank
    registers with nothing to detect is NOT one number, because the two estimators
    disagree:

      hit-level  summed hitPE over all tank hits -- LAPPD ON runs far higher, and
                 the excess is the LAPPD light leak. Reproduces the diagnostics'
                 own Step G `tot_mean` (642.40 vs 421.75).
      cluster-level  summed clusterPE over reconstructed clusters -- the two runs
                 agree. Reproduces Step G's `clus_mean` (279.21 vs 285.32).

    So the leak is charge the clustering does not pick up: it raises the hit stream
    without producing clusters. That is the statement this figure is for, and it is
    the reason mean tank PE cannot be used as a stand-in for cluster PE anywhere in
    the analysis. Both differences are computed and printed rather than one being
    chosen.
    """
    print("\n[constancy] dummy-trigger charge, LAPPD on vs LAPPD off")
    pair = [6254, 6256]

    def diff(col):
        a = tr.loc[tr.run == 6254, col]
        b = tr.loc[tr.run == 6256, col]
        d_ = a.mean() - b.mean()
        e_ = float(np.hypot(a.sem(), b.sem()))
        return a.mean(), b.mean(), d_, e_, (abs(d_ / e_) if e_ else float("nan"))

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.4))
    specs = [("hit_pe_sum", "summed hit PE per trigger (all tank hits)", (0, 4000),
              "{:.1f} PE"),
             ("trigger_pe", "summed cluster PE per trigger", (0, 2000), "{:.1f} PE"),
             ("n_hits", "tank hits per trigger", (0, 600), "{:.1f} hits"),
             ("n_clusters", "clusters per trigger", (0, 11), "{:.3f}")]
    for ax, (col, xlabel, rng, fmt) in zip(axes, specs):
        nb = 11 if col == "n_clusters" else 60
        for run in pair:
            v = tr.loc[tr.run == run, col]
            extra = (f", {100*(v == 0).mean():.1f}% empty"
                     if col == "n_clusters" else "")
            ax.hist(v, bins=nb, range=rng, histtype="step", lw=1.5, density=True,
                    color=DIAG_COLOR[run],
                    label=f"{dlab(run)} (mean {fmt.format(v.mean())}{extra})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalised")
        ax.legend(frameon=False, fontsize=8)
        bare(ax)

    hp, cp, nh = diff("hit_pe_sum"), diff("trigger_pe"), diff("n_hits")
    for name, r_ in (("summed hit PE", hp), ("summed cluster PE", cp),
                     ("tank hits", nh)):
        print(f"  {name:20s} per trigger: LAPPD on {r_[0]:8.2f}, off {r_[1]:8.2f}, "
              f"difference {r_[2]:+8.2f} ± {r_[3]:.2f} ({r_[4]:.2f}σ)")
    fig.suptitle(
        f"Charge with nothing to detect — dummy trigger LAPPD on vs off, ungated\n"
        f"hits {nh[2]:+.0f} ± {nh[3]:.0f} ({nh[4]:.1f}σ), hit PE "
        f"{hp[2]:+.0f} ± {hp[3]:.0f} ({hp[4]:.1f}σ), cluster PE "
        f"{cp[2]:+.1f} ± {cp[3]:.1f} ({cp[4]:.2f}σ)",
        fontsize=12)
    fig.tight_layout()
    save(fig, "nosource_charge_6254_vs_6256", prefix=DIAG_PREFIX)

    pd.DataFrame([
        dict(run=r, role=DIAG_LABEL[r],
             triggers=int((tr.run == r).sum()),
             clusters=int((cl.run == r).sum()),
             clusters_per_trigger=float(tr.loc[tr.run == r, "n_clusters"].mean()),
             empty_trigger_frac=float((tr.loc[tr.run == r, "n_clusters"] == 0).mean()),
             mean_trigger_pe=float(tr.loc[tr.run == r, "trigger_pe"].mean()),
             sem_trigger_pe=float(tr.loc[tr.run == r, "trigger_pe"].sem()),
             mean_hit_pe_sum=float(tr.loc[tr.run == r, "hit_pe_sum"].mean()),
             sem_hit_pe_sum=float(tr.loc[tr.run == r, "hit_pe_sum"].sem()),
             mean_n_hits=float(tr.loc[tr.run == r, "n_hits"].mean()),
             mean_cluster_pe=float(cl.loc[cl.run == r, "clusterPE"].mean()),
             mean_cluster_cb=float(cl.loc[cl.run == r,
                                          "clusterChargeBalance"].mean()))
        for r in DIAG_RUNS
    ]).to_csv(HERE / "boxcut_v4_specialdiag_summary.csv", index=False)
    print("  wrote boxcut_v4_specialdiag_summary.csv")


def fig_appendix_book(wf, cl, tr):
    """One page per special run, everything about that run on it — the deck-5 appendix.

    Replaces APPENDIX_perposition_specialruns_boxcuts.pdf (per SOURCE POSITION, which
    is close to meaningless for a run with no source in the tank) and
    APPENDIX_IC_waveforms_all28runs.pdf (the 28 SOURCE runs, nothing to do with these).
    Both were inherited from deck 2 and neither described the special runs.
    """
    print("\n[appendix] one page per special run")
    out = HERE / "PRESENTATION_PLOTS" / "APPENDIX_specialruns_by_run.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for run in DIAG_RUNS:
            w = wf[wf.run == run]
            d = cl[cl.run == run]
            t = tr[tr.run == run]
            keep = d[box_mask(d)]
            col = DIAG_COLOR[run]

            fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.6))

            ax = axes[0, 0]
            ax.hist(w.IC_adjusted.dropna(), bins=80, range=(-500, 4000),
                    histtype="step", lw=1.5, color=col)
            ax.set_yscale("log")
            ax.set_xlabel("IC_adjusted")
            ax.set_ylabel("waveforms")
            ax.set_title(f"{int(w.in_window.sum()):,} in {IC_LO:.0f}–{IC_HI:.0f}",
                         fontsize=9.5)
            bare(ax)

            for ax, (c, xl, rng, nb, _ra, _cl) in zip(axes[0, 1:], TANK_SPECS[:3]):
                ax.hist(d[c], bins=nb, range=rng, histtype="step", lw=1.5, color=col)
                ax.set_yscale("log")
                ax.set_xlabel(xl)
                ax.set_ylabel("clusters")
                ax.set_title("ungated, before the box", fontsize=9.5)
                bare(ax)

            _panel_2d(fig, axes[1, 0], d, "clusterPE", (0, 300),
                      "clusterChargeBalance", (0, 1), "before the box")
            axes[1, 0].set_xlabel("cluster PE")
            axes[1, 0].set_ylabel("cluster charge balance")
            _panel_2d(fig, axes[1, 1], keep, "clusterPE", (0, 100),
                      "clusterChargeBalance", (0, 0.45),
                      f"after the box, {100*len(keep)/max(len(d),1):.1f} % kept")
            axes[1, 1].set_xlabel("cluster PE")
            axes[1, 1].set_ylabel("cluster charge balance")

            ax = axes[1, 2]
            mx = int(max(3, min(31, t.n_clusters.max() + 1)))
            ax.hist(t.n_clusters, bins=mx, range=(0, mx), histtype="step", lw=1.5,
                    color=col)
            ax.set_yscale("log")
            ax.set_xlabel("clusters per trigger")
            ax.set_ylabel("triggers")
            ax.set_title(f"mean {t.n_clusters.mean():.2f}", fontsize=9.5)
            bare(ax)

            ax = axes[1, 3]
            ax.axis("off")
            n0 = len(d)
            txt = (f"run                  {run}\n"
                   f"what it is           {DIAG_LABEL[run]}\n\n"
                   f"waveforms            {len(w):,}\n"
                   f"  in {IC_LO:.0f}-{IC_HI:.0f}         {int(w.in_window.sum()):,}\n"
                   f"  accepted (Stage 1) {int(w.accepted.sum()):,}\n\n"
                   f"triggers             {len(t):,}\n"
                   f"clusters (ungated)   {n0:,}\n"
                   f"clusters / trigger   {t.n_clusters.mean():.2f}\n"
                   f"empty triggers       {100*(t.n_clusters == 0).mean():.1f} %\n\n"
                   f"survive the box      {len(keep):,} "
                   f"({100*len(keep)/max(n0,1):.1f} %)\n\n"
                   f"median PE            {d.clusterPE.median():.1f}\n"
                   f"median CB            {d.clusterChargeBalance.median():.3f}\n"
                   f"median hits          {d.clusterHits.median():.0f}\n"
                   f"mean hit PE / trig   {t.hit_pe_sum.mean():.1f}\n"
                   f"mean cluster PE/trig {t.trigger_pe.mean():.1f}\n")
            ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=10.5,
                    family="monospace")

            fig.suptitle(f"{dlab(run, with_run=True)} — Stage 1 and Stage 2, "
                         f"ungated, against {BOXCUT}", fontsize=13)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  [page] {dlab(run, with_run=True)}")
    print(f"  wrote {out.name} ({len(DIAG_RUNS)} pages)")


DIAG_PRODUCTS = ["ic1d", "icwindow", "icperrun", "ic2d", "iccutflow",
                 "tank1d", "pecb", "pairs", "pair2d", "dummy2d", "stagecutflow",
                 "tankcutflow", "constancy", "appendix"]

# Reachable by name but NOT part of `--do diag`:
#   ic2d    dropped from the running order as not adding to icwindow/ic1d, which
#           already carry both of its projections.
#   pecb    the four-run mixed 2D frames, superseded by pair2d + dummy2d. They put
#   pairs   dummy triggers and source runs in one frame and contrasted 6264 (y = 0)
#           with 6266 (y = 100), so half the difference was geometry. Kept buildable
#           because they are what the earlier version of the deck showed.
DIAG_NOT_IN_DECK = {"ic2d", "pecb", "pairs"}
DIAG_IN_DECK = [p for p in DIAG_PRODUCTS if p not in DIAG_NOT_IN_DECK]

# Which runs each product needs loaded, so asking for the dummy figures does not read
# 6266's 15M-hit ntuple. Anything unlisted gets DIAG_RUNS.
PRODUCT_RUNS = {"pair2d": PAIR_RUNS, "dummy2d": DUMMY_RUNS,
                "stagecutflow": PAIR_RUNS + DUMMY_RUNS}


def do_diagnostic(want):
    """Deck 5: the special runs described stage by stage rather than scored."""
    if want == {"all"}:
        want = set(DIAG_IN_DECK)
    # The waveform parquet is cheap and every cluster product now needs it too, for
    # the per-cluster Stage-1 flag attach_gate() joins on.
    need_wf = bool(want & {"ic1d", "icwindow", "icperrun", "ic2d", "iccutflow",
                           "appendix", "pair2d", "dummy2d", "stagecutflow"})
    need_cl = bool(want & {"tank1d", "pecb", "pairs", "tankcutflow", "constancy",
                           "appendix", "pair2d", "dummy2d", "stagecutflow"})

    runs = sorted({r for p in want for r in PRODUCT_RUNS.get(p, DIAG_RUNS)})
    print(f"\n[runs] products {sorted(want)} need runs {runs}")

    wf = None
    if need_wf:
        print("\n[load] Stage-1 waveform features (pipeline parquet)")
        wf = pd.concat([load_wf(r) for r in runs], ignore_index=True)

    cl = tr = None
    if need_cl:
        print("\n[load] ungated tank clusters (BeamCluster ROOT)")
        frames = [load_clusters_ungated(r) for r in runs]
        cl = pd.concat([f[0] for f in frames], ignore_index=True)
        tr = pd.concat([f[1] for f in frames], ignore_index=True)
        print("\n[gate] joining the Stage-1 decision onto clusters and triggers")
        cl, tr = attach_gate(cl, tr, wf)

    if "ic1d" in want:
        fig_ic_1d(wf)
    if "icwindow" in want:
        fig_ic_window(wf)
    if "icperrun" in want:
        fig_ic_perrun(wf)
    if "ic2d" in want:
        fig_ic_2d(wf)
    if "iccutflow" in want:
        fig_ic_cutflow(wf)
    if "tank1d" in want:
        # Both samples: ungated on wide axes, then the box survivors on the ranges
        # the box admits. Four figures, two per sample (normalised and counts).
        fig_tank_1d(cl, tr, after=False)
        fig_tank_1d(cl, tr, after=True)
    if "pecb" in want:
        fig_pe_vs_cb(cl)
    if "pairs" in want:
        fig_pairs_2d(cl)
    if "pair2d" in want:
        fig_pair_stages_2d(cl)
    if "dummy2d" in want:
        fig_dummy_stages_2d(cl)
    if "stagecutflow" in want:
        fig_stage_cutflow(
            cl, tr, PAIR_RUNS, PAIR_STAGES, plab,
            "pair_stage_cutflow_table", "boxcut_v4_specialdiag_pair_cutflow.csv",
            f"No source vs source in — {PAIR_POSITION}\n"
            f"clusters surviving each stage")
        fig_stage_cutflow(
            cl, tr, DUMMY_RUNS, DUMMY_STAGES, lambda r: dlab(r, with_run=True),
            "dummy_stage_cutflow_table",
            "boxcut_v4_specialdiag_dummy_cutflow.csv",
            "The two dummy triggers, ungated — clusters surviving each stage")
    if "tankcutflow" in want:
        fig_tank_cutflow(cl, tr)
    if "constancy" in want:
        fig_charge_constancy(cl, tr)
    if "appendix" in want:
        fig_appendix_book(wf, cl, tr)


def main():
    ap = argparse.ArgumentParser(prog="boxcut_v4_special")
    # No default, same reason as the campaign script.
    ap.add_argument("--do", required=True,
                    choices=["all", "table", "yield", "cosmic", "shapes"]
                            + ["diag"] + DIAG_PRODUCTS,
                    help="which product to build. 'all' is the four box-cut special "
                         "products only; 'diag' is the whole deck-5 stage-by-stage "
                         "section, which reads different inputs and is therefore kept "
                         "out of 'all' (same rule as boxcut_v4_campaign's "
                         "residuals/captureheat).")
    args = ap.parse_args()
    want = {"all"} if args.do == "all" else {args.do}

    # Deck 5 first, and returning: it needs neither the Stage-1 gate join nor the
    # process_run loop below, and check_pulsers() would assert on inputs it does not
    # use.
    if args.do == "diag":
        do_diagnostic({"all"})
        print("\ndone")
        return 0
    if args.do in DIAG_PRODUCTS:
        do_diagnostic({args.do})
        print("\ndone")
        return 0

    proc = AmBeNeutronProcessing(cuts=CutCriteria())
    pulsers = check_pulsers()

    summaries, cand_frames = [], []
    for run in SPECIAL_RUNS:
        print(f"\n[run {run}] {SPECIAL_NOTE[run]}")
        s, c = process_run(run, proc)
        summaries.append(s)
        if len(c):
            cand_frames.append(c)
    summ = pd.concat([pd.DataFrame(summaries), pulsers], ignore_index=True)
    summ = summ.sort_values("run").reset_index(drop=True)
    cands = (pd.concat(cand_frames, ignore_index=True) if cand_frames
             else pd.DataFrame(columns=["run", "clusterTime", "clusterPE",
                                        "clusterChargeBalance", "clusterHits"]))

    summ.to_csv(HERE / "boxcut_v4_special_summary.csv", index=False)
    print(f"\nwrote boxcut_v4_special_summary.csv ({len(summ)} runs)")
    if len(cands):
        cands.to_csv(HERE / "boxcut_v4_special_candidates.csv", index=False)
        print(f"wrote boxcut_v4_special_candidates.csv ({len(cands):,} clusters)")

    if want & {"all", "table"}:
        fig_table(summ)
    if want & {"all", "yield"}:
        fig_yields(summ)
    if want & {"all", "cosmic"}:
        fig_cosmic(summ)
    if want & {"all", "shapes"}:
        fig_shapes(cands, summ)

    print("\nNOTE: no efficiency and no capture time are produced for any special "
          "run, by design. See the module docstring.")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
