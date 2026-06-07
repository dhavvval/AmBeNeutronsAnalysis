"""
Charged-current (CC) event selection for tank-only neutrino-interaction MC.

These files (ANNIEEvent_cc_neutrino_*.root, produced by the
CC_MC_RECO_ntuple_neutrino ToolAnalysis chain) carry NO MRD or LAPPD
information, so the canonical CC selection's MRD-based cuts
(MRD coincidence, MRD-stop, projected-MRD-hit) cannot be applied truthfully.
We therefore build the CC mask from the MC-truth branches that ARE available,
plus the tank-only reco/flag branches, and EXPLICITLY log every cut that is
omitted because the detector subsystem is absent — no silent drops.

The EventSelector tool in the chain only *stores* flags (eventStatusFlagged,
SaveStatusToStore=1); it does not cut.  All cutting happens here, offline,
so thresholds can be varied without re-running the toolchain.

Truth branches used (written by ANNIEEventTreeMaker when MCTruth_fill=1,
HasGenie=1):
    trueCC           int   1 = charged-current interaction
    truePrimaryPdg   int   PDG of the primary lepton (13 = mu-, -13 = mu+)
    truePi0 / Pi0Count    pion/kaon content for the CC0pi requirement
    trueMultiRing    int   1 = multi-ring topology (single-ring = 0)
    trueVtxX/Y/Z     double true interaction vertex (cm, ANNIE coords)
    nhits            int   number of tank hits

Fiducial-volume definition matches the ANNIE tank FV convention used
elsewhere in the reco chain (radius and |z| about the tank centre).

EventSelector flag bitmask (UserTools/EventSelector/EventSelector.h), exposed
for callers who prefer flag-based cuts on eventStatusFlagged:
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs

# EventSelector flag bits (eventStatusFlagged) ------------------------------- #
FLAG = {
    "MCFV": 0x01, "MCPMTVol": 0x02, "MCMRD": 0x04, "MCPiK": 0x08,
    "RecoMRD": 0x10, "PromptTrig": 0x20, "NHit": 0x40, "RecoFV": 0x80,
    "RecoPMTVol": 0x100, "MCIsMuon": 0x200, "MCIsElectron": 0x400,
    "MCIsSingleRing": 0x800, "MCIsMultiRing": 0x1000, "MCProjectedMRDHit": 0x2000,
    "MCEnergyCut": 0x4000, "PMTMRDCoinc": 0x8000, "NoVeto": 0x10000,
    "Veto": 0x20000, "Trigger": 0x40000, "ThroughGoing": 0x80000,
    "RecoPDG": 0x100000, "Extended": 0x200000, "BeamOK": 0x400000,
}

# ANNIE tank fiducial volume — mirrors EventSelector::EventSelectionByFV
# (UserTools/EventSelector/EventSelector.cpp): on the true muon start vertex
#   r = sqrt(X^2 + Z^2) < 0.8 * tank_radius
#   |Y| < 50 cm
#   Z < 0                       (upstream-half convention)
# tank_radius = ANNIEGeometry GetCylRadius() ~= 152.4 cm -> fidcutradius ~= 121.9 cm.
TANK_RADIUS_CM = 152.4
FV_RADIUS_CM   = 0.8 * TANK_RADIUS_CM   # ~121.9 cm
FV_HALF_Z_CM   = 50.0                   # |Y| bound (ANNIE Y is the vertical axis)
FV_REQUIRE_Z_NEGATIVE = True            # EventSelector rejects Z > 0

# Event-level branches we try to read (all optional — graceful fallback).
CC_EVENT_BRANCHES = [
    "eventNumber",
    "eventStatusFlagged", "eventStatusApplied",
    "nhits",
    "trueCC", "truePrimaryPdg",
    "truePi0", "Pi0Count",
    "trueMultiRing",
    "trueVtxX", "trueVtxY", "trueVtxZ",
    "trueMuonEnergy", "trueNeutrons",
]

# Cuts that the canonical CC selection applies but which are IMPOSSIBLE on
# tank-only files (no MRD / no veto-paddle simulation).  Reported, never silent.
OMITTED_TANK_ONLY = [
    "PMT-MRD coincidence (kFlagPMTMRDCoinc) — no MRD in tank-only WCSim",
    "MRD-stop / through-going (kFlagMCMRD, kFlagThroughGoing) — no MRD",
    "projected-MRD-hit (kFlagMCProjectedMRDHit) — no MRD",
    "reco-MRD track (kFlagRecoMRD) — FindMrdTracks not run",
]


def _default_cc_cuts() -> dict:
    """Default truth-based CC selection for tank-only neutrino MC."""
    return {
        "require_cc": True,            # trueCC == 1
        "require_muon": True,          # |truePrimaryPdg| == 13
        "require_cc0pi": True,         # no pi0 / pions-kaons (truePi0==0, Pi0Count==0)
        "require_single_ring": True,   # trueMultiRing == 0
        "require_fv": True,            # true vertex inside tank FV (EventSelector convention)
        "fv_radius_cm": FV_RADIUS_CM,
        "fv_y_max_cm": FV_HALF_Z_CM,   # |Y| bound
        "fv_require_z_negative": FV_REQUIRE_Z_NEGATIVE,
        "nhit_min": 4,                 # tank hit multiplicity floor
        "muon_energy_min_mev": 0.0,    # optional lower bound on true muon KE
    }


def _read_event_table(root_path: Path, tree_name: str,
                      max_events: Optional[int], verbose: bool) -> pd.DataFrame:
    """Read the event-level CC branches from one ROOT file into a DataFrame."""
    with uproot.open(str(root_path)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree {tree_name!r} not in {root_path}")
        t = f[tree_name]
        available = set(t.keys())
        wanted = [b for b in CC_EVENT_BRANCHES if b in available]
        missing = set(CC_EVENT_BRANCHES) - available
        if missing and verbose:
            print(f"[cc] WARN {root_path.name}: missing CC branches (cut on these "
                  f"will be skipped): {sorted(missing)}")
        arr = t.arrays(wanted, library="pd", entry_stop=max_events)
    arr["_source_file"] = root_path.name
    return arr


def _compute_cc_pass(df: pd.DataFrame, cuts: dict, verbose: bool) -> pd.DataFrame:
    """Add a boolean `cc_pass` column (and per-cut columns) to the event table."""
    n = len(df)
    keep = np.ones(n, dtype=bool)
    cut_log = []

    def _apply(name: str, mask: np.ndarray):
        nonlocal keep
        before = int(keep.sum())
        keep &= mask
        after = int(keep.sum())
        cut_log.append((name, before, after))

    if cuts.get("require_cc") and "trueCC" in df:
        _apply("trueCC==1", df["trueCC"].to_numpy() == 1)

    if cuts.get("require_muon") and "truePrimaryPdg" in df:
        _apply("|truePrimaryPdg|==13", np.abs(df["truePrimaryPdg"].to_numpy()) == 13)

    if cuts.get("require_cc0pi"):
        pi_mask = np.ones(n, dtype=bool)
        if "truePi0" in df:
            pi_mask &= df["truePi0"].to_numpy() == 0
        if "Pi0Count" in df:
            pi_mask &= df["Pi0Count"].to_numpy() == 0
        _apply("CC0pi (no pi0)", pi_mask)

    if cuts.get("require_single_ring") and "trueMultiRing" in df:
        _apply("single-ring (trueMultiRing==0)", df["trueMultiRing"].to_numpy() == 0)

    if cuts.get("require_fv") and {"trueVtxX", "trueVtxY", "trueVtxZ"} <= set(df.columns):
        vx = df["trueVtxX"].to_numpy(); vy = df["trueVtxY"].to_numpy(); vz = df["trueVtxZ"].to_numpy()
        r = np.hypot(vx, vz)                       # EventSelector uses sqrt(X^2 + Z^2)
        in_fv = (r < cuts["fv_radius_cm"]) & (np.abs(vy) < cuts["fv_y_max_cm"])
        label = f"FV (r<{cuts['fv_radius_cm']:.1f}cm, |Y|<{cuts['fv_y_max_cm']:.0f}cm"
        if cuts.get("fv_require_z_negative"):
            in_fv &= (vz < 0.0)                     # EventSelector rejects Z > 0
            label += ", Z<0"
        _apply(label + ")", in_fv)

    if cuts.get("nhit_min", 0) > 0 and "nhits" in df:
        _apply(f"nhits>={cuts['nhit_min']}", df["nhits"].to_numpy() >= cuts["nhit_min"])

    if cuts.get("muon_energy_min_mev", 0) > 0 and "trueMuonEnergy" in df:
        _apply(f"trueMuonEnergy>={cuts['muon_energy_min_mev']}MeV",
               df["trueMuonEnergy"].to_numpy() >= cuts["muon_energy_min_mev"])

    df = df.copy()
    df["cc_pass"] = keep

    if verbose:
        print(f"[cc] CC selection cut flow ({n} events):")
        for name, before, after in cut_log:
            print(f"[cc]   {name:<42s} {before:>7d} -> {after:>7d}  "
                  f"({100*after/max(before,1):.1f}% kept)")
        print(f"[cc]   ------ FINAL cc_pass: {int(keep.sum())}/{n} "
              f"({100*keep.sum()/max(n,1):.1f}%)")
        print("[cc] OMITTED cuts (tank-only — detector subsystem absent):")
        for o in OMITTED_TANK_ONLY:
            print(f"[cc]   - {o}")
    return df


def build_cc_table(ctx: RunContext, tree_name: str = "Event",
                   max_events: Optional[int] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Read CC branches from all input ROOT files, compute cc_pass per event, and
    return a global event-level table.

    The returned DataFrame uses a global `eventID` (same offsetting scheme as
    ambe.mc.processor) so it can be merged onto the per-pulse parquet.
    Columns include: eventID, _source_file, eventNumber, cc_pass, and the
    truth branches used by the cuts.
    """
    cc_cuts = {**_default_cc_cuts(), **(ctx.cuts.get("cc_cuts", {}) if ctx.cuts else {})}
    root_files = resolve_inputs(ctx.inputs["root_files"])
    if max_events is None and ctx.cuts:
        max_events = ctx.cuts.get("max_events")

    if verbose:
        print(f"[cc] computing CC selection over {len(root_files)} file(s)")
        print(f"[cc] cuts: {cc_cuts}")

    frames = []
    event_id_offset = 0
    for rp in root_files:
        df = _read_event_table(rp, tree_name, max_events, verbose)
        df = _compute_cc_pass(df, cc_cuts, verbose)
        df = df.reset_index(drop=True)
        df["eventID"] = np.arange(len(df)) + event_id_offset
        event_id_offset += len(df)
        frames.append(df)

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table


def run(ctx: RunContext, tree_name: str = "Event",
        max_events: Optional[int] = None, verbose: bool = True) -> Path:
    """
    Build the CC event table and write it to
        <parquet_dir>/<run_name>__cc_events.parquet
    Also writes a one-line CC summary to
        <csv_dir>/<run_name>__cc_summary.csv
    """
    table = build_cc_table(ctx, tree_name=tree_name, max_events=max_events, verbose=verbose)
    out = ctx.parquet_path(f"{ctx.run_name}__cc_events")
    table.to_parquet(out, index=False)

    n = len(table)
    npass = int(table["cc_pass"].sum()) if n else 0
    summary = pd.DataFrame([{
        "run_name": ctx.run_name,
        "n_events": n,
        "n_cc_pass": npass,
        "cc_pass_frac": (npass / n) if n else 0.0,
    }])
    csv_out = ctx.csv_path(f"{ctx.run_name}__cc_summary")
    summary.to_csv(csv_out, index=False)

    if verbose:
        print(f"[cc] wrote {n} events ({npass} CC-passing) -> {out}")
        print(f"[cc] summary -> {csv_out}")
    return out


def apply_residual_filter(pulses: pd.DataFrame, ctx: RunContext,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Reduce the per-hit pulses DataFrame to the *delayed residual* used for the
    neutron search: the hits left after the CC event selection and after the
    prompt window (where the CC muon / prompt particles deposit their light) is
    removed.

    Controlled by an optional `residual:` block in the config:
        residual:
          cc_only: true            # keep only cc_pass==1 events (default true)
          prompt_window_ns: 2000   # drop hits with t <= this (default 2000)

    The prompt-window cut is purely time-based (no truth / BackTracker needed),
    so it is directly applicable to real data. OPTICS and cluster_features call
    this right after loading the pulses parquet.
    """
    block = (ctx.extra.get("residual", {}) if hasattr(ctx, "extra") else {}) or {}
    cc_only       = bool(block.get("cc_only", True))
    prompt_win_ns = float(block.get("prompt_window_ns", 2000.0))

    n0 = len(pulses)
    out = pulses
    if cc_only and "cc_pass" in out.columns:
        out = out[out["cc_pass"].astype(bool)]
    n_cc = len(out)
    if prompt_win_ns > 0 and "t" in out.columns:
        out = out[out["t"] > prompt_win_ns]
    n_res = len(out)

    if verbose:
        print(f"[residual] prompt_window_ns={prompt_win_ns:.0f}  cc_only={cc_only}")
        print(f"[residual]   {n0:,} hits -> {n_cc:,} CC-passing -> "
              f"{n_res:,} delayed residual (t>{prompt_win_ns:.0f} ns)")
        if "truth_class" in out.columns and n_res:
            neut = int(out["truth_class"].isin([1, 2, 3, 4]).sum())
            print(f"[residual]   residual neutron-hit fraction (truth): "
                  f"{100*neut/n_res:.1f}%  (diagnostic only)")
    return out.reset_index(drop=True)


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc cc")
    p.add_argument("--tree", default="Event")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(list(argv) if argv else [])
    run(ctx, tree_name=args.tree, max_events=args.max_events, verbose=not args.quiet)
