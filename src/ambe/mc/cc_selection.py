"""
Charged-current (CC) event selection for neutrino-interaction MC.

Originally written for *tank-only* MC files (ANNIEEvent_cc_neutrino_*.root from
the CC_MC_RECO_ntuple_neutrino ToolAnalysis chain), which carry NO MRD or LAPPD
information so the canonical MRD-based cuts cannot be applied.  This module now
supports several richer files that DO carry MRD / FMV-veto / cluster branches
(e.g. ANNIEEvent_cc_neutrino_FMVMRDTEST*.root).  Rather than hard-code one cut
chain, the selection is expressed as a **registry of named cuts** grouped into
**named streams**, so the marginal effect of every individual cut can be studied
(cut-flow + N-1) and thresholds can be varied from the YAML config without
re-running the toolchain.

What each cut maps onto (James's CCPR / CCInc reference)
-------------------------------------------------------
This file has truth + MRD/veto/cluster branches but NO reco branches
(recoVtxX/Y/Z, recoDirX/Y/Z, recoMuonKE, promptMuonTotalPE, Qij are absent), so
James's reco-level cuts are applied as truth-level equivalents where possible:

    Fiducial volume   -> trueVtxX/Y/Z : hypot(X,Z)<100 cm, |Y|<100 cm
    Is CC             -> trueCC==1
    FSL is muon       -> trueFSLPdg==13       (NOT truePrimaryPdg — see below)
    Muon momentum     -> p = |trueFSLMomentum_(X,Y,Z)| in [600,1200) MeV/c
    Muon angle        -> cos(theta) = pz/p > 0.8   (Z = beam axis)
    PMT-MRD coinc.    -> MRD tag (100%-efficiency assumption, see mrd_source)
    No FMV veto       -> NoVeto==1
    Min PMT hits      -> nhits >= 4
    PromptPE 500-3000 -> sum(hitPE | hitT<=2000ns)  ('prompt_pe' cut)

Two corrections baked in for CCInc
----------------------------------
* The legacy "muon" cut used |truePrimaryPdg|==13.  truePrimaryPdg is the
  *primary* lepton and is set even for many NC events, so it over-counts.  The
  correct "final-state lepton is a muon" cut is trueFSLPdg==13.  CCInc streams
  use trueFSLPdg; the legacy `primary_muon` cut is kept only for cc0pi_legacy.
* trueMuonEnergy is the muon's TOTAL ENERGY, not momentum; James's cut is on
  momentum.  CCInc uses `mu_p` computed from trueFSLMomentum_*.  The legacy
  `muon_energy_min_mev` knob remains but is diagnostic only.

MRD 100%-efficiency tag
-----------------------
Under the "MRD is 100% efficient, count any MRD activity" assumption the
`mrd_tag` cut can be sourced from one of three branches (config `mrd_source`):
    "cluster" (default) : MRDClusterNumber > 0      ("any MRD hit/cluster")
    "tracks"            : sum(NumClusterTracks) > 0  (>=1 reconstructed track)
    "coinc"             : TankMRDCoinc == 1          (Tank-MRD coincidence flag)

Absent-branch handling
----------------------
Each cut declares the branches it needs (`requires`).  If a requested cut's
branches are missing from a file, the cut is SKIPPED and reported per stream —
no silent drops.  On a true tank-only file the ccinc_mrd stream therefore
auto-reports mrd_tag / no_veto as omitted with no code change.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs, filter_files_with_tree

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

# ANNIE tank fiducial volume (James's CCPR reference): on the true muon start vertex
#   r = sqrt(X^2 + Z^2) < 100 cm  (1 m radius)
#   |Y| < 100 cm                  (2 m total height)
TANK_RADIUS_CM = 152.4
TANK_HALF_Y_CM = 198.0   # tank half-height; |Y| < 198 cm spans the full water volume
FV_RADIUS_CM   = 100.0   # 1 m radius (James's CCPR reference)
FV_HALF_Y_CM   = 100.0   # |Y| < 100 cm, 2 m height (James's CCPR reference)
FV_REQUIRE_Z_NEGATIVE = False

# --- Neutrino-interaction vertex, GENIE -> tank coordinates ------------------ #
# There are TWO vertex branch families and they are NOT interchangeable:
#
#   trueVtx{X,Y,Z}        the WCSim PRIMARY-PARTICLE START POINT.  Equal to the
#                         interaction vertex ONLY when the interaction happened
#                         inside the tank.  In world-volume samples it is a
#                         sentinel (0, 14.4602, -168.100) for ~67% of events
#                         (co-occurring with trueMuonEnergy == -9999) and the
#                         TANK-ENTRY POINT on the r=152.4 cm wall for the rest.
#   trueNuIntxVtx_{X,Y,Z} the GENIE interaction vertex, in GENIE/world
#                         coordinates.  Always the real interaction point.
#
# The two families differ by a rigid offset, measured on the productionv3 tank
# sample over events with a valid WCSim primary:
#     dx = 0.000 +- 0.011,  dy = -14.466 +- 0.062,  dz = +168.100 +- 0.013  [cm]
# so tank coords = GENIE coords + (0, +14.466, -168.100).  Re-measure with:
#     mean(trueVtxY - trueNuIntxVtx_Y) over trueMuonEnergy != -9999 events
# The offset is verified at runtime by _origin_offset_report() below, which is
# the guard against a future production changing the convention silently.
NUVTX_OFFSET_X_CM =    0.000
NUVTX_OFFSET_Y_CM =   14.466
NUVTX_OFFSET_Z_CM = -168.100

# Sentinel written when there is no primary muon (all NC events, plus CC events
# whose muon was produced outside the tank).  Not filtered anywhere: it only
# reaches the diagnostic muon_energy_min_mev cut, which defaults to off.
NO_MUON_SENTINEL = -9999.0

# Default prompt window (ns) used by the optional prompt_pe cut; kept consistent
# with apply_residual_filter's `residual.prompt_window_ns`.
PROMPT_WINDOW_NS = 2000.0

# Event-level (scalar) branches we try to read.  All optional — graceful
# fallback if absent (presence-filtered in _read_event_table).
CC_EVENT_BRANCHES = [
    "eventNumber",
    "eventStatusFlagged", "eventStatusApplied",
    "nhits", "numberOfClusters",
    "trueCC", "truePrimaryPdg", "trueFSLPdg",
    "trueFSLEnergy", "trueFSLMomentum_X", "trueFSLMomentum_Y", "trueFSLMomentum_Z",
    "truePi0", "Pi0Count",
    "truePiPlus", "truePiMinus", "PiPlusCount", "PiMinusCount",
    "trueMultiRing",
    "trueVtxX", "trueVtxY", "trueVtxZ",
    # GENIE interaction vertex — all three coordinates.  Needed to decide whether
    # the neutrino interacted inside the tank at all (see _augment_origin_columns).
    "trueNuIntxVtx_X", "trueNuIntxVtx_Y", "trueNuIntxVtx_Z",
    "trueMuonEnergy", "trueNeutrons",
    # MRD / veto scalar branches (richer files only)
    "MRDClusterNumber", "TankMRDCoinc",
    "NoVeto", "vetoHit",
]

# Vector branches read separately with library="ak" and reduced to scalar
# columns (see _augment_vector_columns).
VECTOR_BRANCHES = ["NumClusterTracks", "MRDStop", "hitT", "hitPE"]

# DEPRECATED — kept only so old imports don't break.  Omission is now dynamic
# and per-stream (see _compute_pass / the `omitted` return value).
OMITTED_TANK_ONLY = [
    "PMT-MRD coincidence (kFlagPMTMRDCoinc) — no MRD in tank-only WCSim",
    "MRD-stop / through-going (kFlagMCMRD, kFlagThroughGoing) — no MRD",
    "projected-MRD-hit (kFlagMCProjectedMRDHit) — no MRD",
    "reco-MRD track (kFlagRecoMRD) — FindMrdTracks not run",
]


# --------------------------------------------------------------------------- #
# Cut registry + streams
# --------------------------------------------------------------------------- #
@dataclass
class Cut:
    """One named selection cut.

    name        human-readable label shown in the cut-flow table
    fn          predicate: fn(df, cuts) -> np.ndarray[bool] of length len(df)
    requires    df columns the predicate needs (drives dynamic omission)
    default_on  active unless the matching config toggle is set false
    toggle      config key that switches the whole cut on/off (None = always on)
    """
    name: str
    fn: Callable[[pd.DataFrame, dict], np.ndarray]
    requires: Tuple[str, ...] = ()
    default_on: bool = True
    toggle: Optional[str] = None


def _p_mu(df: pd.DataFrame) -> np.ndarray:
    """Muon (final-state-lepton) momentum magnitude in MeV/c."""
    px = df["trueFSLMomentum_X"].to_numpy(dtype=float)
    py = df["trueFSLMomentum_Y"].to_numpy(dtype=float)
    pz = df["trueFSLMomentum_Z"].to_numpy(dtype=float)
    return np.sqrt(px * px + py * py + pz * pz)


def _cos_theta(df: pd.DataFrame) -> np.ndarray:
    """cos(theta_mu) w.r.t. the beam (Z) axis; -2 where momentum is zero."""
    pz = df["trueFSLMomentum_Z"].to_numpy(dtype=float)
    p = _p_mu(df)
    return np.where(p > 0, pz / p, -2.0)


def _mrd_tag_mask(df: pd.DataFrame, cuts: dict) -> np.ndarray:
    """MRD-tag under the 100%-efficiency assumption; source set by `mrd_source`."""
    source = str(cuts.get("mrd_source", "cluster")).lower()
    if source == "tracks":
        return df["_mrd_ntracks_sum"].to_numpy() > 0
    if source == "coinc":
        return df["TankMRDCoinc"].to_numpy() == 1
    # default "cluster"
    return df["MRDClusterNumber"].to_numpy() > 0


def _mrd_tag_requires(cuts: dict) -> Tuple[str, ...]:
    source = str(cuts.get("mrd_source", "cluster")).lower()
    return {"tracks": ("_mrd_ntracks_sum",),
            "coinc": ("TankMRDCoinc",)}.get(source, ("MRDClusterNumber",))


def _make_cut_registry() -> dict[str, Cut]:
    """Build the registry of all known cuts (keyed by short id)."""
    return {
        # --- truth core ---
        "cc": Cut("trueCC==1",
                  lambda df, c: df["trueCC"].to_numpy() == 1,
                  ("trueCC",), toggle="require_cc"),
        # CCInc: final-state lepton must be a muon
        "fsl_muon": Cut("FSL is muon (trueFSLPdg==13)",
                        lambda df, c: df["trueFSLPdg"].to_numpy() == 13,
                        ("trueFSLPdg",), toggle="require_fsl_muon"),
        # legacy CC0pi muon cut (primary lepton; over-counts NC — diagnostic)
        "primary_muon": Cut("|truePrimaryPdg|==13",
                            lambda df, c: np.abs(df["truePrimaryPdg"].to_numpy()) == 13,
                            ("truePrimaryPdg",), toggle="require_muon"),
        # --- muon kinematics (truth) ---
        "mu_p": Cut("p_mu in [600,1200) MeV/c",
                    lambda df, c: (_p_mu(df) >= c.get("mu_p_min_mev", 600.0))
                                  & (_p_mu(df) < c.get("mu_p_max_mev", 1200.0)),
                    ("trueFSLMomentum_X", "trueFSLMomentum_Y", "trueFSLMomentum_Z")),
        "cos_theta": Cut("cos(theta) > 0.8",
                         lambda df, c: _cos_theta(df) > c.get("cos_theta_min", 0.8),
                         ("trueFSLMomentum_X", "trueFSLMomentum_Y", "trueFSLMomentum_Z")),
        # --- fiducial volume (split so each sub-cut's effect is measurable) ---
        "fv_radius": Cut("FV r < 100 cm",
                         lambda df, c: np.hypot(df["trueVtxX"].to_numpy(),
                                                df["trueVtxZ"].to_numpy())
                                       < c.get("fv_radius_cm", FV_RADIUS_CM),
                         ("trueVtxX", "trueVtxZ"), toggle="require_fv"),
        "fv_y": Cut("FV |Y| < 100 cm",
                    lambda df, c: np.abs(df["trueVtxY"].to_numpy())
                                  < c.get("fv_y_max_cm", FV_HALF_Y_CM),
                    ("trueVtxY",), toggle="require_fv"),
        "fv_z": Cut("FV Z < 0",
                    lambda df, c: df["trueVtxZ"].to_numpy() < 0.0,
                    ("trueVtxZ",), toggle="require_fv_z_negative",
                    default_on=FV_REQUIRE_Z_NEGATIVE),
        # --- detector / activity ---
        "nhit": Cut("nhits >= 4",
                    lambda df, c: df["nhits"].to_numpy() >= c.get("nhit_min", 4),
                    ("nhits",)),
        "mrd_tag": Cut("MRD-tagged (any MRD activity)",
                       _mrd_tag_mask, ("MRDClusterNumber",), toggle="require_mrd"),
        "no_veto": Cut("NoVeto == 1",
                       lambda df, c: df["NoVeto"].to_numpy() == 1,
                       ("NoVeto",), toggle="require_no_veto"),
        "prompt_pe": Cut("promptPE in [500,3000)",
                         lambda df, c: (df["_prompt_pe_sum"].to_numpy() >= c.get("prompt_pe_min", 500.0))
                                       & (df["_prompt_pe_sum"].to_numpy() < c.get("prompt_pe_max", 3000.0)),
                         ("_prompt_pe_sum",), toggle="require_prompt_pe", default_on=True),
        # --- legacy CC0pi-only ---
        "cc0pi": Cut("CC0pi (no π± in final state)",
                     lambda df, c: _cc0pi_mask(df),
                     ("truePiPlus", "truePiMinus"), toggle="require_cc0pi"),
        "single_ring": Cut("single-ring (trueMultiRing==0)",
                           lambda df, c: df["trueMultiRing"].to_numpy() == 0,
                           ("trueMultiRing",), toggle="require_single_ring"),
        # --- interaction-origin cuts (world-volume samples) -------------------
        # NOT part of any stream and default_on=False: origin_in_tank is a LABEL
        # input, not a selection cut.  Registered so an origin-restricted variant
        # can be recombined off a persisted table without reprocessing.
        "origin_in_tank": Cut("interaction vertex inside tank",
                              lambda df, c: df["origin_in_tank"].to_numpy() == 1,
                              ("origin_in_tank",), toggle="require_origin_in_tank",
                              default_on=False),
        "origin_outside_tank": Cut("interaction vertex outside tank",
                                   lambda df, c: df["origin_in_tank"].to_numpy() == 0,
                                   ("origin_in_tank",),
                                   toggle="require_origin_outside_tank",
                                   default_on=False),
    }


def _cc0pi_mask(df: pd.DataFrame) -> np.ndarray:
    """CC0pi charged-pion veto: no π± in the final state.

    The reference cut is P_π± > 210 MeV/c but no per-pion momentum branches
    exist in the MC files; this count-based veto is conservative (removes any
    event with at least one π±, regardless of momentum).
    """
    mask = np.ones(len(df), dtype=bool)
    plus_col = ("truePiPlus" if "truePiPlus" in df.columns
                else "PiPlusCount" if "PiPlusCount" in df.columns else None)
    minus_col = ("truePiMinus" if "truePiMinus" in df.columns
                 else "PiMinusCount" if "PiMinusCount" in df.columns else None)
    if plus_col:
        mask &= df[plus_col].to_numpy() == 0
    if minus_col:
        mask &= df[minus_col].to_numpy() == 0
    return mask


_CUT_REGISTRY = _make_cut_registry()

# Named streams: ordered list of cut ids.  cc0pi_legacy preserves the original
# behaviour; the ccinc_* streams are the CC-inclusive selections.
_STREAMS: dict[str, List[str]] = {
    "cc0pi_legacy": ["cc", "fsl_muon", "cc0pi", "single_ring",
                     "fv_radius", "fv_y", "fv_z", "nhit"],
    "ccinc_truth": ["cc", "fsl_muon", "mu_p", "cos_theta",
                    "fv_radius", "fv_y", "fv_z", "nhit"],
    "ccinc_mrd": ["cc", "fsl_muon", "mrd_tag", "no_veto",
                  "fv_radius", "fv_y", "fv_z", "nhit", "prompt_pe"],
    "ccinc_truth_plus_mrd": ["cc", "fsl_muon", "mu_p", "cos_theta",
                             "fv_radius", "fv_y", "fv_z", "nhit",
                             "mrd_tag", "no_veto", "prompt_pe"],
    # --- The two INDEPENDENT streamlines (2026-08-03). ------------------------
    # These are alternatives, NOT a combination: they share the fiducial-volume
    # and muon-kinematics phase space and differ only in how the muon is TAGGED.
    # ccinc_truth_plus_mrd above ANDs both taggings together, which conflates
    # them — use it only if you specifically want the intersection.
    #
    # Cut order is deliberate and differs from the older streams: fiducial volume
    # first, then muon kinematics, then the tagging cuts. That way the cut-flow's
    # relative-efficiency column reads as "what does the tagging cost me on top of
    # a fixed phase space", which is the comparison the two streamlines exist to
    # make. Final survivor counts are order-independent; only the table changes.
    "ccinc_truthtag": ["fv_radius", "fv_y", "fv_z",
                       "mu_p", "cos_theta",
                       "cc", "fsl_muon",
                       "nhit"],
    "ccinc_recotag": ["fv_radius", "fv_y", "fv_z",
                      "mu_p", "cos_theta",
                      "no_veto", "mrd_tag", "prompt_pe",
                      "nhit"],
    # --- World-volume variants (2026-08-04). ----------------------------------
    # Same tagging as the tank streamlines above, with the fiducial-volume cuts
    # REMOVED.  Two reasons, and neither is a shortcut:
    #
    #  1. The FV cut reads trueVtx*, which in a world sample is not the
    #     interaction vertex at all (sentinel for ~67% of events, tank-entry point
    #     for the rest) — so cutting on it there is meaningless, not just lossy.
    #  2. Even on the correct vertex, requiring r<100 / |Y|<100 keeps only ~5% of
    #     world events and removes exactly the out-of-tank population the world
    #     sample exists to provide.
    #
    # `origin_in_tank` is persisted for every event either way, so an FV- or
    # origin-restricted variant is recoverable from the saved table without
    # reprocessing.  The FV cut is a MUON phase-space cut and plays no part in the
    # signal/background label; see _augment_origin_columns.
    "ccinc_truthtag_world": ["mu_p", "cos_theta",
                             "cc", "fsl_muon",
                             "nhit"],
    "ccinc_recotag_world": ["mu_p", "cos_theta",
                            "no_veto", "mrd_tag", "prompt_pe",
                            "nhit"],
}


def _resolve_stream(name: str, cuts: dict) -> List[Cut]:
    """Return the ordered list of active Cut objects for a stream name.

    Streams can be redefined / added via cuts['cc_streams'][name] (a list of
    cut ids).  Per-cut toggles in `cuts` (e.g. require_fv: false) drop a cut.
    """
    custom = (cuts.get("cc_streams") or {})
    if name in custom:
        ids = list(custom[name])
    elif name in _STREAMS:
        ids = list(_STREAMS[name])
    else:
        raise KeyError(f"unknown stream {name!r}; known: "
                       f"{sorted(set(_STREAMS) | set(custom))}")

    active: List[Cut] = []
    for cid in ids:
        if cid not in _CUT_REGISTRY:
            raise KeyError(f"stream {name!r} references unknown cut id {cid!r}")
        cut = _CUT_REGISTRY[cid]
        on = cut.default_on
        if cut.toggle is not None and cut.toggle in cuts:
            on = bool(cuts[cut.toggle])
        if on:
            # mrd_tag's requires depend on mrd_source — resolve dynamically.
            if cid == "mrd_tag":
                cut = Cut(cut.name, cut.fn, _mrd_tag_requires(cuts),
                          cut.default_on, cut.toggle)
            active.append(cut)
    return active


def _default_cc_cuts() -> dict:
    """Default truth-based CC0pi selection (legacy; tank-only files)."""
    return {
        "require_cc": True,
        "require_fsl_muon": True,
        "require_cc0pi": True,
        "require_single_ring": True,
        "require_fv": True,
        "fv_radius_cm": FV_RADIUS_CM,
        "fv_y_max_cm": FV_HALF_Y_CM,
        "fv_require_z_negative": FV_REQUIRE_Z_NEGATIVE,
        "nhit_min": 4,
        "muon_energy_min_mev": 0.0,   # diagnostic only (TOTAL energy, not momentum)
    }


def _default_ccinc_cuts() -> dict:
    """Default CC-inclusive selection knobs (James's CCPR reference)."""
    return {
        "require_cc": True,
        "require_fsl_muon": True,
        "mu_p_min_mev": 600.0,
        "mu_p_max_mev": 1200.0,
        "cos_theta_min": 0.8,
        "require_fv": True,
        "fv_radius_cm": FV_RADIUS_CM,
        "fv_y_max_cm": FV_HALF_Y_CM,
        "fv_require_z_negative": FV_REQUIRE_Z_NEGATIVE,
        "nhit_min": 4,
        "require_mrd": True,
        "mrd_source": "cluster",      # cluster | tracks | coinc
        "require_no_veto": True,
        "require_prompt_pe": True,
        "prompt_pe_min": 500.0,
        "prompt_pe_max": 3000.0,
        "train_targets": [1000, 5000, 10000, 50000],
    }


def _merged_cuts(ctx: Optional[RunContext]) -> dict:
    """Defaults (legacy + ccinc) overlaid with config's cuts['cc_cuts']."""
    base = {**_default_cc_cuts(), **_default_ccinc_cuts()}
    override = (ctx.cuts.get("cc_cuts", {}) if (ctx and ctx.cuts) else {}) or {}
    return {**base, **override}


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _read_event_table(root_path: Path, tree_name: str,
                      max_events: Optional[int], verbose: bool) -> pd.DataFrame:
    """Read the scalar CC branches from one ROOT file into a DataFrame."""
    with uproot.open(str(root_path)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree {tree_name!r} not in {root_path}")
        t = f[tree_name]
        available = set(t.keys())
        wanted = [b for b in CC_EVENT_BRANCHES if b in available]
        missing = set(CC_EVENT_BRANCHES) - available
        if missing and verbose:
            print(f"[cc] WARN {root_path.name}: missing scalar branches "
                  f"(cuts on these will be skipped): {sorted(missing)}")
        arr = t.arrays(wanted, library="pd", entry_stop=max_events)
    arr["_source_file"] = root_path.name
    return arr


def _augment_origin_columns(df: pd.DataFrame, cuts: Optional[dict] = None) -> pd.DataFrame:
    """Add the interaction-vertex-in-tank-coordinates columns and origin flags.

    Always applied (not stream-dependent), because `origin_in_tank` is the
    signal/background discriminator for world-volume samples and must be present
    in every persisted table:

      _nuvtx_x_tank / _nuvtx_y_tank / _nuvtx_z_tank   GENIE vertex in tank coords
      _nuvtx_r                                        hypot(x, z); ANNIE axis is Y
      origin_in_tank                                  1 = neutrino interacted
                                                      inside the tank water
      origin_in_fv                                    1 = interaction vertex also
                                                      inside the FIDUCIAL volume

    `origin_in_fv` exists so a merged tank+world training can hold the SIGNAL class
    to one homogeneous phase space.  The tank sample's signal is FV-selected; world
    in-tank events are not (the world streams drop the FV cut because it reads
    trueVtx*, which is wrong there).  Measured on the full world sample: only 29.3%
    of world in-tank cc_pass events are also inside the FV, so merging without this
    would make 71% of the world signal near-wall events that the tank signal never
    contains -- and d_wall is a top-4 discriminator.

    Absent branches leave every column absent, so a consumer falls back to its
    "column missing" default rather than getting a wrong answer — the same
    convention the cut registry uses for dynamic omission.

    NOTE the FV *cuts* intentionally still read trueVtx*, not these columns. These
    are LABEL inputs consumed by mva_analysis, never selection cuts: origin_in_tank
    decides signal-vs-background, origin_in_fv decides which clusters are eligible
    for the signal class at all.
    """
    need = ("trueNuIntxVtx_X", "trueNuIntxVtx_Y", "trueNuIntxVtx_Z")
    if not all(b in df.columns for b in need):
        return df
    df = df.copy()
    df["_nuvtx_x_tank"] = df["trueNuIntxVtx_X"].to_numpy(dtype=float) + NUVTX_OFFSET_X_CM
    df["_nuvtx_y_tank"] = df["trueNuIntxVtx_Y"].to_numpy(dtype=float) + NUVTX_OFFSET_Y_CM
    df["_nuvtx_z_tank"] = df["trueNuIntxVtx_Z"].to_numpy(dtype=float) + NUVTX_OFFSET_Z_CM
    df["_nuvtx_r"] = np.hypot(df["_nuvtx_x_tank"].to_numpy(),
                              df["_nuvtx_z_tank"].to_numpy())
    df["origin_in_tank"] = (
        (df["_nuvtx_r"].to_numpy() < TANK_RADIUS_CM)
        & (np.abs(df["_nuvtx_y_tank"].to_numpy()) < TANK_HALF_Y_CM)
    ).astype(int)
    # Same FV thresholds the fv_radius / fv_y cuts use, read from the config so the
    # two cannot drift apart, but evaluated on the GENIE vertex rather than trueVtx.
    c = cuts or {}
    df["origin_in_fv"] = (
        (df["_nuvtx_r"].to_numpy() < float(c.get("fv_radius_cm", FV_RADIUS_CM)))
        & (np.abs(df["_nuvtx_y_tank"].to_numpy()) < float(c.get("fv_y_max_cm", FV_HALF_Y_CM)))
    ).astype(int)
    return df


def _origin_offset_report(df: pd.DataFrame, verbose: bool) -> None:
    """Validate the GENIE->tank vertex offset and report the in/out-of-tank split.

    On events that (a) interacted inside the tank and (b) have a real WCSim
    primary, trueVtx* must equal the offset-corrected GENIE vertex to well under
    a centimetre.  If a future production changes the coordinate convention this
    is the check that catches it — a wrong offset would otherwise mislabel every
    event's origin while looking perfectly plausible.
    """
    if not verbose or "origin_in_tank" not in df.columns or not len(df):
        return
    n = len(df)
    n_in = int(df["origin_in_tank"].sum())
    print(f"[cc] interaction origin: {n_in}/{n} inside tank ({100.0 * n_in / n:.1f}%), "
          f"{n - n_in} outside ({100.0 * (n - n_in) / n:.1f}%)")
    r = df["_nuvtx_r"].to_numpy(dtype=float)
    print(f"[cc]   _nuvtx_r range: {r.min():.1f} .. {r.max():.1f} cm")

    if not all(c in df.columns for c in ("trueVtxX", "trueVtxY", "trueVtxZ")):
        return
    m = df["origin_in_tank"].to_numpy() == 1
    if "trueMuonEnergy" in df.columns:
        # trueVtx is a sentinel when there is no primary muon; those rows say
        # nothing about the offset and must not enter the comparison.
        m &= df["trueMuonEnergy"].to_numpy(dtype=float) != NO_MUON_SENTINEL
    if m.sum() < 10:
        print(f"[cc]   offset check skipped: only {int(m.sum())} in-tank events "
              "with a valid WCSim primary")
        return
    d = np.stack([
        np.abs(df["trueVtxX"].to_numpy(dtype=float)[m] - df["_nuvtx_x_tank"].to_numpy()[m]),
        np.abs(df["trueVtxY"].to_numpy(dtype=float)[m] - df["_nuvtx_y_tank"].to_numpy()[m]),
        np.abs(df["trueVtxZ"].to_numpy(dtype=float)[m] - df["_nuvtx_z_tank"].to_numpy()[m]),
    ])
    worst = d.max(axis=1)
    frac_bad = float((d.max(axis=0) > 1.0).mean())
    print(f"[cc]   GENIE->tank offset check on {int(m.sum())} in-tank events: "
          f"max|dx,dy,dz| = ({worst[0]:.3f}, {worst[1]:.3f}, {worst[2]:.3f}) cm, "
          f"{100.0 * frac_bad:.2f}% exceed 1 cm")
    if frac_bad > 0.02:
        print("[cc]   *** WARNING: the GENIE->tank vertex offset does not hold on this "
              "sample. NUVTX_OFFSET_*_CM in cc_selection.py was measured on productionv3 "
              "tank/fmvmrd; re-measure before trusting origin_in_tank. ***")


def _augment_vector_columns(root_path: Path, df: pd.DataFrame, cuts: dict,
                            stream_ids: Sequence[str], tree_name: str,
                            max_events: Optional[int], verbose: bool) -> pd.DataFrame:
    """Read the few vector branches a stream needs and reduce them to scalars.

    Mirrors processor.py's library="ak" pattern.  Only reads what's needed:
      * NumClusterTracks -> _mrd_ntracks_sum  (when mrd_source == "tracks")
      * hitT, hitPE      -> _prompt_pe_sum    (when 'prompt_pe' in stream)
    Missing branches simply leave the derived column absent, so the dependent
    cut is reported as omitted later.
    """
    import awkward as ak

    # "__all__" forces every derived column regardless of what this stream cuts on.
    # Used when the event table is persisted: a table missing _prompt_pe_sum cannot
    # re-derive a promptPE cut later, so the saved table would silently only support
    # the stream that produced it. Costs one extra vector read per file.
    force_all = "__all__" in stream_ids
    need_tracks = force_all or ("mrd_tag" in stream_ids and
                                str(cuts.get("mrd_source", "cluster")).lower() == "tracks")
    need_prompt = force_all or "prompt_pe" in stream_ids
    if not (need_tracks or need_prompt):
        return df

    prompt_win = float((getattr(_augment_vector_columns, "_prompt_win", None)
                        or PROMPT_WINDOW_NS))

    with uproot.open(str(root_path)) as f:
        t = f[tree_name]
        available = set(t.keys())
        wanted = [b for b in VECTOR_BRANCHES if b in available]
        if not wanted:
            if verbose:
                print(f"[cc] WARN {root_path.name}: no vector branches present; "
                      "derived MRD/promptPE columns unavailable")
            return df
        varr = t.arrays(wanted, library="ak", entry_stop=max_events)

    n = len(df)
    if need_tracks and "NumClusterTracks" in wanted:
        s = ak.to_numpy(ak.sum(varr["NumClusterTracks"], axis=1)).astype(float)
        df = df.copy(); df["_mrd_ntracks_sum"] = s[:n]
    if need_prompt and "hitT" in wanted and "hitPE" in wanted:
        in_win = varr["hitT"] <= prompt_win
        pe_sum = ak.to_numpy(ak.sum(varr["hitPE"][in_win], axis=1)).astype(float)
        df = df.copy(); df["_prompt_pe_sum"] = pe_sum[:n]
    return df


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _compute_pass(df: pd.DataFrame, stream_cuts: List[Cut], cuts: dict,
                  verbose: bool, stream_name: str = "") -> Tuple[pd.DataFrame, list, list]:
    """Apply a stream's cuts in order; return (df+cc_pass, cut_log, omitted).

    cut_log : list of (name, before, after) sequential cut-flow rows.
    omitted : list of strings describing cuts skipped for missing branches.
    """
    n = len(df)
    keep = np.ones(n, dtype=bool)
    cut_log: list = []
    omitted: list = []

    def _apply(name: str, mask: np.ndarray):
        nonlocal keep
        before = int(keep.sum())
        keep &= mask
        after = int(keep.sum())
        cut_log.append((name, before, after))

    cols = set(df.columns)
    for cut in stream_cuts:
        missing = [c for c in cut.requires if c not in cols]
        if missing:
            omitted.append(f"{cut.name}: skipped — missing branch(es) {sorted(missing)}")
            continue
        _apply(cut.name, np.asarray(cut.fn(df, cuts), dtype=bool))

    df = df.copy()
    df["cc_pass"] = keep

    if verbose:
        tag = f" [{stream_name}]" if stream_name else ""
        print(f"[cc] cut flow{tag} ({n} events):")
        for name, before, after in cut_log:
            print(f"[cc]   {name:<34s} {before:>7d} -> {after:>7d}  "
                  f"({100*after/max(before,1):.1f}% kept)")
        print(f"[cc]   ------ FINAL cc_pass: {int(keep.sum())}/{n} "
              f"({100*keep.sum()/max(n,1):.1f}%)")
        if omitted:
            print(f"[cc] OMITTED cuts{tag} (branch absent):")
            for o in omitted:
                print(f"[cc]   - {o}")
    return df, cut_log, omitted


def _compute_cc_pass(df: pd.DataFrame, cuts: dict, verbose: bool) -> pd.DataFrame:
    """Backward-compatible wrapper: legacy CC0pi selection -> `cc_pass` column."""
    stream_cuts = _resolve_stream("cc0pi_legacy", cuts)
    out, _log, _om = _compute_pass(df, stream_cuts, cuts, verbose, "cc0pi_legacy")
    return out


def nminus1_table(df: pd.DataFrame, stream_cuts: List[Cut], cuts: dict,
                  stream_name: str = "") -> pd.DataFrame:
    """N-1 per-cut impact study.

    For each cut k report:
      standalone_pass  : events passing k alone (over all events)
      n_minus_one      : events passing ALL cuts except k
      full_pass        : events passing ALL cuts
      marginal_removed : n_minus_one - full_pass (events ONLY k removes)
      n1_efficiency    : full_pass / n_minus_one (survival of k given the rest)
    """
    cols = set(df.columns)
    masks: list[tuple[str, np.ndarray]] = []
    for cut in stream_cuts:
        if all(c in cols for c in cut.requires):
            masks.append((cut.name, np.asarray(cut.fn(df, cuts), dtype=bool)))

    n = len(df)
    if not masks:
        return pd.DataFrame()
    full = np.ones(n, dtype=bool)
    for _name, m in masks:
        full &= m
    full_pass = int(full.sum())

    rows = []
    for i, (name, m) in enumerate(masks):
        minus = np.ones(n, dtype=bool)
        for j, (_n2, m2) in enumerate(masks):
            if j != i:
                minus &= m2
        n_minus = int(minus.sum())
        rows.append({
            "stream": stream_name,
            "cut_name": name,
            "standalone_pass": int(m.sum()),
            "n_minus_one": n_minus,
            "full_pass": full_pass,
            "marginal_removed": n_minus - full_pass,
            "n1_efficiency": (full_pass / n_minus) if n_minus else float("nan"),
        })
    return pd.DataFrame(rows)


def estimate_training_stats(eff: float, n_pass: int, n_total: int,
                            targets: Sequence[int], events_per_file: int = 0,
                            verbose: bool = True) -> pd.DataFrame:
    """Estimate raw MC stats needed to reach target selected-sample sizes.

    Uses a Wilson score interval (z=1.96) on the binomial efficiency
    eff = n_pass / n_total to bracket the raw requirement.  raw_needed = T/eff;
    the band uses eff_lo/eff_hi.  Relative stat error on the selected count at
    raw N is ~1/sqrt(N*eff).
    """
    z = 1.96
    if n_total > 0:
        p = n_pass / n_total
        denom = 1.0 + z * z / n_total
        centre = (p + z * z / (2 * n_total)) / denom
        half = (z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))) / denom
        eff_lo = max(centre - half, 1e-9)
        eff_hi = min(centre + half, 1.0)
    else:
        eff_lo = eff_hi = eff = max(eff, 1e-9)
    eff = max(eff, 1e-9)

    rows = []
    for T in targets:
        raw = T / eff
        rows.append({
            "target_selected": int(T),
            "eff": eff,
            "eff_lo": eff_lo,
            "eff_hi": eff_hi,
            "raw_needed": int(math.ceil(raw)),
            "raw_needed_lo": int(math.ceil(T / eff_hi)),   # optimistic (high eff)
            "raw_needed_hi": int(math.ceil(T / eff_lo)),   # conservative (low eff)
            "files_needed": (int(math.ceil(raw / events_per_file))
                             if events_per_file > 0 else -1),
            "rel_stat_err_at_target": (1.0 / math.sqrt(T)) if T > 0 else float("nan"),
        })
    out = pd.DataFrame(rows)
    if verbose:
        print(f"[cc] training-stats estimate (eff={eff:.4f} "
              f"[{eff_lo:.4f},{eff_hi:.4f}] from {n_pass}/{n_total}):")
        for _, r in out.iterrows():
            fn = f", ~{r['files_needed']} files" if r["files_needed"] >= 0 else ""
            print(f"[cc]   {int(r['target_selected']):>6d} selected -> "
                  f"~{int(r['raw_needed']):,} raw "
                  f"(band {int(r['raw_needed_lo']):,}–{int(r['raw_needed_hi']):,}){fn}")
    return out


def cutflow_markdown(cut_log: list, n_total: int, stream_name: str = "",
                     omitted: Optional[list] = None) -> str:
    """Render a sequential cut-flow as a markdown table.

    Columns mirror the standard HEP cut-flow style:
        Selection cut | Events | % total | % prev
      * Events  = events surviving AFTER this cut (cumulative)
      * % total = Events / n_total            (overall efficiency at this stage)
      * % prev  = Events / events_above       (this single cut's relative cost)
    """
    title = f"### Cut-flow: {stream_name}\n\n" if stream_name else ""
    lines = ["| Selection cut | Events | % total | % prev |",
             "|---|---:|---:|---:|",
             f"| Total | {n_total:,} | 100.0% | 100.0% |"]
    for name, before, after in cut_log:
        pct_total = 100.0 * after / n_total if n_total else 0.0
        pct_prev = 100.0 * after / before if before else 0.0
        lines.append(f"| {name} | {after:,} | {pct_total:.1f}% | {pct_prev:.1f}% |")
    md = title + "\n".join(lines) + "\n"
    if omitted:
        md += "\n*Omitted (branch absent):* " + "; ".join(
            o.split(":")[0] for o in omitted) + "\n"
    return md


# --------------------------------------------------------------------------- #
# Plotting (optional, Agg-backed)
# --------------------------------------------------------------------------- #
def _plot_cutflow(ctx: RunContext, cut_log: list, stream_name: str, n_total: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[cc] plot skipped (matplotlib unavailable): {e}")
        return None
    labels = ["start"] + [name for name, _b, _a in cut_log]
    counts = [n_total] + [a for _n, _b, a in cut_log]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(counts)), counts, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("events surviving")
    ax.set_title(ctx.title(f"CC cut-flow ({stream_name})"))
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=8)
    fig.tight_layout()
    out = ctx.plot_path(f"cc_{stream_name}_cutflow", "png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def _plot_nminus1(ctx: RunContext, n1: pd.DataFrame, stream_name: str):
    if n1.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[cc] plot skipped (matplotlib unavailable): {e}")
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(n1)), n1["marginal_removed"].to_numpy(), color="indianred")
    ax.set_yticks(range(len(n1)))
    ax.set_yticklabels(n1["cut_name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("events removed by this cut alone (N-1)")
    ax.set_title(ctx.title(f"CC N-1 marginal removal ({stream_name})"))
    fig.tight_layout()
    out = ctx.plot_path(f"cc_{stream_name}_nminus1", "png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def _persist_cc_tables(ctx: RunContext, table: pd.DataFrame, stream: str,
                       stream_cuts: List[Cut], cc_cuts: dict,
                       cut_log: list, omitted: list, verbose: bool = True):
    """
    Write everything needed to reconstruct this stream's selection later, offline.

    Stage 0 previously kept only the final `cc_pass` boolean, so the cut-flow lived
    in the log text and "which cut removed this event" was unrecoverable without
    re-reading every ROOT file (hours). Four artifacts fix that, all built from data
    already in memory, so this costs no extra reading:

      parquet/<run>__cc_<stream>_events.parquet
          The FULL event-level table: every scalar CC branch plus the derived
          _prompt_pe_sum / _mrd_ntracks_sum columns, one row per event, plus one
          `cut_<id>` boolean per cut in the stream and the final cc_pass. Any cut
          value, any cut combination, any other stream, and any N-1 study can be
          recomputed from this file in seconds. This is the no-information-loss
          guarantee — keep it even if the parquet of hits is deleted.
      csv/<run>__cc_<stream>_cutflow.csv / .md
          The sequential cut-flow table, ready to paste into a note or slide.
      csv/<run>__cc_<stream>_omitted.csv
          Only if a cut was skipped for a missing branch — so a silently shorter
          cut list can never masquerade as a real efficiency.

    The `cut_<id>` flags are evaluated INDEPENDENTLY (each cut against the full
    sample, not against the survivors of the previous cut), which is what makes
    arbitrary re-combination possible. The sequential cut-flow is recovered by
    AND-ing them in order; do not read a single column as a sequential efficiency.
    """
    out = table.copy()
    # Resolve IDs from the stream definition rather than zipping against stream_cuts:
    # _resolve_stream drops toggled-off cuts, so the two lists need not align.
    ids = [i for i in _resolve_stream_ids(stream, cc_cuts) if i in _CUT_REGISTRY]
    cols = set(out.columns)
    for cut_id in ids:
        cut = _CUT_REGISTRY[cut_id]
        if any(c not in cols for c in cut.requires):
            continue                      # omitted cut — no flag, reported separately
        try:
            out[f"cut_{cut_id}"] = np.asarray(cut.fn(out, cc_cuts), dtype=bool)
        except Exception as exc:           # never let bookkeeping kill the run
            if verbose:
                print(f"[cc] WARN could not store flag for cut {cut_id!r}: {exc}")

    ev_out = ctx.parquet_path(f"{ctx.run_name}__cc_{stream}_events")
    out.to_parquet(ev_out, index=False)

    n = len(table)
    cf = pd.DataFrame(
        [{"cut": name, "events_before": before, "events_after": after,
          "pct_of_total": round(100.0 * after / n, 4) if n else 0.0,
          "pct_of_previous": round(100.0 * after / before, 4) if before else 0.0}
         for name, before, after in cut_log])
    cf_out = ctx.csv_path(f"{ctx.run_name}__cc_{stream}_cutflow")
    cf.to_csv(cf_out, index=False)
    md_out = ctx.csv_dir / f"{ctx.run_name}__cc_{stream}_cutflow.md"
    md_out.write_text(cutflow_markdown(cut_log, n, stream, omitted))

    if omitted:
        om_out = ctx.csv_path(f"{ctx.run_name}__cc_{stream}_omitted")
        pd.DataFrame({"stream": stream, "omitted_cut": omitted}).to_csv(om_out,
                                                                       index=False)
        if verbose:
            print(f"[cc] [{stream}] WARNING {len(omitted)} cut(s) omitted -> {om_out}")

    if verbose:
        npass = int(table["cc_pass"].sum()) if "cc_pass" in table.columns else -1
        print(f"[cc] [{stream}] persisted event table ({n} events, {npass} passing, "
              f"{len([c for c in out.columns if c.startswith('cut_')])} per-cut flags) "
              f"-> {ev_out}")
        print(f"[cc] [{stream}] persisted cut-flow -> {cf_out}")
        print(f"[cc] [{stream}] persisted cut-flow (markdown) -> {md_out}")


def _resolve_stream_ids(stream_name: str, cc_cuts: dict) -> List[str]:
    """The ordered cut IDs for a stream (config override wins over the builtin)."""
    return list((cc_cuts.get("cc_streams", {}) or {}).get(stream_name)
                or _STREAMS.get(stream_name, []))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_cc_table(ctx: RunContext, tree_name: str = "Event",
                   max_events: Optional[int] = None,
                   verbose: bool = True,
                   stream: Optional[str] = None,
                   root_files: Optional[Sequence[Path]] = None,
                   save_tables: bool = False) -> pd.DataFrame:
    """
    Read CC branches from all input ROOT files, compute cc_pass per event,
    and return a global event-level table (used by processor.py).

    stream: named selection stream (default: reads ctx.cuts['cc_stream'],
            falls back to 'cc0pi_legacy' for backward compatibility).
    root_files: optional pre-resolved, tree-validated file list. processor.run passes
            its own list so both see exactly the same files (see _build_stream_table).
    save_tables: also persist the event table, the cut-flow (CSV + markdown) and the
            per-cut pass flags for this stream — see _persist_cc_tables. Without this
            the cut-flow exists only as log text and is lost with the log.
    """
    cc_cuts = _merged_cuts(ctx)
    if root_files is None:
        root_files = resolve_inputs(ctx.inputs["root_files"])
    if max_events is None and ctx.cuts:
        max_events = ctx.cuts.get("max_events")
    if stream is None:
        stream = (ctx.cuts or {}).get("cc_stream", "cc0pi_legacy")

    if stream == "cc0pi_legacy":
        if verbose:
            print(f"[cc] computing cc0pi_legacy CC selection over {len(root_files)} file(s)")
        frames = []
        event_id_offset = 0
        for rp in root_files:
            df = _read_event_table(rp, tree_name, max_events, verbose)
            df = _compute_cc_pass(df, cc_cuts, verbose)
            df = df.reset_index(drop=True)
            df["eventID"] = np.arange(len(df)) + event_id_offset
            event_id_offset += len(df)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # --- non-legacy stream: use _build_stream_table + _compute_pass ---
    if verbose:
        print(f"[cc] computing stream '{stream}' CC selection over {len(root_files)} file(s)")
    table, stream_cuts, _ = _build_stream_table(
        ctx, stream, cc_cuts, tree_name, max_events, verbose, root_files=root_files,
        augment_all=save_tables)
    table, _log, _omitted = _compute_pass(table, stream_cuts, cc_cuts, verbose, stream)
    if save_tables:
        _persist_cc_tables(ctx, table, stream, stream_cuts, cc_cuts,
                           _log, _omitted, verbose)
    return table


def _build_cc_table_legacy(ctx: RunContext, tree_name: str = "Event",
                            max_events: Optional[int] = None,
                            verbose: bool = True) -> pd.DataFrame:
    """Internal: original cc0pi_legacy loop (kept for reference)."""
    cc_cuts = _merged_cuts(ctx)
    root_files = resolve_inputs(ctx.inputs["root_files"])
    if max_events is None and ctx.cuts:
        max_events = ctx.cuts.get("max_events")

    if verbose:
        print(f"[cc] computing legacy CC selection over {len(root_files)} file(s)")

    frames = []
    event_id_offset = 0
    for rp in root_files:
        df = _read_event_table(rp, tree_name, max_events, verbose)
        df = _compute_cc_pass(df, cc_cuts, verbose)
        df = df.reset_index(drop=True)
        df["eventID"] = np.arange(len(df)) + event_id_offset
        event_id_offset += len(df)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_stream_table(ctx: RunContext, stream_name: str, cc_cuts: dict,
                        tree_name: str, max_events: Optional[int],
                        verbose: bool,
                        root_files: Optional[Sequence[Path]] = None,
                        augment_all: bool = False,
                        ) -> Tuple[pd.DataFrame, List[Cut], int]:
    """Read all files, augment vectors, and return (table, stream_cuts, events_per_file).

    `root_files` lets the caller supply an already-resolved (and tree-validated) file
    list. The CC table and the hit loop in processor.run MUST see the same files, or
    the per-file cc_pass merge silently drops events, so processor passes its list in
    rather than letting this re-resolve the glob independently.
    """
    if root_files is None:
        # Same guard processor.run uses: a zero-key file left by an interrupted grid
        # job (productionv3 ANNIEEvent_cc_neutrino_v3_24.root) otherwise aborts the
        # whole read with a bare KeyError. When the caller supplies root_files it has
        # already filtered, and re-filtering would re-open 498 files for nothing.
        root_files, _ = filter_files_with_tree(
            resolve_inputs(ctx.inputs["root_files"]), tree_name, verbose)
    stream_ids = (cc_cuts.get("cc_streams", {}).get(stream_name)
                  or _STREAMS.get(stream_name, []))
    # When the table will be persisted, derive EVERY vector-reduced column, not just
    # the ones this stream cuts on — otherwise the saved table can only re-derive its
    # own stream (a truth-tag table with no _prompt_pe_sum cannot reproduce a promptPE
    # cut, which defeats the point of saving it).
    aug_ids = list(stream_ids) + (["__all__"] if augment_all else [])
    # propagate the residual prompt window for the prompt_pe augmentation
    prompt_win = float(((ctx.extra.get("residual", {}) if ctx.extra else {})
                        or {}).get("prompt_window_ns", PROMPT_WINDOW_NS))
    _augment_vector_columns._prompt_win = prompt_win  # type: ignore[attr-defined]

    frames = []
    event_id_offset = 0
    first_file_events = 0
    for k, rp in enumerate(root_files):
        df = _read_event_table(rp, tree_name, max_events, verbose)
        df = _augment_origin_columns(df, cc_cuts)
        df = _augment_vector_columns(rp, df, cc_cuts, aug_ids, tree_name,
                                     max_events, verbose)
        df = df.reset_index(drop=True)
        df["eventID"] = np.arange(len(df)) + event_id_offset
        event_id_offset += len(df)
        if k == 0:
            first_file_events = len(df)
        frames.append(df)

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _origin_offset_report(table, verbose)
    stream_cuts = _resolve_stream(stream_name, cc_cuts)
    return table, stream_cuts, first_file_events


def run_streams(ctx: RunContext, stream_names: Sequence[str],
                tree_name: str = "Event", max_events: Optional[int] = None,
                do_nminus1: bool = False, do_plot: bool = False,
                do_trainstats: bool = False,
                train_targets: Optional[Sequence[int]] = None,
                mrd_source: Optional[str] = None,
                verbose: bool = True) -> List[Path]:
    """Run one or more named selection streams, writing per-stream outputs."""
    cc_cuts = _merged_cuts(ctx)
    if mrd_source:
        cc_cuts["mrd_source"] = mrd_source
    if max_events is None and ctx.cuts:
        max_events = ctx.cuts.get("max_events")
    targets = (train_targets or cc_cuts.get("train_targets")
               or [1000, 5000, 10000, 50000])

    if "all" in stream_names:
        known = set(_STREAMS) | set(cc_cuts.get("cc_streams", {}) or {})
        stream_names = sorted(known)

    summary_rows = []
    written: List[Path] = []
    for stream in stream_names:
        table, stream_cuts, evt_per_file = _build_stream_table(
            ctx, stream, cc_cuts, tree_name, max_events, verbose)
        n = len(table)
        table, cut_log, omitted = _compute_pass(
            table, stream_cuts, cc_cuts, verbose, stream)
        npass = int(table["cc_pass"].sum()) if n else 0

        ev_out = ctx.parquet_path(f"{ctx.run_name}__cc_{stream}_events")
        table.to_parquet(ev_out, index=False); written.append(ev_out)

        cf = pd.DataFrame([{"stream": stream, "step": i, "cut_name": nm,
                            "before": b, "after": a,
                            "frac_kept": (a / b) if b else float("nan"),
                            "pct_total": (100.0 * a / n) if n else float("nan"),
                            "pct_prev": (100.0 * a / b) if b else float("nan")}
                           for i, (nm, b, a) in enumerate(cut_log)])
        cf_out = ctx.csv_path(f"{ctx.run_name}__cc_{stream}_cutflow")
        cf.to_csv(cf_out, index=False); written.append(cf_out)

        # Markdown cut-flow table (Selection cut | Events | % total | % prev).
        md = cutflow_markdown(cut_log, n, stream, omitted)
        md_out = ctx.csv_dir / f"{ctx.run_name}__cc_{stream}_cutflow.md"
        md_out.write_text(md); written.append(md_out)

        if omitted:
            om_out = ctx.csv_path(f"{ctx.run_name}__cc_{stream}_omitted")
            pd.DataFrame({"stream": stream, "omitted_cut": omitted}).to_csv(om_out, index=False)
            written.append(om_out)

        if do_nminus1:
            n1 = nminus1_table(table, stream_cuts, cc_cuts, stream)
            n1_out = ctx.csv_path(f"{ctx.run_name}__cc_{stream}_nminus1")
            n1.to_csv(n1_out, index=False); written.append(n1_out)
            if do_plot:
                p = _plot_nminus1(ctx, n1, stream)
                if p: written.append(p)

        if do_plot:
            p = _plot_cutflow(ctx, cut_log, stream, n)
            if p: written.append(p)

        if do_trainstats:
            eff = (npass / n) if n else 0.0
            ts = estimate_training_stats(eff, npass, n, targets,
                                         events_per_file=evt_per_file, verbose=verbose)
            ts.insert(0, "stream", stream)
            ts_out = ctx.csv_path(f"{ctx.run_name}__cc_{stream}_trainstats")
            ts.to_csv(ts_out, index=False); written.append(ts_out)

        summary_rows.append({"run_name": ctx.run_name, "stream": stream,
                             "n_events": n, "n_cc_pass": npass,
                             "cc_pass_frac": (npass / n) if n else 0.0,
                             "n_omitted_cuts": len(omitted)})
        if verbose:
            print(f"[cc] [{stream}] wrote {n} events ({npass} passing) -> {ev_out}")

    summary = pd.DataFrame(summary_rows)
    sum_out = ctx.csv_path(f"{ctx.run_name}__cc_summary")
    summary.to_csv(sum_out, index=False); written.append(sum_out)
    if verbose:
        print(f"[cc] summary -> {sum_out}")
    return written


def run(ctx: RunContext, tree_name: str = "Event",
        max_events: Optional[int] = None, verbose: bool = True) -> Path:
    """
    Legacy entry point: build the CC0pi event table and write
        <parquet_dir>/<run_name>__cc_events.parquet
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
          cc_only: false           # keep only cc_pass==1 events (default false)
          prompt_window_ns: 0      # drop hits with t <= this (default 0 = no cut)

    Defaults are off because AmBe calibration (the primary use of this codebase)
    has no CC events and no prompt-muon window to remove. The CC-neutrino neutron
    search config opts in by setting both values in `configs/cc_neutrino_optics.yaml`.

    The prompt-window cut is purely time-based (no truth / BackTracker needed),
    so it is directly applicable to real data. OPTICS and cluster_features call
    this right after loading the pulses parquet.
    """
    block = (ctx.extra.get("residual", {}) if hasattr(ctx, "extra") else {}) or {}
    cc_only       = bool(block.get("cc_only", False))
    prompt_win_ns = float(block.get("prompt_window_ns", 0.0))

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
    p.add_argument("--stream", action="append", default=None,
                   help="selection stream (repeatable); use 'all' for every stream. "
                        "Default: cc0pi_legacy")
    p.add_argument("--nminus1", action="store_true", help="write N-1 per-cut impact CSV")
    p.add_argument("--plot", action="store_true", help="write cut-flow / N-1 bar plots")
    p.add_argument("--trainstats", action="store_true",
                   help="write raw-MC-stats-needed estimate per stream")
    p.add_argument("--train-targets", default=None,
                   help="comma-separated target selected-sample sizes")
    p.add_argument("--mrd-source", choices=["cluster", "tracks", "coinc"], default=None,
                   help="branch defining an MRD-tagged event (100%% efficiency)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(list(argv) if argv else [])

    streams = args.stream or list(
        (ctx.cuts.get("cc_cuts", {}) or {}).get("streams", ["cc0pi_legacy"]))
    targets = ([int(x) for x in args.train_targets.split(",")]
               if args.train_targets else None)
    run_streams(ctx, stream_names=streams, tree_name=args.tree,
                max_events=args.max_events, do_nminus1=args.nminus1,
                do_plot=args.plot, do_trainstats=args.trainstats,
                train_targets=targets, mrd_source=args.mrd_source,
                verbose=not args.quiet)
