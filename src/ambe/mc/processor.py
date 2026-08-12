"""
MC processor: ROOT -> per-hit Parquet + ClusterFinder sidecar.

Every detector hit (hitChankey / hitT) is included, not just those that
BackTracker could trace.  For each hit we attempt a truth match into the
DirectParent arrays using:
  - chankey identity
  - time proximity: DirectParent_HitTime ≈ hitT + offset
    where offset (~14-16 ns) is the PMT SPE waveform peak delay
    (p1*exp(-p2²) - T0Offset*2ns from PMTWaveformSim lognormal model),
    computed per-event from chankeys that appear in both arrays.

Unmatched hits are assigned truth_class=0 and flagged with is_untraced=True.
Two failure sub-modes are tracked in match_failure_reason:
  "no_entry"      – chankey has no DirectParent record at all (most likely
                    sub-threshold or masked PMT; probable dark noise, but a
                    physics photon that fell below the ADC threshold can also
                    land here — not confirmed dark noise)
  "time_mismatch" – a DirectParent entry exists for the chankey but the best
                    time match exceeds MATCH_TOL (likely a merged-pulse physics
                    hit whose MCHit arrival time doesn't align with the ADCPulse
                    peak; is_darknoise is carried from the nearest DP entry)
                    
This matters for OPTICS training: conflating the two failure modes would label
merged-pulse physics hits as dark noise, corrupting the training set.

Class code conventions (from BackTracker.cpp):
   0 dark noise  ·  1 primary neutron  ·  2 secondary n<-p  ·
   3 secondary n<-n  ·  4 secondary n<-other  ·  -5 non-neutron physics bg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from ..context import RunContext
from ..io import resolve_inputs, filter_files_with_tree


# Branches needed from the full-hit list
HIT_BRANCHES = [
    "eventNumber",
    "hitChankey", "hitT", "hitX", "hitY", "hitZ",
    "hitPE",   # charge per hit — needed for cluster feature extraction
]

# Branches needed for truth lookup (all optional — graceful fallback if absent)
DP_BRANCHES = [
    "DirectParent_PMTID", "DirectParent_HitTime",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
    "DirectParent_NeutronAncestorTrackID", "DirectParent_NeutronAncestorPDG",
    "DirectParent_ImmediateAncestorClass", "DirectParent_ImmediateAncestorPDG",
    "DirectParent_ImmediateAncestorTrackID",
    # Full ancestry chain per hit. Flattened CSR-style on purpose (a
    # vector<vector<int>> would need a custom ROOT dictionary):
    #   hit j's chain = LineagePDG[sum(Depth[:j]) : +Depth[j]], nearest-ancestor first.
    # Only present in MC generated with WCSim >= ac64522 (/WCSimIO/SaveTracksOnDemand);
    # every branch below is optional and falls back to a sentinel if absent, so old
    # productionv1/v2 ntuples still process.
    "DirectParent_LineagePDG", "DirectParent_LineageDepth",
    "DirectParent_LineageStatus", "DirectParent_RootAncestorPDG",
]

# LineageStatus codes (BackTracker): 1 reached a generator primary ·
# 0 truncated at an unsaved track · -1 cycle or depth cap · -5 dark noise.
LINEAGE_STATUS_LABELS = {
    1: "complete", 0: "truncated", -1: "cycle_or_cap", -5: "dark_noise",
}

# PDG -> short name, for the human-readable lineage_chain column.
PDG_SHORT = {
    11: "e-", -11: "e+", 22: "gamma", 13: "mu-", -13: "mu+",
    111: "pi0", 211: "pi+", -211: "pi-", 2112: "n", 2212: "p", -2212: "pbar",
    321: "K+", -321: "K-", 130: "K0L", 310: "K0S", 311: "K0", -311: "K0bar",
    12: "nu_e", -12: "nubar_e", 14: "nu_mu", -14: "nubar_mu",
    3122: "Lambda", 3222: "Sigma+", 3112: "Sigma-", 221: "eta", 113: "rho0",
    1000010020: "d", 1000010030: "t", 1000020030: "He3", 1000020040: "alpha",
    1000060120: "C12", 1000070140: "N14", 1000080160: "O16",
}

# The e+- carrier-skip set. Cherenkov/ionisation electrons are ubiquitous and
# uninformative on their own, so ImmediateAncestor skips through them. PDG 22 is
# deliberately NOT in this set -- "a gamma did this" is real information.
CARRIER_PDGS = frozenset({11, -11})

# The EM-mediator set used by origin_pdg/origin_class (see first_non_em_pdg).
# ImmediateAncestor deliberately stops at a gamma, which is the right call for
# "did a gamma make this light" but hides WHAT made the gamma: a capture gamma, a
# pi0 decay gamma and a proton-induced de-excitation gamma all read as class 5.
# origin_* walks one step further and skips gammas too, so the reported ancestor
# is the first hadron/muon/neutron in the chain. Both are kept: bg_class answers
# "what emitted the light", origin_class answers "what physics put it there".
EM_PDGS = frozenset({11, -11, 22})

MAX_CHAIN_LABEL = 6   # truncate lineage_chain beyond this many generations


def pdg_short(pdg: int) -> str:
    """Short particle name for a PDG code; nuclear codes render as nuc(Z,A)."""
    pdg = int(pdg)
    if pdg in PDG_SHORT:
        return PDG_SHORT[pdg]
    if pdg > 1000000000:                      # nuclear code 10LZZZAAAI
        return f"nuc({(pdg // 10000) % 1000},{(pdg // 10) % 1000})"
    return str(pdg)


def lineage_chain_str(pdgs, maxlen: int = MAX_CHAIN_LABEL) -> str:
    """
    Render a chain as 'e- <- gamma <- pi0' (nearest ancestor first).

    A string column keeps __pulses.parquet free of nested types while still
    allowing a direct value_counts() over chain patterns.
    """
    if len(pdgs) == 0:
        return ""
    s = " <- ".join(pdg_short(p) for p in pdgs[:maxlen])
    return s + " <- ..." if len(pdgs) > maxlen else s

# ImmediateAncestorClass codes (from BackTracker.cpp ClassifyBackgroundPDG):
#   0 dark noise · 1 neutron · 2 muon · 3 charged pion · 4 proton ·
#   5 photon · 6 kaon · 7 electron/positron · 8 other · -5 untraced
BG_CLASS_LABELS = {
    -5: "untraced", 0: "dark_noise", 1: "neutron", 2: "muon", 3: "charged_pion",
    4: "proton", 5: "photon", 6: "kaon", 7: "electron_positron", 8: "other",
}


# origin_class shares BG_CLASS_LABELS' codes but NOT its -5 label. A -5 here means
# the chain is complete and contains no non-EM ancestor at all (e.g. "e- <- gamma"
# where the gamma IS the generator primary) — that is a physics statement, not a
# tracing failure, and reading it as "untraced" would be wrong.
ORIGIN_CLASS_LABELS = {**BG_CLASS_LABELS, -5: "pure_em_or_untraced"}


def classify_pdg(pdg: int) -> int:
    """
    PDG -> BG_CLASS_LABELS code.

    Mirrors BackTracker::ClassifyBackgroundPDG
    (EB_BC_TA/UserTools/BackTracker/BackTracker.cpp:479-491) so origin_class and
    bg_class share one coding scheme and can be tabulated side by side. Any PDG
    the C++ side would call "other" (8) lands on 8 here too.
    """
    pdg = int(pdg)
    if pdg == 2112:
        return 1
    if abs(pdg) == 13:
        return 2
    if abs(pdg) == 211:
        return 3
    if pdg == 2212:
        return 4
    if pdg == 22:
        return 5
    if abs(pdg) in (321, 311):
        return 6
    if abs(pdg) == 11:
        return 7
    if pdg == -5:
        return -5
    return 8


def first_non_em_pdg(chain) -> Tuple[int, int]:
    """
    First ancestor in `chain` that is neither e+- nor gamma.

    `chain` is nearest-ancestor-first (see _decode_lineage_chains). Returns
    (origin_pdg, n_em_steps_skipped). An all-EM or empty chain returns (-5, len)
    -- -5 is the same "untraced/unavailable" sentinel used by bg_class, so the
    two columns share a missing-value convention.
    """
    for k, p in enumerate(chain):
        if int(p) not in EM_PDGS:
            return int(p), k
    return -5, len(chain)

CLUSTER_BRANCHES = [
    "eventNumber",
    "clusterTime", "clusterPE", "clusterHits",
    "Cluster_HitT", "Cluster_HitX", "Cluster_HitY", "Cluster_HitZ",
]

NEUTRON_CLASSES = {1, 2, 3, 4}
DEFAULT_DP_OFFSET = 15.0   # ns fallback if offset can't be computed
MATCH_TOL        = 8.0     # ns half-window for truth matching


# --------------------------------------------------------------------------- #
# Truth-matching helpers
# --------------------------------------------------------------------------- #

def _compute_offset(hit_ck, hit_t, dp_ck, dp_t) -> float:
    """
    Estimate DirectParent_HitTime - hitT per event.

    The offset is the PMT SPE waveform peak delay: p1*exp(-p2²) - T0Offset*2ns
    from PMTWaveformSim (~15.9 ns for typical ANNIE PMTs).  It is NOT the
    Cherenkov photon travel time, which is already embedded in hitT.

    Uses chankeys that appear exactly once in each array for an unambiguous match.
    Falls back to DEFAULT_DP_OFFSET when fewer than 3 pairs exist or the computed
    median is outside the physically plausible range [10, 25] ns.
    """
    shared = set(hit_ck) & set(dp_ck)
    offsets = []
    for ck in shared:
        ht = hit_t[hit_ck == ck]
        dt = dp_t[dp_ck == ck]
        if len(ht) == 1 and len(dt) == 1:
            offsets.append(float(dt[0] - ht[0]))
    if len(offsets) < 3:
        return DEFAULT_DP_OFFSET
    med = float(np.median(offsets))
    return med if 10.0 <= med <= 25.0 else DEFAULT_DP_OFFSET


def _build_dp_lookup(dp_ck, dp_t, dp_class, dp_dn, dp_tid, dp_pdg,
                      dp_bg_cls, dp_bg_pdg, dp_bg_tid,
                      dp_root_pdg=None, dp_lin_status=None, dp_lin_chain=None) -> dict:
    """
    Index DirectParent entries by chankey for O(1) per-hit lookup.
    Returns dict: chankey -> list of
        (dp_time, class, is_darknoise, trackID, pdg, bg_class, bg_pdg, bg_trackID,
         root_pdg, lineage_status, lineage_chain)
    bg_class/bg_pdg/bg_trackID come from DirectParent_ImmediateAncestor* — the
    species-general one-step-skip ancestor (any particle, not just neutrons).

    root_pdg / lineage_status / lineage_chain come from the full-chain branches and
    are optional: pass None for any of them (old ntuples) and a sentinel is stored.
    lineage_chain is the already-decoded list of PDG codes for that entry, nearest
    ancestor first -- decoding happens once per event in the caller, not per hit.
    """
    n = len(dp_ck)
    if dp_root_pdg is None:
        dp_root_pdg = np.full(n, -5, dtype=int)
    if dp_lin_status is None:
        dp_lin_status = np.full(n, -5, dtype=int)
    if dp_lin_chain is None:
        dp_lin_chain = [()] * n

    lookup: dict[int, list] = {}
    for ck, t, cls, dn, tid, pdg, bg_cls, bg_pdg, bg_tid, rpdg, lstat, lchain in zip(
        dp_ck, dp_t, dp_class, dp_dn, dp_tid, dp_pdg, dp_bg_cls, dp_bg_pdg, dp_bg_tid,
        dp_root_pdg, dp_lin_status, dp_lin_chain
    ):
        ck_i = int(ck)
        lookup.setdefault(ck_i, []).append(
            (float(t), int(cls), int(dn), int(tid), int(pdg),
             int(bg_cls), int(bg_pdg), int(bg_tid),
             int(rpdg), int(lstat), lchain)
        )
    return lookup


def _decode_lineage_chains(dp_lin_pdg, dp_lin_depth, n_entries):
    """
    Decode the flattened CSR lineage arrays into one tuple of PDG codes per DP entry.

    hit j's chain = LineagePDG[sum(Depth[:j]) : +Depth[j]], nearest ancestor first.
    Returns None if the arrays are absent or the invariant len(PDG) == sum(Depth)
    does not hold -- a silent misdecode here would mis-attribute every hit's
    ancestry, so it is better to drop the columns than to guess.
    """
    if dp_lin_pdg is None or dp_lin_depth is None:
        return None
    if len(dp_lin_depth) != n_entries:
        return None
    if int(np.sum(dp_lin_depth)) != len(dp_lin_pdg):
        return None
    offs = np.concatenate([[0], np.cumsum(dp_lin_depth)])
    return [tuple(int(p) for p in dp_lin_pdg[offs[j]:offs[j + 1]])
            for j in range(n_entries)]


def _match_hit(ck: int, t: float, dp_lookup: dict, offset: float,
               tol: float = MATCH_TOL):
    """
    Match one hit (chankey, hitT) to its DirectParent truth entry.

    Returns (truth_class, is_darknoise, ancestor_trackID, ancestor_pdg,
             is_untraced, match_failure_reason, bg_class, bg_pdg, bg_trackID,
             root_pdg, lineage_status, lineage_chain).

    match_failure_reason values:
      None            – successful match
      "no_entry"      – chankey absent from DirectParent arrays entirely
                        (sub-threshold / masked PMT; very likely dark noise)
      "time_mismatch" – entry exists but best |dt| > tol; is_darknoise is
                        carried from the nearest DP entry rather than assumed
                        (merged-pulse physics hits fall here)

    bg_class/bg_pdg/bg_trackID are withheld (-5, matching the ROOT branch's own
    "untraced" sentinel) whenever the match itself is unreliable or absent —
    same conservative rule already applied to the neutron-scheme fields. The
    lineage fields (root_pdg, lineage_status, lineage_chain) are withheld under
    exactly the same rule, so a hit never carries a chain the match cannot support.
    """
    entries = dp_lookup.get(ck)
    if not entries:
        return 0, 1, -1, -1, True, "no_entry", -5, -5, -5, -5, -5, ()

    t_shifted = t + offset
    best_dt, best = min(
        ((abs(e[0] - t_shifted), e) for e in entries),
        key=lambda x: x[0],
    )
    if best_dt > tol:
        # Carry is_darknoise from the nearest DP entry instead of assuming 1
        nearest_dn = best[2]
        return 0, nearest_dn, -1, -1, True, "time_mismatch", -5, -5, -5, -5, -5, ()

    (_, cls, dn, tid, pdg, bg_cls, bg_pdg, bg_tid,
     root_pdg, lin_status, lin_chain) = best
    return (cls, dn, tid, pdg, False, None, bg_cls, bg_pdg, bg_tid,
            root_pdg, lin_status, lin_chain)


# --------------------------------------------------------------------------- #
# Per-file processing
# --------------------------------------------------------------------------- #

def _process_single_file(root_path: Path, tree_name: str, verbose: bool,
                          max_events: Optional[int] = None):
    """Read one ROOT file; return (pulse_df, cluster_df)."""
    with uproot.open(str(root_path)) as f:
        if tree_name not in f:
            raise KeyError(f"Tree {tree_name!r} not in {root_path}")
        t = f[tree_name]
        available = set(t.keys())

        wanted_hit     = [b for b in HIT_BRANCHES  if b in available]
        wanted_dp      = [b for b in DP_BRANCHES   if b in available]
        wanted_cluster = [b for b in CLUSTER_BRANCHES if b in available]

        missing_hit = set(HIT_BRANCHES) - available
        if missing_hit and verbose:
            print(f"[mc.processor] WARN {root_path.name}: missing hit branches: {missing_hit}")

        has_dp = len(wanted_dp) > 0
        if not has_dp and verbose:
            print(f"[mc.processor] WARN {root_path.name}: no DirectParent branches — "
                  "all hits will be labeled as untraced dark noise")

        hit_arr     = t.arrays(wanted_hit,     library="ak")
        dp_arr      = t.arrays(wanted_dp,      library="ak") if has_dp else None
        cluster_arr = t.arrays(wanted_cluster, library="ak") if wanted_cluster else None

    n = len(hit_arr)
    if max_events is not None:
        n = min(n, max_events)
    pulse_rows, cluster_rows = [], []

    for i in range(n):
        event_id = int(hit_arr["eventNumber"][i])

        hit_ck = np.array(ak.to_list(hit_arr["hitChankey"][i]), dtype=int)
        hit_t  = np.array(ak.to_list(hit_arr["hitT"][i]),       dtype=float)
        hit_x  = np.array(ak.to_list(hit_arr["hitX"][i]),       dtype=float)
        hit_y  = np.array(ak.to_list(hit_arr["hitY"][i]),       dtype=float)
        hit_z  = np.array(ak.to_list(hit_arr["hitZ"][i]),       dtype=float)
        hit_pe = np.array(ak.to_list(hit_arr["hitPE"][i]),       dtype=float) \
                 if "hitPE" in hit_arr.fields else np.ones(len(hit_ck), dtype=float)

        # Build truth lookup for this event
        if has_dp and "DirectParent_PMTID" in wanted_dp:
            dp_ck  = np.array(ak.to_list(dp_arr["DirectParent_PMTID"][i]),               dtype=int)
            dp_t   = np.array(ak.to_list(dp_arr["DirectParent_HitTime"][i]),             dtype=float)
            dp_cls = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorClass"][i]),dtype=int) \
                     if "DirectParent_NeutronAncestorClass" in wanted_dp \
                     else np.zeros(len(dp_ck), dtype=int)
            dp_dn  = np.array(ak.to_list(dp_arr["DirectParent_IsDarknoise"][i]),         dtype=int) \
                     if "DirectParent_IsDarknoise" in wanted_dp \
                     else np.ones(len(dp_ck), dtype=int)
            dp_tid = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorTrackID"][i]), dtype=int) \
                     if "DirectParent_NeutronAncestorTrackID" in wanted_dp \
                     else np.full(len(dp_ck), -1, dtype=int)
            dp_pdg = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorPDG"][i]),  dtype=int) \
                     if "DirectParent_NeutronAncestorPDG" in wanted_dp \
                     else np.full(len(dp_ck), -1, dtype=int)
            dp_bg_cls = np.array(ak.to_list(dp_arr["DirectParent_ImmediateAncestorClass"][i]), dtype=int) \
                     if "DirectParent_ImmediateAncestorClass" in wanted_dp \
                     else np.full(len(dp_ck), -5, dtype=int)
            dp_bg_pdg = np.array(ak.to_list(dp_arr["DirectParent_ImmediateAncestorPDG"][i]), dtype=int) \
                     if "DirectParent_ImmediateAncestorPDG" in wanted_dp \
                     else np.full(len(dp_ck), -5, dtype=int)
            dp_bg_tid = np.array(ak.to_list(dp_arr["DirectParent_ImmediateAncestorTrackID"][i]), dtype=int) \
                     if "DirectParent_ImmediateAncestorTrackID" in wanted_dp \
                     else np.full(len(dp_ck), -5, dtype=int)

            # Full-chain branches (WCSim >= ac64522 only; absent on old MC).
            dp_root_pdg = np.array(ak.to_list(dp_arr["DirectParent_RootAncestorPDG"][i]), dtype=int) \
                     if "DirectParent_RootAncestorPDG" in wanted_dp else None
            dp_lin_stat = np.array(ak.to_list(dp_arr["DirectParent_LineageStatus"][i]), dtype=int) \
                     if "DirectParent_LineageStatus" in wanted_dp else None
            dp_lin_pdg  = np.array(ak.to_list(dp_arr["DirectParent_LineagePDG"][i]), dtype=int) \
                     if "DirectParent_LineagePDG" in wanted_dp else None
            dp_lin_dep  = np.array(ak.to_list(dp_arr["DirectParent_LineageDepth"][i]), dtype=int) \
                     if "DirectParent_LineageDepth" in wanted_dp else None
            dp_chains   = _decode_lineage_chains(dp_lin_pdg, dp_lin_dep, len(dp_ck))

            offset    = _compute_offset(hit_ck, hit_t, dp_ck, dp_t)
            dp_lookup = _build_dp_lookup(dp_ck, dp_t, dp_cls, dp_dn, dp_tid, dp_pdg,
                                          dp_bg_cls, dp_bg_pdg, dp_bg_tid,
                                          dp_root_pdg, dp_lin_stat, dp_chains)
        else:
            offset    = 0.0
            dp_lookup = {}

        # Emit one row per detector hit
        for j in range(len(hit_ck)):
            ck = int(hit_ck[j])
            t_hit = float(hit_t[j])

            (cls, dn, tid, pdg, untraced, fail_reason, bg_cls, bg_pdg, bg_tid,
             root_pdg, lin_status, lin_chain) = _match_hit(
                ck, t_hit, dp_lookup, offset
            )

            # First non-EM ancestor: resolves the "everything is a photon"
            # degeneracy of bg_class without over-shooting to the generator
            # primary the way root_pdg does.
            origin_pdg, n_em_skipped = first_non_em_pdg(lin_chain)

            pulse_rows.append({
                "eventID":              event_id,
                "eventNumber":          event_id,   # local (per-file) event number, for CC-table join
                "pmtID":                ck,
                "t":                    t_hit,
                "x":                    float(hit_x[j]),
                "y":                    float(hit_y[j]),
                "z":                    float(hit_z[j]),
                "pe":                   float(hit_pe[j]),
                "truth_class":          cls,
                "is_darknoise":         dn,
                "is_neutron":           int(cls in NEUTRON_CLASSES),
                "is_untraced":          int(untraced),
                "match_failure_reason": fail_reason,
                "ancestor_trackID":     tid,
                "ancestor_pdg":         pdg,
                # Species-general immediate-ancestor tag (any particle, not just
                # neutrons) — see DirectParent_ImmediateAncestor* / BG_CLASS_LABELS.
                "bg_class":             bg_cls,
                "bg_pdg":               bg_pdg,
                "bg_trackID":           bg_tid,
                # Full ancestry chain (WCSim >= ac64522). root_pdg is the top of the
                # chain and is only trustworthy where lineage_status == 1.
                # lineage_chain is a readable 'e- <- gamma <- pi0' string so chain
                # patterns can be value_counts()'d directly; lineage_first_pdg is the
                # immediate emitter, and is_gamma_mediated flags a gamma anywhere in
                # the chain (the component old productionv1/v2 cannot see at all).
                "root_pdg":             root_pdg,
                "lineage_status":       lin_status,
                "lineage_depth":        len(lin_chain),
                "lineage_chain":        lineage_chain_str(lin_chain),
                "lineage_first_pdg":    int(lin_chain[0]) if lin_chain else -5,
                "is_gamma_mediated":    int(22 in lin_chain),
                "is_carrier_mediated":  int(bool(lin_chain) and lin_chain[0] in CARRIER_PDGS),
                # First non-EM ancestor (e+- AND gamma skipped) — the particle
                # that actually put the energy there. Only trustworthy where
                # lineage_status == 1; -5 means all-EM chain or no chain.
                "origin_pdg":           origin_pdg,
                "origin_class":         classify_pdg(origin_pdg),
                "n_em_steps_skipped":   n_em_skipped,
            })

        # ClusterFinder sidecar
        if cluster_arr is not None:
            ctimes = np.asarray(cluster_arr["clusterTime"][i]) if "clusterTime" in wanted_cluster else []
            cpes   = np.asarray(cluster_arr["clusterPE"][i])   if "clusterPE"   in wanted_cluster else [0.0] * len(ctimes)
            chits  = np.asarray(cluster_arr["clusterHits"][i]) if "clusterHits" in wanted_cluster else [0]   * len(ctimes)
            for cidx, (ct, cpe, ch) in enumerate(zip(ctimes, cpes, chits)):
                cluster_rows.append({
                    "eventID":     event_id,
                    "cluster_idx": cidx,
                    "clusterTime": float(ct),
                    "clusterPE":   float(cpe),
                    "clusterHits": int(ch),
                })

    return pd.DataFrame(pulse_rows), pd.DataFrame(cluster_rows)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run(ctx: RunContext, tree_name: str = "Event", verbose: bool = True,
        max_events: Optional[int] = None) -> tuple[Path, Path]:
    """
    Process all ROOT files in ctx.inputs['root_files'].
    Writes two Parquet files:
        <parquet_dir>/<run_name>__pulses.parquet
        <parquet_dir>/<run_name>__clusterfinder.parquet

    Streaming write: each file's hits are merged with its CC-pass flags and
    appended to the output parquet immediately, so peak RAM is bounded to one
    file at a time (~90 hits/event × 4000 events × 200 B ≈ 70 MB per file)
    rather than the full dataset (~15 GB for 400 files).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root_files = resolve_inputs(ctx.inputs["root_files"])
    # Drop files with no usable tree BEFORE any reading, and use the same validated
    # list for the CC table below — otherwise a single empty grid output aborts the
    # whole pass, and a per-reader mismatch would desync the cc_pass merge.
    n_globbed = len(root_files)
    root_files, skipped_files = filter_files_with_tree(root_files, tree_name, verbose)
    if max_events is None:
        max_events = ctx.cuts.get("max_events") if ctx.cuts else None
    if verbose:
        print(f"[mc.processor] processing {len(root_files)} file(s)"
              + (f" of {n_globbed} globbed ({len(skipped_files)} skipped)"
                 if skipped_files else "")
              + (f"  (max_events={max_events})" if max_events else ""))

    # Build CC table once upfront (scalar branches only — fits in RAM easily).
    from . import cc_selection
    cc_stream = (ctx.cuts or {}).get("cc_stream", "cc0pi_legacy")
    if verbose:
        print(f"[mc.processor] CC stream: {cc_stream!r}")
    cc_tbl = cc_selection.build_cc_table(ctx, tree_name=tree_name,
                                         max_events=max_events, verbose=verbose,
                                         stream=cc_stream, root_files=root_files,
                                         save_tables=True)
    cc_merge = None
    if len(cc_tbl) and "cc_pass" in cc_tbl.columns:
        cc_cols = ["_source_file", "eventNumber", "cc_pass"]
        # origin_in_tank rides along UNRENAMED (not as event_origin_in_tank): it is
        # the signal/background label input consumed downstream by cluster_features
        # and mva_analysis, which look it up by this exact name.
        for _oc in ("origin_in_tank", "origin_in_fv"):
            if _oc in cc_tbl.columns:
                cc_cols.append(_oc)
        extra = [c for c in ("trueCC", "truePrimaryPdg", "trueMultiRing",
                             "trueNeutrons", "trueMuonEnergy") if c in cc_tbl.columns]
        cc_merge = cc_tbl[cc_cols + extra].rename(
            columns={c: f"event_{c}" for c in extra})

    pulses_path   = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")

    pulse_writer   = None
    cluster_writer = None
    event_id_offset = 0
    total_hits = 0
    total_clusters = 0
    # Accumulators for the end-of-run summary histogram (scalars only — tiny).
    tc_counts: dict = {}
    bg_counts: dict = {}
    lin_status_counts: dict = {}
    lin_chain_counts: dict = {}
    root_counts: dict = {}
    origin_counts: dict = {}          # origin_class over class--5 background hits
    n_bkg_hits = n_gamma_med = n_carrier_med = 0
    n_untraced = n_no_entry = n_time_mm = 0

    for k, rp in enumerate(root_files):
        pf, cf = _process_single_file(rp, tree_name, verbose, max_events=max_events)
        pf["_source_file"] = rp.name
        cf["_source_file"] = rp.name

        if event_id_offset > 0:
            pf["eventID"] = pf["eventID"] + event_id_offset
            if len(cf):
                cf["eventID"] = cf["eventID"] + event_id_offset
            if verbose:
                print(f"[mc.processor]   eventID offset +{event_id_offset} applied to {rp.name}")
        if len(pf):
            event_id_offset = int(pf["eventID"].max()) + 1

        # Merge CC flags for this file's hits.
        if cc_merge is not None:
            file_cc = cc_merge[cc_merge["_source_file"] == rp.name]
            pf = pf.merge(file_cc, on=["_source_file", "eventNumber"], how="left")
            pf["cc_pass"] = pf["cc_pass"].fillna(False).astype(bool)
            # An unmatched event means we know nothing about its origin; -1 keeps it
            # distinguishable from a genuine "outside the tank" (0), which is a
            # positive statement and gets labeled background.
            for _oc in ("origin_in_tank", "origin_in_fv"):
                if _oc in pf.columns:
                    pf[_oc] = pf[_oc].fillna(-1).astype(int)

        # Accumulate summary stats (scalars — negligible memory).
        for cls, cnt in pf["truth_class"].value_counts().items():
            tc_counts[cls] = tc_counts.get(cls, 0) + int(cnt)
        for cls, cnt in pf["bg_class"].value_counts().items():
            bg_counts[cls] = bg_counts.get(cls, 0) + int(cnt)
        # Lineage accumulators, restricted to the class--5 background (truth_class
        # == -5): that is the population the chains were added to characterise.
        # Read truth_class FIRST -- bg_class/lineage never report "neutron", because
        # a capture chain terminates at the gamma.
        if "lineage_status" in pf.columns:
            for st, cnt in pf["lineage_status"].value_counts().items():
                lin_status_counts[st] = lin_status_counts.get(st, 0) + int(cnt)
            bkg = pf[pf["truth_class"] == -5]
            if len(bkg):
                for ch, cnt in bkg["lineage_chain"].value_counts().items():
                    lin_chain_counts[ch] = lin_chain_counts.get(ch, 0) + int(cnt)
                for rp_, cnt in bkg["root_pdg"].value_counts().items():
                    root_counts[rp_] = root_counts.get(rp_, 0) + int(cnt)
                if "origin_class" in bkg.columns:
                    for oc, cnt in bkg["origin_class"].value_counts().items():
                        origin_counts[oc] = origin_counts.get(oc, 0) + int(cnt)
                n_bkg_hits    += len(bkg)
                n_gamma_med   += int(bkg["is_gamma_mediated"].sum())
                n_carrier_med += int(bkg["is_carrier_mediated"].sum())
        n_untraced  += int(pf["is_untraced"].sum())
        n_no_entry  += int((pf["match_failure_reason"] == "no_entry").sum())
        n_time_mm   += int((pf["match_failure_reason"] == "time_mismatch").sum())
        total_hits  += len(pf)
        total_clusters += len(cf)

        # Append to parquet — schema inferred from first batch, enforced after.
        pulse_tbl = pa.Table.from_pandas(pf, preserve_index=False)
        if pulse_writer is None:
            pulse_writer = pq.ParquetWriter(pulses_path, pulse_tbl.schema)
        pulse_writer.write_table(pulse_tbl)

        if len(cf):
            cf_tbl = pa.Table.from_pandas(cf, preserve_index=False)
            if cluster_writer is None:
                cluster_writer = pq.ParquetWriter(clusters_path, cf_tbl.schema)
            cluster_writer.write_table(cf_tbl)

    if pulse_writer:
        pulse_writer.close()
    if cluster_writer:
        cluster_writer.close()
    # Ensure empty output files exist even when no data was written.
    if not pulses_path.exists():
        pd.DataFrame().to_parquet(pulses_path, index=False)
    if not clusters_path.exists():
        pd.DataFrame().to_parquet(clusters_path, index=False)

    if verbose:
        print(f"[mc.processor] wrote {total_hits:>8d} hits     -> {pulses_path}")
        print(f"[mc.processor] wrote {total_clusters:>8d} clusters -> {clusters_path}")
        if total_hits:
            print("[mc.processor] truth_class histogram (all detector hits):")
            for cls in sorted(tc_counts):
                print(f"  {cls:>3d}    {tc_counts[cls]:>8d}")
            print(f"[mc.processor] untraced total : "
                  f"{n_untraced} ({100*n_untraced/total_hits:.1f}%)")
            print(f"[mc.processor]   no_entry     : "
                  f"{n_no_entry} ({100*n_no_entry/total_hits:.1f}%)  "
                  f"[sub-threshold / masked PMT]")
            print(f"[mc.processor]   time_mismatch: "
                  f"{n_time_mm} ({100*n_time_mm/total_hits:.1f}%)  "
                  f"[merged-pulse / offset error]")
            print("[mc.processor] bg_class histogram (species-general immediate "
                  "ancestor, all detector hits):")
            for cls in sorted(bg_counts, key=lambda c: -bg_counts[c]):
                label = BG_CLASS_LABELS.get(cls, "?")
                print(f"  {cls:>3d} {label:<18s} {bg_counts[cls]:>8d}  "
                      f"({100*bg_counts[cls]/total_hits:.1f}%)")

            # ---- full-chain lineage summary (absent on pre-ac64522 MC) ----
            if lin_status_counts:
                print("[mc.processor] lineage_status histogram (all detector hits):")
                for st in sorted(lin_status_counts, key=lambda s: -lin_status_counts[s]):
                    label = LINEAGE_STATUS_LABELS.get(st, "?")
                    print(f"  {st:>3d} {label:<14s} {lin_status_counts[st]:>8d}  "
                          f"({100*lin_status_counts[st]/total_hits:.1f}%)")
                n_trunc = lin_status_counts.get(0, 0)
                if n_trunc:
                    print(f"[mc.processor]   WARNING {n_trunc} hits have TRUNCATED "
                          f"chains (status 0) — save-on-demand did not take for those; "
                          f"expect 0 on WCSim >= ac64522 output")
            if n_bkg_hits:
                print(f"[mc.processor] class--5 BACKGROUND lineage "
                      f"({n_bkg_hits} hits, truth_class == -5):")
                print(f"[mc.processor]   gamma-mediated  : {n_gamma_med:>8d}  "
                      f"({100*n_gamma_med/n_bkg_hits:.1f}%)  "
                      f"[exactly 0 on productionv1/v2]")
                print(f"[mc.processor]   via e+- carrier : {n_carrier_med:>8d}  "
                      f"({100*n_carrier_med/n_bkg_hits:.1f}%)  "
                      f"[the rest radiate directly]")
                if origin_counts:
                    print("[mc.processor]   origin_class (first non-EM ancestor — "
                          "e+- AND gamma skipped):")
                    for oc in sorted(origin_counts, key=lambda c: -origin_counts[c]):
                        label = ORIGIN_CLASS_LABELS.get(oc, "?")
                        print(f"      {oc:>3d} {label:<18s} {origin_counts[oc]:>8d}  "
                              f"({100*origin_counts[oc]/n_bkg_hits:.1f}%)")
                print("[mc.processor]   root ancestor (top of chain):")
                for rp_ in sorted(root_counts, key=lambda r: -root_counts[r])[:8]:
                    print(f"      {pdg_short(rp_):<10s} {root_counts[rp_]:>8d}  "
                          f"({100*root_counts[rp_]/n_bkg_hits:.1f}%)")
                print("[mc.processor]   most common chains (nearest ancestor first):")
                for ch in sorted(lin_chain_counts, key=lambda c: -lin_chain_counts[c])[:10]:
                    print(f"      {100*lin_chain_counts[ch]/n_bkg_hits:5.1f}%  "
                          f"{lin_chain_counts[ch]:>8d}   {ch}")

    return pulses_path, clusters_path


def rederive_cc_pass(ctx: RunContext, source_pulses: Path,
                     tree_name: str = "Event",
                     max_events: Optional[int] = None,
                     verbose: bool = True):
    """
    Build this run's __pulses.parquet from an EXISTING one by recomputing cc_pass
    for this config's cc_stream. The per-hit content is copied unchanged.

    Why this exists: the per-hit pass is the expensive part of Stage 0 (hours for a
    few-hundred-file sample) and it is completely independent of the CC selection --
    every detector hit is written regardless of cc_pass, and the residual/CC filter
    is applied downstream in cluster_features. So two selection streamlines over the
    same input files differ ONLY in the cc_pass column (and the event_* truth columns
    merged alongside it). Re-reading every ROOT file per streamline would repeat hours
    of identical work.

    The CC table itself IS rebuilt here, because different cuts need different
    branches -- that read is unavoidable, but it is a small fraction of Stage 0.

    Guardrails: the source parquet must carry _source_file and eventNumber (the merge
    keys) and must have been produced from the same input glob. Both are checked; a
    mismatch raises rather than silently producing a mis-merged sample.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_pulses = Path(source_pulses)
    if not source_pulses.exists():
        raise FileNotFoundError(f"--reuse-hits source not found: {source_pulses}")

    dest = ctx.parquet_path(f"{ctx.run_name}__pulses")
    if dest.resolve() == source_pulses.resolve():
        raise ValueError(
            f"--reuse-hits source and this run's pulses path are the same file "
            f"({dest}); that would overwrite the source mid-read. Use a different "
            f"run_name.")

    root_files = resolve_inputs(ctx.inputs["root_files"])
    root_files, skipped = filter_files_with_tree(root_files, tree_name, verbose)
    if max_events is None:
        max_events = ctx.cuts.get("max_events") if ctx.cuts else None

    from . import cc_selection
    cc_stream = (ctx.cuts or {}).get("cc_stream", "cc0pi_legacy")
    if verbose:
        print(f"[mc.processor] REUSING hits from {source_pulses}")
        print(f"[mc.processor] recomputing cc_pass for stream {cc_stream!r} "
              f"over {len(root_files)} file(s)")
    cc_tbl = cc_selection.build_cc_table(ctx, tree_name=tree_name,
                                         max_events=max_events, verbose=verbose,
                                         stream=cc_stream, root_files=root_files,
                                         save_tables=True)
    if not len(cc_tbl) or "cc_pass" not in cc_tbl.columns:
        raise RuntimeError(f"CC table for stream {cc_stream!r} is empty — cannot "
                           f"rederive cc_pass")

    extra = [c for c in ("trueCC", "truePrimaryPdg", "trueMultiRing",
                         "trueNeutrons", "trueMuonEnergy") if c in cc_tbl.columns]
    base_cols = ["_source_file", "eventNumber", "cc_pass"]
    # origin_in_tank is stream-independent (it describes the interaction, not the
    # selection), but it is rebuilt rather than carried over so a source parquet
    # produced before this column existed still gains it here.
    for _oc in ("origin_in_tank", "origin_in_fv"):
        if _oc in cc_tbl.columns:
            base_cols.append(_oc)
    cc_merge = cc_tbl[base_cols + extra].rename(
        columns={c: f"event_{c}" for c in extra})
    # Columns the previous run's merge added — dropped so they are rebuilt, never
    # carried over stale from the other streamline.
    stale = ["cc_pass", "origin_in_tank", "origin_in_fv"] + [f"event_{c}" for c in extra]

    pf = pq.ParquetFile(str(source_pulses))
    avail = set(pf.schema_arrow.names)
    for req in ("_source_file", "eventNumber"):
        if req not in avail:
            raise KeyError(f"source parquet lacks {req!r} — cannot merge cc_pass "
                           f"onto it (was it produced by a different processor "
                           f"version?)")
    src_files = set(cc_merge["_source_file"].unique())

    writer = None
    n_rows = n_pass = 0
    seen_files: set = set()
    n_groups = pf.num_row_groups
    for i in range(n_groups):
        d = pf.read_row_group(i).to_pandas()
        if not len(d):
            continue
        seen_files.update(d["_source_file"].unique().tolist())
        d = d.drop(columns=[c for c in stale if c in d.columns])
        d = d.merge(cc_merge, on=["_source_file", "eventNumber"], how="left")
        d["cc_pass"] = d["cc_pass"].fillna(False).astype(bool)
        for _oc in ("origin_in_tank", "origin_in_fv"):
            if _oc in d.columns:
                d[_oc] = d[_oc].fillna(-1).astype(int)
        n_rows += len(d)
        n_pass += int(d["cc_pass"].sum())
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dest, tbl.schema)
        writer.write_table(tbl)
        if verbose:
            print(f"[mc.processor]  rederive chunk {i+1}/{n_groups}  "
                  f"({100*(i+1)/n_groups:.0f}%)  rows={n_rows}", flush=True)
    if writer:
        writer.close()

    missing = seen_files - src_files
    if missing:
        raise RuntimeError(
            f"{len(missing)} source file(s) in the reused parquet have no CC-table "
            f"entry, so their hits would all be cc_pass=False: "
            f"{sorted(missing)[:5]}... The reused parquet was built from a different "
            f"input glob than this config declares.")

    # The ClusterFinder sidecar carries no cc_pass, so it is stream-independent.
    src_cf = source_pulses.parent / source_pulses.name.replace("__pulses",
                                                              "__clusterfinder")
    dest_cf = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if src_cf.exists() and not dest_cf.exists():
        dest_cf.symlink_to(src_cf)
        if verbose:
            print(f"[mc.processor] clusterfinder sidecar (stream-independent) "
                  f"symlinked -> {src_cf}")

    if verbose:
        print(f"[mc.processor] wrote {n_rows} hits     -> {dest}")
        print(f"[mc.processor] cc_pass=True on {n_pass}/{n_rows} hits "
              f"({100*n_pass/max(n_rows,1):.2f}%)")
    return dest, dest_cf


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc process")
    p.add_argument("--tree", default="Event")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--reuse-hits", default=None, metavar="PULSES_PARQUET",
                   help="Skip the per-hit ROOT pass: copy hits from this existing "
                        "__pulses.parquet and only recompute cc_pass for this "
                        "config's cc_stream. Use when running a second selection "
                        "streamline over input files already processed once.")
    args = p.parse_args(list(argv) if argv else [])
    if args.reuse_hits:
        rederive_cc_pass(ctx, Path(args.reuse_hits), tree_name=args.tree,
                         verbose=not args.quiet, max_events=args.max_events)
    else:
        run(ctx, tree_name=args.tree, verbose=not args.quiet,
            max_events=args.max_events)
