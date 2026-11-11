"""
Capture-matched truth for neutron clusters, and a clustering benchmark built on it.

    ambe mc capture --config <cfg.yaml> [--max-files N] [--jobs J] [--vertex] [--tag T]

Why this exists
---------------
The production signal label (`cluster_features._truth_label_cluster`) asks whether one
neutron *trackID* owns >= 50 % of a cluster's hits.  On CCinc v3 that rejects 61 % of
the background class as "merged neutrons", with the trackIDs sharing a cluster only
2.9 ns apart (REPORT_ccinc_v3_cluster_provenance_timewindows.md §5, §8).  BackTracker
assigns each hit to the *nearest* PDG-2112 ancestor, and Geant4-HP inelastic steps
start new neutron trackIDs, so one physical capture can carry several trackIDs.

The Super-K NTag answer (Han 2023, §3.3.2) is to label by the *capture*: a candidate
is signal if a simulated capture lies within 50 ns of it, whatever the trackID.  WCSim
writes every capture (`trueNeutCapTime/VtxX/Y/Z/Nucleus/...`) into the ANNIEEvent tree,
so this needs no reprocessing.  The capture's parent trackID is NOT written, which is
fine: matching is by time, as in the thesis.

What it produces (under the run's parquet/ and csv/ dirs, suffixed by --tag)
---------------------------------------------------------------------------
  __true_captures.parquet         one row per true capture (cached ROOT read)
  __capture_truth_clusters.parquet  one row per cluster per method, with the trackID
                                  breakdown (reproduces the production label) and the
                                  capture breakdown side by side
  __capture_truth_captures.parquet  one row per capture: light seen, clusters that own it
  __capture_truth_summary.csv     per-method recall / split / merge / purity table
  __capture_truth_merged_breakdown.csv  the decisive table: trackID-"merged" clusters
                                  split by how many true captures they really contain
  __capture_truth_tids_per_capture.csv  neutron trackIDs carried by one capture's light
  __capture_truth_vertex.csv      (--vertex) vertex resolution against the true capture
  __capture_truth_offset.json     hit-time minus capture-time calibration

Coordinates.  Pulse x, y, z are tank-centred metres.  trueNeutCapVtx is WCSim world
cm, so capture = (x, y + 14.46, z - 168.1) / 100.  trueVtx (the neutrino vertex) is
already tank-centred cm.  Both were fitted against the hit centroids of single-capture
light before being hard-coded here.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..clustering.peak_finder import SOL_WATER
from ..context import RunContext
from . import clock as mc_clock

NEUTRON_PDG = 2112
CAP_WORLD_OFFSET_CM = np.array([0.0, -14.46, 168.1])   # world - tank-centred frame
GD_NUCLEI = {1000641550, 1000641560, 1000641570, 1000641580}   # 155-158Gd ground states
H_NUCLEI = {1000010020}                                        # deuteron: n + p

# Every cluster hit lands in exactly one of these (they sum to n_hits):
#   cap_dom       light from the cluster's dominant TRUE capture
#   cap_other     light from any other true capture
#   neutron_nocap neutron-lineage hit outside every capture window (scatter/inelastic
#                 light, or a capture outside the analysed window)
#   bg_<species>  non-neutron physics, by processor.py's origin_pdg: the first ancestor
#                 that is neither e+- nor gamma; pure_em = a pure e/gamma chain (-5)
#   darknoise, untraced
BG_SPECIES = {13: "muon", -13: "muon", 211: "pion_charged", -211: "pion_charged",
              111: "pion_zero", 2212: "proton", -5: "pure_em"}
COMP_CATS = (["cap_dom", "cap_other", "neutron_nocap"]
             + [f"bg_{s}" for s in dict.fromkeys(BG_SPECIES.values())]
             + ["bg_other", "darknoise", "untraced"])

DEFAULTS = dict(
    match_ns=50.0,           # thesis TMATCHWINDOW: cluster median vs capture time
    hit_window_ns=(-20.0, 80.0),  # a neutron-lineage hit belongs to a capture in this window
    min_cap_hits=3,          # a capture "contributes" to a cluster with >= this many hits
    min_visible_hits=5,      # a capture counts in the recall denominator above this
    nms_ns=50.0,             # time-NMS merge for the *_nms methods
    peak=dict(twidth_ns=20.0, nhitsth=6, t_min_sep_ns=50.0),
    object_gap_ns=200.0,     # particle mode: split one particle's light at gaps longer than this
    nu_smear_m=0.30,         # per-axis smear of the true neutrino vertex (INIT stand-in)
    window_ns=None,          # [t_lo, t_hi] for captures; None -> from the residual filter
    lowpe_cut=1.0,           # unmatched hits below this PE are counted in comp_untraced_lowpe
    michel_min_delay_ns=100.0,  # --hit-window full: muon-child light this long after the muon is a Michel
    optics_scan=dict(min_samples=[4, 5, 6, 8, 10, 12], xi=[0.01, 0.02, 0.05, 0.1],
                     t_unit_ns=[10, 25, 50, 100], nms=[False, True]),
)


# --------------------------------------------------------------------------- #
# True captures
# --------------------------------------------------------------------------- #

def load_true_captures(root_files: Iterable[str], tree: str = "Event") -> pd.DataFrame:
    """One row per WCSim capture, keyed by (_source_file basename, eventNumber)."""
    import awkward as ak
    import uproot

    cols = ["eventNumber", "trueNeutCapTime", "trueNeutCapVtxX", "trueNeutCapVtxY",
            "trueNeutCapVtxZ", "trueNeutCapNucleus", "trueNeutCapE", "trueNeutCapGammas"]
    vtx = ["trueVtxX", "trueVtxY", "trueVtxZ"]
    out = []
    files = list(root_files)
    t0 = time.time()
    for k, f in enumerate(files):
        try:
            tr = uproot.open(f)[tree]
        except Exception as e:                       # empty / truncated file
            print(f"[capture] SKIP {Path(f).name}: {type(e).__name__}")
            continue
        have = set(tr.keys())
        if "trueNeutCapTime" not in have:
            print(f"[capture] SKIP {Path(f).name}: no trueNeutCap branches")
            continue
        a = tr.arrays([c for c in cols + vtx if c in have])
        n = ak.to_numpy(ak.num(a["trueNeutCapTime"]))
        evn = np.repeat(ak.to_numpy(a["eventNumber"]), n)
        df = pd.DataFrame(dict(
            _source_file=Path(f).name, eventNumber=evn.astype(int),
            cap_idx=np.concatenate([np.arange(m) for m in n]) if n.sum() else np.array([], int),
            cap_t=ak.to_numpy(ak.flatten(a["trueNeutCapTime"])).astype(float),
            cap_x=(ak.to_numpy(ak.flatten(a["trueNeutCapVtxX"])) - CAP_WORLD_OFFSET_CM[0]) / 100.0,
            cap_y=(ak.to_numpy(ak.flatten(a["trueNeutCapVtxY"])) - CAP_WORLD_OFFSET_CM[1]) / 100.0,
            cap_z=(ak.to_numpy(ak.flatten(a["trueNeutCapVtxZ"])) - CAP_WORLD_OFFSET_CM[2]) / 100.0,
            cap_nucleus=ak.to_numpy(ak.flatten(a["trueNeutCapNucleus"])).astype(np.int64),
            cap_E=ak.to_numpy(ak.flatten(a["trueNeutCapE"])).astype(float),
            cap_ngamma=ak.to_numpy(ak.flatten(a["trueNeutCapGammas"])).astype(int),
        ))
        if all(v in have for v in vtx):
            for v, c in zip(vtx, ["nu_x", "nu_y", "nu_z"]):
                df[c] = np.repeat(ak.to_numpy(a[v]).astype(float) / 100.0, n)
        out.append(df)
        if (k + 1) % 50 == 0:
            print(f"[capture] read {k + 1}/{len(files)} files ({time.time() - t0:.0f}s)")
    if not out:
        raise SystemExit("[capture] no capture branches found in any input file")
    caps = pd.concat(out, ignore_index=True)
    caps["cap_medium"] = np.where(caps.cap_nucleus.isin(GD_NUCLEI), "Gd",
                          np.where(caps.cap_nucleus.isin(H_NUCLEI), "H", "other"))
    return caps


def neutrino_vertices(root_files: Iterable[str], tree: str = "Event") -> pd.DataFrame:
    """Per-event true neutrino vertex (tank-centred m), for events without captures too."""
    import awkward as ak
    import uproot
    out = []
    for f in root_files:
        try:
            tr = uproot.open(f)[tree]
        except Exception:
            continue
        if "trueVtxX" not in tr.keys():
            continue
        a = tr.arrays(["eventNumber", "trueVtxX", "trueVtxY", "trueVtxZ"])
        out.append(pd.DataFrame(dict(
            _source_file=Path(f).name, eventNumber=ak.to_numpy(a["eventNumber"]).astype(int),
            nu_x=ak.to_numpy(a["trueVtxX"]) / 100.0, nu_y=ak.to_numpy(a["trueVtxY"]) / 100.0,
            nu_z=ak.to_numpy(a["trueVtxZ"]) / 100.0)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


_PORT5 = re.compile(r"port5_z(minus)?(\d+)")


def source_from_filename(name: str) -> Optional[np.ndarray]:
    """Port-5 AmBe source position in the tank-centred hit frame (m), or None.

    Port 5 is the central port; its z label is the height, i.e. hit-frame y.
    """
    m = _PORT5.search(str(name))
    if not m:
        return None
    z = float(m.group(2)) / 100.0
    return np.array([0.0, -z if m.group(1) else z, 0.0])


# --------------------------------------------------------------------------- #
# Hit -> capture assignment and per-event labelling
# --------------------------------------------------------------------------- #

def assign_hits_to_captures(df_ev: pd.DataFrame, cap_t: np.ndarray, offset_ns: float,
                            window=DEFAULTS["hit_window_ns"]) -> np.ndarray:
    """Capture index for each neutron-lineage hit whose time falls in that capture's
    window (nearest capture wins), else -1.  Non-neutron hits are always -1."""
    out = np.full(len(df_ev), -1, dtype=int)
    if not len(cap_t):
        return out
    is_n = (df_ev["ancestor_pdg"].to_numpy(int) == NEUTRON_PDG) & \
           (df_ev["ancestor_trackID"].to_numpy(int) >= 0)
    if not is_n.any():
        return out
    t = df_ev["t"].to_numpy(float)[is_n]
    d = t[:, None] - (cap_t[None, :] + offset_ns)
    j = np.argmin(np.abs(d), axis=1)
    dj = d[np.arange(len(t)), j]
    ok = (dj >= window[0]) & (dj <= window[1])
    idx = np.where(is_n)[0]
    out[idx[ok]] = j[ok]
    return out


def hit_categories(df_ev: pd.DataFrame, hit_cap: np.ndarray) -> np.ndarray:
    """Per-hit composition category (index into COMP_CATS), except cap_dom vs
    cap_other, which depends on the cluster and is resolved by the caller: every
    capture hit is returned as 'cap_dom' here."""
    n = len(df_ev)
    tid = df_ev["ancestor_trackID"].to_numpy(int)
    is_n = (df_ev["ancestor_pdg"].to_numpy(int) == NEUTRON_PDG) & (tid >= 0)
    is_dn = df_ev["truth_class"].to_numpy(int) == 0
    unt = (df_ev["is_untraced"].to_numpy(int) == 1 if "is_untraced" in df_ev.columns
           else np.zeros(n, bool))
    opdg = (df_ev["origin_pdg"].to_numpy(int) if "origin_pdg" in df_ev.columns
            else np.full(n, -999))
    idx = {c: i for i, c in enumerate(COMP_CATS)}
    cat = np.full(n, idx["bg_other"], dtype=int)
    for pdg, name in BG_SPECIES.items():
        cat[opdg == pdg] = idx[f"bg_{name}"]
    cat[unt] = idx["untraced"]
    cat[is_dn] = idx["darknoise"]
    cat[is_n] = idx["neutron_nocap"]
    cat[hit_cap >= 0] = idx["cap_dom"]
    return cat


# Truth objects: every physics hit belongs to exactly one, keyed by the particle that
# seeded its light (nearest neutron, else the first non-e/gamma ancestor) and split in
# time where that particle's light has a gap longer than object_gap_ns.
OBJ_CLASSES = ("n_capture", "n_other", "muon", "michel", "pion_charged", "pi0", "proton", "em",
               "other")
_KIND = {1: "neutron", 2: "muon", 3: "pion_charged", 4: "proton", 5: "em", 6: "other", 7: "pi0"}


def _source_keys(df_ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kind, source trackID, physics mask) per hit; kind codes as _KIND."""
    n = len(df_ev)
    col = lambda c, fill: (df_ev[c].to_numpy(int) if c in df_ev.columns else np.full(n, fill))
    tid, apdg = col("ancestor_trackID", -5), col("ancestor_pdg", -5)
    opdg, otid = col("origin_pdg", -999), col("origin_trackID", -5)
    is_dn = col("truth_class", 0) == 0
    unt = col("is_untraced", 0) == 1
    phys = ~(is_dn | unt)
    is_n = (apdg == NEUTRON_PDG) & (tid >= 0)
    kind = np.full(n, 6, dtype=int)
    kind[np.abs(opdg) == 13] = 2
    kind[np.abs(opdg) == 211] = 3
    kind[opdg == 111] = 7
    kind[opdg == 2212] = 4
    kind[opdg == -5] = 5
    kind[is_n] = 1
    src = np.where(is_n, tid, np.maximum(otid, 0))
    return kind, src, phys


def build_truth_objects(df_ev: pd.DataFrame, cap_t: np.ndarray, offset_ns: float,
                        cfg: dict, caps_in: Optional[pd.DataFrame] = None
                        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Object index per hit (-1 = dark noise / untraced) and one row per object."""
    n = len(df_ev)
    t = df_ev["t"].to_numpy(float)
    pe = df_ev["pe"].to_numpy(float)
    child = (df_ev["origin_child_trackID"].to_numpy(int)
             if "origin_child_trackID" in df_ev.columns else np.full(n, -5))
    kind, src, phys = _source_keys(df_ev)
    obj = np.full(n, -1, dtype=int)
    rows = []
    gap = float(cfg["object_gap_ns"])
    keys = np.unique(np.stack([kind[phys], src[phys]], axis=1), axis=0) if phys.any() \
        else np.empty((0, 2), int)
    for k, sid in keys:
        idx = np.where(phys & (kind == k) & (src == sid))[0]
        idx = idx[np.argsort(t[idx])]
        cuts = np.where(np.diff(t[idx]) > gap)[0] + 1
        for sub, part in enumerate(np.split(idx, cuts)):
            tm = float(np.median(t[part]))
            cap_j = -1
            if k == 1:
                d = np.abs(cap_t + offset_ns - tm) if len(cap_t) else np.array([])
                if len(d) and d.min() <= cfg["match_ns"]:
                    cap_j = int(np.argmin(d))
                cls = "n_capture" if cap_j >= 0 else "n_other"
            elif k == 2:
                late = (not cfg.get("_full_window")
                        or t[part[0]] - t[idx[0]] >= cfg["michel_min_delay_ns"])
                cls = "michel" if np.mean(child[part] < 0) < 0.5 and late else "muon"
            else:
                cls = _KIND[k] if _KIND[k] in OBJ_CLASSES else "other"
            oid = len(rows)
            obj[part] = oid
            r = dict(obj_id=oid, obj_class=cls, source_kind=_KIND[k], source_trackID=int(sid),
                     sub_idx=sub, n_hits=len(part), pe=float(pe[part].sum()),
                     t_first=float(t[part[0]]), t_med=tm, t_last=float(t[part[-1]]),
                     cap_j=cap_j)
            if cap_j >= 0 and caps_in is not None:
                cr = caps_in.iloc[cap_j]
                r.update(cap_idx=int(cr.cap_idx), cap_medium=cr.cap_medium,
                         dt_cap=tm - float(cr.cap_t) - offset_ns)
            rows.append(r)
    return obj, pd.DataFrame(rows)


def neutron_content(u, c, owner, is_nobj, tid_of, capj_of, cls_of, tmed_of, mh) -> dict:
    """How many distinct neutrons and captures one cluster holds (u, c: its objects and hit counts)."""
    if not len(u):
        return dict(n_neutrons=0, n_neutrons_3=0, n_captures=0, n_hits_other_neutrons=0,
                    dt_captures_min=np.nan)
    nm = is_nobj[u]
    per_tid = pd.Series(c[nm]).groupby(tid_of[u][nm]).sum()
    own_tid = tid_of[owner] if is_nobj[owner] else None
    capm = (cls_of[u] == "n_capture") & (capj_of[u] >= 0)
    cj, first = np.unique(capj_of[u][capm], return_index=True)
    tcap = np.sort(tmed_of[u][capm][first])
    return dict(n_neutrons=int(len(per_tid)), n_neutrons_3=int((per_tid >= mh).sum()),
                n_captures=int(len(cj)),
                n_hits_other_neutrons=int(per_tid.sum() - per_tid.get(own_tid, 0)),
                dt_captures_min=float(np.diff(tcap).min()) if len(tcap) > 1 else np.nan)


def match_clusters_to_objects(lbl: dict, obj: np.ndarray, objs: pd.DataFrame,
                              df_ev: pd.DataFrame, cfg: dict, owner_rule: str
                              ) -> tuple[list, dict]:
    """Per-cluster owner rows and per-object per-method ownership counts."""
    n_obj = len(objs)
    t = df_ev["t"].to_numpy(float)
    kind, _, phys = _source_keys(df_ev)
    is_dn = (df_ev["truth_class"].to_numpy(int) == 0) if "truth_class" in df_ev.columns \
        else np.zeros(len(df_ev), bool)
    unm = (df_ev["is_untraced"].to_numpy(int) == 1) if "is_untraced" in df_ev.columns \
        else np.zeros(len(df_ev), bool)
    pe = df_ev["pe"].to_numpy(float) if "pe" in df_ev.columns else np.ones(len(df_ev))
    nowf = _no_waveform(df_ev)
    ovl = (df_ev["is_overlay"].to_numpy(int) == 1) if "is_overlay" in df_ev.columns \
        else np.zeros(len(df_ev), bool)
    is_nobj = (objs.source_kind.to_numpy() == "neutron") if n_obj else np.array([], bool)
    tid_of = objs.source_trackID.to_numpy(int) if n_obj else np.array([], int)
    capj_of = objs.cap_j.to_numpy(int) if n_obj else np.array([], int)
    tmed_of = objs.t_med.to_numpy(float) if n_obj else np.array([], float)
    cls_of = objs.obj_class.to_numpy() if n_obj else np.array([])
    nh_of = objs.n_hits.to_numpy(int) if n_obj else np.array([], int)
    per_obj = {}
    rows = []
    mh = cfg["min_cap_hits"]
    obj_ok = obj >= 0
    for method, lb in lbl.items():
        owned = np.zeros(n_obj, int)
        best = np.zeros(n_obj, int)
        merged = np.zeros(n_obj, int)
        in_cl = obj_ok & (lb >= 0)
        # Where each object's hits and PE went: clusters it owns, any cluster, (rest) none.
        hits_clustered = np.bincount(obj[in_cl], minlength=n_obj)
        pe_clustered = np.bincount(obj[in_cl], weights=pe[in_cl], minlength=n_obj)
        hits_own = np.zeros(n_obj, int)
        pe_own = np.zeros(n_obj, float)
        for cid in [c for c in np.unique(lb) if c >= 0]:
            mask = lb == cid
            nh = int(mask.sum())
            ids = obj[mask]
            ids = ids[ids >= 0]
            comp = {f"comp_{c}": 0 for c in OBJ_CLASSES}
            owner, n_own, n_contrib = -1, 0, 0
            u = c = np.array([], int)
            if len(ids):
                u, c = np.unique(ids, return_counts=True)
                o = np.argsort(-c)
                u, c = u[o], c[o]
                owner, n_own = int(u[0]), int(c[0])
                n_contrib = int((c >= mh).sum())
                np.maximum.at(best, u, c)
                for uu, cc in zip(u, c):
                    comp[f"comp_{cls_of[uu]}"] += int(cc)
                    if uu != owner and cc >= mh:
                        merged[uu] += 1
            frac = n_own / max(nh, 1)
            tag = cls_of[owner] if owner >= 0 else "noise"
            if owner >= 0 and owner_rule == "majority50" and frac < 0.5:
                tag = "mixed"
            if owner >= 0 and n_own >= mh:
                owned[owner] += 1
            own_mask = mask & (obj == owner) if owner >= 0 else np.zeros_like(mask)
            if owner >= 0:
                hits_own[owner] += n_own
                pe_own[owner] += float(pe[own_mask].sum())
            comp["comp_darknoise"] = int((mask & is_dn & ~unm).sum())
            comp["comp_untraced"] = int((mask & unm).sum())
            assert sum(comp.values()) == nh
            nq = neutron_content(u, c, owner, is_nobj, tid_of, capj_of, cls_of, tmed_of, mh)
            low = mask & nowf & (pe < cfg["lowpe_cut"])
            nq["comp_untraced_lowpe"] = int(low.sum())
            nq["nowf_lowpe_owner"] = int((low & (obj == owner) & (obj >= 0)).sum())
            nq["nowf_lowpe_other_obj"] = int((low & (obj >= 0) & (obj != owner)).sum())
            nq["nowf_lowpe_darknoise"] = int((low & is_dn & ~unm).sum())
            nq["nowf_lowpe_untraced"] = int((low & unm).sum())
            nq["comp_overlay"] = int((mask & ovl).sum())
            pe_cl = float(pe[mask].sum())
            nq["pe_cluster"] = pe_cl
            nq["owner_pe_frac"] = float(pe[own_mask].sum()) / pe_cl if pe_cl > 0 else np.nan
            rows.append(dict(method=method, cluster_id=int(cid), n_hits=nh,
                             t_med=float(np.median(t[mask])), owner_obj=owner,
                             owner_class=cls_of[owner] if owner >= 0 else "noise",
                             tag=tag, owner_frac=frac, n_owner_hits=n_own,
                             owner_completeness=n_own / nh_of[owner] if owner >= 0 else np.nan,
                             n_objects_contrib=n_contrib, **comp, **nq))
        per_obj[method] = dict(owned=owned, best=best, merged=merged, hits_own=hits_own,
                               hits_clustered=hits_clustered, pe_own=pe_own, pe_clustered=pe_clustered)
    return rows, per_obj


def label_event(df_ev: pd.DataFrame, df_cl: Optional[pd.DataFrame], methods: list,
                feat: dict, cfg: dict, ref_m: Optional[np.ndarray]) -> dict:
    """Cluster labels per method for one event (same membership as production)."""
    from ..clustering.peak_finder import find_time_peaks, merge_clusters_in_time
    from .optics import run_optics_on_event, assign_clusterfinder_labels, _prefilter_hits

    t = df_ev["t"].to_numpy(float)
    lbl = {}
    need_opt = any(m in ("optics", "optics_nms") for m in methods)
    need_cf = any(m.startswith("clusterfinder") for m in methods)
    if need_opt:
        pref = float(feat.get("hit_prefilter_ns", 0.0))
        df_opt = _prefilter_hits(df_ev, df_cl, pref) if pref > 0 else df_ev
        lo = run_optics_on_event(df_opt, min_samples=int(feat.get("min_samples", 8)),
                                 xi=float(feat.get("xi", 0.10)),
                                 t_unit_ns=float(feat.get("t_unit_ns", 25.0)),
                                 source_pos_m=feat.get("_source_pos_m"))
        if len(df_opt) < len(df_ev):
            full = np.full(len(df_ev), -1, int)
            full[df_opt.index] = lo
            lo = full
        lbl["optics"] = lo
    if need_cf:
        lbl["clusterfinder"] = (assign_clusterfinder_labels(df_ev, df_cl)
                                if df_cl is not None and len(df_cl)
                                else np.full(len(df_ev), -1, int))
    for base in ("optics", "clusterfinder"):
        if f"{base}_nms" in methods:
            lbl[f"{base}_nms"] = merge_clusters_in_time(lbl[base], t, cfg["nms_ns"])
    xyz = df_ev[["x", "y", "z"]].to_numpy(float)
    pk = cfg["peak"]
    if "peak" in methods:
        lbl["peak"], _ = find_time_peaks(t, xyz=xyz, pmt=df_ev["pmtID"].to_numpy(int),
                                         pe=df_ev["pe"].to_numpy(float), ref_m=None, **pk)
    if "peak_tof" in methods and ref_m is not None:
        lbl["peak_tof"], _ = find_time_peaks(t, xyz=xyz, pmt=df_ev["pmtID"].to_numpy(int),
                                             pe=df_ev["pe"].to_numpy(float), ref_m=ref_m, **pk)
    for m in methods:
        op = parse_optics_scan(m)
        if op is None:
            continue
        pref = float(feat.get("hit_prefilter_ns", 0.0))
        df_opt = _prefilter_hits(df_ev, df_cl, pref) if pref > 0 else df_ev
        lo = run_optics_on_event(df_opt, min_samples=op["min_samples"], xi=op["xi"],
                                 t_unit_ns=op["t_unit_ns"], source_pos_m=feat.get("_source_pos_m"))
        if len(df_opt) < len(df_ev):
            full = np.full(len(df_ev), -1, int)
            full[df_opt.index] = lo
            lo = full
        lbl[m] = merge_clusters_in_time(lo, t, cfg["nms_ns"]) if op["nms"] else lo
    for m in methods:
        sp = parse_scan_method(m)
        if sp is None or (sp.pop("_tof") and ref_m is None):
            continue
        clean = sp.pop("_clean")
        if clean:
            sp.update((cfg.get("peak_scan") or {}).get("clean_params",
                                                       {"pe_max": 10.0, "deadtime_ns": 1000.0}))
        lbl[m], _ = find_time_peaks(t, xyz=xyz, pmt=df_ev["pmtID"].to_numpy(int),
                                    pe=df_ev["pe"].to_numpy(float),
                                    ref_m=ref_m if m.endswith("_tof") else None, **sp)
    return {m: lbl[m] for m in methods if m in lbl}


_SCAN = re.compile(r"^peak_tw(\d+(?:\.\d+)?)_n(\d+)_s(\d+(?:\.\d+)?)(_clean)?(_tof)?$")


def parse_scan_method(name: str) -> Optional[dict]:
    """'peak_tw20_n6_s50[_clean][_tof]' -> find_time_peaks kwargs (+ _clean/_tof flags)."""
    m = _SCAN.match(name)
    if not m:
        return None
    return dict(twidth_ns=float(m.group(1)), nhitsth=int(m.group(2)),
                t_min_sep_ns=float(m.group(3)), _clean=bool(m.group(4)), _tof=bool(m.group(5)))


_OSCAN = re.compile(r"^optics_ms(\d+)_xi(\d+(?:\.\d+)?)_tu(\d+(?:\.\d+)?)(_nms)?$")


def parse_optics_scan(name: str) -> Optional[dict]:
    """'optics_ms8_xi0.02_tu25[_nms]' -> run_optics_on_event kwargs (+ nms flag)."""
    m = _OSCAN.match(name)
    if not m:
        return None
    return dict(min_samples=int(m.group(1)), xi=float(m.group(2)), t_unit_ns=float(m.group(3)),
                nms=bool(m.group(4)))


def optics_scan_methods(cfg: dict) -> list:
    sc = cfg["optics_scan"]
    return [f"optics_ms{ms}_xi{xi:g}_tu{tu:g}{'_nms' if nms else ''}"
            for ms in sc["min_samples"] for xi in sc["xi"] for tu in sc["t_unit_ns"]
            for nms in sc["nms"]]


def scan_methods(cfg: dict) -> list:
    sc = cfg.get("peak_scan", {}) or {}
    out = []
    for tw in sc.get("twidth_ns", [10, 14, 20, 30, 40]):
        for nh in sc.get("nhitsth", [4, 5, 6, 7, 8]):
            for sep in sc.get("t_min_sep_ns", [50, 100]):
                for clean in sc.get("clean", [False, True]):
                    for tof in sc.get("tof", [False, True]):
                        out.append(f"peak_tw{tw:g}_n{nh}_s{sep:g}"
                                   f"{'_clean' if clean else ''}{'_tof' if tof else ''}")
    return out


CLUSTER_FEATURES = ("pe_total", "pe_balance", "charge_bal_legacy", "sigma_t", "sigma_t_mad",
                    "spatial_rms", "d_wall", "n_hits_early")


def scan_table(cl: pd.DataFrame, per_obj: pd.DataFrame, methods: list) -> pd.DataFrame:
    """One row per scanned method: capture-object recall plus capture-cluster purity."""
    ob = per_obj[per_obj.method.isin(methods) & (per_obj.obj_class == "n_capture")]
    t = ob.groupby("method")[["found", "split", "merged_into_other", "hit_completeness"]].mean()
    c = cl[cl.method.isin(methods)]
    n_ev = max(cl.eventID.nunique(), 1)
    cap = c[c.tag == "n_capture"]
    lab = (cap.n_hits - cap.comp_untraced).clip(lower=1)
    g = cap.assign(pur=cap.n_owner_hits / cap.n_hits, pur_lab=cap.n_owner_hits / lab).groupby("method")
    t = t.join(g.size().rename("capture_clusters"))
    t = t.join(g.pur.mean().rename("purity")).join(g.pur_lab.mean().rename("purity_labelled"))
    t = t.join(g.n_neutrons.apply(lambda x: (x <= 1).mean()).rename("one_neutron"))
    t = t.join(g.n_captures.apply(lambda x: (x >= 2).mean()).rename("two_plus_captures"))
    dn = cap.groupby("method").apply(lambda d: d.comp_darknoise.sum() / max(d.n_hits.sum(), 1))
    t = t.join(dn.rename("darknoise_share"))
    nc = (c.assign(nc=~c.tag.eq("n_capture")).groupby("method").nc.sum() / n_ev)
    return t.join(nc.rename("non_capture_clusters_per_event"))


def _mad(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if len(x) > 1 else np.nan


# --------------------------------------------------------------------------- #
# Worker: one row group
# --------------------------------------------------------------------------- #

_G: dict = {}      # per-process state, filled before the Pool forks


class _SliceIndex:
    """Read-only stand-in for {key: group.reset_index(drop=True) for key, group in df.groupby(keys)}:
    one sorted frame plus row ranges, sliced on .get (--low-memory)."""

    def __init__(self, df: pd.DataFrame, keys: list, extra: Optional[dict] = None):
        df = df.sort_values(keys, kind="stable").reset_index(drop=True)
        cols = [df[k].to_numpy() for k in keys]
        n = len(df)
        if n:
            brk = np.zeros(n, bool)
            brk[0] = True
            for c in cols:
                brk[1:] |= c[1:] != c[:-1]
            starts = np.flatnonzero(brk)
            stops = np.append(starts[1:], n)
            if len(keys) == 1:
                ks = cols[0][starts].tolist()
            else:
                ks = list(zip(*(c[starts].tolist() for c in cols)))
            self._pos = dict(zip(ks, zip(starts.tolist(), stops.tolist())))
        else:
            self._pos = {}
        self._df = df
        self._extra = dict(extra or {})

    def get(self, key, default=None):
        if key in self._extra:
            return self._extra[key]
        p = self._pos.get(key)
        return default if p is None else self._df.iloc[p[0]:p[1]].reset_index(drop=True)

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v


PULSE_COLS = ["eventID", "eventNumber", "_source_file", "pmtID", "t", "x", "y", "z", "pe",
              "truth_class", "is_neutron", "is_untraced", "ancestor_trackID", "ancestor_pdg",
              "origin_pdg", "cc_pass", "origin_in_tank", "origin_trackID",
              "origin_child_trackID", "wf_matched"]


def _no_waveform(df: pd.DataFrame) -> np.ndarray:
    """MCHit pulses with no waveform pulse (data would not record them)."""
    if "wf_matched" in df.columns:
        return df["wf_matched"].to_numpy(int) == 0
    if "is_untraced" in df.columns:          # pre-wf_matched pulses: the two coincide
        return df["is_untraced"].to_numpy(int) == 1
    return np.zeros(len(df), bool)


def load_overlay(path: str) -> dict:
    """Random-trigger hits (ambe data randoms) as {trigger index: DataFrame}."""
    ov = pd.read_parquet(path)
    return {i: g[["pmtID", "t", "x", "y", "z", "pe"]].reset_index(drop=True)
            for i, (_, g) in enumerate(ov.groupby("trigger", sort=True))}


def overlay_pick(seed: int, source_file: str, event_number: int, n_trig: int) -> int:
    """Index (into load_overlay's sorted triggers) of the trigger overlaid on one MC event."""
    import zlib
    rng = np.random.default_rng([seed, zlib.crc32(str(source_file).encode()), int(event_number)])
    return int(rng.integers(n_trig))


def add_overlay(chunk: pd.DataFrame, overlay: dict, t_min: float, seed: int,
                empty_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Drop MC dark noise and add one random trigger's hits to every event (seeded per event).
    empty_events (eventID, eventNumber, _source_file) are events with no MC hits left; they
    get the overlay too."""
    mc_dn = (chunk.truth_class == 0) & (chunk.is_untraced == 0)
    chunk = chunk[~mc_dn].assign(is_overlay=0)
    ev_cols = [c for c in ("eventID", "eventNumber", "_source_file", "cc_pass", "origin_in_tank")
               if c in chunk.columns]
    parts = [chunk]
    n_trig = len(overlay)
    firsts = chunk.groupby("eventID", sort=False).head(1)
    if empty_events is not None and len(empty_events):
        firsts = pd.concat([firsts, empty_events.reindex(columns=firsts.columns)], ignore_index=True)
    for e in firsts.to_dict("records"):
        h = overlay[overlay_pick(seed, e["_source_file"], e["eventNumber"], n_trig)]
        h = h[h.t > t_min]
        if not len(h):
            continue
        parts.append(h.assign(**{c: e[c] for c in ev_cols}, truth_class=0, is_neutron=0,
                              is_untraced=0, ancestor_trackID=-5, ancestor_pdg=-5,
                              origin_pdg=-999, origin_trackID=-5, origin_child_trackID=-5,
                              wf_matched=1, is_overlay=1))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["eventID", "t"], kind="stable").reset_index(drop=True)


def _load_chunk(i_grp: int) -> pd.DataFrame:
    import pyarrow.parquet as pq
    from . import cc_selection
    ctx, cfg = _G["ctx"], _G["cfg"]
    pf = pq.ParquetFile(_G["pulses_path"])
    avail = set(pf.schema_arrow.names)
    chunk = pf.read_row_group(i_grp, columns=[c for c in PULSE_COLS if c in avail]).to_pandas()
    if "pe" not in chunk.columns:
        chunk["pe"] = 1.0
    full = _G.get("hit_window") == "full"
    chunk = cc_selection.apply_residual_filter(chunk, ctx, verbose=False,
                                               prompt_window_ns=0.0 if full else None)
    if _G["hit_cleaning"] == "unmatched_lowpe":
        chunk = chunk[~(_no_waveform(chunk) & (chunk.pe.to_numpy() < cfg["lowpe_cut"]))]
    if _G.get("overlay") is not None:
        t_min = -np.inf if full else float((ctx.extra.get("residual") or {}).get("prompt_window_ns", 0.0))
        empty = None
        if _G.get("events") is not None:
            ev = _G["events"].get(_G["group_file"][i_grp])
            have = set(chunk.eventNumber.unique())
            empty = ev[~ev.eventNumber.isin(have)].assign(
                eventID=lambda d: -(i_grp * 100000 + d.eventNumber + 1))
        chunk = add_overlay(chunk, _G["overlay"], t_min, int(cfg.get("overlay_seed", 6256)), empty)
    if cfg.get("hit_t_min_ns") is not None:
        chunk = chunk[chunk.t >= float(cfg["hit_t_min_ns"])]
    chunk["_source_file"] = chunk["_source_file"].map(lambda s: Path(str(s)).name)
    return chunk


def _event_ref(src: str, evn: int, cfg: dict, nuv, rng) -> Optional[np.ndarray]:
    ref = source_from_filename(src) if cfg.get("source_from_filename") else None
    if ref is None and nuv is not None:
        v = nuv.get((src, evn))
        if v is not None:
            ref = np.asarray(v, float) + rng.normal(0.0, cfg["nu_smear_m"], 3)
    return ref


def _process_group_particle(i_grp: int):
    g = _G
    cfg, feat, methods = g["cfg"], g["feat"], g["methods"]
    chunk = _load_chunk(i_grp)
    if not len(chunk):
        return [], [], []
    lo, hi = g["window"]
    off = g["offset_ns"]
    rng = np.random.default_rng(1000 + i_grp)
    cl_rows, obj_rows = [], []
    for evid, df_ev in chunk.groupby("eventID", sort=False):
        df_ev = df_ev.reset_index(drop=True)
        src, evn = df_ev["_source_file"].iloc[0], int(df_ev["eventNumber"].iloc[0])
        caps = g["caps"].get((src, evn), g["caps"][("__empty__", -1)])
        in_win = caps[(caps.cap_t + off >= lo) & (caps.cap_t + off <= hi)].reset_index(drop=True)
        cap_t = in_win.cap_t.to_numpy(float)
        obj, objs = build_truth_objects(df_ev, cap_t, off, cfg, in_win)
        ref = _event_ref(src, evn, cfg, g["nuv"], rng)
        feat_ev = dict(feat)
        feat_ev["_source_pos_m"] = None
        df_cl = g["clusters"].get(evid) if g["clusters"] is not None else None
        lbl = label_event(df_ev, df_cl, methods, feat_ev, cfg, ref)
        rows, per_obj = match_clusters_to_objects(lbl, obj, objs, df_ev, cfg, g["owner_rule"])
        if g["cluster_features"]:
            from .cluster_features import compute_cluster_features
            for r in rows:
                f = compute_cluster_features(df_ev[lbl[r["method"]] == r["cluster_id"]], g["geo"])
                r.update({f"f_{k}": f.get(k, np.nan) for k in CLUSTER_FEATURES})
        ev = dict(row_group=i_grp, eventID=int(evid), _source_file=src, eventNumber=evn,
                  cc_pass=int(df_ev["cc_pass"].iloc[0]) if "cc_pass" in df_ev.columns else -1,
                  n_caps_in_window=len(cap_t))
        cl_rows += [{**ev, **r} for r in rows]
        for j, o in enumerate(objs.to_dict("records")):
            r = {**ev, **o}
            for m, d in per_obj.items():
                r[f"n_owned_{m}"] = int(d["owned"][j])
                r[f"best_hits_{m}"] = int(d["best"][j])
                r[f"n_merged_into_{m}"] = int(d["merged"][j])
                r[f"hits_own_{m}"] = int(d["hits_own"][j])
                r[f"hits_clustered_{m}"] = int(d["hits_clustered"][j])
                r[f"pe_own_{m}"] = float(d["pe_own"][j])
                r[f"pe_clustered_{m}"] = float(d["pe_clustered"][j])
            obj_rows.append(r)
    return cl_rows, obj_rows, []


def _process_group(i_grp: int):
    from .cluster_features import _truth_label_cluster, compute_cluster_features
    from .cluster_matching import _neutron_breakdown
    from .optics import _apply_truth_window

    g = _G
    ctx, cfg, feat, methods = g["ctx"], g["cfg"], g["feat"], g["methods"]
    chunk = _load_chunk(i_grp)
    if not len(chunk):
        return [], [], []
    caps_all, clusters, nuv = g["caps"], g["clusters"], g["nuv"]
    lo, hi = g["window"]
    off = g["offset_ns"]
    geo = g["geo"]
    rng = np.random.default_rng(1000 + i_grp)

    cl_rows, cap_rows, tid_rows = [], [], []
    for evid, df_ev in chunk.groupby("eventID", sort=False):
        df_ev = df_ev.reset_index(drop=True)
        src, evn = df_ev["_source_file"].iloc[0], int(df_ev["eventNumber"].iloc[0])
        caps = caps_all.get((src, evn))
        if caps is None:
            caps = caps_all.get(("__empty__", -1))
        # Captures whose light can fall inside the analysed window.
        in_win = caps[(caps.cap_t + off >= lo) & (caps.cap_t + off <= hi)]
        cap_t = in_win.cap_t.to_numpy(float)
        hit_cap = assign_hits_to_captures(df_ev, cap_t, off, cfg["hit_window_ns"])
        hit_cat = hit_categories(df_ev, hit_cap)

        # INIT reference: AmBe source from the filename, else a smeared neutrino vertex.
        ref = _event_ref(src, evn, cfg, nuv, rng)
        feat_ev = dict(feat)
        feat_ev["_source_pos_m"] = None        # OPTICS is run exactly as in production

        df_cl = clusters.get(evid) if clusters is not None else None
        lbl = label_event(df_ev, df_cl, methods, feat_ev, cfg, ref)
        truth_mask = _apply_truth_window(df_ev, float(feat.get("truth_window_ns", 75.0)))
        t = df_ev["t"].to_numpy(float)
        tids = df_ev["ancestor_trackID"].to_numpy(int)

        # trackIDs carried by each capture's light (the artefact test, method-free)
        for j in range(len(cap_t)):
            m = hit_cap == j
            if m.sum() >= cfg["min_cap_hits"]:
                u, c = np.unique(tids[m], return_counts=True)
                tid_rows.append(dict(eventID=int(evid), cap_idx=int(in_win.cap_idx.iloc[j]),
                                     n_hits=int(m.sum()), n_trackids=len(u),
                                     frac_top_trackid=float(c.max() / m.sum())))

        own = {m: np.zeros(len(cap_t), int) for m in lbl}          # clusters owning cap j
        cap_hits_in = {m: np.zeros(len(cap_t), int) for m in lbl}  # its hits in owners
        for method, lb in lbl.items():
            for cid in [c for c in np.unique(lb) if c >= 0]:
                mask = lb == cid
                n = int(mask.sum())
                bd = _neutron_breakdown(mask, df_ev, truth_mask)
                is_n, _ = _truth_label_cluster(mask, df_ev, truth_mask)
                hc = hit_cap[mask]
                hc = hc[hc >= 0]
                if len(hc):
                    u, c = np.unique(hc, return_counts=True)
                    order = np.argsort(-c)
                    u, c = u[order], c[order]
                    dom, n_dom = int(u[0]), int(c[0])
                    n_contrib = int((c >= cfg["min_cap_hits"]).sum())
                else:
                    dom, n_dom, n_contrib = -1, 0, 0
                cats = hit_cat[mask].copy()
                if dom >= 0:     # capture hits that are NOT the dominant capture
                    cats[(hit_cap[mask] >= 0) & (hit_cap[mask] != dom)] = COMP_CATS.index("cap_other")
                comp_counts = np.bincount(cats, minlength=len(COMP_CATS))
                assert comp_counts.sum() == n
                comp = {f"comp_{c}": int(k) for c, k in zip(COMP_CATS, comp_counts)}
                t_med = float(np.median(t[mask]))
                n_time = int((np.abs(cap_t + off - t_med) <= cfg["match_ns"]).sum())
                row = dict(row_group=i_grp, eventID=int(evid), _source_file=src,
                           eventNumber=evn, method=method, cluster_id=int(cid),
                           t_med=t_med, t_span=float(np.ptp(t[mask])),
                           pe_cluster=float(df_ev["pe"].to_numpy(float)[mask].sum()),
                           n_overlay=int(df_ev["is_overlay"].to_numpy(int)[mask].sum())
                           if "is_overlay" in df_ev.columns else 0,
                           is_truth_neutron=int(is_n), n_caps_in_window=len(cap_t),
                           n_cap_hits=int(len(hc)), cap_dom=dom, n_cap_dom=n_dom,
                           cap_dom_frac=n_dom / max(n, 1), n_caps_contrib=n_contrib,
                           n_caps_time=n_time, **bd, **comp)
                if dom >= 0:
                    cr = in_win.iloc[dom]
                    row.update(cap_dom_idx=int(cr.cap_idx),
                               cap_t=float(cr.cap_t), dt_cap=t_med - float(cr.cap_t) - off,
                               cap_x=float(cr.cap_x), cap_y=float(cr.cap_y),
                               cap_z=float(cr.cap_z), cap_medium=cr.cap_medium,
                               cap_E=float(cr.cap_E))
                    if n_dom >= cfg["min_cap_hits"]:
                        own[method][dom] += 1
                        cap_hits_in[method][dom] += n_dom
                if (g["vertex"] and method in g["vertex_methods"] and dom >= 0
                        and n_contrib == 1 and n >= 4):
                    f = compute_cluster_features(df_ev[mask].reset_index(drop=True), geo)
                    capv = np.array([row["cap_x"], row["cap_y"], row["cap_z"]])
                    cen = np.array([f["vtx_x"], f["vtx_y"], f["vtx_z"]])
                    fit = np.array([f.get("vtx_fit_x", np.nan), f.get("vtx_fit_y", np.nan),
                                    f.get("vtx_fit_z", np.nan)])
                    xyz = df_ev.loc[mask, ["x", "y", "z"]].to_numpy(float)
                    tm = t[mask]
                    sig = {}
                    for name, rv in (("centroid", cen), ("truecap", capv), ("init", ref)):
                        sig[f"sigma_t_mad_tof_{name}"] = (
                            _mad(tm - np.linalg.norm(xyz - rv[None, :], axis=1) / SOL_WATER)
                            if rv is not None else np.nan)
                    row.update(d_centroid=float(np.linalg.norm(cen - capv)),
                               d_fit=float(np.linalg.norm(fit - capv)),
                               fit_converged=int(f.get("fit_converged", 0)),
                               n_fit_hits=int(f.get("n_fit_hits", 0)),
                               d_wall_cap=float(min(geo.tank_radius - np.hypot(
                                   capv[0] - geo.tank_center[0], capv[2] - geo.tank_center[2]),
                                   geo.tank_top - capv[1], capv[1] - geo.tank_bot)),
                               d_init=(float(np.linalg.norm(ref - capv))
                                       if ref is not None else np.nan),
                               sigma_t_mad_raw=_mad(tm), **sig)
                cl_rows.append(row)

        n_light = np.bincount(hit_cap[hit_cap >= 0], minlength=len(cap_t)) if len(cap_t) \
            else np.array([], int)
        for j in range(len(cap_t)):
            cr = in_win.iloc[j]
            r = dict(eventID=int(evid), _source_file=src, eventNumber=evn,
                     cap_idx=int(cr.cap_idx), cap_t=float(cr.cap_t),
                     cap_medium=cr.cap_medium, n_light_hits=int(n_light[j]),
                     visible=int(n_light[j] >= cfg["min_visible_hits"]))
            for method in lbl:
                r[f"n_owners_{method}"] = int(own[method][j])
                r[f"hits_in_owners_{method}"] = int(cap_hits_in[method][j])
            cap_rows.append(r)
    return cl_rows, cap_rows, tid_rows


# --------------------------------------------------------------------------- #
# Offset calibration
# --------------------------------------------------------------------------- #

def calibrate_offset(pulses_path: Path, caps_by_key: dict, ctx: RunContext,
                     n_groups: int = 5) -> dict:
    """Median of (neutron-trackID median hit time - nearest capture time), from
    trackIDs with >= 5 delayed hits.  AmBe QE1.50 gave a control IQR of 2 ns."""
    import pyarrow.parquet as pq
    from . import cc_selection
    pf = pq.ParquetFile(str(pulses_path))
    clock = mc_clock.shifts(ctx.extra)[1]
    dts = []
    for i in range(min(n_groups, pf.num_row_groups)):
        ch = pf.read_row_group(i, columns=["eventID", "eventNumber", "_source_file", "t",
                                           "ancestor_trackID", "ancestor_pdg", "cc_pass"]
                               ).to_pandas()
        ch = cc_selection.apply_residual_filter(ch, ctx, verbose=False)
        ch = ch[(ch.ancestor_pdg == NEUTRON_PDG) & (ch.ancestor_trackID >= 0)]
        ch["_source_file"] = ch["_source_file"].map(lambda s: Path(str(s)).name)
        for (src, evn, tid), h in ch.groupby(["_source_file", "eventNumber",
                                              "ancestor_trackID"]):
            if len(h) < 5:
                continue
            caps = caps_by_key.get((src, int(evn)))
            if caps is None or not len(caps):
                continue
            d = float(h.t.median()) - clock - caps.cap_t.to_numpy(float)
            dts.append(d[np.argmin(np.abs(d))])
    dts = np.array(dts)
    core = dts[np.abs(dts) < 50]
    if len(core) < 20:
        raise SystemExit(f"[capture] offset calibration failed: {len(core)} matched trackIDs")
    q = np.percentile(core, [25, 50, 75])
    return dict(offset_ns=float(q[1]) + clock, iqr_ns=float(q[2] - q[0]), n_trackids=int(len(dts)),
                frac_within_50ns=float(len(core) / max(len(dts), 1)))


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #

def summarise(cl: pd.DataFrame, caps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    n_ev = cl.eventID.nunique()
    vis = caps[caps.visible == 1]
    for method, g in cl.groupby("method"):
        sig = g[g.n_cap_dom >= cfg["min_cap_hits"]]
        rows.append(dict(
            method=method, n_clusters=len(g), clusters_per_event=len(g) / max(n_ev, 1),
            capture_recall=float((vis[f"n_owners_{method}"] >= 1).mean()) if len(vis) else np.nan,
            capture_split_rate=float((vis[f"n_owners_{method}"] >= 2).mean()) if len(vis) else np.nan,
            capture_hit_efficiency=float(vis[f"hits_in_owners_{method}"].sum()
                                         / max(vis.n_light_hits.sum(), 1)),
            true_merge_rate=float((sig.n_caps_contrib >= 2).mean()) if len(sig) else np.nan,
            hit_purity=float(sig.n_cap_dom.sum() / max(sig.n_hits.sum(), 1)),
            spurious_per_event=float((g.n_cap_dom < cfg["min_cap_hits"]).sum() / max(n_ev, 1)),
            frac_trackid_signal=float(g.is_truth_neutron.mean()),
            frac_capture_signal=float((g.n_caps_time >= 1).mean()),
        ))
    return pd.DataFrame(rows)


def summarise_objects(cl: pd.DataFrame, ob: pd.DataFrame, cfg: dict
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per method x object class, per method x sample x cluster tag)."""
    methods = list(dict.fromkeys(cl.method))
    vis = ob[ob.n_hits >= cfg["min_visible_hits"]]
    rows = []
    for m in methods:
        for (sample, cls), g in vis.assign(sample=np.where(vis.cc_pass == 1, "cc_pass", "other")
                                           ).groupby(["sample", "obj_class"]):
            rows.append(dict(method=m, sample=sample, obj_class=cls, n_objects=len(g),
                             found=float((g[f"n_owned_{m}"] >= 1).mean()),
                             split=float((g[f"n_owned_{m}"] >= 2).mean()),
                             merged_into_other=float((g[f"n_merged_into_{m}"] >= 1).mean()),
                             hit_completeness=float(g[f"best_hits_{m}"].sum() / max(g.n_hits.sum(), 1)),
                             median_hits=float(g.n_hits.median())))
    per_obj = pd.DataFrame(rows)
    rows = []
    n_ev = cl.groupby(np.where(cl.cc_pass == 1, "cc_pass", "other")).eventID.nunique()
    for (m, sample, tag), g in cl.assign(sample=np.where(cl.cc_pass == 1, "cc_pass", "other")
                                         ).groupby(["method", "sample", "tag"]):
        rows.append(dict(method=m, sample=sample, tag=tag, n_clusters=len(g),
                         per_event=len(g) / max(int(n_ev.get(sample, 0)), 1),
                         median_owner_frac=float(g.owner_frac.median()),
                         frac_owner_below_50=float((g.owner_frac < 0.5).mean()),
                         frac_multi_object=float((g.n_objects_contrib >= 2).mean()),
                         median_completeness=float(g.owner_completeness.median())))
    return per_obj, pd.DataFrame(rows)


def object_physics_checks(ob: pd.DataFrame, cfg: dict, window) -> dict:
    """Michel lifetime and how many neutron objects each true capture's light forms."""
    out = {}
    lo = float(window[0])
    mi = ob[(ob.obj_class == "michel") & (ob.n_hits >= cfg["min_visible_hits"]) & (ob.t_first >= lo)]
    if len(mi) >= 20:
        out["michel_n"] = int(len(mi))
        out["michel_tau_ns"] = float((mi.t_first - lo).mean())
    nc = ob[(ob.obj_class == "n_capture") & (ob.n_hits >= cfg["min_cap_hits"])]
    if len(nc):
        per_cap = nc.groupby(["eventID", "cap_idx"]).size()
        out["captures_with_objects"] = int(len(per_cap))
        out["frac_capture_one_object"] = float((per_cap == 1).mean())
        out["frac_capture_2plus_objects"] = float((per_cap >= 2).mean())
    return out


def merged_breakdown(cl: pd.DataFrame) -> pd.DataFrame:
    """The decisive table.  For each trackID-label category, how many TRUE captures
    contribute >= min_cap_hits hits?  A 'merged_neutrons' cluster with exactly one
    contributing capture is a labelling artefact, not a clustering merge."""
    rows = []
    cats = {
        "trackid_signal_multi_tid": (cl.is_truth_neutron == 1) & (cl.n_contrib_neutrons >= 2),
        "trackid_signal_single_tid": (cl.is_truth_neutron == 1) & (cl.n_contrib_neutrons == 1),
        "bkg_merged_neutrons": (cl.is_truth_neutron == 0) & (cl.fail_reason == "merged_neutrons"),
        "bkg_diluted": (cl.is_truth_neutron == 0) & (cl.fail_reason == "diluted_by_nonneutron"),
        "bkg_no_neutron": (cl.is_truth_neutron == 0) & (cl.fail_reason == "no_neutron_hits"),
    }
    for method, g in cl.groupby("method"):
        for cat, m in cats.items():
            s = g[m.loc[g.index]]
            if not len(s):
                continue
            nc = s.n_caps_contrib
            rows.append(dict(method=method, category=cat, n_clusters=len(s),
                             frac_0_captures=float((nc == 0).mean()),
                             frac_1_capture=float((nc == 1).mean()),
                             frac_2plus_captures=float((nc >= 2).mean()),
                             frac_time_matched=float((s.n_caps_time >= 1).mean()),
                             median_cap_dom_frac=float(s.cap_dom_frac.median()),
                             median_trackid_dom_frac=float(s.dom_frac.median())))
    return pd.DataFrame(rows)


def vertex_summary(cl: pd.DataFrame) -> pd.DataFrame:
    if "d_fit" not in cl.columns:
        return pd.DataFrame()
    v = cl[cl.d_centroid.notna()]
    rows = []
    for (method, med), g in v.groupby(["method", "cap_medium"]):
        conv = g[g.fit_converged == 1]
        rows.append(dict(
            method=method, capture_medium=med, n=len(g),
            fit_convergence=float(g.fit_converged.mean()),
            d_centroid_median_m=float(g.d_centroid.median()),
            d_centroid_p68_m=float(g.d_centroid.quantile(0.68)),
            d_fit_median_m=float(conv.d_fit.median()) if len(conv) else np.nan,
            d_fit_p68_m=float(conv.d_fit.quantile(0.68)) if len(conv) else np.nan,
            d_init_median_m=float(g.d_init.median()),
            sigma_t_raw_ns=float(g.sigma_t_mad_raw.median()),
            sigma_t_tof_centroid_ns=float(g.sigma_t_mad_tof_centroid.median()),
            sigma_t_tof_init_ns=float(g.sigma_t_mad_tof_init.median()),
            sigma_t_tof_truecap_ns=float(g.sigma_t_mad_tof_truecap.median()),
            median_n_hits=float(g.n_hits.median())))
    return pd.DataFrame(rows)



# --------------------------------------------------------------------------- #
# Hit-composition report (truth tagging)
# --------------------------------------------------------------------------- #

COMP_COLORS = {"cap_dom": "#0077BB", "cap_other": "#33BBEE", "neutron_nocap": "#009988",
               "bg_muon": "#CC3311", "bg_pion_charged": "#EE7733", "bg_pion_zero": "#EE3377",
               "bg_proton": "#AA4499", "bg_pure_em": "#BBBB00", "bg_other": "#882255",
               "darknoise": "#BBBBBB", "untraced": "#555555"}
COMP_NAMES = {"cap_dom": "dominant capture", "cap_other": "other capture(s)",
              "neutron_nocap": "neutron, no capture", "bg_muon": "muon",
              "bg_pion_charged": "charged pion", "bg_pion_zero": "π⁰", "bg_proton": "proton",
              "bg_pure_em": "pure e/γ chain", "bg_other": "other physics",
              "darknoise": "dark noise", "untraced": "untraced"}
FRAC_SCAN = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _capture_tag(cl: pd.DataFrame, f: float = 0.5, min_hits: int = 3) -> pd.Series:
    """Capture truth tag: a true capture within 50 ns of the cluster median owns at
    least a fraction f of ALL its hits (and >= min_hits of them)."""
    return (cl.n_caps_time >= 1) & (cl.cap_dom_frac >= f) & (cl.n_cap_dom >= min_hits)


def _trackid_tag(cl: pd.DataFrame, f: float = 0.5) -> pd.Series:
    """The production tag (_truth_label_cluster) at threshold f."""
    return cl.dom_in_truth.astype(bool) & (cl.dom_frac >= f)


def composition_tables(cl: pd.DataFrame, caps: pd.DataFrame, cfg: dict):
    comp_cols = [f"comp_{c}" for c in COMP_CATS]
    rows = []
    for method, g in cl.groupby("method"):
        tag = _capture_tag(g)
        for name, sub in (("all", g), ("capture_signal", g[tag]), ("capture_background", g[~tag]),
                          ("trackid_signal", g[_trackid_tag(g)]),
                          ("trackid_background", g[~_trackid_tag(g)])):
            if not len(sub):
                continue
            tot = sub.n_hits.sum()
            r = dict(method=method, sample=name, n_clusters=len(sub),
                     mean_hits=float(sub.n_hits.mean()))
            r.update({c: float(sub[f"comp_{c}"].sum() / tot) for c in COMP_CATS})
            rows.append(r)
    comp = pd.DataFrame(rows)

    vis = caps[caps.visible == 1].copy()
    scan = []
    for method, g in cl.groupby("method"):
        for scheme, fn in (("capture", _capture_tag), ("trackid", _trackid_tag)):
            for f in FRAC_SCAN:
                sig = g[fn(g, f)]
                owned = set(zip(sig.eventID, sig.cap_dom_idx.fillna(-1).astype(int)))
                rec = np.mean([(e, c) in owned for e, c in zip(vis.eventID, vis.cap_idx)]) \
                    if len(vis) else np.nan
                tot = sig.n_hits.sum()
                scan.append(dict(method=method, scheme=scheme, threshold=f,
                                 signal_fraction=len(sig) / max(len(g), 1),
                                 capture_recall=float(rec),
                                 signal_hit_purity=float(sig.comp_cap_dom.sum() / max(tot, 1)),
                                 signal_other_capture_frac=float(sig.comp_cap_other.sum() / max(tot, 1)),
                                 signal_bkg_physics_frac=float(sig[[c for c in comp_cols
                                                                   if c.startswith("comp_bg_")]
                                                                  ].sum().sum() / max(tot, 1)),
                                 signal_noise_frac=float((sig.comp_darknoise.sum()
                                                          + sig.comp_untraced.sum()) / max(tot, 1))))
    return comp, pd.DataFrame(scan)


def composition_plots(cl: pd.DataFrame, comp: pd.DataFrame, scan: pd.DataFrame,
                      pdf_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    methods = list(dict.fromkeys(comp.method))
    with PdfPages(pdf_path) as pdf:
        # 1. hit-weighted composition, per method x tag class
        for scheme in ("capture", "trackid"):
            fig, ax = plt.subplots(figsize=(11, 0.55 * 2 * len(methods) + 2))
            labels, lefts = [], None
            rows = []
            for m in methods:
                for cls in (f"{scheme}_signal", f"{scheme}_background"):
                    r = comp[(comp.method == m) & (comp["sample"] == cls)]
                    if len(r):
                        rows.append(r.iloc[0])
                        labels.append(f"{m} — {cls.split('_')[1]} (n={int(r.n_clusters.iloc[0])})")
            y = np.arange(len(rows))
            left = np.zeros(len(rows))
            for c in COMP_CATS:
                v = np.array([r[c] for r in rows])
                ax.barh(y, v, left=left, color=COMP_COLORS[c], label=COMP_NAMES[c])
                left += v
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Fraction of cluster hits (hit-weighted)")
            ax.set_title(f"{title}\nHit composition by {scheme} tag")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=7,
                      frameon=False)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 2. dominant-capture share vs dominant-trackID share
        fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8),
                                 squeeze=False, sharey=True)
        bins = np.linspace(0, 1, 26)
        for ax, m in zip(axes[0], methods):
            g = cl[cl.method == m]
            ax.hist(g.cap_dom_frac, bins, histtype="step", lw=1.8, color="#0077BB",
                    label="top true capture")
            ax.hist(g.dom_frac, bins, histtype="step", lw=1.8, color="#EE7733",
                    label="top neutron trackID")
            ax.axvline(0.5, color="k", ls=":", lw=1)
            ax.set_xlabel("Share of ALL cluster hits")
            ax.set_title(m)
        axes[0][0].set_ylabel("Clusters")
        axes[0][0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{title}\nHow much of each cluster its top capture vs top trackID owns")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 3. threshold scan: recall vs hit purity
        fig, ax = plt.subplots(figsize=(7, 5.5))
        cols = plt.cm.tab10(np.arange(len(methods)))
        for c, m in zip(cols, methods):
            for scheme, ls, mk in (("capture", "-", "o"), ("trackid", "--", "s")):
                d = scan[(scan.method == m) & (scan.scheme == scheme)].sort_values("threshold")
                ax.plot(d.signal_hit_purity, d.capture_recall, ls=ls, marker=mk, color=c,
                        label=f"{m} · {scheme} tag")
                for _, r in d.iterrows():
                    if r.threshold in (0.3, 0.5, 0.8):
                        ax.annotate(f"{r.threshold:g}", (r.signal_hit_purity, r.capture_recall),
                                    fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Signal-cluster hit purity (dominant-capture share)")
        ax.set_ylabel("Capture recall (visible captures with a signal cluster)")
        ax.set_title(f"{title}\nTag threshold scan (labels = share threshold)")
        ax.legend(fontsize=7, frameon=False)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_composition_report(ctx: RunContext, cl: pd.DataFrame, caps: pd.DataFrame,
                             cfg: dict, tag: str) -> None:
    if "comp_cap_dom" not in cl.columns:
        print("[capture] no comp_* columns (table predates the composition breakdown) — "
              "re-run `ambe mc capture` to get the report")
        return
    comp, scan = composition_tables(cl, caps, cfg)
    comp.to_csv(_out(ctx, "capture_truth_composition", tag, "csv"), index=False)
    scan.to_csv(_out(ctx, "capture_truth_tag_scan", tag, "csv"), index=False)
    pdf = ctx.plots_dir / f"{ctx.run_name}__capture_truth_composition{('__' + tag) if tag else ''}.pdf"
    composition_plots(cl, comp, scan, pdf, ctx.title("Cluster hit composition")
                      if hasattr(ctx, "title") else ctx.run_name)
    pd.set_option("display.width", 220)
    show = comp[comp["sample"].isin(["capture_signal", "capture_background"])]
    print("\n== hit-weighted composition (capture tag, f>=0.5) ==")
    print(show.round(3).to_string(index=False))
    print("\n== tag threshold scan ==")
    print(scan[scan.threshold.isin([0.3, 0.5, 0.7])].round(3).to_string(index=False))
    print(f"\n[capture] wrote composition report -> {pdf}")

# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _out(ctx: RunContext, base: str, tag: str, kind: str) -> Path:
    name = f"{ctx.run_name}__{base}" + (f"__{tag}" if tag else "")
    return ctx.parquet_path(name) if kind == "parquet" else ctx.csv_path(name)


def run(ctx: RunContext, args) -> None:
    import glob
    import pyarrow.parquet as pq
    from .cluster_features import load_geometry

    cfg = {**DEFAULTS, **(ctx.extra.get("capture_truth", {}) or {})}
    cfg["peak"] = {**DEFAULTS["peak"], **(cfg.get("peak") or {})}
    cfg["_full_window"] = args.hit_window == "full"
    feat = ctx.extra.get("features", {}) or {}
    methods = args.methods.split(",")
    residual = ctx.extra.get("residual", {}) or {}
    win = cfg.get("window_ns") or [float(residual.get("prompt_window_ns", 2000.0)), 70000.0]

    if args.report_only:
        cl = pd.read_parquet(_out(ctx, "capture_truth_clusters", args.tag, "parquet"))
        cp = pd.read_parquet(_out(ctx, "capture_truth_captures", args.tag, "parquet"))
        write_composition_report(ctx, cl, cp, cfg, args.tag)
        return

    pulses_path = ctx.parquet_path(f"{ctx.run_name}__pulses")
    pf = pq.ParquetFile(str(pulses_path))
    n_groups = pf.num_row_groups
    groups = list(range(min(args.max_files, n_groups) if args.max_files else n_groups))
    if args.merge_shards:
        merge_shards(ctx, groups, cfg, win, args)
        return
    if args.groups:
        groups = [int(g) for g in args.groups.split(",")]
        if max(groups) >= n_groups:
            raise SystemExit(f"[capture] --groups {args.groups}: {pulses_path.name} has {n_groups} row groups")
    print(f"[capture] {pulses_path.name}: {n_groups} row groups, processing {len(groups)}"
          f"{'  *** PARTIAL — pilot only ***' if len(groups) < n_groups else ''}")

    # True captures: cached, and only for the files actually processed.
    cap_path = ctx.parquet_path(f"{ctx.run_name}__true_captures")
    files = sorted({f for pat in ctx.inputs.get("root_files", []) for f in glob.glob(pat)})
    group_file = {i: Path(str(pf.read_row_group(i, columns=["_source_file"]).column(0)[0].as_py())).name
                  for i in groups}
    want = set(group_file.values())
    if cap_path.exists():
        caps = pd.read_parquet(cap_path)
        missing = want - set(caps._source_file.unique())
    else:
        caps, missing = None, want
    if missing and args.cache_only is False and not files:
        raise SystemExit(f"[capture] {len(missing)} files have no cached captures and the "
                         "ROOT inputs are not visible here (EAF has no /pnfs). Run "
                         "`ambe mc capture --cache-only` on a gpvm first.")
    if missing:
        new = load_true_captures([f for f in files if Path(f).name in missing])
        caps = new if caps is None else pd.concat([caps, new], ignore_index=True)
        caps.to_parquet(cap_path, index=False)
        print(f"[capture] cached {len(caps):,} captures -> {cap_path.name}")
    if args.cache_only:
        nv_path = ctx.parquet_path(f"{ctx.run_name}__true_nu_vertices")
        nv = neutrino_vertices([f for f in files if Path(f).name in want])
        if len(nv):
            nv.to_parquet(nv_path, index=False)
        print(f"[capture] --cache-only: {caps._source_file.nunique()} files of captures, "
              f"{len(nv)} neutrino vertices cached. Heavy stage can now run on EAF.")
        return
    caps = caps[caps._source_file.isin(want)]
    if args.low_memory:
        caps_by_key = _SliceIndex(caps, ["_source_file", "eventNumber"],
                                  extra={("__empty__", -1): caps.iloc[0:0]})
    else:
        caps_by_key = {k: g.reset_index(drop=True)
                       for k, g in caps.groupby(["_source_file", "eventNumber"])}
        caps_by_key[("__empty__", -1)] = caps.iloc[0:0]

    nuv = None
    if "nu_x" in caps.columns and not cfg.get("source_from_filename"):
        nv_path = ctx.parquet_path(f"{ctx.run_name}__true_nu_vertices")
        nv = pd.read_parquet(nv_path) if nv_path.exists() else pd.DataFrame()
        miss_nv = want - set(nv._source_file.unique()) if len(nv) else want
        if miss_nv:
            new_nv = neutrino_vertices([f for f in files if Path(f).name in miss_nv])
            if len(new_nv):
                nv = pd.concat([nv, new_nv], ignore_index=True)
                nv.to_parquet(nv_path, index=False)
        nuv = {(f, int(e)): (x, y, z) for f, e, x, y, z in
               zip(nv._source_file, nv.eventNumber, nv.nu_x, nv.nu_y, nv.nu_z)}

    off = (dict(offset_ns=mc_clock.truth_offset_ns(cfg["offset_ns"], ctx.extra), iqr_ns=np.nan, n_trackids=0,
                frac_within_50ns=np.nan, source="config")
           if cfg.get("offset_ns") is not None
           else calibrate_offset(pulses_path, caps_by_key, ctx))
    print(f"[capture] hit-time − capture-time offset {off['offset_ns']:.2f} ns "
          f"(IQR {off['iqr_ns']:.2f} ns, {off['frac_within_50ns']:.1%} of "
          f"{off['n_trackids']} trackIDs within 50 ns)")
    _out(ctx, "capture_truth_offset", args.tag, "csv").with_suffix(".json").write_text(
        json.dumps(off, indent=2))

    geo_block = ctx.extra.get("geometry", {}) or {}
    geo = load_geometry(geo_block.get("pmt_geometry"), geo_block.get("timing_offsets"),
                        frame=args.geometry_frame or geo_block.get("frame", "tank"),
                        apply_offsets=(args.timing_offsets == "on") if args.timing_offsets
                        else bool(geo_block.get("apply_timing_offsets", True)))
    cf_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if cf_path.exists():
        cf = pd.read_parquet(cf_path)
        cf = cf[cf._source_file.isin(want)]
        clusters = (_SliceIndex(cf, ["eventID"]) if args.low_memory
                    else {k: g.reset_index(drop=True) for k, g in cf.groupby("eventID")})
        del cf
    else:
        clusters = None

    if args.scan_peak:
        methods = methods + [m for m in scan_methods(cfg) if m not in methods]
    if args.scan_optics:
        methods = methods + [m for m in optics_scan_methods(cfg) if m not in methods]
    _G.update(ctx=ctx, cfg=cfg, feat=feat, methods=methods, pulses_path=str(pulses_path),
              caps=caps_by_key, clusters=clusters, nuv=nuv, window=win,
              offset_ns=off["offset_ns"], geo=geo, vertex=args.vertex,
              vertex_methods=set(args.vertex_methods.split(",")),
              owner_rule=args.owner_rule, hit_cleaning=args.hit_cleaning,
              cluster_features=args.cluster_features, hit_window=args.hit_window,
              overlay=load_overlay(args.overlay) if args.overlay else None,
              events=None, group_file=group_file)
    if args.overlay and args.truth_unit == "capture":
        ev_path = ctx.parquet_path(f"{ctx.run_name}__events")
        if not ev_path.exists():
            raise SystemExit(f"[capture] --overlay in capture mode needs {ev_path.name} "
                             "(rerun `ambe mc process`): events with no MC hits get the overlay too")
        ev = pd.read_parquet(ev_path)
        _G["events"] = {f: g[["eventNumber", "_source_file"]] for f, g in ev.groupby("_source_file")
                        if f in want}
    if args.overlay:
        print(f"[capture] overlay: {len(_G['overlay']):,} random triggers from {args.overlay}")
    worker = _process_group_particle if args.truth_unit == "particle" else _process_group

    t0 = time.time()
    cl_parts, cap_parts, tid_parts = [], [], []
    def collect(k, a, b, c):
        cl_parts.append(pd.DataFrame(a)); cap_parts.append(pd.DataFrame(b)); tid_parts.append(pd.DataFrame(c))
        if (k + 1) % 10 == 0:
            print(f"[capture] {k + 1}/{len(groups)} groups ({time.time() - t0:.0f}s)")
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            for k, (a, b, c) in enumerate(pool.imap_unordered(worker, groups)):
                collect(k, a, b, c)
    else:
        for k, i in enumerate(groups):
            collect(k, *worker(i))
    cl, cp, tp = (pd.concat(p, ignore_index=True) for p in (cl_parts, cap_parts, tid_parts))
    del cl_parts, cap_parts, tid_parts
    if cl.empty:
        raise SystemExit("[capture] no clusters")
    if args.truth_unit == "particle":
        write_particle_outputs(ctx, cl, cp, cfg, win, args, time.time() - t0)
        mult = event_multiplicity(ctx, cl, cp, caps, want, win, off["offset_ns"], cfg)
        tag = f"{args.tag}_{args.owner_rule}" if args.tag else args.owner_rule
        mult.to_parquet(_out(ctx, "truth_event_multiplicity", tag, "parquet"), index=False)
        print(f"[capture] per-event neutron multiplicity: {len(mult):,} events")
        return
    cl.to_parquet(_out(ctx, "capture_truth_clusters", args.tag, "parquet"), index=False)
    cp.to_parquet(_out(ctx, "capture_truth_captures", args.tag, "parquet"), index=False)

    s = summarise(cl, cp, cfg)
    s.to_csv(_out(ctx, "capture_truth_summary", args.tag, "csv"), index=False)
    mb = merged_breakdown(cl)
    mb.to_csv(_out(ctx, "capture_truth_merged_breakdown", args.tag, "csv"), index=False)
    tpc = (tp.n_trackids.clip(upper=5).value_counts(normalize=True).sort_index()
           .rename("fraction").to_frame()) if len(tp) else pd.DataFrame()
    tpc.to_csv(_out(ctx, "capture_truth_tids_per_capture", args.tag, "csv"))
    vs = vertex_summary(cl)
    if len(vs):
        vs.to_csv(_out(ctx, "capture_truth_vertex", args.tag, "csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"\n[capture] {cl.eventID.nunique():,} events, {len(cp):,} captures in window, "
          f"{(cp.visible == 1).sum():,} visible  ({time.time() - t0:.0f}s)")
    print("\n== neutron trackIDs per capture (captures with >= "
          f"{cfg['min_cap_hits']} hits) ==\n{tpc.T.to_string()}")
    print(f"\n== per-method benchmark ==\n{s.round(4).to_string(index=False)}")
    print(f"\n== trackID label vs true captures ==\n{mb.round(3).to_string(index=False)}")
    if len(vs):
        print(f"\n== vertex ==\n{vs.round(3).to_string(index=False)}")
    write_composition_report(ctx, cl, cp, cfg, args.tag)


def event_multiplicity(ctx: RunContext, cl: pd.DataFrame, ob: pd.DataFrame, caps: pd.DataFrame,
                       files: set, win, offset_ns: float, cfg: dict) -> pd.DataFrame:
    """One row per event (zero-capture events included): true neutron counts and, per base
    method, reco counts of tagged capture clusters, delayed clusters and box-accepted clusters."""
    from ..data.selection import BoxSelection
    stream = (ctx.cuts or {}).get("cc_stream", "cc0pi_legacy")
    ev = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__cc_{stream}_events"),
                         columns=["_source_file", "eventNumber", "cc_pass", "trueNeutrons",
                                  "origin_in_tank"])
    ev["_source_file"] = ev["_source_file"].map(lambda s: Path(str(s)).name)
    ev = ev[ev._source_file.isin(files)].reset_index(drop=True)
    key = ["_source_file", "eventNumber"]
    lo, hi = win
    ci = caps[(caps.cap_t + offset_ns >= lo) & (caps.cap_t + offset_ns <= hi)]
    ev = ev.merge(ci.groupby(key).size().rename("n_true_captures").reset_index(), on=key, how="left")
    nc = ob[(ob.obj_class == "n_capture") & (ob.cap_j >= 0)]
    for k in (1, cfg["min_visible_hits"]):
        v = nc[nc.n_hits >= k].groupby(key).cap_j.nunique().rename(f"n_visible_captures_{k}")
        ev = ev.merge(v.reset_index(), on=key, how="left")
    box = BoxSelection()
    base = [m for m in cl.method.unique() if parse_scan_method(m) is None and parse_optics_scan(m) is None]
    for m in base:
        g = cl[cl.method == m]
        tm, nh = g.t_med.to_numpy(float), g.n_hits.to_numpy(int)
        sel = {"tagged": (g.tag == "n_capture").to_numpy(),
               "delayed": (tm >= box.cuts.ct_min) & (tm <= box.cuts.ct_max) & (nh >= box.cuts.chits_min)}
        if "f_pe_total" in g.columns:
            sel["box"] = np.logical_and.reduce(box.cut_masks(g.f_pe_total.to_numpy(float),
                                                             g.f_charge_bal_legacy.to_numpy(float), tm, nh))
        for name, s in sel.items():
            cnt = g[s].groupby(key).size().rename(f"n_reco_{name}_{m}")
            ev = ev.merge(cnt.reset_index(), on=key, how="left")
    cnt_cols = [c for c in ev.columns if c.startswith("n_")]
    ev[cnt_cols] = ev[cnt_cols].fillna(0).astype(int)
    return ev


def merge_shards(ctx: RunContext, groups: list, cfg: dict, win, args) -> None:
    """Join the `--groups <i> --tag <tag>_g<i>` runs of every row group into the outputs one
    run over all of them writes."""
    parts = {b: [] for b in ("truth_object_clusters", "truth_objects", "truth_event_multiplicity")}
    for i in groups:
        tag = f"{args.tag}_g{i}_{args.owner_rule}"
        paths = {b: _out(ctx, b, tag, "parquet") for b in parts}
        miss = [p.name for p in paths.values() if not p.exists()]
        if miss:
            raise SystemExit(f"[capture] --merge-shards: row group {i} missing {miss}")
        for b, p in paths.items():
            parts[b].append(pd.read_parquet(p))
    cl, ob, mult = (pd.concat(parts[b], ignore_index=True) for b in parts)
    write_particle_outputs(ctx, cl, ob, cfg, win, args, 0.0)
    tag = f"{args.tag}_{args.owner_rule}"
    mult.to_parquet(_out(ctx, "truth_event_multiplicity", tag, "parquet"), index=False)
    print(f"[capture] merged {len(groups)} row groups: {len(cl):,} clusters, {len(ob):,} objects, "
          f"{len(mult):,} events")


def write_particle_outputs(ctx: RunContext, cl: pd.DataFrame, ob: pd.DataFrame, cfg: dict,
                           win, args, dt: float) -> None:
    tag = f"{args.tag}_{args.owner_rule}" if args.tag else args.owner_rule
    cl.to_parquet(_out(ctx, "truth_object_clusters", tag, "parquet"), index=False)
    ob.to_parquet(_out(ctx, "truth_objects", tag, "parquet"), index=False)
    per_obj, per_tag = summarise_objects(cl, ob, cfg)
    per_obj.to_csv(_out(ctx, "truth_object_summary", tag, "csv"), index=False)
    per_tag.to_csv(_out(ctx, "truth_object_cluster_tags", tag, "csv"), index=False)
    chk = object_physics_checks(ob, cfg, win)
    _out(ctx, "truth_object_checks", tag, "csv").with_suffix(".json").write_text(
        json.dumps(chk, indent=2))
    base = [m for m in cl.method.unique() if parse_scan_method(m) is None and parse_optics_scan(m) is None]
    pd.set_option("display.width", 220)
    print(f"\n[capture] particle mode: {cl.eventID.nunique():,} events, {len(ob):,} objects "
          f"({dt:.0f}s)")
    print(f"\n== truth objects (visible, >= {cfg['min_visible_hits']} hits) ==\n"
          f"{per_obj[per_obj.method.isin(base)].round(3).to_string(index=False)}")
    print(f"\n== cluster tags ({args.owner_rule}) ==\n"
          f"{per_tag[per_tag.method.isin(base)].round(3).to_string(index=False)}")
    print(f"\n== physics checks ==\n{json.dumps(chk, indent=2)}")
    if args.scan_optics:
        om = [m for m in cl.method.unique() if parse_optics_scan(m) or m in base]
        so = scan_table(cl, per_obj, om)
        so["score"] = (so.found * (1 - so.split) * (1 - so.merged_into_other)
                       * so.one_neutron * so.purity_labelled)
        so.to_csv(_out(ctx, "truth_object_optics_scan", tag, "csv"))
        print(f"\n== OPTICS scan, top 10 by found x (1-split) x (1-merged) x one_neutron x purity ==\n"
              f"{so.sort_values('score', ascending=False).head(10).round(4).to_string()}")
    if args.scan_peak:
        sc = per_obj[(~per_obj.method.isin(base)) & (per_obj.obj_class == "n_capture")]
        sp = (cl[~cl.method.isin(base)].assign(non_capture=lambda d: ~d.tag.eq("n_capture"))
              .groupby("method").non_capture.sum() / cl.eventID.nunique()).rename("non_capture_clusters_per_event")
        sc = sc.groupby("method")[["found", "split", "merged_into_other", "hit_completeness"]
                                  ].mean().join(sp)
        sc.to_csv(_out(ctx, "truth_object_peak_scan", tag, "csv"))
        print(f"\n== peak scan, n_capture objects (top 10 by found) ==\n"
              f"{sc.sort_values('found', ascending=False).head(10).round(4).to_string()}")


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc capture")
    p.add_argument("--methods", default="optics,clusterfinder,optics_nms,clusterfinder_nms,"
                                        "peak,peak_tof")
    p.add_argument("--max-files", type=int, default=None,
                   help="process only the first N row groups (one per ROOT file)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--groups", default=None,
                   help="process only these row groups (comma list); grid shards use --tag <tag>_g<i>")
    p.add_argument("--merge-shards", action="store_true",
                   help="join the <tag>_g<i> shards of every row group (first --max-files) into <tag>")
    p.add_argument("--vertex", action="store_true",
                   help="also run the production feature/vertex fit on single-capture clusters")
    p.add_argument("--vertex-methods", default="optics,clusterfinder,peak_tof",
                   help="methods whose single-capture clusters get the vertex fit")
    p.add_argument("--geometry-frame", choices=["tank", "world_legacy"], default=None,
                   help="override geometry.frame (world_legacy = pre-fix features)")
    p.add_argument("--timing-offsets", choices=["on", "off"], default=None,
                   help="override geometry.apply_timing_offsets (off = correct for MC)")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild the composition report from saved parquets, no reprocessing")
    p.add_argument("--cache-only", action="store_true",
                   help="only read ROOT truth (captures, neutrino vertices) into parquet "
                        "caches; run this on a gpvm, then the heavy stage on EAF")
    p.add_argument("--tag", default="", help="suffix for every output name (pilots)")
    p.add_argument("--truth-unit", choices=["capture", "particle"], required=True,
                   help="capture = label clusters by true capture only; particle = build a "
                        "truth object for every source particle (captures, muon, Michel, ...)")
    p.add_argument("--owner-rule", choices=["plurality", "majority50"], default=None,
                   help="particle mode: cluster tag = top object (plurality) or only when it "
                        "owns >= 50%% of the hits (majority50, else 'mixed')")
    p.add_argument("--hit-cleaning", choices=["none", "unmatched_lowpe"], required=True,
                   help="none = all MCHit pulses; unmatched_lowpe = drop MCHit pulses with no waveform "
                        "pulse and PE < capture_truth.lowpe_cut before clustering")
    p.add_argument("--hit-window", choices=["delayed", "full"], required=True,
                   help="delayed = hits after residual.prompt_window_ns only; full = every hit, "
                        "so prompt muon/pion/proton/pi0 light is clustered and truth-tagged too")
    p.add_argument("--overlay", default=None, metavar="RANDOM_HITS_PARQUET",
                   help="add one random data trigger (ambe data randoms) to each MC event and "
                        "drop MC dark noise; needs --hit-cleaning unmatched_lowpe")
    p.add_argument("--cluster-features", action="store_true",
                   help="particle mode: add f_<feature> columns (compute_cluster_features) per cluster")
    p.add_argument("--scan-optics", action="store_true",
                   help="particle mode: add the capture_truth.optics_scan grid of OPTICS methods")
    p.add_argument("--scan-peak", action="store_true",
                   help="particle mode: add the capture_truth.peak_scan grid of time-peak methods")
    p.add_argument("--low-memory", action="store_true",
                   help="per-event capture/cluster lookups as slices of one sorted frame instead of "
                        "one DataFrame per event (same outputs; needed for 1,250 files on an 11 GB gpvm)")
    a = p.parse_args(list(argv) if argv else [])
    if a.truth_unit == "particle" and a.owner_rule is None:
        p.error("--truth-unit particle needs --owner-rule")
    if a.overlay and a.hit_cleaning != "unmatched_lowpe":
        p.error("--overlay adds recorded data hits, so the MC side needs --hit-cleaning unmatched_lowpe")
    if a.truth_unit == "capture" and (a.owner_rule or a.scan_peak or a.scan_optics or a.cluster_features):
        p.error("--owner-rule/--scan-peak/--scan-optics/--cluster-features only apply to --truth-unit particle")
    if a.groups and (a.max_files or a.merge_shards):
        p.error("--groups cannot be combined with --max-files or --merge-shards")
    if a.merge_shards and (a.truth_unit != "particle" or not a.tag or a.scan_optics or a.scan_peak):
        p.error("--merge-shards needs --truth-unit particle and --tag, without --scan-optics/--scan-peak")
    run(ctx, a)
