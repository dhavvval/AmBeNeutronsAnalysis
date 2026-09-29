"""
Time-domain neutron-candidate search, after the Super-K NTag Stage 1
(S. Han, PhD thesis, U. Tokyo 2023, §3.3).

A neutron capture is a burst of O(10) single-photon hits inside ~10-20 ns once the
photon travel time to a reference point is removed.  Dark noise is flat in time.  So
the candidate search is one-dimensional: slide a window of width `twidth` over the
(optionally ToF-corrected) hit times, open a candidate wherever the window holds at
least `nhitsth` hits, and keep only the local maxima that are at least `t_min_sep`
apart (non-maximum suppression, NMS).  Space is not used to form candidates at all; it
enters later as features.

Why time-only.  Density clustering in (x, y, z, t) can chain across microseconds
through dark-noise hits and absorbs background light into neutron clusters; it can
also cut one capture's light in two when a few hits sit just outside its density
threshold.  A 1-D NMS cannot merge two captures unless they fall inside `t_min_sep`
of each other (P ~ t_min_sep / tau ~ 0.2 % per pair at 50 ns / 29 us), and cannot
split one capture by construction.

`merge_clusters_in_time` applies the same NMS rule as a post-processing step to any
existing labelling (OPTICS, ClusterFinder), which is the cheap fix for split pairs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

SOL_WATER = 0.299792458 * 0.75          # m/ns, same constant as ambe.mc.optics

DEFAULTS = dict(
    twidth_ns=20.0,        # sliding-window width
    nhitsth=6,             # minimum hits in the window to open a candidate
    t_min_sep_ns=50.0,     # NMS: two accepted peaks are at least this far apart
    tcanwidth_ns=None,     # candidate half-membership width; None -> 1.5 * twidth
    deadtime_ns=0.0,       # same-PMT deadtime for hit cleaning; 0 disables
    pe_max=None,           # drop hits above this charge (single-photon signal); None keeps all
    nres_window_ns=100.0,  # +- window for the NResHits count
)


def clean_hits(t: np.ndarray, pmt: np.ndarray, pe: Optional[np.ndarray] = None,
               deadtime_ns: float = 0.0, pe_max: Optional[float] = None) -> np.ndarray:
    """Boolean mask of hits kept by the NTag hit cleaning.

    `deadtime_ns` gives every PMT the same deadtime: a hit is dropped if the same PMT
    already has a *kept* hit less than `deadtime_ns` earlier.  This suppresses
    afterpulse trains.  `pe_max` drops high-charge hits, which a capture's
    single-photon light does not produce.
    """
    keep = np.ones(len(t), dtype=bool)
    if pe_max is not None and pe is not None:
        keep &= pe <= pe_max
    if deadtime_ns > 0 and len(t):
        order = np.lexsort((t, pmt))
        last_t = {}
        for i in order:
            if not keep[i]:
                continue
            p = int(pmt[i])
            if p in last_t and t[i] - last_t[p] < deadtime_ns:
                keep[i] = False
            else:
                last_t[p] = t[i]
    return keep


def tof_corrected_times(t: np.ndarray, xyz: np.ndarray,
                        ref_m: Optional[np.ndarray]) -> np.ndarray:
    """t - |r_pmt - ref| / c_water, or t unchanged when there is no reference."""
    if ref_m is None:
        return np.asarray(t, dtype=float)
    ref = np.asarray(ref_m, dtype=float)
    return np.asarray(t, dtype=float) - np.linalg.norm(xyz - ref[None, :], axis=1) / SOL_WATER


def _nms(peak_t: np.ndarray, peak_n: np.ndarray, t_min_sep: float) -> np.ndarray:
    """Indices of accepted peaks: highest count first, ties broken by earlier time."""
    order = np.lexsort((peak_t, -peak_n))
    accepted: list[int] = []
    acc_t: list[float] = []
    for i in order:
        if all(abs(peak_t[i] - a) >= t_min_sep for a in acc_t):
            accepted.append(int(i))
            acc_t.append(float(peak_t[i]))
    return np.array(sorted(accepted, key=lambda k: peak_t[k]), dtype=int)


def find_time_peaks(t: np.ndarray,
                    xyz: Optional[np.ndarray] = None,
                    pmt: Optional[np.ndarray] = None,
                    pe: Optional[np.ndarray] = None,
                    ref_m: Optional[np.ndarray] = None,
                    twidth_ns: float = DEFAULTS["twidth_ns"],
                    nhitsth: int = DEFAULTS["nhitsth"],
                    t_min_sep_ns: float = DEFAULTS["t_min_sep_ns"],
                    tcanwidth_ns: Optional[float] = DEFAULTS["tcanwidth_ns"],
                    deadtime_ns: float = DEFAULTS["deadtime_ns"],
                    pe_max: Optional[float] = DEFAULTS["pe_max"],
                    ) -> tuple[np.ndarray, list[dict]]:
    """Label hits by time-peak candidate.

    Returns (labels, peaks).  labels[i] is the candidate index of hit i, or -1.
    Each peak dict holds t_peak (in the ToF-corrected time base), n_window (hits in
    the best window) and n_members.

    Membership: a hit belongs to the nearest accepted peak whose centre lies within
    tcanwidth/2 of the hit's corrected time.  Cleaned-away hits are always -1.
    """
    t = np.asarray(t, dtype=float)
    n = len(t)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels, []
    if tcanwidth_ns is None:
        tcanwidth_ns = 1.5 * twidth_ns
    if pmt is None:
        pmt = np.arange(n)

    keep = clean_hits(t, np.asarray(pmt), pe, deadtime_ns, pe_max)
    tc = tof_corrected_times(t, xyz, ref_m) if (ref_m is not None and xyz is not None) else t
    idx = np.where(keep)[0]
    if len(idx) < nhitsth:
        return labels, []

    ts = tc[idx]
    order = np.argsort(ts, kind="stable")
    ts_sorted = ts[order]
    # Hits in [t_i, t_i + twidth) for each start hit i.
    ends = np.searchsorted(ts_sorted, ts_sorted + twidth_ns, side="left")
    counts = ends - np.arange(len(ts_sorted))
    starts = np.where(counts >= nhitsth)[0]
    if not len(starts):
        return labels, []

    # A window's centre is the median of its hits: robust to one stray early hit.
    peak_t = np.array([np.median(ts_sorted[s:ends[s]]) for s in starts])
    peak_n = counts[starts]
    acc = _nms(peak_t, peak_n, t_min_sep_ns)
    centres = peak_t[acc]

    half = tcanwidth_ns / 2.0
    d = np.abs(ts[:, None] - centres[None, :])
    best = np.argmin(d, axis=1)
    inside = d[np.arange(len(ts)), best] <= half
    labels[idx[inside]] = best[inside]

    peaks = [dict(t_peak=float(centres[k]), n_window=int(peak_n[acc[k]]),
                  n_members=int((labels == k).sum())) for k in range(len(centres))]
    # A peak can lose all its members to a neighbour closer in time; drop and renumber.
    alive = [k for k, p in enumerate(peaks) if p["n_members"] > 0]
    if len(alive) < len(peaks):
        remap = {old: new for new, old in enumerate(alive)}
        labels = np.array([remap.get(l, -1) if l >= 0 else -1 for l in labels], dtype=int)
        peaks = [peaks[k] for k in alive]
    return labels, peaks


def n_residual_hits(t: np.ndarray, labels: np.ndarray, cluster_id: int,
                    window_ns: float = DEFAULTS["nres_window_ns"]) -> int:
    """NTag NResHits: hits within +-window of the cluster median, minus its members."""
    m = labels == cluster_id
    if not m.any():
        return 0
    tm = float(np.median(t[m]))
    return int((np.abs(t - tm) <= window_ns).sum() - m.sum())


def merge_clusters_in_time(labels: np.ndarray, t: np.ndarray,
                           t_min_sep_ns: float = DEFAULTS["t_min_sep_ns"]) -> np.ndarray:
    """Merge clusters whose median hit times are closer than t_min_sep_ns.

    Clusters are ordered by median time and chained: a cluster joins the previous
    group when its median is within t_min_sep_ns of that group's last member.  The
    returned labels are renumbered 0..K-1 in time order; -1 stays -1.
    """
    labels = np.asarray(labels, dtype=int)
    t = np.asarray(t, dtype=float)
    ids = [c for c in np.unique(labels) if c >= 0]
    if len(ids) < 2:
        return labels.copy()
    med = np.array([np.median(t[labels == c]) for c in ids])
    order = np.argsort(med, kind="stable")
    group = {}
    g = 0
    prev = None
    for k in order:
        if prev is not None and med[k] - prev >= t_min_sep_ns:
            g += 1
        group[ids[k]] = g
        prev = med[k]
    out = np.full(len(labels), -1, dtype=int)
    for c, gi in group.items():
        out[labels == c] = gi
    return out
