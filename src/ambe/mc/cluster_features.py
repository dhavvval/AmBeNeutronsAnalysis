"""
Cluster-level feature extraction for OPTICS clusters.

For each OPTICS cluster, computes a set of physically motivated discriminating
variables analogous to those used in SK neutron signal selection (Section IV of
arXiv:2505.04409):

    n_hits              Number of PMT hits in the cluster
    t_mean              Mean hit time (ns)
    sigma_t             RMS of raw hit times (ns)  [std-based, kept for comparison]
    sigma_t_corr        RMS after per-PMT timing offset correction (ns)
    sigma_t_tof         RMS after offset correction + time-of-flight correction (ns)
    sigma_t_mad         MAD-based robust sigma_t — raw (ns)  *** preferred for MVA ***
    sigma_t_mad_corr    MAD-based robust sigma_t — offset-corrected (ns)
    sigma_t_mad_tof     MAD-based robust sigma_t — ToF-corrected (ns)
    n_hits_early        Hits within ±10 ns of cluster median (direct Cherenkov only)
    sigma_t_early_mad   MAD of direct-light-window hit times (ns)
    t_window_80pct      Narrowest window (ns) containing 80% of cluster hits
    pe_total            Total photoelectrons in cluster (0 if hitPE unavailable)
    pe_balance          Charge balance: (max_quad - min_quad) / total PE  [0, 1]
    spatial_rms         RMS of hit PMT positions around cluster centroid (m)
    d_wall              Distance from estimated vertex to nearest tank wall (m)
    beta1               Legendre P1 isotropy:  0 = isotropic, >0 = directional
    beta2               Legendre P2 isotropy

Note on MAD: sigma_t_mad = 1.4826 × median(|t − median(t)|).  For Gaussian
data this equals sigma_t.  For data with 1–2 thermalization-scatter outlier
hits at microsecond timescales, sigma_t can be inflated 100×; sigma_t_mad
stays correct because the median is insensitive to outliers.

Truth labels (MC only):
    is_truth_neutron    1 if cluster matches a truth neutron (majority vote ≥50%)
    dominant_trackID    Matched neutron trackID (-1 if spurious)

Usage
-----
Run as part of the MC pipeline after `ambe mc optics`:
    ambe mc features --config configs/mc_local_trial.yaml

Or import directly:
    from ambe.mc.cluster_features import extract_all_features, run
    df_feat = extract_all_features(ctx, min_samples=8, xi=0.10, t_unit_ns=25.0)

Config block (add to your YAML under `features:`)
--------------------------------------------------
features:
  min_samples: 8          # OPTICS config to use (best from grid sweep)
  xi: 0.10
  t_unit_ns: 25.0
  hit_prefilter_ns: 0     # 0 = no prefilter (recommended)
  truth_window_ns: 75.0

geometry:
  pmt_geometry:    /path/to/FullTankPMTGeometry.csv
  timing_offsets:  /path/to/TankPMTTimingOffsets.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

from ..context import RunContext
from ..plotting import save_plot, set_style
from .optics import (
    NEUTRON_PDG, MIN_MATCH_FRAC, DEFAULT_T_UNIT, DEFAULT_WINDOW,
    run_optics_on_event, _apply_truth_window, _prefilter_hits,
    assign_clusterfinder_labels,
)


# Speed of light in water (m/ns) — matches processor.py / data/processor.py
SOL_WATER = 0.299792458 * 0.75

# Clusters whose mean time offset from the per-event neutron capture reference
# is earlier than this threshold are tagged as prompt gamma clusters.
# The AmBe 4.44 MeV gamma arrives ~18,000 ns before capture; -500 ns sits
# cleanly in the gap between Population 1 (gamma, ~-18k ns) and Population 2
# (near-capture, ~-1.4 ns).  Confirmed May 2026 from Lucho file analysis.
NEUTRON_CLASSES            = {1, 2, 3, 4}
GAMMA_CLUSTER_THRESHOLD_NS = -500.0


# --------------------------------------------------------------------------- #
# Geometry loading
# --------------------------------------------------------------------------- #

@dataclass
class ANNIEGeometry:
    """PMT positions, directions and timing offsets for ANNIE tank."""
    # chankey → (x, y, z) position in metres
    pmt_pos:    Dict[int, np.ndarray]
    # chankey → (xd, yd, zd) inward-facing unit direction
    pmt_dir:    Dict[int, np.ndarray]
    # chankey → timing offset in ns (fallback: mean of reliable offsets)
    pmt_offset: Dict[int, float]
    # Tank boundary parameters (metres)
    tank_radius: float
    tank_top:    float
    tank_bot:    float
    tank_center: np.ndarray   # (cx, cy, cz)
    fallback_offset: float    # mean reliable offset (used when chankey missing)


def load_geometry(geo_path: str, offsets_path: str) -> ANNIEGeometry:
    """
    Load PMT geometry and per-channel timing offsets.

    Parameters
    ----------
    geo_path :
        Path to FullTankPMTGeometry.csv
    offsets_path :
        Path to TankPMTTimingOffsets.csv
    """
    # ---- Geometry ----
    geo = pd.read_csv(geo_path, skiprows=[0, 2, 3], header=0)
    geo.columns = [c.strip() for c in geo.columns]
    for col in ["x_pos","y_pos","z_pos","xd:=x_dir","yd:=y_dir","zd:=z_dir",
                "x_dir","y_dir","z_dir","detector_num","channel_num"]:
        if col in geo.columns:
            geo[col] = pd.to_numeric(geo[col], errors="coerce")
    # Rename direction columns if they were read with old names
    col_map = {}
    for candidate in ["x_dir","y_dir","z_dir","xd","yd","zd"]:
        if candidate in geo.columns:
            col_map[candidate] = candidate
    geo = geo.dropna(subset=["x_pos","y_pos","z_pos","channel_num"])

    pmt_pos, pmt_dir = {}, {}
    for _, row in geo.iterrows():
        ck = int(row["channel_num"])
        pmt_pos[ck] = np.array([float(row["x_pos"]),
                                 float(row["y_pos"]),
                                 float(row["z_pos"])], dtype=float)
        # Direction columns may be named x_dir/y_dir/z_dir
        xd = float(row.get("x_dir", row.get("xd", 0)))
        yd = float(row.get("y_dir", row.get("yd", 0)))
        zd = float(row.get("z_dir", row.get("zd", 0)))
        pmt_dir[ck] = np.array([xd, yd, zd], dtype=float)

    # Tank boundaries from PMT positions
    xs = np.array([v[0] for v in pmt_pos.values()])
    ys = np.array([v[1] for v in pmt_pos.values()])
    zs = np.array([v[2] for v in pmt_pos.values()])
    xzr = np.sqrt(xs**2 + zs**2)  # radial distance in XZ plane

    # Centre of tank in XZ is where barrel PMTs converge; Y is midpoint
    cx = float(np.median(xs))
    cy = float((ys.max() + ys.min()) / 2)
    cz = float(np.median(zs))
    tank_center = np.array([cx, cy, cz])
    tank_radius  = float(xzr.max()) + 0.01   # slight margin
    tank_top     = float(ys.max())  + 0.01
    tank_bot     = float(ys.min())  - 0.01

    # ---- Timing offsets ----
    toff = pd.read_csv(offsets_path, comment="#", header=None,
                       names=["channel","location","offset_ns","offset_std_ns","notes"])
    reliable = toff[~toff["notes"].str.contains("unreliable", na=False)]
    fallback  = float(reliable["offset_ns"].mean()) if len(reliable) else 10.0

    pmt_offset: Dict[int, float] = {}
    for _, row in toff.iterrows():
        try:
            ck = int(row["channel"])
            pmt_offset[ck] = float(row["offset_ns"])
        except (ValueError, TypeError):
            pass

    print(f"[cluster_features] geometry: {len(pmt_pos)} PMTs  "
          f"| tank R={tank_radius:.3f}m  Y=[{tank_bot:.3f}, {tank_top:.3f}]m")
    print(f"[cluster_features] offsets: {len(pmt_offset)} channels  "
          f"| fallback={fallback:.1f} ns")

    return ANNIEGeometry(
        pmt_pos=pmt_pos, pmt_dir=pmt_dir, pmt_offset=pmt_offset,
        tank_radius=tank_radius, tank_top=tank_top, tank_bot=tank_bot,
        tank_center=tank_center, fallback_offset=fallback,
    )


# --------------------------------------------------------------------------- #
# Per-cluster feature computation
# --------------------------------------------------------------------------- #

def _beta_k(uvecs: np.ndarray, k: int) -> float:
    """
    Compute Legendre polynomial isotropy parameter β_k.

    uvecs : (N, 3) unit vectors from estimated vertex to each hit PMT.
    k     : order (1 or 2).

    β_k = (2k+1)/N² × Σᵢ,ⱼ P_k(cos θᵢⱼ)

    Interpretation:
      β₁ ≈ 0   → isotropic (random background / dark noise)
      β₁ > 0   → hits concentrated on one hemisphere (directional Cherenkov)
    """
    N = len(uvecs)
    if N < 2:
        return np.nan
    cos_th = np.clip(uvecs @ uvecs.T, -1.0, 1.0)   # (N, N)
    if k == 1:
        Pk = cos_th
    elif k == 2:
        Pk = (3.0 * cos_th**2 - 1.0) / 2.0
    else:
        raise ValueError(f"k={k} not implemented (use 1 or 2)")
    return float((2*k + 1) / (N * N) * Pk.sum())


def compute_cluster_features(df_cluster: pd.DataFrame,
                             geo: ANNIEGeometry,
                             source_pos_m: "Optional[np.ndarray]" = None) -> dict:
    """
    Compute all features for a single OPTICS cluster.

    Parameters
    ----------
    df_cluster : DataFrame slice — all hits in one cluster.
                 Required columns: x, y, z, t, pe, pmtID
    geo        : ANNIEGeometry (PMT positions, offsets, tank dimensions)

    Returns
    -------
    dict of feature name → float value
    """
    n = len(df_cluster)
    if n == 0:
        return {}

    xyz = df_cluster[["x", "y", "z"]].to_numpy(float)
    t   = df_cluster["t"].to_numpy(float)
    pe  = df_cluster["pe"].to_numpy(float) if "pe" in df_cluster.columns else np.ones(n)
    ids = df_cluster["pmtID"].to_numpy(int) if "pmtID" in df_cluster.columns else np.zeros(n, int)

    pe_total = float(pe.sum())

    # ---- Vertex estimate ----
    # PE-weighted centroid if charge available, else simple mean
    if pe_total > 0:
        weights = pe / pe_total
        vertex = (xyz * weights[:, None]).sum(axis=0)
    else:
        vertex = xyz.mean(axis=0)

    # ---- Time features ----
    t_mean   = float(t.mean())
    sigma_t  = float(t.std()) if n > 1 else 0.0

    # MAD-based sigma_t: robust to Geant4 thermalization outlier hits at µs scale.
    # sigma_t_mad = 1.4826 × median(|t - median(t)|)  →  consistent estimator of σ
    # for Gaussian data, but insensitive to 1-2 outlier hits that inflate std().
    if n > 1:
        med_t = np.median(t)
        sigma_t_mad = 1.4826 * float(np.median(np.abs(t - med_t)))
    else:
        sigma_t_mad = 0.0

    # Per-PMT timing offset correction
    offsets = np.array([geo.pmt_offset.get(int(ck), geo.fallback_offset)
                        for ck in ids], dtype=float)
    t_corr  = t - offsets
    sigma_t_corr = float(t_corr.std()) if n > 1 else 0.0
    if n > 1:
        med_t_corr = np.median(t_corr)
        sigma_t_mad_corr = 1.4826 * float(np.median(np.abs(t_corr - med_t_corr)))
    else:
        sigma_t_mad_corr = 0.0

    # Time-of-flight correction  (photon travel from vertex to each PMT)
    dist = np.linalg.norm(xyz - vertex[np.newaxis, :], axis=1)
    tof  = dist / SOL_WATER
    t_tof = t_corr - tof
    sigma_t_tof = float(t_tof.std()) if n > 1 else 0.0
    if n > 1:
        med_t_tof = np.median(t_tof)
        sigma_t_mad_tof = 1.4826 * float(np.median(np.abs(t_tof - med_t_tof)))
    else:
        sigma_t_mad_tof = 0.0

    # ---- Spatial spread ----
    spatial_rms = float(np.sqrt(np.mean(np.sum((xyz - vertex[np.newaxis,:])**2, axis=1))))

    # ---- Distance to wall ----
    v_xz  = vertex[[0, 2]] - geo.tank_center[[0, 2]]
    r_vtx = float(np.linalg.norm(v_xz))
    d_barrel = geo.tank_radius - r_vtx
    d_top    = geo.tank_top    - float(vertex[1])
    d_bot    = float(vertex[1]) - geo.tank_bot
    d_wall   = float(min(d_barrel, d_top, d_bot))

    # ---- Distance from vertex to known AmBe source position ----
    # d_source: neutrons travel 10-100 cm from source before capture.
    # Useful AmBe-specific discriminant: spurious clusters have no such constraint.
    # source_pos_m must be in metres, same coordinate system as hit positions.
    if source_pos_m is not None:
        src = np.asarray(source_pos_m, dtype=float)
        d_source = float(np.linalg.norm(vertex - src))
    else:
        d_source = np.nan

    # ---- Charge balance (4 quadrants in XZ plane) ----
    if pe_total > 0 and n >= 4:
        dx = xyz[:, 0] - vertex[0]
        dz = xyz[:, 2] - vertex[2]
        quadrant = ((dz >= 0).astype(int) * 2 + (dx >= 0).astype(int))
        quad_pe  = np.array([pe[quadrant == q].sum() for q in range(4)])
        pe_bal   = float((quad_pe.max() - quad_pe.min()) / pe_total)
    else:
        pe_bal = np.nan

    # ---- Isotropy (beta parameters) ----
    # Unit vectors from vertex to each hit PMT
    vecs  = xyz - vertex[np.newaxis, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 1e-6, norms, 1e-6)
    uvecs = vecs / norms

    beta1 = _beta_k(uvecs, k=1)
    beta2 = _beta_k(uvecs, k=2)

    # ---- Direct-light window features (P3) ----
    # n_hits_early: hits within the first 20 ns of the cluster (direct Cherenkov
    # only, cuts reflections).  Uses offset-corrected times centred on the
    # cluster median to be robust to absolute time offset.
    # 20 ns ≈ maximum direct photon travel time across the tank diagonal.
    DIRECT_LIGHT_NS = 20.0
    if n > 1:
        t_corr_med = float(np.median(t_corr))
        early_mask = np.abs(t_corr - t_corr_med) <= DIRECT_LIGHT_NS / 2.0
        n_hits_early = int(early_mask.sum())
        if n_hits_early > 1:
            t_early = t_corr[early_mask]
            med_early = np.median(t_early)
            sigma_t_early_mad = 1.4826 * float(np.median(np.abs(t_early - med_early)))
        else:
            sigma_t_early_mad = 0.0
    else:
        n_hits_early = n
        sigma_t_early_mad = 0.0

    # t_window_80pct: narrowest time window (ns) containing 80% of cluster hits.
    # Does not require a vertex estimate — purely timing-based cluster compactness.
    if n >= 2:
        t_sorted = np.sort(t_corr)
        k80 = max(1, int(np.ceil(0.8 * n)))   # number of hits in 80%
        # slide a window of width k80 and find minimum span
        windows = t_sorted[k80 - 1:] - t_sorted[:n - k80 + 1]
        t_window_80pct = float(windows.min()) if len(windows) > 0 else float(t_sorted[-1] - t_sorted[0])
    else:
        t_window_80pct = 0.0

    return {
        "n_hits":              n,
        "t_mean":              t_mean,
        # std-based (kept for backward compatibility and comparison)
        "sigma_t":             sigma_t,
        "sigma_t_corr":        sigma_t_corr,
        "sigma_t_tof":         sigma_t_tof,
        # MAD-based (robust — preferred for MVA)
        "sigma_t_mad":         sigma_t_mad,
        "sigma_t_mad_corr":    sigma_t_mad_corr,
        "sigma_t_mad_tof":     sigma_t_mad_tof,
        # Direct-light window
        "n_hits_early":        n_hits_early,
        "sigma_t_early_mad":   sigma_t_early_mad,
        "t_window_80pct":      t_window_80pct,
        # Charge / spatial
        "pe_total":            pe_total,
        "pe_balance":          pe_bal,
        "spatial_rms":         spatial_rms,
        "d_wall":              d_wall,
        "d_source":            d_source,
        # Isotropy
        "beta1":               beta1,
        "beta2":               beta2,
        # Vertex
        "vtx_x":               float(vertex[0]),
        "vtx_y":               float(vertex[1]),
        "vtx_z":               float(vertex[2]),
    }


def _hit_composition(mask: np.ndarray, df_event: pd.DataFrame) -> dict:
    """
    For one cluster defined by boolean mask, count hits by truth_class and
    return a composition dict.

    Returns
    -------
    dict with keys:
        n_neutron       hits from neutron classes (1,2,3,4)
        n_darknoise     hits from class 0
        n_nonneutron    hits from class -5
        n_untraced      hits with is_untraced==1 (BackTracker could not trace)
        frac_neutron    n_neutron / total
        frac_darknoise  n_darknoise / total
        frac_nonneutron n_nonneutron / total
        frac_untraced   n_untraced / total
        dominant_class  class id of the most common component
    """
    cls       = df_event["truth_class"].to_numpy(int)[mask]
    untraced  = df_event["is_untraced"].to_numpy(int)[mask] \
                if "is_untraced" in df_event.columns else np.zeros(mask.sum(), int)
    n = int(mask.sum())
    if n == 0:
        return {k: 0 for k in ["n_neutron","n_darknoise","n_nonneutron","n_untraced",
                                "frac_neutron","frac_darknoise","frac_nonneutron",
                                "frac_untraced","dominant_class"]}

    n_neutron    = int(np.isin(cls, [1, 2, 3, 4]).sum())
    n_darknoise  = int((cls == 0).sum())
    n_nonneutron = int((cls == -5).sum())
    n_untraced   = int(untraced.sum())

    counts = {c: int((cls == c).sum()) for c in np.unique(cls)}
    dominant_class = int(max(counts, key=counts.get))

    return {
        "n_neutron":     n_neutron,
        "n_darknoise":   n_darknoise,
        "n_nonneutron":  n_nonneutron,
        "n_untraced":    n_untraced,
        "frac_neutron":   round(n_neutron    / n, 4),
        "frac_darknoise": round(n_darknoise  / n, 4),
        "frac_nonneutron":round(n_nonneutron / n, 4),
        "frac_untraced":  round(n_untraced   / n, 4),
        "dominant_class": dominant_class,
    }


def _truth_label_cluster(mask: np.ndarray,
                         df_event: pd.DataFrame,
                         truth_neutron_mask: np.ndarray) -> Tuple[int, int]:
    """
    Label one cluster as truth neutron or spurious.

    Returns (is_truth_neutron, dominant_trackID).
    Matches the same logic as _optics_cluster_outcomes in optics.py:
    a cluster is matched if its dominant neutron trackID contributes
    >= MIN_MATCH_FRAC of all member hits.
    """
    track_ids = df_event["ancestor_trackID"].to_numpy(int)
    pdgs      = df_event["ancestor_pdg"].to_numpy(int)

    # Truth neutron trackIDs in this event (limited to in-window hits)
    truth_tids = set(
        int(tid)
        for tid, pdg, inw in zip(track_ids, pdgs, truth_neutron_mask)
        if pdg == NEUTRON_PDG and tid >= 0 and inw
    )

    n_total = int(mask.sum())
    neutron_mask_c = mask & (pdgs == NEUTRON_PDG) & (track_ids >= 0)
    if not neutron_mask_c.any():
        return 0, -1

    tids_in, counts = np.unique(track_ids[neutron_mask_c], return_counts=True)
    dom_tid  = int(tids_in[np.argmax(counts)])
    dom_frac = float(counts.max()) / n_total

    if dom_tid in truth_tids and dom_frac >= MIN_MATCH_FRAC:
        return 1, dom_tid
    return 0, -1


# --------------------------------------------------------------------------- #
# Main extraction driver
# --------------------------------------------------------------------------- #

def extract_all_features(ctx: RunContext,
                         geo: ANNIEGeometry,
                         min_samples:      int   = 8,
                         xi:               float = 0.10,
                         t_unit_ns:        float = DEFAULT_T_UNIT,
                         hit_prefilter_ns: float = 0.0,
                         truth_window_ns:  float = DEFAULT_WINDOW,
                         source_pos_m:     "Optional[np.ndarray]" = None,
                         min_pulses_per_event: int = 3) -> pd.DataFrame:
    """
    Re-run OPTICS with a single config and compute cluster features per event.

    Parameters
    ----------
    geo              : pre-loaded ANNIEGeometry
    min_samples, xi, t_unit_ns : OPTICS hyperparameters (use best from grid)
    hit_prefilter_ns : CF-time prefilter (0 = disabled, recommended)
    truth_window_ns  : truth signal window for labelling
    source_pos_m     : AmBe source position in metres for ToF correction
                       and d_source feature.  None disables both.
    """
    pulses_path   = ctx.parquet_path(f"{ctx.run_name}__pulses")
    clusters_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    if not pulses_path.exists():
        raise FileNotFoundError(f"{pulses_path} not found — run `ambe mc process` first")

    pulses   = pd.read_parquet(pulses_path)
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    # Ensure pe column exists (fallback to 1.0 if processor pre-dates this change)
    if "pe" not in pulses.columns:
        pulses["pe"] = 1.0
        print("[cluster_features] WARN: 'pe' column missing — "
              "re-run `ambe mc process` to include hitPE.  Defaulting to 1.0.")

    rows = []
    event_ids = pulses["eventID"].unique()
    src_str = (f"({source_pos_m[0]:.2f}, {source_pos_m[1]:.2f}, "
               f"{source_pos_m[2]:.2f}) m"
               if source_pos_m is not None else "none")
    print(f"[cluster_features] {len(event_ids)} events  |  "
          f"OPTICS config: ms={min_samples} xi={xi} t_unit={t_unit_ns}ns  "
          f"prefilter={hit_prefilter_ns}ns  source_tof={src_str}")

    for evid in event_ids:
        df_ev = pulses[pulses["eventID"] == evid].reset_index(drop=True)
        if len(df_ev) < min_pulses_per_event:
            continue

        df_cl = (clusters[clusters["eventID"] == evid].reset_index(drop=True)
                 if clusters is not None else None)

        # Pre-filter and truth mask (same logic as optics.py)
        if hit_prefilter_ns > 0:
            df_optics = _prefilter_hits(df_ev, df_cl, hit_prefilter_ns)
        else:
            df_optics = df_ev

        truth_mask = _apply_truth_window(df_ev, truth_window_ns)

        # Run OPTICS (with source ToF correction if source_pos_m provided)
        labels_optics = run_optics_on_event(df_optics, min_samples=min_samples,
                                            xi=xi, t_unit_ns=t_unit_ns,
                                            source_pos_m=source_pos_m)

        # Re-align to full event
        if len(df_optics) < len(df_ev):
            labels_full = np.full(len(df_ev), -1, dtype=int)
            labels_full[df_optics.index] = labels_optics
        else:
            labels_full = labels_optics

        # Also get CF labels for comparison
        cf_labels = (assign_clusterfinder_labels(df_ev, df_cl)
                     if df_cl is not None else np.full(len(df_ev), -1, dtype=int))

        # Per-event neutron capture reference time (for gamma-cluster tagging)
        neu_hits_t = df_ev.loc[df_ev["truth_class"].isin(NEUTRON_CLASSES), "t"]
        neu_ref_t  = float(neu_hits_t.median()) if len(neu_hits_t) >= 1 else float("nan")

        # Extract features per cluster — both OPTICS and CF
        for method, lbls in [("optics", labels_full), ("clusterfinder", cf_labels)]:
            unique_clusters = [c for c in np.unique(lbls) if c >= 0]
            for cid in unique_clusters:
                mask = (lbls == cid)
                df_clust = df_ev[mask].reset_index(drop=True)
                feats = compute_cluster_features(df_clust, geo,
                                                source_pos_m=source_pos_m)
                if not feats:
                    continue
                is_n, dom_tid = _truth_label_cluster(mask, df_ev, truth_mask)
                comp = _hit_composition(mask, df_ev)

                # Gamma-cluster flag: cluster arriving >500 ns before capture
                # = prompt AmBe 4.44 MeV gamma, not a false-positive spurious
                cluster_offset = (feats["t_mean"] - neu_ref_t
                                  if not (neu_ref_t != neu_ref_t) else float("nan"))
                is_gamma = int(cluster_offset < GAMMA_CLUSTER_THRESHOLD_NS
                               if cluster_offset == cluster_offset else False)

                row = {
                    "eventID":                 int(evid),
                    "method":                  method,
                    "cluster_id":              int(cid),
                    "is_truth_neutron":        is_n,
                    "dominant_trackID":        dom_tid,
                    "cluster_time_offset_ns":  round(cluster_offset, 1)
                                               if cluster_offset == cluster_offset
                                               else float("nan"),
                    "is_gamma_cluster":        is_gamma,
                }
                row.update(feats)
                row.update(comp)
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[cluster_features] extracted {len(df)} cluster records  "
          f"({df['method'].value_counts().to_dict()})")
    return df


# --------------------------------------------------------------------------- #
# Diagnostic plots
# --------------------------------------------------------------------------- #

FEATURE_LABELS = {
    # Core timing — std-based (kept for comparison)
    "n_hits":              "Number of PMT hits",
    "sigma_t":             "Hit time RMS — raw (ns)",
    "sigma_t_corr":        "Hit time RMS — offset-corrected (ns)",
    "sigma_t_tof":         "Hit time RMS — ToF-corrected (ns)",
    # MAD-based timing (preferred for MVA — robust to thermalization outliers)
    "sigma_t_mad":         "Hit time MAD — raw (ns)  [robust]",
    "sigma_t_mad_corr":    "Hit time MAD — offset-corrected (ns)  [robust]",
    "sigma_t_mad_tof":     "Hit time MAD — ToF-corrected (ns)  [robust]",
    # Direct-light window
    "n_hits_early":        "Hits within ±10 ns of cluster median (direct light)",
    "sigma_t_early_mad":   "Hit time MAD — direct-light window only (ns)",
    "t_window_80pct":      "Narrowest window containing 80% of hits (ns)",
    # Charge / spatial
    "pe_total":            "Total PE",
    "pe_balance":          "Charge balance  (max−min)/total",
    "spatial_rms":         "Spatial RMS of hit PMTs  (m)",
    "d_wall":              "Distance to nearest wall  (m)",
    "d_source":            "Distance from vertex to AmBe source  (m)",
    # Isotropy
    "beta1":               "Isotropy β₁  (Legendre P₁)",
    "beta2":               "Isotropy β₂  (Legendre P₂)",
}


def feature_plots(df: pd.DataFrame, ctx: RunContext):
    """
    For each feature, plot the distribution for truth-neutron clusters vs
    spurious clusters, separately for OPTICS and CF methods.
    """
    set_style()
    pdf_path = ctx.plots_dir / f"{ctx.run_name}__cluster_features.pdf"
    ctx.plots_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for feat, xlabel in FEATURE_LABELS.items():
            if feat not in df.columns:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
            fig.suptitle(f"{ctx.title(feat)}", fontsize=11)

            for ax, method in zip(axes, ["optics", "clusterfinder"]):
                sub = df[df["method"] == method]
                sig = sub[sub["is_truth_neutron"] == 1][feat].dropna()
                bkg = sub[sub["is_truth_neutron"] == 0][feat].dropna()

                if len(sig) == 0 and len(bkg) == 0:
                    ax.set_visible(False)
                    continue

                all_vals = pd.concat([sig, bkg])
                lo, hi  = float(all_vals.quantile(0.01)), float(all_vals.quantile(0.99))
                if lo == hi:
                    lo, hi = float(all_vals.min()), float(all_vals.max()) + 1e-6
                bins = np.linspace(lo, hi, 40)

                ax.hist(sig, bins=bins, density=True, alpha=0.6,
                        color="tomato",    label=f"Truth neutron  (n={len(sig)})")
                ax.hist(bkg, bins=bins, density=True, alpha=0.6,
                        color="steelblue", label=f"Spurious  (n={len(bkg)})")
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel("Density")
                ax.set_title(method.upper(), fontsize=10)
                ax.legend(fontsize=8)

                # Separation metric: (μ_sig - μ_bkg) / sqrt((σ_sig²+σ_bkg²)/2)
                if len(sig) > 1 and len(bkg) > 1:
                    sep = abs(sig.mean() - bkg.mean()) / np.sqrt((sig.std()**2 + bkg.std()**2) / 2 + 1e-9)
                    ax.text(0.97, 0.95, f"sep={sep:.2f}σ", transform=ax.transAxes,
                            ha="right", va="top", fontsize=8, color="black",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Correlation matrix of features for neutron clusters
        feat_cols = [f for f in FEATURE_LABELS if f in df.columns]
        for method in ["optics", "clusterfinder"]:
            sub_n = df[(df["method"] == method) & (df["is_truth_neutron"] == 1)][feat_cols].dropna()
            if len(sub_n) < 5:
                continue
            corr = sub_n.corr()
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(feat_cols))); ax.set_yticklabels(feat_cols, fontsize=8)
            plt.colorbar(im, ax=ax, label="Pearson r")
            ax.set_title(f"{method.upper()} — Feature correlations (truth neutron clusters)", fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[cluster_features] wrote plots → {pdf_path}")
    return pdf_path


def separation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a ranked table of feature separation power.
    Separates background into:
      - gamma clusters (is_gamma_cluster=1): prompt AmBe gamma, not false positives
      - real spurious  (is_truth_neutron=0 and is_gamma_cluster=0): true background
    Separation is computed against real spurious only.
    """
    has_gamma_flag = "is_gamma_cluster" in df.columns
    rows = []
    for feat in FEATURE_LABELS:
        if feat not in df.columns:
            continue
        for method in ["optics", "clusterfinder"]:
            sub = df[df["method"] == method]
            sig = sub[sub["is_truth_neutron"] == 1][feat].dropna()
            # Exclude gamma clusters from background for cleaner separation metric
            if has_gamma_flag:
                real_bkg = sub[(sub["is_truth_neutron"] == 0) &
                               (sub["is_gamma_cluster"] == 0)]
            else:
                real_bkg = sub[sub["is_truth_neutron"] == 0]
            bkg = real_bkg[feat].dropna()
            if len(sig) < 2 or len(bkg) < 2:
                continue
            sep = abs(sig.mean() - bkg.mean()) / np.sqrt((sig.std()**2 + bkg.std()**2) / 2 + 1e-9)
            rows.append({"feature": feat, "method": method,
                         "separation_sigma": round(sep, 3),
                         "mean_neutron":  round(sig.mean(), 4),
                         "mean_spurious": round(bkg.mean(), 4)})
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    return df_out.sort_values("separation_sigma", ascending=False)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def spurious_composition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method (optics / clusterfinder), break down the hit composition
    of spurious clusters (is_truth_neutron == 0) to understand what they are
    actually made of.

    Answers: "Are spurious clusters pure background, or do they contain
    real neutron hits that OPTICS failed to match?"

    Prints a detailed breakdown and returns a summary DataFrame.
    """
    print("\n" + "=" * 70)
    print("SPURIOUS CLUSTER HIT COMPOSITION")
    print("What are spurious clusters actually made of?")
    print("=" * 70)

    rows = []
    for method in ["optics", "clusterfinder"]:
        sub = df[df["method"] == method]
        n_total   = len(sub)
        spurious  = sub[sub["is_truth_neutron"] == 0]
        matched   = sub[sub["is_truth_neutron"] == 1]
        n_spur    = len(spurious)

        # Split spurious into gamma clusters vs real spurious
        has_gamma_flag = "is_gamma_cluster" in sub.columns
        if has_gamma_flag:
            gamma_clusters = spurious[spurious["is_gamma_cluster"] == 1]
            real_spurious  = spurious[spurious["is_gamma_cluster"] == 0]
        else:
            gamma_clusters = pd.DataFrame()
            real_spurious  = spurious

        print(f"\n[{method.upper()}]  {n_total} clusters total: "
              f"{len(matched)} matched, {n_spur} spurious")
        if has_gamma_flag and len(gamma_clusters) > 0:
            print(f"  ↳ Of spurious: {len(gamma_clusters)} prompt gamma clusters "
                  f"(offset < {GAMMA_CLUSTER_THRESHOLD_NS:.0f} ns) + "
                  f"{len(real_spurious)} real spurious")

        if n_spur == 0:
            print("  No spurious clusters.")
            continue

        # Mean composition of spurious clusters
        frac_cols = ["frac_neutron", "frac_darknoise", "frac_nonneutron", "frac_untraced"]
        n_cols    = ["n_neutron",    "n_darknoise",    "n_nonneutron",    "n_untraced"]
        means = spurious[frac_cols + n_cols].mean()

        print(f"\n  Mean hit fractions inside spurious clusters:")
        print(f"    Neutron (class 1-4):       {means['frac_neutron']:.3f}  "
              f"(mean {means['n_neutron']:.1f} hits/cluster)")
        print(f"    Non-neutron physics (−5):  {means['frac_nonneutron']:.3f}  "
              f"(mean {means['n_nonneutron']:.1f} hits/cluster)")
        print(f"    Dark noise (class 0):      {means['frac_darknoise']:.3f}  "
              f"(mean {means['n_darknoise']:.1f} hits/cluster)")
        print(f"    Untraced:                  {means['frac_untraced']:.3f}  "
              f"(mean {means['n_untraced']:.1f} hits/cluster)")

        # Classify spurious clusters by dominant component
        dom = spurious["dominant_class"].value_counts()
        print(f"\n  Spurious clusters by dominant hit class:")
        labels = {1: "Primary neutron", 2: "Secondary n←p", 3: "Secondary n←n",
                  4: "Secondary n←other", 0: "Dark noise", -5: "Non-neutron physics"}
        for cls_id, count in dom.sort_index().items():
            pct = 100 * count / n_spur
            print(f"    Class {cls_id:>3} [{labels.get(int(cls_id),'unknown'):25s}]: "
                  f"{count:5d}  ({pct:.1f}%)")

        # What fraction of spurious clusters contain ANY neutron hits?
        has_neutron = (spurious["n_neutron"] > 0).sum()
        print(f"\n  Spurious clusters containing ≥1 neutron hit: "
              f"{has_neutron} / {n_spur}  ({100*has_neutron/n_spur:.1f}%)")
        print(f"  → These are 'contaminated' clusters where a real neutron hit")
        print(f"    merged with background but wasn't the majority component.")

        # Pure vs mixed
        pure_bg  = (spurious["frac_neutron"] == 0).sum()
        mixed    = (spurious["frac_neutron"] > 0).sum()
        print(f"\n  Pure background (0% neutron):  {pure_bg} / {n_spur}  "
              f"({100*pure_bg/n_spur:.1f}%)")
        print(f"  Mixed (>0% neutron hits):      {mixed} / {n_spur}  "
              f"({100*mixed/n_spur:.1f}%)")

        if has_gamma_flag and len(real_spurious) > 0:
            real_means = real_spurious[frac_cols + n_cols].mean()
            print(f"\n  Real spurious only ({len(real_spurious)} clusters, "
                  f"excluding gamma):")
            print(f"    Neutron (class 1-4):       {real_means['frac_neutron']:.3f}  "
                  f"(mean {real_means['n_neutron']:.1f} hits/cluster)")
            print(f"    Non-neutron physics (−5):  {real_means['frac_nonneutron']:.3f}  "
                  f"(mean {real_means['n_nonneutron']:.1f} hits/cluster)")
            real_pur = len(matched) / (len(matched) + len(real_spurious))
            print(f"  Corrected purity (excl. gamma clusters): {real_pur:.3f}")

        rows.append({
            "method":               method,
            "n_spurious":           n_spur,
            "n_gamma_clusters":     len(gamma_clusters) if has_gamma_flag else 0,
            "n_real_spurious":      len(real_spurious)  if has_gamma_flag else n_spur,
            "mean_frac_neutron":    round(float(means["frac_neutron"]), 4),
            "mean_frac_nonneutron": round(float(means["frac_nonneutron"]), 4),
            "mean_frac_darknoise":  round(float(means["frac_darknoise"]), 4),
            "mean_frac_untraced":   round(float(means["frac_untraced"]), 4),
            "pct_dominant_nonneutron": round(100 * float(dom.get(-5, 0)) / n_spur, 1),
            "pct_dominant_darknoise":  round(100 * float(dom.get(0, 0)) / n_spur, 1),
            "pct_dominant_neutron":    round(100 * sum(float(dom.get(c, 0)) for c in [1,2,3,4]) / n_spur, 1),
            "pct_contains_any_neutron": round(100 * float(has_neutron) / n_spur, 1),
            "pct_pure_background":     round(100 * float(pure_bg) / n_spur, 1),
        })

    return pd.DataFrame(rows)


def _config_from_ctx(ctx: RunContext):
    feat_block = ctx.extra.get("features", {}) or {}
    geo_block  = ctx.extra.get("geometry",  {}) or {}

    ms        = int(feat_block.get("min_samples",      8))
    xi        = float(feat_block.get("xi",             0.10))
    t_unit    = float(feat_block.get("t_unit_ns",      DEFAULT_T_UNIT))
    prefilter = float(feat_block.get("hit_prefilter_ns", 0.0))
    truth_win = float(feat_block.get("truth_window_ns",  DEFAULT_WINDOW))

    geo_path  = geo_block.get("pmt_geometry",   None)
    off_path  = geo_block.get("timing_offsets", None)

    # Source position for ToF correction and d_source feature (metres).
    # Set via  features: source_position_m: [x, y, z]  in config YAML.
    # data/processor.py source_positions dict uses cm → divide by 100 in config.
    src_raw = feat_block.get("source_position_m", None)
    if src_raw is not None:
        source_pos_m = np.array([float(v) for v in src_raw], dtype=float)
        print(f"[cluster_features] source ToF / d_source enabled: "
              f"source at {tuple(source_pos_m)} m")
    else:
        source_pos_m = None

    return ms, xi, t_unit, prefilter, truth_win, geo_path, off_path, source_pos_m


def run(ctx: RunContext, argv: Optional[Iterable[str]] = None) -> Path:
    p = argparse.ArgumentParser(prog="ambe mc features")
    p.add_argument("--min-pulses", type=int,
                   default=int(ctx.cuts.get("min_pulses_per_event", 3)))
    args = p.parse_args(list(argv) if argv else [])

    ms, xi, t_unit, prefilter, truth_win, geo_path, off_path, src_pos = _config_from_ctx(ctx)

    if geo_path is None or off_path is None:
        raise ValueError(
            "geometry.pmt_geometry and geometry.timing_offsets must be set in config.\n"
            "Example:\n"
            "  geometry:\n"
            "    pmt_geometry: /path/to/FullTankPMTGeometry.csv\n"
            "    timing_offsets: /path/to/TankPMTTimingOffsets.csv"
        )

    geo = load_geometry(geo_path, off_path)

    df = extract_all_features(
        ctx, geo,
        min_samples=ms, xi=xi, t_unit_ns=t_unit,
        hit_prefilter_ns=prefilter, truth_window_ns=truth_win,
        source_pos_m=src_pos,
        min_pulses_per_event=args.min_pulses,
    )

    # Save parquet
    out_parquet = ctx.parquet_path(f"{ctx.run_name}__cluster_features")
    df.to_parquet(out_parquet, index=False)
    print(f"[cluster_features] wrote features  → {out_parquet}")

    # Save separation summary CSV
    sep = separation_summary(df)
    out_csv = ctx.csv_path(f"{ctx.run_name}__feature_separation")
    sep.to_csv(out_csv, index=False)
    print(f"[cluster_features] feature separation summary:")
    print(sep.to_string(index=False))
    print(f"[cluster_features] wrote separation → {out_csv}")

    # Spurious cluster composition breakdown
    comp_summary = spurious_composition_summary(df)
    comp_csv = ctx.csv_path(f"{ctx.run_name}__spurious_composition")
    comp_summary.to_csv(comp_csv, index=False)
    print(f"\n[cluster_features] wrote spurious composition → {comp_csv}")

    # Diagnostic plots
    feature_plots(df, ctx)

    return out_parquet


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)
