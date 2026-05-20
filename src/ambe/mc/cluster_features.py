"""
Cluster-level feature extraction for OPTICS clusters.

Physics features (computable from both MC and real data)
---------------------------------------------------------
Timing — std-based (kept for comparison with legacy tools):
    n_hits              Number of PMT hits in the cluster
    t_mean              Mean hit time (ns)
    sigma_t             RMS of raw hit times (ns)
    sigma_t_corr        RMS after per-PMT timing offset correction (ns)
    sigma_t_tof         RMS after offset + PE-weighted-centroid ToF correction (ns)

Timing — MAD-based (robust, preferred for MVA):
    sigma_t_mad         1.4826 × median(|t − median(t)|) on raw times (ns)
    sigma_t_mad_corr    Same after per-PMT timing offset correction (ns)
    sigma_t_mad_tof     Same after offset + PE-weighted-centroid ToF correction (ns)

Direct-light window:
    n_hits_early        Hits within ±10 ns of the offset-corrected cluster median
    sigma_t_early_mad   MAD of the direct-light-window hits only (ns)
    t_window_80pct      Narrowest window (ns) containing 80% of offset-corrected hits

Charge:
    pe_total            Total photoelectrons (0 if hitPE unavailable — re-run processor)
    pe_balance          Quadrant charge asymmetry: (max_quad − min_quad) / pe_total  [0,1]
    charge_bal_legacy   Legacy CB = sqrt(ΣQ²/(ΣQ)² − 1/121) per-PMT  [ClusterFinder convention]

Spatial:
    spatial_rms         RMS of hit PMT positions around PE-weighted centroid (m)
    d_wall              Distance from PE-weighted centroid to nearest tank surface (m)
    d_source            Distance from PE-weighted centroid to AmBe source (m); NaN if no source

Isotropy — SK Legendre convention, β_k = 2/(N(N−1)) Σᵢ≠ⱼ P_k(cos θᵢⱼ):
    beta1 … beta5       β₁–β₅ computed from directions to hit PMTs (PE-weighted centroid vertex)

PE-weighted centroid vertex:
    vtx_x, vtx_y, vtx_z   Centroid position in metres (Y = vertical axis)

Gauss-Newton fitted vertex (ported from VertexLeastSquares.cpp):
    vtx_fit_x/y/z       Best-fit vertex position (NaN if fit did not converge)
    fit_rms_ns          RMS of timing residuals at best-fit vertex (ns)
    fit_converged       1 if any seed produced an in-tank converged vertex, else 0
    n_fit_hits          Number of causally compatible hits used in the fit
    d_wall_fit          Distance to nearest wall — fitted vertex (m)
    d_source_fit        Distance to AmBe source — fitted vertex (m)
    beta1_fit … beta5_fit  β₁–β₅ using fitted-vertex directions
    sigma_t_mad_tof_fit    MAD of ToF-corrected times using fitted vertex (ns)
    fit_goodness_init   SK FitGoodness (eq. 3.1) evaluated at PE-weighted centroid
    fit_goodness_reco   SK FitGoodness evaluated at Gauss-Newton fitted vertex

Truth labels (MC only — not available for real data):
    is_truth_neutron    1 if cluster matches a truth neutron (dominant trackID ≥ MIN_MATCH_FRAC)
    dominant_trackID    Matched neutron trackID (-1 if spurious)
    cluster_time_offset_ns  t_mean − median(neutron hit times) for the event (ns)
    is_prompt_cluster   1 if cluster is a prompt signal (AmBe gamma or other early physics)
    n_neutron           Neutron hits (truth_class ∈ {1,2,3,4})
    n_darknoise         Dark-noise hits (truth_class == 0)
    n_nonneutron        Non-neutron physics hits (truth_class == −5; AmBe prompt gamma)
    n_untraced          Hits BackTracker could not trace (~11%)
    frac_neutron/darknoise/nonneutron/untraced   Hit fractions
    dominant_class      Most frequent truth_class in the cluster

Note on MAD
-----------
sigma_t_mad = 1.4826 × median(|t − median(t)|).  For Gaussian data this equals
sigma_t (std).  Geant4 thermalization tracking adds 1–2 outlier hits per cluster
at microsecond timescales; these inflate sigma_t by up to 100× while leaving
sigma_t_mad unaffected.  Use MAD variants for any MVA input.

Usage
-----
Run as part of the MC pipeline after `ambe mc optics`:
    ambe mc features --config configs/mc_local_trial.yaml

Or import directly:
    from ambe.mc.cluster_features import extract_all_features, run
    df_feat = extract_all_features(ctx, geo, min_samples=8, xi=0.10, t_unit_ns=25.0)

Config block (add to your YAML under `features:`)
--------------------------------------------------
features:
  min_samples: 8          # OPTICS config to use (best from grid sweep)
  xi: 0.10
  t_unit_ns: 25.0
  hit_prefilter_ns: 0     # 0 = no prefilter (required for clean sigma_t)
  truth_window_ns: 75.0
  source_position_m: [x, y, z]   # AmBe source in metres; omit to disable ToF/d_source

geometry:
  pmt_geometry:    /path/to/FullTankPMTGeometry.csv
  timing_offsets:  /path/to/TankPMTTimingOffsets.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from ..context import RunContext
from ..plotting import save_plot, set_style
from .optics import (
    NEUTRON_PDG, MIN_MATCH_FRAC, DEFAULT_T_UNIT, DEFAULT_WINDOW,
    PROMPT_SIGNAL_THRESHOLD_NS, PROMPT_SIGNAL_MAX_NEUTRON_HITS,
    run_optics_on_event, _apply_truth_window, _prefilter_hits,
    assign_clusterfinder_labels,
)


# Speed of light in water (m/ns)
SOL_WATER = 0.299792458 * 0.75

NEUTRON_CLASSES = {1, 2, 3, 4}

# Plot colour palette — change here to affect all cluster-feature plots
C_SIG = "#0077BB"   # signal (truth neutron)
C_GAM = "#EE7733"   # prompt gamma clusters
C_SPU = "#BBBBBB"   # real spurious (true false positives)


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
    Legendre isotropy parameter β_k (SK convention, arXiv:2505.04409 eq. 3.4).

    uvecs : (N, 3) unit vectors from estimated vertex to each hit PMT.
    k     : Legendre degree (1–5).

    β_k = 2/(N*(N-1)) × Σᵢ≠ⱼ P_k(cos θᵢⱼ)

    β_k > 0  →  hits clustered on one side (directional Cherenkov cone)
    β_k ≈ 0  →  isotropic (random background / dark noise)
    β_k < 0  →  hits anticorrelated (two opposing clusters)
    """
    N = len(uvecs)
    if N < 2:
        return np.nan
    cos_th = np.clip(uvecs @ uvecs.T, -1.0, 1.0)   # (N, N)
    if k == 1:
        Pk = cos_th
    elif k == 2:
        Pk = (3.0 * cos_th**2 - 1.0) / 2.0
    elif k == 3:
        Pk = (5.0 * cos_th**3 - 3.0 * cos_th) / 2.0
    elif k == 4:
        Pk = (35.0 * cos_th**4 - 30.0 * cos_th**2 + 3.0) / 8.0
    elif k == 5:
        Pk = (63.0 * cos_th**5 - 70.0 * cos_th**3 + 15.0 * cos_th) / 8.0
    else:
        raise ValueError(f"k={k} not implemented (use 1–5)")
    # P_k(1) = 1 for all k, so the diagonal (i=j) contributes exactly N.
    off_diag = float(Pk.sum()) - N
    return float(2.0 / (N * (N - 1)) * off_diag)


# --------------------------------------------------------------------------- #
# Gauss-Newton vertex fitter
# Ported from VertexLeastSquares.cpp by A. Sutton.
# --------------------------------------------------------------------------- #

_PHI_SQ = ((1.0 + np.sqrt(5.0)) / 2.0) ** 2   # golden-ratio squared for sunflower


def _filter_hits_for_vertex(xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Select causally compatible hits for vertex fitting (port of FilterHitsMC).

    For each hit i, finds all hits j where |t_i - t_j| <= |r_i - r_j| / c_w
    (photons from a common point source cannot arrive in a time shorter than
    the PMT-to-PMT ToF).  Keeps the largest mutually compatible set, then
    restricts to hits within 10 ns of the earliest hit (removes reflections).

    Returns boolean mask of length N.
    """
    n = len(t)
    if n < 4:
        return np.ones(n, dtype=bool)

    diff    = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]   # (N, N, 3)
    pmt_tof = np.linalg.norm(diff, axis=2) / SOL_WATER         # (N, N)

    dt      = np.abs(t[:, np.newaxis] - t[np.newaxis, :])      # (N, N)
    compat  = dt <= pmt_tof

    best    = int(np.argmax(compat.sum(axis=1)))
    mask    = compat[best].copy()
    t0      = float(t[mask].min())
    mask   &= np.abs(t - t0) <= 10.0
    return mask


def _generate_seed_vertices(geo: ANNIEGeometry,
                             y_spacing: float = 0.5,
                             n_planar:  int   = 15,
                             n_boundary: int  = 1,
                             rng_seed:  int   = 0) -> np.ndarray:
    """
    Sunflower seed grid for Gauss-Newton initialisation (port of GenerateVetices).

    Parameters
    ----------
    y_spacing  : vertical spacing between Y levels (metres)
    n_planar   : number of XZ points per Y level
    n_boundary : boundary ring count (passed through to sunflower formula)
    rng_seed   : numpy random seed for per-level rotations (fixed → reproducible)

    Returns (N_seeds, 3) array of positions in metres.
    """
    rng  = np.random.default_rng(rng_seed)
    ys   = np.arange(geo.tank_bot, geo.tank_top + y_spacing / 2.0, y_spacing)

    xs_raw, zs_raw = [], []
    for n in range(1, n_planar + 1):
        denom = max(n_planar - (n_boundary + 1.0) / 2.0, 1e-9)
        rad   = 1.0 if n > n_planar + n_boundary else np.sqrt((n + 0.5) / denom)
        rad   = min(rad, 1.0) * geo.tank_radius
        angle = 2.0 * np.pi * n / _PHI_SQ
        xs_raw.append(rad * np.cos(angle))
        zs_raw.append(rad * np.sin(angle))
    xs_raw = np.array(xs_raw)
    zs_raw = np.array(zs_raw)

    cx, cz = geo.tank_center[0], geo.tank_center[2]
    seeds  = []
    for y in ys:
        rot     = rng.uniform(0.0, 2.0 * np.pi)
        cr, sr  = np.cos(rot), np.sin(rot)
        xr      = xs_raw * cr - zs_raw * sr + cx
        zr      = zs_raw * cr + xs_raw * sr + cz
        for xi, zi in zip(xr, zr):
            seeds.append([xi, y, zi])
    return np.array(seeds)


def _in_tank(pos: np.ndarray, geo: ANNIEGeometry) -> bool:
    """True if pos is within the cylindrical tank boundary."""
    v_xz = pos[[0, 2]] - geo.tank_center[[0, 2]]
    return (float(np.linalg.norm(v_xz)) < geo.tank_radius and
            geo.tank_bot < float(pos[1]) < geo.tank_top)


def _jac_residual(guess: np.ndarray, pmt_xyz: np.ndarray, t: np.ndarray,
                  regularizer: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmented Jacobian A and residual b for one Gauss-Newton step.
    Port of EvalAtGuessVertexMC.

    f_i(vtx, T) = dist_i/c + T - t_i = 0

    Jacobian row i:
      [∂f/∂x, ∂f/∂y, ∂f/∂z, ∂f/∂T] = [(vtx-r_i)/(dist*c), 1]

    Emission time T is analytically eliminated per iteration as mean(t_i - ToF_i).
    Residual b[i] = (t_i - ToF_i) - mean_T  →  -f_i with T = mean_T.

    Rows N..N+3 are Tikhonov regularization: λI to keep the step bounded.
    Returns A (N+4, 4) and b (N+4,).
    """
    n    = len(t)
    diff = pmt_xyz - guess[np.newaxis, :]               # r_det - r_vtx  (N, 3)
    dist = np.linalg.norm(diff, axis=1)
    dist = np.where(dist < 1e-6, 1e-6, dist)

    A         = np.zeros((n + 4, 4))
    A[:n, :3] = -diff / (dist[:, np.newaxis] * SOL_WATER)  # (vtx - r_det)/(dist*c)
    A[:n,  3] = 1.0

    tof    = dist / SOL_WATER
    mean_T = float(np.mean(t - tof))
    b      = np.zeros(n + 4)
    b[:n]  = (t - tof) - mean_T

    for k in range(4):
        A[n + k, k] = regularizer

    return A, b


def fit_vertex_gauss_newton(
        xyz:           np.ndarray,
        t:             np.ndarray,
        geo:           ANNIEGeometry,
        regularizer:   float = 0.05,
        break_dist_m:  float = 0.005,
        max_iter:      int   = 100,
        y_spacing:     float = 0.5,
        n_planar:      int   = 15,
) -> Tuple[np.ndarray, float, bool, int]:
    """
    Gauss-Newton vertex fitter for neutron capture clusters.

    Port of VertexLeastSquares::RunLoopMC.  Generates a sunflower grid of seed
    vertices, runs Gauss-Newton from each seed, and returns the fitted vertex
    with the smallest RMS of timing residuals.

    Parameters
    ----------
    xyz           : (N, 3) PMT positions for all cluster hits (metres)
    t             : (N,)   offset-corrected hit times (ns)
    geo           : ANNIEGeometry for containment checking
    regularizer   : Tikhonov λ — penalises large Gauss-Newton steps
    break_dist_m  : convergence threshold (metres; ~5 mm matches C++ default)
    max_iter      : max iterations per seed

    Returns
    -------
    vtx        : (3,) best-fit vertex in metres  (NaN vector if failed)
    fit_rms_ns : RMS of timing residuals at best vertex (ns)
    converged  : True if any seed produced an in-tank vertex
    n_fit_hits : number of hits used after causal-compatibility filtering
    """
    _NAN_VTX = np.full(3, np.nan)

    fit_mask = _filter_hits_for_vertex(xyz, t)
    xyz_fit  = xyz[fit_mask]
    t_fit    = t[fit_mask]
    n_fit    = int(fit_mask.sum())

    if n_fit < 4:
        return _NAN_VTX, np.nan, False, n_fit

    seeds = _generate_seed_vertices(geo, y_spacing=y_spacing, n_planar=n_planar)

    best_vtx  = _NAN_VTX.copy()
    best_rms  = np.inf
    converged = False

    for seed in seeds:
        guess   = seed.copy()
        in_tank = True

        for _ in range(max_iter):
            last_guess = guess.copy()
            A, b       = _jac_residual(guess, xyz_fit, t_fit, regularizer)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            if not np.all(np.isfinite(sol)):
                in_tank = False
                break

            guess = last_guess + sol[:3]

            if not _in_tank(guess, geo):
                in_tank = False
                break

            if float(np.linalg.norm(sol[:3])) < break_dist_m:
                break

        if not in_tank:
            continue

        _, b_eval = _jac_residual(guess, xyz_fit, t_fit, regularizer)
        rms = float(np.sqrt(np.mean(b_eval[:n_fit] ** 2)))

        if rms < best_rms:
            best_rms  = rms
            best_vtx  = guess.copy()
            converged = True

    return best_vtx, (best_rms if converged else np.nan), converged, n_fit


def _fit_goodness(xyz: np.ndarray, t: np.ndarray,
                  vtx: np.ndarray, t_emit: float,
                  sigma_tight: float = 5.0,
                  sigma_wide:  float = 60.0) -> float:
    """
    SK FitGoodness metric (arXiv:2505.04409 eq. 3.1–3.2).

    g = Σᵢ wᵢ × exp(−tᵣₑₛᵢ² / (2σ_tight²))
    wᵢ ∝ exp(−tᵣₑₛᵢ² / (2σ_wide²))   [normalised to Σwᵢ = 1]

    σ_tight = 5 ns  (PMT timing resolution)
    σ_wide  = 60 ns (wide window for outlier suppression)
    tᵣₑₛᵢ = tᵢ − |rᵢ − vtx|/c_w − t_emit

    Larger g → tighter timing consistency → more signal-like.
    """
    dist   = np.linalg.norm(xyz - vtx[np.newaxis, :], axis=1)
    t_res  = t - dist / SOL_WATER - t_emit

    log_w  = -(t_res ** 2) / (2.0 * sigma_wide ** 2)
    log_w -= log_w.max()
    w      = np.exp(log_w)
    w     /= w.sum()

    return float(np.sum(w * np.exp(-(t_res ** 2) / (2.0 * sigma_tight ** 2))))


def compute_cluster_features(df_cluster: pd.DataFrame,
                             geo: ANNIEGeometry,
                             source_pos_m: "Optional[np.ndarray]" = None) -> dict:
    """
    Compute all features for a single OPTICS cluster.

    Parameters
    ----------
    df_cluster   : DataFrame slice — all hits in one cluster.
                   Required columns: x, y, z, t, pe, pmtID
    geo          : ANNIEGeometry (PMT positions, offsets, tank dimensions)
    source_pos_m : AmBe source position in metres as (3,) array, or None.
                   When provided, d_source and d_source_fit are computed and
                   the source ToF is available for OPTICS (passed separately).

    Returns
    -------
    dict of feature name → float (empty dict if df_cluster is empty)
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

    # ---- Legacy ANNIE charge balance ----
    # CB = sqrt( sum_pmt(Q_i²) / (sum_pmt(Q_i))² − 1/121 )
    # Q_i is total charge deposited on PMT tube i (hits summed per tube).
    # 1/121 is the baseline for uniform distribution (legacy ClusterFinder constant).
    # arg can go slightly negative for >121 PMTs with equal charge → clamp to 0.
    if pe_total > 0:
        pmt_q: dict = {}
        for pid, p in zip(ids, pe):
            pmt_q[int(pid)] = pmt_q.get(int(pid), 0.0) + float(p)
        q_arr   = np.array(list(pmt_q.values()), dtype=float)
        sum_q   = float(q_arr.sum())
        sum_q2  = float((q_arr ** 2).sum())
        arg     = sum_q2 / (sum_q * sum_q) - 1.0 / 121.0
        charge_bal_legacy = float(np.sqrt(max(arg, 0.0)))
    else:
        charge_bal_legacy = np.nan

    # ---- Isotropy (beta parameters, SK convention β1–β5) ----
    # Unit vectors from vertex to each hit PMT
    vecs  = xyz - vertex[np.newaxis, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 1e-6, norms, 1e-6)
    uvecs = vecs / norms

    beta1 = _beta_k(uvecs, k=1)
    beta2 = _beta_k(uvecs, k=2)
    beta3 = _beta_k(uvecs, k=3)
    beta4 = _beta_k(uvecs, k=4)
    beta5 = _beta_k(uvecs, k=5)

    # ---- Direct-light window features ----
    # n_hits_early: hits within ±10 ns of the cluster median (direct Cherenkov
    # only, cuts reflections). Uses offset-corrected times centred on the
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

    # ---- Gauss-Newton vertex fitter ----
    vtx_fit, fit_rms_ns, fit_converged, n_fit_hits = fit_vertex_gauss_newton(
        xyz, t_corr, geo)

    if fit_converged:
        v_xz_fit   = vtx_fit[[0, 2]] - geo.tank_center[[0, 2]]
        r_vtx_fit  = float(np.linalg.norm(v_xz_fit))
        d_wall_fit = float(min(geo.tank_radius - r_vtx_fit,
                               geo.tank_top - float(vtx_fit[1]),
                               float(vtx_fit[1]) - geo.tank_bot))

        vecs_fit  = xyz - vtx_fit[np.newaxis, :]
        norms_fit = np.linalg.norm(vecs_fit, axis=1, keepdims=True)
        norms_fit = np.where(norms_fit > 1e-6, norms_fit, 1e-6)
        uvecs_fit = vecs_fit / norms_fit
        beta1_fit = _beta_k(uvecs_fit, k=1)
        beta2_fit = _beta_k(uvecs_fit, k=2)
        beta3_fit = _beta_k(uvecs_fit, k=3)
        beta4_fit = _beta_k(uvecs_fit, k=4)
        beta5_fit = _beta_k(uvecs_fit, k=5)

        dist_fit            = np.linalg.norm(xyz - vtx_fit[np.newaxis, :], axis=1)
        t_tof_fit           = t_corr - dist_fit / SOL_WATER
        sigma_t_mad_tof_fit = 1.4826 * float(
            np.median(np.abs(t_tof_fit - np.median(t_tof_fit))))
        t_emit_fit          = float(np.mean(t_tof_fit))
        fit_goodness_reco   = _fit_goodness(xyz, t_corr, vtx_fit, t_emit_fit)

        d_source_fit = (float(np.linalg.norm(vtx_fit - np.asarray(source_pos_m, dtype=float)))
                        if source_pos_m is not None else np.nan)
    else:
        d_wall_fit = beta1_fit = beta2_fit = beta3_fit = beta4_fit = beta5_fit = np.nan
        sigma_t_mad_tof_fit = fit_goodness_reco = d_source_fit = np.nan

    # SK FitGoodness using PE-weighted centroid (INIT mode — no fitter needed)
    dist_init   = np.linalg.norm(xyz - vertex[np.newaxis, :], axis=1)
    t_emit_init = float(np.mean(t_corr - dist_init / SOL_WATER))
    fit_goodness_init = _fit_goodness(xyz, t_corr, vertex, t_emit_init)

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
        "charge_bal_legacy":   charge_bal_legacy,
        "spatial_rms":         spatial_rms,
        "d_wall":              d_wall,
        "d_source":            d_source,
        # Isotropy β1–β5 (SK convention: 2/(N*(N-1)) × Σᵢ≠ⱼ P_k(cos θᵢⱼ))
        "beta1":               beta1,
        "beta2":               beta2,
        "beta3":               beta3,
        "beta4":               beta4,
        "beta5":               beta5,
        # Vertex — PE-weighted centroid
        "vtx_x":               float(vertex[0]),
        "vtx_y":               float(vertex[1]),
        "vtx_z":               float(vertex[2]),
        # Vertex — Gauss-Newton fitted
        "vtx_fit_x":           float(vtx_fit[0]) if fit_converged else np.nan,
        "vtx_fit_y":           float(vtx_fit[1]) if fit_converged else np.nan,
        "vtx_fit_z":           float(vtx_fit[2]) if fit_converged else np.nan,
        "fit_rms_ns":          fit_rms_ns,
        "fit_converged":       int(fit_converged),
        "n_fit_hits":          n_fit_hits,
        "d_wall_fit":          d_wall_fit,
        "d_source_fit":        d_source_fit,
        # β1–β5 using fitted-vertex directions
        "beta1_fit":           beta1_fit,
        "beta2_fit":           beta2_fit,
        "beta3_fit":           beta3_fit,
        "beta4_fit":           beta4_fit,
        "beta5_fit":           beta5_fit,
        # Timing spread: ToF-corrected with fitted vertex
        "sigma_t_mad_tof_fit": sigma_t_mad_tof_fit,
        # SK FitGoodness — larger = tighter timing consistency = more signal-like
        "fit_goodness_init":   fit_goodness_init,
        "fit_goodness_reco":   fit_goodness_reco,
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
    ctx                  : RunContext (parquet paths, run name, config)
    geo                  : pre-loaded ANNIEGeometry
    min_samples, xi, t_unit_ns : OPTICS hyperparameters (use best config from grid sweep)
    hit_prefilter_ns     : CF-time prefilter window (0 = disabled — required for clean sigma_t)
    truth_window_ns      : truth hit window for neutron labelling (75 ns recommended)
    source_pos_m         : AmBe source in metres for source ToF correction and d_source.
                           None disables both.
    min_pulses_per_event : skip events with fewer hits than this (0 = keep all)

    Returns
    -------
    DataFrame with one row per cluster (both OPTICS and ClusterFinder methods).
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

    t_start     = time.time()
    n_total     = len(event_ids)
    print_every = max(1, n_total // 20)   # ~20 progress lines over the full run

    for i_ev, evid in enumerate(event_ids):
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

                # Prompt-signal flag: cluster is prompt AmBe gamma, muon light, etc.
                # Tagged by timing (early arrival) OR composition (class-5 dominant
                # with only stray neutron contamination ≤ PROMPT_SIGNAL_MAX_NEUTRON_HITS).
                cluster_offset = (feats["t_mean"] - neu_ref_t
                                  if not (neu_ref_t != neu_ref_t) else float("nan"))
                timing_prompt = (cluster_offset < PROMPT_SIGNAL_THRESHOLD_NS
                                 if cluster_offset == cluster_offset else False)
                comp_prompt   = (comp["frac_nonneutron"] > 0.5 and
                                 comp["n_neutron"] <= PROMPT_SIGNAL_MAX_NEUTRON_HITS)
                is_prompt = int(timing_prompt or comp_prompt)

                row = {
                    "eventID":                 int(evid),
                    "method":                  method,
                    "cluster_id":              int(cid),
                    "is_truth_neutron":        is_n,
                    "dominant_trackID":        dom_tid,
                    "cluster_time_offset_ns":  round(cluster_offset, 1)
                                               if cluster_offset == cluster_offset
                                               else float("nan"),
                    "is_prompt_cluster":       is_prompt,
                }
                row.update(feats)
                row.update(comp)
                rows.append(row)

        if (i_ev + 1) % print_every == 0 or (i_ev + 1) == n_total:
            elapsed   = time.time() - t_start
            rate      = (i_ev + 1) / elapsed
            remaining = (n_total - i_ev - 1) / rate if rate > 0 else 0
            print(f"[cluster_features]  {i_ev+1:5d}/{n_total}  "
                  f"({100*(i_ev+1)/n_total:.0f}%)  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.1f} ev/s  "
                  f"ETA={remaining:.0f}s  clusters={len(rows)}", flush=True)

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
    "pe_balance":          "Charge balance  (max−min)/total  [quadrant]",
    "charge_bal_legacy":   "Legacy charge balance  sqrt(ΣQ²/ΣQ² − 1/121)  [per-PMT]",
    "spatial_rms":         "Spatial RMS of hit PMTs  (m)",
    "d_wall":              "Distance to nearest wall  (m)",
    "d_source":            "Distance from vertex to AmBe source  (m)",
    # Isotropy β1–β5 (SK convention, PE-weighted centroid vertex)
    "beta1":               "Isotropy β₁  (Legendre P₁, SK)",
    "beta2":               "Isotropy β₂  (Legendre P₂, SK)",
    "beta3":               "Isotropy β₃  (Legendre P₃, SK)",
    "beta4":               "Isotropy β₄  (Legendre P₄, SK)",
    "beta5":               "Isotropy β₅  (Legendre P₅, SK)",
    # Gauss-Newton fitted vertex
    "fit_rms_ns":          "Vertex fit timing RMS (ns)",
    "fit_converged":       "Vertex fit converged  (0/1)",
    "n_fit_hits":          "N hits used in vertex fit",
    "d_wall_fit":          "Distance to wall — fitted vertex  (m)",
    "d_source_fit":        "Distance to AmBe source — fitted vertex  (m)",
    "beta1_fit":           "Isotropy β₁ — fitted vertex",
    "beta2_fit":           "Isotropy β₂ — fitted vertex",
    "beta3_fit":           "Isotropy β₃ — fitted vertex",
    "beta4_fit":           "Isotropy β₄ — fitted vertex",
    "beta5_fit":           "Isotropy β₅ — fitted vertex",
    "sigma_t_mad_tof_fit": "σ_t MAD — ToF-corrected (fitted vertex)  (ns)",
    "fit_goodness_init":   "SK FitGoodness — centroid vertex  (0–1)",
    "fit_goodness_reco":   "SK FitGoodness — fitted vertex  (0–1)",
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

                ax.hist(sig, bins=bins, density=True, alpha=0.6, histtype="step",
                        color=C_SIG,    label=f"Truth neutron  (n={len(sig)})")
                ax.hist(bkg, bins=bins, density=True, alpha=0.6,
                        color=C_SPU, label=f"Spurious  (n={len(bkg)})")
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
      - prompt signal clusters (is_prompt_cluster=1): AmBe gamma / muon / beam light
      - real spurious  (is_truth_neutron=0 and is_prompt_cluster=0): true background
    Separation is computed against real spurious only.
    """
    has_prompt_flag = "is_prompt_cluster" in df.columns
    rows = []
    for feat in FEATURE_LABELS:
        if feat not in df.columns:
            continue
        for method in ["optics", "clusterfinder"]:
            sub = df[df["method"] == method]
            sig = sub[sub["is_truth_neutron"] == 1][feat].dropna()
            # Exclude prompt signal clusters from background for cleaner separation metric
            if has_prompt_flag:
                real_bkg = sub[(sub["is_truth_neutron"] == 0) &
                               (sub["is_prompt_cluster"] == 0)]
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

        # Split spurious into prompt signal clusters vs real spurious
        has_prompt_flag = "is_prompt_cluster" in sub.columns
        if has_prompt_flag:
            prompt_clusters = spurious[spurious["is_prompt_cluster"] == 1]
            real_spurious   = spurious[spurious["is_prompt_cluster"] == 0]
        else:
            prompt_clusters = pd.DataFrame()
            real_spurious   = spurious

        print(f"\n[{method.upper()}]  {n_total} clusters total: "
              f"{len(matched)} matched, {n_spur} spurious")
        if has_prompt_flag and len(prompt_clusters) > 0:
            print(f"  ↳ Of spurious: {len(prompt_clusters)} prompt signal clusters "
                  f"(early timing or class-5 dominant ≤{PROMPT_SIGNAL_MAX_NEUTRON_HITS} "
                  f"neutron hits) + {len(real_spurious)} real spurious")

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

        if has_prompt_flag and len(real_spurious) > 0:
            real_means = real_spurious[frac_cols + n_cols].mean()
            print(f"\n  Real spurious only ({len(real_spurious)} clusters, "
                  f"excluding prompt signal):")
            print(f"    Neutron (class 1-4):       {real_means['frac_neutron']:.3f}  "
                  f"(mean {real_means['n_neutron']:.1f} hits/cluster)")
            print(f"    Non-neutron physics (−5):  {real_means['frac_nonneutron']:.3f}  "
                  f"(mean {real_means['n_nonneutron']:.1f} hits/cluster)")
            real_pur = len(matched) / (len(matched) + len(real_spurious))
            print(f"  Corrected purity (excl. prompt signal clusters): {real_pur:.3f}")

        rows.append({
            "method":               method,
            "n_spurious":           n_spur,
            "n_prompt_clusters":    len(prompt_clusters) if has_prompt_flag else 0,
            "n_real_spurious":      len(real_spurious)   if has_prompt_flag else n_spur,
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


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

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


def make_separation_plots(df: pd.DataFrame, ctx: "RunContext") -> Path:
    """
    Generate per-feature distribution plots comparing three cluster categories:
      - Signal         (is_truth_neutron == 1)
      - Prompt signal  (is_prompt_cluster == 1)  — AmBe gamma / muon / beam light
      - Real spurious  (is_truth_neutron == 0, is_prompt_cluster == 0)
                                                 — true false-positive clusters

    Produces one PDF page per method (optics / clusterfinder) with a 6×3 grid
    of subplots — one per physics feature.  Each subplot shows overlapping
    density histograms with vertical median lines and the separation σ values
    printed in the title.

    Saved to: <plots_dir>/<run_name>__feature_separation_plots.pdf
    """
    # ── features & display ranges (xlim may clip tails for readability) ───────
    FEAT_CFG = [
        ("n_hits",            "N hits in cluster",                  (0,   55)),
        ("pe_total",          "Total PE",                           (0,  110)),
        ("n_hits_early",      "N hits early (±10 ns window)",       (0,   50)),
        ("sigma_t_mad",       "σ_t MAD — raw (ns)  [robust]",      (0,   30)),
        ("sigma_t_mad_corr",  "σ_t MAD — offset-corrected (ns)",   (0,   30)),
        ("sigma_t_mad_tof",   "σ_t MAD — ToF-corrected (ns)",      (0,   30)),
        ("sigma_t_early_mad", "σ_t MAD — direct-light window (ns)",(0,   20)),
        ("t_window_80pct",    "80% hit window width (ns)",          (0,  200)),
        ("sigma_t",           "σ_t std — raw (ns)",                 (0,  500)),
        ("sigma_t_corr",      "σ_t std — offset-corrected (ns)",   (0,  500)),
        ("sigma_t_tof",       "σ_t std — ToF-corrected (ns)",      (0,  500)),
        ("pe_balance",        "Charge balance  [0–1]  (quadrant)",  (0,    1)),
        ("charge_bal_legacy", "Legacy charge balance  CB",          (0,  0.5)),
        ("spatial_rms",       "Spatial RMS (m)",                    (0,    2)),
        ("d_wall",            "Distance to wall (m)",               (0,  1.6)),
        ("d_source",          "Distance to AmBe source (m)",        (0,  1.9)),
        ("beta1",             "β₁ — Legendre P1 (SK)",             (-0.5, 2)),
        ("beta2",             "β₂ — Legendre P2 (SK)",             (-0.5, 2)),
        ("beta3",             "β₃ — Legendre P3 (SK)",             (-0.5, 2)),
        ("beta4",             "β₄ — Legendre P4 (SK)",             (-0.5, 2)),
        ("beta5",             "β₅ — Legendre P5 (SK)",             (-0.5, 2)),
        # Gauss-Newton fitted vertex
        ("fit_rms_ns",          "Vertex fit timing RMS (ns)",          (0,   30)),
        ("fit_goodness_reco",   "SK FitGoodness — fitted vtx (0–1)",   (0,    1)),
        ("fit_goodness_init",   "SK FitGoodness — centroid (0–1)",     (0,    1)),
        ("d_wall_fit",          "Distance to wall — fitted vtx (m)",   (0,  1.6)),
        ("sigma_t_mad_tof_fit", "σ_t MAD ToF — fitted vtx (ns)",      (0,   20)),
        ("beta1_fit",           "β₁ — fitted vertex",                 (-0.5, 2)),
    ]

    def _sep(a, b):
        """Separation in units of pooled std."""
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return abs(a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-9)

    has_prompt = "is_prompt_cluster" in df.columns

    pdf_path = ctx.plots_dir / f"{ctx.run_name}__feature_separation_plots.pdf"
    ctx.plots_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for method in ["optics", "clusterfinder"]:
            sub = df[df["method"] == method]
            if len(sub) == 0:
                continue

            sig = sub[sub["is_truth_neutron"] == 1]
            if has_prompt:
                gam = sub[(sub["is_truth_neutron"] == 0) & (sub["is_prompt_cluster"] == 1)]
                spu = sub[(sub["is_truth_neutron"] == 0) & (sub["is_prompt_cluster"] == 0)]
            else:
                gam = pd.DataFrame()
                spu = sub[sub["is_truth_neutron"] == 0]

            NCOLS, NROWS = 3, 10
            fig, axes = plt.subplots(NROWS, NCOLS, figsize=(15, 22))
            fig.suptitle(
                f"{ctx.run_name.upper()} — {method.upper()} clusters\n"
                f"Feature distributions: Signal / Prompt signal / Real spurious\n"
                f"n = {len(sig)} signal  |  {len(gam)} prompt  |  {len(spu)} real spurious",
                fontsize=11, y=1.001,
            )

            for idx, (col, label, xlim) in enumerate(FEAT_CFG):
                row, c = divmod(idx, NCOLS)
                ax = axes[row, c]

                if col not in sub.columns:
                    ax.set_visible(False)
                    continue

                s = sig[col].dropna().clip(*xlim)
                g = gam[col].dropna().clip(*xlim) if len(gam) > 0 else pd.Series(dtype=float)
                r = spu[col].dropna().clip(*xlim)

                bins = np.linspace(xlim[0], xlim[1], 40)
                kw = dict(bins=bins, density=True)

                ax.hist(s, histtype="stepfilled", color=C_SIG, alpha=0.4, edgecolor=C_SIG, linewidth=1.2, **kw)
                if len(g) > 1:
                    ax.hist(g, histtype="step", color=C_GAM, linewidth=1.8, **kw)
                if len(r) > 1:
                    ax.hist(r, histtype="step", color=C_SPU, linewidth=1.8, linestyle="--", **kw)

                # Median lines
                if len(s): ax.axvline(s.median(), color=C_SIG, lw=1.8, ls="--")
                if len(g) > 1: ax.axvline(g.median(), color=C_GAM, lw=1.8, ls="--")
                if len(r) > 1: ax.axvline(r.median(), color=C_SPU, lw=1.8, ls="--")

                sep_sr = _sep(s, r)
                sep_sg = _sep(s, g) if len(g) > 1 else float("nan")
                sep_str = f"sep(sig|spu)={sep_sr:.2f}σ"
                if not np.isnan(sep_sg):
                    sep_str += f"  sep(sig|prompt)={sep_sg:.2f}σ"

                ax.set_title(f"{label}\n{sep_str}", fontsize=8)
                ax.set_xlabel(label, fontsize=7)
                ax.set_ylabel("Density", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.set_xlim(*xlim)

            # Hide unused subplots (27 features in 10×3 grid → 3 spare)
            for spare in range(len(FEAT_CFG), NROWS * NCOLS):
                r2, c2 = divmod(spare, NCOLS)
                axes[r2, c2].set_visible(False)

            # Shared legend in spare cell
            legend_elements = [
                Line2D([0], [0], color=C_SIG, lw=8, alpha=0.7,
                       label=f"Signal — neutron capture  (n={len(sig)})"),
                Line2D([0], [0], color=C_GAM, lw=8, alpha=0.7,
                       label=f"Prompt signal — γ / muon / beam  (n={len(gam)})"),
                Line2D([0], [0], color=C_SPU, lw=8, alpha=0.7,
                       label=f"Real spurious — true false positives  (n={len(spu)})"),
                Line2D([0], [0], color="gray", lw=1.8, ls="--",
                       label="Median of each category"),
            ]
            spare_ax = axes[NROWS - 1, NCOLS - 1]
            spare_ax.set_visible(True)
            spare_ax.axis("off")
            spare_ax.legend(handles=legend_elements, loc="center",
                            fontsize=9, framealpha=0.9)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[cluster_features] wrote separation plots → {pdf_path}")
    return pdf_path


def extract_features_from_background_hits(
        hits_parquet: str,
        geo: ANNIEGeometry,
        source_pos_m: "Optional[np.ndarray]" = None,
) -> pd.DataFrame:
    """
    Compute cluster features for background data that has already been through
    OPTICS pre-selection (output of analyze_optics_data_rate.py).

    Reads background_optics_hits.parquet — columns:
        run, event_number, cluster_id, is_background, x, y, z, t, pe, pmtID

    For each (run, event_number, cluster_id) group it calls
    compute_cluster_features() directly, skipping truth labelling (real data
    has no MC truth).  The is_background=1 label is carried through unchanged.

    Returns a DataFrame with the same feature columns as extract_all_features()
    except the MC-only columns (is_truth_neutron, dominant_trackID,
    is_prompt_cluster, hit composition fractions).
    """
    hits = pd.read_parquet(hits_parquet)
    required = {"run", "event_number", "cluster_id", "is_background",
                "x", "y", "z", "t", "pe", "pmtID"}
    missing = required - set(hits.columns)
    if missing:
        raise ValueError(f"background hits parquet is missing columns: {missing}")

    rows = []
    groups = list(hits.groupby(["run", "event_number", "cluster_id"]))
    print(f"[cluster_features] background mode: {len(groups)} clusters from {hits_parquet}")

    for i, ((run, ev, cid), df_cl) in enumerate(groups):
        feats = compute_cluster_features(
            df_cl.reset_index(drop=True), geo, source_pos_m=source_pos_m)
        if not feats:
            continue
        row = {
            "run":           run,
            "event_number":  ev,
            "cluster_id":    int(cid),
            "is_background": 1,
        }
        row.update(feats)
        rows.append(row)

        if (i + 1) % max(1, len(groups) // 20) == 0 or (i + 1) == len(groups):
            print(f"[cluster_features]  {i+1}/{len(groups)} clusters processed",
                  flush=True)

    df = pd.DataFrame(rows)
    print(f"[cluster_features] background: extracted features for {len(df)} clusters")
    return df


def extract_features_from_background_hits_optics(
        hits_parquet: str,
        geo: ANNIEGeometry,
        min_samples:  int   = 3,
        xi:           float = 0.05,
        t_unit_ns:    float = 25.0,
        source_pos_m: "Optional[np.ndarray]" = None,
) -> pd.DataFrame:
    """
    Re-run OPTICS on each saved background cluster's hits, then compute features
    on the resulting OPTICS sub-clusters.

    Unlike extract_features_from_background_hits(), this does NOT treat the
    ClusterFinder cluster boundaries as fixed.  Instead it passes the raw hits
    through run_optics_on_event() first, so that OPTICS can split, merge, or
    reject hits before feature computation.  This is consistent with how signal
    neutron clusters are processed in extract_all_features().

    Parameters
    ----------
    hits_parquet : path to michel_background_hits.parquet (or equivalent)
    geo          : pre-loaded ANNIEGeometry
    min_samples  : OPTICS min_samples — use 3–5 for small Michel clusters
                   (default 8 for neutrons would label all hits as noise)
    xi           : OPTICS xi steepness parameter
    t_unit_ns    : time axis scale (ns per OPTICS distance unit)
    source_pos_m : source position in metres for ToF correction; None = disabled

    Returns
    -------
    DataFrame with one row per OPTICS sub-cluster found within each input cluster.
    Columns are the same as extract_features_from_background_hits() plus
    optics_cluster_id (sub-cluster index within the original CF cluster) and
    n_optics_noise (hits OPTICS labelled as noise within that CF cluster).
    """
    hits = pd.read_parquet(hits_parquet)
    required = {"run", "event_number", "cluster_id", "is_background",
                "x", "y", "z", "t", "pe", "pmtID"}
    missing = required - set(hits.columns)
    if missing:
        raise ValueError(f"background hits parquet is missing columns: {missing}")

    groups = list(hits.groupby(["run", "event_number", "cluster_id"]))
    print(f"[cluster_features] OPTICS background mode: {len(groups)} input clusters "
          f"| ms={min_samples} xi={xi} t_unit={t_unit_ns}ns")

    rows = []
    n_optics_clusters_total = 0
    n_noise_total = 0

    for i, ((run, ev, cid), df_cl) in enumerate(groups):
        df_cl = df_cl.reset_index(drop=True)

        labels = run_optics_on_event(
            df_cl,
            min_samples=min_samples,
            xi=xi,
            t_unit_ns=t_unit_ns,
            source_pos_m=source_pos_m,
        )

        n_noise = int((labels == -1).sum())
        n_noise_total += n_noise
        unique_clusters = [c for c in np.unique(labels) if c >= 0]
        n_optics_clusters_total += len(unique_clusters)

        for optics_cid in unique_clusters:
            mask = (labels == optics_cid)
            df_sub = df_cl[mask].reset_index(drop=True)
            feats = compute_cluster_features(df_sub, geo, source_pos_m=source_pos_m)
            if not feats:
                continue
            row = {
                "run":              run,
                "event_number":     ev,
                "cf_cluster_id":    int(cid),
                "optics_cluster_id": int(optics_cid),
                "n_optics_noise":   n_noise,
                "is_background":    1,
            }
            row.update(feats)
            rows.append(row)

        if (i + 1) % max(1, len(groups) // 10) == 0 or (i + 1) == len(groups):
            print(f"[cluster_features]  {i+1}/{len(groups)} CF clusters processed  "
                  f"→ {n_optics_clusters_total} OPTICS clusters so far  "
                  f"({n_noise_total} noise hits total)", flush=True)

    df = pd.DataFrame(rows)
    print(f"[cluster_features] OPTICS background: {len(df)} OPTICS clusters extracted "
          f"from {len(groups)} input CF clusters")
    if len(df) == 0:
        print(f"[cluster_features] WARNING: 0 clusters survived OPTICS. "
              f"Try lowering min_samples (currently {min_samples}).")
    return df


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

    # Beamoff mode: hits parquet already has OPTICS cluster assignments —
    # skip OPTICS re-run and truth labelling, just compute features.
    background_hits = ctx.extra.get("features", {}).get("background_hits", None)
    if background_hits:
        df = extract_features_from_background_hits(background_hits, geo,
                                                   source_pos_m=src_pos)
        out_parquet = ctx.parquet_path(f"{ctx.run_name}__background_cluster_features")
        df.to_parquet(out_parquet, index=False)
        print(f"[cluster_features] wrote background features → {out_parquet}")
        return out_parquet

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

    # Diagnostic plots — existing feature distributions
    feature_plots(df, ctx)

    # Signal / gamma / real-spurious separation plots (all 17 physics features)
    make_separation_plots(df, ctx)

    return out_parquet


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)
