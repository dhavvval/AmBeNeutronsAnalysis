"""
IC_adjusted window scan of the AmBe neutron efficiency.

WHAT THIS ANSWERS. The production AmBe efficiency is quoted with a FIXED Stage-1 IC
gate (700 < IC_adjusted < 1200). Moving that gate moves the efficiency, so the
question is where -- if anywhere -- the efficiency stops responding to the window.
This module steps the window edges by `icscan.step` across
[`icscan.ic_lo_start`, `icscan.ic_hi_start`] and re-tallies the efficiency for each
window, entirely offline.

WHY IT IS OFFLINE, AND THE ONE THING THAT MAKES IT POSSIBLE. A waveform rejected at
Stage 1 is never Stage-2 processed, so it can never produce a neutron candidate under
any window -- it returns "zero neutrons" by construction rather than by physics.
Therefore the Stage-1+2 pass must be run ONCE at the widest window in the scan, and
every scan point taken as a subset of it. Given that pass, the three ingredients of

    efficiency = unique_neutron_triggers / ambe_triggers
    ambe_triggers = total_events - cosmic_events

come from two files per run:

  * TriggerSummary/WaveformFeatures_<tag>_<run>.parquet -- one row per waveform,
    carrying IC_adjusted. This supplies total_events for any sub-window.
  * TriggerSummary/EventIndex_<tag>_<run>.parquet -- one row per Stage-1-admitted
    event, carrying the cosmic verdict and the accepted-cluster counts. This is the
    file that makes the scan possible at all: cosmic_events is the ONE ingredient the
    older published dumps cannot supply, because PromptAmBeNeutronCandidates carries
    no event key and so a cosmic event cannot be assigned an IC value. Written only
    when stage1.dump_event_index is set.

The two join 1:1 on eventTankTime == timestamp.

WHAT "PLATEAU" CAN MEAN HERE. Both the numerator AND the denominator of the ratio move
with the window, so what is being scanned is not a detector efficiency -- it is the
neutron-bearing FRACTION of whatever tag sample the window selects. A cumulative curve
flattening is therefore weak evidence. The real plateau condition is stationarity: the
tags newly admitted by the last step must be neutron-bearing at the same rate as the
tags already inside. That is what --mode differential measures, on DISJOINT slices,
which also makes its points statistically independent -- unlike the nested cumulative
windows of --mode left / --mode right, which share most of their events. Do all
flatness testing on the differential curve.

fprompt is deliberately never read, plotted or passed anywhere in this module.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import textwrap
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import RunContext
from ..plotting import save_plot, set_style

# The five centre-port (Port 5) v4 positions, from processor.py source_positions.
# Asserted rather than assumed: this scan is only defined for the centre port, and a
# stray non-Port-5 position in the sample would silently average two different
# geometries into one curve.
PORT5_POSITIONS = {
    (0.0, 100.0, 0.0),
    (0.0, 50.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, -50.0, 0.0),
    (0.0, -100.0, 0.0),
}

MODES = ("left", "right", "differential", "closure", "widthsweep", "grid")

# Slice widths for --mode widthsweep. The scan range is 1000 wide, so 50/100/125/200/
# 250/500 tile it exactly; 150/300/400 leave a short final slice, which is kept and
# flagged rather than dropped -- silently truncating would make the last slice look
# like a low outlier when it is really just a smaller exposure.
SWEEP_WIDTHS = (50.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0)

SUM_LABEL = "port5_sum"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_table(path: str) -> pd.DataFrame:
    """Read a dumped table, tolerating the processor's CSV parquet-fallback."""
    if path.endswith(".parquet") and not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            return pd.read_csv(alt)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _expand(patterns) -> List[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    out: List[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits and pat.endswith(".parquet"):
            hits = sorted(glob.glob(os.path.splitext(pat)[0] + ".csv"))
        out.extend(hits)
    return out


def load_features(ctx: RunContext) -> pd.DataFrame:
    """
    Per-waveform Stage-1 table: every acquisition, accepted or not.

    Used only for the acceptance curve -- the Stage-1 exposure cost of a window,
    which is the denominator-side context the efficiency ratio hides. Note that
    `accepted` here is frozen at the PASS window; the scan re-derives acceptance from
    IC_adjusted and second_pulse instead of trusting that column.
    """
    paths = _expand(ctx.inputs.get("waveform_features"))
    if not paths:
        raise SystemExit(
            "icscan: no waveform feature tables found. Expected "
            "inputs.waveform_features to glob "
            "TriggerSummary/WaveformFeatures_<tag>_<run>.parquet -- run "
            "`ambe data process` on this config first."
        )
    cols = ["timestamp", "accepted", "IC_adjusted", "second_pulse"]
    frames = []
    for p in paths:
        df = _read_table(p)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"icscan: {p} is missing columns {missing}")
        df = df[cols].copy()
        df["run"] = _run_from_path(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_sample(ctx: RunContext) -> pd.DataFrame:
    """
    One row per Stage-2-processed event, carrying its IC_adjusted.

    This is the scan's working table. Every quantity in `tally` is a groupby over it.
    """
    idx_paths = _expand(ctx.inputs.get("event_index"))
    if not idx_paths:
        raise SystemExit(
            "icscan: no event index found. Expected inputs.event_index to glob "
            "TriggerSummary/EventIndex_<tag>_<run>.parquet. That file is written "
            "only when stage1.dump_event_index is true -- without it cosmic_events "
            "cannot be re-sliced by IC window and the efficiency denominator is "
            "unavailable. Set the flag and re-run `ambe data process`."
        )

    feat_paths = _expand(ctx.inputs.get("waveform_features"))
    feat_by_run = {_run_from_path(p): p for p in feat_paths}

    frames = []
    for p in idx_paths:
        run = _run_from_path(p)
        idx = _read_table(p)
        if run not in feat_by_run:
            raise SystemExit(
                f"icscan: event index for run {run} has no matching waveform "
                f"feature table; cannot attach IC_adjusted."
            )
        feat = _read_table(feat_by_run[run])[["timestamp", "IC_adjusted"]]

        if feat["timestamp"].duplicated().any():
            raise SystemExit(
                f"icscan: run {run} has duplicate waveform timestamps; the "
                f"event<->waveform join is not 1:1 and the scan would double count."
            )

        merged = idx.merge(feat, left_on="eventTankTime", right_on="timestamp",
                           how="left", validate="one_to_one")
        unmatched = int(merged["IC_adjusted"].isna().sum())
        if unmatched:
            raise SystemExit(
                f"icscan: run {run} has {unmatched} indexed events with no matching "
                f"waveform. Every Stage-2 event came from an accepted waveform, so "
                f"this means the index and the feature table are from different passes."
            )
        merged["run"] = run
        frames.append(merged)

    sample = pd.concat(frames, ignore_index=True)

    if "n_accepted_clusters" not in sample.columns:
        sample["n_accepted_clusters"] = (sample["n_accepted_single"]
                                         + sample["n_accepted_multiple"])

    positions = {(float(r.sourceX), float(r.sourceY), float(r.sourceZ))
                 for r in sample.itertuples()}
    stray = positions - PORT5_POSITIONS
    if stray:
        raise SystemExit(
            f"icscan: sample contains non-Port-5 source positions {sorted(stray)}. "
            f"This scan is defined for the centre port only -- summing it over other "
            f"ports would average different geometries into one curve."
        )

    print(f"icscan: loaded {len(sample)} Stage-2 events over {len(idx_paths)} runs, "
          f"{len(positions)} Port-5 positions")
    return sample


def _run_from_path(path: str) -> int:
    """Run number is the trailing _<run> of the dumped filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("_", 1)[1])


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def windows(mode: str, lo0: float, hi0: float, step: float,
            closure: Tuple[float, float] = (700.0, 1200.0),
            width: Optional[float] = None
            ) -> List[Tuple[float, float, str, float]]:
    """
    The list of (ic_min, ic_max, edge_moved, edge_value) for a mode.

    `edge_value` is the x-axis of the scan: the edge actually being moved, so every
    mode plots against a meaningful abscissa.

    `width` applies to the disjoint-slice modes and sets the slice width; it defaults
    to `step`. Slice width is the RESOLUTION of the scan -- see slices() -- and is
    independent of `step`, which is the cumulative modes' edge increment.
    """
    if mode == "left":
        # Upper edge fixed at the top of the range; walk the lower edge up.
        return [(lo, hi0, "lower", lo)
                for lo in np.arange(lo0, hi0, step)]
    if mode == "right":
        # Lower edge fixed at the bottom; walk the upper edge down.
        return [(lo0, hi, "upper", hi)
                for hi in np.arange(hi0, lo0, -step)]
    if mode == "differential":
        return slices(lo0, hi0, width if width else step)
    if mode == "grid":
        # Every arbitrary (lo, hi) pair on the step grid with lo < hi. edge_value is
        # the window centre, which is only a label here -- read this mode off the
        # 2-D map, not the 1-D curve.
        edges = list(np.arange(lo0, hi0 + step / 2.0, step))
        return [(lo, hi, "both", (lo + hi) / 2.0)
                for lo in edges for hi in edges if hi > lo]
    if mode == "closure":
        return [(closure[0], closure[1], "both", closure[0])]
    if mode == "widthsweep":
        raise SystemExit("icscan: widthsweep builds its windows per width; "
                         "call slices() directly")
    raise SystemExit(f"icscan: unknown mode {mode!r}; choose from {MODES}")


def slices(lo0: float, hi0: float, width: float
           ) -> List[Tuple[float, float, str, float]]:
    """
    Disjoint slices of the given width tiling [lo0, hi0].

    These are the statistically independent points -- disjoint event sets, no
    window-dependent denominator -- and the only ones a flatness test may be run on.

    If `width` does not divide the range, the final slice is SHORT rather than
    dropped. It is still a valid measurement, just on a smaller exposure; its width
    is recorded in the output so it is never mistaken for a full-width point.
    """
    out = []
    lo = lo0
    while lo < hi0 - 1e-9:
        hi = min(lo + width, hi0)
        out.append((lo, hi, "slice", (lo + hi) / 2.0))
        lo = hi
    return out


# --------------------------------------------------------------------------- #
# The calculation
# --------------------------------------------------------------------------- #
def binom_errors(k: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-point binomial error and Wilson interval.

    The sigma matches src/ambe/plots/heatmap.py err_B, so a scan point at the
    production window is directly comparable to the published heatmap cell. The
    Wilson interval is carried alongside because the interesting ends of the scan are
    exactly where n collapses and the normal approximation stops being honest.

    These are PER-POINT uncertainties. They are NOT valid for differencing two nested
    windows -- those share most of their events, so the shared part cancels and
    sqrt(s1^2 + s2^2) badly overstates the error. Take a difference from the
    differential mode instead.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        sigma = np.where(n > 0, np.sqrt(p * (1.0 - p) / n), np.nan)
        z = 1.0
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        lo = np.where(n > 0, centre - half, np.nan)
        hi = np.where(n > 0, centre + half, np.nan)
    return sigma, lo, hi


def acceptance(features: pd.DataFrame, lo: float, hi: float) -> float:
    """
    Stage-1 acceptance for a window: the fraction of ALL acquisitions it admits.

    Re-derived from IC_adjusted and second_pulse rather than read off the frozen
    `accepted` column, which reflects the pass window only. Strict inequalities to
    match processor.passes_selection.
    """
    if not len(features):
        return float("nan")
    m = ((features["IC_adjusted"] > lo) & (features["IC_adjusted"] < hi)
         & (~features["second_pulse"].astype(bool)))
    return float(m.sum()) / float(len(features))


def tally(sample: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """
    Re-tally the efficiency ingredients for one IC window, per position + summed.

    Strict inequalities, matching processor.passes_selection line for line.
    second_pulse needs no re-test: every row in the index already passed it at
    Stage 1, and that veto is window-independent.
    """
    m = (sample["IC_adjusted"] > lo) & (sample["IC_adjusted"] < hi)
    win = sample.loc[m]

    rows = []
    for (x, y, z), grp in win.groupby(["sourceX", "sourceY", "sourceZ"], sort=True):
        rows.append(_counts(grp, x, y, z))
    # Sum over positions on the COUNTS, never by averaging the per-position
    # efficiencies: the five positions differ in exposure by nearly 2x, so an
    # unweighted mean would not be the Port-5 efficiency of anything.
    rows.append(_counts(win, np.nan, np.nan, np.nan, label=SUM_LABEL))

    out = pd.DataFrame(rows)
    sigma, wlo, whi = binom_errors(out["unique_neutron_triggers"], out["ambe_triggers"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["efficiency"] = np.where(out["ambe_triggers"] > 0,
                                     out["unique_neutron_triggers"] / out["ambe_triggers"],
                                     np.nan)
    out["eff_err_binom"] = sigma
    out["eff_wilson_lo"] = wlo
    out["eff_wilson_hi"] = whi
    # The stricter reading, carried at every window so the size of the
    # numerator/denominator inconsistency is never invisible. See _counts.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["efficiency_noncosmic"] = np.where(
            out["ambe_triggers"] > 0,
            out["unique_neutron_triggers_noncosmic"] / out["ambe_triggers"],
            np.nan)
    out.insert(0, "ic_max", hi)
    out.insert(0, "ic_min", lo)
    out["width"] = hi - lo
    return out


def _counts(grp: pd.DataFrame, x, y, z, label: Optional[str] = None) -> dict:
    """
    The efficiency ingredients for one group of events.

    `unique_neutron_triggers` reproduces the production accumulator in
    process_events_efficient EXACTLY, which means it does NOT exclude cosmic-vetoed
    events. That is not an oversight here -- it is what the published number counts.
    In the production cluster loop, box-passing clusters are appended as the loop
    iterates and the cosmic `break` only fires when the loop REACHES the cosmic
    cluster, so a neutron-like cluster ordered before it is already banked and the
    event increments both cosmic_events and the candidate counter. On run 6062 at the
    production window that is 353 events (all with numberOfClusters >= 4; a
    single-cluster event cannot do it): they are subtracted from the denominator via
    cosmic_events while still counted in the numerator.

    Matching it is what lets --mode closure verify this scan against the frozen
    summary cell for cell, and what keeps every scan point comparable to the
    published 700-1200 efficiency. `unique_neutron_triggers_noncosmic` carries the
    stricter reading -- events that are neutron-bearing AND not cosmic-vetoed -- so
    the size of the inconsistency is visible at every window instead of buried.
    """
    total = len(grp)
    cosmic_mask = grp["is_cosmic"].astype(bool)
    has_cand = grp["n_accepted_clusters"] > 0
    cosmic = int(cosmic_mask.sum())
    return {
        "position": label if label else f"({x:g}, {y:g}, {z:g})",
        "x_pos": x, "y_pos": y, "z_pos": z,
        "total_events": total,
        "cosmic_events": cosmic,
        "ambe_triggers": total - cosmic,
        # cn == 1 and cn != 1 are mutually exclusive in the production selection, so
        # these two partition the numerator rather than overlapping.
        "single_neutron_candidates": int((grp["n_accepted_single"] > 0).sum()),
        "multiple_neutron_candidates": int((grp["n_accepted_multiple"] > 0).sum()),
        "unique_neutron_triggers": int(has_cand.sum()),
        "unique_neutron_triggers_noncosmic": int((~cosmic_mask & has_cand).sum()),
        "cosmic_with_candidate": int((cosmic_mask & has_cand).sum()),
    }


def scan(sample: pd.DataFrame, features: pd.DataFrame,
         wins: List[Tuple[float, float, str, float]]) -> pd.DataFrame:
    """Run `tally` over every window and stack the results."""
    frames = []
    for lo, hi, edge, edge_val in wins:
        t = tally(sample, lo, hi)
        t["edge_moved"] = edge
        t["edge_value"] = edge_val
        t["acceptance"] = acceptance(features, lo, hi)
        t["n_waveforms_in_window"] = int(
            ((features["IC_adjusted"] > lo) & (features["IC_adjusted"] < hi)
             & (~features["second_pulse"].astype(bool))).sum())
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Closure test
# --------------------------------------------------------------------------- #
def closure_check(ctx: RunContext, df: pd.DataFrame, lo: float, hi: float) -> bool:
    """
    Compare the offline re-tally of the production window against the frozen summary.

    This is the gate for the whole study. If the offline filter is not equivalent to
    the Stage-1 gate, every scan point is wrong in the same invisible way, so nothing
    downstream may be believed until this passes cell for cell.
    """
    ref_paths = _expand(ctx.inputs.get("reference_trigger_summary"))
    if not ref_paths:
        print("icscan: no reference_trigger_summary configured; skipping comparison")
        return False
    ref = pd.concat([pd.read_csv(p) for p in ref_paths], ignore_index=True)

    cols = ["total_events", "cosmic_events", "ambe_triggers", "unique_neutron_triggers"]
    mine = df[df["position"] != SUM_LABEL].copy()

    print(f"\nClosure test: offline re-tally of ({lo:g}, {hi:g}) vs "
          f"{os.path.basename(ref_paths[0])}")
    print(f"{'position':>18}  " + "  ".join(f"{c:>24}" for c in cols))
    ok = True
    for _, row in mine.sort_values("y_pos").iterrows():
        r = ref[(ref["x_pos"] == row["x_pos"]) & (ref["y_pos"] == row["y_pos"])
                & (ref["z_pos"] == row["z_pos"])]
        if r.empty:
            print(f"{row['position']:>18}  NOT IN REFERENCE")
            ok = False
            continue
        r = r.iloc[0]
        cells = []
        for c in cols:
            match = int(row[c]) == int(r[c])
            ok = ok and match
            cells.append(f"{int(row[c]):>10} vs {int(r[c]):>8} {'ok' if match else 'MISMATCH'}")
        print(f"{row['position']:>18}  " + "  ".join(f"{c:>24}" for c in cells))

    print(f"\nClosure: {'PASS -- offline re-tally reproduces the frozen summary exactly' if ok else 'FAIL'}")
    if not ok:
        print("Do NOT interpret any scan output. Either the wide pass changed "
              "something it should not have, or the offline re-filter is not "
              "equivalent to the Stage-1 gate.")
    return ok


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
_XLABEL = {
    "lower": "lower IC_adjusted edge",
    "upper": "upper IC_adjusted edge",
    "slice": "slice centre in IC_adjusted",
    "both": "IC_adjusted window",
}


def _edge_label(mode: str, df: pd.DataFrame) -> str:
    edge = df["edge_moved"].iloc[0]
    return _XLABEL.get(edge, "IC_adjusted edge")


def _title(ctx: RunContext, base: str, mode: str, df: pd.DataFrame) -> str:
    """Thresholds folded into the title, per the house plot style."""
    lo, hi = df["ic_min"].min(), df["ic_max"].max()
    if mode == "left":
        window = f"upper edge fixed at {hi:g}"
    elif mode == "right":
        window = f"lower edge fixed at {lo:g}"
    elif mode == "differential":
        step = int(df["width"].iloc[0])
        window = f"disjoint {step}-wide slices over {lo:g}-{hi:g}"
    else:
        window = f"{lo:g} < IC_adjusted < {hi:g}"
    label = ctx.cuts.get("selection_label", "box cuts")
    # Wrapped, not shortened: savefig bbox="tight" grows the canvas to fit a
    # single-line title, which squashes the axes into a letterbox. The thresholds
    # belong in the title (house style), so wrap instead of dropping them.
    return "\n".join(textwrap.wrap(
        ctx.title(f"{base}, {window} – Port 5 centre, {label}"), width=64))


def _plot_curve(ctx: RunContext, df: pd.DataFrame, mode: str):
    """Efficiency vs the moved edge: the summed curve plus the five positions."""
    fig, ax = plt.subplots()
    for pos, grp in df.groupby("position", sort=True):
        grp = grp.sort_values("edge_value")
        if pos == SUM_LABEL:
            ax.errorbar(grp["edge_value"], grp["efficiency"], yerr=grp["eff_err_binom"],
                        marker="o", color="black", label="Port 5 summed", zorder=5)
        else:
            ax.errorbar(grp["edge_value"], grp["efficiency"], yerr=grp["eff_err_binom"],
                        marker=".", alpha=0.75, label=pos)
    ax.set_xlabel(_edge_label(mode, df))
    ax.set_ylabel("neutron-bearing fraction of AmBe triggers")
    ax.set_title(_title(ctx, "AmBe efficiency vs IC window", mode, df))
    ax.legend(frameon=False, ncol=2)
    save_plot(fig, ctx, f"ic_scan_{mode}_efficiency")


def _plot_yield(ctx: RunContext, df: pd.DataFrame, mode: str):
    """
    Numerator and denominator on their own axes.

    Not optional: the ratio alone cannot distinguish "the efficiency fell because we
    admitted junk triggers" from "it fell because we lost good ones".
    """
    s = df[df["position"] == SUM_LABEL].sort_values("edge_value")
    fig, ax = plt.subplots()
    ax.plot(s["edge_value"], s["ambe_triggers"], marker="o", label="AmBe triggers (denominator)")
    ax.plot(s["edge_value"], s["unique_neutron_triggers"], marker="s",
            label="unique neutron triggers (numerator)")
    ax.set_xlabel(_edge_label(mode, df))
    ax.set_ylabel("events")
    ax.set_title(_title(ctx, "AmBe trigger yield vs IC window", mode, df))
    ax.legend(frameon=False)
    save_plot(fig, ctx, f"ic_scan_{mode}_yield")


def _plot_acceptance(ctx: RunContext, df: pd.DataFrame, mode: str):
    """Stage-1 acceptance: what fraction of all acquisitions a window costs."""
    s = df[df["position"] == SUM_LABEL].sort_values("edge_value")
    fig, ax = plt.subplots()
    ax.plot(s["edge_value"], 100.0 * s["acceptance"], marker="o", color="black")
    ax.set_xlabel(_edge_label(mode, df))
    ax.set_ylabel("Stage-1 acceptance [%]")
    ax.set_title(_title(ctx, "Stage-1 acceptance vs IC window", mode, df))
    save_plot(fig, ctx, f"ic_scan_{mode}_acceptance")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def flatness(eff: np.ndarray, n: np.ndarray, err: np.ndarray
             ) -> Tuple[float, float, int]:
    """
    Weighted mean and chi2 of a set of slice efficiencies against a constant.

    Only meaningful on DISJOINT slices. Returns (mean, chi2, dof).
    """
    good = np.isfinite(eff) & np.isfinite(err) & (err > 0) & (n > 0)
    if good.sum() < 2:
        return float("nan"), float("nan"), 0
    w = 1.0 / np.square(err[good])
    mu = float(np.sum(w * eff[good]) / np.sum(w))
    chi2 = float(np.sum(w * np.square(eff[good] - mu)))
    return mu, chi2, int(good.sum() - 1)


def _widthsweep(ctx: RunContext, sample: pd.DataFrame, features: pd.DataFrame,
                lo0: float, hi0: float,
                plateau: Tuple[float, float]) -> pd.DataFrame:
    """
    Repeat the disjoint-slice scan at several slice widths.

    READ THE CAVEAT. Slice width is the RESOLUTION of the scan, and widening it
    pulls chi2/dof in two OPPOSING directions:
      - fewer, larger slices average over more of the curve, washing out any
        structure narrower than the slice (pushes chi2/dof down);
      - each slice holds more events, so its error bar shrinks like 1/sqrt(n)
        (pushes chi2/dof up, for structure that survives the averaging).
    Which one wins depends on the scale of the real structure relative to the width,
    so chi2/dof is NOT monotonic in width and a number from one width cannot be
    compared against another. Measured here: over the full range chi2/dof RISES
    (27 -> 193 from width 50 to 400), because the >1250 cliff spans ~250 units and
    so survives averaging while the errors keep shrinking.

    What the sweep is genuinely good for: a plateau must survive at EVERY width. If
    a range is flat at width 50 and stays flat at 250 -- where the error bars are
    several times smaller -- that is a much stronger statement than either alone.
    Locate the plateau edges at the narrowest width; test the plateau's flatness
    hardest at the widest width that still fits inside it.

    Two series are produced per width: slices tiling the FULL range, and slices
    tiling the plateau range. The second is necessary because full-range slices are
    anchored at lo0 and at large widths none of them land inside the plateau, which
    would report the plateau test as unmeasurable when it is merely misaligned.
    """
    frames = []
    for width in SWEEP_WIDTHS:
        for series, (a, b) in (("full", (lo0, hi0)), ("plateau", plateau)):
            wins = slices(a, b, width)
            if len(wins) < 2:
                continue
            t = scan(sample, features, wins)
            t["slice_width"] = width
            t["series"] = series
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _report_widthsweep(df: pd.DataFrame, plateau: Tuple[float, float]):
    s = df[df["position"] == SUM_LABEL]
    print("\nDisjoint-slice scan repeated at several slice widths.")
    print("chi2/dof is NOT comparable across rows: widening a slice averages away "
          "narrow structure\n(pushes it down) but shrinks the error bars (pushes it "
          "up). Read each row against its\nown resolution. The plateau columns use "
          f"slices tiling {plateau[0]:g}-{plateau[1]:g} directly.")
    print(f"\n{'width':>6} | {'full range':>32} | {'plateau %g-%g' % plateau:>32}")
    print(f"{'':>6} | {'n':>3} {'min evt':>8} {'mean':>8} {'chi2/dof':>10} | "
          f"{'n':>3} {'min evt':>8} {'mean':>8} {'chi2/dof':>10}")
    rows = []
    for width in sorted(s["slice_width"].unique()):
        cells, rec = [], {"slice_width": width}
        for series in ("full", "plateau"):
            g = s[(s.slice_width == width) & (s.series == series)].sort_values("ic_min")
            if len(g) < 2:
                cells.append(f"{'--':>3} {'':>8} {'':>8} {'too few':>10}")
                rec[f"n_{series}"] = len(g)
                continue
            eff = g["efficiency"].to_numpy(float)
            n = g["ambe_triggers"].to_numpy(float)
            er = g["eff_err_binom"].to_numpy(float)
            mu, c, d = flatness(eff, n, er)
            cells.append(f"{len(g):>3d} {int(n.min()):>8d} {mu:>8.4f} "
                         f"{c / d:>10.2f}")
            rec.update({f"n_{series}": len(g), f"mean_{series}": mu,
                        f"chi2_{series}": c, f"dof_{series}": d,
                        f"chi2dof_{series}": c / d})
        print(f"{width:>6.0f} | {cells[0]} | {cells[1]}")
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot_widthsweep(ctx: RunContext, df: pd.DataFrame):
    """Overlay the disjoint-slice curve at every width, on one axis."""
    s = df[df["position"] == SUM_LABEL]
    fig, ax = plt.subplots()
    for width, grp in s.groupby("slice_width"):
        g = grp.sort_values("edge_value")
        ax.errorbar(g["edge_value"], g["efficiency"], yerr=g["eff_err_binom"],
                    marker="o", markersize=4, alpha=0.85, label=f"{width:.0f} wide")
    ax.set_xlabel("slice centre in IC_adjusted")
    ax.set_ylabel("neutron-bearing fraction of AmBe triggers")
    ax.set_title("\n".join(textwrap.wrap(ctx.title(
        "AmBe efficiency on disjoint IC slices, slice width 50 to 500 – "
        "Port 5 centre, " + ctx.cuts.get("selection_label", "box cuts")), width=64)))
    ax.legend(frameon=False, ncol=2)
    save_plot(fig, ctx, "ic_scan_widthsweep_efficiency")

    _plot_widthsweep_panels(ctx, df)
    _plot_widthsweep_single(ctx, df)


def _widthsweep_ylim(s: pd.DataFrame) -> Tuple[float, float]:
    """
    Common y-range for the per-width figures.

    Shared deliberately: the whole point of splitting by width is to compare the
    curves, and per-panel autoscaling would rescale each one to fill its own axes,
    making a flat curve and a cliff look equally dramatic.
    """
    lo = float((s["efficiency"] - s["eff_err_binom"]).min())
    hi = float((s["efficiency"] + s["eff_err_binom"]).max())
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _plot_widthsweep_panels(ctx: RunContext, df: pd.DataFrame):
    """
    One panel per slice width, on a shared axis range.

    The overlay figure gets crowded at nine widths; this is the same data with one
    curve per panel so each is legible on its own.
    """
    s = df[(df["position"] == SUM_LABEL) & (df["series"] == "full")]
    widths = sorted(s["slice_width"].unique())
    ncol = 3
    nrow = int(math.ceil(len(widths) / ncol))
    ylim = _widthsweep_ylim(s)

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, width in zip(axes, widths):
        g = s[s.slice_width == width].sort_values("edge_value")
        ax.errorbar(g["edge_value"], g["efficiency"], yerr=g["eff_err_binom"],
                    marker="o", markersize=4, color="black")
        mu, chi2, dof = flatness(g["efficiency"].to_numpy(float),
                                 g["ambe_triggers"].to_numpy(float),
                                 g["eff_err_binom"].to_numpy(float))
        # Width and its own flatness in the panel title -- the number is only
        # interpretable next to the resolution it was measured at.
        ax.set_title(f"{width:.0f} wide, {len(g)} slices, "
                     f"chi2/dof {chi2 / dof:.1f}" if dof else f"{width:.0f} wide")
        ax.set_ylim(*ylim)
    for ax in axes[len(widths):]:
        ax.set_visible(False)
    for ax in axes[:len(widths)]:
        if not ax.get_subplotspec().is_last_row():
            continue
        ax.set_xlabel("slice centre in IC_adjusted")
    for i, ax in enumerate(axes[:len(widths)]):
        if i % ncol == 0:
            ax.set_ylabel("neutron-bearing fraction")

    fig.suptitle("\n".join(textwrap.wrap(ctx.title(
        "AmBe efficiency on disjoint IC slices, one panel per slice width – "
        "Port 5 centre, " + ctx.cuts.get("selection_label", "box cuts")), width=90)))
    fig.tight_layout()
    save_plot(fig, ctx, "ic_scan_widthsweep_panels")


def _plot_widthsweep_single(ctx: RunContext, df: pd.DataFrame):
    """
    One standalone figure per slice width, into plots/widthsweep/.

    Same shared y-range as the panel figure, so flipping between files is a fair
    comparison rather than nine differently-scaled axes.
    """
    s = df[(df["position"] == SUM_LABEL) & (df["series"] == "full")]
    ylim = _widthsweep_ylim(s)
    label = ctx.cuts.get("selection_label", "box cuts")
    for width in sorted(s["slice_width"].unique()):
        g = s[s.slice_width == width].sort_values("edge_value")
        mu, chi2, dof = flatness(g["efficiency"].to_numpy(float),
                                 g["ambe_triggers"].to_numpy(float),
                                 g["eff_err_binom"].to_numpy(float))
        fig, ax = plt.subplots()
        ax.errorbar(g["edge_value"], g["efficiency"], yerr=g["eff_err_binom"],
                    marker="o", color="black")
        ax.set_xlabel("slice centre in IC_adjusted")
        ax.set_ylabel("neutron-bearing fraction of AmBe triggers")
        ax.set_ylim(*ylim)
        flat_txt = f", flatness chi2/dof {chi2 / dof:.2f}" if dof else ""
        ax.set_title("\n".join(textwrap.wrap(ctx.title(
            f"AmBe efficiency on disjoint {width:.0f}-wide IC slices"
            f"{flat_txt} – Port 5 centre, {label}"), width=64)))
        save_plot(fig, ctx, f"ic_scan_slices_w{width:.0f}", subdir="widthsweep")


def _plot_grid(ctx: RunContext, df: pd.DataFrame):
    """2-D map of efficiency over arbitrary (lower, upper) window pairs."""
    s = df[df["position"] == SUM_LABEL]
    piv = s.pivot(index="ic_min", columns="ic_max", values="efficiency")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto",
                   extent=[piv.columns.min(), piv.columns.max(),
                           piv.index.min(), piv.index.max()])
    fig.colorbar(im, ax=ax, label="neutron-bearing fraction of AmBe triggers")
    ax.set_xlabel("upper IC_adjusted edge")
    ax.set_ylabel("lower IC_adjusted edge")
    ax.set_title("\n".join(textwrap.wrap(ctx.title(
        "AmBe efficiency over arbitrary IC windows – Port 5 centre, "
        + ctx.cuts.get("selection_label", "box cuts")), width=64)))
    save_plot(fig, ctx, "ic_scan_grid_efficiency")


def _report_grid(df: pd.DataFrame, step: float):
    """
    Rank arbitrary windows. Efficiency alone is maximised by a tiny window with no
    statistics, so the table is sorted by efficiency but reports yield beside it --
    the choice is a trade, not a maximum.
    """
    s = df[df["position"] == SUM_LABEL].copy()
    ref = s[(s.ic_min == 700) & (s.ic_max == 1200)]
    print(f"\nArbitrary windows: {len(s)} (lower, upper) pairs on a {step:g} grid.")
    if len(ref):
        r = ref.iloc[0]
        print(f"Production 700-1200: eff {r.efficiency:.4f}, "
              f"{int(r.unique_neutron_triggers)} neutrons")
    wide = s[s.width >= 400]
    print(f"\nTop 12 by efficiency among windows at least 400 wide "
          f"(narrow windows win on ratio but have no statistics):")
    print(f"{'window':>12} {'width':>6} {'neutrons':>9} {'eff':>8} {'+-':>7}")
    for _, r in wide.nlargest(12, "efficiency").iterrows():
        print(f"{r.ic_min:5.0f}-{r.ic_max:5.0f} {r.width:>6.0f} "
              f"{int(r.unique_neutron_triggers):>9} {r.efficiency:>8.4f} "
              f"{r.eff_err_binom:>7.4f}")
    print(f"\nMost neutrons among windows within 0.5 pp of the production efficiency:")
    if len(ref):
        thr = ref.iloc[0].efficiency - 0.005
        near = s[s.efficiency >= thr]
        print(f"{'window':>12} {'width':>6} {'neutrons':>9} {'eff':>8} {'vs prod':>9}")
        for _, r in near.nlargest(8, "unique_neutron_triggers").iterrows():
            d = 100 * (r.unique_neutron_triggers / ref.iloc[0].unique_neutron_triggers - 1)
            print(f"{r.ic_min:5.0f}-{r.ic_max:5.0f} {r.width:>6.0f} "
                  f"{int(r.unique_neutron_triggers):>9} {r.efficiency:>8.4f} {d:>+8.1f}%")


def _report(df: pd.DataFrame, mode: str):
    s = df[df["position"] == SUM_LABEL].sort_values("edge_value")
    print(f"\nPort-5 summed, mode={mode}")
    print(f"{'ic_min':>8} {'ic_max':>8} {'triggers':>10} {'neutrons':>10} "
          f"{'eff':>8} {'+-':>7} {'accept%':>8} {'eff_strict':>11}")
    for _, r in s.iterrows():
        print(f"{r['ic_min']:>8.0f} {r['ic_max']:>8.0f} {int(r['ambe_triggers']):>10} "
              f"{int(r['unique_neutron_triggers']):>10} {r['efficiency']:>8.4f} "
              f"{r['eff_err_binom']:>7.4f} {100 * r['acceptance']:>8.2f} "
              f"{r['efficiency_noncosmic']:>11.4f}")
    # Never let the production definition's cosmic double-count pass silently.
    dbl = int(s["cosmic_with_candidate"].max())
    if dbl:
        print(f"  eff uses the production definition, which counts up to {dbl} events "
              f"per window in BOTH cosmic_events and the numerator; eff_strict excludes "
              f"them. See icscan._counts.")

    if mode == "differential":
        # The interpretation the cumulative modes cannot support. These slices are
        # disjoint, so this is the one curve whose scatter is real statistics.
        f = s["efficiency"].to_numpy(dtype=float)
        n = s["ambe_triggers"].to_numpy(dtype=float)
        good = np.isfinite(f) & (n > 0)
        if good.sum() > 1:
            w = 1.0 / np.square(s["eff_err_binom"].to_numpy(dtype=float)[good])
            mean = float(np.sum(w * f[good]) / np.sum(w))
            chi2 = float(np.sum(w * np.square(f[good] - mean)))
            dof = int(good.sum() - 1)
            print(f"\nFlatness of the disjoint-slice curve (the only mode where this "
                  f"test is legitimate):")
            print(f"  weighted mean {mean:.4f}, chi2/dof = {chi2:.1f}/{dof} = "
                  f"{chi2 / dof:.2f} over the full {s['ic_min'].min():g}-"
                  f"{s['ic_max'].max():g} range")
            print(f"  slice range {np.nanmin(f[good]):.4f} to {np.nanmax(f[good]):.4f}, "
                  f"peak at slice centred {s['edge_value'].to_numpy()[good][np.nanargmax(f[good])]:g}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe data icscan")
    # No default. Each mode answers a different question and they are not
    # interchangeable -- a silent default would let the cumulative curve be read as
    # if it were the independent one.
    p.add_argument("--mode", required=True, choices=MODES,
                   help="left: walk the lower edge with the upper fixed. "
                        "right: walk the upper edge with the lower fixed. "
                        "differential: disjoint slices, the statistically "
                        "independent curve. widthsweep: differential repeated at "
                        "several slice widths. grid: every arbitrary (lower, upper) "
                        "pair. closure: re-tally the production window and check it "
                        "against the frozen summary.")
    p.add_argument("--width", type=float, default=None,
                   help="slice width for --mode differential (default: icscan.step). "
                        "This is the scan RESOLUTION: structure narrower than the "
                        "slice is averaged away.")
    p.add_argument("--no-plots", action="store_true",
                   help="write the CSV only")
    args = p.parse_args(list(argv) if argv else [])

    set_style()

    cfg = ctx.extra.get("icscan") or {}
    lo0 = float(cfg.get("ic_lo_start", 500))
    hi0 = float(cfg.get("ic_hi_start", 1500))
    step = float(cfg.get("step", 50))
    closure = (float(cfg.get("closure_ic_min", 700)),
               float(cfg.get("closure_ic_max", 1200)))

    stage1 = ctx.extra.get("stage1") or {}
    pass_lo = float(stage1.get("ic_min", lo0))
    pass_hi = float(stage1.get("ic_max", hi0))

    plateau = (float(cfg.get("plateau_min", 800)), float(cfg.get("plateau_max", 1250)))

    if args.mode == "widthsweep":
        wins = slices(lo0, hi0, min(SWEEP_WIDTHS))
    else:
        wins = windows(args.mode, lo0, hi0, step, closure, args.width)
    want_lo = min(w[0] for w in wins)
    want_hi = max(w[1] for w in wins)
    # The zero-by-construction guard. A window wider than the pass contains
    # waveforms Stage 2 never saw, which can only ever report zero neutrons.
    if want_lo < pass_lo or want_hi > pass_hi:
        raise SystemExit(
            f"icscan: requested windows span ({want_lo:g}, {want_hi:g}) but this tag "
            f"was processed at ({pass_lo:g}, {pass_hi:g}). Waveforms outside the pass "
            f"window were never Stage-2 processed, so they would report zero neutrons "
            f"by construction rather than by physics. Re-run `ambe data process` with "
            f"a stage1 window at least as wide as the scan."
        )

    features = load_features(ctx)
    sample = load_sample(ctx)

    if args.mode == "widthsweep":
        print(f"icscan: mode=widthsweep, widths {[int(w) for w in SWEEP_WIDTHS]}, "
              f"pass window ({pass_lo:g}, {pass_hi:g})")
        df = _widthsweep(ctx, sample, features, lo0, hi0, plateau)
    else:
        width_note = (f", slice width {args.width or step:g}"
                      if args.mode == "differential" else "")
        print(f"icscan: mode={args.mode}, {len(wins)} windows{width_note}, "
              f"pass window ({pass_lo:g}, {pass_hi:g})")
        df = scan(sample, features, wins)
        if args.mode == "differential":
            df["slice_width"] = args.width or step

    out = ctx.csv_path(f"ic_window_scan_{args.mode}")
    df.to_csv(out, index=False)
    print(f"✓ wrote {len(df)} rows -> {out}")

    if args.mode == "widthsweep":
        summary = _report_widthsweep(df, plateau)
        summary.to_csv(ctx.csv_path("ic_window_scan_widthsweep_flatness"), index=False)
        if not args.no_plots:
            _plot_widthsweep(ctx, df)
            print(f"✓ plots -> {ctx.plots_dir}")
        return df

    if args.mode == "grid":
        _report_grid(df, step)
        if not args.no_plots:
            _plot_grid(ctx, df)
            print(f"✓ plots -> {ctx.plots_dir}")
        return df

    _report(df, args.mode)

    if args.mode == "closure":
        closure_check(ctx, df[df["position"] != SUM_LABEL], closure[0], closure[1])
        return df

    if not args.no_plots:
        _plot_curve(ctx, df, args.mode)
        _plot_yield(ctx, df, args.mode)
        _plot_acceptance(ctx, df, args.mode)
        print(f"✓ plots -> {ctx.plots_dir}")

    return df


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    run(ctx, argv)


if __name__ == "__main__":
    raise SystemExit("Use: ambe data icscan --config <yaml> --mode <left|right|differential|closure>")
