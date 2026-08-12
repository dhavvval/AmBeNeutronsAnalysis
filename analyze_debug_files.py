"""
analyze_debug_files.py
======================
Comparative hit-distribution analysis of the two debug MC files:

  1. ANNIEEvent_MC_AmBe_wcsim_oneneutron.root
       → pure single-neutron capture, no beam background
       → establishes the "clean signal" baseline

  2. ANNIEEvent_MC_AmBe_wcsim_oneneutrononepion.root
       → one neutron + one pion per event
       → directly characterises pion-induced contamination
       → pion hits = Population 2 class -5 from the Lucho beam study

Key questions answered:
  Q1  How many hits does a clean neutron capture produce?  (oneneutron)
  Q2  What does the pion add?  How many extra hits, what timing?
  Q3  Do pion hits overlap the neutron capture time window (±75 ns)?
  Q4  Are pion hits spatially separated from neutron hits?
  Q5  Would OPTICS separate the pion cluster from the neutron cluster?
  Q6  Which pion PDG contributes most to contamination?
  Q7  Does the pion inflate sigma_t of the neutron cluster?

Run with:
    python analyze_debug_files.py

⚠  Update ROOT_FILE_1 / ROOT_FILE_2 below if the files live elsewhere.

Outputs:
    debug_files_analysis.pdf   — 9 figures, side-by-side comparison
    debug_files_summary.txt    — key numbers printed and saved
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import awkward as ak

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE — update paths if files are not at these locations
# ─────────────────────────────────────────────────────────────────────────────
ROOT_FILE_1 = "/Users/dajana/Documents/ANNIEEvent_MC_AmBe_wcsim_oneneutron.root"
ROOT_FILE_2 = "/Users/dajana/Documents/ANNIEEvent_MC_AmBe_wcsim_oneneutrononepion.root"

OUT_DIR   = Path(__file__).parent
PDF_OUT   = OUT_DIR / "debug_files_analysis.pdf"
TXT_OUT   = OUT_DIR / "debug_files_summary.txt"
MAX_EVENTS = None   # set e.g. 200 for a quick test; None = all

TRUTH_WINDOW_NS    = 75.0   # our established truth window
HALF_WINDOW        = TRUTH_WINDOW_NS / 2.0
MATCH_TOL          = 8.0    # ns — hit↔DirectParent matching tolerance
DEFAULT_DP_OFFSET  = 15.0   # ns — fallback offset when matching fails

NEUTRON_CLASSES = {1, 2, 3, 4}

CLASS_NAMES = {
    1:  "Primary n (class 1)",
    2:  "Secondary n←p (class 2)",
    3:  "Secondary n←n (class 3)",
    4:  "Secondary n←other (class 4)",
    0:  "Dark noise (class 0)",
   -5:  "Non-neutron bg (class -5)",
   -99: "Untraced",
}
CLASS_COLORS = {
    1: "tomato", 2: "steelblue", 3: "seagreen",
    4: "orange", 0: "gray", -5: "purple", -99: "silver",
}

# PDG → particle name for class -5 ancestor breakdown
PDG_NAMES = {
    211: "π+", -211: "π−", 111: "π⁰",
    2212: "proton", -13: "µ+", 13: "µ−",
    22: "γ", 11: "e−", -11: "e+",
    2112: "neutron",
}


# ─────────────────────────────────────────────────────────────────────────────
# ROOT reading helpers  (same logic as processor.py)
# ─────────────────────────────────────────────────────────────────────────────

HIT_BRANCHES = [
    "eventNumber", "hitChankey", "hitT", "hitX", "hitY", "hitZ", "hitPE",
]
DP_BRANCHES = [
    "DirectParent_PMTID", "DirectParent_HitTime",
    "DirectParent_NeutronAncestorClass", "DirectParent_IsDarknoise",
    "DirectParent_NeutronAncestorTrackID", "DirectParent_NeutronAncestorPDG",
]


def _compute_offset(hit_ck, hit_t, dp_ck, dp_t):
    shared = set(np.unique(hit_ck)) & set(np.unique(dp_ck))
    offsets = []
    for ck in shared:
        ht = hit_t[hit_ck == ck]
        dt = dp_t[dp_ck == ck]
        if len(ht) == 1 and len(dt) == 1:
            offsets.append(float(dt[0] - ht[0]))
    return float(np.median(offsets)) if len(offsets) >= 3 else DEFAULT_DP_OFFSET


def _build_dp_lookup(dp_ck, dp_t, dp_cls, dp_dn, dp_tid, dp_pdg):
    lookup = {}
    for ck, t, cls, dn, tid, pdg in zip(dp_ck, dp_t, dp_cls, dp_dn, dp_tid, dp_pdg):
        lookup.setdefault(int(ck), []).append(
            (float(t), int(cls), int(dn), int(tid), int(pdg))
        )
    return lookup


def _match_hit(ck, t, dp_lookup, offset):
    entries = dp_lookup.get(int(ck))
    if not entries:
        return -99, 1, -1, -1          # untraced
    t_sh = t + offset
    best_dt, best = min(((abs(e[0] - t_sh), e) for e in entries), key=lambda x: x[0])
    if best_dt > MATCH_TOL:
        return -99, 1, -1, -1
    _, cls, dn, tid, pdg = best
    return cls, dn, tid, pdg


def load_hits(root_file: str, label: str, max_events=None) -> pd.DataFrame:
    """
    Read one ROOT file and return a per-hit DataFrame with truth labels.
    Columns: eventID, pmtID, t, x, y, z, pe, truth_class, ancestor_pdg, is_neutron
    """
    print(f"\nReading {label}: {root_file}")
    with uproot.open(root_file) as f:
        tree = f["Event"]
        available = set(tree.keys())

        wanted_hit = [b for b in HIT_BRANCHES if b in available]
        wanted_dp  = [b for b in DP_BRANCHES  if b in available]
        has_dp     = "DirectParent_PMTID" in available

        hit_arr = tree.arrays(wanted_hit, library="ak")
        dp_arr  = tree.arrays(wanted_dp,  library="ak") if has_dp else None

        if not has_dp:
            print(f"  WARNING: No DirectParent branches — all hits labeled untraced")

    n = min(len(hit_arr), max_events) if max_events else len(hit_arr)
    print(f"  Processing {n} events...")

    rows = []
    for i in range(n):
        evid   = int(hit_arr["eventNumber"][i])
        hit_ck = np.array(ak.to_list(hit_arr["hitChankey"][i]), dtype=int)
        hit_t  = np.array(ak.to_list(hit_arr["hitT"][i]),       dtype=float)
        hit_x  = np.array(ak.to_list(hit_arr["hitX"][i]),       dtype=float)
        hit_y  = np.array(ak.to_list(hit_arr["hitY"][i]),       dtype=float)
        hit_z  = np.array(ak.to_list(hit_arr["hitZ"][i]),       dtype=float)
        hit_pe = (np.array(ak.to_list(hit_arr["hitPE"][i]), dtype=float)
                  if "hitPE" in hit_arr.fields else np.ones(len(hit_ck)))

        if has_dp:
            dp_ck  = np.array(ak.to_list(dp_arr["DirectParent_PMTID"][i]),               dtype=int)
            dp_t   = np.array(ak.to_list(dp_arr["DirectParent_HitTime"][i]),             dtype=float)
            dp_cls = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorClass"][i]),dtype=int) \
                     if "DirectParent_NeutronAncestorClass" in wanted_dp else np.zeros(len(dp_ck), int)
            dp_dn  = np.array(ak.to_list(dp_arr["DirectParent_IsDarknoise"][i]),         dtype=int) \
                     if "DirectParent_IsDarknoise"          in wanted_dp else np.ones(len(dp_ck),  int)
            dp_tid = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorTrackID"][i]),dtype=int) \
                     if "DirectParent_NeutronAncestorTrackID" in wanted_dp else np.full(len(dp_ck),-1,int)
            dp_pdg = np.array(ak.to_list(dp_arr["DirectParent_NeutronAncestorPDG"][i]),  dtype=int) \
                     if "DirectParent_NeutronAncestorPDG"   in wanted_dp else np.full(len(dp_ck),-1,int)

            offset    = _compute_offset(hit_ck, hit_t, dp_ck, dp_t)
            dp_lookup = _build_dp_lookup(dp_ck, dp_t, dp_cls, dp_dn, dp_tid, dp_pdg)
        else:
            offset, dp_lookup = 0.0, {}

        for j in range(len(hit_ck)):
            cls, dn, tid, pdg = _match_hit(hit_ck[j], hit_t[j], dp_lookup, offset)
            rows.append({
                "eventID":     evid,
                "pmtID":       hit_ck[j],
                "t":           hit_t[j],
                "x":           hit_x[j],
                "y":           hit_y[j],
                "z":           hit_z[j],
                "pe":          hit_pe[j],
                "truth_class": cls,
                "ancestor_pdg":pdg,
                "is_neutron":  int(cls in NEUTRON_CLASSES),
            })

    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} hits, {df['eventID'].nunique():,} events")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-event summary builder
# ─────────────────────────────────────────────────────────────────────────────

def build_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event counts: n_hits_total, n_neutron, n_bg, n_class_minus5,
    n_darknoise, neutron_median_t, sigma_t_mad_neutron,
    n_pion_in_window (class -5 hits within HALF_WINDOW of neutron median).
    """
    rows = []
    for evid, grp in df.groupby("eventID"):
        n_t   = len(grp)
        n_neu  = int((grp["truth_class"].isin(NEUTRON_CLASSES)).sum())
        n_c5   = int((grp["truth_class"] == -5).sum())
        n_dn   = int((grp["truth_class"] == 0).sum())
        n_unt  = int((grp["truth_class"] == -99).sum())

        # Neutron hits for timing
        neu_t = grp.loc[grp["truth_class"].isin(NEUTRON_CLASSES), "t"].values
        if len(neu_t) >= 2:
            med_t    = float(np.median(neu_t))
            mad      = float(np.median(np.abs(neu_t - med_t)))
            sigma_mad = 1.4826 * mad
        else:
            med_t, sigma_mad = (float(neu_t[0]) if len(neu_t) == 1 else np.nan), np.nan

        # How many class -5 hits fall inside the truth window around the neutron capture?
        c5_t = grp.loc[grp["truth_class"] == -5, "t"].values
        n_c5_in_window = int(np.sum(np.abs(c5_t - med_t) <= HALF_WINDOW)) if (len(c5_t) > 0 and not np.isnan(med_t)) else 0

        rows.append({
            "eventID":           evid,
            "n_hits_total":      n_t,
            "n_neutron":         n_neu,
            "n_class_minus5":    n_c5,
            "n_darknoise":       n_dn,
            "n_untraced":        n_unt,
            "neutron_median_t":  med_t,
            "sigma_t_mad_neutron": sigma_mad,
            "n_c5_in_window":    n_c5_in_window,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Hit-level delta_t  (offset from per-event neutron cluster median)
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_t(df: pd.DataFrame, ev_summary: pd.DataFrame) -> pd.DataFrame:
    """Merge per-event neutron_median_t onto hits and compute delta_t."""
    merged = df.merge(ev_summary[["eventID", "neutron_median_t"]], on="eventID", how="left")
    merged["delta_t"] = merged["t"] - merged["neutron_median_t"]
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics (printed + saved)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(label, df, ev_sum, lines):
    def p(s=""):
        print(s); lines.append(s)

    p(f"\n{'='*70}")
    p(f"  {label}")
    p(f"{'='*70}")
    p(f"  Events:         {df['eventID'].nunique():>8,}")
    p(f"  Total hits:     {len(df):>8,}  (mean {len(df)/df['eventID'].nunique():.1f}/event)")

    # Class breakdown
    p(f"\n  Truth class breakdown:")
    vc = df["truth_class"].value_counts().sort_index()
    total = len(df)
    for cls, cnt in vc.items():
        name = CLASS_NAMES.get(cls, f"class {cls}")
        p(f"    {name:<35} {cnt:>8,}  ({100*cnt/total:.1f}%)")

    # Per-event stats
    p(f"\n  Per-event neutron hit count:")
    p(f"    mean={ev_sum['n_neutron'].mean():.1f}  "
      f"median={ev_sum['n_neutron'].median():.1f}  "
      f"p5={np.percentile(ev_sum['n_neutron'],5):.0f}  "
      f"p95={np.percentile(ev_sum['n_neutron'],95):.0f}")

    p(f"\n  Per-event sigma_t_mad (neutron hits only):")
    sm = ev_sum["sigma_t_mad_neutron"].dropna()
    p(f"    mean={sm.mean():.1f} ns  median={sm.median():.1f} ns  "
      f"p5={np.percentile(sm,5):.1f} ns  p95={np.percentile(sm,95):.1f} ns")

    c5 = df[df["truth_class"] == -5]
    if len(c5) > 0:
        p(f"\n  Class -5 (non-neutron bg):")
        p(f"    Total hits:  {len(c5):,}  ({100*len(c5)/total:.1f}% of all hits)")
        p(f"    Per event:   mean={ev_sum['n_class_minus5'].mean():.1f}  "
          f"median={ev_sum['n_class_minus5'].median():.1f}")
        p(f"\n  Class -5 hits inside ±{HALF_WINDOW:.0f} ns of neutron capture:")
        p(f"    Per event:   mean={ev_sum['n_c5_in_window'].mean():.2f}  "
          f"median={ev_sum['n_c5_in_window'].median():.2f}")
        n_contaminated = int((ev_sum["n_c5_in_window"] > 0).sum())
        p(f"    Events with ≥1 pion hit in window: {n_contaminated}/{len(ev_sum)} "
          f"({100*n_contaminated/len(ev_sum):.1f}%)")

        # PDG breakdown
        pdg_vc = c5["ancestor_pdg"].value_counts()
        p(f"\n  Class -5 ancestor PDG breakdown:")
        for pdg, cnt in pdg_vc.head(10).items():
            name = PDG_NAMES.get(int(pdg), f"PDG {pdg}")
            p(f"    {name:<15} {cnt:>8,}  ({100*cnt/len(c5):.1f}%)")
    else:
        p(f"\n  No class -5 hits (clean signal file ✓)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Per-event hit multiplicity comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_multiplicity(ev1, ev2, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Fig 1 — Per-event hit multiplicity by truth class", fontsize=13)

    cols = ["n_hits_total", "n_neutron", "n_class_minus5"]
    titles = ["Total hits/event", "Neutron hits/event (class 1–4)", "Class -5 hits/event"]
    max_bins = [120, 60, 50]

    for col_i, (col, title, mb) in enumerate(zip(cols, titles, max_bins)):
        for row_i, (ev, label, color) in enumerate(
            [(ev1, "One neutron (clean)", "tomato"),
             (ev2, "One n + one pion",    "steelblue")]
        ):
            ax = axes[row_i, col_i]
            vals = ev[col].values
            bins = np.arange(0, max(vals.max() if len(vals) > 0 else 1, mb) + 2, 1)
            ax.hist(vals, bins=bins, color=color, alpha=0.7, edgecolor="none")
            ax.axvline(np.median(vals), color="black", lw=1.5, ls="--",
                       label=f"median={np.median(vals):.1f}")
            ax.set_xlabel(title)
            ax.set_ylabel("Events")
            ax.set_title(f"{label}\nmean={vals.mean():.1f}, p5={np.percentile(vals,5):.0f}–p95={np.percentile(vals,95):.0f}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Hit time distributions (absolute hitT), stacked by class
# ─────────────────────────────────────────────────────────────────────────────

def fig_absolute_times(df1, df2, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fig 2 — Absolute hit time distributions by truth class", fontsize=13)

    for ax, df, label in [
        (axes[0], df1, "One neutron (clean)"),
        (axes[1], df2, "One n + one pion"),
    ]:
        bins = np.linspace(df["t"].quantile(0.01), df["t"].quantile(0.99), 100)
        for cls in sorted(df["truth_class"].unique()):
            sub = df.loc[df["truth_class"] == cls, "t"]
            if len(sub) == 0:
                continue
            name  = CLASS_NAMES.get(cls, f"class {cls}")
            color = CLASS_COLORS.get(cls, "black")
            ax.hist(sub.clip(bins[0], bins[-1]), bins=bins, histtype="step",
                    lw=1.5, color=color, label=f"{name} (n={len(sub):,})")

        ax.set_xlabel("Hit time (ns)")
        ax.set_ylabel("Hits")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Delta_t from neutron capture median, by class
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_t(dt1, dt2, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Fig 3 — Hit timing relative to neutron cluster median (δt)\n"
        f"Dashed lines = ±{HALF_WINDOW:.0f} ns truth window",
        fontsize=12,
    )

    for ax, dt, label in [
        (axes[0], dt1, "One neutron (clean)"),
        (axes[1], dt2, "One n + one pion"),
    ]:
        bins = np.linspace(-200, 200, 80)
        for cls in [1, 2, 3, 4, 0, -5, -99]:
            sub = dt.loc[dt["truth_class"] == cls, "delta_t"]
            if len(sub) < 5:
                continue
            name  = CLASS_NAMES.get(cls, f"class {cls}")
            color = CLASS_COLORS.get(cls, "black")
            ax.hist(sub.clip(-200, 200), bins=bins, histtype="step",
                    lw=1.8, color=color, label=f"{name} ({len(sub):,})")
        ax.axvline(-HALF_WINDOW, color="black", ls="--", lw=1.2)
        ax.axvline( HALF_WINDOW, color="black", ls="--", lw=1.2,
                    label=f"±{HALF_WINDOW:.0f} ns window")
        ax.set_xlabel("δt from neutron cluster median (ns)")
        ax.set_ylabel("Hits")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — KEY: Pion hit timing relative to neutron capture (zoom ±200 ns)
# ─────────────────────────────────────────────────────────────────────────────

def fig_pion_timing_zoom(dt2, pdf):
    """Pion-file only — zoom on ±200 ns and ±50 ns to show overlap with capture."""
    c5  = dt2[dt2["truth_class"] == -5]["delta_t"].dropna()
    sig = dt2[dt2["truth_class"].isin(NEUTRON_CLASSES)]["delta_t"].dropna()

    if len(c5) == 0:
        return  # nothing to plot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Fig 4 [KEY] — Pion hit timing vs neutron signal\n"
        "(One n + one pion file only)",
        fontsize=12,
    )

    for ax, xlim, subtitle in [
        (axes[0], (-200, 200), "Wide view ±200 ns"),
        (axes[1], (-50,   80), "Capture region ±50 ns"),
    ]:
        bins = np.linspace(xlim[0], xlim[1], 60)
        ax.hist(sig.clip(*xlim), bins=bins, density=True, alpha=0.6,
                color="tomato",    label=f"Neutron hits (n={len(sig):,})")
        ax.hist(c5.clip(*xlim),  bins=bins, density=True, alpha=0.6,
                color="purple",    label=f"Pion hits (n={len(c5):,})")
        ax.axvline(-HALF_WINDOW, color="black", ls="--", lw=1.2)
        ax.axvline( HALF_WINDOW, color="black", ls="--", lw=1.2,
                    label=f"±{HALF_WINDOW:.0f} ns window")
        frac_in = float((np.abs(c5) <= HALF_WINDOW).mean())
        ax.set_xlabel("δt from neutron cluster median (ns)")
        ax.set_ylabel("Density")
        ax.set_title(f"{subtitle}\n{100*frac_in:.1f}% of pion hits inside window")
        ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-event pion contamination in truth window
# ─────────────────────────────────────────────────────────────────────────────

def fig_pion_contamination_per_event(ev1, ev2, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Fig 5 — Pion contamination inside ±{HALF_WINDOW:.0f} ns truth window\n"
        "(per event)",
        fontsize=12,
    )

    for ax, ev, label, color in [
        (axes[0], ev1, "One neutron (clean)", "tomato"),
        (axes[1], ev2, "One n + one pion",    "purple"),
    ]:
        vals = ev["n_c5_in_window"].values
        bins = np.arange(0, vals.max() + 2, 1)
        ax.hist(vals, bins=bins - 0.5, color=color, alpha=0.7, edgecolor="white")
        ax.set_xlabel(f"Pion hits inside ±{HALF_WINDOW:.0f} ns per event")
        ax.set_ylabel("Events")
        n_contam = int((vals > 0).sum())
        ax.set_title(f"{label}\n{n_contam}/{len(vals)} events ({100*n_contam/len(vals):.1f}%) "
                     f"have ≥1 pion hit in window\nmean={vals.mean():.2f}")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Truth window scan: what fraction of pion hits fall inside window?
# ─────────────────────────────────────────────────────────────────────────────

def fig_window_scan(dt1, dt2, pdf):
    windows = np.array([10, 20, 35, 50, 75, 100, 150, 200, 500, 1000])

    results = {}
    for label, dt in [("One neutron (clean)", dt1), ("One n + one pion", dt2)]:
        sig_dt = dt.loc[dt["truth_class"].isin(NEUTRON_CLASSES), "delta_t"].abs().dropna()
        c5_dt  = dt.loc[dt["truth_class"] == -5,                 "delta_t"].abs().dropna()
        sig_in, c5_in = [], []
        for w in windows:
            hw = w / 2
            sig_in.append(float((sig_dt <= hw).mean()) if len(sig_dt) > 0 else 0.0)
            c5_in.append( float((c5_dt  <= hw).mean()) if len(c5_dt)  > 0 else 0.0)
        results[label] = (sig_in, c5_in)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Fig 6 — Truth window scan: fraction of hits inside window\n"
        "(signal = class 1–4, pion bg = class -5)",
        fontsize=12,
    )

    # Left: signal fraction in window (should plateau ~75 ns)
    ax = axes[0]
    for label, (sig_in, _) in results.items():
        color = "tomato" if "clean" in label else "steelblue"
        ax.semilogx(windows, sig_in, "o-", color=color, label=label)
    ax.axvline(TRUTH_WINDOW_NS, color="black", ls="--", lw=1.2, label=f"{TRUTH_WINDOW_NS:.0f} ns")
    ax.set_xlabel("Window half-width (ns)")
    ax.set_ylabel("Fraction of neutron hits inside window")
    ax.set_title("Signal hit capture vs window size")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: pion contamination fraction
    ax = axes[1]
    for label, (_, c5_in) in results.items():
        color = "tomato" if "clean" in label else "purple"
        ax.semilogx(windows, c5_in, "s--", color=color, label=label)
    ax.axvline(TRUTH_WINDOW_NS, color="black", ls="--", lw=1.2, label=f"{TRUTH_WINDOW_NS:.0f} ns")
    ax.set_xlabel("Window half-width (ns)")
    ax.set_ylabel("Fraction of class -5 hits inside window")
    ax.set_title("Pion hit contamination vs window size")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — PDG breakdown of class -5 ancestors (pion file)
# ─────────────────────────────────────────────────────────────────────────────

def fig_pdg_breakdown(df2, pdf):
    c5 = df2[df2["truth_class"] == -5]
    if len(c5) == 0:
        return

    pdg_vc = c5["ancestor_pdg"].value_counts().head(12)
    names  = [PDG_NAMES.get(int(p), f"PDG {p}") for p in pdg_vc.index]
    fracs  = 100 * pdg_vc.values / len(c5)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(names)), fracs, color="purple", alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("% of class -5 hits")
    ax.set_title("Fig 7 — Ancestor PDG of class -5 hits\n(one n + one pion file)")
    for i, (bar, val) in enumerate(zip(bars, fracs)):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Spatial distribution of hits: neutron vs pion (Y vs R)
# ─────────────────────────────────────────────────────────────────────────────

def fig_spatial_distribution(df1, df2, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Fig 8 — Hit spatial distribution (Y vs radial R)\n"
        "Neutron hits (red) vs pion hits (purple)",
        fontsize=12,
    )

    for row_i, (df, label) in enumerate([
        (df1, "One neutron (clean)"),
        (df2, "One n + one pion"),
    ]):
        df_r  = df.copy()
        df_r["R"] = np.sqrt(df_r["x"]**2 + df_r["z"]**2)   # horizontal radius (m)

        neu = df_r[df_r["truth_class"].isin(NEUTRON_CLASSES)]
        c5  = df_r[df_r["truth_class"] == -5]

        # Y vs R scatter
        ax = axes[row_i, 0]
        ax.scatter(neu["R"], neu["y"], s=2, alpha=0.15, color="tomato",
                   label=f"Neutron ({len(neu):,})")
        if len(c5) > 0:
            ax.scatter(c5["R"], c5["y"], s=3, alpha=0.25, color="purple",
                       label=f"Pion bg ({len(c5):,})")
        # Tank boundary
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(1.12 * np.abs(np.cos(theta)), 1.5 * np.sin(theta),
                "k-", lw=0.5, alpha=0.4, label="Tank boundary")
        ax.set_xlabel("Radial R (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8, markerscale=5)
        ax.set_xlim(0, 1.3)
        ax.set_ylim(-1.6, 1.6)

        # Y distribution (projection)
        ax = axes[row_i, 1]
        bins_y = np.linspace(-1.5, 1.5, 40)
        ax.hist(neu["y"].clip(-1.5, 1.5), bins=bins_y, density=True,
                alpha=0.6, color="tomato", label=f"Neutron ({len(neu):,})")
        if len(c5) > 0:
            ax.hist(c5["y"].clip(-1.5, 1.5), bins=bins_y, density=True,
                    alpha=0.6, color="purple", label=f"Pion bg ({len(c5):,})")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — Y projection")
        ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — sigma_t_mad comparison: does pion inflate neutron cluster spread?
# ─────────────────────────────────────────────────────────────────────────────

def fig_sigma_t_comparison(ev1, ev2, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Fig 9 — Neutron cluster sigma_t_mad comparison\n"
        "(computed from neutron-class hits only)",
        fontsize=12,
    )

    ax = axes[0]
    sm1 = ev1["sigma_t_mad_neutron"].dropna()
    sm2 = ev2["sigma_t_mad_neutron"].dropna()
    bins = np.linspace(0, 80, 60)
    ax.hist(sm1.clip(0, 80), bins=bins, alpha=0.6, color="tomato",
            label=f"One neutron  median={sm1.median():.1f} ns")
    ax.hist(sm2.clip(0, 80), bins=bins, alpha=0.6, color="steelblue",
            label=f"One n+pion   median={sm2.median():.1f} ns")
    ax.set_xlabel("sigma_t_mad of neutron hits (ns)")
    ax.set_ylabel("Events")
    ax.set_title("sigma_t_mad — neutron hits only\n(pion hits excluded from this calculation)")
    ax.legend(fontsize=9)

    # Right: scatter — sigma_t_mad vs n_neutron_hits
    ax = axes[1]
    for ev, label, color in [
        (ev1, "One neutron",  "tomato"),
        (ev2, "One n+pion",   "steelblue"),
    ]:
        ax.scatter(ev["n_neutron"], ev["sigma_t_mad_neutron"],
                   s=5, alpha=0.3, color=color, label=label)
    ax.set_xlabel("Neutron hits per event (n_neutron)")
    ax.set_ylabel("sigma_t_mad of neutron hits (ns)")
    ax.set_title("sigma_t_mad vs cluster size\n(higher n_hits → more thermalization scatter outliers?)")
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Validate file paths
    for fpath, name in [(ROOT_FILE_1, "oneneutron"), (ROOT_FILE_2, "oneneutrononepion")]:
        if not Path(fpath).exists():
            sys.exit(
                f"\n❌  File not found: {fpath}\n"
                f"    Update ROOT_FILE_1 / ROOT_FILE_2 at the top of this script.\n"
            )

    # Load both files
    df1 = load_hits(ROOT_FILE_1, "ONE-NEUTRON (clean signal)", MAX_EVENTS)
    df2 = load_hits(ROOT_FILE_2, "ONE-N + ONE-PION",           MAX_EVENTS)

    # Per-event summaries
    ev1 = build_event_summary(df1)
    ev2 = build_event_summary(df2)

    # Delta_t (hit timing relative to neutron cluster median)
    dt1 = compute_delta_t(df1, ev1)
    dt2 = compute_delta_t(df2, ev2)

    # Print + save summary text
    lines = ["DEBUG FILE ANALYSIS SUMMARY", "=" * 70,
             f"Truth window used: ±{HALF_WINDOW:.0f} ns  ({TRUTH_WINDOW_NS:.0f} ns full)"]
    print_summary("ONE-NEUTRON (clean signal)",   df1, ev1, lines)
    print_summary("ONE-NEUTRON + ONE-PION",        df2, ev2, lines)

    # Key comparison paragraph
    def p(s=""): print(s); lines.append(s)
    p("\n" + "="*70)
    p("  KEY COMPARISON FINDINGS")
    p("="*70)

    n1 = ev1["n_neutron"].median()
    n2 = ev2["n_neutron"].median()
    s1 = ev1["sigma_t_mad_neutron"].median()
    s2 = ev2["sigma_t_mad_neutron"].median()
    c5_in = ev2["n_c5_in_window"].mean()
    c5_evfrac = 100 * (ev2["n_c5_in_window"] > 0).mean()

    p(f"  Neutron hits/event:  clean={n1:.1f}  pion={n2:.1f}  "
      f"(pion adds {n2-n1:.1f} neutron hits on average — should be ~0)")
    p(f"  sigma_t_mad (ns):    clean={s1:.1f}  pion={s2:.1f}  "
      f"(inflation from pion = {s2-s1:.1f} ns)")
    p(f"  Pion hits in ±{HALF_WINDOW:.0f} ns window: {c5_in:.2f}/event mean, "
      f"{c5_evfrac:.1f}% of events have ≥1")

    # Window at which 50% of pion hits are included
    c5_dt = dt2.loc[dt2["truth_class"] == -5, "delta_t"].abs().dropna()
    if len(c5_dt) > 0:
        p(f"  Median |δt| of pion hits: {c5_dt.median():.1f} ns  "
          f"(75th-pct: {np.percentile(c5_dt,75):.1f} ns)")
        p(f"  At 75 ns window: {100*(c5_dt <= HALF_WINDOW).mean():.1f}% of pion hits are inside")

    TXT_OUT.write_text("\n".join(lines))
    print(f"\nSaved summary: {TXT_OUT}")

    # Generate all figures
    print(f"\nGenerating figures → {PDF_OUT} ...")
    with PdfPages(PDF_OUT) as pdf:
        fig_multiplicity(ev1, ev2, pdf)
        fig_absolute_times(df1, df2, pdf)
        fig_delta_t(dt1, dt2, pdf)
        fig_pion_timing_zoom(dt2, pdf)
        fig_pion_contamination_per_event(ev1, ev2, pdf)
        fig_window_scan(dt1, dt2, pdf)
        fig_pdg_breakdown(df2, pdf)
        fig_spatial_distribution(df1, df2, pdf)
        fig_sigma_t_comparison(ev1, ev2, pdf)

    print(f"Done. Saved: {PDF_OUT}")


if __name__ == "__main__":
    main()
