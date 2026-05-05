"""
analyze_truth_timing.py
=======================
Investigates the hypothesis:
  "Most neutron capture hits (class 1) fall within a ~50 ns window;
   the 500 ns tail is dominated by secondary recoil hits (class 2/3/4)
   with higher trackIDs."

Run with:
    python analyze_truth_timing.py

Reads:  /Users/dajana/Documents/ANNIETree_MC.root  (or ROOT_FILE below)
Writes: truth_timing_analysis.pdf  (saved to this script's directory)
        truth_timing_summary.txt   (printed + saved)

Outputs six figures:
  Fig 1 — Per-class hit-time spread (delta_t) distribution
  Fig 2 — Hit-time distributions relative to cluster median, split by class
  Fig 3 — TrackID vs hit time within each neutron truth cluster
  Fig 4 — Fraction of hits inside 50 ns window, by class
  Fig 5 — What fraction of >50 ns tail hits come from each class?
  Fig 6 — Per-event: n_hits_inside_50ns vs n_hits_outside_50ns by class
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import awkward as ak

# ---------------------------------------------------------------------------
# CONFIGURE THIS
# ---------------------------------------------------------------------------
ROOT_FILE   = "/Users/dajana/Documents/ANNIEEvent_MC_AmBe_oneutron.root"
OUT_DIR     = Path(__file__).parent
WINDOW_NS   = 50.0   # the hypothesis window to test
MAX_EVENTS  = None   # set to e.g. 200 to do a quick test; None = all events

NEUTRON_PDG = 2112
CLASS_NAMES = {
    0:  "Dark noise",
    1:  "Primary neutron",
    2:  "Secondary n←p",
    3:  "Secondary n←n",
    4:  "Secondary n←other",
    -5: "Non-neutron bg",
}
CLASS_COLORS = {1: "tomato", 2: "steelblue", 3: "seagreen", 4: "orange", 0: "gray", -5: "purple"}

BRANCHES = [
    "eventNumber",
    "DirectParent_HitTime",
    "DirectParent_NeutronAncestorTrackID",
    "DirectParent_NeutronAncestorPDG",
    "DirectParent_NeutronAncestorClass",
    "DirectParent_IsDarknoise",
]

# Raw hit branches — needed to count ALL detector hits including untraced dark noise
RAW_HIT_BRANCHES = [
    "eventNumber",
    "hitT",         # all PMT hit times (includes 11% untraced dark noise)
]


# ---------------------------------------------------------------------------
# Load ROOT data
# ---------------------------------------------------------------------------
def load_data(root_file: str, max_events=None):
    print(f"Reading {root_file} ...")
    with uproot.open(root_file) as f:
        tree = f["Event"]
        arr  = tree.arrays(BRANCHES, library="ak")
        # Raw hit times — graceful fallback if branch absent
        try:
            arr_raw = tree.arrays(RAW_HIT_BRANCHES, library="ak")
        except Exception:
            arr_raw = None
    if max_events is not None:
        arr     = arr[:max_events]
        if arr_raw is not None:
            arr_raw = arr_raw[:max_events]
    n_raw = f" + {len(arr_raw)} raw-hit events" if arr_raw is not None else " (raw hitT unavailable)"
    print(f"  Loaded {len(arr)} events{n_raw}.")
    return arr, arr_raw


# ---------------------------------------------------------------------------
# Build per-event, per-neutron-cluster hit records
# ---------------------------------------------------------------------------
def build_cluster_records(arr):
    """
    Returns a list of dicts, one per (event, neutron trackID):
      evid, trackID, n_class, hit_times (array), delta_t, n_hits,
      n_in_window, n_out_window, frac_in_window,
      tail_class_counts (dict: class->count among out-window hits)
    """
    records = []

    for i in range(len(arr)):
        evid  = int(arr["eventNumber"][i])
        times = np.asarray(ak.to_list(arr["DirectParent_HitTime"][i]),              dtype=float)
        tids  = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorTrackID"][i]), dtype=int)
        pdgs  = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorPDG"][i]),   dtype=int)
        cls   = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorClass"][i]),  dtype=int)

        # Only look at neutron-origin hits
        neutron_mask = (pdgs == NEUTRON_PDG)
        if not neutron_mask.any():
            continue

        for tid in np.unique(tids[neutron_mask]):
            m = neutron_mask & (tids == tid)
            hit_t  = times[m]
            hit_cls = cls[m]

            if len(hit_t) < 2:
                continue

            # "Core" of the cluster = median hit time
            median_t = np.median(hit_t)
            dt_from_median = np.abs(hit_t - median_t)

            in_win  = dt_from_median <= (WINDOW_NS / 2)
            out_win = ~in_win

            # Majority class among in-window hits
            if in_win.any():
                classes_in, counts_in = np.unique(hit_cls[in_win], return_counts=True)
                majority_class = int(classes_in[np.argmax(counts_in)])
            else:
                majority_class = int(hit_cls[0])

            tail_class_counts = {}
            if out_win.any():
                for c, cnt in zip(*np.unique(hit_cls[out_win], return_counts=True)):
                    tail_class_counts[int(c)] = int(cnt)

            records.append({
                "evid":             evid,
                "trackID":          int(tid),
                "majority_class":   majority_class,
                "n_hits":           len(hit_t),
                "delta_t":          float(np.ptp(hit_t)),
                "t_min":            float(hit_t.min()),
                "t_max":            float(hit_t.max()),
                "median_t":         float(median_t),
                "hit_times":        hit_t,
                "hit_classes":      hit_cls,
                "n_in_window":      int(in_win.sum()),
                "n_out_window":     int(out_win.sum()),
                "frac_in_window":   float(in_win.mean()),
                "tail_class_counts": tail_class_counts,
            })

    return records


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------
def analyse_and_plot(records, out_dir: Path, pdf_name: str = "truth_timing_analysis.pdf"):
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("hit_times", "hit_classes", "tail_class_counts")}
                        for r in records])

    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("TRUTH HIT TIMING ANALYSIS")
    summary_lines.append(f"Window tested: ±{WINDOW_NS/2:.0f} ns of per-cluster median  (={WINDOW_NS} ns total)")
    summary_lines.append(f"Total neutron truth clusters analysed: {len(df)}")
    summary_lines.append("=" * 65)

    for cls_id, cls_name in CLASS_NAMES.items():
        sub = df[df["majority_class"] == cls_id]
        if len(sub) == 0:
            continue
        summary_lines.append(f"\nClass {cls_id}  [{cls_name}]  — {len(sub)} clusters")
        summary_lines.append(f"  delta_t   mean={sub['delta_t'].mean():.1f} ns  median={sub['delta_t'].median():.1f} ns  max={sub['delta_t'].max():.1f} ns")
        summary_lines.append(f"  frac_in_{int(WINDOW_NS)}ns   mean={sub['frac_in_window'].mean():.3f}  median={sub['frac_in_window'].median():.3f}")
        summary_lines.append(f"  n_in_win  mean={sub['n_in_window'].mean():.1f}  n_out_win mean={sub['n_out_window'].mean():.1f}")

    # What makes up the tail (out-of-window hits)?
    tail_by_class = defaultdict(int)
    total_tail = 0
    for r in records:
        for c, cnt in r["tail_class_counts"].items():
            tail_by_class[c] += cnt
            total_tail += cnt

    summary_lines.append(f"\n--- Composition of hits OUTSIDE the {WINDOW_NS} ns window ---")
    summary_lines.append(f"Total out-of-window hits: {total_tail}")
    for c, cnt in sorted(tail_by_class.items()):
        pct = 100.0 * cnt / total_tail if total_tail > 0 else 0
        summary_lines.append(f"  Class {c:>3}  [{CLASS_NAMES.get(c, '?'):25s}]: {cnt:6d} hits  ({pct:.1f}%)")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (out_dir / "truth_timing_summary.txt").write_text(summary_text)

    # ---- Plots ----
    pdf_path = out_dir / pdf_name
    with PdfPages(pdf_path) as pdf:

        # Fig 1 — delta_t distribution per class
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, min(df["delta_t"].quantile(0.99), 600), 60)
        for cls_id in [1, 2, 3, 4]:
            sub = df[df["majority_class"] == cls_id]
            if len(sub) == 0:
                continue
            ax.hist(sub["delta_t"], bins=bins, alpha=0.55,
                    color=CLASS_COLORS[cls_id],
                    label=f"Class {cls_id}: {CLASS_NAMES[cls_id]}  (n={len(sub)})",
                    density=True)
        ax.axvline(WINDOW_NS, color="black", ls="--", lw=1.5, label=f"{WINDOW_NS} ns window")
        ax.set_xlabel("delta_t = t_max − t_min per truth cluster  (ns)")
        ax.set_ylabel("Density")
        ax.set_title("Fig 1 — Hit-time spread (Δt) by neutron truth class")
        ax.legend(fontsize=9)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 2 — Hit times relative to cluster median, by class (violin / histogram)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        axes = axes.flatten()
        class_list = [1, 2, 3, 4]
        for idx, cls_id in enumerate(class_list):
            sub_recs = [r for r in records if r["majority_class"] == cls_id]
            if not sub_recs:
                axes[idx].set_visible(False)
                continue
            all_offsets = []
            for r in sub_recs:
                offsets = r["hit_times"] - r["median_t"]
                all_offsets.extend(offsets.tolist())
            all_offsets = np.array(all_offsets)
            bins = np.linspace(-300, 300, 80)
            axes[idx].hist(all_offsets, bins=bins, color=CLASS_COLORS[cls_id],
                           alpha=0.7, density=True)
            axes[idx].axvline(-WINDOW_NS/2, color="black", ls="--", lw=1.2)
            axes[idx].axvline( WINDOW_NS/2, color="black", ls="--", lw=1.2,
                               label=f"±{WINDOW_NS/2:.0f} ns")
            axes[idx].set_title(f"Class {cls_id}: {CLASS_NAMES[cls_id]}")
            axes[idx].set_ylabel("Density")
            axes[idx].legend(fontsize=8)
        for ax in axes[-2:]:
            ax.set_xlabel("Hit time − cluster median  (ns)")
        fig.suptitle("Fig 2 — Hit-time offset from cluster median, by class", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 3 — trackID vs relative hit time scatter (sample of events)
        fig, ax = plt.subplots(figsize=(9, 6))
        sample = records[:min(100, len(records))]
        for r in sample:
            offsets = r["hit_times"] - r["median_t"]
            for ot, cls in zip(offsets, r["hit_classes"]):
                ax.scatter(ot, r["trackID"], s=6, alpha=0.3,
                           color=CLASS_COLORS.get(int(cls), "gray"))
        ax.axvline(-WINDOW_NS/2, color="black", ls="--", lw=1, label=f"±{WINDOW_NS/2:.0f} ns")
        ax.axvline( WINDOW_NS/2, color="black", ls="--", lw=1)
        ax.set_xlabel("Hit time − cluster median  (ns)")
        ax.set_ylabel("Neutron ancestor trackID")
        ax.set_title("Fig 3 — TrackID vs time offset (sample of 100 clusters)\nLow trackID = primary; high trackID = secondary recoil")
        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=CLASS_COLORS[c], label=f"Class {c}: {CLASS_NAMES[c]}")
                           for c in [1,2,3,4] if c in CLASS_COLORS]
        ax.legend(handles=legend_elements, fontsize=9)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 4 — Fraction of hits inside window, by class
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls_id in [1, 2, 3, 4]:
            sub = df[df["majority_class"] == cls_id]["frac_in_window"]
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=np.linspace(0, 1, 30), alpha=0.6,
                    color=CLASS_COLORS[cls_id],
                    label=f"Class {cls_id}: {CLASS_NAMES[cls_id]}",
                    density=True)
        ax.set_xlabel(f"Fraction of hits inside ±{WINDOW_NS/2:.0f} ns window")
        ax.set_ylabel("Density")
        ax.set_title(f"Fig 4 — What fraction of each cluster's hits fall inside {WINDOW_NS} ns?")
        ax.legend(fontsize=9)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 5 — Tail composition pie / bar
        if total_tail > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            labels = [f"Class {c}: {CLASS_NAMES.get(c,'?')}" for c in sorted(tail_by_class)]
            sizes  = [tail_by_class[c] for c in sorted(tail_by_class)]
            colors = [CLASS_COLORS.get(c, "gray") for c in sorted(tail_by_class)]
            ax.bar(labels, sizes, color=colors, alpha=0.8)
            ax.set_ylabel("Hit count")
            ax.set_title(f"Fig 5 — Class breakdown of hits OUTSIDE the {WINDOW_NS} ns window")
            ax.tick_params(axis='x', rotation=20)
            for i, (lbl, s) in enumerate(zip(labels, sizes)):
                pct = 100*s/total_tail
                ax.text(i, s + total_tail*0.005, f"{pct:.1f}%", ha='center', fontsize=9)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 6 — in-window vs out-window hit counts
        fig, ax = plt.subplots(figsize=(7, 6))
        for cls_id in [1, 2, 3, 4]:
            sub = df[df["majority_class"] == cls_id]
            if len(sub) == 0:
                continue
            ax.scatter(sub["n_in_window"], sub["n_out_window"], s=15, alpha=0.35,
                       color=CLASS_COLORS[cls_id],
                       label=f"Class {cls_id}: {CLASS_NAMES[cls_id]}")
        ax.set_xlabel(f"Hits inside {WINDOW_NS} ns window")
        ax.set_ylabel(f"Hits outside {WINDOW_NS} ns window")
        ax.set_title("Fig 6 — In-window vs out-of-window hit counts per cluster")
        ax.legend(fontsize=9)
        ax.plot([0, df["n_in_window"].max()], [0, 0], "k--", lw=0.8, alpha=0.5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved:  {pdf_path}")
    print(f"Saved:  {out_dir / 'truth_timing_summary.txt'}")
    return df


# ---------------------------------------------------------------------------
# Main  (legacy block — superseded by the block at end of file)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BONUS: test multiple window sizes to confirm bimodal gap
# ---------------------------------------------------------------------------
def test_multiple_windows(records, windows_ns=(20, 35, 50, 75, 100, 200, 500)):
    """
    For each window size, compute mean/median frac_in_window for class 1 clusters.
    If the distribution is bimodal with a gap, increasing the window beyond ~50ns
    should give essentially no improvement.
    """
    print("\n" + "=" * 60)
    print("WINDOW SIZE SENSITIVITY — Class 1 only")
    print(f"{'Window (ns)':>12} | {'mean frac_in':>13} | {'median frac_in':>15} | {'mean n_in':>10} | {'mean n_out':>11}")
    print("-" * 70)
    class1 = [r for r in records if r["majority_class"] == 1]
    for w in windows_ns:
        hw = w / 2.0
        fracs, n_ins, n_outs = [], [], []
        for r in class1:
            offsets = np.abs(r["hit_times"] - r["median_t"])
            n_in  = int((offsets <= hw).sum())
            n_out = int((offsets  > hw).sum())
            frac  = n_in / len(r["hit_times"]) if len(r["hit_times"]) > 0 else 0
            fracs.append(frac); n_ins.append(n_in); n_outs.append(n_out)
        print(f"{w:>12.0f} | {np.mean(fracs):>13.4f} | {np.median(fracs):>15.4f} | "
              f"{np.mean(n_ins):>10.1f} | {np.mean(n_outs):>11.1f}")
    print()

    # Also show: what fraction of clusters have ZERO out-of-window hits at each window size?
    print(f"{'Window (ns)':>12} | {'% clusters fully inside window':>32}")
    print("-" * 48)
    for w in windows_ns:
        hw = w / 2.0
        n_perfect = 0
        for r in class1:
            offsets = np.abs(r["hit_times"] - r["median_t"])
            if (offsets <= hw).all():
                n_perfect += 1
        print(f"{w:>12.0f} | {100*n_perfect/len(class1):>32.1f}%")


# ---------------------------------------------------------------------------
# Background / residual hit analysis  (Figs 7–11)
# ---------------------------------------------------------------------------

def build_background_records(arr, arr_raw=None):
    """
    Collect ALL DirectParent hits (all classes) and raw untraced hits,
    with timing offset relative to the truth neutron capture time
    (median of class-1 hits for each event).

    Returns a DataFrame with one row per hit:
        evid, cls, is_darknoise, hit_time, offset_ns
        (offset_ns = hit_time − neutron_reference_time; NaN if no class-1 hits)
    Also returns per-event summary dict for window analysis.
    """
    rows      = []
    raw_rows  = []
    per_event = []   # (evid, ref_time, n_neutron, n_darknoise_dp, n_nonneutron, n_raw_total)

    for i in range(len(arr)):
        evid  = int(arr["eventNumber"][i])
        times = np.asarray(ak.to_list(arr["DirectParent_HitTime"][i]),              dtype=float)
        cls   = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorClass"][i]),  dtype=int)
        pdgs  = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorPDG"][i]),   dtype=int)
        dn    = np.asarray(ak.to_list(arr["DirectParent_IsDarknoise"][i]),           dtype=int)

        if len(times) == 0:
            continue

        # Reference time = median of class-1 (primary neutron) hits
        neutron_mask = (pdgs == NEUTRON_PDG) & np.isin(cls, [1, 2, 3, 4])
        neutron_t    = times[neutron_mask]
        ref_t        = float(np.median(neutron_t)) if len(neutron_t) > 0 else np.nan

        # All DirectParent hits
        for t, c, d in zip(times, cls, dn):
            rows.append({
                "evid":       evid,
                "cls":        int(c),
                "is_darknoise": int(d),
                "hit_time":   float(t),
                "offset_ns":  float(t - ref_t) if not np.isnan(ref_t) else np.nan,
            })

        # Count by category for per-event summary
        n_neutron   = int(neutron_mask.sum())
        n_darknoise = int(dn.sum())
        n_nonneutron = int((~neutron_mask & (cls == -5)).sum())
        n_raw_total = 0

        # Raw untraced hits (hitT) — everything not in BackTracker
        if arr_raw is not None:
            raw_t = np.asarray(ak.to_list(arr_raw["hitT"][i]), dtype=float)
            n_raw_total = len(raw_t)
            for t in raw_t:
                raw_rows.append({
                    "evid":      evid,
                    "cls":       999,   # untraced — no class label
                    "hit_time":  float(t),
                    "offset_ns": float(t - ref_t) if not np.isnan(ref_t) else np.nan,
                })

        per_event.append({
            "evid":          evid,
            "ref_time":      ref_t,
            "n_neutron":     n_neutron,
            "n_darknoise_dp": n_darknoise,
            "n_nonneutron":  n_nonneutron,
            "n_raw_total":   n_raw_total,
        })

    df_dp  = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows) if raw_rows else None
    df_ev  = pd.DataFrame(per_event)
    return df_dp, df_raw, df_ev


def signal_to_background_table(df_dp, df_raw, windows_ns=(20, 50, 75, 100, 200, 500, 1000, 2000)):
    """
    For each window size (±window/2 of neutron reference time), count:
      - n_signal   : class 1 hits inside window
      - n_bg_dp    : class 0 + class -5 DirectParent hits inside window
      - n_raw      : all raw hitT hits inside window (if available)
      - S/B_dp     : n_signal / n_bg_dp
      - S/B_raw    : n_signal / n_raw (if available)
    """
    print("\n" + "=" * 75)
    print("SIGNAL vs BACKGROUND — Hit counts vs time window size")
    print(f"{'Window (ns)':>12} | {'n_sig':>8} | {'n_bg_dp':>9} | {'S/B_dp':>8} | "
          f"{'n_raw':>8} | {'S/B_raw':>8} | {'sig_efficiency':>16}")
    print("-" * 75)

    # Per-event reference times
    ev_refs = df_dp.groupby("evid")["offset_ns"].apply(lambda x: 0.0)  # offset is already relative

    sig_all  = df_dp[df_dp["cls"] == 1]["offset_ns"].dropna()
    bg_all   = df_dp[(df_dp["cls"].isin([0, -5]))]["offset_ns"].dropna()
    sig_total = len(sig_all)

    results = []
    for w in windows_ns:
        hw = w / 2.0
        n_sig = int((np.abs(sig_all) <= hw).sum())
        n_bg  = int((np.abs(bg_all)  <= hw).sum())
        sb    = n_sig / n_bg if n_bg > 0 else np.inf
        eff   = 100.0 * n_sig / sig_total if sig_total > 0 else 0.0

        n_raw, sb_raw = np.nan, np.nan
        if df_raw is not None:
            raw_offsets = df_raw["offset_ns"].dropna()
            n_raw = int((np.abs(raw_offsets) <= hw).sum())
            sb_raw = n_sig / n_raw if n_raw > 0 else np.inf

        print(f"{w:>12.0f} | {n_sig:>8d} | {n_bg:>9d} | {sb:>8.3f} | "
              f"{str(int(n_raw)) if not np.isnan(n_raw) else 'N/A':>8} | "
              f"{sb_raw if not np.isnan(sb_raw) else 'N/A':>8.3f} | "
              f"{eff:>15.1f}%")
        results.append({"window_ns": w, "n_signal": n_sig, "n_bg_dp": n_bg,
                         "sb_dp": sb, "sig_efficiency_pct": eff,
                         "n_raw": n_raw, "sb_raw": sb_raw})

    return pd.DataFrame(results)


def background_plots(df_dp, df_raw, df_ev, out_dir: Path, pdf_name: str = "background_timing_analysis.pdf"):
    """
    Figs 7–11: Signal vs background hit timing comparison.
    """
    pdf_path = out_dir / pdf_name
    with PdfPages(pdf_path) as pdf:

        # -- Fig 7: Time offset distributions for ALL classes (relative to neutron) --
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_classes = {1: "Primary neutron (signal)", 0: "Dark noise (BackTracker)", -5: "Non-neutron physics"}
        colors_map   = {1: "tomato", 0: "gray", -5: "steelblue"}
        clip_ns = 500.0   # clip display range for clarity
        for cls_id, label in plot_classes.items():
            sub = df_dp[(df_dp["cls"] == cls_id) & (df_dp["offset_ns"].notna())]
            offsets = sub["offset_ns"].clip(-clip_ns, clip_ns)
            if len(offsets) == 0:
                continue
            bins = np.linspace(-clip_ns, clip_ns, 80)
            ax.hist(offsets, bins=bins, density=True, alpha=0.55,
                    color=colors_map[cls_id], label=f"{label}  (n={len(offsets):,})")
        if df_raw is not None:
            offsets_raw = df_raw["offset_ns"].dropna().clip(-clip_ns, clip_ns)
            ax.hist(offsets_raw, bins=np.linspace(-clip_ns, clip_ns, 80),
                    density=True, alpha=0.3, color="orange",
                    label=f"All raw hits (hitT)  (n={len(offsets_raw):,})")
        ax.axvline(-WINDOW_NS/2, color="black", ls="--", lw=1.2, label=f"±{WINDOW_NS/2:.0f} ns window")
        ax.axvline( WINDOW_NS/2, color="black", ls="--", lw=1.2)
        ax.set_xlabel("Hit time offset from neutron capture (ns)")
        ax.set_ylabel("Density")
        ax.set_title("Fig 7 — Signal vs background: time offset from truth neutron capture")
        ax.legend(fontsize=9)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # -- Fig 8: Wide view (full event window) --
        fig, ax = plt.subplots(figsize=(10, 5))
        wide_ns = 5000.0
        bins_wide = np.linspace(-wide_ns, wide_ns, 100)
        for cls_id, label in plot_classes.items():
            sub = df_dp[(df_dp["cls"] == cls_id) & (df_dp["offset_ns"].notna())]
            offsets = sub["offset_ns"].clip(-wide_ns, wide_ns)
            if len(offsets) == 0: continue
            ax.hist(offsets, bins=bins_wide, density=True, alpha=0.55,
                    color=colors_map[cls_id], label=label)
        ax.axvspan(-WINDOW_NS/2, WINDOW_NS/2, alpha=0.1, color="green",
                   label=f"Signal window (±{WINDOW_NS/2:.0f} ns)")
        ax.set_xlabel("Hit time offset from neutron capture (ns)")
        ax.set_ylabel("Density")
        ax.set_title("Fig 8 — Wide view: background spread across event window (clipped to ±5000 ns)")
        ax.legend(fontsize=9)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # -- Fig 9: Per-event hit counts by class --
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col, title, color in zip(
            axes,
            ["n_neutron", "n_darknoise_dp", "n_nonneutron"],
            ["Neutron hits (class 1-4)", "Dark noise hits (DP)", "Non-neutron physics (class -5)"],
            ["tomato", "gray", "steelblue"]
        ):
            vals = df_ev[col].dropna()
            if vals.sum() == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                ax.set_title(title, fontsize=9)
                continue
            ax.hist(vals, bins=np.arange(0, vals.quantile(0.99)+2, 1), color=color, alpha=0.7)
            ax.set_xlabel("Hits per event")
            ax.set_ylabel("Events")
            ax.set_title(f"{title}\nmean={vals.mean():.1f}", fontsize=9)
        fig.suptitle("Fig 9 — Per-event hit count distributions by class", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # -- Fig 10: S/B ratio vs window size --
        sb_data = signal_to_background_table(df_dp, df_raw,
                   windows_ns=(10, 20, 50, 75, 100, 200, 500, 1000, 2000, 5000))
        sb_data_plot = sb_data[sb_data["sb_dp"].replace([np.inf], np.nan).notna()]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        windows_arr = sb_data_plot["window_ns"].values

        ax1.plot(windows_arr, sb_data_plot["sb_dp"], "o-", color="steelblue",
                 label="S/B (DirectParent classes)")
        if not sb_data_plot["sb_raw"].replace([np.inf], np.nan).isna().all():
            ax1.plot(windows_arr, sb_data_plot["sb_raw"].replace([np.inf], np.nan),
                     "s--", color="orange", label="S/B (all raw hitT)")
        ax1.axhline(1.0, color="red", ls=":", lw=1, label="S/B = 1")
        ax1.axvline(WINDOW_NS, color="black", ls="--", lw=1, label=f"{WINDOW_NS} ns")
        ax1.set_ylabel("Signal / Background ratio")
        ax1.set_title("Fig 10 — Signal-to-background ratio vs time window size")
        ax1.legend(fontsize=9); ax1.set_xscale("log")

        ax2.plot(windows_arr, sb_data_plot["sig_efficiency_pct"], "o-", color="tomato")
        ax2.axvline(WINDOW_NS, color="black", ls="--", lw=1)
        ax2.set_xlabel("Time window (ns)  [log scale]")
        ax2.set_ylabel("Signal efficiency (%)")
        ax2.set_title("Signal efficiency (fraction of class-1 hits captured)")
        ax2.set_ylim(0, 105)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # -- Fig 11: Cumulative hit density comparison --
        fig, ax = plt.subplots(figsize=(9, 5))
        sig_offsets = df_dp[(df_dp["cls"] == 1) & (df_dp["offset_ns"].notna())]["offset_ns"]
        bg_offsets  = df_dp[(df_dp["cls"].isin([0, -5])) & (df_dp["offset_ns"].notna())]["offset_ns"]

        windows_fine = np.logspace(0.5, 4, 60)   # 3 to 10000 ns
        sig_density, bg_density = [], []
        for w in windows_fine:
            hw = w / 2
            n_s = (np.abs(sig_offsets) <= hw).sum()
            n_b = (np.abs(bg_offsets)  <= hw).sum()
            # density = hits per ns in the window
            sig_density.append(n_s / w if w > 0 else 0)
            bg_density.append(n_b  / w if w > 0 else 0)

        ax.plot(windows_fine, sig_density, color="tomato",    label="Signal (class 1) density")
        ax.plot(windows_fine, bg_density,  color="steelblue", label="Background density")
        ax.axvline(WINDOW_NS, color="black", ls="--", lw=1, label=f"{WINDOW_NS} ns window")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Time window (ns)  [log scale]")
        ax.set_ylabel("Hits per ns  [log scale]")
        ax.set_title("Fig 11 — Hit density (hits/ns) vs window size\n"
                     "Signal cluster appears as plateau; background rises with window")
        ax.legend(fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved:  {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Non-neutron physics population analysis  (Figs 12–13)
# Separately studies the two class -5 populations visible in Figs 7/8:
#   Population 1 — prompt gamma hits  (~-4600 ns, far before capture)
#   Population 2 — near-capture hits  (~0 ns, actual contamination for OPTICS)
# ---------------------------------------------------------------------------

PROMPT_GAMMA_THRESHOLD_NS = -500.0   # offsets below this are Population 1

def analyze_nonneutron_populations(df_dp: pd.DataFrame):
    """
    Separate class -5 hits into two populations and report statistics.
    Prints a summary and returns a dict of DataFrames.
    """
    nonneutron = df_dp[(df_dp["cls"] == -5) & df_dp["offset_ns"].notna()].copy()
    if len(nonneutron) == 0:
        print("No class -5 hits found.")
        return {}

    pop1 = nonneutron[nonneutron["offset_ns"] <  PROMPT_GAMMA_THRESHOLD_NS]
    pop2 = nonneutron[nonneutron["offset_ns"] >= PROMPT_GAMMA_THRESHOLD_NS]

    print("\n" + "=" * 70)
    print("NON-NEUTRON PHYSICS (class -5) — TWO-POPULATION BREAKDOWN")
    print("=" * 70)
    print(f"Total class -5 hits: {len(nonneutron):,}")
    print()
    print(f"Population 1 — Prompt gamma (offset < {PROMPT_GAMMA_THRESHOLD_NS:.0f} ns):")
    print(f"  Count:  {len(pop1):,}  ({100*len(pop1)/len(nonneutron):.1f}% of class -5)")
    if len(pop1) > 0:
        print(f"  Offset: mean={pop1['offset_ns'].mean():.1f} ns  "
              f"median={pop1['offset_ns'].median():.1f} ns  "
              f"std={pop1['offset_ns'].std():.1f} ns")
        print(f"  → Interpretation: hits from the AmBe prompt gamma (4.44 MeV)")
        print(f"    emitted ~{abs(pop1['offset_ns'].median()):.0f} ns BEFORE neutron capture")
        print(f"    Trivially excluded by any time window < {abs(PROMPT_GAMMA_THRESHOLD_NS):.0f} ns")

    print()
    print(f"Population 2 — Near-capture contamination (offset ≥ {PROMPT_GAMMA_THRESHOLD_NS:.0f} ns):")
    print(f"  Count:  {len(pop2):,}  ({100*len(pop2)/len(nonneutron):.1f}% of class -5)")
    if len(pop2) > 0:
        print(f"  Offset: mean={pop2['offset_ns'].mean():.1f} ns  "
              f"median={pop2['offset_ns'].median():.1f} ns  "
              f"std={pop2['offset_ns'].std():.1f} ns")
        print(f"  → Interpretation: physics correlated with or produced BY the")
        print(f"    neutron capture itself (secondary gammas, scattered particles)")
        print(f"    These are the hits that form spurious OPTICS clusters.")
        print(f"    Cannot be separated by timing alone — spatial/MVA features needed")

    # Per-window S/B using ONLY Population 2 (the real contamination)
    sig_all = df_dp[(df_dp["cls"] == 1) & df_dp["offset_ns"].notna()]["offset_ns"]
    print()
    print(f"  Effective S/B when only counting Population 2 (real contamination):")
    print(f"  {'Window (ns)':>12} | {'n_sig':>8} | {'n_pop2_bg':>10} | {'S/B_pop2':>10}")
    print("  " + "-" * 50)
    for w in [20, 50, 75, 100, 200]:
        hw = w / 2.0
        ns  = int((np.abs(sig_all) <= hw).sum())
        nb2 = int((np.abs(pop2["offset_ns"]) <= hw).sum())
        sb  = ns / nb2 if nb2 > 0 else np.inf
        print(f"  {w:>12.0f} | {ns:>8d} | {nb2:>10d} | {sb:>10.2f}")

    return {"pop1": pop1, "pop2": pop2}


def population_plots(df_dp: pd.DataFrame, pop_dict: dict, out_dir: Path, pdf_name: str = "nonneutron_population_analysis.pdf"):
    """
    Fig 12 — Zoom into ±200 ns showing only Population 2 contamination.
    Fig 13 — Per-event Population 2 hit count distribution.
    """
    if not pop_dict:
        return

    pop2 = pop_dict.get("pop2", pd.DataFrame())
    sig  = df_dp[(df_dp["cls"] == 1) & df_dp["offset_ns"].notna()]

    pdf_path = out_dir / pdf_name
    with PdfPages(pdf_path) as pdf:

        # Fig 12 — tight zoom, Pop2 vs signal
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, xlim, title_suffix in zip(
            axes,
            [(-200, 200), (-50, 80)],
            ["±200 ns", "±50 ns (capture region)"]
        ):
            bins = np.linspace(xlim[0], xlim[1], 60)
            ax.hist(sig["offset_ns"].clip(*xlim), bins=bins, density=True,
                    alpha=0.6, color="tomato", label=f"Signal (class 1)  n={len(sig):,}")
            if len(pop2):
                ax.hist(pop2["offset_ns"].clip(*xlim), bins=bins, density=True,
                        alpha=0.6, color="steelblue",
                        label=f"Pop 2 class -5  n={len(pop2):,}")
            ax.axvline(-WINDOW_NS/2, color="black", ls="--", lw=1)
            ax.axvline( WINDOW_NS/2, color="black", ls="--", lw=1,
                        label=f"±{WINDOW_NS/2:.0f} ns window")
            ax.set_xlabel("Hit time offset from neutron capture (ns)")
            ax.set_ylabel("Density")
            ax.set_title(f"Fig 12 — Near-capture class -5 vs signal  ({title_suffix})")
            ax.legend(fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 13 — per-event Population 2 count
        if len(pop2):
            pop2_per_event = pop2.groupby("evid").size()
            sig_per_event  = sig.groupby("evid").size()

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(pop2_per_event.values, bins=np.arange(0, 30, 1),
                         color="steelblue", alpha=0.7)
            axes[0].set_xlabel("Population 2 class -5 hits per event")
            axes[0].set_ylabel("Events")
            axes[0].set_title(f"Fig 13a — Near-capture contamination per event\n"
                              f"mean={pop2_per_event.mean():.1f}, "
                              f"median={pop2_per_event.median():.1f}")

            axes[1].hist(sig_per_event.values, bins=np.arange(0, 40, 1),
                         color="tomato", alpha=0.7)
            axes[1].set_xlabel("Signal class-1 hits per event")
            axes[1].set_ylabel("Events")
            axes[1].set_title(f"Fig 13b — Signal hit count per event\n"
                              f"mean={sig_per_event.mean():.1f}, "
                              f"median={sig_per_event.median():.1f}")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved:  {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    import sys
    rfile = sys.argv[1] if len(sys.argv) > 1 else ROOT_FILE

    # Stamp output filenames with the input ROOT file stem so multiple runs
    # don't overwrite each other.
    # e.g. ANNIEEvent_MC_AmBe_wcsim_oneneutron.root  →  stem = "oneneutron"
    stem = Path(rfile).stem
    # keep just the last meaningful part (after last underscore group)
    # e.g. "ANNIEEvent_MC_AmBe_wcsim_oneneutron" → "oneneutron"
    _parts = stem.split("_")
    short_stem = "_".join(_parts[-2:]) if len(_parts) >= 2 else stem

    print(f"\nInput:  {rfile}")
    print(f"Prefix: {short_stem}")

    arr, arr_raw = load_data(rfile, max_events=MAX_EVENTS)
    records = build_cluster_records(arr)
    print(f"Built {len(records)} neutron truth cluster records.")

    # Override OUT_DIR to write stamped filenames alongside this script
    import types, inspect
    _out = OUT_DIR

    # Monkey-patch output paths used inside functions that write PDFs
    # by setting module-level OUT_DIR to a namespace with the stem prefix
    df = analyse_and_plot(records, _out, pdf_name=f"truth_timing_analysis_{short_stem}.pdf")
    test_multiple_windows(records)

    # --- Background analysis ---
    print("\n" + "=" * 65)
    print("BUILDING BACKGROUND RECORDS (all hit classes)...")
    df_dp, df_raw, df_ev = build_background_records(arr, arr_raw)
    print(f"  DirectParent hits: {len(df_dp):,}  "
          f"(classes: {dict(df_dp['cls'].value_counts().sort_index())})")
    if df_raw is not None:
        print(f"  Raw hitT entries:  {len(df_raw):,}")
    print(f"  Events with reference time: "
          f"{df_ev['ref_time'].notna().sum()} / {len(df_ev)}")
    print()
    signal_to_background_table(df_dp, df_raw)
    background_plots(df_dp, df_raw, df_ev, _out,
                     pdf_name=f"background_timing_analysis_{short_stem}.pdf")

    # --- Non-neutron physics population decomposition ---
    print("\n" + "=" * 70)
    print("DECOMPOSING CLASS -5 INTO TWO POPULATIONS...")
    pop_dict = analyze_nonneutron_populations(df_dp)
    population_plots(df_dp, pop_dict, _out,
                     pdf_name=f"nonneutron_population_analysis_{short_stem}.pdf")
