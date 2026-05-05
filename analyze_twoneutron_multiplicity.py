"""
analyze_twoneutron_multiplicity.py
===================================
Deep-dive into truth neutron multiplicity for the two-neutron MC file.

Answers:
  1. What is the n_truth distribution? Are 0, 1, 2, 3+ neutrons correctly identified?
  2. For n_truth=2 events that are under-counted (n_pred=1 or n_pred=0):
       - Did OPTICS MERGE the two neutrons into one cluster?
       - Or did it MISS one neutron entirely?
  3. What separates the 70% correct vs 30% under-counted two-neutron events?
     (capture time difference, spatial separation, hit counts per neutron)
  4. Is the 75ns truth window causing any neutron clusters to be conflated?

Run:
    cd /Users/dajana/Documents/AmBe/AmBeNeutronsAnalysis
    source ~/venvs/annie/bin/activate
    python analyze_twoneutron_multiplicity.py

Reads:
    /Users/dajana/Documents/ambe_output/mc_twoneutron/parquet/*
    /Users/dajana/Documents/AmBe/AmBeNeutronsAnalysis/ANNIEEvent_MC_AmBe_twoneutrons.root
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import awkward as ak

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path("/Users/dajana/Documents/ambe_output")
RUN_NAME    = "mc_twoneutron"
ROOT_FILE   = "/Users/dajana/Documents/AmBe/AmBeNeutronsAnalysis/ANNIEEvent_MC_AmBe_twoneutrons.root"
OUT_DIR     = Path(__file__).parent
MAX_EVENTS  = None

NEUTRON_PDG    = 2112
TRUTH_WINDOW   = 75.0   # ns — must match what was used in optics.py
MIN_MATCH_FRAC = 0.50


# ---------------------------------------------------------------------------
# PART 1 — Truth multiplicity distribution directly from ROOT
# ---------------------------------------------------------------------------
def compute_truth_multiplicity(root_file: str, max_events=None) -> pd.DataFrame:
    """
    For each event, count the number of distinct neutron trackIDs
    with at least 2 in-window (±37.5 ns of their own median) hits.
    Returns per-event DataFrame with n_truth and per-neutron timing stats.
    """
    print(f"\nReading truth from {root_file} ...")
    branches = [
        "eventNumber",
        "DirectParent_HitTime",
        "DirectParent_NeutronAncestorTrackID",
        "DirectParent_NeutronAncestorPDG",
        "DirectParent_NeutronAncestorClass",
    ]
    with uproot.open(root_file) as f:
        arr = f["Event"].arrays(branches, library="ak")
    if max_events:
        arr = arr[:max_events]

    event_rows = []
    neutron_rows = []

    for i in range(len(arr)):
        evid  = int(arr["eventNumber"][i])
        times = np.asarray(ak.to_list(arr["DirectParent_HitTime"][i]),               dtype=float)
        tids  = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorTrackID"][i]), dtype=int)
        pdgs  = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorPDG"][i]),     dtype=int)
        cls   = np.asarray(ak.to_list(arr["DirectParent_NeutronAncestorClass"][i]),   dtype=int)

        neutron_mask = (pdgs == NEUTRON_PDG) & np.isin(cls, [1, 2, 3, 4])
        if not neutron_mask.any():
            event_rows.append({"evid": evid, "n_truth_raw": 0,
                                "n_truth_windowed": 0, "ref_t": np.nan})
            continue

        # Reference time = median of all neutron hits (for relative timing)
        all_neut_t = times[neutron_mask]
        event_ref_t = float(np.median(all_neut_t))

        n_truth_raw = 0
        n_truth_windowed = 0

        for tid in np.unique(tids[neutron_mask]):
            m = neutron_mask & (tids == tid)
            hit_t = times[m]
            if len(hit_t) < 2:
                continue

            n_truth_raw += 1
            median_t = float(np.median(hit_t))
            in_win = np.abs(hit_t - median_t) <= TRUTH_WINDOW / 2
            n_in_win = int(in_win.sum())

            if n_in_win >= 2:
                n_truth_windowed += 1

            neutron_rows.append({
                "evid":          evid,
                "trackID":       int(tid),
                "n_hits_total":  len(hit_t),
                "n_hits_inwin":  n_in_win,
                "median_t":      median_t,
                "delta_t":       float(np.ptp(hit_t)),
                "offset_from_event_ref": median_t - event_ref_t,
            })

        event_rows.append({"evid": evid, "n_truth_raw": n_truth_raw,
                            "n_truth_windowed": n_truth_windowed,
                            "ref_t": event_ref_t})

    df_ev = pd.DataFrame(event_rows)
    df_neut = pd.DataFrame(neutron_rows)

    # For two-neutron events: compute time separation between the two neutrons
    for evid, grp in df_neut.groupby("evid"):
        if len(grp) == 2:
            t_sep = abs(float(grp["median_t"].iloc[0]) - float(grp["median_t"].iloc[1]))
            df_ev.loc[df_ev["evid"] == evid, "t_sep_ns"] = t_sep

    print(f"  {len(df_ev)} events analysed")
    print(f"\n  n_truth_raw distribution:")
    print(df_ev["n_truth_raw"].value_counts().sort_index().to_string())
    print(f"\n  n_truth_windowed (≥2 in-window hits) distribution:")
    print(df_ev["n_truth_windowed"].value_counts().sort_index().to_string())

    return df_ev, df_neut


# ---------------------------------------------------------------------------
# PART 2 — Load OPTICS predictions and join with truth
# ---------------------------------------------------------------------------
def load_optics_predictions(output_root: Path, run_name: str) -> pd.DataFrame:
    """
    Load per-event OPTICS metrics and return a DataFrame with
    n_truth, n_predicted, n_matched, n_spurious, n_missed per event.
    Uses the best config: ms=8, xi=0.10, t_unit=25.
    """
    metrics_path = output_root / run_name / "parquet" / f"{run_name}__metrics.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Not found: {metrics_path}")

    met = pd.read_parquet(metrics_path)

    # Best OPTICS config
    optics = met[
        (met["method"] == "optics") &
        (met["min_samples"] == 8) &
        (met["xi"].round(2) == 0.10) &
        (met["t_unit_ns"].round(0) == 25)
    ].copy()

    cf = met[met["method"] == "clusterfinder"].copy()

    print(f"\n  OPTICS events: {len(optics)}  |  CF events: {len(cf)}")
    return optics, cf


# ---------------------------------------------------------------------------
# PART 3 — Merge and analyse two-neutron events
# ---------------------------------------------------------------------------
def analyse_two_neutron_events(df_ev_truth: pd.DataFrame,
                               df_neut: pd.DataFrame,
                               optics: pd.DataFrame,
                               cf: pd.DataFrame) -> dict:
    """
    For events with n_truth=2, classify OPTICS outcome:
      CORRECT  : n_matched=2, n_spurious=0
      MERGED   : n_predicted=1 and that 1 cluster has mixed trackIDs (both neutrons in one)
      MISSED_1 : n_predicted=1 and only one neutron is found (the other is absent)
      MISSED_2 : n_predicted=0 (both neutrons missed)
    """
    two_n_events = df_ev_truth[df_ev_truth["n_truth_windowed"] == 2]["evid"].values
    optics_2n = optics[optics["eventID"].isin(two_n_events)].copy()

    print(f"\n  Two-neutron events: {len(two_n_events)}")
    print(f"  OPTICS rows for these events: {len(optics_2n)}")

    # Classify outcome
    correct  = optics_2n[(optics_2n["n_matched"] == 2) & (optics_2n["n_spurious"] == 0)]
    under    = optics_2n[optics_2n["n_predicted"] < 2]
    over     = optics_2n[optics_2n["n_predicted"] > 2]

    # For under-counted: is it merged (n_pred=1) or totally missed (n_pred=0)?
    merged   = under[under["n_predicted"] == 1]
    both_miss= under[under["n_predicted"] == 0]
    partial  = merged[merged["n_matched"] == 1]  # found 1, missed 1 cleanly
    fused    = merged[merged["n_matched"] != 1]  # found 1 but it matched how?

    print(f"\n  n_truth=2 event outcomes (OPTICS ms=8, xi=0.10, t=25):")
    print(f"    Correct   (n_pred=2, n_matched=2): {len(correct):4d}  ({100*len(correct)/len(optics_2n):.1f}%)")
    print(f"    Under-count total:                 {len(under):4d}  ({100*len(under)/len(optics_2n):.1f}%)")
    print(f"      n_pred=1 (one cluster found):    {len(merged):4d}  ({100*len(merged)/len(optics_2n):.1f}%)")
    print(f"        → n_matched=1 (missed 1):      {len(partial):4d}  ({100*len(partial)/len(optics_2n):.1f}%)")
    print(f"        → n_matched≠1 (merge/other):   {len(fused):4d}  ({100*len(fused)/len(optics_2n):.1f}%)")
    print(f"      n_pred=0 (both missed):          {len(both_miss):4d}  ({100*len(both_miss)/len(optics_2n):.1f}%)")
    print(f"    Over-count (n_pred>2):             {len(over):4d}  ({100*len(over)/len(optics_2n):.1f}%)")

    # Time separation for correct vs under-counted
    t_sep_col = df_ev_truth.set_index("evid")["t_sep_ns"].to_dict()
    correct_t_sep = [t_sep_col.get(evid, np.nan) for evid in correct["eventID"]]
    under_t_sep   = [t_sep_col.get(evid, np.nan) for evid in under["eventID"]]
    correct_t_sep = [x for x in correct_t_sep if not np.isnan(x)]
    under_t_sep   = [x for x in under_t_sep   if not np.isnan(x)]

    if correct_t_sep and under_t_sep:
        print(f"\n  Time separation between 2 neutron captures (ns):")
        print(f"    Correct events:       mean={np.mean(correct_t_sep):8.1f}  "
              f"median={np.median(correct_t_sep):8.1f}  "
              f"std={np.std(correct_t_sep):7.1f}")
        print(f"    Under-count events:   mean={np.mean(under_t_sep):8.1f}  "
              f"median={np.median(under_t_sep):8.1f}  "
              f"std={np.std(under_t_sep):7.1f}")
        print(f"\n  → If under-counted events have SMALLER time separation,")
        print(f"    it confirms OPTICS merges temporally-close neutron captures.")

    return {
        "two_n_events": two_n_events,
        "optics_2n":    optics_2n,
        "correct":      correct,
        "under":        under,
        "merged":       merged,
        "both_miss":    both_miss,
        "correct_t_sep": correct_t_sep,
        "under_t_sep":   under_t_sep,
    }


# ---------------------------------------------------------------------------
# PART 4 — Truth window validation: are the two neutrons clearly separated?
# ---------------------------------------------------------------------------
def validate_truth_window(df_neut: pd.DataFrame):
    """
    For events with 2 neutron trackIDs, check whether their hit-time windows
    overlap. If both neutrons capture within 75 ns of each other, the truth
    window definition may conflate them.
    """
    print(f"\n{'='*60}")
    print("TRUTH WINDOW VALIDATION FOR TWO-NEUTRON EVENTS")
    print(f"Truth window: ±{TRUTH_WINDOW/2:.0f} ns per cluster median")
    print(f"{'='*60}")

    two_n = df_neut.groupby("evid").filter(lambda g: len(g) == 2)
    n_events = two_n["evid"].nunique()
    print(f"Events with exactly 2 neutron trackIDs: {n_events}")

    overlap_count = 0
    t_seps = []

    for evid, grp in two_n.groupby("evid"):
        t1, t2 = grp["median_t"].values
        t_sep = abs(t1 - t2)
        t_seps.append(t_sep)

        # Do the ±37.5 ns windows around each neutron overlap?
        window_overlap = t_sep < TRUTH_WINDOW
        if window_overlap:
            overlap_count += 1

    t_seps = np.array(t_seps)
    print(f"\nCapture time separation between the 2 neutrons:")
    print(f"  Mean:   {t_seps.mean():12.1f} ns")
    print(f"  Median: {np.median(t_seps):12.1f} ns")
    print(f"  Min:    {t_seps.min():12.1f} ns")
    print(f"  Max:    {t_seps.max():12.1f} ns")
    print(f"\n  Events where t_sep < {TRUTH_WINDOW:.0f} ns (windows OVERLAP):")
    print(f"    {overlap_count} / {n_events}  ({100*overlap_count/n_events:.1f}%)")
    print(f"  → These are events where truth labels for the 2 neutrons may be ambiguous")
    print(f"\n  Distribution of t_sep:")
    bins = [0, 50, 100, 200, 500, 1000, 5000, 50000, 500000]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((t_seps >= lo) & (t_seps < hi)).sum())
        pct = 100 * n / len(t_seps)
        bar = "█" * int(pct / 2)
        print(f"    [{lo:8.0f} – {hi:8.0f}) ns:  {n:5d} events ({pct:5.1f}%)  {bar}")

    return t_seps


# ---------------------------------------------------------------------------
# PART 5 — Plots
# ---------------------------------------------------------------------------
def make_plots(df_ev_truth, df_neut, outcome_dict, t_seps, out_dir):
    pdf_path = out_dir / "twoneutron_multiplicity_analysis.pdf"
    with PdfPages(pdf_path) as pdf:

        # Fig 1 — n_truth distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, title in zip(axes,
            ["n_truth_raw", "n_truth_windowed"],
            ["Raw (≥1 hit)", f"After truth window (≥2 hits, ≤±{TRUTH_WINDOW/2:.0f} ns)"]
        ):
            vc = df_ev_truth[col].value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color="steelblue", alpha=0.8)
            ax.set_xlabel("Number of truth neutrons per event")
            ax.set_ylabel("Events")
            ax.set_title(f"Fig 1 — n_truth ({title})")
            for x, v in zip(range(len(vc)), vc.values):
                ax.text(x, v + 2, str(v), ha="center", fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 2 — Capture time separation for correct vs under-counted
        ct = np.array(outcome_dict["correct_t_sep"])
        ut = np.array(outcome_dict["under_t_sep"])
        if len(ct) > 0 and len(ut) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            vmax = max(np.percentile(ct, 99), np.percentile(ut, 99)) if len(ct) and len(ut) else 1e5
            bins = np.logspace(1, np.log10(max(vmax, 1e4)), 50)
            ax.hist(ct, bins=bins, alpha=0.6, density=True, color="tomato",
                    label=f"Correctly identified  (n={len(ct)})")
            ax.hist(ut, bins=bins, alpha=0.6, density=True, color="steelblue",
                    label=f"Under-counted  (n={len(ut)})")
            ax.set_xscale("log")
            ax.axvline(TRUTH_WINDOW, color="black", ls="--", lw=1.2,
                       label=f"Truth window = {TRUTH_WINDOW:.0f} ns")
            ax.set_xlabel("Time separation between 2 neutron captures (ns)  [log scale]")
            ax.set_ylabel("Density")
            ax.set_title("Fig 2 — Capture time separation: correctly found vs under-counted\n"
                         "Smaller t_sep → harder for OPTICS to separate the two clusters")
            ax.legend(fontsize=9)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 3 — t_sep distribution with annotation
        fig, ax = plt.subplots(figsize=(9, 5))
        bins_log = np.logspace(np.log10(max(t_seps.min(), 1)), np.log10(t_seps.max()), 60)
        ax.hist(t_seps, bins=bins_log, color="steelblue", alpha=0.7)
        ax.axvline(TRUTH_WINDOW, color="red", ls="--", lw=1.5,
                   label=f"Truth window ({TRUTH_WINDOW:.0f} ns)\nevents left of this: windows overlap")
        ax.set_xscale("log")
        ax.set_xlabel("Time separation between 2 neutron captures (ns)  [log scale]")
        ax.set_ylabel("Events")
        ax.set_title("Fig 3 — Two-neutron capture time separation distribution\n"
                     "The thermalization time spread determines how separable the clusters are")
        ax.legend(fontsize=9)
        # Annotate thermalization timescales
        for t_label, t_val, color in [
            ("~40 ns\n(detector\ncrossing)", 40, "green"),
            ("~1 µs", 1e3, "orange"),
            ("~100 µs", 1e5, "red"),
        ]:
            if t_val < t_seps.max():
                ax.axvline(t_val, color=color, ls=":", lw=1, alpha=0.6)
                ax.text(t_val * 1.1, ax.get_ylim()[1] * 0.8, t_label,
                        color=color, fontsize=8, va="top")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Fig 4 — Per-neutron hit count distribution for two-neutron events
        two_n_data = df_neut[df_neut["evid"].isin(outcome_dict["two_n_events"])]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(two_n_data["n_hits_inwin"], bins=np.arange(0, 35, 1),
                     color="tomato", alpha=0.7)
        axes[0].set_xlabel("In-window hits per neutron cluster")
        axes[0].set_ylabel("Neutron clusters")
        axes[0].set_title(f"Fig 4a — In-window hit count per neutron\n"
                          f"mean={two_n_data['n_hits_inwin'].mean():.1f}")

        axes[1].hist(two_n_data["delta_t"], bins=np.linspace(0, 200, 50),
                     color="steelblue", alpha=0.7)
        axes[1].set_xlabel("Hit time spread δt (ns) per neutron cluster")
        axes[1].set_ylabel("Neutron clusters")
        axes[1].set_title(f"Fig 4b — Time spread per neutron\nmedian={two_n_data['delta_t'].median():.1f} ns")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved: {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root_file = sys.argv[1] if len(sys.argv) > 1 else ROOT_FILE

    print("=" * 65)
    print("TWO-NEUTRON MULTIPLICITY ANALYSIS")
    print("=" * 65)

    # Part 1: Truth from ROOT
    df_ev_truth, df_neut = compute_truth_multiplicity(root_file, MAX_EVENTS)

    # Part 2: OPTICS predictions
    optics, cf = load_optics_predictions(OUTPUT_ROOT, RUN_NAME)

    # Part 3: Two-neutron event analysis
    print(f"\n{'='*60}")
    print("TWO-NEUTRON EVENT OUTCOME ANALYSIS")
    print("="*60)
    outcome = analyse_two_neutron_events(df_ev_truth, df_neut, optics, cf)

    # Part 4: Truth window validation
    t_seps = validate_truth_window(df_neut)

    # Part 5: Plots
    make_plots(df_ev_truth, df_neut, outcome, t_seps, OUT_DIR)

    print("\n" + "="*65)
    print("KEY QUESTION ANSWERED:")
    print("="*65)
    two_n = df_ev_truth[df_ev_truth["n_truth_windowed"] == 2]
    one_n = df_ev_truth[df_ev_truth["n_truth_windowed"] == 1]
    zero_n = df_ev_truth[df_ev_truth["n_truth_windowed"] == 0]
    total = len(df_ev_truth)
    print(f"  Events with 0 reconstructable neutrons: {len(zero_n):5d} ({100*len(zero_n)/total:.1f}%)")
    print(f"  Events with 1 reconstructable neutron:  {len(one_n):5d} ({100*len(one_n)/total:.1f}%)")
    print(f"  Events with 2 reconstructable neutrons: {len(two_n):5d} ({100*len(two_n)/total:.1f}%)")
    print(f"\n  Of the {len(outcome['two_n_events'])} two-neutron events OPTICS sees:")
    print(f"    Correctly finds both:    {len(outcome['correct']):5d}  ({100*len(outcome['correct'])/len(outcome['optics_2n']):.1f}%)")
    print(f"    Merges/misses one:       {len(outcome['merged']):5d}  ({100*len(outcome['merged'])/len(outcome['optics_2n']):.1f}%)")
    print(f"    Misses both:             {len(outcome['both_miss']):5d}  ({100*len(outcome['both_miss'])/len(outcome['optics_2n']):.1f}%)")
    print()
