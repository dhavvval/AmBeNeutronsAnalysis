"""
analyze_gamma_tagging.py
========================
Checkpoint analysis: in-tank prompt gamma coincidence tagging.

Background
----------
The AmBe source emits a neutron + 4.44 MeV prompt gamma simultaneously.
Confirmed from MC truth: ALL class -5 hits in the Lucho file come from this
gamma (files without the gamma, e.g. oneneutron / Steven's file, have zero
class -5 hits).

The gamma produces Cherenkov light via Compton scattering, arriving at PMTs
~18,000 ns BEFORE the neutron capture (Population 1).  This prompt signature
can be used as a coincidence tag: events where a gamma cluster is detected
~15-20 µs before the neutron capture window are high-confidence true AmBe
events.

What this script computes
--------------------------
1. Gamma detectability rate — what fraction of events have ≥N gamma hits?
2. Gamma cluster properties — n_hits, timing spread, spatial extent
3. Purity comparison — signal cluster purity in gamma-tagged vs untagged events
4. Population 2 characterisation — near-capture class-5 (offset ≥ -500 ns)
5. Summary table: gamma-tag efficiency vs neutron detection improvement

Note on Population 2 (offset ~-1.4 ns)
--------------------------------------
Origin is unclear — possibly secondary Compton scatters from the same gamma
that arrive coincident with capture by chance, or gamma captures on H.
Gamma tagging does NOT help reject Population 2 because those hits arrive
inside the capture window. Left as open question.

Run with:
    python analyze_gamma_tagging.py --run-name mc_local_trial \\
        --output-root /path/to/ambe_output

Reads:  <output-root>/<run-name>/parquet/*__pulses.parquet
        <output-root>/<run-name>/parquet/*__cluster_features.parquet
Writes: gamma_tagging_<run_name>.pdf
        gamma_tagging_<run_name>.txt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NEUTRON_CLASSES  = {1, 2, 3, 4}
POP1_CUTOFF_NS   = -500.0    # offset < this → Population 1 (prompt gamma)
MIN_GAMMA_HITS   = [1, 3, 5, 8]   # thresholds for "gamma detected"


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def build_event_gamma_table(pulses: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event summary:
      ref_t           — median time of neutron-class hits (capture reference)
      n_neutron_hits  — neutron hits
      n_c5_total      — total class-5 hits
      n_pop1          — Population 1 gamma hits (offset < POP1_CUTOFF_NS)
      n_pop2          — Population 2 near-capture hits (offset ≥ POP1_CUTOFF_NS)
      pop1_median_offset — median offset of Pop1 hits
      pop1_mad        — timing MAD of Pop1 hits (cluster tightness)
    """
    # Reference time = neutron-cluster median
    def neu_median(grp):
        neu = grp[grp["truth_class"].isin(NEUTRON_CLASSES)]["t"]
        return float(neu.median()) if len(neu) >= 2 else np.nan

    ref = pulses.groupby("eventID").apply(
        lambda g: g[g["truth_class"].isin(NEUTRON_CLASSES)]["t"].median(),
        include_groups=False
    ).rename("ref_t")

    merged = pulses.merge(ref, on="eventID")
    merged["offset_ns"] = merged["t"] - merged["ref_t"]

    rows = []
    for evid, grp in merged.groupby("eventID"):
        n_neu  = int(grp["truth_class"].isin(NEUTRON_CLASSES).sum())
        c5     = grp[grp["truth_class"] == -5]
        pop1   = c5[c5["offset_ns"] <  POP1_CUTOFF_NS]
        pop2   = c5[c5["offset_ns"] >= POP1_CUTOFF_NS]
        ref_t  = float(grp["ref_t"].iloc[0])

        pop1_med_off = float(pop1["offset_ns"].median()) if len(pop1) > 0 else np.nan
        pop1_mad     = float(1.4826 * (pop1["offset_ns"] -
                             pop1["offset_ns"].median()).abs().median()
                             ) if len(pop1) >= 3 else np.nan

        rows.append({
            "eventID":          evid,
            "ref_t":            ref_t,
            "n_neutron_hits":   n_neu,
            "n_c5_total":       len(c5),
            "n_pop1":           len(pop1),
            "n_pop2":           len(pop2),
            "pop1_median_offset": pop1_med_off,
            "pop1_sigma_mad":   pop1_mad,
        })
    return pd.DataFrame(rows)


def gamma_detectability_table(ev_tbl: pd.DataFrame, n_events_total: int):
    """Print and return gamma detectability rates."""
    lines = []
    def p(s=""): print(s); lines.append(s)

    p("=" * 65)
    p("  PROMPT GAMMA DETECTABILITY")
    p("=" * 65)
    p(f"  Total events analysed:        {n_events_total}")
    p(f"  Events with any class-5 hit:  "
      f"{(ev_tbl['n_c5_total'] > 0).sum()}  "
      f"({100*(ev_tbl['n_c5_total'] > 0).sum()/n_events_total:.1f}%)")
    p()
    p(f"  Population 1 (prompt gamma, offset < {POP1_CUTOFF_NS:.0f} ns):")
    pop1_present = ev_tbl[ev_tbl["n_pop1"] > 0]
    p(f"    Events with ≥1 gamma hit:   "
      f"{len(pop1_present)}/{n_events_total}  "
      f"({100*len(pop1_present)/n_events_total:.1f}%)")
    for thr in MIN_GAMMA_HITS:
        n = int((ev_tbl["n_pop1"] >= thr).sum())
        p(f"    Events with ≥{thr:<2} gamma hits: {n:>4}/{n_events_total}  "
          f"({100*n/n_events_total:.1f}%)  ← tagable at ms={thr}")
    p()
    if len(pop1_present) > 0:
        p(f"    When gamma is present:")
        p(f"      hits/event: mean={pop1_present['n_pop1'].mean():.1f}  "
          f"median={pop1_present['n_pop1'].median():.1f}  "
          f"p5={pop1_present['n_pop1'].quantile(.05):.0f}  "
          f"p95={pop1_present['n_pop1'].quantile(.95):.0f}")
        p(f"      median offset: {pop1_present['pop1_median_offset'].median():.0f} ns  "
          f"(= {pop1_present['pop1_median_offset'].median()/1000:.1f} µs before capture)")

    p()
    p(f"  Population 2 (near-capture, offset ≥ {POP1_CUTOFF_NS:.0f} ns):")
    p(f"    Events with ≥1 Pop2 hit:    "
      f"{(ev_tbl['n_pop2'] > 0).sum()}/{n_events_total}  "
      f"({100*(ev_tbl['n_pop2'] > 0).sum()/n_events_total:.1f}%)")
    pop2 = ev_tbl[ev_tbl["n_pop2"] > 0]
    if len(pop2) > 0:
        p(f"    Pop2 hits/event (when present): "
          f"mean={pop2['n_pop2'].mean():.1f}  median={pop2['n_pop2'].median():.1f}")
    p("=" * 65)
    return lines


def purity_by_gamma_tag(feat: pd.DataFrame, ev_tbl: pd.DataFrame,
                        gamma_min_hits: int = 5) -> dict:
    """
    Compare cluster purity between gamma-tagged and untagged events.
    gamma_min_hits: minimum Population 1 hits to call event "gamma-tagged".
    """
    tagged_ev   = set(ev_tbl[ev_tbl["n_pop1"] >= gamma_min_hits]["eventID"])
    untagged_ev = set(ev_tbl[ev_tbl["n_pop1"] <  gamma_min_hits]["eventID"])

    op = feat[feat["method"] == "optics"]
    results = {}
    for label, ev_set in [("gamma_tagged", tagged_ev), ("untagged", untagged_ev)]:
        sub = op[op["eventID"].isin(ev_set)]
        if len(sub) == 0:
            continue
        n_sig = int((sub["is_truth_neutron"] == 1).sum())
        n_spu = int((sub["is_truth_neutron"] == 0).sum())
        purity = n_sig / (n_sig + n_spu) if (n_sig + n_spu) > 0 else np.nan
        results[label] = {
            "n_events":   len(ev_set),
            "n_clusters": len(sub),
            "n_signal":   n_sig,
            "n_spurious": n_spu,
            "purity":     purity,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(ev_tbl: pd.DataFrame, feat: pd.DataFrame,
                 purity_results: dict, pdf: PdfPages):

    # Fig 1 — Pop1 hits per event distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fig 1 — Prompt gamma (Pop1) hit count per event", fontsize=12)

    pop1_counts = ev_tbl["n_pop1"]
    ax = axes[0]
    bins = np.arange(0, pop1_counts.max() + 2, 1)
    ax.hist(pop1_counts, bins=bins - 0.5, color="purple", alpha=0.7, edgecolor="white")
    for thr, col in zip([3, 5, 8], ["green", "orange", "red"]):
        n = int((pop1_counts >= thr).sum())
        frac = 100 * n / len(pop1_counts)
        ax.axvline(thr, color=col, ls="--", lw=1.5,
                   label=f"≥{thr} hits: {n} events ({frac:.0f}%)")
    ax.set_xlabel("Population 1 gamma hits per event")
    ax.set_ylabel("Events")
    ax.set_title("All events (0 = gamma not detectable)")
    ax.legend(fontsize=9)

    ax = axes[1]
    n_hits = ev_tbl["n_neutron_hits"]
    ax.scatter(n_hits, pop1_counts, s=8, alpha=0.4, color="steelblue")
    ax.set_xlabel("Neutron capture hits per event")
    ax.set_ylabel("Gamma (Pop1) hits per event")
    ax.set_title("Gamma hits vs neutron hits\n(correlation check)")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Fig 2 — Pop1 timing offset distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Fig 2 — Population 1 gamma timing offset from capture median", fontsize=12)
    offsets = ev_tbl.loc[ev_tbl["n_pop1"] > 0, "pop1_median_offset"].dropna()
    ax.hist(offsets / 1000, bins=40, color="purple", alpha=0.7)
    ax.axvline(offsets.median() / 1000, color="black", ls="--", lw=1.5,
               label=f"Median = {offsets.median()/1000:.1f} µs")
    ax.set_xlabel("Median gamma hit offset from neutron capture (µs)")
    ax.set_ylabel("Events")
    ax.set_title(f"n={len(offsets)} events with ≥1 gamma hit")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Fig 3 — Purity comparison: tagged vs untagged
    if len(purity_results) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        labels  = list(purity_results.keys())
        purities = [purity_results[k]["purity"] for k in labels]
        n_events = [purity_results[k]["n_events"] for k in labels]
        colors   = ["seagreen", "steelblue"]
        bars = ax.bar(labels, purities, color=colors[:len(labels)], alpha=0.8)
        for bar, p, n in zip(bars, purities, n_events):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{p:.3f}\n(n={n} events)", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("OPTICS cluster purity (fraction signal)")
        ax.set_title("Fig 3 — Cluster purity: gamma-tagged vs untagged events\n"
                     "(gamma-tagged = events with ≥5 Pop1 hits)")
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="red", ls=":", lw=1, label="50% purity")
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Fig 4 — Population 2 characterisation
    pop2 = ev_tbl[ev_tbl["n_pop2"] > 0]
    pop1_only = ev_tbl[ev_tbl["n_pop1"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, max(ev_tbl["n_pop2"].max(), 1) + 2, 1)
    ax.hist(ev_tbl["n_pop2"], bins=bins - 0.5, color="red", alpha=0.7,
            label=f"All events (mean={ev_tbl['n_pop2'].mean():.1f})")
    ax.set_xlabel("Population 2 (near-capture class-5) hits per event")
    ax.set_ylabel("Events")
    ax.set_title("Fig 4 — Population 2 near-capture contamination per event\n"
                 "(origin unclear — NOT removed by gamma tagging)")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Prompt gamma coincidence tagging checkpoint analysis.")
    ap.add_argument("--run-name",    required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--gamma-min-hits", type=int, default=5,
                    help="Min Pop1 hits to call event gamma-tagged (default: 5)")
    args = ap.parse_args()

    parquet_dir = Path(args.output_root) / args.run_name / "parquet"
    out_dir     = Path(args.output_root) / args.run_name

    pulses_path = parquet_dir / f"{args.run_name}__pulses.parquet"
    feat_path   = parquet_dir / f"{args.run_name}__cluster_features.parquet"

    if not pulses_path.exists():
        sys.exit(f"Parquet not found: {pulses_path}")
    if not feat_path.exists():
        sys.exit(f"Features parquet not found: {feat_path}\n"
                 "Run `ambe mc features` first.")

    print(f"\n[gamma_tagging] Loading parquets for {args.run_name} ...")
    pulses = pd.read_parquet(pulses_path)
    feat   = pd.read_parquet(feat_path)
    n_ev   = pulses["eventID"].nunique()
    print(f"  {n_ev} events, {len(pulses):,} hits")

    # Check if class -5 exists
    n_c5 = (pulses["truth_class"] == -5).sum()
    if n_c5 == 0:
        print(f"\n  ⚠  No class -5 hits found in {args.run_name}.")
        print(f"  This file was simulated without the AmBe prompt gamma.")
        print(f"  Gamma tagging is not applicable. Exiting.")
        sys.exit(0)

    print(f"  {n_c5:,} class -5 hits present — proceeding with gamma analysis")

    # Build per-event table
    print(f"[gamma_tagging] Building per-event gamma table ...")
    ev_tbl = build_event_gamma_table(pulses)

    # Detectability summary
    lines = gamma_detectability_table(ev_tbl, n_ev)

    # Purity by gamma tag
    print(f"\n[gamma_tagging] Comparing purity: tagged vs untagged ...")
    purity_results = purity_by_gamma_tag(feat, ev_tbl, args.gamma_min_hits)
    print(f"\n  Gamma-tagged (≥{args.gamma_min_hits} Pop1 hits):")
    for label, r in purity_results.items():
        print(f"    {label}: {r['n_events']} events, "
              f"{r['n_clusters']} clusters, purity={r['purity']:.3f}")

    # Append to lines
    lines += ["", f"  PURITY COMPARISON (gamma_min_hits={args.gamma_min_hits}):"]
    for label, r in purity_results.items():
        lines.append(f"    {label:<16}: events={r['n_events']:>4}  "
                     f"clusters={r['n_clusters']:>4}  "
                     f"signal={r['n_signal']:>4}  spurious={r['n_spurious']:>4}  "
                     f"purity={r['purity']:.4f}")

    # Save text summary
    txt_path = out_dir / f"gamma_tagging_{args.run_name}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"\nSaved: {txt_path}")

    # Figures
    pdf_path = out_dir / f"gamma_tagging_{args.run_name}.pdf"
    print(f"[gamma_tagging] Writing figures → {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        make_figures(ev_tbl, feat, purity_results, pdf)
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
