#!/usr/bin/env python3
"""
plot_multiplicity_compare.py

Neutron-cluster multiplicity per AmBe event, for the two selections, drawn on
SEPARATE pages (not overlaid):

  Selection 1 : OPTICS + frozen MVA            (rf_score >= --score-cut)
  Selection 2 : OPTICS + legacy box-cut        (passes_stage1: PE<80/CB<0.45/nHits>9)

Design follows the established toolkit style (set_style; integer-CENTERED bars via
np.arange + ax.bar — NOT bins=range(...)+align='left', which mis-aligns integer
multiplicities; that was the "weird spaced bins" issue in combined.py/basic.py).

Each page is normalised to FRACTION of events so the physics reads directly: an
AmBe decay emits ~1 neutron, so a cleaner selection peaks harder at multiplicity=1
and has a smaller >=2 tail.  A side-by-side fraction overlay page is included LAST
for direct comparison, but the per-selection pages are the primary deliverable.

Usage (myboy venv):
  python plot_multiplicity_compare.py \
      --scored .../ambe_all__data_features__scored.parquet \
      --score-col rf_score --score-cut 0.643 \
      --out .../ambe_all__multiplicity_compare.pdf
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.plotting import set_style  # noqa: E402

EVENT_KEYS = ["run", "event_tank_time"]
SEL1_COLOR = "#0077BB"   # OPTICS+MVA   (toolkit signal blue)
SEL2_COLOR = "#EE7733"   # OPTICS+cut   (toolkit prompt orange)


def per_event_multiplicity(sel: pd.DataFrame, all_events_index) -> np.ndarray:
    """Clusters-per-event, reindexed onto the full event list (0 where none)."""
    return (sel.groupby(EVENT_KEYS).size()
            .reindex(all_events_index, fill_value=0).to_numpy())


def mult_fractions(mult: np.ndarray, max_n: int):
    """Integer-centered (1..max_n) fraction of events; last bin is an overflow >=max_n."""
    centers = np.arange(1, max_n + 1)
    n_ev = len(mult)
    frac = np.array([(mult == n).mean() for n in centers[:-1]]
                    + [(mult >= max_n).mean()])
    return centers, frac, n_ev


def _stats(mult: np.ndarray) -> dict:
    m1 = mult[mult >= 1]   # events with >=1 selected cluster
    return {
        "n_events": int((mult >= 1).sum()),
        "single_frac": float((m1 == 1).mean()) if len(m1) else np.nan,
        "mean_mult": float(m1.mean()) if len(m1) else np.nan,
        "n2_frac": float((m1 == 2).mean()) if len(m1) else np.nan,
        "ge3_frac": float((m1 >= 3).mean()) if len(m1) else np.nan,
    }


def selection_page(pdf, mult: np.ndarray, color: str, title: str,
                   subtitle: str, max_n: int):
    """One selection, its own page. Fraction-of-events vs integer multiplicity."""
    # restrict to events that produced >=1 cluster (the physical 'neutron events')
    m1 = mult[mult >= 1]
    centers, frac, _ = mult_fractions(m1, max_n)
    st = _stats(mult)

    fig, ax = plt.subplots()
    bars = ax.bar(centers, frac, width=0.7, color=color, edgecolor="black",
                  linewidth=0.6)
    for b, f in zip(bars, frac):
        if f > 0:
            ax.text(b.get_x() + b.get_width() / 2, f + 0.01, f"{f:.3f}",
                    ha="center", va="bottom", fontsize=8)
    labels = [str(n) for n in centers[:-1]] + [f"≥{max_n}"]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Neutron-cluster multiplicity / event")
    ax.set_ylabel("Fraction of neutron events")
    ax.set_ylim(0, min(1.0, frac.max() * 1.18))
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
    ax.text(0.97, 0.95,
            f"neutron events: {st['n_events']:,}\n"
            f"single-neutron frac: {st['single_frac']:.3f}\n"
            f"mean multiplicity: {st['mean_mult']:.3f}\n"
            f"n=2: {st['n2_frac']:.3f}   n≥3: {st['ge3_frac']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.text(0.5, -0.16, "AmBe emits ~1 neutron/event → ideal is a hard peak at 1",
            transform=ax.transAxes, ha="center", fontsize=8, color="0.4")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return st


def overlay_page(pdf, m1: np.ndarray, m2: np.ndarray, max_n: int, run_label: str):
    """Final page: the two selections' fractions side by side for direct comparison."""
    c1, f1, _ = mult_fractions(m1[m1 >= 1], max_n)
    _, f2, _ = mult_fractions(m2[m2 >= 1], max_n)
    x = np.arange(1, max_n + 1)
    w = 0.4
    fig, ax = plt.subplots()
    ax.bar(x - w / 2, f1, width=w, color=SEL1_COLOR, edgecolor="black",
           linewidth=0.6, label="Selection 1 — OPTICS + MVA")
    ax.bar(x + w / 2, f2, width=w, color=SEL2_COLOR, edgecolor="black",
           linewidth=0.6, label="Selection 2 — OPTICS + legacy cut")
    labels = [str(n) for n in x[:-1]] + [f"≥{max_n}"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Neutron-cluster multiplicity / event")
    ax.set_ylabel("Fraction of neutron events")
    ax.set_title(f"Multiplicity — Selection 1 vs Selection 2\n{run_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--score-col", default="rf_score")
    p.add_argument("--score-cut", type=float, default=0.643)
    p.add_argument("--max-n", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    set_style()
    d = pd.read_parquet(args.scored)
    if "passes_stage1" not in d.columns:
        raise SystemExit("scored parquet needs 'passes_stage1' for Selection 2")
    all_idx = d.groupby(EVENT_KEYS).size().index
    run_label = (f"AmBe data — {d['run'].nunique()} runs"
                 if "run" in d else "AmBe data")

    sel1 = d[d[args.score_col] >= args.score_cut]
    sel2 = d[d["passes_stage1"]]
    m1 = per_event_multiplicity(sel1, all_idx)
    m2 = per_event_multiplicity(sel2, all_idx)

    with PdfPages(args.out) as pdf:
        s1 = selection_page(pdf, m1, SEL1_COLOR, "Selection 1 — OPTICS + MVA",
                            f"{run_label} · {args.score_col} ≥ {args.score_cut:.2f}",
                            args.max_n)
        s2 = selection_page(pdf, m2, SEL2_COLOR, "Selection 2 — OPTICS + legacy cut",
                            f"{run_label} · PE<80 / CB<0.45 / nHits>9",
                            args.max_n)
        overlay_page(pdf, m1, m2, args.max_n, run_label)

    print(f"[mult] wrote {args.out}")
    print(f"{'selection':<28}{'n_events':>10}{'single':>9}{'mean':>8}{'n2':>8}{'n>=3':>8}")
    for tag, s in [("Sel1 OPTICS+MVA", s1), ("Sel2 OPTICS+cut", s2)]:
        print(f"{tag:<28}{s['n_events']:>10,}{s['single_frac']:>9.3f}"
              f"{s['mean_mult']:>8.3f}{s['n2_frac']:>8.3f}{s['ge3_frac']:>8.3f}")


if __name__ == "__main__":
    main()
