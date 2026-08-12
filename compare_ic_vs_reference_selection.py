#!/usr/bin/env python
"""
One-on-one: our IC-window selection vs the reference's structure
(fprompt + over-threshold width, NO integrated-charge window).

  A  = 700 < IC_adjusted < 1200                  AND not second_pulse
  B  = 0.15 < fprompt < 0.30 AND width > W       AND not second_pulse

B mirrors `PMT_wvfm_selection` from lmlepin9/2x2_neutron_sources: a shape gate
plus a width gate, with nothing cutting on total charge. The *thresholds* are
not transferable from the reference -- their record is ~1000 samples with the
peak at bin ~115 and their width counts samples above 70 ADC over 115 bins,
while ours is 34992 samples, peak ~366, counting above 10 ADC over 850 bins --
so W is scanned rather than translated, and the headline comparison is made at
**equal acceptance**, where the question becomes: for the same number of tags,
which selection captures more neutrons?

COVERAGE CAVEAT, and it is the main limitation here. "Did this event yield a
neutron?" is only knowable for waveforms some Stage-2 pass actually processed:

    processed = (700<IC<1200 & ~veto)                     [the gated pass]
              | (500<IC<2000 & 0.15<fp<0.30 & ~veto)      [the fprompt pass]

B keeps waveforms with IC < 500 or IC > 2000 that **neither** pass ever saw, so
their neutron content is unmeasurable without a new pipeline run. Every table
below reports the processed fraction so the blind part is explicit, and P is
quoted only over the processed subset.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python -u compare_ic_vs_reference_selection.py
"""
import sys

import numpy as np
import pandas as pd

RUNS = [6046, 6056, 6060, 6061, 6062, 6165, 6166, 6186, 6187, 6188, 6189,
        6230, 6231, 6232, 6234, 6235, 6237, 6239, 6241, 6242]
FEAT = "TriggerSummary/WaveformFeatures_AmBe2.0v4_fprompt_{run}.parquet"
CAND = "EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_AmBe2.0v4_{tag}_{run}.csv"
IC_LO, IC_HI = 700, 1200
FP_LO, FP_HI = 0.15, 0.30


def load():
    out = []
    for run in RUNS:
        d = pd.read_parquet(FEAT.format(run=run),
                            columns=["timestamp", "IC_adjusted", "fprompt",
                                     "width_over_thresh", "second_pulse"])
        seen = set()
        for tag in ("gated", "fprompt"):
            seen |= set(pd.read_csv(CAND.format(tag=tag, run=run),
                                    usecols=["eventTankTime"]).eventTankTime)
        d["has_n"] = d.timestamp.isin(seen)
        out.append(d)
    a = pd.concat(out, ignore_index=True)
    a["fp_in"] = (a.fprompt > FP_LO) & (a.fprompt < FP_HI)
    a["veto_ok"] = ~a.second_pulse
    # where a Stage-2 pass actually ran, so has_n is meaningful
    a["processed"] = (
        ((a.IC_adjusted > IC_LO) & (a.IC_adjusted < IC_HI) & a.veto_ok)
        | ((a.IC_adjusted > 500) & (a.IC_adjusted < 2000) & a.fp_in & a.veto_ok))
    return a


def describe(a, mask, label):
    s = a[mask]
    pr = s[s.processed]
    p = pr.has_n.mean() if len(pr) else np.nan
    e = np.sqrt(p * (1 - p) / len(pr)) if len(pr) else np.nan
    return dict(label=label, N=len(s), cover=len(pr) / max(len(s), 1),
                P=p, P_err=e, n_neutron=int(pr.has_n.sum()))


def row(r, total):
    return (f"  {r['label']:34s} N={r['N']:7d} ({r['N']/total:5.1%})  "
            f"processed={r['cover']:6.1%}  P={r['P']:.4f}+-{r['P_err']:.4f}  "
            f"neutrons={r['n_neutron']:7d}")


def main():
    a = load()
    T = len(a)
    A = (a.IC_adjusted > IC_LO) & (a.IC_adjusted < IC_HI) & a.veto_ok
    nA = int(A.sum())
    print(f"total waveforms: {T}\n")

    print("=" * 96)
    print("W scan for selection B = fprompt band AND width > W AND veto  (no IC window)")
    print("=" * 96)
    print(f"  {'selection':34s} {'N':>9s}          {'coverage':>8s}  "
          f"{'P(neutron|processed)':>22s}  {'neutrons':>8s}")
    print(row(describe(a, A, f"A: IC {IC_LO}-{IC_HI} (ours)"), T))
    print("  " + "-" * 92)
    best, bestgap = None, None
    for W in (0, 100, 200, 300, 350, 400, 425, 450, 475, 500, 550, 600):
        m = a.fp_in & (a.width_over_thresh > W) & a.veto_ok
        r = describe(a, m, f"B: fp band, width>{W}")
        print(row(r, T))
        gap = abs(int(m.sum()) - nA)
        if bestgap is None or gap < bestgap:
            best, bestgap, bestmask = W, gap, m.copy()

    print(f"\n  closest to A's acceptance: W={best}  "
          f"(N={int(bestmask.sum())} vs A's {nA})")

    print("\n" + "=" * 96)
    print(f"HEAD-TO-HEAD at matched acceptance:  A  vs  B(width>{best})")
    print("=" * 96)
    B = bestmask
    for m, lab in ((A, "A: IC window (ours)"), (B, f"B: fprompt+width (reference structure)")):
        print(row(describe(a, m, lab), T))

    print("\n  where they disagree:")
    for m, lab in (((A & B), "kept by BOTH"),
                   ((A & ~B), "A only  (IC keeps, reference drops)"),
                   ((B & ~A), "B only  (reference keeps, IC drops)")):
        print(row(describe(a, m, lab), T))

    bonly = a[B & ~A]
    blind = bonly[~bonly.processed]
    print(f"\n  B-only is {len(bonly)} waveforms; {len(blind)} of them "
          f"({len(blind)/max(len(bonly),1):.1%}) sit outside 500-2000 and were")
    print("  never Stage-2 processed, so their neutron content is UNKNOWN, not zero.")
    if len(blind):
        print(f"    their IC range: {blind.IC_adjusted.min():.0f} - "
              f"{blind.IC_adjusted.max():.0f}, median {blind.IC_adjusted.median():.0f}")
        lo = int((blind.IC_adjusted <= 500).sum()); hi = int((blind.IC_adjusted >= 2000).sum())
        print(f"    {lo} below IC 500, {hi} above IC 2000")

    print("\n" + "=" * 96)
    print("Does the width gate add anything ON TOP of the IC window?")
    print("=" * 96)
    for W in (0, 300, 400, 450, 500):
        m = A & a.fp_in & (a.width_over_thresh > W)
        print(row(describe(a, m, f"A AND fp band AND width>{W}"), T))
    print("\n  (first row = our shipped ic+fprompt mode; the rest add the reference's")
    print("   width gate on top, to see whether it buys purity the IC window missed)")


if __name__ == "__main__":
    sys.exit(main())
