#!/usr/bin/env python
"""
Why the efficiency ratio fell when the IC window was widened, and which of the
two changes (fprompt cut vs IC widening) caused it.

Joins the Stage-1 feature dumps to the Stage-2 candidate CSVs on
`timestamp` == `eventTankTime` -- an exact 1:1 key -- so every accepted tag can
be labelled "did this event yield >=1 neutron cluster?". That gives
P(neutron | tag) as a function of IC_adjusted, which is the curve the IC window
is really sliding along, and it is the right tool for picking the final bounds
(REPORT §7.4 item 1).

P here keeps cosmics in the denominator, unlike the official
`unique_neutron_triggers / ambe_triggers`, so it sits ~22% low in absolute
terms. Every statement below is a ratio between bands, where that cancels.

One trap this script is written to avoid: a waveform rejected at Stage 1 was
never Stage-2 processed, so it *cannot* appear in that tag's candidate CSV.
Asking "do fprompt's rejects contain neutrons?" against the fprompt CSV
therefore returns 0 by construction, not by physics. The fprompt-rejected set
has to be scored against the **gated** CSVs, which did process it.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python -u analyze_efficiency_vs_ic.py
"""
import sys

import numpy as np
import pandas as pd

RUNS = [6046, 6056, 6060, 6061, 6062, 6165, 6166, 6186, 6187, 6188, 6189,
        6230, 6231, 6232, 6234, 6235, 6237, 6239, 6241, 6242]
FEAT = "TriggerSummary/WaveformFeatures_AmBe2.0v4_fprompt_{run}.parquet"
CAND = "EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_AmBe2.0v4_{tag}_{run}.csv"
COSMIC_FRAC = 0.22          # measured across the campaign; only used to rescale
FP_LO, FP_HI = 0.15, 0.30


def load():
    out = []
    for run in RUNS:
        d = pd.read_parquet(FEAT.format(run=run),
                            columns=["timestamp", "IC_adjusted", "fprompt",
                                     "second_pulse", "accepted"])
        for tag, col in (("gated", "n_gated"), ("fprompt", "n_fprompt")):
            ts = pd.read_csv(CAND.format(tag=tag, run=run),
                             usecols=["eventTankTime"]).eventTankTime
            d[col] = d.timestamp.isin(set(ts))
        d["run"] = run
        out.append(d)
    return pd.concat(out, ignore_index=True)


def rate(s, col):
    if not len(s):
        return np.nan, np.nan
    p = s[col].mean()
    return p, np.sqrt(p * (1 - p) / len(s))


def main():
    a = load()
    acc = a[a.accepted]                       # the fprompt2d-accepted set

    print("=" * 74)
    print("P(>=1 neutron cluster | accepted tag) vs IC_adjusted")
    print("=" * 74)
    edges = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1600, 2000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = acc[(acc.IC_adjusted > lo) & (acc.IC_adjusted < hi)]
        p, e = rate(s, "n_fprompt")
        flag = "  <- inside the old 700-1200 window" if lo >= 700 and hi <= 1200 else ""
        print(f"  {lo:5d}-{hi:5d}  N={len(s):7d}  P={p:.4f} +- {e:.4f}  "
              f"{'#' * int(p * 55)}{flag}")
    print("\n  The curve peaks at 1100-1200 and falls off BOTH sides. The old window")
    print("  sat on that peak, which is why widening it can only lower the ratio.")

    print("\n" + "=" * 74)
    print("Which change caused the drop?")
    print("=" * 74)
    base = a[(a.IC_adjusted > 700) & (a.IC_adjusted < 1200) & (~a.second_pulse)]
    inb = (base.fprompt > FP_LO) & (base.fprompt < FP_HI)
    p1, _ = rate(base, "n_gated")
    p2, _ = rate(base[inb], "n_gated")
    p3, _ = rate(acc, "n_fprompt")
    print(f"  1. baseline IC 700-1200, no fprompt     N={len(base):7d}  P={p1:.4f}")
    print(f"  2. + fprompt {FP_LO}-{FP_HI}, same IC window   N={len(base[inb]):7d}  "
          f"P={p2:.4f}  {(p2-p1)*100:+.3f} pp")
    print(f"  3. + widen IC to 500-2000               N={len(acc):7d}  "
          f"P={p3:.4f}  {(p3-p2)*100:+.3f} pp")
    print(f"\n  fprompt's own contribution : {(p2-p1)*100:+.3f} pp   (essentially zero)")
    print(f"  the IC widening's          : {(p3-p2)*100:+.3f} pp   (the whole effect)")
    print(f"  rescaled by 1/(1-{COSMIC_FRAC}) for the official definition: "
          f"{(p3-p2)/(1-COSMIC_FRAC)*100:+.2f} pp  vs -1.83 pp observed")

    print("\n" + "=" * 74)
    print("Is fprompt discarding good tags? (scored on the GATED CSVs -- see docstring)")
    print("=" * 74)
    for lab, s in (("fprompt KEEPS", base[inb]), ("fprompt REJECTS", base[~inb])):
        p, e = rate(s, "n_gated")
        print(f"  {lab:16s} N={len(s):7d}  P={p:.4f} +- {e:.4f}")
    lost_tags = (~inb).sum() / len(base)
    lost_n = base[~inb].n_gated.sum() / base.n_gated.sum()
    print(f"\n  discards {lost_tags:.2%} of baseline-accepted tags but only "
          f"{lost_n:.2%} of neutron-bearing ones")
    print("  -> mildly purifying. It is not what lowered the efficiency.")

    print("\n" + "=" * 74)
    print("Newly admitted flanks, AFTER fprompt has already cleaned them")
    print("=" * 74)
    for lo, hi, lab in ((500, 700, "500-700   (new, low)"),
                        (700, 1200, "700-1200  (old window)"),
                        (1200, 2000, "1200-2000 (new, high)")):
        s = acc[(acc.IC_adjusted > lo) & (acc.IC_adjusted < hi)]
        p, e = rate(s, "n_fprompt")
        print(f"  {lab:24s} N={len(s):7d}  P={p:.4f} +- {e:.4f}")
    print("\n  Every tag above already passed fprompt, so the residual inefficiency")
    print("  in the flanks is NOT identifiable from pulse shape -- fprompt cannot")
    print("  recover it. Low IC = weak/partial-energy tags; high IC = pile-up.")


if __name__ == "__main__":
    sys.exit(main())
