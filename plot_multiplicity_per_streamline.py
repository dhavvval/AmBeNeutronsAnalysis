#!/usr/bin/env python3
"""
plot_multiplicity_per_streamline.py

One standalone neutron-multiplicity histogram per selection streamline, mirroring
the per-streamline capture-time plots.  Each PDF is a SINGLE titled figure showing
tagged-neutron-clusters-per-event for exactly one (clustering x selection):

    OPTICS / CF   x   RF (rf_score>=0.547) / NN (nn_score>=0.538) / box (80/0.45/9)

Event key is ["run","event_tank_time"] (the unique per-AmBe-event id used across the
toolkit).  The legacy box is recomputed LIVE from raw features (pe_total<80 &
(charge_bal_legacy<0.45 | NaN) & n_hits>9) — NOT read from passes_stage1, which is
stale on the OPTICS data parquet (baked at the old 60/0.5/10 box).

AmBe is a single-neutron source, so each plot is dominated by the multiplicity=1 bin;
y-axis is log so the small >=2 tail stays visible.

Run (myboy venv):
    python plot_multiplicity_per_streamline.py --out-dir <staging dir>
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVT = ["run", "event_tank_time"]
ROOT = "/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/ambe_data"
STREAMS = {
    "optics": f"{ROOT}/ccinc_xi02cc_optics_mcbkg/*__scored.parquet",
    "cf":     f"{ROOT}/ccinc_xi02cc_cf_mcbkg/*__scored.parquet",
}
# (column, cut, pretty label, threshold-string) per selection
SELECTIONS = {
    "RF":  ("rf_score", 0.547, "Random Forest", "rf_score $\\geq$ 0.547"),
    "NN":  ("nn_score", 0.538, "Neural Net",    "nn_score $\\geq$ 0.538"),
    "box": (None,       None,  "legacy box cut", "PE<80, CB<0.45, nHits>9"),
}
CLUSTER_LABEL = {"optics": "OPTICS", "cf": "ClusterFinder"}
COLOR = {"RF": "#0077BB", "NN": "#33BBEE", "box": "#EE7733"}


def box_mask(df: pd.DataFrame) -> pd.Series:
    cb = df["charge_bal_legacy"]
    return (df["pe_total"] < 80) & ((cb < 0.45) | cb.isna()) & (df["n_hits"] > 9)


def select(df: pd.DataFrame, sel: str) -> pd.DataFrame:
    if sel == "box":
        return df[box_mask(df)]
    col, cut = SELECTIONS[sel][0], SELECTIONS[sel][1]
    return df[df[col] >= cut]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/slide_plots_ccinc_xi02")
    p.add_argument("--max-n", type=int, default=6, help="last explicit bin; rest folded into >=max-n")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    maxn = args.max_n

    for stream, pat in STREAMS.items():
        df = pd.concat((pd.read_parquet(f) for f in sorted(glob.glob(pat))),
                       ignore_index=True)
        n_events = df.groupby(EVT).ngroups
        for sel in ("RF", "NN", "box"):
            sub = select(df, sel)
            per_evt = sub.groupby(EVT).size()
            n_with = int((per_evt >= 1).sum())
            single = (per_evt == 1).sum() / max(n_with, 1)

            # histogram 1..maxn-1 explicit, last bin = >= maxn
            centers = np.arange(1, maxn + 1)
            h = np.array([(per_evt == n).sum() for n in centers[:-1]], dtype=float)
            h = np.append(h, (per_evt >= maxn).sum())
            labels = [str(n) for n in centers[:-1]] + [f"$\\geq${maxn}"]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(centers, h, width=0.7, color=COLOR[sel],
                   edgecolor="black", linewidth=0.6)
            for c, v in zip(centers, h):
                if v > 0:
                    ax.text(c, v * 1.15, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylim(0.7, h.max() * 3)
            ax.set_xticks(centers); ax.set_xticklabels(labels)
            ax.set_xlabel("Tagged neutron clusters per event", fontsize=11)
            ax.set_ylabel("Number of events", fontsize=11)
            pretty = SELECTIONS[sel][2]
            ax.set_title(f"AmBe neutron multiplicity — {CLUSTER_LABEL[stream]} + {pretty}",
                         fontsize=12, fontweight="bold")
            ax.text(0.97, 0.93,
                    f"{SELECTIONS[sel][3]}\n"
                    f"events with $\\geq$1: {n_with:,} / {n_events:,}\n"
                    f"single-neutron fraction: {single*100:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
            if sel == "box":
                fig.text(0.5, 0.005, "legacy-box reference (not an MVA result)",
                         ha="center", fontsize=8, color="0.45")
            ax.grid(axis="y", alpha=0.3)

            fname = out / f"{CLUSTER_LABEL[stream].upper().replace('CLUSTERFINDER','CF')}_{sel}__multiplicity_standalone.pdf"
            fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
            print(f"[mult] {fname.name:42s} single={single*100:5.1f}%  events>=1={n_with:,}")


if __name__ == "__main__":
    main()
