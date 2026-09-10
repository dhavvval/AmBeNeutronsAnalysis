#!/usr/bin/env python3
"""
Selection-1-vs-Selection-2 benchmark — the clean 2x2 the earlier comparison
conflated.  Two axes, evaluated independently on the SAME MC truth:

    clustering axis :  OPTICS        vs  ClusterFinder
    discriminant axis: legacy box-cut (PE<80, CB<0.45, nHits>9)  vs  frozen MVA

The earlier report applied the legacy cut to OPTICS clusters and CALLED it the
"ClusterFinder" baseline — so it measured MVA-vs-cut on identical OPTICS inputs,
not Selection-1 (OPTICS+MVA) vs Selection-2 (ClusterFinder+cuts).  This script
fills the missing cell: it scores the ClusterFinder clusters with the SAME
frozen MVA artifact (reusing mva_analysis.score_data, no divergent code), then
reports all four (clustering x discriminant) combinations.

Truth definition: cluster-level, dominant_class in {1,2,3,4} == neutron-dominated,
identical to how mva_analysis.prepare_data defines MVA signal (mva_analysis.py:188).
This is the consistent label for BOTH methods; we do NOT mix in is_truth_neutron
(the stricter trackID match) on one side only — that was a separate trap.

Run (in the `myboy` venv):
    python benchmark_selection_2x2.py \
        --features  .../cc_neutrino__cluster_features.parquet \
        --model     .../cc_neutrino__mva_frozen.pkl \
        --out-dir   .../cc_neutrino/plots \
        --score-cut 0.643
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

# Two box-cut definitions, selected by --box. They are NOT interchangeable and the
# difference is the reason --box has no default:
#
#   legacy  PE<80, CB<0.45, nHits>9. The MC preselection, kept byte-identical to
#           analyze_optics_beamcluster_data.py:87-89 and
#           plot_ambe_neutron_multiplicity.py:50-52. Do not retune it here -- those
#           two files define the sample this script's MC numbers were measured on.
#
#   ambe    0<PE<=100, 0<CB<0.45, nHits>=5. The box the AmBe DATA analysis actually
#           applies (src/ambe/data/selection.py BoxSelection, and the deck-2 config).
#           This is the one to use when the question is "what efficiency and purity
#           does the selection I am running on data have?", which is what deck 1 has
#           to answer. The Stage-4 table in PRESENTATION_DEEP_DIVE_MC_TO_OUTPUTS.md
#           was produced with `legacy` and therefore does NOT describe the deck-2
#           selection.
#
# Note the cluster-TIME cut of the AmBe box has no analogue here: these are MC
# cluster features with no delayed-window time column, and the MC sample is already
# restricted to the delayed window upstream. So `ambe` is the PE/CB/hits box only.
BOXES = {
    "legacy": dict(pe_max=80.0, pe_min=None, cb_max=0.45, cb_min=None,
                   hits_min=9, hits_strict=True,
                   note="PE<80, CB<0.45, nHits>9"),
    "ambe":   dict(pe_max=100.0, pe_min=0.0, cb_max=0.45, cb_min=0.0,
                   hits_min=5, hits_strict=False,
                   note="0<PE<=100, 0<CB<0.45, nHits>=5"),
}
NEUTRON_CLASSES = (1, 2, 3, 4)
COLORS = {"optics": "#0077BB", "clusterfinder": "#EE7733"}


def box_cut(sub: pd.DataFrame, box: dict) -> pd.Series:
    """Box-cut mask. NaN CB is kept, matching passes_preselection."""
    cb = sub["charge_bal_legacy"]
    pe = sub["pe_total"]
    # legacy uses strict < on PE and > on hits; the AmBe box is inclusive on both
    # bounds (`pe_min < pe <= pe_max`, `hits >= hits_min`). Getting this wrong moves
    # the efficiency by a fraction of a percent and the fake count by more.
    m = (pe < box["pe_max"]) if box["pe_min"] is None else (pe <= box["pe_max"])
    if box["pe_min"] is not None:
        m &= pe > box["pe_min"]
    m &= (cb < box["cb_max"]) | cb.isna()
    if box["cb_min"] is not None:
        m &= (cb > box["cb_min"]) | cb.isna()
    m &= (sub["n_hits"] > box["hits_min"]) if box["hits_strict"] \
        else (sub["n_hits"] >= box["hits_min"])
    return m


def metrics(sub: pd.DataFrame, sel: pd.Series, truth: pd.Series) -> dict:
    """Efficiency (recall), purity, and counts for a boolean selection mask."""
    sel = sel.to_numpy(bool)
    truth = truth.to_numpy(bool)
    tp = int((sel & truth).sum())
    pred = int(sel.sum())
    tot = int(truth.sum())
    return {
        "eff": tp / tot if tot else np.nan,
        "purity": tp / pred if pred else np.nan,
        "tp": tp, "predicted": pred, "fakes": pred - tp, "total_neutron": tot,
    }


def mva_threshold_at_eff(score: pd.Series, truth: pd.Series, sig_eff: float) -> float:
    """Score threshold giving `sig_eff` recall on this (truth) sample."""
    sig = score[truth.to_numpy(bool)]
    return float(np.quantile(sig, 1.0 - sig_eff))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-col", default="rf_score")
    p.add_argument("--score-cut", type=float, default=0.643,
                   help="frozen Stage-B cut (rf @ 80%% MC sig eff = 0.643)")
    p.add_argument("--sig-eff", type=float, default=0.80,
                   help="also report the self-consistent threshold at this eff")
    # No default: `legacy` and `ambe` are different selections and answer different
    # questions (see BOXES). Silently picking one is how the Stage-4 purity table
    # came to describe a box the data analysis does not apply.
    p.add_argument("--box", required=True, choices=sorted(BOXES),
                   help="which box-cut to benchmark against the MVA")
    args = p.parse_args()
    box = BOXES[args.box]
    print(f"[bench] box-cut = {args.box}: {box['note']}")

    set_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    df["is_neutron"] = df["dominant_class"].isin(NEUTRON_CLASSES)

    # --- Fill the missing 2x2 cell: score CF clusters with the SAME frozen MVA ---
    # Reuse mva_analysis.score_data so the feature ordering / median imputation is
    # byte-for-byte identical to how the OPTICS clusters were scored for data.
    from mva_analysis import score_data
    base = Path(args.features).with_name(
        Path(args.features).stem + "__allmethods_scored.parquet")
    final = base.with_name(base.stem + "__scored.parquet")
    if final.exists():
        print(f"[bench] reusing scored parquet {final.name}")
    else:
        df.to_parquet(base, index=False)
        final = score_data(Path(args.model), base)
    df = pd.read_parquet(final)
    df["is_neutron"] = df["dominant_class"].isin(NEUTRON_CLASSES)

    sc = args.score_col
    # Two truth definitions, reported side by side so nothing is hidden:
    #   composition : dominant_class in {1,2,3,4}  (the MVA's own signal def)
    #   strict      : is_truth_neutron==1          (stricter trackID-purity match,
    #                 what the earlier report used — gives lower purities)
    rows = []
    for truth_def, truth_col in [("composition", "is_neutron"),
                                 ("strict", "is_truth_neutron")]:
      for method in ("optics", "clusterfinder"):
        sub = df[df["method"] == method].copy()
        truth = sub[truth_col].astype(bool)
        # self-consistent threshold on THIS method at the target efficiency
        thr_self = mva_threshold_at_eff(sub[sc], truth, args.sig_eff)
        # discriminant KEY is shared across methods (so the grouped bar plot can
        # pair OPTICS vs CF); the per-method threshold value lives in `note`.
        for disc, sel, note in [
            (f"{args.box} box-cut", box_cut(sub, box), box["note"]),
            (f"MVA frozen @{args.score_cut:.3f}", sub[sc] >= args.score_cut, "frozen rf cut"),
            (f"MVA self {int(args.sig_eff*100)}% eff", sub[sc] >= thr_self,
             f"rf>={thr_self:.3f}"),
        ]:
            m = metrics(sub, sel, truth)
            rows.append({"truth_def": truth_def, "clustering": method,
                         "discriminant": disc, "note": note, **m})

    res = pd.DataFrame(rows)
    # Tag the outputs with the box and the score column: running --box legacy and
    # --box ambe into one filename would leave whichever ran last, silently.
    stem = f"selection_2x2_benchmark__{args.box}__{args.score_col.split('_')[0]}"
    csv_path = out_dir / f"{stem}.csv"
    res.to_csv(csv_path, index=False)

    # --- Console table ---
    for truth_def, blurb in [("composition", "dominant_class in {1,2,3,4} (MVA signal def)"),
                             ("strict", "is_truth_neutron==1 (strict trackID match)")]:
        print("\n" + "=" * 90)
        print("SELECTION 1 (OPTICS) vs SELECTION 2 (ClusterFinder) — 2x2 benchmark")
        print(f"MC truth = {blurb}")
        print("=" * 90)
        hdr = f"{'clustering':<14}{'discriminant':<24}{'eff':>7}{'purity':>9}{'fakes':>8}{'tp/pred':>12}{'total_n':>9}"
        print(hdr)
        print("-" * len(hdr))
        for _, r in res[res.truth_def == truth_def].iterrows():
            print(f"{r.clustering:<14}{r.discriminant:<24}{r.eff:>7.3f}{r.purity:>9.3f}"
                  f"{r.fakes:>8d}{f'{r.tp}/{r.predicted}':>12}{r.total_neutron:>9d}")
    print("=" * 90)
    print(f"[bench] wrote table -> {csv_path}")

    # --- Plot: grouped eff & purity bars, OPTICS vs CF, per discriminant ---
    _plot_2x2(res, out_dir, args, stem, box)


def _plot_2x2(res: pd.DataFrame, out_dir: Path, args, stem: str, box: dict) -> None:
    # primary plot uses the composition truth def (the MVA's own signal def)
    res = res[res.truth_def == "composition"]
    # one row of discriminants, OPTICS vs CF side by side, eff & purity panels
    discs = list(res["discriminant"].unique())
    x = np.arange(len(discs))
    w = 0.38
    # One PDF+PNG PAIR per metric, plus the combined multi-page PDF. The pairs exist
    # because collect_presentation_plots.py resolves a figure as <stem>.pdf AND
    # <stem>.png and fails if either is missing -- a single multi-page PDF has no
    # matching PNG, so it can never be picked up as a deck figure.
    pdf_path = out_dir / f"{stem}.pdf"
    with PdfPages(pdf_path) as pdf:
        for metric, ylabel in [("eff", "Efficiency (recall)"), ("purity", "Purity")]:
            fig, ax = plt.subplots()
            for i, method in enumerate(("optics", "clusterfinder")):
                vals = [res[(res.clustering == method) & (res.discriminant == d)][metric].iloc[0]
                        for d in discs]
                bars = ax.bar(x + (i - 0.5) * w, vals, width=w,
                              color=COLORS[method], edgecolor="black", linewidth=0.6,
                              label="OPTICS" if method == "optics" else "ClusterFinder")
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([d.replace(" ", "\n", 1) for d in discs], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            # Title names the box literally and states the population, because a
            # purity is meaningless without knowing which denominator it used.
            ax.set_title(f"{box['note']}  vs  MVA neutron ({args.score_col})\n"
                         f"{ylabel} on the natural MC cluster population")
            ax.legend(frameon=False)
            pdf.savefig(fig, bbox_inches="tight")
            pair = out_dir / f"{stem}__{metric}"
            fig.savefig(f"{pair}.pdf", bbox_inches="tight")
            fig.savefig(f"{pair}.png", dpi=200, bbox_inches="tight")
            print(f"[bench] wrote pair  -> {pair.name}.pdf / .png")
            plt.close(fig)
    print(f"[bench] wrote plot  -> {pdf_path}")


if __name__ == "__main__":
    main()
