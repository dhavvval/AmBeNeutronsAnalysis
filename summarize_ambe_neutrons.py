"""
summarize_ambe_neutrons.py

Per-run AmBe neutron summary from the scored all-runs parquet, using an
MC-anchored score threshold (per-model threshold set to a fixed MC signal
efficiency, then applied to data — comparable across models and physically
anchored, unlike a flat cut).

Outputs:
  <out>/ambe_all__neutron_rate_by_run.csv     per-run table:
        run, port, x_cm,y_cm,z_cm, n_events, n_clusters, n_neutron_clusters,
        n_events_with_neutron, frac_events_with_neutron, mean_mult, single_neutron_frac
  <out>/ambe_all__neutron_summary.pdf
        p1 neutron-event fraction vs run (colored by port)
        p2 mean neutron multiplicity vs |source position|  (position-dependence control)
        p3 stacked multiplicity distribution (all runs combined)
        p4 score distribution (all runs)

Usage:
  python summarize_ambe_neutrons.py \
      --scored <out>/ambe_all__data_features__scored.parquet \
      --model  <mc>/cc_neutrino__mva_frozen.pkl \
      --mc-scores <mc>/cc_neutrino__mva_scores.parquet \
      --score-col rf_score --sig-eff 0.80 --out <out>
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.data.processor import AmBeNeutronProcessing
SRC = AmBeNeutronProcessing().source_positions
EVENT_KEYS = ["run", "event_tank_time"]
NEUT = [1, 2, 3, 4]


def mc_threshold(mc_scores_path: Path, score_col: str, sig_eff: float) -> float:
    """Threshold on score that keeps `sig_eff` of MC test-set signal."""
    mc = pd.read_parquet(mc_scores_path)
    y = mc["dominant_class"].isin(NEUT).to_numpy().astype(int)
    test = mc["in_test"].to_numpy().astype(bool) if "in_test" in mc else np.ones(len(mc), bool)
    sig = mc[score_col].to_numpy()[test & (y == 1)]
    return float(np.quantile(sig, 1 - sig_eff))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--mc-scores", required=True, help="MC scored parquet (for MC-anchored threshold)")
    p.add_argument("--score-col", default="rf_score")
    p.add_argument("--sig-eff", type=float, default=0.80)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    thr = mc_threshold(Path(args.mc_scores), args.score_col, args.sig_eff)
    print(f"[summary] {args.score_col} threshold for {args.sig_eff:.0%} MC signal eff = {thr:.3f}")

    df = pd.read_parquet(args.scored)
    df["is_neutron"] = df[args.score_col] >= thr
    print(f"[summary] {len(df)} clusters over {df['run'].nunique()} runs")

    rows = []
    for run, g in df.groupby("run"):
        pos = SRC.get(int(run), (np.nan, np.nan, np.nan))
        nev = g.groupby(EVENT_KEYS).ngroups
        sel = g[g["is_neutron"]]
        mult = sel.groupby(EVENT_KEYS).size()
        nev_n = len(mult)
        rows.append({
            "run": int(run), "port": g["port"].iloc[0],
            "x_cm": pos[0], "y_cm": pos[1], "z_cm": pos[2],
            "n_events": nev, "n_clusters": len(g),
            "n_neutron_clusters": int(g["is_neutron"].sum()),
            "n_events_with_neutron": nev_n,
            "frac_events_with_neutron": nev_n / max(nev, 1),
            "mean_mult": float(mult.mean()) if nev_n else 0.0,
            "single_neutron_frac": float((mult == 1).sum() / max(nev_n, 1)),
        })
    tab = pd.DataFrame(rows).sort_values("run")
    csv = out / "ambe_all__neutron_rate_by_run.csv"
    tab.to_csv(csv, index=False)
    print(f"[summary] wrote {csv}")
    print(tab.to_string(index=False))

    with PdfPages(out / "ambe_all__neutron_summary.pdf") as pdf:
        # p1: neutron-event fraction vs run
        fig, ax = plt.subplots(figsize=(9, 5))
        ports = tab["port"].unique()
        cmap = plt.cm.tab10(np.linspace(0, 1, len(ports)))
        for c, port in zip(cmap, ports):
            s = tab[tab["port"] == port]
            ax.scatter(s["run"], s["frac_events_with_neutron"], color=c, label=port, s=40)
        ax.set_xlabel("Run number"); ax.set_ylabel("Fraction of events with ≥1 neutron")
        ax.set_title("AmBe Neutron-Event Fraction by Run", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        fig.text(0.5, 0.005, f"OPTICS + MVA selection · {args.score_col} at {args.sig_eff:.0%} MC signal efficiency (cut = {thr:.2f})",
                 ha="center", fontsize=8, color="0.4")
        pdf.savefig(fig); plt.close(fig)

        # p2: mean multiplicity vs source distance from tank center (position-dependence control)
        fig, ax = plt.subplots(figsize=(8, 5))
        rcm = np.sqrt(tab["x_cm"]**2 + tab["y_cm"]**2 + tab["z_cm"]**2)
        sc = ax.scatter(rcm, tab["mean_mult"], c=tab["frac_events_with_neutron"],
                        cmap="viridis", s=50)
        fig.colorbar(sc, ax=ax, label="frac events w/ neutron")
        for _, r in tab.iterrows():
            ax.annotate(int(r["run"]), (np.sqrt(r["x_cm"]**2+r["y_cm"]**2+r["z_cm"]**2), r["mean_mult"]),
                        fontsize=6, alpha=0.6)
        ax.set_xlabel("Source distance from tank center (cm)")
        ax.set_ylabel("Mean neutron multiplicity per event")
        ax.set_title("Neutron Yield vs Source Position", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        fig.text(0.5, 0.005, "Physics control: yield should track source location · point labels = run number",
                 ha="center", fontsize=8, color="0.4")
        pdf.savefig(fig); plt.close(fig)

        # p3: combined multiplicity distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        allmult = df[df["is_neutron"]].groupby(EVENT_KEYS).size()
        centers = np.arange(1, 9)
        h = np.array([(allmult == n).sum() for n in centers], float)
        ax.bar(centers, h, color="#9be8a0", edgecolor="darkgreen")
        ax.set_yscale("log"); ax.set_xticks(centers)
        ax.set_xlabel("Neutron multiplicity per event"); ax.set_ylabel("Counts")
        ax.set_title("AmBe Neutron Multiplicity — All Runs", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.005, f"OPTICS + MVA selection · {args.score_col} at {args.sig_eff:.0%} MC signal efficiency (cut = {thr:.2f})",
                 ha="center", fontsize=8, color="0.4")
        pdf.savefig(fig); plt.close(fig)

        # p4: score distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        for col in [c for c in ["rf_score", "gbt_score", "xgb_score"] if c in df]:
            ax.hist(df[col], bins=50, range=(0, 1), histtype="step", linewidth=1.4, label=col)
        ax.axvline(thr, color="red", ls="--", label=f"{args.score_col} cut {thr:.2f}")
        ax.set_xlabel("MVA score"); ax.set_ylabel("Clusters"); ax.legend(fontsize=8)
        ax.set_title("MVA Score Distribution — All Runs", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.005, f"Frozen MC-trained model applied to data · dashed line = {args.score_col} cut at {args.sig_eff:.0%} MC signal efficiency",
                 ha="center", fontsize=8, color="0.4")
        pdf.savefig(fig); plt.close(fig)
    print(f"[summary] wrote {out / 'ambe_all__neutron_summary.pdf'}")


if __name__ == "__main__":
    main()
