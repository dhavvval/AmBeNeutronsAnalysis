"""
analyze_optics_data_rate.py

Apply OPTICS clustering (Stage 1 pre-selection) to real ANNIE off-beam data.

Accepts one ROOT file OR a directory/glob of R{run}_extracted_off_beam_data.ntuple.root
files.  All events from all runs are pooled together; no per-run breakdown.

OPTICS is run per event on the full combined hit set (all CF cluster rows for
that event merged — CF cluster boundaries discarded).  Parameters mirror the
mc_lucho_full features config exactly: ms=8, xi=0.10, t_unit=25 ns, no prefilter.

Outputs:
    optics_data_rate.pdf          — 4-panel comparison plot (CF vs OPTICS)
    optics_data_stats.csv         — per-event table with run, n_cf, n_optics, n_hits
    optics_summary.txt            — headline statistics for all runs combined
    background_optics_hits.parquet — per-hit table for every OPTICS cluster that
                                    passed pre-selection, columns:
                                      run, event_number, cluster_id, is_background,
                                      x, y, z, t, pe, pmtID
                                    Feed cluster slices to compute_cluster_features()
                                    from ambe.mc.cluster_features to get physics
                                    features without duplicating that logic here.

Usage:
    # single file
    python analyze_optics_data_rate.py /path/to/R4432_extracted_off_beam_data.ntuple.root

    # all runs in a directory
    python analyze_optics_data_rate.py /Users/dajana/Documents/BeamCluster/

    # explicit glob
    python analyze_optics_data_rate.py "/Users/dajana/Documents/BeamCluster/R*_extracted_off_beam_data.ntuple.root"
"""

import sys
import re
import argparse
import glob
from pathlib import Path
from collections import Counter, defaultdict

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.mc.optics import run_optics_on_event

# ── OPTICS hyperparameters — mirrors mc_lucho_full features config ────────────
MIN_SAMPLES = 8
XI          = 0.10
T_UNIT_NS   = 25.0

_RUN_RE = re.compile(r"R(\d+)_extracted_off_beam_data\.ntuple\.root")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?",
                   default="/pnfs/annie/persistent/users/doran/datasets/NCQE_BEAMCLUSTER_DATA/EXTRACTED_FULL_OFFBEAM_DATA/",
                   help="ROOT file, directory, or glob pattern")
    p.add_argument("--output-dir", default=".",
                   help="Directory for outputs (default: current dir)")
    return p.parse_args()


def resolve_files(input_path: str) -> list[Path]:
    """Return sorted list of ROOT files matching the run pattern."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("R*_extracted_off_beam_data.ntuple.root"))
    else:
        files = sorted(Path(f) for f in glob.glob(input_path))
    if not files:
        raise FileNotFoundError(f"No matching ROOT files found for: {input_path}")
    return files


def run_number_from_path(path: Path) -> int:
    m = _RUN_RE.search(path.name)
    return int(m.group(1)) if m else -1


# ── Data loading ──────────────────────────────────────────────────────────────

def load_file(root_file: Path) -> list[dict]:
    """
    Load one ROOT file.  Returns list of event dicts:
      run, event_number, n_cf_clusters, df_hits (x,y,z,t,pe,pmtID)
    CF cluster boundaries are discarded — all hits for the same event_number
    are merged into one DataFrame, exactly as the MC pipeline sees df_ev.
    """
    run = run_number_from_path(root_file)
    f   = uproot.open(str(root_file))
    tree = f["data"]
    data = tree.arrays(
        ["event_number", "number_of_clusters",
         "hitX", "hitY", "hitZ", "hitT", "hitPE", "hitID"],
        library="ak"
    )

    ev_nums = ak.to_numpy(data["event_number"])

    hit_chunks  = defaultdict(list)
    cf_count_ev = {}
    for i, ev in enumerate(ev_nums):
        cf_count_ev[ev] = int(data["number_of_clusters"][i])
        chunk = pd.DataFrame({
            "x":     ak.to_numpy(data["hitX"][i]).astype(float),
            "y":     ak.to_numpy(data["hitY"][i]).astype(float),
            "z":     ak.to_numpy(data["hitZ"][i]).astype(float),
            "t":     ak.to_numpy(data["hitT"][i]).astype(float),
            "pe":    ak.to_numpy(data["hitPE"][i]).astype(float),
            "pmtID": ak.to_numpy(data["hitID"][i]).astype(int),
        })
        hit_chunks[ev].append(chunk)

    events = []
    for ev, chunks in hit_chunks.items():
        df_ev = pd.concat(chunks, ignore_index=True)
        events.append({
            "run":          run,
            "event_number": ev,
            "n_cf":         cf_count_ev[ev],
            "df":           df_ev,
        })
    return events


# ── OPTICS ────────────────────────────────────────────────────────────────────

def run_optics_all(events: list[dict]) -> tuple[list[dict], list[pd.DataFrame]]:
    """
    Run OPTICS on every event.

    Returns
    -------
    results    : per-event summary dicts (for rate comparison plots/CSV)
    hit_frames : list of DataFrames, one per OPTICS cluster, each carrying
                 run, event_number, cluster_id, is_background plus the hit
                 columns (x, y, z, t, pe, pmtID).  Concatenate and pass
                 cluster slices to compute_cluster_features() for features.
    """
    results    = []
    hit_frames = []

    for ev in events:
        df    = ev["df"]
        nhits = len(df)
        base  = {"run": ev["run"], "event_number": ev["event_number"],
                 "n_cf": ev["n_cf"], "n_hits_total": nhits}

        labels   = run_optics_on_event(df, min_samples=MIN_SAMPLES, xi=XI,
                                       t_unit_ns=T_UNIT_NS, source_pos_m=None)
        cl_ids   = [l for l in np.unique(labels) if l >= 0]
        n_optics = len(cl_ids)
        nhits_cl = [int((labels == c).sum()) for c in cl_ids]

        results.append({**base, "n_optics": n_optics,
                        "n_hits_per_cluster": nhits_cl})

        for cid in cl_ids:
            df_cl = df[labels == cid].copy()
            df_cl["run"]          = ev["run"]
            df_cl["event_number"] = ev["event_number"]
            df_cl["cluster_id"]   = int(cid)
            df_cl["is_background"] = 1
            hit_frames.append(df_cl)

    return results, hit_frames


# ── Summary + plots ───────────────────────────────────────────────────────────

def summarise(results: list[dict], run_list: list[int]) -> dict:
    cf_arr  = np.array([r["n_cf"]     for r in results])
    opt_arr = np.array([r["n_optics"] for r in results])
    all_nhits = [h for r in results for h in r["n_hits_per_cluster"]]

    agree    = int((cf_arr == opt_arr).sum())
    opt_more = int((opt_arr > cf_arr).sum())
    opt_less = int((opt_arr < cf_arr).sum())
    n = len(results)

    stats = {
        "runs":              sorted(run_list),
        "n_events_total":    n,
        "cf_mean":           float(cf_arr.mean()),
        "cf_median":         float(np.median(cf_arr)),
        "optics_mean":       float(opt_arr.mean()),
        "optics_median":     float(np.median(opt_arr)),
        "n_agree":           agree,
        "pct_agree":         100 * agree / n,
        "n_optics_more":     opt_more,
        "pct_optics_more":   100 * opt_more / n,
        "n_optics_less":     opt_less,
        "pct_optics_less":   100 * opt_less / n,
        "optics_clusters_total": int(opt_arr.sum()),
        "nhits_per_cluster_median": float(np.median(all_nhits)) if all_nhits else np.nan,
        "nhits_per_cluster_mean":   float(np.mean(all_nhits))   if all_nhits else np.nan,
    }
    return stats, cf_arr, opt_arr, all_nhits


def print_summary(stats: dict, cf_arr, opt_arr):
    print("=" * 60)
    print("OPTICS vs ClusterFinder — all runs combined")
    print(f"  OPTICS params: ms={MIN_SAMPLES}, xi={XI}, t_unit={T_UNIT_NS} ns")
    print("=" * 60)
    print(f"Runs processed      : {stats['runs']}")
    print(f"Events total        : {stats['n_events_total']}")
    print()
    print(f"CF     mean/median  : {stats['cf_mean']:.2f} / {stats['cf_median']:.1f}")
    print(f"OPTICS mean/median  : {stats['optics_mean']:.2f} / {stats['optics_median']:.1f}")
    print()
    n = stats["n_events_total"]
    print(f"Exact agreement     : {stats['n_agree']}/{n}  ({stats['pct_agree']:.1f}%)")
    print(f"OPTICS > CF         : {stats['n_optics_more']}/{n}  ({stats['pct_optics_more']:.1f}%)")
    print(f"OPTICS < CF         : {stats['n_optics_less']}/{n}  ({stats['pct_optics_less']:.1f}%)")
    print()
    print(f"Total OPTICS clusters found : {stats['optics_clusters_total']}")
    print(f"Hits/cluster median/mean    : "
          f"{stats['nhits_per_cluster_median']:.1f} / {stats['nhits_per_cluster_mean']:.1f}")
    print()
    print(f"{'n_cf':>6}  {'events':>7}  {'mean n_optics':>14}  {'agree%':>8}")
    for nc in sorted(set(cf_arr))[:15]:
        mask = cf_arr == nc
        mo   = opt_arr[mask]
        ag   = (mo == nc).mean() * 100
        print(f"  {nc:4d}   {mask.sum():6d}   {mo.mean():13.2f}   {ag:7.1f}%")


def make_plots(stats: dict, cf_arr, opt_arr, all_nhits, output_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    mx   = int(min(max(cf_arr.max(), opt_arr.max()), 20))
    bins = np.arange(-0.5, mx + 1.5)
    axes[0].hist(cf_arr,  bins=bins, alpha=0.65, color="#0077BB",
                 label=f"ClusterFinder  (mean={cf_arr.mean():.2f})")
    axes[0].hist(opt_arr, bins=bins, alpha=0.65, color="#EE7733",
                 label=f"OPTICS ms={MIN_SAMPLES}  (mean={opt_arr.mean():.2f})")
    axes[0].set_xlabel("Clusters per event")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Cluster count distribution")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-0.5, mx + 0.5)

    rng    = np.random.default_rng(0)
    jit    = rng.uniform(-0.2, 0.2, len(cf_arr))
    lim    = int(min(max(cf_arr.max(), opt_arr.max()), 20))
    axes[1].scatter(cf_arr + jit, opt_arr + jit, alpha=0.2, s=6, color="steelblue")
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="CF = OPTICS")
    axes[1].set_xlabel("CF clusters / event")
    axes[1].set_ylabel("OPTICS clusters / event")
    axes[1].set_title(f"Event-level scatter\n(agree={stats['pct_agree']:.1f}%)")
    axes[1].set_xlim(-0.5, lim + 0.5)
    axes[1].set_ylim(-0.5, lim + 0.5)
    axes[1].legend(fontsize=8)

    diff  = opt_arr.astype(int) - cf_arr.astype(int)
    dcnt  = Counter(diff)
    dk    = sorted(dcnt)
    dk_cl = ["#EE7733" if k > 0 else "#0077BB" if k < 0 else "#AAAAAA" for k in dk]
    axes[2].bar(dk, [dcnt[k] for k in dk], color=dk_cl)
    axes[2].axvline(0, color="k", lw=1, ls="--")
    axes[2].set_xlabel("OPTICS − CF  (clusters/event)")
    axes[2].set_ylabel("Events")
    axes[2].set_title(f"Difference\nmean={diff.mean():.2f}  median={np.median(diff):.1f}")
    dk_range = max(abs(dk[0]), abs(dk[-1]))
    axes[2].set_xlim(-min(dk_range, 15) - 0.5, min(dk_range, 15) + 0.5)

    if all_nhits:
        axes[3].hist(all_nhits, bins=40, color="darkorange", edgecolor="white")
        axes[3].axvline(np.median(all_nhits), color="k", ls="--",
                        label=f"median={np.median(all_nhits):.0f}")
        axes[3].set_xlabel("Hits per OPTICS cluster")
        axes[3].set_ylabel("Clusters")
        axes[3].set_title("n_hits per cluster")
        axes[3].legend(fontsize=8)

    runs_str = ", ".join(f"R{r}" for r in stats["runs"][:6])
    if len(stats["runs"]) > 6:
        runs_str += f" … ({len(stats['runs'])} runs)"
    fig.suptitle(
        f"OPTICS (ms={MIN_SAMPLES}, xi={XI}, t={T_UNIT_NS} ns) vs ClusterFinder\n"
        f"{runs_str}  |  {stats['n_events_total']} events",
        fontsize=10)
    fig.tight_layout()

    out = output_dir / "optics_data_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Plot  → {out}")


def save_outputs(results: list[dict], stats: dict,
                 hit_frames: list[pd.DataFrame], output_dir: Path):
    # Per-event CSV
    rows = []
    for r in results:
        rows.append({
            "run":           r["run"],
            "event_number":  r["event_number"],
            "n_cf":          r["n_cf"],
            "n_optics":      r["n_optics"],
            "n_hits_total":  r["n_hits_total"],
            "n_hits_cluster_median": (np.median(r["n_hits_per_cluster"])
                                      if r["n_hits_per_cluster"] else np.nan),
        })
    csv_out = output_dir / "optics_data_stats.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"Stats → {csv_out}")

    # Headline summary text
    txt_out = output_dir / "optics_summary.txt"
    lines = [
        f"OPTICS params: min_samples={MIN_SAMPLES}, xi={XI}, t_unit_ns={T_UNIT_NS}",
        f"Runs: {stats['runs']}",
        f"Events processed: {stats['n_events_total']}",
        f"CF  mean={stats['cf_mean']:.3f}  median={stats['cf_median']:.1f}",
        f"OPTICS mean={stats['optics_mean']:.3f}  median={stats['optics_median']:.1f}",
        f"Agreement: {stats['pct_agree']:.1f}%  OPTICS>CF: {stats['pct_optics_more']:.1f}%  OPTICS<CF: {stats['pct_optics_less']:.1f}%",
        f"Total OPTICS clusters: {stats['optics_clusters_total']}",
        f"Hits/cluster median={stats['nhits_per_cluster_median']:.1f}  mean={stats['nhits_per_cluster_mean']:.1f}",
    ]
    txt_out.write_text("\n".join(lines) + "\n")
    print(f"Summary → {txt_out}")

    # Per-cluster hit parquet — feed to compute_cluster_features() for features
    if hit_frames:
        hits_out = output_dir / "background_optics_hits.parquet"
        col_order = ["run", "event_number", "cluster_id", "is_background",
                     "x", "y", "z", "t", "pe", "pmtID"]
        pd.concat(hit_frames, ignore_index=True)[col_order].to_parquet(
            hits_out, index=False)
        n_clusters = sum(1 for r in results for _ in r["n_hits_per_cluster"])
        print(f"Hits   → {hits_out}  ({n_clusters} clusters, "
              f"{sum(len(f) for f in hit_frames)} hits)")


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = resolve_files(args.input)
    print(f"Found {len(files)} ROOT file(s):")
    for f in files:
        print(f"  {f.name}")

    all_events = []
    run_list   = []
    for f in files:
        run = run_number_from_path(f)
        run_list.append(run)
        evs = load_file(f)
        print(f"  R{run}: {len(evs)} events, "
              f"{sum(len(e['df']) for e in evs)} total hits")
        all_events.extend(evs)

    print(f"\nTotal across all runs: {len(all_events)} events")

    print(f"\nRunning OPTICS (ms={MIN_SAMPLES}, xi={XI}, t_unit={T_UNIT_NS} ns) ...")
    results, hit_frames = run_optics_all(all_events)

    stats, cf_arr, opt_arr, all_nhits = summarise(results, run_list)
    print_summary(stats, cf_arr, opt_arr)
    make_plots(stats, cf_arr, opt_arr, all_nhits, output_dir)
    save_outputs(results, stats, hit_frames, output_dir)


if __name__ == "__main__":
    main()
