#!/usr/bin/env python3
"""Pick ONE OPTICS config that serves both the tank and world ccv3 samples.

The two samples are swept separately (they have different cc_only / FV semantics,
and a joint sweep would be ~95% tank by hit count, drowning the near-wall
population the world sample exists to provide). But cluster features must be
extracted with a UNIFORM config, so that any tank-vs-world feature difference in
the merged training set is physics rather than a clustering artifact.

So this does not pick a winner per sample. It ranks configs by how well they do
on BOTH, and reports where the two samples disagree.

Usage:
    python3 compare_sweeps_pick_uniform.py \
        ambe_output/sweep_v3_tank_truthtag/csv/sweep_v3_tank_truthtag__optics_summary.csv \
        ambe_output/sweep_v3_world_truthtag/csv/sweep_v3_world_truthtag__optics_summary.csv
"""
import sys
import pandas as pd

KEYS = ["min_samples", "xi", "t_unit_ns"]
# The frozen production point, carried as the anchor the sweep must beat to matter.
BASELINE = (8, 0.02, 25.0)


def load(path, tag):
    df = pd.read_csv(path)
    opt = df[df["method"] == "optics"].copy()
    if opt.empty:
        sys.exit(f"no method=optics rows in {path}")
    for k in KEYS:
        opt[k] = opt[k].astype(float)
    cf = df[df["method"] == "clusterfinder"]
    ref = float(cf["mean_purity"].iloc[0]) if len(cf) else float("nan")
    print(f"{tag:6s}: {len(opt)} configs, {int(opt['events'].iloc[0])} events, "
          f"ClusterFinder reference purity = {ref:.4f}")
    return opt.set_index(KEYS), ref


def main(tank_csv, world_csv):
    tank, tank_cf = load(tank_csv, "tank")
    world, world_cf = load(world_csv, "world")

    common = tank.index.intersection(world.index)
    if not len(common):
        sys.exit("the two sweeps share no configs — were the same grids used?")
    print(f"\n{len(common)} configs swept on both samples\n")

    j = pd.DataFrame(index=common)
    for col in ("mean_purity", "mean_recall", "mean_real_spurious"):
        j[f"tank_{col}"] = tank.loc[common, col]
        j[f"world_{col}"] = world.loc[common, col]

    # A uniform config has to be acceptable on BOTH samples, so score it by the
    # WORSE of the two rather than the average -- an average lets a config that is
    # excellent on tank and poor on world win on tank's weight alone.
    j["min_purity"] = j[["tank_mean_purity", "world_mean_purity"]].min(axis=1)
    j["min_recall"] = j[["tank_mean_recall", "world_mean_recall"]].min(axis=1)
    j["max_spurious"] = j[["tank_mean_real_spurious",
                           "world_mean_real_spurious"]].max(axis=1)
    j["purity_gap"] = (j["tank_mean_purity"] - j["world_mean_purity"]).abs()

    base = BASELINE if BASELINE in j.index else None
    if base:
        b = j.loc[base]
        print(f"BASELINE ms={base[0]:.0f} xi={base[1]} t={base[2]:.0f}: "
              f"tank purity {b['tank_mean_purity']:.4f} / "
              f"world purity {b['world_mean_purity']:.4f} / "
              f"min recall {b['min_recall']:.4f}")
        # Do not trade away signal to buy purity: require recall within 0.05 of
        # baseline's worse sample, the same guard the overnight sweep used.
        floor = b["min_recall"] - 0.05
    else:
        floor = j["min_recall"].max() - 0.05
        print(f"baseline {BASELINE} not in grid; recall floor from best = {floor:.4f}")

    elig = j[j["min_recall"] >= floor]
    print(f"\n{len(elig)}/{len(j)} configs keep recall within 0.05 of baseline "
          f"on BOTH samples (floor {floor:.4f})\n")
    if elig.empty:
        sys.exit("no config satisfies the recall floor on both samples")

    rank = elig.sort_values("min_purity", ascending=False)
    print("Top 8 by worst-case purity (the uniform-config criterion):")
    print(f"{'ms':>4} {'xi':>5} {'t':>4} | {'tank pur':>9} {'world pur':>10} "
          f"{'MIN pur':>8} | {'tank rec':>9} {'world rec':>10} | {'gap':>6}")
    for (ms, xi, t), r in rank.head(8).iterrows():
        mark = "  <- baseline" if (ms, xi, t) == BASELINE else ""
        print(f"{ms:4.0f} {xi:5.2f} {t:4.0f} | {r['tank_mean_purity']:9.4f} "
              f"{r['world_mean_purity']:10.4f} {r['min_purity']:8.4f} | "
              f"{r['tank_mean_recall']:9.4f} {r['world_mean_recall']:10.4f} | "
              f"{r['purity_gap']:6.4f}{mark}")

    best = rank.index[0]
    print(f"\nRECOMMENDED UNIFORM CONFIG: min_samples={best[0]:.0f} "
          f"xi={best[1]} t_unit_ns={best[2]:.0f}")
    if base and best == BASELINE:
        print("  -> this IS the frozen production point; the sweep found nothing better.")
    elif base:
        d = rank.iloc[0]["min_purity"] - j.loc[base, "min_purity"]
        print(f"  -> worst-case purity gain over baseline: {d:+.4f}")
        if abs(d) < 0.01:
            print("     (under 0.01 — treat as noise and keep the baseline)")

    # Where the two samples disagree most: a large gap means the config is not
    # really uniform in effect, even if it is uniform in parameters.
    print("\nLargest tank/world purity gaps (uniformity is worst here):")
    for (ms, xi, t), r in j.sort_values("purity_gap", ascending=False).head(4).iterrows():
        print(f"  ms={ms:.0f} xi={xi} t={t:.0f}: tank {r['tank_mean_purity']:.4f} "
              f"vs world {r['world_mean_purity']:.4f}  (gap {r['purity_gap']:.4f})")

    out = "sweep_uniform_comparison.csv"
    j.sort_values("min_purity", ascending=False).to_csv(out)
    print(f"\nwrote full joint table -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
