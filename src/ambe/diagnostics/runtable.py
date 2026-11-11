"""Build run-comparison tables from saved waveform and trigger summaries.

Percentages use total waveforms. A tag must have one run per position because
the processor aggregates summaries by position. Ambiguous joins raise an error."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TS = Path("TriggerSummary")

# (label, key, is_percent). Percentages are all of total_waveforms.
ROWS = [
    ("Total waveforms",                              "total_waveforms",   False),
    ("Accepted waveforms",                           "accepted_waveforms", False),
    ("% Accepted waveforms",                         "accepted_waveforms", True),
    ("AmBe triggers",                                "ambe_triggers",     False),
    ("% AmBe triggers (of total waveforms)",         "ambe_triggers",     True),
    ("% Cosmic events",                              "cosmic_events",     True),
    ("% Unique neutron triggers",                    "unique_neutron_triggers", True),
    ("% Single neutron candidates",                  "single_neutron_candidates", True),
    ("% Multiple neutron candidates",                "multiple_neutron_candidates", True),
]


def load_tag(tag):
    """Per-run metrics for one Stage-1/2 tag, joined on source position."""
    wf = pd.read_csv(TS / f"AmBeWaveformResults_{tag}.csv")
    tr = pd.read_csv(TS / f"AmBeTriggerSummary_{tag}.csv")

    wf["pos"] = list(zip(wf.sourceX.astype(float), wf.sourceY.astype(float),
                         wf.sourceZ.astype(float)))
    tr["pos"] = list(zip(tr.x_pos.astype(float), tr.y_pos.astype(float),
                         tr.z_pos.astype(float)))

    # The aggregation happens BEFORE the CSV is written, so a duplicate-position check
    # on the file finds nothing -- there is already just one row per position. What
    # survives is the summed run number, and ANNIE run numbers are 4 digits, so a
    # 5-digit "run" is the fingerprint of N runs collapsed into one row.
    bogus = wf[(wf.run < 1000) | (wf.run > 9999)]
    if len(bogus):
        raise SystemExit(
            f"[runtable] tag {tag!r}: {len(bogus)} row(s) carry an impossible run "
            f"number: {sorted(bogus.run.tolist())}.\n"
            "  This is several runs sharing one source position and being aggregated "
            "into a single row, with the run column summed along with the counts "
            "(e.g. 25039 = 6254+6256+6264+6265 in the v4 special tag). Every count in "
            "such a row is an N-run total, not a run.\n"
            "  Reprocess the runs you want under a tag where each has its own source "
            "position -- see configs/data_ambe2v4_refpair_box.yaml.")

    dup = wf.groupby("pos").size()
    bad = dup[dup > 1]
    if len(bad):
        raise SystemExit(
            f"[runtable] tag {tag!r}: {len(bad)} source position(s) hold more than one "
            f"run, so the per-run and per-position files cannot be joined:\n" +
            "\n".join(f"    {p}: runs {sorted(wf[wf.pos == p].run.tolist())}"
                      for p in bad.index) +
            "\n  Process those runs under a tag where each has its own position. "
            "(This is the check that catches the aggregated 'run 25039' row.)")

    m = wf.merge(tr, on="pos", how="inner", validate="one_to_one")
    if len(m) != len(wf):
        missing = sorted(set(wf.run) - set(m.run))
        print(f"[runtable] tag {tag!r}: {len(missing)} run(s) have no trigger-summary "
              f"row (no surviving candidates?): {missing}")
    m["tag"] = tag
    return m


def build(tags, order=None):
    frames = [load_tag(t) for t in tags]
    d = pd.concat(frames, ignore_index=True)
    d["run"] = d.run.astype(int)

    if order:
        want = [int(r) for r in order]
        missing = [r for r in want if r not in set(d.run)]
        if missing:
            raise SystemExit(f"[runtable] --order names run(s) not present: {missing}")
        d = d.set_index("run").loc[want].reset_index()
    else:
        d = d.sort_values("run").reset_index(drop=True)
    return d


def render(d):
    runs = list(d.run)
    header = "| Metric | " + " | ".join(f"Run {r}" for r in runs) + " |"
    sep = "| --- |" + " --- |" * len(runs)
    lines = [header, sep]
    out_rows = {}
    for label, key, pct in ROWS:
        vals = []
        for _, r in d.iterrows():
            if pct:
                v = 100.0 * r[key] / r["total_waveforms"]
                vals.append(f"{v:.2f}%")
                out_rows.setdefault(label, []).append(round(v, 4))
            else:
                vals.append(f"{int(r[key]):,}")
                out_rows.setdefault(label, []).append(int(r[key]))
        lines.append(f"| **{label}** | " + " | ".join(vals) + " |")
    return "\n".join(lines), out_rows


def run(ctx, argv=None):
    ap = argparse.ArgumentParser(prog="ambe data runtable")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="Stage-1 tags to include; default is the config's own tag")
    ap.add_argument("--order", nargs="+", default=None,
                    help="explicit run order for the columns")
    ap.add_argument("--name", default=None, help="slug for the output CSV")
    args = ap.parse_args(argv or [])

    tags = args.tags or [ctx.extra.get("stage1", {}).get("tag", ctx.run_name)]
    d = build(tags, args.order)
    table, rows = render(d)

    print()
    print(table)
    print()
    print("Every percentage is of TOTAL WAVEFORMS.")
    print(f"Tags: {', '.join(tags)}")

    name = args.name or "_".join(tags)
    out = TS / f"RunTable_{name}.csv"
    pd.DataFrame(rows, index=[f"run_{r}" for r in d.run]).T.to_csv(out)
    print(f"\n[runtable] wrote {out}")
    return 0


def cli(ctx, argv=None):
    return run(ctx, argv)
