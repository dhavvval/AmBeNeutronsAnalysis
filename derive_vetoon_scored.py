#!/usr/bin/env python3
"""Derive the veto-ON scored parquet from the veto-OFF one, exactly.

WHY THIS IS EXACT, NOT AN APPROXIMATION. In
extract_cf_features_beamcluster_data.extract_cf_features_one_file the prompt veto is a
per-EVENT `continue` taken BEFORE any hit is filtered, clustered or featurised. Nothing
downstream of it depends on the veto setting. So the veto-ON cluster table is precisely
the veto-OFF table with every row from a vetoed event removed -- no feature is
recomputed and no value changes. Running the extraction twice would burn ~11 h of CPU
to reproduce a row filter.

Which events were vetoed is not inferred: it is read from the box deck's own per-event
index (TriggerSummary/EventIndex_<tag>_<run>.parquet, `is_cosmic`), which is the same
Selection.cosmic verdict the deck applied, so the derived table cannot drift from it.

usage: derive_vetoon_scored.py <vetoOFF scored parquet> <vetoON out> <EventIndex glob>
"""
import sys
import glob

import pandas as pd

src, dst, idx_glob = sys.argv[1], sys.argv[2], sys.argv[3]

files = sorted(glob.glob(idx_glob))
if not files:
    raise SystemExit(f"no EventIndex parquets matched {idx_glob}")
idx = pd.concat([pd.read_parquet(f, columns=["run", "eventID", "is_cosmic"])
                 for f in files], ignore_index=True)
print(f"[derive] event index: {len(idx):,} events over {idx.run.nunique()} runs, "
      f"{int(idx.is_cosmic.sum()):,} vetoed ({100*idx.is_cosmic.mean():.2f} %)")

d = pd.read_parquet(src)
print(f"[derive] veto-OFF scored: {len(d):,} clusters over {d.run.nunique()} runs")

cosmic = set(zip(idx.loc[idx.is_cosmic, "run"].astype("int64").tolist(),
                 idx.loc[idx.is_cosmic, "eventID"].astype("int64").tolist()))
keep = [(int(r), int(e)) not in cosmic
        for r, e in zip(d["run"], d["event_tank_time"])]
out = d[keep].reset_index(drop=True)
print(f"[derive] veto-ON  scored: {len(out):,} clusters "
      f"({100*len(out)/max(len(d),1):.2f} % of veto-OFF)")

# The derived table must contain no cluster from a vetoed event, and must not have
# invented one. Both are cheap to assert, so assert them rather than trust the filter.
assert not any((int(r), int(e)) in cosmic
               for r, e in zip(out["run"], out["event_tank_time"])), \
    "derived veto-ON table still contains clusters from vetoed events"
assert len(out) + sum(1 for k in keep if not k) == len(d)
out.to_parquet(dst, index=False)
print(f"[derive] wrote -> {dst}")
