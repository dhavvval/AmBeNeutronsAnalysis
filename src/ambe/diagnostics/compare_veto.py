"""Compare legacy and current event indices on identical waveform-selected events."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(legacy, current):
    """Return event and candidate losses; fractions are not fitted efficiencies."""
    keys = ["run", "eventTankTime"]
    required = keys + ["is_cosmic", "numberOfClusters", "n_accepted_single",
                       "n_accepted_multiple"]
    frames = []
    for name, frame in (("legacy", legacy), ("current", current)):
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name}: duplicate event keys")
        frame = frame[required].copy().set_index(keys).sort_index()
        if not frame.is_cosmic.isin([True, False, 0, 1]).all():
            raise ValueError(f"{name}: is_cosmic must be boolean or 0/1")
        frame["is_cosmic"] = frame.is_cosmic.astype(bool)
        frame["accepted"] = frame.n_accepted_single + frame.n_accepted_multiple
        frames.append(frame)
    old, new = frames
    if not old.index.equals(new.index):
        raise ValueError("Event cohorts differ; use identical inputs and waveform gates")
    if not old.numberOfClusters.equals(new.numberOfClusters):
        raise ValueError("Total cluster counts differ; this is not an ordering-only comparison")
    if not old.is_cosmic.equals(new.is_cosmic):
        raise ValueError("Veto decisions differ; this is not an ordering-only comparison")
    if (new.loc[new.is_cosmic, "accepted"] > 0).any():
        raise ValueError("Current index still contains candidates in vetoed events")
    rows = []
    for run in [None, *old.index.get_level_values("run").unique()]:
        a = old if run is None else old.xs(run, level="run")
        b = new if run is None else new.xs(run, level="run")
        denominator = int((~a.is_cosmic).sum())
        n_old, n_new = int((a.accepted > 0).sum()), int((b.accepted > 0).sum())
        lost = (a.accepted > 0) & (b.accepted == 0)
        rows.append(dict(
            run="all" if run is None else int(run), waveform_selected=len(a),
            cosmic_events=int(a.is_cosmic.sum()), ambe_triggers=denominator,
            legacy_neutron_events=n_old, current_neutron_events=n_new,
            lost_one_accepted=int((lost & (a.accepted == 1)).sum()),
            lost_multiple_accepted=int((lost & (a.accepted > 1)).sum()),
            lost_original_single=int((lost & (a.numberOfClusters == 1)).sum()),
            lost_original_multiple=int((lost & (a.numberOfClusters != 1)).sum()),
            lost_clusters=int(a.accepted.sum() - b.accepted.sum()),
            candidate_fraction_change_pp=(100 * (n_new - n_old) / denominator
                                          if denominator else np.nan),
        ))
    return pd.DataFrame(rows)


def _read(paths):
    return pd.concat([pd.read_csv(p) if Path(p).suffix == ".csv"
                      else pd.read_parquet(p) for p in paths], ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", nargs="+", required=True)
    parser.add_argument("--current", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(_read(args.legacy), _read(args.current))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
