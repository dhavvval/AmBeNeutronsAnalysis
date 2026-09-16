# Michel samples

There is no tracked `Michel_tuning.py` in this repository. It is a historical
reference in comments, not a runtime dependency. The two existing selectors are
not equivalent, so this cleanup keeps their sample definitions separate.

| Property | `tag_michel_background.py --mode dirt` | `extract_michel_features.py` default |
|---|---|---|
| Input/output | ROOT → selected hit parquet | ROOT → cluster-feature parquet |
| Event flags | `TankMRDCoinc == 0`, `NoVeto == 0`, `Extended == 1`, more than one cluster | None |
| Muon | Brightest cluster; >100 hits; 1000 < PE < 4000; 0 ≤ CB ≤ 0.2; sum(z·PE) ≥ 0 | Brightest cluster with PE ≥ configured minimum (default 500) |
| Michel timing | 200 < dt < 5000 ns | 200 ≤ dt ≤ 70000 ns by default |
| Michel charge | 0 < PE < 650 | PE < 650; no lower bound in the existing code |
| Michel hits | ≥20 | ≥5 |
| Michel CB | 0 < CB < 0.18 after tightening | 0 < CB < 0.2 |

The relaxed extractor's old prose omitted its CB cut and described strict time/PE
edges that did not match the implementation. The table above describes the actual
code. Its thresholds are configurable through `michel`; this cleanup does not
retune them. The strict tagger's `--mode electron` skips the muon and relative-time
selection and applies only the Michel PE/hit/CB criteria to electron-only MC.

`--exact-cuts` on the relaxed extractor is a historical partial diagnostic. It
checks/reports `hadExtended`, then computes cluster cuts independently of that
flag. It does not implement all the event-flag cuts its former docstring claimed,
and uses a ≥50-hit muon criterion rather than the strict tagger's >100. It must
not be used as evidence that the full dirt-muon selection has been validated.

## Tagged hits to OPTICS features

```bash
python michel/tag_michel_background.py /path/to/dirtmuon --mode dirt --output-dir ambe_output/michel
python michel/extract_michel_optics_features.py \
  --config configs/mc_michel_mva.yaml \
  --hits ambe_output/michel/michel_background_hits.parquet \
  --min-samples 3 --xi 0.05 --t-unit-ns 25 \
  --output ambe_output/michel/optics_features.parquet
```

The second command now calls the maintained `ambe mc features` implementation;
its former target `extract_cluster_features.py` did not exist. It runs one OPTICS
setting separately on each already selected ClusterFinder cluster's saved hits,
then computes features for the resulting subclusters. It does not select dirt muons,
cluster the whole event anew, or perform a hyperparameter sweep. Use `mc optics`
for grid searches. Omit `--rerun-optics` from the equivalent `ambe mc features
--background-hits ...` command to retain the input cluster boundaries. These
background extractors assign `is_background=1` for neutron-classifier training,
including if the input electron-only tagger labelled its own electron signal as 0.

For the separate relaxed sample:

```bash
python michel/extract_michel_features.py --config configs/mc_michel_mva.yaml
```

Both paths already reuse `ambe.mc.cluster_features`. Combining their selectors
would change the training sample and requires a deliberate choice of sample definition.
Before changing cuts, inspect the ROOT branch conventions, run a cumulative cutflow,
compare truth-labelled efficiency/purity and delayed-time shapes, and test model
performance on held-out samples. No ROOT files were supplied, so none of those
physics checks has been performed here.
