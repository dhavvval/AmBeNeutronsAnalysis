# `ambe.mc` module guide

This folder contains the Monte Carlo analysis pipeline used by the `ambe mc ...` CLI commands.

## Commands and modules

| CLI command | Module | What it does |
|---|---|---|
| `ambe mc process --config <yaml>` | `processor.py` | Reads ROOT files and writes per-hit parquet with truth matching. |
| `ambe mc optics --config <yaml>` | `optics.py` | Runs OPTICS sweeps and ClusterFinder baseline, then writes per-event metrics. |
| `ambe mc features --config <yaml>` | `cluster_features.py` | Extracts per-cluster physics features for MVA and diagnostic plots. |
| `ambe mc discriminate --config <yaml>` | `cluster_discrimination.py` | Cluster-level discrimination helper command. |
| `ambe mc match --config <yaml>` | `cluster_matching.py` | Cluster-to-truth matching diagnostics. |
| `ambe mc timespread --config <yaml>` | `time_spread.py` | Timing spread analysis utilities. |
| `ambe mc hitcomp --config <yaml>` | `hit_accounting.py` | Hit accounting and class-composition checks. |

## Typical run order

```bash
ambe mc process  --config configs/mc_lucho_full.yaml
ambe mc optics   --config configs/mc_lucho_full.yaml
ambe mc features --config configs/mc_lucho_full.yaml
```

This produces outputs under:

```text
<output_root>/<run_name>/
  parquet/
  csv/
  plots/
```

## Input expectations

- Config file must include `run_name`, input ROOT paths, and `output_root`.
- ROOT content is expected to contain ANNIE hit and DirectParent branches used by `processor.py`.
- For feature extraction, geometry CSV paths should be available in the config (`geometry` block).

## Notes for contributors

- Keep MC command modules callable from CLI via a `cli(ctx, argv)` entrypoint.
- If you add a new MC command, register it in `src/ambe/cli.py` under the `("mc", "...")` dispatch table.
- Prefer adding new analysis outputs under the configured output root, not inside the repository root.
