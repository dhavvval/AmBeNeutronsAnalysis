# Lucho 100k AmBe pipeline — remote server run instructions

Run the full 100k-event Lucho AmBe MC pipeline (process → features → MVA freeze)
on a remote server with the same codebase. Produces two frozen MVA artifacts
(OPTICS + ClusterFinder methods, both trained vs. all MC non-neutron background)
ready for application to real AmBe data with `mva_analysis.py --score-data`.

No `ambe mc optics` grid-sweep step is run — features step does its own internal
OPTICS clustering with the production parameters.

---

## 0. Prerequisites on the server

Two Python virtual envs:

| Venv | Used by | Required packages |
|---|---|---|
| `~/venvs/annie`   | `ambe mc process`, `ambe mc features` | `uproot`, `awkward`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `pyyaml`, plus the local `ambe` package (`pip install -e .` from the repo root) |
| `~/venvs/annie311`| `python mva_analysis.py --save-model`    | everything above **plus** `tensorflow` (for the NN), `xgboost`, `joblib`, `matplotlib` |

If the venvs are at different paths on this server, edit the two `VENV_*` lines
near the top of `run_lucho_100k.sh` (see step 3).

---

## 1. Pull the latest code

The local-side branch this was prepared on is **`feature/blessedPlotcuts`**. Two
new files and one source-file change are needed:

| File | Status | Purpose |
|---|---|---|
| `configs/mc_lucho_100k.yaml` | new | 100k-event run config (10-file Lucho glob, no CC residual filter) |
| `run_lucho_100k.sh`          | new | end-to-end runner |
| `src/ambe/mc/cc_selection.py` | modified | flips `apply_residual_filter()` defaults to `cc_only=False`, `prompt_window_ns=0` so AmBe configs no longer need to opt out of the CC residual filter |

On the server:

```bash
cd <repo-root>
git fetch origin
git checkout feature/blessedPlotcuts
git pull
```

If those changes haven't been pushed yet, ask Dhaval for a patch — every change
is unstaged on his laptop right now.

---

## 2. Adjust paths in `configs/mc_lucho_100k.yaml`

Three path keys will differ between Dhaval's laptop and the server. Edit them
before running:

```yaml
output_root: <SERVER_PATH>/ambe_output            # where parquets and frozen models land
inputs:
  root_files:
    - <SERVER_PATH>/Lucho_ANNIEEvent/ANNIEEvent_MC_wcsimlucho_*.root  # 10 ROOT files, 10k events each
geometry:
  pmt_geometry:   <SERVER_PATH>/FullTankPMTGeometry.csv
  timing_offsets: <SERVER_PATH>/TankPMTTimingOffsets.csv
```

(For reference, the laptop paths are:
`/Users/dajana/Documents/ANNIEEvent/Lucho_ANNIEEvent/`,
`/Users/dajana/Documents/AmBe/AmBeNeutronsAnalysis/ambe_output/`,
and the two geometry CSVs under `/Users/dajana/Documents/ToolAnalysis/configfiles/LoadGeometry/`.)

Nothing else in the config needs to change. Source position
`[0, -0.1446, 1.681]` is the WCSim Port-5 position used by Lucho's MC generation
and is the same on the server.

---

## 3. (Optional) Adjust venv paths in `run_lucho_100k.sh`

Top of the script:

```bash
VENV_PIPE="$HOME/venvs/annie/bin/activate"
VENV_MVA="$HOME/venvs/annie311/bin/activate"
```

Change to wherever the venvs live on the server.

---

## 4. Run it

```bash
cd <repo-root>
mkdir -p logs
bash run_lucho_100k.sh 2>&1 | tee logs/lucho_100k.log
```

Or detach if it'll outlive your SSH session:

```bash
nohup bash run_lucho_100k.sh > logs/lucho_100k.log 2>&1 &
disown
tail -f logs/lucho_100k.log   # watch progress
```

---

## 5. Expected wall-clock (rough, scales with CPU count)

On Dhaval's M-series laptop (~6 cores used):

| Step | Wall time |
|---|---|
| `ambe mc process`   (10 ROOT files → 2.84M hits)  | ~1.5 min |
| `ambe mc features`  (OPTICS+CF on 95k events)     | ~50 min (~33 ev/s) |
| `mva_analysis.py --method optics --save-model`    | ~10–20 min |
| `mva_analysis.py --method clusterfinder --save-model` | ~10–20 min |
| **Total** | **~90–120 min** |

A 16-core server should run features in ~20 min and total wall in ~45–60 min.

---

## 6. Deliverables — what to send back

Inside `<SERVER_PATH>/ambe_output/mc_lucho_100k/parquet/`:

```
mc_lucho_100k__pulses.parquet                    # 2.84M per-hit rows
mc_lucho_100k__clusterfinder.parquet             # 148k CF clusters from the processor
mc_lucho_100k__cluster_features.parquet          # both OPTICS + CF cluster features
mc_lucho_100k_optics_frozen.pkl                  # frozen OPTICS MVA (RF + GBT + XGB)
mc_lucho_100k_optics_frozen__nn.keras            # companion NN for the OPTICS freeze
mc_lucho_100k_cf_frozen.pkl                      # frozen CF MVA
mc_lucho_100k_cf_frozen__nn.keras                # companion NN for the CF freeze
```

The four files starting with `mc_lucho_100k_*_frozen` are the artifacts to ship
back. With them in hand, scoring real data is one command per method:

```bash
python mva_analysis.py --config configs/mc_lucho_100k.yaml \
    --score-data <real_data_features>.parquet \
    --model ambe_output/mc_lucho_100k/parquet/mc_lucho_100k_optics_frozen.pkl
```

`score_data()` auto-loads the companion `.keras` NN if it's next to the `.pkl`.

---

## 7. Notes / gotchas

- **CC residual filter is OFF by default now.** Earlier commit `6e8f1ad` added a
  CC + prompt-window filter inside `cluster_features.py` for the CC-neutrino
  neutron search. Its defaults silently zeroed every event in an AmBe run
  (no CC events possible). The current code in `cc_selection.py:875` defaults to
  `cc_only=False` / `prompt_window_ns=0`. The CC-neutrino config
  (`configs/cc_neutrino_optics.yaml`) opts in explicitly, so it's unaffected.
  If the server is still on a pre-2026-06-09 commit, either pull or set this
  block in `mc_lucho_100k.yaml`:

      residual:
        cc_only: false
        prompt_window_ns: 0

- **MVA freeze flags.** The runner uses `--method {optics,clusterfinder} --bkg-mode all
  --all-events --save-model <path>`. `--all-events` disables the default
  `--cc-only` filter inside `mva_analysis.py` (same reason as above — AmBe has
  no CC events).

- **Process step is single-threaded per file** but `awkward`/`uproot` are fast.
  Features step parallelises via OPTICS internals and uses ~6 cores typically.

- **If features crashes mid-run**, the processor outputs (`__pulses.parquet`,
  `__clusterfinder.parquet`) survive on disk. Re-running `ambe mc features`
  alone is safe — it reads only those two files and overwrites
  `__cluster_features.parquet`.
