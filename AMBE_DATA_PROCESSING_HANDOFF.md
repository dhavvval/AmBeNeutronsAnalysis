# AmBe v4 Data Processing — Handoff (2026-07-06)

Pick this up in a fresh chat by pointing Claude Code at this file. It covers
everything done today on the AmBe v4 real-data campaign: processing, the IC
window bug/fix, efficiency, capture-time fits, and the two background jobs
running when the SSH session was about to be closed.

## Dataset

- `/pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4/` — 15 runs, two
  sub-campaigns: March 2026 (6046, 6056, 6060, 6061, 6062) and June-July 2026
  "continued" (6165, 6166, 6186, 6187, 6188, 6189, 6230, 6231, 6232, 6234).
- Port/position table for all 15 runs is in the conversation history and in
  `src/ambe/data/processor.py`'s `source_positions` dict (already updated with
  the v4 entries) and `src/ambe/data/eff.py` (legacy copy, also updated).

## Code changes made today (all committed except where noted)

1. `src/ambe/data/processor.py` — added v4 run/port positions (10 new runs
   under `##AmBe v4 Campaign 3 continued`), flipped `WaveformConfig` default
   to v4 (`pulse_gamma=700`, `pulse_max=1200`), hardened
   `process_run_waveforms()` to recursively glob `.root` files and
   try/except around `uproot.open()` (was: flat `os.listdir` + unguarded
   open — would crash on any nested-subfolder layout).
2. `src/ambe/data/eff.py` — synced the same v3+v4 position blocks (legacy
   module, not on the live import path, but kept consistent per your call).
3. `src/ambe/plots/heatmap.py` — added missing `(0, -60, 102): "Port 3"` to
   `PORT_INFO` (run 6187's y=-60 doesn't sit on the historical y-grid).
4. **`run_waveform_gated_pipeline.py` — real bug found and fixed**: this
   script hardcoded `wf.pulse_gamma = 400` / `wf.pulse_max = 575` (v1's
   window) regardless of `WaveformConfig`'s active default. This silently
   applied the WRONG cut to the first attempt at processing v4
   (acceptance ~0.6-0.7%, should be ~40%). Fixed to defer to whatever
   `WaveformConfig()` default is active, and added `--pulse-gamma`/
   `--pulse-max` CLI overrides for one-off tests without touching the shared
   default.
5. `configs/data_ambe2v4.yaml` — new config for `ambe plots heatmap`
   (efficiency) against the v4 real-data trigger summary. **Not yet
   committed** (created mid-session).

## Pipeline used for real DATA (not MC, not MVA)

Per your explicit correction mid-session: **do not use the MVA/frozen-model/
score-cut pipeline for this.** Use `src/ambe/plots` directly on the raw
waveform-gated candidate CSVs:

1. **Processing**: `run_waveform_gated_pipeline.py --dataset <dir> --runinfo
   <tag> [--pulse-gamma N --pulse-max N]` → writes
   `EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_<tag>_<run>.csv`
   and `TriggerSummary/AmBeTriggerSummary_<tag>.csv` /
   `AmBeWaveformResults_<tag>.csv`.
2. **Efficiency**: `ambe plots heatmap --config configs/data_ambe2v4.yaml`
   (or a copy with a different `trigger_summary` path for a different tag)
   → `efficiency = unique_neutron_triggers / ambe_triggers` per position,
   written to `ambe_output/ambe_data/<run_name>/plots/`.
3. **Capture-time**: build `AmBeNeutronAnalyzer` directly (NOT the
   `ambe plots basic` CLI wrapper — its `run(ctx)` hardcodes
   `tasks=["2d_histograms","1d_histograms"]`, no lmfit). Needs three
   workarounds, all already applied in the commands below:
   - `MPLBACKEND=Agg` (or `matplotlib.use('Agg')` before any other
     matplotlib import) — `DISPLAY` is set over this SSH session and
     without this the script hangs on Tkinter cleanup after finishing
     (`delayed_destroy` errors), even though the actual analysis completes.
   - `scipy.signal.gaussian` monkeypatch before importing `ambe.plots.basic`
     (it unconditionally `import pymc`, which pulls `arviz`, which imports
     the scipy<1.13 name `scipy.signal.gaussian` — removed in newer scipy).
   - `lmfit.Model.fit` monkeypatched to force `method='leastsq'` —
     `lmfit_analysis()` hardcodes `method='basinhopping'`, a global
     optimizer that doesn't populate parameter stderr (uncertainties come
     back as exactly 0.0 otherwise).

Reusable command template (edit `data_directory`/`file_pattern`/`output_pdf`
for a different tag):

```bash
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
source /exp/annie/app/users/dajana/myboy/bin/activate
MPLBACKEND=Agg python -u -c "
import sys; sys.path.insert(0, 'src')
import scipy.signal, scipy.signal.windows as _w
if not hasattr(scipy.signal, 'gaussian'): scipy.signal.gaussian = _w.gaussian
import matplotlib; matplotlib.use('Agg')
from ambe.plots.basic import AmBeNeutronAnalyzer
import lmfit
_orig_fit = lmfit.Model.fit
def _fit_leastsq(self, *a, **kw):
    kw['method'] = 'leastsq'; return _orig_fit(self, *a, **kw)
lmfit.Model.fit = _fit_leastsq
analyzer = AmBeNeutronAnalyzer(data_directory='EventAmBeNeutronCandidatesData/',
    output_pdf='<OUTPUT_PDF_PATH>')
analyzer.update_fitting_config(fit_min_time=2.0, fit_max_time=70.0)  # see note below
analyzer.run_analysis(file_pattern='EventAmBeNeutronCandidates_<TAG>_*.csv',
    tasks=['1d_histograms', 'lmfit_fit', 'summary'])
print('SCRIPT DONE')
"
```

## fit_min_time / fit_max_time — IMPORTANT, unresolved

- `AmBeNeutronAnalyzer`'s hardcoded default is `fit_min_time=10.0,
  fit_max_time=67.0`. This traces back to an undocumented edit sitting in
  your own working tree before this session (no comment/commit message)
  and is inconsistent with `src/ambe/plots/combined.py`'s default
  (`fit_min_time=2.0`) and the README's example config (also `2.0`).
- Your instruction: for real AmBe DATA (not MC), always fit `2.0` to `70.0`
  (saved to memory as a standing preference for future sessions).
- **BUT**: when actually run on the v4 `AmBe2.0v4_gated` (700-1200 IC
  window) candidate data, the 2-70 fit was MUCH WORSE than 10-67:
  - 10-67 window: χ²/ndof ≈ 0.9-2.1 across all 14 positions, clean
    convergence, weighted avg τ=29.16±0.44 µs, thermal=5.71±0.46 µs.
  - 2-70 window: χ²/ndof exploded to 40-360, most `tau` collapsed to the
    initial guess (25.0), most `therm` pinned near its lower bound (0.1),
    uncertainties came back nonsensical (e.g. ±2×10⁷).
  - **This suggests the original `10.0` lower bound, despite being
    undocumented, may be empirically necessary** — something in the 2-10 µs
    (and/or 67-70 µs tail) region isn't described by the current
    `NeutCapture(t) = A·(1-exp(-t/therm))·exp(-t/tau) + B` model, even
    though that model already has a `therm` rise-term that should in
    principle handle early-time turn-on.
  - **Not yet investigated**: why the wider window breaks convergence.
    Worth checking the 2-10 µs histogram bins directly (afterpulsing?
    cosmic leakage right at the `ct_min=2000ns` selection edge? reflected
    light?) before deciding whether to keep 10-67, patch the model, or
    add per-position custom initial guesses/bounds for a 2-70 fit.
  - Full log: `/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/logs/capture_time_AmBe2.0v4_gated_fit2to70.log`

## Results so far (AmBe2.0v4_gated, 700 < IC < 1200, the "production" run)

**Processing**: all 15 runs, 635,681 total waveforms, 261,731 accepted
(41.2% overall), 128,352 total neutron candidate clusters. Per-run table and
full port/position breakdown with efficiency are in the conversation
history (also reconstructable from
`TriggerSummary/AmBeTriggerSummary_AmBe2.0v4_gated.csv`).

**Efficiency** (via `ambe plots heatmap`, real v4 positions only): overall
mean 56.3% across 14 positions, ranging 40.9% (Port 3, run 6186) to 71.8%
(Port 5, run 6046). You flagged this as "not satisfactory," which is what
triggered the 600-1400 IC window test below.

**Capture-time** (10-67 µs fit, the one with good χ²): weighted average
τ = 29.16 ± 0.44 µs, thermal time = 5.71 ± 0.46 µs. Two positions (runs
6231, 6232 — both Port 1) had `therm` pinned near the 0.1 lower bound with
`leastsq` — likely a real fit-quality issue for those two, not just a
reporting artifact.

## Background jobs running when this was written

**tmux session `AmBev4`** — testing a WIDER IC window (600 < IC < 1400,
tag `AmBe2.0v4_test`) to see if narrowing at 700-1200 was leaving efficiency
on the table:
```
python -u run_waveform_gated_pipeline.py --dataset /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4 \
    --runinfo AmBe2.0v4_test --pulse-gamma 600 --pulse-max 1400
```
Log: `/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/logs/waveform_gated_AmBe2.0v4_test_stage12.log`
Progress as of last check (laptop about to go offline — tmux session is
unaffected, it runs server-side on anniegpvm02): 10/15 runs done (6186,
6187, 6188, 6189, 6056, 6060, 6061, 6062, 6230, 6231), acceptance 45-51%
(vs 37-44% with 700-1200) — confirms the narrower window WAS clipping real
efficiency. Currently processing 6232 (the largest run, ~156k waveforms —
will take the longest of the 15). Still need after that: 6234, 6165, 6166,
6046. Once done: rebuild the efficiency heatmap (new
config pointing at `AmBeTriggerSummary_AmBe2.0v4_test.csv` +
`EventAmBeNeutronCandidates_AmBe2.0v4_test_*.csv`) and rerun capture-time
(same command template above, tag `AmBe2.0v4_test`) — decide the
fit-window question above before doing that.

**tmux session `AmBev4_capture`** — the 2-70 µs refit described above.
**Already finished** ("SCRIPT DONE" in its log) — the bad-fit finding is
already captured above, nothing left running there.

To check on `AmBev4` after reconnecting: `tmux attach -t AmBev4` (or
`tmux capture-pane -t AmBev4 -p | tail -40` without attaching).

## Next steps (in likely priority order)

1. Let `AmBev4`'s 600-1400 test finish (~7 more runs).
2. Build efficiency heatmap + capture-time for the test tag, compare against
   the 700-1200 production numbers directly (does wider window actually
   raise `unique_neutron_triggers/ambe_triggers`, or just accept more
   background that gets vetoed downstream by `ambe_single_cut`?).
3. Investigate the 2-70 µs fit failure before deciding whether to keep
   10-67 or fix something else.
4. Commit `configs/data_ambe2v4.yaml` and the `run_waveform_gated_pipeline.py`
   fix if not already committed (check `git status`).
5. Decide on the final IC window for v4 (700-1200 vs 600-1400 vs something
   else derived from where efficiency actually plateaus).
