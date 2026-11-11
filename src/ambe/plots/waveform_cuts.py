"""
Stage-1 waveform selection figures: source vs no-source comparison.

Reads TriggerSummary/WaveformFeatures_<tag>_<run>.parquet (written by
data/processor.py) and the event index. Rationale and results:
docs/efficiency/WAVEFORM_STUDY.md.

Usage:  ambe plots waveformcuts --config configs/data_ambe2v4_effpair_box.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..plotting import save_plot, set_style
from ..paths import REPOSITORY_ROOT
from ..data.processor import AmBeNeutronProcessing, WaveformConfig


ROLE_COLOUR = {"source": "C0", "no-source": "C3", "cross-check": "C2"}
SOURCE_COLOURS = ["C0", "C1", "C4", "C5", "C6", "C8", "C9"]


def _style(runs):
    """Per-run (colour, linestyle). The i-th cross-check run shares the i-th source colour, dashed."""
    out, n_src, n_xc = {}, 0, 0
    multi = sum(role == "source" for _, role, _ in runs) > 1
    for run, role, _ in runs:
        if role == "source":
            out[run] = (SOURCE_COLOURS[n_src % len(SOURCE_COLOURS)] if multi
                        else ROLE_COLOUR[role], "-")
            n_src += 1
        elif role == "cross-check":
            out[run] = ((SOURCE_COLOURS[n_xc % len(SOURCE_COLOURS)], "--") if multi
                        else (ROLE_COLOUR[role], "-"))
            n_xc += 1
        else:
            out[run] = (ROLE_COLOUR[role], "-")
    return out


def _label(run, role):
    if role == "no-source":
        return f"{run} ({role})"
    try:
        x, y, z = AmBeNeutronProcessing().get_source_location(run)
    except ValueError:
        return f"{run} ({role})"
    return f"{run} ({role}, x={x:g} y={y:g} z={z:g})"


def _features_path(ctx, run: int) -> Path:
    import glob

    pats = ctx.inputs.get("waveform_features") or []
    if isinstance(pats, (str, Path)):
        pats = [pats]
    for pat in pats:
        for p in sorted(glob.glob(str(pat))):
            if f"_{run}." in p:
                return Path(p)

    tag = (ctx.extra.get("stage1") or {}).get("tag")
    if not tag:
        raise SystemExit("[plots.waveformcuts] config has no stage1.tag and no "
                         "inputs.waveform_features")
    cand = REPOSITORY_ROOT / "TriggerSummary" / f"WaveformFeatures_{tag}_{run}.parquet"
    if not cand.exists() and cand.with_suffix(".csv").exists():
        cand = cand.with_suffix(".csv")
    if not cand.exists():
        raise SystemExit(f"[plots.waveformcuts] missing {cand}. Run "
                         f"`ambe data process --config <this config> --selection box` first.")
    return cand


def load_features(ctx, run: int):
    """One row per recorded waveform, minus the timestamp=0 placeholder row.

    The saved gate is checked against stage1 ic_min/ic_max only for files of this
    config's own stage1.tag; reference runs from other tags keep their own gate.
    """
    import pandas as pd
    path = _features_path(ctx, run)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    required = {"timestamp", "IC_adjusted", "second_pulse", "accepted"}
    if required - set(df):
        raise ValueError(f"{path}: missing waveform feature columns {sorted(required - set(df))}")
    n_raw = len(df)
    df = df[df["timestamp"] > 0].copy()
    if n_raw != len(df):
        print(f"[plots.waveformcuts] run {run}: dropped {n_raw - len(df)} "
              f"placeholder row(s) with timestamp = 0")
    if df.empty:
        raise ValueError(f"{path}: no recorded waveforms after removing placeholders")
    lo, hi = _ic_window(ctx)
    processor = AmBeNeutronProcessing(WaveformConfig(pulse_gamma=lo, pulse_max=hi))
    for column in ("second_pulse", "accepted"):
        if not df[column].isin([True, False, 0, 1]).all():
            raise ValueError(f"{path}: {column} must contain booleans")
        df[column] = df[column].astype(bool)
    tag = (ctx.extra.get("stage1") or {}).get("tag")
    df.attrs["own_tag"] = bool(tag) and path.name.startswith(f"WaveformFeatures_{tag}_")
    if not df.attrs["own_tag"]:
        return df
    expected = df.apply(processor.passes_selection, axis=1).astype(bool)
    if not df["accepted"].equals(expected):
        raise ValueError(f"{path}: saved gate differs from stage1 IC bounds/veto; regenerate with the matching processor config")
    return df


def _ic_window(ctx) -> tuple[float, float]:
    s1 = ctx.extra.get("stage1") or {}
    if "ic_min" not in s1 or "ic_max" not in s1:
        raise SystemExit("[plots.waveformcuts] config stage1 block has no "
                         "ic_min/ic_max")
    return float(s1["ic_min"]), float(s1["ic_max"])


def plot_ic_overlay(ctx, runs, lo, hi, log_x=True):
    """IC_adjusted over the full range, normalised per recorded waveform."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    edges = (np.logspace(0, np.log10(3e4), 120) if log_x
             else np.linspace(0, 12000, 120))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    sty = _style(runs)
    for run, role, df in runs:
        h, _ = np.histogram(df["IC_adjusted"], bins=edges)
        c, ls = sty[run]
        ax.step(ctr, h / len(df), where="mid", color=c, ls=ls,
                label=f"{_label(run, role)}, {len(df):,} waveforms")
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IC$_{adjusted}$")
    ax.set_ylabel("Fraction of recorded waveforms per bin")
    ax.set_title(ctx.title(f"AmBe small-PMT integrated charge, "
                           f"Stage-1 window {lo:g}–{hi:g}"))
    ax.legend(fontsize=8 if len(runs) > 3 else None)
    fig.tight_layout()
    return save_plot(fig, ctx, "wfcut_ic_overlay")


def plot_ic_zoom(ctx, runs, lo, hi):
    """Zoom on the window; the Stage-1 accepted subset is filled under each curve."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    edges = np.linspace(0, 2500, 101)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    sty = _style(runs)
    for run, role, df in runs:
        c, ls = sty[run]
        h, _ = np.histogram(df["IC_adjusted"], bins=edges)
        ax.step(ctr, h / len(df), where="mid", color=c, ls=ls,
                label=f"{_label(run, role)}, all")
        if not df.attrs.get("own_tag", True):
            continue
        ha, _ = np.histogram(df.loc[df["accepted"], "IC_adjusted"], bins=edges)
        ax.fill_between(ctr, ha / len(df), step="mid", color=c, alpha=.30,
                        label=f"{run} ({role}), Stage-1 accepted")
    ax.set_yscale("log")
    ax.set_xlabel("IC$_{adjusted}$")
    ax.set_ylabel("Fraction of recorded waveforms per bin")
    ax.set_title(ctx.title(f"Stage-1 acceptance, IC window {lo:g}–{hi:g} "
                           f"+ second-pulse veto"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_plot(fig, ctx, "wfcut_ic_zoom")


def plot_cutflow(ctx, rows):
    """Survival through Stage 1 and Stage 2 on one axis."""
    import matplotlib.pyplot as plt
    stages = ["recorded", "IC window", "+ 2nd-pulse veto\n(= Stage-1 accepted)",
              "− cosmic veto\n(= ambe_triggers)"]
    keys = ["n_waveforms", "n_ic_window", "n_accepted", "n_ambe_triggers"]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(stages))
    w = 0.8 / max(len(rows), 1)
    sty = _style([(r["run"], r["role"], None) for r in rows])
    for i, r in enumerate(rows):
        vals = [r[k] if r[k] is not None else 0 for k in keys]
        frac = [100 * v / r["n_waveforms"] for v in vals]
        ax.bar(x + i * w - 0.4 + w / 2, frac, width=w,
               color=sty[r["run"]][0], hatch="//" if sty[r["run"]][1] == "--" else None,
               label=f"{r['run']} ({r['role']})")
        for xi, (f, v) in enumerate(zip(frac, vals)):
            ax.text(xi + i * w - 0.4 + w / 2, f, f"{v:,}", ha="center",
                    va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Percent of recorded waveforms surviving")
    ax.set_title(ctx.title("Stage-1 and Stage-2 cut flow"))
    ax.legend(fontsize=8 if len(rows) > 3 else None)
    fig.tight_layout()
    return save_plot(fig, ctx, "wfcut_cutflow")


def plot_sideband(ctx, src, bkg, lo, hi, sb):
    """No-source IC shape scaled to the source run's high-IC sideband.

    Overestimates by construction: the source sideband contains its own peak's
    tail. Every number returned is an upper bound.
    """
    import matplotlib.pyplot as plt
    sb_lo, sb_hi = sb
    edges = np.linspace(0, 2500, 101)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    def nin(df, a, b, acc=False):
        m = (df["IC_adjusted"] > a) & (df["IC_adjusted"] < b)
        if acc:
            m &= df["accepted"]
        return int(m.sum())

    s_run, _, s_df = src
    b_run, _, b_df = bkg
    n_sb_src, n_sb_bkg = nin(s_df, sb_lo, sb_hi), nin(b_df, sb_lo, sb_hi)
    if n_sb_bkg == 0:
        raise SystemExit(f"[plots.waveformcuts] no-source run {b_run} has no "
                         f"waveforms in the sideband [{sb_lo:g}, {sb_hi:g}]")
    scale = n_sb_src / n_sb_bkg

    hs, _ = np.histogram(s_df["IC_adjusted"], bins=edges)
    hb, _ = np.histogram(b_df["IC_adjusted"], bins=edges)
    hb_scaled = hb * scale

    fig, ax = plt.subplots()
    ax.step(ctr, hs, where="mid", color=ROLE_COLOUR["source"],
            label=f"{s_run} (source)")
    ax.step(ctr, hb_scaled, where="mid", color=ROLE_COLOUR["no-source"],
            label=f"{b_run} (no-source) × {scale:.3f}")
    ax.step(ctr, np.maximum(hs - hb_scaled, 0), where="mid", color="k",
            label="source − scaled no-source")
    ax.set_yscale("log")
    ax.set_xlabel("IC$_{adjusted}$")
    ax.set_ylabel("Waveforms per bin")
    ax.set_title(ctx.title(
        f"Sideband-normalised no-source subtraction, sideband "
        f"{sb_lo:g}–{sb_hi:g}, window {lo:g}–{hi:g}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    paths = save_plot(fig, ctx, f"wfcut_sideband_{s_run}")

    win_src, win_bkg = nin(s_df, lo, hi), nin(b_df, lo, hi)
    acc_src, acc_bkg = nin(s_df, lo, hi, True), nin(b_df, lo, hi, True)
    out = {
        "source_run": s_run, "nosource_run": b_run,
        "sideband_lo": sb_lo, "sideband_hi": sb_hi,
        "n_sideband_source": n_sb_src, "n_sideband_nosource": n_sb_bkg,
        "scale": scale,
        "n_in_window_source": win_src,
        "flat_in_window_predicted": scale * win_bkg,
        "flat_frac_in_window": scale * win_bkg / win_src if win_src else np.nan,
        "n_accepted_source": acc_src,
        "flat_accepted_predicted": scale * acc_bkg,
        "flat_frac_accepted": scale * acc_bkg / acc_src if acc_src else np.nan,
        "bound": "UPPER -- the source sideband contains the source peak's own tail",
    }
    return paths, out


def run(ctx, argv=None):
    import pandas as pd

    p = argparse.ArgumentParser(prog="ambe plots waveformcuts")
    p.add_argument("--linear-x", action="store_true",
                   help="linear IC axis on the overlay instead of log")
    args, _ = p.parse_known_args(argv or [])

    set_style()
    wc = ctx.extra.get("waveform_cuts") or {}
    if "source_runs" in wc:
        src_runs = [int(r) for r in wc["source_runs"]]
        bkg_run = int(wc["background_run"])
        xchk_runs = [int(r) for r in wc.get("crosscheck_runs") or []]
    else:
        fp = dict(ctx.fit_params or {})
        if not fp:
            raise SystemExit("[plots.waveformcuts] config has neither "
                             "waveform_cuts.source_runs nor a fit_params block")
        src_runs = [int(fp["source_run"])]
        bkg_run = int(fp["background_run"])
        xchk_runs = [int(fp["crosscheck_run"])] if fp.get("crosscheck_run") else []
    lo, hi = _ic_window(ctx)

    runs = ([(r, "source", load_features(ctx, r)) for r in src_runs]
            + [(bkg_run, "no-source", load_features(ctx, bkg_run))]
            + [(r, "cross-check", load_features(ctx, r)) for r in xchk_runs])
    bkg = next(x for x in runs if x[1] == "no-source")

    ambe_trig = {}
    try:
        from ..efficiency_extraction.efficiency_fit import _resolve_event_index
        ei = _resolve_event_index(ctx)
        for run_, _, _ in runs:
            g = ei[ei["run"] == run_]
            if len(g):
                ambe_trig[run_] = int((~g["is_cosmic"].astype(bool)).sum())
    except SystemExit as exc:
        print(f"[plots.waveformcuts] no event index ({exc}); cut flow stops at "
              f"Stage-1 accepted")

    rows = []
    for run_, role, df in runs:
        ic = df["IC_adjusted"]
        in_win = (ic > lo) & (ic < hi)
        rows.append({
            "run": run_, "role": role,
            "n_waveforms": len(df),
            "n_ic_window": int(in_win.sum()),
            "n_accepted": int(df["accepted"].sum()),
            "n_ambe_triggers": ambe_trig.get(run_),
            "frac_ic_window": float(in_win.mean()),
            "frac_accepted": float(df["accepted"].mean()),
            "second_pulse_frac_all": float(df["second_pulse"].mean()),
            "second_pulse_frac_in_window": float(
                df.loc[in_win, "second_pulse"].mean()),
            "ic_median": float(ic.median()),
            "gate_from_this_config": bool(df.attrs.get("own_tag", True)),
        })
        r = rows[-1]
        print(f"[plots.waveformcuts] run {run_} ({role}): {r['n_waveforms']:,} "
              f"waveforms, {r['n_ic_window']:,} in window "
              f"({100*r['frac_ic_window']:.2f} %), {r['n_accepted']:,} accepted "
              f"({100*r['frac_accepted']:.2f} %), median IC {r['ic_median']:.0f}")

    plot_ic_overlay(ctx, runs, lo, hi, log_x=not args.linear_x)
    plot_ic_zoom(ctx, runs, lo, hi)
    plot_cutflow(ctx, rows)

    ctx.csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ctx.csv_dir / "waveform_cutflow.csv", index=False)

    sb = wc.get("sideband_ic")
    if not sb:
        print("[plots.waveformcuts] no waveform_cuts.sideband_ic in the config; "
              "the sideband-subtraction figure is SKIPPED. It has no default "
              "because the answer moves by a factor of several across plausible "
              "sidebands; set e.g. `waveform_cuts: {sideband_ic: [1600, 2500]}`.")
    else:
        sb = (float(sb[0]), float(sb[1]))
        subs = []
        for run_, role, df in runs:
            if role == "no-source":
                continue
            _, info = plot_sideband(ctx, (run_, role, df), bkg, lo, hi, sb)
            info["role"] = role
            if run_ in ambe_trig and bkg_run in ambe_trig:
                info["flat_ambe_triggers_predicted"] = info["scale"] * ambe_trig[bkg_run]
                info["n_ambe_triggers_source"] = ambe_trig[run_]
                info["flat_frac_ambe_triggers"] = (
                    info["flat_ambe_triggers_predicted"] / ambe_trig[run_])
            subs.append(info)
            print(f"[plots.waveformcuts] {run_} ({role}) vs {bkg_run}: "
                  f"scale {info['scale']:.3f}, flat component "
                  f"{100*info['flat_frac_accepted']:.2f} % of Stage-1 accepted, "
                  f"{100*info.get('flat_frac_ambe_triggers', float('nan')):.2f} % of "
                  f"ambe_triggers after the cosmic veto (both UPPER BOUNDS)")
        pd.DataFrame(subs).to_csv(ctx.csv_dir / "waveform_sideband.csv", index=False)

    print(f"[plots.waveformcuts] figures in {ctx.plots_dir}")
    print(f"[plots.waveformcuts] tables  in {ctx.csv_dir}")


def cli(ctx, argv=None):
    run(ctx, argv)
