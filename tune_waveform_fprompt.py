"""
tune_waveform_fprompt.py

Offline tuning of the Stage-1 AmBe waveform cuts (IC window and fprompt window)
from the per-waveform feature tables written by run_waveform_gated_pipeline.py.

This reads TriggerSummary/WaveformFeatures_<runinfo>_<run>.parquet ONLY -- never
the ROOT files. Everything below is a pandas query over a few MB, so a full cut
retune takes seconds instead of another multi-hour read over dCache.

Optionally (--waveform-dir) it takes one extra pass over the ROOT files to draw
mean-waveform overlays for each region of the fprompt-vs-IC plane, which is what
actually shows that the populations are noise / BGO / pile-up rather than just
differently-numbered.

Usage:
  python tune_waveform_fprompt.py --runinfo AmBe2.0v4_gated
  python tune_waveform_fprompt.py --runinfo AmBe2.0v4_gated \\
      --waveform-dir /pnfs/annie/persistent/users/dajana/AmBe/AmBe2.0v4
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # SSH session: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.data.processor import WaveformConfig     # noqa: E402  (after sys.path)


def load_features(runinfo, feature_dir='TriggerSummary'):
    """Load every per-run feature table for a runinfo tag into one DataFrame."""
    frames = []
    for ext in ('parquet', 'csv'):
        pattern = os.path.join(feature_dir, f'WaveformFeatures_{runinfo}_*.{ext}')
        for path in sorted(glob.glob(pattern)):
            match = re.search(rf'WaveformFeatures_{re.escape(runinfo)}_(\d+)\.', path)
            if not match:
                continue
            df = pd.read_parquet(path) if ext == 'parquet' else pd.read_csv(path)
            df['run'] = int(match.group(1))
            frames.append(df)

    if not frames:
        raise SystemExit(
            f"No feature tables found matching "
            f"{feature_dir}/WaveformFeatures_{runinfo}_*.parquet\n"
            f"Run run_waveform_gated_pipeline.py with the feature dump enabled first."
        )

    df = pd.concat(frames, ignore_index=True)
    df['fprompt'] = df['fprompt'].replace([np.inf, -np.inf], np.nan)
    return df


def accept_mask(df, ic_lo, ic_hi, fp_lo=None, fp_hi=None, use_veto=True):
    """Recompute the Stage-1 decision from features, mirroring passes_selection."""
    mask = (df['IC_adjusted'] > ic_lo) & (df['IC_adjusted'] < ic_hi)
    if use_veto:
        mask &= ~df['second_pulse'].astype(bool)
    if fp_lo is not None:
        mask &= df['fprompt'].notna() & (df['fprompt'] > fp_lo) & (df['fprompt'] < fp_hi)
    return mask


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_fprompt_1d(df, cfg, pdf):
    fig, ax = plt.subplots(figsize=(9, 5))
    in_ic = (df['IC_adjusted'] > cfg.pulse_gamma) & (df['IC_adjusted'] < cfg.pulse_max)
    bins = np.linspace(0, 1.0, 101)

    ax.hist(df['fprompt'].dropna(), bins=bins, histtype='step', lw=1.6,
            label=f'all waveforms (N={df["fprompt"].notna().sum()})')
    ax.hist(df.loc[in_ic, 'fprompt'].dropna(), bins=bins, histtype='stepfilled',
            alpha=0.35, label=f'inside IC window (N={(in_ic & df["fprompt"].notna()).sum()})')

    ax.axvspan(cfg.fprompt_min, cfg.fprompt_max, color='green', alpha=0.12,
               label=f'fprompt cut {cfg.fprompt_min}-{cfg.fprompt_max}')
    ax.set_xlabel('fprompt = prompt-peak integral / total-pulse integral')
    ax.set_ylabel('waveforms')
    ax.set_yscale('log')
    ax.set_title('fprompt distribution')
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_fprompt_vs_ic(df, cfg, pdf, ic_max=2500, proposed=None):
    """The money plot: the 2D plane with the current and proposed cut boxes."""
    sel = df[df['fprompt'].notna()]
    fig, ax = plt.subplots(figsize=(9, 6))

    h = ax.hist2d(sel['IC_adjusted'].clip(-100, ic_max), sel['fprompt'].clip(0, 1.0),
                  bins=[np.linspace(-100, ic_max, 130), np.linspace(0, 1.0, 100)],
                  norm=LogNorm(), cmap='viridis')
    fig.colorbar(h[3], ax=ax, label='waveforms')

    def box(lo, hi, fp_lo, fp_hi, color, label):
        ax.add_patch(plt.Rectangle((lo, fp_lo), min(hi, ic_max) - lo, fp_hi - fp_lo,
                                   fill=False, ec=color, lw=2.2, label=label))

    box(cfg.pulse_gamma, cfg.pulse_max, 0.0, 1.0, 'red',
        f'current IC window {cfg.pulse_gamma}-{cfg.pulse_max}')
    if proposed:
        box(proposed[0], proposed[1], proposed[2], proposed[3], 'white',
            f'proposed {proposed[0]}-{proposed[1]} x fprompt {proposed[2]}-{proposed[3]}')

    ax.set_xlabel('IC_adjusted')
    ax.set_ylabel('fprompt')
    ax.set_title('fprompt vs IC_adjusted — the two axes are independent')
    ax.legend(loc='upper right', framealpha=0.85)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_veto_independence(df, cfg, pdf):
    """
    fprompt for veto-passing vs veto-failing waveforms.

    If these overlay, the second-pulse veto and fprompt are catching different
    failures and both are needed -- which is what the design assumes.
    """
    in_ic = (df['IC_adjusted'] > cfg.pulse_gamma) & (df['IC_adjusted'] < cfg.pulse_max)
    sub = df[in_ic & df['fprompt'].notna()]
    veto = sub['second_pulse'].astype(bool)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.05, 0.55, 80)
    for mask, label, style in ((~veto, 'veto PASS', '-'), (veto, 'veto FAIL', '--')):
        vals = sub.loc[mask, 'fprompt']
        if not len(vals):
            continue
        ax.hist(vals, bins=bins, histtype='step', lw=1.8, ls=style, density=True,
                label=f'{label} (N={len(vals)}, median {vals.median():.3f})')
    ax.axvspan(cfg.fprompt_min, cfg.fprompt_max, color='green', alpha=0.12)
    ax.set_xlabel('fprompt')
    ax.set_ylabel('normalised')
    ax.set_title('Second-pulse veto vs fprompt (inside the IC window)\n'
                 'overlapping curves = orthogonal cuts, both needed')
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_acceptance_scans(df, cfg, pdf):
    """
    Acceptance vs each window with the other held fixed.

    The IC scan is the thing that answers the open 700-1200 vs 600-1400 question
    from AMBE_DATA_PROCESSING_HANDOFF.md: with fprompt doing the shape rejection,
    look for where the curve stops rising steeply rather than guessing a bound.
    """
    total = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lows = np.arange(200, 1001, 50)
    for ic_hi in (cfg.pulse_max, 1400, 2000, 2500):
        acc = [100 * accept_mask(df, lo, ic_hi, cfg.fprompt_min, cfg.fprompt_max).sum() / total
               for lo in lows]
        axes[0].plot(lows, acc, marker='o', ms=3, label=f'IC upper = {ic_hi}')
    axes[0].axvline(cfg.pulse_gamma, color='red', ls=':', label=f'current lower ({cfg.pulse_gamma})')
    axes[0].set_xlabel('IC lower bound')
    axes[0].set_ylabel('Stage-1 acceptance [%]')
    axes[0].set_title(f'IC window scan at fprompt {cfg.fprompt_min}-{cfg.fprompt_max}')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    fp_los = np.arange(0.00, 0.26, 0.01)
    for fp_hi in (0.25, 0.30, 0.35, 0.40):
        acc = [100 * accept_mask(df, cfg.pulse_gamma, cfg.pulse_max, lo, fp_hi).sum() / total
               for lo in fp_los]
        axes[1].plot(fp_los, acc, marker='o', ms=3, label=f'fprompt upper = {fp_hi}')
    axes[1].axvline(cfg.fprompt_min, color='red', ls=':', label=f'current lower ({cfg.fprompt_min})')
    axes[1].set_xlabel('fprompt lower bound')
    axes[1].set_ylabel('Stage-1 acceptance [%]')
    axes[1].set_title(f'fprompt scan at IC {cfg.pulse_gamma}-{cfg.pulse_max}')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_per_run_dq(df, pdf):
    """
    Per-run fprompt median as a data-quality monitor.

    A run whose gain has drifted shows up here as an fprompt outlier, where the
    IC cut would instead have quietly turned it into a low-acceptance run
    (v3 run 5740: median IC 39, fprompt 0.618, acceptance 20.9%).
    """
    stats = (df.groupby('run')['fprompt']
               .agg(median='median', p5=lambda s: s.quantile(0.05),
                    p95=lambda s: s.quantile(0.95), n='size')
               .reset_index())
    if stats.empty:
        return stats

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(stats)), 5))
    x = np.arange(len(stats))
    median = stats['median'].to_numpy()
    ax.errorbar(x, median,
                yerr=[median - stats['p5'].to_numpy(), stats['p95'].to_numpy() - median],
                fmt='o', capsize=3)
    grand = stats['median'].median()
    ax.axhline(grand, color='green', ls='--', label=f'campaign median {grand:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(stats['run'].astype(str), rotation=60, ha='right')
    ax.set_xlabel('run')
    ax.set_ylabel('fprompt (median, 5-95%)')
    ax.set_title('Per-run fprompt — outliers indicate gain/threshold drift')
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return stats


def plot_region_waveforms(df, cfg, waveform_dir, campaign, pdf, n_per_region=150):
    """
    Mean waveform for each region of the fprompt-vs-IC plane.

    Requires one pass over the ROOT files. This is the plot that shows the
    populations really are what the cut assumes: low-IC/high-fprompt should be a
    narrow spike with no tail, high-IC/low-fprompt should show extra late light.
    """
    import uproot

    regions = [
        ('low-IC / high-fprompt (noise)',   lambda d: (d['IC_adjusted'] < 250) & (d['fprompt'] > 0.30)),
        ('in-box (BGO)',                    lambda d: accept_mask(d, cfg.pulse_gamma, cfg.pulse_max,
                                                                  cfg.fprompt_min, cfg.fprompt_max)),
        ('high-IC / low-fprompt (pile-up)', lambda d: (d['IC_adjusted'] > 2000) & (d['fprompt'] < 0.15)),
    ]

    prefix = 'RWM_' if campaign == 1 else 'BRF_'
    run = int(df['run'].mode().iat[0])
    files = sorted(glob.glob(os.path.join(waveform_dir, str(run), '**', '*.root'), recursive=True))
    if not files:
        print(f'  no ROOT files under {waveform_dir}/{run}, skipping waveform overlays')
        return

    wanted = {}
    for label, fn in regions:
        ts = df.loc[fn(df) & (df['run'] == run), 'timestamp'].head(n_per_region)
        for t in ts:
            wanted[int(t)] = label
    if not wanted:
        print('  no waveforms matched the region definitions, skipping overlays')
        return

    sums, counts = {label: None for label, _ in regions}, {label: 0 for label, _ in regions}
    n_bins = cfg.fprompt_total_end + 200

    for path in files:
        try:
            with uproot.open(path) as root:
                for key in root.keys():
                    if not key.startswith(prefix):
                        continue
                    try:
                        ts = int(key.split('_')[-1].split(';')[0])
                    except ValueError:
                        continue
                    label = wanted.pop(ts, None)
                    if label is None:
                        continue
                    values = root[key].values()
                    if len(values) < n_bins:
                        continue
                    values = values[:n_bins] - np.median(values[:cfg.fprompt_baseline_end])
                    sums[label] = values if sums[label] is None else sums[label] + values
                    counts[label] += 1
        except Exception as e:                      # noqa: BLE001 - dCache reads fail in many ways
            print(f'  could not read {path}: {e}')
        if not wanted:
            break

    # Left: absolute amplitude. Right: each curve scaled to its own peak, which
    # is what fprompt actually measures -- the pile-up curve otherwise dominates
    # the y-scale and flattens the noise curve into the axis.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, normalise in zip(axes, (False, True)):
        for label, _ in regions:
            if not counts[label]:
                continue
            mean_wf = sums[label] / counts[label]
            if normalise:
                peak = mean_wf.max()
                if peak <= 0:
                    continue
                mean_wf = mean_wf / peak
            ax.plot(mean_wf, lw=1.4, label=f'{label}  (N={counts[label]})')
        ax.axvspan(cfg.fprompt_prompt_start, cfg.fprompt_prompt_end, color='blue', alpha=0.18,
                   label='fprompt numerator')
        ax.axvspan(cfg.fprompt_prompt_end, cfg.fprompt_total_end, color='orange', alpha=0.10,
                   label='rest of denominator')
        ax.set_xlim(cfg.fprompt_baseline_end, n_bins)
        ax.set_xlabel('ADC sample (bin index; 2 ns/sample)')
        ax.set_ylabel('peak-normalised' if normalise else 'baseline-subtracted ADC')
        ax.set_title('shape only (peak-normalised)' if normalise else 'absolute amplitude')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Mean waveform per region of the fprompt-IC plane (run {run})')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runinfo', required=True, help='Tag used when the features were written')
    p.add_argument('--feature-dir', default='TriggerSummary')
    p.add_argument('--output', default=None, help='Output PDF (default: verbose/fprompt_tuning_<tag>.pdf)')
    p.add_argument('--pulse-gamma', type=int, default=None, help='IC lower bound to draw as "current"')
    p.add_argument('--pulse-max', type=int, default=None, help='IC upper bound to draw as "current"')
    p.add_argument('--fprompt-min', type=float, default=None)
    p.add_argument('--fprompt-max', type=float, default=None)
    p.add_argument('--proposed', nargs=4, type=float, metavar=('IC_LO', 'IC_HI', 'FP_LO', 'FP_HI'),
                   default=None, help='Overlay a proposed 2D box on the main plot')
    p.add_argument('--waveform-dir', default=None,
                   help='Dataset dir with <run>/*.root — enables the mean-waveform overlays '
                        '(one extra ROOT pass)')
    p.add_argument('--campaign', type=int, default=1, choices=[1, 2],
                   help='1 = RWM_ channel (v1, v4), 2 = BRF_ channel (v3)')
    args = p.parse_args()

    cfg = WaveformConfig()
    for attr in ('pulse_gamma', 'pulse_max', 'fprompt_min', 'fprompt_max'):
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)

    df = load_features(args.runinfo, args.feature_dir)
    output = args.output or f'verbose/fprompt_tuning_{args.runinfo}.pdf'
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    n_undef = df['fprompt'].isna().sum()
    print(f'Loaded {len(df)} waveforms from {df["run"].nunique()} runs '
          f'({n_undef} with undefined fprompt / low light)')

    # Text summary of the 2D plane -- the same table the design was based on.
    print('\nfprompt vs IC_adjusted:')
    edges = [0, 250, 500, cfg.pulse_gamma, cfg.pulse_max, 1400, 2000, np.inf]
    edges = sorted(set(e for e in edges if e >= 0))
    header = f'{"IC_adjusted":>20s}{"fp<min":>10s}{"in band":>10s}{"fp>max":>10s}{"undef":>10s}{"total":>10s}'
    print(header)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df['IC_adjusted'] >= lo) & (df['IC_adjusted'] < hi)
        fp = df['fprompt']
        print(f'{f"[{lo:g}, {hi:g})":>20s}'
              f'{(m & (fp < cfg.fprompt_min)).sum():>10d}'
              f'{(m & (fp >= cfg.fprompt_min) & (fp <= cfg.fprompt_max)).sum():>10d}'
              f'{(m & (fp > cfg.fprompt_max)).sum():>10d}'
              f'{(m & fp.isna()).sum():>10d}'
              f'{m.sum():>10d}')

    # The design assumes the second-pulse veto and fprompt catch different
    # failures, so both are kept. Print the check rather than trusting it.
    in_ic = (df['IC_adjusted'] > cfg.pulse_gamma) & (df['IC_adjusted'] < cfg.pulse_max)
    sub = df[in_ic & df['fprompt'].notna()]
    veto = sub['second_pulse'].astype(bool)
    if veto.any() and (~veto).any():
        m_pass, m_fail = sub.loc[~veto, 'fprompt'].median(), sub.loc[veto, 'fprompt'].median()
        print(f'\nVeto/fprompt independence (inside IC window): '
              f'median fprompt {m_pass:.3f} (veto pass) vs {m_fail:.3f} (veto fail), '
              f'delta {abs(m_pass - m_fail):.3f} -- '
              f'{"orthogonal, keep both cuts" if abs(m_pass - m_fail) < 0.01 else "NOT orthogonal, revisit"}')

    ic_only = accept_mask(df, cfg.pulse_gamma, cfg.pulse_max)
    ic_fp = accept_mask(df, cfg.pulse_gamma, cfg.pulse_max, cfg.fprompt_min, cfg.fprompt_max)
    print(f'\nAcceptance, IC only          : {ic_only.sum():7d}  ({100*ic_only.mean():.2f}%)')
    print(f'Acceptance, IC + fprompt     : {ic_fp.sum():7d}  ({100*ic_fp.mean():.2f}%)')
    if args.proposed:
        prop = accept_mask(df, *args.proposed)
        print(f'Acceptance, proposed 2D box  : {prop.sum():7d}  ({100*prop.mean():.2f}%)  '
              f'[{100*(prop.sum()/max(ic_only.sum(), 1) - 1):+.1f}% vs IC only]')

    with PdfPages(output) as pdf:
        plot_fprompt_1d(df, cfg, pdf)
        plot_fprompt_vs_ic(df, cfg, pdf, proposed=args.proposed)
        plot_veto_independence(df, cfg, pdf)
        plot_acceptance_scans(df, cfg, pdf)
        stats = plot_per_run_dq(df, pdf)
        if args.waveform_dir:
            plot_region_waveforms(df, cfg, args.waveform_dir, args.campaign, pdf)

    if stats is not None and not stats.empty:
        print('\nPer-run fprompt median:')
        print(stats.to_string(index=False))

    print(f'\nWrote {output}')


if __name__ == '__main__':
    main()
