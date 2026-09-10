#!/usr/bin/env python3
"""
plot_illustrative_cc_hit_timing.py

Illustrative (schematic) hit-timing structure for a CC-passing event: a sharp
prompt muon burst (0-2 us) followed by the delayed neutron-capture-on-Gd
window (2-70 us). NOT drawn from one real event's raw hits -- a single real
CC-passing event only has ~17 delayed hits total (see grounding below), far
too sparse to render as a clean decay curve. Instead this is a deterministic,
stacked/summed illustration, explicitly labelled as such.

Grounding (2026-07-21, from cc_neutrino_xi02_compcut__cluster_features.parquet,
optics method, cc_pass & is_truth_neutron & not is_prompt_cluster):
    mean hits per neutron-capture cluster : 17.11  (used range in prompt: ~15-20)
    mean captures per event (>=1 capture) : 1.27   (simplified here to 1/event)
    cluster t_mean range                  : 2.0 - 66.5 us
Capture-time constant (30 us) taken as given/confirmed MC-truth value per
instructions, not re-fit here.

Usage (myboy venv):
    python plot_illustrative_cc_hit_timing.py [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.plotting import set_style  # noqa: E402

# --- house palette ------------------------------------------------------ #
MID_BLUE = "#2E5F8A"
DARK_BLUE = "#1A3752"
PROMPT_SHADE = "#ECECEC"
DELAYED_SHADE = "#DCE7F0"

# --- physics / grounding constants --------------------------------------- #
TAU_CAPTURE_US = 30.0          # confirmed MC-truth mean capture time
PROMPT_WINDOW_END_US = 2.0
DELAYED_WINDOW_END_US = 70.0
HITS_PER_CAPTURE = 17.11        # grounded: mean n_hits, truth-neutron delayed clusters
N_STACK = 40                     # illustrative number of stacked CC-passing captures
BKG_RATE_PER_US = 3.0            # illustrative flat background (not fit from data)
PROMPT_TOTAL_HITS = 750          # target total muon-burst hits (spec: ~500-1000)
PROMPT_GEOM_RATIO = 0.16         # bin-to-bin falloff -> concentrates burst << 1 us

RNG_SEED = 42
JITTER_FRAC = 0.03               # small seeded jitter for visual naturalism only


def build_histogram():
    rng = np.random.default_rng(RNG_SEED)

    # Prompt window: 0.05 us bins, geometric-decay burst (deterministic shape).
    prompt_bin_w = 0.05
    prompt_edges = np.arange(0.0, PROMPT_WINDOW_END_US + 1e-9, prompt_bin_w)
    n_prompt_bins = len(prompt_edges) - 1
    prompt_heights = PROMPT_TOTAL_HITS * (1 - PROMPT_GEOM_RATIO) * PROMPT_GEOM_RATIO ** np.arange(n_prompt_bins)

    # Delayed window: 1 us bins, analytic dN/dt = A*exp(-t/tau) + flat bkg.
    delayed_bin_w = 1.0
    delayed_edges = np.arange(PROMPT_WINDOW_END_US, DELAYED_WINDOW_END_US + 1e-9, delayed_bin_w)
    centers = 0.5 * (delayed_edges[:-1] + delayed_edges[1:])
    n_delayed_bins = len(centers)

    total_delayed = N_STACK * HITS_PER_CAPTURE
    bkg_total = BKG_RATE_PER_US * n_delayed_bins * delayed_bin_w
    exp_total = max(total_delayed - bkg_total, 0.0)
    exp_integral_unit = TAU_CAPTURE_US * (np.exp(-PROMPT_WINDOW_END_US / TAU_CAPTURE_US)
                                           - np.exp(-DELAYED_WINDOW_END_US / TAU_CAPTURE_US))
    amp = exp_total / exp_integral_unit
    delayed_heights = amp * np.exp(-centers / TAU_CAPTURE_US) * delayed_bin_w + BKG_RATE_PER_US

    print(f"[grounding] hits/capture={HITS_PER_CAPTURE}  N_STACK={N_STACK}  "
          f"-> target delayed total={total_delayed:.1f}")
    print(f"[delayed]   bin range: {delayed_heights.min():.1f} - {delayed_heights.max():.1f} hits/bin "
          f"(spec: ~5-20)")
    print(f"[prompt]    peak bin={prompt_heights.max():.1f}  total={prompt_heights.sum():.1f} "
          f"(spec: peak ~600-900, total ~500-1000)")

    # small seeded jitter, visual naturalism only -- shape/decay is unaffected
    prompt_heights = prompt_heights * (1 + rng.normal(0, JITTER_FRAC, n_prompt_bins))
    delayed_heights = delayed_heights * (1 + rng.normal(0, JITTER_FRAC, n_delayed_bins))
    prompt_heights = np.clip(prompt_heights, 0, None)
    delayed_heights = np.clip(delayed_heights, 0, None)

    return (prompt_edges, prompt_heights), (delayed_edges, delayed_heights)


def make_plot(out_dir: Path):
    (prompt_edges, prompt_heights), (delayed_edges, delayed_heights) = build_histogram()

    set_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.grid": False,
    })

    fig, (ax_hi, ax_lo) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 6.2),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.06},
    )

    peak = prompt_heights.max()
    ax_hi.set_ylim(peak * 0.78, peak * 1.12)
    ax_lo.set_ylim(0, 27)

    for ax in (ax_hi, ax_lo):
        ax.axvspan(0, PROMPT_WINDOW_END_US, color=PROMPT_SHADE, lw=0, zorder=0)
        ax.axvspan(PROMPT_WINDOW_END_US, DELAYED_WINDOW_END_US, color=DELAYED_SHADE, lw=0, zorder=0)
        ax.axvline(PROMPT_WINDOW_END_US, color=DARK_BLUE, ls="--", lw=1.2, zorder=1)
        ax.bar(prompt_edges[:-1], prompt_heights, width=np.diff(prompt_edges),
               align="edge", color=DARK_BLUE, edgecolor="none", zorder=2,
               label="Prompt (muon) hits")
        ax.bar(delayed_edges[:-1], delayed_heights, width=np.diff(delayed_edges),
               align="edge", color=MID_BLUE, edgecolor="none", zorder=2,
               label="Delayed hits (Gd capture + background)")
        ax.grid(False)

    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)
    ax_lo.xaxis.tick_bottom()

    d = 0.012
    kwargs = dict(transform=ax_hi.transAxes, color="k", clip_on=False, lw=1)
    ax_hi.plot((-d, +d), (-d, +d), **kwargs)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_lo.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # window labels, placed in empty space (not on top of bars)
    ax_lo.text(1.05, 22.5, "Prompt\nwindow", ha="center", va="top",
               fontsize=9, color="dimgray")
    ax_hi.text(36, (ax_hi.get_ylim()[0] + ax_hi.get_ylim()[1]) / 2,
               "Delayed window (2–70 μs)", ha="center", va="center",
               fontsize=9, color="dimgray")

    # capture-time annotation, away from the top-panel legend
    t_ann = 30.0
    y_at_30 = np.interp(t_ann, 0.5 * (delayed_edges[:-1] + delayed_edges[1:]), delayed_heights)
    ax_lo.annotate(
        r"$\langle\tau_{capture}\rangle \approx 30\ \mu s$" + "\n(MC truth)",
        xy=(t_ann, y_at_30), xytext=(46, y_at_30 + 9),
        arrowprops=dict(arrowstyle="->", color=DARK_BLUE, lw=1.1),
        fontsize=9, ha="left", color=DARK_BLUE,
    )

    ax_hi.legend(loc="upper right", frameon=False, fontsize=8.5)

    ax_lo.set_xlabel("Time since interaction trigger (μs)")
    fig.supylabel(f"Hits / bin  (stacked over N={N_STACK} CC-passing captures)", fontsize=10)

    fig.suptitle("Muon Prompt Burst vs. Delayed Neutron-Capture Hits", fontsize=13, y=0.985)
    fig.text(0.5, 0.945,
             "Illustrative sketch — not measured data (stacked/summed, not a single raw event)",
             ha="center", fontsize=8.5, style="italic", color="dimgray")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "cc_hit_timing_illustrative"
    fig.savefig(stem.with_suffix(".pdf"), dpi=160)
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"[out] wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default="/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/cc_neutrino_xi02_compcut/plots",
    )
    args = p.parse_args()
    make_plot(Path(args.out_dir))
