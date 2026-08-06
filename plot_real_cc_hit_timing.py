#!/usr/bin/env python3
"""
plot_real_cc_hit_timing.py

Real (not illustrative) hit-timing structure for one actual CC-passing event:
the prompt muon-burst window (0-2 us) and the delayed neutron-capture window
(2-70 us), drawn directly from that event's raw hits -- no stacking, no
analytic curve, no synthetic jitter.

Event chosen (grounded, 2026-07-21, from
cc_neutrino_xi02_compcut__pulses.parquet, cc_pass==True population,
n=29223 events):
    eventID 685824 (ANNIEEvent_cc_neutrino_49.root)
    prompt_n=118 (population median 121)
    delayed_n=21, in TWO separated clusters: ~5875-5893 ns (9 hits) and
    ~11030-11044 ns (12 hits) -- 14/21 delayed hits truth-tagged is_neutron.
Picked over a single-tight-burst event (originally eventID 74362) because a
population check found 47% of CC-passing events with >=2 delayed hits have
them spread over >5 us (scattered/multi-cluster), not one tight burst --
74362's single ~13 ns cluster was atypically clean. This event shows the more
common two-cluster/residual-hit structure while keeping a population-typical
prompt-hit count.

Usage (myboy venv):
    python plot_real_cc_hit_timing.py [--event-id ID] [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambe.plotting import set_style  # noqa: E402

PULSES_PARQUET = (
    "/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/"
    "cc_neutrino_xi02_compcut/parquet/cc_neutrino_xi02_compcut__pulses.parquet"
)
DEFAULT_EVENT_ID = 685824
PROMPT_WINDOW_END_NS = 2000.0
DELAYED_WINDOW_END_NS = 70000.0

MUON_COLOR = "blue"
DELAYED_COLOR = "red"
PROMPT_SHADE = "0.92"


def load_event_hits(event_id: int):
    tab = pq.read_table(
        PULSES_PARQUET,
        columns=["eventID", "t", "pe", "cc_pass", "is_neutron", "_source_file"],
        filters=[("eventID", "=", event_id)],
    )
    df = tab.to_pandas()
    df = df[df["cc_pass"]].sort_values("t")
    if df.empty:
        raise SystemExit(f"event {event_id}: no cc_pass hits found")
    return df


def make_plot(event_id: int, out_dir: Path):
    df = load_event_hits(event_id)

    prompt = df[df["t"] <= PROMPT_WINDOW_END_NS]
    delayed = df[df["t"] > PROMPT_WINDOW_END_NS]

    set_style()
    plt.rcParams.update({"axes.grid": False})

    prompt_end_us = PROMPT_WINDOW_END_NS / 1000.0
    delayed_end_us = DELAYED_WINDOW_END_NS / 1000.0

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.axvspan(0, prompt_end_us, color=PROMPT_SHADE, lw=0, zorder=0)
    ax.axvline(prompt_end_us, color="black", ls="--", lw=1.0, zorder=1)

    ax.vlines(prompt["t"] / 1000.0, 0, prompt["pe"], color=MUON_COLOR, lw=1.2, label="Prompt (muon) hits")
    ax.vlines(delayed["t"] / 1000.0, 0, delayed["pe"], color=DELAYED_COLOR, lw=1.2, label="Delayed hits")

    ax.set_yscale("log")
    ax.set_xlim(0, delayed_end_us)
    ax.set_ylim(0.3, 400)

    ax.text(prompt_end_us / 2, 300, "Prompt window", ha="center", va="top", fontsize=10)
    ax.text((prompt_end_us + delayed_end_us) / 2, 300, "Delayed window (2–70 μs)",
            ha="center", va="top", fontsize=10)

    ax.set_xlabel("Time since interaction trigger (μs)")
    ax.set_ylabel("Total charge")
    ax.set_title("CC inclusive event charge distribution")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "cc_hit_timing_real_event"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[out] wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--event-id", type=int, default=DEFAULT_EVENT_ID)
    p.add_argument(
        "--out-dir",
        default="/exp/annie/app/users/dajana/AmBeNeutronAnalysis/ambe_output/cc_neutrino_xi02_compcut/plots",
    )
    args = p.parse_args()
    make_plot(args.event_id, Path(args.out_dir))
