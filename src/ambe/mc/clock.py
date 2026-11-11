"""MC time frame, opt-in config block ``mc_clock``.

WCSim truth times start at the neutron emission. An AmBe source-trigger data window starts
about 1,074 ns earlier, so data hit and cluster times sit ~1,074 ns later than emission-clock MC.

    mc_clock:
      sample_time_shift_ns: 0       # PMTWaveformSim TimeShift of the sample (wf_nosmear: 0, wf_dtho dataclock: 1074)
      analysis_time_shift_ns: 1074  # clock of the analysis: 0 = emission clock, 1074 = AmBe data clock

`ambe mc process` adds analysis - sample to every MC hit, DirectParent and cluster time, so the
pulse and cluster parquets are in the analysis clock. Truth (captures) stays in the emission clock;
a configured capture_truth.offset_ns gets + analysis (calibrate_offset measures it directly).
Without the block both shifts are 0. Modules that read hit or cluster times straight from the
ANNIEEvent ROOT files stay in the sample clock.
"""
from __future__ import annotations

AMBE_DATA_CLOCK_NS = 1074.0


def shifts(extra: dict) -> tuple[float, float]:
    """(shift added to MC reco times, analysis clock relative to the emission) in ns."""
    c = (extra or {}).get("mc_clock") or {}
    sample = float(c.get("sample_time_shift_ns", 0.0))
    analysis = float(c.get("analysis_time_shift_ns", sample))
    return analysis - sample, analysis


def truth_offset_ns(cfg_offset_ns: float, extra: dict) -> float:
    """A configured hit-time minus capture-time offset, moved into the analysis clock."""
    return float(cfg_offset_ns) + shifts(extra)[1]
