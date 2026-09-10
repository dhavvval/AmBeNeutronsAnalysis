"""
The neutron cluster definition, as a swappable object.

WHY THIS MODULE EXISTS. The AmBe analysis is run twice over the same data, changing
only what counts as a neutron cluster:

    box   0 < clusterPE <= 100,  0 < chargeBalance < 0.45,
          clusterTime >= 2000 ns,  clusterHits >= 5
    mva   gbt_score > threshold, and NO PE / CB / time / hits box at all

Before this module the box was written out twice, in two files that had to agree and
had no mechanism forcing them to:

    data/processor.py:165-182   ambe_single_cut / ambe_multiple_cut / cosmic_cut
    data/eff.py:15-48           AmBe() / AmBeMultiple() / cosmic()

Both now delegate here, so there is one definition of the box and one place the MVA
alternative had to be added.

THE COSMIC VETO IS DELIBERATELY NOT SWAPPABLE. `cosmic()` lives on the base class and
neither subclass overrides it. The veto (any cluster with clusterTime < 2 us or
clusterPE > 100 drops the whole event) sits UPSTREAM of the neutron definition, in
Stage 1, alongside the IC gate. If it differed between the two selections the two
analyses would no longer share a denominator and their efficiencies would not be
comparable -- which is the entire point of running both.

WHAT `cn` MEANS, AND WHY IT IS THE SAME IN BOTH. `cn` is the event's TOTAL cluster
count (`numberOfClusters`), not the number of accepted clusters. So `single` means
"this event has exactly one cluster, and it is a neutron" for both selections alike.
Keeping that identical is what lets the single/multiple split be compared across the
two definitions; the multiplicity of *accepted* clusters is a separate quantity and is
computed downstream from the candidate lists.

Usage:
    from .selection import get_selection
    sel = get_selection("box")                      # or "mva", with mva_* kwargs
    if sel.cosmic(ct, cpe): ...
    if sel.single(cpe, ccb, ct, cn, chits, key): ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# (run, eventTankTime, clusterTime in ns rounded to 3 dp). The same key
# boxcut_v4_campaign.do_matched and boxcut_v4_appendix._scored already join on, so a
# cluster identified here is the same cluster identified there.
ClusterKey = Tuple[int, int, float]


@dataclass
class CutCriteria:
    """Box-selection thresholds.

    Defined here rather than in processor.py because the selection objects are the
    things that consume it. `from ambe.data.processor import CutCriteria` still works
    -- processor.py re-exports it -- since reprocess_ambe_v1.py, boxcut_v4_special.py
    and run_waveform_gated_pipeline.py all import it from there.
    """
    pe_min: float = 0
    pe_max: float = 100
    ccb_min: float = 0
    ccb_max: float = 0.45
    ct_min: float = 2000
    chits_min: int = 5
    cosmic_ct_threshold: float = 2000
    cosmic_pe_threshold: float = 100


def cluster_key(run, event_tank_time, cluster_time_ns) -> ClusterKey:
    """Build the join key. Kept in one place so both sides round identically."""
    return (int(run), int(event_tank_time), round(float(cluster_time_ns), 3))


class Selection:
    """Base class: owns the cosmic veto, which is common to every selection."""

    name = "base"
    label = "base"

    def __init__(self, cuts: Optional[CutCriteria] = None):
        self.cuts = cuts or CutCriteria()

    def cosmic(self, ct: float, cpe: float) -> bool:
        """Prompt/high-charge cluster => veto the whole event. NOT overridable."""
        return (ct < self.cuts.cosmic_ct_threshold or
                cpe > self.cuts.cosmic_pe_threshold)

    def accepts(self, cpe, ccb, ct, chits, key) -> bool:
        raise NotImplementedError

    def single(self, cpe, ccb, ct, cn, chits, key=None) -> bool:
        return cn == 1 and self.accepts(cpe, ccb, ct, chits, key)

    def multiple(self, cpe, ccb, ct, cn, chits, key=None) -> bool:
        return cn != 1 and self.accepts(cpe, ccb, ct, chits, key)


class BoxSelection(Selection):
    """The traditional AmBe box. Reproduces the published 57.23 % / 30.535 us."""

    name = "box"

    def __init__(self, cuts: Optional[CutCriteria] = None):
        super().__init__(cuts)
        c = self.cuts
        self.label = (f"box cuts PE ≤ {c.pe_max:g}, CB < {c.ccb_max:g}, "
                      f"t ≥ {c.ct_min/1000:g} µs, hits ≥ {c.chits_min:g}")

    def accepts(self, cpe, ccb, ct, chits, key=None) -> bool:
        c = self.cuts
        return (c.pe_min < cpe <= c.pe_max and
                c.ccb_min < ccb < c.ccb_max and
                ct >= c.ct_min and chits >= c.chits_min)


class MvaSelection(Selection):
    """MVA neutron: score above threshold, and no box of any kind.

    This is what makes the second analysis independent rather than a tightening of
    the first. Clusters that FAIL the box (too much light, poor charge balance, too
    few hits) are eligible here, so the two selections are genuinely different
    samples rather than one nested inside the other.

    Membership is precomputed into a set rather than tested arithmetically because
    the score lives in a scored parquet keyed by cluster, not in the BeamCluster
    quantities the event loop has to hand.
    """

    name = "mva"

    def __init__(self, scored: Sequence[Path] | Path, threshold: float,
                 score_col: str = "gbt_score", point: str = "eff80",
                 cuts: Optional[CutCriteria] = None):
        super().__init__(cuts)
        self.threshold = float(threshold)
        self.score_col = score_col
        self.label = f"MVA neutron ({score_col.split('_')[0].upper()} {point})"
        paths = [scored] if isinstance(scored, (str, Path)) else list(scored)
        # `scored_keys` is every cluster the scoring stage saw, at any score;
        # `accepted` is the subset above threshold. Keeping both is what lets
        # "no score" be told apart from "scored and rejected".
        self.accepted, self.scored_keys, self.n_scored = self._load(paths)
        self.n_queried = 0
        self.n_unscored = 0
        self.unscored_runs = {}

    def _load(self, paths: Iterable[Path]):
        cols = ["run", "event_tank_time", "cf_clusterTime", self.score_col]
        frames = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                raise SystemExit(f"[selection] missing scored parquet: {p}")
            frames.append(pd.read_parquet(p, columns=cols))
        d = pd.concat(frames, ignore_index=True)
        n_scored = len(d)

        def keyset(t):
            return set(zip(t.run.astype("int64").tolist(),
                           t.event_tank_time.astype("int64").tolist(),
                           np.round(t.cf_clusterTime.to_numpy(float), 3).tolist()))

        all_keys = keyset(d)
        keep = d[d[self.score_col] > self.threshold]
        accepted = keyset(keep)
        print(f"[selection] mva: {len(keep):,} of {n_scored:,} scored clusters above "
              f"{self.score_col} > {self.threshold:.6f} "
              f"({100*len(keep)/max(n_scored,1):.2f} %)")
        print(f"[selection] mva: scored sample covers {len(all_keys):,} distinct "
              f"clusters over runs {sorted(d.run.unique().tolist())}")
        return accepted, all_keys, n_scored

    def accepts(self, cpe, ccb, ct, chits, key=None) -> bool:
        if key is None:
            raise ValueError(
                "MvaSelection needs a cluster key -- the caller must pass "
                "cluster_key(run, eventTankTime, clusterTime_ns). A box-style call "
                "with key=None would silently accept everything.")
        self.n_queried += 1
        if key in self.accepted:
            return True
        # CRITICAL BOOKKEEPING. A cluster this pipeline finds but the scoring stage
        # never saw has NO score, and returning False for it is indistinguishable
        # from "scored and below threshold". Those are different statements: one is
        # a rejection, the other is missing information, and silently merging them
        # depresses the MVA's apparent efficiency by the extraction's dropout rate.
        # Measured on the 28-run campaign: 11,780 of 237,590 box candidates (4.96 %)
        # are unscored, which is the documented -4.6 % Stage-2 closure gap of the
        # MVA extraction. It moves the keep rate from 79.01 % of SCORED clusters
        # (the published number) to 75.09 % of ALL clusters.
        # So count the misses and make the caller report them.
        if key not in self.scored_keys:
            self.n_unscored += 1
            self.unscored_runs[key[0]] = self.unscored_runs.get(key[0], 0) + 1
        return False

    def coverage(self) -> dict:
        """How much of what we were asked about the scored sample actually covers."""
        q = max(self.n_queried, 1)
        return dict(queried=self.n_queried, unscored=self.n_unscored,
                    covered_frac=1.0 - self.n_unscored / q,
                    per_run=dict(sorted(self.unscored_runs.items())))

    def report_coverage(self, min_covered=0.90):
        """Print coverage and say plainly whether the result is quotable."""
        c = self.coverage()
        print(f"[selection] mva coverage: {c['queried'] - c['unscored']:,} of "
              f"{c['queried']:,} clusters had a score "
              f"({100*c['covered_frac']:.2f} %)")
        if c["unscored"]:
            print(f"[selection] {c['unscored']:,} clusters had NO score and were "
                  f"counted as rejected. Quote the keep rate over SCORED clusters, "
                  f"not over all clusters, or this dropout enters your efficiency.")
            worst = sorted(c["per_run"].items(), key=lambda kv: -kv[1])[:5]
            print(f"[selection] worst runs: "
                  + ", ".join(f"{r}:{n:,}" for r, n in worst))
        if c["covered_frac"] < min_covered:
            print(f"[selection] *** COVERAGE {100*c['covered_frac']:.1f} % IS BELOW "
                  f"{100*min_covered:.0f} % -- this selection is NOT usable as an "
                  f"efficiency. The scoring stage was run on a different (smaller) "
                  f"Stage-1 sample than this one. Re-extract and re-score against "
                  f"THIS tag before quoting anything. ***")
        return c


def get_selection(name: str, cuts: Optional[CutCriteria] = None, **kw) -> Selection:
    """Factory. `name` is required by every caller -- there is deliberately no
    default, because defaulting to one neutron definition is how you end up with a
    deck labelled as the other one."""
    if name == "box":
        return BoxSelection(cuts)
    if name == "mva":
        missing = {"scored", "threshold"} - set(kw)
        if missing:
            raise SystemExit(f"[selection] mva needs {sorted(missing)}")
        return MvaSelection(cuts=cuts, **kw)
    raise SystemExit(f"[selection] unknown selection {name!r}; use 'box' or 'mva'")
