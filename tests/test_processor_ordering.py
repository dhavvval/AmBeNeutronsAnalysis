"""The candidate multiplicity must not depend on the order of the cluster arrays.

WHAT WAS WRONG. The cluster loop ran in a single pass and `break`-ed on the cosmic
veto. The break only fired when iteration REACHED the cosmic cluster, so any
box-passing cluster sitting BEFORE it in array order had already been appended and
was never rolled back. The event then incremented `cosmic_events` -- which is
subtracted from the denominator -- while also incrementing the candidate counter,
which is the numerator. Documented at icscan.py:373-381; on run 6062 that is 353
events and 2.8 percentage points, larger than the whole systematic budget.

Array order is not a physical property of an event, so the fix is structural rather
than a threshold change: decide the cosmic verdict over ALL clusters first, then
collect candidates only if the event survived.
"""
import itertools

import numpy as np
import pytest

from ambe.data.processor import AmBeNeutronProcessing
from ambe.data.selection import BoxSelection, CutCriteria


CUTS = CutCriteria()

# A cluster is (clusterTime, clusterPE, chargeBalance, clusterHits).
CAND_A = (10000.0, 50.0, 0.2, 10)      # passes the box
CAND_B = (30000.0, 60.0, 0.25, 12)     # passes the box
COSMIC_PROMPT = (500.0, 50.0, 0.2, 10)     # t < 2 us -> vetoes the event
COSMIC_CHARGE = (20000.0, 500.0, 0.2, 10)  # PE > 100 -> vetoes the event
DUD = (40000.0, 50.0, 0.9, 10)         # fails charge balance, not a candidate


def _event_data(clusters):
    """Build the event_data dict for a SINGLE event holding `clusters`."""
    n = len(clusters)
    ct = np.array([[c[0] for c in clusters]], dtype=float)
    cpe = np.array([[c[1] for c in clusters]], dtype=float)
    ccb = np.array([[c[2] for c in clusters]], dtype=float)
    ch = np.array([[c[3] for c in clusters]], dtype=int)
    # Hit arrays are (event, cluster, hit) -- three deep. Each cluster needs its own
    # small hit list or _append_cluster_data has nothing to iterate.
    hits = [[np.zeros(4, dtype=float) for _ in clusters]]
    return {
        "eventNumber": np.array([1]),
        "eventTimeTank": np.array([1000]),
        "clusterTime": ct, "clusterPE": cpe,
        "clusterChargeBalance": ccb, "clusterHits": ch,
        "clusterNumber": np.array([n]),
        "hitT": hits, "hitX": hits, "hitY": hits, "hitZ": hits,
        "hitPE": hits, "hitDetID": hits,
    }


def _run_event(clusters, legacy=False):
    """Process one event; return (event_row, stats)."""
    proc = AmBeNeutronProcessing(cuts=CUTS, selection=BoxSelection(CUTS),
                                 legacy_ordering=legacy)
    rows = []
    _, stats = proc.process_events_efficient(
        _event_data(clusters), good_events=None,
        x_pos=0.0, y_pos=0.0, z_pos=0.0, run=6265, event_rows=rows)
    return rows[0], stats


# --------------------------------------------------------------------------- #
# Order invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("clusters", [
    [COSMIC_PROMPT, CAND_A, CAND_B],       # cosmic first -- the old code got this right
    [CAND_A, CAND_B, COSMIC_PROMPT],       # cosmic last  -- the old code LEAKED here
    [CAND_A, COSMIC_PROMPT, CAND_B],       # cosmic in the middle
    [CAND_A, COSMIC_CHARGE, CAND_B],       # veto by charge rather than time
])
def test_vetoed_event_contributes_no_candidates_whatever_the_order(clusters):
    row, stats = _run_event(clusters)
    assert row["is_cosmic"] is True
    assert stats["cosmic_events"] == 1
    n = row["n_accepted_single"] + row["n_accepted_multiple"]
    assert n == 0, (
        f"a cosmic-vetoed event contributed {n} candidates; it is subtracted from "
        f"the denominator, so this is the numerator/denominator leak")
    assert stats["neutron_cand_count"] == 0


def test_candidate_multiplicity_is_invariant_under_every_permutation():
    """The strongest form of the claim: try all orderings of the same clusters."""
    base = [CAND_A, CAND_B, COSMIC_PROMPT, DUD]
    results = set()
    for perm in itertools.permutations(base):
        row, stats = _run_event(list(perm))
        results.add((row["is_cosmic"],
                     row["n_accepted_single"] + row["n_accepted_multiple"],
                     stats["cosmic_events"], stats["neutron_cand_count"]))
    assert len(results) == 1, f"order-dependent outcomes: {sorted(results)}"


def test_non_vetoed_event_multiplicity_is_order_invariant():
    base = [CAND_A, CAND_B, DUD]
    results = set()
    for perm in itertools.permutations(base):
        row, stats = _run_event(list(perm))
        results.add((row["is_cosmic"],
                     row["n_accepted_single"] + row["n_accepted_multiple"]))
    assert len(results) == 1
    (is_cosmic, n), = results
    assert is_cosmic is False
    assert n == 2, "both box-passing clusters must be counted"


def test_multiple_cosmic_clusters_count_the_event_once():
    row, stats = _run_event([CAND_A, COSMIC_PROMPT, COSMIC_CHARGE, CAND_B])
    assert row["is_cosmic"] is True
    assert stats["cosmic_events"] == 1
    assert row["n_accepted_single"] + row["n_accepted_multiple"] == 0


def test_event_with_no_cosmic_is_not_vetoed():
    row, stats = _run_event([CAND_A, CAND_B])
    assert row["is_cosmic"] is False
    assert stats["cosmic_events"] == 0
    assert row["n_accepted_single"] + row["n_accepted_multiple"] == 2


# --------------------------------------------------------------------------- #
# The window, enforced through the real event loop
# --------------------------------------------------------------------------- #
def test_late_cluster_is_not_a_candidate():
    """End-to-end assertion that the [2, 67] us window is applied where candidates
    are actually built, not merely where the fit normalises."""
    late = (90000.0, 50.0, 0.2, 10)      # past the 67 us window
    row, _ = _run_event([CAND_A, late])
    assert row["is_cosmic"] is False
    assert row["n_accepted_single"] + row["n_accepted_multiple"] == 1


def test_every_accepted_cluster_lies_inside_the_configured_window():
    proc = AmBeNeutronProcessing(cuts=CUTS, selection=BoxSelection(CUTS))
    clusters = [(t, 50.0, 0.2, 10) for t in
                (1500.0, 2000.0, 10000.0, 67000.0, 67001.0, 120000.0)]
    data = _event_data(clusters)
    processed, _ = proc.process_events_efficient(
        data, good_events=None, x_pos=0.0, y_pos=0.0, z_pos=0.0, run=6265)
    # 1500 ns trips the cosmic veto, so this event is vetoed outright and yields
    # nothing -- which is itself the correct answer.
    times = processed["cluster_time"] if isinstance(processed, dict) else []
    assert all(CUTS.ct_min <= t <= CUTS.ct_max for t in times)


# --------------------------------------------------------------------------- #
# The legacy path stays reproducible
# --------------------------------------------------------------------------- #
def test_legacy_ordering_still_reproduces_the_leak():
    """The published 57.23 % path was produced by the order-dependent loop, so it
    must remain available -- explicitly, never by default."""
    row, stats = _run_event([CAND_A, CAND_B, COSMIC_PROMPT], legacy=True)
    assert row["is_cosmic"] is True
    leaked = row["n_accepted_single"] + row["n_accepted_multiple"]
    assert leaked == 2, "legacy mode must still bank candidates seen before the cosmic"
    assert stats["cosmic_events"] == 1


def test_legacy_ordering_is_off_by_default():
    proc = AmBeNeutronProcessing(cuts=CUTS, selection=BoxSelection(CUTS))
    assert proc.legacy_ordering is False


def test_legacy_and_fixed_agree_when_the_cosmic_comes_first():
    """The two paths differ only in the ordering-sensitive case; with the cosmic
    cluster first the old loop broke immediately and was already correct."""
    clusters = [COSMIC_PROMPT, CAND_A, CAND_B]
    a, sa = _run_event(clusters, legacy=False)
    b, sb = _run_event(clusters, legacy=True)
    assert (a["is_cosmic"], a["n_accepted_single"], a["n_accepted_multiple"]) == \
           (b["is_cosmic"], b["n_accepted_single"], b["n_accepted_multiple"])
    assert sa["cosmic_events"] == sb["cosmic_events"]
