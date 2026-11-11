"""Cut-boundary tests for the neutron definitions.

The fitter had a thorough test suite and the selection that FEEDS it had none, so
the delayed window could be -- and was -- unbounded above while every fit test
passed. These tests pin each edge of the box explicitly, because an off-by-one on a
boundary is exactly the kind of change that alters an efficiency by a fraction of a
percent and never crashes.
"""
import numpy as np
import pytest

from ambe.data.selection import (BoxSelection, CutCriteria, MvaSelection, Selection,
                                 cluster_key, get_selection)


CUTS = CutCriteria()          # the defaults: PE<=100, CB<0.45, 2000<=t<=67000, hits>=5


def _accepts(sel, cpe=50.0, ccb=0.2, ct=10000.0, chits=10, key=None):
    """Accept with everything else comfortably inside the box."""
    return sel.accepts(cpe, ccb, ct, chits, key)


# --------------------------------------------------------------------------- #
# The delayed search window -- the cut that did not exist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ct,expected", [
    (1999.0, False),      # below the window
    (2000.0, True),       # closed at the lower edge
    (2001.0, True),
    (66999.0, True),
    (67000.0, True),      # closed at the upper edge -- the intended convention
    (67001.0, False),     # above the window
    (200000.0, False),    # far outside; previously ACCEPTED
])
def test_box_time_window_boundaries(ct, expected):
    assert _accepts(BoxSelection(CUTS), ct=ct) is expected


def test_time_window_is_configurable_not_hardcoded():
    """ct_min/ct_max used to be dataclass defaults that no config could reach."""
    narrow = BoxSelection(CutCriteria(ct_min=5000, ct_max=30000))
    assert _accepts(narrow, ct=4999.0) is False
    assert _accepts(narrow, ct=5000.0) is True
    assert _accepts(narrow, ct=30000.0) is True
    assert _accepts(narrow, ct=30001.0) is False


def test_window_lives_on_the_base_class():
    """in_window sits beside the cosmic veto so both selections share it. If it were
    only on BoxSelection, the two analyses would search different spans and their
    efficiencies would stop being comparable."""
    assert hasattr(Selection, "in_window")
    assert "in_window" not in BoxSelection.__dict__
    assert "in_window" not in MvaSelection.__dict__


# --------------------------------------------------------------------------- #
# The rest of the box
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cpe,expected", [
    (0.0, False),         # pe_min is exclusive
    (0.001, True),
    (99.99, True),
    (100.0, True),        # pe_max is inclusive
    (100.01, False),
])
def test_box_pe_boundaries(cpe, expected):
    assert _accepts(BoxSelection(CUTS), cpe=cpe) is expected


@pytest.mark.parametrize("ccb,expected", [
    (0.0, False),         # ccb_min is exclusive
    (0.001, True),
    (0.449, True),
    (0.45, False),        # ccb_max is exclusive
    (0.46, False),
])
def test_box_charge_balance_boundaries(ccb, expected):
    assert _accepts(BoxSelection(CUTS), ccb=ccb) is expected


@pytest.mark.parametrize("chits,expected", [
    (4, False),
    (5, True),            # chits_min is inclusive
    (6, True),
])
def test_box_cluster_hits_boundaries(chits, expected):
    assert _accepts(BoxSelection(CUTS), chits=chits) is expected


def test_label_is_derived_from_the_cuts():
    """The label used to be free text in the config that nothing checked against the
    thresholds actually applied."""
    sel = BoxSelection(CutCriteria(pe_max=80, ccb_max=0.4, ct_min=3000, ct_max=50000,
                                   chits_min=7))
    assert "80" in sel.label and "0.4" in sel.label
    assert "3" in sel.label and "50" in sel.label and "7" in sel.label


# --------------------------------------------------------------------------- #
# The cosmic veto -- shared, and distinct from the window
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ct,cpe,expected", [
    (1999.0, 50.0, True),      # prompt
    (2000.0, 50.0, False),
    (10000.0, 100.01, True),   # high charge
    (10000.0, 100.0, False),
])
def test_cosmic_veto_boundaries(ct, cpe, expected):
    assert BoxSelection(CUTS).cosmic(ct, cpe) is expected


def test_cosmic_veto_is_not_overridden_by_either_subclass():
    assert "cosmic" not in BoxSelection.__dict__
    assert "cosmic" not in MvaSelection.__dict__


# --------------------------------------------------------------------------- #
# MVA: no box, but still the window
# --------------------------------------------------------------------------- #
class _FakeMva(MvaSelection):
    """MvaSelection without the parquet I/O, so the window logic can be tested."""
    def __init__(self, accepted_keys, cuts=None):
        Selection.__init__(self, cuts)
        self.threshold = 0.5
        self.score_col = "gbt_score"
        self.label = "fake mva"
        self.accepted = set(accepted_keys)
        self.scored_keys = set(accepted_keys)
        self.n_scored = len(accepted_keys)
        self.n_queried = 0
        self.n_unscored = 0
        self.unscored_runs = {}


def test_mva_applies_the_window_even_though_it_has_no_box():
    """MvaSelection previously applied NO time cut at all, so its span was whatever
    the scoring stage happened to cover."""
    key_in = cluster_key(6265, 123, 10000.0)
    key_out = cluster_key(6265, 123, 90000.0)
    sel = _FakeMva([key_in, key_out], CUTS)
    # In-window scored cluster: accepted.
    assert sel.accepts(9999.0, 0.99, 10000.0, 1, key_in) is True
    # Out-of-window scored cluster: rejected despite being above threshold.
    assert sel.accepts(9999.0, 0.99, 90000.0, 1, key_out) is False


def test_mva_out_of_window_is_not_counted_as_unscored():
    """A cluster outside the window is not 'missing information' -- it is outside the
    measurement. Counting it as unscored would corrupt the coverage diagnostic."""
    key = cluster_key(6265, 123, 90000.0)
    sel = _FakeMva([], CUTS)
    sel.accepts(50.0, 0.2, 90000.0, 10, key)
    assert sel.n_unscored == 0


def test_mva_ignores_the_box_but_not_the_window():
    """PE and CB well outside the box must still be accepted by the MVA."""
    key = cluster_key(6265, 1, 5000.0)
    sel = _FakeMva([key], CUTS)
    assert sel.accepts(5000.0, 0.99, 5000.0, 1, key) is True


def test_mva_without_a_key_is_an_error():
    sel = _FakeMva([], CUTS)
    with pytest.raises(ValueError, match="cluster key"):
        sel.accepts(50.0, 0.2, 10000.0, 10, None)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_get_selection_requires_a_known_name():
    with pytest.raises(SystemExit, match="unknown selection"):
        get_selection("boxx")


def test_get_selection_box_passes_cuts_through():
    cuts = CutCriteria(ct_max=40000)
    sel = get_selection("box", cuts=cuts)
    assert sel.cuts.ct_max == 40000
    assert _accepts(sel, ct=40001.0) is False
