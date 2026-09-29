"""Synthetic regression checks for shared waveform and event selection."""
import numpy as np
import pandas as pd
import pytest

from ambe.data.processor import AmBeNeutronProcessing, WaveformConfig
from ambe.data.selection import BoxSelection
from ambe.diagnostics.compare_veto import compare


def test_waveform_ic_matches_original_formula_and_veto():
    proc = AmBeNeutronProcessing()
    edges = np.arange(1501, dtype=float)
    values = np.full(1500, 100.0)
    values[350:400] += 1200.0
    features = proc.compute_waveform_features(values, edges)
    baseline = values.mean()
    expected = 2 / 50 * (values[301:1200] - baseline).sum()
    assert features["baseline"] == pytest.approx(baseline)
    assert features["IC_adjusted"] == pytest.approx(expected)
    assert not features["second_pulse"]
    assert set(features) == {"IC_adjusted", "baseline", "baseline_sigma", "second_pulse"}
    values[1300] = 10000
    assert proc.compute_waveform_features(values, edges)["second_pulse"]


@pytest.mark.parametrize("ic,second,expected", [
    (700, False, False), (700.1, False, True), (1199.9, False, True),
    (1200, False, False), (900, True, False), (float("nan"), False, False),
])
def test_ic_gate_boundaries(ic, second, expected):
    proc = AmBeNeutronProcessing(WaveformConfig())
    assert proc.passes_selection({"IC_adjusted": ic, "second_pulse": second}) is expected


def test_plot_masks_match_scalar_box_at_boundaries():
    selection = BoxSelection()
    pe = np.array([0, 50, 100, 100.01, 50, 50, 50, 50, 50])
    cb = np.array([.2, .2, .2, .2, .45, 0, .2, .2, .2])
    times = np.array([2000, 67000, 2000, 2000, 2000, 2000, 67001, 1999, 2000])
    hits = np.array([5, 5, 5, 5, 5, 5, 5, 5, 4])
    vector = np.logical_and.reduce(selection.cut_masks(pe, cb, times, hits))
    scalar = [selection.accepts(*row) for row in zip(pe, cb, times, hits)]
    np.testing.assert_array_equal(vector, scalar)


def test_veto_comparison_separates_event_and_cluster_losses():
    old = pd.DataFrame({"run": [1, 1, 1], "eventTankTime": [10, 20, 30],
                        "is_cosmic": [True, True, False], "numberOfClusters": [2, 3, 1],
                        "n_accepted_single": [0, 0, 1], "n_accepted_multiple": [1, 2, 0]})
    new = old.copy()
    new.loc[new.is_cosmic, "n_accepted_multiple"] = 0
    row = compare(old, new).iloc[0]
    assert row.ambe_triggers == 1
    assert row.lost_one_accepted == 1
    assert row.lost_multiple_accepted == 1
    assert row.lost_original_single == 0
    assert row.lost_original_multiple == 2
    assert row.lost_clusters == 3
    with pytest.raises(ValueError, match="cohorts differ"):
        compare(old, new.iloc[1:])
    with pytest.raises(ValueError, match="duplicate"):
        compare(pd.concat([old, old.iloc[:1]]), new)


def test_background_optics_cli_uses_shared_extractor(tmp_path, monkeypatch):
    from ambe.cli import main
    from ambe.mc import cluster_features as features

    config = tmp_path / "background.yaml"
    config.write_text("run_name: synthetic\ngeometry:\n  pmt_geometry: geometry.csv\n"
                      "  timing_offsets: offsets.csv\n")
    monkeypatch.setenv("AMBE_OUT", str(tmp_path / "outputs"))
    output = tmp_path / "features.parquet"
    calls = []
    geometry = object()
    monkeypatch.setattr(features, "load_geometry", lambda *args, **kwargs: geometry)

    def extract(path, geo, **kwargs):
        calls.append((path, geo, kwargs))
        return pd.DataFrame({"n_hits": [5], "is_background": [1]})

    monkeypatch.setattr(features, "extract_features_from_background_hits_optics", extract)
    main(["mc", "features", "--config", str(config), "--hits", "tagged.parquet",
          "--rerun-optics", "--min-samples", "3", "--xi", "0.05",
          "--t-unit-ns", "25", "--output", str(output)])
    assert calls == [("tagged.parquet", geometry,
                      {"min_samples": 3, "xi": .05, "t_unit_ns": 25.,
                       "source_pos_m": None})]
    assert pd.read_parquet(output).n_hits.tolist() == [5]


def test_special_run_diagnostics_share_configured_box():
    from scripts.presentation.boxcut_v4_special import box_mask, SPECIAL_CUTS

    frame = pd.DataFrame({"clusterPE": [50, 50], "clusterChargeBalance": [.2, .2],
                          "clusterTime": [SPECIAL_CUTS.ct_max / 1000,
                                          (SPECIAL_CUTS.ct_max + 1) / 1000],
                          "clusterHits": [5, 5]})
    assert box_mask(frame).tolist() == [True, False]
