"""Synthetic checks for shared plot inputs, fit engines and output ownership."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ambe.context import RunContext
from ambe.plots import basic, waveform_cuts


@pytest.fixture
def capture_times():
    centers = np.arange(70, dtype=float) + 0.5
    counts = np.rint(basic.NeutCapture(centers, 2000., 5., 30., 3.)).astype(int)
    return np.repeat(centers, counts)


@pytest.mark.parametrize("backend", ["scipy", "lmfit"])
def test_capture_fit_recovers_model_and_respects_fixed_background(capture_times, backend):
    analyzer = basic.AmBeNeutronAnalyzer()
    analyzer.fitting_config["initial_background"] = 3.
    result = basic.fit(capture_times, False, backend, analyzer=analyzer)
    assert result is not None
    assert result["tau"] == pytest.approx(30., abs=.03)
    assert result["therm"] == pytest.approx(5., abs=.03)
    assert result["B"] == 3.
    assert result["B_err"] == 0.
    assert "_result" not in result


def test_capture_page_and_table_share_the_same_fit(capture_times):
    analyzer = basic.AmBeNeutronAnalyzer()
    expected = basic.fit(capture_times, False, "lmfit", analyzer=analyzer)
    pages = []
    pdf = SimpleNamespace(savefig=lambda **kwargs: pages.append(kwargs))
    analyzer.lmfit_analysis({"CT": capture_times}, (0, 0, 0), pdf)
    assert len(pages) == 1
    assert len(analyzer.capture_fit_rows) == 1
    row = analyzer.capture_fit_rows[0]
    assert row["pos"] == (0, 0, 0)
    for key, value in expected.items():
        assert row[key] == pytest.approx(value)
    assert analyzer.lmfit_summary[0]["Tau"] == row["tau"]


def test_empty_capture_fit_is_not_a_measurement():
    assert basic.fit(np.array([]), False, "scipy") is None
    assert basic.fit(np.array([]), False, "lmfit") is None


@pytest.fixture
def waveform_products(tmp_path):
    path = tmp_path / "WaveformFeatures_example_123.csv"
    frame = pd.DataFrame({"timestamp": [0, 1, 2, 3, 4, 5],
                          "IC_adjusted": [0, 700, 900, 1200, 900, 1100],
                          "second_pulse": [False, False, False, False, True, False],
                          "accepted": [False, False, True, False, False, True]})
    frame.to_csv(path, index=False)
    ctx = SimpleNamespace(inputs={"waveform_features": str(path)},
                          extra={"stage1": {"ic_min": 700, "ic_max": 1200}})
    return path, frame, ctx


def test_waveform_overlay_uses_processor_decisions(waveform_products):
    _, _, ctx = waveform_products
    data = waveform_cuts.load_features(ctx, 123)
    assert data.timestamp.tolist() == [1, 2, 3, 4, 5]
    assert data.accepted.tolist() == [False, True, False, False, True]


def test_waveform_overlay_rejects_mismatched_saved_gate(waveform_products):
    path, frame, ctx = waveform_products
    frame.loc[1, "accepted"] = True  # IC exactly on the strict lower bound
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="saved gate differs"):
        waveform_cuts.load_features(ctx, 123)


def test_waveform_overlay_rejects_empty_products(waveform_products):
    path, frame, ctx = waveform_products
    frame.iloc[:1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="no recorded waveforms"):
        waveform_cuts.load_features(ctx, 123)


def test_output_defaults_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    from ambe import paths
    repo = tmp_path / "AmBeNeutronsAnalysis"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", repo)
    monkeypatch.setattr(paths, "DEFAULT_OUTPUT", repo / "ambe_output")
    monkeypatch.delenv("AMBE_OUT", raising=False)
    assert paths.output_root() == repo / "ambe_output"
    assert paths.output_root("ambe_output/ambe_data") == repo / "ambe_output/ambe_data"
    monkeypatch.setenv("AMBE_OUT", "custom")
    assert paths.output_root() == repo / "custom"
    assert paths.output_root(str(tmp_path / "explicit")) == tmp_path / "explicit"
    with pytest.raises(ValueError, match="AmBeNeutronAnalysis"):
        paths.output_root(str(tmp_path / "AmBeNeutronAnalysis/ambe_output"))


def test_generated_markdown_has_a_dedicated_location(tmp_path, monkeypatch):
    from ambe import context
    monkeypatch.setattr(context, "REPOSITORY_ROOT", tmp_path)
    ctx = RunContext(run_name="example", output_root=str(tmp_path / "outputs"))
    assert ctx.markdown_path("cutflow") == tmp_path / "docs/generated/example/cutflow.md"


def test_efficiency_plot_passes_numerical_config_to_shared_fitter(monkeypatch):
    from ambe.efficiency_extraction import efficiency_thesis as plots
    fp = dict(source_run=1, background_run=2, seed=7,
              eff_grid={"min": .2, "max": .8, "step": .1},
              lambda_grid={"min": .01, "max": .2, "step": .01},
              likelihood="multinomial", background_uncertainty="bootstrap",
              overflow_policy="clip", n_bootstrap=12, template_fluctuate=True)
    ctx = SimpleNamespace(fit_params=fp, plots_dir="unused")
    hist_calls, fit_calls = [], {}
    monkeypatch.setattr(plots, "_resolve_event_index", lambda ctx: object())
    def histogram(*args, **kwargs):
        hist_calls.append(kwargs)
        return object()
    monkeypatch.setattr(plots, "build_multiplicity", histogram)
    def fit_model(name):
        def call(*args, **kwargs):
            fit_calls[name] = kwargs
            return SimpleNamespace(summary=lambda: name)
        return call
    monkeypatch.setattr(plots, "fit_poisson", fit_model("poisson"))
    monkeypatch.setattr(plots, "fit_datadriven", fit_model("data"))
    monkeypatch.setattr(plots, "assert_interior", lambda *args: None)
    for name in ("plot_multiplicity", "plot_bestfit_overlay", "plot_chi2_1d",
                 "plot_chi2_2d", "plot_chi2_projections"):
        monkeypatch.setattr(plots, name, lambda *args: None)
    plots.run(ctx)
    assert hist_calls == [{"overflow_policy": "clip"}] * 2
    assert fit_calls["poisson"]["likelihood"] == "multinomial"
    assert fit_calls["data"]["likelihood"] == "multinomial"
    assert fit_calls["data"]["bkg_unc_mode"] == "bootstrap"
    assert fit_calls["data"]["n_bootstrap"] == 12
    assert fit_calls["data"]["template_fluctuate"] is True


def test_presentation_adapters_import_the_shared_capture_model():
    from scripts.presentation import boxcut_v4_campaign, boxcut_v4_appendix
    from scripts.presentation import ccinc_v3_ambe_closure as closure
    assert closure.fit_neutcapture is basic.fit
    assert (closure.TAU_LO, closure.TAU_HI) == basic.AmBeNeutronAnalyzer().fitting_config["capture_bounds"]


def test_stageab_uses_shared_fit_without_patching_lmfit(capture_times, tmp_path, monkeypatch):
    import lmfit
    from ambe.diagnostics import capture_time_stageAB_from_parquet as stage
    original_fit = lmfit.Model.fit
    source = tmp_path / "scored.parquet"
    pd.DataFrame({"run": 1, "rf_score": 1., "t_mean": capture_times * 1000,
                  "event_tank_time": np.arange(len(capture_times))}).to_parquet(source)
    out = tmp_path / "plots"
    monkeypatch.setattr(stage, "SOURCE_POSITIONS", {1: (0, 0, 0)})
    monkeypatch.setattr(basic.AmBeNeutronAnalyzer, "generate_summary_plots", lambda self: None)
    monkeypatch.setattr("sys.argv", ["stage", "--scored", str(source), "--out-dir", str(out),
                                     "--methods", "lmfit"])
    stage.main()
    assert lmfit.Model.fit is original_fit
    table = pd.read_csv(out / "stageAB_capture_lmfit_summary.csv")
    expected = basic.fit(capture_times, False, "lmfit")
    assert table.Tau.iloc[0] == pytest.approx(expected["tau"])
    assert table.Tau_err.iloc[0] == pytest.approx(expected["tau_err"])


def test_styled_capture_page_uses_shared_fit(capture_times):
    from scripts.presentation import make_presentable_optics_rf_plots as styled
    analyzer = basic.AmBeNeutronAnalyzer()
    pages = []
    pdf = SimpleNamespace(savefig=lambda fig: pages.append(True))
    result = styled._fit_and_plot_capture(pdf, analyzer, capture_times, "Synthetic sample")
    expected = basic.fit(capture_times, False, "lmfit")
    assert len(pages) == 1
    assert result["tau"] == pytest.approx(expected["tau"])
    assert result["tau_err"] == pytest.approx(expected["tau_err"])
