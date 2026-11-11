import joblib
import numpy as np
import pandas as pd
import pytest

from ambe.mc import mva_analysis as mva


def test_primary_score_col_prefers_nn_then_falls_back():
    assert mva.primary_score_col(pd.DataFrame(columns=["rf_score", "xgb_score", "nn_score"])) == "nn_score"
    assert mva.primary_score_col(pd.DataFrame(columns=["rf_score", "gbt_score"])) == "gbt_score"
    assert mva.primary_score_col(pd.DataFrame(columns=["nn_score", "xgb_score"]),
                                 prefer="xgb") == "xgb_score"
    with pytest.raises(KeyError):
        mva.primary_score_col(pd.DataFrame(columns=["n_hits"]))


def test_working_points_fix_the_right_axis():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(1000, int), np.zeros(1000, int)]
    sc = np.r_[rng.uniform(0.3, 1.0, 1000), rng.uniform(0.0, 0.7, 1000)]
    ev = np.repeat(np.arange(500), 4)
    wp = mva.working_points(y, {"nn": sc}, ev, fp_targets=(0.1,)).set_index("point")
    assert abs(wp.loc["eff80", "signal_eff"] - 0.80) < 0.01
    assert wp.loc["fp0.1_per_event", "fp_per_event"] <= 0.1 + 1e-9


def test_frozen_working_point_reads_primary(tmp_path):
    art = {"meta": {"primary_model": "nn", "working_points": [
        {"model": "xgb", "point": "eff80", "threshold": 0.9},
        {"model": "nn", "point": "eff80", "threshold": 0.42}]}}
    p = tmp_path / "m.pkl"
    joblib.dump(art, p)
    assert mva.frozen_working_point(p) == ("nn_score", 0.42)
    assert mva.frozen_working_point(p, model="xgb") == ("xgb_score", 0.9)
    joblib.dump({"meta": {}}, p)
    with pytest.raises(SystemExit):
        mva.frozen_working_point(p)


def test_grouped_permutation_sees_information_split_across_twins():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 2, n)
    sig = y + rng.normal(0, 0.5, n)
    X = np.c_[sig, sig + rng.normal(0, 0.01, n), rng.normal(0, 1, n)]
    feats = ["beta1", "beta1_fit", "n_hits"]
    grp, per = mva.permutation_importance_auc(lambda Z: Z[:, 0] + Z[:, 1], X, y, feats,
                                              n_repeats=2)
    assert set(grp.group) == {"isotropy_centroid", "isotropy_fit", "multiplicity"}
    assert per.set_index("feature").loc["n_hits", "auc_drop"] == pytest.approx(0, abs=1e-9)


def test_parse_models_rejects_unknown():
    with pytest.raises(SystemExit):
        mva._parse_models("xgb,svm")
