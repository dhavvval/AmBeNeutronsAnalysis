import types

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages

from ambe.data.selection import BoxSelection
from ambe.plots import cluster_truth, provenance_deck

FOUR = "clusterfinder,clusterfinder_nms,optics_nms,peak"


def _ctx(tmp_path, name="r"):
    return types.SimpleNamespace(run_name=name, parquet_path=lambda b: tmp_path / f"{b}.parquet")


def test_clustertruth_needs_known_plot_methods():
    base = ["--world-config", "w.yaml", "--tag", "t", "--out", "o"]
    with pytest.raises(SystemExit):
        cluster_truth.cli(None, base)
    with pytest.raises(SystemExit):
        cluster_truth.cli(None, base + ["--plot-methods", "clusterfinder,dbscan"])


@pytest.mark.parametrize("deck", ["neutrino", "clusterpurity"])
def test_provenance_decks_need_plot_methods(deck):
    with pytest.raises(SystemExit):
        provenance_deck.cli(None, ["--deck", deck, "--world-config", "w.yaml", "--tag", "nu",
                                   "--clean-tag", "nuclean", "--out", "o"])


def test_cf_noise_adds_tree_and_nms_clusters(tmp_path):
    c = BoxSelection().cuts
    t_in = 0.5 * (c.ct_min + c.ct_max)
    pd.DataFrame(dict(trigger=[1, 1, 2], pmtID=1, t=t_in)).to_parquet(tmp_path / "r__random_hits_6256.parquet")
    pd.DataFrame(dict(trigger=[1, 2], clusterTime=[t_in, c.ct_max + 1e4], clusterHits=c.chits_min,
                      clusterPE=1.0, clusterCB=0.1)).to_parquet(tmp_path / "r__random_cfclusters_6256.parquet")
    pd.DataFrame(dict(trigger=[1, 1, 2], method=["clusterfinder_nms", "peak", "clusterfinder_nms"],
                      t_med=t_in, n_hits=c.chits_min, f_pe_total=1.0, f_charge_bal_legacy=0.1)
                 ).to_parquet(tmp_path / "r__random_recoclusters_6256__x.parquet")
    ev = pd.DataFrame({"_source_file": ["a.root"] * 4, "eventNumber": range(4),
                       **{f"n_reco_{k}_{m}": 0 for k in ("delayed", "box")
                          for m in ("clusterfinder", "clusterfinder_nms")}})
    methods = {m: m for m in FOUR.split(",")}
    out = cluster_truth._cf_noise(ev, _ctx(tmp_path), 6256, methods, "x")
    for m in ("clusterfinder", "clusterfinder_nms"):
        assert set(out[f"n_reco_delayed_{m}"]) <= {0, 1}
    assert out.n_reco_delayed_clusterfinder_nms.sum() >= out.n_reco_delayed_clusterfinder.sum()
    assert out.n_reco_delayed_clusterfinder_nms.sum() == 4
    with pytest.raises(FileNotFoundError):
        cluster_truth._cf_noise(ev, _ctx(tmp_path), 6256, methods, "")


def test_multiplicity_pages_for_four_methods(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    e = pd.DataFrame(dict(cc_pass=1, origin_in_tank=1, n_true_captures=rng.integers(0, 3, n),
                          n_visible_captures_1=1, n_visible_captures_5=1))
    for m in FOUR.split(","):
        e[f"n_reco_delayed_{m}"] = rng.integers(0, 3, n)
        e[f"n_reco_box_{m}"] = rng.integers(0, 2, n)
    methods = {m: cluster_truth.METHOD_LABELS[m] for m in FOUR.split(",")}
    with PdfPages(tmp_path / "m.pdf") as pdf:
        t = cluster_truth.multiplicity({"tank": e, "world": e}, methods, pdf)
    assert set(t.clusterer) == set(methods.values())
    assert len(t) == 3 * 4 * 2
