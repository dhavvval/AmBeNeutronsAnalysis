import pandas as pd

from ambe.context import RunContext
from ambe.efficiency_extraction.recluster_index import build


def _ctx(tmp_path, gate_glob):
    return RunContext(run_name="t", output_root=str(tmp_path / "out"),
                      inputs={"root_files": ["/x/BeamCluster_6046.root"]},
                      extra={"clusterindex": {"window_ns": [2000, 67000],
                                              "gate_event_index": [gate_glob]}})


def test_data_index_zero_bin_veto_and_gate(tmp_path):
    gate = tmp_path / "gate.parquet"
    pd.DataFrame({"run": 6046, "eventTankTime": [10, 11, 12, 13]}).to_parquet(gate)
    ctx = _ctx(tmp_path, str(gate))
    base = f"{ctx.run_name}__random_%s_6046"
    pd.DataFrame({"trigger": range(5), "event_tank_time": [10, 11, 12, 13, 14],
                  "n_clusters": [0, 1, 2, 1, 1]}).to_parquet(ctx.parquet_path(base % "triggers"))
    # trigger 2 has a prompt CF cluster -> vetoed; trigger 4 is outside the gate
    pd.DataFrame({"trigger": [1, 2, 2, 3, 4], "clusterTime": [5e3, 500., 9e3, 7e4, 5e3],
                  "clusterPE": [20., 30., 20., 10., 20.]}).to_parquet(ctx.parquet_path(base % "cfclusters"))
    pd.DataFrame({"trigger": [1, 1, 2, 3, 4], "method": "peak",
                  "t_med": [5e3, 6e3, 9e3, 7e4, 5e3], "f_pe_total": [20., 15., 20., 10., 20.],
                  "n_hits": 10}).to_parquet(ctx.parquet_path(base % "recoclusters" + "__x"))
    for g, n in (("ic", 4), ("none", 5)):
        build(ctx, "data", g, ["peak"], "x")
        ei = pd.read_parquet(ctx.parquet_path(f"t__eventindex_{g}_peak"))
        assert len(ei) == n
        assert ei.set_index("eventTankTime").is_cosmic.to_dict()[12]
        assert ei.n_accepted_clusters.tolist()[:4] == [0, 2, 0, 0]   # veto zeroes 12, 13 out of window
    cand = pd.read_parquet(ctx.parquet_path("t__candidates_none"))
    assert sorted(cand.eventTankTime.tolist()) == [11, 11, 14]


def test_overlay_pick_matches_add_overlay():
    from ambe.mc.capture_truth import add_overlay, overlay_pick
    ov = {i: pd.DataFrame({"pmtID": [i], "t": [5e3], "x": 0., "y": 0., "z": 0., "pe": [float(i)]})
          for i in range(50)}
    chunk = pd.DataFrame({"eventID": [1, 2], "eventNumber": [7, 8], "_source_file": "f_0_999.root",
                          "t": 4e3, "pe": 1., "truth_class": 1, "is_untraced": 0})
    out = add_overlay(chunk, ov, 2000., 6256)
    got = out[out.is_overlay == 1].set_index("eventNumber").pe.to_dict()
    assert got == {n: float(overlay_pick(6256, "f_0_999.root", n, 50)) for n in (7, 8)}


def _fold_sample(n_ev=40000, eff=0.7, seed=1):
    import numpy as np
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    ev = pd.DataFrame({"_source_file": "f_0_999.root", "eventNumber": np.arange(n_ev), "run": 1,
                       "sourceX": 0., "sourceY": 0., "sourceZ": 0., "found": False})
    hit = ev[rng.random(n_ev) < eff]
    m_c = hit[["_source_file", "eventNumber"]].assign(cap=True)
    m_c = pd.concat([m_c, ev[["_source_file", "eventNumber"]].iloc[:500].assign(cap=False)])
    j = rng.choice([0, 1, 2], size=6000, p=[0.9, 0.08, 0.02])
    b_e = pd.DataFrame({"run": 6264, "is_cosmic": False, "n_accepted_clusters": j,
                        "sourceX": 0., "sourceY": 0., "sourceZ": -999.})
    return SimpleNamespace(m_e=ev, m_c=m_c, b_e=b_e), j


def test_fold_index_seeded_and_convolution():
    import numpy as np
    from ambe.efficiency_extraction.recluster_index import fold_index
    from ambe.plots.datamc_clustering import convolve_pmf, pmf
    S, j = _fold_sample()
    a, b = fold_index(S, 6264), fold_index(S, 6264)
    assert a.n_accepted_clusters.equals(b.n_accepted_clusters)
    assert (a.n_accepted_clusters >= a.truth_k).all() and a.truth_k.max() == 1   # noise clusters not counted
    want = convolve_pmf(pmf(a.truth_k.to_numpy(), 3), pmf(j, 3), 3)
    got = pmf(a.n_accepted_clusters.to_numpy(), 3)
    assert np.allclose(got, want, atol=0.006)
    g = fold_index(S, 6264, np.full(len(S.m_e), 2.0))
    assert abs(len(g) / len(a) - 2.0) < 0.03


def test_fold_fit_recovers_efficiency(tmp_path):
    import numpy as np
    from ambe.efficiency_extraction.recluster_index import _fit, fold_index
    S, _ = _fold_sample(eff=0.7)
    fi = fold_index(S, 6264)
    cols = ["run", "is_cosmic", "n_accepted_clusters", "sourceX", "sourceY", "sourceZ"]
    ei = pd.concat([fi[cols], S.b_e[cols]], ignore_index=True)
    ctx = RunContext(run_name="t", output_root=str(tmp_path / "out"))
    fp = {"background_run": 6264, "source_runs": "all", "n_bins": 4, "likelihood": "multinomial",
          "background_uncertainty": "profile", "overflow_policy": "clip", "n_gof_toys": 50,
          "eff_grid": {"min": 0.5, "max": 0.9, "step": 0.002},
          "lambda_grid": {"min": 0.0, "max": 0.4, "step": 0.004},
          "s_grid": {"min": 0.0, "max": 1.5, "step": 0.01}, "seed": 1}
    df = _fit(ctx, fp, ei, 1, "_t")
    r = df[df.level == "position"].iloc[0]
    truth = (fi.truth_k > 0).mean()
    sig = 0.5 * (r.dds_eff_hi - r.dds_eff_lo)
    assert abs(r.dds_eff - truth) < 2 * sig + 0.002
    assert abs(r.dds_s - 1.0) < 0.2
    assert r.naive_eff > truth + 0.02
