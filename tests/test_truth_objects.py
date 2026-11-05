import numpy as np
import pandas as pd

from ambe.mc.capture_truth import DEFAULTS, build_truth_objects, match_clusters_to_objects


def _event():
    rows = []
    def add(t, **kw):
        base = dict(t=t, pe=1.0, truth_class=5, is_untraced=0, ancestor_trackID=-5,
                    ancestor_pdg=-5, origin_pdg=-5, origin_trackID=-5, origin_child_trackID=-5)
        base.update(kw)
        rows.append(base)
    for t in np.linspace(40, 60, 12):                       # muon Cherenkov
        add(t, origin_pdg=13, origin_trackID=2)
    for t in np.linspace(2500, 2520, 10):                   # its Michel electron
        add(t, origin_pdg=13, origin_trackID=2, origin_child_trackID=900)
    for t in np.linspace(10010, 10030, 8):                  # capture of neutron 7
        add(t, truth_class=1, ancestor_trackID=7, ancestor_pdg=2112, origin_pdg=2112, origin_trackID=7)
    for t in np.linspace(30010, 30030, 6):                  # capture of neutron 9
        add(t, truth_class=1, ancestor_trackID=9, ancestor_pdg=2112, origin_pdg=2112, origin_trackID=9)
    for t in (5000.0, 20000.0):                             # dark noise
        add(t, truth_class=0)
    return pd.DataFrame(rows)


def test_objects_and_ownership():
    df = _event()
    cfg = dict(DEFAULTS)
    cap_t = np.array([10000.0, 30000.0])
    obj, objs = build_truth_objects(df, cap_t, 10.0, cfg)
    assert (obj >= 0).sum() == len(df) - 2
    assert sorted(objs.obj_class) == ["michel", "muon", "n_capture", "n_capture"]
    caps = objs[objs.obj_class == "n_capture"].sort_values("t_med")
    assert list(caps.cap_j) == [0, 1]

    lbl = {"truth": obj.copy(), "merged": np.where(df.t > 9000, 0, -1)}
    rows, per_obj = match_clusters_to_objects(lbl, obj, objs, df, cfg, "plurality")
    merged = [r for r in rows if r["method"] == "merged"][0]
    assert merged["n_objects_contrib"] == 2 and merged["comp_darknoise"] == 1
    assert merged["tag"] == "n_capture"
    maj = match_clusters_to_objects(lbl, obj, objs, df, cfg, "majority50")[0]
    assert [r for r in maj if r["method"] == "merged"][0]["tag"] == "n_capture"
    assert per_obj["truth"]["owned"].tolist() == [1, 1, 1, 1]


def test_neutron_content_and_unmatched_split():
    df = _event()
    extra = pd.DataFrame([dict(t=10020.0, pe=0.4, truth_class=0, is_untraced=1),
                          dict(t=10025.0, pe=2.0, truth_class=0, is_untraced=1)])
    df = pd.concat([df, extra], ignore_index=True).fillna(-5)
    cfg = dict(DEFAULTS)
    obj, objs = build_truth_objects(df, np.array([10000.0, 30000.0]), 10.0, cfg)
    lbl = {"merged": np.where(df.t > 9000, 0, -1), "single": np.where((df.t > 9000) & (df.t < 11000), 0, -1)}
    rows = {r["method"]: r for r in match_clusters_to_objects(lbl, obj, objs, df, cfg, "plurality")[0]}
    m, s = rows["merged"], rows["single"]
    assert (m["n_neutrons"], m["n_neutrons_3"], m["n_captures"]) == (2, 2, 2)
    assert m["n_hits_other_neutrons"] == 6 and 19000 < m["dt_captures_min"] < 21000
    assert (s["n_neutrons"], s["n_captures"], s["n_hits_other_neutrons"]) == (1, 1, 0)
    assert s["comp_darknoise"] == 0 and s["comp_untraced"] == 2 and s["comp_untraced_lowpe"] == 1
    assert m["comp_darknoise"] == 1 and m["comp_untraced"] == 2


def test_optics_scan_names():
    from ambe.mc.capture_truth import optics_scan_methods, parse_optics_scan
    cfg = dict(DEFAULTS, optics_scan=dict(min_samples=[5, 8], xi=[0.02], t_unit_ns=[25], nms=[False, True]))
    names = optics_scan_methods(cfg)
    assert names == ["optics_ms5_xi0.02_tu25", "optics_ms5_xi0.02_tu25_nms",
                     "optics_ms8_xi0.02_tu25", "optics_ms8_xi0.02_tu25_nms"]
    assert parse_optics_scan(names[1]) == dict(min_samples=5, xi=0.02, t_unit_ns=25.0, nms=True)
    assert parse_optics_scan("optics") is None and parse_optics_scan("peak_tw20_n6_s50") is None


def test_mchit_truth_from_lineage():
    from ambe.mc.processor import _mchit_truth
    # e- <- gamma <- n(7) <- p(3), complete chain: secondary neutron from a proton
    r = _mchit_truth((11, 22, 2112, 2212), (50, 40, 7, 3), 1, 0)
    assert r[0] == 2 and r[2] == 7 and r[3] == 2112 and r[4] is False
    assert r[7] == 22 and r[9] == 2212 and r[10] == 1
    # primary neutron at the top of a complete chain
    assert _mchit_truth((11, 22, 2112), (5, 4, 2), 1, 0)[0] == 1
    # muon light: no neutron, immediate ancestor skips the e-
    r = _mchit_truth((11, 13), (8, 1), 1, 0)
    assert r[0] == -5 and r[7] == 13 and r[8] == 1
    assert _mchit_truth((), (), -5, 1)[:2] == (0, 1)             # dark noise
    assert _mchit_truth((), (), 0, 0)[4] is True                 # unresolved


def test_pion_split_and_prompt_michel_rule():
    rows = []
    for t in np.linspace(0, 20, 10):                             # muon light
        rows.append(dict(t=t, origin_pdg=13, origin_trackID=2, origin_child_trackID=-5))
    for t in np.linspace(5, 25, 6):                              # its delta ray, same time
        rows.append(dict(t=t, origin_pdg=13, origin_trackID=2, origin_child_trackID=77))
    for t in np.linspace(1500, 1520, 8):                         # Michel at 1.5 us
        rows.append(dict(t=t, origin_pdg=13, origin_trackID=2, origin_child_trackID=900))
    for t in np.linspace(2, 12, 5):
        rows.append(dict(t=t, origin_pdg=211, origin_trackID=4, origin_child_trackID=-5))
    for t in np.linspace(3, 13, 5):
        rows.append(dict(t=t, origin_pdg=111, origin_trackID=5, origin_child_trackID=60))
    df = pd.DataFrame(rows).assign(pe=1.0, truth_class=5, is_untraced=0,
                                   ancestor_trackID=-5, ancestor_pdg=-5)
    cfg = dict(DEFAULTS, _full_window=True)
    _, objs = build_truth_objects(df, np.array([]), 0.0, cfg)
    assert sorted(objs.obj_class) == ["michel", "muon", "pi0", "pion_charged"]
    assert objs[objs.obj_class == "michel"].t_first.iloc[0] >= 1500


def test_overlay_drops_mc_noise_and_is_seeded():
    from ambe.mc.capture_truth import add_overlay
    mc = pd.DataFrame(dict(eventID=[0, 0, 0, 1], eventNumber=[0, 0, 0, 1], _source_file="f.root",
                           cc_pass=1, t=[100.0, 3000.0, 4000.0, 5000.0], pe=1.0, pmtID=1,
                           x=0.0, y=0.0, z=0.0, truth_class=[5, 0, 0, 1], is_untraced=[0, 0, 1, 0]))
    ov = {0: pd.DataFrame(dict(pmtID=[9, 9], t=[50.0, 2500.0], x=0.0, y=0.0, z=0.0, pe=2.0)),
          1: pd.DataFrame(dict(pmtID=[8], t=[6000.0], x=0.0, y=0.0, z=0.0, pe=2.0))}
    a = add_overlay(mc, ov, 2000.0, seed=1)
    assert not ((a.is_overlay == 0) & (a.truth_class == 0) & (a.is_untraced == 0)).any()
    assert (a.is_overlay == 1).sum() >= 1 and (a[a.is_overlay == 1].t > 2000).all()
    assert a.equals(add_overlay(mc, ov, 2000.0, seed=1))


def test_hit_efficiency_accounting():
    df = _event()
    cfg = dict(DEFAULTS)
    obj, objs = build_truth_objects(df, np.array([10000.0, 30000.0]), 10.0, cfg)
    cap7 = int(objs[(objs.obj_class == "n_capture")].sort_values("t_med").obj_id.iloc[0])
    lb = np.full(len(df), -1)
    lb[np.where(obj == cap7)[0][:6]] = 0          # 6 of its 8 hits in its own cluster
    lb[np.where(obj == cap7)[0][6:7]] = 1         # 1 hit inside the muon's cluster
    mu = int(objs[objs.obj_class == "muon"].obj_id.iloc[0])
    lb[np.where(obj == mu)[0]] = 1                # muon cluster
    rows, per_obj = match_clusters_to_objects({"m": lb}, obj, objs, df, cfg, "plurality")
    d = per_obj["m"]
    assert d["hits_own"][cap7] == 6 and d["hits_clustered"][cap7] == 7
    assert objs.n_hits[cap7] - d["hits_clustered"][cap7] == 1
    cl1 = [r for r in rows if r["cluster_id"] == 1][0]
    assert cl1["n_owner_hits"] == 12 and abs(cl1["owner_pe_frac"] - 12 / 13) < 1e-9
