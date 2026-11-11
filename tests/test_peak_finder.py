import numpy as np

from ambe.clustering.peak_finder import (find_time_peaks, merge_clusters_in_time,
                                         n_residual_hits, clean_hits)


def _burst(t0, n, rng, spread=4.0):
    return t0 + rng.normal(0.0, spread, n)


def test_two_captures_far_apart_give_two_candidates():
    rng = np.random.default_rng(1)
    t = np.concatenate([_burst(5_000, 12, rng), _burst(25_000, 10, rng)])
    labels, peaks = find_time_peaks(t, twidth_ns=20, nhitsth=6, t_min_sep_ns=50)
    assert len(peaks) == 2
    assert (labels[:12] == 0).sum() >= 10 and (labels[12:] == 1).sum() >= 8


def test_one_capture_is_never_split():
    # Two sub-bursts 15 ns apart: the AmBe split-pair signature (median dt 15 ns).
    rng = np.random.default_rng(2)
    t = np.concatenate([_burst(8_000, 8, rng, 2.0), _burst(8_015, 8, rng, 2.0)])
    labels, peaks = find_time_peaks(t, twidth_ns=20, nhitsth=6, t_min_sep_ns=50)
    assert len(peaks) == 1
    assert (labels == 0).sum() >= 14


def test_flat_noise_gives_almost_no_candidates():
    rng = np.random.default_rng(3)
    # ~130 PMTs x ~1 kHz x 65 us ~ 8 hits: far below 6 hits in 20 ns.
    t = rng.uniform(2_000, 67_000, 40)
    _, peaks = find_time_peaks(t, twidth_ns=20, nhitsth=6)
    assert len(peaks) == 0


def test_nms_is_deterministic_under_permutation():
    rng = np.random.default_rng(4)
    t = np.concatenate([_burst(5_000, 10, rng), _burst(5_120, 7, rng),
                        rng.uniform(2_000, 67_000, 30)])
    l1, p1 = find_time_peaks(t)
    perm = rng.permutation(len(t))
    l2, p2 = find_time_peaks(t[perm])
    assert [p["t_peak"] for p in p1] == [p["t_peak"] for p in p2]
    assert np.array_equal(l1[perm], l2)


def test_tof_correction_collapses_a_point_source():
    rng = np.random.default_rng(5)
    xyz = rng.normal(0, 1, (12, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]          # PMTs on a 1 m sphere
    xyz[:6] *= 0.2                                       # half of them much closer
    src = np.zeros(3)
    t = 10_000 + np.linalg.norm(xyz - src, axis=1) / (0.299792458 * 0.75)
    _, peaks = find_time_peaks(t, xyz=xyz, ref_m=src, twidth_ns=2, nhitsth=12)
    assert len(peaks) == 1


def test_merge_clusters_in_time_joins_split_pairs_only():
    t = np.array([100., 101, 102, 115, 116, 117, 30_000, 30_001])
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    out = merge_clusters_in_time(labels, t, t_min_sep_ns=50)
    assert len(set(out[:6])) == 1 and out[6] != out[0]
    assert len(np.unique(out)) == 2


def test_merge_keeps_noise_label():
    out = merge_clusters_in_time(np.array([-1, 0, 0]), np.array([5., 10, 11]), 50)
    assert out[0] == -1


def test_nres_and_deadtime():
    t = np.array([0., 1, 2, 50, 90, 500])
    labels = np.array([0, 0, 0, -1, -1, -1])
    assert n_residual_hits(t, labels, 0, window_ns=100) == 2
    keep = clean_hits(np.array([0., 300, 2000]), np.array([7, 7, 7]), deadtime_ns=1000)
    assert keep.tolist() == [True, False, True]
