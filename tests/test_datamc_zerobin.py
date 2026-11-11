import numpy as np
import pandas as pd

from ambe.plots.datamc_zerobin import _found_curve, _thinned


def _caps():
    n = np.array([0, 2, 4, 5, 5, 6, 10, 10, 20, 20])
    own = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    return pd.DataFrame({"n_light_hits": n, "n_owners_x": own})


def test_keep_one_reproduces_the_found_curve():
    cp = _caps()
    curve = _found_curve(cp, "x", nmax=30)
    assert curve[5] == 0.5 and curve[10] == 1.0 and curve[3] == 1.0   # unpopulated bins default to 1
    pf, eh = _thinned(cp.n_light_hits.to_numpy(), 1.0, curve)
    assert np.allclose(pf, curve[cp.n_light_hits.to_numpy()])
    assert np.allclose(eh, pf * cp.n_light_hits.to_numpy())


def test_thinning_lowers_efficiency_and_hits():
    cp = _caps()
    curve = _found_curve(cp, "x", nmax=30)
    curve[1:5] = 0.0
    pf1, eh1 = _thinned(cp.n_light_hits.to_numpy(), 1.0, curve)
    pf8, eh8 = _thinned(cp.n_light_hits.to_numpy(), 0.8, curve)
    assert (pf8 <= pf1 + 1e-12).all() and pf8.sum() < pf1.sum()
    assert eh8.sum() / pf8.sum() < eh1.sum() / pf1.sum()
