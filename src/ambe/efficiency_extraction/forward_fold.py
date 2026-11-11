"""
Forward fold of the background-normalisation plan (step 3, plan P4):

    ambe stats foldeff --config configs/step3_ambe2v4_bkgnorm.yaml --fold-model forward --stage closure|data|checks

Per source run r at position pos(r) in group g (port by default), n = TA ClusterFinder clusters with time in
fold.window_ns per veto-passing IC-gated trigger:

    nu_r(n) = beta_rn * (N_r - F_r * sum(theta)) * [M_pos(s_g) (*) Poisson(mu_g)](n) + F_r * theta_n

M_pos(s): setup-A MC multiplicity (the 6256 overlay is inside the MC), the clusters of each true capture kept
together with probability s; a polynomial in s, continued past s = 1 (and Poisson past mu = 0) so the fit is
not cut at the physical edge. theta: the background-run (6264) gated veto-passing bins, Poisson-constrained by
its counts and shared by every run. beta: Barlow-Beeston-lite for the MC template, profiled per bin.
F_r: the frozen factors CSV (md5 checked). Efficiency at a position: eps(s) = P(>= 1 kept capture cluster).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context

PORT_OF = {(0.0, 0.0): 5, (75.0, 0.0): 4, (0.0, -75.0): 1, (0.0, 75.0): 2, (0.0, 102.0): 3}
NPOIS = 30
BIG = 1e12


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def poisson_coeffs(mu: float, n: int = NPOIS) -> np.ndarray:
    """e^-mu mu^k / k!, k < n; valid as a signed series for mu < 0."""
    k = np.arange(n)
    out = np.empty(n)
    out[0] = np.exp(-mu)
    for i in range(1, n):
        out[i] = out[i - 1] * mu / i
    return out


def poisson_matrix(mu: float, K1: int) -> np.ndarray:
    """P[j, n]: bin j -> bin n under (*) Poisson(mu); the last bin is the overflow and stays one."""
    K = K1 - 1
    c = poisson_coeffs(mu, max(NPOIS, K1))
    P = np.zeros((K1, K1))
    for j in range(K):
        P[j, j:K] = c[:K - j]
        P[j, K] = 1.0 - P[j, :K].sum()
    P[K, K] = 1.0
    return P


def fold_poisson(p: np.ndarray, mu: float) -> np.ndarray:
    """p (..., K+1) with the last bin = overflow; returns p (*) Poisson(mu), overflow kept as overflow."""
    if mu == 0.0:
        return p
    return p @ poisson_matrix(mu, p.shape[-1])


# ---------------------------------------------------------------------------------------------- templates
@dataclass
class Template:
    """MC multiplicity of one position as signatures: m = clusters of each true capture, ko = other clusters."""
    sig_m: list                     # tuple of per-capture cluster counts, per signature
    sig_ko: np.ndarray
    w: np.ndarray                   # events per signature
    K: int                          # overflow bin index
    A: np.ndarray = field(init=False)
    k: np.ndarray = field(init=False)

    def __post_init__(self):
        S = len(self.sig_m)
        self.k = np.array([len(m) for m in self.sig_m], int)
        kmax = int(self.k.max()) if S else 0
        self.A = np.zeros((S, kmax + 1, self.K + 1))
        for i, (m, ko) in enumerate(zip(self.sig_m, self.sig_ko)):
            for j in range(len(m) + 1):
                for sub in itertools.combinations(m, j):
                    self.A[i, j, min(int(ko) + sum(sub), self.K)] += 1.0
        self.w = np.asarray(self.w, float)

    @property
    def W(self) -> float:
        return float(self.w.sum())

    def with_weights(self, w: np.ndarray) -> "Template":
        t = object.__new__(Template)
        t.sig_m, t.sig_ko, t.K, t.A, t.k = self.sig_m, self.sig_ko, self.K, self.A, self.k
        t.w = np.asarray(w, float)
        return t

    def per_event(self, s: float, mu: float) -> np.ndarray:
        j = np.arange(self.A.shape[1])
        e = self.k[:, None] - j[None, :]
        coef = np.where(e >= 0, s ** j[None, :] * (1.0 - s) ** np.clip(e, 0, None), 0.0)
        p = np.einsum("sj,sjn->sn", coef, self.A)
        return fold_poisson(p, mu)

    def pmf(self, s: float, mu: float) -> tuple:
        """(M, V): mean pmf and the variance of its estimate from W events (for Barlow-Beeston-lite)."""
        q = self.per_event(s, mu)
        W = self.W
        M = self.w @ q / W
        V = np.clip(self.w @ (q * q) / W - M * M, 0.0, None) / W
        return M, V

    def eff(self, s: float) -> float:
        return float(self.w @ (1.0 - (1.0 - s) ** self.k) / self.W)

    def deff(self, s: float) -> float:
        return float(self.w @ (self.k * (1.0 - s) ** np.clip(self.k - 1, 0, None)) / self.W)


def mc_signatures(mctx: RunContext, method: str, window: tuple) -> pd.DataFrame:
    """One row per (label, file index, signature) with the number of veto-passing MC events."""
    ei = pd.read_parquet(mctx.parquet_path(f"{mctx.run_name}__eventindex_mc_{method}"),
                         columns=["_source_file", "eventNumber", "label", "file_run", "sourceX", "sourceY", "sourceZ",
                                  "is_cosmic"])
    ei = ei[~ei.is_cosmic].drop(columns="is_cosmic")
    c = pd.read_parquet(mctx.parquet_path(f"{mctx.run_name}__candidates_mc"),
                        columns=["_source_file", "eventNumber", "method", "t_med", "cap_dom"])
    c = c[(c.method == method) & (c.t_med >= window[0]) & (c.t_med <= window[1])]
    key = ["_source_file", "eventNumber"]
    ko = c[c.cap_dom < 0].groupby(key).size().rename("ko")
    size = c[c.cap_dom >= 0].groupby(key + ["cap_dom"]).size().rename("size").reset_index()
    size["size"] = size["size"].clip(upper=4)
    nsz = size.groupby(key + ["size"]).size().unstack(fill_value=0)
    nsz = nsz.reindex(columns=[1, 2, 3, 4], fill_value=0)
    nsz.columns = [f"n{k}" for k in nsz.columns]
    ev = ei.set_index(key).join(ko).join(nsz)
    ev["ko"] = ev.ko.fillna(0).astype(int)
    for k in range(1, 5):
        ev[f"n{k}"] = ev[f"n{k}"].fillna(0).astype(int)
    combos = ev[["n1", "n2", "n3", "n4"]].drop_duplicates()
    combos["m"] = [tuple(sorted(sum(([k] * int(r[f"n{k}"]) for k in range(1, 5)), []))) for _, r in combos.iterrows()]
    ev = ev.merge(combos, on=["n1", "n2", "n3", "n4"], how="left").drop(columns=["n1", "n2", "n3", "n4"])
    ev["file_idx"] = ev.file_run % 1000
    g = ev.groupby(["label", "sourceX", "sourceY", "sourceZ", "file_idx", "m", "ko"]).size().rename("w")
    return g.reset_index()


def template(sig: pd.DataFrame, label: str, K: int, files: Optional[tuple] = None) -> Template:
    q = sig[sig.label == label]
    if files is not None:
        q = q[(q.file_idx >= files[0]) & (q.file_idx <= files[1])]
    q = q.groupby(["m", "ko"]).w.sum().reset_index()
    return Template(list(q.m), q.ko.to_numpy(int), q.w.to_numpy(float), K)


# ---------------------------------------------------------------------------------------------- data
def factors(blk: dict) -> pd.DataFrame:
    p = Path(blk["factors_csv"])
    got = md5(p)
    if got != blk["factors_md5"]:
        raise SystemExit(f"[forwardfold] {p} md5 {got} is not the frozen {blk['factors_md5']}")
    f = pd.read_csv(p)
    f["F_shared_rel"] = 1.0 / np.sqrt(f.n_ic3000_bkg_g)
    f["F_src_rel"] = 1.0 / np.sqrt(f.n_ic3000)
    return f


def gated_triggers(dctx: RunContext, method: str, gate: str) -> pd.DataFrame:
    ei = pd.read_parquet(dctx.parquet_path(f"{dctx.run_name}__eventindex_{gate}_{method}"),
                         columns=["run", "trigger", "eventTankTime", "is_cosmic", "n_accepted_clusters"])
    return ei


def cluster_counts(dctx: RunContext, trig: pd.DataFrame, window: tuple) -> pd.DataFrame:
    """Veto-passing triggers of trig with the number of tree clusters with clusterTime in window."""
    out = []
    for run, t in trig[~trig.is_cosmic].groupby("run"):
        cf = pd.read_parquet(dctx.parquet_path(f"{dctx.run_name}__random_cfclusters_{run}"),
                             columns=["trigger", "clusterTime"])
        cf = cf[(cf.clusterTime >= window[0]) & (cf.clusterTime <= window[1])]
        n = cf.groupby("trigger").size()
        out.append(t.assign(n=t.trigger.map(n).fillna(0).astype(int).to_numpy()))
    return pd.concat(out, ignore_index=True)


def cluster_times(dctx: RunContext, trig: pd.DataFrame, window: tuple) -> pd.DataFrame:
    out = []
    for run, t in trig[~trig.is_cosmic].groupby("run"):
        cf = pd.read_parquet(dctx.parquet_path(f"{dctx.run_name}__random_cfclusters_{run}"),
                             columns=["trigger", "clusterTime"])
        cf = cf[(cf.clusterTime >= window[0]) & (cf.clusterTime <= window[1]) & cf.trigger.isin(t.trigger)]
        out.append(cf.assign(run=run))
    return pd.concat(out, ignore_index=True)


def hist(n: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(np.minimum(np.asarray(n, int), K), minlength=K + 1).astype(float)


# ---------------------------------------------------------------------------------------------- model
@dataclass
class Problem:
    runs: list
    groups: list
    gidx: np.ndarray        # group index per run
    tmpl: list              # Template per run
    d: np.ndarray           # (R, K+1)
    F: np.ndarray           # (R,)
    c: np.ndarray           # (K+1,) background-run counts
    pois_on_F: bool = False
    bb_share: str = "template"      # template: one Barlow-Beeston factor per (MC template, bin); run: per (run, bin)
    bb_model: str = "diag"          # diag: independent factor per bin; cov: one shift of the template pmf per
                                    # (template, group) with the template's multinomial covariance (bins anti-correlated)

    def __post_init__(self):
        uniq, self.t_idx = {}, np.empty(len(self.tmpl), int)
        for r, t in enumerate(self.tmpl):
            self.t_idx[r] = uniq.setdefault(id(t), len(uniq))
        ts = sorted(uniq, key=uniq.get)
        ts = [next(t for t in self.tmpl if id(t) == i) for i in ts]
        jmax = max(t.A.shape[1] for t in ts)
        K1 = self.d.shape[1]
        self._A = np.concatenate([np.pad(t.A, ((0, 0), (0, jmax - t.A.shape[1]), (0, 0))) for t in ts])
        self._k = np.concatenate([t.k for t in ts])
        self._W = np.array([t.W for t in ts])
        self._Wn = np.zeros((len(ts), len(self._k)))
        o = 0
        for i, t in enumerate(ts):
            self._Wn[i, o:o + len(t.w)] = t.w / t.W
            o += len(t.w)
        j = np.arange(jmax)
        self._e = self._k[:, None] - j[None, :]
        self._j = j
        assert self._A.shape[2] == K1
        if self.bb_share not in ("template", "run"):
            raise ValueError(f"bb_share {self.bb_share}")
        # runs with the same template and the same (s, mu) share one Barlow-Beeston factor per bin
        pairs = {}
        self._bkey = np.array([pairs.setdefault((int(t), int(g)), len(pairs)) for t, g in zip(self.t_idx, self.gidx)])
        self._nshare = np.bincount(self._bkey)[self._bkey].astype(float)
        if self.bb_model not in ("diag", "cov"):
            raise ValueError(f"bb_model {self.bb_model}")
        q, _ = np.linalg.qr(np.column_stack([np.ones(K1), np.eye(K1)[:, :K1 - 1]]))
        self._B = q[:, 1:]                       # orthonormal basis of the sum-zero shifts
        self._first = np.array([np.flatnonzero(self._bkey == t)[0] for t in range(int(self._bkey.max()) + 1)])
        self._C = None
        self._onehot = (np.arange(len(self._first))[:, None] == self._bkey[None, :]).astype(float)
        self._z0 = None

    def _MV(self, s: float, mu: float) -> tuple:
        coef = np.where(self._e >= 0, s ** self._j[None, :] * (1.0 - s) ** np.clip(self._e, 0, None), 0.0)
        q = (coef[:, :, None] * self._A).sum(axis=1)
        if mu != 0.0:
            q = q @ poisson_matrix(mu, q.shape[1])
        M = self._Wn @ q
        V = np.clip(self._Wn @ (q * q) - M * M, 0.0, None) / self._W[:, None]
        if self.bb_model == "cov":
            K1 = q.shape[1]
            Eqq = (self._Wn @ (q[:, :, None] * q[:, None, :]).reshape(len(q), K1 * K1)).reshape(-1, K1, K1)
            C = (Eqq - M[:, :, None] * M[:, None, :]) / self._W[:, None, None]
            return M, V, C
        return M, V

    @property
    def G(self):
        return len(self.groups)

    def unpack(self, x):
        G = self.G
        return x[:G], x[G:2 * G], np.exp(x[2 * G:])

    def predict(self, x):
        s, mu, th = self.unpack(x)
        R, K1 = self.d.shape
        M = np.empty((R, K1)); V = np.empty((R, K1))
        if self.bb_model == "cov":
            self._C = np.empty((R, K1, K1))
        for g in range(self.G):
            sel = self.gidx == g
            out = self._MV(s[g], mu[g])
            M[sel], V[sel] = out[0][self.t_idx[sel]], out[1][self.t_idx[sel]]
            if self.bb_model == "cov":
                self._C[sel] = out[2][self.t_idx[sel]]
        NA = self.d.sum(axis=1) - self.F * th.sum()
        nuA = NA[:, None] * M
        if self.pois_on_F:
            nuF = self.F[:, None] * np.array([fold_poisson(th, mu[g]) for g in self.gidx])
        else:
            nuF = self.F[:, None] * th[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            sig2 = np.where(M > 0, V / (M * M), 0.0)
        return nuA, nuF, sig2, NA

    @staticmethod
    def beta(nuA, nuF, d, sig2):
        a = nuA
        b = nuA * nuA * sig2 + nuF - nuA
        c0 = sig2 * nuA * (nuF - d) - nuF
        with np.errstate(invalid="ignore", divide="ignore"):
            disc = np.sqrt(np.clip(b * b - 4 * a * c0, 0, None))
            beta = np.where((sig2 > 0) & (a > 0), (-b + disc) / (2 * a), 1.0)
        return np.where(np.isfinite(beta) & (beta > 0), beta, 1.0)

    def beta_shared(self, nuA, nuF, sig2):
        """One factor per (template, bin), profiled over the runs that share the template (Newton, convex)."""
        key = self._bkey
        T = int(key.max()) + 1
        first = np.array([np.flatnonzero(key == t)[0] for t in range(T)])
        s2 = sig2[first]
        on = s2 > 0
        bt = np.ones_like(s2)
        d = self.d
        for _ in range(200):
            nu = bt[key] * nuA + nuF
            with np.errstate(divide="ignore", invalid="ignore"):
                g1 = np.where(nu > 0, nuA * (1.0 - d / nu), 0.0)
                g2 = np.where(nu > 0, d * nuA * nuA / (nu * nu), 0.0)
            G1 = np.zeros_like(bt); G2 = np.zeros_like(bt)
            np.add.at(G1, key, g1); np.add.at(G2, key, g2)
            with np.errstate(divide="ignore", invalid="ignore"):
                G1 = G1 + np.where(on, (bt - 1.0) / s2, 0.0)
                G2 = G2 + np.where(on, 1.0 / s2, 0.0)
                step = np.where(on & (G2 > 0), G1 / G2, 0.0)
            bt = np.clip(bt - step, 1e-3, None)
            if np.abs(step).max() < 1e-13:
                break
        with np.errstate(divide="ignore", invalid="ignore"):
            pen_t = np.where(on, (bt - 1.0) ** 2 / s2, 0.0)
        return bt[key], pen_t[key] / self._nshare[:, None]

    def shift_cov(self, nuA, nuF, NA):
        """Template pmf shift delta = B z per (template, group), Gaussian with the template covariance,
        profiled jointly over the runs sharing it (Newton with step halving)."""
        key, first, B, d = self._bkey, self._first, self._B, self.d
        T, K = len(first), B.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            Mt = nuA[first] / NA[first, None]
        S = np.einsum("nk,tnm,ml->tkl", B, self._C[first], B) + 1e-14 * np.eye(K)[None]
        P = np.linalg.inv(S)
        oh = self._onehot
        z = self._z0.copy() if self._z0 is not None and self._z0.shape == (T, K) else np.zeros((T, K))

        def f_of(zz):
            nu = NA[:, None] * (Mt[key] + (zz @ B.T)[key]) + nuF
            if (nu <= 0).any():
                return np.full(T, np.inf), nu
            with np.errstate(divide="ignore", invalid="ignore"):
                v = nu - d - np.where(d > 0, d * np.log(nu / d), 0.0)
            return oh @ v.sum(axis=1) + 0.5 * np.einsum("tk,tkl,tl->t", zz, P, zz), nu

        ft, nu = f_of(z)
        if not np.all(np.isfinite(ft)):
            z = np.zeros((T, K))
            ft, nu = f_of(z)
        for _ in range(60):
            with np.errstate(divide="ignore", invalid="ignore"):
                u = NA[:, None] * (1.0 - d / nu)
                w = NA[:, None] ** 2 * d / (nu * nu)
            U, Wd = oh @ u, oh @ w
            grad = U @ B + np.einsum("tkl,tl->tk", P, z)
            H = np.einsum("nk,tn,nl->tkl", B, Wd, B) + P
            step = np.linalg.solve(H, grad[..., None])[..., 0]
            lam = np.ones(T)
            for _h in range(40):
                zn = z - lam[:, None] * step
                fn, nun = f_of(zn)
                bad = ~np.isfinite(fn) | (fn > ft + 1e-9 * (1.0 + np.abs(ft)))
                if not bad.any():
                    break
                lam = np.where(bad, lam * 0.5, lam)
            z, ft, nu = zn, fn, nun
            if np.abs(step).max() < 1e-10:
                break
        self._z0 = z
        delta = (z @ B.T)[key]
        pen = np.zeros_like(nuA)
        pen[:, 0] = np.einsum("tk,tkl,tl->t", z, P, z)[key] / self._nshare
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(Mt[key] > 0, (Mt[key] + delta) / Mt[key], 1.0)
        return NA[:, None] * (Mt[key] + delta), beta, pen

    def terms(self, x):
        nuA, nuF, sig2, NA = self.predict(x)
        if self.bb_model == "cov":
            if (NA <= 0).any():
                return nuA + nuF, np.ones_like(nuA), np.zeros_like(nuA), nuA, nuF, NA
            amb, beta, pen = self.shift_cov(nuA, nuF, NA)
            return amb + nuF, beta, pen, nuA, nuF, NA
        if self.bb_share == "template":
            beta, pen = self.beta_shared(nuA, nuF, sig2)
        else:
            beta = self.beta(nuA, nuF, self.d, sig2)
            with np.errstate(divide="ignore", invalid="ignore"):
                pen = np.where(sig2 > 0, (beta - 1.0) ** 2 / sig2, 0.0)
        nu = beta * nuA + nuF
        return nu, beta, pen, nuA, nuF, NA

    def nll(self, x):
        """-ln L minus its saturated value (= deviance / 2), so the scale stays O(bins)."""
        nu, beta, pen, nuA, nuF, NA = self.terms(x)
        _, _, th = self.unpack(x)
        if (NA <= 0).any() or not np.all(np.isfinite(nu)) or (nu <= 0).any():
            bad = np.clip(-nu, 0, None).sum() + np.clip(-NA, 0, None).sum()
            return BIG * (1.0 + bad)
        d, c = self.d, self.c
        with np.errstate(divide="ignore", invalid="ignore"):
            v = float((nu - d - np.where(d > 0, d * np.log(nu / d), 0.0)).sum() + 0.5 * pen.sum())
            v += float((th - c - np.where(c > 0, c * np.log(th / c), 0.0)).sum())
        return v

    def deviance(self, x) -> tuple:
        nu, beta, pen, *_ = self.terms(x)
        _, _, th = self.unpack(x)
        d = self.d
        with np.errstate(divide="ignore", invalid="ignore"):
            dev_r = 2 * (nu - d + np.where(d > 0, d * np.log(d / nu), 0.0)).sum(axis=1) + pen.sum(axis=1)
            dev_c = 2 * float((th - self.c + np.where(self.c > 0, self.c * np.log(self.c / th), 0.0)).sum())
        return dev_r, dev_c


@dataclass
class FitResult:
    x: np.ndarray
    cov: np.ndarray
    nll: float
    ok: bool
    msg: str


def fit(pb: Problem, x0: Optional[np.ndarray] = None, hessian: bool = True, polish: bool = True) -> FitResult:
    from scipy.optimize import minimize
    G = pb.G
    if x0 is None:
        x0 = np.concatenate([np.full(G, 0.85), np.full(G, 0.02), np.log(pb.c + 0.5)])
    bounds = [(0.02, 1.2)] * G + [(-0.1, 2.0)] * G + [(None, None)] * len(pb.c)
    r = minimize(pb.nll, x0, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=5000, maxfun=200000, ftol=1e-14, gtol=1e-7, eps=1e-7))
    best = r
    if polish:
        r2 = minimize(pb.nll, r.x, method="Powell", bounds=bounds,
                      options=dict(maxiter=20000, xtol=1e-8, ftol=1e-13))
        best = r2 if r2.fun <= r.fun else r
    x = best.x
    cov = np.full((len(x), len(x)), np.nan)
    if hessian:
        # empty background-run bins fit to theta -> 0: a flat, uncoupled direction in log theta; the
        # pseudo-inverse drops it and leaves the s, mu block as the inverse of the rest
        H = num_hessian(pb.nll, x)
        if np.all(np.isfinite(H)):
            cov = np.linalg.pinv(H, rcond=1e-10, hermitian=True)
    sm = np.diag(cov)[:2 * G]
    ok = bool(np.isfinite(best.fun) and best.fun < BIG and np.all(np.isfinite(sm)) and
              np.all(sm > 0)) if hessian else bool(best.fun < BIG)
    return FitResult(x, cov, float(best.fun), ok, str(best.message))


def num_hessian(f, x, h: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(x)
    h = np.full(n, 1e-3) if h is None else h
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


# ---------------------------------------------------------------------------------------------- set-up
@dataclass
class Setup:
    blk: dict
    K: int
    window: tuple
    fac: pd.DataFrame
    sig: pd.DataFrame
    labels: dict            # run -> MC label
    label_dy: dict          # run -> MC y - data y
    counts: pd.DataFrame    # source runs + background run, veto-passing gated triggers with n
    bkg_run: int
    dctx: RunContext
    mctx: RunContext


def nearest_label(sig: pd.DataFrame, x: float, y: float, z: float) -> tuple:
    p = sig.drop_duplicates("label")[["label", "sourceX", "sourceY", "sourceZ"]]
    q = p[(np.abs(p.sourceX - x) < 1) & (np.abs(p.sourceZ - z) < 1)]
    if q.empty:
        raise SystemExit(f"[forwardfold] no MC position at x={x:g}, z={z:g}")
    i = int(np.argmin(np.abs(q.sourceY.to_numpy() - y)))
    return str(q.label.iloc[i]), float(q.sourceY.iloc[i] - y)


def setup(ctx: RunContext, window: Optional[tuple] = None) -> Setup:
    blk = dict((ctx.extra or {}).get("fold") or {})
    if not blk:
        raise SystemExit(f"[forwardfold] {ctx.config_path} has no fold: block")
    K = int(blk["n_bins"]) - 1
    window = tuple(float(v) for v in (window or blk["window_ns"]))
    dctx, mctx = load_context(blk["data_config"]), load_context(blk["mc_config"])
    fac = factors(blk)
    sig = mc_signatures(mctx, blk["mc_method"], window)
    labels, dy = {}, {}
    for r in fac.itertuples():
        labels[r.run], dy[r.run] = nearest_label(sig, r.x, r.y, r.z)
    bkg = int(blk["background_run"])
    trig = gated_triggers(dctx, blk["data_method"], "ic")
    trig = trig[trig.run.isin(list(fac.run) + [bkg])]
    counts = cluster_counts(dctx, trig, window)
    nominal = tuple(float(v) for v in blk["window_ns"])
    if window == nominal:
        bad = (counts.n != counts.n_accepted_clusters).sum()
        if bad:
            raise SystemExit(f"[forwardfold] {bad} triggers: cluster count differs from the clusterindex table")
    missing = set(fac.run) - set(counts.run)
    if missing:
        raise SystemExit(f"[forwardfold] runs without gated triggers: {sorted(missing)}")
    return Setup(blk, K, window, fac, sig, labels, dy, counts, bkg, dctx, mctx)


def group_of(fac: pd.DataFrame, by: str) -> pd.Series:
    if by == "port":
        return fac.port.astype(int).map(lambda p: f"port{p}")
    if by == "position":
        return fac.apply(lambda r: f"port{int(r.port)}_y{r.y:+g}", axis=1)
    raise SystemExit(f"[forwardfold] unknown grouping {by}")


def problem(su: Setup, by: str = "port", F_col: str = "F_rsb_g", F_scale: Optional[np.ndarray] = None,
            pois_on_F: bool = False, tmpls: Optional[dict] = None, d: Optional[np.ndarray] = None,
            c: Optional[np.ndarray] = None, runs: Optional[list] = None, bb_share: Optional[str] = None) -> Problem:
    fac = su.fac if runs is None else su.fac[su.fac.run.isin(runs)]
    fac = fac.reset_index(drop=True)
    gs = group_of(fac, by)
    groups = sorted(gs.unique())
    if tmpls is None:
        tmpls = {lab: template(su.sig, lab, su.K) for lab in sorted(set(su.labels.values()))}
    if d is None:
        d = np.array([hist(su.counts.n[su.counts.run == r], su.K) for r in fac.run])
    if c is None:
        c = hist(su.counts.n[su.counts.run == su.bkg_run], su.K)
    F = fac[F_col].to_numpy(float) * (1.0 if F_scale is None else np.asarray(F_scale))
    return Problem(list(fac.run), groups, np.array([groups.index(g) for g in gs]),
                   [tmpls[su.labels[r]] for r in fac.run], d, F, c, pois_on_F,
                   bb_share or su.blk.get("bb_share", "template"), su.blk.get("bb_model", "diag"))


def params_table(pb: Problem, fr: FitResult, variant: str) -> pd.DataFrame:
    G = pb.G
    err = np.sqrt(np.clip(np.diag(fr.cov), 0, None))
    rows = []
    dev_r, dev_c = pb.deviance(fr.x)
    for g, name in enumerate(pb.groups):
        cv = fr.cov[g, G + g] / (err[g] * err[G + g]) if err[g] > 0 and err[G + g] > 0 else np.nan
        sel = pb.gidx == g
        rows.append(dict(variant=variant, group=name, runs=" ".join(str(r) for r, k in zip(pb.runs, sel) if k),
                         s=fr.x[g], s_err=err[g], mu_ind=fr.x[G + g], mu_ind_err=err[G + g], corr_s_mu=cv,
                         deviance=float(dev_r[sel].sum()), ndf=int(sel.sum() * (pb.d.shape[1] - 1) - 2),
                         nll=fr.nll, fit_ok=fr.ok))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- closure
def _toy_one(args):
    pb_kw, seed, s_true, mu_true, W_tmpl, parents, d_tot, c_obs, F = args
    rng = np.random.default_rng(seed)
    K1 = len(c_obs)
    N_F = c_obs.sum()
    fF = c_obs / N_F
    c_toy = rng.multinomial(int(N_F), fF).astype(float)
    tm = {lab: parents[lab].with_weights(rng.multinomial(W_tmpl[lab], parents[lab].w / parents[lab].W))
          for lab in sorted(parents)}
    d = np.empty((len(d_tot), K1))
    for r, (lab, N) in enumerate(zip(pb_kw["labels"], d_tot)):
        M, _ = parents[lab].pmf(s_true, mu_true)
        nF = min(rng.poisson(F[r] * N_F), int(N))
        d[r] = rng.multinomial(int(N) - nF, np.clip(M, 0, None) / np.clip(M, 0, None).sum()) + rng.multinomial(nF, fF)
    pb = Problem(pb_kw["runs"], pb_kw["groups"], pb_kw["gidx"], [tm[l] for l in pb_kw["labels"]], d, F, c_toy,
                 bb_share=pb_kw["bb_share"], bb_model=pb_kw["bb_model"])
    fr = fit(pb)
    G = pb.G
    err = np.sqrt(np.clip(np.diag(fr.cov), 0, None))
    out = []
    for g, name in enumerate(pb.groups):
        out.append(dict(seed=seed, s_true=s_true, mu_true=mu_true, group=name, s=fr.x[g], s_err=err[g],
                        mu_ind=fr.x[G + g], mu_ind_err=err[G + g], fit_ok=fr.ok))
    return out


def closure(ctx: RunContext, su: Setup, n_toys: int, jobs: int) -> None:
    from multiprocessing import Pool
    cb = su.blk["closure"]
    tf, pf = tuple(cb["template_files"]), tuple(cb["pseudo_files"])
    base = problem(su)
    labs = [su.labels[r] for r in base.runs]
    tA = {lab: template(su.sig, lab, su.K, tf) for lab in sorted(set(labs))}
    tB = {lab: template(su.sig, lab, su.K, pf) for lab in sorted(set(labs))}
    parent = {lab: template(su.sig, lab, su.K) for lab in sorted(set(labs))}
    N = base.d.sum(axis=1)
    N_F = base.c.sum()
    rows_a = []
    for s_true in cb["s"]:
        for mu_true in cb["mu"]:
            d = np.array([(N[r] - base.F[r] * N_F) * tB[lab].pmf(s_true, mu_true)[0] + base.F[r] * base.c
                          for r, lab in enumerate(labs)])
            pb = Problem(base.runs, base.groups, base.gidx, [tA[l] for l in labs], d, base.F, base.c.copy(),
                         bb_share=base.bb_share, bb_model=base.bb_model)
            fr = fit(pb)
            t = params_table(pb, fr, "asimov").assign(s_true=s_true, mu_true=mu_true)
            rows_a.append(t)
            print(f"[forwardfold] asimov s={s_true:g} mu={mu_true:g}: "
                  + ", ".join(f"{r.group} s {r.s:.4f}±{r.s_err:.4f} mu {r.mu_ind:+.4f}±{r.mu_ind_err:.4f}"
                              for r in t.itertuples()), flush=True)
    asimov = pd.concat(rows_a, ignore_index=True)
    asimov["s_pull"] = (asimov.s - asimov.s_true) / asimov.s_err
    asimov["mu_pull"] = (asimov.mu_ind - asimov.mu_true) / asimov.mu_ind_err
    asimov.to_csv(ctx.csv_path("forwardfold_closure_asimov"), index=False)
    W_tmpl = {lab: int(round(tA[lab].W)) for lab in tA}
    kw = dict(runs=base.runs, groups=base.groups, gidx=base.gidx, labels=labs, bb_share=base.bb_share,
              bb_model=base.bb_model)
    seed0 = int(cb["seed"])
    tasks = []
    for i, (s_true, mu_true) in enumerate(itertools.product(cb["s"], cb["mu"])):
        for t in range(n_toys):
            tasks.append((kw, seed0 + 100000 * i + t, s_true, mu_true, W_tmpl, parent, N, base.c, base.F))
    rows = []
    with Pool(jobs) as pool:
        for k, out in enumerate(pool.imap_unordered(_toy_one, tasks, chunksize=4)):
            rows += out
            if (k + 1) % 100 == 0:
                print(f"[forwardfold] toys {k + 1}/{len(tasks)}", flush=True)
    toys = pd.DataFrame(rows).sort_values(["s_true", "mu_true", "seed", "group"]).reset_index(drop=True)
    toys["s_pull"] = (toys.s - toys.s_true) / toys.s_err
    toys["mu_pull"] = (toys.mu_ind - toys.mu_true) / toys.mu_ind_err
    toys.to_csv(ctx.csv_path("forwardfold_closure_toys"), index=False)
    summ = closure_summary(toys, su.blk.get("pass") or {})
    summ.to_csv(ctx.csv_path("forwardfold_closure_summary"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ.to_string(index=False))


def closure_summary(toys: pd.DataFrame, rule: dict) -> pd.DataFrame:
    mp, wl, wh = float(rule.get("mean_pull_max", 0.2)), float(rule.get("width_min", 0.9)), float(rule.get("width_max", 1.1))
    rows = []
    ok = toys[toys.fit_ok]
    for (s_true, mu_true), g in toys.groupby(["s_true", "mu_true"]):
        q = g[g.fit_ok]
        for par in ("s", "mu"):
            col = f"{par}_pull"
            for name, h in [("pooled", q)] + list(q.groupby("group")):
                p = h[col].to_numpy()
                n = len(p)
                m, sd = float(np.mean(p)), float(np.std(p, ddof=1))
                val = h["s" if par == "s" else "mu_ind"]
                row = dict(s_true=s_true, mu_true=mu_true, param=par, group=name, n_toys=n,
                           n_failed=int((~g.fit_ok).sum()) if name == "pooled" else int((~g[g.group == name].fit_ok).sum()),
                           mean_pull=m, mean_pull_err=sd / np.sqrt(n), pull_width=sd, pull_width_err=sd / np.sqrt(2 * (n - 1)),
                           mean_fit=float(val.mean()), mean_err=float(h[f"{'s' if par == 's' else 'mu_ind'}_err"].mean()),
                           rms_fit=float(val.std(ddof=1)))
                if name == "pooled":
                    row["pass"] = bool(abs(m) < mp and wl <= sd <= wh)
                else:
                    row["flag_3se"] = bool(abs(m) > 3 * row["mean_pull_err"] or abs(sd - 1) > 3 * row["pull_width_err"])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- data fit
VARIANTS = [
    ("nominal", {}),
    ("F_alpha", dict(F_col="alpha")),
    ("F_raw", dict(F_col="F_rsb_raw")),
    ("F_shared_up", dict(F_scale="shared_up")),
    ("F_shared_down", dict(F_scale="shared_down")),
    ("pois_on_F", dict(pois_on_F=True)),
    ("by_position", dict(by="position")),
]


def data_fit(ctx: RunContext, su: Setup) -> None:
    rows, effs = [], []
    nominal_pb = nominal_fr = None
    for name, kw in VARIANTS:
        kw = dict(kw)
        sc = kw.pop("F_scale", None)
        if sc == "shared_up":
            kw["F_scale"] = 1.0 + su.fac.F_shared_rel.to_numpy()
        elif sc == "shared_down":
            kw["F_scale"] = 1.0 - su.fac.F_shared_rel.to_numpy()
        pb = problem(su, **kw)
        fr = fit(pb)
        t = params_table(pb, fr, name)
        rows.append(t)
        effs.append(efficiency_table(su, pb, fr, name))
        print(f"[forwardfold] {name}: " + ", ".join(f"{r.group} s {r.s:.4f}±{r.s_err:.4f} mu {r.mu_ind:+.4f}±"
                                                     f"{r.mu_ind_err:.4f}" for r in t.itertuples()), flush=True)
        if name == "nominal":
            nominal_pb, nominal_fr = pb, fr
    # tail robustness: bins 0, 1, >= 2 only
    K2 = 2
    t2 = {lab: template(su.sig, lab, K2) for lab in sorted(set(su.labels.values()))}
    d2 = np.array([hist(su.counts.n[su.counts.run == r], K2) for r in su.fac.run])
    c2 = hist(su.counts.n[su.counts.run == su.bkg_run], K2)
    pb2 = problem(su, tmpls=t2, d=d2, c=c2)
    fr2 = fit(pb2)
    rows.append(params_table(pb2, fr2, "bins_0_1_2plus"))
    effs.append(efficiency_table(su, pb2, fr2, "bins_0_1_2plus"))
    # start-point check of the nominal minimum
    ms = []
    G = nominal_pb.G
    for s0, m0 in ((0.70, 0.0), (1.0, 0.05), (0.9, 0.0), (0.75, 0.04)):
        x0 = np.concatenate([np.full(G, s0), np.full(G, m0), np.log(nominal_pb.c + 0.5)])
        f2 = fit(nominal_pb, x0=x0, hessian=False)
        ms.append(dict(s0=s0, mu0=m0, dnll=f2.nll - nominal_fr.nll, max_ds=float(np.abs(f2.x[:G] - nominal_fr.x[:G]).max()),
                       max_dmu=float(np.abs(f2.x[G:2 * G] - nominal_fr.x[G:2 * G]).max())))
    pd.DataFrame(ms).to_csv(ctx.csv_path("forwardfold_multistart"), index=False)
    print("[forwardfold] start-point check:\n" + pd.DataFrame(ms).to_string(index=False), flush=True)
    # per-run source part of F, one run at a time, shifts added in quadrature per group
    shifts = []
    for i, r in enumerate(su.fac.run):
        sc = np.ones(len(su.fac)); sc[i] += su.fac.F_src_rel.iloc[i]
        pb = problem(su, F_scale=sc)
        fr = fit(pb, x0=nominal_fr.x, hessian=False)
        G = pb.G
        shifts.append(pd.DataFrame(dict(run=r, group=pb.groups, ds=fr.x[:G] - nominal_fr.x[:G],
                                        dmu=fr.x[G:2 * G] - nominal_fr.x[G:2 * G])))
    sh = pd.concat(shifts, ignore_index=True)
    sh.to_csv(ctx.csv_path("forwardfold_F_src_shifts"), index=False)
    q = sh.groupby("group").agg(ds=("ds", lambda v: float(np.sqrt((v ** 2).sum()))),
                                dmu=("dmu", lambda v: float(np.sqrt((v ** 2).sum())))).reset_index()
    params = pd.concat(rows, ignore_index=True)
    params.to_csv(ctx.csv_path("forwardfold_params"), index=False)
    eff = pd.concat(effs, ignore_index=True)
    eff.to_csv(ctx.csv_path("forwardfold_efficiency"), index=False)
    bins_table(nominal_pb, nominal_fr, su).to_csv(ctx.csv_path("forwardfold_bins"), index=False)
    syst = systematics(params, q)
    syst.to_csv(ctx.csv_path("forwardfold_systematics"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        print(params.to_string(index=False))
        print(syst.to_string(index=False))


def efficiency_table(su: Setup, pb: Problem, fr: FitResult, variant: str) -> pd.DataFrame:
    G = pb.G
    err = np.sqrt(np.clip(np.diag(fr.cov), 0, None))
    rows = []
    fac = su.fac.set_index("run")
    seen = {}
    for r, g, t in zip(pb.runs, pb.gidx, pb.tmpl):
        lab = su.labels[r]
        k = (lab, g)
        seen.setdefault(k, dict(variant=variant, group=pb.groups[g], mc_label=lab, x=fac.x[r], y=fac.y[r] + su.label_dy[r],
                                z=fac.z[r], runs=[], data_y=[], s=fr.x[g], s_err=err[g], eps_mc=t.eff(1.0),
                                eps_data=t.eff(fr.x[g]), eps_data_err=abs(t.deff(fr.x[g])) * err[g], mc_events=t.W))
        seen[k]["runs"].append(str(r))
        seen[k]["data_y"].append(f"{fac.y[r]:g}")
    for v in seen.values():
        v["runs"] = " ".join(v["runs"])
        v["data_y"] = " ".join(v["data_y"])
        rows.append(v)
    return pd.DataFrame(rows)


def bins_table(pb: Problem, fr: FitResult, su: Setup) -> pd.DataFrame:
    nu, beta, pen, nuA, nuF, NA = pb.terms(fr.x)
    dev_r, dev_c = pb.deviance(fr.x)
    rows = []
    for i, r in enumerate(pb.runs):
        for n in range(pb.d.shape[1]):
            rows.append(dict(run=r, group=pb.groups[pb.gidx[i]], n=n, overflow=n == pb.d.shape[1] - 1, data=pb.d[i, n],
                             pred=nu[i, n], pred_ambe=beta[i, n] * nuA[i, n], pred_falsestart=nuF[i, n], beta=beta[i, n],
                             N_trig=pb.d[i].sum(), N_A=NA[i], F=pb.F[i], run_deviance=dev_r[i]))
    _, _, th = pb.unpack(fr.x)
    for n in range(pb.d.shape[1]):
        rows.append(dict(run=su.bkg_run, group="background", n=n, overflow=n == pb.d.shape[1] - 1, data=pb.c[n],
                         pred=th[n], N_trig=pb.c.sum(), run_deviance=dev_c))
    return pd.DataFrame(rows)


def systematics(params: pd.DataFrame, fsrc: pd.DataFrame) -> pd.DataFrame:
    nom = params[params.variant == "nominal"].set_index("group")
    rows = []
    for g in nom.index:
        row = dict(group=g, s=nom.s[g], s_stat=nom.s_err[g], mu_ind=nom.mu_ind[g], mu_stat=nom.mu_ind_err[g])
        for v in ("F_alpha", "F_raw", "pois_on_F"):
            q = params[(params.variant == v) & (params.group == g)]
            row[f"ds_{v}"] = float(q.s.iloc[0] - nom.s[g]) if len(q) else np.nan
            row[f"dmu_{v}"] = float(q.mu_ind.iloc[0] - nom.mu_ind[g]) if len(q) else np.nan
        up = params[(params.variant == "F_shared_up") & (params.group == g)]
        dn = params[(params.variant == "F_shared_down") & (params.group == g)]
        row["ds_F_shared"] = float((up.s.iloc[0] - dn.s.iloc[0]) / 2)
        row["dmu_F_shared"] = float((up.mu_ind.iloc[0] - dn.mu_ind.iloc[0]) / 2)
        f = fsrc.set_index("group")
        row["ds_F_src"], row["dmu_F_src"] = float(f.ds[g]), float(f.dmu[g])
        row["s_syst_F"] = float(np.sqrt(row["ds_F_shared"] ** 2 + row["ds_F_src"] ** 2 + max(row["ds_F_raw"] ** 2, 0)))
        row["mu_syst_F"] = float(np.sqrt(row["dmu_F_shared"] ** 2 + row["dmu_F_src"] ** 2 + row["dmu_F_raw"] ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- checks
def _expflat(h: np.ndarray, var: np.ndarray, t: np.ndarray, w: float) -> dict:
    from scipy.optimize import least_squares
    ok = var > 0
    def res(p):
        A, tau, C = p
        return ((A * np.exp(-t / tau) + C * w - h) / np.sqrt(var))[ok]
    C0 = max(float(np.mean(h[-10:])) / w, 1e-6)
    r = least_squares(res, [max(h[0] - C0 * w, 1.0), 29.0, C0], bounds=([0, 1, -np.inf], [np.inf, 200, np.inf]))
    J = r.jac
    cov = np.linalg.pinv(J.T @ J)
    chi2 = float((r.fun ** 2).sum())
    return dict(A=r.x[0], tau_us=r.x[1], tau_err=float(np.sqrt(cov[1, 1])), C_per_us=r.x[2],
                C_err=float(np.sqrt(cov[2, 2])), chi2=chi2, ndf=int(ok.sum()) - 3)


def _neutcapture(h: np.ndarray, var: np.ndarray, t: np.ndarray) -> dict:
    """The campaign capture-time model A (1 - exp(-t/therm)) exp(-t/tau) + B (plots.basic NeutCapture), 1 us bins."""
    from scipy.optimize import least_squares
    ok = var > 0

    def res(p):
        A, th, tau, B = p
        return ((A * (1 - np.exp(-t / th)) * np.exp(-t / tau) + B - h) / np.sqrt(var))[ok]
    p0 = [2.0 * h.max(), 3.0, 29.0, 0.2 * float(np.mean(h[-8:]))]
    r = least_squares(res, p0, bounds=([0, 0.05, 1, -np.inf], [np.inf, 30, 200, np.inf]))
    cov = np.linalg.pinv(r.jac.T @ r.jac)
    e = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(A=r.x[0], therm_us=r.x[1], therm_err=e[1], tau_us=r.x[2], tau_err=e[2], C_per_us=r.x[3], C_err=e[3],
                chi2=float((r.fun ** 2).sum()), ndf=int(ok.sum()) - 4)


def _neutcapture_lmfit(h: np.ndarray, var: np.ndarray, t: np.ndarray) -> dict:
    """_neutcapture through lmfit (method least_squares, same start values, bounds and weights)."""
    import lmfit
    from ..plots.basic import NeutCapture
    ok = var > 0
    m = lmfit.Model(NeutCapture)
    p = m.make_params(A=2.0 * h.max(), therm=3.0, tau=29.0, B=0.2 * float(np.mean(h[-8:])))
    for k, lo, hi in (("A", 0, np.inf), ("therm", 0.05, 30), ("tau", 1, 200), ("B", -np.inf, np.inf)):
        p[k].min, p[k].max = lo, hi
    r = m.fit(h[ok], p, t=t[ok], weights=1.0 / np.sqrt(var[ok]), method="least_squares", scale_covar=False)
    e = {k: float(r.params[k].stderr) if r.params[k].stderr is not None else np.nan for k in r.params}
    v = {k: float(r.params[k].value) for k in r.params}
    return dict(A=v["A"], therm_us=v["therm"], therm_err=e["therm"], tau_us=v["tau"], tau_err=e["tau"],
                C_per_us=v["B"], C_err=e["B"], chi2=float(r.chisqr), ndf=int(ok.sum()) - 4)


TIME_FITTERS = {"scipy": ("scipy.optimize.least_squares", _neutcapture),
                "lmfit": ("lmfit least_squares", _neutcapture_lmfit)}


def time_spectrum(ctx: RunContext, su: Setup, fit_params: Optional[pd.DataFrame]) -> pd.DataFrame:
    w_us = 1.0
    lo = su.window[0] / 1000.0
    hi = float(su.blk.get("time_fit_max_us", su.window[1] / 1000.0))
    t_eff = float(su.blk.get("flat_window_us", (su.window[1] - su.window[0]) / 1000.0))
    edges = np.arange(lo, hi + 1e-9, w_us)
    t = 0.5 * (edges[1:] + edges[:-1])
    trig = gated_triggers(su.dctx, su.blk["data_method"], "ic")
    bkg = trig[trig.run == su.bkg_run]
    tb = cluster_times(su.dctx, bkg, su.window)
    hb = np.histogram(tb.clusterTime / 1000.0, edges)[0].astype(float)
    nb = int((~bkg.is_cosmic).sum())
    dummy = int(su.blk["dummy_run"])
    tn = gated_triggers(su.dctx, su.blk["data_method"], "none")
    dm = tn[tn.run == dummy]
    td = cluster_times(su.dctx, dm, su.window)
    hd = np.histogram(td.clusterTime / 1000.0, edges)[0].astype(float)
    nd = int((~dm.is_cosmic).sum())
    c_dummy, c_dummy_err = hd.sum() / nd / (hi - lo), np.sqrt(hd.sum()) / nd / (hi - lo)
    sl = np.polyfit(t, hd / nd, 1, w=1 / np.sqrt(np.clip(hd, 1, None)) * nd, cov=True)
    rd = dict(slope=sl[0][0], slope_err=float(np.sqrt(sl[1][0, 0])))
    flat_dummy = c_dummy
    backend = str(su.blk.get("time_fit_backend", "scipy"))
    if backend not in TIME_FITTERS:
        raise SystemExit(f"[forwardfold] fold.time_fit_backend must be one of {sorted(TIME_FITTERS)}, got {backend}")
    fitter_name, fitter = TIME_FITTERS[backend]
    rows, spec = [], []
    fac = su.fac.assign(group=group_of(su.fac, "port"))
    for g, f in fac.groupby("group"):
        tr = trig[trig.run.isin(f.run)]
        ts = cluster_times(su.dctx, tr, su.window)
        hs = np.histogram(ts.clusterTime / 1000.0, edges)[0].astype(float)
        N = int((~tr.is_cosmic).sum())
        Fsum = float(f.F_rsb_g.sum())
        h = hs - Fsum * hb
        var = hs + Fsum ** 2 * hb
        N_A = N - Fsum * nb
        r = fitter(h, np.clip(var, 1, None), t)
        c = r["C_per_us"] / N_A
        mu_t = (c - c_dummy) * t_eff
        mu_t_err = np.hypot(r["C_err"] / N_A, c_dummy_err) * t_eff
        row = dict(group=g, runs=" ".join(str(x) for x in f.run), N_trig=N, N_A=N_A, A_fit=r["A"], C_fit=r["C_per_us"],
                   therm_us=r["therm_us"], therm_err=r["therm_err"], flat_window_us=t_eff, tau_us=r["tau_us"],
                   tau_err=r["tau_err"], flat_per_trig_us=c, flat_err=r["C_err"] / N_A, chi2=r["chi2"], ndf=r["ndf"],
                   dummy_flat_per_trig_us=c_dummy, dummy_flat_err=c_dummy_err, dummy_mean_per_trig_us=flat_dummy,
                   dummy_slope_per_us2=rd["slope"], dummy_slope_err=rd["slope_err"], dummy_trig=nd,
                   mu_ind_time=mu_t, mu_ind_time_err=mu_t_err)
        if backend != "scipy":
            row["fitter"] = fitter_name
        if fit_params is not None:
            q = fit_params[(fit_params.variant == "nominal") & (fit_params.group == g)]
            if len(q):
                row["mu_ind_fold"], row["mu_ind_fold_err"] = float(q.mu_ind.iloc[0]), float(q.mu_ind_err.iloc[0])
                row["diff_time_minus_fold"] = mu_t - row["mu_ind_fold"]
        rows.append(row)
        spec += [dict(group=g, t_lo_us=edges[i], t_hi_us=edges[i + 1], source=hs[i], background_scaled=Fsum * hb[i],
                      dummy=hd[i], dummy_trig=nd, N_A=N_A) for i in range(len(t))]
    pd.DataFrame(spec).to_csv(ctx.csv_path("forwardfold_timespectrum_bins"), index=False)
    out = pd.DataFrame(rows)
    out.to_csv(ctx.csv_path("forwardfold_timespectrum"), index=False)
    return out


def ic3000_tank(ctx: RunContext, su: Setup) -> pd.DataFrame:
    """Veto-passing IC > 3000 triggers (source-free BGO region): source run vs F_r x background run (gain g_r)."""
    ic_cut = float(su.blk.get("ic_cosmic", 3000.0))
    tag = su.blk["features_tag"]
    tn = gated_triggers(su.dctx, su.blk["data_method"], "none")

    def ic(run):
        w = pd.read_parquet(f"TriggerSummary/WaveformFeatures_{tag}_{run}.parquet", columns=["timestamp", "IC_adjusted"])
        w = w[w.timestamp > 0].drop_duplicates("timestamp")
        t = tn[tn.run == run].merge(w, left_on="eventTankTime", right_on="timestamp", how="inner")
        return t

    b = ic(su.bkg_run)
    rows = []
    for r in su.fac.itertuples():
        s = ic(r.run)
        sv = s[(s.IC_adjusted > ic_cut)]
        bv = b[(b.IC_adjusted * r.g > ic_cut)]
        ns_all, nb_all = len(sv), len(bv)
        fs, fb = float((~sv.is_cosmic).mean()), float((~bv.is_cosmic).mean())
        sv, bv = sv[~sv.is_cosmic], bv[~bv.is_cosmic]
        ms, mb = float(sv.n_accepted_clusters.mean()), float(bv.n_accepted_clusters.mean())
        rows.append(dict(run=r.run, port=r.port, x=r.x, y=r.y, z=r.z, F=r.F_rsb_g, n_ic3000=ns_all, n_ic3000_bkg_g=nb_all,
                         join_frac=ns_all / r.n_ic3000, join_frac_bkg=nb_all / r.n_ic3000_bkg_g,
                         pass_frac=fs, pass_frac_bkg=fb, pass_frac_ratio=fs / fb,
                         n_pass=len(sv), n_pass_bkg_g=len(bv), count_closure=len(sv) / (r.F_rsb_g * len(bv)),
                         count_closure_err=len(sv) / (r.F_rsb_g * len(bv)) * np.sqrt(1 / max(len(sv), 1) + 1 / max(len(bv), 1)),
                         mean_clusters=ms, mean_clusters_err=float(sv.n_accepted_clusters.std(ddof=1) / np.sqrt(max(len(sv), 1))),
                         mean_clusters_bkg=mb, mean_clusters_bkg_err=float(bv.n_accepted_clusters.std(ddof=1) / np.sqrt(max(len(bv), 1))),
                         p0=float((sv.n_accepted_clusters == 0).mean()), p0_bkg=float((bv.n_accepted_clusters == 0).mean())))
    df = pd.DataFrame(rows)
    df["mu_ind_ic3000"] = df.mean_clusters - df.mean_clusters_bkg
    df["mu_ind_ic3000_err"] = np.hypot(df.mean_clusters_err, df.mean_clusters_bkg_err)
    df.to_csv(ctx.csv_path("forwardfold_ic3000tank_runs"), index=False)
    prow = []
    for p, g in df.groupby("port"):
        w = 1 / g.mu_ind_ic3000_err ** 2
        prow.append(dict(port=int(p), runs=" ".join(str(x) for x in g.run), n_pass=int(g.n_pass.sum()),
                         expected_pass=float((g.F * g.n_pass_bkg_g).sum()),
                         count_closure=float(g.n_pass.sum() / (g.F * g.n_pass_bkg_g).sum()),
                         mu_ind_ic3000=float((w * g.mu_ind_ic3000).sum() / w.sum()), mu_ind_ic3000_err=float(1 / np.sqrt(w.sum()))))
    pdf = pd.DataFrame(prow)
    pdf.to_csv(ctx.csv_path("forwardfold_ic3000tank_ports"), index=False)
    return pdf


# ---------------------------------------------------------------------------------------------- plots
def plots(ctx: RunContext) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ..plotting import two_titles
    out = ctx.plots_dir / "forwardfold_step3.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    rd = lambda b: pd.read_csv(ctx.csv_path(b)) if Path(ctx.csv_path(b)).exists() else None  # noqa: E731
    bins, eff, ts, tsb, cs = (rd(b) for b in ("forwardfold_bins", "forwardfold_efficiency", "forwardfold_timespectrum",
                                                "forwardfold_timespectrum_bins", "forwardfold_closure_summary"))
    with PdfPages(out) as pdf:
        if bins is not None:
            b = bins[bins.group != "background"].groupby(["group", "n"])[["data", "pred"]].sum().reset_index()
            fig, ax = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
            for a, (g, q) in zip(ax, b.groupby("group")):
                a.errorbar(q.n, q.data, np.sqrt(q.data), fmt="o", ms=4, color="k", label="data")
                a.plot(q.n, q.pred, "-", color="C0", label="fit")
                a.set_yscale("log"); a.set_xlabel("clusters per trigger (last bin ≥ 4)"); a.set_title(g)
            ax[0].set_ylabel("veto-passing IC-gated triggers"); ax[0].legend(frameon=False)
            two_titles(fig, "Cluster multiplicity per port: data vs forward fold",
                       "TA ClusterFinder + C++ merge clusters, 2-67 µs, no box; fit = N_A·[MC(s)⊛Poisson(μ_ind)] + F·6264")
            pdf.savefig(fig); plt.close(fig)
        if eff is not None:
            e = eff[eff.variant == "nominal"]
            fig, ax = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
            for a, (g, q) in zip(ax, e.groupby("group")):
                q = q.sort_values("y")
                a.plot(q.y, q.eps_mc, "s-", ms=4, color="C1", label="MC truth")
                a.errorbar(q.y, q.eps_data, q.eps_data_err, fmt="o-", ms=4, color="k", label="data (s·MC)")
                a.set_xlabel("source height y [cm]"); a.set_title(g)
            ax[0].set_ylabel("efficiency"); ax[0].legend(frameon=False)
            two_titles(fig, "Neutron detection efficiency per position",
                       "fraction of veto-passing AmBe triggers with ≥ 1 capture cluster in 2-67 µs; data = MC thinned by s_port")
            pdf.savefig(fig); plt.close(fig)
        if tsb is not None and ts is not None:
            fig, ax = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
            for a, (g, q) in zip(ax, tsb.groupby("group")):
                t = 0.5 * (q.t_lo_us + q.t_hi_us)
                NA = q.N_A.iloc[0]
                y = (q.source - q.background_scaled) / NA
                a.errorbar(t, y, np.sqrt(q.source) / NA, fmt="o", ms=2, color="k", label="source − F·6264")
                a.plot(t, q.dummy / q.dummy_trig, "-", color="C2", label="6256")
                r = ts[ts.group == g].iloc[0]
                a.plot(t, (r.A_fit * (1 - np.exp(-t / r.therm_us)) * np.exp(-t / r.tau_us) + r.C_fit) / NA, "-",
                       color="C0", label="capture fit")
                a.set_yscale("log"); a.set_xlabel("cluster time [µs]"); a.set_title(g)
            ax[0].set_ylabel("clusters per AmBe trigger per µs"); ax[0].legend(frameon=False)
            two_titles(fig, "Cluster time spectra per port", "veto-passing IC-gated triggers; 6256 = dummy triggers, per trigger")
            pdf.savefig(fig); plt.close(fig)
        if cs is not None:
            q = cs[cs.group == "pooled"]
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
            for a, par in zip(ax, ("s", "mu")):
                z = q[q.param == par].reset_index(drop=True)
                x = np.arange(len(z))
                a.errorbar(x - 0.1, z.mean_pull, z.mean_pull_err, fmt="o", color="k", label="mean pull")
                a.errorbar(x + 0.1, z.pull_width, z.pull_width_err, fmt="s", color="C0", label="pull width")
                a.set_xticks(x, [f"s={r.s_true:g}\nμ={r.mu_true:g}" for r in z.itertuples()])
                a.set_title("efficiency scale s" if par == "s" else "source-induced μ_ind")
            ax[0].legend(frameon=False)
            two_titles(fig, "MC closure: pulls pooled over the 5 ports",
                       "pass: |mean| < 0.2 and width 0.9-1.1; 200 toys per point, template and pseudo-data drawn independently")
            pdf.savefig(fig); plt.close(fig)
    print(f"[forwardfold] wrote {out}")


# ---------------------------------------------------------------------------------------------- CLI
def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe stats foldeff --fold-model forward")
    p.add_argument("--stage", choices=["closure", "data", "checks", "plots", "datamc", "deckplots"], required=True,
                   help="closure: Asimov + toys on MC and the background run only; data: the fit with its refits; "
                        "checks: time-spectrum mu_ind and the IC > 3000 tank closure; datamc / deckplots: the "
                        "Data/MC and Data/Data tables and the deck figures with residual panels (forward_fold_datamc)")
    p.add_argument("--toys", type=int, default=None, help="toys per injection point (fold.closure.n_toys otherwise)")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args(list(argv) if argv else [])
    if a.stage == "plots":
        return plots(ctx)
    if a.stage == "deckplots":
        from .forward_fold_datamc import deckplots
        return deckplots(ctx)
    su = setup(ctx)
    print(f"[forwardfold] factors {su.blk['factors_csv']} md5 OK; window {su.window[0]:g}-{su.window[1]:g} ns; "
          f"bins 0..{su.K - 1} + >= {su.K}; {len(su.fac)} source runs; background run {su.bkg_run} "
          f"({int((su.counts.run == su.bkg_run).sum())} veto-passing gated triggers)", flush=True)
    off = {r: dy for r, dy in su.label_dy.items() if dy != 0}
    if off:
        print(f"[forwardfold] off-grid runs (nearest MC height, MC y - data y): {off}")
    if a.stage == "closure":
        closure(ctx, su, a.toys or int(su.blk["closure"]["n_toys"]), a.jobs)
    elif a.stage == "data":
        data_fit(ctx, su)
    elif a.stage == "datamc":
        from .forward_fold_datamc import datamc
        datamc(ctx, su)
    else:
        prm = ctx.csv_path("forwardfold_params")
        fp = pd.read_csv(prm) if Path(prm).exists() else None
        ts = time_spectrum(ctx, su, fp)
        ic = ic3000_tank(ctx, su)
        with pd.option_context("display.width", 220, "display.float_format", "{:.5f}".format):
            print(ts.to_string(index=False))
            print(ic.to_string(index=False))
