"""
EventIndex tables from re-clustered hits, for `ambe stats efficiency` and `ambe plots datamc`.

    ambe stats clusterindex --config <data cfg> --sample data --gate ic|none --methods a,b
    ambe stats clusterindex --config <mc cfg>   --sample mc   --methods a,b
    ambe stats foldeff      --config <data cfg> --mc-config <mc cfg> --methods a,b ...

Data reads `ambe data randoms --cluster` outputs (triggers, tree CF clusters, recoclusters);
MC reads `ambe mc process` (__events, __clusterfinder) and `ambe mc capture --truth-unit capture`.
The cosmic veto is the production one on the tree ClusterFinder clusters; a candidate is any
cluster of the method with t_med inside clusterindex.window_ns. No other cut.

Writes, under the config's parquet dir:
  <run_name>__eventindex_<gate>_<method>.parquet   one row per denominator trigger
  <run_name>__candidates_<gate>.parquet            candidate clusters of every method
MC uses gate "mc" and one pseudo-run per label (run = label_index), so the per-position fit
does not fit 1250 files one by one.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext
from ..data.randoms import run_number
from ..data.selection import CutCriteria


def _veto(cf: pd.DataFrame, key: list) -> pd.Series:
    c = CutCriteria()
    bad = (cf.clusterTime < c.cosmic_ct_threshold) | (cf.clusterPE > c.cosmic_pe_threshold)
    return cf.assign(_v=bad).groupby(key)["_v"].any()


def _overlay_veto(ctx: RunContext, spec: dict, ev: pd.DataFrame) -> np.ndarray:
    """Events whose overlaid random trigger has a tree CF cluster past the overlay cut with
    PE above the cosmic threshold (that light is in the MC hits but not in the MC tree CF)."""
    from ..mc.capture_truth import overlay_pick
    trig = np.sort(pd.read_parquet(spec["random_hits"], columns=["trigger"]).trigger.unique())
    cf = pd.read_parquet(spec["cfclusters"], columns=["trigger", "clusterTime", "clusterPE"])
    t_min = float((ctx.extra.get("residual") or {})["prompt_window_ns"])
    seed = int((ctx.extra.get("capture_truth") or {}).get("overlay_seed", 6256))
    big = cf[(cf.clusterTime > t_min) & (cf.clusterPE > CutCriteria().cosmic_pe_threshold)].trigger
    pick = np.fromiter((overlay_pick(seed, f, n, len(trig))
                        for f, n in zip(ev._source_file, ev.eventNumber)), np.int64, len(ev))
    v = np.isin(trig[pick], big.unique())
    print(f"[clusterindex] overlay veto: {v.mean():.4f} of {len(ev):,} MC events "
          f"({np.isin(trig, big.unique()).mean():.4f} of {len(trig):,} overlay triggers)")
    return v


def _data_tables(ctx: RunContext, blk: dict, gate: str, tag: str):
    from ..data.processor import AmBeNeutronProcessing
    pos = AmBeNeutronProcessing().source_positions
    pos.update({int(k): tuple(float(x) for x in v)
                for k, v in (blk.get("extra_source_positions") or {}).items()})
    gate_set = None
    if gate == "ic":
        paths = [p for pat in blk["gate_event_index"] for p in sorted(glob.glob(pat))]
        if not paths:
            raise SystemExit(f"[clusterindex] no gate EventIndex matched {blk['gate_event_index']}")
        g = pd.concat([pd.read_parquet(p, columns=["run", "eventTankTime"]) for p in paths])
        gate_set = g
    trig_all, cand_all = [], []
    for rf in ctx.inputs["root_files"]:
        run = run_number(Path(rf))
        base = f"{ctx.run_name}__random_%s_{run}"
        tr = pd.read_parquet(ctx.parquet_path(base % "triggers"))
        cf = pd.read_parquet(ctx.parquet_path(base % "cfclusters"))
        rc = pd.read_parquet(ctx.parquet_path(base % "recoclusters" + f"__{tag}"))
        tr = tr.rename(columns={"event_tank_time": "eventTankTime", "n_clusters": "numberOfClusters"})
        if gate_set is not None:
            gs = gate_set[gate_set.run == run]
            if gs.empty:
                raise SystemExit(f"[clusterindex] run {run} has no rows in the gate EventIndex")
            n_ref = len(gs)
            tr = tr[tr.eventTankTime.isin(set(gs.eventTankTime))]
            print(f"[clusterindex] run {run}: gate keeps {len(tr):,} triggers "
                  f"(gate EventIndex has {n_ref:,} rows)")
        v = _veto(cf, ["trigger"])
        tr = tr.assign(run=run, is_cosmic=tr.trigger.map(v).fillna(False).astype(bool))
        x, y, z = pos[run]
        tr = tr.assign(sourceX=float(x), sourceY=float(y), sourceZ=float(z))
        rc = rc.rename(columns={"f_pe_total": "pe"})[["trigger", "method", "t_med", "pe", "n_hits"]]
        trig_all.append(tr[["run", "trigger", "eventTankTime", "numberOfClusters", "is_cosmic",
                            "sourceX", "sourceY", "sourceZ"]])
        cand_all.append(rc.assign(run=run))
    return pd.concat(trig_all, ignore_index=True), pd.concat(cand_all, ignore_index=True), ["run", "trigger"]


def _mc_tables(ctx: RunContext, blk: dict, tag: str):
    from ..data.processor import mc_label_position
    lab_idx = {}
    for line in Path(blk["label_map"]).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            i, lab = line.split()
            lab_idx[lab] = int(i)
    sfx = f"__{tag}" if tag else ""
    ev = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__events"))
    cf = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__clusterfinder"),
                         columns=["_source_file", "eventID", "clusterTime", "clusterPE"])
    cf = cf.merge(ev[["_source_file", "eventID", "eventNumber"]], on=["_source_file", "eventID"])
    cl = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__capture_truth_clusters{sfx}"),
                         columns=["_source_file", "eventNumber", "method", "t_med", "pe_cluster",
                                  "n_hits", "n_overlay", "cap_dom"])
    cp = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__capture_truth_captures{sfx}"))
    rx = re.compile(r"ANNIEEvent_MC_AmBe_(?P<lab>.+QE_[\d.]+)_(?P<a>\d+)_\d+\.root$")
    files = pd.DataFrame({"_source_file": ev._source_file.unique()})
    m = files._source_file.str.extract(rx)
    if m.lab.isna().any():
        raise SystemExit(f"[clusterindex] unparsed MC file names: {files[m.lab.isna()].head()}")
    files["label"] = m.lab
    files["run"] = files.label.map(lab_idx)
    if files.run.isna().any():
        raise SystemExit(f"[clusterindex] labels missing from {blk['label_map']}: "
                         f"{sorted(files.label[files.run.isna()].unique())}")
    files["file_run"] = files.run.astype(int) * 1000 + m.a.astype(int) // 1000
    p = files.label.map(mc_label_position)
    files["sourceX"], files["sourceY"], files["sourceZ"] = zip(*p)
    tr = ev.merge(files, on="_source_file")
    tr["eventTankTime"] = tr.file_run.astype(np.int64) * 1000 + tr.eventNumber
    key = ["_source_file", "eventNumber"]
    v = _veto(cf, key).rename("is_cosmic").reset_index()
    tr = tr.merge(v, on=key, how="left")
    tr["is_cosmic"] = tr.is_cosmic.fillna(False).astype(bool)
    if blk.get("overlay_veto"):
        tr["is_cosmic"] |= _overlay_veto(ctx, blk["overlay_veto"], tr)
    tr["numberOfClusters"] = tr.merge(cf.groupby(key).size().rename("n").reset_index(),
                                      on=key, how="left").n.fillna(0).astype(int).to_numpy()
    tr["n_caps_window"] = tr.merge(cp.groupby(key).size().rename("n").reset_index(),
                                   on=key, how="left").n.fillna(0).astype(int).to_numpy()
    owners = [c for c in cp.columns if c.startswith("n_owners_")]
    found = (cp[owners] >= 1).groupby([cp[k] for k in key]).any()
    found.columns = [c.replace("n_owners_", "found_") for c in owners]
    tr = tr.merge(found.reset_index(), on=key, how="left")
    for c in found.columns:
        tr[c] = tr[c].astype("boolean").fillna(False).astype(bool)
    tr["run"] = tr.run.astype(int)
    cand = cl.rename(columns={"pe_cluster": "pe"}).merge(tr[key + ["run", "eventTankTime"]], on=key)
    return tr, cand, ["run", "eventTankTime"]


def build(ctx: RunContext, sample: str, gate: str, methods: list, tag: str):
    blk = (ctx.extra or {}).get("clusterindex") or {}
    lo, hi = (float(x) for x in blk["window_ns"])
    tr, cand, key = (_data_tables(ctx, blk, gate, tag) if sample == "data"
                     else _mc_tables(ctx, blk, tag))
    known = set(cand.method.unique())
    if set(methods) - known:
        raise SystemExit(f"[clusterindex] methods {sorted(set(methods) - known)} not in the "
                         f"clusters (have {sorted(known)})")
    cand = cand[cand.method.isin(methods) & (cand.t_med >= lo) & (cand.t_med <= hi)]
    cols = list(dict.fromkeys(key + ["eventTankTime", "is_cosmic", "sourceX", "sourceY", "sourceZ"]))
    cand = cand.drop(columns=[c for c in cols if c in cand.columns and c not in key]).merge(tr[cols], on=key)
    cand = cand[~cand.is_cosmic].drop(columns="is_cosmic")
    cand.to_parquet(ctx.parquet_path(f"{ctx.run_name}__candidates_{gate}"), index=False)
    for m in methods:
        n = cand[cand.method == m].groupby(key).size().rename("n_accepted_clusters").reset_index()
        ei = tr.merge(n, on=key, how="left")
        ei["n_accepted_clusters"] = ei.n_accepted_clusters.fillna(0).astype(int)
        ei = ei.drop(columns=[c for c in ei.columns
                              if c.startswith("found_") and c != f"found_{m}"])
        if f"found_{m}" in ei.columns:
            ei = ei.rename(columns={f"found_{m}": "found"})
        path = ctx.parquet_path(f"{ctx.run_name}__eventindex_{gate}_{m}")
        ei.to_parquet(path, index=False)
        ok = ~ei.is_cosmic
        msg = (f"[clusterindex] {m:>18}: {len(ei):,} triggers, veto {(~ok).mean():.4f}, "
               f"naive eff {(ei.n_accepted_clusters[ok] > 0).mean():.4f}, "
               f"mean cand {ei.n_accepted_clusters[ok].mean():.4f}")
        if "found" in ei.columns:
            msg += f", truth eff {ei.found[ok].mean():.4f}"
        print(msg + f"  -> {path.name}")


TREE_METHODS = ("clusterfinder_tree", "clusterfinder_v4box")


def _poisson(ctx: RunContext, run: int, counts: tuple) -> tuple:
    from . import efficiency_fit as ef
    fp = ctx.fit_params or {}
    n = int(sum(counts))
    h = ef.multiplicity_from_counts(int(run), {k: int(v) for k, v in enumerate(counts) if k > 0}, n,
                                    n_bins=len(counts), overflow_policy="clip")
    r = ef.fit_poisson(h, ef.make_grid(fp["eff_grid"]), ef.make_grid(fp["lambda_grid"]), likelihood=fp["likelihood"])
    return r.best_eff, r.eff_lo, r.eff_hi, r.best_lambda


def _v4_check(ctx: RunContext, blk: dict, tr: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Per run: the v4 Stage-2 EventIndex (gate_event_index) against this rebuild with the v4 box —
    triggers, cosmic veto and the box multiplicity, trigger by trigger."""
    paths = [p for pat in blk["gate_event_index"] for p in sorted(glob.glob(pat))]
    out = []
    box = cand[cand.method == "clusterfinder_v4box"].groupby(["run", "trigger"]).size()
    for p in paths:
        v = pd.read_parquet(p, columns=["run", "eventTankTime", "is_cosmic", "n_accepted_clusters"])
        run = int(v.run.iloc[0])
        t = tr[tr.run == run]
        if t.empty:
            out.append(dict(run=run, in_rebuild=False, n_v4=len(v)))
            continue
        t = t.assign(n_box=[box.get((run, x), 0) for x in t.trigger])
        m = v.merge(t[["eventTankTime", "is_cosmic", "n_box"]], on="eventTankTime", how="outer", suffixes=("_v4", "_now"), indicator=True)
        both = m[m._merge == "both"]
        ok = ~both.is_cosmic_v4.astype(bool)
        out.append(dict(run=run, in_rebuild=True, n_v4=len(v), n_now=len(t), only_v4=int((m._merge == "left_only").sum()),
                        only_now=int((m._merge == "right_only").sum()),
                        cosmic_v4=int(both.is_cosmic_v4.astype(bool).sum()), cosmic_now=int(both.is_cosmic_now.astype(bool).sum()),
                        cosmic_differs=int((both.is_cosmic_v4.astype(bool) != both.is_cosmic_now.astype(bool)).sum()),
                        box_differs=int((both.n_accepted_clusters[ok] != both.n_box[ok]).sum())))
    return pd.DataFrame(out)


def build_treebox(ctx: RunContext, gate: str, methods: list, tag: str):
    """Data only: EventIndex tables from the production (tree) ClusterFinder clusters, without
    (clusterfinder_tree) and with (clusterfinder_v4box) the v4 box of data.selection.CutCriteria.
    Writes only eventindex_<gate>_<method>; the shared candidates_<gate> table is left alone."""
    from ..data.selection import CutCriteria
    blk = (ctx.extra or {}).get("clusterindex") or {}
    lo, hi = (float(x) for x in blk["window_ns"])
    c = CutCriteria()
    tr, _, key = _data_tables(ctx, blk, gate, tag)
    rows, flow = [], []
    for run in tr.run.unique():
        cf = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__random_cfclusters_{run}"))
        w = cf[(cf.clusterTime >= lo) & (cf.clusterTime <= hi)]
        pe = (w.clusterPE > c.pe_min) & (w.clusterPE <= c.pe_max)
        cb = (w.clusterCB > c.ccb_min) & (w.clusterCB < c.ccb_max)
        nh = w.clusterHits >= c.chits_min
        box = pe & cb & nh
        rows.append(w.assign(run=run, method="clusterfinder_tree")[["run", "trigger", "method"]])
        rows.append(w[box].assign(run=run, method="clusterfinder_v4box")[["run", "trigger", "method"]])
        t = tr[(tr.run == run) & ~tr.is_cosmic]
        for step, sel in (("window", np.ones(len(w), bool)), ("+PE", pe), ("+PE+CB", pe & cb), ("+PE+CB+hits", box),
                          ("CB only", cb)):
            n = t.trigger.map(w[np.asarray(sel)].groupby("trigger").size()).fillna(0).astype(int).to_numpy()
            h = np.bincount(np.minimum(n, 3), minlength=4)
            flow.append(dict(run=run, step=step, sourceX=t.sourceX.iloc[0], sourceY=t.sourceY.iloc[0],
                             sourceZ=t.sourceZ.iloc[0], n_trig=len(t), m0=h[0], m1=h[1], m2=h[2], m3=h[3],
                             p0=h[0] / max(len(t), 1), counting=1 - h[0] / max(len(t), 1),
                             clusters=len(w[np.asarray(sel) & w.trigger.isin(t.trigger).to_numpy()])))
    flow = pd.DataFrame(flow)
    pos = flow.groupby(["sourceX", "sourceY", "sourceZ", "step"])[["n_trig", "m0", "m1", "m2", "m3", "clusters"]].sum().reset_index()
    pos["runs"] = flow.groupby(["sourceX", "sourceY", "sourceZ", "step"]).run.apply(
        lambda r: " ".join(str(x) for x in sorted(r))).to_numpy()
    for d in (flow, pos):
        d[["poisson_eff", "poisson_eff_lo", "poisson_eff_hi", "poisson_lambda"]] = [
            _poisson(ctx, r.run if "run" in d else 0, (r.m0, r.m1, r.m2, r.m3)) for r in d.itertuples()]
    pos["p0"], pos["counting"] = pos.m0 / pos.n_trig, 1 - pos.m0 / pos.n_trig
    flow.to_csv(ctx.csv_path(f"treebox_cutflow_{gate}"), index=False)
    pos.to_csv(ctx.csv_path(f"treebox_cutflow_pos_{gate}"), index=False)
    if gate == "ic":
        _v4_check(ctx, blk, tr, pd.concat(rows, ignore_index=True)).to_csv(ctx.csv_path("treebox_v4check"), index=False)
    cand = pd.concat(rows, ignore_index=True)
    for m in methods:
        n = cand[cand.method == m].groupby(key).size().rename("n_accepted_clusters").reset_index()
        ei = tr.merge(n, on=key, how="left")
        ei["n_accepted_clusters"] = ei.n_accepted_clusters.fillna(0).astype(int).where(~ei.is_cosmic, 0)
        path = ctx.parquet_path(f"{ctx.run_name}__eventindex_{gate}_{m}")
        ei.to_parquet(path, index=False)
        ok = ~ei.is_cosmic
        print(f"[clusterindex] {m:>20}: {len(ei):,} triggers, veto {(~ok).mean():.4f}, "
              f"naive eff {(ei.n_accepted_clusters[ok] > 0).mean():.4f}  -> {path.name}")


def fit(ctx: RunContext, gate: str, methods: list, data_ctx: RunContext, bkg_gate: str,
        n_jobs: int, s_max: Optional[float] = None):
    """Per-position Poisson + profile data-driven fits (efficiency_fit.run_by_position) on the
    clusterindex tables; fit_params and the 6264 template always come from the data config."""
    from . import efficiency_fit as ef
    fp = dict(data_ctx.fit_params or {})
    if not fp:
        raise SystemExit(f"[clustereff] {data_ctx.run_name} has no fit_params")
    if s_max is not None:
        fp["s_grid"] = {**fp["s_grid"], "max": s_max}
    lo, hi = (float(x) for x in (ctx.extra.get("clusterindex") or {})["window_ns"])
    if (lo, hi) != (float(fp["signal_window_start_ns"]), float(fp["acquisition_window_ns"])):
        raise SystemExit(f"[clustereff] clusterindex.window_ns {lo:g}-{hi:g} differs from the fit window")
    bkg_run = int(fp["background_run"])
    for m in methods:
        src = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__eventindex_{gate}_{m}"))
        bkg = pd.read_parquet(data_ctx.parquet_path(f"{data_ctx.run_name}__eventindex_{bkg_gate}_{m}"))
        bkg = bkg[bkg.run == bkg_run]
        if bkg.empty:
            raise SystemExit(f"[clustereff] no run {bkg_run} rows in the {bkg_gate} background index")
        ei = pd.concat([src[src.run != bkg_run], bkg], ignore_index=True)
        gtag = gate if gate in ("mc", bkg_gate) else f"{gate}_bkg{bkg_gate}"
        sfx = f"_{gtag}{'' if s_max is None else f'_smax{s_max:g}'}_{m}"
        print(f"\n[clustereff] {ctx.run_name} gate={gate} method={m} background={bkg_run} ({bkg_gate})")
        df = ef.run_by_position(ctx, fp, ei, n_bins=int(fp["n_bins"]),
                                eff_grid=ef.make_grid(fp["eff_grid"]),
                                lam_grid=ef.make_grid(fp["lambda_grid"]),
                                likelihood=fp["likelihood"],
                                bkg_unc_mode=fp["background_uncertainty"],
                                overflow_policy=fp["overflow_policy"],
                                n_gof_toys=int(fp.get("n_gof_toys", 2000)), n_jobs=n_jobs,
                                out_suffix=sfx)
        if "found" in src.columns:
            ok = src[~src.is_cosmic]
            t = ok.groupby(["sourceX", "sourceY", "sourceZ"]).found.agg(["mean", "size"])
            key = list(zip(df.x_pos, df.y_pos, df.z_pos))
            df["truth_eff"] = [t["mean"].get(k, np.nan) for k in key]
            df["truth_eff_err"] = [np.sqrt(t["mean"].get(k, np.nan) * (1 - t["mean"].get(k, np.nan))
                                           / t["size"].get(k, np.nan)) for k in key]
            df.to_csv(ctx.csv_path(f"efficiency_fit_by_position{sfx}"), index=False)


def cli_fit(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    from ..context import load_context
    p = argparse.ArgumentParser(prog="ambe stats clustereff")
    p.add_argument("--sample", choices=["data", "mc"], required=True)
    p.add_argument("--gate", choices=["ic", "none"], default=None, help="data only")
    p.add_argument("--background-gate", choices=["ic", "none"], default=None,
                   help="which 6264 index (data gate) is the background template; required for mc, "
                        "optional for data (default: --gate; a different gate writes _<gate>_bkg<gate>_)")
    p.add_argument("--data-config", required=True,
                   help="data config: fit_params and the 6264 background index")
    p.add_argument("--methods", required=True)
    p.add_argument("--s-max", type=float, default=None,
                   help="override fit_params.s_grid.max; outputs carry _smax<value>_")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args(list(argv) if argv else [])
    if a.sample == "data" and a.gate is None:
        p.error("--sample data needs --gate")
    if a.sample == "mc" and (a.gate is not None or a.background_gate is None):
        p.error("--sample mc needs --background-gate and refuses --gate")
    gate = a.gate or "mc"
    fit(ctx, gate, a.methods.split(","), load_context(a.data_config), a.background_gate or a.gate,
        a.jobs, a.s_max)


FOLD_SEED = 6264


def fold_index(S, seed: int, gd_w: Optional[np.ndarray] = None,
               survive: Optional[np.ndarray] = None) -> pd.DataFrame:
    """EventIndex rows for the MC "data-like" sample: per non-vetoed MC event (Poisson-replicated by
    gd_w when given), k capture-owned clusters plus the cluster count j of one no-source trigger drawn
    at random, n = k + j. truth_k = k. survive: per-event probability that k is kept (else 0), for the
    light-yield correction; arrays are aligned with S.m_e sorted by (_source_file, eventNumber)."""
    key = ["_source_file", "eventNumber"]
    e = S.m_e.sort_values(key).reset_index(drop=True)
    nc = S.m_c[S.m_c.cap.to_numpy(bool)].groupby(key).size().rename("k").reset_index()
    k = e[key].merge(nc, on=key, how="left").k.fillna(0).to_numpy(int)
    if survive is not None:
        k = np.where(np.random.default_rng(seed + 1).random(len(k)) < survive, k, 0)
    rng = np.random.default_rng(seed)
    rep = np.ones(len(e), int) if gd_w is None else rng.poisson(gd_w)
    idx = np.repeat(np.arange(len(e)), rep)
    j = S.b_e.n_accepted_clusters.to_numpy(int)
    jj = j[rng.integers(len(j), size=len(idx))]
    out = e.loc[idx, ["run", "sourceX", "sourceY", "sourceZ", "found"]].reset_index(drop=True)
    out["truth_k"] = k[idx]
    out["n_accepted_clusters"] = k[idx] + jj
    out["is_cosmic"] = False
    return out


def _template_rows(data_ctx: RunContext, m: str, gate: str, bkg_run: int) -> pd.DataFrame:
    b = pd.read_parquet(data_ctx.parquet_path(f"{data_ctx.run_name}__eventindex_{gate}_{m}"))
    b = b[b.run == bkg_run]
    if b.empty:
        raise SystemExit(f"[foldeff] no run {bkg_run} rows in eventindex_{gate}_{m}")
    return b[["run", "is_cosmic", "n_accepted_clusters", "sourceX", "sourceY", "sourceZ"]]


def _dummy_rows(dummy: RunContext, m: str, tag: str, like: pd.DataFrame, bkg_run: int, win: tuple) -> pd.DataFrame:
    """Dummy run 6256 triggers as extra template rows (relabelled to the background run): same cosmic veto,
    re-clustered with `tag`; vetoed triggers kept as is_cosmic with no candidates."""
    from ..plots.datamc_zerobin import WIN, _bkg_counts
    if tuple(float(v) for v in win) != WIN:
        raise SystemExit(f"[foldeff] dummy template window {WIN} differs from the fit window {win}")
    n = _bkg_counts(dummy, 6256, m, tag)[0]
    n_all = len(pd.read_parquet(dummy.parquet_path(f"{dummy.run_name}__random_triggers_6256"), columns=["trigger"]))
    r = like.iloc[0]
    return pd.DataFrame(dict(run=bkg_run, is_cosmic=np.r_[np.zeros(len(n), bool), np.ones(n_all - len(n), bool)],
                             n_accepted_clusters=np.r_[n, np.zeros(n_all - len(n), int)],
                             sourceX=r.sourceX, sourceY=r.sourceY, sourceZ=r.sourceZ))


def _fit(ctx: RunContext, fp: dict, ei: pd.DataFrame, n_jobs: int, suffix: str) -> pd.DataFrame:
    from . import efficiency_fit as ef
    df = ef.run_by_position(ctx, fp, ei, n_bins=int(fp["n_bins"]), eff_grid=ef.make_grid(fp["eff_grid"]),
                              lam_grid=ef.make_grid(fp["lambda_grid"]), likelihood=fp["likelihood"],
                              bkg_unc_mode=fp["background_uncertainty"],
                              overflow_policy=fp["overflow_policy"],
                              n_gof_toys=int(fp.get("n_gof_toys", 2000)), n_jobs=n_jobs, out_suffix=suffix)
    df["s_grid_max"] = float(fp["s_grid"]["max"])
    df.to_csv(ctx.csv_path("efficiency_fit_by_position" + suffix), index=False)
    return df


def _pid(df: pd.DataFrame) -> np.ndarray:
    from ..plots.datamc_clustering import pos_id
    return pos_id(df.rename(columns={"x_pos": "sourceX", "y_pos": "sourceY", "z_pos": "sourceZ"}))


def foldeff(data_ctx: RunContext, mc: RunContext, methods: list, versions: list, hc_tag: Optional[str],
            template_gate: str, gd: Optional[float], fit_data: bool, n_jobs: int,
            light_keep: Optional[float] = None, s_max: Optional[float] = None,
            dummy: Optional[RunContext] = None, dummy_tag: Optional[str] = None, fold_gate: Optional[str] = None):
    """Counting / Poisson / data-driven efficiency on data IC triggers and on MC neutron clusters folded
    with the no-source run; the same 6264 template (gate template_gate) in every fit. With dummy, the
    6256 dummy triggers join both the template and the MC fold pool (outputs tagged <gate>_plus6256)."""
    import dataclasses
    from ..plots import datamc_clustering as dmc
    fp = dict(data_ctx.fit_params or {})
    if s_max is not None:
        fp["s_grid"] = {**fp["s_grid"], "max": s_max}
    bkg_run = int(fp["background_run"])
    win = tuple(float(v) for v in (mc.extra.get("clusterindex") or {})["window_ns"])
    hcl = None
    if "hitcorr" in versions:
        hcl = pd.read_parquet(mc.parquet_path(f"{mc.run_name}__candidates_hitcorr__{hc_tag}"))
    gd_tag = "" if gd is None else f"_gd{gd:g}".replace(".", "p")
    if light_keep is not None:
        gd_tag += f"_keep{light_keep:.2f}".replace(".", "p")
    tg = template_gate if dummy is None else f"{template_gate}_plus6256"
    if fold_gate == "ic":
        tg += "_icfold"
    rows = []
    for m in methods:
        S0 = dmc.load(data_ctx, mc, m, False, bkg_run)
        tmpl = _template_rows(data_ctx, m, template_gate, bkg_run)
        if dummy is not None:
            dr = _dummy_rows(dummy, m, dummy_tag, tmpl, bkg_run, win)
            tmpl = pd.concat([tmpl, dr], ignore_index=True)
            S0 = dataclasses.replace(S0, b_e=pd.concat([S0.b_e, dr[~dr.is_cosmic]], ignore_index=True))
            print(f"[foldeff] {m}: template {bkg_run} + 6256, {int((~tmpl.is_cosmic).sum()):,} triggers after the veto")
        if fold_gate == "ic":
            S0 = dataclasses.replace(S0, b_e=tmpl[~tmpl.is_cosmic].reset_index(drop=True))
            print(f"[foldeff] {m}: MC folded with the template triggers, {len(S0.b_e):,} after the veto")
        sfx = f"_fold_data_{tg}_{m}"
        if fit_data:
            print(f"\n[foldeff] data, {m}, template {bkg_run} gate {tg}")
            df = _fit(data_ctx, fp, pd.concat([S0.d_e[tmpl.columns], tmpl], ignore_index=True), n_jobs, sfx)
        else:
            df = pd.read_csv(data_ctx.csv_path("efficiency_fit_by_position" + sfx))
        rows.append(df[df.level == "position"].assign(method=m, sample="data", mc=""))
        for v in versions:
            S = S0 if v == "nominal" else dmc.hitcorr_sample(S0, hcl, win)
            gw = None
            if gd is not None:
                tc, model = dmc.gd_model(mc)
                w, w0, _ = dmc.gd_event_weights(tc, model, gd)
                gw = dmc._attach(S.m_e.sort_values(["_source_file", "eventNumber"]), tc, w, w0)
            sv = None
            if light_keep is not None:
                from ..plots.datamc_zerobin import light_survival
                sv = light_survival(mc, m, light_keep, S.m_e.sort_values(["_source_file", "eventNumber"]))
            fi = fold_index(S, FOLD_SEED, gw, sv)
            label = f"{mc.run_name}:{v}{gd_tag}"
            print(f"\n[foldeff] MC {label}, {m}: {len(fi):,} events, P(k>=1) {(fi.truth_k > 0).mean():.4f}, "
                  f"P(n=0) {(fi.n_accepted_clusters == 0).mean():.4f}")
            df = _fit(mc, fp, pd.concat([fi[tmpl.columns], tmpl], ignore_index=True), n_jobs,
                      f"_fold_mc{v}{gd_tag}_{tg}_{m}")
            t = fi.assign(ge1=fi.truth_k > 0).groupby(["sourceX", "sourceY", "sourceZ"]).agg(
                truth_eff=("ge1", "mean"), truth_n=("ge1", "size"), found_eff=("found", "mean"))
            key = list(zip(df.x_pos, df.y_pos, df.z_pos))
            for c in t.columns:
                df[c] = [t[c].get(q, np.nan) for q in key]
            df["truth_eff_err"] = np.sqrt(df.truth_eff * (1 - df.truth_eff) / df.truth_n)
            df.to_csv(mc.csv_path(f"efficiency_fit_by_position_fold_mc{v}{gd_tag}_{tg}_{m}"), index=False)
            rows.append(df[df.level == "position"].assign(method=m, sample="mc", mc=label))
    new = pd.concat(rows, ignore_index=True)
    new["pid"] = _pid(new)
    new["template_gate"] = tg
    path = data_ctx.csv_path("foldeff_summary" + ("_corrected" if light_keep is not None else ""))
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, na_values=[""])
        k = ["method", "sample", "mc", "template_gate"]
        old = old[~old.set_index(k).index.isin(new.set_index(k).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"\n[foldeff] wrote {path}")


def cli_foldeff(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    from ..context import load_context
    p = argparse.ArgumentParser(prog="ambe stats foldeff")
    p.add_argument("--mc-config", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--mc-versions", required=True, help="comma list of nominal,hitcorr, or none (data only)")
    p.add_argument("--hitcorr-tag", default=None, help="required with hitcorr, e.g. ov6256_deaddoc")
    p.add_argument("--template-gate", choices=["ic", "none"], required=True,
                   help="6264 eventindex gate used as the background template in every fit")
    p.add_argument("--gd", required=True, help="Gd loading [%%] for the MC reweight, or 'none'")
    p.add_argument("--data-fit", choices=["run", "reuse"], required=True,
                   help="run: fit the data; reuse: read the existing data fold CSVs")
    p.add_argument("--jobs", type=int, required=True)
    p.add_argument("--s-max", type=float,
                   help="upper end of the template-scale grid (fit_params.s_grid.max otherwise)")
    p.add_argument("--light-keep", type=float,
                   help="MC light-yield correction: keep this fraction of hits (per-event survival from the MC "
                        "P(found|hits) curve); writes csv/foldeff_summary_corrected.csv instead of foldeff_summary")
    p.add_argument("--template-dummy",
                   help="dummy randoms config: add the 6256 dummy triggers to the template and the MC fold pool")
    p.add_argument("--dummy-tag", help="recoclusters tag of the dummy run (with --template-dummy)")
    p.add_argument("--fold-gate", choices=["ic"], default=None,
                   help="ic: fold the MC with the IC-gated template triggers too (outputs tagged ic_icfold); "
                        "absent: the MC fold pool is the ungated background run, as before")
    a = p.parse_args(list(argv) if argv else [])
    if a.fold_gate == "ic" and (a.template_gate != "ic" or a.template_dummy):
        p.error("--fold-gate ic needs --template-gate ic and no --template-dummy")
    if bool(a.template_dummy) != bool(a.dummy_tag):
        p.error("--template-dummy and --dummy-tag go together")
    if a.template_dummy and a.template_gate != "none":
        p.error("--template-dummy needs --template-gate none (the dummy run has no IC gate)")
    vs = [] if a.mc_versions == "none" else a.mc_versions.split(",")
    if set(vs) - {"nominal", "hitcorr"}:
        p.error("--mc-versions takes nominal,hitcorr")
    if ("hitcorr" in vs) != (a.hitcorr_tag is not None):
        p.error("--hitcorr-tag is required with hitcorr and refused without it")
    foldeff(ctx, load_context(a.mc_config), a.methods.split(","), vs, a.hitcorr_tag, a.template_gate,
            None if a.gd == "none" else float(a.gd), a.data_fit == "run", a.jobs, a.light_keep, a.s_max,
            load_context(a.template_dummy) if a.template_dummy else None, a.dummy_tag, a.fold_gate)


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe stats clusterindex")
    p.add_argument("--sample", choices=["data", "mc"], required=True)
    p.add_argument("--gate", choices=["ic", "none"], default=None,
                   help="data only: ic = the Stage-1 trigger list of clusterindex.gate_event_index; "
                        "none = every trigger")
    p.add_argument("--methods", required=True)
    p.add_argument("--tag", required=True,
                   help="suffix of the recoclusters (data) / capture_truth outputs (mc)")
    p.add_argument("--tree-box", action="store_true",
                   help=f"data only: methods {','.join(TREE_METHODS)} from the production ClusterFinder "
                        "clusters, without and with the v4 box")
    a = p.parse_args(list(argv) if argv else [])
    if (a.sample == "data") != (a.gate is not None):
        p.error("--gate is required for --sample data and refused for --sample mc")
    if a.tree_box:
        if a.sample != "data" or set(a.methods.split(",")) - set(TREE_METHODS):
            p.error(f"--tree-box is data only and takes --methods from {','.join(TREE_METHODS)}")
        return build_treebox(ctx, a.gate, a.methods.split(","), a.tag)
    build(ctx, a.sample, a.gate or "mc", a.methods.split(","), a.tag)
