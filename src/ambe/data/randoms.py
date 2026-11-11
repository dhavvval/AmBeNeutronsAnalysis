"""
Random-trigger (dummy-trigger) runs as a data-driven noise sample.

`ambe data randoms --config <randoms config>` writes, per run in inputs.root_files:
  <run_name>__random_hits_<run>.parquet      trigger, pmtID (chankey), t, x, y, z, pe
  <run_name>__random_triggers_<run>.parquet  trigger, event_tank_time, nhits, pe_sum, n_clusters
  <run_name>__random_cfclusters_<run>.parquet  the tree's ClusterFinder clusters
With --cluster --mc-config <mc config> it also clusters every trigger with the MC
config's clusterers (same parameters as `ambe mc capture`) into
  <run_name>__random_recoclusters_<run>[__<tag>].parquet
The hits parquet is what `ambe mc capture --overlay` adds to MC events.
With --pool <name> --pool-veto {cosmic,none} it also writes the runs of the config as one
background sample <run_name>__random_{hits,triggers,cfclusters}_<name> (triggers renumbered,
`run` / `run_trigger` columns); --cluster then clusters the pool instead of each run.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context
from ..io import resolve_inputs


def _load_file():
    """The BeamCluster reader of scripts/workflows/analyze_optics_beamcluster_data.py."""
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "scripts" / "workflows" / "analyze_optics_beamcluster_data.py"
    spec = importlib.util.spec_from_file_location("analyze_optics_beamcluster_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_file


def run_number(path: Path) -> int:
    m = re.search(r"(\d{4})(?!.*\d{4})", Path(path).stem)
    if not m:
        raise SystemExit(f"[randoms] no run number in {path}")
    return int(m.group(1))


def extract(root_file: Path):
    """(hits, triggers, cf clusters) for one BeamCluster file; triggers with tank data only,
    one per eventTimeTank."""
    hits, trig, cfc = [], [], []
    seen = set()
    for ev in _load_file()(Path(root_file)):
        ett = ev["event_tank_time"]
        if ett == 0 or ett in seen:
            continue
        seen.add(ett)
        k = len(trig)
        h = ev["df_hits"].rename(columns={"chankey": "pmtID"})
        hits.append(h[["pmtID", "t", "x", "y", "z", "pe"]].assign(trigger=k))
        cfc.append(ev["df_cf"].assign(trigger=k))
        trig.append(dict(trigger=k, event_tank_time=ett, event_number=ev["event_number"],
                         nhits=len(h), pe_sum=float(h.pe.sum()), n_clusters=len(ev["df_cf"])))
    return (pd.concat(hits, ignore_index=True), pd.DataFrame(trig),
            pd.concat(cfc, ignore_index=True))


RUN_LABELS = {6256: "6256 (LAPPD off)", 6254: "6254 (LAPPD on)"}


def pool_label(name) -> str:
    return str(name).replace("p", "+")


def _cosmic_vetoed(cfc: pd.DataFrame) -> set:
    from .selection import CutCriteria
    c = CutCriteria()
    return set(cfc.trigger[(cfc.clusterTime < c.cosmic_ct_threshold) | (cfc.clusterPE > c.cosmic_pe_threshold)])


def build_pool(parts: list, veto: str) -> tuple:
    """parts = [(run, hits, triggers, cf clusters)] -> pooled (hits, triggers, cf clusters), one trigger
    index over all runs in the given order; veto 'cosmic' drops triggers the cosmic veto rejects."""
    hs, ts, cs = [], [], []
    k0 = 0
    for run, h, t, c in parts:
        keep = t.trigger
        if veto == "cosmic":
            keep = keep[~keep.isin(_cosmic_vetoed(c))]
        new = pd.Series(np.arange(k0, k0 + len(keep)), index=keep.to_numpy())
        k0 += len(keep)
        t = t[t.trigger.isin(new.index)]
        ts.append(t.assign(run=run, run_trigger=t.trigger, trigger=t.trigger.map(new).to_numpy()))
        h = h[h.trigger.isin(new.index)]
        hs.append(h.assign(run=run, trigger=h.trigger.map(new).to_numpy()))
        c = c[c.trigger.isin(new.index)]
        cs.append(c.assign(run=run, trigger=c.trigger.map(new).to_numpy()))
    return (pd.concat(hs, ignore_index=True), pd.concat(ts, ignore_index=True),
            pd.concat(cs, ignore_index=True))


def random_samples(rctx: RunContext, overlay_run, tag: str = "") -> list:
    """[(key, label, triggers, hits, recoclusters, cf clusters)] for the plots of an overlay sample:
    overlay_run 6256 = the dummy runs 6256 and 6254; a --pool name = the pool, then each of its runs."""
    def rd(base, run, t=""):
        return pd.read_parquet(rctx.parquet_path(f"{rctx.run_name}__{base}_{run}" + (f"__{t}" if t else "")))
    if str(overlay_run) == "6256":
        return [(r, RUN_LABELS[r], rd("random_triggers", r), rd("random_hits", r), rd("random_recoclusters", r, tag),
                 rd("random_cfclusters", r)) for r in (6256, 6254)]
    tr = rd("random_triggers", overlay_run)
    if "run" not in tr:
        raise SystemExit(f"[randoms] {overlay_run} is not a --pool sample of {rctx.run_name}")
    h, rc, cf = rd("random_hits", overlay_run), rd("random_recoclusters", overlay_run, tag), rd("random_cfclusters", overlay_run)
    out = [(str(overlay_run), pool_label(overlay_run), tr, h, rc, cf)]
    for r in tr.run.unique():
        k = set(tr.trigger[tr.run == r])
        out.append((int(r), str(r), tr[tr.trigger.isin(k)], h[h.trigger.isin(k)], rc[rc.trigger.isin(k)],
                    cf[cf.trigger.isin(k)]))
    return out


def cluster_triggers(hits: pd.DataFrame, cfc: pd.DataFrame, mc_ctx: RunContext,
                     methods: list, t_min: float, pe_only: bool = False,
                     skip: Optional[set] = None) -> pd.DataFrame:
    """Cluster each random trigger exactly as `ambe mc capture` clusters an MC event.
    pe_only: f_pe_total is the only feature; skip: triggers not clustered."""
    from ..mc.capture_truth import DEFAULTS, CLUSTER_FEATURES, label_event
    from ..mc.cluster_features import compute_cluster_features, load_geometry
    cfg = {**DEFAULTS, **(mc_ctx.extra.get("capture_truth", {}) or {})}
    cfg["peak"] = {**DEFAULTS["peak"], **(cfg.get("peak") or {})}
    feat = dict(mc_ctx.extra.get("features", {}) or {}, _source_pos_m=None)
    gb = mc_ctx.extra.get("geometry", {}) or {}
    geo = load_geometry(gb.get("pmt_geometry"), gb.get("timing_offsets"),
                        frame=gb.get("frame", "tank"),
                        apply_offsets=bool(gb.get("apply_timing_offsets", True)))
    cf_by = {k: g.reset_index(drop=True) for k, g in cfc.groupby("trigger")}
    rows, parts = [], []
    sel = hits.t > t_min
    if skip:
        sel &= ~hits.trigger.isin(skip)
    for trig, df in hits[sel].groupby("trigger", sort=False):
        df = df.sort_values("t").reset_index(drop=True)
        lbl = label_event(df, cf_by.get(trig), methods, feat, cfg, None)
        for m, lb in lbl.items():
            for cid in np.unique(lb[lb >= 0]):
                sub = df[lb == cid]
                f = ({"f_pe_total": float(sub.pe.sum())} if pe_only else
                     {f"f_{k}": v for k, v in compute_cluster_features(sub, geo).items()
                      if k in CLUSTER_FEATURES})
                rows.append(dict(trigger=int(trig), method=m, cluster_id=int(cid),
                                 n_hits=len(sub), t_med=float(sub.t.median()),
                                 **{k: f.get(k, np.nan) for k in
                                    (["f_pe_total"] if pe_only else [f"f_{k}" for k in CLUSTER_FEATURES])}))
        if len(rows) >= 200_000:
            parts.append(pd.DataFrame(rows))
            rows = []
    parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe data randoms")
    p.add_argument("--cluster", action="store_true",
                   help="also cluster every trigger with the --mc-config clusterers")
    p.add_argument("--mc-config", default=None,
                   help="MC config whose capture_truth/features/geometry blocks set the clusterers")
    p.add_argument("--methods", default="clusterfinder,optics,peak")
    p.add_argument("--from-parquet", action="store_true",
                   help="skip the ROOT read and cluster the parquets a previous run wrote (EAF: no /pnfs)")
    p.add_argument("--runs", default=None, help="comma list of run numbers to do (default: all inputs); "
                                                 "with --from-parquet --cluster also a --pool name")
    p.add_argument("--hit-window", choices=["delayed", "full"], default=None,
                   help="with --cluster: delayed = hits after the MC config's residual.prompt_window_ns")
    p.add_argument("--tag", default="",
                   help="with --cluster: suffix for the random_recoclusters output")
    p.add_argument("--pe-only", action="store_true",
                   help="with --cluster: write f_pe_total only, no compute_cluster_features")
    p.add_argument("--skip-vetoed", action="store_true",
                   help="with --cluster: do not cluster triggers the cosmic veto (tree CF cluster "
                        "t < 2 us or PE > 100) rejects")
    p.add_argument("--pool", default=None, help="name of one background sample made of all runs of the config, "
                                                 "e.g. 6264p6256; --cluster then clusters the pool only")
    p.add_argument("--pool-veto", choices=["cosmic", "none"], default=None,
                   help="with --pool: cosmic = drop triggers the cosmic veto rejects, none = keep all")
    p.add_argument("--no-hits", action="store_true",
                   help="write only random_triggers and random_cfclusters (no hits parquet); "
                        "for tree-cluster counts (`ambe stats clusterindex --tree-box`)")
    a = p.parse_args(list(argv) if argv else [])
    if a.no_hits and (a.cluster or a.pool or a.from_parquet):
        p.error("--no-hits is for the ROOT read alone (no --cluster, --pool or --from-parquet)")
    if a.cluster and (a.mc_config is None or a.hit_window is None):
        p.error("--cluster needs --mc-config and --hit-window")
    if a.from_parquet and not a.cluster and not a.pool:
        p.error("--from-parquet only makes sense with --cluster or --pool")
    if bool(a.pool) != bool(a.pool_veto):
        p.error("--pool and --pool-veto go together")
    if a.from_parquet and a.runs and not a.cluster:
        p.error("--from-parquet --runs needs --cluster")
    if a.pool and (a.skip_vetoed or a.runs):
        p.error("--pool takes every run of the config; the veto is --pool-veto")
    names = ("random_hits", "random_triggers", "random_cfclusters")
    if a.from_parquet and a.runs:
        for run in a.runs.split(","):
            hits, trig, cfc = (pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__{b}_{run}")) for b in names)
            _report(f"{ctx.run_name} {run}", hits, trig, cfc)
            _cluster(ctx, a, run, hits, trig, cfc)
        return
    files = ([Path(f) for f in ctx.inputs["root_files"]] if a.from_parquet
             else resolve_inputs(ctx.inputs["root_files"]))
    want = {int(r) for r in a.runs.split(",")} if a.runs else None
    parts = []
    for rf in files:
        run = run_number(rf)
        if want is not None and run not in want:
            continue
        if a.from_parquet:
            hits, trig, cfc = (pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__{b}_{run}")) for b in names)
        else:
            hits, trig, cfc = extract(rf)
            for base, df in zip(names, (hits, trig, cfc)):
                if a.no_hits and base == "random_hits":
                    continue
                df.to_parquet(ctx.parquet_path(f"{ctx.run_name}__{base}_{run}"), index=False)
        _report(rf.name, hits, trig, cfc)
        if a.pool:
            parts.append((run, hits, trig, cfc))
        elif a.cluster:
            _cluster(ctx, a, run, hits, trig, cfc)
    if a.pool:
        hits, trig, cfc = build_pool(parts, a.pool_veto)
        for base, df in zip(names, (hits, trig, cfc)):
            df.to_parquet(ctx.parquet_path(f"{ctx.run_name}__{base}_{a.pool}"), index=False)
        print(f"[randoms] pool {a.pool} (veto {a.pool_veto}): "
              + ", ".join(f"{r} {int((trig.run == r).sum()):,}/{len(t):,}" for r, _, t, _ in parts) + " triggers kept")
        _report(f"pool {a.pool}", hits, trig, cfc)
        if a.cluster:
            _cluster(ctx, a, a.pool, hits, trig, cfc)


def _report(name, hits, trig, cfc):
    dur_us = 70.0
    print(f"[randoms] {name}: {len(trig):,} triggers, {len(hits):,} hits "
          f"(median {trig.nhits.median():.0f}/trigger, "
          f"{len(hits) / len(trig) / hits.pmtID.nunique() / dur_us * 1e3:.2f} kHz/PMT), "
          f"{len(cfc) / len(trig):.3f} tree ClusterFinder clusters/trigger")


def _cluster(ctx, a, run, hits, trig, cfc):
    mc_ctx = load_context(a.mc_config)
    t_min = (-np.inf if a.hit_window == "full" else
             float((mc_ctx.extra.get("residual") or {}).get("prompt_window_ns", 0.0)))
    skip = None
    if a.skip_vetoed:
        skip = _cosmic_vetoed(cfc)
        print(f"[randoms]   --skip-vetoed: {len(skip):,} of {len(trig):,} triggers not clustered")
    rc = cluster_triggers(hits, cfc, mc_ctx, a.methods.split(","), t_min, pe_only=a.pe_only, skip=skip)
    name = f"{ctx.run_name}__random_recoclusters_{run}" + (f"__{a.tag}" if a.tag else "")
    rc.to_parquet(ctx.parquet_path(name), index=False)
    print(f"[randoms]   clusters/trigger by method:\n"
          f"{(rc.groupby('method').size() / len(trig)).round(4).to_string()}")
