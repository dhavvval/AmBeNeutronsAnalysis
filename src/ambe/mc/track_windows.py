"""
Geant4-track-level time windows in CCinc MC: every saved WCSim track, before any cut.

    ambe mc trackwindows --config configs/cc_neutrino_v3fixnufull_truthtag.yaml \
        --world-config configs/cc_neutrino_v3fixnufullworld_truthtag.yaml \
        --wcsim-dir /pnfs/annie/persistent/users/dajana/output/genie_wcsim_tank/productionv3fix/tank \
        --world-wcsim-dir /pnfs/annie/persistent/users/dajana/output/genie_wcsim_world/productionv3fix/world \
        --image /cvmfs/singularity.opensciencegrid.org/anniesoft/toolanalysis:latest \
        --wcsim-lib /exp/annie/app/users/dajana/WCSim/WCSim --ta-repo /exp/annie/app/users/dajana/EB_BC_TA \
        --out ambe_output/cc_neutrino_v3fixnufull_truthtag/g4trackwindows \
        --fig-out PRESENTATION_PLOTS/ccinc_g4track_windows

Step 1 dumps the WCSim file behind each of the config's ANNIEEvent files (wcsim_<n>.root for
ANNIEEvent_<sample>_<n>.root) with root/dump_wcsim_tracks.C in the container: all saved tracks
and all tank digits. Dumps are cached in <out>/dumps.
Step 2 places every saved track (flag 0) in the windows <2, 2-10, 10-70 us (and 2-70) by its
creation time t0, by its Geant4 lifetime (t0 -> t1), and by its light: the MCHits in the ANNIEEvent
tree whose lineage contains the track (its own light or its descendants').
Samples: tank and world, all events of each (no event selection).

  g4tw_events.csv            events per sample
  g4tw_creation.csv          tracks per particle and creation window
  g4tw_creation_process.csv  tracks created at 2-10 / 10-70 us by creator process and parent particle
  g4tw_creation_hist.csv     creation time, 1 us bins, per particle
  g4tw_lifetime.csv          tracks per particle, start window x end window (a capturing neutron ends at
                             its capture: WCSim's stop time is the last elastic scatter for some of them)
  g4tw_light.csv             tracks per particle and set of windows holding their light
  g4tw_carriers.csv          which daughter carries a track's 10-70 us light
  g4tw_captures.csv          captures (nuclei created by nCapture) vs trueNeutCap, per window
  g4tw_digits.csv            true tank digits (photons) per time bin
  g4tw_origin_tracks.csv     tracks per source (origin(): nearest nCapture, mu decay, mu- stop, n scatter,
                             pi/K decay or primary ancestor), particle and creation window
  g4tw_origin_light.csv      true photons and events with light per source and time bin (digit parent's source)
  g4tw_neutron_end.csv       neutron tracks by start window, end window and end reason
  g4tw_units.parquet         one row per source unit at 2-70 us (captured neutron, decaying mu, stopped mu-, ...)
                             and per dark-noise clump, with its true-photon features
--origin-only rebuilds these four from the cached dumps, then redraws the figures.
Figures and tables: track_windows_figures.make_figures, from the CSVs only (--figures-only).
"""
from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..context import RunContext, load_context

SAMPLES = ("tank", "world")
WINDOWS = ("<2", "2-10", "10-70")
EDGES_NS = (2000.0, 10000.0, 70000.0)
LIFE_LABELS = ("<2", "2-10", "10-70", ">=70")
DIGIT_EDGES_NS = (2000.0, 10000.0, 66500.0, 70000.0)
DIGIT_LABELS = ("<2", "2-10", "10-66.5", "66.5-70", ">=70")
KEY = ["_source_file", "eventNumber"]
MACRO = Path(__file__).parent / "root" / "dump_wcsim_tracks.C"


def particle(pdg: np.ndarray) -> np.ndarray:
    pdg = np.asarray(pdg, np.int64)
    names = {11: "e-", -11: "e+", 22: "gamma", 13: "mu-", -13: "mu+", 211: "pi+", -211: "pi-",
             111: "pi0", 2212: "p", 2112: "n"}
    out = np.full(len(pdg), "other", dtype=object)
    for k, v in names.items():
        out[pdg == k] = v
    out[np.isin(np.abs(pdg), [12, 14, 16])] = "nu"
    out[np.isin(np.abs(pdg), [321, 311, 130, 310])] = "K"
    out[pdg > 1000000000] = "nucleus"
    return out


def window(t_ns: np.ndarray, edges=EDGES_NS) -> np.ndarray:
    """0, 1, 2 for <2, 2-10, 10-70 us; 3 for >= 70 us (other edges: one code per bin)."""
    return np.searchsorted(np.asarray(edges, float), np.asarray(t_ns, float), side="right")


TAGS = ("n capture", "mu decay", "mu- stop", "n scatter", "pi/K decay", "late primary", "prompt primary",
        "parent not saved")


TANK_CENTER_CM = np.array([0.0, -14.46, 168.1])
TANK_R_CM, TANK_HALF_H_CM = 152.4, 198.0
CF_HALF_WINDOW_NS, CF_MIN_HITS, NOISE_GAP_NS, N_LIVE_PMTS = 10.0, 5, 50.0, 121


def origin(tr: pd.DataFrame) -> pd.DataFrame:
    """Source of each track: the nearest tag (TAGS) on the track or its ancestors. Products of an nCapture
    are split by the source of the captured neutron ("n capture: <source>"). src_id is the source unit: the
    track that underwent the tagged process (captured neutron, decaying mu, stopped mu-, scattering n,
    decaying pi/K), or the primary itself; src_t/x/y/z/pdg/e0 describe the tagged track."""
    ent = tr.entry.to_numpy(np.int64)
    key, pkey = ent * 10**9 + tr.id.to_numpy(np.int64), ent * 10**9 + tr.parent.to_numpy(np.int64)
    order = np.argsort(key)
    pos = np.searchsorted(key[order], pkey).clip(0, len(key) - 1)
    pidx = np.where(key[order][pos] == pkey, order[pos], -1)
    part, sp = tr.particle.to_numpy(), tr.sproc.to_numpy()
    ppart = np.where(pidx >= 0, part[pidx], "")
    prim = tr.parent.to_numpy() <= 0
    tag = np.full(len(tr), -1)
    for i, m in reversed(list(enumerate([
            sp == "nCapture",
            (sp == "Decay") & np.isin(ppart, ["mu-", "mu+"]),
            sp == "muMinusCaptureAtRest",
            (ppart == "n") & (sp != "nCapture"),
            (sp == "Decay") & np.isin(ppart, ["pi+", "pi-", "pi0", "K"]),
            prim & (tr.t0.to_numpy() >= EDGES_NS[0]),
            prim & (tr.t0.to_numpy() < EDGES_NS[0]),
            ~prim & (pidx < 0)]))):
        tag[m] = i
    o, root = tag.copy(), np.where(tag >= 0, np.arange(len(tr)), -1)
    while (m := o < 0).any():
        o[m], root[m] = o[pidx[m]], root[pidx[m]]
    lab = np.array(TAGS, dtype=object)[o]
    cap = o == 0
    lab[cap] = "n capture: " + lab[pidx[root[cap]]]
    ids = tr.id.to_numpy()
    unit = np.where(o <= 4, pidx[root], root)
    return pd.DataFrame({"origin": lab, "src_tag": o, "src_id": ids[unit], "src_t": tr.t0.to_numpy()[root],
                         "src_x": tr.x0.to_numpy()[root], "src_y": tr.y0.to_numpy()[root],
                         "src_z": tr.z0.to_numpy()[root], "src_pdg": tr.pdg.to_numpy()[root],
                         "src_e0": tr.e0.to_numpy()[root]}, index=tr.index)


def wall_distance_cm(x, y, z) -> np.ndarray:
    r = np.hypot(np.asarray(x) - TANK_CENTER_CM[0], np.asarray(z) - TANK_CENTER_CM[2])
    return np.minimum(TANK_R_CM - r, TANK_HALF_H_CM - np.abs(np.asarray(y) - TANK_CENTER_CM[1]))


def _light_features(ph: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per unit: photons, PE, PMTs, charge balance (ClusterClassifiers MC form), time rms, PMTs within the
    ClusterFinder integration window around the median photon time."""
    tube = ph.groupby(keys + ["tube"]).q.sum().rename("qt").reset_index()
    tube["q2"] = tube.qt ** 2
    f = tube.groupby(keys).agg(pe=("qt", "sum"), q2=("q2", "sum"), n_tubes=("tube", "size"))
    f["charge_balance"] = np.sqrt(np.clip(f.q2 / f.pe ** 2 - 1.0 / N_LIVE_PMTS, 0, None))
    g = ph.groupby(keys).t
    f = f.join(g.size().rename("n_photons")).join(g.std(ddof=0).rename("t_rms_ns")).join(g.median().rename("t_med"))
    ph = ph.join(f.t_med, on=keys)
    near = ph[(ph.t - ph.t_med).abs() <= CF_HALF_WINDOW_NS]
    f = f.join(near.groupby(keys).tube.nunique().rename("n_tubes_cf"))
    f["n_tubes_cf"] = f.n_tubes_cf.fillna(0).astype(int)
    return f.drop(columns=["q2", "t_med"])


def unit_table(tr: pd.DataFrame, dg: pd.DataFrame, s: str) -> pd.DataFrame:
    """One row per source unit created at 2-70 us: one tagged process at one time (a neutron capture, a mu decay,
    a mu- capture or decay in orbit after its atomic cascade, one neutron scatter, ...), and one per dark-noise clump (noise digits closer than NOISE_GAP_NS), with the features of its true photons."""
    k = ["entry", "src_tag", "src_id", "src_t"]
    u = (tr.groupby(k)
         .agg(origin=("origin", "first"), x=("src_x", "first"), y=("src_y", "first"), z=("src_z", "first"),
              pdg=("src_pdg", "first"), e0=("src_e0", "first")))
    u["t_unit"] = u.index.get_level_values("src_t")
    u = u[(u.t_unit >= EDGES_NS[0]) & (u.t_unit < EDGES_NS[2])]
    nc = tr[(tr.sproc == "nCapture") & (tr.particle == "nucleus")]
    z = pd.Series((nc.pdg.to_numpy(np.int64) // 10000) % 1000,
                  index=pd.MultiIndex.from_arrays([nc.entry, np.zeros(len(nc), int), nc.parent, nc.t0], names=k))
    u = u.join(z.groupby(level=k).max().rename("capture_Z"))
    tk = pd.MultiIndex.from_arrays([tr.entry, tr.id])
    pk = pd.MultiIndex.from_arrays([dg.entry, dg.dparent])
    ph = dg.assign(**{c: pd.Series(tr[c].to_numpy(), index=tk).reindex(pk).to_numpy() for c in k[1:]})
    ph = ph[(ph.noise == 0) & ph.src_id.notna()].astype({"src_id": np.int64, "src_tag": np.int64})
    ph = ph.join(pd.Series(True, index=u.index, name="keep"), on=k)
    u = u.join(_light_features(ph[ph.keep.notna()], k))
    u["d_wall_cm"] = wall_distance_cm(u.x, u.y, u.z)

    nz = dg[(dg.noise > 0) & (dg.t >= EDGES_NS[0]) & (dg.t < EDGES_NS[2])].sort_values(["entry", "t"])
    new = (nz.entry.diff().fillna(1).to_numpy() != 0) | (nz.t.diff().fillna(np.inf).to_numpy() > NOISE_GAP_NS)
    nz = nz.assign(src_tag=-1, src_id=-np.cumsum(new))
    nz["src_t"] = nz.groupby(["entry", "src_id"]).t.transform("min")
    nu = nz.groupby(k).t.min().rename("t_unit").to_frame().assign(origin="dark noise").join(_light_features(nz, k))
    u = pd.concat([u.reset_index(), nu.reset_index()], ignore_index=True)
    for c in ("n_photons", "n_tubes", "n_tubes_cf"):
        u[c] = u[c].fillna(0).astype(int)
    u["pe"] = u.pe.fillna(0.0)
    u["w"] = window(u.t_unit.to_numpy())
    return u.assign(sample=s)


def neutron_end_table(tr: pd.DataFrame, s: str) -> pd.DataFrame:
    """Every neutron track: start window, end window and how it ends (capture, neutronInelastic, stops outside
    the tank, other). A capturing neutron ends at its capture."""
    n = tr[tr.particle == "n"]
    k = pd.MultiIndex.from_arrays([n.entry, n.id])
    nc = tr[(tr.sproc == "nCapture") & (tr.particle == "nucleus")]
    tc = (pd.Series(nc.t0.to_numpy(), index=pd.MultiIndex.from_arrays([nc.entry, nc.parent])).groupby(level=[0, 1])
          .min().reindex(k).to_numpy())
    inel = tr[tr.sproc == "neutronInelastic"]
    has_inel = k.isin(pd.MultiIndex.from_arrays([inel.entry, inel.parent]))
    cap = ~np.isnan(tc)
    reason = np.select([cap, has_inel, n.stopvol.to_numpy() != 10], ["capture", "neutronInelastic", "stops outside tank"],
                       "other")
    d = pd.DataFrame({"w0": window(n.t0.to_numpy()), "w1": window(np.where(cap, tc, n.t1.to_numpy())),
                      "reason": reason})
    return d.groupby(["w0", "w1", "reason"]).size().rename("n").reset_index().assign(sample=s)


def origin_tables(tr: pd.DataFrame, dg: pd.DataFrame, s: str) -> dict:
    """Tracks per source and creation window; true photons (digits) and events with light per source and bin;
    neutron end reasons; source units with their light features."""
    src = pd.Series(tr.origin.to_numpy(), index=pd.MultiIndex.from_arrays([tr.entry, tr.id]))
    o = src.reindex(pd.MultiIndex.from_arrays([dg.entry, dg.dparent])).fillna("photon parent not saved").to_numpy()
    dgo = dg.assign(origin=np.where(dg.noise.to_numpy() > 0, "dark noise", o))
    ph = dgo.groupby(["origin", "bin", "entry"]).size().rename("n_photons").reset_index()
    return {"origin_tracks": tr.groupby(["origin", "particle", "w0"]).size().rename("n").reset_index().assign(sample=s),
            "origin_light": ph.groupby(["origin", "bin"]).agg(n_photons=("n_photons", "sum"),
                                                              n_events=("entry", "size")).reset_index().assign(sample=s),
            "neutron_end": neutron_end_table(tr, s),
            "units": unit_table(tr, dg, s)}


def origin_file(tracks_root: Path, s: str) -> dict:
    tr = _read(tracks_root, "tracks")
    tr = tr[tr.flag == 0].reset_index(drop=True)
    tr["particle"] = particle(tr.pdg.to_numpy())
    tr = tr.join(origin(tr))
    tr["w0"] = window(tr.t0.to_numpy())
    dg = _read(tracks_root, "digits")
    dg["bin"] = window(dg.t.to_numpy(), DIGIT_EDGES_NS)
    return origin_tables(tr, dg, s)


def dump_file(wcsim: Path, out: Path, a) -> Path:
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_name(out.stem + ".tmp.root")
    cmd = (f"cd {a.ta_repo}; source Setup.sh >/dev/null 2>&1; "
           f"export LD_LIBRARY_PATH={a.wcsim_lib}:$LD_LIBRARY_PATH; "
           f"export ROOT_INCLUDE_PATH={a.wcsim_lib}/include:$ROOT_INCLUDE_PATH; "
           f"root -l -b -q '{MACRO}(\"{wcsim}\",\"{tmp}\")'")
    r = subprocess.run(["singularity", "exec", "-B", "/exp/annie:/exp/annie", "-B", "/pnfs:/pnfs",
                        a.image, "bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or "Error" in r.stderr:
        raise SystemExit(f"[trackwindows] dump failed for {wcsim}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    tmp.rename(out)
    print(f"[trackwindows] dumped {wcsim.name} -> {out.name}", flush=True)
    return out


def _read(path: Path, tree: str) -> pd.DataFrame:
    import uproot
    a = uproot.open(path)[tree].arrays(library="np")
    return pd.DataFrame({k: (v.astype(str) if v.dtype == object else v) for k, v in a.items()})


def _hit_lineage(annie: Path) -> pd.DataFrame:
    """One row per (hit, lineage step): entry, hit time, step trackID, its child trackID (-1 = emitter)."""
    import awkward as ak
    import uproot
    a = uproot.open(annie)["Event"].arrays(["eventNumber", "hitT", "hitLineageTrackID", "hitLineageDepth"])
    nh = ak.to_numpy(ak.num(a.hitT))
    ev = np.repeat(ak.to_numpy(a.eventNumber).astype(np.int64), nh)
    t = ak.to_numpy(ak.flatten(a.hitT)).astype(float)
    d = ak.to_numpy(ak.flatten(a.hitLineageDepth)).astype(np.int64)
    tid = ak.to_numpy(ak.flatten(a.hitLineageTrackID)).astype(np.int64)
    if d.sum() != len(tid) or len(d) != len(t):
        raise SystemExit(f"[trackwindows] {annie.name}: lineage depth does not match trackIDs")
    hit = np.repeat(np.arange(len(t)), d)
    pos = np.arange(len(tid)) - np.repeat(np.cumsum(d) - d, d)
    child = np.where(pos > 0, np.roll(tid, 1), -1)
    return pd.DataFrame({"entry": ev[hit], "t": t[hit], "tid": tid, "child": child})


def process_file(tracks_root: Path, annie: Path, s: str) -> dict:
    tr = _read(tracks_root, "tracks")
    ents = set(tr.entry.astype(int))
    tr = tr[tr.flag == 0].reset_index(drop=True)
    tr["particle"] = particle(tr.pdg.to_numpy())
    tr["origin"] = origin(tr)
    cap =(tr[(tr.particle == "nucleus") & (tr.sproc == "nCapture")].groupby(["entry", "parent"]).t0.min()
           .rename("t_capture").reset_index().rename(columns={"parent": "id"}))
    tr = tr.merge(cap, on=["entry", "id"], how="left")
    is_cap = (tr.particle == "n") & tr.t_capture.notna()
    n_fixed = int((is_cap & ((tr.t_capture - tr.t1).abs() > 1.0)).sum())
    tr["t_end"] = np.where(is_cap, tr.t_capture, tr.t1)
    tr["w0"] = window(tr.t0.to_numpy())
    tr["w1"] = window(tr.t_end.to_numpy())
    par = tr[["entry", "id", "particle"]].rename(columns={"id": "parent", "particle": "parent_particle"})
    tr = tr.merge(par, on=["entry", "parent"], how="left")
    tr["parent_particle"] = tr.parent_particle.fillna(pd.Series(np.where(tr.parent <= 0, "primary", "not saved"),
                                                                index=tr.index))
    ptype = pd.Series(tr.particle.to_numpy(), index=pd.MultiIndex.from_arrays([tr.entry, tr.id]))

    hl = _hit_lineage(annie)
    hl["w"] = window(hl.t.to_numpy())
    hl = hl[hl.w < 3]
    known = pd.MultiIndex.from_arrays([hl.entry, hl.tid]).isin(ptype.index)
    n_missing = int((~known).sum())
    light = (hl.assign(bit=np.left_shift(1, hl.w.to_numpy())).groupby(["entry", "tid"]).bit.agg(np.bitwise_or.reduce)
             .rename("light_mask").reset_index())
    for w in range(3):
        n = hl[hl.w == w].groupby(["entry", "tid"]).size().rename(f"hits_{w}")
        light = light.merge(n.reset_index(), on=["entry", "tid"], how="left")
    late = hl[hl.w == 2]
    cidx = pd.MultiIndex.from_arrays([late.entry, late.child])
    carrier = np.where(late.child.to_numpy() < 0, "own light",
                       ptype.reindex(cidx).fillna("not saved").to_numpy())
    carr = (late.assign(carrier=carrier).groupby(["entry", "tid"]).carrier
            .agg(lambda s: s.value_counts().index[0]).rename("carrier").reset_index())
    tr = (tr.merge(light.rename(columns={"tid": "id"}), on=["entry", "id"], how="left")
          .merge(carr.rename(columns={"tid": "id"}), on=["entry", "id"], how="left"))
    tr["light_mask"] = tr.light_mask.fillna(0).astype(int)
    for w in range(3):
        tr[f"hits_{w}"] = tr[f"hits_{w}"].fillna(0).astype(int)

    dg = _read(tracks_root, "digits")
    dg["bin"] = window(dg.t.to_numpy(), DIGIT_EDGES_NS)

    out = {k: [] for k in ("creation", "creation_process", "creation_hist", "lifetime", "light",
                           "carriers", "captures", "digits", "events")}
    hb = np.arange(0.0, 70.01, 1.0)
    g = tr
    out["events"].append(pd.DataFrame([dict(sample=s, n_events=len(ents), n_tracks=len(g),
                                             lineage_steps=int(len(hl)), lineage_steps_not_in_tracks=n_missing,
                                             capturing_neutrons_end_time_replaced=n_fixed)]))
    out["creation"].append(g.groupby(["particle", "w0"]).size().rename("n").reset_index().assign(sample=s))
    out["creation_process"].append(g[g.w0.isin([1, 2])].groupby(["particle", "w0", "sproc", "parent_particle"])
                                   .size().rename("n").reset_index().assign(sample=s))
    for p, gp in g.groupby("particle"):
        h, _ = np.histogram(gp.t0.to_numpy() / 1000.0, bins=hb)
        out["creation_hist"].append(pd.DataFrame(dict(sample=s, particle=p, lo=hb[:-1], hi=hb[1:], n=h)))
    out["lifetime"].append(g.groupby(["particle", "w0", "w1"]).size().rename("n").reset_index().assign(sample=s))
    out["light"].append(g.groupby(["particle", "light_mask"])
                        .agg(n=("id", "size"), hits_0=("hits_0", "sum"), hits_1=("hits_1", "sum"),
                             hits_2=("hits_2", "sum")).reset_index().assign(sample=s))
    lt = g[(g.light_mask & 4) > 0]
    lt = lt.assign(group=np.where((lt.light_mask & 1) > 0, "prompt + late", "late, no prompt"))
    out["carriers"].append(lt.groupby(["particle", "group", "carrier"]).size().rename("n").reset_index()
                           .assign(sample=s))
    nc = g[(g.particle == "nucleus") & (g.sproc == "nCapture")]
    out["captures"].append(nc.groupby("w0").size().rename("n_capture_nuclei").reset_index()
                           .rename(columns={"w0": "w1"}).assign(sample=s))
    out["digits"].append(dg.groupby("bin").size().rename("n").reset_index().assign(sample=s))
    for k, v in origin_tables(g, dg, s).items():
        out[k] = [v]
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def true_captures(ctxs: dict, files: dict) -> pd.DataFrame:
    rows = []
    for src, (c, _) in ctxs.items():
        caps = pd.read_parquet(c.parquet_path(f"{c.run_name}__true_captures"), columns=KEY + ["cap_t"])
        w = window(caps[caps._source_file.isin(files[src])].cap_t.to_numpy())
        rows += [dict(sample=src, window=lab, n_trueNeutCap=int((w == k).sum())) for k, lab in enumerate(LIFE_LABELS)]
    return pd.DataFrame(rows)


def run(ctx: RunContext, a) -> Path:
    out = Path(a.out).resolve()
    (out / "dumps").mkdir(parents=True, exist_ok=True)
    ctxs = {"tank": (ctx, Path(a.wcsim_dir)), "world": (load_context(a.world_config), Path(a.world_wcsim_dir))}
    jobs = []
    for src, (c, wdir) in ctxs.items():
        for f in [Path(f) for f in c.inputs["root_files"]][: a.max_files]:
            n = int(re.search(r"_(\d+)\.root$", f.name).group(1))
            jobs.append((src, c, f, wdir / f"wcsim_{n}.root", out / "dumps" / f"{src}_{n}.root"))
    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(lambda j: dump_file(j[3], j[4], a), jobs))

    parts = {}
    if a.origin_only:
        for src, _, annie, _, dump in jobs:
            for k, v in origin_file(dump, src).items():
                parts.setdefault(k, []).append(v)
            print(f"[trackwindows] origin {dump.name}", flush=True)
    for src, c, annie, _, dump in ([] if a.origin_only else jobs):
        for k, v in process_file(dump, annie, src).items():
            parts.setdefault(k, []).append(v)
        print(f"[trackwindows] processed {annie.name}", flush=True)

    keys = {"creation": ["sample", "particle", "w0"],
            "creation_process": ["sample", "particle", "w0", "sproc", "parent_particle"],
            "creation_hist": ["sample", "particle", "lo", "hi"], "lifetime": ["sample", "particle", "w0", "w1"],
            "light": ["sample", "particle", "light_mask"], "carriers": ["sample", "particle", "group", "carrier"],
            "captures": ["sample", "w1"], "digits": ["sample", "bin"], "events": ["sample"],
            "origin_tracks": ["sample", "origin", "particle", "w0"], "origin_light": ["sample", "origin", "bin"],
            "neutron_end": ["sample", "w0", "w1", "reason"]}
    units = pd.concat(parts.pop("units"), ignore_index=True)
    units.to_parquet(out / "g4tw_units.parquet", index=False)
    tables = {k: pd.concat(v, ignore_index=True).groupby(keys[k], as_index=False).sum(numeric_only=True)
              for k, v in parts.items()}
    lab = dict(enumerate(LIFE_LABELS))
    for k, col, new in (("creation", "w0", "window"), ("creation_process", "w0", "window"),
                        ("lifetime", "w0", "start_window"), ("lifetime", "w1", "end_window"),
                        ("captures", "w1", "window"), ("origin_tracks", "w0", "window"),
                        ("neutron_end", "w0", "start_window"), ("neutron_end", "w1", "end_window")):
        if k in tables:
            tables[k][new] = tables[k][col].map(lab)
    for k in ("digits", "origin_light"):
        if k in tables:
            tables[k]["window"] = tables[k].bin.map(dict(enumerate(DIGIT_LABELS)))
    if a.origin_only:
        for k, t in tables.items():
            t.to_csv(out / f"g4tw_{k}.csv", index=False)
        print(f"[trackwindows] origin tables in {out}")
    else:
        files = {src: {j[2].name for j in jobs if j[0] == src} for src in ctxs}
        tables["captures"] = tables["captures"].merge(true_captures(ctxs, files), on=["sample", "window"],
                                                      how="outer")
        for k, t in tables.items():
            t.to_csv(out / f"g4tw_{k}.csv", index=False)
        pd.set_option("display.width", 200)
        print(tables["events"].to_string(index=False))
        print(tables["captures"].to_string(index=False))
        print(f"[trackwindows] CSVs in {out}")
    from .track_windows_figures import make_figures
    make_figures(out, Path(a.fig_out) if a.fig_out else out / "figs")
    return out


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc trackwindows")
    p.add_argument("--world-config")
    p.add_argument("--wcsim-dir", help="WCSim files of the tank config (wcsim_<n>.root)")
    p.add_argument("--world-wcsim-dir", help="WCSim files of the world config")
    p.add_argument("--image", help="container with ROOT for the dump (the ToolAnalysis image)")
    p.add_argument("--wcsim-lib", help="directory with libWCSimRoot.so built for that image, and include/")
    p.add_argument("--ta-repo", help="ToolAnalysis checkout whose Setup.sh sets up ROOT in the image")
    p.add_argument("--max-files", type=int, default=None, help="pilot: first N files per sample")
    p.add_argument("--jobs", type=int, default=3, help="parallel dumps")
    p.add_argument("--out", required=True, help="CSV directory")
    p.add_argument("--fig-out", help="figure directory (default <out>/figs)")
    p.add_argument("--figures-only", action="store_true", help="redraw the figures from the CSVs in --out")
    p.add_argument("--origin-only", action="store_true", help="only the origin, neutron-end and unit tables (cached dumps), then the figures")
    a = p.parse_args(list(argv) if argv else [])
    if a.figures_only:
        from .track_windows_figures import make_figures
        make_figures(Path(a.out), Path(a.fig_out) if a.fig_out else Path(a.out) / "figs")
        return
    need = ("world_config", "wcsim_dir", "world_wcsim_dir") + (() if a.origin_only else ("image", "wcsim_lib", "ta_repo"))
    missing = [k for k in need if not getattr(a, k)]
    if missing:
        p.error(f"missing {missing}")
    run(ctx, a)
