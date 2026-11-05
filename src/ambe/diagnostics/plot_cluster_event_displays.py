"""
Per-event cluster displays: which true neutron capture each hit came from, and what
OPTICS / ClusterFinder / time-NMS / the time-peak finder made of it.

    ambe mc displays --config <cfg.yaml> --mode categories [--tag T] [--n-per-category 4]
                     [--methods optics,clusterfinder,optics_nms,peak]
    ambe mc displays --config <cfg.yaml> --mode particles --owner-rule plurality [--tag T]

Runs on the output of `ambe mc capture` (same --tag).  Events are picked
automatically, N per category, from `__capture_truth_clusters.parquet`:

  A  true_merge        one cluster holds light from >= 2 distinct TRUE captures
  B  trackid_artefact  the trackID label calls it "merged neutrons" (>= 2 neutron
                       trackIDs, none >= 50 %) but all the neutron light is ONE capture
  C  split             one true capture owns >= 2 clusters of the same method
  D  bkg_absorbed      a capture cluster with > 40 % non-capture (background/noise) hits

Each page has:
  (a) the delayed window: hit time vs PMT height, coloured by true capture, with the
      true capture times (after the calibrated offset) as dashed lines;
  (b) +-150 ns around the selected cluster, coloured by neutron trackID, so a
      multi-trackID-but-one-capture cluster is visible as several colours on ONE line;
  (c) the membership strip: one row per clustering method, each hit coloured by the
      cluster it was put in (grey = unclustered), so splits and merges read directly;
  (d) unrolled tank (phi vs height) with the true capture vertex, the PE centroid,
      the Gauss-Newton fit and the INIT reference;
  (e) the numbers.
Output: <run>/plots/displays[_<tag>]/ one PNG per event + one multi-page PDF + index CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from ..context import RunContext
from ..plotting import set_style

CAT_TITLES = {
    "A": "True merge — one cluster, two or more true captures",
    "B": "trackID artefact — several neutron trackIDs, one true capture",
    "C": "Split — one true capture, two or more clusters",
    "D": "Background absorbed — capture cluster with >40% non-capture hits",
}
METHOD_NAMES = {"optics": "OPTICS", "clusterfinder": "ClusterFinder",
                "optics_nms": "OPTICS + time-NMS", "clusterfinder_nms": "CF + time-NMS",
                "peak": "Time-peak", "peak_tof": "Time-peak (TOF)"}
CAP_COLORS = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377",
              "#BBBB00", "#AA4499"]


def _pick(cl: pd.DataFrame, n: int, base_methods: list, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = cl[cl.method.isin(base_methods)].copy()
    nonsig = (g.n_nonneutron + g.n_darknoise + g.n_untraced
              + (g.n_neutron - g.n_cap_hits).clip(lower=0))
    sel = {
        "A": g[g.n_caps_contrib >= 2],
        "B": g[(g.fail_reason == "merged_neutrons") & (g.n_caps_contrib == 1)
               & (g.n_contrib_neutrons >= 2) & (g.cap_dom_frac >= 0.5)],
        "D": g[(g.n_caps_contrib >= 1) & (nonsig / g.n_hits.clip(lower=1) > 0.4)
               & (g.n_hits >= 8)],
    }
    own = g[g.n_cap_dom >= 3].groupby(["method", "eventID", "cap_dom"]).cluster_id.transform("count")
    sel["C"] = g.loc[own[own >= 2].index]
    out = []
    for cat, s in sel.items():
        for m in base_methods:
            sm = s[s.method == m]
            if not len(sm):
                continue
            # one cluster per event, spread over events
            sm = sm.drop_duplicates("eventID")
            take = sm.iloc[rng.permutation(len(sm))[:n]].copy()
            take["category"] = cat
            take["n_in_category"] = len(s[s.method == m])
            out.append(take)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _unroll(x, y, z):
    return np.degrees(np.arctan2(x, z)), y


def draw(row: pd.Series, df_ev: pd.DataFrame, lbl: dict, hit_cap: np.ndarray,
         caps_in: pd.DataFrame, off: float, ref: Optional[np.ndarray], geo,
         methods: list) -> plt.Figure:
    from ..mc.cluster_features import compute_cluster_features

    t = df_ev["t"].to_numpy(float)
    y = df_ev["y"].to_numpy(float)
    tids = df_ev["ancestor_trackID"].to_numpy(int)
    is_n = (df_ev["ancestor_pdg"].to_numpy(int) == 2112) & (tids >= 0)
    is_dn = df_ev["truth_class"].to_numpy(int) == 0
    sel_m = row.method
    mask = lbl[sel_m] == row.cluster_id
    t_c = float(np.median(t[mask]))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.1, 1.0], hspace=0.45, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1:, 2])
    ax_c = fig.add_subplot(gs[2, :2])

    # (a) the whole delayed window, coloured by true capture
    def cap_color(j):
        return CAP_COLORS[j % len(CAP_COLORS)]
    ax_a.scatter(t[is_dn] / 1e3, y[is_dn], s=6, c="#DDDDDD", marker="x", lw=0.6,
                 label="dark noise")
    oth = ~is_dn & (hit_cap < 0)
    ax_a.scatter(t[oth] / 1e3, y[oth], s=8, c="#999999", label="not from a capture")
    for j in np.unique(hit_cap[hit_cap >= 0]):
        m = hit_cap == j
        ax_a.scatter(t[m] / 1e3, y[m], s=14, c=cap_color(j),
                     label=f"capture {j} ({caps_in.cap_medium.iloc[j]})")
    for j, ct in enumerate(caps_in.cap_t.to_numpy(float)):
        ax_a.axvline((ct + off) / 1e3, color=cap_color(j), ls="--", lw=0.8, alpha=0.8)
    ax_a.axvspan((t_c - 150) / 1e3, (t_c + 150) / 1e3, color="#FFDD88", alpha=0.35, lw=0)
    ax_a.set_xlabel("Hit time (µs)")
    ax_a.set_ylabel("PMT height y (m)")
    ax_a.set_title("(a) Delayed window — hits coloured by the true capture they came from")
    ax_a.legend(loc="upper right", fontsize=7, ncol=4, frameon=False)
    ax_a.grid(False)

    # (b) zoom, coloured by neutron trackID
    z = np.abs(t - t_c) <= 150
    utid = [u for u in np.unique(tids[z & is_n])]
    for k, u in enumerate(utid):
        m = z & is_n & (tids == u)
        mk = np.where(hit_cap[m] >= 0, "o", "s")
        for shape in ("o", "s"):
            mm = np.where(m)[0][mk == shape]
            if len(mm):
                ax_b.scatter(t[mm] - t_c, y[mm], s=40, marker=shape,
                             color=plt.cm.tab20(k % 20), edgecolor="k", lw=0.4,
                             label=f"trackID {u}" if shape == "o" else None)
    m = z & ~is_n
    ax_b.scatter(t[m] - t_c, y[m], s=18, c="#999999", marker="x", label="non-neutron / noise")
    for j, ct in enumerate(caps_in.cap_t.to_numpy(float)):
        if abs(ct + off - t_c) <= 150:
            ax_b.axvline(ct + off - t_c, color=cap_color(j), ls="--", lw=1.2,
                         label=f"true capture {j}")
    ax_b.set_xlim(-150, 150)
    ax_b.set_xlabel(f"Hit time − cluster median (ns)   [cluster median {t_c / 1e3:.3f} µs]")
    ax_b.set_ylabel("PMT height y (m)")
    ax_b.set_title("(b) ±150 ns around the cluster — colour = neutron trackID "
                   "(square = hit not in any capture window)")
    ax_b.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)
    ax_b.grid(False)

    # (c) membership strip
    for r, meth in enumerate(methods):
        if meth not in lbl:
            continue
        lb = lbl[meth]
        ids_here = [c for c in np.unique(lb[z]) if c >= 0]
        cmap = {c: plt.cm.Set1(i % 9) for i, c in enumerate(ids_here)}
        col = [cmap.get(l, (0.8, 0.8, 0.8, 1)) for l in lb[z]]
        ax_c.scatter(t[z] - t_c, np.full(z.sum(), r) + np.linspace(-0.25, 0.25, z.sum()),
                     s=26, c=col, edgecolor="k", lw=0.3)
        ax_c.text(152, r, f"{len(ids_here)} cl.", va="center", fontsize=8)
    for j, ct in enumerate(caps_in.cap_t.to_numpy(float)):
        if abs(ct + off - t_c) <= 150:
            ax_c.axvline(ct + off - t_c, color=cap_color(j), ls="--", lw=1.2)
    ax_c.set_yticks(range(len(methods)))
    ax_c.set_yticklabels([METHOD_NAMES.get(m, m) for m in methods])
    ax_c.set_xlim(-150, 150)
    ax_c.set_ylim(-0.6, len(methods) - 0.4)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Hit time − cluster median (ns)")
    ax_c.set_title("(c) Cluster membership per method — colour = cluster, grey = unclustered")
    ax_c.grid(False)

    # (d) unrolled tank
    xs, ys, zs = (df_ev[c].to_numpy(float) for c in ("x", "y", "z"))
    ph, hy = _unroll(xs, ys, zs)
    ax_d.scatter(ph[mask], hy[mask], s=np.clip(df_ev["pe"].to_numpy(float)[mask] * 25, 15, 150),
                 c=[plt.cm.tab20(utid.index(u) % 20) if (u in utid and n) else (0.6, 0.6, 0.6, 1)
                    for u, n in zip(tids[mask], is_n[mask])], edgecolor="k", lw=0.4)
    f = compute_cluster_features(df_ev[mask].reset_index(drop=True), geo)
    marks = []
    if pd.notna(row.get("cap_x", np.nan)):
        marks.append(("true capture", (row.cap_x, row.cap_y, row.cap_z), "*", "#CC3311", 260))
    marks.append(("PE centroid", (f["vtx_x"], f["vtx_y"], f["vtx_z"]), "o", "#000000", 90))
    if f.get("fit_converged"):
        marks.append(("Gauss–Newton fit", (f["vtx_fit_x"], f["vtx_fit_y"], f["vtx_fit_z"]),
                      "P", "#0077BB", 150))
    if ref is not None:
        marks.append(("INIT reference", tuple(ref), "D", "#009988", 80))
    for name, (vx, vy, vz), mk, c, s in marks:
        p, h = _unroll(np.array([vx]), np.array([vy]), np.array([vz]))
        ax_d.scatter(p, h, marker=mk, s=s, c=c, edgecolor="w", lw=0.8, label=name, zorder=5)
    ax_d.set_xlim(-180, 180)
    ax_d.set_ylim(-2.0, 2.0)
    ax_d.set_xlabel("φ = atan2(x, z) (deg)")
    ax_d.set_ylabel("Height y (m)")
    ax_d.set_title("(d) Unrolled tank — selected cluster")
    ax_d.legend(loc="lower left", fontsize=7, frameon=False)
    ax_d.grid(False)
    d_fit = (np.linalg.norm(np.array([f["vtx_fit_x"], f["vtx_fit_y"], f["vtx_fit_z"]])
                            - np.array([row.cap_x, row.cap_y, row.cap_z]))
             if f.get("fit_converged") and pd.notna(row.get("cap_x", np.nan)) else np.nan)
    d_cen = (np.linalg.norm(np.array([f["vtx_x"], f["vtx_y"], f["vtx_z"]])
                            - np.array([row.cap_x, row.cap_y, row.cap_z]))
             if pd.notna(row.get("cap_x", np.nan)) else np.nan)
    txt = (f"{METHOD_NAMES.get(sel_m, sel_m)} cluster {row.cluster_id}:  {row.n_hits} hits\n"
           f"neutron {row.n_neutron} | non-neutron {row.n_nonneutron} | noise {row.n_darknoise}"
           f" | untraced {row.n_untraced}\n"
           f"trackIDs {row.n_contrib_neutrons}, top trackID share {row.dom_frac:.2f}"
           f"  → production label: {'SIGNAL' if row.is_truth_neutron else 'background'}"
           f" ({row.fail_reason})\n"
           f"true captures contributing {row.n_caps_contrib}, top capture share "
           f"{row.cap_dom_frac:.2f}, within 50 ns: {row.n_caps_time}\n"
           f"|centroid − capture| {d_cen:.2f} m   |fit − capture| {d_fit:.2f} m")
    fig.text(0.02, 0.005, txt, fontsize=9, family="monospace", va="bottom")
    fig.suptitle(f"{CAT_TITLES[row.category]}\nevent {row.eventID} "
                 f"({row['_source_file']} #{row.eventNumber})", fontsize=13, y=0.995)
    return fig


PARTICLE_STYLE = {   # class -> (legend label, colour, stem-top marker)
    "muon": ("muon", "#1f5fa8", ""),
    "n_capture": ("neutron capture", "#c62828", "o"),
    "michel": ("Michel e±", "#1b8a5a", "^"),
    "other": ("other background", "#d97a00", "s"),
    "darknoise": ("dark noise", "#b8b8b8", ""),
}
_TO_STYLE = {"n_other": "other", "pion": "other", "pion_charged": "other", "pi0": "other", "proton": "other", "em": "other",
             "mixed": "other", "noise": "darknoise"}
PARTICLE_CATEGORIES = {
    "1cap": "CCinc-selected, 1 neutron capture",
    "2cap": "CCinc-selected, 2 neutron captures",
    "3cap": "CCinc-selected, 3+ neutron captures",
    "michel": "CCinc-selected, with a delayed Michel",
    "notcc": "not CCinc-selected",
}


def _style_key(cls: str) -> str:
    return _TO_STYLE.get(cls, cls)


def _pick_particle_events(ob: pd.DataFrame, n: int, min_hits: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vis = ob[ob.n_hits >= min_hits]
    ev = vis.groupby(["eventID", "row_group", "_source_file", "eventNumber", "cc_pass"]).agg(
        n_cap=("obj_class", lambda c: int((c == "n_capture").sum())),
        n_michel=("obj_class", lambda c: int((c == "michel").sum()))).reset_index()
    sel = {"1cap": ev[(ev.cc_pass == 1) & (ev.n_cap == 1)],
           "2cap": ev[(ev.cc_pass == 1) & (ev.n_cap == 2)],
           "3cap": ev[(ev.cc_pass == 1) & (ev.n_cap >= 3)],
           "michel": ev[(ev.cc_pass == 1) & (ev.n_michel >= 1)],
           "notcc": ev[(ev.cc_pass != 1) & (ev.n_cap + ev.n_michel >= 1)]}
    out = []
    for cat, g in sel.items():
        take = g.iloc[rng.permutation(len(g))[:n]].copy()
        take["category"] = cat
        take["n_in_category"] = len(g)
        out.append(take)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _part(path) -> int:
    import re
    return int(re.search(r"_(\d+)\.root$", Path(str(path)).name).group(1))


def _events_from_index(ob: pd.DataFrame, index_csv: str) -> pd.DataFrame:
    want = pd.read_csv(index_csv)
    ev = ob.groupby(["eventID", "row_group", "_source_file", "eventNumber", "cc_pass"]).agg(
        n_cap=("obj_class", lambda c: int((c == "n_capture").sum())),
        n_michel=("obj_class", lambda c: int((c == "michel").sum()))).reset_index()
    ev["_part"] = ev["_source_file"].map(_part)
    want["_part"] = want["source_file"].map(_part)
    m = want[["category", "_part", "eventNumber"]].merge(ev, on=["_part", "eventNumber"], how="left")
    if m.eventID.isna().any():
        print(f"[displays] {int(m.eventID.isna().sum())} listed events have no objects here, skipped")
    m = m.dropna(subset=["eventID"]).astype({"eventID": int, "row_group": int})
    m["n_in_category"] = -1
    return m.drop(columns="_part")


def _stems(ax, t, h, keys, x_scale=1.0):
    for k in ("darknoise", "other", "muon", "michel", "n_capture"):
        m = keys == k
        if not m.any():
            continue
        label, col, mk = PARTICLE_STYLE[k]
        ax.vlines(t[m] * x_scale, 0, h[m], color=col, lw=1.2, label=label)
        if mk:
            ax.plot(t[m] * x_scale, h[m], ls="none", marker=mk, ms=4, color=col)


def draw_particles(row: pd.Series, df_ev: pd.DataFrame, obj_cls: np.ndarray,
                   cl_ev: pd.DataFrame, methods: list, prompt_ns: float) -> plt.Figure:
    t = df_ev["t"].to_numpy(float)
    pe = df_ev["pe"].to_numpy(float)
    keys = np.array([_style_key(c) for c in obj_cls])
    fig = plt.figure(figsize=(15, 3.2 + 1.3 * len(methods)))
    gs = fig.add_gridspec(1 + len(methods), 2, width_ratios=[1, 3.2],
                          height_ratios=[2.2] + [1] * len(methods), hspace=0.55, wspace=0.12)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    pm = t < prompt_ns
    _stems(ax_p, t[pm], pe[pm], keys[pm])
    _stems(ax_d, t[~pm], pe[~pm], keys[~pm], x_scale=1e-3)
    ax_p.set_xlim(0, prompt_ns)
    ax_p.set_xlabel("Hit time (ns), prompt")
    ax_p.set_ylabel("Hit charge (PE)")
    ax_d.set_xlim(prompt_ns * 1e-3, 70)
    ax_d.set_xlabel("Hit time (µs), delayed")
    ax_d.set_ylabel("Hit charge (PE)")
    if (~pm).any():
        ax_d.set_ylim(0, 1.15 * float(pe[~pm].max()))
    handles = {}
    for ax in (ax_p, ax_d):
        for hdl, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, hdl)
    ax_d.legend(handles.values(), handles.keys(), loc="upper right", fontsize=8, frameon=False)
    for r, m in enumerate(methods):
        ax = fig.add_subplot(gs[1 + r, 1], sharex=ax_d)
        c = cl_ev[cl_ev.method == m]
        ck = np.array([_style_key(x) for x in c.tag])
        _stems(ax, c.t_med.to_numpy(float), c.n_hits.to_numpy(float), ck, x_scale=1e-3)
        ax.set_ylabel(f"{METHOD_NAMES.get(m, m)}\nhits", fontsize=8)
        if r == len(methods) - 1:
            ax.set_xlabel("Cluster median time (µs), coloured by the truth object that owns it")
    for ax in fig.axes:
        ax.grid(False)
    n_cap = int(row.n_cap)
    fig.suptitle(f"{PARTICLE_CATEGORIES[row.category]} — event {row.eventID} "
                 f"({row['_source_file']} #{row.eventNumber}): {n_cap} capture"
                 f"{'s' if n_cap != 1 else ''}, {int(row.n_michel)} Michel", fontsize=12)
    return fig


def run_particles(ctx: RunContext, args) -> None:
    import json
    import pyarrow.parquet as pq
    from ..mc.capture_truth import DEFAULTS, build_truth_objects

    tag = f"{args.tag}_{args.owner_rule}" if args.tag else args.owner_rule
    sfx = f"__{tag}"
    ob = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__truth_objects{sfx}"))
    cl = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__truth_object_clusters{sfx}"))
    off = json.loads(ctx.csv_path(f"{ctx.run_name}__capture_truth_offset"
                                  f"{('__' + args.tag) if args.tag else ''}")
                     .with_suffix(".json").read_text())["offset_ns"]
    caps = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__true_captures"))
    cfg = {**DEFAULTS, **(ctx.extra.get("capture_truth", {}) or {})}
    cfg["_full_window"] = True
    prompt_ns = float((ctx.extra.get("residual", {}) or {}).get("prompt_window_ns", 2000.0))
    methods = [m for m in args.methods.split(",") if m in set(cl.method)]
    picks = (_events_from_index(ob, args.events_from) if args.events_from
             else _pick_particle_events(ob, args.n_per_category, cfg["min_visible_hits"]))
    if picks.empty:
        raise SystemExit("[displays] no events matched any particle category")
    out_dir = ctx.plots_dir / f"displays_particles_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(ctx.parquet_path(f"{ctx.run_name}__pulses")))
    set_style()
    pdf_path = out_dir / f"{ctx.run_name}__particle_event_displays{sfx}.pdf"
    index, cache = [], {}
    with PdfPages(pdf_path) as pdf:
        for _, row in picks.iterrows():
            if row.row_group not in cache:
                ch = pf.read_row_group(int(row.row_group)).to_pandas()
                if "pe" not in ch.columns:
                    ch["pe"] = 1.0
                cache = {row.row_group: ch}
            ch = cache[row.row_group]
            df_ev = ch[ch.eventID == row.eventID].reset_index(drop=True)
            src = Path(str(row["_source_file"])).name
            c_ev = caps[(caps._source_file == src) & (caps.eventNumber == row.eventNumber)]
            obj, objs = build_truth_objects(df_ev, c_ev.cap_t.to_numpy(float), off, cfg)
            cls = np.where(obj >= 0, objs.obj_class.to_numpy()[np.maximum(obj, 0)]
                           if len(objs) else "darknoise", "darknoise")
            fig = draw_particles(row, df_ev, cls, cl[cl.eventID == row.eventID], methods,
                                 prompt_ns)
            name = f"{row.category}_ev{row.eventID}"
            fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            index.append(dict(category=row.category, eventID=row.eventID, source_file=src,
                              eventNumber=row.eventNumber, n_cap=row.n_cap,
                              n_michel=row.n_michel, n_in_category=row.n_in_category,
                              png=f"{name}.png"))
    idx = pd.DataFrame(index)
    idx.to_csv(out_dir / f"{ctx.run_name}__particle_event_displays_index{sfx}.csv", index=False)
    print(f"[displays] {len(idx)} events -> {pdf_path}")
    print(idx.groupby("category").agg(shown=("eventID", "count"),
                                      in_category=("n_in_category", "first")).to_string())


TRUE_STYLE = {"michel": ("Michel e\u00b1", "#1b8a5a"), "muon": ("muon", "#1f5fa8"),
              "other": ("other true particle", "#d97a00")}


def _membership_hits(ctx: RunContext, src: str, evn: int, cfg: dict, cleaning: str, overlay) -> pd.DataFrame:
    """One event's hits exactly as `ambe mc capture --hit-window full` saw them."""
    from ..mc import cc_selection
    from ..mc.capture_truth import PULSE_COLS, _no_waveform, add_overlay
    import pyarrow.parquet as pq
    path = ctx.parquet_path(f"{ctx.run_name}__pulses")
    avail = set(pq.read_schema(path).names)
    ch = pd.read_parquet(path, columns=[c for c in PULSE_COLS if c in avail],
                         filters=[("eventNumber", "==", int(evn))])
    ch = ch[ch._source_file.map(lambda v: Path(str(v)).name) == src]
    if "pe" not in ch.columns:
        ch["pe"] = 1.0
    ch = cc_selection.apply_residual_filter(ch, ctx, verbose=False, prompt_window_ns=0.0)
    if cleaning == "unmatched_lowpe":
        ch = ch[~(_no_waveform(ch) & (ch.pe.to_numpy() < cfg["lowpe_cut"]))]
    if overlay is not None:
        ch = add_overlay(ch, overlay, -np.inf, int(cfg.get("overlay_seed", 6256)))
    else:
        ch = ch.assign(is_overlay=0)
    if cfg.get("hit_t_min_ns") is not None:
        ch = ch[ch.t >= float(cfg["hit_t_min_ns"])]
    ch["_source_file"] = ch["_source_file"].map(lambda v: Path(str(v)).name)
    return ch.reset_index(drop=True)


def draw_membership(df_ev, obj, objs, lbl, methods, t_lo, t_hi, title):
    t = df_ev["t"].to_numpy(float) / 1e3
    y = df_ev["y"].to_numpy(float)
    ov = df_ev["is_overlay"].to_numpy(int) == 1
    win = (t >= t_lo) & (t <= t_hi)
    fig, axes = plt.subplots(1 + len(methods), 1, figsize=(13, 2.1 * (1 + len(methods))), sharex=True)
    ax = axes[0]
    cls = np.where(obj >= 0, objs.obj_class.to_numpy()[np.maximum(obj, 0)] if len(objs) else "", "")
    capj = np.where(obj >= 0, objs.cap_j.to_numpy()[np.maximum(obj, 0)] if len(objs) else -1, -1)
    dn = win & (obj < 0) & ~ov
    if dn.any():
        ax.scatter(t[dn], y[dn], s=14, marker="x", c="#bdbdbd", lw=0.8, label="MC dark noise")
    m = win & ov
    if m.any():
        ax.scatter(t[m], y[m], s=16, marker="x", c="#111111", lw=0.9, label="6256 dummy-trigger hit")
    cjs = set(capj[win & (cls == "n_capture")])
    for j, cj in enumerate(sorted(cjs, key=lambda c: np.median(t[(cls == "n_capture") & (capj == c)]))):
        m = win & (cls == "n_capture") & (capj == cj)
        ax.scatter(t[m], y[m], s=22, c=CAP_COLORS[j % len(CAP_COLORS)], label=f"neutron capture {chr(65 + j)}")
    for k, (lab, col) in TRUE_STYLE.items():
        m = win & (obj >= 0) & (cls != "n_capture") & (np.array([_style_key(c) for c in cls]) == k)
        if m.any():
            ax.scatter(t[m], y[m], s=18, c=col, label=lab)
    ax.set_ylabel("PMT y [m]")
    ax.set_title("hits coloured by the true particle that made them", fontsize=10, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    pal = list(plt.cm.tab10.colors)
    for r, meth in enumerate(methods):
        ax = axes[1 + r]
        lb = lbl[meth]
        ids = [c for c in np.unique(lb[win]) if c >= 0]
        un = win & (lb < 0)
        if un.any():
            ax.scatter(t[un], y[un], s=10, c="#d9d9d9", label="not in any cluster")
        for i, c in enumerate(ids):
            mm = win & (lb == c)
            ax.scatter(t[mm], y[mm], s=22, c=[pal[i % len(pal)]], marker="ox^sD"[i % 5],
                       label=f"cluster {i + 1}: {int((lb == c).sum())} hits")
        ax.set_ylabel("PMT y [m]")
        ax.set_title(f"{METHOD_NAMES.get(meth, meth)}: {len(ids)} cluster{'s' if len(ids) != 1 else ''} in this window "
                     f"(colour = cluster)", fontsize=10, loc="left")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    axes[-1].set_xlim(t_lo, t_hi)
    axes[-1].set_xlabel("hit time [\u00b5s]")
    for a in axes:
        a.grid(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def run_membership(ctx: RunContext, args) -> None:
    from ..mc.capture_truth import DEFAULTS, build_truth_objects, label_event, load_overlay
    cfg = {**DEFAULTS, **(ctx.extra.get("capture_truth", {}) or {})}
    cfg["peak"] = {**DEFAULTS["peak"], **(cfg.get("peak") or {})}
    cfg["_full_window"] = True
    feat = dict(ctx.extra.get("features", {}) or {})
    feat["_source_pos_m"] = None
    off = float(cfg["offset_ns"])
    methods = args.methods.split(",")
    overlay = None if args.overlay == "none" else load_overlay(args.overlay)
    caps = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__true_captures"))
    cf_all = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__clusterfinder"))
    ref = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__truth_object_clusters__{args.tag}_plurality"),
                          columns=["_source_file", "eventNumber", "method", "n_hits", "t_med"])
    out_dir = ctx.plots_dir / f"displays_membership_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    for row in pd.read_csv(args.events_from).itertuples():
        df_ev = _membership_hits(ctx, row.source_file, row.eventNumber, cfg, args.hit_cleaning, overlay)
        evid = int(df_ev.eventID.iloc[0])
        c_ev = caps[(caps._source_file == row.source_file) & (caps.eventNumber == row.eventNumber)]
        obj, objs = build_truth_objects(df_ev, c_ev.cap_t.to_numpy(float), off, cfg)
        df_cl = cf_all[cf_all.eventID == evid].reset_index(drop=True)
        lbl = label_event(df_ev, df_cl, methods, feat, cfg, None)
        r = ref[(ref._source_file == row.source_file) & (ref.eventNumber == row.eventNumber)]
        t = df_ev.t.to_numpy(float)
        for meth in methods:
            mine = sorted((int((lbl[meth] == c).sum()), round(float(np.median(t[lbl[meth] == c])), 1))
                          for c in np.unique(lbl[meth][lbl[meth] >= 0]))
            prod = sorted((int(a), round(float(b), 1)) for a, b in r[r.method == meth][["n_hits", "t_med"]].values)
            if mine != prod:
                raise SystemExit(f"[displays] {row.source_file} #{row.eventNumber} {meth}: re-clustered "
                                 f"{len(mine)} clusters differ from the production {len(prod)}")
        fig = draw_membership(df_ev, obj, objs, lbl, methods, row.t_lo_us, row.t_hi_us,
                              f"{row.label} ({row.source_file} #{row.eventNumber})")
        name = f"{row.name}"
        fig.savefig(out_dir / f"{name}.png", dpi=150)
        fig.savefig(out_dir / f"{name}.pdf")
        plt.close(fig)
        print(f"[displays] {name}: clusters identical to production -> {out_dir / name}.png")


def run(ctx: RunContext, args) -> None:
    import pyarrow.parquet as pq
    from ..mc import cc_selection
    from ..mc.capture_truth import (DEFAULTS, assign_hits_to_captures, label_event,
                                    source_from_filename)
    from ..mc.cluster_features import load_geometry
    import json

    sfx = f"__{args.tag}" if args.tag else ""
    cl = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__capture_truth_clusters{sfx}"))
    off = json.loads(ctx.csv_path(f"{ctx.run_name}__capture_truth_offset{sfx}")
                     .with_suffix(".json").read_text())["offset_ns"]
    caps = pd.read_parquet(ctx.parquet_path(f"{ctx.run_name}__true_captures"))
    cfg = {**DEFAULTS, **(ctx.extra.get("capture_truth", {}) or {})}
    cfg["peak"] = {**DEFAULTS["peak"], **(cfg.get("peak") or {})}
    feat = dict(ctx.extra.get("features", {}) or {})
    feat["_source_pos_m"] = None
    methods = args.methods.split(",")
    residual = ctx.extra.get("residual", {}) or {}
    lo, hi = cfg.get("window_ns") or [float(residual.get("prompt_window_ns", 2000.0)), 70000.0]
    geo_block = ctx.extra.get("geometry", {}) or {}
    geo = load_geometry(geo_block.get("pmt_geometry"), geo_block.get("timing_offsets"),
                        frame=geo_block.get("frame", "tank"),
                        apply_offsets=(args.timing_offsets == "on") if args.timing_offsets
                        else bool(geo_block.get("apply_timing_offsets", True)))
    cf_path = ctx.parquet_path(f"{ctx.run_name}__clusterfinder")
    cf_all = pd.read_parquet(cf_path) if cf_path.exists() else None

    picks = _pick(cl, args.n_per_category, [m for m in ("optics", "clusterfinder")
                                            if m in cl.method.unique()])
    if picks.empty:
        raise SystemExit("[displays] no events matched any category")
    out_dir = ctx.plots_dir / f"displays{('_' + args.tag) if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(ctx.parquet_path(f"{ctx.run_name}__pulses")))
    set_style()
    rng = np.random.default_rng(0)
    pdf_path = out_dir / f"{ctx.run_name}__cluster_event_displays{sfx}.pdf"
    index = []
    cache = {}
    with PdfPages(pdf_path) as pdf:
        for _, row in picks.sort_values(["category", "method"]).iterrows():
            if row.row_group not in cache:
                ch = pf.read_row_group(int(row.row_group)).to_pandas()
                if "pe" not in ch.columns:
                    ch["pe"] = 1.0
                ch = cc_selection.apply_residual_filter(ch, ctx, verbose=False)
                if cfg.get("hit_t_min_ns") is not None:
                    ch = ch[ch.t >= float(cfg["hit_t_min_ns"])]
                cache = {row.row_group: ch}
            ch = cache[row.row_group]
            df_ev = ch[ch.eventID == row.eventID].reset_index(drop=True)
            src = Path(str(row["_source_file"])).name
            c_ev = caps[(caps._source_file == src) & (caps.eventNumber == row.eventNumber)]
            caps_in = c_ev[(c_ev.cap_t + off >= lo) & (c_ev.cap_t + off <= hi)].reset_index(drop=True)
            hit_cap = assign_hits_to_captures(df_ev, caps_in.cap_t.to_numpy(float), off,
                                              cfg["hit_window_ns"])
            ref = source_from_filename(src) if cfg.get("source_from_filename") else None
            if ref is None and "nu_x" in caps_in.columns and len(caps_in):
                ref = (caps_in[["nu_x", "nu_y", "nu_z"]].iloc[0].to_numpy(float)
                       + rng.normal(0, cfg["nu_smear_m"], 3))
            df_cl = (cf_all[cf_all.eventID == row.eventID].reset_index(drop=True)
                     if cf_all is not None else None)
            lbl = label_event(df_ev, df_cl, methods, feat, cfg, ref)
            fig = draw(row, df_ev, lbl, hit_cap, caps_in, off, ref, geo, methods)
            name = f"{row.category}_{row.method}_ev{row.eventID}_cl{row.cluster_id}"
            fig.savefig(out_dir / f"{name}.png", dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
            index.append(dict(category=row.category, method=row.method,
                              eventID=row.eventID, cluster_id=row.cluster_id,
                              source_file=src, eventNumber=row.eventNumber,
                              n_in_category=row.n_in_category, png=f"{name}.png"))
    idx = pd.DataFrame(index)
    idx.to_csv(out_dir / f"{ctx.run_name}__cluster_event_displays_index{sfx}.csv", index=False)
    print(f"[displays] {len(idx)} events -> {pdf_path}")
    print(idx.groupby(["category", "method"]).agg(shown=("eventID", "count"),
                                                  in_category=("n_in_category", "first"))
          .to_string())


def cli(ctx: RunContext, argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(prog="ambe mc displays")
    p.add_argument("--tag", default="")
    p.add_argument("--timing-offsets", choices=["on", "off"], default=None,
                   help="override geometry.apply_timing_offsets (off = correct for MC)")
    p.add_argument("--n-per-category", type=int, default=4)
    p.add_argument("--methods", default="optics,clusterfinder,optics_nms,clusterfinder_nms,peak")
    p.add_argument("--mode", choices=["categories", "particles", "membership"], required=True,
                   help="categories = A-D capture pages from `ambe mc capture --truth-unit "
                        "capture`; particles = time vs PE coloured by truth object, from "
                        "`--truth-unit particle`")
    p.add_argument("--owner-rule", choices=["plurality", "majority50"], default=None,
                   help="particles mode: which `ambe mc capture` owner-rule output to read")
    p.add_argument("--events-from", default=None, metavar="INDEX_CSV",
                   help="particles mode: draw the events listed in another run's display index "
                        "(matched on file part number + eventNumber) instead of picking new ones")
    p.add_argument("--hit-cleaning", choices=["none", "unmatched_lowpe"],
                   help="membership mode: the hit list of the `ambe mc capture` run given by --tag")
    p.add_argument("--overlay", help="membership mode: random-trigger hits parquet of that run, or 'none'")
    a = p.parse_args(list(argv) if argv else [])
    if a.mode == "membership":
        miss = [f for f, v in (("--events-from", a.events_from), ("--hit-cleaning", a.hit_cleaning),
                               ("--overlay", a.overlay), ("--tag", a.tag)) if not v]
        if miss:
            p.error(f"--mode membership needs {' '.join(miss)}")
        run_membership(ctx, a)
    elif a.mode == "particles":
        if a.owner_rule is None:
            p.error("--mode particles needs --owner-rule")
        run_particles(ctx, a)
    else:
        run(ctx, a)
