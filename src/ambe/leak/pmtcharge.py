"""
Per-PMT charge deposition, before and after the box cuts --
`ambe leak pmtcharge --config <cfg>`.

Reproduces the tank-information study built for the LAPPD light-leak pair 6249/6250
(/exp/annie/data/users/dajana/LAPPDAmBedebug_PMTcharge/lappd_report_ICgated_clustering.md)
but STANDALONE per run: every figure shows only the runs of the campaign in the
config, with no external reference run drawn on it.

WHAT IS COMPUTED, per run, per PMT DetID
  before      raw hit charge in IC-gated events -- every digitized hit, cluster or not
  clustered   the subset of those hits ClusterFinder put into some cluster
  unclustered the rest; clustered + unclustered == before, exactly
  after       hit charge in clusters that pass the box cuts, in events that survive
              the cosmic veto -- i.e. the neutron-candidate sample
  fraction    clustered / before, per PMT ("charge fraction")

UNITS ARE PER IC-GATED EVENT FOR BOTH before AND after. That is deliberate and it is
the one thing to get right: the original study first normalised `before` per event and
`after` per candidate-cluster, which are different denominators, and the numbers were
not comparable until that was fixed (hence its
`pmt_charge_per_event_before_after_CONSISTENT_UNITS.csv`). Per-selected-cluster values
are also written, as separate `_per_cluster` columns, clearly labelled.

TWO TRAPS THIS AVOIDS
1. `after` is recomputed from the ROOT file, NOT from the candidate CSVs. In those
   CSVs `hitID`/`hitPE` were written via numpy's `str(array)`, which truncates long
   lists with `...`, and for v5sandi that affects 18.8-36.2 % of rows. Because
   `hitID` is sorted, the truncation drops the MIDDLE DetIDs specifically -- a
   systematic, PMT-dependent bias, which is exactly the quantity this module
   measures. Recomputing from ROOT loses nothing.
2. The clustered/unclustered split is done by HIT-LEVEL MATCHING on
   `(hitChankey, round(hitT,6), round(hitPE,6))`, never by summing the
   `Cluster_HitPE` branch. A naive sum double-counts hits claimed by overlapping
   clusters -- measured at +41 % (6249) and +46 % (6250) in the original study.

Also note `clusterPE` and summed `hitPE` are DIFFERENT ESTIMATORS. Everything here is
hit-level PE throughout, so it is internally consistent; do not compare these numbers
against a `clusterPE` mean.

The box cuts and the cosmic veto come from `build_selection(ctx, "box")`, the same
call `ambe data process` makes, so `after` is definitionally the pipeline's selection
rather than a restatement of it.

Output:
  TriggerSummary/PMTCharge_<tag>_<run>.csv          one row per PMT
  <output_root>/<run_name>/plots/pmtcharge_*.{png,pdf}

Usage:
    ambe leak pmtcharge --config configs/data_ambe2v5sandi_box.yaml
    ambe leak pmtcharge --config <cfg> --runs 6273
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BRANCHES = ["eventTimeTank", "hitDetID", "hitPE", "hitT", "hitChankey",
            "hitX", "hitY", "hitZ",
            "Cluster_HitDetID", "Cluster_HitPE", "Cluster_HitT", "Cluster_HitChankey",
            "clusterPE", "clusterChargeBalance", "clusterTime", "clusterHits"]

BATCH = 1000  # jagged hit branches are several GB per run unread in batches

BLUE, ORANGE, GREY, GREEN = "#1f77b4", "#e08214", "#999999", "#2ca02c"


def _ytag(height):
    """Source height as a filename-safe tag: y000, yp100, ym100."""
    h = int(round(height))
    if h == 0:
        return "y000"
    return f"y{'p' if h > 0 else 'm'}{abs(h)}"


def bare(ax):
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def per_pmt_charge(run, dataset, good_set, sel, src=(np.nan, np.nan, np.nan)):
    """Per-PMT before / clustered / unclustered / after for one run.

    `src` is the SOURCE position, written into every row as src_x/src_y/src_z. It is
    stored per run so that a cross-campaign comparison figure can pair runs by
    position without needing either campaign's config -- the CSV is self-describing.
    """
    import uproot

    f = uproot.open(f"{dataset}/BeamCluster_{run}.root")
    t = f["Event;3"] if "Event;3" in f else f["Event"]

    all_ett = t["eventTimeTank"].array(library="np")
    mask = np.array([int(e) in good_set for e in all_ett])
    n_ic = int(mask.sum())

    acc = {}          # DetID -> dict of running sums
    pos = {}          # DetID -> (x, y, z), first seen
    n_sel_clusters = 0
    n_sel_events = 0
    n_cosmic_events = 0

    def bucket(d):
        return acc.setdefault(int(d), dict(before_pe=0.0, clustered_pe=0.0,
                                           unclustered_pe=0.0, after_pe=0.0,
                                           n_hits=0, n_ev_with_hit=0,
                                           n_selclu_with_hit=0))

    start = 0
    for arrs in t.iterate(BRANCHES, library="np", step_size=BATCH):
        n_this = len(arrs["eventTimeTank"])
        for j in range(n_this):
            if not mask[start + j]:
                continue

            hck, ht_, hpe = arrs["hitChankey"][j], arrs["hitT"][j], arrs["hitPE"][j]
            hid = arrs["hitDetID"][j]
            hx, hy, hz = arrs["hitX"][j], arrs["hitY"][j], arrs["hitZ"][j]

            # --- clustered-hit key set for this event (hit-level, see docstring) ---
            clus_keys = set()
            for c in range(len(arrs["Cluster_HitChankey"][j])):
                c_ck = np.asarray(arrs["Cluster_HitChankey"][j][c])
                c_t = np.asarray(arrs["Cluster_HitT"][j][c])
                c_pe = np.asarray(arrs["Cluster_HitPE"][j][c])
                clus_keys.update(zip(c_ck.tolist(), np.round(c_t, 6).tolist(),
                                     np.round(c_pe, 6).tolist()))

            seen_this_event = set()
            for k in range(len(hck)):
                d = int(hid[k])
                b = bucket(d)
                pe = float(hpe[k])
                b["before_pe"] += pe
                b["n_hits"] += 1
                if d not in seen_this_event:
                    b["n_ev_with_hit"] += 1
                    seen_this_event.add(d)
                if d not in pos:
                    pos[d] = (float(hx[k]), float(hy[k]), float(hz[k]))
                key = (hck[k], round(float(ht_[k]), 6), round(pe, 6))
                if key in clus_keys:
                    b["clustered_pe"] += pe
                else:
                    b["unclustered_pe"] += pe

            # --- AFTER: exactly the pipeline's event loop, including its BREAK ---
            # process_events_efficient walks clusters IN ORDER and, on the first
            # cosmic cluster, counts the event as cosmic and `break`s. It does NOT
            # retroactively discard candidates already collected from earlier
            # clusters of the same event. So an event can be both cosmic-tagged and
            # contribute candidates. Rejecting the whole event instead gives 8,857
            # clusters against the pipeline's 9,257 on run 6273 -- i.e. it would be a
            # different sample from the one the analysis is built on, which is the
            # one thing `after` has to match. (The config prose says the veto "drops
            # the whole event"; the code breaks. They coincide only when the cosmic
            # cluster is the earliest, which is usual but not guaranteed for the
            # >100 PE branch of the veto.)
            cpe = np.asarray(arrs["clusterPE"][j], dtype=float)
            ccb = np.asarray(arrs["clusterChargeBalance"][j], dtype=float)
            ct = np.asarray(arrs["clusterTime"][j], dtype=float)
            chits = np.asarray(arrs["clusterHits"][j])

            kept_any = False
            for c in range(len(cpe)):
                if sel.cosmic(ct[c], cpe[c]):
                    n_cosmic_events += 1
                    break
                if not sel.accepts(cpe[c], ccb[c], ct[c], chits[c]):
                    continue
                kept_any = True
                n_sel_clusters += 1
                c_id = np.asarray(arrs["Cluster_HitDetID"][j][c])
                c_pe = np.asarray(arrs["Cluster_HitPE"][j][c], dtype=float)
                seen_this_cluster = set()
                for m in range(len(c_id)):
                    d = int(c_id[m])
                    b = bucket(d)
                    b["after_pe"] += float(c_pe[m])
                    if d not in seen_this_cluster:
                        b["n_selclu_with_hit"] += 1
                        seen_this_cluster.add(d)
            if kept_any:
                n_sel_events += 1
        start += n_this
        del arrs

    rows = []
    for d in sorted(acc):
        b = acc[d]
        x, y, z = pos.get(d, (np.nan, np.nan, np.nan))
        rows.append(dict(
            DetID=d, run=int(run), x=x, y=y, z=z,
            src_x=src[0], src_y=src[1], src_z=src[2],
            n_ic_events=n_ic, n_sel_clusters=n_sel_clusters,
            n_sel_events=n_sel_events, n_cosmic_events=n_cosmic_events,
            # per IC-gated event -- before and after share this denominator
            before_PE_per_event=b["before_pe"] / n_ic if n_ic else np.nan,
            after_PE_per_event=b["after_pe"] / n_ic if n_ic else np.nan,
            clustered_PE_per_event=b["clustered_pe"] / n_ic if n_ic else np.nan,
            unclustered_PE_per_event=b["unclustered_pe"] / n_ic if n_ic else np.nan,
            # per selected cluster -- a different denominator, labelled as such
            after_PE_per_cluster=(b["after_pe"] / n_sel_clusters
                                  if n_sel_clusters else np.nan),
            occupancy_selclusters=(b["n_selclu_with_hit"] / n_sel_clusters
                                   if n_sel_clusters else np.nan),
            occupancy_events=b["n_ev_with_hit"] / n_ic if n_ic else np.nan,
            clustered_fraction=(b["clustered_pe"] / b["before_pe"]
                                if b["before_pe"] > 0 else np.nan),
            before_PE_total=b["before_pe"], after_PE_total=b["after_pe"],
            n_hits=b["n_hits"],
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures -- one run per figure, no external reference run drawn on any of them
# --------------------------------------------------------------------------- #
def _save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}",
                    dpi=160 if ext == "png" else None, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  [fig] {name}.png / .pdf")


def fig_bars(df, run, outdir, label):
    import matplotlib.pyplot as plt
    d = df.sort_values("DetID")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(16, 5.4))
    ax.bar(x - 0.2, d.before_PE_per_event, 0.4, color=BLUE,
           label="before cuts (all hits, IC-gated)")
    ax.bar(x + 0.2, d.after_PE_per_event, 0.4, color=ORANGE,
           label="after box cuts (candidate clusters)")
    ax.set_xticks(x)
    ax.set_xticklabels(d.DetID.astype(int), rotation=90, fontsize=5.5)
    ax.set_xlabel("PMT DetID")
    ax.set_ylabel("PE per IC-gated event")
    ax.set_yscale("log")
    ax.set_title(f"Run {run} — per-PMT charge before vs after the box cuts, "
                 f"same denominator\n{label}")
    ax.legend(frameon=False, fontsize=9)
    bare(ax)
    _save(fig, outdir, f"pmtcharge_bars_{run}")


def fig_scatter(df, run, outdir, label):
    """Before vs after per PMT.

    PMTs whose surviving charge is exactly zero are drawn on a floor row rather than
    dropped: on a log axis they would vanish silently, and "sees light, keeps none of
    it" is a result, not a missing point.
    """
    import matplotlib.pyplot as plt
    d = df[df.before_PE_per_event > 0].copy()
    d["keep"] = d.after_PE_per_event / d.before_PE_per_event
    frac = 100 * d.after_PE_per_event.sum() / d.before_PE_per_event.sum()
    med = d.keep[d.keep > 0].median()

    nz = d[d.after_PE_per_event > 0]
    zr = d[d.after_PE_per_event <= 0]
    floor = nz.after_PE_per_event.min() * 0.25

    fig, ax = plt.subplots(figsize=(7.6, 7))
    ax.scatter(nz.before_PE_per_event, nz.after_PE_per_event, s=34, c=BLUE,
               alpha=0.85, edgecolor="k", linewidth=0.3)
    if len(zr):
        ax.scatter(zr.before_PE_per_event, np.full(len(zr), floor), s=60, c=ORANGE,
                   marker="v", edgecolor="k", linewidth=0.3, zorder=5,
                   label=f"{len(zr)} PMTs with zero surviving charge")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(d.before_PE_per_event.min() * 0.6,
                d.before_PE_per_event.max() * 1.6)
    ax.set_ylim(floor * 0.5, nz.after_PE_per_event.max() * 3)

    for _, r in d.iterrows():
        yv = r.after_PE_per_event if r.after_PE_per_event > 0 else floor
        if r.keep == 0 or r.keep > 3 * med or r.keep < med / 3:
            ax.annotate(str(int(r.DetID)), (r.before_PE_per_event, yv),
                        fontsize=7, alpha=0.9,
                        textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("before cuts: PE per IC-gated event")
    ax.set_ylabel("after box cuts: PE per IC-gated event")
    ax.set_title(f"Run {run} — per-PMT charge survival through the box cuts\n"
                 f"{frac:.2f} % of tank charge kept, median per-PMT retention "
                 f"{100*med:.2f} %; labelled PMTs are 3x off that")
    bare(ax)
    _save(fig, outdir, f"pmtcharge_scatter_before_after_{run}")


def fig_fraction(df, run, outdir, label):
    """Charge fraction per PMT: how much of each PMT's light is clustered at all."""
    import matplotlib.pyplot as plt
    d = df[df.before_PE_per_event > 0].sort_values("DetID")
    x = np.arange(len(d))
    mean_f = (d.clustered_PE_per_event.sum() / d.before_PE_per_event.sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                                 gridspec_kw={"width_ratios": [2, 1]})
    a1.bar(x, 100 * d.clustered_fraction, 0.8, color=GREEN)
    a1.set_xticks(x[::2])
    a1.set_xticklabels(d.DetID.astype(int)[::2], rotation=90, fontsize=5.5)
    a1.set_xlabel("PMT DetID")
    a1.set_ylabel("clustered charge fraction (%)")
    a1.set_title(f"Run {run} — per-PMT clustered charge fraction\n"
                 f"tank-wide {100*mean_f:.1f} % of hit charge is clustered")
    bare(a1)

    a2.scatter(d.before_PE_per_event, 100 * d.clustered_fraction, s=30, c=GREEN,
               alpha=0.8, edgecolor="k", linewidth=0.3)
    a2.set_xscale("log")
    a2.set_xlabel("PE per IC-gated event (before cuts)")
    a2.set_ylabel("clustered charge fraction (%)")
    a2.set_title("Clustered fraction vs how much light the PMT sees")
    bare(a2)
    fig.tight_layout(w_pad=2.5)
    _save(fig, outdir, f"pmtcharge_clustered_fraction_{run}")


def fig_spatial(df, run, outdir, label):
    import matplotlib.pyplot as plt
    d = df[(df.before_PE_per_event > 0) & df.x.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, col, ttl in (
            (axes[0], "before_PE_per_event", "before cuts"),
            (axes[1], "after_PE_per_event", "after box cuts"),
            (axes[2], "clustered_fraction", "clustered fraction")):
        v = d[col]
        sc = ax.scatter(d.z, d.y, c=v, s=70, cmap="viridis",
                        edgecolor="k", linewidth=0.3)
        fig.colorbar(sc, ax=ax, label=("PE / IC-gated event"
                                       if col != "clustered_fraction" else "fraction"))
        # hitX/Y/Z are in METRES (tank spans about -1.55..+1.48 in y), unlike the
        # source positions in the efficiency maps, which are in cm.
        ax.set_xlabel("PMT z (m)")
        ax.set_ylabel("PMT y (m)")
        ax.set_title(ttl)
        bare(ax)
    fig.suptitle(f"Run {run} — per-PMT charge in tank coordinates, "
                 f"before vs after the box cuts")
    fig.tight_layout()
    _save(fig, outdir, f"pmtcharge_spatial_{run}")


def fig_campaign(frames, outdir, label):
    """The campaign's own runs against each other. No external reference run."""
    import matplotlib.pyplot as plt
    runs = sorted(frames)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    cols = [BLUE, ORANGE, GREEN]
    for i, r in enumerate(runs):
        d = frames[r][frames[r].before_PE_per_event > 0].sort_values("DetID")
        a1.plot(d.DetID, d.before_PE_per_event, ".", ms=6, color=cols[i % 3],
                label=f"{r}")
        a2.plot(d.DetID, 100 * d.clustered_fraction, ".", ms=6, color=cols[i % 3],
                label=f"{r}")
    a1.set_yscale("log")
    a1.set_xlabel("PMT DetID"); a1.set_ylabel("PE per IC-gated event")
    a1.set_title("Per-PMT charge before cuts, the three runs together")
    a1.legend(frameon=False, fontsize=9)
    bare(a1)
    a2.set_xlabel("PMT DetID"); a2.set_ylabel("clustered charge fraction (%)")
    a2.set_title("Per-PMT clustered charge fraction")
    a2.legend(frameon=False, fontsize=9)
    bare(a2)
    fig.tight_layout(w_pad=2.5)
    _save(fig, outdir, "pmtcharge_campaign_overlay")


def fig_allruns(frames, outdir, stage, label, notes=None):
    """One figure, ALL runs, grouped bars over every PMT DetID.

    `stage` is "before" or "after". Runs are ordered by source height and then by
    campaign, so each v5sandi run sits next to the reference run at the same Port 5
    position and the pairing is readable off the legend without a second figure.
    Hue encodes the position; the reference run of each pair is drawn hatched.
    """
    import matplotlib.pyplot as plt

    col = "before_PE_per_event" if stage == "before" else "after_PE_per_event"

    def srcy(df):
        v = df.src_y.dropna()
        return float(v.iloc[0]) if len(v) else np.nan

    # Group by source height; inside a group the campaign run first, reference second.
    # "reference" comes from the dict key prefix set by --compare, NOT from sort
    # order -- ordering by run number put the reference first for y=+100 (6266 <
    # 6274) and mislabelled the pair.
    items = []
    for key, df in frames.items():
        run = int(df.run.iloc[0])
        is_ref = str(key).startswith("ref:")
        items.append((srcy(df), is_ref, run, df))
    items.sort(key=lambda t: (-t[0] if not np.isnan(t[0]) else 0, t[1], t[2]))

    ids = sorted({int(i) for _, _, _, df in items for i in df.DetID})
    xpos = np.arange(len(ids))
    n = len(items)
    width = 0.9 / n

    # one hue per source height, so a pair shares a colour
    heights = sorted({t[0] for t in items}, reverse=True)
    hue = {h: c for h, c in zip(heights, [BLUE, ORANGE, GREEN, "#7b3294"])}

    fig, ax = plt.subplots(figsize=(23, 6.4))
    for i, (h, is_ref, run, df) in enumerate(items):
        s = df.set_index(df.DetID.astype(int))[col]
        vals = np.array([s.get(d, 0.0) for d in ids])
        ax.bar(xpos + (i - (n - 1) / 2) * width, vals, width,
               color=hue.get(h, GREY), alpha=0.95 if not is_ref else 0.55,
               hatch="" if not is_ref else "///",
               edgecolor="k", linewidth=0.25,
               label=f"{run}  (y={h:+.0f}){'  reference' if is_ref else '  v5sandi'}")

    ax.set_xticks(xpos[::2])
    ax.set_xticklabels([ids[k] for k in range(0, len(ids), 2)], rotation=90,
                       fontsize=5.5)
    ax.set_xlabel("PMT DetID")
    ax.set_ylabel("PE per IC-gated event")
    ax.set_yscale("log")
    ttl = ("BEFORE the box cuts — all hits in IC-gated events"
           if stage == "before" else
           "AFTER the box cuts — hits in surviving candidate clusters")
    ax.set_title(f"Per-PMT charge, {ttl}\n"
                 f"solid = AmBe 2.0v5sandi, hatched = same-position reference run; "
                 f"colour = Port 5 height. {label}")
    ax.legend(frameon=False, fontsize=9, ncol=min(n, 6), loc="upper center")
    bare(ax)
    _save(fig, outdir, f"pmtcharge_ALLRUNS_{stage}")


def fig_pair(cdf, rdf, height, outdir, label):
    """One port position, one figure: before on top, after below, two runs each.

    The per-position view the all-runs figure cannot give -- with only two runs per
    axes the bars are wide enough to read a single DetID off, and the before/after
    panels share the x order so a PMT can be tracked between them.
    """
    import matplotlib.pyplot as plt

    crun, rrun = int(cdf.run.iloc[0]), int(rdf.run.iloc[0])
    ids = sorted({int(i) for i in cdf.DetID} | {int(i) for i in rdf.DetID})
    xpos = np.arange(len(ids))

    def series(df, col):
        s = df.set_index(df.DetID.astype(int))[col]
        return np.array([s.get(d, 0.0) for d in ids])

    fig, axes = plt.subplots(2, 1, figsize=(21, 10.5), sharex=True)
    for ax, col, stage in ((axes[0], "before_PE_per_event", "BEFORE the box cuts"),
                           (axes[1], "after_PE_per_event", "AFTER the box cuts")):
        cv, rv = series(cdf, col), series(rdf, col)
        ax.bar(xpos - 0.21, cv, 0.42, color=BLUE, edgecolor="k", linewidth=0.25,
               label=f"{crun}  (v5sandi)")
        ax.bar(xpos + 0.21, rv, 0.42, color=ORANGE, alpha=0.9, hatch="///",
               edgecolor="k", linewidth=0.25, label=f"{rrun}  (leak-free reference)")
        ax.set_ylabel("PE per IC-gated event")
        ax.set_yscale("log")
        # ratio quoted in the panel title, since that is the comparison being made
        m = (cv > 0) & (rv > 0)
        med = np.median(cv[m] / rv[m])
        ax.set_title(f"{stage} — median per-PMT ratio {crun}/{rrun} = {med:.2f}",
                     fontsize=12)
        ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center")
        bare(ax)

    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(ids, rotation=90, fontsize=5.5)
    axes[1].set_xlabel("PMT DetID")
    fig.suptitle(f"Per-PMT charge at Port 5, y = {height:+.0f} — "
                 f"run {crun} vs leak-free run {rrun}\n{label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # Name carries the plot TYPE last and avoids "+" in filenames: the old
    # pmtcharge_PAIR_/PAIRSCATTER_ pair sorted with the scatter first ("S" <
    # "_" in ASCII), which buried the bar plots in a directory listing.
    _save(fig, outdir, f"pmtcharge_pair_{_ytag(height)}_{crun}_vs_{rrun}_BARS")


def fig_pair_scatter(cdf, rdf, height, outdir, label):
    """Run-vs-run scatter for one port position, before and after the cuts.

    Same axes convention as the original 6249/6250 study: one point per PMT,
    reference run on x, campaign run on y, with the identity line -- on a
    run-vs-run scatter that line is what the eye reads the comparison against.
    PMTs with zero charge on either axis are drawn on a floor/edge row rather than
    dropped silently by the log scale.
    """
    import matplotlib.pyplot as plt

    crun, rrun = int(cdf.run.iloc[0]), int(rdf.run.iloc[0])
    ids = sorted({int(i) for i in cdf.DetID} & {int(i) for i in rdf.DetID})

    def series(df, col):
        s = df.set_index(df.DetID.astype(int))[col]
        return np.array([s.get(d, 0.0) for d in ids])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for ax, col, stage in ((axes[0], "before_PE_per_event", "BEFORE the box cuts"),
                           (axes[1], "after_PE_per_event", "AFTER the box cuts")):
        y, x = series(cdf, col), series(rdf, col)
        both = (x > 0) & (y > 0)
        med = np.median(y[both] / x[both])
        r = np.corrcoef(np.log(x[both]), np.log(y[both]))[0, 1]

        lo = min(x[both].min(), y[both].min()) * 0.5
        hi = max(x[both].max(), y[both].max()) * 2.0
        ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1.2, zorder=0,
                label="equal charge")
        ax.scatter(x[both], y[both], s=34, c=BLUE, alpha=0.85, edgecolor="k",
                   linewidth=0.3)

        # zero on one axis only -- park on the corresponding edge, do not drop
        edge = lo * 1.4
        zx, zy = (x <= 0) & (y > 0), (y <= 0) & (x > 0)
        if zx.any():
            ax.scatter(np.full(zx.sum(), edge), y[zx], s=55, c=ORANGE, marker="<",
                       edgecolor="k", linewidth=0.3, zorder=5,
                       label=f"{zx.sum()} PMT(s) zero in {rrun}")
        if zy.any():
            ax.scatter(x[zy], np.full(zy.sum(), edge), s=55, c=GREEN, marker="v",
                       edgecolor="k", linewidth=0.3, zorder=5,
                       label=f"{zy.sum()} PMT(s) zero in {crun}")

        for i, d in enumerate(ids):
            if not both[i]:
                continue
            rr = (y[i] / x[i]) / med
            if rr > 2.5 or rr < 1 / 2.5:
                ax.annotate(str(d), (x[i], y[i]), fontsize=7, alpha=0.9,
                            textcoords="offset points", xytext=(4, 3))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"run {rrun} (leak-free reference): PE per IC-gated event")
        ax.set_ylabel(f"run {crun} (v5sandi): PE per IC-gated event")
        ax.set_title(f"{stage}\nmedian ratio {med:.2f}, log-charge r = {r:.2f}; "
                     f"labelled PMTs are 2.5x off the median")
        # upper left: the data sits on the diagonal, so that corner is always free
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        bare(ax)

    fig.suptitle(f"Per-PMT charge at Port 5, y = {height:+.0f} — "
                 f"run {crun} vs leak-free run {rrun}\n{label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5)
    _save(fig, outdir, f"pmtcharge_pair_{_ytag(height)}_{crun}_vs_{rrun}_SCATTER")


def run(ctx, argv=None):
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(prog="ambe leak pmtcharge")
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--refit", action="store_true",
                    help="recompute runs whose CSV already exists")
    ap.add_argument("--compare", nargs="+", default=None, metavar="CSV",
                    help="extra PMTCharge_*.csv files (any campaign) to include in "
                         "the combined ALLRUNS before/after bar figures. They are "
                         "read, never recomputed; each carries its own source "
                         "position, so runs pair up by position automatically.")
    ap.add_argument("--only-allruns", action="store_true",
                    help="skip the per-run figures and draw only the combined pair")
    ap.add_argument("--paired", action="store_true",
                    help="draw one bar figure and one scatter figure PER PORT "
                         "POSITION (campaign run vs its --compare reference, before "
                         "and after) instead of the single all-runs pair. Positions "
                         "with no reference supplied are skipped.")
    args = ap.parse_args(argv or [])

    from .features import _good_events, _runs_from_config
    from ..data.processor import build_selection

    st = ctx.extra.get("stage1", {})
    tag = st.get("tag", ctx.run_name)
    dataset = st["waveform_dir"]
    sel, cuts_obj = build_selection(ctx, "box")
    label = ctx.cuts.get("selection_label", sel.label)

    # Source position per run, so the CSV is self-describing for cross-campaign
    # comparison. Uses the same resolution order the pipeline does: the config's
    # extra_source_positions first, then the shared campaign map.
    from ..data.processor import AmBeNeutronProcessing, CutCriteria
    _extra = {int(k): tuple(float(x) for x in v)
              for k, v in (st.get("extra_source_positions") or {}).items()}
    _proc = AmBeNeutronProcessing(cuts=CutCriteria(), extra_positions=_extra)

    outdir = Path(ctx.run_dir) / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = Path("TriggerSummary")
    ts.mkdir(parents=True, exist_ok=True)

    runs = [str(r) for r in (args.runs or _runs_from_config(ctx))]
    print(f"[pmtcharge] tag {tag}, selection: {label}")
    print(f"[pmtcharge] before AND after are per IC-gated event (same denominator)")
    print(f"[pmtcharge] 'after' is recomputed from ROOT, not from the candidate CSVs "
          f"(their hitID lists are truncated)")

    frames = {}
    for r in runs:
        csv_path = ts / f"PMTCharge_{tag}_{r}.csv"
        if csv_path.exists() and not args.refit:
            print(f"[pmtcharge] run {r}: reusing {csv_path.name}")
            frames[r] = pd.read_csv(csv_path)
        else:
            good, how = _good_events(ctx, r)
            try:
                src = _proc.get_source_location(int(r))
            except ValueError:
                print(f"[pmtcharge] run {r}: no source position on record; "
                      f"src_* columns will be blank and this run cannot be paired "
                      f"by position in the combined figures")
                src = (np.nan, np.nan, np.nan)
            df = per_pmt_charge(r, dataset, good, sel, src=src)
            df.to_csv(csv_path, index=False)
            n = df.iloc[0]
            print(f"[pmtcharge] run {r}: {len(df)} PMTs, "
                  f"{int(n.n_ic_events)} IC-gated events, "
                  f"{int(n.n_sel_clusters)} selected clusters, "
                  f"{int(n.n_cosmic_events)} cosmic-vetoed events "
                  f"({how}) -> {csv_path.name}")
            frames[r] = df

        d = frames[r]
        kept = 100 * d.after_PE_per_event.sum() / d.before_PE_per_event.sum()
        clf = 100 * d.clustered_PE_per_event.sum() / d.before_PE_per_event.sum()
        print(f"[pmtcharge] run {r}: tank charge kept by the cuts {kept:.2f} %, "
              f"clustered fraction {clf:.1f} %")

        if not args.only_allruns:
            fig_bars(d, r, outdir, label)
            fig_scatter(d, r, outdir, label)
            fig_fraction(d, r, outdir, label)
            fig_spatial(d, r, outdir, label)

    if len(frames) > 1 and not args.only_allruns:
        fig_campaign(frames, outdir, label)

    # Combined before/after bars, this campaign plus any --compare CSVs.
    combined = dict(frames)
    for p in (args.compare or []):
        pp = Path(p)
        if not pp.exists():
            raise SystemExit(f"[pmtcharge] --compare file not found: {pp}")
        cdf = pd.read_csv(pp)
        if "src_y" not in cdf.columns:
            raise SystemExit(
                f"[pmtcharge] {pp.name} has no src_y column, so it cannot be paired "
                f"by position -- regenerate it with the current version")
        key = f"ref:{int(cdf.run.iloc[0])}"
        combined[key] = cdf
        print(f"[pmtcharge] comparison run {int(cdf.run.iloc[0])} "
              f"(y={cdf.src_y.iloc[0]:+.0f}) from {pp.name}")

    if args.paired:
        # Pair each campaign run with the reference at the same source height.
        def _y(df):
            v = df.src_y.dropna()
            return float(v.iloc[0]) if len(v) else None

        refs = {}
        for k, df in combined.items():
            if str(k).startswith("ref:"):
                h = _y(df)
                if h is not None:
                    refs[h] = df
        drawn = 0
        for k, df in combined.items():
            if str(k).startswith("ref:"):
                continue
            h = _y(df)
            if h is None or h not in refs:
                print(f"[pmtcharge] run {int(df.run.iloc[0])} (y={h}): no reference "
                      f"supplied at this position, skipping the paired figures")
                continue
            fig_pair(df, refs[h], h, outdir, label)
            fig_pair_scatter(df, refs[h], h, outdir, label)
            drawn += 1
        if not drawn:
            raise SystemExit("[pmtcharge] --paired drew nothing: no --compare "
                             "reference matched any campaign run's position")
    elif len(combined) > 1:
        for stage in ("before", "after"):
            fig_allruns(combined, outdir, stage, label)

    print(f"[pmtcharge] figures -> {outdir}")
    return 0


def cli(ctx, argv=None):
    return run(ctx, argv)
