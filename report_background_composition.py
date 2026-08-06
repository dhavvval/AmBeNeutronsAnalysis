#!/usr/bin/env python3
"""
report_background_composition.py — quantify WHAT the delayed background is made of.

The MVA is binary (neutron-dominated cluster vs everything else). This script does
not change that; it answers the separate question of what the "everything else" is
physically, using the per-hit lineage that BackTracker stores.

Three ancestry axes are reported side by side, because each answers a different
question and only reading all three avoids a wrong conclusion:

  bg_class      DirectParent_ImmediateAncestorClass — the particle that emitted the
                Cherenkov light, with e+- carriers skipped but gammas NOT skipped.
                Reads "photon" for the large majority of hits, so on its own it says
                almost nothing about the physics origin.
  origin_pdg    first ancestor that is neither e+- NOR gamma (added in processor.py).
                This is the one that separates "a proton did this" from "a pi0 did
                this" from "a capture gamma did this".
  root_pdg      DirectParent_RootAncestorPDG — the generator-level primary at the top
                of the chain. Correct but coarse: a mu- root covers both direct muon
                light and everything the muon spawned.

Populations (each reported separately, never mixed):
  all           every detector hit in the processed files
  cc            hits in CC-passing events (cc_pass == 1)
  delayed       cc, and t > residual.prompt_window_ns — the window the neutron
                search actually runs in, i.e. the background that costs us efficiency

Everything is normalised two ways: as a percentage of its population, and as a rate
per CC-passing event, which is the number that says how much of this background a
single beam event actually delivers.

Outputs (under <output_root>/<run_name>/):
  csv/<run>__bkg_hit_ancestry<tag>.csv      hit-level, all three axes, all populations
  csv/<run>__bkg_lineage_chains<tag>.csv    top-N literal chains ("e- <- gamma <- pi0")
  csv/<run>__bkg_cluster_yields<tag>.csv    cluster-level dominant-species yields
  csv/<run>__bkg_compcut_impact<tag>.csv    what the composition cut removes, by origin
  plots/<run>__bkg_composition<tag>.pdf     the same, as a multi-page PDF

Usage:
    python report_background_composition.py --config configs/<run>.yaml \
        [--methods optics clusterfinder] [--top-chains 30] \
        [--comp-n-neutron-max 10] [--comp-frac-nonneutron-min 0.40] [--out-tag ""]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ambe.context import load_context                      # noqa: E402
from ambe.mc.processor import BG_CLASS_LABELS, pdg_short    # noqa: E402

NEUTRON_CLASSES = [1, 2, 3, 4]

TRUTH_CLASS_LABELS = {
    0: "dark_noise", 1: "primary_neutron", 2: "secondary_n_from_p",
    3: "secondary_n_from_n", 4: "secondary_n_from_other",
    -5: "nonneutron_physics",
}

# Hit columns the aggregation touches. Kept explicit and checked against the file
# schema so a rename upstream fails loudly instead of producing an empty table.
HIT_COLS = ["eventID", "t", "truth_class", "is_untraced", "cc_pass",
            "bg_class", "root_pdg", "lineage_status", "lineage_chain",
            "origin_pdg", "origin_class"]

POPULATIONS = ["all", "cc", "delayed"]


def _origin_label(pdg: int) -> str:
    """
    Label for origin_pdg. The -5 sentinel does NOT mean "untraced" on this axis: it
    means the chain is complete but purely electromagnetic, i.e. a gamma sits at the
    top and there is no non-EM ancestor to report. Calling it untraced would read as
    a tracing failure, which it is not.
    """
    return "pure_em (no non-EM ancestor)" if int(pdg) == -5 else pdg_short(int(pdg))


# --------------------------------------------------------------------------- #
# Hit-level aggregation (chunked — the pulses parquet does not fit in RAM)
# --------------------------------------------------------------------------- #
def aggregate_hits(pulses_path: Path, prompt_window_ns: float, top_chains: int):
    """
    One streaming pass over the pulses parquet.

    Returns (counters, meta) where counters[population][axis] is a Counter over
    category labels, plus per-population hit totals and the CC event count.
    """
    pf = pq.ParquetFile(str(pulses_path))
    avail = set(pf.schema_arrow.names)
    cols = [c for c in HIT_COLS if c in avail]
    missing = [c for c in HIT_COLS if c not in avail]
    has_origin = "origin_pdg" in avail

    counters = {p: {ax: Counter() for ax in
                    ["truth_class", "bg_class", "origin", "root", "chain",
                     "lineage_status"]}
                for p in POPULATIONS}
    totals = {p: 0 for p in POPULATIONS}
    bkg_totals = {p: 0 for p in POPULATIONS}
    # origin split per chain, only for the delayed background — this is what
    # resolves a chain like "e- <- gamma <- ..." into its real progenitor.
    chain_origin: dict[str, Counter] = {}
    cc_events: set[int] = set()
    all_events: set[int] = set()

    n_groups = pf.num_row_groups
    for i in range(n_groups):
        d = pf.read_row_group(i, columns=cols).to_pandas()
        if not len(d):
            continue

        all_events.update(d["eventID"].unique().tolist())
        cc = (d["cc_pass"].to_numpy().astype(bool) if "cc_pass" in d.columns
              else np.ones(len(d), bool))
        if "cc_pass" in d.columns:
            cc_events.update(d.loc[cc, "eventID"].unique().tolist())

        delayed = cc & (d["t"].to_numpy(float) > prompt_window_ns)
        masks = {"all": np.ones(len(d), bool), "cc": cc, "delayed": delayed}

        cls    = d["truth_class"].to_numpy(int)
        is_bkg = (cls == -5)

        for pop, m in masks.items():
            totals[pop] += int(m.sum())
            sub = d[m]
            sm_bkg = is_bkg[m]
            bkg_totals[pop] += int(sm_bkg.sum())

            for c, n in sub["truth_class"].value_counts().items():
                counters[pop]["truth_class"][TRUTH_CLASS_LABELS.get(int(c), str(c))] += int(n)
            if "lineage_status" in sub.columns:
                for s, n in sub["lineage_status"].value_counts().items():
                    counters[pop]["lineage_status"][int(s)] += int(n)

            # All ancestry axes are background-only (truth_class == -5). Including
            # neutron hits would drown them: 99.4% of capture hits report
            # bg_class == photon, which is a capture artefact, not background.
            b = sub[sm_bkg]
            if not len(b):
                continue
            for c, n in b["bg_class"].value_counts().items():
                counters[pop]["bg_class"][BG_CLASS_LABELS.get(int(c), str(c))] += int(n)
            for r, n in b["root_pdg"].value_counts().items():
                counters[pop]["root"][pdg_short(int(r))] += int(n)
            if has_origin:
                for o, n in b["origin_pdg"].value_counts().items():
                    counters[pop]["origin"][_origin_label(int(o))] += int(n)
            if "lineage_chain" in b.columns:
                for ch, n in b["lineage_chain"].value_counts().items():
                    counters[pop]["chain"][ch] += int(n)

        # per-chain origin breakdown, delayed background only
        if has_origin and "lineage_chain" in d.columns:
            bd = d[delayed & is_bkg]
            if len(bd):
                for (ch, o), n in bd.groupby(
                        ["lineage_chain", "origin_pdg"]).size().items():
                    chain_origin.setdefault(ch, Counter())[_origin_label(int(o))] += int(n)

        print(f"[bkg-report] hits chunk {i+1}/{n_groups}  "
              f"({100*(i+1)/n_groups:.0f}%)  total={totals['all']}", flush=True)

    meta = {
        "totals": totals,
        "bkg_totals": bkg_totals,
        "n_cc_events": len(cc_events) if cc_events else len(all_events),
        "n_events": len(all_events),
        "has_origin": has_origin,
        "missing_cols": missing,
        "chain_origin": chain_origin,
        "top_chains": top_chains,
    }
    return counters, meta


def hit_ancestry_table(counters, meta) -> pd.DataFrame:
    """Long-format hit-level table: one row per (population, axis, category)."""
    rows = []
    n_cc = max(meta["n_cc_events"], 1)
    for pop in POPULATIONS:
        for axis in ["truth_class", "bg_class", "origin", "root"]:
            c = counters[pop][axis]
            # truth_class is normalised to ALL hits in the population; the ancestry
            # axes are background-only and must use the background denominator.
            denom = (meta["totals"][pop] if axis == "truth_class"
                     else meta["bkg_totals"][pop])
            denom = max(denom, 1)
            for cat, n in c.most_common():
                rows.append({
                    "population": pop,
                    "axis": axis,
                    "denominator": ("all_hits" if axis == "truth_class"
                                    else "background_hits"),
                    "category": cat,
                    "n_hits": n,
                    "pct_of_denominator": round(100.0 * n / denom, 3),
                    "hits_per_cc_event": round(n / n_cc, 4),
                })
    return pd.DataFrame(rows)


def chain_table(counters, meta) -> pd.DataFrame:
    """Top-N literal lineage chains in the delayed background."""
    c = counters["delayed"]["chain"]
    denom = max(meta["bkg_totals"]["delayed"], 1)
    n_cc = max(meta["n_cc_events"], 1)
    rows = []
    for rank, (ch, n) in enumerate(c.most_common(meta["top_chains"]), start=1):
        oc = meta["chain_origin"].get(ch, Counter())
        dom, dom_n = (oc.most_common(1)[0] if oc else ("", 0))
        rows.append({
            "rank": rank,
            "lineage_chain": ch,
            "n_hits": n,
            "pct_of_delayed_bkg": round(100.0 * n / denom, 3),
            "hits_per_cc_event": round(n / n_cc, 4),
            "dominant_origin": dom,
            "dominant_origin_pct_of_chain": (round(100.0 * dom_n / n, 1) if n else 0.0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cluster-level aggregation
# --------------------------------------------------------------------------- #
CLUSTER_AXES = ["dominant_class", "bg_dominant_species",
                "origin_dominant_species", "root_dominant_species"]


def cluster_tables(feat_path: Path, methods, n_cc_events: int,
                   comp_n_max: int, comp_frac_min: float):
    """
    Cluster-level yields per method and the composition-cut impact.

    Returns (yields_df, compcut_df, n_clusters_by_method).
    """
    df = pd.read_parquet(feat_path)
    if "cc_pass" in df.columns:
        df = df[df["cc_pass"] == 1]
    n_cc = max(n_cc_events, 1)

    yield_rows, cut_rows, counts = [], [], {}
    for method in methods:
        sub = df[df["method"] == method]
        counts[method] = len(sub)
        if not len(sub):
            print(f"[bkg-report] WARNING: no clusters for method={method}")
            continue

        is_sig = sub["dominant_class"].isin(NEUTRON_CLASSES)
        prompt = (sub["is_prompt_cluster"] == 1 if "is_prompt_cluster" in sub.columns
                  else pd.Series(False, index=sub.index))

        for axis in CLUSTER_AXES:
            if axis not in sub.columns:
                continue
            # The ancestry axes describe the background, so restrict to the
            # background clusters; dominant_class describes everything.
            scope = sub if axis == "dominant_class" else sub[~is_sig]
            if not len(scope):
                continue
            scope_prompt = prompt.reindex(scope.index).fillna(False)
            vals = scope[axis].fillna("none")
            for cat, n in vals.value_counts().items():
                sel = (vals == cat)
                lab = (TRUTH_CLASS_LABELS.get(int(cat), str(cat))
                       if axis == "dominant_class" else str(cat))
                yield_rows.append({
                    "method": method,
                    "axis": axis,
                    "scope": ("all_clusters" if axis == "dominant_class"
                              else "background_clusters"),
                    "category": lab,
                    "n_clusters": int(n),
                    "pct_of_scope": round(100.0 * n / len(scope), 3),
                    "clusters_per_cc_event": round(n / n_cc, 5),
                    "n_prompt": int((sel & scope_prompt).sum()),
                    "n_delayed": int((sel & ~scope_prompt).sum()),
                })

        # MVA pool sizes, using mva_analysis.py's own definitions.
        yield_rows.append({
            "method": method, "axis": "mva_pool", "scope": "all_clusters",
            "category": "signal (neutron-dominated)", "n_clusters": int(is_sig.sum()),
            "pct_of_scope": round(100.0 * is_sig.sum() / len(sub), 3),
            "clusters_per_cc_event": round(is_sig.sum() / n_cc, 5),
            "n_prompt": int((is_sig & prompt).sum()),
            "n_delayed": int((is_sig & ~prompt).sum()),
        })
        yield_rows.append({
            "method": method, "axis": "mva_pool", "scope": "all_clusters",
            "category": "background (all non-neutron)", "n_clusters": int((~is_sig).sum()),
            "pct_of_scope": round(100.0 * (~is_sig).sum() / len(sub), 3),
            "clusters_per_cc_event": round((~is_sig).sum() / n_cc, 5),
            "n_prompt": int((~is_sig & prompt).sum()),
            "n_delayed": int((~is_sig & ~prompt).sum()),
        })

        # Composition cut, broken down by origin species.
        if {"n_neutron", "frac_nonneutron"} <= set(sub.columns):
            removed = ((sub["n_neutron"] <= comp_n_max) &
                       (sub["frac_nonneutron"] >= comp_frac_min))
            axis = ("origin_dominant_species"
                    if "origin_dominant_species" in sub.columns
                    else "bg_dominant_species")
            cats = sub[axis].fillna("none") if axis in sub.columns else None
            groups = ([("ALL", pd.Series(True, index=sub.index))] +
                      ([(str(c), cats == c) for c in cats.value_counts().index]
                       if cats is not None else []))
            for lab, g in groups:
                n_tot = int(g.sum())
                if not n_tot:
                    continue
                n_rm = int((g & removed).sum())
                n_sig_rm = int((g & removed & is_sig).sum())
                cut_rows.append({
                    "method": method,
                    "split_axis": axis,
                    "category": lab,
                    "n_before": n_tot,
                    "n_removed": n_rm,
                    "n_kept": n_tot - n_rm,
                    "pct_removed": round(100.0 * n_rm / n_tot, 2),
                    "n_signal_removed": n_sig_rm,
                })

    return pd.DataFrame(yield_rows), pd.DataFrame(cut_rows), counts


# --------------------------------------------------------------------------- #
# Plots — title only, no gridlines / reference lines / annotations
# --------------------------------------------------------------------------- #
def _barh(ax, labels, values, title):
    y = np.arange(len(labels))
    ax.barh(y, values, color="#3b6ea5")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.grid(False)


def make_pdf(pdf_path: Path, hits_df, chains_df, yields_df, cut_df, meta, ctx,
             prompt_window_ns, comp_n_max, comp_frac_min):
    with PdfPages(pdf_path) as pdf:
        # Page 1 — the three ancestry axes for the delayed background, side by side
        d = hits_df[hits_df["population"] == "delayed"]
        axes_present = [a for a in ["bg_class", "origin", "root"]
                        if len(d[d["axis"] == a])]
        if axes_present:
            fig, axs = plt.subplots(1, len(axes_present),
                                    figsize=(5 * len(axes_present), 5.5))
            axs = np.atleast_1d(axs)
            names = {"bg_class": "immediate ancestor (gamma not skipped)",
                     "origin": "first non-EM ancestor",
                     "root": "generator primary"}
            for ax, a in zip(axs, axes_present):
                s = d[d["axis"] == a].head(12)
                _barh(ax, s["category"].tolist(), s["pct_of_denominator"].tolist(),
                      f"{names[a]} — % of delayed background hits")
            fig.suptitle(ctx.title(f"Delayed background ancestry, t > {prompt_window_ns:.0f} ns, "
                                   f"CC-passing events"), fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Page 2 — hit population composition by truth_class
        t = hits_df[hits_df["axis"] == "truth_class"]
        if len(t):
            fig, axs = plt.subplots(1, len(POPULATIONS), figsize=(15, 4.5))
            for ax, pop in zip(np.atleast_1d(axs), POPULATIONS):
                s = t[t["population"] == pop]
                _barh(ax, s["category"].tolist(), s["pct_of_denominator"].tolist(),
                      f"{pop} hits (n={meta['totals'][pop]}) — % by truth class")
            fig.suptitle(ctx.title("Hit truth-class composition by population"), fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Page 3 — top lineage chains
        if len(chains_df):
            s = chains_df.head(25)
            fig, ax = plt.subplots(figsize=(10, 0.32 * len(s) + 2))
            _barh(ax, s["lineage_chain"].tolist(), s["pct_of_delayed_bkg"].tolist(),
                  ctx.title("Most common delayed-background lineage chains "
                            "(% of background hits, nearest ancestor first)"))
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Page 4+ — cluster-level dominant species per method
        for method in yields_df["method"].unique() if len(yields_df) else []:
            m = yields_df[yields_df["method"] == method]
            axes_c = [a for a in CLUSTER_AXES if len(m[m["axis"] == a])]
            if not axes_c:
                continue
            fig, axs = plt.subplots(1, len(axes_c), figsize=(5 * len(axes_c), 5))
            for ax, a in zip(np.atleast_1d(axs), axes_c):
                s = m[m["axis"] == a].sort_values("n_clusters", ascending=False).head(12)
                _barh(ax, s["category"].tolist(), s["clusters_per_cc_event"].tolist(),
                      f"{a} — clusters per CC event")
            fig.suptitle(ctx.title(f"Cluster background composition — {method}"), fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Last page — composition-cut impact
        if len(cut_df):
            methods = cut_df["method"].unique()
            fig, axs = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
            for ax, method in zip(np.atleast_1d(axs), methods):
                s = cut_df[cut_df["method"] == method].sort_values(
                    "n_before", ascending=False).head(12)
                _barh(ax, s["category"].tolist(), s["pct_removed"].tolist(),
                      f"{method} — % of clusters removed")
            fig.suptitle(ctx.title(f"Composition cut impact by origin species "
                                   f"(n_neutron <= {comp_n_max} and "
                                   f"frac_nonneutron >= {comp_frac_min})"), fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run YAML (same one Stage 0/1 used)")
    p.add_argument("--methods", nargs="+", default=["optics", "clusterfinder"],
                   help="clustering methods to break down (default: both)")
    p.add_argument("--top-chains", type=int, default=30,
                   help="how many literal lineage chains to tabulate (default 30)")
    p.add_argument("--comp-n-neutron-max", type=int, default=10,
                   help="composition cut: n_neutron <= this (default 10)")
    p.add_argument("--comp-frac-nonneutron-min", type=float, default=0.40,
                   help="composition cut: frac_nonneutron >= this (default 0.40)")
    p.add_argument("--out-tag", default="",
                   help="suffix appended to every output filename")
    args = p.parse_args()

    ctx = load_context(args.config)
    tag = args.out_tag
    prompt_window_ns = float(
        (ctx.extra.get("residual") or {}).get("prompt_window_ns", 2000.0))

    pulses_path = ctx.parquet_path(f"{ctx.run_name}__pulses")
    feat_path   = ctx.parquet_path(f"{ctx.run_name}__cluster_features")
    if not pulses_path.exists():
        sys.exit(f"[bkg-report] {pulses_path} not found — run `ambe mc process` first")

    print(f"[bkg-report] run          : {ctx.run_name}")
    print(f"[bkg-report] pulses       : {pulses_path}")
    print(f"[bkg-report] prompt window: {prompt_window_ns:.0f} ns "
          f"(delayed population is t > this, CC-passing events only)")

    counters, meta = aggregate_hits(pulses_path, prompt_window_ns, args.top_chains)
    if meta["missing_cols"]:
        print(f"[bkg-report] NOTE: columns absent from the pulses parquet, their "
              f"axes are omitted: {meta['missing_cols']}")
    if not meta["has_origin"]:
        print("[bkg-report] WARNING: no origin_pdg column — this parquet predates the "
              "first-non-EM-ancestor addition. The origin axis will be empty; "
              "re-run Stage 0 to populate it.")

    print(f"[bkg-report] events: {meta['n_events']}  CC-passing: {meta['n_cc_events']}")
    for pop in POPULATIONS:
        print(f"[bkg-report]   {pop:<8s} hits={meta['totals'][pop]:>12d}  "
              f"background(-5)={meta['bkg_totals'][pop]:>12d}")

    hits_df   = hit_ancestry_table(counters, meta)
    chains_df = chain_table(counters, meta)

    if feat_path.exists():
        yields_df, cut_df, counts = cluster_tables(
            feat_path, args.methods, meta["n_cc_events"],
            args.comp_n_neutron_max, args.comp_frac_nonneutron_min)
        print(f"[bkg-report] clusters by method: {counts}")
    else:
        print(f"[bkg-report] NOTE: {feat_path} absent — cluster-level sections skipped "
              f"(run `ambe mc features` first)")
        yields_df, cut_df = pd.DataFrame(), pd.DataFrame()

    out = {
        "bkg_hit_ancestry":   hits_df,
        "bkg_lineage_chains": chains_df,
        "bkg_cluster_yields": yields_df,
        "bkg_compcut_impact": cut_df,
    }
    for base, df in out.items():
        if not len(df):
            continue
        path = ctx.csv_path(f"{ctx.run_name}__{base}{tag}")
        df.to_csv(path, index=False)
        print(f"[bkg-report] wrote {len(df):>5d} rows -> {path}")

    pdf_path = ctx.plots_dir / f"{ctx.run_name}__bkg_composition{tag}.pdf"
    make_pdf(pdf_path, hits_df, chains_df, yields_df, cut_df, meta, ctx,
             prompt_window_ns, args.comp_n_neutron_max, args.comp_frac_nonneutron_min)
    print(f"[bkg-report] wrote plots -> {pdf_path}")

    # ---- console summary: the delayed background, resolved three ways --------
    n_cc = max(meta["n_cc_events"], 1)
    print(f"\n[bkg-report] ===== DELAYED BACKGROUND (t > {prompt_window_ns:.0f} ns, "
          f"CC events) =====")
    print(f"[bkg-report] {meta['bkg_totals']['delayed']} non-neutron hits over "
          f"{n_cc} CC events = "
          f"{meta['bkg_totals']['delayed']/n_cc:.2f} hits/event")
    for axis, header in [("bg_class", "immediate ancestor (gamma NOT skipped)"),
                         ("origin",   "first non-EM ancestor  <-- physics origin"),
                         ("root",     "generator primary (top of chain)")]:
        s = hits_df[(hits_df["population"] == "delayed") & (hits_df["axis"] == axis)]
        if not len(s):
            continue
        print(f"[bkg-report] {header}:")
        for _, r in s.head(10).iterrows():
            print(f"      {r['category']:<20s} {r['n_hits']:>10d}  "
                  f"{r['pct_of_denominator']:>6.1f}%   "
                  f"{r['hits_per_cc_event']:>8.3f} hits/CC evt")
    if len(chains_df):
        print("[bkg-report] top chains (nearest ancestor first):")
        for _, r in chains_df.head(10).iterrows():
            print(f"      {r['pct_of_delayed_bkg']:>5.1f}%  {r['n_hits']:>9d}   "
                  f"{r['lineage_chain']}   [origin: {r['dominant_origin']}]")
    if len(yields_df):
        print("[bkg-report] cluster background composition (per CC event):")
        for method in yields_df["method"].unique():
            m = yields_df[(yields_df["method"] == method) &
                          (yields_df["axis"] == "origin_dominant_species")]
            if not len(m):
                m = yields_df[(yields_df["method"] == method) &
                              (yields_df["axis"] == "bg_dominant_species")]
            print(f"    {method}:")
            for _, r in m.sort_values("n_clusters", ascending=False).head(10).iterrows():
                print(f"      {r['category']:<20s} {r['n_clusters']:>8d}  "
                      f"{r['pct_of_scope']:>6.1f}%   "
                      f"{r['clusters_per_cc_event']:>8.4f} clusters/CC evt")


if __name__ == "__main__":
    main()
