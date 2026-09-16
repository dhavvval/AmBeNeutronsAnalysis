#!/usr/bin/env python
"""Collect figures listed in docs/FIGURES.md into the five presentation decks.

Run from the repository root: python -m scripts.presentation.collect_presentation_plots
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
FIGURES_MD = HERE / "docs/FIGURES.md"
OUT = HERE / "PRESENTATION_PLOTS"

# Both source trees. Order matters only for the "first match wins" lookup.
SOURCES = [
    HERE / "slide_plots_ccinc_v3_merged",
    Path("/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/ambe_data"),
]

DECKS = {
    "deck1": dict(
        dirname="deck1_mva_neutron_definition",
        title="Deck 1 — Defining a neutron cluster with the MVA (MC, Stage 0 to Stage 3/4)",
        selection='MC neutron definition from separate truth-tag and reco-tag samples. OPTICS and ClusterFinder are evaluated separately; the frozen model is applied downstream.',
        order=["A1", "A2", "A4",
               "B1", "B3", "B4", "B5", "B6", "B7", "B8",
               "C0", "C4", "C1", "C2", "D8", "D9", "D6", "D7", "D0", "D10", "D1", "D2", "D3", "D4", "D5"],
        backup=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9",
                "Z10", "Z11", "Z12", "Z13", "D11",
                "D12", "D13", "D14", "D15", "D16", "D17", "D18"],
        appendix=[],
    ),
    "deck2": dict(
        dirname="deck2_ambe_boxcuts",
        title="Deck 2 — AmBe neutron analysis with the traditional box cuts (data, standalone)",
        selection='IC waveform gate and second-pulse veto, shared event cosmic veto, then the box selection inside the 2–67 us delayed window. Thresholds come from data_ambe2v4_all28_box.yaml.',
        order=["G1", "J1", "J2", "J3", "G2", "J9", "J10", "J4", "J5",
               "J6", "J7", "J8", "G6", "G7"],
        # G10-G12 are the SAME three heatmaps from the older published tag, so they
        # would duplicate J1-J3 in the running order. Kept as the published
        # reference rather than shown twice.
        backup=["G10", "G11", "G12", "Z17", "Z19", "Z20", "Z21", "Z22", "Z23"],
        appendix=["APPENDIX_perposition_boxcuts.pdf",
                  "APPENDIX_boxcut_v4_by_position.pdf",
                  "APPENDIX_IC_waveforms_all28runs.pdf"],
    ),
    "deck3": dict(
        dirname="deck3_ambe_mva_neutron",
        title="Deck 3 — The same AmBe analysis with the MVA neutron definition (data, standalone)",
        selection='The same waveform gate, event veto and 2–67 us delayed window as deck 2. The frozen MVA score replaces the PE/CB/hits box. Report score coverage on the same event cohort.',
        order=["K1", "K2", "K3", "K9", "K10", "K4", "K5", "K6", "K7", "K8",
               "E1", "E2", "E3"],
        # F4/F5 have MOVED to deck 5's backup. They are the MVA applied to the SPECIAL
        # runs, so they belong wherever the special runs are described; deck 3 keeps
        # F1/F2/F3/F6/F7, which are the 28 source runs.
        backup=["F1", "F2", "F3", "F6", "F7",
                "Z14", "Z15", "Z16", "Z18"],
        appendix=["APPENDIX_perposition_mvaneutron.pdf",
                  "APPENDIX_IC_waveforms_all28runs.pdf"],
    ),
    "deck4": dict(
        dirname="deck4_boxcut_vs_mva",
        title="Deck 4 — Box cuts against the MVA neutron definition",
        selection='Compare box and MVA results on matched runs and event cohorts, with common waveform gate, event veto and delayed window.',
        order=["L1", "L2", "L3", "L4", "L5", "L6",
               "G13", "G8", "G9", "G15", "G14", "G16"],
        backup=[],
        appendix=[],
    ),
    "deck5": dict(
        dirname="deck5_special_runs",
        title="Deck 5 — The special runs, described stage by stage (data, standalone)",
        selection='Special runs shown separately from campaign efficiencies. Waveform features and gates come from the processor. Cluster-only diagnostic panels apply the shared box without an event veto; the gated summary applies the processor event selector. Dummy-trigger and source/no-source comparisons remain separate.',
        order=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
               "M9", "M10", "M11", "M12",
               "M13", "M14", "M15", "M16", "M25",
               "M21", "M22", "M23", "M24", "M26",
               "M17", "M18", "M19",
               "H1"],
        backup=["H2", "H3", "H4", "F4", "F5"],
        appendix=["APPENDIX_specialruns_by_run.pdf",
                  "APPENDIX_IC_waveforms_specialruns.pdf"],
    ),
}

# Group heading:  "## F. The new definition ... — 4 figures"
# I is skipped deliberately -- "I1" against "l1"/"11" in a slide margin is unreadable.
RE_GROUP = re.compile(r"^##\s+([A-HJ-NZ])\.\s+(.*?)\s*$")
# Table row:      "| F1 | `V3AMBEN__classified_fraction` | says ... |"
RE_ROW = re.compile(r"^\|\s*([A-HJ-NZ]\d+)\s*\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*$")
# A stem prefix like V4BOX__, V3AMBE_AMBE6266__, TT_CF_GBT__ — all caps, digits,
# dots and underscores only. Used to strip the prefix off for the deck filename.
RE_PREFIX = re.compile(r"^[A-Z0-9_.]+$")


def parse_figures_md():
    """Ordered [(slot, group, stem, says)] from the group tables in docs/FIGURES.md."""
    if not FIGURES_MD.exists():
        raise SystemExit(f"missing {FIGURES_MD}")
    rows, group = [], None
    for line in FIGURES_MD.read_text().splitlines():
        m = RE_GROUP.match(line)
        if m:
            group = m.group(1)
            continue
        m = RE_ROW.match(line)
        if m and group:
            slot, stem, says = m.group(1), m.group(2).strip(), m.group(3).strip()
            if not slot.startswith(group):
                # A row whose slot letter disagrees with its heading means the file
                # has been edited into an inconsistent state; better to stop than to
                # file a figure under the wrong deck.
                raise SystemExit(f"docs/FIGURES.md: slot {slot} sits under group {group}")
            rows.append((slot, group, stem, says))
    if not rows:
        raise SystemExit("docs/FIGURES.md: parsed no figure rows — has the table format "
                         "changed? This script reads the `| A1 | `name` | says |` rows.")
    return rows


def check_assignment(rows):
    """Every slot in exactly one deck, in exactly one of its order/backup lists.

    The whole point of the rewrite is that no figure shows up twice. If that can
    drift silently the decks go back to being near-copies of each other, so it is
    checked rather than trusted.
    """
    seen, dupes = {}, []
    for key, d in DECKS.items():
        for slot in list(d["order"]) + list(d["backup"]):
            if slot in seen:
                dupes.append(f"{slot} in both {seen[slot]} and {key}")
            seen[slot] = key
    if dupes:
        raise SystemExit("deck assignment is not a partition:\n  "
                         + "\n  ".join(dupes))

    named = {r[0] for r in rows}
    unassigned = sorted(named - set(seen), key=slot_key)
    unknown = sorted(set(seen) - named, key=slot_key)
    if unassigned:
        raise SystemExit(
            f"{len(unassigned)} slot(s) in docs/FIGURES.md belong to no deck — add them "
            f"to a DECKS order/backup list: {', '.join(unassigned)}")
    if unknown:
        raise SystemExit(
            f"{len(unknown)} slot(s) are assigned to a deck but do not exist in "
            f"docs/FIGURES.md: {', '.join(unknown)}")
    return seen


def find(stem, ext):
    """First existing <source>/**/<stem>.<ext>. Returns None if nowhere."""
    name = f"{stem}.{ext}"
    for root in SOURCES:
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def slot_key(slot):
    return (slot[0], int(slot[1:]))


def plain_name(stem, taken):
    """Deck filename for a stem: prefix stripped, de-collided if it has to be.

    V4BOX__efficiency_by_position                 -> efficiency_by_position
    efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_... -> efficiency_heatmap
    TT_CF_GBT__presentable_features_byspecies     -> presentable_features_byspecies

    Z10/Z11/Z12 differ ONLY by their prefix (RT_CF_GBT, TT_OPTICS_XGB, RT_OPTICS_GBT),
    so stripping it would give three identical names inside one backup folder. When
    that happens the prefix is put back, lowercased, as a suffix.
    """
    parts = stem.split("__")
    if len(parts) > 1 and RE_PREFIX.match(parts[0]):
        # Keep every remaining segment: V3BG__truthtag_cf__top6_shape must not
        # collapse to "truthtag_cf", which says nothing about what is plotted.
        name, prefix = "__".join(parts[1:]), parts[0]
    else:
        # No prefix to strip, so the first segment IS the name and anything after
        # it is a dataset tag: efficiency_heatmap__AmBe_2.0v4__AmBe2.0v4_all28.
        name, prefix = parts[0], ""
    if name in taken:
        extra = prefix.lower() if prefix else "__".join(parts[1:])
        if extra:
            name = f"{name}__{extra}"
    taken.add(name)
    return name


TS_DIR = HERE / "TriggerSummary"

# Which trigger summaries each deck's statistics block should quote. Deck 2 gets its
# FULL sample because it needs no MVA; decks 3 and 4 get the COVERED pair, which is
# the only basis on which the two selections share a denominator.
STATS_TAGS = {
    "deck2": [("box cuts", "AmBe2.0v4_all28_box", "full sample")],
    "deck3": [("MVA neutron", "AmBe2.0v4_all28_mva_covered", "covered triggers")],
    "deck4": [("box cuts", "AmBe2.0v4_all28_box_covered", "covered triggers"),
              ("MVA neutron", "AmBe2.0v4_all28_mva_covered", "covered triggers")],
}


def _read_stats(tag):
    """Event-level clustering statistics from one trigger summary.

    Everything here is a fraction of AmBe TRIGGERS (events surviving the IC gate and
    the cosmic veto), not of clusters. That is the distinction that caused trouble:
    the pipeline's own `multiple_neutron_candidates` counts CLUSTERS sitting in
    events that had more than one cluster in total -- 19.74 % campaign-wide, and
    almost independent of which selection you run, because "clusters in the event"
    does not depend on the selection. The numbers below count EVENTS by how many
    neutrons the selection ACCEPTED, which is what actually responds to changing the
    neutron definition.
    """
    import csv as _csv
    p = TS_DIR / f"AmBeTriggerSummary_{tag}.csv"
    if not p.exists():
        return None
    tot = dict(ambe=0, one=0, multi=0, any=0, gated=0, cosmic=0, npos=0)
    with p.open() as fh:
        for r in _csv.DictReader(fh):
            tot["npos"] += 1
            tot["gated"] += int(float(r["total_events"]))
            tot["cosmic"] += int(float(r["cosmic_events"]))
            tot["ambe"] += int(float(r["ambe_triggers"]))
            tot["one"] += int(float(r["single_neutron_candidates"]))
            tot["multi"] += int(float(r["multiple_neutron_candidates"]))
            tot["any"] += int(float(r["unique_neutron_triggers"]))
    return tot


def stats_block(deck_key):
    """Markdown table of the event-level clustering statistics for a deck."""
    specs = STATS_TAGS.get(deck_key)
    if not specs:
        return []
    got = [(lbl, basis, _read_stats(tag)) for lbl, tag, basis in specs]
    got = [g for g in got if g[2]]
    if not got:
        return []
    L = ["", "## Clustering statistics", ""]
    n0 = got[0][2]["ambe"]
    same = all(g[2]["ambe"] == n0 for g in got)
    L.append(f"Denominator: **{n0:,} AmBe triggers** "
             f"({got[0][1]}) — events passing the IC gate 700–1200 and surviving the "
             f"cosmic veto." + ("  Identical for every column below, which is what "
                               "makes them comparable." if same and len(got) > 1 else ""))
    L += ["",
          "| | " + " | ".join(g[0] for g in got) + " |",
          "|---|" + "---:|" * len(got)]

    def row(name, fn):
        cells = []
        for _, _, s in got:
            v = fn(s)
            cells.append(f"{v:,} ({100*v/max(s['ambe'],1):.2f} %)")
        return f"| {name} | " + " | ".join(cells) + " |"

    L.append(row("events with **no** neutron cluster",
                 lambda s: s["ambe"] - s["any"]))
    L.append(row("events with **exactly one** neutron",
                 lambda s: s["one"]))
    L.append(row("events with **more than one** neutron",
                 lambda s: s["multi"]))
    L.append(row("**events with ≥1 neutron** (signal)",
                 lambda s: s["any"]))
    L += ["", "Of the signal events alone:", ""]
    for lbl, _, s in got:
        a = max(s["any"], 1)
        L.append(f"- **{lbl}** — {100*s['one']/a:.2f} % single, "
                 f"{100*s['multi']/a:.2f} % multiple")
    L += ["", "Single/multiple here counts accepted neutron clusters per event. "
          "The trigger-summary categories instead use numberOfClusters before selection.", ""]
    return L


def write_readme(path, deck, entries, appendix_ok, deck_key=""):
    """One README per deck: the selection in words, the statistics, the order."""
    L = [f"# {deck['title']}", "",
         f"Figure directory: `PRESENTATION_PLOTS/{deck['dirname']}/`.", "",
         "## Selection used in this deck", "", deck["selection"], ""]
    L += stats_block(deck_key)
    L += ["## Running order", "",
         "| # | file | slot | original figure | what it shows |",
         "|---|---|---|---|---|"]
    for n, (fname, slot, stem, says) in enumerate(entries["order"], 1):
        L.append(f"| {n} | `{fname}.pdf` | {slot} | `{stem}` | {says} |")
    if entries["backup"]:
        L += ["", "## Backup — `backup/`, not in the running order", "",
              "| file | slot | original figure | what it shows |",
              "|---|---|---|---|"]
        for fname, slot, stem, says in entries["backup"]:
            L.append(f"| `backup/{fname}.pdf` | {slot} | `{stem}` | {says} |")
    if appendix_ok:
        L += ["", "## Appendix — `appendix/`, multi-page PDFs", "",
              "| file | pages |", "|---|---:|"]
        for fname, pages in appendix_ok:
            L.append(f"| `appendix/{fname}` | {pages} |")
    L += ["", "---", "",
          "Every file here is also in the flat `PRESENTATION_PLOTS/` folder under its "
          "slot-prefixed original name, and `MANIFEST.csv` maps between the two. "
          "Regenerate with `python -u -m scripts.presentation.collect_presentation_plots --do build`.", ""]
    path.write_text("\n".join(L))


RE_DECK_ORDERED = re.compile(r"^\d{2}_.*\.(pdf|png)$")
RE_FLAT_SLOT = re.compile(r"^[A-HJ-NZ]\d+_.*\.(pdf|png)$")


def sweep_stale():
    """Delete the previous build's output so renumbering cannot leave duplicates."""
    gone = 0
    for f in OUT.glob("*"):
        if f.is_file() and RE_FLAT_SLOT.match(f.name):
            f.unlink()
            gone += 1
    for d in DECKS.values():
        ddir = OUT / d["dirname"]
        for f in ddir.glob("*"):
            if f.is_file() and RE_DECK_ORDERED.match(f.name):
                f.unlink()
                gone += 1
        for sub in ("backup", "appendix"):
            for f in (ddir / sub).glob("*"):
                if f.is_file() and f.suffix in (".pdf", ".png"):
                    f.unlink()
                    gone += 1
    print(f"[sweep] removed {gone} files from the previous build")


def do_build(dry_run=False):
    rows = parse_figures_md()
    rows.sort(key=lambda r: slot_key(r[0]))
    print(f"[figures] {len(rows)} figures named in docs/FIGURES.md, "
          f"groups {''.join(sorted({r[1] for r in rows}))}")

    check_assignment(rows)
    print(f"[decks] {len(DECKS)} decks, assignment is a clean partition "
          f"(no figure appears in two decks)")

    by_slot = {r[0]: r for r in rows}

    if not dry_run:
        for d in DECKS.values():
            for sub in ("", "backup", "appendix"):
                (OUT / d["dirname"] / sub).mkdir(parents=True, exist_ok=True)
        sweep_stale()

    manifest, missing, copied = [], [], 0

    # 1. the flat folder — unchanged behaviour, slot-prefixed original stems
    for slot, group, stem, says in rows:
        for ext in ("pdf", "png"):
            src = find(stem, ext)
            if src is None:
                missing.append(f"{slot}  {stem}.{ext}")
                continue
            newname = f"{slot}_{stem}.{ext}"
            if not dry_run:
                shutil.copy2(src, OUT / newname)
            copied += 1
            manifest.append(dict(
                slot=slot, group=group, deck="", role="flat", filename=newname,
                source_path=str(src.resolve()),
                source_mtime=dt.datetime.fromtimestamp(
                    src.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                size_bytes=src.stat().st_size, says=says))

    # 2. the decks — plain names, running order, backup and appendix separated
    for key, deck in DECKS.items():
        ddir = OUT / deck["dirname"]
        taken, entries = set(), {"order": [], "backup": []}
        for role, slots in (("order", deck["order"]), ("backup", deck["backup"])):
            for i, slot in enumerate(slots, 1):
                _, group, stem, says = by_slot[slot]
                name = plain_name(stem, taken)
                fname = f"{i:02d}_{name}" if role == "order" else name
                for ext in ("pdf", "png"):
                    src = find(stem, ext)
                    if src is None:
                        continue
                    dest = (ddir if role == "order" else ddir / "backup") / \
                        f"{fname}.{ext}"
                    if not dry_run:
                        shutil.copy2(src, dest)
                    copied += 1
                    manifest.append(dict(
                        slot=slot, group=group, deck=key, role=role,
                        filename=str(dest.relative_to(OUT)),
                        source_path=str(src.resolve()),
                        source_mtime=dt.datetime.fromtimestamp(
                            src.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                        size_bytes=src.stat().st_size, says=says))
                entries[role].append((fname, slot, stem, says))

        # 3. the multi-page appendices, which the old collector never routed anywhere
        app_ok = []
        for fname in deck["appendix"]:
            src = OUT / fname
            if not src.exists():
                missing.append(f"{key} appendix  {fname}")
                continue
            if not dry_run:
                shutil.copy2(src, ddir / "appendix" / fname)
            copied += 1
            app_ok.append((fname, n_pages(src)))
            manifest.append(dict(
                slot="", group="", deck=key, role="appendix",
                filename=str((ddir / "appendix" / fname).relative_to(OUT)),
                source_path=str(src.resolve()),
                source_mtime=dt.datetime.fromtimestamp(
                    src.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                size_bytes=src.stat().st_size, says="multi-page appendix"))

        if not dry_run:
            readme = HERE / "docs" / "presentations" / f"{deck['dirname']}.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            write_readme(readme, deck, entries, app_ok, key)
        print(f"[deck] {deck['dirname']:32s} {len(entries['order']):2d} in order, "
              f"{len(entries['backup']):2d} backup, {len(app_ok)} appendix")

    # Anything present in the source trees but not named in docs/FIGURES.md. This is how
    # V3AMBESPEC__classified_by_position went unnoticed — warn, do not fail.
    named = {r[2] for r in rows}
    orphans = set()
    for root in SOURCES[:1]:                  # only the curated slide dir
        for p in root.glob("V*__*.pdf"):
            if p.stem not in named:
                orphans.add(p.stem)

    if not dry_run:
        mpath = OUT / "MANIFEST.csv"
        with mpath.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)
        print(f"[manifest] wrote {mpath}  ({len(manifest)} rows)")

    print(f"[copy] {copied} files "
          f"-> {OUT}{'  (DRY RUN, nothing written)' if dry_run else ''}")

    if orphans:
        print(f"\n[warn] {len(orphans)} figure(s) exist in slide_plots_ccinc_v3_merged "
              f"but are NOT named in docs/FIGURES.md, so they are not in any deck:")
        for o in sorted(orphans):
            print(f"         {o}")

    if missing:
        print(f"\n[FAIL] {len(missing)} figure(s) named in docs/FIGURES.md are missing "
              f"on disk:")
        for m in missing:
            print(f"         {m}")
        return 1

    print(f"\n[ok] every figure named in docs/FIGURES.md was found and copied")
    print(f"\ncopy the whole set with:")
    print(f"  rsync -av <user>@<node>:{OUT}/ ./PRESENTATION_PLOTS/")
    return 0


def n_pages(pdf_path):
    """Page count of a PDF, read off the raw bytes — no PyPDF dependency here."""
    try:
        blob = pdf_path.read_bytes()
    except OSError:
        return "?"
    n = blob.count(b"/Type /Page") - blob.count(b"/Type /Pages")
    if n <= 0:
        n = blob.count(b"/Type/Page") - blob.count(b"/Type/Pages")
    return n if n > 0 else "?"


def main():
    ap = argparse.ArgumentParser(prog="collect_presentation_plots")
    # No default: `build` writes ~500 files, `check` does not. Picking one silently
    # is how you end up either surprised or convinced nothing happened.
    ap.add_argument("--do", required=True, choices=["build", "check"],
                    help="build = copy everything, write the four decks and "
                         "MANIFEST.csv. check = report what would be copied and what "
                         "is missing, writing nothing.")
    a = ap.parse_args()
    return do_build(dry_run=(a.do == "check"))


if __name__ == "__main__":
    sys.exit(main())
