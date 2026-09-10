#!/usr/bin/env python
"""
collect_presentation_plots.py — put every talk figure in ONE folder, and build the
five decks, so the whole set can be copied off the node in a single rsync.

THE PROBLEM THIS SOLVES. The figures currently live in two unrelated trees whose
top-level directory names differ by exactly one letter:

  /exp/annie/app/users/dajana/AmBeNeutronsAnalysis/slide_plots_ccinc_v3_merged/
                                          ^^^^^^^ note the "s"
  /exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/ambe_data/<run>/plots/
                                          ^^^^^^ no "s"

Both exist and both are real. Anyone copying by hand gets this wrong.

WHY FIVE DECKS, AND WHY NOTHING IS IN TWO OF THEM. The previous version assigned
groups F, G and ALL of Z to both decks, so deck1 held 68 figures and deck2 held 63
with ~55 in common: opening either one showed mostly the same plots. The decks are
now split by the QUESTION each answers, and every figure lands in EXACTLY ONE deck:

  deck1_mva_neutron_definition   MC only. How a neutron cluster is defined, Stage 0
                                 through Stage 3/4, and what the classifier delivers.
  deck2_ambe_boxcuts             AmBe data, traditional selection, standalone. The
                                 neutron is  0 < PE <= 100, 0 < CB < 0.45,
                                 t >= 2 us, hits >= 5  after the IC gate and the
                                 cosmic veto. NO MVA anywhere in this deck.
  deck3_ambe_mva_neutron         AmBe data, the same analysis run with the MVA
                                 neutron definition instead of the box cuts.
  deck4_boxcut_vs_mva            The two definitions measured against each other.
                                 The only place both appear.
  deck5_special_runs             The special runs, described stage by stage -- IC
                                 waveform stage, tank cluster stage, Stage-2 box --
                                 ungated as well as gated, because the two dummy
                                 triggers have ZERO IC-passing triggers and are
                                 therefore invisible in any gated figure. Deck 2's H1-H4 and deck 3's
                                 F4/F5 moved here: the special runs are now described
                                 in one place instead of appearing as four bars at the
                                 end of two other decks. Legends here name what a
                                 curve IS ("dummy trigger (LAPPD on)", "no source",
                                 ...) rather than its run number; 1D figures come in
                                 normalised and raw-counts pairs; and after-the-box
                                 panels are re-ranged to the range the cut admits.

That split is asserted, not assumed: a figure assigned to two decks, or to none, is
a hard error (see check_assignment).

WHAT ELSE IS DIFFERENT FROM A FLAT DUMP

  * Running order first, backup out of the way. Each deck is <NN>_<name>.pdf in the
    order it is presented; anything FIGURES.md calls backup/zoom/optional goes to
    <deck>/backup/ instead of being interleaved.
  * The multi-page APPENDIX_*.pdf documents are now COPIED INTO the deck they belong
    to, under <deck>/appendix/. They were previously written to PRESENTATION_PLOTS/
    and never routed anywhere, which is why the port-position material looked missing.
  * Plain filenames. The V4BOX__ / V3AMBEN__ / TT_CF_GBT__ prefixes are stripped, so
    a file is named for what it shows. The slot and the original stem are preserved
    in each deck's README.md and in MANIFEST.csv, so every reference in
    ANALYSIS_ccinc_v3_FULL.md / FIGURES.md still resolves by search.

Files are COPIED, not symlinked: symlinks into two source trees do not survive an
rsync or an scp off this node, which is the entire point of the folder.

Figure names, order and the "says" line are read from FIGURES.md, which stays the
single source of truth for what is in the talk. This script never keeps its own list
of figures. It keeps the deck assignment and the selection wording, which are the two
things FIGURES.md does not encode per-figure.

Usage:
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python -u collect_presentation_plots.py --do build
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
FIGURES_MD = HERE / "FIGURES.md"
OUT = HERE / "PRESENTATION_PLOTS"

# Both source trees. Order matters only for the "first match wins" lookup.
SOURCES = [
    HERE / "slide_plots_ccinc_v3_merged",
    Path("/exp/annie/app/users/dajana/AmBeNeutronsAnalysis/ambe_output/ambe_data"),
]

# ════════════════════════════════════════════════════════════════════════════
# the deck definitions
# ════════════════════════════════════════════════════════════════════════════
# Per-SLOT, not per-group: group G splits across two decks. G1-G7 and G10-G12 are
# the box cuts standing on their own; G8, G9 and G13-G16 put the MVA next to them
# and belong with the comparison.
DECKS = {
    "deck1": dict(
        dirname="deck1_mva_neutron_definition",
        title="Deck 1 — Defining a neutron cluster with the MVA (MC, Stage 0 to Stage 3/4)",
        selection="MVA neutron. Trained on CC-inclusive MC; the deliverable is the "
                  "GBT on ClusterFinder clusters, truth-tag streamline, taken at the "
                  "MC 80 % signal-efficiency working point.",
        order=["A1", "A2", "A3", "A4",
               "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
               "C0", "C4", "C1", "C2", "C3",
               "D8", "D9", "D6", "D7", "D0", "D10", "D1", "D2", "D3", "D4", "D5"],
        # D11 is the co-occurrence matrix behind D3's top-ten bar chart. D12-D18 are
        # the other seven renderings of D5 -- {linear, log} y x {solid, dashed} lines
        # x {with, without} the dark-noise class, same data and same classes
        # throughout -- so the deck can be swapped to whichever reads best in the room
        # by editing D5's stem here, without rerunning anything.
        backup=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9",
                "Z10", "Z11", "Z12", "Z13", "D11",
                "D12", "D13", "D14", "D15", "D16", "D17", "D18"],
        appendix=[],
    ),
    "deck2": dict(
        dirname="deck2_ambe_boxcuts",
        title="Deck 2 — AmBe neutron analysis with the traditional box cuts (data, standalone)",
        selection="Box cuts. IC waveform gate 700 < IC_adjusted < 1200 plus the "
                  "second-pulse veto, then the cosmic veto (any cluster with "
                  "clusterTime < 2 us or clusterPE > 100 drops the whole event), then "
                  "a neutron is  0 < clusterPE <= 100,  0 < charge balance < 0.45,  "
                  "clusterTime >= 2 us,  clusterHits >= 5. No MVA and no MC anywhere "
                  "in this deck. "
                  "CAPTURE-TIME FIT WINDOW: 2-67 us, changed 2026-08-27 from 10-67 so "
                  "that one window is used everywhere and the fit sees the "
                  "thermalisation rise the model exists to describe. The anchor is now "
                  "29.417 +- 0.221 us (19 positions) / 29.477 +- 0.193 us (26); the old "
                  "30.53 / 30.535 numbers were 10-67 results and are NOT comparable.",
        # J9/J10 sit immediately after G5, i.e. with the capture-time block and not
        # with the heatmaps at the top: they ARE capture time, drawn on the heatmap
        # grid. Putting them next to J1-J3 would read as three efficiency maps.
        # H1-H4 have MOVED to deck 5. They are the special runs scored by the box, and
        # deck 5 is now the one place the special runs are described; four bars at the
        # end of the campaign deck was exactly the presentation this rewrite removes.
        order=["G1", "J1", "J2", "J3", "G2", "G4", "G5", "J9", "J10", "J4", "J5",
               "J6", "J7", "J8", "G3", "G6", "G7"],
        # G10-G12 are the SAME three heatmaps from the older published tag, so they
        # would duplicate J1-J3 in the running order. Kept as the published
        # reference rather than shown twice.
        backup=["G10", "G11", "G12", "Z17", "Z19", "Z20", "Z21", "Z22", "Z23"],
        appendix=["APPENDIX_perposition_boxcuts.pdf",
                  "APPENDIX_boxcut_v4_by_position.pdf",
                  # The IC gate is Stage 1, upstream of the neutron definition, so
                  # this SAME file is also deck 3's appendix. It is the one figure
                  # legitimately shared between decks. Deck 5 does NOT take it -- it
                  # is the 28 SOURCE runs and says nothing about the special runs,
                  # which is why deck 5 now builds its own IC book.
                  "APPENDIX_IC_waveforms_all28runs.pdf"],
    ),
    "deck3": dict(
        dirname="deck3_ambe_mva_neutron",
        title="Deck 3 — The same AmBe analysis with the MVA neutron definition (data, standalone)",
        selection="MVA neutron. The frozen classifier from Deck 1 applied to AmBe "
                  "data: a cluster is a neutron iff GBT score > 0.423621, the MC "
                  "80 % signal-efficiency point. There is NO PE, charge-balance, "
                  "time or hits box -- the MVA REPLACES the box cuts rather than "
                  "being applied on top of them. Shared upstream with Deck 2 is "
                  "only the IC gate 700-1200 and the cosmic veto. Where a figure "
                  "shows a split, 'MVA-rejected' means score <= 0.423621. "
                  "CAVEAT: the scoring stage covered 225,810 of the 237,590 "
                  "clusters Stage 1 finds (95.04 %), so efficiency here is quoted "
                  "on COVERED triggers -- compare 46.15 % with Deck 2's 55.99 % "
                  "on the same triggers, never with its full-sample 57.25 %. "
                  "CAPTURE-TIME FIT WINDOW: 2-67 us (was 10-67 before "
                  "2026-08-27) -- see deck 2.",
        # F1-F7 and Z14-Z18 are the NESTED study: the MVA applied ON TOP of the box
        # cuts (its own titles say "MVA neutron applied to box cuts ... 225,945
        # candidates"). This deck applies the MVA INSTEAD of the box, so those
        # figures contradict the deck they were in. They now live in deck4, which is
        # where a both-cuts figure belongs, or in backup.
        #
        # The per-feature books are gone too. Features are the TRAINING input, frozen
        # in deck 1; this deck only applies the model. Worse, they were built by
        # boxcut_v4_appendix --do features, whose _scored() applies the box cuts
        # internally, so they describe the nested sample and not this selection.
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
        selection="Both. Every figure here puts the box cuts (PE <= 100, CB < 0.45, "
                  "t >= 2 us, hits >= 5) and the MVA neutron (GBT eff80) on the same "
                  "clusters or the same source positions. This is the only deck in "
                  "which the two definitions appear together. "
                  "CAPTURE-TIME FIT WINDOW: 2-67 us (was 10-67 before "
                  "2026-08-27). Every tau on these slides moved; the matched "
                  "box-to-MVA shift is now -1.546 +- 0.311 us (4.98 sigma) "
                  "against the old -0.989 +- 0.363 (2.73 sigma).",
        order=["L1", "L2", "L3", "L4", "L5", "L6",
               "G13", "G8", "G9", "G15", "G14", "G16"],
        backup=[],
        appendix=[],
    ),
    "deck5": dict(
        dirname="deck5_special_runs",
        title="Deck 5 — The special runs, described stage by stage (data, standalone)",
        selection="The SAME flow decks 2 and 3 use -- Stage-1 IC waveform gate "
                  "700 < IC_adjusted < 1200 plus the second-pulse veto, then the tank "
                  "cluster stage, then the Stage-2 box PE <= 100, CB < 0.45, t >= 2 "
                  "us, hits >= 5 -- but reporting the DISTRIBUTIONS at each stage "
                  "rather than the survivor count, and reporting them UNGATED as well "
                  "as gated. THREE DIFFERENT KINDS OF RUN, and the labels must not "
                  "blur them: 6254/6256 are DUMMY (pulser) TRIGGERS with the LAPPD on "
                  "and off -- the readout fires on a schedule, so each event is a "
                  "random slice of the tank and they measure the ACCIDENTAL cluster "
                  "rate (0.159 box-accepted clusters per trigger, the two agreeing to "
                  "0.3 %); 6264 is a genuine NO-SOURCE run with the real trigger logic "
                  "live, so its 808 in-gate waveforms are FALSE STARTS of the IC "
                  "trigger, which a dummy-trigger run can never measure because it "
                  "never exercises the trigger logic. Ungated is not a relaxation: a "
                  "scheduled readout carries no BGO gamma, so 6254 and 6256 have ZERO "
                  "waveforms in the IC window (0 of 7,933 and 0 of 25,525) and every "
                  "gated plot of those two is blank; what they contain is still a "
                  "measurement. THE STAGE FIGURES USE TWO RUN SETS AND NEVER MIX THEM "
                  "IN ONE FRAME. M13-M16 + M25 are 6264 (no source) against 6265 "
                  "(source in), both at Port 5, (0,0,0) -- the same position to the "
                  "centimetre, so a difference between them is the source and not the "
                  "geometry -- through all three stages: before the AmBe waveform cut, "
                  "after it, after the box (176,634 -> 11,210 -> 787 against 75,259 -> "
                  "31,081 -> 15,165). M21-M24 + M26 are the two DUMMY triggers alone, "
                  "ungated, BOX ONLY: they have zero IC-passing waveforms, so the "
                  "waveform-cut stage does not exist for them and cannot be a shared "
                  "row. The earlier version showed all four runs in one frame and used "
                  "6266 (Port 5, y = 100) as 6264's contrast, which put a ~10 pp "
                  "geometry difference inside the comparison. M1-M26 are these runs "
                  "stage by stage; H1 is "
                  "the campaign's own box-cut scoring of them; backup carries the "
                  "figures that restate those numbers in bar form (H2/H3), the "
                  "narrower-sample shape plot (H4), the IC 2D (M20) and the MVA "
                  "neutron applied to these runs (F4/F5). NO efficiency "
                  "and NO capture time is quoted for any run in this deck, by design "
                  "-- these are not source runs and a campaign number computed on "
                  "them would be wrong however it came out. "
                  "ONE DENOMINATOR MISMATCH TO EXPECT, IT IS NOT A CONTRADICTION: "
                  "M1 says 6256 has 25,525 waveforms and H1 says 4,254. The M slots read "
                  "the PIPELINE's own WaveformFeatures parquet (every part file "
                  "`ambe data process` saw); H1 reads the special-runs diagnostics "
                  "join, which covers a narrower part range. Both report ZERO "
                  "IC-passing triggers, which is the claim. Quote M1's denominator "
                  "for anything about what the run contains.",
        # Stage 1 (M1-M8), then Stage 2 (M9-M18), then the no-source charge answer
        # (M19), then H1 -- the campaign's own scoring of the same runs, which only
        # means something once the distributions behind it have been shown.
        #
        # H2 (yield per trigger), H3 (cosmic fraction) and H4 (cluster shapes) are in
        # backup, not the running order: H4 duplicates M9/M11 on a narrower sample,
        # and H2/H3 restate in bar form what M1 and M17 already give as numbers.
        # M20 (IC vs fprompt 2D) likewise -- M2 and M3 carry both its projections.
        # The two stage blocks are kept whole and consecutive: the 6264-vs-6265 pair
        # (M13-M16) closed by its cut flow (M25), then the dummy pair (M21-M24) closed
        # by its own (M26). Interleaving them by plane would put a dummy trigger next
        # to a source run again, which is the thing this split exists to stop.
        order=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
               "M9", "M10", "M11", "M12",
               "M13", "M14", "M15", "M16", "M25",
               "M21", "M22", "M23", "M24", "M26",
               "M17", "M18", "M19",
               "H1"],
        backup=["M20", "H2", "H3", "H4", "F4", "F5"],
        # Both previous entries were inherited from deck 2 and neither described these
        # runs: APPENDIX_IC_waveforms_all28runs.pdf is the 28 SOURCE runs, and
        # APPENDIX_perposition_specialruns_boxcuts.pdf is per SOURCE POSITION, which is
        # close to meaningless for a run with no source in the tank. Replaced by two
        # books built from these four runs, by boxcut_v4_special.py --do diag.
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
    """Ordered [(slot, group, stem, says)] from the group tables in FIGURES.md."""
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
                raise SystemExit(f"FIGURES.md: slot {slot} sits under group {group}")
            rows.append((slot, group, stem, says))
    if not rows:
        raise SystemExit("FIGURES.md: parsed no figure rows — has the table format "
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
            f"{len(unassigned)} slot(s) in FIGURES.md belong to no deck — add them "
            f"to a DECKS order/backup list: {', '.join(unassigned)}")
    if unknown:
        raise SystemExit(
            f"{len(unknown)} slot(s) are assigned to a deck but do not exist in "
            f"FIGURES.md: {', '.join(unknown)}")
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
        # Two stems can reduce to the same plain name. Put back whatever
        # distinguishes them: the all-caps prefix when there was one (Z10/Z11/Z12
        # differ only by RT_CF_GBT / TT_OPTICS_XGB / RT_OPTICS_GBT), otherwise the
        # trailing segments that were dropped (B7/B8 are
        # selection_2x2_benchmark__ambe__gbt__eff and __purity, which would both
        # have become "selection_2x2_benchmark" and been told apart only by their
        # running-order number).
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
    L += ["",
          "*Do not confuse the \"more than one\" row with the 19.74 % that appears in "
          "G7 and in the published tables.* **Both count events** — what differs is "
          "which. G7 counts events that had more than one cluster **in total before "
          "any selection** (`numberOfClusters != 1`, straight off the ntuple) and "
          "yielded at least one accepted neutron; because that condition does not "
          "depend on the selection, the number barely moves between them "
          "(19.74 % box, 17.23 % MVA). The rows above count events by how many "
          "neutrons the selection **accepted**, which is the quantity that responds "
          "to changing the neutron definition.", ""]
    return L


def write_readme(path, deck, entries, appendix_ok, deck_key=""):
    """One README per deck: the selection in words, the statistics, the order."""
    L = [f"# {deck['title']}", "",
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
          "Regenerate with `python -u collect_presentation_plots.py --do build`.", ""]
    path.write_text("\n".join(L))


# Deck filenames carry their POSITION (07_roc_by_config.pdf), so inserting one
# figure renumbers everything after it. Copying on top of the previous build then
# leaves the old numbers behind and the deck shows two "13"s — the same figure
# under its old and new number, indistinguishable in a file browser. Every build
# therefore removes what the previous build wrote, first. Only files matching the
# patterns this script itself produces are touched; anything a human dropped into
# the folder under another name survives.
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
    print(f"[figures] {len(rows)} figures named in FIGURES.md, "
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
            write_readme(ddir / "README.md", deck, entries, app_ok, key)
        print(f"[deck] {deck['dirname']:32s} {len(entries['order']):2d} in order, "
              f"{len(entries['backup']):2d} backup, {len(app_ok)} appendix")

    # Anything present in the source trees but not named in FIGURES.md. This is how
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
              f"but are NOT named in FIGURES.md, so they are not in any deck:")
        for o in sorted(orphans):
            print(f"         {o}")

    if missing:
        print(f"\n[FAIL] {len(missing)} figure(s) named in FIGURES.md are missing "
              f"on disk:")
        for m in missing:
            print(f"         {m}")
        return 1

    print(f"\n[ok] every figure named in FIGURES.md was found and copied")
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
