"""
ccinc_v3_stats.py
=================
Classifier selection for the CCinc v3 merged tank+world campaign, and the
cluster-level truth-tag vs reco-tag discrepancy.

Answers the question the campaign was built for and the two existing reports stop
short of: **which classifier, in which streamline, with which clustering method?**
The reports quote AUC only; efficiency/purity sits in one unreferenced CSV at a
single working point, and only the NN has a loss number. This puts all four models
(RF / GBT / XGBoost / NN) on one scale — ROC, a working-point scan, and two proper
losses — and then measures how much the two streamlines actually disagree.

READ-ONLY over existing artifacts. No retraining, no reprocessing. Everything comes
from the four merged score tables
    <BASE>/cc_neutrino_v3_{truthtag,recotag}/parquet/
        *__mva_scores__keepprompt__merged__{optics,cf}.parquet
plus the training logs. Constants, palette, loaders and the log parser are imported
from make_plots_ccinc_v3_merged.py, which owns them.

THE LABEL GUARD (the reason this file starts where it does)
-----------------------------------------------------------
Three of the four merged score tables carry `mva_label`. The truth-tag/OPTICS one
does not — it was written at 11:05 on 2026-08-05, before the fix that added the
column at 11:26. Its label has to be re-derived, and re-deriving it from
`dominant_class` ALONE is wrong: an out-of-tank neutron capture is neutron-dominated
and still background, and that mistake reads AUC 0.54 instead of 0.66.

So `load_scores()` re-derives with the documented vertex rule and then **asserts that
every reproduced AUC matches the published value**. If the label is wrong, this file
refuses to run rather than quietly producing a table of plausible nonsense. Every
number downstream — here, in the differential background study, and in the AmBe
closure — rests on that assertion.

Run (no silent default — pick a section):
    source /exp/annie/app/users/dajana/myboy/bin/activate
    python ccinc_v3_stats.py --do guard        # label guard only
    python ccinc_v3_stats.py --do models       # + working points, loss, figures
    python ccinc_v3_stats.py --do truthreco    # + cluster-level streamline overlap
    python ccinc_v3_stats.py --do dirtn        # + dirt-neutron decomposition
    python ccinc_v3_stats.py --do spurious     # + is a background cluster one particle?
    python ccinc_v3_stats.py --do all
Output:
    ccinc_v3_model_comparison.csv
    ccinc_v3_truthreco_discrepancy.csv
    ccinc_v3_dirtn_*.csv
    ccinc_v3_spurious_{purity,byspecies,cooccurrence}.csv
    slide_plots_ccinc_v3_merged/V3STATS__*.{pdf,png}
    slide_plots_ccinc_v3_merged/V3SPUR__*.{pdf,png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, log_loss,
                             brier_score_loss)

import make_plots_ccinc_v3_merged as V3
from make_plots_ccinc_v3_merged import (
    BASE, TANK, MSFX, CONFIGS, CFG_LBL, NEUTRON_CLASSES, OUTD,
    BLUE, GREY, ORANGE, GREEN, LINE_COLORS, LBLUE, LGREY,
    bare, table_axes)

HERE = Path(__file__).parent
PREFIX = "V3STATS__"

# score column -> the model name the training logs and the report use
SCORE_COLS = {"rf_score": "Random Forest", "gbt_score": "GBT",
              "xgb_score": "XGBoost", "nn_score": "Neural Network"}
SHORT = {"Random Forest": "RF", "GBT": "GBT", "XGBoost": "XGB",
         "Neural Network": "NN"}

# Published merged test AUCs — REPORT_ccinc_v3_world_merged.md §5. These are the
# guard's reference values, not an input to any calculation.
REPORTED_AUC = {
    ("truthtag", "optics"):        {"Random Forest": 0.656, "GBT": 0.661,
                                    "XGBoost": 0.662, "Neural Network": 0.649},
    ("truthtag", "clusterfinder"): {"Random Forest": 0.665, "GBT": 0.672,
                                    "XGBoost": 0.669, "Neural Network": 0.667},
    ("recotag", "optics"):         {"Random Forest": 0.684, "GBT": 0.689,
                                    "XGBoost": 0.688, "Neural Network": 0.664},
    ("recotag", "clusterfinder"):  {"Random Forest": 0.688, "GBT": 0.694,
                                    "XGBoost": 0.690, "Neural Network": 0.687},
}
AUC_TOL = 1e-3          # the report quotes 3 decimals, so this is exact agreement

# Pre-capping class counts per merged training, from the training logs' own
# "[mva] N clusters | signal=.. background=.." line. The background column is the
# "background per class" row of REPORT_ccinc_v3_world_merged.md §5.
#
# These exist to attribute a log block to a configuration. The truth-tag/CF re-run
# was launched by hand rather than through run_world_merged.sh, so its log carries
# neither the launcher's "MERGED TRAIN" header nor the pipeline rename line, and the
# class counts are the only thing in it that identifies the configuration. Without
# this map that training's cross-validation spread comes back NaN.
TRAIN_COUNTS = {
    ("truthtag", "optics"):        (50375, 36008),
    ("truthtag", "clusterfinder"): (51609, 39133),
    ("recotag", "optics"):         (25494, 30633),
    ("recotag", "clusterfinder"):  (26045, 33798),
}

# Signal efficiencies the report quotes purity at. 0.80 is the working point the
# pre-existing ccinc_v3_effpurity_by_config.csv uses and is cross-checked against it.
TARGET_EFF = [0.50, 0.80, 0.90]
# Floor for the max-purity / max-significance search. Without it the optimum runs
# away to the single highest-scoring cluster, where purity is 1.0 on one event.
MIN_EFF_FOR_OPTIMUM = 0.05


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — the loader and its guard
# ════════════════════════════════════════════════════════════════════════════
def score_table_path(stream: str, method: str) -> Path:
    run = TANK[stream]
    return (BASE / run / "parquet" /
            f"{run}__mva_scores__keepprompt__merged__{MSFX[method]}.parquet")


def load_scores(stream: str, method: str, verbose: bool = True) -> pd.DataFrame:
    """
    The merged score table with a trustworthy label in column `y`.

    `mva_label` when the training wrote it; otherwise the documented vertex rule
    (REPORT_ccinc_v3_world_merged.md §0): out-of-tank interaction -> background
    regardless of composition, in-tank -> neutron-dominated. The table is already
    the training frame, so the in-tank/outside-FV rows were dropped upstream and
    the FV term of the rule has nothing left to do here.
    """
    p = score_table_path(stream, method)
    if not p.exists():
        sys.exit(f"[stats] missing merged score table: {p}")
    d = pd.read_parquet(p)

    if "mva_label" in d.columns:
        d["y"] = d["mva_label"].astype(int)
        origin = "mva_label (written by the training)"
    else:
        in_tank = d["origin_in_tank"].to_numpy(int)
        d["y"] = np.where(in_tank == 0, 0,
                          d["dominant_class"].isin(NEUTRON_CLASSES).astype(int))
        origin = "re-derived (vertex rule)"
    if verbose:
        te = int(d["in_test"].astype(bool).sum())
        print(f"  [load] {stream}/{method}: {len(d):,} clusters "
              f"({te:,} test), label from {origin}")
    return d


def guard(verbose: bool = True) -> dict:
    """
    Recompute all 16 merged test AUCs from the stored scores and assert they
    reproduce REPORT_ccinc_v3_world_merged.md §5. Exits non-zero on any mismatch.

    This is the primary test of the whole analysis: it is what proves the
    re-derived truth-tag/OPTICS label is the same label the model was trained on.
    """
    print("[guard] reproducing the 16 published merged AUCs from the score tables")
    frames, rows, bad = {}, [], []
    for stream, method in CONFIGS:
        d = load_scores(stream, method, verbose=verbose)
        frames[(stream, method)] = d
        te = d[d["in_test"].astype(bool)]
        y = te["y"].to_numpy(int)
        for col, model in SCORE_COLS.items():
            if col not in te.columns:
                bad.append(f"{stream}/{method}: score column {col} absent")
                continue
            got = roc_auc_score(y, te[col].to_numpy(float))
            want = REPORTED_AUC[(stream, method)][model]
            ok = abs(round(got, 3) - want) <= AUC_TOL
            rows.append(dict(streamline=stream, method=method, model=model,
                             auc=got, reported=want, ok=ok))
            if not ok:
                bad.append(f"{stream}/{method} {model}: got {got:.4f}, "
                           f"report says {want:.3f}")
        if verbose:
            print(f"         nS={int((y == 1).sum()):,} nB={int((y == 0).sum()):,}")

    chk = pd.DataFrame(rows)
    if bad:
        print("\n[guard] FAILED — refusing to continue:")
        for b in bad:
            print("   ", b)
        print("\n  A mismatch on truth-tag/OPTICS means the label re-derivation is "
              "wrong (that table has no mva_label).\n  A mismatch elsewhere means the "
              "score table changed under this script.")
        sys.exit(1)
    print(f"[guard] PASS — all {len(chk)} AUCs reproduce the report to "
          f"{AUC_TOL:g}\n")
    return {"frames": frames, "check": chk}


def collect_merged_trainings() -> dict:
    """
    (streamline, method) -> the training log block, for the merged trainings only.

    Reuses V3.parse_log_blocks but does its own attribution: blocks are keyed by the
    pre-capping class counts in TRAIN_COUNTS, which is the only identifier present in
    a hand-launched re-run's log. Newest log wins; logs_merged_tankworld_run1.log is
    excluded because it predates the 2026-08-05 09:53 FV-drop fix and its labels are
    wrong.
    """
    by_counts = {v: k for k, v in TRAIN_COUNTS.items()}
    logs = sorted((f for f in HERE.glob("logs_merged_*.log")
                   if f.name != "logs_merged_tankworld_run1.log"),
                  key=lambda f: f.stat().st_mtime)
    out = {}
    for path in logs:
        for b in V3.parse_log_blocks(path):
            cfg = by_counts.get(b["counts"])
            if cfg is None:
                continue
            b["log"] = path.name
            out[cfg] = b
    missing = [c for c in CONFIGS if c not in out]
    if missing:
        print(f"[logs] WARNING — no training block found for {missing}; "
              f"cross-validation spread will read NaN for those.")
    else:
        print("[logs] cross-validation spread found for all four merged trainings "
              + ", ".join(f"{s}/{m}={out[(s, m)]['log']}" for s, m in CONFIGS))
    return out


def n_features_used(trainings: dict) -> dict:
    """
    How many features each merged training actually used. Not cosmetic: the two
    reco-tag trainings fell back to 21 of 31 because the world rows' fitted-vertex
    block is too sparse to pass the non-NaN threshold, so the best-AUC models are
    also the ones with the fewest inputs.
    """
    import re as _re
    out = {}
    for cfg, b in trainings.items():
        p = HERE / b.get("log", "")
        n = None
        if p.exists():
            sig, bkg = TRAIN_COUNTS[cfg]
            m = _re.search(rf"signal={sig}\s+background={bkg}\s*\|\s*(\d+) features",
                           p.read_text(errors="replace"))
            if m:
                n = int(m.group(1))
        out[cfg] = n
    return out


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — working points, loss, and a verdict
# ════════════════════════════════════════════════════════════════════════════
def working_points(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """
    Purity / rejection / significance along the ROC, at fixed signal efficiencies
    and at the two optima.

    purity       = S / (S + B) among selected clusters
    rejection    = 1 - fpr  (fraction of background removed)
    significance = S / sqrt(S + B)

    S and B are absolute test-set counts, so purity is the purity of THIS test set
    (nS ~ nB by construction: the training is 1:1 capped). It is not a physical
    purity — the true signal:background ratio in data is not 1:1 — and must be read
    as a relative figure of merit between models on a common sample.
    """
    nS = int((y == 1).sum())
    nB = int((y == 0).sum())
    fpr, tpr, thr = roc_curve(y, s)

    S = tpr * nS
    B = fpr * nB
    with np.errstate(invalid="ignore", divide="ignore"):
        pur = np.where(S + B > 0, S / (S + B), np.nan)
        sig = np.where(S + B > 0, S / np.sqrt(S + B), 0.0)

    rows = []

    def row(kind, i):
        rows.append(dict(point=kind, threshold=float(thr[i]),
                         eff_signal=float(tpr[i]), rej_bkg=float(1.0 - fpr[i]),
                         purity=float(pur[i]), significance=float(sig[i]),
                         n_sig_sel=int(round(S[i])), n_bkg_sel=int(round(B[i])),
                         nS=nS, nB=nB))

    for target in TARGET_EFF:
        cand = np.nonzero(tpr >= target)[0]
        if len(cand):
            row(f"eff{int(target * 100)}", int(cand[0]))

    ok = tpr >= MIN_EFF_FOR_OPTIMUM
    if ok.any():
        idx = np.nonzero(ok)[0]
        row("max_purity", int(idx[np.nanargmax(pur[idx])]))
        row("max_significance", int(idx[np.argmax(sig[idx])]))
    return pd.DataFrame(rows)


def model_comparison(frames: dict, trainings: dict) -> pd.DataFrame:
    """One row per (streamline, method, model, working point), plus AUC and loss."""
    out = []
    for (stream, method), d in frames.items():
        te = d[d["in_test"].astype(bool)]
        y = te["y"].to_numpy(int)
        blk = trainings.get((stream, method), {})
        for col, model in SCORE_COLS.items():
            if col not in te.columns:
                continue
            s = te[col].to_numpy(float)
            auc = roc_auc_score(y, s)
            # Loss on a scale every model shares. The trees have no loss number in
            # the reports at all, and the NN's is a training curve, not a test value.
            ll = log_loss(y, np.clip(s, 1e-15, 1 - 1e-15))
            br = brier_score_loss(y, s)
            cv = (blk.get("cv") or {}).get(model)
            wp = working_points(y, s)
            for _, r in wp.iterrows():
                out.append(dict(
                    streamline=stream, method=method, model=model,
                    model_short=SHORT[model], config=CFG_LBL[(stream, method)],
                    auc=auc, cv_mean=(cv[0] if cv else np.nan),
                    cv_sigma=(cv[1] if cv else np.nan),
                    log_loss=ll, brier=br,
                    nn_best_epoch=(blk.get("best_epoch")
                                   if model == "Neural Network" else np.nan),
                    **r.to_dict()))
    return pd.DataFrame(out)


def verdict(cmp: pd.DataFrame) -> pd.DataFrame:
    """
    Best model per configuration, and whether the win is resolvable.

    A lead smaller than the fold-to-fold CV scatter is not a result. The merged
    trainings sit at CV sigma 0.003-0.005, so a 0.001 AUC lead means nothing —
    exactly the trap the tank-only truth-vs-reco differences fell into.
    """
    rows = []
    for cfg in CONFIGS:
        sub = (cmp[(cmp.streamline == cfg[0]) & (cmp.method == cfg[1])]
               .drop_duplicates(subset=["model"])
               .sort_values("auc", ascending=False))
        if sub.empty:
            continue
        best, second = sub.iloc[0], sub.iloc[1]
        margin = best.auc - second.auc
        sigma = best.cv_sigma if np.isfinite(best.cv_sigma) else np.nan
        by_ll = sub.sort_values("log_loss").iloc[0]
        rows.append(dict(
            config=CFG_LBL[cfg], streamline=cfg[0], method=cfg[1],
            best_auc_model=best.model, best_auc=best.auc,
            runner_up=second.model, margin=margin, cv_sigma=sigma,
            resolvable=(bool(margin > sigma) if np.isfinite(sigma) else None),
            best_logloss_model=by_ll.model, best_logloss=by_ll.log_loss,
            agree=(best.model == by_ll.model)))
    return pd.DataFrame(rows)


def crosscheck_effpurity(cmp: pd.DataFrame) -> None:
    """
    The 80%-efficiency purities must reproduce ccinc_v3_effpurity_by_config.csv,
    which was built independently. Any disagreement is a bug in the scan above.
    """
    ref_path = HERE / "ccinc_v3_effpurity_by_config.csv"
    if not ref_path.exists():
        print("[xcheck] ccinc_v3_effpurity_by_config.csv absent — skipped")
        return
    ref = pd.read_csv(ref_path)
    ref = ref[ref["campaign"] == "tank+world"].copy()
    sl = {"Truth CC-inc": "truthtag", "Reco CC-inc": "recotag"}
    me = {"OPTICS": "optics", "ClusterFinder": "clusterfinder"}
    mo = {"Random Forest": "Random Forest", "GBT": "GBT", "XGBoost": "XGBoost",
          "Neural Net": "Neural Network"}
    ref["streamline"] = ref["streamline"].map(sl)
    ref["method"] = ref["method"].map(me)
    ref["model"] = ref["model"].map(mo)

    mine = cmp[cmp["point"] == "eff80"][
        ["streamline", "method", "model", "purity", "auc"]]
    j = ref.merge(mine, on=["streamline", "method", "model"],
                  suffixes=("_ref", "_new"))
    j["d_pur"] = (j["pur"] - j["purity"]).abs()
    j["d_auc"] = (j["auc_ref"] - j["auc_new"]).abs()
    worst_p, worst_a = j["d_pur"].max(), j["d_auc"].max()
    print(f"[xcheck] vs ccinc_v3_effpurity_by_config.csv on {len(j)} rows: "
          f"max |dpurity| = {worst_p:.2e}, max |dAUC| = {worst_a:.2e}")
    if worst_p > 5e-3 or worst_a > 5e-3:
        print("[xcheck] WARNING — does not reproduce the existing CSV. "
              "Investigate before quoting either.")
        print(j[j["d_pur"] > 5e-3][["streamline", "method", "model",
                                    "pur", "purity"]].to_string(index=False))
    else:
        print("[xcheck] reproduces the existing CSV (that CSV used a fixed 80% "
              "signal-efficiency working point)")


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — truth-tag vs reco-tag, at cluster and score level
# ════════════════════════════════════════════════════════════════════════════
def event_overlap_check() -> dict:
    """
    Re-confirm the event-level overlap from REPORT_ccinc_v3_streamlines.md §1.1
    (truthtag 81,790 / recotag 41,418 / both 40,442). Cached by the plots module.
    Guards against the cluster-level join below changing the population silently.
    """
    o = V3.overlap()
    print(f"[truthreco] event level: truthtag {int(o['truthtag']):,}  "
          f"recotag {int(o['recotag']):,}  both {int(o['both']):,}")
    if int(o["both"]) != 40442:
        print(f"[truthreco] WARNING — intersection is {int(o['both']):,}, "
              f"the report says 40,442. Population changed; do not trust the "
              f"cluster-level numbers below until this is understood.")
    return o


def _keyed(d: pd.DataFrame) -> pd.DataFrame:
    """
    Add the cluster join key and a streamline-independent source tag.

    The key is (sample, eventID, cluster_id) with the streamline suffix stripped from
    the run name, so a tank cluster seen by both streamlines gets the same key. Note
    this is a join BETWEEN two cluster tables built by the same processor over the
    same input files — not the cluster-to-event-table join on eventID, which is the
    documented 74%-agreement trap and is not performed anywhere in this file.
    """
    d = d.copy()
    d["_base"] = (d["_source_run"].astype(str)
                  .str.replace("_truthtag", "", regex=False)
                  .str.replace("_recotag", "", regex=False))
    d["_key"] = (d["_base"] + "#" + d["eventID"].astype(str)
                 + "#" + d["cluster_id"].astype(str))
    d["_is_world"] = d["_base"].str.contains("world")
    return d


def overlap_breakdown(frames: dict, method: str) -> pd.DataFrame:
    """
    The shared / truthtag-only / recotag-only populations split by sample (tank vs
    world) and by class. This is what makes the overlap interpretable: it separates
    "the two streams select different events" from "the two streams label differently".
    """
    a = _keyed(frames[("truthtag", method)])
    b = _keyed(frames[("recotag", method)])
    ka, kb = set(a["_key"]), set(b["_key"])
    pops = {"shared": a[a["_key"].isin(kb)],
            "truthtag_only": a[~a["_key"].isin(kb)],
            "recotag_only": b[~b["_key"].isin(ka)]}
    rows = []
    for name, sub in pops.items():
        w = sub["_is_world"].to_numpy(bool)
        rows.append(dict(method=method, population=name, n=len(sub),
                         n_tank=int((~w).sum()), n_world=int(w.sum()),
                         n_signal=int((sub["y"] == 1).sum()),
                         n_background=int((sub["y"] == 0).sum())))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def cluster_overlap(frames: dict, method: str) -> dict:
    """
    Cluster-level overlap of the two streamlines' merged training sets for one
    clustering method, and how much the two disagree on the clusters they share.

    Key: (_source_run, eventID, cluster_id). eventID is the authoritative key
    WITHIN the cluster tables — the documented 74%-agreement trap is joining
    cluster_features to the cc_*_events table on eventID, which is not what this
    does. The two frames here are both cluster tables produced by the same
    processor, so eventID is consistent between them; the check below verifies it.
    """
    a = _keyed(frames[("truthtag", method)])
    b = _keyed(frames[("recotag", method)])
    ka, kb = set(a["_key"]), set(b["_key"])
    shared = ka & kb

    # Sanity on the key: a shared key must describe the same cluster. n_hits and
    # pe_total are produced by the same clustering pass, so they must agree exactly.
    m = (a[a["_key"].isin(shared)].set_index("_key")[["n_hits", "pe_total", "y"]]
         .join(b[b["_key"].isin(shared)].set_index("_key")[["n_hits", "pe_total", "y"]],
               lsuffix="_tt", rsuffix="_rt", how="inner"))
    m = m[~m.index.duplicated()]
    key_ok = float((m["n_hits_tt"] == m["n_hits_rt"]).mean()) if len(m) else np.nan

    out = dict(method=method, n_truthtag=len(ka), n_recotag=len(kb),
               n_shared=len(shared), n_truthtag_only=len(ka - kb),
               n_recotag_only=len(kb - ka),
               frac_recotag_shared=(len(shared) / max(len(kb), 1)),
               key_nhits_agreement=key_ok,
               label_agreement=(float((m["y_tt"] == m["y_rt"]).mean())
                                if len(m) else np.nan))
    print(f"[truthreco] cluster level, {method}: truthtag {len(ka):,}  "
          f"recotag {len(kb):,}  shared {len(shared):,} "
          f"({out['frac_recotag_shared'] * 100:.1f}% of recotag)")
    print(f"            key check (n_hits agree on shared): "
          f"{key_ok * 100:.2f}%   label agreement: "
          f"{out['label_agreement'] * 100:.2f}%")
    if np.isfinite(key_ok) and key_ok < 0.999:
        print("            WARNING — the join key does not describe the same "
              "cluster on both sides. Do not read the score comparison below.")

    # Score agreement on shared clusters, per model.
    for col, model in SCORE_COLS.items():
        if col not in a.columns or col not in b.columns:
            continue
        j = (a[a["_key"].isin(shared)].set_index("_key")[[col]]
             .join(b[b["_key"].isin(shared)].set_index("_key")[[col]],
                   lsuffix="_tt", rsuffix="_rt", how="inner"))
        j = j[~j.index.duplicated()].dropna()
        if len(j) > 2:
            out[f"score_corr_{SHORT[model]}"] = float(
                np.corrcoef(j[f"{col}_tt"], j[f"{col}_rt"])[0, 1])
    return out


def disagreement_composition(frames: dict, method: str) -> pd.DataFrame:
    """
    What physics the two streamlines disagree about: the background species mix of
    the clusters each stream keeps and the other does not, next to the shared pool.

    Species axis is `origin_dominant_species` — the first non-EM ancestor
    (world report §0.0). Not bg_class, and never ImmediateAncestorClass.
    """
    a = _keyed(frames[("truthtag", method)])
    b = _keyed(frames[("recotag", method)])
    ka, kb = set(a["_key"]), set(b["_key"])
    pops = {"shared": a[a["_key"].isin(ka & kb)],
            "truthtag_only": a[~a["_key"].isin(kb)],
            "recotag_only": b[~b["_key"].isin(ka)]}
    rows = []
    for name, d in pops.items():
        bkg = d[d["y"] == 0]
        vc = bkg["origin_dominant_species"].value_counts(normalize=True)
        for sp, f in vc.items():
            rows.append(dict(method=method, population=name, n_bkg=len(bkg),
                             species=sp, frac=float(f)))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# figures
# ════════════════════════════════════════════════════════════════════════════
def save(fig, name):
    OUTD.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTD / f"{PREFIX}{name}.{ext}",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  [fig] {PREFIX}{name}.pdf/.png")


def fig_roc_by_config(frames):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        te = frames[cfg]
        te = te[te["in_test"].astype(bool)]
        y = te["y"].to_numpy(int)
        for (col, model), c in zip(SCORE_COLS.items(), LINE_COLORS):
            if col not in te.columns:
                continue
            fpr, tpr, _ = roc_curve(y, te[col].to_numpy(float))
            ax.plot(fpr, tpr, lw=1.8, color=c,
                    label=f"{model}   AUC = "
                          f"{roc_auc_score(y, te[col]):.3f}")
        ax.set_xlabel("False positive rate", fontsize=10)
        ax.set_ylabel("True positive rate", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("ROC by Configuration — Merged Tank+World, Held-Out Test Set",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "roc_by_config")


def fig_effpurity_scan(frames):
    """Purity vs signal efficiency — the curve the fixed 80% point is one slice of."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        te = frames[cfg]
        te = te[te["in_test"].astype(bool)]
        y = te["y"].to_numpy(int)
        nS, nB = int((y == 1).sum()), int((y == 0).sum())
        for (col, model), c in zip(SCORE_COLS.items(), LINE_COLORS):
            if col not in te.columns:
                continue
            fpr, tpr, _ = roc_curve(y, te[col].to_numpy(float))
            S, B = tpr * nS, fpr * nB
            with np.errstate(invalid="ignore", divide="ignore"):
                pur = np.where(S + B > 0, S / (S + B), np.nan)
            keep = tpr >= 0.02
            ax.plot(tpr[keep], pur[keep], lw=1.8, color=c, label=model)
        ax.set_xlabel("Signal efficiency", fontsize=10)
        ax.set_ylabel("Purity (1:1 test set)", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("Purity vs Signal Efficiency — Merged Models, 1:1 Balanced "
                 "Test Set", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "effpurity_scan")


def fig_loss_comparison(cmp):
    """Log-loss and Brier side by side — the two losses all four models share."""
    u = cmp.drop_duplicates(subset=["streamline", "method", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(CONFIGS))
    w = 0.2
    for metric, ax, name in ((["log_loss"][0], axes[0], "Log-loss"),
                             (["brier"][0], axes[1], "Brier score")):
        for k, (model, c) in enumerate(zip(SCORE_COLS.values(), LINE_COLORS)):
            vals = [u[(u.streamline == s) & (u.method == m) &
                      (u.model == model)][metric]
                    for s, m in CONFIGS]
            vals = [float(v.iloc[0]) if len(v) else np.nan for v in vals]
            ax.bar(x + (k - 1.5) * w, vals, w, color=c, alpha=0.85,
                   edgecolor="none", label=SHORT[model])
        ax.set_xticks(x)
        ax.set_xticklabels([CFG_LBL[c].replace(" / ", "\n") for c in CONFIGS],
                           fontsize=8.5)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(f"{name}  (lower is better)", fontsize=11)
        ax.tick_params(labelsize=9)
        bare(ax)
    axes[0].legend(fontsize=9, frameon=False, ncol=4)
    fig.suptitle("Test-Set Loss by Model — All Four Classifiers on One Scale",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "loss_comparison")


def fig_verdict_table(cmp, vd):
    """A table, matching the cut-flow convention rather than a bar chart."""
    u = cmp.drop_duplicates(subset=["streamline", "method", "model"])
    e80 = cmp[cmp["point"] == "eff80"]
    cols = ["Configuration", "Model", "AUC", "CV ± σ", "Purity @80% eff",
            "Log-loss", "Brier"]
    cell = []
    for s, m in CONFIGS:
        sub = u[(u.streamline == s) & (u.method == m)].sort_values(
            "auc", ascending=False)
        for i, (_, r) in enumerate(sub.iterrows()):
            p = e80[(e80.streamline == s) & (e80.method == m) &
                    (e80.model == r.model)]
            cv = (f"{r.cv_mean:.4f} ± {r.cv_sigma:.4f}"
                  if np.isfinite(r.cv_mean) else "—")
            cell.append([CFG_LBL[(s, m)] if i == 0 else "",
                         SHORT[r.model] + ("  (best)" if i == 0 else ""),
                         f"{r.auc:.4f}", cv,
                         f"{float(p['purity'].iloc[0]):.4f}" if len(p) else "—",
                         f"{r.log_loss:.4f}", f"{r.brier:.4f}"])
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(cell) + 1.6))
    table_axes(ax, cell, cols, fs=9.0, emphasise_last=False)
    res = vd.dropna(subset=["resolvable"])
    n_res = int(res["resolvable"].sum()) if len(res) else 0
    fig.suptitle("Merged Model Comparison — best AUC per configuration marked\n"
                 f"{n_res} of {len(res)} leads exceed that training's "
                 f"cross-validation σ", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "model_verdict_table")


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — the dirt-neutron background: composition and separability
# ════════════════════════════════════════════════════════════════════════════
# Species columns are the per-cluster origin counts: background hits whose
# lineage walk reached a first non-EM ancestor. Pure-EM and untraced chains are
# in neither numerator nor denominator, which is why the budget below carries an
# explicit "no complete non-EM chain" row rather than folding that light into a
# species. frac_bg_* is never used — it can exceed 1 (n_bg_photon counts capture
# gammas). See REPORT_ccinc_v3_world_merged.md §4.2.
ORIGIN_SPECIES = ["muminus", "piplus", "pizero", "proton", "piminus", "muplus",
                  "kplus", "kminus", "other"]
# Features the budget/census reports. The first block is what actually separates
# signal from dirt neutrons; d_wall and vtx_y are carried to keep the geometry
# question answerable from this table alone.
CENSUS_FEATURES = ["n_hits", "n_hits_early", "n_fit_hits", "pe_total",
                   "charge_bal_legacy", "beta1", "sigma_t_mad",
                   "t_window_80pct", "d_wall", "vtx_y"]
FRAC_COLS = ["frac_neutron", "frac_nonneutron", "frac_darknoise", "frac_untraced"]


def dirtn_groups(d: pd.DataFrame) -> pd.Series:
    """
    Split the merged frame into the five populations the label rule creates.

    The background splits into three physically different things that the class
    label alone hides: tank background (muon and pion light), world background
    that is neutron-dominated (the dirt neutrons), and world background that is
    not. `SIGNAL world in-tank` is the control — same sample and processing as
    the dirt neutrons, in-tank interaction — so anything that separates signal
    from dirt neutrons but does NOT separate it from this group is a property of
    the world sample, not of where the neutron came from.
    """
    is_tank = d["_source_run"] == d["_source_run"].mode().iat[0]
    neu = d["dominant_class"].isin(NEUTRON_CLASSES)
    sig = d["y"].astype(bool)
    g = pd.Series(index=d.index, dtype=object)
    g[sig & is_tank] = "SIGNAL tank"
    g[sig & ~is_tank] = "SIGNAL world in-tank"
    g[~sig & is_tank] = "BKG tank"
    g[~sig & ~is_tank & neu] = "BKG world neutron-dominated (dirt n)"
    g[~sig & ~is_tank & ~neu] = "BKG world non-neutron"
    return g


def dirtn_budget(d: pd.DataFrame, stream: str, method: str) -> pd.DataFrame:
    """
    The background light budget, normalised to 100% of all background hits and
    split by tank vs world contribution.

    This is the table to quote for "what fraction of the background is muons /
    dirt neutrons". Every row is a percentage of the SAME denominator (total
    background hits), so the tank and world columns add to the total column and
    the whole table adds to 100.
    """
    g = dirtn_groups(d)
    b = d[~d["y"].astype(bool)].copy()
    b["side"] = np.where(g[b.index].str.startswith("BKG tank"), "tank", "world")
    tot = float(b["n_hits"].sum())

    def split(series_name, mask=None):
        sub = b if mask is None else b[mask]
        by = sub.groupby("side")[series_name].sum()
        return (float(by.get("tank", 0.0)), float(by.get("world", 0.0)))

    rows = []

    def add(label, tank_hits, world_hits):
        rows.append(dict(streamline=stream, method=method, component=label,
                         hits_tank=tank_hits, hits_world=world_hits,
                         hits_total=tank_hits + world_hits,
                         pct_tank=100 * tank_hits / tot,
                         pct_world=100 * world_hits / tot,
                         pct_total=100 * (tank_hits + world_hits) / tot))

    add("neutron-capture light", *split("n_neutron"))
    traced_t = traced_w = 0.0
    for sp in ORIGIN_SPECIES:
        col = f"n_origin_{sp}"
        if col not in b.columns:
            continue
        t, w = split(col)
        traced_t += t
        traced_w += w
        add(f"non-neutron: {sp}", t, w)
    nn_t, nn_w = split("n_nonneutron")
    add("non-neutron: no complete non-EM chain (pure EM / untraced)",
        nn_t - traced_t, nn_w - traced_w)
    h_t, h_w = split("n_hits")
    add("dark noise / other", h_t - nn_t - split("n_neutron")[0],
        h_w - nn_w - split("n_neutron")[1])
    out = pd.DataFrame(rows)
    add_tot = out["pct_total"].sum()
    if abs(add_tot - 100.0) > 0.5:                 # rounding aside, this must close
        sys.exit(f"[dirtn] budget does not close for {stream}/{method}: "
                 f"{add_tot:.2f}% — the component definitions overlap or leak")
    return out


def dirtn_census(d: pd.DataFrame, stream: str, method: str) -> pd.DataFrame:
    """Per-population cluster counts, hit composition, features and median score."""
    g = dirtn_groups(d)
    agg = {c: "mean" for c in CENSUS_FEATURES + FRAC_COLS}
    out = d.groupby(g).agg(agg)
    out.insert(0, "clusters", d.groupby(g).size())
    for c in FRAC_COLS:
        out[c + "_median"] = d.groupby(g)[c].median()
    out["gbt_score_median"] = d.groupby(g)["gbt_score"].median()
    out.insert(0, "method", method)
    out.insert(0, "streamline", stream)
    return out.reset_index(names="population")


def dirtn_separability(d: pd.DataFrame, stream: str, method: str) -> pd.DataFrame:
    """
    How well the trained model separates signal from EACH background population,
    and which features do it for the dirt neutrons specifically.

    The per-population AUCs share a common signal set but sit on unbalanced
    subsets, so they rank difficulty against each other — they are not standalone
    performance figures for a selection.
    """
    g = dirtn_groups(d)
    te = d[d["in_test"].astype(bool)]
    gte = g[te.index]
    sig = te[gte.str.startswith("SIGNAL")]
    rows = []
    for pop in ["BKG tank", "BKG world neutron-dominated (dirt n)",
                "BKG world non-neutron"]:
        bkg = te[gte == pop]
        if len(bkg) < 50:                          # too few to quote an AUC
            continue
        y = np.r_[np.ones(len(sig)), np.zeros(len(bkg))]
        for col, model in SCORE_COLS.items():
            if col not in te.columns:
                continue
            s = np.r_[sig[col].to_numpy(float), bkg[col].to_numpy(float)]
            rows.append(dict(streamline=stream, method=method, population=pop,
                             model=model, n_signal=len(sig), n_background=len(bkg),
                             auc=roc_auc_score(y, s),
                             bkg_median_score=float(bkg[col].median()),
                             sig_median_score=float(sig[col].median())))
    auc = pd.DataFrame(rows)

    # Feature-by-feature separation, signal vs dirt neutrons only.
    S = d[d["y"].astype(bool)]
    D = d[g == "BKG world neutron-dominated (dirt n)"]
    frows = []
    for f in CENSUS_FEATURES:
        a, b = S[f].astype(float), D[f].astype(float)
        sd = float(np.sqrt((a.var() + b.var()) / 2))
        frows.append(dict(streamline=stream, method=method, feature=f,
                          signal_median=float(a.median()),
                          dirtn_median=float(b.median()),
                          signal_mean=float(a.mean()), dirtn_mean=float(b.mean()),
                          sep_sigma=abs(a.mean() - b.mean()) / sd if sd > 0 else 0.0))
    return auc, pd.DataFrame(frows).sort_values("sep_sigma", ascending=False)


def dirtn_tank_neutron_light(d: pd.DataFrame, stream: str,
                             method: str) -> pd.DataFrame:
    """
    Why the TANK side of the background carries neutron light at all, given that
    it contains no neutron-dominated cluster by construction.

    `dominant_class` is the most frequent single truth_class in the cluster, so a
    cluster goes to background as soon as neutron hits stop being the plurality —
    it keeps whatever neutron light it had. Neutron light is also split across
    four class codes (1,2,3,4), so a cluster can be majority-neutron overall and
    still have a non-neutron plurality; the last row of the returned table counts
    exactly those.
    """
    g = dirtn_groups(d)
    b = d[g == "BKG tank"]
    edges = [0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1.01]
    cut = pd.cut(b["frac_neutron"], edges, right=False)
    t = b.groupby(cut, observed=False).agg(clusters=("n_hits", "size"),
                                           hits=("n_hits", "sum"),
                                           neutron_hits=("n_neutron", "sum"))
    t["pct_of_tank_bkg_clusters"] = 100 * t["clusters"] / len(b)
    t["pct_of_tank_bkg_neutron_light"] = 100 * t["neutron_hits"] / b["n_neutron"].sum()
    t = t.reset_index(names="frac_neutron_bin")
    t["frac_neutron_bin"] = t["frac_neutron_bin"].astype(str)
    maj = b[b["frac_neutron"] > b["frac_nonneutron"]]
    t = pd.concat([t, pd.DataFrame([dict(
        frac_neutron_bin="TOTAL", clusters=len(b), hits=int(b["n_hits"].sum()),
        neutron_hits=int(b["n_neutron"].sum()), pct_of_tank_bkg_clusters=100.0,
        pct_of_tank_bkg_neutron_light=100.0)]), pd.DataFrame([dict(
        frac_neutron_bin="of which majority-neutron (mislabelled by plurality)",
        clusters=len(maj), hits=int(maj["n_hits"].sum()),
        neutron_hits=int(maj["n_neutron"].sum()),
        pct_of_tank_bkg_clusters=100 * len(maj) / len(b),
        pct_of_tank_bkg_neutron_light=100 * maj["n_neutron"].sum()
        / b["n_neutron"].sum())])], ignore_index=True)
    t.insert(0, "method", method)
    t.insert(0, "streamline", stream)
    return t


# ════════════════════════════════════════════════════════════════════════════
# Step 6 — background spuriousness: is a background cluster ONE particle?
# ════════════════════════════════════════════════════════════════════════════
# The reports quote the background composition INTEGRATED over all clusters
# (world report §4.2: mu- 53.3%, pi+ 25.5%, ...). That is a statement about the
# light, not about the clusters, and the two are only the same thing if a typical
# background cluster is made by a single particle. This section measures that
# directly, because the answer decides how the per-species feature plots may be
# read: if clusters are single-species, a per-species curve is the distribution of
# a physically distinct population; if they are mixtures, it is the distribution of
# clusters that particle merely dominates.
#
# The axis is `n_origin_<species>` — BACKGROUND hits (truth_class == -5) whose
# lineage walk reached a first non-EM ancestor, per cluster. Neutron-capture light
# (classes 1-4) is NOT on this axis by construction, so a cluster can have
# n_origin_traced == 0 and still be full of light. Those clusters are a real and
# large population, not a tracing failure, and get their own class below.
SPUR_MIN_CLUSTERS = 200        # below this a per-species row is noise, not a result
SPUR_SINGLE_CUT = 0.9          # top-species share at or above which we call it "one particle"

# The class label shared by this section and make_presentable_optics_rf_plots.py's
# --bkg-split origin. Defined once, here, so the plots and the tables cannot drift.
NO_ORIGIN_LABEL = "no non-EM origin"
SPUR_SPECIES_ORDER = ["muminus", "piplus", "pizero", "proton", "piminus",
                      "muplus", "kplus", "kminus", "neutron", "other"]
SPUR_PRETTY = {"muminus": r"$\mu^-$", "muplus": r"$\mu^+$",
               "piplus": r"$\pi^+$", "piminus": r"$\pi^-$",
               "pizero": r"$\pi^0$", "proton": "p", "neutron": "n",
               "kplus": r"$K^+$", "kminus": r"$K^-$", "other": "other",
               NO_ORIGIN_LABEL: "no non-EM origin"}
# The integrated shares this section must reproduce — REPORT_ccinc_v3_world_merged.md
# §4.2, as percentages of the traced non-neutron background light. Same role as
# REPORTED_AUC: a reference to check against, never an input.
REPORTED_SPECIES_PCT = {"muminus": 53.3, "piplus": 25.5, "pizero": 7.2,
                        "proton": 5.8, "piminus": 4.6, "muplus": 3.4}
SPECIES_PCT_TOL = 0.5          # the report quotes one decimal


def origin_class(d: pd.DataFrame) -> pd.Series:
    """
    One background-particle class per cluster, for tables and for plotting.

    `origin_dominant_species` where the cluster has traced non-EM background light;
    NO_ORIGIN_LABEL where it has none. The residual class is not a leftover bin to
    be dropped — it is ~65-70% of background clusters and is median ~84%
    neutron-capture light, i.e. it is where the out-of-tank captures live. Dropping
    it would delete the majority of the background from every per-species figure.
    """
    s = d["origin_dominant_species"].astype("object")
    return s.where(d["n_origin_traced"].to_numpy(int) > 0,
                   NO_ORIGIN_LABEL).fillna(NO_ORIGIN_LABEL)


def _origin_matrix(d: pd.DataFrame):
    """(n_clusters x n_species) count matrix and its row sums, in a fixed order."""
    cols = [f"n_origin_{s}" for s in SPUR_SPECIES_ORDER
            if f"n_origin_{s}" in d.columns]
    m = d[cols].to_numpy(float)
    return m, m.sum(axis=1), [c.removeprefix("n_origin_") for c in cols]


# ── the same question asked over ALL the light in the cluster ───────────────
# _origin_matrix() only carries charged-parent light, so 70% of background
# clusters have nothing on it and drop out of every figure built on it. That is
# a correct answer to "how mixed is the charged-particle light" and a misleading
# one to "how mixed is the cluster". _light_matrix() partitions every hit into
# exactly one bucket:
#     neutron capture (truth_class 1-4)  |  charged parent (n_origin_<species>)
#     gamma/e+- with no non-EM ancestor  |  dark noise (truth_class 0)
# There is no untraced bucket: n_lineage_complete == n_hits - n_darknoise holds
# for every cluster in this campaign, i.e. every physics hit has a complete
# chain, so the EM bucket is genuine pure-EM light and not a tracing failure.
# The closure below enforces that; if a future sample breaks it the figures stop
# rather than quietly renormalising.
CAPTURE_LABEL = "neutron capture"
PUREEM_LABEL = r"$\gamma$/$e^\pm$ (no non-EM ancestor)"
DARKNOISE_LABEL = "dark noise"


def _light_matrix(d: pd.DataFrame):
    """(n_clusters x n_sources) hit-count matrix over ALL light, and its row sums."""
    cols = [f"n_origin_{s}" for s in SPUR_SPECIES_ORDER
            if f"n_origin_{s}" in d.columns]
    names = [CAPTURE_LABEL] + [SPUR_PRETTY.get(c.removeprefix("n_origin_"),
                                               c.removeprefix("n_origin_"))
                               for c in cols] + [PUREEM_LABEL, DARKNOISE_LABEL]
    pure_em = (d["n_nonneutron"] - d["n_origin_traced"]).to_numpy(float)
    m = np.column_stack([d["n_neutron"].to_numpy(float),
                         d[cols].to_numpy(float),
                         pure_em,
                         d["n_darknoise"].to_numpy(float)])
    tot = d["n_hits"].to_numpy(float)
    if np.abs(m.sum(axis=1) - tot).max() > 0:
        sys.exit("[spurious] the light buckets do not partition n_hits — "
                 "capture / charged / pure-EM / dark noise overlap or leak")
    incomplete = (d["n_hits"] - d["n_darknoise"] - d["n_lineage_complete"])
    if int((incomplete != 0).sum()):
        sys.exit("[spurious] physics hits without a complete lineage chain: the "
                 f"pure-EM bucket is not pure EM for {int((incomplete != 0).sum())} "
                 "clusters — regenerate the MC or split the bucket")
    return m, tot, names


def light_purity(d: pd.DataFrame, stream: str, method: str) -> pd.DataFrame:
    """
    "Is a cluster one thing or several?" over all its light, signal and background.

    Companion to spurious_purity(), which asks the same on the charged-parent axis
    only. Both are needed: this one says what the cluster IS, that one says what
    its charged-particle content is made of.
    """
    # Reported for the whole capped frame AND for the train and test halves
    # separately. The composition is a property of the sample, not of the fit, so
    # the three should agree; showing them is what makes that checkable instead of
    # assumed, and it is the split the audience will ask about.
    te = d["in_test"].astype(bool)
    sig = d["y"].astype(bool)
    rows = []
    for (pop, pmask), (split, smask) in [
            (p, s) for p in (("SIGNAL", sig), ("BKG", ~sig))
            for s in (("all", np.ones(len(d), bool)), ("train", ~te), ("test", te))]:
        sub = d[pmask & smask]
        m, tot, names = _light_matrix(sub)
        share = m / tot[:, None]
        top1 = share.max(axis=1)
        nsrc = (m > 0).sum(axis=1)
        dom = pd.Series(np.take(names, m.argmax(axis=1)))
        rows.append(dict(
            streamline=stream, method=method, population=pop, split=split,
            clusters=len(sub),
            median_hits=float(np.median(tot)),
            median_top1=float(np.median(top1)),
            pct_top1_ge90=100.0 * float((top1 >= 0.9).mean()),
            pct_top1_ge50=100.0 * float((top1 >= 0.5).mean()),
            mean_n_sources=float(nsrc.mean()),
            pct_one_source=100.0 * float((nsrc == 1).mean()),
            dominant_source=dom.value_counts().index[0],
            pct_dominant_source=100.0 * float(dom.value_counts(normalize=True).iat[0]),
            mean_share_capture=float(share[:, 0].mean()),
            mean_share_charged=float(share[:, 1:-2].sum(axis=1).mean()),
            mean_share_pureem=float(share[:, -2].mean()),
            mean_share_darknoise=float(share[:, -1].mean()),
        ))
    return pd.DataFrame(rows)


def spurious_purity(d: pd.DataFrame, stream: str, method: str) -> pd.DataFrame:
    """
    Per-population answer to "is a cluster one particle or several?".

    Signal is carried as the control. The claim "background clusters are made of one
    particle" means nothing on its own — it only becomes a result next to how mixed
    the signal clusters are on the same axis.
    """
    g = dirtn_groups(d)
    pops = {"SIGNAL": g.str.startswith("SIGNAL"),
            "BKG tank": g == "BKG tank",
            "BKG world (dirt n)": g == "BKG world neutron-dominated (dirt n)",
            "BKG world non-neutron": g == "BKG world non-neutron",
            "BKG all": ~d["y"].astype(bool)}
    rows = []
    for pop, mask in pops.items():
        sub = d[mask]
        if not len(sub):
            continue
        m, tr, _ = _origin_matrix(sub)
        has = tr > 0
        mm, tt = m[has], tr[has]
        if len(tt):
            share = mm / tt[:, None]
            top1 = share.max(axis=1)
            srt = np.sort(share, axis=1)[:, ::-1]
            top2 = srt[:, 1] if srt.shape[1] > 1 else np.zeros(len(srt))
            nsp = (mm > 0).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -np.where(share > 0, share * np.log(share), 0.0).sum(axis=1)
            hmax = np.log(max(len(SPUR_SPECIES_ORDER), 2))
        else:
            top1 = top2 = nsp = h = np.array([np.nan])
            hmax = 1.0
        rows.append(dict(
            streamline=stream, method=method, population=pop,
            clusters=len(sub),
            clusters_no_traced=int((~has).sum()),
            pct_no_traced=100.0 * float((~has).mean()),
            clusters_with_traced=int(has.sum()),
            median_purity_top1=float(np.median(top1)),
            mean_purity_top1=float(np.mean(top1)),
            median_share_top2=float(np.median(top2)),
            pct_single_species=100.0 * float(np.mean(top1 >= SPUR_SINGLE_CUT)),
            pct_exactly_one_species=100.0 * float(np.mean(nsp == 1)),
            mean_n_species=float(np.mean(nsp)),
            median_entropy_norm=float(np.median(h) / hmax),
            median_frac_neutron=float(sub["frac_neutron"].median()),
        ))
    return pd.DataFrame(rows)


def spurious_by_species(d: pd.DataFrame, stream: str, method: str) -> pd.DataFrame:
    """
    Per dominant-species census of the background, plus the one-vs-signal GBT AUC.

    The AUC is what turns the per-species feature plots into a statement about the
    selection: it says which species the trained model already removes and which it
    does not. Like §4.4's per-population AUCs these share one signal set on
    unbalanced subsets, so they rank difficulty against each other and are not
    standalone performance figures.
    """
    cls = origin_class(d)
    te = d[d["in_test"].astype(bool)]
    cls_te = cls[te.index]
    sig = te[te["y"].astype(bool)]
    bkg_all = d[~d["y"].astype(bool)]
    rows = []
    for sp in SPUR_SPECIES_ORDER + [NO_ORIGIN_LABEL]:
        sub = bkg_all[cls[bkg_all.index] == sp]
        if not len(sub):
            continue
        m, tr, _ = _origin_matrix(sub)
        has = tr > 0
        top1 = (m[has].max(axis=1) / tr[has]) if has.any() else np.array([np.nan])
        nsp = (m[has] > 0).sum(axis=1) if has.any() else np.array([np.nan])
        r = dict(streamline=stream, method=method, species=sp,
                 clusters=len(sub),
                 pct_of_background=100.0 * len(sub) / len(bkg_all),
                 median_purity_top1=float(np.median(top1)),
                 pct_single_species=100.0 * float(np.mean(top1 >= SPUR_SINGLE_CUT)),
                 mean_n_species=float(np.mean(nsp)),
                 median_frac_neutron=float(sub["frac_neutron"].median()),
                 median_n_hits=float(sub["n_hits"].median()),
                 median_pe_total=float(sub["pe_total"].median()),
                 median_d_wall=float(sub["d_wall"].median()),
                 median_gbt_score=float(sub["gbt_score"].median()))
        b_te = te[(cls_te == sp) & ~te["y"].astype(bool)]
        if len(b_te) >= 50 and len(sig):
            y = np.r_[np.ones(len(sig)), np.zeros(len(b_te))]
            for col, model in SCORE_COLS.items():
                if col not in te.columns:
                    continue
                s = np.r_[sig[col].to_numpy(float), b_te[col].to_numpy(float)]
                r[f"auc_{SHORT[model].lower()}"] = float(roc_auc_score(y, s))
            r["n_test_background"] = len(b_te)
            r["n_test_signal"] = len(sig)
        rows.append(r)
    return pd.DataFrame(rows)


def spurious_cooccurrence(d: pd.DataFrame, stream: str, method: str,
                          basis: str = "origin") -> pd.DataFrame:
    """
    Which sources actually share a cluster, among the mixed ones.

    Counting mixtures says only that they exist; this says what they are made of,
    which is the difference between "the background is dirty" and "pi+ light rides
    on mu- light". Hit-weighted, so a species contributing one hit to a large
    mu- cluster does not count the same as an even split.

    basis="origin" is the charged-parent axis, _origin_matrix. Neutron-capture light
    is not on it by construction (capture hits are truth_class 1-4, never -5), so the
    `tr > 0` filter below drops every cluster with no traced charged parent -- 27,393
    of 39,133 on truth-tag / ClusterFinder, about 70%. That is the right answer to
    "what is the charged-particle light made of" and a misleading one to "what is a
    background cluster made of", because neutron capture is in fact the single
    largest background source (61.1% of the light, dominating 76.8% of clusters).

    basis="light" asks the second question, on _light_matrix -- the same partition
    the light-source census uses, over ALL the light: neutron capture, each charged
    parent, pure EM, dark noise. No cluster drops out, because every cluster has
    hits, so the denominator is the whole background. Both are kept: the tables and
    figures downstream of the charged-parent axis (spurious_by_species, D1/D2/D4)
    still want "origin".
    """
    if basis not in ("origin", "light"):
        raise ValueError(f"basis must be 'origin' or 'light', got {basis!r}")
    b = d[~d["y"].astype(bool)]
    if basis == "light":
        m, tr, names = _light_matrix(b)
        has = tr > 0          # every cluster has hits; kept for shape symmetry only
    else:
        m, tr, names = _origin_matrix(b)
        has = tr > 0
    mm, tt = m[has], tr[has]
    share = mm / tt[:, None]
    mixed = share.max(axis=1) < SPUR_SINGLE_CUT
    mx = share[mixed]
    rows = []
    for i, a_ in enumerate(names):
        for j, b_ in enumerate(names):
            if j <= i:
                continue
            both = (mx[:, i] > 0) & (mx[:, j] > 0)
            if not both.any():
                continue
            rows.append(dict(streamline=stream, method=method,
                             species_a=a_, species_b=b_,
                             clusters=int(both.sum()),
                             pct_of_mixed=100.0 * float(both.mean()),
                             mean_share_a=float(mx[both, i].mean()),
                             mean_share_b=float(mx[both, j].mean())))
    out = pd.DataFrame(rows).sort_values("clusters", ascending=False)
    out.insert(3, "n_mixed_clusters", int(mixed.sum()))
    # Carried so the figure can say what the mixed count is a fraction OF. On the
    # origin basis that denominator is not the background -- it is the ~30% of it with
    # traced charged light -- and a figure that only prints the mixed count reads as
    # if it were the whole thing.
    out.insert(4, "n_axis_clusters", int(has.sum()))
    out.insert(5, "n_bkg_clusters", int(len(b)))
    out.insert(6, "basis", basis)
    return out


def spurious_species_shares(d: pd.DataFrame, stream: str,
                            method: str) -> pd.DataFrame:
    """
    Hit-weighted share of the traced non-neutron background light, per species.

    On (truthtag, optics) this is asserted against world report §4.2 — that is the
    single configuration §4.2 was measured on, and the four configurations select
    genuinely different cluster sets, so the others are reported, not checked. Same
    discipline as `dirtn_budget`'s closing check: if the species axis has moved
    under this script, every per-species figure downstream is wrong and it must fail
    loudly rather than produce a plausible new composition.
    """
    b = d[~d["y"].astype(bool)]
    m, tr, names = _origin_matrix(b)
    tot = m.sum(axis=0)
    T = tot.sum()
    got = {n: 100.0 * v / T for n, v in zip(names, tot)}

    if (stream, method) == ("truthtag", "optics"):
        bad = [f"{sp}: got {got.get(sp, 0.0):.2f}%, report says {want:.1f}%"
               for sp, want in REPORTED_SPECIES_PCT.items()
               if abs(got.get(sp, 0.0) - want) > SPECIES_PCT_TOL]
        if bad:
            print("[spurious] species closure FAILED for truthtag/optics:")
            for x in bad:
                print("   ", x)
            sys.exit("[spurious] the origin-species axis no longer matches "
                     "REPORT_ccinc_v3_world_merged.md §4.2 — refusing to continue")
        print("[spurious] species closure PASS — truth-tag/OPTICS reproduces "
              "world report §4.2")

    return pd.DataFrame([dict(streamline=stream, method=method, species=n,
                              hits=float(v), pct_of_traced_bkg_light=got[n])
                         for n, v in zip(names, tot)])


def save_spur(fig, name):
    OUTD.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTD / f"V3SPUR__{name}.{ext}",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  [fig] V3SPUR__{name}.pdf/.png")


def fig_spur_n_species(frames):
    """How many distinct particles make the traced light in one background cluster."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        d = frames[cfg]
        b = d[~d["y"].astype(bool)]
        s = d[d["y"].astype(bool)]
        for sub, c, lbl in ((s, BLUE, "Signal"), (b, GREY, "Background")):
            m, tr, _ = _origin_matrix(sub)
            has = tr > 0
            if not has.any():
                continue
            nsp = (m[has] > 0).sum(axis=1)
            k = np.arange(1, 7)
            frac = [100.0 * float((nsp == i).mean()) for i in k]
            ax.plot(k, frac, "o-", lw=1.8, ms=5, color=c,
                    label=f"{lbl}  ({int(has.sum()):,} clusters)")
        ax.set_xlabel("Distinct particles making the traced light", fontsize=10)
        ax.set_ylabel("Clusters [%]", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("Particles per cluster — clusters with traced non-EM light only",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_spur(fig, "n_species")


def fig_spur_purity(frames):
    """Distribution of the leading species' share of the traced light."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    edges = np.linspace(0, 1, 21)
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        d = frames[cfg]
        for sub, c, lbl in ((d[d["y"].astype(bool)], BLUE, "Signal"),
                            (d[~d["y"].astype(bool)], GREY, "Background")):
            m, tr, _ = _origin_matrix(sub)
            has = tr > 0
            if not has.any():
                continue
            top1 = m[has].max(axis=1) / tr[has]
            ax.hist(top1, bins=edges, histtype="step", lw=1.8, color=c,
                    density=True, label=f"{lbl}  median {np.median(top1):.2f}")
        ax.set_xlabel("Leading particle's share of the traced light", fontsize=10)
        ax.set_ylabel("Clusters (normalised)", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("Leading-particle purity of a cluster — single-particle clusters "
                 "sit at 1.0", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_spur(fig, "purity")


def fig_light_n_sources(frames):
    """How many distinct light sources make up a cluster — ALL light, not just charged."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        d = frames[cfg]
        for sub, c, lbl in ((d[d["y"].astype(bool)], BLUE, "Signal"),
                            (d[~d["y"].astype(bool)], GREY, "Background")):
            m, _, _ = _light_matrix(sub)
            nsrc = (m > 0).sum(axis=1)
            k = np.arange(1, 7)
            ax.plot(k, [100.0 * float((nsrc == i).mean()) for i in k], "o-",
                    lw=1.8, ms=5, color=c,
                    label=f"{lbl}  ({len(sub):,} clusters, mean {nsrc.mean():.2f})")
        ax.set_xlabel("Distinct light sources in the cluster", fontsize=10)
        ax.set_ylabel("Clusters [%]", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("Light sources per cluster — every cluster, all of its light "
                 "(capture / charged parent / pure EM / dark noise)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_spur(fig, "n_sources_alllight")


def fig_light_purity(frames):
    """Distribution of the leading source's share of ALL the light in a cluster."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    edges = np.linspace(0, 1, 21)
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        d = frames[cfg]
        for sub, c, lbl in ((d[d["y"].astype(bool)], BLUE, "Signal"),
                            (d[~d["y"].astype(bool)], GREY, "Background")):
            m, tot, _ = _light_matrix(sub)
            top1 = m.max(axis=1) / tot
            ax.hist(top1, bins=edges, histtype="step", lw=1.8, color=c,
                    density=True, label=f"{lbl}  median {np.median(top1):.2f}")
        ax.set_xlabel("Leading source's share of the cluster's light", fontsize=10)
        ax.set_ylabel("Clusters (normalised)", fontsize=10)
        ax.set_title(CFG_LBL[cfg], fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.tick_params(labelsize=9)
        bare(ax)
    fig.suptitle("Leading-source purity of a cluster — all light, so dark noise "
                 "and in-time EM light count against it", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_spur(fig, "purity_alllight")


def fig_light_sources_census(frames):
    """Share of the light and share of the clusters it dominates, per source."""
    d = frames[("truthtag", "clusterfinder")]
    b = d[~d["y"].astype(bool)]
    m, tot, names = _light_matrix(b)
    pct_light = 100.0 * m.sum(axis=0) / m.sum()
    dom = np.take(names, m.argmax(axis=1))
    pct_clu = np.array([100.0 * float((dom == nm).mean()) for nm in names])
    keep = [i for i in range(len(names)) if pct_light[i] >= 0.1 or pct_clu[i] >= 0.1]
    idx = sorted(keep, key=lambda i: -pct_light[i])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    yy = np.arange(len(idx))[::-1]
    ax.barh(yy + 0.19, [pct_light[i] for i in idx], height=0.36,
            color=GREY, alpha=0.85, edgecolor="none",
            label="% of all background hits")
    ax.barh(yy - 0.19, [pct_clu[i] for i in idx], height=0.36,
            color=BLUE, alpha=0.85, edgecolor="none",
            label="% of background clusters it dominates")
    ax.set_yticks(yy)
    ax.set_yticklabels([names[i] for i in idx], fontsize=11)
    ax.set_xlabel("%", fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    bare(ax)
    # "light" reads as an intensity and "leads" as a ranking; the quantity is a HIT
    # COUNT and the claim is that neutron capture is the largest single BACKGROUND
    # source. Say both literally -- this figure is read in a technical report where
    # "61.1% of the light" is the sort of phrase that gets requoted as a PE fraction.
    ax.set_title("What Makes the Background Light — truth-tag / ClusterFinder\n"
                 "{:,} clusters, {:,} hits;  neutron capture (background) is "
                 "{:.1f}% of the background hits and dominates {:.1f}% of the "
                 "clusters".format(len(b), int(m.sum()), pct_light[0], pct_clu[0]),
                 fontsize=12)
    fig.tight_layout()
    save_spur(fig, "light_sources_census")


def fig_light_sources_table(frames):
    """The all-light census as numbers: what the light is, and what it dominates."""
    d = frames[("truthtag", "clusterfinder")]
    b = d[~d["y"].astype(bool)]
    m, tot, names = _light_matrix(b)
    pct_light = 100.0 * m.sum(axis=0) / m.sum()
    share = m / tot[:, None]
    dom = np.take(names, m.argmax(axis=1))
    rows = []
    for i, nm in enumerate(names):
        sel = dom == nm
        if pct_light[i] < 0.1 and not sel.any():
            continue
        led = share[sel, i] if sel.any() else np.array([np.nan])
        # The full pure-EM label is a sentence; it overruns the cell at this width.
        short = r"pure EM ($\gamma$/$e^\pm$)" if nm == PUREEM_LABEL else nm
        rows.append([short, f"{pct_light[i]:.2f}", f"{int(sel.sum()):,}",
                     f"{100.0 * sel.mean():.2f}",
                     "—" if not sel.any() else f"{np.median(led):.2f}",
                     "—" if not sel.any() else f"{100.0 * (led >= 0.9).mean():.1f}"])
    rows.sort(key=lambda r: -float(r[1]))
    rows.append(["total", f"{pct_light.sum():.0f}", f"{len(b):,}", "100", "—", "—"])
    # "hits" not "light", matching the census figure this table is the numbers for.
    cols = ["light source", "% of all\nbkg hits", "clusters it\ndominates",
            "% of bkg\nclusters", "median share\nwhere it leads",
            "% of those\n≥90% pure"]
    fig, ax = plt.subplots(figsize=(10.4, 0.46 * len(rows) + 1.5))
    table_axes(ax, rows, cols, fs=10.0)
    ax.set_title("What Makes the Background Light — truth-tag / ClusterFinder\n"
                 "{:,} clusters, {:,} hits; every hit is in exactly one row".format(
                     len(b), int(m.sum())), fontsize=12.5, pad=14)
    fig.tight_layout()
    save_spur(fig, "light_sources_table")


def fig_light_purity_table(lp):
    """Signal against background on the all-light mixture metrics."""
    t = lp.query("streamline=='truthtag' and method=='clusterfinder' "
                 "and split=='all'").set_index("population")
    spec = [("clusters", "clusters", lambda v: f"{int(v):,}"),
            ("median hits per cluster", "median_hits", lambda v: f"{v:.0f}"),
            ("median share held by the leading source", "median_top1",
             lambda v: f"{v:.2f}"),
            ("clusters ≥90% from one source [%]", "pct_top1_ge90",
             lambda v: f"{v:.1f}"),
            ("clusters ≥50% from one source [%]", "pct_top1_ge50",
             lambda v: f"{v:.1f}"),
            ("mean number of sources present", "mean_n_sources",
             lambda v: f"{v:.2f}"),
            ("clusters with exactly one source [%]", "pct_one_source",
             lambda v: f"{v:.1f}"),
            ("mean share: neutron capture", "mean_share_capture",
             lambda v: f"{v:.3f}"),
            ("mean share: charged parent", "mean_share_charged",
             lambda v: f"{v:.3f}"),
            ("mean share: dark noise", "mean_share_darknoise",
             lambda v: f"{v:.3f}"),
            (r"mean share: pure EM $\gamma$/$e^\pm$", "mean_share_pureem",
             lambda v: f"{v:.3f}")]
    cell = [[lbl, fmt(t.loc["SIGNAL", col]), fmt(t.loc["BKG", col])]
            for lbl, col, fmt in spec]
    fig, ax = plt.subplots(figsize=(8.8, 0.46 * len(cell) + 1.5))
    table_axes(ax, cell, ["", "signal", "background"], fs=10.5,
               emphasise_last=False)
    ax.set_title("Is a Cluster One Thing or Several? — all light, "
                 "truth-tag / ClusterFinder\n"
                 "composition purity is nearly the same for both classes",
                 fontsize=12.5, pad=14)
    fig.tight_layout()
    save_spur(fig, "light_purity_table")


def fig_light_split_table(lp):
    """The same composition numbers on the train half and the test half separately.

    Every other figure in this block is quoted on the whole capped frame. That is
    the right denominator for "what is the background made of", but it invites the
    question of whether the test half looks like what the model was fitted on. This
    table answers it with numbers instead of an assurance.
    """
    t = lp.query("streamline=='truthtag' and method=='clusterfinder'")
    cell = []
    for pop in ("SIGNAL", "BKG"):
        for split in ("all", "train", "test"):
            r = t[(t["population"] == pop) & (t["split"] == split)].iloc[0]
            cell.append([f"{'signal' if pop == 'SIGNAL' else 'background'}"
                         f" — {split}", f"{int(r.clusters):,}",
                         f"{r.median_hits:.0f}", f"{r.median_top1:.2f}",
                         f"{r.pct_top1_ge90:.1f}", f"{r.mean_n_sources:.2f}",
                         f"{r.mean_share_capture:.3f}",
                         f"{r.mean_share_charged:.3f}"])
    cols = ["population", "clusters", "median\nhits", "median\ntop-1 share",
            "≥90% from\none source [%]", "mean\nsources",
            "mean share\ncapture", "mean share\ncharged"]
    fig, ax = plt.subplots(figsize=(11.2, 0.46 * len(cell) + 1.6))
    table_axes(ax, cell, cols, fs=10.0, emphasise_last=False)
    ax.set_title("Composition on the Train and Test Halves — "
                 "truth-tag / ClusterFinder\n"
                 "80/20 split of the 1:1-capped frame; the halves agree, so the "
                 "block may be quoted on either", fontsize=12.5, pad=14)
    fig.tight_layout()
    save_spur(fig, "light_split_table")


def fig_spur_census(bysp):
    """Per-species census as a table: how much of the background, how pure, how hard."""
    t = bysp.query("streamline=='truthtag' and method=='clusterfinder'")
    t = t[t["clusters"] >= SPUR_MIN_CLUSTERS]
    # The mixture columns are undefined for the no-non-EM-origin class by
    # construction (it has no traced light to be a mixture of). Print an em dash,
    # not a NaN or a 0 — a 0 in "% single" reads as "never single-particle".
    def _mix(v, fmt):
        return "—" if not np.isfinite(v) else format(v, fmt)

    cell = [[SPUR_PRETTY.get(r.species, r.species), f"{r.clusters:,}",
             f"{r.pct_of_background:.1f}",
             _mix(r.median_purity_top1, ".2f"),
             ("—" if not np.isfinite(r.mean_n_species)
              else f"{r.pct_single_species:.0f}"),
             _mix(r.mean_n_species, ".2f"),
             f"{r.median_frac_neutron:.2f}", f"{r.median_gbt_score:.3f}",
             (f"{r.auc_gbt:.3f}" if pd.notna(getattr(r, "auc_gbt", np.nan))
              else "—")]
            for r in t.itertuples()]
    cols = ["particle", "clusters", "% of bkg", "median\npurity", "% single",
            "mean\nspecies", "median\nfrac n", "median\nGBT", "GBT AUC\nvs signal"]
    fig, ax = plt.subplots(figsize=(11.0, 0.52 * len(cell) + 1.5))
    table_axes(ax, cell, cols, fs=10.0, emphasise_last=False)
    ax.set_title("Background by dominant particle — truth-tag / ClusterFinder",
                 fontsize=12.5, pad=14)
    fig.tight_layout()
    save_spur(fig, "species_census")


def _co_tt_cf(co):
    """The truth-tag / ClusterFinder rows, with the denominators the title needs."""
    t = co.query("streamline=='truthtag' and method=='clusterfinder'")
    if not len(t):
        return None, {}
    r0 = t.iloc[0]
    return t, dict(mixed=int(r0["n_mixed_clusters"]),
                   axis=int(r0["n_axis_clusters"]),
                   bkg=int(r0["n_bkg_clusters"]))


def fig_spur_cooccurrence(co):
    """The commonest source pairs inside mixed background clusters.

    Fed the ALL-LIGHT co-occurrence table (basis="light"), so neutron capture, dark
    noise and pure EM are eligible as partners. On the charged-parent basis this
    figure described only the ~30% of background clusters that have traced charged
    light, while printing a bare mixed-cluster count that read as if it covered the
    background -- and it structurally could not show the largest background source.
    """
    t, n = _co_tt_cf(co)
    if t is None:
        return
    t = t.head(10)
    lbl = [f"{SPUR_PRETTY.get(r.species_a, r.species_a)} + "
           f"{SPUR_PRETTY.get(r.species_b, r.species_b)}" for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(t) + 2.2))
    ypos = np.arange(len(t))[::-1]
    ax.barh(ypos, t["pct_of_mixed"].to_numpy(float), color=GREY, height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(lbl, fontsize=10)
    ax.set_xlabel("Share of mixed background clusters containing both [%]",
                  fontsize=10)
    ax.tick_params(labelsize=9)
    bare(ax)
    ax.set_title(f"Commonest source pairs in mixed background clusters — "
                 f"{n['mixed']:,} mixed of {n['bkg']:,} background clusters, "
                 f"all light, truth-tag / ClusterFinder", fontsize=12)
    fig.tight_layout()
    save_spur(fig, "cooccurrence")


def fig_spur_cooccurrence_matrix(co):
    """Every source pair at once, as a matrix. Backup for fig_spur_cooccurrence.

    The top-10 bar chart is the slide; this is the thing to put up when someone asks
    about a pair that is not in the top 10. Same long-form table, pivoted -- the
    matrix is symmetric and only the upper triangle is populated, so it is mirrored
    here rather than drawn half-empty.
    """
    t, n = _co_tt_cf(co)
    if t is None:
        return
    names = sorted(set(t.species_a) | set(t.species_b),
                   key=lambda s: -t.loc[(t.species_a == s) | (t.species_b == s),
                                        "pct_of_mixed"].sum())
    idx = {s: i for i, s in enumerate(names)}
    M = np.full((len(names), len(names)), np.nan)
    for r in t.itertuples():
        i, j = idx[r.species_a], idx[r.species_b]
        M[i, j] = M[j, i] = r.pct_of_mixed

    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3.0,
                                    1.0 * len(names) + 2.2))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="YlOrBr", aspect="equal")
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=8)
    pretty = [SPUR_PRETTY.get(s, s) for s in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(pretty, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(pretty, fontsize=9)
    fig.colorbar(im, ax=ax, label="% of mixed background clusters with both",
                 fraction=0.046)
    ax.set_title(f"Source co-occurrence in mixed background clusters — "
                 f"{n['mixed']:,} mixed of {n['bkg']:,},\nall light, "
                 f"truth-tag / ClusterFinder", fontsize=11.5)
    fig.tight_layout()
    save_spur(fig, "cooccurrence_matrix")


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(prog="ccinc_v3_stats")
    ap.add_argument("--do", required=True,
                    choices=["guard", "models", "truthreco", "dirtn", "spurious",
                             "all"],
                    help="Which section to run. No default on purpose.")
    a = ap.parse_args()

    g = guard()
    frames = g["frames"]
    if a.do == "guard":
        return

    if a.do in ("models", "all"):
        trainings = collect_merged_trainings()
        nfeat = n_features_used(trainings)
        print("[logs] features used per training: "
              + ", ".join(f"{CFG_LBL[c]} {nfeat.get(c)}" for c in CONFIGS))
        cmp = model_comparison(frames, trainings)
        cmp["n_features"] = cmp.apply(
            lambda r: nfeat.get((r.streamline, r.method)), axis=1)
        cmp.to_csv(HERE / "ccinc_v3_model_comparison.csv", index=False)
        print(f"[csv] ccinc_v3_model_comparison.csv  ({len(cmp)} rows)")
        crosscheck_effpurity(cmp)
        vd = verdict(cmp)
        print("\n=== best model per configuration ===")
        print(vd.to_string(index=False))
        fig_roc_by_config(frames)
        fig_effpurity_scan(frames)
        fig_loss_comparison(cmp)
        fig_verdict_table(cmp, vd)

    if a.do in ("truthreco", "all"):
        print("\n=== truth-tag vs reco-tag ===")
        event_overlap_check()
        rows, comps, brk = [], [], []
        for method in ("optics", "clusterfinder"):
            rows.append(cluster_overlap(frames, method))
            brk.append(overlap_breakdown(frames, method))
            comps.append(disagreement_composition(frames, method))
        pd.DataFrame(rows).to_csv(
            HERE / "ccinc_v3_truthreco_discrepancy.csv", index=False)
        pd.concat(brk).to_csv(
            HERE / "ccinc_v3_truthreco_overlap_breakdown.csv", index=False)
        pd.concat(comps).to_csv(
            HERE / "ccinc_v3_truthreco_disagreement_composition.csv", index=False)
        print("[csv] ccinc_v3_truthreco_discrepancy.csv, "
              "ccinc_v3_truthreco_overlap_breakdown.csv, "
              "ccinc_v3_truthreco_disagreement_composition.csv")

    if a.do in ("dirtn", "all"):
        print("\n=== dirt-neutron background: composition and separability ===")
        bud, cen, auc, sep, tnk = [], [], [], [], []
        for stream, method in CONFIGS:
            d = frames[(stream, method)]
            bud.append(dirtn_budget(d, stream, method))
            cen.append(dirtn_census(d, stream, method))
            a_, s_ = dirtn_separability(d, stream, method)
            auc.append(a_)
            sep.append(s_)
            tnk.append(dirtn_tank_neutron_light(d, stream, method))
        pd.concat(bud).to_csv(HERE / "ccinc_v3_dirtn_budget.csv", index=False)
        pd.concat(cen).to_csv(HERE / "ccinc_v3_dirtn_census.csv", index=False)
        pd.concat(auc).to_csv(HERE / "ccinc_v3_dirtn_separability.csv", index=False)
        pd.concat(sep).to_csv(HERE / "ccinc_v3_dirtn_feature_separation.csv",
                              index=False)
        pd.concat(tnk).to_csv(HERE / "ccinc_v3_dirtn_tankbkg_neutron_light.csv",
                              index=False)
        print("[csv] ccinc_v3_dirtn_{budget,census,separability,"
              "feature_separation,tankbkg_neutron_light}.csv")

        b0 = pd.concat(bud).query("streamline=='truthtag' and method=='optics'")
        print("\n  truth-tag/OPTICS background light budget "
              "(% of all background hits):")
        for _, r in b0.iterrows():
            print(f"    {r.component:60s} tank {r.pct_tank:5.2f}  "
                  f"world {r.pct_world:5.2f}  total {r.pct_total:5.2f}")
        a0 = pd.concat(auc).query("streamline=='truthtag' and method=='optics' "
                                  "and model=='GBT'")
        print("\n  GBT AUC against each background population:")
        for _, r in a0.iterrows():
            print(f"    signal vs {r.population:40s} n={r.n_background:6d}  "
                  f"AUC={r.auc:.3f}")

    if a.do in ("spurious", "all"):
        print("\n=== background spuriousness: one particle, or several? ===")
        pur, bysp, co, col, shr = [], [], [], [], []
        for stream, method in CONFIGS:
            d = frames[(stream, method)]
            shr.append(spurious_species_shares(d, stream, method))
            pur.append(spurious_purity(d, stream, method))
            bysp.append(spurious_by_species(d, stream, method))
            co.append(spurious_cooccurrence(d, stream, method))
            # The all-light basis is written alongside the charged-parent one, not
            # over it: spurious_by_species and the D1/D2/D4 figures still read the
            # charged-parent table, and the two answer different questions.
            col.append(spurious_cooccurrence(d, stream, method, basis="light"))
        pur = pd.concat(pur, ignore_index=True)
        bysp = pd.concat(bysp, ignore_index=True)
        co = pd.concat(co, ignore_index=True)
        col = pd.concat(col, ignore_index=True)
        pd.concat(shr, ignore_index=True).to_csv(
            HERE / "ccinc_v3_spurious_species_shares.csv", index=False)
        pur.to_csv(HERE / "ccinc_v3_spurious_purity.csv", index=False)
        bysp.to_csv(HERE / "ccinc_v3_spurious_byspecies.csv", index=False)
        co.to_csv(HERE / "ccinc_v3_spurious_cooccurrence.csv", index=False)
        col.to_csv(HERE / "ccinc_v3_spurious_cooccurrence_alllight.csv",
                   index=False)
        print("[csv] ccinc_v3_spurious_{species_shares,purity,byspecies,"
              "cooccurrence,cooccurrence_alllight}.csv")

        p0 = pur.query("streamline=='truthtag' and method=='clusterfinder'")
        print("\n  truth-tag/ClusterFinder — composition of a cluster:")
        print(f"    {'population':24s} {'clusters':>9s} {'no non-EM':>10s} "
              f"{'median':>8s} {'% single':>9s} {'mean':>6s}")
        print(f"    {'':24s} {'':>9s} {'origin':>10s} {'purity':>8s} "
              f"{'species':>9s} {'nsp':>6s}")
        for _, r in p0.iterrows():
            print(f"    {r.population:24s} {r.clusters:9,d} "
                  f"{r.pct_no_traced:9.1f}% {r.median_purity_top1:8.2f} "
                  f"{r.pct_single_species:8.1f}% {r.mean_n_species:6.2f}")

        b0 = bysp.query("streamline=='truthtag' and method=='clusterfinder'")
        b0 = b0[b0["clusters"] >= SPUR_MIN_CLUSTERS]
        print("\n  GBT AUC against each background particle "
              "(signal vs that particle only):")
        for _, r in b0.iterrows():
            auc_s = (f"{r.auc_gbt:.3f}" if "auc_gbt" in b0.columns
                     and pd.notna(r.auc_gbt) else "  --")
            print(f"    signal vs {r.species:22s} n={int(r.clusters):6d} "
                  f"({r.pct_of_background:5.1f}% of bkg)  AUC={auc_s}")

        lp = pd.concat([light_purity(frames[c], *c) for c in CONFIGS],
                       ignore_index=True)
        lp.to_csv(HERE / "ccinc_v3_light_purity.csv", index=False)
        print("\n  all-light view — is a CLUSTER one thing or several?")
        l0 = lp.query("streamline=='truthtag' and method=='clusterfinder'")
        for _, r in l0.iterrows():
            print(f"    {r.population:6s} {r.split:5s} {r.clusters:8,d}  median top-1 "
                  f"{r.median_top1:.2f}  >=90% {r.pct_top1_ge90:5.1f}%  "
                  f">=50% {r.pct_top1_ge50:5.1f}%  mean sources "
                  f"{r.mean_n_sources:.2f}  led by {r.dominant_source} "
                  f"({r.pct_dominant_source:.1f}%)")

        fig_spur_n_species(frames)
        fig_spur_purity(frames)
        fig_light_n_sources(frames)
        fig_light_purity(frames)
        fig_light_sources_census(frames)
        fig_light_sources_table(frames)
        fig_light_purity_table(lp)
        fig_light_split_table(lp)
        fig_spur_census(bysp)
        fig_spur_cooccurrence(col)
        fig_spur_cooccurrence_matrix(col)


if __name__ == "__main__":
    main()
