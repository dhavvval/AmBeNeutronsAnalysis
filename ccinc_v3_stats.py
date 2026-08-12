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
    python ccinc_v3_stats.py --do all
Output:
    ccinc_v3_model_comparison.csv
    ccinc_v3_truthreco_discrepancy.csv
    slide_plots_ccinc_v3_merged/V3STATS__*.{pdf,png}
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
def main():
    ap = argparse.ArgumentParser(prog="ccinc_v3_stats")
    ap.add_argument("--do", required=True,
                    choices=["guard", "models", "truthreco", "all"],
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


if __name__ == "__main__":
    main()
