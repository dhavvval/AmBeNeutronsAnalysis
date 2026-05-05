"""
analyze_containment.py
======================
Computes neutron containment and detection threshold efficiencies for any
AmBe MC ROOT file that has been processed through `ambe mc process`.

Three efficiency components are separated:

  ε_geo     — Geometric containment:
              Fraction of fired neutrons that thermalize and capture inside
              the active tank volume, producing at least one PMT hit.
              Neutrons that escape or capture outside leave no hits at all
              and appear as missing event IDs in the parquet.

  ε_det(ms) — Detection threshold efficiency (given containment):
              Of the events that DO produce hits, what fraction have enough
              neutron-class hits for OPTICS to form a cluster at min_samples=ms.
              This separates algorithm threshold from geometry.

  ε_total(ms) = ε_geo × ε_det(ms)   — end-to-end efficiency

Usage:
    python analyze_containment.py \\
        --root-file /path/to/file.root \\
        --run-name  mc_steven_port5 \\
        --output-root /path/to/ambe_output

Output:
    Prints efficiency table to stdout.
    Writes containment_<run_name>.txt summary.
    Appends one row to containment_summary.csv (for multi-file comparison).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import awkward as ak

NEUTRON_CLASSES = {1, 2, 3, 4}


# ─────────────────────────────────────────────────────────────────────────────
# ROOT reader — just needs total event count + per-event neutron hit presence
# ─────────────────────────────────────────────────────────────────────────────

def count_root_events(root_file: str) -> int:
    """Return total number of events in the ROOT file Event tree."""
    with uproot.open(root_file) as f:
        tree = f["Event"]
        return int(tree.num_entries)


def read_root_neutron_presence(root_file: str, max_events=None) -> dict:
    """
    For each event in the ROOT file, determine if the neutron produced
    any PMT hit at all (using DirectParent_NeutronAncestorClass).
    Returns dict: {event_id: bool_has_any_neutron_hit}
    """
    print(f"  Reading ROOT file: {root_file}")
    with uproot.open(root_file) as f:
        tree = f["Event"]
        available = set(tree.keys())

        # Minimum branches needed
        has_dp_class = "DirectParent_NeutronAncestorClass" in available
        has_hit_t    = "hitT" in available

        if not has_hit_t:
            print("  WARNING: hitT branch not found — cannot count hits per event")
            return {}

        branches = ["eventNumber", "hitT"]
        if has_dp_class:
            branches.append("DirectParent_NeutronAncestorClass")
            branches.append("DirectParent_PMTID")

        arr = tree.arrays(branches, library="ak")

    n = min(len(arr), max_events) if max_events else len(arr)
    print(f"  Total events in ROOT file: {n}")

    result = {}
    for i in range(n):
        evid = int(arr["eventNumber"][i])
        n_hits = len(ak.to_list(arr["hitT"][i]))

        has_neutron_hit = False
        if has_dp_class and n_hits > 0:
            dp_cls = ak.to_list(arr["DirectParent_NeutronAncestorClass"][i])
            has_neutron_hit = any(c in NEUTRON_CLASSES for c in dp_cls)

        result[evid] = {
            "n_total_hits": n_hits,
            "has_any_hit":  n_hits > 0,
            "has_neutron_hit": has_neutron_hit,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Parquet reader — per-event neutron hit counts
# ─────────────────────────────────────────────────────────────────────────────

def read_parquet_neutron_counts(parquet_dir: Path, run_name: str) -> pd.DataFrame:
    """
    Read pulses parquet and return per-event neutron hit counts.
    Returns DataFrame: eventID | n_total_hits | n_neutron_hits
    """
    path = parquet_dir / f"{run_name}__pulses.parquet"
    if not path.exists():
        sys.exit(f"Parquet not found: {path}\nRun `ambe mc process` first.")

    pulses = pd.read_parquet(path)
    per_ev = pulses.groupby("eventID").agg(
        n_total_hits  = ("truth_class", "count"),
        n_neutron_hits = ("is_neutron", "sum"),
    ).reset_index()
    per_ev["n_neutron_hits"] = per_ev["n_neutron_hits"].astype(int)
    return per_ev


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiencies(root_presence: dict, parquet_counts: pd.DataFrame,
                         run_name: str) -> dict:
    """
    Combine ROOT event count with parquet hit counts to compute all efficiencies.
    """
    n_total_root = len(root_presence)

    # Events in parquet (had ≥1 hit of any kind — wrote at least one row)
    parquet_ev_ids  = set(parquet_counts["eventID"])
    n_with_any_hit  = len(parquet_ev_ids)

    # Events with ≥1 neutron-class hit
    n_with_neutron  = int((parquet_counts["n_neutron_hits"] >= 1).sum())

    # Detection threshold efficiencies
    det = {}
    for ms in [1, 2, 3, 5, 8, 10]:
        n = int((parquet_counts["n_neutron_hits"] >= ms).sum())
        det[ms] = n

    # Events that fired but left no trace (escaped / captured outside tank)
    root_ev_ids = set(root_presence.keys())
    n_zero_hit  = len(root_ev_ids - parquet_ev_ids)

    # Geometric containment
    eps_geo = n_with_neutron / n_total_root if n_total_root > 0 else 0

    lines = []
    def p(s=""): print(s); lines.append(s)

    p("=" * 65)
    p(f"  CONTAINMENT EFFICIENCY — {run_name}")
    p("=" * 65)
    p(f"  Total events in ROOT file:             {n_total_root:>6}")
    p(f"  Events with ≥1 PMT hit (any class):    {n_with_any_hit:>6}")
    p(f"  Events with ≥1 neutron-class hit:      {n_with_neutron:>6}")
    p(f"  Events with zero hits (escaped/lost):  {n_zero_hit:>6}  "
      f"({100*n_zero_hit/n_total_root:.1f}%)")
    p()
    p(f"  ε_geo  (geometric containment)         {eps_geo:.4f}  "
      f"= {n_with_neutron}/{n_total_root}  ({100*eps_geo:.1f}%)")
    p()
    p(f"  Detection threshold efficiency ε_det(ms) — given containment:")
    p(f"  {'min_samples':>12} | {'N events':>9} | {'ε_det':>8} | {'ε_total':>8}")
    p("  " + "-" * 46)
    for ms in [1, 2, 3, 5, 8, 10]:
        n = det[ms]
        eps_det   = n / n_with_neutron if n_with_neutron > 0 else 0
        eps_total = eps_geo * eps_det
        p(f"  {ms:>12} | {n:>9} | {eps_det:>8.4f} | {eps_total:>8.4f}  "
          f"({100*eps_total:.1f}%)")
    p()
    p("  Interpretation:")
    p(f"  - Of {n_total_root} fired neutrons, {n_with_neutron} captured inside "
      f"tank ({100*eps_geo:.1f}% geometric containment)")
    p(f"  - Of those {n_with_neutron} captures, {det[8]} produce ≥8 hits "
      f"(OPTICS ms=8 detectable: {100*det[8]/n_with_neutron:.1f}%)")
    p(f"  - End-to-end efficiency at ms=8: "
      f"{100*eps_geo*det[8]/n_with_neutron:.1f}%")
    p("=" * 65)

    return {
        "run_name":         run_name,
        "n_total_root":     n_total_root,
        "n_with_neutron":   n_with_neutron,
        "n_zero_hit":       n_zero_hit,
        "eps_geo":          round(eps_geo, 4),
        "eps_det_ms3":      round(det[3] / n_with_neutron, 4) if n_with_neutron else 0,
        "eps_det_ms5":      round(det[5] / n_with_neutron, 4) if n_with_neutron else 0,
        "eps_det_ms8":      round(det[8] / n_with_neutron, 4) if n_with_neutron else 0,
        "eps_total_ms3":    round(eps_geo * det[3] / n_with_neutron, 4) if n_with_neutron else 0,
        "eps_total_ms5":    round(eps_geo * det[5] / n_with_neutron, 4) if n_with_neutron else 0,
        "eps_total_ms8":    round(eps_geo * det[8] / n_with_neutron, 4) if n_with_neutron else 0,
        "_lines":           lines,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compute neutron containment and detection efficiency.")
    ap.add_argument("--root-file",    required=True,
                    help="Path to WCSim ROOT file")
    ap.add_argument("--run-name",     required=True,
                    help="Pipeline run_name (matches parquet filenames)")
    ap.add_argument("--output-root",  required=True,
                    help="ambe_output root directory (same as config output_root)")
    ap.add_argument("--max-events",   type=int, default=None,
                    help="Cap number of events to read from ROOT (debug)")
    args = ap.parse_args()

    root_file    = args.root_file
    run_name     = args.run_name
    parquet_dir  = Path(args.output_root) / run_name / "parquet"
    out_dir      = Path(args.output_root) / run_name

    if not Path(root_file).exists():
        sys.exit(f"ROOT file not found: {root_file}")

    # Step 1 — read ROOT file
    print(f"\n[containment] Reading ROOT file ...")
    root_presence = read_root_neutron_presence(root_file, args.max_events)

    # Step 2 — read parquet
    print(f"[containment] Reading parquet ...")
    parquet_counts = read_parquet_neutron_counts(parquet_dir, run_name)
    print(f"  Events in parquet: {len(parquet_counts)}")

    # Step 3 — compute
    print(f"[containment] Computing efficiencies ...\n")
    result = compute_efficiencies(root_presence, parquet_counts, run_name)

    # Step 4 — save
    txt_path = out_dir / f"containment_{run_name}.txt"
    txt_path.write_text("\n".join(result["_lines"]))
    print(f"\nSaved: {txt_path}")

    # Append to summary CSV for multi-file comparison
    csv_path = Path(args.output_root) / "containment_summary.csv"
    row = {k: v for k, v in result.items() if k != "_lines"}
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Replace existing row for this run_name if present
        existing = existing[existing["run_name"] != run_name]
        df_row = pd.concat([existing, df_row], ignore_index=True)
    df_row.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
