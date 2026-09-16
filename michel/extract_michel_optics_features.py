"""Recluster tagged Michel hits using the maintained MC feature extractor.

Example: python michel/extract_michel_optics_features.py --config CONFIG.yaml
         --hits michel_background_hits.parquet --min-samples 3 --output features.parquet
This runs one OPTICS setting on previously selected clusters, not a sweep.
"""
from ambe.cli import main
import sys

if __name__ == "__main__":
    main(["mc", "features", "--rerun-optics", *sys.argv[1:]])
