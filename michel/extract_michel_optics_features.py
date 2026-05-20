"""
Thin wrapper kept for backward compatibility.
Use extract_cluster_features.py directly instead:

    python3 extract_cluster_features.py --hits <path> --output <path> --rerun-optics
"""
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.exit(subprocess.call([sys.executable, str(root / "extract_cluster_features.py"),
                          "--rerun-optics", *sys.argv[1:]]))
