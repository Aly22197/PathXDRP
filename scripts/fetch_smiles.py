"""
CLI shim: python scripts/fetch_smiles.py [--dry-run]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathxdrp.data.fetch_smiles import fetch_all
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
fetch_all(dry_run=args.dry_run)
