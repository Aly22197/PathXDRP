"""
Build all five split regimes for seeds 0-4.

Prerequisites:
  1. python pathxdrp/data/fetch_smiles.py      (SMILES already fetched)
  2. python scripts/download_expression.py     (expression matrix downloaded)

Usage: python scripts/build_splits.py
"""
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathxdrp.data.loader import build_master_df
from pathxdrp.data.splits import build_all_splits

print("Loading data")
df, _expr = build_master_df(version="GDSC2", require_smiles=True)
print(f"Building splits for {len(df):,} rows")
build_all_splits(df, seeds=[0, 1, 2, 3, 4])
