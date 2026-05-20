"""
Phase 1 gate: inspect all raw data files, report coverage, flag missing SMILES.
Run:  python scripts/inspect_dataset.py
"""
import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    sep("GDSC2 dose-response (primary label set)")
    gdsc2 = pd.read_csv(ROOT / "GDSC2-dataset.csv")
    print(f"  Rows         : {len(gdsc2):,}")
    print(f"  Unique drugs : {gdsc2.DRUG_ID.nunique()}")
    print(f"  Unique cells : {gdsc2.COSMIC_ID.nunique()}")
    print(f"  LN_IC50 range: {gdsc2.LN_IC50.min():.3f} to {gdsc2.LN_IC50.max():.3f}")
    print(f"  AUC range    : {gdsc2.AUC.min():.3f} to {gdsc2.AUC.max():.3f}")
    print(f"  Missing IC50 : {gdsc2.LN_IC50.isna().sum()}")

    sep("GDSC combined (merged metadata)")
    gdsc = pd.read_csv(ROOT / "GDSC_DATASET.csv")
    print(f"  Rows         : {len(gdsc):,}")
    print(f"  Columns      : {list(gdsc.columns)}")
    print(f"  Tissues      : {gdsc['GDSC Tissue descriptor 2'].nunique()} unique")
    print(f"  TCGA labels  : {gdsc['Cancer Type (matching TCGA label)'].nunique()} unique")

    sep("Compounds annotation (drug metadata)")
    cpd = pd.read_csv(ROOT / "Compounds-annotation.csv")
    print(f"  Rows         : {len(cpd)}")
    print(f"  Columns      : {list(cpd.columns)}")
    has_smiles = any("smiles" in c.lower() for c in cpd.columns)
    print(f"  Has SMILES   : {has_smiles}  ← {'OK' if has_smiles else 'MISSING — will fetch from PubChem'}")
    print(f"  Pathways     : {cpd.TARGET_PATHWAY.value_counts().head(5).to_dict()}")

    sep("Cell Lines Details")
    cells = pd.read_excel(ROOT / "Cell_Lines_Details.xlsx")
    print(f"  Rows         : {len(cells)}")
    print(f"  Columns      : {list(cells.columns)}")
    gene_expr_col = [c for c in cells.columns if "Gene Expression" in c]
    if gene_expr_col:
        col = gene_expr_col[0]
        y_count = (cells[col] == "Y").sum()
        print(f"  Gene Expr Y  : {y_count}/{len(cells)} cell lines")
        print(f"  ⚠  Expression MATRIX not in Kaggle dump.")
        print(f"     Download: https://www.cancerrxgene.org/downloads/bulk_download")
        print(f"     File: Cell_line_RMA_proc_basalExp.txt.gz")

    sep("GDSC2 × Compounds overlap")
    gdsc2_drug_ids = set(gdsc2.DRUG_ID.unique())
    cpd_drug_ids = set(cpd.DRUG_ID.unique())
    overlap = gdsc2_drug_ids & cpd_drug_ids
    print(f"  GDSC2 drugs          : {len(gdsc2_drug_ids)}")
    print(f"  Compounds annotation : {len(cpd_drug_ids)}")
    print(f"  Overlap              : {len(overlap)}")
    print(f"  GDSC2 drugs missing  : {gdsc2_drug_ids - cpd_drug_ids}")

    sep("Summary / action items")
    print("  [1] SMILES strings   → run scripts/fetch_smiles.py")
    print("  [2] Expression matrix → download from cancerrxgene.org (see above)")
    print("  [3] Splits           → run scripts/build_splits.py after step 1")
    print("  [4] Pathway mask     → run scripts/build_pathway_mask.py after step 2")


if __name__ == "__main__":
    main()
