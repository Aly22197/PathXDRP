"""
GDSC data loading utilities.

Produces:
  1. Master DataFrame (merged IC50 + SMILES + cell metadata), filtered to rows
     that have both SMILES and DepMap expression coverage.
  2. Expression matrix — (COSMIC_ID × gene_symbol), Z-scored log2(TPM+1), float32.

Expression source: DepMap 24Q4
  File:   data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv  (~507 MB)
  Rows:   ModelID (ACH-XXXXXX)
  Cols:   "GENE_SYMBOL (entrez_id)"  e.g. "TSPAN6 (7105)"
  Values: log2(TPM + 1)

COSMIC -> ModelID mapping: data/processed/cosmic_to_depmap.csv
  (946/969 GDSC2 cell lines covered, verified against DepMap 24Q4 Model.csv)

Download expression with:
  python scripts/download_expression.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"


# --- Individual loaders ---

def load_response(version: str = "GDSC2") -> pd.DataFrame:
    """Load the raw IC50 response file; deduplicate (DRUG_ID, COSMIC_ID)."""
    path = ROOT / f"{version}-dataset.csv"
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["DRUG_ID", "COSMIC_ID"])
    return df


def load_cell_metadata() -> pd.DataFrame:
    """Load Cell_Lines_Details.xlsx and normalise column names."""
    path = ROOT / "Cell_Lines_Details.xlsx"
    cells = pd.read_excel(path)
    rename = {
        "COSMIC identifier":                    "COSMIC_ID",
        "Sample Name":                          "CELL_LINE_NAME",
        "GDSC\nTissue descriptor 1":            "tissue_1",
        "GDSC\nTissue\ndescriptor 2":           "tissue_2",
        "Cancer Type\n(matching TCGA label)":   "cancer_type",
        "Microsatellite \ninstability Status (MSI)": "msi_status",
        "Growth Properties":                    "growth_props",
    }
    cells = cells.rename(columns={k: v for k, v in rename.items() if k in cells.columns})
    cells["COSMIC_ID"] = pd.to_numeric(cells["COSMIC_ID"], errors="coerce")
    return cells


def load_drugs_with_smiles() -> pd.DataFrame:
    """Load pre-fetched SMILES parquet. Raises if not present."""
    path = PROCESSED / "drugs_with_smiles.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"SMILES file not found: {path}\n"
            "Run: python pathxdrp/data/fetch_smiles.py"
        )
    return pd.read_parquet(path)


def load_expression() -> pd.DataFrame:
    """
    Load and return the DepMap expression matrix indexed by COSMIC_ID.

    Steps:
      1. Read cosmic_to_depmap.csv for COSMIC_ID <-> ModelID mapping.
      2. Load OmicsExpressionProteinCodingGenesTPMLogp1.csv (rows=ModelID).
      3. Strip entrez IDs from column names: "TSPAN6 (7105)" -> "TSPAN6".
      4. Filter to ModelIDs with known COSMIC_IDs; reindex by COSMIC_ID.
      5. Z-score normalise per gene (mean/std across cell lines).

    Returns:
        pd.DataFrame  shape (n_cells, n_genes)
                      index : COSMIC_ID (int)
                      columns : gene symbol (str)
                      dtype : float32

    Raises:
        FileNotFoundError if the expression CSV or the mapping CSV are absent.
        RuntimeError     if there is zero overlap between the two.
    """
    expr_path = ROOT / "data" / "raw" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    mapping_path = PROCESSED / "cosmic_to_depmap.csv"

    if not expr_path.exists():
        raise FileNotFoundError(
            f"Expression matrix not found:\n  {expr_path}\n"
            "Run: python scripts/download_expression.py"
        )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"COSMIC->DepMap mapping not found:\n  {mapping_path}\n"
            "This file is produced during the initial data fetch pipeline."
        )

    # -- COSMIC_ID <-> ModelID mapping --
    # Column may be "COSMICID" (float) or "COSMIC_ID" depending on how the file was created.
    mapping = pd.read_csv(mapping_path)
    cosmic_col = "COSMICID" if "COSMICID" in mapping.columns else "COSMIC_ID"
    mapping = mapping[mapping[cosmic_col].notna()].copy()
    mapping[cosmic_col] = mapping[cosmic_col].astype(int)
    model_to_cosmic: dict[str, int] = dict(zip(mapping["ModelID"], mapping[cosmic_col]))
    print(f"  COSMIC<->DepMap mapping: {len(model_to_cosmic)} cell lines")

    # -- Load expression matrix --
    file_size_mb = expr_path.stat().st_size / (1024 * 1024)
    print(f"  Loading DepMap expression matrix ({file_size_mb:.0f} MB)")
    t0 = time.time()
    expr = pd.read_csv(expr_path, index_col=0)  # index = ModelID
    print(f"  {expr.shape[0]} models x {expr.shape[1]:,} genes loaded in {time.time()-t0:.1f}s")

    # Strip " (entrez_id)" suffix from column names
    expr.columns = pd.Index([c.split(" (")[0].strip() for c in expr.columns])

    # Filter to ModelIDs that have a COSMIC mapping
    valid_ids = [m for m in expr.index if m in model_to_cosmic]
    if not valid_ids:
        raise RuntimeError(
            "No overlap between expression matrix ModelIDs and COSMIC mapping. "
            "Verify that cosmic_to_depmap.csv matches the DepMap release."
        )
    print(f"  Filtering to {len(valid_ids)}/{expr.shape[0]} models with COSMIC mapping")
    expr = expr.loc[valid_ids].copy()

    # Reindex by COSMIC_ID
    expr.index = pd.Index([model_to_cosmic[m] for m in expr.index], name="COSMIC_ID", dtype=int)

    # Z-score per gene across cell lines
    print("  Z-scoring expression matrix per gene")
    t0 = time.time()
    mean = expr.mean(axis=0)
    std = expr.std(axis=0)
    std[std == 0] = 1.0
    expr = (expr - mean) / std
    print(f"  Z-scored in {time.time()-t0:.1f}s")

    print(f"  Expression matrix ready: {expr.shape[0]} cell lines x {expr.shape[1]:,} genes")
    return expr.astype("float32")


# --- Master merge ---

def build_master_df(
    version: str = "GDSC2",
    require_smiles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge IC50 responses, drug SMILES, cell metadata, and expression data.
    Filters to rows with valid SMILES AND expression coverage.

    Returns:
        (master_df, expr_matrix)

        master_df   : pd.DataFrame
            Columns include: DRUG_ID, DRUG_NAME, COSMIC_ID, CELL_LINE_NAME,
            LN_IC50, AUC, SMILES, TARGET, TARGET_PATHWAY,
            tissue_1, tissue_2, cancer_type, msi_status.

        expr_matrix : pd.DataFrame
            Shape (n_cells, n_genes), index=COSMIC_ID, columns=gene_symbol.
            Float32, Z-scored log2(TPM+1).
    """
    print("Loading response IC50 file")
    response    = load_response(version)
    print(f"  Response: {len(response):,} rows, "
          f"{response['DRUG_ID'].nunique()} drugs, "
          f"{response['COSMIC_ID'].nunique()} cells")

    print("Loading cell line metadata")
    cells       = load_cell_metadata()

    print("Loading drug SMILES")
    drugs       = load_drugs_with_smiles()
    print(f"  Drugs with SMILES: {len(drugs)}")

    print("Loading expression matrix")
    expr_matrix = load_expression()

    # Merge response + drug annotations
    df = response.merge(
        drugs[["DRUG_ID", "SMILES", "TARGET", "TARGET_PATHWAY"]],
        on="DRUG_ID",
        how="left",
    )
    # Merge cell metadata
    df = df.merge(
        cells[["COSMIC_ID", "tissue_1", "tissue_2", "cancer_type", "msi_status"]],
        on="COSMIC_ID",
        how="left",
    )

    # Filter: SMILES
    if require_smiles:
        before = len(df)
        df = df[df["SMILES"].notna()].copy()
        print(f"  Kept {len(df):,}/{before:,} rows with valid SMILES")

    # Filter: expression coverage
    cells_with_expr = set(expr_matrix.index)
    before = len(df)
    df = df[df["COSMIC_ID"].isin(cells_with_expr)].copy()
    print(
        f"  Kept {len(df):,}/{before:,} rows with expression data "
        f"({df['COSMIC_ID'].nunique()} cell lines)",
    )

    # Reset to 0-based positional index so iloc[k] == loc[k] everywhere.
    # Splits are built with np.arange(len(df)) (positional), so this is required
    # for df.iloc[split_idx] to be consistent with any df.loc[] access.
    df = df.reset_index(drop=True)

    print(
        f"Master DF: {len(df):,} rows | "
        f"{df['DRUG_ID'].nunique()} drugs | "
        f"{df['COSMIC_ID'].nunique()} cells",
    )
    return df, expr_matrix


# --- Utilities ---

def tissue_vocab(df: pd.DataFrame) -> dict[str, int]:
    """Build tissue_2 -> integer index vocabulary."""
    tissues = sorted(df["tissue_2"].dropna().unique())
    return {t: i for i, t in enumerate(tissues)}
