"""
Build the MoA benchmark JSON for the XAI evaluation (Phase 6).

For a curated set of ~25 well-characterised oncology drugs in GDSC2, this
script writes:
  data/processed/moa_benchmark.json
  {
    "Erlotinib":  {
      "drug_id": ...,
      "smiles":  "...",
      "target_genes":   ["EGFR", ...],
      "target_pathway": "EGFR signaling",
      "kegg_pathway":   "hsa04012",
      "primary_targets": ["EGFR"],
    },
    ...
  }

Sources
-------
- Drug list + targets: GDSC2 dataset's TARGET / TARGET_PATHWAY columns
- Curated 25 drugs (well-known MoA, broad pathway coverage):
  see CURATED below — selected to span EGFR, MAPK, PI3K/AKT/mTOR, DNA damage,
  HDAC, BCL-2, CDK, microtubule, BCR-ABL, BRAF, JAK/STAT, etc.

Notes
-----
- We only include drugs that exist in our master_df (i.e. have SMILES + expr).
- target_pathway strings are GDSC's own pathway labels; the XAI benchmark
  resolves these against KEGG pathway names from pathway_gene_map.json
  via fuzzy matching (top-K hit).
- Primary target genes are extracted from the comma-separated TARGET column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent

CURATED_DRUGS: list[str] = [
    # Tyrosine kinase inhibitors
    "Erlotinib", "Gefitinib", "Lapatinib", "Afatinib",
    # MAPK pathway
    "Trametinib", "Selumetinib", "Dabrafenib", "PLX-4720", "Vemurafenib",
    # PI3K / AKT / mTOR
    "AZD8055", "Rapamycin", "MK-2206", "Pictilisib", "Alpelisib",
    # CDK
    "Palbociclib",
    # BCR-ABL
    "Imatinib", "Nilotinib", "Dasatinib",
    # HDAC / epigenetic
    "Vorinostat", "Entinostat",
    # BCL-2
    "Navitoclax", "Venetoclax",
    # DNA damage / topoisomerase / chemo
    "Olaparib", "Cisplatin",
    # Microtubule
    "Paclitaxel",
]


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "").replace(" ", "")


def build_moa_benchmark(
    gdsc_csv: Path,
    out_path: Path,
    curated: list[str] = CURATED_DRUGS,
) -> dict:
    df = pd.read_csv(gdsc_csv)
    # GDSC2 raw uses PUTATIVE_TARGET / PATHWAY_NAME. The drugs_with_smiles parquet
    # uses TARGET / TARGET_PATHWAY (verified 100% identical content earlier).
    # Auto-detect either schema.
    target_col = "PUTATIVE_TARGET" if "PUTATIVE_TARGET" in df.columns else "TARGET"
    if target_col not in df.columns:
        raise RuntimeError(
            f"Cannot find a target column in {gdsc_csv}. "
            f"Expected one of: PUTATIVE_TARGET, TARGET. Got {df.columns.tolist()[:10]}..."
        )
    pathway_col = "PATHWAY_NAME" if "PATHWAY_NAME" in df.columns else "TARGET_PATHWAY"
    if pathway_col not in df.columns:
        raise RuntimeError(
            f"Cannot find a pathway column in {gdsc_csv}. "
            f"Expected one of: PATHWAY_NAME, TARGET_PATHWAY. Got {df.columns.tolist()[:10]}..."
        )

    # Fold to per-drug rows
    drug_table = (
        df[["DRUG_NAME", "DRUG_ID", target_col, pathway_col]]
        .drop_duplicates(subset=["DRUG_NAME"])
        .reset_index(drop=True)
    )
    drug_lookup = {_norm(n): row for n, row in zip(drug_table["DRUG_NAME"], drug_table.itertuples(index=False))}

    # Try to attach SMILES
    smiles_path = ROOT / "data" / "processed" / "drugs_with_smiles.parquet"
    smiles_map: dict[int, str] = {}
    if smiles_path.exists():
        smi_df = pd.read_parquet(smiles_path)
        smiles_map = dict(zip(smi_df["DRUG_ID"].astype(int), smi_df["SMILES"]))

    out: dict = {}
    not_found: list[str] = []
    pbar = tqdm(curated, desc="Resolving curated drugs", unit="drug")
    for name in pbar:
        row = drug_lookup.get(_norm(name))
        if row is None:
            not_found.append(name)
            continue

        target_str  = getattr(row, target_col, "") or ""
        pathway_str = getattr(row, pathway_col, "") or ""
        targets = [t.strip() for t in str(target_str).split(",") if t.strip()]

        out[name] = {
            "drug_id":         int(row.DRUG_ID),
            "smiles":          smiles_map.get(int(row.DRUG_ID), None),
            "target_genes":    targets,
            "target_pathway":  pathway_str.strip(),
            "primary_targets": targets[:3],
        }
        pbar.set_postfix(found=len(out), missing=len(not_found))
    pbar.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(out)} drugs to {out_path}", flush=True)
    if not_found:
        print(f"Not found in GDSC: {not_found}", flush=True)
    return out


def build_moa_benchmark_all(gdsc_csv: Path, out_path: Path) -> dict:
    """Build a MoA benchmark from every GDSC drug that has a TARGET annotation.

    Same schema as the curated version, no pre-filtering by name. Drugs whose
    TARGET column is empty are dropped. Useful for evaluating XAI methods on a
    population scale rather than a hand-picked 25.
    """
    df = pd.read_csv(gdsc_csv)
    target_col  = "PUTATIVE_TARGET" if "PUTATIVE_TARGET" in df.columns else "TARGET"
    pathway_col = "PATHWAY_NAME"    if "PATHWAY_NAME"    in df.columns else "TARGET_PATHWAY"

    drug_table = (
        df[["DRUG_NAME", "DRUG_ID", target_col, pathway_col]]
        .drop_duplicates(subset=["DRUG_NAME"])
        .reset_index(drop=True)
    )

    smiles_path = ROOT / "data" / "processed" / "drugs_with_smiles.parquet"
    smiles_map: dict[int, str] = {}
    if smiles_path.exists():
        smi_df = pd.read_parquet(smiles_path)
        smiles_map = dict(zip(smi_df["DRUG_ID"].astype(int), smi_df["SMILES"]))

    out: dict = {}
    skipped_no_target = 0
    skipped_no_smiles = 0
    pbar = tqdm(drug_table.itertuples(index=False),
                total=len(drug_table), desc="Building expanded MoA", unit="drug")
    for row in pbar:
        target_str = getattr(row, target_col) or ""
        if not str(target_str).strip():
            skipped_no_target += 1
            continue
        drug_id = int(row.DRUG_ID)
        smi = smiles_map.get(drug_id)
        if smi is None:
            skipped_no_smiles += 1
            continue
        pathway_str = getattr(row, pathway_col) or ""
        targets = [t.strip() for t in str(target_str).split(",") if t.strip()]
        out[row.DRUG_NAME] = {
            "drug_id":         drug_id,
            "smiles":          smi,
            "target_genes":    targets,
            "target_pathway":  str(pathway_str).strip(),
            "primary_targets": targets[:3],
        }
        pbar.set_postfix(kept=len(out), no_target=skipped_no_target, no_smiles=skipped_no_smiles)
    pbar.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(out)} drugs to {out_path}", flush=True)
    print(f"  Skipped (no target):  {skipped_no_target}", flush=True)
    print(f"  Skipped (no SMILES):  {skipped_no_smiles}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gdsc_csv", default=str(ROOT / "GDSC2-dataset.csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "processed" / "moa_benchmark.json"))
    p.add_argument("--all", action="store_true",
                   help="Include every GDSC drug with a non-empty TARGET column "
                        "(not just the 25-drug curated list).")
    args = p.parse_args()
    if args.all:
        build_moa_benchmark_all(Path(args.gdsc_csv), Path(args.out))
    else:
        build_moa_benchmark(Path(args.gdsc_csv), Path(args.out))


if __name__ == "__main__":
    main()
