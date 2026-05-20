"""
Build KEGG pathway -> gene-symbol map for PathwaySetEncoder.

Steps
-----
1. Read gene symbols from the DepMap expression matrix header (fast — nrows=0).
2. Build entrez_id -> HGNC gene symbol from gene_identifiers_20241212.csv.
3. Fetch all human KEGG pathways via REST API (list + bulk gene-link endpoint).
4. Intersect KEGG entrez IDs with expression-matrix genes.
5. Save data/processed/pathway_gene_map.json
   Format: { "Pathway name": ["GENE1", "GENE2", ...], ... }

The gene symbols in the JSON are the same strings that appear as columns in the
expression matrix (after stripping the " (entrez_id)" suffix).

In train.py, load the JSON and convert to gene indices:
  gene_to_idx = {g: i for i, g in enumerate(expr_matrix.columns)}
  pathway_gene_map = {
      pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
      for pw, genes in pathway_gene_symbols.items()
  }

Requires: download_expression.py must have been run first.
Usage: python scripts/build_pathway_mask.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

EXPR_FILE = ROOT / "data" / "raw" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
GENE_ID_FILE = ROOT / "gene_identifiers_20241212.csv"
OUT_FILE = PROCESSED / "pathway_gene_map.json"

KEGG_BASE = "https://rest.kegg.jp"
REQUEST_DELAY = 0.4  # seconds between KEGG requests (stay well under 3 req/s limit)


# --- Helpers ---

def _get(url: str, desc: str) -> str:
    """GET with retry and polite delay."""
    time.sleep(REQUEST_DELAY)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "pathxdrp/1.0"})
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            print(f"  [{desc}] attempt {attempt + 1}/3 failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts.")


def get_kegg_pathway_names() -> dict[str, str]:
    """Return {pathway_id: clean_pathway_name} for all human KEGG pathways."""
    print("Fetching KEGG pathway list")
    text = _get(f"{KEGG_BASE}/list/pathway/hsa", "pathway list")
    pathways: dict[str, str] = {}
    for line in text.strip().split("\n"):
        if "\t" not in line:
            continue
        pid, name = line.split("\t", 1)
        clean = name.split(" - Homo sapiens")[0].strip()
        pathways[pid.replace("path:", "")] = clean
    print(f"  {len(pathways)} human KEGG pathways found.")
    return pathways


def get_kegg_gene_pathway_links() -> dict[str, list[int]]:
    """
    Return {pathway_id: [entrez_ids]} using bulk link endpoint.
    Each line of the response is: hsa:{entrez_id}\tpath:{pathway_id}
    """
    print("Fetching KEGG gene-pathway links (may take ~30 s)")
    text = _get(f"{KEGG_BASE}/link/pathway/hsa", "gene-pathway link")
    pathway_genes: dict[str, list[int]] = defaultdict(list)
    n_pairs = 0
    for line in text.strip().split("\n"):
        if "\t" not in line:
            continue
        gene_part, path_part = line.strip().split("\t", 1)
        try:
            entrez_id = int(gene_part.replace("hsa:", ""))
        except ValueError:
            continue
        pathway_id = path_part.replace("path:", "")
        pathway_genes[pathway_id].append(entrez_id)
        n_pairs += 1
    print(f"  Parsed {n_pairs:,} gene-pathway pairs across {len(pathway_genes)} pathways.")
    return dict(pathway_genes)


def build_entrez_to_symbol(expr_gene_set: set[str]) -> dict[int, str]:
    """
    Build {entrez_id (int): hgnc_symbol} using COSMIC gene_identifiers file.
    Only includes symbols that appear in the expression matrix.
    """
    if not GENE_ID_FILE.exists():
        raise FileNotFoundError(
            f"Gene identifiers file not found: {GENE_ID_FILE}\n"
            "This file should have been downloaded from COSMIC census."
        )
    print("Building entrez -> gene symbol map")
    df = pd.read_csv(GENE_ID_FILE, low_memory=False)

    result: dict[int, str] = {}
    for _, row in df.iterrows():
        eid_raw = row.get("entrez_id")
        # Prefer hgnc_symbol (matches DepMap column names), fall back to cosmic_gene_symbol
        sym = row.get("hgnc_symbol") or row.get("cosmic_gene_symbol")
        if pd.isna(eid_raw) or pd.isna(sym):
            continue
        sym = str(sym).strip()
        if not sym or sym not in expr_gene_set:
            continue
        try:
            result[int(float(eid_raw))] = sym
        except (ValueError, TypeError):
            pass

    print(f"  Mapped {len(result):,} entrez IDs to expression-matrix gene symbols.")
    return result


def load_expr_gene_symbols() -> list[str]:
    """Read only the header of the expression CSV to get gene symbols."""
    if not EXPR_FILE.exists():
        raise FileNotFoundError(
            f"Expression matrix not found: {EXPR_FILE}\n"
            "Run: python scripts/download_expression.py"
        )
    print("Reading expression matrix columns")
    header_df = pd.read_csv(EXPR_FILE, nrows=0, index_col=0)
    symbols = [c.split(" (")[0].strip() for c in header_df.columns]
    print(f"  {len(symbols):,} genes in expression matrix.")
    return symbols


# --- Main ---

def build_pathway_gene_map() -> None:
    # 1. Gene symbols from expression matrix
    expr_symbols = load_expr_gene_symbols()
    expr_gene_set = set(expr_symbols)

    # 2. entrez_id -> gene_symbol (filtered to expr genes)
    entrez_to_sym = build_entrez_to_symbol(expr_gene_set)

    # 3. KEGG data
    pathway_names = get_kegg_pathway_names()   # {hsa00010: "Glycolysis..."}
    pathway_entrez = get_kegg_gene_pathway_links()  # {hsa00010: [entrez_ids]}

    # 4. Build pathway -> [gene_symbols] intersected with expression matrix
    pathway_gene_map: dict[str, list[str]] = {}
    n_total_pairs = 0
    n_skipped = 0

    for pid, entrez_ids in pathway_entrez.items():
        if pid not in pathway_names:
            continue
        name = pathway_names[pid]

        # Map entrez IDs to gene symbols, deduplicate (preserve order)
        seen: set[str] = set()
        symbols: list[str] = []
        for eid in entrez_ids:
            sym = entrez_to_sym.get(eid)
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)

        if symbols:
            pathway_gene_map[name] = symbols
            n_total_pairs += len(symbols)
        else:
            n_skipped += 1

    # 5. Report
    gene_coverage = len({g for genes in pathway_gene_map.values() for g in genes})
    print("\n=== Pathway gene map summary ===")
    print(f"  {len(pathway_gene_map)} pathways with >= 1 expr-matrix gene")
    print(f"  {n_skipped} pathways skipped (zero overlap)")
    print(f"  {n_total_pairs:,} total (pathway, gene) pairs")
    print(f"  {gene_coverage:,} unique genes covered ({100 * gene_coverage / len(expr_symbols):.1f}% of expr matrix)")

    # 6. Save
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pathway_gene_map, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_FILE}")
    print("\nNext: python -m pathxdrp.train --split random --seed 0 --fold 0")


if __name__ == "__main__":
    build_pathway_gene_map()
