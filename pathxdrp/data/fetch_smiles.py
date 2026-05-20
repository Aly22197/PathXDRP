"""
Fetch canonical SMILES for all GDSC drugs from PubChem REST API.

Strategy:
  1. Search PubChem by drug name (fast path).
  2. If not found, try first synonym from Compounds-annotation.csv.
  3. Store result in data/processed/drugs_with_smiles.parquet.

Usage:
  python -m pathxdrp.data.fetch_smiles          # fetch all
  python -m pathxdrp.data.fetch_smiles --dry-run # just show what would be fetched
"""
import argparse
import io
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Force UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# PubChem REST API — note: response key is "SMILES" (not "CanonicalSMILES") as of 2025
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,IsomericSMILES/JSON"
SLEEP_BETWEEN = 0.22  # PubChem rate limit: 5 requests/sec


def fetch_smiles_for_name(name: str, session: requests.Session) -> Optional[str]:
    """Return SMILES from PubChem, or None on failure.

    PubChem API returns the key as 'SMILES', 'IsomericSMILES', or 'CanonicalSMILES'
    depending on the version — we check all possible key names.
    """
    safe = requests.utils.quote(name)
    url = PUBCHEM_URL.format(name=safe)
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            # Try all known SMILES key variants returned by PubChem
            for key in ("IsomericSMILES", "CanonicalSMILES", "SMILES", "ConnectivitySMILES"):
                if key in props and props[key]:
                    return props[key]
    except Exception as e:
        pass
    return None


def fetch_all(dry_run: bool = False) -> pd.DataFrame:
    cpd_path = ROOT / "Compounds-annotation.csv"
    cpd = pd.read_csv(cpd_path)

    out_path = PROCESSED / "drugs_with_smiles.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        fetched_ids = set(existing.loc[existing.SMILES.notna(), "DRUG_ID"])
        missing = cpd[~cpd.DRUG_ID.isin(fetched_ids)]
        print(f"Resuming: {len(fetched_ids)} already fetched, {len(missing)} remaining.")
        cpd_to_fetch = missing
        result_rows = existing.to_dict("records")
    else:
        cpd_to_fetch = cpd
        result_rows = []

    if dry_run:
        print(f"Dry run — would fetch {len(cpd_to_fetch)} drugs.")
        return cpd

    session = requests.Session()
    session.headers.update({"User-Agent": "PathXDRP-research/0.1 (mailto:user@example.com)"})

    for _, row in cpd_to_fetch.iterrows():
        drug_name = row["DRUG_NAME"]
        smiles = fetch_smiles_for_name(drug_name, session)

        # Fallback: try first synonym
        if smiles is None and pd.notna(row.get("SYNONYMS", None)):
            first_syn = str(row["SYNONYMS"]).split(",")[0].strip()
            if first_syn and first_syn != drug_name:
                smiles = fetch_smiles_for_name(first_syn, session)

        status = "OK" if smiles else "MISSING"
        print(f"  [{status}] {drug_name}: {smiles[:60] if smiles else '—'}")

        result_rows.append({
            "DRUG_ID": row["DRUG_ID"],
            "DRUG_NAME": drug_name,
            "TARGET": row.get("TARGET"),
            "TARGET_PATHWAY": row.get("TARGET_PATHWAY"),
            "SMILES": smiles,
        })
        time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(result_rows)
    df.to_parquet(out_path, index=False)

    total = len(df)
    found = df.SMILES.notna().sum()
    print(f"\nDone: {found}/{total} drugs have SMILES ({100*found/total:.1f}%)")
    print(f"Saved to: {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fetch_all(dry_run=args.dry_run)
