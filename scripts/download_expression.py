"""
Download the DepMap 24Q4 gene expression matrix via Figshare public API.

Source article: https://figshare.com/articles/dataset/27993248
Target file:    OmicsExpressionProteinCodingGenesTPMLogp1.csv  (~507 MB)
Download URL:   https://ndownloader.figshare.com/files/51065489

Values: log2(TPM + 1) for protein-coding genes, one row per DepMap ModelID (ACH-XXXXXX).

Usage: python scripts/download_expression.py
"""
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"

# Direct Figshare download URL (DepMap 24Q4, article 27993248, file 51065489)
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/51065489"
EXPECTED_SIZE_MB = 507  # approximate — used for sanity check only


def _file_is_complete(path: Path, min_mb: int = 400) -> bool:
    """True if the file exists and is at least min_mb in size."""
    return path.exists() and path.stat().st_size >= min_mb * 1024 * 1024


def download() -> None:
    if _file_is_complete(OUT_FILE):
        size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
        print(f"Already downloaded ({size_mb:.0f} MB): {OUT_FILE}")
        print("Delete the file and re-run to force re-download.")
        return

    print(f"Downloading DepMap expression matrix ({EXPECTED_SIZE_MB} MB):")
    print(f"  URL: {DOWNLOAD_URL}")
    print(f"  Destination: {OUT_FILE}\n")

    headers = {"User-Agent": "pathxdrp/1.0 (research)"}
    r = requests.get(DOWNLOAD_URL, stream=True, timeout=300, headers=headers)
    r.raise_for_status()

    total_bytes = int(r.headers.get("content-length", 0))

    tmp_file = OUT_FILE.with_suffix(".tmp")
    with (
        open(tmp_file, "wb") as fout,
        tqdm(total=total_bytes or None, unit="B", unit_scale=True, desc="Downloading") as pbar,
    ):
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            fout.write(chunk)
            pbar.update(len(chunk))

    # Atomic rename
    tmp_file.rename(OUT_FILE)
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nSaved ({size_mb:.0f} MB) -> {OUT_FILE}")
    print("\nNext step: python scripts/build_pathway_mask.py")


if __name__ == "__main__":
    download()
