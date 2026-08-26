"""
Build the PRISM external-validation set from the Repurposing 20Q2 release.

Reviewer #4 point 5 asks for calibration on PRISM and Reviewer #5 point 11
asks for a significance analysis of the per-drug PRISM correlation. Both need
per-pair predictions, which need the PRISM response table in the layout
scripts/external_validation.py expects. This builds it.

Which PRISM file, and why
-------------------------
The 20Q2 *secondary* screen is a multi-dose screen with fitted dose-response
curves, so it reports an IC50 per (cell line, compound) that is comparable with
the GDSC2 LN-IC50 the model is trained on. The 24Q2 release is a single-dose
screen at 2.5 uM and reports log-fold-change viability only, so it cannot
supply an IC50 and is not used here.

Input:
    data/PRISM/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv
Output:
    data/external/PRISM_response.csv   (ModelID, DRUG_NAME, LN_IC50, ...)
    data/external/prism_manifest.json  (what was kept and what was dropped)

Usage:
    python scripts/prepare_prism.py
    python scripts/prepare_prism.py --min-r2 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CURVES = ROOT / "data" / "PRISM" / (
    "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")
EXT = ROOT / "data" / "external"
DRUGS = ROOT / "data" / "processed" / "drugs_with_smiles.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-r2", type=float, default=0.0,
                    help="drop curve fits below this R^2 (0 keeps everything)")
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--no-collapse", dest="collapse", action="store_false",
                    default=True,
                    help="keep one row per screen instead of median-collapsing "
                         "duplicate (cell line, compound) measurements")
    ap.add_argument("--out", default=None,
                    help="output filename inside data/external/")
    args = ap.parse_args()

    src = Path(args.curves)
    if not src.exists():
        raise SystemExit(f"PRISM curves not found: {src}")

    cols = ["depmap_id", "ccle_name", "screen_id", "name", "smiles",
            "moa", "target", "ic50", "ec50", "auc", "r2"]
    df = pd.read_csv(src, usecols=cols, low_memory=False)
    n_raw = len(df)

    # A fitted IC50 is the quantity comparable with the training target.
    df = df[df["ic50"].notna() & np.isfinite(df["ic50"])]
    df = df[df["ic50"] > 0]
    n_ic50 = len(df)

    if args.min_r2 > 0:
        df = df[df["r2"].fillna(0) >= args.min_r2]
    n_r2 = len(df)

    # PRISM doses are in micromolar, as is the GDSC2 target, so the model's
    # LN-IC50 scale is the natural log of this column.
    df["LN_IC50"] = np.log(df["ic50"].astype(float))

    # One row per (cell line, compound): the release screens some pairs in more
    # than one screen_id, and a median is the stable summary.
    if args.collapse:
        grp = (df.groupby(["depmap_id", "name"], as_index=False)
                 .agg(LN_IC50=("LN_IC50", "median"),
                      ic50=("ic50", "median"),
                      auc=("auc", "median"),
                      r2=("r2", "median"),
                      ccle_name=("ccle_name", "first"),
                      smiles=("smiles", "first"),
                      moa=("moa", "first"),
                      n_screens=("screen_id", "nunique")))
    else:
        grp = df.assign(n_screens=1)[
            ["depmap_id", "name", "LN_IC50", "ic50", "auc", "r2",
             "ccle_name", "smiles", "moa", "n_screens"]].copy()

    # Restrict to drugs our pipeline knows, matched on name.
    drugs = pd.read_parquet(DRUGS)
    ours = (drugs.assign(key=drugs["DRUG_NAME"].astype(str).str.strip().str.lower())
                 .drop_duplicates("key"))
    grp["key"] = grp["name"].astype(str).str.strip().str.lower()
    merged = grp.merge(ours[["key", "DRUG_NAME", "DRUG_ID", "SMILES"]],
                       on="key", how="inner")

    out = merged.rename(columns={"depmap_id": "ModelID"})[
        ["ModelID", "ccle_name", "DRUG_NAME", "DRUG_ID", "LN_IC50",
         "ic50", "auc", "r2", "n_screens", "SMILES", "moa"]]

    EXT.mkdir(parents=True, exist_ok=True)
    dest = EXT / (args.out or "PRISM_response.csv")
    out.to_csv(dest, index=False)

    manifest = {
        "source_file": src.name,
        "release": "PRISM Repurposing 20Q2 secondary screen (multi-dose)",
        "why_this_file": ("the secondary screen fits dose-response curves and "
                          "reports IC50; the 24Q2 release is single-dose and "
                          "reports log-fold change only"),
        "rows_in_source": int(n_raw),
        "rows_with_fitted_ic50": int(n_ic50),
        "rows_after_r2_filter": int(n_r2),
        "min_r2": args.min_r2,
        "pairs_after_median_collapse": int(len(grp)),
        "pairs_matched_to_our_drugs": int(len(out)),
        "drugs": int(out["DRUG_NAME"].nunique()),
        "cell_lines": int(out["ModelID"].nunique()),
        "target": "LN_IC50 = natural log of the fitted IC50 in micromolar",
    }
    (EXT / "prism_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    for k, v in manifest.items():
        print(f"  {k:28s} {v}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
