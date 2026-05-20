"""
CDRScan baseline (Chang et al. 2018 / Góis & Gonçalves 2021 revisit).

Drug:  Extended-connectivity (Morgan) fingerprint (radius=2, 2048 bits) -> MLP.
Cell:  Gene expression vector -> MLP.
Head:  Concat [drug_emb || cell_emb] -> 3-layer MLP -> IC50 scalar.
Loss:  MSE.

This is the canonical "fingerprint + expression MLP" baseline that appears
in virtually every DRP comparison table.  It is fast to train (no graph ops)
and serves as a lower-bound reference for graph-based methods.

References
----------
Chang Y et al. "Cancer Drug Response Profile scan (CDRscan): A Deep Learning
Model That Predicts Drug Effectiveness from Cancer Genomic Signature."
Scientific Reports, 2018. DOI: 10.1038/s41598-018-27214-6

Góis A, Gonçalves A. "Predicting drug sensitivity of cancer cell lines using
deep learning with a multi-omics approach." arXiv, 2021.

Differences from PathXDRP
--------------------------
- No molecular graph; uses 2048-bit binary Morgan fingerprint.
- No pathway structure.
- No cross-attention.
- No uncertainty.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048


# ---
# Fingerprint cache builder
# ---

def build_fp_cache(drugs_df) -> dict[int, np.ndarray]:
    """
    Build {DRUG_ID: Morgan fingerprint (np.uint8, 2048 bits)} for every drug
    with a valid SMILES string.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from tqdm.auto import tqdm

    cache: dict[int, np.ndarray] = {}
    n_failed = 0
    pbar = tqdm(
        drugs_df.iterrows(), total=len(drugs_df),
        desc="  Building Morgan fingerprints", unit="drug",
        disable=not sys.stdout.isatty(),
    )
    for _, row in pbar:
        smi = row["SMILES"]
        if not isinstance(smi, str) or not smi.strip():
            n_failed += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_BITS)
        arr = np.zeros(MORGAN_BITS, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        cache[int(row["DRUG_ID"])] = arr
        pbar.set_postfix(ok=len(cache), failed=n_failed)
    pbar.close()
    return cache


# ---
# Dataset  (fingerprint + expression — no graph)
# ---

class CDRScanDataset(Dataset):
    """
    Dataset for CDRScan: returns Morgan fingerprint + expression per IC50 row.
    Same interface as GDSCDataset but substitutes drug fingerprint for graph.
    """

    def __init__(
        self,
        df,
        fp_cache: dict[int, np.ndarray],
        expr_matrix,           # pd.DataFrame indexed by COSMIC_ID
    ) -> None:
        self.df      = df.reset_index(drop=True)
        self.fp_cache = fp_cache
        self.expr_np  = expr_matrix.values.astype("float32")
        self.cosmic_to_row: dict[int, int] = {
            int(cid): i for i, cid in enumerate(expr_matrix.index)
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row       = self.df.iloc[idx]
        drug_id   = int(row["DRUG_ID"])
        cosmic_id = int(row["COSMIC_ID"])
        fp   = self.fp_cache[drug_id]
        expr = self.expr_np[self.cosmic_to_row[cosmic_id]]
        return {
            "fp":       fp,
            "expr":     expr,
            "y":        float(row["LN_IC50"]),
            "drug_id":  drug_id,
            "cosmic_id": cosmic_id,
        }


def cdrscan_collate(batch: list[dict]) -> dict:
    return {
        "fp":   torch.tensor(np.stack([b["fp"]   for b in batch]), dtype=torch.float),
        "expr": torch.tensor(np.stack([b["expr"] for b in batch]), dtype=torch.float),
        "y":    torch.tensor([b["y"] for b in batch], dtype=torch.float),
        "drug_ids":   torch.tensor([b["drug_id"]  for b in batch], dtype=torch.long),
        "cosmic_ids": torch.tensor([b["cosmic_id"] for b in batch], dtype=torch.long),
    }


# ---
# Model
# ---

class CDRScan(nn.Module):
    """
    CDRScan: fingerprint MLP + expression MLP + joint head.

    Forward takes (fp, expr, y=None) — note: no drug_batch.
    When used inside train_baseline.py, the collate function supplies fp.
    """

    def __init__(
        self,
        n_genes: int,
        fp_dim: int = MORGAN_BITS,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_enc = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        fp:   torch.Tensor,               # (B, 2048)
        expr: torch.Tensor,               # (B, n_genes)
        y:    Optional[torch.Tensor] = None,
        **_,
    ) -> dict:
        h_drug = self.drug_enc(fp)        # (B, hidden//2)
        h_cell = self.cell_enc(expr)      # (B, hidden//2)
        z      = torch.cat([h_drug, h_cell], dim=-1)   # (B, hidden)
        pred   = self.head(z).squeeze(-1)              # (B,)

        out: dict = {"pred": pred}
        if y is not None:
            out["loss"] = F.mse_loss(pred, y)
        return out
