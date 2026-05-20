"""
Baseline drug response prediction models.

All baselines share the same data pipeline as PathXDRP:
  - GDSC2 IC50 labels
  - DepMap 24Q4 gene expression (19,193 genes, Z-scored log2 TPM+1)
  - Drug molecular graphs (same featurisation as PathXDRP)

Models implemented
------------------
GraphDRP    Nguyen et al. 2021 — GIN drug encoder + 1D-CNN cell encoder.
DRPreter    Shin et al. 2023   — GATv2 drug + flat Transformer cell (no pathway mask).
CDRScan     Chang et al. 2018  — MLP on Morgan fingerprints + expression.

Each model exposes:
  model.forward(drug_batch, expr, y=None) -> {"pred": (B,), "loss": tensor or None}

Use scripts/train_baseline.py to train any of these.
"""
from pathxdrp.baselines.graphdrp import GraphDRP
from pathxdrp.baselines.drpreter import DRPreter
from pathxdrp.baselines.cdrscan import CDRScan

__all__ = ["GraphDRP", "DRPreter", "CDRScan"]
