"""
GraphDRP baseline (Nguyen et al. 2021, Bioinformatics).

Drug:  Graph Isomorphism Network (GINConv, 5 layers) -> global mean pool.
Cell:  Three-layer 1-D convolutional network over gene expression,
       followed by adaptive max-pool and a projector MLP.
Head:  Concat [drug || cell] -> 3-layer MLP -> IC50 scalar.
Loss:  MSE (standard; no evidential uncertainty).

Reference
---------
Nguyen T-T et al. "GraphDRP: predicting drug response in cancer cell lines using graph
convolutional networks." Bioinformatics, 2021.
DOI: 10.1093/bioinformatics/btab100

Differences from PathXDRP
--------------------------
- No KEGG pathway structure or masking.
- No cross-attention mechanism.
- No calibrated uncertainty.
- MSE loss instead of evidential NIG-NLL.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool


# ---
# Drug encoder: GIN
# ---

class GINDrugEncoder(nn.Module):
    """
    5-layer GIN encoder for molecular graphs.
    Each layer: GINConv(MLP) -> BatchNorm -> ReLU.
    Output: (B, hidden_dim) global mean-pooled graph embedding.
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.drop  = nn.Dropout(dropout)

        in_dim = node_in_dim
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

    def forward(self, drug_batch: Batch, batch_idx: torch.Tensor) -> torch.Tensor:
        x = drug_batch.x
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, drug_batch.edge_index)))
            x = self.drop(x)
        return global_mean_pool(x, batch_idx)   # (B, hidden_dim)


# ---
# Cell encoder: 1D CNN over expression
# ---

class ExprCNN(nn.Module):
    """
    Three-layer 1-D convolutional encoder for gene expression profiles.

    Treats the expression vector as a single-channel 1-D signal of length n_genes,
    applies strided convolutions to compress it, then projects to hidden_dim.

    Architecture for n_genes ~ 19 193:
      Conv1d(1 -> 32, k=8, s=4)  ->  shape ~4 800
      Conv1d(32 -> 64, k=8, s=4) ->  shape ~1 200
      Conv1d(64 -> 128, k=8, s=4)->  shape ~  298
      AdaptiveMaxPool1d(64)       ->  shape =   64
      flatten                     ->  128 * 64 = 8 192
      Linear(8192 -> hidden_dim)
    """

    def __init__(self, n_genes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(64),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 64, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        """expr: (B, n_genes)  ->  (B, hidden_dim)"""
        x = expr.unsqueeze(1)    # (B, 1, n_genes)
        x = self.conv(x)         # (B, 128, 64)
        return self.proj(x)      # (B, hidden_dim)


# ---
# Full model
# ---

class GraphDRP(nn.Module):
    """
    GraphDRP: GIN drug encoder + 1D-CNN cell encoder + MLP head.

    Compatible with the GDSCDataset / collate_fn from pathxdrp.train:
      out = model(drug_batch=batch, expr=expr, y=y)
      loss = out["loss"]
      pred = out["pred"]   # (B,) raw IC50 scalar
    """

    def __init__(
        self,
        node_in_dim: int,
        n_genes: int,
        hidden_dim: int = 128,
        n_gin_layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_enc = GINDrugEncoder(node_in_dim, hidden_dim, n_gin_layers, dropout)
        self.cell_enc = ExprCNN(n_genes, hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        drug_batch: Batch,
        expr: torch.Tensor,
        y: torch.Tensor | None = None,
        **_,
    ) -> dict:
        h_drug = self.drug_enc(drug_batch, drug_batch.batch)  # (B, D)
        h_cell = self.cell_enc(expr)                           # (B, D)
        z    = torch.cat([h_drug, h_cell], dim=-1)             # (B, 2D)
        pred = self.head(z).squeeze(-1)                        # (B,)

        out: dict = {"pred": pred}
        if y is not None:
            out["loss"] = F.mse_loss(pred, y)
        return out
