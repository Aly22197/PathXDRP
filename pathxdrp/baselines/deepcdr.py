"""
DeepCDR baseline (Liu et al. 2020, Bioinformatics).

Added during the Knowledge-Based Systems revision in response to Reviewer #3
(point 4) and Reviewer #5 (point 7), who asked for more recent baselines. DeepCDR
was already cited in our introduction, so its absence from the comparison was
conspicuous.

Architecture
------------
Drug:  Uniform Graph Convolutional Network (UGCN) over the molecular graph --
       three GCNConv layers with batch norm, then global max pool. The original
       work uses a "uniform" graph convolution over a padded adjacency matrix;
       GCNConv on the sparse PyG batch is the same operator without the padding.
Cell:  The original DeepCDR consumes three omics channels (mutation, expression,
       methylation), each through its own subnetwork, concatenated before
       fusion. Our benchmark provides expression only, so this is the
       expression-only variant of DeepCDR. The expression branch follows the
       paper: a two-layer MLP (256 -> 100) with tanh activation and batch norm.
Fusion: concatenate [drug || cell], then the paper's 1-D CNN tower over the
       fused vector (Conv1d 150/5 -> maxpool -> Conv1d 5/10 -> maxpool ->
       Conv1d 5/5 -> maxpool), flatten, dropout, dense -> scalar.
Loss:  MSE.

Deviation from the published model, stated plainly
--------------------------------------------------
Only the expression channel is used, because mutation and methylation matrices
are not part of this benchmark and adding them for one model would break the
controlled comparison. This is the single-omics DeepCDR configuration; it is
expected to score below the full multi-omics model reported in the original
paper, and we do not present it as a reproduction of that number.

Reference
---------
Liu Q, Hu Z, Jiang R, Zhou M. "DeepCDR: a hybrid graph convolutional network for
predicting cancer drug response." Bioinformatics 36(Suppl_2):i911-i918, 2020.
DOI: 10.1093/bioinformatics/btaa822
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_max_pool


class UGCNDrugEncoder(nn.Module):
    """Three-layer graph convolution with batch norm, global max pool."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_dim = node_in_dim
        for _ in range(n_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.drop = nn.Dropout(dropout)

    def forward(self, data: Batch, batch_idx: torch.Tensor) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)
        return global_max_pool(x, batch_idx)


class ExpressionEncoder(nn.Module):
    """DeepCDR's expression subnetwork: Dense(256) -> Dense(100), tanh."""

    def __init__(self, n_genes: int, out_dim: int = 100, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.Tanh(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        return self.net(expr)


class DeepCDR(nn.Module):
    """DeepCDR (expression-only variant).

    Interface matches the other baselines so it runs through the same loaders,
    splits, fold-wise normalisation and metric pipeline:
        out = model(drug_batch=batch, expr=expr, y=y)
    """

    def __init__(
        self,
        node_in_dim: int,
        n_genes: int,
        hidden_dim: int = 128,
        n_gcn_layers: int = 3,
        dropout: float = 0.1,
        **_,
    ) -> None:
        super().__init__()
        self.drug_enc = UGCNDrugEncoder(
            node_in_dim, hidden_dim=hidden_dim,
            n_layers=n_gcn_layers, dropout=dropout,
        )
        self.cell_enc = ExpressionEncoder(n_genes, out_dim=100, dropout=dropout)

        fused_dim = hidden_dim + 100

        # The paper's 1-D convolutional tower over the fused representation.
        self.conv_tower = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(150, 5, kernel_size=10, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(5, 5, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(3),
        )
        with torch.no_grad():
            flat = self.conv_tower(torch.zeros(1, 1, fused_dim)).numel()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        drug_batch: Batch,
        expr: torch.Tensor,
        y: torch.Tensor | None = None,
        **_,
    ) -> dict:
        h_drug = self.drug_enc(drug_batch, drug_batch.batch)   # (B, H)
        h_cell = self.cell_enc(expr)                            # (B, 100)
        z = torch.cat([h_drug, h_cell], dim=-1).unsqueeze(1)    # (B, 1, H+100)
        z = self.conv_tower(z).flatten(1)                       # (B, flat)
        pred = self.head(z).squeeze(-1)                         # (B,)

        out: dict = {"pred": pred}
        if y is not None:
            out["loss"] = F.mse_loss(pred, y)
        return out
