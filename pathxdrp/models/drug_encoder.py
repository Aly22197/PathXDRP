"""
Drug encoder: GATv2 over molecular graph → per-atom embeddings.

Optionally concatenates a frozen MolFormer-XL CLS token projected to `hidden_dim`.
Output: (N_atoms, hidden_dim) — passed as Query to cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class DrugGATEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_molformer: bool = False,
        molformer_dim: int = 768,
        use_morgan_fp: bool = False,
        morgan_fp_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_molformer = use_molformer
        self.use_morgan_fp = use_morgan_fp
        # h_mol dimension at output: 2*hidden_dim from mean+max pool, +hidden_dim
        # if a global Morgan fingerprint projection is concatenated.
        self.h_mol_dim = 2 * hidden_dim + (hidden_dim if use_morgan_fp else 0)

        # Input projection
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    edge_dim=edge_in_dim,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Global Morgan fingerprint projection (optional).
        # When enabled, adds an explicit substructure prior on top of the GAT's
        # learned features. Same encoding as CDRScan: 2048-bit radius-2 Morgan FP.
        if use_morgan_fp:
            self.fp_proj = nn.Sequential(
                nn.Linear(morgan_fp_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )

        # MolFormer integration (optional)
        if use_molformer:
            self.molformer_proj = nn.Sequential(
                nn.Linear(molformer_dim, hidden_dim),
                nn.GELU(),
            )
            self._molformer = self._load_molformer()

    def _load_molformer(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
            )
            for p in model.parameters():
                p.requires_grad_(False)
            return (tokenizer, model)
        except Exception as e:
            print(f"MolFormer load failed ({e}), disabling.")
            self.use_molformer = False
            return None

    @torch.no_grad()
    def _encode_molformer(self, smiles_list: list[str]) -> torch.Tensor:
        tokenizer, model = self._molformer
        enc = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=202)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        out = model(**enc)
        return out.pooler_output  # (B, 768)

    def forward(
        self,
        data: Data,
        smiles_list: list[str] | None = None,
        batch: torch.Tensor | None = None,
        morgan_fp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: PyG batched molecular graphs.
            smiles_list: optional SMILES list for MolFormer injection.
            batch: explicit batch index (defaults to data.batch).
            morgan_fp: optional (B, morgan_fp_dim) tensor of 2048-bit Morgan FPs.
                       Required if use_morgan_fp=True; ignored otherwise.

        Returns:
            h_atom: (N_atoms_total, hidden_dim)  per-atom embeddings
            h_mol:  (B, h_mol_dim)  graph-level embeddings
                    h_mol_dim = 2*hidden_dim   if use_morgan_fp=False
                              = 3*hidden_dim   if use_morgan_fp=True
        """
        x = self.act(self.node_proj(data.x))

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + self.dropout(x_new))
            x = self.act(x)

        if batch is None:
            batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_mol = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=-1)  # (B, 2 * hidden_dim) — mean captures average, max captures peak activations

        # Concatenate the projected global Morgan fingerprint, if provided.
        # This adds an explicit substructure prior (the same one CDRScan exploits)
        # on top of the learned GAT features.
        if self.use_morgan_fp:
            if morgan_fp is None:
                raise ValueError(
                    "use_morgan_fp=True but no morgan_fp tensor was passed to forward(). "
                    "Pass morgan_fp through the data loader / collate_fn."
                )
            fp_emb = self.fp_proj(morgan_fp.to(h_mol.dtype))   # (B, hidden_dim)
            h_mol = torch.cat([h_mol, fp_emb], dim=-1)         # (B, 3 * hidden_dim)

        # Inject MolFormer CLS embedding (broadcast to all atoms of each molecule)
        if self.use_molformer and smiles_list is not None:
            mf_emb = self._encode_molformer(smiles_list).to(x.device)  # (B, hidden_dim)
            mf_emb = self.molformer_proj(mf_emb)  # (B, hidden_dim)
            atom_mf = mf_emb[batch]  # (N_atoms, hidden_dim)
            x = x + atom_mf

        return x, h_mol
