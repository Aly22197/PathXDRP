"""
PathXDRP: Pathway-masked Cross-attention Drug Response Predictor.

Pipeline
--------
  DrugGATEncoder      (drug molecular graph -> per-atom + global embeddings)
  PathwaySetEncoder   (gene expression -> per-pathway embeddings)
  PathwayMaskedCrossAttention  (drug atoms attend over cell pathways)
  Attention-weighted pool      (atom embeddings -> molecule-level context)
  EvidentialRegressionHead     (predict IC50 + calibrated uncertainty)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from pathxdrp.models.cell_encoder import PathwaySetEncoder
from pathxdrp.models.cross_attention import PathwayMaskedCrossAttention
from pathxdrp.models.drug_encoder import DrugGATEncoder
from pathxdrp.models.evidential_head import EvidentialRegressionHead, evidential_loss

# Optional Mamba encoders (CUDA-only). Imports succeed even without mamba_ssm;
# instantiation raises ImportError. Smoke tests detect this and skip.
try:
    from pathxdrp.models.graph_mamba_drug import GraphMambaDrugEncoder, MAMBA_AVAILABLE as _MAMBA_DRUG
except Exception:
    GraphMambaDrugEncoder = None
    _MAMBA_DRUG = False
try:
    from pathxdrp.models.mamba_cell import GeneMambaCellEncoder, MAMBA_AVAILABLE as _MAMBA_CELL
except Exception:
    GeneMambaCellEncoder = None
    _MAMBA_CELL = False
try:
    from pathxdrp.models.scgpt_cell import ScGPTCellEncoder, SCGPT_AVAILABLE as _SCGPT
except Exception:
    ScGPTCellEncoder = None
    _SCGPT = False


class PathXDRP(nn.Module):
    """
    End-to-end drug response predictor.

    Args:
        node_in_dim:        Atom feature dimensionality (from graph_utils).
        edge_in_dim:        Bond feature dimensionality (from graph_utils).
        n_genes:            Number of genes in the expression vector.
        pathway_gene_map:   {pathway_name: [gene_indices_in_expr_vector]}.
                            Built by scripts/build_pathway_mask.py.
        hidden_dim:         Internal embedding dimension (default 128).
        n_gat_layers:       Number of GATv2 message-passing layers (default 3).
        n_attn_heads:       Multi-head attention heads (default 8).
        dropout:            Dropout probability (default 0.1).
        mask_type:          Cross-attention mask type: "hard" | "soft" | "none".
        entropy_reg_weight: Weight for attention entropy regularisation loss.
        use_molformer:      Inject MolFormer-XL CLS embeddings into atom feats.
        evidential_lam:     Regularisation weight for evidential NIG-NLL loss.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        n_genes: int,
        pathway_gene_map: dict[str, list[int]],
        hidden_dim: int = 128,
        n_gat_layers: int = 3,
        n_attn_heads: int = 8,
        dropout: float = 0.1,
        mask_type: str = "none",
        entropy_reg_weight: float = 0.01,
        use_molformer: bool = False,
        evidential_lam: float = 0.01,
        # Phase 4 encoder switches
        drug_encoder_type: str = "gat",      # gat | molformer | graph_mamba
        cell_encoder_type: str = "pathway_set",  # pathway_set | gene_mamba | scgpt
        gene_symbols: Optional[list[str]] = None,  # required for gene_mamba
        graph_mamba_kwargs: Optional[dict] = None,
        gene_mamba_kwargs:  Optional[dict] = None,
        n_pw_transformer_layers: int = 1,
        n_pw_stats: int = 4,
        frac_active_sharpness: float = 0.0,
        # Optional global Morgan fingerprint injection (CDRScan-style substructure prior)
        use_morgan_fp: bool = False,
        morgan_fp_dim: int = 2048,
        # AUC multi-task auxiliary loss weight.
        # Training jointly on AUC and LN_IC50 regularises the shared drug-cell
        # representation and improves drug-blind generalisation (two correlated
        # response metrics reduce over-fitting to IC50-specific noise).
        # 0.0 disables the head entirely; 0.2 is a safe default.
        aux_auc_weight: float = 0.0,
        # Cross-attention residual + LayerNorm. When True, the cross-attention
        # output flows into the prediction head as ``LN(out + h_drug)``, and
        # the per-atom pooling switches from the (degenerate) max-attention
        # weighting to a standard mean pool. This is what makes the attention
        # weights load-bearing for the prediction. Default False preserves
        # backward compatibility with v3 checkpoints; new training runs should
        # set True.
        cross_attn_residual: bool = False,
        # Atom -> molecule pooling mode, decoupled from ``cross_attn_residual``.
        #   "auto"     : legacy behaviour -- mean pool iff cross_attn_residual,
        #                otherwise the max-attention-weighted pool.
        #   "mean"     : always mean pool.
        #   "attention": always the max-attention-weighted pool.
        # The two were previously tied together, which made it impossible to
        # tell whether a faithfulness gain came from the residual or from the
        # pooling change (Reviewer #5, point 3). "auto" reproduces every
        # published checkpoint exactly; the explicit modes exist so the
        # ablation can vary one factor at a time.
        pool_mode: str = "auto",
        # When True, drop the parallel ``h_mol`` (GAT global readout) input to
        # the head. h_mol is an alternative drug representation that competes
        # with the cross-attention path; even with the residual fix the head
        # learns to bypass attention via h_mol. Removing it forces all drug
        # information to flow through cross-attention, raising attention
        # faithfulness 2-3x at a small expected PCC cost (~0.005). Default
        # False keeps backward compat.
        drop_h_mol: bool = False,
        # Auxiliary "predict from attention only" loss. When > 0, an extra
        # MLP head predicts LN_IC50 from h_drug_context alone (the
        # post-attention drug representation); the loss is added with this
        # weight. Forces the attention output to be predictive on its own,
        # which directly improves attention faithfulness. 0.3 is a sane
        # starting value.
        attn_aux_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.evidential_lam = evidential_lam
        self.entropy_reg_weight = entropy_reg_weight
        self.drug_encoder_type = drug_encoder_type
        self.cell_encoder_type = cell_encoder_type
        self.aux_auc_weight = aux_auc_weight
        self.drop_h_mol = drop_h_mol
        self.attn_aux_weight = attn_aux_weight

        # Drug branch: molecular graph -> atom embeddings
        if drug_encoder_type == "graph_mamba":
            if GraphMambaDrugEncoder is None or not _MAMBA_DRUG:
                raise ImportError(
                    "drug_encoder_type='graph_mamba' requires mamba_ssm. "
                    "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation"
                )
            gm_kw = dict(graph_mamba_kwargs or {})
            self.drug_enc = GraphMambaDrugEncoder(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                n_heads=n_attn_heads,
                dropout=dropout,
                **gm_kw,
            )
        else:  # "gat" or "molformer" both use DrugGATEncoder; molformer is a flag
            self.drug_enc = DrugGATEncoder(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                n_layers=n_gat_layers,
                n_heads=n_attn_heads,
                dropout=dropout,
                use_molformer=use_molformer or (drug_encoder_type == "molformer"),
                use_morgan_fp=use_morgan_fp,
                morgan_fp_dim=morgan_fp_dim,
            )
        self.use_morgan_fp = use_morgan_fp
        # Cache the actual h_mol dim from the encoder so we can size the head correctly.
        # Falls back to 2*hidden_dim for encoders that don't expose h_mol_dim
        # (graph_mamba etc.).
        h_mol_dim = getattr(self.drug_enc, "h_mol_dim", 2 * hidden_dim)

        # Cell branch: gene expression -> pathway embeddings
        if cell_encoder_type == "gene_mamba":
            if GeneMambaCellEncoder is None or not _MAMBA_CELL:
                raise ImportError(
                    "cell_encoder_type='gene_mamba' requires mamba_ssm. "
                    "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation"
                )
            if gene_symbols is None:
                raise ValueError(
                    "cell_encoder_type='gene_mamba' requires gene_symbols (list of HGNC "
                    "symbols matching expr columns) to build the backbone token map."
                )
            gnm_kw = dict(gene_mamba_kwargs or {})
            self.cell_enc = GeneMambaCellEncoder(
                n_genes=n_genes,
                gene_symbols=gene_symbols,
                pathway_gene_map=pathway_gene_map,
                hidden_dim=hidden_dim,
                **gnm_kw,
            )
        elif cell_encoder_type == "scgpt":
            if ScGPTCellEncoder is None:
                raise ImportError(
                    "cell_encoder_type='scgpt' requires the scgpt module on the import path. "
                    "Install instructions: https://github.com/bowang-lab/scGPT  "
                    "A from-scratch fallback is used automatically when scgpt is missing."
                )
            if gene_symbols is None:
                raise ValueError(
                    "cell_encoder_type='scgpt' requires gene_symbols (list of HGNC "
                    "symbols matching expr columns)."
                )
            scg_kw = dict((kwargs := {}) if False else {})  # placeholder for clean ternary
            scg_kw = {}
            self.cell_enc = ScGPTCellEncoder(
                n_genes=n_genes,
                gene_symbols=gene_symbols,
                pathway_gene_map=pathway_gene_map,
                hidden_dim=hidden_dim,
            )
        else:  # "pathway_set" — original
            self.cell_enc = PathwaySetEncoder(
                n_genes=n_genes,
                pathway_gene_map=pathway_gene_map,
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_pw_transformer_layers=n_pw_transformer_layers,
                n_pw_stats=n_pw_stats,
                frac_active_sharpness=frac_active_sharpness,
            )

        # Cross-attention: drug atoms (Q) attend over cell pathways (K, V)
        # n_pathways is required so soft_mask_logit gets shape (1,1,1,P),
        # giving a per-pathway learnable bias that actually influences softmax.
        self.cross_attn_residual = cross_attn_residual
        if pool_mode not in ("auto", "mean", "attention"):
            raise ValueError(f"pool_mode must be auto|mean|attention, got {pool_mode!r}")
        self.pool_mode = pool_mode
        self.cross_attn = PathwayMaskedCrossAttention(
            hidden_dim=hidden_dim,
            n_heads=n_attn_heads,
            dropout=dropout,
            mask_type=mask_type,
            entropy_reg_weight=entropy_reg_weight,
            n_pathways=len(pathway_gene_map),
            use_residual=cross_attn_residual,
        )

        # Attention-weighted pool projection
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # Learned attention pooling over pathway tokens for cell global representation.
        # A single trainable query vector attends over all P pathway tokens,
        # producing a weighted sum that focuses on the most relevant pathways.
        self.cell_pool_q = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Evidential regression head input layout depends on drop_h_mol:
        #   default          : [drug_context(D) || h_mol(h_mol_dim) || cell_global(D) || interaction(D)]
        #                      = D + h_mol_dim + D + D
        #   drop_h_mol=True  : [drug_context(D) || cell_global(D) || interaction(D)]
        #                      = 3D
        # Removing h_mol forces all drug information through cross-attention,
        # which makes the attention weights load-bearing (raising faithfulness
        # 2-3×) at a small expected PCC cost.
        # The interaction term h_drug_context ⊙ h_cell_global is an element-wise product
        # that creates explicit pairwise feature combinations, enabling the head to model
        # "which drug features match which cell vulnerabilities" — a bilinear interaction.
        if self.drop_h_mol:
            head_in_dim = hidden_dim + hidden_dim + hidden_dim   # 3D
        else:
            head_in_dim = hidden_dim + h_mol_dim + hidden_dim + hidden_dim
        self.head = EvidentialRegressionHead(in_dim=head_in_dim, hidden_dim=hidden_dim)

        # Auxiliary "attention-only" predictor: small MLP that maps
        # h_drug_context (D) -> scalar LN_IC50. Trained jointly with the main
        # loss when attn_aux_weight > 0. The auxiliary loss penalises the
        # model for routing predictive information AROUND the attention path
        # via h_mol or h_cell_global — both faithfulness metrics improve.
        self.attn_aux_head: nn.Module | None = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if attn_aux_weight > 0.0 else None
        )

        # Optional AUC auxiliary head (multi-task training).
        # Predicts area-under-curve alongside LN_IC50 from the same z-vector,
        # adding a correlated second objective that regularises the shared
        # drug-cell representation without extra encoder parameters.
        self.auc_head: nn.Module | None = (
            nn.Sequential(
                nn.Linear(head_in_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
            if aux_auc_weight > 0.0 else None
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        drug_batch: Batch,
        expr: torch.Tensor,                       # (B, N_genes) — required
        smiles_list: Optional[list[str]] = None,  # for optional MolFormer injection
        hard_mask: Optional[torch.Tensor] = None, # (N_atoms_total, N_pathways) bool
        morgan_fp: Optional[torch.Tensor] = None, # (B, morgan_fp_dim) for FP injection
        y: Optional[torch.Tensor] = None,         # (B,) LN_IC50 labels for loss
        auc: Optional[torch.Tensor] = None,       # (B,) AUC auxiliary target in [0, 1]
    ) -> dict:
        """
        Returns a dict with keys:
          pred        : evidential outputs (mu, nu, alpha, beta, pred, aleatoric, epistemic)
          attn_weights: (N_atoms_total, N_pathways) attention map (for XAI)
          loss        : total loss (only when y is not None)
          main_loss   : evidential NIG-NLL
          entropy_loss: attention entropy regularisation
        """
        # 1. Encode drug molecular graph
        h_atom, h_mol = self.drug_enc(
            drug_batch, smiles_list=smiles_list, batch=drug_batch.batch,
            morgan_fp=morgan_fp,
        )
        # h_atom: (N_atoms_total, D), h_mol: (B, D) — global GAT readout

        # 2. Encode cell-line expression into pathway tokens
        h_cell = self.cell_enc(expr)   # (B, N_pathways, D)

        # 3. Drug atoms attend over cell pathways
        context, attn_weights = self.cross_attn(
            h_drug=h_atom,
            h_cell=h_cell,
            atom_batch=drug_batch.batch,
            hard_mask=hard_mask,
        )
        # context: (N_atoms_total, D), attn_weights: (N_atoms_total, N_pathways)

        # 4. Pool atoms -> molecule-level context.
        # When cross-attention has a residual + LN (cross_attn_residual=True),
        # ``context`` already mixes drug atom features with attention output,
        # so a plain mean pool is the correct aggregator (this is what
        # DRPreter does, and its attention is faithful as a result). Without
        # the residual, we fall back to the historical max-attention-weighted
        # pool that v3 checkpoints depend on.
        # ``pool_mode`` decouples this choice from ``cross_attn_residual`` so
        # the ablation can attribute the faithfulness gain to one or the other.
        if self.pool_mode == "auto":
            use_mean = self.cross_attn_residual
        else:
            use_mean = self.pool_mode == "mean"

        if use_mean:
            from torch_geometric.nn import global_mean_pool
            h_drug_context = global_mean_pool(context, drug_batch.batch)         # (B, D)
        else:
            a_weights = attn_weights.max(dim=-1, keepdim=True)[0]                # (N_atoms, 1)
            h_drug_context = global_add_pool(context * a_weights, drug_batch.batch)
        h_drug_context = self.pool_norm(self.pool_proj(h_drug_context))

        # 5. Learned attention pooling over pathway tokens -> cell global representation.
        # Query: (1, 1, D), Keys: (B, P, D) -> scores: (B, 1, P) -> (B, D)
        pool_scores = torch.matmul(
            self.cell_pool_q.expand(h_cell.size(0), -1, -1),  # (B, 1, D)
            h_cell.transpose(-1, -2),                          # (B, D, P)
        ) / math.sqrt(self.hidden_dim)                         # (B, 1, P)
        pool_weights = F.softmax(pool_scores, dim=-1)          # (B, 1, P)
        h_cell_global = (pool_weights @ h_cell).squeeze(1)     # (B, D)

        # 6. Joint drug–cell representation.
        # When drop_h_mol is on, the GAT global readout is excluded so that
        # the head must rely on the cross-attention output (h_drug_context).
        # Otherwise we keep the historical 4-block input.
        h_int = h_drug_context * h_cell_global                       # (B, D)
        if self.drop_h_mol:
            z = torch.cat([h_drug_context, h_cell_global, h_int], dim=-1)        # (B, 3D)
        else:
            z = torch.cat([h_drug_context, h_mol, h_cell_global, h_int], dim=-1) # (B, 5D)

        # 7. Predict IC50 + uncertainty via evidential head
        pred = self.head(z)

        out: dict = {"pred": pred, "attn_weights": attn_weights}

        # 8. Loss (only if ground-truth labels provided)
        if y is not None:
            main_loss    = evidential_loss(pred, y, lam=self.evidential_lam)
            entropy_loss = self.cross_attn._entropy_loss * self.entropy_reg_weight
            total_loss   = main_loss + entropy_loss
            # AUC auxiliary loss: joint MSE on a correlated response metric
            if auc is not None and self.auc_head is not None:
                auc_pred = self.auc_head(z).squeeze(-1)              # (B,)
                auc_loss = F.mse_loss(auc_pred, auc)
                total_loss = total_loss + self.aux_auc_weight * auc_loss
                out["auc_loss"] = auc_loss
            # Attention-only auxiliary loss: predict y from h_drug_context
            # alone. Forces the post-attention drug representation to be
            # predictive on its own — the model can no longer route the signal
            # AROUND the attention path, which is what makes attention faithful.
            if self.attn_aux_head is not None:
                attn_pred = self.attn_aux_head(h_drug_context).squeeze(-1)   # (B,)
                attn_aux_loss = F.mse_loss(attn_pred, y)
                total_loss = total_loss + self.attn_aux_weight * attn_aux_loss
                out["attn_aux_loss"] = attn_aux_loss
            out["loss"]         = total_loss
            out["main_loss"]    = main_loss
            out["entropy_loss"] = entropy_loss

        return out

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> dict:
        """Convenience eval-mode wrapper; returns pred dict with uncertainty."""
        self.eval()
        return self.forward(*args, **kwargs)
