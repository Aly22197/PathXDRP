"""
Smoke tests: verify PathXDRP and baseline model forward passes on synthetic data.

All PathXDRP tests use expression mode (the only supported mode) with a tiny
synthetic pathway_gene_map so they run fast on CPU.

Run: pytest tests/test_smoke.py -v
"""
import pytest
import torch
from torch_geometric.data import Batch, Data

from pathxdrp.models.cross_attention import PathwayMaskedCrossAttention
from pathxdrp.models.drug_encoder import DrugGATEncoder
from pathxdrp.models.evidential_head import EvidentialRegressionHead, evidential_loss
from pathxdrp.models.pathxdrp import PathXDRP
from pathxdrp.baselines.graphdrp import GraphDRP
from pathxdrp.baselines.drpreter import DRPreter
from pathxdrp.baselines.cdrscan import CDRScan

# Optional Mamba encoders — gated on (mamba_ssm AND CUDA) availability.
# mamba_ssm builds CUDA kernels that only run on NVIDIA GPUs at runtime.
try:
    from pathxdrp.models.graph_mamba_drug import (
        GraphMambaDrugEncoder, MAMBA_AVAILABLE as _MAMBA_DRUG_PKG,
    )
except Exception:
    GraphMambaDrugEncoder = None
    _MAMBA_DRUG_PKG = False
try:
    from pathxdrp.models.mamba_cell import (
        GeneMambaCellEncoder, MAMBA_AVAILABLE as _MAMBA_CELL_PKG,
    )
except Exception:
    GeneMambaCellEncoder = None
    _MAMBA_CELL_PKG = False

_MAMBA_RUNNABLE = torch.cuda.is_available()
_MAMBA_DRUG = _MAMBA_DRUG_PKG and _MAMBA_RUNNABLE
_MAMBA_CELL = _MAMBA_CELL_PKG and _MAMBA_RUNNABLE
MAMBA_SKIP_REASON = "mamba_ssm not installed or no CUDA; skipping Mamba encoder tests"


# ---
# Helpers
# ---

def make_fake_graph(n_atoms: int = 10, node_dim: int = 60, edge_dim: int = 9) -> Data:
    x          = torch.randn(n_atoms, node_dim)
    edge_index = torch.randint(0, n_atoms, (2, n_atoms * 2))
    edge_attr  = torch.randn(n_atoms * 2, edge_dim)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Shared constants
NODE_DIM   = 60
EDGE_DIM   = 9
HIDDEN     = 64
B          = 4
N_PATHWAYS = 10
N_GENES    = 200

# Synthetic pathway_gene_map: 10 pathways, 10 genes each (non-overlapping)
PATHWAY_GENE_MAP = {f"PW{i}": list(range(i * 10, i * 10 + 10)) for i in range(N_PATHWAYS)}


# ---
# Tests
# ---

def test_drug_encoder() -> None:
    enc = DrugGATEncoder(
        node_in_dim=NODE_DIM, edge_in_dim=EDGE_DIM,
        hidden_dim=HIDDEN, n_layers=2, n_heads=4,
    )
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs)
    h_atom, h_mol = enc(batch, batch=batch.batch)

    assert h_atom.shape == (batch.num_nodes, HIDDEN)
    # h_mol = concat(global_mean_pool, global_max_pool) → 2 × hidden_dim
    assert h_mol.shape  == (B, HIDDEN * 2)


def test_cross_attention() -> None:
    attn    = PathwayMaskedCrossAttention(hidden_dim=HIDDEN, n_heads=4, mask_type="none")
    n_atoms = 40
    h_drug  = torch.randn(n_atoms, HIDDEN)
    h_cell  = torch.randn(B, N_PATHWAYS, HIDDEN)
    atom_batch = torch.repeat_interleave(torch.arange(B), n_atoms // B)

    context, weights = attn(h_drug, h_cell, atom_batch)
    assert context.shape == (n_atoms, HIDDEN)
    assert weights.shape == (n_atoms, N_PATHWAYS)


def test_evidential_head() -> None:
    head = EvidentialRegressionHead(in_dim=HIDDEN * 2)
    z    = torch.randn(B, HIDDEN * 2)
    out  = head(z)

    assert out["pred"].shape == (B,)
    assert (out["nu"] > 0).all()
    assert (out["alpha"] > 1).all()

    y    = torch.randn(B)
    loss = evidential_loss(out, y)
    assert loss.item() > 0


def test_pathxdrp_forward() -> None:
    """Full end-to-end forward pass with synthetic data."""
    model = PathXDRP(
        node_in_dim=NODE_DIM,
        edge_in_dim=EDGE_DIM,
        n_genes=N_GENES,
        pathway_gene_map=PATHWAY_GENE_MAP,
        hidden_dim=HIDDEN,
        n_gat_layers=2,
        n_attn_heads=4,
        mask_type="none",
    )
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs)
    expr   = torch.randn(B, N_GENES)
    y      = torch.randn(B)

    out = model(drug_batch=batch, expr=expr, y=y)

    assert "loss" in out
    assert out["loss"].item() > 0
    assert out["pred"]["pred"].shape == (B,)
    assert out["attn_weights"].shape[1] == N_PATHWAYS


def test_pathxdrp_uncertainty() -> None:
    """Check that evidential head produces non-negative uncertainty estimates."""
    model = PathXDRP(
        node_in_dim=NODE_DIM,
        edge_in_dim=EDGE_DIM,
        n_genes=N_GENES,
        pathway_gene_map=PATHWAY_GENE_MAP,
        hidden_dim=HIDDEN,
        n_gat_layers=2,
        n_attn_heads=4,
        mask_type="soft",
    )
    model.eval()
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs)
    expr   = torch.randn(B, N_GENES)

    with torch.no_grad():
        out = model(drug_batch=batch, expr=expr)

    assert (out["pred"]["aleatoric"] >= 0).all(), "Aleatoric uncertainty must be >= 0"
    assert (out["pred"]["epistemic"] >= 0).all(), "Epistemic uncertainty must be >= 0"


# ---
# Baseline smoke tests
# ---

def test_graphdrp() -> None:
    """GraphDRP: GIN drug encoder + 1D-CNN cell encoder."""
    model = GraphDRP(node_in_dim=NODE_DIM, n_genes=N_GENES, hidden_dim=HIDDEN, n_gin_layers=3)
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs)
    expr   = torch.randn(B, N_GENES)
    y      = torch.randn(B)

    out = model(drug_batch=batch, expr=expr, y=y)
    assert out["pred"].shape == (B,)
    assert out["loss"].item() > 0


def test_drpreter() -> None:
    """DRPreter: GATv2 + unmasked pathway cross-attention + MSE."""
    model = DRPreter(
        node_in_dim=NODE_DIM,
        edge_in_dim=EDGE_DIM,
        n_genes=N_GENES,
        pathway_gene_map=PATHWAY_GENE_MAP,
        hidden_dim=HIDDEN,
        n_gat_layers=2,
        n_attn_heads=4,
    )
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs)
    expr   = torch.randn(B, N_GENES)
    y      = torch.randn(B)

    out = model(drug_batch=batch, expr=expr, y=y)
    assert out["pred"].shape == (B,)
    assert out["loss"].item() >= 0


def test_cdrscan() -> None:
    """CDRScan: Morgan fingerprint MLP + expression MLP."""
    model = CDRScan(n_genes=N_GENES, hidden_dim=HIDDEN)
    fp    = torch.randint(0, 2, (B, 2048)).float()
    expr  = torch.randn(B, N_GENES)
    y     = torch.randn(B)

    out = model(fp=fp, expr=expr, y=y)
    assert out["pred"].shape == (B,)
    assert out["loss"].item() >= 0


# ---
# Mamba encoder smoke tests (skipped if mamba_ssm unavailable)
# ---

@pytest.mark.skipif(not _MAMBA_DRUG, reason=MAMBA_SKIP_REASON)
def test_graph_mamba_drug_encoder() -> None:
    """Graph-Mamba drug encoder: GAT + Bi-Mamba over atom sequence."""
    device = torch.device("cuda")
    enc = GraphMambaDrugEncoder(
        node_in_dim=NODE_DIM,
        edge_in_dim=EDGE_DIM,
        hidden_dim=HIDDEN,
        n_gat_layers=2,
        n_mamba_layers=2,
        n_heads=4,
        ordering="degree",
    ).to(device)
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs).to(device)
    h_atom, h_mol = enc(batch, batch=batch.batch)

    assert h_atom.shape == (batch.num_nodes, HIDDEN)
    assert h_mol.shape  == (B, HIDDEN)
    assert h_atom.dtype == torch.float32


@pytest.mark.skipif(not _MAMBA_CELL, reason=MAMBA_SKIP_REASON)
def test_gene_mamba_cell_encoder() -> None:
    """GeneMamba cell encoder: rank-encode genes -> backbone -> pathway pool."""
    device = torch.device("cuda")
    gene_symbols = [f"GENE{i}" for i in range(N_GENES)]
    enc = GeneMambaCellEncoder(
        n_genes=N_GENES,
        gene_symbols=gene_symbols,
        pathway_gene_map=PATHWAY_GENE_MAP,
        hidden_dim=HIDDEN,
        top_k=64,                       # tiny for fast smoke test
        backbone_id="__nonexistent__",  # forces fallback path
        n_layers_fallback=2,
        backbone_dim_fallback=32,
    ).to(device)
    expr = torch.randn(B, N_GENES, device=device)
    out  = enc(expr)
    assert out.shape == (B, N_PATHWAYS, HIDDEN)


@pytest.mark.skipif(not (_MAMBA_DRUG and _MAMBA_CELL), reason=MAMBA_SKIP_REASON)
def test_pathxdrp_with_mamba_encoders() -> None:
    """End-to-end PathXDRP using both Graph-Mamba drug + GeneMamba cell encoders."""
    device = torch.device("cuda")
    gene_symbols = [f"GENE{i}" for i in range(N_GENES)]
    model = PathXDRP(
        node_in_dim=NODE_DIM,
        edge_in_dim=EDGE_DIM,
        n_genes=N_GENES,
        pathway_gene_map=PATHWAY_GENE_MAP,
        hidden_dim=HIDDEN,
        n_attn_heads=4,
        mask_type="none",
        drug_encoder_type="graph_mamba",
        cell_encoder_type="gene_mamba",
        gene_symbols=gene_symbols,
        graph_mamba_kwargs={"n_gat_layers": 1, "n_mamba_layers": 1, "ordering": "degree"},
        gene_mamba_kwargs={
            "top_k":                 64,
            "backbone_id":           "__nonexistent__",
            "n_layers_fallback":     2,
            "backbone_dim_fallback": 32,
        },
    ).to(device)
    graphs = [make_fake_graph(node_dim=NODE_DIM) for _ in range(B)]
    batch  = Batch.from_data_list(graphs).to(device)
    expr   = torch.randn(B, N_GENES, device=device)
    y      = torch.randn(B, device=device)

    out = model(drug_batch=batch, expr=expr, y=y)
    assert "loss" in out
    assert out["loss"].item() > 0
    assert out["pred"]["pred"].shape == (B,)
    assert out["attn_weights"].shape[1] == N_PATHWAYS
