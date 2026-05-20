"""
Quantitative XAI Benchmark (Contribution C4).

For each of the curated MoA drugs, runs six attribution methods and scores them
against ground-truth known targets.

Methods implemented:
  1. attention_rollout  — from cross-attention weights (free, architectural)
  2. integrated_gradients — via Captum IntegratedGradients on atom features
  3. gnnexplainer       — mask optimisation (via PyG GNNExplainer)
  4. pgexplainer        — parametric amortised explainer (via PyG PGExplainer)

Scoring metrics:
  - target_auroc        — AUROC of gene attribution vs. known target genes
  - pathway_hitatk      — top-k pathway overlap with known KEGG pathway
  - faithfulness_suff   — delta-prediction on removing low-attribution atoms
  - faithfulness_comp   — delta-prediction on removing high-attribution atoms
  - stability           — cosine sim of maps across 5 seeds
  - sparsity            — entropy of attribution distribution

Usage:
  from pathxdrp.explain.benchmark import ExplainBenchmark
  bench = ExplainBenchmark(model, moa_json="data/processed/moa_benchmark.json")
  results = bench.run(drug_name="Erlotinib", cell_ids=[...])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

ROOT = Path(__file__).parent.parent.parent


class ExplainBenchmark:
    def __init__(
        self,
        model: nn.Module,
        moa_json: str | Path = ROOT / "data" / "processed" / "moa_benchmark.json",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.moa_data = {}
        moa_path = Path(moa_json)
        if moa_path.exists():
            with open(moa_path) as f:
                self.moa_data = json.load(f)

    # ------------------------------------------------------------------
    # Method 1: Cross-attention rollout (architectural, free)
    # ------------------------------------------------------------------

    def attention_rollout(self, drug_batch: Batch, **cell_kwargs) -> torch.Tensor:
        """Returns (N_atoms, N_pathways) attention map from last forward pass."""
        self.model.eval()
        with torch.no_grad():
            self.model(drug_batch.to(self.device), **cell_kwargs)
        return self.model.cross_attn._last_attn.cpu()

    # ------------------------------------------------------------------
    # Method 2: Integrated Gradients (manual midpoint Riemann sum)
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        drug_batch: Batch,
        n_steps: int = 20,
        **cell_kwargs,
    ) -> torch.Tensor:
        """Returns (N_atoms, node_dim) atom-feature IG attributions.

        Captum's IntegratedGradients cannot be used here because it internally
        stacks ``n_steps`` copies of ``drug_batch.x`` into a single tensor of
        shape ``(n_steps * N_atoms, node_dim)`` and feeds that back through
        ``forward_fn``. Our model uses ``drug_batch.batch`` (length ``N_atoms``)
        to map atoms to molecules, so the stacked input desynchronises with
        the batch index and the GAT layer raises a shape-mismatch.

        We instead loop manually over the integration path and call the model
        once per alpha, preserving the original ``(N_atoms, ...)`` shape.

        Baseline: zero atom features. Uses ``(i + 0.5) / n_steps`` midpoint
        alphas so we never sample exactly ``alpha=0`` (which can produce
        NaN gradients through normalisation layers).
        """
        drug_batch = drug_batch.to(self.device)
        self.model.eval()

        orig_x = drug_batch.x.detach().clone()
        baseline_x = torch.zeros_like(orig_x)
        alphas = (
            torch.arange(n_steps, device=orig_x.device, dtype=orig_x.dtype) + 0.5
        ) / n_steps

        total_grads = torch.zeros_like(orig_x)
        for a in alphas:
            x = baseline_x + a * (orig_x - baseline_x)
            x = x.detach().requires_grad_(True)
            drug_batch.x = x
            out = self.model(drug_batch, **cell_kwargs)
            pred = out["pred"]["pred"].sum()
            grads = torch.autograd.grad(
                pred, x, retain_graph=False, create_graph=False
            )[0]
            total_grads = total_grads + grads.detach()

        # Restore so subsequent calls (faithfulness, attention rollout) see the
        # original features rather than the final interpolated state.
        drug_batch.x = orig_x

        avg_grads = total_grads / n_steps
        ig = (orig_x - baseline_x) * avg_grads
        return ig.detach().cpu()  # (N_atoms, node_dim)

    # ------------------------------------------------------------------
    # Method 3: GNNExplainer (PyG built-in)
    # ------------------------------------------------------------------

    def gnnexplainer(self, drug_batch: Batch, **cell_kwargs) -> torch.Tensor:
        """
        Returns (N_atoms,) node importance scores from GNNExplainer.
        Requires PyG >= 2.5 with GNNExplainer.

        Note: GNNExplainer is per-graph. For batch inputs, runs on the
        first graph only (use a loop over drugs in practice).
        """
        try:
            from torch_geometric.explain import Explainer, GNNExplainer
        except ImportError:
            raise ImportError("Upgrade PyG: pip install torch-geometric>=2.5")

        single_graph = drug_batch.get_example(0)
        single_graph = single_graph.to(self.device)

        def model_wrapper(x, edge_index, **kwargs):
            single_graph.x = x
            single_graph.edge_index = edge_index
            out = self.model(Batch.from_data_list([single_graph]), **cell_kwargs)
            return out["pred"]["pred"]

        explainer = Explainer(
            model=model_wrapper,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="regression", task_level="graph", return_type="raw"),
        )
        explanation = explainer(single_graph.x, single_graph.edge_index)
        return explanation.node_mask.sum(dim=-1).cpu()  # (N_atoms,)

    # ------------------------------------------------------------------
    # Scoring metrics
    # ------------------------------------------------------------------

    def target_auroc(
        self,
        gene_attributions: np.ndarray,  # (N_genes,) importance scores
        known_target_genes: list[str],
        all_gene_names: list[str],
    ) -> float:
        """AUROC: rank genes by attribution, label=1 for known targets."""
        from sklearn.metrics import roc_auc_score

        target_set = set(known_target_genes)
        labels = np.array([1 if g in target_set else 0 for g in all_gene_names])
        if labels.sum() == 0 or labels.sum() == len(labels):
            return float("nan")
        return float(roc_auc_score(labels, gene_attributions))

    def pathway_hitatk(
        self,
        pathway_attributions: np.ndarray,  # (N_pathways,) importance
        known_pathway: str,
        pathway_names: list[str],
        k: int = 5,
    ) -> float:
        """Fraction of top-k pathways that include the known pathway."""
        top_k_idx = np.argsort(pathway_attributions)[::-1][:k]
        top_k_names = [pathway_names[i] for i in top_k_idx]
        return float(known_pathway in top_k_names)

    def faithfulness_sufficiency(
        self,
        model: nn.Module,
        drug_batch: Batch,
        atom_scores: np.ndarray,
        frac: float = 0.2,
        **cell_kwargs,
    ) -> float:
        """Delta-prediction when masking LOW-attribution atoms (small = faithful)."""
        model.eval()
        with torch.no_grad():
            base_pred = model(drug_batch.to(self.device), **cell_kwargs)["pred"]["pred"].mean().item()

        n = len(atom_scores)
        low_idx = np.argsort(atom_scores)[:int(n * frac)]
        drug_batch_perturbed = drug_batch.clone()
        drug_batch_perturbed.x[low_idx] = 0.0
        with torch.no_grad():
            perturbed_pred = model(drug_batch_perturbed.to(self.device), **cell_kwargs)["pred"]["pred"].mean().item()
        return abs(base_pred - perturbed_pred)

    def sparsity(self, attributions: np.ndarray) -> float:
        """Normalised entropy of attribution distribution (lower = sparser = better)."""
        p = np.abs(attributions)
        p = p / (p.sum() + 1e-9)
        entropy = -(p * np.log(p + 1e-9)).sum()
        max_entropy = np.log(len(p) + 1e-9)
        return float(entropy / max_entropy)

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(
        self,
        drug_name: str,
        drug_batch: Batch,
        gene_attributions: Optional[np.ndarray] = None,
        all_gene_names: Optional[list[str]] = None,
        pathway_names: Optional[list[str]] = None,
        **cell_kwargs,
    ) -> dict[str, Any]:
        moa = self.moa_data.get(drug_name, {})
        results: dict[str, Any] = {"drug": drug_name, "moa": moa}

        # Attention rollout
        attn = self.attention_rollout(drug_batch, **cell_kwargs)
        atom_scores = attn.mean(dim=-1).numpy()  # (N_atoms,)
        results["attention_sparsity"] = self.sparsity(atom_scores)

        if gene_attributions is not None and all_gene_names and moa.get("target_genes"):
            results["attention_target_auroc"] = self.target_auroc(
                gene_attributions, moa["target_genes"], all_gene_names
            )

        if pathway_names and moa.get("target_pathway"):
            pathway_scores = attn.mean(dim=0).numpy()  # (N_pathways,)
            results["attention_pathway_hit5"] = self.pathway_hitatk(
                pathway_scores, moa["target_pathway"], pathway_names, k=5
            )

        return results
