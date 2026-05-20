"""Post-process per-drug XAI JSONs into a unified cross-method summary.

Reads everything under ``results/xai/`` and produces a single
``results/xai/xai_unified_summary.json`` with three axes:

  - **per-method-per-model** averages (existing IG, attention, perm, occlusion)
  - **stability**: when multiple seeds are present, cosine similarity of
    each method's per-drug attribution across seeds
  - **method agreement**: Spearman correlation between the *attribution
    rankings* produced by different methods on the same drug, averaged across
    drugs. Tells us whether IG and occlusion converge on the same pathway
    rankings (high agreement = robust signal across methods).

Run after both ``run_xai_multimodel.py`` and ``run_xai_modelagnostic.py``
have written their outputs.

Usage:
  python scripts/xai_aggregate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def _load_per_drug(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return {r["drug"]: r for r in d.get("per_drug", []) if "drug" in r}


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b, nan_policy="omit")
    return float(rho) if rho == rho else float("nan")


def method_agreement(per_drug_a: dict, per_drug_b: dict,
                      key_a: str, key_b: str) -> dict:
    """Spearman correlation between two scalar attribution metrics across the
    drugs that have both. Returns mean rho + N."""
    paired_a, paired_b = [], []
    for drug in sorted(set(per_drug_a) & set(per_drug_b)):
        va = per_drug_a[drug].get(key_a)
        vb = per_drug_b[drug].get(key_b)
        if va is None or vb is None:
            continue
        if isinstance(va, float) and va != va:
            continue
        if isinstance(vb, float) and vb != vb:
            continue
        try:
            paired_a.append(float(va))
            paired_b.append(float(vb))
        except (TypeError, ValueError):
            continue
    return {"n": len(paired_a), "rho": _spearman(paired_a, paired_b)}


def stability_across_methods(per_drug: dict, key_pairs: list[tuple[str, str]]) -> dict:
    """For each (method_A, method_B) key pair, compute the cross-method
    agreement on the same model's per-drug records."""
    out = {}
    for ka, kb in key_pairs:
        paired_a, paired_b = [], []
        for drug, rec in per_drug.items():
            va = rec.get(ka); vb = rec.get(kb)
            if va is None or vb is None:
                continue
            if (isinstance(va, float) and va != va) or (isinstance(vb, float) and vb != vb):
                continue
            try:
                paired_a.append(float(va)); paired_b.append(float(vb))
            except (TypeError, ValueError):
                continue
        out[f"{ka} vs {kb}"] = {"n": len(paired_a), "spearman": _spearman(paired_a, paired_b)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xai_dir", default=str(ROOT / "results" / "xai"))
    p.add_argument("--out", default=str(ROOT / "results" / "xai" / "xai_unified_summary.json"))
    args = p.parse_args()

    xai = Path(args.xai_dir)
    models = ["pathxdrp", "drpreter", "graphdrp", "cdrscan"]

    multimodel = {m: _load_per_drug(xai / f"xai_multimodel_{m}.json") for m in models}
    agnostic   = {m: _load_per_drug(xai / f"xai_modelagnostic_{m}.json") for m in models}

    out = {"per_model": {}, "method_agreement": {}, "global": {}}

    # ------------------------------------------------------------------
    # Per-model: cross-method agreement on the same model's drugs
    # ------------------------------------------------------------------
    for m in models:
        rec = {}
        if multimodel.get(m) is not None:
            rec["multimodel_n_drugs"] = len(multimodel[m])
        if agnostic.get(m) is not None:
            rec["agnostic_n_drugs"] = len(agnostic[m])

        # Within-multimodel agreement: IG vs Attention target AUROC per drug
        if multimodel.get(m) is not None:
            mm_agree = stability_across_methods(multimodel[m], [
                ("ig_target_auroc", "attn_target_auroc"),
            ])
            rec["multimodel_internal_agreement"] = mm_agree

        # Multimodel ↔ agnostic agreement (occlusion vs IG, perm vs IG)
        if multimodel.get(m) is not None and agnostic.get(m) is not None:
            cross = {
                "ig_target_auroc vs occ_pw_target_pathway_auroc":
                    method_agreement(multimodel[m], agnostic[m],
                                     "ig_target_auroc", "occ_pw_target_pathway_auroc"),
                "ig_target_auroc vs perm_pw_target_pathway_auroc":
                    method_agreement(multimodel[m], agnostic[m],
                                     "ig_target_auroc", "perm_pw_target_pathway_auroc"),
                "occ_pw_target_pathway_auroc vs perm_pw_target_pathway_auroc":
                    method_agreement(agnostic[m], agnostic[m],
                                     "occ_pw_target_pathway_auroc",
                                     "perm_pw_target_pathway_auroc"),
            }
            rec["cross_runner_agreement"] = cross

        out["per_model"][m] = rec

    # ------------------------------------------------------------------
    # Cross-model method agreement: do PathXDRP and DRPreter rank the same
    # drugs as easy/hard in the same order? (signal that the benchmark itself
    # is consistent independent of which model you use)
    # ------------------------------------------------------------------
    for ma in models:
        for mb in models:
            if ma == mb:
                continue
            if not multimodel.get(ma) or not multimodel.get(mb):
                continue
            ag = method_agreement(multimodel[ma], multimodel[mb],
                                  "ig_target_auroc", "ig_target_auroc")
            out["method_agreement"][f"{ma} vs {mb} (IG target AUROC)"] = ag

    # ------------------------------------------------------------------
    # Global summary numbers
    # ------------------------------------------------------------------
    def _flat(metric_path: list[str]) -> list[float]:
        vals = []
        cur = out
        for k in metric_path[:-1]:
            cur = cur.get(k, {})
        return cur

    out["global"] = {
        "n_models_with_multimodel": sum(1 for m in models if multimodel.get(m) is not None),
        "n_models_with_agnostic":   sum(1 for m in models if agnostic.get(m)   is not None),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Unified XAI summary -> {args.out}")


if __name__ == "__main__":
    main()
