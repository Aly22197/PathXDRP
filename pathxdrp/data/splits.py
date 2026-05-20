"""
Five deterministic split builders with frozen hashes.

All splits write train/val/test index files to data/processed/splits/<name>/<seed>/.
Re-running is idempotent if hashes match.

Usage:
  from pathxdrp.data.splits import build_all_splits
  build_all_splits(df, seeds=[0,1,2,3,4])
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).parent.parent.parent
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

SplitName = Literal["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]

N_FOLDS = 5


def _save_fold(fold_dir: Path, train_idx, val_idx, test_idx) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "train.npy", np.array(train_idx))
    np.save(fold_dir / "val.npy", np.array(val_idx))
    np.save(fold_dir / "test.npy", np.array(test_idx))


def _df_hash(df: pd.DataFrame) -> str:
    key = str(sorted(df.columns.tolist())) + str(len(df))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def random_split(df: pd.DataFrame, seed: int = 0) -> list[dict]:
    n = len(df)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in kf.split(np.arange(n)):
        rng = np.random.default_rng(seed)
        rng.shuffle(te)
        val_size = len(te) // 2
        folds.append({"train": tr, "val": te[:val_size], "test": te[val_size:]})
    return folds


def cell_blind_split(df: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Hold out entire cell lines (COSMIC_ID) from training."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["COSMIC_ID"].values
    folds = []
    for tr_idx, te_idx in gkf.split(np.arange(len(df)), groups=groups):
        rng = np.random.default_rng(seed)
        rng.shuffle(te_idx)
        val_size = len(te_idx) // 2
        folds.append({"train": tr_idx, "val": te_idx[:val_size], "test": te_idx[val_size:]})
    return folds


def drug_blind_split(df: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Hold out entire drugs (DRUG_ID) from training."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["DRUG_ID"].values
    folds = []
    for tr_idx, te_idx in gkf.split(np.arange(len(df)), groups=groups):
        rng = np.random.default_rng(seed)
        rng.shuffle(te_idx)
        val_size = len(te_idx) // 2
        folds.append({"train": tr_idx, "val": te_idx[:val_size], "test": te_idx[val_size:]})
    return folds


def scaffold_blind_split(df: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Bemis-Murcko scaffold split — hardest drug generalisation test."""
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(df["SMILES"].values):
        mol = Chem.MolFromSmiles(smi) if pd.notna(smi) else None
        if mol is None:
            scaffold = "__invalid__"
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = "__error__"
        scaffolds.setdefault(scaffold, []).append(i)

    scaffold_list = sorted(scaffolds.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_list)

    all_idx = np.arange(len(df))
    n_test_scaffolds = max(1, len(scaffold_list) // N_FOLDS)

    folds = []
    for fold in range(N_FOLDS):
        test_scaffolds = set(scaffold_list[fold * n_test_scaffolds: (fold + 1) * n_test_scaffolds])
        te_idx = np.array([i for s, idxs in scaffolds.items() if s in test_scaffolds for i in idxs])
        tr_idx = np.setdiff1d(all_idx, te_idx)
        rng2 = np.random.default_rng(seed + fold)
        rng2.shuffle(te_idx)
        val_size = len(te_idx) // 2
        folds.append({"train": tr_idx, "val": te_idx[:val_size], "test": te_idx[val_size:]})
    return folds


def tissue_blind_split(df: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Leave-one-cancer-type-out."""
    tissues = df["tissue_2"].fillna("unknown").values
    unique_tissues = sorted(set(tissues))
    all_idx = np.arange(len(df))
    # Use the N_FOLDS most-represented tissues as test folds
    tissue_counts = df["tissue_2"].value_counts()
    top_tissues = tissue_counts.index[:N_FOLDS].tolist()

    folds = []
    for tissue in top_tissues:
        te_mask = tissues == tissue
        te_idx = all_idx[te_mask]
        tr_idx = all_idx[~te_mask]
        rng = np.random.default_rng(seed)
        rng.shuffle(te_idx)
        val_size = len(te_idx) // 2
        folds.append({"train": tr_idx, "val": te_idx[:val_size], "test": te_idx[val_size:]})
    return folds


def build_all_splits(df: pd.DataFrame, seeds: list[int] | None = None) -> None:
    """Build and persist all five split regimes for each seed."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    df_hash = _df_hash(df)
    meta = {"df_hash": df_hash, "n_rows": len(df), "seeds": seeds, "n_folds": N_FOLDS}

    builders = {
        "random": random_split,
        "cell_blind": cell_blind_split,
        "drug_blind": drug_blind_split,
        "scaffold_blind": scaffold_blind_split,
        "tissue_blind": tissue_blind_split,
    }

    from tqdm.auto import tqdm
    jobs = [(name, builder, seed)
            for name, builder in builders.items()
            for seed in seeds]
    pbar = tqdm(jobs, desc="Building splits", unit="job")
    for name, builder, seed in pbar:
        pbar.set_postfix(current=f"{name}/seed{seed}")
        split_root = SPLITS_DIR / name / f"seed{seed}"
        lock = split_root / "meta.json"
        if lock.exists():
            saved = json.loads(lock.read_text())
            if saved.get("df_hash") == df_hash:
                tqdm.write(f"  {name}/seed{seed}: cached (hash match)")
                continue

        folds = builder(df, seed=seed)
        for fold_i, fold in enumerate(folds):
            fold_dir = split_root / f"fold{fold_i}"
            _save_fold(fold_dir, fold["train"], fold["val"], fold["test"])

        meta_out = {**meta, "name": name, "seed": seed}
        split_root.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(meta_out, indent=2))
        tqdm.write(f"  {name}/seed{seed}: built {len(folds)} folds")
    pbar.close()

    print(f"\nAll splits written to {SPLITS_DIR}", flush=True)


def load_split(
    name: SplitName,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_dir = SPLITS_DIR / name / f"seed{seed}" / f"fold{fold}"
    train = np.load(fold_dir / "train.npy")
    val = np.load(fold_dir / "val.npy")
    test = np.load(fold_dir / "test.npy")
    return train, val, test
