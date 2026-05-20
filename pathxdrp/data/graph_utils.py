"""
Build PyTorch Geometric Data objects from SMILES.

Node features per atom (334-dim total):
  - Atomic number (one-hot, 43 elements + other = 44)
  - Degree (0–10 = 11)
  - Formal charge (-3 to 3 bucketed = 7)
  - Hybridisation (sp, sp2, sp3, sp3d, sp3d2, other = 6)
  - Aromaticity (bool = 1)
  - In-ring (bool = 1)
  - Chirality (R, S, unspecified = 3)
  - Num Hs (0–4 = 5)
  - Morgan FG identifier (one-hot top-256 Morgan-radius-2 fragments = 256)

Edge features per bond (10-dim total):
  - Bond type (single, double, triple, aromatic, other = 5)
  - Conjugated (bool = 1)
  - In-ring (bool = 1)
  - Stereo (E/Z/none = 3)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import warnings

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")  # silence RDKit deprecation + info messages

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
    "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb",
]
_ATOM_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
_N_ATOMS = len(ATOM_TYPES) + 1  # +1 for "other"

HYBRIDISATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_HYBRIDS = {h: i for i, h in enumerate(HYBRIDISATION)}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_IDX = {b: i for i, b in enumerate(BOND_TYPES)}

MORGAN_RADIUS = 2
MORGAN_BITS = 256
# FG vocab is learned/fixed at preprocessing time — see build_fg_vocab()

# Modern Morgan generator instance (created once at module load, thread-safe for reads)
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


def _one_hot(val, vocab: list) -> list[int]:
    vec = [0] * (len(vocab) + 1)
    idx = vocab.index(val) if val in vocab else len(vocab)
    vec[idx] = 1
    return vec


def atom_features(atom: Chem.Atom, fg_vocab: Optional[dict] = None) -> list[int]:
    sym = atom.GetSymbol()
    atom_oh = _one_hot(sym, ATOM_TYPES)

    degree_oh = [0] * 11
    degree_oh[min(atom.GetDegree(), 10)] = 1

    charge = atom.GetFormalCharge()
    charge_bucket = min(max(charge + 3, 0), 6)
    charge_oh = [0] * 7
    charge_oh[charge_bucket] = 1

    hyb = atom.GetHybridization()
    hyb_oh = [0] * (len(HYBRIDISATION) + 1)
    hyb_oh[_HYBRIDS.get(hyb, len(HYBRIDISATION))] = 1

    arom = [int(atom.GetIsAromatic())]
    in_ring = [int(atom.IsInRing())]

    chiral = atom.GetChiralTag()
    chiral_oh = [0, 0, 0]
    if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_oh[0] = 1
    elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_oh[1] = 1
    else:
        chiral_oh[2] = 1

    num_hs = [0] * 5
    num_hs[min(atom.GetTotalNumHs(), 4)] = 1

    feats = atom_oh + degree_oh + charge_oh + hyb_oh + arom + in_ring + chiral_oh + num_hs

    if fg_vocab is not None:
        fg_vec = [0] * len(fg_vocab)
        atom_idx = atom.GetIdx()
        mol = atom.GetOwningMol()
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        _MORGAN_GEN.GetFingerprint(mol, additionalOutput=ao)
        bit_map = ao.GetBitInfoMap()
        for bit, origins in (bit_map.items() if bit_map else []):
            for center, radius in origins:
                if center == atom_idx and radius == MORGAN_RADIUS:
                    if bit in fg_vocab:
                        fg_vec[fg_vocab[bit]] = 1
        feats += fg_vec

    return feats


def bond_features(bond: Chem.Bond) -> list[int]:
    bt = bond.GetBondType()
    bt_oh = [0] * (len(BOND_TYPES) + 1)
    bt_oh[_BOND_IDX.get(bt, len(BOND_TYPES))] = 1

    conj = [int(bond.GetIsConjugated())]
    ring = [int(bond.IsInRing())]

    stereo = bond.GetStereo()
    stereo_oh = [0, 0, 0]
    if stereo == Chem.rdchem.BondStereo.STEREOE:
        stereo_oh[0] = 1
    elif stereo == Chem.rdchem.BondStereo.STEREOZ:
        stereo_oh[1] = 1
    else:
        stereo_oh[2] = 1

    return bt_oh + conj + ring + stereo_oh


def smiles_to_graph(
    smiles: str,
    fg_vocab: Optional[dict] = None,
    label: Optional[float] = None,
) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [atom_features(a, fg_vocab) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edges_src, edges_dst, edge_attr_list = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        for u, v in [(i, j), (j, i)]:
            edges_src.append(u)
            edges_dst.append(v)
            edge_attr_list.append(bf)

    if not edges_src:  # single-atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(bond_features(mol.GetBonds()[0])) if mol.GetNumBonds() else 9), dtype=torch.float)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


def build_fg_vocab(smiles_list: list[str], top_k: int = MORGAN_BITS) -> dict[int, int]:
    """Build a Morgan-bit → index vocabulary from a list of SMILES."""
    from collections import Counter
    counter: Counter = Counter()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=top_k)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        bit_map = ao.GetBitInfoMap()
        if bit_map:
            counter.update(bit_map.keys())
    top = [bit for bit, _ in counter.most_common(top_k)]
    return {bit: idx for idx, bit in enumerate(top)}
