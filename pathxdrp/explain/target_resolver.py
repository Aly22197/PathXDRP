"""Drug-target name resolution for the XAI benchmark.

GDSC's TARGET column is a free-text mix of:
  - HGNC gene symbols (`EGFR`, `MTOR`)
  - mutated isoforms with parenthetical suffixes (`IDH1 (R132H)`, `BRAF (V600E)`)
  - protein family names (`PI3Kalpha`, `VEGFR`, `BCR-ABL`)
  - drug-class labels with no gene mapping (`Antimetabolite`, `DNA crosslinker`)
  - drug-class labels that DO map to a known set of genes
    (`DNA methyltransferases` -> DNMT1/3A/3B; `Topoisomerase II` -> TOP2A/2B)

This module exposes a single ``resolve_targets(raw, gene_set)`` function used by
the XAI benchmark scripts to map TARGET strings into a list of HGNC symbols
present in the expression matrix. The previous resolver lived inline in two
script files and dropped ~94/237 MoA drugs from the AUROC computation.
"""
from __future__ import annotations

import re
from typing import Iterable

# ---
# Aliases for protein families / fusions whose name is not a direct HGNC symbol.
# Order matters within a list (first hit per raw target wins downstream).
# ---

_FAMILY_ALIASES: dict[str, list[str]] = {
    # mTOR complexes
    "MTORC":  ["MTOR"],
    "MTORC1": ["MTOR"],
    "MTORC2": ["MTOR"],
    # ERBB / HER family
    "HER1":   ["EGFR"],
    "HER2":   ["ERBB2"],
    "HER3":   ["ERBB3"],
    "HER4":   ["ERBB4"],
    "ERBB":   ["EGFR", "ERBB2", "ERBB3", "ERBB4"],
    # BCR-ABL fusion -> ABL1 (the kinase side)
    "BCRABL": ["ABL1"],
    # VEGFR / PDGFR / FGFR / IGFR families
    "VEGFR":  ["KDR", "FLT1", "FLT4"],
    "VEGFR1": ["FLT1"],
    "VEGFR2": ["KDR"],
    "VEGFR3": ["FLT4"],
    "PDGFR":  ["PDGFRA", "PDGFRB"],
    "FGFR":   ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "IGF1R":  ["IGF1R"],
    "IGFR":   ["IGF1R"],
    # PI3K family
    "PI3K":   ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    "PIK3":   ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    # JAK / STAT
    "JAK":    ["JAK1", "JAK2", "JAK3", "TYK2"],
    # SRC / RAF / RAS families
    "SRC":    ["SRC", "FYN", "LCK", "LYN", "YES1"],
    "RAF":    ["BRAF", "RAF1", "ARAF"],
    "RAS":    ["KRAS", "HRAS", "NRAS"],
    # AKT / GSK3
    "AKT":    ["AKT1", "AKT2", "AKT3"],
    "GSK3":   ["GSK3A", "GSK3B"],
    # CDK family — Palbociclib hits 4/6, Dinaciclib hits broader
    "CDK":    ["CDK1", "CDK2", "CDK4", "CDK6", "CDK7", "CDK9"],
    # MEK / ERK
    "MEK":    ["MAP2K1", "MAP2K2"],
    "MEK1":   ["MAP2K1"],
    "MEK2":   ["MAP2K2"],
    "ERK":    ["MAPK1", "MAPK3"],
    "ERK1":   ["MAPK3"],
    "ERK2":   ["MAPK1"],
}

# ---
# Drug-class labels that map to a known group of HGNC symbols.
# These would otherwise be unresolvable ("DNA methyltransferases" is not a gene).
# ---

_CLASS_ALIASES: dict[str, list[str]] = {
    # Epigenetic erasers / writers / readers
    "DNAMETHYLTRANSFERASES":  ["DNMT1", "DNMT3A", "DNMT3B"],
    "DNAMETHYLTRANSFERASE":   ["DNMT1", "DNMT3A", "DNMT3B"],
    "DNMT":                   ["DNMT1", "DNMT3A", "DNMT3B"],
    "HDAC":                   ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5",
                               "HDAC6", "HDAC7", "HDAC8", "HDAC9", "HDAC10"],
    "HDAC1/2":                ["HDAC1", "HDAC2"],
    "HISTONEDEACETYLASE":     ["HDAC1", "HDAC2", "HDAC3"],
    "HISTONEDEACETYLASES":    ["HDAC1", "HDAC2", "HDAC3"],
    "HMT":                    ["EZH2", "EHMT1", "EHMT2", "DOT1L"],
    "HISTONEMETHYLTRANSFERASE": ["EZH2", "EHMT1", "EHMT2", "DOT1L"],
    "BROMODOMAIN":            ["BRD2", "BRD3", "BRD4", "BRDT"],
    "BET":                    ["BRD2", "BRD3", "BRD4", "BRDT"],
    # DNA damage / repair targets
    "PARP":                   ["PARP1", "PARP2", "PARP3"],
    "TOPOISOMERASEI":         ["TOP1"],
    "TOPOISOMERASEII":        ["TOP2A", "TOP2B"],
    "TOPOISOMERASE":          ["TOP1", "TOP2A", "TOP2B"],
    "TOP1":                   ["TOP1"],
    "TOP2":                   ["TOP2A", "TOP2B"],
    # Microtubule
    "MICROTUBULE":            ["TUBB", "TUBB2A", "TUBB3", "TUBB4A"],
    "TUBULIN":                ["TUBB", "TUBB2A", "TUBB3", "TUBB4A"],
    # Proteasome
    "PROTEASOME":             ["PSMB1", "PSMB2", "PSMB5"],
    # Aromatase / steroid
    "AROMATASE":              ["CYP19A1"],
    # Hsp90
    "HSP90":                  ["HSP90AA1", "HSP90AB1", "HSP90B1"],
    # Antimetabolites: deliberately NOT mapped — too broad
    # ("Antimetabolite (DNA & RNA)" hits dozens of genes; would inflate AUROC noise).
}

# Tokens we strip when normalising a raw target name.
_STRIP_PUNCT = re.compile(r"[\s\-_/]+")
_PAREN_RE    = re.compile(r"\([^)]*\)")            # "IDH1 (R132H)" -> "IDH1 "
# Mutation-suffix patterns: "IDH2 R140Q mutant", "BRAF V600E", etc.
# The mutation token itself ("R140Q") followed by an optional "mutant" word.
_MUT_RE      = re.compile(
    r"\s+[A-Z]\d{1,4}[A-Z*](?:\s+mutant|\s+mutation)?\s*$",
    re.IGNORECASE,
)
_TRAIL_WORDS = re.compile(r"\s+(mutant|mutation|inhibitor|wt|wildtype)\s*$", re.IGNORECASE)
_GREEK_MAP   = str.maketrans({"α": "A", "β": "B", "γ": "G", "δ": "D",
                              "ε": "E", "ζ": "Z", "η": "H", "θ": "T",
                              "ι": "I", "κ": "K", "λ": "L", "μ": "M",
                              "ν": "N", "ξ": "X", "ο": "O", "π": "P",
                              "ρ": "R", "σ": "S", "τ": "T", "υ": "U",
                              "φ": "F", "χ": "C", "ψ": "Y", "ω": "W"})


def _normalise(name: str) -> str:
    """Upper-case, strip whitespace/punctuation/mutation/parenthetical suffixes."""
    s = name.strip()
    s = _PAREN_RE.sub("", s)            # drop "(R132H)" etc
    s = _MUT_RE.sub("", s)              # drop bare "R140Q" / "R140Q mutant"
    s = _TRAIL_WORDS.sub("", s)         # drop trailing "mutant"/"inhibitor"
    s = s.translate(_GREEK_MAP)         # PI3Kα -> PI3KA
    s = _STRIP_PUNCT.sub("", s)         # collapse whitespace + dashes
    return s.upper()


def _candidates(name: str) -> list[str]:
    """Generate plausible HGNC candidates from one raw target string.

    Strategy: emit the exact normalised name first (handles ``EGFR``), then
    the stem with trailing digits removed (handles ``IDH1 (R132H)`` -> ``IDH``
    -> nothing useful, but doesn't hurt), then any family / class aliases that
    match the stem.
    """
    if not name:
        return []
    norm = _normalise(name)
    if not norm:
        return []
    out: list[str] = [norm]

    # Stem (drop trailing non-letters: PIK3CA -> PIK3CA; PIK3 -> PIK3)
    stem = norm
    while stem and not stem[-1].isalpha():
        stem = stem[:-1]
    if stem and stem != norm:
        out.append(stem)

    # Class aliases (DNA methyltransferases, HDAC, ...)
    for k in (norm, stem):
        if k in _CLASS_ALIASES:
            out.extend(_CLASS_ALIASES[k])

    # Family aliases (PI3K, VEGFR, ...)
    for k in (norm, stem):
        if k in _FAMILY_ALIASES:
            out.extend(_FAMILY_ALIASES[k])

    # Common simple substring expansions kept for backwards-compat.
    if "ABL" in stem:
        out.append("ABL1")

    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def resolve_targets(
    raw_targets: Iterable[str],
    gene_set: set[str] | list[str],
) -> list[str]:
    """Return the subset of HGNC symbols in ``gene_set`` that ``raw_targets``
    resolves to, preserving input order, no duplicates.

    Semantics depend on the kind of label:

    - **Drug-class label** (``DNA methyltransferases`` -> DNMT1/3A/3B,
      ``HDAC`` -> HDAC1..10): all members present in ``gene_set`` are added,
      since the drug acts on the whole class.
    - **Family label** (``PI3K``, ``VEGFR``, ``BCR-ABL``): the first matching
      family-member symbol is added. Catalytic-subunit choice is otherwise
      arbitrary and inflating the positive set with all subunits would inflate
      AUROC for any model that learns the family broadly.
    - **Direct gene symbol** (``EGFR``, ``IDH1 (R132H)``): the first candidate
      is added.
    """
    gene_set = set(gene_set)
    resolved: list[str] = []

    def _add(g: str) -> None:
        if g in gene_set and g not in resolved:
            resolved.append(g)

    for raw in raw_targets:
        if not raw:
            continue
        norm = _normalise(raw)
        # Drug-class branch: add ALL class members
        cls_hits = _CLASS_ALIASES.get(norm)
        if cls_hits is None:
            stem = norm
            while stem and not stem[-1].isalpha():
                stem = stem[:-1]
            cls_hits = _CLASS_ALIASES.get(stem)
        if cls_hits is not None:
            for g in cls_hits:
                _add(g)
            continue

        # Otherwise: first matching candidate
        for cand in _candidates(raw):
            if cand in gene_set and cand not in resolved:
                resolved.append(cand)
                break

    return resolved


# Kept for backwards compatibility with callers that imported the old name.
def normalise_target_name(name: str) -> list[str]:
    return _candidates(name)
