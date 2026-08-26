"""
Trace the attention-faithfulness numbers quoted in the manuscript.

The submitted manuscript describes the same quantity -- the gain in attention
faithfulness from the head redesign -- as "roughly a threefold increase" in
Section 3.2 and "roughly sixfold" in the Discussion. Only one can be right, and
the Conclusion separately quotes 0.603 vs 0.407, which is 1.5x and is a
PathXDRP-vs-DRPreter comparison rather than a before-vs-after one.

This script finds every faithfulness number in results/ and results/archive/ so
the revised text can quote the correct one against the correct baseline.

Usage:
    python revision/scripts/faithfulness_provenance.py
Outputs:
    outputs/faithfulness_provenance.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def walk(obj, prefix=""):
    """Yield (dotted_key, value) for every scalar under obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, (int, float)):
        yield prefix, obj


def main() -> None:
    rows = []
    for f in sorted(ROOT.glob("results/**/*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in walk(d):
            if "faith" in k.lower() and "per_drug" not in k.lower():
                rows.append((str(f.relative_to(ROOT)).replace("\\", "/"), k, v))

    L = ["# Faithfulness numbers: provenance\n",
         "The submitted manuscript quotes the same gain as both *threefold* "
         "(Section 3.2) and *sixfold* (Discussion), and separately quotes "
         "0.603 vs 0.407 in the Conclusion. This lists every faithfulness "
         "number actually present in `results/`.\n",
         "| File | Key | Value |", "|---|---|---|"]
    for f, k, v in rows:
        L.append(f"| `{f}` | `{k}` | {v:.4f} |")
    L.append("")

    comp = [(f, k, v) for f, k, v in rows if "comp" in k.lower()]
    if comp:
        L.append("## Comprehensiveness values only\n")
        L.append("| File | Key | Value |")
        L.append("|---|---|---|")
        for f, k, v in comp:
            L.append(f"| `{f}` | `{k}` | {v:.4f} |")
        L.append("")

    L.append("## Action\n")
    L.append(
        "The revised manuscript must quote one ratio, against a named baseline, "
        "and use it consistently in Section 3.2, the Discussion and the "
        "Conclusion. Where the before-vs-after ancestor number is not present "
        "in `results/`, the comparison is not reproducible from the released "
        "artefacts and the claim must either be re-measured by the ablation "
        "(variant A of the redesigned Table 11) or dropped.\n"
    )
    (OUT / "faithfulness_provenance.md").write_text("\n".join(L), encoding="utf-8")
    for f, k, v in rows:
        print(f"{v:10.4f}  {k:52s} {f}")
    print(f"\nwrote {OUT/'faithfulness_provenance.md'}")


if __name__ == "__main__":
    main()
