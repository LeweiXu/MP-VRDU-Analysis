"""Split the MMLongBench documents into per-doc_type folders for eyeballing.

Purpose:
    The paper groups MMLongBench's seven native `doc_type` labels into three
    "Option-A" bins (text_heavy / in_between / visual_heavy, see
    `data/binning.py`). Before trusting that grouping we want to look at the PDFs
    in each doc_type and judge whether the text/visual assignment holds up. This
    script copies each document into `.data/mmlongbench_docs_split/<doc_type>/` so
    you can open a folder and flip through the real docs. It only reads the staged
    dataset and copies PDFs; it does not touch the parquet or any cache. Re-running
    is safe (it overwrites the split dir).

Pipeline role:
    A standalone browsing utility, companion to `scripts/annotate_docs.py`. Not
    part of the run pipeline.

Arguments:
    None. Run with `envs/mpvrdu/bin/python scripts/split_docs_by_type.py`.
"""

from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Put the repo root on the path so `data.*` imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binning import DOC_TYPE_TO_BIN  # noqa: E402
from data.loader import (  # noqa: E402
    find_mmlongbench_root,
    load_raw_mmlongbench,
    resolve_pdf,
)

# Sub-dir under .data/ that holds the split. Sits next to `mmlongbench/`.
SPLIT_DIRNAME = "mmlongbench_docs_split"


def safe_dirname(doc_type: str) -> str:
    """Turn a doc_type label into a filesystem-safe folder name.

    doc_type values contain slashes and spaces ("Administration/Industry file",
    "Tutorial/Workshop"), so collapse those to underscores.
    """

    cleaned = " ".join(str(doc_type).split())
    for ch in "/\\":
        cleaned = cleaned.replace(ch, "-")
    return cleaned.replace(" ", "_")


def main() -> None:
    dataset_root = find_mmlongbench_root()
    out_root = dataset_root.parent / SPLIT_DIRNAME

    # doc_id -> doc_type (each doc has exactly one doc_type in this corpus).
    doc_types: dict[str, str] = {}
    for row in load_raw_mmlongbench():
        doc_id = str(row.get("doc_id") or "")
        doc_type = str(row.get("doc_type") or "")
        if not doc_id:
            continue
        prev = doc_types.get(doc_id)
        if prev is not None and prev != doc_type:
            print(f"  ! {doc_id}: multiple doc_types ({prev!r} vs {doc_type!r}), keeping first")
            continue
        doc_types[doc_id] = doc_type

    # Fresh split dir each run so it's always an exact mirror of the corpus.
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    per_type: dict[str, int] = defaultdict(int)
    missing: list[str] = []
    for doc_id, doc_type in sorted(doc_types.items()):
        try:
            pdf = resolve_pdf(doc_id)
        except FileNotFoundError:
            missing.append(doc_id)
            continue
        dest_dir = out_root / safe_dirname(doc_type)
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(pdf, dest_dir / pdf.name)
        per_type[doc_type] += 1

    print(f"\nwrote {sum(per_type.values())} docs to {out_root}")
    for doc_type in sorted(per_type):
        assumed_bin = DOC_TYPE_TO_BIN.get(doc_type, "UNMAPPED")
        print(f"  {per_type[doc_type]:>3}  {doc_type:<32} -> {assumed_bin}")
    if missing:
        print(f"\n{len(missing)} doc_ids had no PDF on disk:")
        for doc_id in missing:
            print(f"  - {doc_id}")


if __name__ == "__main__":
    main()
