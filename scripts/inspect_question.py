#!/usr/bin/env python3
"""Eyeball one question: its evidence pages + the rendered evidence page (Stage 1).

Exists to catch the 1-based/0-based off-by-one in evidence_pages by letting a
human look at the actual rendered page next to the question.

    # synthetic fixture (offline):
    python scripts/inspect_question.py --dataset synthetic
    # dev slice:
    python scripts/inspect_question.py --dataset dev --qid s1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.config import DataConfig
from mpvrdu.data.load import load_dataset
from mpvrdu.data.render import render_page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "dev", "full"])
    ap.add_argument("--qid", default=None, help="question id; default = first")
    ap.add_argument("--dpi", type=int, default=144)
    args = ap.parse_args()

    if args.dataset == "synthetic":
        cfg = DataConfig(name="synthetic")
    else:
        cfg = DataConfig(name="mmlongbench-doc", slice=args.dataset)
    ds = load_dataset(cfg)

    q = (next((x for x in ds.questions if x.qid == args.qid), None)
         if args.qid else ds.questions[0])
    if q is None:
        raise SystemExit(f"qid {args.qid!r} not found")

    print("=" * 70)
    print(f"qid:            {q.qid}")
    print(f"doc_id:         {q.doc_id}")
    print(f"question:       {q.question}")
    print(f"answer:         {q.answer}")
    print(f"answer_format:  {q.answer_format}")
    print(f"question_type:  {q.question_type.value}")
    print(f"evidence_pages: {q.evidence_pages} (1-based)  "
          f"-> {q.evidence_pages_zero_based} (0-based)")
    print(f"evidence_src:   {q.evidence_sources}")
    print("=" * 70)

    doc = ds.get_document(q.doc_id)
    if q.is_unanswerable:
        print("UNANSWERABLE: no evidence page to render (oracle feeds nothing).")
        return
    for p0 in q.evidence_pages_zero_based:
        rp = render_page(doc.pdf_path, p0, dpi=args.dpi, doc_id=doc.doc_id)
        print(f"rendered page {p0} (0-based) -> {rp.path}  "
              f"[{rp.width}x{rp.height}px, empty={rp.is_empty}]")


if __name__ == "__main__":
    main()
