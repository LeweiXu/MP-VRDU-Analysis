#!/usr/bin/env python3
"""Carve the dev slice from a downloaded MMLongBench-Doc (Stage 1 deliverable).

    python scripts/build_dev_slice.py --src data/mmlongbench

Produces data/dev_slice/ (samples.json + documents/), which all local tests and
the `slice: dev` configs load.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.data.dataset import load_samples_json
from mpvrdu.data.slice import build_dev_slice
from mpvrdu.logging_utils import get_logger

log = get_logger("build_dev_slice")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mmlongbench",
                    help="dir holding samples.json + documents/")
    ap.add_argument("--max-docs", type=int, default=5)
    ap.add_argument("--max-questions", type=int, default=30)
    args = ap.parse_args()

    src = Path(args.src)
    ds = load_samples_json(src / "samples.json", src / "documents")
    slice_ds = build_dev_slice(ds, max_docs=args.max_docs,
                               max_questions=args.max_questions)
    log.info("dev slice built: %d questions, types=%s",
             len(slice_ds), slice_ds.type_counts())


if __name__ == "__main__":
    main()
