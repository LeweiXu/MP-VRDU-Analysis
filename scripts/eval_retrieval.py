#!/usr/bin/env python3
"""Retrieval-recall only, no generator (plan Stage 4 — fast, local-able).

Validate a retriever on recall@k BEFORE spending compute on downstream accuracy.

    python scripts/eval_retrieval.py --config configs/retr_bm25_image.yaml \
        --ks 1 2 4 8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.config import load_config
from mpvrdu.data.load import load_dataset
from mpvrdu.logging_utils import get_logger
from mpvrdu.results import ResultsWriter, results_path
from mpvrdu.retrieve import build_selector, evaluate_retrieval

log = get_logger("eval_retrieval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = load_dataset(cfg.data)
    selector = build_selector(cfg)
    result = evaluate_retrieval(selector, ds, ks=args.ks)

    log.info("recall@k for %s: %s (n_answerable=%d)",
             result["selector"],
             {k: round(v, 3) for k, v in result["recall_at_k"].items()},
             result["n_answerable"])

    out = Path(args.out) if args.out else results_path(cfg).with_suffix(".recall.jsonl")
    with ResultsWriter(out, config=cfg) as w:
        w.write({"kind": "recall_summary", "recall_at_k": result["recall_at_k"],
                 "n_answerable": result["n_answerable"]}, _is_row=False)
        for row in result["rows"]:
            w.write(row)
    print(json.dumps(result["recall_at_k"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
