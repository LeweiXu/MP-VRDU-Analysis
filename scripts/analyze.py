#!/usr/bin/env python3
"""Regenerate the results tables (+ optional figures) from raw JSONL (Stage 7).

One command turns a results/ directory into a Markdown comparison table:

    python scripts/analyze.py --results results --out results/summary.md

This is the reproducibility entry point: the table is a pure function of the
JSONL files, each of which embeds its own config.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.analysis import aggregate_dir, to_markdown_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="dir of *.jsonl results")
    ap.add_argument("--out", default=None, help="write Markdown table here")
    ap.add_argument("--json", default=None, help="also dump full summaries as JSON")
    args = ap.parse_args()

    summaries = aggregate_dir(args.results)
    if not summaries:
        print(f"no result files under {args.results}")
        return
    table = to_markdown_table(summaries)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\n-> wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"-> wrote {args.json}")


if __name__ == "__main__":
    main()
