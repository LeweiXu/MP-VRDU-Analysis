"""Inspect cached inference results: the document, gold pages, and both answers.

Purpose:
    Pick one or many cached cells for a generation task and dump a viewing packet
    you can open directly in VSCode: the source PDF, the rendered pages the model
    was fed, and an `info.md` with the question, gold pages, gold answer, model
    answer, every cached generate/judge field, and (if the judge phase ran) the
    verdict. Output goes to a gitignored `inspect/` folder at the repo root. This
    is a thin wrapper over `gates.viewer`.

Pipeline role:
    A standalone debugging utility; it reads the prediction/result caches a run
    produced and never touches them. Not part of generate/judge/build.

Arguments:
    `--generation TASK` (required), `--full`, `--run-tag`, and the selectors
    `--question`, `--doc`, `--representation`, `--condition`, `--incorrect-only`,
    `--abstained-only`, `--limit`, `--out`. See `--help`.

Examples:
    python -m scripts.inspect_results --run-tag bf16-lowres --full \
        --generation G1_sufficiency --limit 5
    python -m scripts.inspect_results --run-tag bf16-lowres --full \
        --generation G1_sufficiency --incorrect-only --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT, ExperimentConfig  # noqa: E402
from gates.viewer import select_items, write_item  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True, help="generation task, e.g. G1_sufficiency")
    parser.add_argument("--full", action="store_true", help="read the full-corpus cache (default: smoke)")
    parser.add_argument("--run-tag", help="run tag namespacing the cache tree")
    parser.add_argument("--question", help="filter to one question_id (e.g. mmlongbench:000123)")
    parser.add_argument("--doc", help="filter to one doc_id")
    parser.add_argument("--representation", help="filter to one rung: T/TL/TLV/V")
    parser.add_argument("--condition", help="filter to one conditioner, e.g. oracle")
    parser.add_argument("--incorrect-only", action="store_true", help="only judged-incorrect cells (needs judge phase)")
    parser.add_argument("--abstained-only", action="store_true", help="only abstained cells (needs judge phase)")
    parser.add_argument("--limit", type=int, help="cap the number of cells written")
    parser.add_argument("--out", type=Path, default=ROOT / "inspect", help="output dir (default: ./inspect)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=not args.full, run_tag=args.run_tag)
    items = select_items(
        config,
        args.generation,
        question_id=args.question,
        doc_id=args.doc,
        representation=args.representation,
        condition=args.condition,
        incorrect_only=args.incorrect_only,
        abstained_only=args.abstained_only,
        limit=args.limit,
    )
    if not items:
        print("no cells matched the filters")
        return 1
    for item in items:
        dest = write_item(item, args.out, config)
        print(f"wrote {dest}")
    print(f"\n{len(items)} cell(s) -> {args.out}  (open the info.md / PNGs in VSCode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
