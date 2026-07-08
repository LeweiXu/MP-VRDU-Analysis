"""GPU generation entry point: runs a task over a small corpus and caches rows."""

from __future__ import annotations

import argparse

from config import DEPLOYMENT_RESOLUTION, ExperimentConfig
from data.binning import stamp_bins
from data.loader import load_mmlongbench
from experiments.engine.driver import generate
from experiments.engine.paths import configure_logging
from experiments.registry import resolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", help="task name or group (e.g. G1_oracle_ladder, reasoners, all)")
    parser.add_argument("--reasoner-spec", default="qwen3vl-2b-local")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--visual-resolution", default=DEPLOYMENT_RESOLUTION,
                        help="resolution preset (min/low/med/high/full)")
    parser.add_argument("--judge-spec", default="stub")
    parser.add_argument("--limit", type=int, default=None, help="cap questions per task (smoke/debug)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = ExperimentConfig(
        reasoner_spec=args.reasoner_spec,
        quantization=args.quantization,
        visual_resolution=args.visual_resolution,
        judge_spec=args.judge_spec,
    )
    questions = stamp_bins(load_mmlongbench(config.paths.data_dir))
    for task in resolve(args.task):
        generate(config, task, questions, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
