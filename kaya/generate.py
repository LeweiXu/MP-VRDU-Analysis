"""Generate (GPU phase) for one experiment or a group, on a Kaya compute node.

Purpose:
    The GPU half of the experiment pipeline: it runs a chosen experiment's cells
    (reasoner over conditioned pages) and any GPU side work (retrievers, the
    doc-type classifier) and caches the predictions per experiment. The judge and
    the table build run later, off the cluster (`cli.experiments --phase judge`
    after `kaya.kaya pull`). Nothing here needs the internet.

Pipeline role:
    Thin GPU entry point over `experiments.driver.run_generate`. Submitted with
    `kaya.kaya submit`; runs one experiment per job so a single table re-runs as
    its own small, fast-queueing job (or `--experiment all` in one job).

CLI:
    `python -m kaya.kaya submit kaya/generate.py -- --experiment SEL [--full]`

Arguments:
    --experiment SEL: experiment name (e.g. T1_headline) or group
        (all, rq1, rq2, rq3, appendix). Default: all.
    --full: use the full config/corpus (8B, all questions). Default: smoke.
    --questions N: cap the corpus to the first N questions.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=generate

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.corpus import load_questions
from experiments.driver import run_generate
from kaya.prestage import prepare_tool_cache_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", help="experiment name or group (default: all)")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--questions", type=int, help="cap the corpus to the first N questions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=not args.full)
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config, limit=args.questions)
    run_generate(config, args.experiment, questions)
    print(f"generated {args.experiment}: {len(questions)} questions ({'full' if args.full else 'smoke'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
