"""Judge phase (local): score a task's cached predictions with an LLM judge.

Purpose:
    The thin local entry point over `experiments.driver.run_judge`. It reads a
    generation task's cached predictions and scores each with a real judge,
    writing `results.jsonl` per task. It builds no tables (that is `cli/build.py`)
    and loads no models.

CLI:
    `python -m cli.judge [--generation SEL] [--full] [--judge SPEC]`

    Pass the same corpus/model flags as the generate phase (`--full`,
    `--per-bin-questions`, `--sample-seed`, `--quantization`, `--run-tag`) so the
    judge re-resolves the exact same cells.

Arguments:
    --generation SEL: which task(s) to score (name, group, or comma list).
        --judge picks the scorer (gemini default, gpt-4o-mini, stub).
        --continue-on-error skips cells with no cached prediction (partial cache)
        and continues past a task failure. See `build_parser`.
"""

from __future__ import annotations

import argparse

from experiments.corpus import load_questions
from experiments.driver import config_from_args, run_judge
from experiments.paths import configure_logging
from pipeline.judge import get_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="all", help="generation task or group to score (default: all)")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--judge", default="gemini", help="judge: gemini (default), gpt-4o-mini, or stub")
    parser.add_argument("--questions", type=int, help="global cap: first N questions (overrides --per-bin-questions)")
    parser.add_argument("--per-bin-questions", type=int, help="full mmlongbench: ~N questions per Option-A bin (default 100; 0 = whole corpus)")
    parser.add_argument("--sample-seed", type=int, help="which documents land in the per-bin subset (default 0)")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), help="quantized reasoner spec suffix; must match the generate phase")
    parser.add_argument("--visual-resolution", choices=("full", "high", "med", "low", "min"), help="ignored for scoring; kept so judge flags can mirror generate")
    parser.add_argument("--run-tag", help="cache namespace to read; must match the generate phase")
    parser.add_argument("--continue-on-error", action="store_true", help="skip cells with no cached prediction (partial cache) and continue past task failures")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    questions = load_questions(config, limit=args.questions)
    statuses = run_judge(
        config,
        args.generation,
        questions,
        judge_impl=get_judge(args.judge),
        continue_on_error=args.continue_on_error,
    )
    failed = [status for status in statuses if status.status != "success"]
    print(
        f"judged {args.generation}: {len(statuses) - len(failed)} scored, {len(failed)} failed. "
        f"Build tables with: python -m cli.build"
        f"{' --full' if args.full else ''}{f' --run-tag {args.run_tag}' if args.run_tag else ''}"
    )
    for status in failed:
        print(f"failed {status.experiment}: {status.error_type}: {status.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
