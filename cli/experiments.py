"""Run paper-table experiments: generate on a GPU, judge/build anywhere.

Purpose:
    The single entry point for running one experiment, an RQ group, or all of
    them, for the smoke or the full corpus. It drives the two phases:

    - `--phase generate` (needs a GPU): caches predictions per experiment.
    - `--phase judge` (needs internet + a judge key, no GPU): scores the cached
      predictions and writes the table CSVs.
    - `--phase all`: both, in one process (a machine with GPU + internet).

    The intended Kaya flow keeps the heavy half on the cluster and the light half
    local: `kaya.kaya submit kaya/generate.py -- --experiment X`, then
    `kaya.kaya pull`, then `python -m cli.experiments --phase judge --experiment X`
    locally.

Pipeline role:
    Thin wrapper over `experiments.driver` + `experiments.registry`. Same code
    serves smoke and full; `--full` selects the full config/corpus.

CLI:
    `python -m cli.experiments [--experiment SEL] [--phase P] [--full] [options]`

Arguments:
    --experiment SEL: an experiment name (e.g. T1_headline) or a group
        (all, section2, rq1, rq2, rq3, appendix). Default: all.
    --phase {generate,judge,all}: which phase(s) to run. Default: all.
    --full: use the full config/corpus (8B, all questions). Default: smoke.
    --judge SPEC: judge for the judge phase: gemini (default), gpt-4o-mini, stub.
    --questions N: cap the corpus to the first N questions (smoke defaults to all).
    --continue-on-error: for generate/all grouped runs, continue after an
        experiment failure and record its phase status.
"""

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.corpus import load_questions
from experiments.driver import configure_logging, run_generate, run_judge
from kaya.prestage import prepare_tool_cache_env
from pipeline.judge import get_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", help="experiment name or group (default: all)")
    parser.add_argument("--phase", choices=("generate", "judge", "all"), default="all")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--judge", default="gemini", help="judge: gemini (default), gpt-4o-mini, or stub")
    parser.add_argument("--questions", type=int, help="cap the corpus to the first N questions")
    parser.add_argument("--continue-on-error", action="store_true", help="continue generate phase after failures")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level per-cell/per-stage logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=not args.full)
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    # Point tool/model caches at the root-relative staged weights for local runs.
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config, limit=args.questions)

    if args.phase in ("generate", "all"):
        run_generate(config, args.experiment, questions, continue_on_error=args.continue_on_error)
        print(f"generated: {args.experiment} ({len(questions)} questions, mode={'full' if args.full else 'smoke'})")
    if args.phase in ("judge", "all"):
        written = run_judge(config, args.experiment, questions, judge_impl=get_judge(args.judge))
        for key, path in sorted(written.items()):
            print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
