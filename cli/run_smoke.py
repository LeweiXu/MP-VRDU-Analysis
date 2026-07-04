"""Run the full end-to-end smoke experiment and build all 8 tables.

Purpose:
    Local/driver entry point for the Section-1 MVP "full run, tiny data" sweep.
    It runs the real pipeline (Qwen3-VL-2B reasoner, BM25+BGE / ColQwen
    retrievers, Qwen doc-type classifier, GPT-4o-mini judge) over the frozen
    smoke corpus and writes the eight paper table CSVs. Nothing here is a stub.

Pipeline role:
    Thin wrapper over `experiments.smoke_run`. The two phases (`generate` on a
    GPU, `judge` on the internet) are the same code that `kaya/smoke_generate.py`
    and `kaya/smoke_judge.py` run on Kaya; locally you run `--phase all` in one
    env that has both a GPU and internet.

CLI:
    `python -m cli.run_smoke [--phase {generate,judge,all}] [options]`

Arguments:
    --phase {generate,judge,all}: which phase(s) to run (default: all).
        `generate` needs a GPU; `judge` needs internet + an API key.
    --judge SPEC: judge for the judge phase: `gemini` (default, free tier,
        needs GEMINI_API_KEY), `gpt-4o-mini` (paid, needs OPENAI_API_KEY), or
        `stub` (offline heuristic, plumbing only).
    --questions N: run only the first N smoke questions (default: all).
    --k N: top-k for the matched/cross retrieval cells (default: config k).
    --bootstrap N: document-level bootstrap resamples for the tables
        (default: 200; smoke tables are tiny).
"""

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.smoke import load_smoke_questions
from experiments.smoke_run import run_generate, run_judge
from kaya.prestage import prepare_tool_cache_env
from pipeline.judge import get_judge


def build_parser() -> argparse.ArgumentParser:
    """Return the smoke-run CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("generate", "judge", "all"),
        default="all",
        help="which phase(s) to run",
    )
    parser.add_argument(
        "--judge",
        default="gemini",
        help="judge for the judge phase: gemini (default), gpt-4o-mini, or stub",
    )
    parser.add_argument("--questions", type=int, help="run only the first N smoke questions")
    parser.add_argument("--k", type=int, help="top-k for matched/cross retrieval cells")
    parser.add_argument("--bootstrap", type=int, default=200, help="document-level bootstrap resamples")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=True)
    # Point Marker/Surya/Paddle/HF caches at the root-relative staged weights so
    # a local run reuses prestaged models instead of re-downloading.
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_smoke_questions(config.paths.data_dir)
    if args.questions is not None:
        questions = questions[: max(1, args.questions)]

    if args.phase in ("generate", "all"):
        run_generate(config, questions, k=args.k)
    if args.phase in ("judge", "all"):
        run_judge(config, questions, judge=get_judge(args.judge), k=args.k, n_bootstrap=args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
